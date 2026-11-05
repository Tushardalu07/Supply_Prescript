from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg

from backend.database import get_db_connection

app = FastAPI(title="Supply Prescript API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ShipmentCreate(BaseModel):
    shipment_id: str = Field(min_length=1, max_length=50)
    shipping_days: int = Field(gt=0)
    expected_days: int = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Supply Prescript API is running"}


@app.get("/shipments")
def get_shipments():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, shipment_id, shipping_days,
                           expected_days, created_at
                    FROM shipments
                    ORDER BY id DESC
                """)
                rows = cur.fetchall()

        return {
            "shipments": [
                {
                    "id": row[0],
                    "shipment_id": row[1],
                    "shipping_days": row[2],
                    "expected_days": row[3],
                    "created_at": row[4].isoformat()
                    if row[4] else None
                }
                for row in rows
            ]
        }
    except psycopg.Error:
        raise HTTPException(
            status_code=500,
            detail="Database error while fetching shipments"
        )


@app.post("/shipments")
def create_shipment(shipment: ShipmentCreate):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO shipments
                        (shipment_id, shipping_days, expected_days)
                    VALUES (%s, %s, %s)
                    RETURNING id, shipment_id,
                              shipping_days, expected_days, created_at
                """, (
                    shipment.shipment_id,
                    shipment.shipping_days,
                    shipment.expected_days
                ))
                row = cur.fetchone()

        return {
            "message": "Shipment created successfully",
            "shipment": {
                "id": row[0],
                "shipment_id": row[1],
                "shipping_days": row[2],
                "expected_days": row[3],
                "created_at": row[4].isoformat() if row[4] else None
            }
        }
    except psycopg.errors.UniqueViolation:
        raise HTTPException(
            status_code=409,
            detail="This shipment ID already exists"
        )
    except psycopg.Error:
        raise HTTPException(
            status_code=500,
            detail="Database error while creating shipment"
        )

@app.delete("/shipments/{shipment_id}")
def delete_shipment(shipment_id: str):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM shipments WHERE shipment_id = %s RETURNING shipment_id",
                    (shipment_id,)
                )
                deleted = cur.fetchone()

                if not deleted:
                    raise HTTPException(
                        status_code=404,
                        detail="Shipment not found"
                    )

        return {
            "message": "Shipment deleted successfully",
            "shipment_id": deleted[0]
        }

    except psycopg.Error:
        raise HTTPException(
            status_code=500,
            detail="Database error while deleting shipment"
        )


class ExecuteDecisionRequest(BaseModel):
    decision_id: int = Field(gt=0)


class ActualOutcomeRequest(BaseModel):
    actual_days: int = Field(gt=0)
    actual_cost: float = Field(ge=0)


# Baseline recommendation options
# These are sample rules, not ML predictions.

def get_recommendation_options(shipment):
    shipping_days = shipment["shipping_days"]
    expected_days = shipment["expected_days"]

    return [
        {
            "id": 1,
            "type": "Air Freight",
            "extra_cost": 12000,
            "delivery_days": 2,
            "score": 0.91
        },
        {
            "id": 2,
            "type": "Secondary Supplier",
            "extra_cost": 7000,
            "delivery_days": 4,
            "score": 0.78
        },
        {
            "id": 3,
            "type": "Delay Launch",
            "extra_cost": 0,
            "delivery_days": max(shipping_days, expected_days),
            "score": 0.55
        }
    ]


# 1. Get recommendations

@app.get("/shipments/{shipment_id}/recommendations")
def get_recommendations(shipment_id: str):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT shipment_id, shipping_days, expected_days
                    FROM shipments
                    WHERE shipment_id = %s
                """, (shipment_id,))

                row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Shipment not found")

        shipment = {
            "shipment_id": row[0],
            "shipping_days": row[1],
            "expected_days": row[2]
        }

        return {
            "shipment_id": shipment_id,
            "options": get_recommendation_options(shipment),
            "note": "Baseline sample recommendations; ML model not connected yet."
        }

    except psycopg.Error:
        raise HTTPException(status_code=500, detail="Database error")


# 2. Execute and save a decision

@app.post("/shipments/{shipment_id}/execute")
def execute_decision(shipment_id: str, request: ExecuteDecisionRequest):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT shipment_id, shipping_days, expected_days
                    FROM shipments
                    WHERE shipment_id = %s
                """, (shipment_id,))

                row = cur.fetchone()

                if not row:
                    raise HTTPException(status_code=404, detail="Shipment not found")

                shipment = {
                    "shipment_id": row[0],
                    "shipping_days": row[1],
                    "expected_days": row[2]
                }

                options = get_recommendation_options(shipment)

                selected = next(
                    (option for option in options
                     if option["id"] == request.decision_id),
                    None
                )

                if not selected:
                    raise HTTPException(status_code=404, detail="Recommendation not found")

                cur.execute("""
                    INSERT INTO decisions (
                        shipment_id, decision_id, decision_type,
                        predicted_days, predicted_cost, score
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    shipment_id,
                    selected["id"],
                    selected["type"],
                    selected["delivery_days"],
                    selected["extra_cost"],
                    selected["score"]
                ))

                decision_record_id = cur.fetchone()[0]

                cur.execute("""
                    INSERT INTO outcomes (decision_record_id)
                    VALUES (%s)
                """, (decision_record_id,))

        return {
            "status": "success",
            "message": "Decision saved successfully",
            "decision_record_id": decision_record_id,
            "decision": selected["type"]
        }

    except psycopg.Error:
        raise HTTPException(status_code=500, detail="Database error while saving decision")


# 3. Get decision outcomes

@app.get("/outcomes")
def get_outcomes():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        d.shipment_id,
                        d.decision_type,
                        d.predicted_cost,
                        o.actual_cost,
                        d.predicted_days,
                        o.actual_days
                    FROM decisions d
                    JOIN outcomes o
                        ON d.id = o.decision_record_id
                    ORDER BY d.id DESC
                """)

                rows = cur.fetchall()

        return {
            "outcomes": [
                {
                    "shipment_id": row[0],
                    "decision": row[1],
                    "predicted_cost": float(row[2]),
                    "actual_cost": float(row[3]) if row[3] is not None else None,
                    "predicted_days": row[4],
                    "actual_days": row[5]
                }
                for row in rows
            ]
        }

    except psycopg.Error:
        raise HTTPException(status_code=500, detail="Database error while fetching outcomes")


# 4. Save actual outcome later

@app.put("/outcomes/{decision_record_id}")
def update_actual_outcome(
    decision_record_id: int,
    outcome: ActualOutcomeRequest
):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE outcomes
                    SET actual_days = %s, actual_cost = %s
                    WHERE decision_record_id = %s
                    RETURNING id
                """, (
                    outcome.actual_days,
                    outcome.actual_cost,
                    decision_record_id
                ))

                if not cur.fetchone():
                    raise HTTPException(
                        status_code=404,
                        detail="Decision outcome not found"
                    )

        return {"message": "Actual outcome saved successfully"}

    except psycopg.Error:
        raise HTTPException(
            status_code=500,
            detail="Database error while updating outcome"
        )