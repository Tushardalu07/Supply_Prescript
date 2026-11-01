from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg

from backend.database import get_db_connection

app = FastAPI(title="Supply Prescript API")


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