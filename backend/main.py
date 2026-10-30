from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Supply Prescript API")


class ShipmentData(BaseModel):
    shipment_id: str = Field(min_length=1)
    shipping_days: int = Field(gt=0)
    expected_days: int = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Supply Prescript backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/shipment-delay")
def predict_shipment_delay(shipment: ShipmentData):
    delayed = shipment.shipping_days > shipment.expected_days

    return {
        "shipment_id": shipment.shipment_id,
        "predicted_delay": delayed,
        "status": "Delayed" if delayed else "On time"
    }