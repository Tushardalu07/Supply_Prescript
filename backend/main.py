from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Supply Prescript API")


class Prescription(BaseModel):
    patient_name: str = Field(min_length=2, max_length=100)
    medicine_name: str = Field(min_length=2, max_length=100)
    dosage: str = Field(min_length=1, max_length=50)
    instructions: str = Field(min_length=1, max_length=300)


@app.get("/")
def home():
    return {"message": "Supply Prescript backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/prescriptions")
def create_prescription(prescription: Prescription):
    return {
        "message": "Prescription received successfully",
        "prescription": prescription.model_dump()
    }