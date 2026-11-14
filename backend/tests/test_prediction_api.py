from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.main import app

client = TestClient(app)

SAMPLE_PREDICTION = {
    "delay_probability": 0.3687,
    "predicted_late_risk": 1,
    "classification_threshold": 0.35,
    "predicted_days_late": 0.57,
    "risk_category": "Medium",
    "model_version": "xgboost-v2",
}

VALID_SHIPMENT = {
    "Type": "DEBIT",
    "Category Id": 10,
    "Category Name": "Accessories",
    "Customer Segment": "Consumer",
    "Department Id": 5,
    "Department Name": "Golf",
    "Market": "USCA",
    "Order Item Discount": 5.0,
    "Order Item Discount Rate": 0.05,
    "Order Item Product Price": 100.0,
    "Order Item Quantity": 1,
    "Product Category Id": 10,
    "Product Price": 100.0,
    "Shipping Mode": "Standard Class",
    "order_month": 10,
    "order_dayofweek": 2,
}


def test_valid_prediction():
    with patch("backend.main.predict_shipment", return_value=SAMPLE_PREDICTION):
        response = client.post("/predict", json={"shipment": VALID_SHIPMENT})

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_prediction_response_contains_required_fields():
    with patch("backend.main.predict_shipment", return_value=SAMPLE_PREDICTION):
        response = client.post("/predict", json={"shipment": VALID_SHIPMENT})

    prediction = response.json()["prediction"]

    expected_fields = {
        "delay_probability",
        "predicted_late_risk",
        "classification_threshold",
        "predicted_days_late",
        "risk_category",
        "model_version",
    }

    assert set(prediction.keys()) == expected_fields


def test_missing_feature_returns_422():
    response = client.post("/predict", json={"shipment": {"Type": "DEBIT"}})
    assert response.status_code == 422


def test_invalid_shipment_type_returns_422():
    response = client.post("/predict", json={"shipment": ["invalid"]})
    assert response.status_code == 422


def test_prediction_failure_returns_500():
    with patch("backend.main.predict_shipment", side_effect=RuntimeError("Model failure")):
        response = client.post("/predict", json={"shipment": VALID_SHIPMENT})

    assert response.status_code == 500
    assert response.json()["detail"] == "Shipment prediction failed"