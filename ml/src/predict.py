
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

classifier = joblib.load(ARTIFACTS / "delay_classifier.joblib")
regressor = joblib.load(ARTIFACTS / "days_late_regressor.joblib")


def predict_shipment(shipment: dict) -> dict:
    """
    shipment must contain the same feature columns used in training.
    Example fields include Shipping Mode, Type, Category Name,
    Order Item Quantity, Product Price, order_month and order_dayofweek.
    """
    row = pd.DataFrame([shipment])

    probability = float(classifier.predict_proba(row)[0, 1])
    days_late = max(0.0, float(regressor.predict(row)[0]))

    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "delay_probability": round(probability, 4),
        "predicted_days_late": round(days_late, 2),
        "risk_category": risk,
        "model_version": "xgboost-v1",
    }


if __name__ == "__main__":
    example = {
        "Type": "DEBIT",
        "Benefit per order": 20.0,
        "Sales per customer": 100.0,
        "Category Id": 10,
        "Category Name": "Accessories",
        "Customer City": "Chicago",
        "Customer Country": "Estados Unidos",
        "Customer Segment": "Consumer",
        "Customer State": "Illinois",
        "Department Id": 5,
        "Department Name": "Golf",
        "Market": "USCA",
        "Order City": "Chicago",
        "Order Country": "Estados Unidos",
        "Order Item Cardprod Id": 100,
        "Order Item Discount": 5.0,
        "Order Item Discount Rate": 0.05,
        "Order Item Product Price": 100.0,
        "Order Item Profit Ratio": 0.2,
        "Order Item Quantity": 1,
        "Sales": 100.0,
        "Order Item Total": 95.0,
        "Order Profit Per Order": 20.0,
        "Order Region": "Central",
        "Order State": "Illinois",
        "Product Card Id": 100,
        "Product Category Id": 10,
        "Product Name": "Example Product",
        "Product Price": 100.0,
        "Product Status": 0,
        "Shipping Mode": "Standard Class",
        "order_month": 10,
        "order_dayofweek": 2,
    }

    print(predict_shipment(example))
