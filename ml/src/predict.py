
from pathlib import Path
import json

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

CLASSIFIER_PATH = ARTIFACTS / "delay_classifier.joblib"
REGRESSOR_PATH = ARTIFACTS / "days_late_regressor.joblib"
METADATA_PATH = ARTIFACTS / "model_metadata.json"


# ---------------------------------------------------------
# LOAD MODEL METADATA
# ---------------------------------------------------------

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Model metadata not found: {METADATA_PATH}"
    )

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

MODEL_VERSION = metadata["model_version"]
CLASSIFICATION_THRESHOLD = float(
    metadata["classification_threshold"]
)

EXPECTED_FEATURES = metadata["features"]

if not 0.0 <= CLASSIFICATION_THRESHOLD <= 1.0:
    raise ValueError("Invalid classification threshold in metadata.")

if not EXPECTED_FEATURES:
    raise ValueError("Model feature metadata is empty.")


# ---------------------------------------------------------
# LOAD TRAINED MODELS
# ---------------------------------------------------------

classifier = joblib.load(CLASSIFIER_PATH)
regressor = joblib.load(REGRESSOR_PATH)


# Verify that saved models use the metadata feature schema.
for model_name, model in [
    ("classifier", classifier),
    ("regressor", regressor),
]:
    model_features = list(model.feature_names_in_)

    if model_features != EXPECTED_FEATURES:
        raise ValueError(
            f"{model_name} feature schema does not match "
            "model_metadata.json."
        )


# ---------------------------------------------------------
# VALIDATE SHIPMENT INPUT
# ---------------------------------------------------------

def validate_shipment(shipment: dict) -> dict:
    """Validate and arrange input using the training feature schema."""

    if not isinstance(shipment, dict):
        raise TypeError(
            "Shipment input must be a dictionary."
        )

    missing = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in shipment
    ]

    if missing:
        raise ValueError(
            f"Missing required shipment features: {missing}"
        )

    return {
        feature: shipment[feature]
        for feature in EXPECTED_FEATURES
    }


# ---------------------------------------------------------
# PREDICT SHIPMENT DELAY
# ---------------------------------------------------------

def predict_shipment(shipment: dict) -> dict:
    """Predict late-delivery risk and expected delay duration."""

    validated = validate_shipment(shipment)

    row = pd.DataFrame(
        [validated],
        columns=EXPECTED_FEATURES,
    )

    # Classification probability.
    probability = float(
        classifier.predict_proba(row)[0, 1]
    )

    # Apply the threshold selected using validation data.
    predicted_late_risk = int(
        probability >= CLASSIFICATION_THRESHOLD
    )

    # Regression prediction.
    days_late = max(
        0.0,
        float(regressor.predict(row)[0]),
    )

    # Risk categories aligned with the classification threshold.
    if probability >= 0.70:
        risk = "High"
    elif probability >= CLASSIFICATION_THRESHOLD:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "delay_probability": round(probability, 4),
        "predicted_late_risk": predicted_late_risk,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "predicted_days_late": round(days_late, 2),
        "risk_category": risk,
        "model_version": MODEL_VERSION,
    }



# ---------------------------------------------------------
# LOCAL TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    example = {
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

    result = predict_shipment(example)

    print(json.dumps(result))
