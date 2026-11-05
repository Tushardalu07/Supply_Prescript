
import ast
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREDICT_SCRIPT = ROOT / "ml" / "src" / "predict.py"

MODEL_FILES = [
    ROOT / "ml" / "artifacts" / "delay_classifier.joblib",
    ROOT / "ml" / "artifacts" / "days_late_regressor.joblib",
]


class TestShipmentPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for path in [PREDICT_SCRIPT, *MODEL_FILES]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {path}"
                )

        result = subprocess.run(
            [sys.executable, str(PREDICT_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        output = next(
            (
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip().startswith("{")
                and line.strip().endswith("}")
            ),
            None,
        )

        if output is None:
            raise AssertionError(
                f"No prediction dictionary found:\n{result.stdout}"
            )

        cls.prediction = ast.literal_eval(output)

    def test_required_fields(self):
        required = {
            "delay_probability",
            "predicted_days_late",
            "risk_category",
            "model_version",
        }
        self.assertTrue(required.issubset(self.prediction))

    def test_probability_range(self):
        probability = float(
            self.prediction["delay_probability"]
        )
        self.assertTrue(math.isfinite(probability))
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)

    def test_predicted_delay(self):
        days = float(
            self.prediction["predicted_days_late"]
        )
        self.assertTrue(math.isfinite(days))
        self.assertGreaterEqual(days, 0)

    def test_risk_category(self):
        self.assertIn(
            self.prediction["risk_category"],
            {"Low", "Medium", "High"},
        )

    def test_model_version(self):
       self.assertEqual(
           self.prediction["model_version"],
           "xgboost-v2",
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)

# Additional prediction input validation tests

import pytest

from ml.src.predict import (
    EXPECTED_FEATURES,
    predict_shipment,
    validate_shipment,
)


@pytest.fixture
def sample_shipment():
    return {
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


def test_valid_input_has_exact_training_features(sample_shipment):
    result = validate_shipment(sample_shipment)
    assert list(result.keys()) == EXPECTED_FEATURES


def test_missing_feature_is_rejected(sample_shipment):
    sample_shipment.pop("Shipping Mode")
    with pytest.raises(ValueError, match="Missing required"):
        predict_shipment(sample_shipment)


def test_extra_features_are_ignored(sample_shipment):
    sample_shipment["Unexpected Column"] = "extra"
    result = validate_shipment(sample_shipment)
    assert "Unexpected Column" not in result


def test_prediction_output_is_valid(sample_shipment):
    result = predict_shipment(sample_shipment)
    assert 0 <= result["delay_probability"] <= 1
    assert result["predicted_days_late"] >= 0
    assert result["risk_category"] in ["Low", "Medium", "High"]
    assert result["model_version"] == "xgboost-v2"


def test_non_dictionary_input_is_rejected():
    with pytest.raises(TypeError):
        validate_shipment(["invalid"])
def test_classification_threshold_is_loaded():
    from ml.src.predict import CLASSIFICATION_THRESHOLD

    assert CLASSIFICATION_THRESHOLD == 0.35


def test_prediction_contains_binary_late_risk():
    from ml.src.predict import (
        predict_shipment,
        CLASSIFICATION_THRESHOLD,
    )

    shipment = {
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

    result = predict_shipment(shipment)

    assert result["predicted_late_risk"] in (0, 1)
    assert result["classification_threshold"] == CLASSIFICATION_THRESHOLD
    assert result["model_version"] == "xgboost-v2"