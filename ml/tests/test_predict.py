
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
            "xgboost-v1",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
