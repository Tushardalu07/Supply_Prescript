
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    ROOT / "ml" / "reports" / "test_predictions.csv"
)

REPORT_DIR = ROOT / "ml" / "reports"

df = pd.read_csv(PREDICTIONS_PATH)

required = [
    "order_id",
    "actual_late_risk",
    "predicted_late_risk",
    "delay_probability",
    "actual_days_late",
    "predicted_days_late",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required].isna().any().any():
    raise ValueError("Missing values detected in prediction data.")

grouped = df.groupby("order_id").agg(
    rows=("order_id", "size"),
    actual_class_values=("actual_late_risk", "nunique"),
    predicted_class_values=("predicted_late_risk", "nunique"),
    actual_delay_values=("actual_days_late", "nunique"),
    predicted_delay_values=("predicted_days_late", "nunique"),
    min_probability=("delay_probability", "min"),
    max_probability=("delay_probability", "max"),
)

report = {
    "prediction_rows": int(len(df)),
    "unique_test_orders": int(df["order_id"].nunique()),
    "orders_with_multiple_rows": int((grouped["rows"] > 1).sum()),
    "orders_with_inconsistent_actual_class": int(
        (grouped["actual_class_values"] > 1).sum()
    ),
    "orders_with_inconsistent_predicted_class": int(
        (grouped["predicted_class_values"] > 1).sum()
    ),
    "orders_with_inconsistent_actual_delay": int(
        (grouped["actual_delay_values"] > 1).sum()
    ),
    "orders_with_inconsistent_predicted_delay": int(
        (grouped["predicted_delay_values"] > 1).sum()
    ),
    "probabilities_within_valid_range": bool(
        df["delay_probability"].between(0, 1).all()
    ),
}

path = REPORT_DIR / "prediction_consistency_audit.json"

path.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8",
)

print("SUPPLYPRESCRIPT PREDICTION CONSISTENCY AUDIT")
print(json.dumps(report, indent=4))
print(f"\nReport saved: {path}")
