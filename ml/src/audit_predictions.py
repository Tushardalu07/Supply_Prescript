
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT / "ml" / "reports"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"
METRICS_PATH = REPORT_DIR / "metrics.json"
METADATA_PATH = ROOT / "ml" / "artifacts" / "model_metadata.json"

# Load prediction data and reference metadata.
df = pd.read_csv(PREDICTIONS_PATH)

with open(METRICS_PATH, encoding="utf-8") as file:
    metrics = json.load(file)

with open(METADATA_PATH, encoding="utf-8") as file:
    metadata = json.load(file)

threshold = float(metadata["classification_threshold"])
expected_rows = int(metrics["test_rows"])
expected_orders = int(metrics["test_orders"])

required = [
    "order_id",
    "actual_late_risk",
    "predicted_late_risk",
    "delay_probability",
    "actual_days_late",
    "predicted_days_late",
]

missing_columns = [
    column for column in required if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Validate missing values.
missing_values = int(df[required].isna().sum().sum())

if missing_values:
    raise ValueError("Missing values detected in prediction data.")

# Validate numeric fields.
numeric_columns = [
    "actual_late_risk",
    "predicted_late_risk",
    "delay_probability",
    "actual_days_late",
    "predicted_days_late",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="raise")

# Item-level validation.
probability_valid = bool(
    df["delay_probability"].between(0, 1).all()
)

actual_class_valid = bool(
    df["actual_late_risk"].isin([0, 1]).all()
)

predicted_class_valid = bool(
    df["predicted_late_risk"].isin([0, 1]).all()
)

expected_predictions = (
    df["delay_probability"] >= threshold
).astype(int)

threshold_consistent = bool(
    (df["predicted_late_risk"] == expected_predictions).all()
)

actual_delay_valid = bool(
    (df["actual_days_late"] >= 0).all()
)

predicted_delay_valid = bool(
    (df["predicted_days_late"] >= 0).all()
)

# Dataset and order-level descriptive checks.
row_count_matches = len(df) == expected_rows
order_count_matches = df["order_id"].nunique() == expected_orders

group_sizes = df.groupby("order_id").size()

report = {
    "prediction_granularity": "item_level",
    "model_version": metadata["model_version"],
    "classification_threshold": threshold,
    "prediction_rows": int(len(df)),
    "expected_test_rows": expected_rows,
    "row_count_matches_metrics": bool(row_count_matches),
    "unique_test_orders": int(df["order_id"].nunique()),
    "expected_test_orders": expected_orders,
    "order_count_matches_metrics": bool(order_count_matches),
    "orders_with_multiple_items": int((group_sizes > 1).sum()),
    "missing_values": missing_values,
    "probabilities_within_valid_range": probability_valid,
    "actual_classes_are_binary": actual_class_valid,
    "predicted_classes_are_binary": predicted_class_valid,
    "predictions_match_classification_threshold": threshold_consistent,
    "actual_delays_are_non_negative": actual_delay_valid,
    "predicted_delays_are_non_negative": predicted_delay_valid,
}

checks = [
    row_count_matches,
    order_count_matches,
    probability_valid,
    actual_class_valid,
    predicted_class_valid,
    threshold_consistent,
    actual_delay_valid,
    predicted_delay_valid,
    missing_values == 0,
]

report["overall_audit_passed"] = bool(all(checks))

output_path = REPORT_DIR / "prediction_consistency_audit.json"

output_path.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8",
)

print("SUPPLYPRESCRIPT ITEM-LEVEL PREDICTION AUDIT")
print(json.dumps(report, indent=4))
print(f"\nReport saved: {output_path}")
