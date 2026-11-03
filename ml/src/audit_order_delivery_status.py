
from pathlib import Path
import pandas as pd
import json

ML_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ML_DIR / "data/raw/DataCoSupplyChainDataset.csv"
REPORT_DIR = ML_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

df.columns = df.columns.astype(str).str.strip()

required = ["Order Id", "Delivery Status"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()

order_status = df.groupby("Order Id")["Delivery Status"].agg(
    status_count="nunique",
    statuses=lambda x: sorted(x.astype(str).unique().tolist()),
    rows="size"
)

mixed_orders = order_status[order_status["status_count"] > 1]
cancelled_orders = order_status[
    order_status["statuses"].apply(
        lambda values: "Shipping canceled" in values
    )
]

report = {
    "total_orders": int(len(order_status)),
    "orders_with_multiple_statuses": int(len(mixed_orders)),
    "orders_with_cancelled_status": int(len(cancelled_orders)),
    "orders_with_only_cancelled_status": int(
        sum(
            values == ["Shipping canceled"]
            for values in order_status["statuses"]
        )
    ),
    "orders_with_mixed_cancelled_status": int(
        sum(
            "Shipping canceled" in values and len(values) > 1
            for values in order_status["statuses"]
        )
    )
}

output = REPORT_DIR / "order_delivery_status_audit.json"

with open(output, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("ORDER DELIVERY STATUS AUDIT")
print(json.dumps(report, indent=4))
print("\nReport saved:", output)

if len(mixed_orders):
    print("\nSample mixed-status orders:")
    print(mixed_orders.head(10).to_string())

