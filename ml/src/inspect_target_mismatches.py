
from pathlib import Path
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ML_DIR / "data/raw/DataCoSupplyChainDataset.csv"

try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

df.columns = df.columns.astype(str).str.strip()

for col in [
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Status"
]).copy()

df["days_late_target"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
).clip(lower=0)

df["regression_indicates_delay"] = (
    df["days_late_target"] > 0
).astype(int)

df["target_mismatch"] = (
    df["Late_delivery_risk"].astype(int)
    != df["regression_indicates_delay"]
)

print("\nDELIVERY STATUS DISTRIBUTION")
print(df["Delivery Status"].value_counts(dropna=False))

print("\nTARGET MISMATCHES BY DELIVERY STATUS")
print(
    df.groupby("Delivery Status")["target_mismatch"]
    .agg(["count", "sum"])
    .sort_values("sum", ascending=False)
)

print("\nREGRESSION TRAINING POPULATION")
eligible = df[df["Delivery Status"] != "Shipping canceled"].copy()

print("All rows:", len(df))
print("Cancelled rows:", len(df) - len(eligible))
print("Eligible regression rows:", len(eligible))
print("Eligible unique orders:", eligible["Order Id"].nunique())

print("\nTARGET AGREEMENT AFTER EXCLUDING CANCELLATIONS")
print(
    pd.crosstab(
        eligible["Late_delivery_risk"],
        eligible["regression_indicates_delay"]
    )
)
