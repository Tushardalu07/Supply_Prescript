
from pathlib import Path
import json
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ML_DIR / "data/raw/DataCoSupplyChainDataset.csv"
REPORT_DIR = ML_DIR / "reports"

DATE_COL = "order date (DateOrders)"
ORDER_COL = "Order Id"
TEST_FRACTION = 0.20


def main():
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")

    df.columns = df.columns.astype(str).str.strip()

    required = [DATE_COL, ORDER_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL], errors="coerce"
    )

    df = df.dropna(subset=[DATE_COL, ORDER_COL]).copy()
    df[ORDER_COL] = df[ORDER_COL].astype(str)

    # Keep each order together, using its earliest order date.
    orders = (
        df.groupby(ORDER_COL, as_index=False)
        [DATE_COL].min()
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    split_index = int(len(orders) * (1 - TEST_FRACTION))

    train_orders = set(
        orders.iloc[:split_index][ORDER_COL]
    )
    test_orders = set(
        orders.iloc[split_index:][ORDER_COL]
    )

    overlap = train_orders & test_orders

    train_end = orders.iloc[split_index - 1][DATE_COL]
    test_start = orders.iloc[split_index][DATE_COL]

    result = {
        "total_orders": int(len(orders)),
        "train_orders": len(train_orders),
        "test_orders": len(test_orders),
        "overlapping_orders": len(overlap),
        "training_end_date": str(train_end),
        "testing_start_date": str(test_start),
        "chronological_order": bool(train_end <= test_start),
        "order_overlap_free": len(overlap) == 0,
        "status": (
            "INITIAL_CHECKS_PASSED"
            if len(overlap) == 0 and train_end <= test_start
            else "REVIEW_REQUIRED"
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "order_split_validation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("\nSUPPLYPRESCRIPT ORDER SPLIT VALIDATION")
    print("-" * 45)

    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
