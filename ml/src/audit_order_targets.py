
from pathlib import Path
import json
import pandas as pd


# 1. PATHS AND CONFIGURATION

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "ml" / "data" / "raw"
    / "DataCoSupplyChainDataset.csv"
)

REPORT_DIR = ROOT / "ml" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ORDER_COL = "Order Id"
DATE_COL = "order date (DateOrders)"
LABEL_COL = "Late_delivery_risk"
ACTUAL_COL = "Days for shipping (real)"
SCHEDULED_COL = "Days for shipment (scheduled)"


# 2. LOAD AND CLEAN DATA

def main():

    print("=" * 60)
    print("SUPPLYPRESCRIPT ORDER TARGET AUDIT")
    print("=" * 60)

    required_columns = [
        ORDER_COL,
        DATE_COL,
        LABEL_COL,
        ACTUAL_COL,
        SCHEDULED_COL,
    ]

    # Try UTF-8 first, then Latin-1.
    try:
        df = pd.read_csv(
            DATA_PATH,
            usecols=required_columns,
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        print("UTF-8 decoding failed. Retrying with latin-1...")

        df = pd.read_csv(
            DATA_PATH,
            usecols=required_columns,
            encoding="latin-1",
        )

    print(f"\nOriginal rows: {len(df):,}")

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    for col in [LABEL_COL, ACTUAL_COL, SCHEDULED_COL]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    df = df[df[LABEL_COL].isin([0, 1])].copy()

    df["days_late_target"] = (
        df[ACTUAL_COL] - df[SCHEDULED_COL]
    ).clip(lower=0)

    # 3. GROUP BY ORDER ID

    grouped = df.groupby(ORDER_COL).agg(
        rows=(ORDER_COL, "size"),
        class_values=(LABEL_COL, "nunique"),
        days_late_values=("days_late_target", "nunique"),
        earliest_order_date=(DATE_COL, "min"),
        latest_order_date=(DATE_COL, "max"),
    )

    inconsistent_class = grouped[
        grouped["class_values"] > 1
    ]

    inconsistent_regression = grouped[
        grouped["days_late_values"] > 1
    ]

    multiple_dates = grouped[
        grouped["earliest_order_date"]
        != grouped["latest_order_date"]
    ]

    # 4. CREATE AUDIT REPORT

    report = {
        "total_valid_rows": int(len(df)),
        "unique_orders": int(df[ORDER_COL].nunique()),

        "orders_with_multiple_rows": int(
            (grouped["rows"] > 1).sum()
        ),

        "orders_with_inconsistent_classification_target": int(
            len(inconsistent_class)
        ),

        "orders_with_inconsistent_regression_target": int(
            len(inconsistent_regression)
        ),

        "orders_with_multiple_order_dates": int(
            len(multiple_dates)
        ),

        "classification_target_values": sorted(
            df[LABEL_COL].unique().astype(int).tolist()
        ),

        "regression_target_definition": (
            "max(actual shipping days - scheduled shipping days, 0)"
        ),
    }

    report_path = REPORT_DIR / "order_target_audit.json"

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # 5. DISPLAY RESULTS

    print("\nAUDIT RESULTS")
    print("-" * 60)

    for key, value in report.items():
        print(f"{key}: {value}")

    print(f"\nReport saved: {report_path}")

    if (
        report["orders_with_inconsistent_classification_target"] == 0
        and report["orders_with_inconsistent_regression_target"] == 0
    ):
        print("\nTARGET CONSISTENCY CHECK PASSED.")
    else:
        print("\nWARNING: Inconsistent targets detected.")

    print("=" * 60)


if __name__ == "__main__":
    main()
