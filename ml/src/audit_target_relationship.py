
from pathlib import Path
import json
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ML_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"
REPORT_DIR = ML_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Late_delivery_risk"
ACTUAL_COL = "Days for shipping (real)"
SCHEDULED_COL = "Days for shipment (scheduled)"
ORDER_COL = "Order Id"


def main():
    print("SUPPLYPRESCRIPT TARGET RELATIONSHIP AUDIT")

    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")

    df.columns = df.columns.astype(str).str.strip()

    required = [ORDER_COL, LABEL_COL, ACTUAL_COL, SCHEDULED_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in [LABEL_COL, ACTUAL_COL, SCHEDULED_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()
    df = df[df[LABEL_COL].isin([0, 1])].copy()

    df["days_late_target"] = (
        df[ACTUAL_COL] - df[SCHEDULED_COL]
    ).clip(lower=0)

    df["regression_indicates_delay"] = (
        df["days_late_target"] > 0
    ).astype(int)

    df["targets_agree"] = (
        df[LABEL_COL].astype(int)
        == df["regression_indicates_delay"]
    )

    total_rows = len(df)
    matching_rows = int(df["targets_agree"].sum())
    mismatching_rows = total_rows - matching_rows

    relationship = pd.crosstab(
        df[LABEL_COL].astype(int),
        df["regression_indicates_delay"],
        rownames=["Late_delivery_risk"],
        colnames=["days_late_target > 0"]
    )

    order_summary = df.groupby(ORDER_COL).agg(
        classification_values=(LABEL_COL, "nunique"),
        regression_values=("days_late_target", "nunique"),
        rows=("targets_agree", "size"),
        matching_rows=("targets_agree", "sum")
    )

    report = {
        "total_rows": int(total_rows),
        "unique_orders": int(df[ORDER_COL].nunique()),
        "matching_target_rows": matching_rows,
        "mismatching_target_rows": mismatching_rows,
        "target_agreement_percentage": round(
            matching_rows / total_rows * 100, 2
        ) if total_rows else None,
        "classification_target_values": sorted(
            df[LABEL_COL].astype(int).unique().tolist()
        ),
        "regression_target_min": float(
            df["days_late_target"].min()
        ),
        "regression_target_max": float(
            df["days_late_target"].max()
        ),
        "target_relationship": relationship.to_dict(),
        "orders_with_inconsistent_class": int(
            (order_summary["classification_values"] > 1).sum()
        ),
        "orders_with_inconsistent_delay": int(
            (order_summary["regression_values"] > 1).sum()
        ),
        "interpretation": (
            "Agreement is measured, not assumed. "
            "A mismatch does not automatically mean data leakage."
        )
    }

    output_path = REPORT_DIR / "target_relationship_audit.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
