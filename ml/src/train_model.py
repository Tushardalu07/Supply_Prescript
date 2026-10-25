
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error,
    confusion_matrix,
)
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

ARTIFACTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

DATE = "order date (DateOrders)"
LABEL = "Late_delivery_risk"
REAL_DAYS = "Days for shipping (real)"
SCHEDULED_DAYS = "Days for shipment (scheduled)"


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA, encoding_errors="replace")
    df.columns = df.columns.str.strip()

    required = [DATE, LABEL, REAL_DAYS, SCHEDULED_DAYS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    for col in [LABEL, REAL_DAYS, SCHEDULED_DAYS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=required)
    df = df[df[LABEL].isin([0, 1])]
    df = df[(df[REAL_DAYS] >= 0) & (df[SCHEDULED_DAYS] >= 0)]

    # Features known at order time only.
    df["order_month"] = df[DATE].dt.month
    df["order_dayofweek"] = df[DATE].dt.dayofweek

    # Duration target: positive days late, otherwise zero.
    df["days_late_target"] = (
        df[REAL_DAYS] - df[SCHEDULED_DAYS]
    ).clip(lower=0)

    # Remove targets, identifiers, sensitive fields and post-outcome data.
    excluded = {
        LABEL, REAL_DAYS, SCHEDULED_DAYS,
        "days_late_target", DATE,
        "shipping date (DateOrders)",
        "Delivery Status", "Order Status",
        "Customer Email", "Customer Password",
        "Customer Fname", "Customer Lname",
        "Customer Street", "Customer Id",
        "Order Customer Id", "Order Id", "Order Item Id",
        "Product Description", "Product Image",
        "Customer Zipcode", "Order Zipcode",
        "Latitude", "Longitude",
    }

    # Exclude additional identifiers or potentially post-outcome columns.
    for col in df.columns:
        lower = col.lower()
        if (
            "shipping date" in lower
            or "actual delivery" in lower
            or "delivery status" in lower
            or lower.endswith("password")
        ):
            excluded.add(col)

    feature_cols = [c for c in df.columns if c not in excluded]
    X = df[feature_cols].copy()
    y_class = df[LABEL].astype(int)
    y_days = df["days_late_target"].astype(float)

    # Chronological split: older orders train, newer orders test.
    df = df.sort_values(DATE).reset_index(drop=True)
    X = X.loc[df.index].reset_index(drop=True)
    y_class = y_class.loc[df.index].reset_index(drop=True)
    y_days = y_days.loc[df.index].reset_index(drop=True)

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    yc_train, yc_test = y_class.iloc[:split], y_class.iloc[split:]
    yd_train, yd_test = y_days.iloc[:split], y_days.iloc[split:]

    # Build preprocessing from training data only.
    cat_cols = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    def make_pipeline(model):
        prep = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ])
        return Pipeline([("prep", prep), ("model", model)])

    classifier = make_pipeline(XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    ))
    regressor = make_pipeline(XGBRegressor(
        n_estimators=250, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    ))

    print(f"Training rows: {len(X_train):,}; test rows: {len(X_test):,}")
    print("Training delay-risk classifier...")
    classifier.fit(X_train, yc_train)

    print("Training days-late regressor...")
    regressor.fit(X_train, yd_train)

    probabilities = classifier.predict_proba(X_test)[:, 1]
    class_pred = (probabilities >= 0.5).astype(int)
    days_pred = regressor.predict(X_test).clip(0)

    metrics = {
        "classification": {
            "accuracy": float(accuracy_score(yc_test, class_pred)),
            "precision": float(precision_score(
                yc_test, class_pred, zero_division=0
            )),
            "recall": float(recall_score(
                yc_test, class_pred, zero_division=0
            )),
            "f1": float(f1_score(
                yc_test, class_pred, zero_division=0
            )),
            "confusion_matrix": confusion_matrix(
                yc_test, class_pred
            ).tolist(),
        },
        "days_late_regression": {
            "mae_days": float(mean_absolute_error(yd_test, days_pred)),
            "rmse_days": float(
                mean_squared_error(yd_test, days_pred) ** 0.5
            ),
        },
        "metadata": {
            "dataset_rows_after_cleaning": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": feature_cols,
            "split": "chronological 80/20",
        },
    }

    if yc_test.nunique() == 2:
        metrics["classification"]["roc_auc"] = float(
            roc_auc_score(yc_test, probabilities)
        )

    joblib.dump(classifier, ARTIFACTS / "delay_classifier.joblib")
    joblib.dump(regressor, ARTIFACTS / "days_late_regressor.joblib")

    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    pd.DataFrame({
        "actual_late_risk": yc_test,
        "predicted_late_probability": probabilities,
        "predicted_late_risk": class_pred,
        "actual_days_late": yd_test,
        "predicted_days_late": days_pred,
    }).to_csv(REPORTS / "test_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"\nModels saved in: {ARTIFACTS}")
    print(f"Reports saved in: {REPORTS}")


if __name__ == "__main__":
    main()
