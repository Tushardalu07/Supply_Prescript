
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)

from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# 1. PATHS AND CONFIGURATION
# ---------------------------------------------------------

ML_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ML_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"
ARTIFACT_DIR = ML_DIR / "artifacts"
REPORT_DIR = ML_DIR / "reports"

ORDER_COL = "Order Id"
DATE_COL = "order date (DateOrders)"
LABEL_COL = "Late_delivery_risk"
ACTUAL_DAYS_COL = "Days for shipping (real)"
SCHEDULED_DAYS_COL = "Days for shipment (scheduled)"
STATUS_COL = "Delivery Status"

RANDOM_STATE = 42
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.10

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# ---------------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\\n"
            "Place DataCoSupplyChainDataset.csv inside "
            "ml/data/raw/."
        )

    print(f"Loading dataset: {DATA_PATH}")

    # DataCo dataset files are commonly UTF-8 or Latin-1 encoded.
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Normalize accidental spaces around column names.
    df.columns = df.columns.astype(str).str.strip()

    required = [
        ORDER_COL,
        DATE_COL,
        LABEL_COL,
        ACTUAL_DAYS_COL,
        SCHEDULED_DAYS_COL,
        STATUS_COL,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns):,}")

    # Parse the order date.
    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce"
    )

    # Convert targets to numeric values.
    for col in [
        LABEL_COL,
        ACTUAL_DAYS_COL,
        SCHEDULED_DAYS_COL,
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize delivery status before filtering.
    df[STATUS_COL] = (
        df[STATUS_COL].astype("string").str.strip()
    )

    # Remove records that cannot support training.
    df = df.dropna(
        subset=[
            ORDER_COL,
            DATE_COL,
            LABEL_COL,
            ACTUAL_DAYS_COL,
            SCHEDULED_DAYS_COL,
            STATUS_COL,
        ]
    ).copy()

    # Retain valid binary labels only.
    df = df[df[LABEL_COL].isin([0, 1])].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Exclude cancelled shipments from both model populations.
    cancelled_mask = (
        df[STATUS_COL].str.casefold() == "shipping canceled"
    )
    cancelled_rows_excluded = int(cancelled_mask.sum())
    df = df.loc[~cancelled_mask].copy()

    # Remove exact duplicate rows.
    df = df.drop_duplicates().copy()

    # Sort BEFORE generating X and y so that every row stays aligned.
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            "Too few valid rows remain after data cleaning."
        )

    # Retrospective regression target:
    # positive number of days beyond the scheduled shipping duration.
    df["days_late_target"] = (
        df[ACTUAL_DAYS_COL] - df[SCHEDULED_DAYS_COL]
    ).clip(lower=0)

    # Order-time calendar features.
    df["order_month"] = df[DATE_COL].dt.month
    df["order_dayofweek"] = df[DATE_COL].dt.dayofweek

    print(f"Cancelled rows excluded: {cancelled_rows_excluded:,}")
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Eligible unique orders: {df[ORDER_COL].nunique():,}")
    print("\\nClass distribution:")
    print(df[LABEL_COL].value_counts().sort_index())

    return df, cancelled_rows_excluded


# ---------------------------------------------------------
# 3. SELECT APPROVED FEATURES
# ---------------------------------------------------------

def select_features(df):
    # Only use features intended to be available at order time.
    # Do NOT use actual shipping duration, delivery dates,
    # actual delivery status, or the target label as features.
    allowed_features = [
        "Type",
        "Category Id",
        "Category Name",
        "Customer Segment",
        "Department Id",
        "Department Name",
        "Market",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Quantity",
        "ProductCategory Id",
        "Product Category Id",
        "Product Price",
        "Shipping Mode",
        "order_month",
        "order_dayofweek",
    ]

    # Use only columns actually present in the CSV.
    feature_cols = []
    for col in allowed_features:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)

    missing_features = [
        col for col in allowed_features
        if col not in df.columns
    ]

    if not feature_cols:
        raise ValueError(
            "No approved features were found. "
            "Check the dataset column names."
        )

    X = df[feature_cols].copy()
    y_class = df[LABEL_COL].astype(int).copy()
    y_days = df["days_late_target"].astype(float).copy()

    # Normalize numeric features and convert infinite values to NaN.
    numeric_cols = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = [
        col for col in feature_cols
        if col not in numeric_cols
    ]

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    for col in categorical_cols:
        X[col] = X[col].astype("string").astype(object)
        X[col] = X[col].where(X[col].notna(), np.nan)

    print("\nSelected features:")
    for col in feature_cols:
        print(f"  - {col}")

    if missing_features:
        print("\nOptional features not found in this CSV:")
        for col in missing_features:
            print(f"  - {col}")

    print(f"\nNumeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    return X, y_class, y_days, numeric_cols, categorical_cols


# ---------------------------------------------------------
# 4. BUILD PREPROCESSING PIPELINE
# ---------------------------------------------------------

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ---------------------------------------------------------
# 5. TRAIN AND EVALUATE MODELS
# ---------------------------------------------------------

def main():
    df, cancelled_rows_excluded = load_data()

    # Features and targets are created from the SAME sorted dataframe.
    X, y_class, y_days, numeric_cols, categorical_cols = (
        select_features(df)
    )

    # -----------------------------------------------------
    # ORDER-AWARE CHRONOLOGICAL 70/10/20 SPLIT
    # -----------------------------------------------------

    target_audit = df.groupby(ORDER_COL).agg(
        class_values=(LABEL_COL, "nunique"),
        days_late_values=("days_late_target", "nunique"),
    )
    inconsistent_class_orders = int(
        (target_audit["class_values"] > 1).sum()
    )
    inconsistent_regression_orders = int(
        (target_audit["days_late_values"] > 1).sum()
    )

    if inconsistent_class_orders or inconsistent_regression_orders:
        raise ValueError(
            "Inconsistent targets found within orders: "
            f"classification={inconsistent_class_orders}, "
            f"regression={inconsistent_regression_orders}."
        )

    # Sort unique orders by their first order date. All item rows
    # belonging to an order are kept in the same partition.
    order_dates = (
        df.groupby(ORDER_COL)[DATE_COL]
        .min()
        .sort_values(kind="stable")
    )
    total_orders = len(order_dates)

    test_start_index = int(total_orders * (1 - TEST_FRACTION))
    validation_start_index = int(
        total_orders * (1 - TEST_FRACTION - VALIDATION_FRACTION)
    )

    if not (
        0 < validation_start_index < test_start_index < total_orders
    ):
        raise ValueError("Invalid chronological split boundaries.")

    train_order_ids = set(order_dates.index[:validation_start_index])
    validation_order_ids = set(
        order_dates.index[validation_start_index:test_start_index]
    )
    test_order_ids = set(order_dates.index[test_start_index:])

    train_mask = df[ORDER_COL].isin(train_order_ids)
    validation_mask = df[ORDER_COL].isin(validation_order_ids)
    test_mask = df[ORDER_COL].isin(test_order_ids)

    assignment_count = (
        train_mask.astype(int)
        + validation_mask.astype(int)
        + test_mask.astype(int)
    )
    if not assignment_count.eq(1).all():
        raise AssertionError(
            "Rows are missing from or duplicated across partitions."
        )

    overlap_count = (
        len(train_order_ids.intersection(validation_order_ids))
        + len(train_order_ids.intersection(test_order_ids))
        + len(validation_order_ids.intersection(test_order_ids))
    )
    if overlap_count:
        raise AssertionError("Order leakage detected.")

    X_train = X.loc[train_mask].copy()
    X_validation = X.loc[validation_mask].copy()
    X_test = X.loc[test_mask].copy()

    yc_train = y_class.loc[train_mask].copy()
    yc_validation = y_class.loc[validation_mask].copy()
    yc_test = y_class.loc[test_mask].copy()

    yd_train = y_days.loc[train_mask].copy()
    yd_validation = y_days.loc[validation_mask].copy()
    yd_test = y_days.loc[test_mask].copy()

    test_order_ids_by_row = df.loc[test_mask, ORDER_COL].copy()

    training_start = df.loc[train_mask, DATE_COL].min()
    training_end = df.loc[train_mask, DATE_COL].max()
    validation_start = df.loc[validation_mask, DATE_COL].min()
    validation_end = df.loc[validation_mask, DATE_COL].max()
    testing_start = df.loc[test_mask, DATE_COL].min()
    testing_end = df.loc[test_mask, DATE_COL].max()

    if not (
        training_end <= validation_start
        and validation_end <= testing_start
    ):
        raise AssertionError(
            "Chronological partition order is invalid."
        )

    if yc_train.nunique() < 2:
        raise ValueError("Training partition contains only one class.")
    if yc_validation.nunique() < 2:
        raise ValueError("Validation partition contains only one class.")
    if yc_test.nunique() < 2:
        print(
            "WARNING: Test partition contains only one class; "
            "ROC-AUC will be unavailable."
        )

    print("\n" + "=" * 60)
    print("SUPPLYPRESCRIPT - XGBOOST MODEL TRAINING")
    print("=" * 60)
    print("ORDER-AWARE CHRONOLOGICAL SPLIT VALIDATION")
    print(f"Total unique orders:      {total_orders:,}")
    print(f"Training orders:          {len(train_order_ids):,}")
    print(f"Validation orders:        {len(validation_order_ids):,}")
    print(f"Testing orders:           {len(test_order_ids):,}")
    print(f"Training rows:            {len(X_train):,}")
    print(f"Validation rows:          {len(X_validation):,}")
    print(f"Testing rows:             {len(X_test):,}")
    print(f"Training period:          {training_start} to {training_end}")
    print(f"Validation period:        {validation_start} to {validation_end}")
    print(f"Testing period:           {testing_start} to {testing_end}")
    print("Order leakage check:      PASSED")
    print("Chronological check:      PASSED")
    print("Target consistency check: PASSED")

    # -----------------------------------------------------
    # 5A. DELAY-RISK CLASSIFIER
    # -----------------------------------------------------

    print("\nTraining delay-risk classifier...")

    classifier_preprocessor = build_preprocessor(
        numeric_cols,
        categorical_cols,
    )

    classifier = Pipeline(
        steps=[
            ("preprocessor", classifier_preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    classifier.fit(X_train, yc_train)

    # Select the binary decision threshold using validation data only.
    validation_probabilities = classifier.predict_proba(
        X_validation
    )[:, 1]

    threshold_candidates = np.round(
        np.arange(0.20, 0.801, 0.01),
        2,
    )
    validation_f1_scores = []

    for threshold in threshold_candidates:
        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)
        validation_f1_scores.append(
            f1_score(
                yc_validation,
                validation_predictions,
                zero_division=0,
            )
        )

    best_validation_f1 = float(max(validation_f1_scores))
    best_indices = [
        i for i, score in enumerate(validation_f1_scores)
        if np.isclose(score, best_validation_f1)
    ]
    # If F1 ties, prefer the higher threshold.
    best_index = max(
        best_indices,
        key=lambda i: threshold_candidates[i],
    )
    classification_threshold = float(
        threshold_candidates[best_index]
    )

    print("\nVALIDATION THRESHOLD SELECTION")
    print("-" * 40)
    print("Selection metric: F1-score")
    print(f"Selected threshold: {classification_threshold:.2f}")
    print(f"Best validation F1: {best_validation_f1:.4f}")

    # Evaluate once on the untouched final test partition.
    class_probabilities = classifier.predict_proba(X_test)[:, 1]
    class_predictions = (
        class_probabilities >= classification_threshold
    ).astype(int)

    classifier_metrics = {
        "decision_threshold": classification_threshold,
        "threshold_selection": "validation_maximum_f1",
        "validation_best_f1": best_validation_f1,
        "accuracy": float(
            accuracy_score(yc_test, class_predictions)
        ),
        "precision": float(
            precision_score(
                yc_test, class_predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                yc_test, class_predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                yc_test, class_predictions, zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            yc_test,
            class_predictions,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            yc_test,
            class_predictions,
            labels=[0, 1],
            target_names=[
                "On-time/low-risk label",
                "Late-risk label",
            ],
            output_dict=True,
            zero_division=0,
        ),
    }

    if yc_test.nunique() == 2:
        classifier_metrics["roc_auc"] = float(
            roc_auc_score(yc_test, class_probabilities)
        )
    else:
        classifier_metrics["roc_auc"] = None

    print("\nCLASSIFICATION RESULTS")
    print("-" * 40)
    for metric, value in classifier_metrics.items():
        if metric != "classification_report":
            print(f"{metric}: {value}")

    # -----------------------------------------------------
    # 5B. DAYS-LATE REGRESSOR
    # -----------------------------------------------------

    print("\nTraining days-late regressor...")

    regressor_preprocessor = build_preprocessor(
        numeric_cols,
        categorical_cols,
    )

    regressor = Pipeline(
        steps=[
            ("preprocessor", regressor_preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=250,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    regressor.fit(X_train, yd_train)

    days_predictions = regressor.predict(X_test)
    days_predictions = np.clip(days_predictions, 0, None)

    regression_metrics = {
        "mae_days": float(
            mean_absolute_error(yd_test, days_predictions)
        ),
        "rmse_days": float(
            np.sqrt(mean_squared_error(yd_test, days_predictions))
        ),
        "mean_actual_days_late": float(yd_test.mean()),
        "mean_predicted_days_late": float(
            np.mean(days_predictions)
        ),
    }

    print("\nREGRESSION RESULTS")
    print("-" * 40)
    for metric, value in regression_metrics.items():
        print(f"{metric}: {value:.4f}")

    # -----------------------------------------------------
    # 6. SAVE MODELS
    # -----------------------------------------------------

    classifier_path = ARTIFACT_DIR / "delay_classifier.joblib"
    regressor_path = ARTIFACT_DIR / "days_late_regressor.joblib"
    metadata_path = ARTIFACT_DIR / "model_metadata.json"

    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    model_metadata = {
        "model_version": "xgboost-v2",
        "classification_threshold": classification_threshold,
        "threshold_selection": "validation_maximum_f1",
        "features": list(X.columns),
        "split_strategy": "order_aware_chronological_70_10_20",
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(model_metadata, file, indent=2)

    print("\nModels and metadata saved:")
    print(f"  Classifier: {classifier_path}")
    print(f"  Regressor:  {regressor_path}")
    print(f"  Metadata:   {metadata_path}")

    # -----------------------------------------------------
    # 7. SAVE METRICS AND TEST PREDICTIONS
    # -----------------------------------------------------

    metrics = {
        "project": "SupplyPrescript",
        "model_version": "xgboost-v2",
        "dataset": DATA_PATH.name,
        "training_scope": "non_cancelled_shipments_only",
        "cancelled_rows_excluded": int(cancelled_rows_excluded),
        "split_strategy": "order_aware_chronological_70_10_20",
        "train_fraction": 0.70,
        "validation_fraction": VALIDATION_FRACTION,
        "test_fraction": TEST_FRACTION,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "total_orders": int(total_orders),
        "train_orders": int(len(train_order_ids)),
        "validation_orders": int(len(validation_order_ids)),
        "test_orders": int(len(test_order_ids)),
        "overlapping_orders": int(overlap_count),
        "training_start_date": str(training_start),
        "training_end_date": str(training_end),
        "validation_start_date": str(validation_start),
        "validation_end_date": str(validation_end),
        "testing_start_date": str(testing_start),
        "testing_end_date": str(testing_end),
        "chronological_order": True,
        "threshold_selection": {
            "method": "maximum_validation_f1",
            "selected_threshold": classification_threshold,
            "best_validation_f1": best_validation_f1,
            "candidate_min": float(threshold_candidates.min()),
            "candidate_max": float(threshold_candidates.max()),
            "candidate_step": 0.01,
            "tie_break": "higher_threshold",
        },
        "order_target_consistency": {
            "inconsistent_classification_orders": inconsistent_class_orders,
            "inconsistent_regression_orders": inconsistent_regression_orders,
        },
        "features": list(X.columns),
        "classification": classifier_metrics,
        "regression": regression_metrics,
    }

    metrics_path = REPORT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    split_audit = {
        "split_strategy": "order_aware_chronological_70_10_20",
        "total_orders": int(total_orders),
        "train_orders": int(len(train_order_ids)),
        "validation_orders": int(len(validation_order_ids)),
        "test_orders": int(len(test_order_ids)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "overlapping_orders": int(overlap_count),
        "training_start": str(training_start),
        "training_end": str(training_end),
        "validation_start": str(validation_start),
        "validation_end": str(validation_end),
        "testing_start": str(testing_start),
        "testing_end": str(testing_end),
        "chronological_order": True,
        "order_target_consistency": True,
    }
    split_audit_path = REPORT_DIR / "order_split_validation.json"
    with open(split_audit_path, "w", encoding="utf-8") as file:
        json.dump(split_audit, file, indent=2)

    predictions = pd.DataFrame(
        {
            "order_id": test_order_ids_by_row.to_numpy(),
            "actual_late_risk": yc_test.to_numpy(),
            "predicted_late_risk": class_predictions,
            "delay_probability": class_probabilities,
            "actual_days_late": yd_test.to_numpy(),
            "predicted_days_late": days_predictions,
        }
    )

    predictions_path = REPORT_DIR / "test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    print("\nReports saved:")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Split audit: {split_audit_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
