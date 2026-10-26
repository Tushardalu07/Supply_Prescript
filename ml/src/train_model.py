
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

DATE_COL = "order date (DateOrders)"
LABEL_COL = "Late_delivery_risk"
ACTUAL_DAYS_COL = "Days for shipping (real)"
SCHEDULED_DAYS_COL = "Days for shipment (scheduled)"

RANDOM_STATE = 42
TEST_FRACTION = 0.20

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# ---------------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
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
        DATE_COL,
        LABEL_COL,
        ACTUAL_DAYS_COL,
        SCHEDULED_DAYS_COL,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
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

    # Remove records that cannot support training.
    df = df.dropna(
        subset=[
            DATE_COL,
            LABEL_COL,
            ACTUAL_DAYS_COL,
            SCHEDULED_DAYS_COL,
        ]
    ).copy()

    # Retain valid binary labels only.
    df = df[df[LABEL_COL].isin([0, 1])].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

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

    print(f"Rows after cleaning: {len(df):,}")
    print("\nClass distribution:")
    print(df[LABEL_COL].value_counts().sort_index())

    return df


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
    df = load_data()

    # Features and targets are created from the SAME sorted dataframe.
    X, y_class, y_days, numeric_cols, categorical_cols = (
        select_features(df)
    )

    # Chronological 80/20 split: older records train the models;
    # newer records are held out for evaluation.
    split_index = int(len(df) * (1 - TEST_FRACTION))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    yc_train = y_class.iloc[:split_index].copy()
    yc_test = y_class.iloc[split_index:].copy()

    yd_train = y_days.iloc[:split_index].copy()
    yd_test = y_days.iloc[split_index:].copy()

    if yc_train.nunique() < 2:
        raise ValueError(
            "The training portion has only one class. "
            "Check label distribution or the chronological split."
        )

    print("\n" + "=" * 60)
    print("SUPPLYPRESCRIPT - XGBOOST MODEL TRAINING")
    print("=" * 60)
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")
    print(f"Training period: {df[DATE_COL].iloc[0]} "
          f"to {df[DATE_COL].iloc[split_index - 1]}")
    print(f"Test period:     {df[DATE_COL].iloc[split_index]} "
          f"to {df[DATE_COL].iloc[-1]}")

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

    class_predictions = classifier.predict(X_test).astype(int)
    class_probabilities = classifier.predict_proba(X_test)[:, 1]

    classifier_metrics = {
        "accuracy": float(
            accuracy_score(yc_test, class_predictions)
        ),
        "precision": float(
            precision_score(
                yc_test,
                class_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                yc_test,
                class_predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                yc_test,
                class_predictions,
                zero_division=0,
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
            target_names=["On-time/low-risk label", "Late-risk label"],
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
        if metric == "classification_report":
            continue
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

    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    print("\nModels saved:")
    print(f"  Classifier: {classifier_path}")
    print(f"  Regressor:  {regressor_path}")

    # -----------------------------------------------------
    # 7. SAVE METRICS AND TEST PREDICTIONS
    # -----------------------------------------------------

    metrics = {
        "project": "SupplyPrescript",
        "model_version": "xgboost-v1",
        "dataset": DATA_PATH.name,
        "split_strategy": "chronological",
        "train_fraction": 1 - TEST_FRACTION,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": list(X.columns),
        "classification": classifier_metrics,
        "regression": regression_metrics,
    }

    metrics_path = REPORT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    predictions = pd.DataFrame(
        {
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

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
