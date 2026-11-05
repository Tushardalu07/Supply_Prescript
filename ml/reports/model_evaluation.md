# SupplyPrescript — ML Model Evaluation Report

## 1. Project Overview

**Project:** SupplyPrescript  
**Model Version:** xgboost-v2  
**Prediction Granularity:** Item-level  
**Model Type:** XGBoost Classifier and Regressor  
**Dataset:** DataCoSupplyChainDataset.csv

The machine learning module predicts shipment late-delivery risk and estimates the expected number of days late for individual shipment items.

## 2. Dataset and Preprocessing

- Original dataset: 180,519 rows and 53 columns.
- Cancelled shipment rows excluded: 7,754.
- Training scope: Non-cancelled shipments.
- Cleaned dataset: 172,765 rows.
- Unique eligible orders: 62,897.

## 3. Data Splitting Strategy

An order-aware chronological 70:10:20 split was used.

| Split | Orders | Rows |
|---|---:|---:|
| Training | 44,027 | 132,175 |
| Validation | 6,290 | 18,807 |
| Testing | 12,580 | 21,783 |
| Total | 62,897 | 172,765 |

No order overlap was detected between the splits. Chronological ordering was validated.

## 4. Model Architecture

Two XGBoost models were developed:

- XGBoost Classifier: Predicts late-delivery risk.
- XGBoost Regressor: Estimates expected days late.

The models use 16 selected input features covering shipment type, product information, customer segment, market, discounts, quantity, shipping mode and order calendar information.

## 5. Classification Evaluation

The classification threshold was selected using maximum F1-score on validation data.

**Selected threshold:** 0.35

| Metric | Test Result |
|---|---:|
| Accuracy | 57.999% |
| Precision | 57.894% |
| Recall | 98.062% |
| F1-score | 72.806% |
| ROC-AUC | 74.429% |

### Confusion Matrix

| Actual / Predicted | On-time | Late |
|---|---:|---:|
| On-time | 387 | 8,907 |
| Late | 242 | 12,247 |

The classifier achieves high recall for late-risk shipments but produces a substantial number of false positives. On-time classification recall is approximately 4.16%.

## 6. Regression Evaluation

| Metric | Result |
|---|---:|
| Mean Absolute Error | 0.702 days |
| Root Mean Squared Error | 0.891 days |
| Mean Actual Delay | 0.927 days |
| Mean Predicted Delay | 0.930 days |

The regression model's mean prediction is close to the mean actual delay. Individual prediction errors should still be considered.

## 7. Prediction Validation

The item-level prediction audit verified:

- 21,783 prediction rows.
- 12,580 unique test orders.
- Zero missing values.
- All probabilities within the 0–1 range.
- Binary classification outputs restricted to 0 and 1.
- Predictions consistent with the selected classification threshold.
- Actual and predicted delay values non-negative.

Overall item-level prediction audit: PASSED.

## 8. Automated Testing

The prediction module passed all 16 automated tests.

Tests cover model version, prediction output, probability validation, feature validation, missing and extra inputs, classification threshold, binary late-risk output and risk-category boundaries.

**Test result:** 16 passed, 0 failed.

## 9. Limitations

1. Classification precision is relatively low, resulting in many false-positive late-risk predictions.
2. The classifier has substantially lower recall for on-time shipments than for late shipments.
3. Regression predictions have measurable individual-level errors.
4. Historical dataset patterns may not represent future shipment conditions.
5. Model performance should be monitored when new shipment data becomes available.

## 10. Conclusion

The SupplyPrescript ML module successfully implements item-level shipment delay classification and regression using XGBoost.

The chronological data split, order-level target consistency checks, item-level prediction audit and automated tests provide a structured validation process.

The model is technically functional and ready for further controlled evaluation. Its classification trade-offs and regression errors should be considered before operational deployment.
