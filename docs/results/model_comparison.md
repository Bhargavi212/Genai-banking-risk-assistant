# Fraud Model Comparison

Four classification models were evaluated using the same stratified
80/20 train-test split on the synthetic transaction dataset.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6700 | 0.5573 | 0.6948 | 0.6185 | 0.7229 |
| XGBoost | 0.6800 | 0.6000 | 0.5065 | 0.5493 | 0.7183 |
| Gradient Boosting | 0.6325 | 0.5259 | 0.4610 | 0.4913 | 0.6816 |
| Random Forest | 0.6275 | 0.5172 | 0.4870 | 0.5017 | 0.6630 |

## Interpretation

Logistic Regression achieved the highest ROC-AUC, recall, and F1-score
among the evaluated models.

XGBoost achieved the highest accuracy and precision but detected a
smaller proportion of fraud cases.

For this prototype, Logistic Regression is the strongest baseline when
fraud recall and overall discrimination are prioritized.

> These results are based on synthetically generated transaction data
> and should not be interpreted as production banking performance.
