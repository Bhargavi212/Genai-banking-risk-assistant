"""
Train and evaluate a Random Forest fraud-classification pipeline.

The script:
1. Loads transaction data.
2. Preprocesses categorical and numerical features.
3. Uses a stratified train/test split.
4. Trains a Random Forest classifier.
5. Evaluates multiple classification metrics.
6. Logs results to MLflow.
7. Saves the complete preprocessing + model pipeline.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

DATA_PATH = PROJECT_ROOT / "Dataset" / "transactions.csv"
MODEL_PATH = PROJECT_ROOT / "model.pkl"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 200


# ---------------------------------------------------------
# MLflow
# ---------------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud_risk")


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "amount",
    "txn_type",
    "location",
    "device_type",
]

TARGET = "is_fraud"

X = df[FEATURES]
y = df[TARGET]


print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())
print("\nClass proportions:")
print(y.value_counts(normalize=True))


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

categorical_features = [
    "txn_type",
    "location",
    "device_type",
]

numeric_features = [
    "amount",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ]
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

classifier = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
)

model = make_pipeline(
    preprocessor,
    classifier,
)


# ---------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

conf_matrix = confusion_matrix(
    y_test,
    predictions,
)


print("\nEvaluation Metrics")
print("-------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)


# ---------------------------------------------------------
# MLflow Logging
# ---------------------------------------------------------

with mlflow.start_run() as run:

    mlflow.log_param(
        "model_type",
        "RandomForestClassifier",
    )

    mlflow.log_param(
        "n_estimators",
        N_ESTIMATORS,
    )

    mlflow.log_param(
        "test_size",
        TEST_SIZE,
    )

    mlflow.log_param(
        "class_weight",
        "balanced",
    )

    mlflow.log_metric(
        "accuracy",
        accuracy,
    )

    mlflow.log_metric(
        "precision",
        precision,
    )

    mlflow.log_metric(
        "recall",
        recall,
    )

    mlflow.log_metric(
        "f1",
        f1,
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc,
    )


    print(
        f"\nMLflow run logged: "
        f"{run.info.run_id}"
    )


# ---------------------------------------------------------
# Save Model Pipeline
# ---------------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH,
)

print(
    f"\nSaved model pipeline to: "
    f"{MODEL_PATH}"
)
