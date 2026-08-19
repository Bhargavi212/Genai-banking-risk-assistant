from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "Dataset" / "transactions.csv"
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "amount",
    "txn_type",
    "location",
    "device_type",
]

TARGET = "is_fraud"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def build_preprocessor():
    categorical_features = [
        "txn_type",
        "location",
        "device_type",
    ]

    numeric_features = [
        "amount",
    ]

    return ColumnTransformer(
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


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    pipeline = make_pipeline(
        build_preprocessor(),
        model,
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


def main():
    df = pd.read_csv(
        DATA_PATH
    )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"Evaluating: {name}")

        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="roc_auc",
        ascending=False,
    )

    print("\nModel Comparison")
    print("----------------")
    print(
        results_df.to_string(
            index=False
        )
    )

    csv_path = (
        RESULTS_DIR
        / "model_comparison.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    md_path = (
        RESULTS_DIR
        / "model_comparison.md"
    )

    with md_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "# Fraud Model Comparison\n\n"
        )

        file.write(
            "Models were evaluated using the same "
            "stratified 80/20 train-test split.\n\n"
        )

        file.write(
            "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n"
        )

        file.write(
            "|---|---:|---:|---:|---:|---:|\n"
        )

        for _, row in results_df.iterrows():
            file.write(
                f"| {row['model']} "
                f"| {row['accuracy']:.4f} "
                f"| {row['precision']:.4f} "
                f"| {row['recall']:.4f} "
                f"| {row['f1']:.4f} "
                f"| {row['roc_auc']:.4f} |\n"
            )

        file.write(
            "\n> Dataset is synthetically generated "
            "for research and demonstration purposes.\n"
        )

    print(
        f"\nResults saved to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
