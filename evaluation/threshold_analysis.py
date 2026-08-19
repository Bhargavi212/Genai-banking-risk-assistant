from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


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

THRESHOLDS = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    categorical_features = [
        "txn_type",
        "location",
        "device_type",
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
                ["amount"],
            ),
        ]
    )

    model = make_pipeline(
        preprocessor,
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        predictions = (
            probabilities >= threshold
        ).astype(int)

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

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nThreshold Analysis")
    print("------------------")
    print(results_df.to_string(index=False))

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\nBest threshold by F1:")
    print(
        f"Threshold: {best_row['threshold']:.2f}"
    )
    print(
        f"Precision: {best_row['precision']:.4f}"
    )
    print(
        f"Recall: {best_row['recall']:.4f}"
    )
    print(
        f"F1: {best_row['f1']:.4f}"
    )

    results_df.to_csv(
        RESULTS_DIR / "threshold_analysis.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
