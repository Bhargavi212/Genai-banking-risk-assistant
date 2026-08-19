from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "Dataset" / "transactions.csv"
MODEL_PATH = PROJECT_ROOT / "model.pkl"

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


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = joblib.load(MODEL_PATH)

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

    print("\nFraud Model Evaluation")
    print("----------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    metrics_path = RESULTS_DIR / "fraud_model_metrics.md"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("# Fraud Model Evaluation\n\n")
        file.write(
            "Evaluation performed on a stratified "
            "20% holdout split.\n\n"
        )

        file.write("| Metric | Score |\n")
        file.write("|---|---:|\n")
        file.write(f"| Accuracy | {accuracy:.4f} |\n")
        file.write(f"| Precision | {precision:.4f} |\n")
        file.write(f"| Recall | {recall:.4f} |\n")
        file.write(f"| F1-score | {f1:.4f} |\n")
        file.write(f"| ROC-AUC | {roc_auc:.4f} |\n")

        file.write(
            "\n> Dataset is synthetically generated "
            "for research and demonstration purposes.\n"
        )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
    )

    plt.title("Fraud Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
    )

    plt.title("Fraud Model ROC Curve")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "roc_curve.png",
        dpi=200,
    )
    plt.close()

    print(
        f"\nResults saved to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
