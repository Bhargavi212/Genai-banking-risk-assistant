from pathlib import Path

import joblib
import pandas as pd


# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parents[2] / "model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Risk model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# Transaction Risk Scoring
# ---------------------------------------------------------

def score_transaction(txn):
    """
    Predict the probability that a transaction belongs
    to the model's positive (risk/fraud) class.

    Returns:
        tuple:
            risk_probability (float)
            risk_level (str)
    """

    input_data = pd.DataFrame(
        [
            {
                "amount": txn.amount,
                "txn_type": txn.txn_type,
                "location": getattr(txn, "location", "US"),
                "device_type": getattr(
                    txn,
                    "device_type",
                    "web",
                ),
            }
        ]
    )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not support probability prediction."
        )

    probabilities = model.predict_proba(input_data)

    risk_probability = float(probabilities[0][1])

    risk_level = classify_risk(risk_probability)

    return round(risk_probability, 4), risk_level


def classify_risk(probability):
    """
    Convert model probability into an interpretable risk category.

    Thresholds are demonstration thresholds and should be
    validated experimentally before production use.
    """

    if probability >= 0.75:
        return "High risk"

    if probability >= 0.40:
        return "Medium risk"

    return "Low risk"
