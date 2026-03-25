import joblib
import numpy as np
import pandas as pd
import os

# Load model when module is loaded
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../model.pkl")
model = joblib.load(MODEL_PATH)

def score_transaction(txn):
    input_data = pd.DataFrame([{
    "amount": txn.amount,
    "txn_type": txn.txn_type,
    "location": getattr(txn, "location", "US"),
    "device_type": getattr(txn, "device_type", "web")
    }])

    # Predict probability of fraud
    proba = model.predict_proba(input_data)[0][1]  # prob of class 1 (fraud)
    reason = "High risk" if proba > 0.5 else "Low risk"
    return round(float(proba), 4), reason
