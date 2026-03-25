import shap
import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException
from app.models.schemas import Transaction
from app.services.risk_engine import model

explainer = shap.Explainer(model)

router = APIRouter()

@router.post("/fraud_explain")
def explain_fraud(txn: Transaction):
    try:
        # Create DataFrame for input
        df = pd.DataFrame([{
            "amount": txn.amount,
            "txn_type": txn.txn_type,
            "location": getattr(txn, "location", "US"),
            "device_type": getattr(txn, "device_type", "web")
        }])

        shap_values = explainer(df)
        explanation = shap_values.values[0].tolist()
        feature_names = df.columns.tolist()

        return {
            "features": feature_names,
            "contributions": explanation,
            "base_value": shap_values.base_values[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
