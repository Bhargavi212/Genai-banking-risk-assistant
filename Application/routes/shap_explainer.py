# appp/routes/shap_explainer.py
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import shap, joblib
from pathlib import Path
import numpy as np

router = APIRouter()

# Pydantic schema  
class TxnFeatures(BaseModel):
    amount: float
    txn_type: str
    location: str
    device_type: str


# Load pipeline  
BASE_DIR   = Path(__file__).resolve().parents[2]          # routes - appp - project-root
MODEL_PATH = BASE_DIR / "model.pkl"
print(f" Loading model from: {MODEL_PATH}")

pipeline       = joblib.load(MODEL_PATH)
preprocessor   = pipeline.named_steps["columntransformer"]
rf_clf         = pipeline.named_steps["randomforestclassifier"]


# Build SHAP explainer  
background_raw = pd.DataFrame(
    [{
        "amount": 10_000,
        "txn_type": "domestic",
        "location": "US",
        "device_type": "web",
    }]
)
background_mat = preprocessor.transform(background_raw)

try:
    # SHAP ≥ 0.42 (takes masker=)
    masker    = shap.maskers.Independent(background_mat, max_samples=100)
    explainer = shap.TreeExplainer(
        rf_clf,
        data=masker,
        feature_names=preprocessor.get_feature_names_out(),
    )
except TypeError:
    # SHAP < 0.42
    explainer = shap.TreeExplainer(rf_clf, data=background_mat)
    explainer.feature_names = preprocessor.get_feature_names_out().tolist()


# API route  
@router.post("/fraud_explain")
def fraud_explain(txn: TxnFeatures):
    """
    Return SHAP values for a single transaction.
    """
    try:
        # ------- Encode sample -------
        raw_df      = pd.DataFrame([txn.model_dump()])   # Pydantic v2 
        numeric_mat = preprocessor.transform(raw_df)

        shap_out    = explainer(numeric_mat)

        # ------- Pick class index (1 = fraud-prob column) -------
        multi_output = shap_out.values.ndim == 3     # (n_samples, n_features, n_classes)
        cls_idx      = 1 if multi_output else None

        # SHAP values
        if multi_output:
            shap_vals = shap_out.values[0][:, cls_idx]   # (n_features,)
        else:
            shap_vals = shap_out.values[0]               # (n_features,)

        # Expected value(s)
        if shap_out.base_values.ndim == 2:               # (n_samples, n_classes)
            expected_val = float(shap_out.base_values[0, cls_idx])
        else:                                            # (n_samples,)
            expected_val = float(shap_out.base_values[0])

        return {
            "feature_names": list(explainer.feature_names),
            "shap_values":   shap_vals.tolist(),
            "expected_value": expected_val,
        }

    except Exception as e:
        return {"error": f"SHAP explanation failed: {e}"}
