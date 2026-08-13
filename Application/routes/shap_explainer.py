import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import APIRouter, HTTPException

from Application.models.schemas import TxnFeatures


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/explanations",
    tags=["Explainability"],
)


# ---------------------------------------------------------
# Load Model Pipeline
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}"
    )

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["columntransformer"]
classifier = pipeline.named_steps["randomforestclassifier"]


# ---------------------------------------------------------
# Initialize SHAP Explainer
# ---------------------------------------------------------

explainer = shap.TreeExplainer(classifier)

feature_names = list(
    preprocessor.get_feature_names_out()
)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def extract_fraud_class_shap(
    shap_values,
    sample_index=0,
    class_index=1,
):
    """
    Extract SHAP values for the positive/fraud class while
    supporting multiple SHAP output formats.
    """

    # Older SHAP versions may return one array per class
    if isinstance(shap_values, list):
        return np.asarray(
            shap_values[class_index][sample_index]
        )

    values = np.asarray(shap_values)

    # Newer SHAP:
    # (samples, features, classes)
    if values.ndim == 3:
        return values[
            sample_index,
            :,
            class_index,
        ]

    # Binary/single-output case:
    # (samples, features)
    if values.ndim == 2:
        return values[sample_index]

    raise ValueError(
        f"Unexpected SHAP output shape: {values.shape}"
    )


def get_expected_value(class_index=1):
    expected = np.asarray(
        explainer.expected_value
    )

    if expected.ndim == 0:
        return float(expected)

    if len(expected) > class_index:
        return float(expected[class_index])

    return float(expected[0])


# ---------------------------------------------------------
# SHAP Explanation API
# ---------------------------------------------------------

@router.post("/fraud")
def fraud_explain(txn: TxnFeatures):
    """
    Explain the model prediction for a transaction using
    SHAP feature contributions.
    """

    try:
        raw_df = pd.DataFrame(
            [txn.model_dump()]
        )

        transformed = preprocessor.transform(
            raw_df
        )

        # Some preprocessors can produce sparse matrices.
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        shap_output = explainer.shap_values(
            transformed
        )

        fraud_shap_values = (
            extract_fraud_class_shap(
                shap_output,
                class_index=1,
            )
        )

        expected_value = get_expected_value(
            class_index=1
        )

        prediction_probability = float(
            classifier.predict_proba(
                transformed
            )[0][1]
        )

        contributions = []

        for feature, value in zip(
            feature_names,
            fraud_shap_values,
        ):
            contributions.append(
                {
                    "feature": feature,
                    "shap_value": round(
                        float(value),
                        6,
                    ),
                }
            )

        contributions.sort(
            key=lambda item: abs(
                item["shap_value"]
            ),
            reverse=True,
        )

        return {
            "fraud_probability": round(
                prediction_probability,
                4,
            ),
            "expected_value": round(
                expected_value,
                6,
            ),
            "feature_contributions": contributions,
            "top_contributors": contributions[:5],
        }

    except Exception as exc:
        logger.exception(
            "SHAP explanation failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate model explanation."
            ),
        ) from exc
