import logging

from fastapi import APIRouter, HTTPException

from Application.models.schemas import Transaction
from Application.services.risk_engine import score_transaction


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/transactions",
    tags=["Transaction Risk"],
)


@router.post("/score")
def analyze_transaction(txn: Transaction):
    try:
        score, reason = score_transaction(txn)

        return {
            "risk_score": score,
            "risk_reason": reason,
            "status": "success",
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid transaction input.",
        ) from exc

    except Exception as exc:
        logger.exception("Transaction scoring failed")

        raise HTTPException(
            status_code=500,
            detail="Unable to score transaction.",
        ) from exc
