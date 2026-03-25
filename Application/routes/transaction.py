from fastapi import APIRouter, HTTPException
from Application.models.schemas import Transaction
from Application.services.risk_engine import score_transaction

router = APIRouter()

@router.post("/txn")
def analyze_transaction(txn: Transaction):
    try:
        score, reason = score_transaction(txn)
        return {"score": score, "reason": reason}
    except Exception as e:
        print(f" Error in score_transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
