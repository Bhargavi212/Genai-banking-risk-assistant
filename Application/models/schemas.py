from datetime import datetime

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    user_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)

    txn_type: str = Field(
        ...,
        min_length=1,
        description="Transaction type used by the fraud model.",
    )

    location: str = Field(
        ...,
        min_length=1,
        description="Transaction location.",
    )

    device_type: str = Field(
        ...,
        min_length=1,
        description="Device or channel used for the transaction.",
    )

    timestamp: datetime


class TxnFeatures(BaseModel):
    amount: float = Field(..., gt=0)
    txn_type: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    device_type: str = Field(..., min_length=1)
