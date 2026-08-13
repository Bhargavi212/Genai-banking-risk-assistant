import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_RECORDS = 2000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

OUTPUT_PATH = Path(__file__).resolve().parent / "transactions.csv"


# ---------------------------------------------------------
# Synthetic Feature Options
# ---------------------------------------------------------

TRANSACTION_TYPES = [
    "domestic",
    "international",
]

LOCATIONS = [
    "US",
    "UK",
    "India",
    "Canada",
    "Germany",
]

DEVICE_TYPES = [
    "web",
    "mobile",
    "atm",
]


# ---------------------------------------------------------
# Synthetic Transaction Generator
# ---------------------------------------------------------

def generate_transaction(index):
    amount = round(
        random.uniform(10, 50000),
        2,
    )

    txn_type = random.choice(
        TRANSACTION_TYPES
    )

    location = random.choice(
        LOCATIONS
    )

    device_type = random.choice(
        DEVICE_TYPES
    )

    timestamp = (
        datetime.now()
        - timedelta(
            minutes=random.randint(
                0,
                60 * 24 * 30,
            )
        )
    )

    # -----------------------------------------------------
    # Synthetic Risk Logic
    # -----------------------------------------------------

    risk_points = 0

    if amount > 25000:
        risk_points += 2

    if txn_type == "international":
        risk_points += 2

    if location != "US":
        risk_points += 1

    if device_type == "atm":
        risk_points += 1

    if amount > 40000:
        risk_points += 1

    # Baseline fraud probability
    fraud_probability = 0.03

    # Increase probability based on risk features
    fraud_probability += (
        risk_points * 0.10
    )

    # Cap probability
    fraud_probability = min(
        fraud_probability,
        0.85,
    )

    is_fraud = int(
        random.random()
        < fraud_probability
    )

    return {
        "transaction_id": str(
            uuid.uuid4()
        ),
        "user_id": f"user_{index:05d}",
        "amount": amount,
        "txn_type": txn_type,
        "location": location,
        "device_type": device_type,
        "timestamp": timestamp.isoformat(),
        "is_fraud": is_fraud,
    }


# ---------------------------------------------------------
# Generate Dataset
# ---------------------------------------------------------

transactions = [
    generate_transaction(i)
    for i in range(NUM_RECORDS)
]

df = pd.DataFrame(
    transactions
)


# ---------------------------------------------------------
# Save Dataset
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print(
    f"Generated {len(df)} transactions."
)

print(
    f"Saved dataset to: {OUTPUT_PATH}"
)

print(
    "\nFraud distribution:"
)

print(
    df["is_fraud"].value_counts()
)

print(
    "\nFraud rate:"
)

print(
    df["is_fraud"].mean()
)
