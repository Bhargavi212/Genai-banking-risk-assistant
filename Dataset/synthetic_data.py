import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

# Settings
num_records = 1000
fraud_ratio = 0.1  # 10% fraud

txn_types = ["domestic", "international"]
locations = ["US", "UK", "IN", "CA", "AU"]
device_types = ["mobile", "web", "atm"]
start_time = datetime(2024, 1, 1)

transactions = []

for i in range(num_records):
    is_fraud = 1 if i < int(num_records * fraud_ratio) else 0
    amount = round(random.uniform(10, 50000), 2)
    txn_type = random.choice(txn_types)
    location = random.choice(locations)
    device_type = random.choice(device_types)
    timestamp = (start_time + timedelta(minutes=random.randint(0, 100000))).isoformat()
    user_id = f"user_{random.randint(1, 200)}"
    txn_id = str(uuid.uuid4())

    transactions.append({
        "txn_id": txn_id,
        "user_id": user_id,
        "amount": amount,
        "txn_type": txn_type,
        "location": location,
        "device_type": device_type,
        "timestamp": timestamp,
        "is_fraud": is_fraud
    })

# Save to CSV
df = pd.DataFrame(transactions)
df.to_csv("transactions.csv", index=False)
print("✅ transactions.csv created successfully.")
