"""
ML_Model/fraud_detection.py
Train a Random-Forest fraud classifier, log metrics to MLflow,
and save the trained scikit-learn pipeline as ../model.pkl
"""

from pathlib import Path
import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Paths 
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]         # one level up from ML_Model/
DATA_PATH = PROJECT_ROOT / "Dataset" / "transactions.csv"
MODEL_PATH = PROJECT_ROOT / "model.pkl"

#  MLflow 
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("fraud_risk")

# Data 
df = pd.read_csv(DATA_PATH)

X = df[["amount", "txn_type", "location", "device_type"]]
y = df["is_fraud"]

categorical = ["txn_type", "location", "device_type"]
numeric = ["amount"]

preprocess = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
     ("num", "passthrough", numeric)]
)

model = make_pipeline(
    preprocess,
    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
)

#  Train 
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
model.fit(X_tr, y_tr)
acc = accuracy_score(y_te, model.predict(X_te))

# Logging 
with mlflow.start_run() as run:
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="model")
    print(f"✔ MLflow run logged → {run.info.run_id}")

#  Persist 
joblib.dump(model, MODEL_PATH)          # this both writes the file and returns its path
print(f"Saved model pipeline to: {MODEL_PATH}")
