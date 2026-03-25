from pathlib import Path
import pandas as pd
import optuna, mlflow
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH  = ROOT / "Dataset" / "transactions.csv"
MODEL_PATH = ROOT / "model.pkl"


# Load Data 
df = pd.read_csv(DATA_PATH)
FEATURES = ["amount", "txn_type", "location", "device_type"]
X = df[FEATURES]
y = df["is_fraud"]

#  Preprocessing 
cat_cols = ["txn_type", "location", "device_type"]
num_cols = ["amount"]
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# MLflow Setup 
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Fraud Detection Optuna")

# Objective Function 
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
    }

    with mlflow.start_run(nested=True):
        clf = make_pipeline(
            preprocess,
            RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        )
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

        mlflow.log_params(params)
        mlflow.log_metric("auc", auc)

    return auc

# Run Optimization
if __name__ == "__main__":
    study = optuna.create_study(
        study_name="fraud_risk",
        storage="sqlite:///optuna/optuna.db",  
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=25, show_progress_bar=True)

    print("Best params:", study.best_params)

    # Final model training and save
    best_model = make_pipeline(
        preprocess,
        RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
    )
    best_model.fit(X_tr, y_tr)
    dump(best_model, MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")
