from fastapi import FastAPI
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator

from Application.routes import transaction, compliance, shap_explainer

load_dotenv()

app = FastAPI(
    title="GenAI Banking Risk & Compliance Assistant",
    description=(
        "AI/ML application for transaction risk prediction, "
        "SHAP-based explainability, and RAG-powered compliance question answering."
    ),
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)

app.include_router(transaction.router)
app.include_router(compliance.router)
app.include_router(shap_explainer.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "GenAI Banking Risk & Compliance Assistant",
        "status": "running",
    }
