![image](https://github.com/user-attachments/assets/0c49f9d0-d4d5-4560-8769-7896642d678f)
![image](https://github.com/user-attachments/assets/cc89d691-3d57-429f-a6a3-2eca55e79965)
![image](https://github.com/user-attachments/assets/2363bcde-d68a-4313-82b6-e675f50c9164)

# GenAI Banking Risk & Compliance Assistant

![CI](https://github.com/Bhargavi212/Genai-banking-risk-assistant/actions/workflows/ci.yml/badge.svg)

An end-to-end AI/ML prototype for banking risk analysis and compliance
question answering, combining machine learning, explainable AI, and
retrieval-augmented generation (RAG).

## 🎯 Motivation

Financial institutions need to identify potentially risky transactions
while also providing analysts with reliable access to complex compliance
and regulatory information.

This project explores how traditional machine learning and Generative AI
can be combined to support these two tasks:

1. **Transaction Risk Prediction** – Use supervised machine learning to
   identify potentially risky transactions from structured transaction data.

2. **Compliance Question Answering** – Use Retrieval-Augmented Generation
   (RAG) to retrieve relevant information from compliance documents and
   generate context-grounded responses.

## 🔬 Research Questions

This project investigates:

- How effectively can machine learning models identify high-risk transactions?
- Which transaction features contribute most strongly to model predictions?
- Can SHAP explanations improve the interpretability of transaction risk scores?
- Can RAG provide more grounded compliance answers by retrieving relevant
  policy information before generating a response?
- How can retrieval quality and answer faithfulness be evaluated in a
  compliance-focused RAG system?

## 🧠 System Components

- **Risk Prediction:** Random Forest-based transaction risk classification
- **Explainable AI:** SHAP-based feature attribution
- **Document Processing:** PDF extraction and text chunking
- **Embeddings:** Sentence Transformers
- **Vector Search:** FAISS
- **Generative AI:** LLM-based compliance response generation
- **Backend:** FastAPI
- **Frontend:** Gradio
- **Experiment Tracking:** MLflow
- **Monitoring:** Prometheus and Grafana
- **Deployment:** Docker / Docker Compose

# GenAI Banking Compliance & Risk Assistant

An end-to-end AI/ML prototype for banking risk analysis and compliance question answering, combining machine learning, explainable AI, and retrieval-augmented generation (RAG).

---

## Key Highlights

- Real-time fraud risk scoring using ML model (RandomForest)
- SHAP - based explainability (`/fraud_explain`)
- Compliance Q&A over uploaded policy PDFs (RAG using FAISS + Groq)
- Upload support for new documents via Streamlit + FastAPI
- Prometheus + Grafana observability dashboard
- Tracked experiments via MLflow
- Containerized with Docker Compose for local orchestration

---

## Tech Stack

| Domain         | Tools / Frameworks                                      |
|----------------|---------------------------------------------------------|
| Backend        | FastAPI, Uvicorn                                        |
| Frontend       | Gradio                                               |
| ML & Explain   | Scikit-Learn, SHAP, Optuna (planned), MLflow            |
| GenAI & RAG    | FAISS, SentenceTransformers, pdfplumber, Groq API       |
| Observability  | Prometheus, Grafana, prometheus-fastapi-instrumentator |
| Orchestration  | Docker, Docker Compose                                  |

---

## Core Features

### Fraud Transaction Risk Analysis (`/txn`)
- Input: `amount`, `txn_type`, `location`, `device_type`
- ML model (RandomForest) returns fraud **risk score** and reason
- SHAP visualization via `/fraud_explain` for transparency

### Compliance Q&A with RAG (`/compliance-qa`)
- Upload PDFs (AML/KYC/etc.)
- Documents embedded with `sentence-transformers` and stored in FAISS
- Questions are answered using Groq-hosted LLM with retrieved context

### MLFlow Tracking
- ML experiments are logged to `mlflow.db`
- View via: [http://localhost:5001](http://localhost:5001)

### Observability
- FastAPI metrics exposed to Prometheus
- Dashboards available at: [http://localhost:3000](http://localhost:3000)

---

## Streamlit UI

```bash
python gradio_ui.py
```
## 🧪 Evaluation Strategy

The system is evaluated across two components.

### Machine Learning Evaluation

The transaction risk model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- SHAP-based feature importance

### RAG Evaluation

The compliance assistant is evaluated using:

- Retrieval Hit Rate
- Recall@K
- Answer Relevance
- Groundedness / Faithfulness
- Source Correctness
- Response Latency

Future experiments will compare:

**LLM without retrieval vs. LLM with RAG**

to measure whether retrieval improves the reliability and grounding of
compliance responses.
