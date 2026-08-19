# GenAI Banking Risk & Compliance Assistant

![CI](https://github.com/Bhargavi212/Genai-banking-risk-assistant/actions/workflows/ci.yml/badge.svg)

An end-to-end AI/ML research prototype combining **machine learning, explainable AI, and Retrieval-Augmented Generation (RAG)** for banking transaction risk analysis and compliance question answering.

The project explores two related problems:

1. **Transaction Risk Analysis** — identifying potentially fraudulent transactions using supervised machine learning.
2. **Compliance Question Answering** — retrieving relevant evidence from banking compliance documents and generating context-grounded answers using an LLM.

> **Research prototype:** The transaction dataset used in the current experiments is synthetically generated. Results should not be interpreted as production banking performance.

---

## 🎯 Motivation

Financial institutions must identify suspicious transactions while allowing analysts to efficiently navigate complex regulatory and compliance information.

Traditional machine learning can help identify patterns associated with transaction risk, while Generative AI and RAG can help analysts retrieve relevant information from regulatory documents.

This project investigates how these approaches can be combined into a single explainable AI system.

---

## 🔬 Research Questions

This project investigates:

- How effectively can machine learning models identify high-risk transactions?
- How do different classification algorithms compare on the same transaction dataset?
- Why can accuracy be misleading when evaluating fraud detection systems?
- How does classification-threshold selection affect precision and recall?
- Which transaction features contribute to model predictions?
- Can SHAP improve interpretability of transaction-risk predictions?
- Can RAG generate compliance answers grounded in retrieved policy documents?
- How can retrieval quality, source correctness, groundedness, and faithfulness be evaluated?

---

## 🏗️ System Architecture

The system contains two primary AI pipelines.

### Transaction Risk Pipeline

```text
Transaction
    ↓
Feature preprocessing
    ↓
ML classifier
    ↓
Fraud probability
    ↓
Decision threshold
    ↓
Risk prediction
    ↓
SHAP explanation
```

### Compliance RAG Pipeline

```text
Compliance PDFs
      ↓
PDF text extraction
      ↓
Text chunking
      ↓
SentenceTransformer embeddings
      ↓
FAISS vector index
      ↓
User question
      ↓
Semantic retrieval
      ↓
Relevant document context
      ↓
Groq-hosted LLM
      ↓
Context-grounded answer
```

---

## ✨ Key Features

### Transaction Risk Scoring

FastAPI exposes a transaction-risk endpoint:

```text
POST /txn
```

The model processes:

- Transaction amount
- Transaction type
- Location
- Device type

and returns a fraud-risk score and risk classification.

### Explainable AI

The project includes a SHAP-based explanation endpoint:

```text
POST /fraud_explain
```

SHAP values provide feature-level attribution for individual transaction predictions.

### Compliance Question Answering

The RAG endpoint:

```text
POST /compliance-qa
```

retrieves relevant information from indexed compliance documents before sending the retrieved context to the language model.

### Dynamic PDF Indexing

New compliance documents can be uploaded through:

```text
POST /upload-pdf
```

Uploaded PDFs are:

1. Parsed with `pdfplumber`
2. Split into text chunks
3. Embedded using Sentence Transformers
4. Added to the FAISS index

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Backend | FastAPI, Uvicorn |
| Frontend | Gradio |
| Machine Learning | Scikit-learn |
| Model Comparison | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| Explainability | SHAP |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Document Processing | pdfplumber |
| Generative AI | Groq-hosted LLM |
| Experiment Tracking | MLflow |
| Monitoring | Prometheus, Grafana |
| Deployment | Docker, Docker Compose |
| Testing | Pytest |
| CI | GitHub Actions |

---

# 📊 Machine Learning Experiments

## Dataset

The current experiments use a **synthetically generated transaction dataset** containing features such as:

- Amount
- Transaction type
- Location
- Device type
- Fraud label

The synthetic dataset allows the complete ML pipeline to be demonstrated without exposing real financial or personally identifiable information.

---

## Baseline Model Comparison

Four classification algorithms were evaluated using the same **stratified 80/20 train-test split**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.670 | 0.557 | **0.695** | **0.618** | **0.723** |
| XGBoost | **0.680** | **0.600** | 0.506 | 0.549 | 0.718 |
| Gradient Boosting | 0.633 | 0.526 | 0.461 | 0.491 | 0.682 |
| Random Forest | 0.628 | 0.517 | 0.487 | 0.502 | 0.663 |

### Key Finding

Logistic Regression achieved the strongest baseline performance when prioritizing:

- ROC-AUC
- Fraud-class recall
- F1-score

Although XGBoost achieved slightly higher overall accuracy and precision, Logistic Regression detected a substantially larger proportion of fraud cases.

This experiment demonstrates that **increased model complexity does not automatically produce better performance**.

It also illustrates why accuracy alone can be misleading for fraud-detection problems.

---

## 🎚️ Classification Threshold Analysis

The Logistic Regression model was further evaluated across different classification thresholds.

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.20 | 0.407 | 0.994 | 0.577 |
| 0.30 | 0.450 | 0.974 | 0.616 |
| **0.40** | **0.500** | **0.831** | **0.624** |
| 0.50 | 0.557 | 0.695 | 0.618 |
| 0.60 | 0.591 | 0.442 | 0.506 |
| 0.70 | 0.704 | 0.247 | 0.365 |

### Key Finding

Among the tested thresholds, **0.40 achieved the highest F1-score of 0.624**.

Lowering the threshold from the default `0.50` to `0.40` increased fraud recall from:

**69.5% → 83.1%**

while precision decreased from:

**55.7% → 50.0%**

This demonstrates the practical **precision-recall trade-off** involved in fraud screening.

A lower threshold detects more potentially fraudulent transactions but also produces more false-positive alerts.

A production system would require threshold selection based on independent validation data and the operational costs of false positives and false negatives.

---

## 🔍 Explainability

SHAP is used to examine how individual transaction features influence model predictions.

The explainability component is designed to help answer questions such as:

- Why was this transaction considered risky?
- Which features contributed most strongly to the prediction?
- Did transaction amount significantly affect the score?
- Did transaction type, location, or device type increase predicted risk?

This is particularly important for financial AI systems where model decisions may require investigation and human review.

---

# 🤖 Retrieval-Augmented Generation

The compliance assistant uses Retrieval-Augmented Generation to ground LLM responses in uploaded compliance documents.

## Document Processing

Compliance PDFs are processed using:

```text
PDF
 ↓
pdfplumber
 ↓
Recursive text splitting
 ↓
SentenceTransformer embeddings
 ↓
FAISS vector index
```

The embedding model used by the current implementation is:

```text
all-MiniLM-L6-v2
```

---

## Semantic Retrieval

For each compliance question:

1. The question is converted into an embedding.
2. FAISS searches for semantically similar document chunks.
3. The most relevant chunks are retrieved.
4. Retrieved evidence is supplied to the LLM.
5. The LLM is instructed to answer using the supplied context.

This architecture is intended to reduce unsupported generation and improve answer grounding.

---

## 📏 RAG Evaluation

The repository contains an initial RAG evaluation framework.

Current evaluation work considers:

- Retrieval hit rate
- Retrieval latency
- Evidence retrieval

Planned experiments include:

- Recall@1
- Recall@3
- Recall@5
- Source correctness
- Answer relevance
- Groundedness
- Faithfulness
- RAG vs. no-RAG comparison

The current RAG evaluation should be considered an **initial prototype benchmark**, not a comprehensive evaluation of compliance-answer reliability.

---

# 🧪 Reproducible Evaluation

Evaluation scripts are available under:

```text
evaluation/
```

### Fraud Model Evaluation

```bash
python evaluation/evaluate_fraud_model.py
```

Evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrix
- ROC curve

### Model Comparison

```bash
python evaluation/compare_models.py
```

Compares:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

### Threshold Analysis

```bash
python evaluation/threshold_analysis.py
```

Measures precision, recall, and F1 across multiple classification thresholds.

### RAG Evaluation

```bash
python evaluation/evaluate_rag.py
```

Provides the initial retrieval evaluation framework.

---

# 🌐 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Application health/root endpoint |
| `/txn` | POST | Transaction risk scoring |
| `/fraud_explain` | POST | SHAP-based prediction explanation |
| `/compliance-qa` | POST | Compliance RAG question answering |
| `/upload-pdf` | POST | Upload and index a compliance PDF |
| `/metrics` | GET | Prometheus application metrics |

---

# 🖥️ Gradio Interface

The project includes a Gradio interface for interacting with the backend.

Start the FastAPI backend first:

```bash
uvicorn main:app --reload
```

Then launch the interface:

```bash
python gradio_ui.py
```

The interface provides:

- Transaction risk scoring
- Compliance question answering
- Compliance PDF upload

---

# 🧪 Testing and Continuous Integration

The project uses **Pytest** for automated testing and **GitHub Actions** for continuous integration.

The CI workflow runs automatically on pushes and pull requests to the `main` branch.

Current CI checks include:

```text
Code checkout
     ↓
Python environment
     ↓
Dependency installation
     ↓
Flake8 validation
     ↓
Pytest
```

The CI badge at the top of this README reflects the current workflow status.

---

# 📈 Experiment Tracking

ML experiments are tracked using **MLflow**.

The training workflow records model parameters and evaluation metrics using a local SQLite tracking backend:

```text
sqlite:///mlflow.db
```

This makes the experiment-tracking configuration portable across local and notebook environments.

---

# 📊 Monitoring

FastAPI application metrics are instrumented using:

```text
prometheus-fastapi-instrumentator
```

The project also includes Prometheus/Grafana components for observability experiments.

---

# 🐳 Containerization

The project includes Docker/Docker Compose components for local containerized execution.

This supports separation of application and observability services and provides a foundation for future deployment experiments.

---

# ⚠️ Limitations

This repository is an **AI/ML research and engineering prototype**.

Important limitations include:

- Transaction data is synthetically generated.
- Model performance does not represent real banking fraud performance.
- The current dataset is relatively small.
- The current RAG evaluation benchmark is limited.
- Compliance answers should not be treated as legal or regulatory advice.
- Real-world deployment would require stronger security, governance, validation, monitoring, and human oversight.

---

# 🔭 Future Research

Future work includes:

- Recall@K evaluation for RAG retrieval
- Groundedness and faithfulness evaluation
- RAG vs. no-RAG experiments
- Larger labeled compliance QA benchmark
- Embedding-model comparison
- Chunk-size and overlap experiments
- Retrieval reranking
- Hyperparameter optimization
- Probability calibration
- Cost-sensitive fraud classification
- Cross-validation
- Model drift monitoring
- Human-in-the-loop compliance review
- Evaluation on appropriately governed real-world datasets

---

# 📁 Project Structure

```text
Genai-banking-risk-assistant/
│
├── Application/
│   ├── routes/
│   ├── services/
│   └── models/
│
├── Compliance_files/
│
├── Dataset/
│   ├── synthetic_data.py
│   └── transactions.csv
│
├── ML_Model/
│   └── fraud_detection.py
│
├── evaluation/
│   ├── evaluate_fraud_model.py
│   ├── compare_models.py
│   ├── threshold_analysis.py
│   └── evaluate_rag.py
│
├── docs/
│   └── results/
│
├── .github/
│   └── workflows/
│
├── gradio_ui.py
├── main.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

## Research Perspective

This project is designed not only as an application prototype but also as an experimental environment for studying the intersection of:

**Machine Learning + Explainable AI + Generative AI + Information Retrieval + Responsible AI**

The goal is to progressively evaluate each component rather than relying only on end-to-end demonstrations.
