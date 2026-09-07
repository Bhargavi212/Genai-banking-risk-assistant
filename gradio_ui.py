"""
Gradio front-end for the GenAI Banking Risk & Compliance Assistant.

Standalone:
    python gradio_ui.py

Backend:
    FastAPI must be running at http://127.0.0.1:8000
"""

from pathlib import Path

import gradio as gr
import requests


BACKEND_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------
# Backend API Helpers
# ---------------------------------------------------------

def score_txn(
    user_id,
    amount,
    txn_type,
    location,
    device_type,
    timestamp,
):
    payload = {
        "user_id": user_id,
        "amount": amount,
        "txn_type": txn_type,
        "location": location,
        "device_type": device_type,
        "timestamp": timestamp,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/transactions/score",
            json=payload,
            timeout=20,
        )

        response.raise_for_status()
        data = response.json()

        score = data.get("risk_score")
        risk_level = data.get("risk_reason")

        if score is None:
            return "Unable to retrieve a risk score."

        return (
            f"### Transaction Risk Result\n\n"
            f"**Risk Probability:** {score:.2%}\n\n"
            f"**Risk Level:** {risk_level}"
        )

    except requests.RequestException as exc:
        return (
            "Unable to connect to the transaction "
            f"risk scoring service.\n\nError: {exc}"
        )


def ask_compliance(question):
    if not question or len(question.strip()) < 3:
        return "Please enter a valid compliance question."

    try:
        response = requests.post(
            f"{BACKEND_URL}/compliance/qa",
            json={"question": question},
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        answer = data.get(
            "answer",
            "No answer was returned.",
        )

        retrieval_status = data.get(
            "retrieval_status",
            "unknown",
        )

        return (
            f"### Compliance Answer\n\n"
            f"{answer}\n\n"
            f"**Retrieval status:** {retrieval_status}"
        )

    except requests.RequestException as exc:
        return (
            "Unable to connect to the compliance "
            f"question-answering service.\n\nError: {exc}"
        )


def upload_pdf(file_path):
    if not file_path:
        return "Please select a PDF file."

    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        return "Only PDF files are supported."

    try:
        with path.open("rb") as file:
            files = {
                "file": (
                    path.name,
                    file,
                    "application/pdf",
                )
            }

            response = requests.post(
                f"{BACKEND_URL}/compliance/upload",
                files=files,
                timeout=60,
            )

        response.raise_for_status()
        data = response.json()

        return (
            f"### Upload Successful\n\n"
            f"**File:** {data.get('filename', path.name)}\n\n"
            f"**Chunks indexed:** "
            f"{data.get('chunks_added', 'N/A')}"
        )

    except requests.RequestException as exc:
        return (
            "Unable to upload or index the PDF."
            f"\n\nError: {exc}"
        )


# ---------------------------------------------------------
# UI Construction
# ---------------------------------------------------------

def build_ui():
    with gr.Blocks(
        title="GenAI Banking Risk & Compliance Assistant"
    ) as ui:

        gr.Markdown(
            """
# 🏦 GenAI Banking Risk & Compliance Assistant

### End-to-End AI Engineering Prototype

Combining **Machine Learning**, **Explainable AI**, and
**Retrieval-Augmented Generation** for financial risk and
compliance workflows.

---

**Core capabilities**

- Transaction risk prediction
- Explainable AI
- Banking compliance Q&A
- PDF ingestion and vector retrieval
"""
        )

        # -------------------------------------------------
        # Transaction Risk
        # -------------------------------------------------

        with gr.Tab("🔍 Transaction Risk Scoring"):

            gr.Markdown(
                """
### Analyze a Banking Transaction

Enter transaction details below to generate a machine-learning
risk score.
"""
            )

            with gr.Row():
                user_id = gr.Textbox(
                    label="User ID",
                    value="user001",
                )

                timestamp = gr.Textbox(
                    label="Timestamp",
                    value="2026-08-13T10:00:00",
                    placeholder="YYYY-MM-DDTHH:MM:SS",
                )

            with gr.Row():

                amount = gr.Number(
                    label="Transaction Amount ($)",
                    value=5000.0,
                )

                txn_type = gr.Dropdown(
                    choices=[
                        "domestic",
                        "international",
                    ],
                    label="Transaction Type",
                    value="domestic",
                )

            with gr.Row():

                location = gr.Textbox(
                    label="Location",
                    value="US",
                )

                device_type = gr.Dropdown(
                    choices=[
                        "web",
                        "mobile",
                        "atm",
                    ],
                    label="Device Type",
                    value="web",
                )

            score_btn = gr.Button(
                "Analyze Transaction",
                variant="primary",
            )

            score_out = gr.Markdown()

            score_btn.click(
                fn=score_txn,
                inputs=[
                    user_id,
                    amount,
                    txn_type,
                    location,
                    device_type,
                    timestamp,
                ],
                outputs=score_out,
            )

        # -------------------------------------------------
        # Compliance Q&A
        # -------------------------------------------------

        with gr.Tab("📚 Compliance Q&A"):

            gr.Markdown(
                """
### RAG-Powered Banking Compliance Assistant

Ask questions based on the compliance documents indexed by
the backend.
"""
            )

            question = gr.Textbox(
                lines=4,
                label="Compliance Question",
                placeholder=(
                    "Example: What are the key components "
                    "of a BSA/AML risk assessment?"
                ),
            )

            ask_btn = gr.Button(
                "Ask Compliance Assistant",
                variant="primary",
            )

            answer = gr.Markdown()

            ask_btn.click(
                fn=ask_compliance,
                inputs=question,
                outputs=answer,
            )

        # -------------------------------------------------
        # PDF Upload
        # -------------------------------------------------

        with gr.Tab("📄 Upload Compliance PDF"):

            gr.Markdown(
                """
### Add a New Compliance Document

Upload a PDF to add its contents to the compliance
retrieval pipeline.
"""
            )

            pdf_file = gr.File(
                label="Compliance Document",
                file_types=[".pdf"],
                type="filepath",
            )

            upload_btn = gr.Button(
                "Upload and Index PDF",
                variant="primary",
            )

            upload_status = gr.Markdown()

            upload_btn.click(
                fn=upload_pdf,
                inputs=pdf_file,
                outputs=upload_status,
            )

        gr.Markdown(
            """
---

### Technology Stack

**ML:** Scikit-learn • SHAP  
**GenAI / RAG:** Sentence Transformers • FAISS • LLM  
**Backend:** FastAPI  
**Frontend:** Gradio  
**MLOps:** MLflow • Docker • Prometheus • Grafana • GitHub Actions

*Research and engineering prototype using synthetic transaction data.*
"""
        )

    return ui


# ---------------------------------------------------------
# Standalone Run
# ---------------------------------------------------------

if __name__ == "__main__":
    build_ui().launch(
        show_error=True,
        share=True,
    )
