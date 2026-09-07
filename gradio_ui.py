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
            f"risk scoring service.\n\n**Error:** {exc}"
        )


# ---------------------------------------------------------
# Compliance Q&A
# ---------------------------------------------------------

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
            f"question-answering service.\n\n**Error:** {exc}"
        )


# ---------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------

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
            f"\n\n**Error:** {exc}"
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

### AI-powered Banking Risk Detection & Compliance Intelligence

An end-to-end AI engineering prototype combining
**Machine Learning**, **Explainable AI**, and
**Retrieval-Augmented Generation (RAG)**.

**Capabilities**
- 🔍 Transaction Risk Prediction
- 🧠 Explainable Machine Learning
- 📚 RAG-powered Compliance Q&A
- 📄 Compliance Document Ingestion
"""
        )

        # -------------------------------------------------
        # Transaction Risk
        # -------------------------------------------------

        with gr.Tab("🔍 Transaction Risk Scoring"):

            gr.Markdown(
                """
### Analyze Transaction Risk

Enter transaction information to generate an ML-powered
risk probability.
"""
            )

            with gr.Row():

                user_id = gr.Textbox(
                    label="User ID",
                    value="user777",
                )

                timestamp = gr.Textbox(
                    label="Timestamp",
                    value="2026-09-07T13:20:00",
                    placeholder="YYYY-MM-DDTHH:MM:SS",
                )

            with gr.Row():

                amount = gr.Number(
                    label="Transaction Amount ($)",
                    value=25000.0,
                )

                txn_type = gr.Dropdown(
                    choices=[
                        "domestic",
                        "international",
                    ],
                    label="Transaction Type",
                    value="international",
                )

            with gr.Row():

                location = gr.Textbox(
                    label="Location",
                    value="Germany",
                )

                device_type = gr.Dropdown(
                    choices=[
                        "web",
                        "mobile",
                        "atm",
                    ],
                    label="Device Type",
                    value="atm",
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
### RAG-Powered Compliance Assistant

Ask questions based on the banking compliance documents
available to the retrieval system.
"""
            )

            question = gr.Textbox(
                lines=4,
                label="Compliance Question",
                value=(
                    "What are the key components of "
                    "a BSA/AML risk assessment?"
                ),
                placeholder=(
                    "Ask a question based on the "
                    "compliance documents."
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
### Compliance Document Ingestion

Upload a PDF compliance document for indexing and
retrieval.
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

        # -------------------------------------------------
        # Footer
        # -------------------------------------------------

        gr.Markdown(
            """
---

### ⚙️ Technology Stack

**Machine Learning:** Scikit-learn • SHAP  
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