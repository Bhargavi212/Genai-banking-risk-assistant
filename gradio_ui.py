"""
gradio_ui.py  –  front-end for the GenAI Banking Compliance project
-------------------------------------------------------------------
Launch as a standalone app:

    python gradio_ui.py           # http://127.0.0.1:7860

…or mount inside FastAPI:

    from gradio.routes import mount_gradio_app
    from gradio_ui import build_ui
    app = FastAPI()
    mount_gradio_app(app, build_ui(), path="/ui")
"""

import gradio as gr
import requests

BACKEND_URL = "http://127.0.0.1:8000"        # FastAPI base URL


# ───────────────────────────── helper calls ──────────────────────────────
def score_txn(user_id, amount, txn_type, location, device_type, timestamp):
    payload = {
        "user_id": user_id,
        "amount": amount,
        "txn_type": txn_type,
        "location": location,
        "device_type": device_type,
        "timestamp": timestamp,
    }
    try:
        res = requests.post(f"{BACKEND_URL}/txn", json=payload, timeout=20)
        res.raise_for_status()
        data = res.json()
        return f"✅ **Risk Score {data['score']}**  — {data['reason']}"
    except Exception as e:
        return f"❌ Error: {e}"


def ask_compliance(question):
    try:
        res = requests.post(
            f"{BACKEND_URL}/compliance-qa", json={"question": question}, timeout=20
        )
        res.raise_for_status()
        return res.json()["answer"]
    except Exception as e:
        return f"❌ Error: {e}"


def upload_pdf(file):
    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name, f, "application/pdf")}
            res = requests.post(f"{BACKEND_URL}/upload-pdf", files=files, timeout=30)
            res.raise_for_status()
            return f"✅ {res.json()['message']}"
    except Exception as e:
        return f"❌ Upload failed: {e}"


# ─────────────────────────── UI construction ─────────────────────────────
def build_ui():
    with gr.Blocks(title="GenAI Banking Compliance & Risk Assistant") as ui:
        gr.Markdown("# 🏦 GenAI Banking Compliance & Risk Assistant")

        with gr.Tab("Transaction Risk Scoring"):
            with gr.Row():
                user_id = gr.Textbox(label="User ID", value="user001")
                timestamp = gr.Textbox(
                    label="Timestamp (YYYY-MM-DDTHH:MM:SS)",
                    value="2025-05-25T10:00:00",
                )
            amount = gr.Number(label="Transaction Amount", value=5000.0)
            txn_type = gr.Dropdown(
                ["domestic", "international"], label="Transaction Type", value="domestic"
            )
            location = gr.Textbox(label="Location", value="UK")
            device_type = gr.Dropdown(
                ["web", "mobile", "atm"], label="Device Type", value="web"
            )
            score_btn = gr.Button("Check Risk", variant="primary")
            score_out = gr.Markdown()

            score_btn.click(
                score_txn,
                [user_id, amount, txn_type, location, device_type, timestamp],
                score_out,
            )

        with gr.Tab("Compliance Q&A"):
            question = gr.Textbox(
                lines=2,
                placeholder="Ask a compliance-related question (e.g., AML, KYC rules)…",
            )
            ask_btn = gr.Button("Get Answer", variant="primary")
            answer = gr.Markdown()

            ask_btn.click(ask_compliance, question, answer)

        with gr.Tab("Upload Compliance PDF"):
            pdf_file = gr.File(
                label="Upload new policy PDF",
                file_types=[".pdf"],
                type="filepath",
            )
            upload_btn = gr.Button("Upload", variant="primary")
            upload_status = gr.Markdown()

            upload_btn.click(upload_pdf, pdf_file, upload_status)

        gr.Markdown(
            "© 2025 GenAI Banking Compliance Demo &nbsp;|&nbsp; Powered by FastAPI + Gradio"
        )

    return ui


# ──────────────────────────── standalone run ─────────────────────────────
if __name__ == "__main__":
    build_ui().launch(show_error=True, share=False)
