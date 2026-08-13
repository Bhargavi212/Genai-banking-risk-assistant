from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


# ---------------------------------------------------------
# Root / Health Check
# ---------------------------------------------------------

def test_root():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "GenAI Banking Risk & Compliance Assistant",
        "status": "running",
    }


# ---------------------------------------------------------
# Transaction API Validation
# ---------------------------------------------------------

def test_transaction_rejects_negative_amount():
    payload = {
        "user_id": "test_user",
        "amount": -100,
        "txn_type": "domestic",
        "location": "US",
        "device_type": "web",
        "timestamp": "2026-08-13T10:00:00",
    }

    response = client.post(
        "/transactions/score",
        json=payload,
    )

    assert response.status_code == 422


def test_transaction_rejects_invalid_timestamp():
    payload = {
        "user_id": "test_user",
        "amount": 1000,
        "txn_type": "domestic",
        "location": "US",
        "device_type": "web",
        "timestamp": "not-a-date",
    }

    response = client.post(
        "/transactions/score",
        json=payload,
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Compliance API Validation
# ---------------------------------------------------------

def test_compliance_rejects_short_question():
    response = client.post(
        "/compliance/qa",
        json={
            "question": "a",
        },
    )

    assert response.status_code == 422


def test_pdf_upload_rejects_non_pdf():
    response = client.post(
        "/compliance/upload",
        files={
            "file": (
                "test.txt",
                b"This is not a PDF.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
