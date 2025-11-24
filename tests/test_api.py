import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

@pytest.fixture(scope="session")
def sample_texts():
    return {
        "spam": "Congratulations, you won a free lottery prize!",
        "ham": "Just checking if you are coming to the meeting tomorrow"
    }

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_loaded" in data and data["model_loaded"] is True
    assert "vectorizer_loaded" in data and data["vectorizer_loaded"] is True


def test_predict_spam(sample_texts):
    resp = client.post("/predict", json={"text": sample_texts["spam"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ("spam", "ham")
    assert isinstance(data["spam_probability"], float)


def test_predict_ham(sample_texts):
    resp = client.post("/predict", json={"text": sample_texts["ham"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ("spam", "ham")


def test_feedback(sample_texts, tmp_path):
    # Feedback endpoint should accept payload
    resp = client.post("/feedback", json={"text": sample_texts["ham"], "true_label": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "feedback saved"
