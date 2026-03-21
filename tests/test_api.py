import os

from fastapi.testclient import TestClient

from src.api import app


class DummyEngine:
    def predict(self, text: str) -> dict:
        return {"category": "Technical Support", "confidence": 0.98}


def test_predict_endpoint_success(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")

    with TestClient(app) as client:
        app.state.engine = DummyEngine()
        response = client.post("/predict", json={"text": "Mi luz inteligente no funciona"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "Technical Support"
    assert payload["confidence"] == 0.98
    assert "Mi luz inteligente no funciona" in payload["cleaned_text"]


def test_predict_endpoint_model_not_loaded(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")

    with TestClient(app) as client:
        app.state.engine = None
        app.state.startup_error = "No model found"
        response = client.post("/predict", json={"text": "hello"})

    assert response.status_code == 503
