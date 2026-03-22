import os

from fastapi.testclient import TestClient

from src.api import app


class DummyEngine:
    def predict(self, text: str) -> dict:
        return {"category": "Technical Support", "confidence": 0.98}


def test_predict_endpoint_success(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")
    monkeypatch.setenv("ENABLE_AUTH", "0")
    monkeypatch.setenv("TICKET_DB_PATH", str(tmp_path / "tickets_test.db"))

    with TestClient(app) as client:
        app.state.engine = DummyEngine()
        response = client.post("/predict", json={"text": "Mi luz inteligente no funciona"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "Technical Support"
    assert payload["confidence"] == 0.98
    assert "Mi luz inteligente no funciona" in payload["cleaned_text"]


def test_predict_endpoint_model_not_loaded(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")
    monkeypatch.setenv("ENABLE_AUTH", "0")
    monkeypatch.setenv("TICKET_DB_PATH", str(tmp_path / "tickets_test.db"))

    with TestClient(app) as client:
        app.state.engine = None
        app.state.startup_error = "No model found"
        response = client.post("/predict", json={"text": "hello"})

    assert response.status_code == 503


def test_auth_login_and_me(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")
    monkeypatch.setenv("ENABLE_AUTH", "1")
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth_test.db"))
    monkeypatch.setenv("TICKET_DB_PATH", str(tmp_path / "tickets_test.db"))
    monkeypatch.setenv("AUTH_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("AUTH_ADMIN_PASSWORD", "pass123")

    with TestClient(app) as client:
        login = client.post("/auth/login", json={"username": "admin", "password": "pass123"})
        assert login.status_code == 200
        token = login.json()["access_token"]

        me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["role"] == "admin"


def test_viewer_cannot_predict(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIP_MODEL_LOAD", "1")
    monkeypatch.setenv("ENABLE_AUTH", "1")
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth_test.db"))
    monkeypatch.setenv("TICKET_DB_PATH", str(tmp_path / "tickets_test.db"))
    monkeypatch.setenv("AUTH_VIEWER_USERNAME", "viewer")
    monkeypatch.setenv("AUTH_VIEWER_PASSWORD", "viewpass")

    with TestClient(app) as client:
        app.state.engine = DummyEngine()
        login = client.post("/auth/login", json={"username": "viewer", "password": "viewpass"})
        token = login.json()["access_token"]

        response = client.post(
            "/predict",
            json={"text": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 403
