import pytest

from app import create_app
from app.services import ai_service


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    monkeypatch.setattr(ai_service, "client", None)
    app = create_app()
    app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        DATA_FOLDER=str(tmp_path / "data"),
        DATASET_REGISTRY_PATH=str(tmp_path / "data" / "datasets.json"),
    )
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_security_headers_are_set_on_api_responses(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"


def test_debug_classify_row_is_hidden_when_debug_is_disabled(client):
    response = client.get("/api/debug/classify-row?dataset_id=x&name=Ada&industry=tech")

    assert response.status_code == 404
    assert response.get_json()["error"] == "Route not found."


def test_secret_key_can_be_configured_from_environment(monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret")

    app = create_app()

    assert app.config["SECRET_KEY"] == "test-secret"
