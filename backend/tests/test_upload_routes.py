from io import BytesIO

import pandas as pd
import pytest

from app import create_app
from app.services import ai_service


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)
    app = create_app()
    app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        DATA_FOLDER=str(tmp_path / "data"),
        DATASET_REGISTRY_PATH=str(tmp_path / "data" / "datasets.json"),
    )
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_upload_missing_file_field(client):
    response = client.post("/api/upload", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Missing file field" in response.get_json()["error"]


def test_upload_empty_filename(client):
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(b"data"), "")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "No file selected" in response.get_json()["error"]


def test_upload_unsupported_type(client):
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(b"data"), "readme.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported" in response.get_json()["error"]


def test_upload_empty_file(client):
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(b""), "empty.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "empty" in response.get_json()["error"].lower()


def test_upload_valid_csv(client):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    payload = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(payload), "data.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    data = response.get_json()
    assert "dataset_id" in data
    assert data["filename"] == "data.csv"
    assert "summary" in data
    assert data["metadata"]["row_count"] == 2
    assert data["metadata"]["column_count"] == 2


def test_upload_valid_xlsx(client):
    stream = BytesIO()
    pd.DataFrame({"X": [10], "Y": [20]}).to_excel(stream, index=False, engine="openpyxl")
    stream.seek(0)
    response = client.post(
        "/api/upload",
        data={"file": (stream, "data.xlsx")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    data = response.get_json()
    assert data["metadata"]["file_type"] == "xlsx"
