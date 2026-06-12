import os
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


def upload_dataframe(client, df, filename="sample.csv"):
    payload = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )
    assert response.status_code in {200, 201}, response.get_data(as_text=True)
    return response.get_json()


def sample_df(rows=2):
    return pd.DataFrame(
        {
            "First Name": [f"Person{i}" for i in range(rows)],
            "Employer": ["Acme"] * rows,
        }
    )


def test_list_datasets_returns_uploads_newest_first(client):
    first = upload_dataframe(client, sample_df(1), "first.csv")
    second = upload_dataframe(client, sample_df(3), "second.csv")

    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 2
    datasets = data["datasets"]
    ids = [item["dataset_id"] for item in datasets]
    # Newest first, even when uploads land within the same second.
    assert ids == [second["dataset_id"], first["dataset_id"]]

    item = next(entry for entry in datasets if entry["dataset_id"] == second["dataset_id"])
    assert item["display_name"] == "second.csv"
    assert item["original_filename"] == "second.csv"
    assert item["stored_filename"].endswith("second.csv")
    assert item["uploaded_at"]
    assert item["row_count"] == 3
    assert item["column_count"] == 2
    assert item["columns"] == ["First Name", "Employer"]
    assert item["file_type"] == "csv"
    assert item["status"] == "ready"


def test_list_datasets_empty_registry(client):
    response = client.get("/api/datasets")
    assert response.status_code == 200
    assert response.get_json() == {"datasets": [], "count": 0}


def test_datasets_persist_across_simulated_restart(client, tmp_path, monkeypatch):
    uploaded = upload_dataframe(client, sample_df(), "persist.csv")

    restarted = create_app()
    restarted.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        DATA_FOLDER=str(tmp_path / "data"),
        DATASET_REGISTRY_PATH=str(tmp_path / "data" / "datasets.json"),
    )
    response = restarted.test_client().get("/api/datasets")
    data = response.get_json()
    assert data["count"] == 1
    assert data["datasets"][0]["dataset_id"] == uploaded["dataset_id"]
    assert data["datasets"][0]["status"] == "ready"


def test_rename_dataset_persists_and_validates(client):
    uploaded = upload_dataframe(client, sample_df(), "rename-me.csv")
    dataset_id = uploaded["dataset_id"]

    response = client.patch(f"/api/datasets/{dataset_id}", json={"display_name": "Alumni 2026"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["display_name"] == "Alumni 2026"
    assert data["original_filename"] == "rename-me.csv"

    listed = client.get("/api/datasets").get_json()["datasets"]
    assert listed[0]["display_name"] == "Alumni 2026"

    # Empty/blank names are rejected and do not change state.
    assert client.patch(f"/api/datasets/{dataset_id}", json={"display_name": "  "}).status_code == 400
    assert client.patch(f"/api/datasets/{dataset_id}", json={}).status_code == 400
    assert client.get("/api/datasets").get_json()["datasets"][0]["display_name"] == "Alumni 2026"

    # Unknown dataset is a clean 404.
    assert client.patch("/api/datasets/nope", json={"display_name": "X"}).status_code == 404


def test_delete_dataset_removes_metadata_and_file(client, app):
    uploaded = upload_dataframe(client, sample_df(), "delete-me.csv")
    dataset_id = uploaded["dataset_id"]
    stored = uploaded["metadata"]["stored_filename"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], stored)
    assert os.path.exists(file_path)

    response = client.delete(f"/api/datasets/{dataset_id}")
    assert response.status_code == 200
    assert response.get_json() == {"deleted": True, "dataset_id": dataset_id}
    assert not os.path.exists(file_path)
    assert client.get("/api/datasets").get_json()["count"] == 0

    # Deleting again (or unknown ids) is a clean JSON 404.
    repeat = client.delete(f"/api/datasets/{dataset_id}")
    assert repeat.status_code == 404
    assert "error" in repeat.get_json()


def test_missing_file_marks_status_missing_without_crashing(client, app):
    uploaded = upload_dataframe(client, sample_df(), "vanishing.csv")
    stored = uploaded["metadata"]["stored_filename"]
    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], stored))

    response = client.get("/api/datasets")
    assert response.status_code == 200
    item = response.get_json()["datasets"][0]
    assert item["dataset_id"] == uploaded["dataset_id"]
    assert item["status"] == "missing"

    # Preview of the missing dataset is a clean JSON error, not a crash.
    preview = client.get(f"/api/datasets/{uploaded['dataset_id']}/preview")
    assert preview.status_code == 404
    assert "missing" in preview.get_json()["error"].lower()


def test_upload_preview_summary_and_ask_still_work_with_library(client):
    df = pd.DataFrame(
        {
            "First Name": ["Ada"],
            "Last Name": ["Lovelace"],
            "Occupation": ["Software Engineer"],
            "Employer": ["Google"],
        }
    )
    uploaded = upload_dataframe(client, df, "ask.csv")
    dataset_id = uploaded["dataset_id"]
    assert uploaded["metadata"]["display_name"] == "ask.csv"

    assert client.get(f"/api/datasets/{dataset_id}/preview").status_code == 200
    assert client.get(f"/api/datasets/{dataset_id}/summary").status_code == 200

    ask = client.post("/api/ask", json={"dataset_id": dataset_id, "question": "Which alumni work in tech?"})
    assert ask.status_code == 200
    assert ask.get_json()["result"]["total_matches"] == 1
