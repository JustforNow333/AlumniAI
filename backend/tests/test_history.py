"""Automatic analysis history snapshots."""

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from app import create_app
from app.services import ai_service


def configure(app, tmp_path):
    app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        DATA_FOLDER=str(tmp_path / "data"),
        DATASET_REGISTRY_PATH=str(tmp_path / "data" / "datasets.json"),
        HISTORY_REGISTRY_PATH=str(tmp_path / "data" / "history.json"),
        INSIGHTS_REGISTRY_PATH=str(tmp_path / "data" / "saved_insights.json"),
    )
    return app


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)
    yield configure(create_app(), tmp_path)


@pytest.fixture
def client(app):
    return app.test_client()


def upload_dataframe(client, df, filename="sample.csv"):
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(df.to_csv(index=False).encode("utf-8")), filename)},
        content_type="multipart/form-data",
    )
    assert response.status_code in {200, 201}, response.get_data(as_text=True)
    return response.get_json()["dataset_id"]


def sample_df():
    return pd.DataFrame(
        {
            "First Name": ["Ada", "Grace", "Katherine"],
            "Occupation": ["Software Engineer", "Consultant", "Data Scientist"],
            "Employer": ["Acme", "McKinsey", "NASA"],
            "Gift": [10, 20, 30],
        }
    )


def create_history(client, dataset_id, **overrides):
    payload = {
        "dataset_id": dataset_id,
        "dataset_filename": "alumni.csv",
        "question": "Which alumni work in tech?",
        "answer_text": "2 alumni match tech criteria.",
        "response_payload": {
            "answer": {
                "title": "Tech alumni",
                "summary": "2 alumni match tech criteria.",
                "blocks": [
                    {"type": "metrics", "items": [{"label": "Alumni matching criteria", "value": "2"}]},
                    {"type": "table", "columns": ["First Name"], "rows": [["Ada"]]},
                ],
                "followups": [],
            },
            "answer_text": "2 alumni match tech criteria.",
            "operation": {"type": "contains_any"},
            "result": {"status": "ok", "total_matches": 2},
        },
        "metadata": {"searched_columns": ["Occupation", "Employer"]},
    }
    payload.update(overrides)
    return client.post("/api/history", json=payload)


def test_create_history_item(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")

    response = create_history(client, dataset_id)
    assert response.status_code == 201
    item = response.get_json()
    assert item["history_id"]
    assert item["id"] == item["history_id"]
    assert item["dataset_id"] == dataset_id
    assert item["dataset_filename"] == "alumni.csv"
    assert item["dataset_status"] == "ready"
    assert item["title"] == "Which alumni work in tech"
    assert item["question"] == "Which alumni work in tech?"
    assert item["answer_text"] == "2 alumni match tech criteria."
    assert item["status"] == "success"
    assert item["metadata"]["searched_columns"] == ["Occupation", "Employer"]
    assert item["created_at"]
    assert item["updated_at"] == item["created_at"]


def test_create_history_validation_errors_are_clean(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")

    missing_dataset = create_history(client, "", question="Question?")
    assert missing_dataset.status_code == 400
    assert "dataset_id" in missing_dataset.get_json()["error"]

    missing_question = create_history(client, dataset_id, question="   ")
    assert missing_question.status_code == 400
    assert "question" in missing_question.get_json()["error"]

    missing_answer = create_history(client, dataset_id, answer_text="")
    assert missing_answer.status_code == 400
    assert "answer_text" in missing_answer.get_json()["error"]

    bad_payload = create_history(client, dataset_id, response_payload=["not", "an", "object"])
    assert bad_payload.status_code == 400
    assert "response_payload" in bad_payload.get_json()["error"]


def test_list_history_newest_first(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    first = create_history(client, dataset_id, question="First question?").get_json()
    second = create_history(client, dataset_id, question="Second question?").get_json()

    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 2
    assert [item["history_id"] for item in data["history"]] == [second["history_id"], first["history_id"]]


def test_clear_history_removes_all_items_and_is_idempotent(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    create_history(client, dataset_id, question="First question?")
    create_history(client, dataset_id, question="Second question?")
    assert client.get("/api/history").get_json()["count"] == 2

    response = client.delete("/api/history")
    assert response.status_code == 200
    assert response.get_json() == {"deleted": True, "count": 0}
    assert client.get("/api/history").get_json() == {"history": [], "count": 0}

    repeat = client.delete("/api/history")
    assert repeat.status_code == 200
    assert repeat.get_json() == {"deleted": True, "count": 0}


def test_get_single_history_item(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    created = create_history(client, dataset_id).get_json()

    response = client.get(f"/api/history/{created['history_id']}")
    assert response.status_code == 200
    item = response.get_json()
    assert item["history_id"] == created["history_id"]
    assert item["response_payload"] == created["response_payload"]

    missing = client.get("/api/history/not-history")
    assert missing.status_code == 404
    assert missing.get_json()["error"] == "History item not found."


def test_delete_history_item(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    created = create_history(client, dataset_id).get_json()

    response = client.delete(f"/api/history/{created['history_id']}")
    assert response.status_code == 200
    assert response.get_json() == {"deleted": True, "history_id": created["history_id"]}
    assert client.get("/api/history").get_json()["count"] == 0

    repeat = client.delete(f"/api/history/{created['history_id']}")
    assert repeat.status_code == 404


def test_old_and_malformed_records_do_not_crash(client, app):
    registry_path = Path(app.config["HISTORY_REGISTRY_PATH"])
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "old-id": {
                    "dataset_id": "deleted-dataset",
                    "dataset_filename": "old.csv",
                    "question": "Old question?",
                    "answer_text": "Old answer.",
                    "created_at": "2026-06-13T10:00:00",
                },
                "bad-id": "not a record",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 1
    item = data["history"][0]
    assert item["history_id"] == "old-id"
    assert item["response_payload"] is None
    assert item["dataset_status"] == "deleted"
    assert item["title"] == "Old question"


def test_invalid_history_registry_returns_clean_error(client, app):
    registry_path = Path(app.config["HISTORY_REGISTRY_PATH"])
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("{not valid json", encoding="utf-8")

    response = client.get("/api/history")
    assert response.status_code == 500
    assert response.get_json()["error"] == "History registry is invalid JSON."


def test_history_preserves_response_payload_exactly(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    payload = {
        "answer": {
            "title": "Consulting alumni",
            "summary": "1 alumnus matches consulting.",
            "blocks": [
                {"type": "metrics", "items": [{"label": "Alumni matching criteria", "value": "1"}]},
                {
                    "type": "table",
                    "columns": ["First Name", "Occupation", "Employer"],
                    "rows": [["Grace", "Consultant", "McKinsey"]],
                    "caption": "Searched columns: Occupation, Employer",
                },
            ],
            "followups": ["Show consulting-adjacent alumni"],
        },
        "answer_text": "1 alumnus matches consulting.",
        "operation": {"type": "contains_any", "params": {"filter_mode": "people"}},
        "result": {"status": "ok", "total_matches": 1, "visible_columns": ["First Name"]},
        "analysis_plan": {"operations": [{"type": "contains_any"}]},
        "operation_results": [{"status": "ok", "total_matches": 1}],
    }

    created = create_history(
        client,
        dataset_id,
        question="Who works in consulting?",
        answer_text=payload["answer_text"],
        response_payload=payload,
    ).get_json()
    fetched = client.get(f"/api/history/{created['history_id']}").get_json()
    listed = client.get("/api/history").get_json()["history"][0]

    assert created["response_payload"] == payload
    assert fetched["response_payload"] == payload
    assert listed["response_payload"] == payload


def test_history_sanitizes_internal_response_payload_columns(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    payload = {
        "question": "Show me alumni in tech",
        "answer": {
            "summary": "One match.",
            "blocks": [
                {
                    "type": "table",
                    "columns": ["First Name", "Major", "MATCH REASON", "eval_case_id"],
                    "rows": [["Ada", "Math", "matched title", "case-1"]],
                }
            ],
            "followups": [],
        },
        "result": {
            "columns": ["First Name", "Major", "MATCH REASON", "eval_case_id"],
            "rows": [["Ada", "Math", "matched title", "case-1"]],
            "debug": {"confidence": 0.99},
        },
    }

    created = create_history(
        client,
        dataset_id,
        question="Show me alumni in tech",
        answer_text="One match.",
        response_payload=payload,
    ).get_json()
    fetched = client.get(f"/api/history/{created['history_id']}").get_json()

    for item in (created, fetched):
        response_payload = item["response_payload"]
        assert response_payload["answer"]["blocks"][0]["columns"] == ["First Name"]
        assert response_payload["result"]["columns"] == ["First Name"]
        assert "debug" not in response_payload["result"]


def test_history_survives_restart_and_deleted_dataset_is_marked(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    created = create_history(client, dataset_id).get_json()

    restarted = configure(create_app(), tmp_path).test_client()
    assert restarted.get("/api/history").get_json()["history"][0]["history_id"] == created["history_id"]

    delete_response = restarted.delete(f"/api/datasets/{dataset_id}")
    assert delete_response.status_code == 200

    listing = restarted.get("/api/history")
    assert listing.status_code == 200
    item = listing.get_json()["history"][0]
    assert item["history_id"] == created["history_id"]
    assert item["dataset_status"] == "deleted"
    assert item["answer_text"] == created["answer_text"]
    assert item["response_payload"] == created["response_payload"]


def test_ask_success_automatically_creates_history(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")

    response = client.post("/api/ask", json={"dataset_id": dataset_id, "question": "Summarize this dataset."})
    assert response.status_code == 200
    ask_payload = response.get_json()
    assert ask_payload["history_item"]["history_id"]

    history = client.get("/api/history").get_json()
    assert history["count"] == 1
    item = history["history"][0]
    expected_payload = {
        key: ask_payload[key]
        for key in [
            "dataset_id",
            "question",
            "answer",
            "answer_text",
            "operation",
            "result",
            "analysis_intent",
            "analysis_plan",
            "operation_results",
        ]
    }
    assert item["question"] == "Summarize this dataset."
    assert item["dataset_id"] == dataset_id
    assert item["dataset_filename"] == "alumni.csv"
    assert item["response_payload"] == expected_payload
    assert item["metadata"]["row_count"] == 3
    assert item["metadata"]["column_count"] == 4


def test_ask_validation_and_missing_dataset_do_not_create_history(client):
    missing_question = client.post("/api/ask", json={"dataset_id": "missing"})
    assert missing_question.status_code == 400

    missing_dataset = client.post("/api/ask", json={"dataset_id": "missing", "question": "Summarize this dataset."})
    assert missing_dataset.status_code == 404

    assert client.get("/api/history").get_json()["count"] == 0


def test_ask_planner_failures_do_not_create_history(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")

    unsupported = client.post("/api/ask", json={"dataset_id": dataset_id, "question": "Write me a poem."})
    assert unsupported.status_code == 200
    assert unsupported.get_json()["operation"] is None

    mutation = client.post("/api/ask", json={"dataset_id": dataset_id, "question": "Delete all rows."})
    assert mutation.status_code == 200
    assert mutation.get_json()["operation"] is None

    assert client.get("/api/history").get_json()["count"] == 0


def test_save_history_item_as_insight(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    created = create_history(client, dataset_id).get_json()

    response = client.post(f"/api/history/{created['history_id']}/save-insight")
    assert response.status_code == 201
    insight = response.get_json()
    assert insight["dataset_id"] == dataset_id
    assert insight["question"] == created["question"]
    assert insight["answer_text"] == created["answer_text"]
    assert insight["response_payload"] == created["response_payload"]


def test_save_history_item_as_insight_requires_existing_dataset(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    created = create_history(client, dataset_id).get_json()
    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 200

    response = client.post(f"/api/history/{created['history_id']}/save-insight")
    assert response.status_code == 404
    assert "Dataset not found" in response.get_json()["error"]

    history = client.get(f"/api/history/{created['history_id']}")
    assert history.status_code == 200
    assert history.get_json()["dataset_status"] == "deleted"
