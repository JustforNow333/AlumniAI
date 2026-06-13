"""Saved insights: manually saved AI answers tied to a dataset.

Insights are snapshots — the answer is never recomputed, and nothing is saved
automatically (history is a separate, future feature).
"""

from io import BytesIO

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
    payload = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )
    assert response.status_code in {200, 201}, response.get_data(as_text=True)
    return response.get_json()["dataset_id"]


def sample_df(rows=2):
    return pd.DataFrame(
        {
            "First Name": [f"Person{i}" for i in range(rows)],
            "Occupation": ["Software Engineer"] * rows,
            "Employer": ["Acme"] * rows,
        }
    )


def save_insight(client, dataset_id, **overrides):
    payload = {
        "dataset_id": dataset_id,
        "title": "Alumni working in tech",
        "question": "Which alumni work in tech?",
        "answer": "2 alumni work in tech: Person0 and Person1.",
    }
    payload.update(overrides)
    response = client.post("/api/insights", json=payload)
    return response


def test_create_insight_returns_full_snapshot(client):
    dataset_id = upload_dataframe(client, sample_df(3), "alumni.csv")

    response = save_insight(client, dataset_id)
    assert response.status_code == 201
    insight = response.get_json()
    assert insight["insight_id"]
    assert insight["dataset_id"] == dataset_id
    assert insight["dataset_name_snapshot"] == "alumni.csv"
    assert insight["dataset_status"] == "ready"
    assert insight["title"] == "Alumni working in tech"
    assert insight["question"] == "Which alumni work in tech?"
    assert insight["answer"].startswith("2 alumni work in tech")
    assert insight["answer_text"] == insight["answer"]
    assert insight["response_payload"] is None
    assert insight["dataset_filename"] == "alumni.csv"
    assert insight["created_at"]
    assert insight["updated_at"] == insight["created_at"]
    assert insight["tags"] == []
    assert insight["metadata"]["row_count"] == 3
    assert insight["metadata"]["column_count"] == 3


def test_create_insight_persists_full_response_payload(client):
    dataset_id = upload_dataframe(client, sample_df(3), "alumni.csv")
    response_payload = {
        "answer": {
            "title": "Tech alumni",
            "summary": "2 alumni match tech criteria.",
            "blocks": [
                {"type": "metrics", "items": [{"label": "Alumni matching criteria", "value": "2"}]},
                {
                    "type": "table",
                    "columns": ["First Name", "Occupation", "Employer"],
                    "rows": [["Person0", "Software Engineer", "Acme"], ["Person1", "Software Engineer", "Acme"]],
                    "caption": "Searched columns: Occupation, Employer",
                },
                {"type": "markdown", "content": "Assumptions: strict people filter."},
            ],
            "followups": [],
        },
        "answer_text": "2 alumni match tech criteria.",
        "operation": {"type": "contains_any"},
        "result": {
            "intent": "people_filter",
            "entity": "alumni",
            "total_matches": 2,
            "uncertain_count": 1,
            "adjacent_count": 4,
            "total_dataset_rows": 3,
        },
        "metadata": {"searched_columns": ["Occupation", "Employer"]},
    }

    created = save_insight(client, dataset_id, response_payload=response_payload).get_json()
    assert created["response_payload"]["answer"]["blocks"][1]["columns"] == ["First Name", "Occupation", "Employer"]
    assert created["response_payload"]["answer"]["blocks"][1]["rows"][0][0] == "Person0"
    assert created["response_payload"]["result"]["total_matches"] == 2

    fetched = client.get(f"/api/insights/{created['insight_id']}").get_json()
    assert fetched["response_payload"] == created["response_payload"]

    listed = client.get("/api/insights").get_json()["insights"][0]
    assert listed["response_payload"] == created["response_payload"]


def test_create_insight_generates_title_from_question_when_missing(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")

    response = save_insight(client, dataset_id, title="", question="Which alumni work in consulting?")
    assert response.status_code == 201
    assert response.get_json()["title"] == "Which alumni work in consulting"


def test_create_insight_validation_errors_are_clean(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")

    missing_question = save_insight(client, dataset_id, question="  ")
    assert missing_question.status_code == 400
    assert "question" in missing_question.get_json()["error"]

    missing_answer = save_insight(client, dataset_id, answer="")
    assert missing_answer.status_code == 400
    assert "answer" in missing_answer.get_json()["error"]

    missing_dataset = save_insight(client, "")
    assert missing_dataset.status_code == 400
    assert "dataset_id" in missing_dataset.get_json()["error"]

    unknown_dataset = save_insight(client, "not-a-dataset")
    assert unknown_dataset.status_code == 404
    assert "Dataset not found" in unknown_dataset.get_json()["error"]


def test_list_insights_newest_first(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    first = save_insight(client, dataset_id, title="First").get_json()
    second = save_insight(client, dataset_id, title="Second").get_json()

    response = client.get("/api/insights")
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 2
    # Newest first, even when both saves land within the same second.
    assert [item["insight_id"] for item in data["insights"]] == [second["insight_id"], first["insight_id"]]
    for item in data["insights"]:
        for key in ["insight_id", "dataset_id", "dataset_name_snapshot", "title", "question", "answer", "created_at", "updated_at", "tags"]:
            assert key in item


def test_list_insights_filters_by_dataset_id(client):
    dataset_a = upload_dataframe(client, sample_df(), "a.csv")
    dataset_b = upload_dataframe(client, sample_df(), "b.csv")
    insight_a = save_insight(client, dataset_a, title="A").get_json()
    save_insight(client, dataset_b, title="B")

    response = client.get(f"/api/insights?dataset_id={dataset_a}")
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 1
    assert data["insights"][0]["insight_id"] == insight_a["insight_id"]
    assert data["insights"][0]["title"] == "A"

    empty = client.get("/api/insights?dataset_id=no-such-dataset").get_json()
    assert empty == {"insights": [], "count": 0}


def test_get_single_insight_and_clean_404(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    created = save_insight(client, dataset_id).get_json()

    response = client.get(f"/api/insights/{created['insight_id']}")
    assert response.status_code == 200
    insight = response.get_json()
    assert insight["insight_id"] == created["insight_id"]
    assert insight["question"] == "Which alumni work in tech?"
    assert insight["answer"] == created["answer"]

    missing = client.get("/api/insights/not-an-insight")
    assert missing.status_code == 404
    assert missing.get_json()["error"] == "Saved insight not found."


def test_patch_updates_title_tags_and_updated_at(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    created = save_insight(client, dataset_id).get_json()

    response = client.patch(
        f"/api/insights/{created['insight_id']}",
        json={"title": "Tech alumni snapshot", "tags": ["tech", "  tech ", "alumni"]},
    )
    assert response.status_code == 200
    updated = response.get_json()
    assert updated["title"] == "Tech alumni snapshot"
    assert updated["tags"] == ["tech", "alumni"]
    assert updated["updated_at"] >= created["updated_at"]
    # The snapshot fields are immutable.
    assert updated["question"] == created["question"]
    assert updated["answer"] == created["answer"]
    assert updated["dataset_id"] == created["dataset_id"]

    empty_title = client.patch(f"/api/insights/{created['insight_id']}", json={"title": "   "})
    assert empty_title.status_code == 400

    nothing = client.patch(f"/api/insights/{created['insight_id']}", json={})
    assert nothing.status_code == 400

    unknown = client.patch("/api/insights/not-an-insight", json={"title": "X"})
    assert unknown.status_code == 404


def test_delete_insight_removes_it_and_repeat_is_404(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    created = save_insight(client, dataset_id).get_json()

    response = client.delete(f"/api/insights/{created['insight_id']}")
    assert response.status_code == 200
    assert response.get_json() == {"deleted": True, "insight_id": created["insight_id"]}

    assert client.get("/api/insights").get_json()["count"] == 0
    repeat = client.delete(f"/api/insights/{created['insight_id']}")
    assert repeat.status_code == 404
    assert repeat.get_json()["error"] == "Saved insight not found."


def test_insights_survive_app_restart(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    created = save_insight(client, dataset_id).get_json()

    restarted = configure(create_app(), tmp_path).test_client()
    data = restarted.get("/api/insights").get_json()
    assert data["count"] == 1
    assert data["insights"][0]["insight_id"] == created["insight_id"]
    assert data["insights"][0]["answer"] == created["answer"]


def test_insight_for_deleted_dataset_is_kept_and_marked(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")
    created = save_insight(client, dataset_id).get_json()

    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 200

    listing = client.get("/api/insights")
    assert listing.status_code == 200
    item = listing.get_json()["insights"][0]
    assert item["insight_id"] == created["insight_id"]
    assert item["dataset_status"] == "deleted"
    assert item["dataset_name_snapshot"] == "alumni.csv"
    assert item["answer"] == created["answer"]

    single = client.get(f"/api/insights/{created['insight_id']}")
    assert single.status_code == 200
    assert single.get_json()["dataset_status"] == "deleted"


def test_saving_is_manual_only_ask_does_not_create_insights(client):
    dataset_id = upload_dataframe(client, sample_df(), "alumni.csv")

    ask = client.post("/api/ask", json={"dataset_id": dataset_id, "question": "Which alumni work in tech?"})
    assert ask.status_code == 200

    assert client.get("/api/insights").get_json()["count"] == 0
