from io import BytesIO

import pandas as pd
import pytest

from app import create_app
from app.services import ai_service
from app.services.spreadsheet_service import DATASETS


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)
    DATASETS.clear()

    app = create_app()
    app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        DATA_FOLDER=str(tmp_path / "data"),
        DATASET_REGISTRY_PATH=str(tmp_path / "data" / "datasets.json"),
    )

    yield app

    DATASETS.clear()


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


def ask(client, dataset_id, question):
    response = client.post("/api/ask", json={"dataset_id": dataset_id, "question": question})
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()


def assert_valid_answer(data):
    answer = data["answer"]
    assert isinstance(answer, dict)
    assert isinstance(answer["summary"], str)
    assert isinstance(answer["blocks"], list)
    assert isinstance(answer["followups"], list)
    for block in answer["blocks"]:
        assert block["type"] in {"markdown", "table", "metrics", "ranked_list"}
    return answer


def table_blocks(answer):
    return [block for block in answer["blocks"] if block["type"] == "table"]


def metrics_labels(answer):
    labels = []
    for block in answer["blocks"]:
        if block["type"] == "metrics":
            labels.extend(item["label"] for item in block.get("items", []))
    return labels


def test_bad_planner_json_falls_back_safely(client, monkeypatch):
    class FakeResponses:
        def create(self, **kwargs):
            return type("FakeResponse", (), {"output_text": "this is not json"})()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(ai_service, "client", FakeClient())
    df = pd.DataFrame({"Name": ["A"], "Industry": ["Tech"]})
    dataset_id = upload_dataframe(client, df)

    data = ask(client, dataset_id, "Show me all alumni who work in tech")

    answer = assert_valid_answer(data)
    assert data["operation"]["type"] == "contains_any"
    assert data["operation_results"][0]["status"] == "ok"
    assert "invalid json" in " ".join(data["analysis_plan"]["assumptions"]).lower()


def test_ask_route_uses_full_persisted_dataset_not_preview_rows(client):
    rows = []
    for i in range(15):
        rows.append(
            {
                "Name": f"Person {i}",
                "Occupation": "Teacher",
                "Employer": "School",
                "Industry": "Education",
                "Major": "History",
            }
        )
    rows.append(
        {
            "Name": "Late Tech Match",
            "Occupation": "Software Engineer",
            "Employer": "AI Lab",
            "Industry": "Technology",
            "Major": "Computer Science",
        }
    )
    dataset_id = upload_dataframe(client, pd.DataFrame(rows), "full-data.csv")

    data = ask(client, dataset_id, "Show me all alumni who work in tech")

    answer = assert_valid_answer(data)
    assert data["operation"]["type"] == "contains_any"
    assert data["operation_results"][0]["status"] == "ok"
    flattened_rows = [" ".join(str(cell) for cell in row) for row in data["operation_results"][0]["rows"]]
    assert any("Late Tech Match" in row for row in flattened_rows)
    assert any(block["type"] == "table" for block in answer["blocks"])


def test_ask_route_searches_uppercase_occupation_and_employer_for_tech_alumni(client):
    rows = []
    for i in range(15):
        rows.append(
            {
                "NICKNAME": f"Person {i}",
                "OCCUPATION": "Teacher",
                "EMPLOYER": "School",
            }
        )
    rows.append({"NICKNAME": "Ada", "OCCUPATION": "Software Engineer", "EMPLOYER": "Community Org"})
    rows.append({"NICKNAME": "Grace", "OCCUPATION": "Product Manager", "EMPLOYER": "Google"})
    dataset_id = upload_dataframe(client, pd.DataFrame(rows), "uppercase-tech.csv")

    data = ask(
        client,
        dataset_id,
        "Which alumni work in tech as either software engineers or any other role in a tech company?",
    )

    answer = assert_valid_answer(data)
    result = data["operation_results"][0]
    assert data["operation"]["type"] == "contains_any"
    groups = data["operation"]["params"]["column_term_groups"]
    assert next(group for group in groups if group["concept"] == "software_engineer_role")["columns"] == ["OCCUPATION"]
    assert next(group for group in groups if group["concept"] == "tech_company")["columns"] == ["EMPLOYER"]
    assert result["status"] == "ok"
    assert result["is_filtered"] is True
    assert result["search_columns"] == ["OCCUPATION", "EMPLOYER"]
    assert result["display_columns"] == ["NICKNAME", "OCCUPATION", "EMPLOYER", "MATCH REASON"]
    names = [row[0] for row in result["rows"]]
    assert names == ["Ada", "Grace"]
    table = table_blocks(answer)[0]
    assert table["columns"] == ["NICKNAME", "OCCUPATION", "EMPLOYER", "MATCH REASON"]
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert "I treated tech alumni as records whose occupation or employer matched" in rendered
    assert "Searched columns: OCCUPATION, EMPLOYER" in rendered
    assert "no search terms" not in rendered.lower()
    assert "Unique alumni matched" in metrics_labels(answer)
    assert "Rows shown" in metrics_labels(answer)
    assert "Total dataset rows" in metrics_labels(answer)
    assert "Person 0" not in rendered


def test_tech_alumni_query_does_not_display_major_unless_requested(client):
    df = pd.DataFrame(
        {
            "NICKNAME": ["Ada", "Grace"],
            "OCCUPATION": ["Software Engineer", "Teacher"],
            "EMPLOYER": ["Community Org", "School"],
            "MAJOR": ["Computer Science", "History"],
        }
    )
    dataset_id = upload_dataframe(client, df, "major-default.csv")

    data = ask(client, dataset_id, "Show me tech alumni")

    answer = assert_valid_answer(data)
    result = data["operation_results"][0]
    assert "MAJOR" in result["search_columns"]
    assert result["display_columns"] == ["NICKNAME", "OCCUPATION", "EMPLOYER", "MATCH REASON"]
    assert table_blocks(answer)[0]["columns"] == ["NICKNAME", "OCCUPATION", "EMPLOYER", "MATCH REASON"]
    assert result["rows"][0][-1] == "Matched OCCUPATION: Software Engineer"


def test_tech_alumni_query_displays_major_when_requested(client):
    df = pd.DataFrame(
        {
            "NICKNAME": ["Ada", "Grace"],
            "OCCUPATION": ["Software Engineer", "Teacher"],
            "EMPLOYER": ["Community Org", "School"],
            "MAJOR": ["Computer Science", "History"],
        }
    )
    dataset_id = upload_dataframe(client, df, "major-requested.csv")

    data = ask(client, dataset_id, "Show me tech alumni and their majors")

    answer = assert_valid_answer(data)
    result = data["operation_results"][0]
    assert result["display_columns"] == ["NICKNAME", "OCCUPATION", "EMPLOYER", "MAJOR", "MATCH REASON"]
    assert table_blocks(answer)[0]["columns"] == ["NICKNAME", "OCCUPATION", "EMPLOYER", "MAJOR", "MATCH REASON"]


def test_display_limit_is_explained_when_rows_are_capped(client):
    df = pd.DataFrame(
        {
            "NICKNAME": ["Ada", "Grace", "Linus"],
            "OCCUPATION": ["Software Engineer", "Software Engineer", "Software Engineer"],
            "EMPLOYER": ["A", "B", "C"],
        }
    )
    dataset_id = upload_dataframe(client, df, "limit.csv")

    data = ask(client, dataset_id, "Show me the 1 tech alumni")

    answer = assert_valid_answer(data)
    result = data["operation_results"][0]
    assert result["matched_row_count"] == 3
    assert result["returned_row_count"] == 1
    assert result["display_limit"] == 1
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert "Display limit" in rendered
    assert "Showing 1 rows because the display limit is 1" in rendered


def test_presenter_invalid_json_falls_back_to_structured_answer(client, monkeypatch):
    class FakeResponses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return type(
                    "FakeResponse",
                    (),
                    {
                        "output_text": '{"intent":"rank_records","target_entity":"rows","user_goal":"Find top donors by lifetime giving.","concepts":[],"semantic_columns":{"person_name":["name"],"lifetime_giving":["lifetime giving"]},"filters":[],"sort":{"semantic_column":"lifetime_giving","direction":"desc"},"aggregation":null,"desired_output":{"format":"ranked_list","semantic_columns":["person_name","lifetime_giving"],"limit":2},"assumptions":[],"clarification_needed":false,"clarifying_question":null}'
                    },
                )()
            return type("FakeResponse", (), {"output_text": "not presenter json"})()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(ai_service, "client", FakeClient())
    df = pd.DataFrame(
        {
            "Name": ["A", "B", "C"],
            "Lifetime Giving": ["$100", "$18,500", "$4,200"],
        }
    )
    dataset_id = upload_dataframe(client, df, "presenter-fallback.csv")

    data = ask(client, dataset_id, "Who are the top donors by lifetime giving?")

    answer = assert_valid_answer(data)
    assert data["operation"]["type"] == "top_n"
    assert data["operation_results"][0]["rows"][0][0] == "B"
    assert any(block["type"] in {"table", "ranked_list"} for block in answer["blocks"])


def test_presenter_answer_keeps_backend_assumptions(client, monkeypatch):
    class FakeResponses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return type(
                    "FakeResponse",
                    (),
                    {
                        "output_text": '{"intent":"find_records","target_entity":"rows","user_goal":"Show tech alumni.","concepts":[{"name":"tech_related","definition":"Technology-related people.","search_terms":["software","engineer"],"known_entities":["Google"]}],"semantic_columns":{"person_name":["name"],"occupation":["occupation"],"employer":["employer","company"],"industry":["industry"],"major":["major"]},"filters":[{"concept":"tech_related","apply_to_semantic_columns":["occupation","employer","industry","major"],"match_mode":"contains_any"}],"sort":null,"aggregation":null,"desired_output":{"format":"table","semantic_columns":["person_name","occupation","employer","matched_reason"],"limit":100},"assumptions":["Tech-related alumni are identified using occupation, employer, industry, or major text."],"clarification_needed":false,"clarifying_question":null}'
                    },
                )()
            return type(
                "FakeResponse",
                (),
                {
                    "output_text": '{"answer":{"title":"Tech alumni","summary":"Found matching rows.","blocks":[{"type":"markdown","content":"Found matching rows."}],"followups":[]}}'
                },
            )()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(ai_service, "client", FakeClient())
    df = pd.DataFrame(
        {
            "Name": ["Ada"],
            "OCCUPATION": ["Software Engineer"],
            "EMPLOYER": ["Google"],
        }
    )
    dataset_id = upload_dataframe(client, df, "assumptions.csv")

    data = ask(client, dataset_id, "Show me tech alumni")

    answer = assert_valid_answer(data)
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert "Tech-related alumni are identified" in rendered
    assert "occupation -> OCCUPATION" in rendered
