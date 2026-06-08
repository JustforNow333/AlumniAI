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
    df = pd.DataFrame({"Name": ["A"], "Occupation": ["Software Engineer"], "Employer": ["Local Bakery"]})
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
    flattened_rows = [" ".join(str(cell) for cell in row.values()) for row in data["operation_results"][0]["rows"]]
    assert any("Late Tech Match" in row for row in flattened_rows)
    assert any(block["type"] == "table" for block in answer["blocks"])


def test_ask_route_searches_uppercase_occupation_and_employer_for_tech_alumni(client):
    rows = []
    for i in range(15):
        rows.append(
            {
                "First Name": f"Person{i}",
                "LastName": "Example",
                "NICKNAME": f"Person {i}",
                "OCCUPATION": "Teacher",
                "EMPLOYER": "School",
                "LinkedinURL": "",
            }
        )
    rows.append(
        {
            "First Name": "Ada",
            "LastName": "Lovelace",
            "NICKNAME": "Ada",
            "OCCUPATION": "Software Engineer",
            "EMPLOYER": "Community Org",
            "LinkedinURL": "https://linkedin.com/in/ada",
        }
    )
    rows.append(
        {
            "First Name": "Grace",
            "LastName": "Hopper",
            "NICKNAME": "Grace",
            "OCCUPATION": "Product Manager",
            "EMPLOYER": "Google",
            "LinkedinURL": "",
        }
    )
    dataset_id = upload_dataframe(client, pd.DataFrame(rows), "uppercase-tech.csv")

    data = ask(
        client,
        dataset_id,
        "Which alumni work in tech as either software engineers or any other role in a tech company?",
    )

    answer = assert_valid_answer(data)
    result = data["operation_results"][0]
    assert data["operation"]["type"] == "contains_any"
    assert data["operation"]["params"]["filter_mode"] == "tech_people"
    groups = data["operation"]["params"]["column_term_groups"]
    assert next(group for group in groups if group["concept"] == "software_engineer_role")["columns"] == ["OCCUPATION"]
    assert next(group for group in groups if group["concept"] == "tech_company")["columns"] == ["EMPLOYER"]
    assert result["status"] == "ok"
    assert result["is_filtered"] is True
    assert result["search_columns"] == ["OCCUPATION", "EMPLOYER"]
    assert result["intent"] == "people_filter"
    assert result["entity"] == "alumni"
    assert result["criteria_label"] == "working in tech or technical roles"
    assert result["answer_label"] == "Alumni matching criteria"
    assert result["total_matches"] == 2
    assert result["displayed_count"] == 2
    assert result["display_limit"] == 100
    assert result["display_columns"] == ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"]
    first_names = [row["First Name"] for row in result["rows"]]
    assert first_names == ["Ada", "Grace"]
    assert result["rows"][0]["LinkedIn URL"] == "https://linkedin.com/in/ada"
    assert result["rows"][1]["LinkedIn URL"] == ""
    table = table_blocks(answer)[0]
    assert table["columns"] == ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"]
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert "I used the default alumni tech filter" in rendered
    assert "Searched columns: OCCUPATION, EMPLOYER" in rendered
    assert "no search terms" not in rendered.lower()
    assert "Alumni matching criteria" in metrics_labels(answer)
    assert "Rows shown" not in metrics_labels(answer)
    assert "Display limit" not in metrics_labels(answer)
    assert "NICKNAME" not in rendered
    assert "MATCH REASON" not in rendered
    assert "Person 0" not in rendered


def test_tech_alumni_query_does_not_display_major_unless_requested(client):
    df = pd.DataFrame(
        {
            "First Name": ["Ada", "Grace"],
            "Last Name": ["Lovelace", "Hopper"],
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
    assert result["display_columns"] == ["First Name", "Last Name", "Occupation", "Employer"]
    assert table_blocks(answer)[0]["columns"] == ["First Name", "Last Name", "Occupation", "Employer"]
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert "MAJOR" not in table_blocks(answer)[0]["columns"]
    assert "MATCH REASON" not in rendered
    assert "NICKNAME" not in rendered


def test_tech_alumni_query_displays_major_when_requested(client):
    df = pd.DataFrame(
        {
            "First Name": ["Ada", "Grace"],
            "LastName": ["Lovelace", "Hopper"],
            "NICKNAME": ["Ada", "Grace"],
            "OCCUPATION": ["Software Engineer", "Teacher"],
            "EMPLOYER": ["Community Org", "School"],
            "MAJOR": ["Computer Science", "History"],
            "LinkedIn": ["linkedin.com/in/ada", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "major-requested.csv")

    data = ask(client, dataset_id, "Show me tech alumni and their majors")

    answer = assert_valid_answer(data)
    result = data["operation_results"][0]
    assert result["display_columns"] == ["First Name", "Last Name", "Occupation", "Employer", "MAJOR", "LinkedIn URL"]
    assert table_blocks(answer)[0]["columns"] == ["First Name", "Last Name", "Occupation", "Employer", "MAJOR", "LinkedIn URL"]
    assert result["rows"][0]["LinkedIn URL"] == "linkedin.com/in/ada"


@pytest.mark.parametrize(
    "question",
    [
        "How many alumni are working in tech either as software engineers or as other roles in a tech company",
        "what alumni work in tech as either a software engineer or some other role in a tech company",
        "show me alumni in software engineering or at tech companies",
        "which alumni are in tech",
    ],
)
def test_broad_alumni_tech_queries_do_not_return_analysis_plan_errors(client, question):
    df = pd.DataFrame(
        {
            "First Name": ["Ada", "Grace", "Marie"],
            "Last Name": ["Lovelace", "Hopper", "Curie"],
            "Occupation": ["Software Engineer", "Founder", "Director of Hematologic Oncology"],
            "Employer": ["Local Bakery", "Google", "Holy Name Medical Center"],
            "LinkedIn URL": ["linkedin.com/in/ada", "", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "broad-tech.csv")

    data = ask(client, dataset_id, question)

    answer = assert_valid_answer(data)
    result = data["result"]
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert answer.get("title") != "Analysis Plan Error"
    assert "could not create a valid analysis plan" not in rendered.lower()
    assert result["intent"] == "people_filter"
    assert result["entity"] == "alumni"
    assert "total_matches" in result
    assert "displayed_count" in result
    assert "rows" in result
    assert result["visible_columns"] == ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"]
    assert result["total_matches"] == 2


def test_model_clarification_for_alumni_tech_query_uses_default_people_filter(client, monkeypatch):
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
                        "output_text": '{"intent":"unknown","target_entity":"dataset","user_goal":"How many alumni are working in tech either as software engineers or other roles in a tech company?","concepts":[],"semantic_columns":{},"filters":[],"sort":null,"aggregation":null,"desired_output":{"format":"markdown","semantic_columns":[],"limit":100},"assumptions":[],"clarification_needed":true,"clarifying_question":"Should I count only explicit software engineer titles and clearly identifiable tech companies, or use a broader keyword-based definition?"}'
                    },
                )()
            return type("FakeResponse", (), {"output_text": "not presenter json"})()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(ai_service, "client", FakeClient())
    df = pd.DataFrame(
        {
            "First Name": ["Ada", "Grace", "Erin"],
            "Last Name": ["Lovelace", "Hopper", "Doctor"],
            "Occupation": ["Software Engineer", "Founder", "Director of Hematologic Oncology"],
            "Employer": ["Local Bakery", "FanAmp", "Holy Name Medical Center"],
            "LinkedInURL": ["linkedin.com/in/ada", "", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "clarification-fallback.csv")

    data = ask(
        client,
        dataset_id,
        "How many alumni are working in tech either as software engineers or as other roles in a tech company?",
    )

    answer = assert_valid_answer(data)
    result = data["result"]
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert answer.get("title") != "Analysis Plan Error"
    assert "could not create a valid analysis plan" not in rendered.lower()
    assert data["analysis_intent"]["intent"] == "people_filter"
    assert data["operation"]["params"]["filter_mode"] == "tech_people"
    assert result["intent"] == "people_filter"
    assert result["entity"] == "alumni"
    assert result["total_matches"] == 2
    assert result["uncertain_count"] == 0
    assert [row["First Name"] for row in result["rows"]] == ["Ada", "Grace"]


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
    assert result["total_matches"] == 3
    assert result["displayed_count"] == 1
    assert result["display_limit"] == 1
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert "Alumni matching criteria" in rendered
    assert "Showing" in rendered
    assert "Display limit" not in rendered


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
