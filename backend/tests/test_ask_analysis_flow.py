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
    ("question", "expect_major"),
    [
        ("Show me alumni who work in tech.", False),
        ("Show me alumni who work in tech and include their majors.", True),
    ],
)
def test_ask_response_boundary_sanitizes_internal_columns(client, monkeypatch, question, expect_major):
    df = pd.DataFrame(
        {
            "First Name": ["Ada"],
            "Last Name": ["Lovelace"],
            "Employer": ["Google"],
            "Title": ["Software Engineer"],
            "Major": ["Mathematics"],
            "LinkedIn URL": ["linkedin.com/in/ada"],
            "MATCH REASON": ["source data should not leak"],
            "expected_industry": ["Tech"],
        }
    )
    dataset_id = upload_dataframe(client, df, "unsafe-display-columns.csv")

    unsafe_result = {
        "operation_type": "contains_any",
        "status": "ok",
        "is_filtered": True,
        "summary": "One matching alumnus.",
        "columns": [
            "First Name",
            "Last Name",
            "Employer",
            "Title",
            "Major",
            "LinkedIn URL",
            "MATCH REASON",
            "expected_industry",
        ],
        "rows": [
            [
                "Ada",
                "Lovelace",
                "Google",
                "Software Engineer",
                "Mathematics",
                "linkedin.com/in/ada",
                "matched title",
                "Tech",
            ]
        ],
        "metrics": {"matched_row_count": 1, "returned_row_count": 1, "total_rows": 1},
        "debug": {
            "rows": [
                {
                    "classification": "direct_match",
                    "confidence": 0.99,
                    "internal_reason": "matched title",
                }
            ]
        },
    }
    monkeypatch.setattr(
        "app.routes.chat_routes.execute_analysis_plan",
        lambda _df, _plan: [unsafe_result],
    )

    data = ask(client, dataset_id, question)

    result = data["operation_results"][0]
    table = table_blocks(data["answer"])[0]
    all_column_sets = [
        result["columns"],
        data["result"]["columns"],
        table["columns"],
    ]
    for columns in all_column_sets:
        normalized = {str(column).lower().replace("_", " ") for column in columns}
        assert "match reason" not in normalized
        assert "expected industry" not in normalized
        assert ("major" in normalized) is expect_major
    assert len(result["rows"]) == 1
    assert "debug" not in result
    assert "classification" not in str(data["operation_results"]).lower()


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


def test_consulting_query_uses_taxonomy_without_plan_error(client):
    df = pd.DataFrame(
        {
            "First Name": ["Pat", "Ada", "Sam", "Lee"],
            "Last Name": ["Partner", "Lovelace", "Strategy", "Analyst"],
            "Occupation": ["Partner", "Software Engineer", "Strategy Consultant", "Analyst"],
            "Employer": ["McKinsey", "Google", "Family Business", "Local Bakery"],
            "LinkedIn URL": ["linkedin.com/in/pat", "", "", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "consulting.csv")

    data = ask(client, dataset_id, "Which alumni work in consulting?")

    answer = assert_valid_answer(data)
    result = data["result"]
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert answer.get("title") != "Analysis Plan Error"
    assert "could not create a valid analysis plan" not in rendered.lower()
    assert data["operation"]["params"]["filter_mode"] == "people"
    assert result["intent"] == "people_filter"
    assert result["filter_type"] == "industry"
    assert result["industry"] == "consulting"
    assert result["total_matches"] == 2
    assert result["visible_columns"] == ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"]
    first_names = {row["First Name"] for row in result["rows"]}
    assert first_names == {"Pat", "Sam"}


def consulting_precision_df():
    return pd.DataFrame(
        {
            "First Name": ["Riley", "Kim", "Pat", "Ind", "Tess", "Hank", "Sofi", "Zoe", "Mona", "Ann", "Jud", "Ivan", "Pri"],
            "Last Name": [
                "Risk", "Consult", "Advisory", "Solo", "Tech",
                "Hershey", "Spotify", "Zoom", "Morgan", "Attorney", "Clerk", "Banker", "Equity",
            ],
            "Occupation": [
                "Senior Manager, Risk Consulting",
                "Management Consultant",
                "Transaction Advisory",
                "Independent Consultant",
                "Technology Consultant",
                "Head of Strategy",
                "Director, Premium Subscription Strategy",
                "Director, Product Strategy & Operations",
                "Product Manager",
                "Attorney",
                "Judicial Law Clerk",
                "Investment Banking Analyst",
                "Private Equity Associate",
            ],
            "Employer": [
                "EY",
                "KPMG",
                "Deloitte",
                "",
                "Acme Corp",
                "Hershey",
                "Spotify",
                "ZoomInfo",
                "Morgan Stanley",
                "Smith Law",
                "District Court",
                "Goldman Sachs",
                "Blackstone",
            ],
            "LinkedIn URL": ["linkedin.com/in/riley"] + [""] * 12,
        }
    )


def test_consulting_query_counts_direct_matches_only_not_keyword_hits(client):
    dataset_id = upload_dataframe(client, consulting_precision_df(), "consulting-precision.csv")

    data = ask(client, dataset_id, "What alumni work in consulting?")

    answer = assert_valid_answer(data)
    result = data["result"]
    assert data["operation"]["params"]["filter_mode"] == "people"
    assert result["industry"] == "consulting"
    # Only the five real consulting/advisory rows count.
    assert result["total_matches"] == 5
    first_names = {row["First Name"] for row in result["rows"]}
    assert first_names == {"Riley", "Kim", "Pat", "Ind", "Tess"}
    # Strategy/product/finance/legal rows matched broad keywords but are excluded.
    for excluded in ["Hank", "Sofi", "Zoe", "Mona", "Ann", "Jud", "Ivan", "Pri"]:
        assert excluded not in first_names
    # Broad retrieval found more candidates than direct matches.
    assert result["raw_candidate_count"] > result["total_matches"]
    assert result["direct_match_count"] == 5
    assert result["adjacent_count"] >= 4
    assert result["classification_version"] == "multi_label_v1"
    # Adjacent/non-match counts stay out of the headline count.
    assert result["total_matches"] == result["direct_match_count"]
    assert result["visible_columns"] == ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"]
    assert result["rows"][0]["LinkedIn URL"] == "linkedin.com/in/riley"
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert "adjacent rows matched broad keywords but were not counted" in rendered
    assert "internal_reason" not in rendered


def test_consulting_or_strategy_query_includes_internal_strategy_roles(client):
    dataset_id = upload_dataframe(client, consulting_precision_df(), "consulting-or-strategy.csv")

    data = ask(client, dataset_id, "What alumni work in consulting or strategy?")

    result = data["result"]
    assert_valid_answer(data)
    first_names = {row["First Name"] for row in result["rows"]}
    # Direct consulting plus internal strategy roles.
    assert {"Riley", "Kim", "Pat", "Ind", "Tess", "Hank", "Sofi"}.issubset(first_names)
    # Legal and pure finance rows still excluded.
    assert "Ann" not in first_names
    assert "Ivan" not in first_names


def test_consulting_adjacent_query_includes_adjacent_rows(client):
    dataset_id = upload_dataframe(client, consulting_precision_df(), "consulting-adjacent.csv")

    data = ask(client, dataset_id, "Show consulting-adjacent alumni too")

    result = data["result"]
    assert_valid_answer(data)
    first_names = {row["First Name"] for row in result["rows"]}
    # Adjacent strategy/product rows are now included and reported separately.
    assert {"Hank", "Sofi", "Zoe"}.issubset(first_names)
    assert result["adjacent_included_count"] >= 3
    assert result["adjacent_included"] is True
    # Clear non-matches stay out even in adjacent mode.
    assert "Ann" not in first_names
    assert "Jud" not in first_names


def test_finance_query_returns_finance_roles_even_if_not_consulting(client):
    dataset_id = upload_dataframe(client, consulting_precision_df(), "finance-roles.csv")

    data = ask(client, dataset_id, "What alumni work in finance?")

    result = data["result"]
    assert_valid_answer(data)
    assert result["industry"] == "finance"
    first_names = {row["First Name"] for row in result["rows"]}
    assert {"Ivan", "Pri", "Mona"}.issubset(first_names)
    assert "Ann" not in first_names


def test_finance_consulting_query_returns_only_the_intersection(client):
    dataset_id = upload_dataframe(client, consulting_precision_df(), "finance-consulting.csv")

    data = ask(client, dataset_id, "Who works in finance consulting?")

    result = data["result"]
    assert_valid_answer(data)
    assert result["industry"] == "consulting"
    first_names = {row["First Name"] for row in result["rows"]}
    # Consulting rows with finance context (risk consulting, transaction advisory,
    # Big Four professional services) count; pure finance does not.
    assert "Riley" in first_names
    assert "Pat" in first_names
    assert "Ivan" not in first_names
    assert "Pri" not in first_names
    assert "Mona" not in first_names


def test_confident_model_keyword_plan_for_consulting_still_uses_strict_classifier(client, monkeypatch):
    """Even when the intent model confidently returns a broad keyword search,
    a people/industry question must run through the strict classifier so raw
    keyword hits are not presented as final matches."""

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
                        "output_text": '{"intent":"find_records","target_entity":"rows","user_goal":"What alumni work in consulting?","concepts":[{"name":"consulting_related","definition":"Consulting-related people.","search_terms":["consultant","consulting","strategy","management","operations","advisory","transaction"],"known_entities":["McKinsey","Deloitte"]}],"semantic_columns":{"occupation":["occupation"],"employer":["employer"]},"filters":[{"concept":"consulting_related","apply_to_semantic_columns":["occupation","employer"],"match_mode":"contains_any"}],"sort":null,"aggregation":null,"desired_output":{"format":"table","semantic_columns":["first_name","last_name","occupation","employer"],"limit":100},"assumptions":[],"clarification_needed":false,"clarifying_question":null}'
                    },
                )()
            return type("FakeResponse", (), {"output_text": "not presenter json"})()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(ai_service, "client", FakeClient())
    dataset_id = upload_dataframe(client, consulting_precision_df(), "model-keyword-consulting.csv")

    data = ask(client, dataset_id, "What alumni work in consulting?")

    result = data["result"]
    assert_valid_answer(data)
    assert data["operation"]["params"]["filter_mode"] == "people"
    assert result["intent"] == "people_filter"
    assert result["industry"] == "consulting"
    assert result["total_matches"] == 5
    first_names = {row["First Name"] for row in result["rows"]}
    assert first_names == {"Riley", "Kim", "Pat", "Ind", "Tess"}


def test_investment_banking_query_excludes_generic_analyst(client):
    df = pd.DataFrame(
        {
            "First Name": ["Gail", "Lee", "Mia"],
            "Last Name": ["Golden", "Analyst", "Merger"],
            "Occupation": ["Analyst", "Analyst", "M&A Associate"],
            "Employer": ["Goldman Sachs", "Community Food Pantry", "Evercore"],
            "LinkedIn URL": ["", "", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "banking.csv")

    data = ask(client, dataset_id, "Which alumni work in investment banking?")

    result = data["result"]
    assert_valid_answer(data)
    assert result["industry"] == "banking"
    assert result["total_matches"] == 2
    assert {row["First Name"] for row in result["rows"]} == {"Gail", "Mia"}


def test_employer_query_uses_employer_filter_not_industry(client):
    df = pd.DataFrame(
        {
            "First Name": ["Neil", "Ada", "Sam"],
            "Last Name": ["Wusu", "Lovelace", "Stream"],
            "Occupation": ["Head of Growth", "Software Engineer", "Recruiter"],
            "Employer": ["Spotify", "Google", "Spotify"],
            "LinkedIn URL": ["linkedin.com/in/neil", "", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "employer.csv")

    data = ask(client, dataset_id, "Who works at Spotify?")

    result = data["result"]
    assert_valid_answer(data)
    assert result["intent"] == "people_filter"
    assert result["filter_type"] == "employer"
    assert result["industry"] is None
    # All Spotify rows are returned regardless of occupation.
    assert result["total_matches"] == 2
    assert {row["First Name"] for row in result["rows"]} == {"Neil", "Sam"}


def test_founders_query_uses_occupation_filter_regardless_of_industry(client):
    df = pd.DataFrame(
        {
            "First Name": ["Bo", "Fi", "Ada"],
            "Last Name": ["Baker", "Founder", "Lovelace"],
            "Occupation": ["Founder", "Co-Founder", "Software Engineer"],
            "Employer": ["Local Bakery", "FanAmp", "Google"],
            "LinkedIn URL": ["", "", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "founders.csv")

    data = ask(client, dataset_id, "Who are founders?")

    result = data["result"]
    assert_valid_answer(data)
    assert result["filter_type"] == "occupation"
    assert result["total_matches"] == 2
    assert {row["First Name"] for row in result["rows"]} == {"Bo", "Fi"}


def test_startup_founders_query_requires_startup_like_employer(client):
    df = pd.DataFrame(
        {
            "First Name": ["Bo", "Fi"],
            "Last Name": ["Baker", "Founder"],
            "Occupation": ["Founder", "Founder"],
            "Employer": ["Local Bakery", "FanAmp"],
            "LinkedIn URL": ["", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "startup-founders.csv")

    data = ask(client, dataset_id, "Which alumni are startup founders?")

    result = data["result"]
    assert_valid_answer(data)
    assert result["filter_type"] == "industry"
    assert result["industry"] == "startups"
    assert result["total_matches"] == 1
    assert result["rows"][0]["First Name"] == "Fi"


def test_neil_wusu_at_spotify_is_included_in_tech_query(client):
    df = pd.DataFrame(
        {
            "First Name": ["Neil", "Marie", "Bo"],
            "LastName": ["Wusu", "Curie", "Baker"],
            "Occupation": ["Head of Growth", "Director of Hematologic Oncology", "Founder"],
            "Employer": ["Spotify", "Holy Name Medical Center", "Local Bakery"],
            "LinkedinURL": ["https://linkedin.com/in/neilwusu", "", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "neil.csv")

    data = ask(client, dataset_id, "Which alumni are in tech?")

    result = data["result"]
    assert_valid_answer(data)
    assert result["total_matches"] == 1
    row = result["rows"][0]
    assert row["First Name"] == "Neil"
    assert row["Last Name"] == "Wusu"
    assert row["LinkedIn URL"] == "https://linkedin.com/in/neilwusu"


def test_healthcare_query_includes_hospital_rows_excluded_from_tech(client):
    df = pd.DataFrame(
        {
            "First Name": ["Marie", "Bob", "Ada"],
            "Last Name": ["Curie", "Hopper", "Lovelace"],
            "Occupation": ["Director of Hematologic Oncology", "Data Scientist", "Founder"],
            "Employer": ["Holy Name Medical Center", "Hospital for Special Surgery", "Local Bakery"],
            "LinkedIn URL": ["", "", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "healthcare.csv")

    data = ask(client, dataset_id, "Which alumni work in healthcare?")

    result = data["result"]
    assert_valid_answer(data)
    assert result["industry"] == "healthcare"
    assert result["total_matches"] == 2
    assert {row["First Name"] for row in result["rows"]} == {"Marie", "Bob"}


def test_debug_classify_row_endpoint_explains_classification(client, monkeypatch):
    monkeypatch.setenv("FLASK_DEBUG", "1")
    df = pd.DataFrame(
        {
            "First Name": ["Neil", "Bo"],
            "Last Name": ["Wusu", "Baker"],
            "Occupation": ["Head of Growth", "Founder"],
            "Employer": ["Spotify", "Local Bakery"],
            "LinkedIn URL": ["", ""],
        }
    )
    dataset_id = upload_dataframe(client, df, "debug.csv")

    response = client.get(
        f"/api/debug/classify-row?dataset_id={dataset_id}&name=Neil%20Wusu&industry=tech"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["match_count"] == 1
    match = data["matches"][0]
    assert match["status"] == "confirmed"
    assert "known_company" in match["match_sources"]
    assert match["internal_reason"]

    excluded = client.get(
        f"/api/debug/classify-row?dataset_id={dataset_id}&name=Bo%20Baker&industry=tech"
    ).get_json()
    assert excluded["matches"][0]["status"] == "excluded"


def test_people_filter_rows_do_not_contain_debug_fields(client):
    df = pd.DataFrame(
        {
            "First Name": ["Pat"],
            "Last Name": ["Partner"],
            "Occupation": ["Partner"],
            "Employer": ["McKinsey"],
            "LinkedIn URL": [""],
        }
    )
    dataset_id = upload_dataframe(client, df, "nodebug.csv")

    data = ask(client, dataset_id, "Which alumni work in consulting?")

    result = data["result"]
    for row in result["rows"]:
        for forbidden in ["match_reason", "match_sources", "confidence", "internal_reason", "classification"]:
            assert forbidden not in row
    answer = assert_valid_answer(data)
    rendered = " ".join(str(block) for block in answer["blocks"])
    assert "internal_reason" not in rendered
    assert "match_sources" not in rendered
