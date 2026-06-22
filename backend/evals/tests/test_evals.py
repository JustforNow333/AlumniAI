import json
from pathlib import Path

import pandas as pd

from evals.report import build_report, render_markdown_report, write_json_report, write_markdown_report
from evals.scoring import (
    apply_execution_expectations,
    check_forbidden_columns,
    check_required_columns,
    compute_expected_rows,
    compute_precision_recall,
    detect_hallucinated_names,
    extract_normalized_response,
    load_cases,
    make_app_view_dataset,
    score_case,
)
from evals.tracing import ModelCallTrace, TracedAIClient


def sample_gold_df():
    return pd.DataFrame(
        {
            "first_name": ["Neil", "Sarah", "Maya"],
            "last_name": ["Wusu", "Patel", "Chen"],
            "preferred_name": ["Nei", "", ""],
            "graduation_year": [2021, 2019, 2023],
            "major": ["Information Science", "English", "Economics"],
            "employer": ["Spotify", "Westchester High School", ""],
            "title": ["Head of Growth", "Teacher", ""],
            "expected_industry": ["Tech", "Education", "Finance"],
            "eval_bucket": ["edge", "edge", "missing"],
            "location": ["New York, NY", "White Plains, NY", "Boston, MA"],
            "linkedin_url": ["https://linkedin.example/neil", "", "https://linkedin.example/maya"],
            "notes": ["seeded", "seeded", "display-test"],
        }
    )


def test_load_cases_reads_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"case_1","question":"Show tech alumni"}\n\n'
        '{"id":"case_2","question":"How many rows?"}\n',
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert [case["id"] for case in cases] == ["case_1", "case_2"]


def test_committed_cases_have_execution_metadata():
    cases = load_cases(Path(__file__).resolve().parents[1] / "cases.jsonl")
    direct_cases = [case for case in cases if case.get("eval_kind") == "direct_classifier"]

    assert len(cases) >= 60
    assert direct_cases
    assert any(case["execution"]["model_calls"] == "required" for case in direct_cases)
    for case in cases:
        assert case.get("modes")
        assert case.get("execution")
        assert case["execution"].get("model_calls") in {"allowed", "disallowed", "required"}


def test_make_app_view_dataset_removes_gold_columns_but_keeps_display_test_columns():
    app_df = make_app_view_dataset(sample_gold_df())

    assert "expected_industry" not in app_df.columns
    assert "eval_bucket" not in app_df.columns
    assert "notes" in app_df.columns
    assert "employer" in app_df.columns


def test_compute_expected_rows_from_industry_and_filter():
    gold_df = sample_gold_df()

    tech = compute_expected_rows({"expected_industry": "Tech"}, gold_df)
    missing_employer = compute_expected_rows(
        {"expected_filter": {"column": "employer", "op": "missing"}},
        gold_df,
    )
    school = compute_expected_rows(
        {"expected_filter": {"column": "employer", "op": "contains", "value": "school"}},
        gold_df,
    )

    assert tech["first_name"].tolist() == ["Neil"]
    assert missing_employer["first_name"].tolist() == ["Maya"]
    assert school["first_name"].tolist() == ["Sarah"]


def test_compute_precision_recall_handles_empty_sets():
    assert compute_precision_recall(["Neil Wusu"], ["Neil Wusu", "Sarah Patel"]) == (1.0, 0.5)
    assert compute_precision_recall([], ["Neil Wusu"]) == (1.0, 0.0)
    assert compute_precision_recall(["Fake Person"], []) == (0.0, 0.0)
    assert compute_precision_recall([], []) == (1.0, 1.0)


def test_detect_hallucinated_names():
    hallucinations = detect_hallucinated_names(
        ["Neil Wusu", "Imaginary Alum"],
        {"neil wusu", "sarah patel"},
    )

    assert hallucinations == ["Imaginary Alum"]


def test_extract_normalized_response_prefers_operation_results():
    response = {
        "answer_text": "Alumni matching criteria: 1",
        "answer": {
            "summary": "Alumni matching criteria: 1",
            "blocks": [
                {"type": "metrics", "items": [{"label": "Alumni matching criteria", "value": "1"}]},
                {
                    "type": "table",
                    "columns": ["Wrong"],
                    "rows": [["Fallback"]],
                },
            ],
        },
        "operation_results": [
            {
                "status": "ok",
                "columns": ["First Name", "Last Name", "Employer", "Occupation"],
                "rows": [["Neil", "Wusu", "Spotify", "Head of Growth"]],
                "metrics": {"total_matches": 1, "displayed_count": 1},
            }
        ],
    }

    normalized = extract_normalized_response(response)

    assert normalized["extraction_source"] == "operation_results"
    assert normalized["scored_from"] == "operation_results"
    assert normalized["count"] == 1
    assert normalized["displayed_count"] == 1
    assert normalized["displayed_columns"] == ["First Name", "Last Name", "Employer", "Occupation"]
    assert normalized["rows"][0]["Employer"] == "Spotify"


def test_column_checks_use_display_aliases_and_forbidden_columns():
    displayed = ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL", "Match Reason"]

    assert check_required_columns(displayed, ["Title", "LinkedIn URL"]) == []
    assert check_forbidden_columns(displayed, ["expected_industry", "Match Reason"]) == ["Match Reason"]


def test_score_case_detects_count_mismatch_for_exact_case():
    gold_df = sample_gold_df()
    case = {
        "id": "spotify",
        "question": "Show me alumni who work at Spotify.",
        "expected_filter": {"column": "employer", "op": "equals", "value": "Spotify"},
        "exact_match": True,
    }
    normalized = {
        "answer_text": "Alumni matching criteria: 2",
        "rows": [{"First Name": "Neil", "Last Name": "Wusu", "Employer": "Spotify", "Occupation": "Head of Growth"}],
        "displayed_columns": ["First Name", "Last Name", "Employer", "Occupation"],
        "count": 2,
        "displayed_count": 1,
        "raw_response": {},
        "metrics": {},
        "extraction_source": "operation_results",
    }

    result = score_case(case, normalized, gold_df)

    assert result["passed"] is False
    assert "Expected count 1, but app reported 2." in result["failures"]
    assert "count_mismatch" in result["failure_categories"]


def test_score_case_detects_hallucinated_structured_name():
    gold_df = sample_gold_df()
    case = {
        "id": "tech",
        "question": "Show me alumni who work in tech.",
        "expected_industry": "Tech",
        "expected_count_mode": "from_expected_industry",
    }
    normalized = {
        "answer_text": "Alumni matching criteria: 1",
        "rows": [{"First Name": "Fake", "Last Name": "Person", "Employer": "Spotify"}],
        "displayed_columns": ["First Name", "Last Name", "Employer"],
        "count": 1,
        "displayed_count": 1,
        "raw_response": {},
        "metrics": {},
        "extraction_source": "operation_results",
    }

    result = score_case(case, normalized, gold_df)

    assert result["passed"] is False
    assert result["hallucinated_names"] == ["Fake Person"]
    assert "hallucinated_names" in result["failure_categories"]


def test_execution_expectations_disallow_model_calls():
    scored = {
        "passed": True,
        "failures": [],
        "failure_categories": [],
        "scored_from": "operation_results",
    }
    case = {"execution": {"model_calls": "disallowed", "llm_classifier": "disallowed"}}
    trace = {"total_model_calls": 1, "llm_classifier_calls": 1, "final_model_synthesis_calls": 0}

    result = apply_execution_expectations(case, scored, trace)

    assert result["passed"] is False
    assert "execution_behavior" in result["failure_categories"]
    assert any("Model calls were disallowed" in failure for failure in result["failures"])


def test_execution_expectations_require_classifier_calls():
    scored = {
        "passed": True,
        "failures": [],
        "failure_categories": [],
        "scored_from": "direct_classifier",
    }
    case = {"execution": {"model_calls": "required", "llm_classifier": "required", "scored_from": "direct_classifier"}}
    trace = {"total_model_calls": 0, "llm_classifier_calls": 0, "final_model_synthesis_calls": 0}

    result = apply_execution_expectations(case, scored, trace)

    assert result["passed"] is False
    assert any("LLM classifier calls were required" in failure for failure in result["failures"])


def test_model_call_trace_classifies_response_create_calls():
    class FakeResponses:
        def create(self, **_kwargs):
            return {"ok": True}

    class FakeClient:
        responses = FakeResponses()

    trace = ModelCallTrace()
    client = TracedAIClient(FakeClient(), trace)

    client.responses.create(model="test-model", instructions="You classify whether an employer belongs to a target industry.")
    client.responses.create(model="test-model", instructions="You present spreadsheet analysis results.")

    assert trace.total_model_calls == 2
    assert trace.llm_classifier_calls == 1
    assert trace.final_model_synthesis_calls == 1
    assert trace.model_name == "test-model"


def test_report_generation_and_writes(tmp_path):
    case_results = [
        {
            "id": "case_1",
            "category": "industry_classification",
            "question": "Show tech alumni",
            "passed": False,
            "precision": 0.5,
            "recall": 0.25,
            "expected_count": 2,
            "returned_count": 1,
            "displayed_count": 1,
            "app_count": 1,
            "failures": ["Precision too low."],
            "failure_categories": ["classification"],
            "hallucinated_names": [],
            "false_positives": ["Wrong Person"],
            "false_negatives_sample": ["Neil Wusu"],
            "raw_answer_excerpt": "Alumni matching criteria: 1",
            "duration_ms": 12.5,
            "answer_source": "deterministic_presenter",
            "scored_from": "operation_results",
            "total_model_calls": 0,
            "llm_classifier_calls": 0,
            "final_model_synthesis_calls": 0,
        }
    ]

    report = build_report(case_results, metadata={"dataset": "gold.csv", "app_view_dataset": "app.csv"})
    markdown = render_markdown_report(report)
    json_path = tmp_path / "latest.json"
    markdown_path = tmp_path / "latest.md"
    write_json_report(report, json_path)
    write_markdown_report(report, markdown_path)

    assert report["summary"]["total_cases"] == 1
    assert report["summary"]["failed"] == 1
    assert report["summary"]["total_model_calls"] == 0
    assert report["summary"]["answer_source_breakdown"] == {"deterministic_presenter": 1}
    assert "## Failed Cases" in markdown
    assert "## Answer Sources" in markdown
    assert "`case_1`" in markdown
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["failed"] == 1
    assert "Slowest Cases" in markdown_path.read_text(encoding="utf-8")
