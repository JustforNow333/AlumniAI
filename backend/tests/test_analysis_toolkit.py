import numpy as np
import pandas as pd

from app.services.analysis_toolkit import MAX_LIMIT, execute_operation
from app.services.answer_schema import deterministic_answer_from_results


def test_contains_any_searches_full_dataframe_and_returns_matches():
    df = pd.DataFrame(
        {
            "Name": [f"Person {i}" for i in range(15)] + ["Full Dataset Match"],
            "Occupation": ["Teacher"] * 15 + ["Senior Software Engineer"],
            "Employer": ["School"] * 15 + ["Tech Corp"],
            "Industry": ["Education"] * 15 + ["Technology"],
            "Major": ["History"] * 15 + ["Computer Science"],
        }
    )

    result = execute_operation(
        df,
        {
            "type": "contains_any",
            "params": {
                "columns": ["Occupation", "Employer", "Industry", "Major"],
                "terms": ["software", "engineer", "developer", "data", "ai", "machine learning", "tech"],
                "limit": 100,
            },
        },
    )

    assert result["status"] == "ok"
    assert result["is_filtered"] is True
    assert result["matched_row_count"] == 1
    assert result["returned_row_count"] == 1
    assert result["total_rows"] == 16
    assert result["search_columns"] == ["Occupation", "Employer", "Industry", "Major"]
    assert result["display_columns"] == ["Name", "Occupation", "Employer", "MATCH REASON"]
    assert result["metrics"]["matched_row_count"] == 1
    assert result["rows"][0][0] == "Full Dataset Match"
    assert "MATCH REASON" in result["columns"]
    assert "Major" not in result["columns"]


def test_top_n_sorts_numeric_like_currency_strings():
    df = pd.DataFrame(
        {
            "Name": ["Small", "Large", "Medium"],
            "Lifetime Giving": ["$900", "$18,500", "$4,200"],
            "Employer": ["A", "B", "C"],
        }
    )

    result = execute_operation(
        df,
        {
            "type": "top_n",
            "params": {
                "column": "Lifetime Giving",
                "n": 2,
                "return_columns": ["Name", "Lifetime Giving", "Employer"],
            },
        },
    )

    assert result["status"] == "ok"
    assert result["rows"] == [["Large", "$18,500", "B"], ["Medium", "$4,200", "C"]]


def test_group_by_average_ignores_missing_and_invalid_numeric_values():
    df = pd.DataFrame(
        {
            "Industry": ["Tech", "Tech", "Finance", "Finance", "Finance"],
            "Lifetime Giving": ["$100", "", "bad", "$300", None],
        }
    )

    result = execute_operation(
        df,
        {
            "type": "group_by_average",
            "params": {"group_by": "Industry", "value_column": "Lifetime Giving", "limit": 10},
        },
    )

    assert result["status"] == "ok"
    assert result["rows"] == [["Finance", 300.0], ["Tech", 100.0]]
    assert result["metrics"]["invalid_or_missing_numeric_values"] == 3


def test_missing_values_counts_null_nan_and_empty_strings():
    df = pd.DataFrame(
        {
            "Name": ["A", "", None, "D"],
            "Score": [1, np.nan, 3, ""],
        }
    )

    result = execute_operation(df, {"type": "missing_values", "params": {"columns": None}})

    assert result["status"] == "ok"
    rows = {row[0]: row[1] for row in result["rows"]}
    assert rows["Name"] == 2
    assert rows["Score"] == 2


def test_unknown_columns_are_rejected_safely():
    df = pd.DataFrame({"Name": ["A"], "Score": [1]})

    result = execute_operation(
        df,
        {"type": "top_n", "params": {"column": "Missing Column", "n": 5}},
    )

    assert result["status"] == "error"
    assert "was not found" in result["error"]
    assert result["is_filtered"] is False


def test_unknown_operation_type_is_rejected_safely():
    df = pd.DataFrame({"Name": ["A"], "Score": [1]})

    result = execute_operation(df, {"type": "run_python", "params": {"code": "print(1)"}})

    assert result["status"] == "error"
    assert "Unknown operation type" in result["error"]


def test_limit_is_capped():
    df = pd.DataFrame({"Name": [f"Person {i}" for i in range(MAX_LIMIT + 25)]})

    result = execute_operation(
        df,
        {"type": "preview", "params": {"limit": MAX_LIMIT + 25}},
    )

    assert result["status"] == "ok"
    assert len(result["rows"]) == MAX_LIMIT


def test_occupation_resolves_to_uppercase_occupation_for_contains_any():
    df = pd.DataFrame(
        {
            "NAME": ["Ada", "Grace"],
            "OCCUPATION": ["Software Engineer", "Teacher"],
            "EMPLOYER": ["Google", "School"],
        }
    )

    result = execute_operation(
        df,
        {"type": "contains_any", "params": {"columns": ["Occupation"], "terms": ["software"]}},
    )

    assert result["status"] == "ok"
    assert result["metrics"]["searched_columns"] == ["OCCUPATION"]
    assert result["metrics"]["matched_row_count"] == 1
    assert result["columns"] == ["NAME", "OCCUPATION", "EMPLOYER", "MATCH REASON"]
    assert result["rows"][0] == ["Ada", "Software Engineer", "Google", "Matched OCCUPATION: Software Engineer"]


def test_employer_resolves_to_uppercase_employer_for_contains_any():
    df = pd.DataFrame(
        {
            "NAME": ["Ada", "Grace"],
            "OCCUPATION": ["Engineer", "Teacher"],
            "EMPLOYER": ["Google", "School"],
        }
    )

    result = execute_operation(
        df,
        {"type": "contains_any", "params": {"columns": ["Employer"], "terms": ["google"]}},
    )

    assert result["status"] == "ok"
    assert result["metrics"]["searched_columns"] == ["EMPLOYER"]
    assert result["metrics"]["matched_row_count"] == 1


def test_company_resolves_to_employer_when_company_column_is_absent():
    df = pd.DataFrame(
        {
            "NAME": ["Ada", "Grace"],
            "EMPLOYER": ["Google", "School"],
        }
    )

    result = execute_operation(
        df,
        {"type": "contains_any", "params": {"columns": ["Company"], "terms": ["google"]}},
    )

    assert result["status"] == "ok"
    assert result["metrics"]["searched_columns"] == ["EMPLOYER"]
    assert result["metrics"]["matched_row_count"] == 1


def test_contains_any_resolves_requested_columns_against_uppercase_dataframe():
    df = pd.DataFrame(
        {
            "NAME": ["Ada", "Grace"],
            "OCCUPATION": ["Software Engineer", "Teacher"],
            "EMPLOYER": ["Google", "School"],
        }
    )

    result = execute_operation(
        df,
        {
            "type": "contains_any",
            "params": {
                "columns": ["Occupation", "Employer"],
                "terms": ["software", "google"],
                "return_columns": ["NAME", "Occupation", "Employer"],
            },
        },
    )

    assert result["status"] == "ok"
    assert result["metrics"]["searched_columns"] == ["OCCUPATION", "EMPLOYER"]
    assert result["columns"][:3] == ["NAME", "OCCUPATION", "EMPLOYER"]
    assert result["metrics"]["matched_row_count"] == 1


def test_contains_any_infers_searchable_columns_when_requested_columns_are_invalid():
    df = pd.DataFrame(
        {
            "NAME": ["Ada", "Grace"],
            "OCCUPATION": ["Software Engineer", "Teacher"],
            "EMPLOYER": ["Google", "School"],
            "NOTES": ["", "technology club"],
        }
    )

    result = execute_operation(
        df,
        {"type": "contains_any", "params": {"columns": ["Missing Column"], "terms": ["software"]}},
    )

    assert result["status"] == "ok"
    assert result["is_filtered"] is True
    assert result["metrics"]["searched_columns"] == ["OCCUPATION", "EMPLOYER"]
    assert result["metrics"]["matched_row_count"] == 1
    assert any(warning["type"] == "inferred_search_columns" for warning in result["warnings"])


def test_filter_contains_accepts_columns_and_query_params():
    df = pd.DataFrame(
        {
            "NAME": ["Ada", "Grace"],
            "OCCUPATION": ["Software Engineer", "Teacher"],
            "EMPLOYER": ["Google", "School"],
        }
    )

    result = execute_operation(
        df,
        {
            "type": "filter_contains",
            "params": {
                "columns": ["job-title", "company"],
                "query": "GOOGLE",
                "return_columns": ["name", "occupation", "employer"],
            },
        },
    )

    assert result["status"] == "ok"
    assert result["is_filtered"] is True
    assert result["metrics"]["searched_columns"] == ["OCCUPATION", "EMPLOYER"]
    assert result["columns"][:3] == ["NAME", "OCCUPATION", "EMPLOYER"]
    assert result["rows"][0][:3] == ["Ada", "Software Engineer", "Google"]


def test_text_search_counts_raw_keyword_hits_unique_rows_and_limit():
    df = pd.DataFrame(
        {
            "NICKNAME": ["Ada", "Grace", "Linus"],
            "OCCUPATION": ["Software Engineer", "Data Engineer", "Teacher"],
            "EMPLOYER": ["Google", "OpenAI", "School"],
        }
    )

    result = execute_operation(
        df,
        {
            "type": "contains_any",
            "params": {
                "columns": ["OCCUPATION", "EMPLOYER"],
                "terms": ["engineer", "google", "openai"],
                "limit": 1,
            },
        },
    )

    assert result["raw_match_count"] == 4
    assert result["matched_row_count"] == 2
    assert result["returned_row_count"] == 1
    assert result["display_limit"] == 1
    assert result["deduplicated"] is True
    assert any(warning["type"] == "deduplicated_text_matches" for warning in result["warnings"])
    assert any(warning["type"] == "display_limit_applied" for warning in result["warnings"])


def test_failed_filter_does_not_return_unfiltered_rows():
    df = pd.DataFrame({"ID": [1, 2], "Score": [10, 20]})

    result = execute_operation(
        df,
        {"type": "contains_any", "params": {"columns": ["Missing Column"], "terms": ["software"]}},
    )

    assert result["status"] == "error"
    assert result["is_filtered"] is False
    assert "rows" not in result
    assert "No valid searchable columns" in result["error"]

    answer = deterministic_answer_from_results(
        "find software matches",
        {"operations": [{"type": "contains_any", "params": {}}], "presentation_hint": "table"},
        [result],
        {"columns": [{"name": "ID", "type": "number"}, {"name": "Score", "type": "number"}]},
    )
    assert all(block["type"] != "table" for block in answer["blocks"])
    assert "No valid searchable columns" in answer["summary"]
