import pandas as pd
import pytest

from app.services.analysis_executor import MAX_OPERATIONS, execute_analysis_plan


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Carol"],
            "Score": [90, 80, 70],
        }
    )


def test_operations_not_a_list(sample_df):
    plan = {"operations": "not-a-list"}
    results = execute_analysis_plan(sample_df, plan)
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert "list" in results[0]["error"].lower()


def test_plan_missing_operations_key(sample_df):
    plan = {"not_operations": True}
    results = execute_analysis_plan(sample_df, plan)
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert "list" in results[0]["error"].lower()


def test_plan_with_empty_operations(sample_df):
    plan = {"operations": []}
    results = execute_analysis_plan(sample_df, plan)
    assert results == []


def test_invalid_operation_produces_error(sample_df):
    plan = {"operations": [{"type": "nonexistent_op", "column": "Score"}]}
    results = execute_analysis_plan(sample_df, plan)
    assert len(results) == 1
    assert results[0]["status"] == "error"


def test_invalid_operation_uses_type_field(sample_df):
    plan = {"operations": [{"type": "bad_op"}]}
    results = execute_analysis_plan(sample_df, plan)
    assert results[0]["operation_type"] == "bad_op"


def test_invalid_operation_non_dict(sample_df):
    plan = {"operations": ["not-a-dict"]}
    results = execute_analysis_plan(sample_df, plan)
    assert results[0]["operation_type"] == "unknown"
    assert results[0]["status"] == "error"


def test_exceeds_max_operations_appends_warning(sample_df):
    ops = [{"type": "preview", "params": {}}] * (MAX_OPERATIONS + 2)
    plan = {"operations": ops}
    results = execute_analysis_plan(sample_df, plan)
    assert len(results) == MAX_OPERATIONS + 1
    last = results[-1]
    assert last["status"] == "error"
    assert "Only" in last["error"]
    assert last["warnings"]


def test_assumptions_passed_through(sample_df):
    plan = {
        "operations": [{"type": "preview", "params": {}}],
        "assumptions": ["Data is clean"],
    }
    results = execute_analysis_plan(sample_df, plan)
    assert results[0]["status"] == "ok"
    assert results[0]["assumptions"] == ["Data is clean"]


def test_assumptions_non_list_ignored(sample_df):
    plan = {
        "operations": [{"type": "preview", "params": {}}],
        "assumptions": "not-a-list",
    }
    results = execute_analysis_plan(sample_df, plan)
    assert results[0]["status"] == "ok"
    assert results[0]["assumptions"] == []


def test_executor_runs_only_validated_filter_predicates():
    df = pd.DataFrame({"Name": ["A", "B"], "Email": ["a@cornell.edu", "b@gmail.com"]})
    plan = {
        "operations": [
            {
                "type": "filter_predicates",
                "params": {
                    "logic": "and",
                    "predicates": [
                        {
                            "columns": ["Email"],
                            "operator": "email_domain_not_in",
                            "values": ["cornell.edu"],
                            "quantifier": "any",
                            "include_subdomains": True,
                            "require_valid_value": True,
                        }
                    ],
                },
            }
        ]
    }

    result = execute_analysis_plan(df, plan)[0]

    assert result["status"] == "ok"
    assert result["operation_type"] == "filter_predicates"
    assert result["metrics"]["matched_row_count"] == 1


def test_executor_runs_one_validated_composite_people_filter():
    df = pd.DataFrame(
        {
            "First Name": ["A", "B"],
            "Occupation": ["Software Engineer", "Product Manager"],
            "Employer": ["Google", "Google"],
            "Email": ["a@gmail.com", "b@cornell.edu"],
        }
    )
    plan = {
        "operations": [
            {
                "type": "composite_people_filter",
                "params": {
                    "people_operation": {
                        "type": "contains_any",
                        "params": {
                            "columns": ["Occupation", "Employer"],
                            "terms": ["software", "google"],
                            "filter_mode": "people",
                            "people_filter": {
                                "filter_type": "industry",
                                "industry": "tech",
                                "industries": ["tech"],
                                "query_scope": "industry",
                            },
                        },
                    },
                    "constraint_logic": "and",
                    "logic": "and",
                    "predicates": [
                        {
                            "columns": ["Email"],
                            "operator": "email_domain_not_in",
                            "values": ["cornell.edu"],
                            "quantifier": "any",
                            "include_subdomains": True,
                            "require_valid_value": True,
                        }
                    ],
                },
            }
        ]
    }

    result = execute_analysis_plan(df, plan)[0]

    assert result["status"] == "ok"
    assert result["operation_type"] == "composite_people_filter"
    assert result["total_matches"] == 1
    assert result["rows"][0]["First Name"] == "A"


def test_executor_rejects_composite_that_bypasses_people_classifier(sample_df):
    plan = {
        "operations": [
            {
                "type": "composite_people_filter",
                "params": {
                    "people_operation": {
                        "type": "contains_any",
                        "params": {"columns": ["Name"], "terms": ["Alice"]},
                    },
                    "logic": "and",
                    "predicates": [
                        {
                            "columns": ["Name"],
                            "operator": "exists",
                            "values": [],
                        }
                    ],
                },
            }
        ]
    }

    result = execute_analysis_plan(sample_df, plan)[0]

    assert result["status"] == "error"
    assert "people-classifier" in result["error"]
