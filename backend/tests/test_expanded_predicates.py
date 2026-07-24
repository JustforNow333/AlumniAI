from datetime import date, datetime, timezone
from io import BytesIO

import pandas as pd
import pytest

from app import create_app
from app.services import ai_service
from app.services.analysis_intent import (
    heuristic_intent,
    intent_to_analysis_plan,
    validate_analysis_intent,
)
from app.services.analysis_toolkit import (
    build_dataset_context,
    execute_operation,
)
from app.services.intent_filter import (
    apply_intent_filter,
    evaluate_predicate_row,
    normalize_predicate,
    normalize_row_predicates,
    predicate_group_depth,
    predicate_group_leaves,
    row_satisfies_predicate_group,
    validate_filter_predicate_params,
)
from app.services.predicate_values import parse_date_value, parse_numeric_value


@pytest.fixture
def predicate_df():
    return pd.DataFrame(
        {
            "First Name": ["Ava", "Noah", "Mia", "Liam"],
            "Last Name": ["Lee", "Kim", "Patel", "Jones"],
            "Grad Year": [2018, 2020, 2023, 2024],
            "Lifetime Giving": ["$5,000", "$5,001", "10,000", "N/A"],
            "Event Count": [3, 2, 5, 0],
            "Last Contact Date": ["2024-12-15", "2025-01-01", "", "2026-06-01"],
            "Updated At": ["2026-07-01", "2026-01-01", "2026-05-01", "bad date"],
            "City": ["New York", "Boston", "Philadelphia", "Chicago"],
            "Employer": ["Google", "Deloitte", "Goldman Sachs", "Morgan Stanley"],
            "Occupation": ["Software Engineer", "Consultant", "Analyst", "Data Engineer"],
            "Relationship Manager": ["", "Jordan Smith", "", "Taylor Reed"],
            "Email": [
                "ava@example.com",
                "noah@cornell.edu",
                "mia@gmail.com",
                "liam@example.com",
            ],
        }
    )


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)
    app = create_app()
    app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        DATA_FOLDER=str(tmp_path / "data"),
        DATASET_REGISTRY_PATH=str(tmp_path / "data" / "datasets.json"),
        HISTORY_REGISTRY_PATH=str(tmp_path / "data" / "history.json"),
        INSIGHTS_REGISTRY_PATH=str(tmp_path / "data" / "insights.json"),
    )
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def _upload(client, df):
    response = client.post(
        "/api/upload",
        data={"file": (BytesIO(df.to_csv(index=False).encode()), "predicates.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code in {200, 201}, response.get_data(as_text=True)
    return response.get_json()["dataset_id"]


def _ask(client, dataset_id, question):
    response = client.post("/api/ask", json={"dataset_id": dataset_id, "question": question})
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()


def _names(response):
    return {
        row.get("First Name")
        for row in (response.get("result") or {}).get("rows") or []
        if isinstance(row, dict)
    }


def _operators(response):
    return (response.get("intent_filter_trace") or {}).get("intent_filter_operators") or []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (5000, 5000),
        (5000.25, 5000.25),
        ("5000", 5000),
        ("5,000", 5000),
        ("$5,000", 5000),
        ("$5,000.00", 5000),
        ("N/A", None),
        ("5 thousand", None),
        ("", None),
        (True, None),
    ],
)
def test_numeric_parser_is_bounded_and_deterministic(raw, expected):
    assert parse_numeric_value(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (date(2025, 1, 2), date(2025, 1, 2)),
        (datetime(2025, 1, 2, 22, 30, tzinfo=timezone.utc), date(2025, 1, 2)),
        (pd.Timestamp("2025-01-02"), date(2025, 1, 2)),
        ("2025-01-02", date(2025, 1, 2)),
        ("01/02/2025", date(2025, 1, 2)),
        ("January 2025", date(2025, 1, 1)),
        (45659, date(2025, 1, 2)),
        ("bad date", None),
        ("", None),
    ],
)
def test_date_parser_normalizes_supported_values(raw, expected):
    assert parse_date_value(raw) == expected


@pytest.mark.parametrize(
    ("operator", "values", "valid"),
    [
        ("exists", [], True),
        ("exists", ["unexpected"], False),
        ("greater_than", [1], True),
        ("greater_than", [], False),
        ("greater_than", [1, 2], False),
        ("between", [1, 2], True),
        ("between", [2, 1], False),
        ("between", [1], False),
        ("in", ["A"], True),
        ("in", [], False),
        ("date_before", ["2025-01-01"], True),
        ("date_before", ["not a date"], False),
        ("date_between", ["2025-01-01", "2025-12-31"], True),
        ("starts_with", ["Morgan"], True),
        ("ends_with", ["Engineer"], True),
    ],
)
def test_new_operator_arity_and_types_are_validated(predicate_df, operator, values, valid):
    context = build_dataset_context(predicate_df)
    semantic = {
        "date_before": "last_contact_date",
        "date_between": "last_contact_date",
        "starts_with": "employer",
        "ends_with": "occupation",
        "in": "city",
        "exists": "city",
    }.get(operator, "lifetime_giving")
    predicate, error = normalize_predicate(
        {"semantic_column": semantic, "operator": operator, "values": values},
        context,
    )
    assert (predicate is not None) is valid
    assert bool(error) is not valid


def test_numeric_comparison_and_range_boundaries_are_exact():
    row = {"Giving": "$5,000", "Year": 2018}
    base = {
        "columns": ["Giving"],
        "values": [5000],
        "quantifier": "any",
        "require_valid_value": True,
    }
    assert not evaluate_predicate_row(row, {**base, "operator": "greater_than"})
    assert evaluate_predicate_row(row, {**base, "operator": "greater_than_or_equal"})
    assert not evaluate_predicate_row(row, {**base, "operator": "less_than"})
    assert evaluate_predicate_row(row, {**base, "operator": "less_than_or_equal"})

    inclusive = {
        "columns": ["Year"],
        "operator": "between",
        "values": [2018, 2023],
        "lower_inclusive": True,
        "upper_inclusive": True,
        "require_valid_value": True,
    }
    assert evaluate_predicate_row(row, inclusive)
    assert not evaluate_predicate_row(row, {**inclusive, "lower_inclusive": False})
    assert evaluate_predicate_row({"Year": 2023}, inclusive)
    assert not evaluate_predicate_row({"Year": 2023}, {**inclusive, "upper_inclusive": False})


def test_membership_boundaries_and_missing_values_are_safe():
    predicate = {
        "columns": ["City"],
        "operator": "in",
        "values": ["New York", " Boston "],
        "require_valid_value": True,
    }
    assert evaluate_predicate_row({"City": "  new   york "}, predicate)
    assert evaluate_predicate_row({"City": "BOSTON"}, predicate)
    assert not evaluate_predicate_row({"City": "Chicago"}, predicate)
    assert not evaluate_predicate_row(
        {"City": ""},
        {**predicate, "operator": "not_in"},
    )
    assert evaluate_predicate_row(
        {"City": "Philadelphia"},
        {**predicate, "operator": "not_in"},
    )


def test_date_prefix_suffix_and_invalid_values_are_safe():
    before = {
        "columns": ["Contact"],
        "operator": "date_before",
        "values": ["2025-01-01"],
        "require_valid_value": True,
    }
    assert evaluate_predicate_row({"Contact": "2024-12-31"}, before)
    assert not evaluate_predicate_row({"Contact": "2025-01-01"}, before)
    assert not evaluate_predicate_row({"Contact": "bad"}, before)
    assert not evaluate_predicate_row({"Contact": ""}, before)

    assert evaluate_predicate_row(
        {"Employer": "Morgan Stanley"},
        {
            "columns": ["Employer"],
            "operator": "starts_with",
            "values": ["morgan"],
            "require_valid_value": True,
        },
    )
    assert not evaluate_predicate_row(
        {"Employer": "J.P. Morgan"},
        {
            "columns": ["Employer"],
            "operator": "starts_with",
            "values": ["Morgan"],
            "require_valid_value": True,
        },
    )
    assert evaluate_predicate_row(
        {"Title": "Software Engineer"},
        {
            "columns": ["Title"],
            "operator": "ends_with",
            "values": ["engineer"],
            "require_valid_value": True,
        },
    )
    assert not evaluate_predicate_row(
        {"Title": "Engineering Manager"},
        {
            "columns": ["Title"],
            "operator": "ends_with",
            "values": ["Engineer"],
            "require_valid_value": True,
        },
    )


def test_multicolumn_quantifiers_avoid_vacuous_truth():
    base = {
        "columns": ["Giving 1", "Giving 2"],
        "operator": "greater_than",
        "values": [5000],
        "require_valid_value": True,
    }
    row = {"Giving 1": "$6,000", "Giving 2": "$4,000"}
    assert evaluate_predicate_row(row, {**base, "quantifier": "any"})
    assert not evaluate_predicate_row(row, {**base, "quantifier": "all"})
    assert not evaluate_predicate_row(row, {**base, "quantifier": "none"})
    assert not evaluate_predicate_row(
        {"Giving 1": "", "Giving 2": "N/A"},
        {**base, "quantifier": "all"},
    )


def test_legacy_flat_roots_normalize_to_recursive_clauses(predicate_df):
    root, errors = normalize_row_predicates(
        {
            "logic": "and",
            "predicates": [
                {
                    "semantic_column": "event_count",
                    "operator": "greater_than_or_equal",
                    "values": [3],
                }
            ],
        },
        build_dataset_context(predicate_df),
    )
    assert errors == []
    assert root["clauses"][0]["type"] == "predicate"
    assert root["predicates"] == predicate_group_leaves(root)
    assert predicate_group_depth(root) == 1


def test_recursive_groups_preserve_inner_or_and_enforce_depth(predicate_df):
    group = {
        "logic": "and",
        "clauses": [
            {
                "type": "predicate",
                "predicate": {
                    "columns": ["Lifetime Giving"],
                    "operator": "greater_than",
                    "values": [5000],
                    "require_valid_value": True,
                },
            },
            {
                "type": "group",
                "logic": "or",
                "clauses": [
                    {
                        "type": "predicate",
                        "predicate": {
                            "columns": ["Relationship Manager"],
                            "operator": "missing",
                            "values": [],
                        },
                    },
                    {
                        "type": "predicate",
                        "predicate": {
                            "columns": ["Last Contact Date"],
                            "operator": "date_before",
                            "values": ["2025-01-01"],
                            "require_valid_value": True,
                        },
                    },
                ],
            },
        ],
    }
    valid, error = validate_filter_predicate_params(
        {"predicate_group": group},
        predicate_df.columns,
    )
    assert valid, error
    matches = predicate_df.apply(lambda row: row_satisfies_predicate_group(row, group), axis=1)
    assert predicate_df.loc[matches, "First Name"].tolist() == ["Mia"]

    too_deep = group
    for _ in range(3):
        too_deep = {"logic": "and", "clauses": [{"type": "group", **too_deep}]}
    valid, error = validate_filter_predicate_params({"predicate_group": too_deep})
    assert not valid
    assert "at most 3" in error


@pytest.mark.parametrize(
    ("question", "expected_operator", "semantic", "values"),
    [
        ("Show alumni who graduated between 2018 and 2023.", "between", "grad_year", [2018, 2023]),
        ("Find donors who have given more than $1,000.", "greater_than", "lifetime_giving", [1000]),
        ("Who attended at least three events?", "greater_than_or_equal", "event_count", [3]),
        ("Find alumni in New York, Boston, or Philadelphia.", "in", "city", ["New York", "Boston", "Philadelphia"]),
        ("Find alumni outside New York and Boston.", "not_in", "city", ["New York", "Boston"]),
        ("Find alumni whose employer starts with Morgan.", "starts_with", "employer", ["Morgan"]),
        ("Show alumni whose title ends with Engineer.", "ends_with", "occupation", ["Engineer"]),
        ("Show alumni contacted before January 2025.", "date_before", "last_contact_date", ["2025-01-01"]),
    ],
)
def test_deterministic_extraction_is_field_anchored(
    predicate_df,
    question,
    expected_operator,
    semantic,
    values,
):
    context = build_dataset_context(predicate_df)
    intent, _trace = apply_intent_filter(question, context, heuristic_intent(question, context))
    predicate = next(
        item
        for item in predicate_group_leaves(intent["row_predicates"])
        if item["semantic_column"] == semantic
    )
    assert predicate["operator"] == expected_operator
    assert predicate["values"] == values


def test_relative_date_is_resolved_once_to_a_stable_boundary(predicate_df):
    question = "Show records updated within the past 90 days."
    context = build_dataset_context(predicate_df)
    intent, trace = apply_intent_filter(
        question,
        context,
        heuristic_intent(question, context),
        now=datetime(2026, 7, 23, 23, 30, tzinfo=timezone.utc),
    )
    predicate = predicate_group_leaves(intent["row_predicates"])[0]
    assert predicate["operator"] == "date_after"
    assert predicate["values"] == ["2026-04-24"]
    assert trace["relative_dates_resolved"] == {"past_90_days": "2026-04-24"}


@pytest.mark.parametrize(
    ("question", "model_operator", "model_values", "expected_operator", "expected_values"),
    [
        ("Show alumni with at least 3 events.", "greater_than", [3], "greater_than_or_equal", [3]),
        ("Show alumni with giving more than 5000.", "greater_than_or_equal", [5000], "greater_than", [5000]),
        ("Show alumni contacted before 2025.", "date_after", ["2025-01-01"], "date_before", ["2025-01-01"]),
        (
            "Find alumni in New York or Boston.",
            "in",
            ["New York", "Boston", "Chicago"],
            "in",
            ["New York", "Boston"],
        ),
    ],
)
def test_source_comparisons_and_values_repair_incorrect_model_output(
    predicate_df,
    question,
    model_operator,
    model_values,
    expected_operator,
    expected_values,
):
    context = build_dataset_context(predicate_df)
    semantic = (
        "event_count"
        if "event" in question
        else "lifetime_giving"
        if "giving" in question
        else "last_contact_date"
        if "contacted" in question
        else "city"
    )
    model_intent = {
        "intent": "find_records",
        "filters": [],
        "row_predicates": {
            "logic": "and",
            "predicates": [
                {
                    "semantic_column": semantic,
                    "operator": model_operator,
                    "values": model_values,
                }
            ],
        },
        "desired_output": {"format": "table", "semantic_columns": [], "limit": 100},
    }
    intent, trace = apply_intent_filter(question, context, model_intent)
    predicate = predicate_group_leaves(intent["row_predicates"])[0]
    assert predicate["operator"] == expected_operator
    assert predicate["values"] == expected_values
    assert trace["intent_filter_conflicts_removed"] == 1
    assert trace["intent_filter_repaired_model_output"] is True


def test_display_only_field_does_not_become_a_model_predicate(predicate_df):
    context = build_dataset_context(predicate_df)
    model_intent = {
        "intent": "find_records",
        "filters": [],
        "row_predicates": {
            "logic": "and",
            "predicates": [
                {
                    "semantic_column": "email",
                    "operator": "exists",
                    "values": [],
                }
            ],
        },
        "desired_output": {
            "format": "table",
            "semantic_columns": ["first_name", "last_name", "email"],
            "limit": 100,
        },
    }
    intent, trace = apply_intent_filter("Show alumni names and email addresses.", context, model_intent)
    assert predicate_group_leaves(intent.get("row_predicates")) == []
    assert trace["intent_filter_conflicts_removed"] == 1


def test_model_validator_accepts_recursive_groups_and_rejects_too_deep_groups():
    value = {
        "intent": "find_records",
        "target_entity": "rows",
        "row_predicates": {
            "logic": "and",
            "clauses": [
                {
                    "type": "group",
                    "logic": "or",
                    "clauses": [
                        {
                            "type": "predicate",
                            "predicate": {
                                "semantic_column": "city",
                                "operator": "in",
                                "values": ["Boston", "New York"],
                            },
                        }
                    ],
                }
            ],
        },
    }
    intent, valid, _error = validate_analysis_intent(value)
    assert valid
    assert predicate_group_depth(intent["row_predicates"]) == 2
    assert predicate_group_leaves(intent["row_predicates"])[0]["operator"] == "in"

    root = value["row_predicates"]
    for _ in range(3):
        root = {"logic": "and", "clauses": [{"type": "group", **root}]}
    value["row_predicates"] = root
    intent, _valid, _error = validate_analysis_intent(value)
    assert any("at most 3" in item for item in intent["row_predicate_validation_errors"])


@pytest.mark.parametrize(
    ("question", "expected_names", "expected_operators"),
    [
        ("Show alumni who graduated between 2018 and 2023.", {"Ava", "Noah", "Mia"}, ["between"]),
        ("Find donors who have given more than $5,000.", {"Noah", "Mia"}, ["greater_than"]),
        ("Find donors with lifetime giving at least $5,000.", {"Ava", "Noah", "Mia"}, ["greater_than_or_equal"]),
        ("Who attended at least three events?", {"Ava", "Mia"}, ["greater_than_or_equal"]),
        ("Find alumni in New York, Boston, or Philadelphia.", {"Ava", "Noah", "Mia"}, ["in"]),
        ("Find alumni outside New York and Boston.", {"Mia", "Liam"}, ["not_in"]),
        ("Show alumni contacted before January 2025.", {"Ava"}, ["date_before"]),
        ("Find alumni whose employer starts with Morgan.", {"Liam"}, ["starts_with"]),
        ("Show alumni whose title ends with Engineer.", {"Ava", "Liam"}, ["ends_with"]),
        (
            "Find alumni with lifetime giving over $5,000 who do not have an assigned relationship manager.",
            {"Mia"},
            ["missing", "greater_than"],
        ),
        (
            "Find alumni with giving over $5,000 or at least five events.",
            {"Noah", "Mia"},
            ["greater_than", "greater_than_or_equal"],
        ),
    ],
)
def test_api_executes_expanded_predicates_end_to_end(
    client,
    predicate_df,
    question,
    expected_names,
    expected_operators,
):
    dataset_id = _upload(client, predicate_df)
    data = _ask(client, dataset_id, question)
    assert data["operation"]["type"] == "filter_predicates"
    assert _names(data) == expected_names
    assert _operators(data) == expected_operators
    assert data["result"]["metrics"]["matched_row_count"] == len(expected_names)
    assert data["result"]["post_verification_removed_count"] == 0
    assert "__alumniai_row_id__" not in str(data)
    assert "predicate" not in " ".join(data["result"]["columns"]).casefold()


def test_api_relative_date_and_parse_failure_metadata(client, predicate_df, monkeypatch):
    monkeypatch.setattr(
        "app.services.intent_filter.relative_date_boundary",
        lambda days, now=None: date(2026, 4, 24),
    )
    dataset_id = _upload(client, predicate_df)
    data = _ask(client, dataset_id, "Show records updated within the past 90 days.")
    assert _names(data) == {"Ava", "Mia"}
    assert data["intent_filter_trace"]["relative_dates_resolved"] == {
        "past_90_days": "2026-04-24"
    }
    assert data["result"]["date_parse_failure_count"] == 1


@pytest.mark.parametrize(
    ("question", "expected_names"),
    [
        ("Find tech alumni who graduated after 2020.", {"Liam"}),
        ("Find finance alumni with lifetime giving over $5,000.", {"Zoe"}),
        ("Find consultants in New York or Boston.", {"Noah"}),
        ("Find tech alumni with a non-Cornell email and at least three events.", {"Ava"}),
        ("Find software engineers outside New York and Boston.", set()),
    ],
)
def test_fuzzy_and_expanded_exact_predicates_compose(client, predicate_df, question, expected_names):
    if question.startswith("Find finance"):
        predicate_df = pd.concat(
            [
                predicate_df,
                pd.DataFrame(
                    [
                        {
                            "First Name": "Zoe",
                            "Last Name": "Reed",
                            "Grad Year": 2022,
                            "Lifetime Giving": "$8,000",
                            "Event Count": 1,
                            "Last Contact Date": "2025-06-01",
                            "Updated At": "2026-06-01",
                            "City": "Austin",
                            "Employer": "BlackRock",
                            "Occupation": "Portfolio Manager",
                            "Relationship Manager": "Alex Rivera",
                            "Email": "zoe@example.com",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    dataset_id = _upload(client, predicate_df)
    data = _ask(client, dataset_id, question)
    assert data["operation"]["type"] == "composite_people_filter"
    assert _names(data) == expected_names
    trace = data["intent_filter_trace"]
    assert trace["has_fuzzy_people_filter"] is True
    assert trace["has_exact_row_predicates"] is True
    assert trace["fuzzy_clause_dropped"] is False
    assert trace["fuzzy_direct_count_before_predicates"] >= len(expected_names)
    assert data["result"]["post_verification_removed_count"] == 0
    assert "__alumniai_row_id__" not in str(data)
    for row in data["result"].get("direct_rows") or []:
        assert not any(str(key).startswith("_") for key in row)


def test_nested_group_query_is_not_flattened_and_persists_in_history(client, predicate_df):
    dataset_id = _upload(client, predicate_df)
    question = (
        "Find alumni with giving over $5,000 who either have no relationship manager "
        "or were contacted before January 2025."
    )
    data = _ask(client, dataset_id, question)
    assert _names(data) == {"Mia"}
    assert data["intent_filter_trace"]["predicate_group_depth"] == 2
    group = data["analysis_intent"]["row_predicates"]
    assert group["logic"] == "and"
    assert next(clause for clause in group["clauses"] if clause["type"] == "group")["logic"] == "or"

    history = client.get("/api/history").get_json()["history"]
    assert history[0]["response_payload"]["analysis_intent"]["row_predicates"]["clauses"]
    assert history[0]["response_payload"]["result"]["metrics"]["matched_row_count"] == 1


def test_filtering_precedes_numeric_sorting_and_limit(client):
    df = pd.DataFrame(
        {
            "First Name": ["A", "B", "C", "D"],
            "Last Name": ["One", "Two", "Three", "Four"],
            "Lifetime Giving": ["$5,000", "$7,000", "$6,000", "$10,000"],
        }
    )
    dataset_id = _upload(client, df)
    data = _ask(client, dataset_id, "Show the top 2 donors with more than $5,000 in lifetime giving.")
    assert _names(data) == {"B", "D"}
    assert [row["First Name"] for row in data["result"]["rows"]] == ["D", "B"]
    assert data["result"]["metrics"]["matched_row_count"] == 3
    assert data["result"]["metrics"]["returned_row_count"] == 2


def test_direct_execution_reports_numeric_and_date_parse_failures(predicate_df):
    operation = {
        "type": "filter_predicates",
        "params": {
            "logic": "and",
            "predicates": [
                {
                    "columns": ["Lifetime Giving"],
                    "operator": "greater_than",
                    "values": [0],
                    "require_valid_value": True,
                },
                {
                    "columns": ["Updated At"],
                    "operator": "date_after",
                    "values": ["2025-01-01"],
                    "require_valid_value": True,
                },
            ],
        },
    }
    result = execute_operation(predicate_df, operation)
    assert result["status"] == "ok"
    assert result["numeric_parse_failure_count"] == 1
    assert result["date_parse_failure_count"] == 1
