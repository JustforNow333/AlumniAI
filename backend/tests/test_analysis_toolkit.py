import numpy as np
import pandas as pd

from app.services import analysis_toolkit as toolkit
from app.services.analysis_toolkit import (
    MAX_LIMIT,
    classify_employer_tech_status,
    execute_operation,
    is_explicit_technical_title,
    is_strong_non_tech_context,
)
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


def test_contains_all_with_groups_requires_every_distinct_term():
    df = pd.DataFrame(
        {
            "OCCUPATION": ["Software Engineer", "Cloud Architect", "Teacher"],
            "EMPLOYER": ["Software Inc", "Cloud Software Co", "School"],
        }
    )

    # Row 0 matches "software" in both groups but never matches "cloud",
    # so it must not count as matching all terms. Row 1 matches both terms.
    result = execute_operation(
        df,
        {
            "type": "contains_all",
            "params": {
                "column_term_groups": [
                    {"columns": ["OCCUPATION"], "terms": ["software", "cloud"]},
                    {"columns": ["EMPLOYER"], "terms": ["software", "cloud"]},
                ],
            },
        },
    )

    assert result["status"] == "ok"
    assert result["matched_row_count"] == 1
    assert result["rows"][0][0] == "Cloud Architect"


def test_contains_all_with_shared_term_across_groups_still_matches():
    df = pd.DataFrame(
        {
            "OCCUPATION": ["Software Engineer", "Teacher"],
            "EMPLOYER": ["Software Inc", "School"],
        }
    )

    # The single required term matches in both groups; the duplicate group
    # hits must not prevent the row from counting as a match.
    result = execute_operation(
        df,
        {
            "type": "contains_all",
            "params": {
                "column_term_groups": [
                    {"columns": ["OCCUPATION"], "terms": ["software"]},
                    {"columns": ["EMPLOYER"], "terms": ["software"]},
                ],
            },
        },
    )

    assert result["status"] == "ok"
    assert result["matched_row_count"] == 1
    assert result["rows"][0][0] == "Software Engineer"


def test_tech_people_filter_uses_clean_person_columns_counts_and_strict_matching():
    df = pd.DataFrame(
        {
            "First Name": [
                "Alice",
                "Bob",
                "Carol",
                "Dan",
                "Erin",
                "Frank",
                "Gina",
                "Hank",
                "Ivy",
                "Jack",
                "Jack",
                "Ken",
                "Lee",
            ],
            "LastName": [
                "School",
                "Hospital",
                "University",
                "School",
                "Hospital",
                "Bakery",
                "Rune",
                "Fanamp",
                "Dao",
                "Bakery",
                "Bakery",
                "Ventures",
                "Google",
            ],
            "Nickname": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "J2",
                "K",
                "L",
            ],
            "Occupation": [
                "Mathematics Department Chair (Middle School)",
                "Data Scientist",
                "Software Engineer",
                "IT Director",
                "Director of Hematologic Oncology",
                "Founder",
                "Founder",
                "Founder",
                "Founder",
                "Software Engineer",
                "Software Engineer",
                "CEO",
                "CEO",
            ],
            "Employer": [
                "Northview Middle School",
                "Hospital for Special Surgery",
                "Local University",
                "Public High School",
                "Hospital for Special Surgery",
                "Local Bakery",
                "Rune Technologies",
                "FanAmp",
                "Cogni DAO",
                "Local Bakery",
                "Local Bakery",
                "Bright Ventures",
                "Google",
            ],
            "LinkedinURL": [
                "",
                "https://linkedin.com/in/bob",
                "",
                "linkedin.com/in/dan",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
        }
    )

    result = execute_operation(
        df,
        {
            "type": "contains_any",
            "params": {
                "columns": ["Occupation", "Employer"],
                "terms": ["software", "data scientist", "it", "technologies", "google"],
                "display_columns": ["First Name", "LastName", "Occupation", "Employer", "LinkedinURL", "MATCH REASON"],
                "filter_mode": "tech_people",
                "limit": 100,
            },
        },
    )

    assert result["status"] == "ok"
    assert result["intent"] == "people_filter"
    assert result["entity"] == "alumni"
    assert result["answer_label"] == "Alumni matching criteria"
    assert result["columns"] == ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"]
    assert result["visible_columns"] == ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"]
    assert "Nickname" not in result["columns"]
    assert "MATCH REASON" not in result["columns"]
    assert result["total_dataset_rows"] == 13
    assert result["total_matches"] == 8
    assert result["displayed_count"] == 8
    assert result["display_limit"] == 100
    assert result["uncertain_count"] == 1
    assert result["metrics"]["total_matches"] == 8
    assert result["metrics"]["displayed_count"] == 8
    assert result["metrics"]["display_limit"] == 100
    assert result["metrics"]["uncertain_count"] == 1

    rows_by_first = {row["First Name"]: row for row in result["rows"]}
    assert set(rows_by_first) == {"Bob", "Carol", "Dan", "Gina", "Hank", "Ivy", "Jack", "Lee"}
    assert rows_by_first["Bob"]["LinkedIn URL"] == "https://linkedin.com/in/bob"
    assert rows_by_first["Carol"]["LinkedIn URL"] == ""
    assert "Alice" not in rows_by_first
    assert "Erin" not in rows_by_first
    assert "Frank" not in rows_by_first
    assert "Ken" not in rows_by_first
    assert "debug" in result
    assert all("match_reason" not in row for row in result["rows"])


def test_people_filter_returns_all_scored_rows_separate_from_display_limit():
    df = pd.DataFrame(
        {
            "First Name": ["Ada", "Grace", "Katherine"],
            "Last Name": ["Lovelace", "Hopper", "Johnson"],
            "Occupation": ["Software Engineer", "Data Scientist", "Cloud Engineer"],
            "Employer": ["Bakery", "Hospital", "School"],
            "LinkedIn URL": ["", "", ""],
        }
    )

    result = execute_operation(
        df,
        {
            "type": "contains_any",
            "params": {
                "columns": ["Occupation", "Employer"],
                "terms": ["engineer", "data scientist"],
                "filter_mode": "people",
                "people_filter": {
                    "filter_type": "industry",
                    "industry": "tech",
                    "industries": ["tech"],
                    "query_scope": "technical_role",
                },
                "limit": 1,
            },
        },
    )

    assert result["status"] == "ok"
    assert result["total_matches"] == 3
    assert result["displayed_count"] == 1
    assert result["scored_result_count"] == 3
    assert result["metrics"]["returned_row_count"] == 3
    assert len(result["rows"]) == 3


def test_technical_title_and_employer_classification_helpers():
    assert is_explicit_technical_title("Software Engineer") is True
    assert is_explicit_technical_title("Data Scientist") is True
    assert is_explicit_technical_title("IT Director") is True
    assert is_explicit_technical_title("Founder") is False
    assert is_explicit_technical_title("Chief Executive Officer") is False

    assert is_strong_non_tech_context("Director of Hematologic Oncology", "Holy Name Medical Center") is True
    assert is_strong_non_tech_context("Software Engineer", "Local Bakery") is False

    assert classify_employer_tech_status("Rune Technologies")["status"] == "confirmed_tech"
    assert classify_employer_tech_status("FanAmp")["status"] == "confirmed_tech"
    assert classify_employer_tech_status("Cogni DAO")["status"] == "confirmed_tech"
    ambiguous = classify_employer_tech_status("Bright Ventures")
    assert ambiguous["status"] == "uncertain"
    assert ambiguous["confidence"] < 0.75


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


def test_filter_missing_returns_matching_rows_not_column_summary():
    df = pd.DataFrame(
        {
            "First Name": ["Ada", "Grace", "Katherine"],
            "Last Name": ["Lovelace", "Hopper", "Johnson"],
            "Title": ["Engineer", "Scientist", "Manager"],
            "Employer": ["Analytical Engines", "Navy", "Google"],
            "LinkedIn URL": ["", "https://linkedin.com/in/grace", None],
        }
    )

    result = execute_operation(
        df,
        {
            "type": "filter_missing",
            "params": {
                "column": "linkedin_url",
                "return_columns": ["first_name", "last_name", "occupation", "employer"],
                "limit": 10,
            },
        },
    )

    assert result["status"] == "ok"
    assert result["operation_type"] == "filter_missing"
    assert result["is_filtered"] is True
    assert result["metrics"]["rows_matched"] == 2
    assert result["metrics"]["rows_returned"] == 2
    assert result["columns"] == ["First Name", "Last Name", "Title", "Employer"]
    assert result["rows"] == [
        ["Ada", "Lovelace", "Engineer", "Analytical Engines"],
        ["Katherine", "Johnson", "Manager", "Google"],
    ]


def test_text_search_can_suppress_match_reason_for_clean_person_filters():
    df = pd.DataFrame(
        {
            "First Name": ["Ada", "Grace"],
            "Last Name": ["Lovelace", "Hopper"],
            "Title": ["Engineer", "Scientist"],
            "Employer": ["Analytical Engines", "Navy"],
        }
    )

    result = execute_operation(
        df,
        {
            "type": "filter_contains",
            "params": {
                "column": "employer",
                "terms": ["Navy"],
                "display_columns": ["first_name", "last_name", "occupation", "employer"],
                "include_match_reason": False,
            },
        },
    )

    assert result["status"] == "ok"
    assert result["columns"] == ["First Name", "Last Name", "Title", "Employer"]
    assert "MATCH REASON" not in result["columns"]
    assert result["rows"] == [["Grace", "Hopper", "Scientist", "Navy"]]


def _people_industry_filter(df, industry, *, excluded=None, query_scope="industry", limit=100):
    spec = {
        "filter_type": "industry",
        "industry": industry,
        "industries": [industry],
        "query_scope": query_scope,
    }
    if excluded is not None:
        spec["excluded_industries"] = list(excluded)
    return execute_operation(
        df,
        {
            "type": "contains_any",
            "params": {
                "columns": ["Occupation", "Employer"],
                "terms": ["analyst", "researcher", "associate", "manager"],
                "filter_mode": "people",
                "people_filter": spec,
                "limit": limit,
            },
        },
    )


_BANKING_FINANCE_DF = pd.DataFrame(
    {
        "First Name": ["Ava", "Ben", "Cara", "Dan", "Eve", "Finn"],
        "Last Name": ["Stone", "Royce", "Kim", "Webb", "Park", "Cole"],
        "Occupation": [
            "Investment Banking Analyst",
            "Corporate Banking Analyst",
            "Wealth Management Analyst",
            "Quantitative Researcher",
            "Portfolio Analyst",
            "M&A Analyst",
        ],
        "Employer": [
            "Goldman Sachs",
            "Bank of America",
            "UBS",
            "Two Sigma",
            "Citadel",
            "Evercore",
        ],
        "LinkedIn URL": ["", "", "", "", "", ""],
    }
)


def _names(result):
    return {(row["First Name"], row["Last Name"]) for row in result["rows"]}


def test_broad_banking_filter_counts_banking_titles_and_employers_not_only_ib():
    result = _people_industry_filter(_BANKING_FINANCE_DF, "banking")
    names = _names(result)
    # Broad banking includes IB, corporate banking, wealth management and M&A.
    assert ("Ava", "Stone") in names
    assert ("Ben", "Royce") in names
    assert ("Cara", "Kim") in names
    assert ("Finn", "Cole") in names
    # Hedge-fund finance rows are not broad banking.
    assert ("Dan", "Webb") not in names
    assert ("Eve", "Park") not in names
    assert result["total_matches"] == 4
    assert len(result["rows"]) == result["total_matches"]


def test_investment_banking_filter_stays_narrow():
    result = _people_industry_filter(_BANKING_FINANCE_DF, "investment_banking", query_scope="subindustry")
    names = _names(result)
    assert ("Ava", "Stone") in names  # explicit "Investment Banking" title
    assert ("Finn", "Cole") in names  # M&A at an advisory bank is investment banking
    # Corporate banking and wealth management are broad banking but not IB,
    # and hedge-fund finance rows are never IB.
    assert ("Ben", "Royce") not in names
    assert ("Cara", "Kim") not in names
    assert ("Dan", "Webb") not in names
    assert ("Eve", "Park") not in names
    # Narrow IB is strictly a subset of broad banking (which counts 4 here).
    assert result["total_matches"] == 2


def test_finance_exclusion_filter_returns_finance_rows_without_banking():
    result = _people_industry_filter(
        _BANKING_FINANCE_DF,
        "finance",
        excluded=["banking", "investment_banking"],
        query_scope="industry_exclusion",
    )
    names = _names(result)
    # Only the hedge-fund finance rows remain; every banking/IB row is excluded.
    assert ("Dan", "Webb") in names
    assert ("Eve", "Park") in names
    assert ("Ava", "Stone") not in names
    assert ("Ben", "Royce") not in names
    assert ("Cara", "Kim") not in names
    assert ("Finn", "Cole") not in names
    assert result["total_matches"] == 2
    # operation_results.rows is the source of truth: the final included set only,
    # never the excluded banking rows or candidate rows.
    assert len(result["rows"]) == result["total_matches"] == 2


def test_world_bank_government_employer_is_not_counted_as_finance():
    df = pd.DataFrame(
        {
            "First Name": ["Gita"],
            "Last Name": ["Rao"],
            "Occupation": ["Research Analyst"],
            "Employer": ["World Bank"],
            "LinkedIn URL": [""],
        }
    )
    result = _people_industry_filter(df, "finance")
    assert result["total_matches"] == 0
    assert result["rows"] == []


def _email_predicate_operation(quantifier="any", operator="email_domain_not_in"):
    return {
        "type": "filter_predicates",
        "params": {
            "logic": "and",
            "predicates": [
                {
                    "semantic_column": "email",
                    "columns": ["email"],
                    "operator": operator,
                    "values": ["cornell.edu"],
                    "include_subdomains": True,
                    "quantifier": quantifier,
                    "require_valid_value": True,
                }
            ],
            "return_columns": ["first_name", "email"],
            "limit": 500,
        },
    }


def test_filter_predicates_non_cornell_any_returns_exact_verified_rows():
    df = pd.DataFrame(
        [
            {"first_name": "A", "email": "a@cornell.edu"},
            {"first_name": "B", "email": "b@gmail.com"},
            {"first_name": "C", "email": "c@alumni.cornell.edu"},
            {"first_name": "D", "email": "d@cornell.edu; d@yahoo.com"},
            {"first_name": "E", "email": ""},
            {"first_name": "F", "email": "not an email"},
            {"first_name": "G", "email": "g@cornell.edu.com"},
            {"first_name": "H", "email": "h@cornelltech.io"},
        ]
    )

    result = execute_operation(df, _email_predicate_operation())

    assert result["status"] == "ok"
    assert [row["First Name"] for row in result["rows"]] == ["B", "D", "G", "H"]
    assert [row["Matching Email"] for row in result["rows"]] == [
        "b@gmail.com",
        "d@yahoo.com",
        "g@cornell.edu.com",
        "h@cornelltech.io",
    ]
    assert result["metrics"]["rows_matched"] == 4
    assert result["metrics"]["rows_returned"] == 4
    assert result["metrics"]["post_verification_removed_count"] == 0
    assert len(result["rows"]) == 4


def test_filter_predicates_email_quantifiers_any_all_and_none_are_distinct():
    df = pd.DataFrame(
        {
            "first_name": ["Cornell", "External", "Mixed", "Blank"],
            "email": [
                "a@cornell.edu",
                "b@gmail.com",
                "c@cornell.edu; c@yahoo.com",
                "",
            ],
        }
    )

    any_result = execute_operation(df, _email_predicate_operation("any", "email_domain_not_in"))
    all_result = execute_operation(df, _email_predicate_operation("all", "email_domain_not_in"))
    none_result = execute_operation(df, _email_predicate_operation("none", "email_domain_in"))

    assert [row["First Name"] for row in any_result["rows"]] == ["External", "Mixed"]
    assert [row["First Name"] for row in all_result["rows"]] == ["External"]
    assert [row["First Name"] for row in none_result["rows"]] == ["External"]


def test_filter_predicates_rejects_unknown_operator_and_column():
    df = pd.DataFrame({"email": ["a@gmail.com"]})
    unknown_operator = _email_predicate_operation()
    unknown_operator["params"]["predicates"][0]["operator"] = "regex_python"
    unknown_column = _email_predicate_operation()
    unknown_column["params"]["predicates"][0]["columns"] = ["Missing Email"]

    first = execute_operation(df, unknown_operator)
    second = execute_operation(df, unknown_column)

    assert first["status"] == "error"
    assert "Unsupported" in first["error"]
    assert second["status"] == "error"
    assert "not found" in second["error"]


def _composite_people_operation(constraint_logic="and", limit=100):
    people_operation = {
        "type": "contains_any",
        "params": {
            "columns": ["Employer", "Occupation"],
            "terms": ["software", "engineer", "google", "microsoft", "technology", "tech"],
            "filter_mode": "people",
            "people_filter": {
                "filter_type": "industry",
                "industry": "tech",
                "industries": ["tech"],
                "query_scope": "industry",
                "capability": "offers_software_engineering_internships",
            },
            "display_columns": [
                "First Name",
                "Last Name",
                "Occupation",
                "Employer",
                "LinkedIn URL",
            ],
            "limit": limit,
        },
    }
    return {
        "type": "composite_people_filter",
        "params": {
            "people_operation": people_operation,
            "constraint_logic": constraint_logic,
            "logic": "and",
            "predicates": [
                {
                    "semantic_column": "email",
                    "columns": ["Email1", "Email2"],
                    "operator": "email_domain_not_in",
                    "values": ["cornell.edu"],
                    "include_subdomains": True,
                    "quantifier": "any",
                    "require_valid_value": True,
                }
            ],
            "limit": limit,
        },
    }


def test_composite_people_filter_is_the_stable_row_intersection_of_standalone_sets():
    df = pd.DataFrame(
        [
            {
                "First Name": "A",
                "Last Name": "One",
                "Employer": "Google",
                "Occupation": "Software Engineer",
                "Email1": "a@gmail.com",
                "Email2": "",
            },
            {
                "First Name": "B",
                "Last Name": "Two",
                "Employer": "Google",
                "Occupation": "Product Manager",
                "Email1": "b@cornell.edu",
                "Email2": "",
            },
            {
                "First Name": "C",
                "Last Name": "Three",
                "Employer": "High School",
                "Occupation": "Teacher",
                "Email1": "c@yahoo.com",
                "Email2": "",
            },
            {
                "First Name": "D",
                "Last Name": "Four",
                "Employer": "Microsoft",
                "Occupation": "Finance Manager",
                "Email1": "d@cornell.edu",
                "Email2": "d@gmail.com",
            },
        ]
    )
    original = df.copy(deep=True)
    operation = _composite_people_operation()
    people_result = execute_operation(df, operation["params"]["people_operation"])
    exact_result = execute_operation(
        df,
        {
            "type": "filter_predicates",
            "params": {
                "logic": "and",
                "predicates": operation["params"]["predicates"],
                "return_columns": ["First Name", "Last Name"],
                "limit": 100,
            },
        },
    )
    combined = execute_operation(df, operation)

    fuzzy_ids = {row["First Name"] for row in people_result["direct_rows"]}
    exact_ids = {row["First Name"] for row in exact_result["rows"]}
    combined_ids = {row["First Name"] for row in combined["direct_rows"]}
    assert fuzzy_ids == {"A", "B", "D"}
    assert exact_ids == {"A", "C", "D"}
    assert combined_ids == fuzzy_ids & exact_ids == {"A", "D"}
    assert [row["Matching Email"] for row in combined["direct_rows"]] == [
        "a@gmail.com",
        "d@gmail.com",
    ]
    assert combined["metrics"]["fuzzy_direct_count_before_predicates"] == 3
    assert combined["metrics"]["exact_predicate_match_count"] == 3
    assert combined["metrics"]["direct_count_after_intersection"] == 2
    assert combined["metrics"]["post_verification_removed_count"] == 0
    assert not any("__alumniai" in str(row).casefold() for row in combined["direct_rows"])
    assert "debug" not in combined
    pd.testing.assert_frame_equal(df, original)


def test_composite_display_limit_does_not_change_the_verified_match_count():
    df = pd.DataFrame(
        {
            "First Name": ["A", "B"],
            "Last Name": ["One", "Two"],
            "Employer": ["Google", "Microsoft"],
            "Occupation": ["Software Engineer", "Product Manager"],
            "Email1": ["a@gmail.com", "b@yahoo.com"],
            "Email2": ["", ""],
        }
    )

    result = execute_operation(df, _composite_people_operation(limit=1))

    assert result["total_matches"] == 2
    assert result["direct_count"] == 2
    assert len(result["direct_rows"]) == 2
    assert result["metrics"]["matched_row_count"] == 2
    assert result["metrics"]["returned_row_count"] == 1
    assert result["metrics"]["displayed_count"] == 1
    assert len(result["rows"]) == 1
    assert any(warning.get("type") == "display_limit_applied" for warning in result["warnings"])


def test_composite_people_filter_intersects_direct_adjacent_and_uncertain_buckets(monkeypatch):
    df = pd.DataFrame(
        {
            "First Name": list("ABCDEF"),
            "Last Name": ["Person"] * 6,
            "Occupation": ["Role"] * 6,
            "Employer": ["Company"] * 6,
            "Email1": [
                "a@gmail.com",
                "b@cornell.edu",
                "c@yahoo.com",
                "d@alumni.cornell.edu",
                "e@cornell.edu.com",
                "f@cs.cornell.edu",
            ],
            "Email2": [""] * 6,
        }
    )

    def fake_people_filter(working, params, assumptions):
        internal_column = params["_internal_row_id_column"]

        def projected(row_id):
            row = working.iloc[row_id]
            return {
                "First Name": row["First Name"],
                "Last Name": row["Last Name"],
                "Occupation": row["Occupation"],
                "Employer": row["Employer"],
                toolkit.INTERNAL_ROW_ID_KEY: int(row[internal_column]),
            }

        direct = [projected(0), projected(1)]
        adjacent = [projected(2), projected(3)]
        uncertain = [projected(4), projected(5)]
        return {
            "status": "ok",
            "intent": "people_filter",
            "operation_type": "contains_any",
            "rows": direct,
            "direct_rows": direct,
            "adjacent_rows": adjacent,
            "uncertain_rows": uncertain,
            "direct_count": 2,
            "adjacent_count": 2,
            "uncertain_count": 2,
            "total_matches": 2,
            "visible_columns": ["First Name", "Last Name", "Occupation", "Employer"],
            "columns": ["First Name", "Last Name", "Occupation", "Employer"],
            "metrics": {},
            "warnings": [],
            "row_sections": [],
        }

    monkeypatch.setattr(toolkit, "_op_contains_any", fake_people_filter)

    result = execute_operation(df, _composite_people_operation())

    assert [row["First Name"] for row in result["direct_rows"]] == ["A"]
    assert [row["First Name"] for row in result["adjacent_rows"]] == ["C"]
    assert [row["First Name"] for row in result["uncertain_rows"]] == ["E"]
    assert result["direct_count"] == 1
    assert result["adjacent_count"] == 1
    assert result["uncertain_count"] == 1
    assert result["metrics"]["post_verification_removed_count"] == 0
    for bucket in ["direct_rows", "adjacent_rows", "uncertain_rows"]:
        assert all(row["Matching Email"] for row in result[bucket])
        assert all(toolkit.INTERNAL_ROW_ID_KEY not in row for row in result[bucket])


def test_composite_people_filter_or_uses_stable_row_union(monkeypatch):
    df = pd.DataFrame(
        {
            "First Name": ["Fuzzy", "Exact", "Neither"],
            "Last Name": ["A", "B", "C"],
            "Occupation": ["Role"] * 3,
            "Employer": ["Company"] * 3,
            "Email1": ["f@cornell.edu", "e@gmail.com", "n@cornell.edu"],
            "Email2": [""] * 3,
        }
    )

    def fake_people_filter(working, params, assumptions):
        internal_column = params["_internal_row_id_column"]
        row = {
            "First Name": "Fuzzy",
            "Last Name": "A",
            "Occupation": "Role",
            "Employer": "Company",
            toolkit.INTERNAL_ROW_ID_KEY: int(working.iloc[0][internal_column]),
        }
        return {
            "status": "ok",
            "intent": "people_filter",
            "rows": [row],
            "direct_rows": [row],
            "adjacent_rows": [],
            "uncertain_rows": [],
            "direct_count": 1,
            "adjacent_count": 0,
            "uncertain_count": 0,
            "total_matches": 1,
            "visible_columns": ["First Name", "Last Name", "Occupation", "Employer"],
            "columns": ["First Name", "Last Name", "Occupation", "Employer"],
            "metrics": {},
            "warnings": [],
            "row_sections": [],
        }

    monkeypatch.setattr(toolkit, "_op_contains_any", fake_people_filter)

    result = execute_operation(df, _composite_people_operation(constraint_logic="or"))

    assert {row["First Name"] for row in result["direct_rows"]} == {"Fuzzy", "Exact"}
    assert result["total_matches"] == 2
    assert result["constraint_logic"] == "or"
