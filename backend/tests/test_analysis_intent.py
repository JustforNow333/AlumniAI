import pandas as pd

from app.services.analysis_intent import (
    heuristic_intent,
    intent_to_analysis_plan,
    resolve_intent_semantic_columns,
    validate_analysis_intent,
)
from app.services.analysis_toolkit import build_dataset_context
from app.services.answer_schema import deterministic_answer_from_results
from app.services.analysis_executor import execute_analysis_plan
from app.services.intent_filter import apply_intent_filter


def tech_intent():
    return {
        "intent": "find_records",
        "target_entity": "rows",
        "user_goal": "Show alumni who appear to work in technology.",
        "concepts": [
            {
                "name": "tech_related",
                "definition": "People connected to software, data, AI, or technology employers.",
                "search_terms": ["software", "engineer", "developer", "data", "AI"],
                "known_entities": ["Google", "Microsoft", "Amazon"],
            }
        ],
        "semantic_columns": {
            "person_name": ["name", "full name", "nickname"],
            "occupation": ["occupation", "job title", "role", "position"],
            "employer": ["employer", "company", "organization"],
            "industry": ["industry", "sector"],
            "major": ["major", "degree", "field of study"],
        },
        "filters": [
            {
                "concept": "tech_related",
                "apply_to_semantic_columns": ["occupation", "employer", "industry", "major"],
                "match_mode": "contains_any",
            }
        ],
        "sort": None,
        "aggregation": None,
        "desired_output": {
            "format": "table",
            "semantic_columns": ["person_name", "occupation", "employer", "matched_reason"],
            "limit": 100,
        },
        "assumptions": [
            "Tech-related alumni are identified using occupation, employer, industry, or major text."
        ],
        "clarification_needed": False,
        "clarifying_question": None,
    }


def uppercase_context():
    df = pd.DataFrame(
        {
            "NICKNAME": ["Ada", "Grace"],
            "OCCUPATION": ["Software Engineer", "Teacher"],
            "EMPLOYER": ["Google", "High School"],
        }
    )
    return df, build_dataset_context(df)


def test_intent_inference_output_can_be_validated():
    intent, valid, error = validate_analysis_intent(tech_intent())

    assert valid is True
    assert error == ""
    assert intent["intent"] == "find_records"
    assert intent["filters"][0]["concept"] == "tech_related"
    assert intent["desired_output"]["format"] == "table"


def test_model_intent_schema_retains_only_allowlisted_row_predicates():
    raw = tech_intent()
    raw["row_predicates"] = {
        "logic": "and",
        "predicates": [
            {
                "semantic_column": "email",
                "operator": "email_domain_not_in",
                "values": ["cornell.edu"],
                "include_subdomains": True,
                "quantifier": "any",
                "require_valid_value": True,
            },
            {
                "semantic_column": "email",
                "operator": "regex_python",
                "values": [".*"],
            },
        ],
    }

    intent, valid, error = validate_analysis_intent(raw)

    assert valid is True
    assert error == ""
    assert [item["operator"] for item in intent["row_predicates"]["predicates"]] == [
        "email_domain_not_in"
    ]
    assert "regex_python" in intent["row_predicate_validation_errors"][0]


def test_semantic_occupation_resolves_to_uppercase_occupation():
    _df, context = uppercase_context()
    intent, _valid, _error = validate_analysis_intent(tech_intent())

    resolved = resolve_intent_semantic_columns(intent, context)

    assert resolved["occupation"] == "OCCUPATION"


def test_semantic_employer_resolves_to_uppercase_employer():
    _df, context = uppercase_context()
    intent, _valid, _error = validate_analysis_intent(tech_intent())

    resolved = resolve_intent_semantic_columns(intent, context)

    assert resolved["employer"] == "EMPLOYER"


def test_tech_related_intent_maps_to_contains_any_on_available_columns():
    _df, context = uppercase_context()
    intent, _valid, _error = validate_analysis_intent(tech_intent())

    plan = intent_to_analysis_plan(intent, context)

    assert plan["operations"][0]["type"] == "contains_any"
    assert plan["operations"][0]["params"]["columns"] == ["OCCUPATION", "EMPLOYER"]


def test_missing_optional_semantic_columns_do_not_fail_operation():
    df, context = uppercase_context()
    intent, _valid, _error = validate_analysis_intent(tech_intent())
    plan = intent_to_analysis_plan(intent, context)

    results = execute_analysis_plan(df, plan)

    assert results[0]["status"] == "ok"
    assert results[0]["metrics"]["rows_matched"] == 1
    assert results[0]["rows"][0]["NICKNAME"] == "Ada"


def test_unavailable_concepts_produce_clarification_instead_of_fake_results():
    df = pd.DataFrame({"NAME": ["A", "B"], "OCCUPATION": ["Engineer", "Teacher"]})
    context = build_dataset_context(df)
    intent = tech_intent()
    intent["concepts"] = [
        {
            "name": "gpa_related",
            "definition": "Student grade point average.",
            "search_terms": [],
            "known_entities": [],
        }
    ]
    intent["filters"] = [
        {
            "concept": "gpa_related",
            "apply_to_semantic_columns": ["gpa"],
            "match_mode": "contains_any",
        }
    ]
    intent["desired_output"]["semantic_columns"] = ["person_name", "gpa"]
    intent, _valid, _error = validate_analysis_intent(intent)

    plan = intent_to_analysis_plan(intent, context)

    assert plan["operations"] == []
    assert "no matching columns" in plan["cannot_answer_reason"].lower()


def test_final_answer_says_what_assumption_was_used_for_fuzzy_concept():
    df, context = uppercase_context()
    intent, _valid, _error = validate_analysis_intent(tech_intent())
    plan = intent_to_analysis_plan(intent, context)
    results = execute_analysis_plan(df, plan)

    answer = deterministic_answer_from_results("show me tech alumni", plan, results, context)

    rendered = " ".join(str(block) for block in answer["blocks"])
    assert "Tech-related alumni are identified" in rendered
    assert "occupation -> OCCUPATION" in rendered
    assert "employer -> EMPLOYER" in rendered


def test_validated_correlation_intent_maps_to_correlation_operation():
    df = pd.DataFrame({"Score": [1, 2, 3], "Giving": [10, 30, 50]})
    context = build_dataset_context(df)
    intent, valid, error = validate_analysis_intent(
        {
            "intent": "compare_groups",
            "target_entity": "columns",
            "user_goal": "Find numeric relationships.",
            "concepts": [],
            "semantic_columns": {},
            "filters": [],
            "sort": None,
            "aggregation": {"operation": "correlation"},
            "desired_output": {"format": "ranked_list", "semantic_columns": [], "limit": 20},
            "assumptions": [],
            "clarification_needed": False,
            "clarifying_question": None,
        }
    )

    plan = intent_to_analysis_plan(intent, context)

    assert valid is True
    assert error == ""
    assert intent["aggregation"]["operation"] == "correlation"
    assert plan["operations"] == [{"type": "correlation", "params": {"columns": None, "limit": 20}}]


def test_validated_date_summary_intent_maps_to_date_summary_operation():
    df = pd.DataFrame({"Last Contact": pd.to_datetime(["2026-01-01", "2026-02-01"])})
    context = build_dataset_context(df)
    intent, _valid, _error = validate_analysis_intent(
        {
            "intent": "aggregate",
            "target_entity": "columns",
            "user_goal": "Summarize dates.",
            "concepts": [],
            "semantic_columns": {},
            "filters": [],
            "sort": None,
            "aggregation": {"operation": "date_summary"},
            "desired_output": {"format": "metrics", "semantic_columns": [], "limit": 20},
            "assumptions": [],
            "clarification_needed": False,
            "clarifying_question": None,
        }
    )

    plan = intent_to_analysis_plan(intent, context)

    assert intent["aggregation"]["operation"] == "date_summary"
    assert plan["operations"] == [{"type": "date_summary", "params": {"columns": None}}]


def test_tech_company_query_infers_tech_company_terms():
    df, context = uppercase_context()

    intent = heuristic_intent("Which alumni work at a tech company?", context)
    plan = intent_to_analysis_plan(intent, context)

    concept_names = {concept["name"] for concept in intent["concepts"]}
    assert "tech_company" in concept_names
    operation = plan["operations"][0]
    assert operation["type"] == "contains_any"
    group = next(group for group in operation["params"]["column_term_groups"] if group["concept"] == "tech_company")
    assert group["columns"] == ["EMPLOYER"]
    assert "Google" in group["terms"]


def test_explicit_people_display_columns_respect_names_and_employers_only():
    df = pd.DataFrame(
        {
            "First Name": ["Ada"],
            "Last Name": ["Lovelace"],
            "Occupation": ["Engineer"],
            "Employer": ["Analytical Engines"],
            "LinkedIn URL": ["https://linkedin.example/ada"],
        }
    )
    context = build_dataset_context(df)

    intent = heuristic_intent("Show tech alumni names and employers only.", context)
    plan = intent_to_analysis_plan(intent, context)

    assert plan["operations"][0]["params"]["display_columns"] == [
        "First Name",
        "Last Name",
        "Employer",
    ]


def test_negated_optional_display_column_is_not_added():
    df = pd.DataFrame(
        {
            "First Name": ["Ada"],
            "Last Name": ["Lovelace"],
            "Occupation": ["Finance Analyst"],
            "Employer": ["Acme Capital"],
            "Major": ["Mathematics"],
            "LinkedIn URL": [""],
        }
    )
    context = build_dataset_context(df)

    intent = heuristic_intent("Show me alumni in finance. Do not include majors.", context)
    plan = intent_to_analysis_plan(intent, context)

    assert "Major" not in plan["operations"][0]["params"]["display_columns"]


def test_software_engineers_query_infers_role_terms():
    _df, context = uppercase_context()

    intent = heuristic_intent("Which alumni are software engineers?", context)
    plan = intent_to_analysis_plan(intent, context)

    assert intent["people_filter_spec"]["filter_type"] == "occupation"
    group = next(group for group in plan["operations"][0]["params"]["column_term_groups"] if group["concept"] == "target_occupation")
    assert group["columns"] == ["OCCUPATION"]
    assert "software engineer" in group["terms"]


def test_deterministic_location_and_year_filters_plan_without_model():
    df = pd.DataFrame(
        {
            "First Name": ["Ada", "Grace"],
            "Last Name": ["Lovelace", "Hopper"],
            "Title": ["Engineer", "Admiral"],
            "Employer": ["Analytical Engines", "Navy"],
            "LinkedIn URL": ["", ""],
            "Location": ["New York, NY", "San Francisco, CA"],
            "Graduation Year": [2023, 2024],
        }
    )
    context = build_dataset_context(df)

    location_plan = intent_to_analysis_plan(heuristic_intent("Show me alumni in New York.", context), context)
    year_plan = intent_to_analysis_plan(heuristic_intent("Show alumni who graduated in 2023.", context), context)

    assert location_plan["operations"][0]["type"] == "filter_contains"
    assert location_plan["operations"][0]["params"]["column"] == "location"
    assert location_plan["operations"][0]["params"]["terms"] == ["New York"]
    assert year_plan["operations"][0]["type"] == "filter_equals"
    assert year_plan["operations"][0]["params"]["column"] == "grad_year"
    assert year_plan["operations"][0]["params"]["value"] == 2023


def test_deterministic_missing_and_name_filters_plan_without_model():
    df = pd.DataFrame(
        {
            "first_name": ["Isabella", "Isabella"],
            "last_name": ["Khan", "Perez"],
            "title": ["Analyst", "Engineer"],
            "employer": ["AQR", "Launch Potato"],
            "linkedin_url": ["", "https://linkedin.com/in/example"],
        }
    )
    context = build_dataset_context(df)

    missing_plan = intent_to_analysis_plan(heuristic_intent("Show alumni with missing LinkedIn URLs.", context), context)
    name_plan = intent_to_analysis_plan(heuristic_intent("Show alumni named Isabella Khan.", context), context)

    assert missing_plan["operations"][0]["type"] == "filter_missing"
    assert missing_plan["operations"][0]["params"]["column"] == "linkedin_url"
    assert name_plan["operations"][0]["type"] == "contains_all"
    assert name_plan["operations"][0]["params"]["column_term_groups"][0]["terms"] == ["Isabella"]
    assert name_plan["operations"][0]["params"]["column_term_groups"][1]["terms"] == ["Khan"]


def test_deterministic_employer_contains_and_zero_result_domain_plans_without_model():
    df = pd.DataFrame(
        {
            "first_name": ["Ada"],
            "last_name": ["Lovelace"],
            "title": ["Engineer"],
            "employer": ["Capital Labs"],
            "major": ["Mathematics"],
            "linkedin_url": [""],
        }
    )
    context = build_dataset_context(df)

    employer_plan = intent_to_analysis_plan(
        heuristic_intent("Show alumni whose employer contains Capital.", context),
        context,
    )
    aerospace_plan = intent_to_analysis_plan(
        heuristic_intent("Show alumni in aerospace.", context),
        context,
    )

    assert employer_plan["operations"][0]["type"] == "filter_contains"
    assert employer_plan["operations"][0]["params"]["column"] == "employer"
    assert employer_plan["operations"][0]["params"]["terms"] == ["Capital"]
    assert employer_plan["operations"][0]["params"]["include_match_reason"] is False
    assert aerospace_plan["operations"][0]["type"] == "contains_any"
    assert aerospace_plan["operations"][0]["params"]["terms"] == ["aerospace"]
    assert aerospace_plan["operations"][0]["params"]["include_match_reason"] is False


def test_known_concept_with_no_terms_expands_from_library():
    _df, context = uppercase_context()
    intent, valid, error = validate_analysis_intent(
        {
            "intent": "find_records",
            "user_goal": "Find software engineers.",
            "concepts": [{"name": "software_engineer_role", "definition": "", "search_terms": [], "known_entities": []}],
            "filters": [
                {
                    "concept": "software_engineer_role",
                    "apply_to_semantic_columns": ["occupation"],
                    "match_mode": "contains_any",
                }
            ],
            "search_columns": {"occupation": ["occupation", "job title", "role"]},
            "display_columns": ["person_name", "occupation", "employer", "match_reason"],
            "limit": 100,
            "assumptions": [],
            "clarification_needed": False,
            "clarifying_question": None,
        }
    )

    plan = intent_to_analysis_plan(intent, context)

    assert valid is True
    assert error == ""
    assert "software engineer" in intent["concepts"][0]["search_terms"]
    assert plan["operations"][0]["params"]["column_term_groups"][0]["terms"]
    assert "no search terms" not in plan["cannot_answer_reason"].lower()


def test_no_relevant_columns_returns_clarification_plan():
    df = pd.DataFrame({"NICKNAME": ["Ada"], "GRAD YR": [2020]})
    context = build_dataset_context(df)

    intent = heuristic_intent("Show tech alumni", context)
    plan = intent_to_analysis_plan(intent, context)

    assert plan["operations"] == []
    assert "no matching columns" in plan["cannot_answer_reason"].lower()


# ---------------------------------------------------------------------------
# Hybrid regression: the model's rephrased user_goal must not distort
# finance/banking people routing. Planning re-classifies the *original*
# question, so misleading words the model adds ("or investment banking",
# "excluding banking") cannot change the deterministic filter spec.
# ---------------------------------------------------------------------------

def _people_dataset_context():
    df = pd.DataFrame(
        {
            "First Name": ["Ada"],
            "Last Name": ["Lovelace"],
            "Occupation": ["Analyst"],
            "Employer": ["Acme"],
            "LinkedIn URL": [""],
        }
    )
    return build_dataset_context(df)


def _model_find_records_intent(original_question, rephrased_user_goal):
    """Mimic a confident model find_records plan (the model never emits the
    people_filter intent) carrying a rephrased user_goal alongside the verbatim
    question stamped by infer_analysis_intent."""
    return {
        "intent": "find_records",
        "target_entity": "rows",
        "original_question": original_question,
        "user_goal": rephrased_user_goal,
        "concepts": [
            {
                "name": "people",
                "definition": "",
                "search_terms": ["analyst", "manager", "associate"],
                "known_entities": [],
            }
        ],
        "semantic_columns": {
            "occupation": ["occupation", "title", "role"],
            "employer": ["employer", "company"],
        },
        "filters": [
            {
                "concept": "people",
                "apply_to_semantic_columns": ["occupation", "employer"],
                "match_mode": "contains_any",
            }
        ],
        "desired_output": {"format": "table", "semantic_columns": ["person_name", "occupation", "employer"]},
    }


def _people_filter_from_plan(plan):
    params = plan["operations"][0]["params"]
    assert params.get("filter_mode") == "people"
    return params["people_filter"]


def test_model_rephrasing_in_banking_to_investment_banking_still_routes_to_broad_banking():
    context = _people_dataset_context()
    intent = _model_find_records_intent(
        "Find alumni in banking.",
        "Find alumni whose employer, title, or notes indicate banking or investment banking.",
    )
    pf = _people_filter_from_plan(intent_to_analysis_plan(intent, context))
    assert pf["industry"] == "banking"
    assert pf["query_scope"] == "industry"
    assert pf["excluded_industries"] == []


def test_model_rephrasing_work_at_banks_routes_to_broad_banking():
    context = _people_dataset_context()
    intent = _model_find_records_intent(
        "Which alumni work at banks?",
        "Find alumni whose employer appears to be a bank or financial institution.",
    )
    pf = _people_filter_from_plan(intent_to_analysis_plan(intent, context))
    assert pf["industry"] == "banking"
    assert pf["query_scope"] == "industry"


def test_model_rephrasing_finance_but_not_banking_keeps_finance_with_exclusions():
    context = _people_dataset_context()
    intent = _model_find_records_intent(
        "Show me alumni in finance but not banking.",
        "Find alumni in finance, while excluding those associated with banking.",
    )
    pf = _people_filter_from_plan(intent_to_analysis_plan(intent, context))
    assert pf["industry"] == "finance"
    assert pf["query_scope"] == "industry_exclusion"
    assert "banking" in pf["excluded_industries"]
    assert "investment_banking" in pf["excluded_industries"]


def test_model_rephrasing_finance_outside_ib_keeps_finance_excluding_ib():
    context = _people_dataset_context()
    intent = _model_find_records_intent(
        "Find finance alumni outside investment banking.",
        "Find alumni with a finance background, excluding investment banking roles or employers.",
    )
    pf = _people_filter_from_plan(intent_to_analysis_plan(intent, context))
    assert pf["industry"] == "finance"
    assert pf["query_scope"] == "industry_exclusion"
    assert "investment_banking" in pf["excluded_industries"]


def test_planner_routes_fuzzy_exact_and_combined_queries_to_distinct_safe_paths():
    df = pd.DataFrame(
        {
            "First Name": ["Ada", "Grace"],
            "Last Name": ["Lovelace", "Hopper"],
            "Occupation": ["Software Engineer", "Product Manager"],
            "Employer": ["Google", "Microsoft"],
            "Email": ["ada@gmail.com", "grace@cornell.edu"],
        }
    )
    context = build_dataset_context(df)
    fuzzy_question = "Which alumni work at a company that offers SWE internships of any kind?"
    exact_question = "Which alumni have a non-Cornell email?"
    combined_question = f"{fuzzy_question[:-1]} and have a non-Cornell email?"

    fuzzy_intent, _ = apply_intent_filter(
        fuzzy_question,
        context,
        heuristic_intent(fuzzy_question, context),
    )
    exact_intent, _ = apply_intent_filter(
        exact_question,
        context,
        heuristic_intent(exact_question, context),
    )
    combined_intent, _ = apply_intent_filter(
        combined_question,
        context,
        heuristic_intent(combined_question, context),
    )

    fuzzy_plan = intent_to_analysis_plan(fuzzy_intent, context)
    exact_plan = intent_to_analysis_plan(exact_intent, context)
    combined_plan = intent_to_analysis_plan(combined_intent, context)

    assert fuzzy_plan["operations"][0]["type"] == "contains_any"
    assert fuzzy_plan["operations"][0]["params"]["filter_mode"] == "people"
    assert exact_plan["operations"][0]["type"] == "filter_predicates"
    operation = combined_plan["operations"][0]
    assert operation["type"] == "composite_people_filter"
    assert operation["params"]["constraint_logic"] == "and"
    assert operation["params"]["people_operation"]["params"]["filter_mode"] == "people"
    assert operation["params"]["predicates"][0]["operator"] == "email_domain_not_in"
    assert len(combined_plan["operations"]) == 1


def test_model_literal_internship_search_cannot_replace_source_people_capability():
    df = pd.DataFrame(
        {
            "Occupation": ["Product Manager"],
            "Employer": ["Google"],
            "Email": ["a@gmail.com"],
        }
    )
    context = build_dataset_context(df)
    question = (
        "Which alumni work at a company that offers SWE internships "
        "and have a non-Cornell email?"
    )
    model_intent = {
        "intent": "find_records",
        "original_question": question,
        "user_goal": "Find literal SWE internship text and an employer email.",
        "concepts": [
            {
                "name": "literal_internship",
                "definition": "literal internship wording",
                "search_terms": ["SWE internship"],
                "known_entities": [],
            }
        ],
        "filters": [
            {
                "concept": "literal_internship",
                "apply_to_semantic_columns": ["employer"],
                "match_mode": "contains_any",
            }
        ],
        "desired_output": {"format": "table", "semantic_columns": []},
    }

    reconciled, trace = apply_intent_filter(question, context, model_intent)
    plan = intent_to_analysis_plan(reconciled, context)
    operation = plan["operations"][0]

    assert reconciled["people_filter_spec"]["capability"] == "offers_software_engineering_internships"
    assert operation["type"] == "composite_people_filter"
    assert operation["params"]["people_operation"]["params"]["people_filter"]["capability"] == (
        "offers_software_engineering_internships"
    )
    assert operation["params"]["people_operation"]["params"]["filter_mode"] == "people"
    assert trace["recognized_constraint_count"] == trace["planned_constraint_count"] == 2
    assert trace["intent_filter_people_filter_repaired"] is True
    result = execute_analysis_plan(df, plan)[0]
    assert result["status"] == "ok"
    assert result["total_matches"] == 1
    assert result["rows"][0]["Employer"] == "Google"
