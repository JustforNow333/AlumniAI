import pandas as pd

from app.services import ai_service
from app.services.analysis_intent import heuristic_intent
from app.services.analysis_toolkit import build_dataset_context
from app.services.intent_filter import apply_intent_filter


def _context(*columns):
    values = {column: ["a@gmail.com", "b@cornell.edu", ""] for column in columns}
    return build_dataset_context(pd.DataFrame(values))


def _apply(question, context, intent=None):
    intent = intent or heuristic_intent(question, context)
    return apply_intent_filter(question, context, intent)


def test_detects_non_cornell_email():
    intent, trace = _apply("Show all alumni with a non-Cornell email.", _context("Email"))
    predicate = intent["row_predicates"]["predicates"][0]
    assert predicate["operator"] == "email_domain_not_in"
    assert predicate["values"] == ["cornell.edu"]
    assert predicate["quantifier"] == "any"
    assert predicate["require_valid_value"] is True
    assert trace["intent_filter_source"] == "deterministic_explicit"


def test_detects_isnt_their_cornell_email_and_not_at_domain():
    context = _context("Email")
    first, _trace = _apply("Which alumni have an email that isn't their Cornell email?", context)
    second, _trace = _apply("Show alumni whose email is not @cornell.edu.", context)
    assert first["row_predicates"]["predicates"][0]["operator"] == "email_domain_not_in"
    assert second["row_predicates"]["predicates"][0]["operator"] == "email_domain_not_in"


def test_source_negation_rejects_contradictory_model_predicate():
    context = _context("Email")
    model_intent = {
        "intent": "find_records",
        "target_entity": "rows",
        "filters": [],
        "row_predicates": {
            "logic": "and",
            "predicates": [
                {
                    "semantic_column": "email",
                    "operator": "contains",
                    "values": ["cornell.edu"],
                    "quantifier": "any",
                }
            ],
        },
        "desired_output": {"format": "table", "semantic_columns": [], "limit": 100},
    }
    intent, trace = _apply("Show alumni whose email is not @cornell.edu.", context, model_intent)
    assert [item["operator"] for item in intent["row_predicates"]["predicates"]] == [
        "email_domain_not_in"
    ]
    assert trace["intent_filter_conflicts_removed"] == 1
    assert trace["intent_filter_repaired_model_output"] is True


def test_source_exact_constraint_removes_conflicting_fuzzy_filter_on_same_field():
    context = _context("Email")
    model_intent = {
        "intent": "find_records",
        "concepts": [
            {
                "name": "cornell_email",
                "definition": "Cornell email text",
                "search_terms": ["cornell.edu"],
                "known_entities": [],
            }
        ],
        "filters": [
            {
                "concept": "cornell_email",
                "apply_to_semantic_columns": ["email"],
                "match_mode": "contains_any",
            }
        ],
        "desired_output": {"format": "table", "semantic_columns": [], "limit": 100},
    }

    intent, trace = _apply("Show alumni whose email is not @cornell.edu.", context, model_intent)

    assert intent["filters"] == []
    assert intent["row_predicates"]["predicates"][0]["operator"] == "email_domain_not_in"
    assert trace["intent_filter_conflicts_removed"] == 1


def test_equivalent_model_predicate_is_deduplicated():
    context = _context("Email")
    model_intent = {
        "intent": "find_records",
        "filters": [],
        "row_predicates": {
            "logic": "and",
            "predicates": [
                {
                    "semantic_column": "email",
                    "operator": "email_domain_not_in",
                    "values": ["cornell.edu"],
                    "include_subdomains": True,
                    "quantifier": "any",
                    "require_valid_value": True,
                }
            ],
        },
        "desired_output": {"format": "table", "semantic_columns": [], "limit": 100},
    }
    intent, trace = _apply("Show alumni with a non-Cornell email.", context, model_intent)
    assert len(intent["row_predicates"]["predicates"]) == 1
    assert trace["intent_filter_deduplicated"] == 1


def test_broad_email_scope_resolves_all_email_columns_in_dataframe_order():
    context = _context("Cornell Email", "Personal Email", "Work Email", "Notes")
    intent, trace = _apply("Show alumni with any non-Cornell email in there.", context)
    assert intent["row_predicates"]["predicates"][0]["resolved_columns"] == [
        "Cornell Email",
        "Personal Email",
        "Work Email",
    ]
    assert trace["intent_filter_resolved_columns"] == ["Cornell Email", "Personal Email", "Work Email"]


def test_explicit_email_column_stays_scoped_to_that_column():
    context = _context("Personal Email", "Work Email")
    intent, _trace = _apply(
        "Show alumni whose Work Email column is outside cornell.edu.",
        context,
    )
    assert intent["row_predicates"]["predicates"][0]["resolved_columns"] == ["Work Email"]


def test_missing_required_semantic_column_returns_controlled_failure():
    context = build_dataset_context(pd.DataFrame({"First Name": ["Ada"]}))
    intent, trace = _apply("Show alumni with a non-Cornell email.", context)
    assert intent["clarification_needed"] is True
    assert "email-like column" in intent["clarifying_question"]
    assert trace["intent_filter_valid"] is False


def test_offline_gate_does_not_call_a_model(monkeypatch):
    class ExplodingClient:
        def __getattr__(self, _name):
            raise AssertionError("intent filter must not call a model")

    monkeypatch.setattr(ai_service, "client", ExplodingClient())
    intent, _trace = apply_intent_filter(
        "Show alumni with a non-Cornell email.",
        _context("Email"),
        {"intent": "unknown", "filters": [], "desired_output": {}},
    )
    assert intent["row_predicates"]["predicates"]


def test_fuzzy_industry_filter_remains_separate_from_exact_predicate(monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)
    df = pd.DataFrame(
        {
            "First Name": ["Ada"],
            "Occupation": ["Software Engineer"],
            "Employer": ["Google"],
            "Email": ["ada@gmail.com"],
        }
    )
    context = build_dataset_context(df)
    question = "Which alumni work in tech and have a non-Cornell email?"
    inferred = heuristic_intent(question, context)
    intent, _trace = apply_intent_filter(question, context, inferred)
    assert intent["filters"]
    assert any(concept["name"].startswith("tech") or concept["name"] == "software_engineer_role" for concept in intent["concepts"])
    assert intent["row_predicates"]["predicates"][0]["operator"] == "email_domain_not_in"


def test_only_gmail_preserves_narrow_domain_constraint():
    intent, _trace = _apply("Show alumni with only Gmail email addresses.", _context("Email"))
    predicate = intent["row_predicates"]["predicates"][0]
    assert predicate["operator"] == "email_domain_in"
    assert predicate["values"] == ["gmail.com"]
    assert predicate["quantifier"] == "all"


def test_exists_missing_equals_and_contains_constraints_are_typed():
    df = pd.DataFrame(
        {
            "Email": ["a@gmail.com"],
            "LinkedIn URL": [""],
            "Employer": ["Google"],
            "Occupation": ["Software Engineer"],
        }
    )
    context = build_dataset_context(df)
    cases = [
        ("Show alumni with an email.", "exists", "email"),
        ("Show alumni without LinkedIn URLs.", "missing", "linkedin_url"),
        ("Show alumni whose employer is Google.", "equals", "employer"),
        ("Show alumni whose employer is not Google.", "not_equals", "employer"),
        ("Show alumni whose title contains engineer.", "contains", "occupation"),
        ("Show alumni whose title does not contain intern.", "not_contains", "occupation"),
    ]
    for question, operator, semantic in cases:
        intent, _trace = _apply(question, context)
        predicate = intent["row_predicates"]["predicates"][0]
        assert predicate["operator"] == operator
        assert predicate["semantic_column"] == semantic


def test_unsupported_model_operator_returns_controlled_validation_failure():
    context = _context("Email")
    model_intent = {
        "intent": "find_records",
        "filters": [],
        "row_predicates": {
            "logic": "and",
            "predicates": [
                {
                    "semantic_column": "email",
                    "operator": "regex_python",
                    "values": [".*"],
                }
            ],
        },
        "row_predicate_validation_errors": ["Unsupported row predicate operator 'regex_python'."],
    }

    intent, trace = _apply("Show rows matching this exact email rule.", context, model_intent)

    assert intent["clarification_needed"] is True
    assert "Unsupported row predicate operator" in intent["clarifying_question"]
    assert trace["intent_filter_valid"] is False


def test_swe_internship_employer_clause_remains_a_fuzzy_people_filter():
    context = build_dataset_context(
        pd.DataFrame(
            {
                "Employer": ["Google", "Microsoft"],
                "Occupation": ["Product Manager", "Finance Manager"],
                "Email": ["a@gmail.com", "b@cornell.edu"],
            }
        )
    )
    question = "Which alumni work at a company that offers SWE internships of any kind?"

    intent, trace = _apply(question, context)

    assert intent["people_filter_spec"]["capability"] == "offers_software_engineering_internships"
    assert intent["people_filter_spec"]["filter_type"] == "industry"
    assert intent["people_filter_spec"]["industry"] == "tech"
    assert intent.get("row_predicates", {}).get("predicates", []) == []
    assert trace["has_fuzzy_people_filter"] is True
    assert trace["has_exact_row_predicates"] is False


def test_combined_source_preserves_both_clauses_and_and_relationship():
    context = build_dataset_context(
        pd.DataFrame(
            {
                "Employer": ["Google"],
                "Occupation": ["Product Manager"],
                "Email1": ["a@cornell.edu"],
                "Email2": ["a@gmail.com"],
            }
        )
    )
    question = (
        "Which alumni work at a company that offers SWE internships of any kind "
        "and have an email in the dataset that is not a Cornell email?"
    )

    intent, trace = _apply(question, context, {"intent": "unknown", "filters": []})

    assert intent["people_filter_spec"]["capability"] == "offers_software_engineering_internships"
    assert intent["row_predicates"]["predicates"][0]["operator"] == "email_domain_not_in"
    assert intent["logical_operator"] == "and"
    assert trace["has_fuzzy_people_filter"] is True
    assert trace["has_exact_row_predicates"] is True
    assert trace["is_composite_filter"] is True
    assert trace["recognized_constraint_count"] == 2
    assert trace["planned_constraint_count"] == 2
    assert trace["fuzzy_clause_dropped"] is False
    assert [clause["type"] for clause in trace["source_clauses"]] == [
        "fuzzy_people_filter",
        "exact_row_predicate",
    ]
    assert all(clause["preserved"] for clause in trace["source_clauses"])


def test_reversed_combined_clause_order_has_the_same_normalized_constraints():
    context = build_dataset_context(
        pd.DataFrame(
            {
                "Employer": ["Google"],
                "Occupation": ["Product Manager"],
                "Email": ["a@gmail.com"],
            }
        )
    )
    fuzzy_first, _ = _apply(
        "Which alumni work at companies offering SWE internships and have a non-Cornell email?",
        context,
        {"intent": "unknown", "filters": []},
    )
    exact_first, _ = _apply(
        "Which alumni have a non-Cornell email and work at companies offering SWE internships?",
        context,
        {"intent": "unknown", "filters": []},
    )

    assert fuzzy_first["people_filter_spec"] == exact_first["people_filter_spec"]
    assert fuzzy_first["row_predicates"] == exact_first["row_predicates"]
    assert fuzzy_first["logical_operator"] == exact_first["logical_operator"] == "and"


def test_model_cannot_reassign_alumni_email_clause_to_the_employer():
    context = build_dataset_context(
        pd.DataFrame(
            {
                "Employer": ["Google"],
                "Occupation": ["Engineer"],
                "Email": ["a@gmail.com"],
            }
        )
    )
    model_intent = {
        "intent": "find_records",
        "filters": [],
        "assumptions": ["The email belongs to the employer or a company recruiter."],
        "desired_output": {"format": "table"},
    }
    question = (
        "Which alumni work at a company that offers SWE internships and have a non-Cornell email?"
    )

    intent, trace = _apply(question, context, model_intent)

    assert intent["row_predicates"]["predicates"][0]["semantic_column"] == "email"
    assert intent["row_predicates"]["predicates"][0]["target_entity"] == "alumni_row"
    assert not any("belongs to the employer" in item.casefold() for item in intent["assumptions"])
    assert trace["intent_filter_model_assumptions_repaired"] == 1
    assert trace["intent_filter_repaired_model_output"] is True


def test_explicit_graduation_year_becomes_typed_predicate():
    context = build_dataset_context(
        pd.DataFrame({"Occupation": ["Software Engineer"], "Graduation Year": [2023]})
    )

    intent, _trace = _apply("Which software engineers graduated in 2023?", context)

    predicate = next(
        item
        for item in intent["row_predicates"]["predicates"]
        if item["semantic_column"] == "grad_year"
    )
    assert predicate["operator"] == "equals"
    assert predicate["values"] == [2023]
