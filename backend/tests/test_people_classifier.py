"""Unit tests for the query-aware multi-label people classifier.

These lock in the core principle: a broad keyword hit only makes a row a
candidate; strict classification decides final inclusion.
"""

import pytest

from app.services.people_classifier import (
    classify_candidate,
    classify_employer_industries,
    classify_job_functions,
    profile_row,
    query_spec_from_filter,
)


CONSULTING = {"industries": ["consulting"]}
FINANCE = {"industries": ["finance"]}
FINANCE_CONSULTING = {"industries": ["consulting"], "required_industries": ["finance"]}
CONSULTING_OR_STRATEGY = {"industries": ["consulting"], "include_functions": ["internal_strategy"]}
CONSULTING_WITH_ADJACENT = {"industries": ["consulting"], "include_adjacent": True}
TECH = {"industries": ["tech"], "query_scope": "industry"}
TECHNICAL_ROLE = {"industries": ["tech"], "query_scope": "technical_role"}
INVESTMENT_BANKING = {"industries": ["investment_banking"], "query_scope": "subindustry"}
FINANCE_NOT_BANKING = {"industries": ["finance"], "excluded_industries": ["banking", "investment_banking"]}
FINANCE_OUTSIDE_IB = {"industries": ["finance"], "excluded_industries": ["investment_banking"]}
MARKETING = {"industries": ["marketing"]}
MARKETING_OR_GROWTH = {"industries": ["marketing"], "include_functions": ["marketing_growth"]}
OPERATIONS = {"industries": ["operations"]}
GOVERNMENT_LEGAL = {"industries": ["government_legal"]}


def consulting(occupation, employer):
    return classify_candidate(occupation, employer, CONSULTING)


# ---------------------------------------------------------------------------
# Consulting: direct matches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("occupation", "employer"),
    [
        ("Senior Manager, Risk Consulting", "EY"),
        ("Management Consultant", "KPMG"),
        ("Principal Consultant", ""),
        ("Independent Consultant", ""),
        ("Compensation Consultant", ""),
        ("Technology Consultant", "Acme Corp"),
        ("Strategy Consultant", "Unknown Family Business"),
        ("Implementation Consultant", ""),
        ("Transaction Advisory", "KPMG"),
        ("Transaction Advisory Associate", "EY"),
        ("Deal Advisory", "Deloitte"),
        ("Valuation Advisory", "Grant Thornton"),
        ("Restructuring Advisory", "Alvarez & Marsal"),
        ("Restructuring Advisory", "FTI Consulting"),
        ("Risk Advisory Manager", "PwC"),
        ("Financial Advisory", "Deloitte"),
    ],
)
def test_consulting_direct_matches(occupation, employer):
    result = consulting(occupation, employer)
    assert result["classification"] == "direct_match", result["internal_reason"]
    assert result["count_as_match"] is True
    assert result["confidence"] >= 0.70


def test_plausible_title_at_known_consulting_firm_is_direct():
    result = consulting("Partner", "McKinsey")
    assert result["classification"] == "direct_match"
    assert result["count_as_match"] is True


# ---------------------------------------------------------------------------
# Consulting: adjacent (never counted by default)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("occupation", "employer"),
    [
        ("Head of Strategy", "Hershey"),
        ("Director, Premium Subscription Strategy", "Spotify"),
        ("Director, Product Strategy & Operations", "ZoomInfo"),
        ("Product Manager", "Morgan Stanley"),
        ("Corporate Strategy Lead", "Delta Air Lines"),
        ("Business Operations Manager", "Hershey"),
        ("Chief of Staff", "ZoomInfo"),
        ("Transaction Strategy", "Hershey"),
        ("General Manager", "Hershey"),
    ],
)
def test_consulting_adjacent_rows_do_not_count(occupation, employer):
    result = consulting(occupation, employer)
    assert result["classification"] == "adjacent", result["internal_reason"]
    assert result["count_as_match"] is False


def test_confident_adjacent_rows_never_sneak_in_via_confidence():
    result = consulting("Director of Strategy", "Spotify")
    assert result["classification"] == "adjacent"
    assert result["count_as_match"] is False


# ---------------------------------------------------------------------------
# Consulting: non-matches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("occupation", "employer"),
    [
        ("Attorney", ""),
        ("Judicial Law Clerk", ""),
        ("Attorney", "Skadden"),
        ("Investment Banking Analyst", "Goldman Sachs"),
        ("Private Equity Associate", "Blackstone"),
        ("Portfolio Manager", "BlackRock"),
        ("Trader", "Jane Street"),
        ("Equity Research Associate", "Morgan Stanley"),
        ("Wealth Management Advisor", "Merrill Lynch"),
        ("Corporate Finance Analyst", "Hershey"),
        ("Software Engineer", "Google"),
        ("Teacher", "Latin School of Chicago"),
    ],
)
def test_consulting_non_matches(occupation, employer):
    result = consulting(occupation, employer)
    assert result["classification"] == "non_match", result["internal_reason"]
    assert result["count_as_match"] is False


@pytest.mark.parametrize("occupation", ["Head of Strategy", "General Manager", "Business Operations", "Transaction Strategy"])
def test_single_weak_keywords_never_count(occupation):
    result = consulting(occupation, "Some Operating Company")
    assert result["count_as_match"] is False
    assert result["classification"] in {"adjacent", "non_match"}


def test_blank_row_is_uncertain_not_counted():
    result = consulting("", "")
    assert result["classification"] == "uncertain"
    assert result["count_as_match"] is False


# ---------------------------------------------------------------------------
# Multi-label intersections (consulting and finance are not mutually exclusive)
# ---------------------------------------------------------------------------

def test_risk_consulting_at_ey_is_consulting_and_finance_related():
    result = consulting("Senior Manager, Risk Consulting", "EY")
    assert result["count_as_match"] is True
    assert "consulting_professional_services" in result["employer_industry"]
    assert "consulting_advisory" in result["job_function"]
    assert "risk_consulting" in result["specialties"]

    intersection = classify_candidate("Senior Manager, Risk Consulting", "EY", FINANCE_CONSULTING)
    assert intersection["count_as_match"] is True


def test_transaction_advisory_at_kpmg_is_consulting_and_finance_related():
    result = classify_candidate("Transaction Advisory", "KPMG", FINANCE_CONSULTING)
    assert result["classification"] == "direct_match"
    assert result["count_as_match"] is True
    assert "transaction_advisory" in result["specialties"]


def test_investment_banker_is_investment_banking_not_plain_finance_or_consulting():
    investment_banking_result = classify_candidate("Investment Banking Analyst", "Goldman Sachs", INVESTMENT_BANKING)
    assert investment_banking_result["count_as_match"] is True

    finance_result = classify_candidate("Investment Banking Analyst", "Goldman Sachs", FINANCE)
    assert finance_result["count_as_match"] is False

    consulting_result = classify_candidate("Investment Banking Analyst", "Goldman Sachs", CONSULTING)
    assert consulting_result["classification"] == "non_match"
    assert consulting_result["count_as_match"] is False


def test_product_manager_at_morgan_stanley_is_financial_services_product_not_consulting():
    profile = profile_row("Product Manager", "Morgan Stanley")
    assert "financial_services" in profile["employer_industry"]
    assert "product" in profile["job_function"]

    result = classify_candidate("Product Manager", "Morgan Stanley", CONSULTING)
    assert result["count_as_match"] is False


def test_pure_consultant_without_finance_context_is_excluded_from_finance_consulting():
    result = classify_candidate("Education Consultant", "Independent", FINANCE_CONSULTING)
    assert result["count_as_match"] is False


# ---------------------------------------------------------------------------
# Query-specific behavior: the query, not just the row, decides inclusion
# ---------------------------------------------------------------------------

def test_consulting_or_strategy_query_includes_internal_strategy():
    result = classify_candidate("Head of Strategy", "Hershey", CONSULTING_OR_STRATEGY)
    assert result["classification"] == "direct_match"
    assert result["count_as_match"] is True

    # Plain consulting query still excludes the same row.
    assert classify_candidate("Head of Strategy", "Hershey", CONSULTING)["count_as_match"] is False


def test_include_adjacent_counts_adjacent_rows_but_keeps_label():
    result = classify_candidate("Director, Product Strategy & Operations", "ZoomInfo", CONSULTING_WITH_ADJACENT)
    assert result["classification"] == "adjacent"
    assert result["count_as_match"] is True


def test_finance_query_counts_finance_roles_that_are_not_consulting():
    for occupation, employer in [
        ("Private Equity Associate", "Blackstone"),
        ("Portfolio Manager", "BlackRock"),
        ("Risk Analyst", "Millennium Management"),
    ]:
        result = classify_candidate(occupation, employer, FINANCE)
        assert result["count_as_match"] is True, (occupation, employer)


# ---------------------------------------------------------------------------
# New taxonomy policies: tech scope, investment banking, finance exclusions,
# marketing, operations, and government/legal.
# ---------------------------------------------------------------------------

def test_tech_known_employer_and_technical_role_scope():
    assert classify_candidate("Head of Growth", "Spotify", TECH)["count_as_match"] is True
    assert classify_candidate("Software Engineer", "Local Bakery", TECH)["count_as_match"] is True

    marketing_at_tech = classify_candidate("Marketing Manager", "Google", TECH)
    assert marketing_at_tech["count_as_match"] is False

    nontechnical = classify_candidate("Head of Growth", "Spotify", TECHNICAL_ROLE)
    assert nontechnical["classification"] == "non_match"
    assert nontechnical["count_as_match"] is False


def test_tech_weak_product_or_analytics_titles_do_not_count_without_context():
    for title in ["Product Manager", "Analytics Manager", "Head of Growth"]:
        result = classify_candidate(title, "Local Bakery", TECH)
        assert result["count_as_match"] is False


def test_investment_banking_requires_title_or_context_not_employer_only():
    assert classify_candidate("Investment Banking Analyst", "Goldman Sachs", INVESTMENT_BANKING)["count_as_match"] is True
    assert classify_candidate("Software Engineer", "Goldman Sachs", INVESTMENT_BANKING)["classification"] == "non_match"
    assert classify_candidate("Risk Analyst", "JPMorgan", INVESTMENT_BANKING)["classification"] == "non_match"
    generic = classify_candidate("Analyst", "Evercore", INVESTMENT_BANKING)
    assert generic["classification"] == "uncertain"
    assert generic["count_as_match"] is False


def test_banking_and_finance_are_separate_with_user_exclusions():
    assert classify_candidate("Corporate Banking Analyst", "Citi", {"industries": ["banking"]})["count_as_match"] is True
    assert classify_candidate("Portfolio Manager", "BlackRock", {"industries": ["banking"]})["count_as_match"] is False

    assert classify_candidate("Portfolio Manager", "BlackRock", FINANCE_NOT_BANKING)["count_as_match"] is True
    assert classify_candidate("Corporate Banking Analyst", "Citi", FINANCE_NOT_BANKING)["count_as_match"] is False
    assert classify_candidate("Investment Banking Analyst", "Goldman Sachs", FINANCE_OUTSIDE_IB)["count_as_match"] is False
    assert classify_candidate("Risk Analyst", "Millennium Management", FINANCE_OUTSIDE_IB)["count_as_match"] is True
    assert classify_candidate("Risk Analyst", "JPMorgan", FINANCE_OUTSIDE_IB)["count_as_match"] is False


def test_marketing_policy_excludes_generic_growth_for_plain_marketing():
    assert classify_candidate("Growth Marketing Manager", "Acme", MARKETING)["count_as_match"] is True
    plain_growth = classify_candidate("Head of Growth", "Spotify", MARKETING)
    assert plain_growth["classification"] == "adjacent"
    assert plain_growth["count_as_match"] is False
    expanded = classify_candidate("Head of Growth", "Spotify", MARKETING_OR_GROWTH)
    assert expanded["count_as_match"] is False
    assert classify_candidate("Product Manager", "Spotify", MARKETING)["count_as_match"] is False


def test_operations_policy_requires_operations_context():
    assert classify_candidate("Operations Manager", "Acme", OPERATIONS)["count_as_match"] is True
    assert classify_candidate("Business Operations Analyst", "Acme", OPERATIONS)["count_as_match"] is True
    assert classify_candidate("Strategy and Operations Manager", "Acme", OPERATIONS)["count_as_match"] is True
    assert classify_candidate("Supply Chain Analyst", "Acme", OPERATIONS)["count_as_match"] is True
    assert classify_candidate("Business Analyst", "Acme", OPERATIONS)["count_as_match"] is False
    assert classify_candidate("Strategy Manager", "Acme", OPERATIONS)["count_as_match"] is False
    assert classify_candidate("General Manager", "Acme", OPERATIONS)["count_as_match"] is False


def test_government_legal_policy_requires_direct_government_or_legal_signal():
    assert classify_candidate("Policy Analyst", "City Government", GOVERNMENT_LEGAL)["count_as_match"] is True
    assert classify_candidate("Attorney", "Skadden", GOVERNMENT_LEGAL)["count_as_match"] is True
    assert classify_candidate("Program Manager", "Public School", GOVERNMENT_LEGAL)["count_as_match"] is False
    assert classify_candidate("Operations Analyst", "Public Health Clinic", GOVERNMENT_LEGAL)["count_as_match"] is False


def test_adjacent_and_uncertain_are_not_counted_by_default():
    assert classify_candidate("Head of Growth", "Spotify", MARKETING)["classification"] == "adjacent"
    assert classify_candidate("Head of Growth", "Spotify", MARKETING)["count_as_match"] is False
    assert classify_candidate("Analyst", "Evercore", INVESTMENT_BANKING)["classification"] == "uncertain"
    assert classify_candidate("Analyst", "Evercore", INVESTMENT_BANKING)["count_as_match"] is False


# ---------------------------------------------------------------------------
# Profiles, output shape, and robustness
# ---------------------------------------------------------------------------

def test_result_shape_matches_structured_output_contract():
    result = classify_candidate("Management Consultant", "KPMG", CONSULTING, row_id=7)
    assert result["row_id"] == 7
    for key in ["classification", "count_as_match", "confidence", "employer_industry", "job_function", "specialties", "internal_reason"]:
        assert key in result
    assert isinstance(result["employer_industry"], list)
    assert isinstance(result["job_function"], list)
    assert isinstance(result["specialties"], list)


def test_employer_can_span_multiple_industries():
    labels = classify_employer_industries("Spotify")
    assert "technology" in labels
    assert "media" in labels


def test_strategy_consultant_is_not_internal_strategy():
    functions = classify_job_functions("Strategy Consultant")
    assert "consulting_advisory" in functions
    assert "internal_strategy" not in functions


def test_missing_and_nan_values_are_handled():
    for value in [None, float("nan"), "nan", "  "]:
        result = classify_candidate(value, value, CONSULTING)
        assert result["classification"] in {"uncertain", "non_match"}
        assert result["count_as_match"] is False


def test_query_spec_from_filter_defaults():
    spec = query_spec_from_filter({"industry": "consulting"})
    assert spec["industries"] == ["consulting"]
    assert spec["include_adjacent"] is False
    assert spec["required_industries"] == []
    assert spec["include_functions"] == []

    rich = query_spec_from_filter(
        {
            "industries": ["consulting"],
            "required_industries": ["finance"],
            "include_functions": ["internal_strategy"],
            "include_adjacent": True,
        }
    )
    assert rich["required_industries"] == ["finance"]
    assert rich["include_functions"] == ["internal_strategy"]
    assert rich["include_adjacent"] is True


def test_model_classifier_only_used_for_uncertain_rows():
    calls = []

    def spying_model(employer, occupation, taxonomy):
        calls.append((employer, occupation))
        return {"belongs_to_industry": True, "confidence": 0.95, "classification": "confirmed", "reason": "boutique consultancy"}

    # Deterministic direct match: the model must not be consulted.
    direct = classify_candidate("Management Consultant", "KPMG", CONSULTING, model_classifier=spying_model)
    assert direct["count_as_match"] is True
    # Deterministic adjacent: the model must not promote it.
    adjacent = classify_candidate("Head of Strategy", "Hershey", CONSULTING, model_classifier=spying_model)
    assert adjacent["classification"] == "adjacent"
    assert calls == []

    # Ambiguous row: the model may confirm it.
    uncertain = classify_candidate("Advisor", "Bluestone Partners", CONSULTING, model_classifier=spying_model)
    assert calls, "expected the model to be consulted for an ambiguous row"
    assert uncertain["classification"] == "direct_match"
