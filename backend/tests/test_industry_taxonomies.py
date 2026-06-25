import pytest

from app.services.industry_taxonomies import (
    classify_people_question,
    get_taxonomy,
    industry_for_question,
    taxonomy_names,
)


EXPECTED_INDUSTRIES = {
    "tech",
    "consulting",
    "investment_banking",
    "banking",
    "finance",
    "healthcare",
    "law",
    "government_legal",
    "education",
    "marketing",
    "operations",
    "media",
    "nonprofit",
    "startups",
    "venture_capital",
    "private_equity",
}


def test_all_initial_taxonomies_exist_and_are_structured():
    assert EXPECTED_INDUSTRIES.issubset(set(taxonomy_names()))
    for name in EXPECTED_INDUSTRIES:
        taxonomy = get_taxonomy(name)
        assert taxonomy["industry"] == name
        assert taxonomy["criteria_label"]
        assert isinstance(taxonomy["aliases"], list)
        assert isinstance(taxonomy["title_keywords"], list)
        assert isinstance(taxonomy["employer_keywords"], list)
        assert isinstance(taxonomy["known_companies"], list)
        assert isinstance(taxonomy["exclusion_keywords"], list)
        assert isinstance(taxonomy["ambiguous_keywords"], list)
        assert 0 < taxonomy["confidence_threshold"] <= 1


def test_tech_taxonomy_merges_known_companies_config():
    taxonomy = get_taxonomy("tech")
    for company in ["Spotify", "Google", "FanAmp", "Cogni DAO", "Rune Technologies", "Twilio", "Workday", "InterSystems", "ICEYE"]:
        assert company in taxonomy["known_companies"]


def test_taxonomy_lookup_accepts_natural_names():
    assert get_taxonomy("VC")["industry"] == "venture_capital"
    assert get_taxonomy("PE")["industry"] == "private_equity"
    assert get_taxonomy("technology")["industry"] == "tech"
    assert get_taxonomy("not-an-industry") is None


@pytest.mark.parametrize(
    ("question", "industry"),
    [
        # Tech
        ("Which alumni work in tech?", "tech"),
        ("How many alumni are in software?", "tech"),
        ("Show me alumni at tech companies.", "tech"),
        ("Which alumni have technical roles?", "tech"),
        # Consulting
        ("Which alumni work in consulting?", "consulting"),
        ("How many alumni are management consultants?", "consulting"),
        # Banking
        ("Which alumni are in investment banking?", "investment_banking"),
        ("Show me alumni in IB.", "investment_banking"),
        ("Which alumni work in banking?", "banking"),
        # Finance
        ("Which alumni work in finance?", "finance"),
        ("Show me alumni in asset management.", "finance"),
        # Marketing / operations / government
        ("Which alumni work in marketing?", "marketing"),
        ("Show alumni in operations.", "operations"),
        ("Which alumni work in government or legal roles?", "government_legal"),
        # Healthcare
        ("Which alumni work in healthcare?", "healthcare"),
        ("Show me alumni who are doctors.", "healthcare"),
        ("Which alumni work at hospitals?", "healthcare"),
        # Law
        ("Which alumni are lawyers?", "law"),
        ("Show me alumni in law.", "law"),
        ("Which alumni work at law firms?", "law"),
        # Education
        ("Which alumni work in education?", "education"),
        ("Show me alumni who are teachers.", "education"),
        ("Which alumni are professors?", "education"),
        # Media
        ("Which alumni work in media?", "media"),
        ("Show me alumni in entertainment.", "media"),
        # Startups
        ("Which alumni work at startups?", "startups"),
        ("Which alumni are startup founders?", "startups"),
        # VC / PE
        ("Which alumni work in VC?", "venture_capital"),
        ("Which alumni are in private equity?", "private_equity"),
        # Nonprofit
        ("Which alumni work at nonprofits?", "nonprofit"),
    ],
)
def test_industry_queries_classify_to_people_filter(question, industry):
    spec = classify_people_question(question)
    assert spec is not None, f"Expected a people filter spec for: {question}"
    assert spec["intent"] == "people_filter"
    assert spec["entity"] == "alumni"
    assert spec["filter_type"] == "industry"
    assert spec["industry"] == industry
    assert spec["criteria_label"]
    assert spec["answer_label"] == "Alumni matching criteria"


@pytest.mark.parametrize(
    ("question", "employer"),
    [
        ("Who works at Spotify?", "Spotify"),
        ("Show me alumni at Goldman Sachs.", "Goldman Sachs"),
        ("How many alumni work at Goldman Sachs or Morgan Stanley?", "Goldman Sachs"),
        ("Show me alumni at McKinsey or BCG.", "McKinsey"),
        ("Which alumni work at Spotify or Netflix?", "Spotify"),
        ("Which alumni work at Cornell?", "Cornell"),
    ],
)
def test_single_employer_queries_classify_as_employer_filter(question, employer):
    spec = classify_people_question(question)
    assert spec is not None
    assert spec["filter_type"] == "employer"
    assert spec["industry"] is None
    assert employer in spec["employer_terms"]


@pytest.mark.parametrize(
    ("question", "expected_term"),
    [
        ("Who are software engineers?", "software engineer"),
        ("Which alumni are founders?", "founder"),
        ("Show me product managers.", "product manager"),
    ],
)
def test_occupation_queries_classify_as_occupation_filter(question, expected_term):
    spec = classify_people_question(question)
    assert spec is not None
    assert spec["filter_type"] == "occupation"
    assert spec["industry"] is None
    assert expected_term in spec["occupation_terms"]


def test_plain_consulting_query_is_strict_direct_matches_only():
    spec = classify_people_question("What alumni work in consulting?")
    assert spec["industry"] == "consulting"
    assert spec["industries"] == ["consulting"]
    assert spec["include_adjacent"] is False
    assert spec["include_functions"] == []
    assert spec["required_industries"] == []


def test_consulting_or_strategy_query_requests_internal_strategy_function():
    spec = classify_people_question("What alumni work in consulting or strategy?")
    assert spec["industry"] == "consulting"
    assert "internal_strategy" in spec["include_functions"]


def test_consulting_adjacent_query_sets_include_adjacent():
    spec = classify_people_question("Show consulting-adjacent alumni too")
    assert spec["industry"] == "consulting"
    assert spec["include_adjacent"] is True


def test_finance_consulting_query_requires_the_intersection():
    spec = classify_people_question("Who works in finance consulting?")
    assert spec["industry"] == "consulting"
    assert spec["required_industries"] == ["finance"]


def test_plain_finance_query_has_no_consulting_requirement():
    spec = classify_people_question("What alumni work in finance?")
    assert spec["industry"] == "finance"
    assert spec["required_industries"] == []
    assert spec["excluded_industries"] == []
    assert spec["include_adjacent"] is False


def test_finance_but_not_banking_sets_exclusions():
    spec = classify_people_question("Show me alumni in finance but not banking.")
    assert spec["industry"] == "finance"
    assert spec["excluded_industries"] == ["banking", "investment_banking"]
    assert spec["query_scope"] == "industry_exclusion"


def test_finance_outside_investment_banking_sets_specific_exclusion():
    spec = classify_people_question("Find finance alumni outside investment banking.")
    assert spec["industry"] == "finance"
    assert spec["excluded_industries"] == ["investment_banking"]


def test_technical_roles_query_sets_technical_scope():
    spec = classify_people_question("Which alumni have technical roles?")
    assert spec["industry"] == "tech"
    assert spec["query_scope"] == "technical_role"


def test_tech_company_query_sets_employer_first_scope():
    spec = classify_people_question("Which alumni work at tech companies?")
    assert spec["industry"] == "tech"
    assert spec["query_scope"] == "tech_company"


def test_software_engineers_in_finance_sets_intersection_scope():
    spec = classify_people_question("Which alumni work as software engineers in finance?")
    assert spec["industry"] == "tech"
    assert spec["query_scope"] == "technical_role"
    assert spec["required_industries"] == ["finance"]


def test_marketing_growth_query_requests_growth_function():
    spec = classify_people_question("Show marketing or growth alumni.")
    assert spec["industry"] == "marketing"
    assert "marketing_growth" in spec["include_functions"]


def test_consulting_taxonomy_carries_broad_retrieval_keywords():
    taxonomy = get_taxonomy("consulting")
    for keyword in ["consultant", "advisory", "strategy", "transaction", "valuation", "restructuring"]:
        assert keyword in taxonomy["retrieval_keywords"]
    # Retrieval keywords are for candidate generation only; they are not
    # title keywords that confirm a match on their own.
    assert "strategy" not in taxonomy["title_keywords"]
    assert "operations" not in taxonomy["title_keywords"]


def test_non_people_questions_are_not_classified():
    assert classify_people_question("Summarize revenue by month") is None
    assert classify_people_question("What is the average GPA?") is None
    assert classify_people_question("") is None


def test_industry_for_question_prefers_more_specific_alias():
    assert industry_for_question("Which alumni are in investment banking?") == "investment_banking"
    assert industry_for_question("Which alumni are in VC or private equity?") == "private_equity"
