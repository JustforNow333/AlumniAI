import pytest

from app.services.industry_matching import (
    debug_classify_person,
    match_row_to_industry,
)
from app.services.industry_taxonomies import get_taxonomy


def status_for(occupation, employer, industry):
    return match_row_to_industry(occupation, employer, get_taxonomy(industry))["status"]


@pytest.mark.parametrize(
    ("occupation", "employer", "expected_status"),
    [
        ("Head of Growth", "Spotify", "confirmed"),
        ("Software Engineer", "Local Bakery", "confirmed"),
        ("Founder", "Local Bakery", "excluded"),
        ("Founder", "Rune Technologies", "confirmed"),
        ("Founder", "FanAmp", "confirmed"),
        ("Founder", "Cogni DAO", "confirmed"),
        ("Mathematics Department Chair", "Latin School of Chicago", "excluded"),
        ("Director of Hematologic Oncology", "Holy Name Medical Center", "excluded"),
        ("Data Scientist", "Hospital for Special Surgery", "confirmed"),
        ("IT Director", "High School", "confirmed"),
        ("Founder", "Bright Ventures", "uncertain"),
    ],
)
def test_tech_examples(occupation, employer, expected_status):
    assert status_for(occupation, employer, "tech") == expected_status


def test_tech_match_sources_for_generic_role_at_known_company():
    match = match_row_to_industry("Head of Growth", "Spotify", get_taxonomy("tech"))
    assert match["status"] == "confirmed"
    assert "known_company" in match["match_sources"]
    assert "generic_business_role_with_matching_employer" in match["match_sources"]
    assert match["confidence"] >= 0.9
    assert "Spotify" in match["internal_reason"]


@pytest.mark.parametrize(
    ("occupation", "employer", "expected_status"),
    [
        ("Associate Consultant", "Bain", "confirmed"),
        ("Partner", "McKinsey", "confirmed"),
        ("Analyst", "Rune Technologies", "excluded"),
        ("Strategy Consultant", "Unknown Family Business", "confirmed"),
    ],
)
def test_consulting_examples(occupation, employer, expected_status):
    assert status_for(occupation, employer, "consulting") == expected_status


@pytest.mark.parametrize(
    ("occupation", "employer", "expected_status"),
    [
        ("Investment Banking Analyst", "Goldman Sachs", "confirmed"),
        ("Analyst", "Goldman Sachs", "confirmed"),
        ("Analyst", "Community Food Pantry", "excluded"),
        ("M&A Associate", "Evercore", "confirmed"),
    ],
)
def test_banking_examples(occupation, employer, expected_status):
    assert status_for(occupation, employer, "banking") == expected_status


@pytest.mark.parametrize(
    ("occupation", "employer", "expected_status"),
    [
        ("Portfolio Manager", "BlackRock", "confirmed"),
        ("Trader", "Jane Street", "confirmed"),
        ("Financial Analyst", "Random Manufacturing", "confirmed"),
        ("Software Engineer", "BlackRock", "confirmed"),
    ],
)
def test_finance_examples(occupation, employer, expected_status):
    assert status_for(occupation, employer, "finance") == expected_status


@pytest.mark.parametrize(
    ("occupation", "employer", "expected_status"),
    [
        ("Doctor", "Mayo Clinic", "confirmed"),
        ("Data Scientist", "Hospital for Special Surgery", "confirmed"),
        ("Director of Hematologic Oncology", "Holy Name Medical Center", "confirmed"),
    ],
)
def test_healthcare_examples(occupation, employer, expected_status):
    assert status_for(occupation, employer, "healthcare") == expected_status


@pytest.mark.parametrize(
    ("occupation", "employer", "expected_status"),
    [
        ("Associate Attorney", "Skadden", "confirmed"),
        ("Partner", "Skadden", "confirmed"),
        ("Partner", "McKinsey", "excluded"),
    ],
)
def test_law_examples(occupation, employer, expected_status):
    assert status_for(occupation, employer, "law") == expected_status


@pytest.mark.parametrize(
    ("occupation", "employer", "expected_status"),
    [
        ("Mathematics Department Chair", "Latin School of Chicago", "confirmed"),
        ("Software Engineer", "Cornell University", "confirmed"),
    ],
)
def test_education_examples(occupation, employer, expected_status):
    assert status_for(occupation, employer, "education") == expected_status


@pytest.mark.parametrize(
    ("occupation", "employer", "expected_status"),
    [
        ("Head of Growth", "Spotify", "confirmed"),
        ("Producer", "Netflix", "confirmed"),
        ("Software Engineer", "Netflix", "confirmed"),
    ],
)
def test_media_examples(occupation, employer, expected_status):
    assert status_for(occupation, employer, "media") == expected_status


@pytest.mark.parametrize(
    ("occupation", "employer", "expected_status"),
    [
        ("Founder", "FanAmp", "confirmed"),
        ("Founder", "Local Bakery", "excluded"),
        ("Founding Engineer", "Acme AI", "confirmed"),
    ],
)
def test_startup_examples(occupation, employer, expected_status):
    assert status_for(occupation, employer, "startups") == expected_status


def test_same_row_can_belong_to_different_industries_depending_on_query():
    occupation, employer = "Data Scientist", "Hospital for Special Surgery"
    assert status_for(occupation, employer, "tech") == "confirmed"
    assert status_for(occupation, employer, "healthcare") == "confirmed"


def test_match_result_shape():
    match = match_row_to_industry("Software Engineer", "Local Bakery", get_taxonomy("tech"))
    assert set(match) == {"status", "match_sources", "confidence", "internal_reason"}
    assert match["match_sources"] == ["title_keyword"]
    assert match["confidence"] == 1.0


def test_uncertain_rows_carry_ambiguous_source():
    match = match_row_to_industry("Founder", "Bright Ventures", get_taxonomy("tech"))
    assert match["status"] == "uncertain"
    assert match["confidence"] < 0.8


def test_model_classifier_confirms_only_at_or_above_threshold():
    taxonomy = get_taxonomy("tech")

    def confident_model(employer, occupation, tax):
        return {"belongs_to_industry": True, "confidence": 0.92, "classification": "confirmed", "reason": "tech startup"}

    def unsure_model(employer, occupation, tax):
        return {"belongs_to_industry": True, "confidence": 0.4, "classification": "uncertain", "reason": "unclear"}

    confirmed = match_row_to_industry("Founder", "Bright Ventures", taxonomy, model_classifier=confident_model)
    assert confirmed["status"] == "confirmed"
    assert "model_classification" in confirmed["match_sources"]

    uncertain = match_row_to_industry("Founder", "Bright Ventures", taxonomy, model_classifier=unsure_model)
    assert uncertain["status"] == "uncertain"


def test_debug_classify_person_explains_inclusion():
    debug = debug_classify_person("Head of Growth", "Spotify", "tech", name="Neil Wusu")
    assert debug["name"] == "Neil Wusu"
    assert debug["target_industry"] == "tech"
    assert debug["status"] == "confirmed"
    assert "known_company" in debug["match_sources"]
    assert debug["internal_reason"]

    unknown = debug_classify_person("Founder", "Somewhere", "not-an-industry")
    assert unknown["status"] == "error"
