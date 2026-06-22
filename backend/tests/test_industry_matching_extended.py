import pytest

from app.services import ai_service
from app.services.industry_matching import (
    _apply_model_classifier,
    _normalize,
    _normalize_company,
    budgeted_model_classifier,
    classify_employer_status,
    default_model_classifier,
    is_strong_exclusion_context,
    is_title_match,
    known_company_match,
    match_row_to_industry,
    matched_term,
)
from app.services.industry_taxonomies import get_taxonomy


@pytest.fixture(autouse=True)
def disable_ai(monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)


# --- default_model_classifier ---

def test_default_model_classifier_returns_none_without_client():
    taxonomy = get_taxonomy("tech")
    result = default_model_classifier("Google", "Engineer", taxonomy)
    assert result is None


# --- budgeted_model_classifier ---

def test_budgeted_model_classifier_returns_none_without_client():
    result = budgeted_model_classifier(budget=5)
    assert result is None


def test_budgeted_model_classifier_with_mock_client(monkeypatch):
    class FakeClient:
        pass

    monkeypatch.setattr(ai_service, "client", FakeClient())
    classifier = budgeted_model_classifier(budget=2)
    assert callable(classifier)


def test_budgeted_model_classifier_respects_budget(monkeypatch):
    class FakeClient:
        pass

    monkeypatch.setattr(ai_service, "client", FakeClient())
    classifier = budgeted_model_classifier(budget=2)

    call_count = 0

    def mock_default(employer, occupation, taxonomy):
        nonlocal call_count
        call_count += 1
        return None

    monkeypatch.setattr(
        "app.services.industry_matching.default_model_classifier", mock_default
    )

    taxonomy = get_taxonomy("tech")
    classifier("A", "Eng", taxonomy)
    classifier("B", "Eng", taxonomy)
    result = classifier("C", "Eng", taxonomy)
    assert result is None
    assert call_count == 2


# --- _apply_model_classifier ---

def test_apply_model_classifier_confirmed_above_threshold():
    taxonomy = get_taxonomy("tech")

    def model(employer, occ, tax):
        return {
            "belongs_to_industry": True,
            "confidence": 0.95,
            "classification": "confirmed",
            "reason": "tech company",
        }

    result = _apply_model_classifier(model, "Acme AI", "Engineer", taxonomy)
    assert result["status"] == "confirmed"
    assert "model_classification" in result["sources"]


def test_apply_model_classifier_excluded_above_threshold():
    taxonomy = get_taxonomy("tech")

    def model(employer, occ, tax):
        return {
            "belongs_to_industry": False,
            "confidence": 0.9,
            "classification": "non_match",
            "reason": "bakery",
        }

    result = _apply_model_classifier(model, "Local Bakery", "Baker", taxonomy)
    assert result["status"] == "excluded"


def test_apply_model_classifier_uncertain_below_threshold():
    taxonomy = get_taxonomy("tech")

    def model(employer, occ, tax):
        return {
            "belongs_to_industry": True,
            "confidence": 0.3,
            "classification": "uncertain",
            "reason": "unclear",
        }

    result = _apply_model_classifier(model, "Unknown Inc", "Manager", taxonomy)
    assert result["status"] == "uncertain"


def test_apply_model_classifier_exception_returns_none():
    taxonomy = get_taxonomy("tech")

    def model(employer, occ, tax):
        raise RuntimeError("model error")

    result = _apply_model_classifier(model, "X", "Y", taxonomy)
    assert result is None


def test_apply_model_classifier_non_dict_returns_none():
    taxonomy = get_taxonomy("tech")

    def model(employer, occ, tax):
        return "invalid"

    result = _apply_model_classifier(model, "X", "Y", taxonomy)
    assert result is None


def test_apply_model_classifier_bad_confidence():
    taxonomy = get_taxonomy("tech")

    def model(employer, occ, tax):
        return {
            "belongs_to_industry": True,
            "confidence": "not-a-number",
            "reason": "test",
        }

    result = _apply_model_classifier(model, "X", "Y", taxonomy)
    assert result is not None
    assert result["confidence"] == 0.0


# --- classify_employer_status ---

def test_classify_employer_status_string_taxonomy():
    result = classify_employer_status("Google", "tech", occupation="Engineer")
    assert result["status"] == "confirmed"


def test_classify_employer_status_descriptor_match():
    taxonomy = get_taxonomy("tech")
    result = classify_employer_status(
        "Unknown Corp", taxonomy, occupation="Manager", descriptor_text="technology solutions"
    )
    assert result["status"] == "confirmed"
    assert result["source"] == "strong_keyword"


def test_classify_employer_status_descriptor_with_exclusion():
    taxonomy = get_taxonomy("tech")
    result = classify_employer_status(
        "City Hospital Tech Wing",
        taxonomy,
        occupation="Hospital Administrator",
        descriptor_text="hospital medical center",
    )
    assert result["status"] in {"excluded", "uncertain"}


def test_classify_employer_status_ambiguous():
    taxonomy = get_taxonomy("tech")
    result = classify_employer_status("Bright Ventures", taxonomy, occupation="Founder")
    assert result["status"] in {"uncertain", "excluded"}


# --- is_title_match ---

def test_is_title_match_positive():
    assert is_title_match("Software Engineer", "tech")


def test_is_title_match_negative():
    assert not is_title_match("Baker", "tech")


def test_is_title_match_string_taxonomy():
    assert is_title_match("Software Engineer", "tech")


# --- is_strong_exclusion_context ---

def test_exclusion_context_with_hospital():
    assert is_strong_exclusion_context("Manager", "Hospital for Special Surgery", "tech")


def test_exclusion_context_tech_title_overrides():
    assert not is_strong_exclusion_context("Software Engineer", "Hospital for Special Surgery", "tech")


def test_exclusion_context_string_taxonomy():
    assert is_strong_exclusion_context("Manager", "School of Arts", "tech")


# --- known_company_match ---

def test_known_company_exact():
    result = known_company_match("Google", ["Google", "Apple", "Microsoft"])
    assert result == "Google"


def test_known_company_case_insensitive():
    result = known_company_match("google", ["Google"])
    assert result == "Google"


def test_known_company_with_suffix():
    result = known_company_match("Google Inc", ["Google"])
    assert result == "Google"


def test_known_company_no_match():
    result = known_company_match("Local Bakery", ["Google", "Apple"])
    assert result == ""


def test_known_company_empty_employer():
    assert known_company_match("", ["Google"]) == ""


def test_known_company_empty_list():
    assert known_company_match("Google", []) == ""


def test_known_company_empty_entry_in_list():
    assert known_company_match("Google", ["", "Google"]) == "Google"


# --- matched_term ---

def test_matched_term_single_word():
    assert matched_term("Software Engineer at Google", ["engineer"])


def test_matched_term_multi_word():
    assert matched_term("machine learning engineer", ["machine learning"])


def test_matched_term_no_match():
    assert matched_term("Baker", ["engineer"]) == ""


def test_matched_term_empty_text():
    assert matched_term("", ["engineer"]) == ""


def test_matched_term_empty_term():
    assert matched_term("engineer", [""]) == ""


# --- _normalize / _normalize_company ---

def test_normalize():
    assert _normalize("  Hello   WORLD  ") == "hello world"
    assert _normalize(None) == ""


def test_normalize_company():
    assert _normalize_company("Google Inc") == "google"
    assert _normalize_company("Apple LLC") == "apple"
    assert _normalize_company("") == ""
    assert _normalize_company(None) == ""


# --- match_row_to_industry with string taxonomy ---

def test_match_row_string_taxonomy():
    result = match_row_to_industry("Software Engineer", "Google", "tech")
    assert result["status"] == "confirmed"


def test_match_row_unknown_taxonomy():
    result = match_row_to_industry("Engineer", "Google", "unknown_industry")
    assert result["status"] == "excluded"
    assert "Unknown" in result["internal_reason"]


def test_match_row_empty_occupation_and_employer():
    result = match_row_to_industry("", "", get_taxonomy("tech"))
    assert result["status"] == "excluded"


def test_match_row_none_values():
    result = match_row_to_industry(None, None, get_taxonomy("tech"))
    assert result["status"] == "excluded"
