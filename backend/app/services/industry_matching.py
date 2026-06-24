"""Layered row-to-industry matching engine.

Matching layers, in order:
1. Title keyword            — an explicitly in-industry occupation confirms on its own.
2. Known company            — employer in the taxonomy's known_companies confirms.
3. Employer keyword         — strong industry wording in the employer name confirms.
4. Generic business role    — generic roles (founder, head of growth, ...) never confirm
                              alone; they ride on layers 2/3 and are reported as a source.
5. Optional model classifier— for employers with no deterministic signal; only confirms
                              at or above the taxonomy confidence_threshold.

Rows hit exclusion keywords (e.g. schools/hospitals for tech) only when the title is
not explicitly in-industry. Ambiguous employer wording yields "uncertain", which is
reported separately and never counted in total_matches.
"""

import json
import os
import re

from app.services import ai_service
from app.services.industry_taxonomies import GENERIC_BUSINESS_ROLES, get_taxonomy


MODEL_CLASSIFIER_INSTRUCTIONS = """
You classify whether an employer belongs to a target industry.
Return only valid JSON, no prose:
{"belongs_to_industry": true/false, "confidence": 0.0-1.0,
 "classification": "direct_match" | "adjacent" | "uncertain" | "non_match",
 "count_as_match": true/false, "reason_code": "short_code", "reason": "short rationale"}
""".strip()

_model_cache = {}


def match_row_to_industry(occupation, employer, taxonomy, descriptor_text="", model_classifier=None):
    """Classify one person row against one industry taxonomy.

    Returns {"status": "confirmed" | "uncertain" | "excluded",
             "match_sources": [...], "confidence": float, "internal_reason": str}
    """
    if isinstance(taxonomy, str):
        taxonomy = get_taxonomy(taxonomy)
    if not taxonomy:
        return _result("excluded", [], 0.0, "Unknown industry taxonomy.")

    industry = taxonomy.get("industry", "industry")
    occupation_text = str(occupation or "")
    employer_text = str(employer or "")

    title_term = matched_term(occupation_text, taxonomy.get("title_keywords") or [])
    if title_term:
        return _result(
            "confirmed",
            ["title_keyword"],
            1.0,
            f"Explicit {industry} title: {occupation_text} (matched '{title_term}')",
        )

    employer_status = classify_employer_status(
        employer_text,
        taxonomy,
        occupation=occupation_text,
        descriptor_text=descriptor_text,
        model_classifier=model_classifier,
    )
    sources = list(employer_status.get("sources") or [])
    if employer_status["status"] == "confirmed" and _is_generic_business_role(occupation_text, taxonomy):
        sources.append("generic_business_role_with_matching_employer")
    return _result(
        employer_status["status"],
        sources,
        employer_status["confidence"],
        employer_status["internal_reason"],
    )


def classify_employer_status(employer, taxonomy, occupation="", descriptor_text="", model_classifier=None):
    """Classify an employer against a taxonomy without the title layer.

    Returns {"status": "confirmed" | "uncertain" | "excluded", "source": str,
             "sources": [...], "confidence": float, "internal_reason": str}
    """
    if isinstance(taxonomy, str):
        taxonomy = get_taxonomy(taxonomy)
    industry = taxonomy.get("industry", "industry")
    employer_text = str(employer or "")
    descriptor_text = str(descriptor_text or "")
    combined_context = " ".join(item for item in [employer_text, descriptor_text] if item).strip()

    known_match = known_company_match(employer_text, taxonomy.get("known_companies") or [])
    if known_match:
        return _employer_result(
            "confirmed",
            "known_company",
            0.95,
            f"Employer matches known {industry} company list: {known_match}",
        )

    strong_match = matched_term(employer_text, taxonomy.get("employer_keywords") or [])
    if strong_match:
        return _employer_result(
            "confirmed",
            "strong_keyword",
            0.9,
            f"Employer name contains strong {industry} indicator: {strong_match}",
        )

    descriptor_match = matched_term(descriptor_text, taxonomy.get("employer_keywords") or [])
    if descriptor_match and not is_strong_exclusion_context(occupation, combined_context, taxonomy):
        return _employer_result(
            "confirmed",
            "strong_keyword",
            0.82,
            f"Employer descriptor contains strong {industry} indicator: {descriptor_match}",
        )

    if is_strong_exclusion_context(occupation, combined_context, taxonomy):
        return _employer_result(
            "excluded",
            "exclusion",
            0.9,
            f"Employer or role context strongly indicates a non-{industry} domain.",
        )

    weak_match = matched_term(employer_text, taxonomy.get("ambiguous_keywords") or [])

    if model_classifier and employer_text.strip():
        model_outcome = _apply_model_classifier(model_classifier, employer_text, occupation, taxonomy)
        if model_outcome:
            return model_outcome

    if weak_match:
        return _employer_result(
            "uncertain",
            "none",
            0.45,
            f"Employer has ambiguous wording for {industry}: {weak_match}",
        )

    return _employer_result(
        "excluded",
        "none",
        0.0,
        f"No strong {industry} title or employer signal was found.",
    )


def is_title_match(occupation, taxonomy):
    if isinstance(taxonomy, str):
        taxonomy = get_taxonomy(taxonomy)
    return bool(matched_term(occupation, taxonomy.get("title_keywords") or []))


def is_strong_exclusion_context(occupation, employer_context, taxonomy):
    if isinstance(taxonomy, str):
        taxonomy = get_taxonomy(taxonomy)
    if is_title_match(occupation, taxonomy):
        return False
    exclusions = taxonomy.get("exclusion_keywords") or []
    combined = " ".join([str(occupation or ""), str(employer_context or "")])
    return bool(matched_term(combined, exclusions))


def debug_classify_person(occupation, employer, industry, name="", descriptor_text=""):
    """Development helper: explain why a person matches/misses an industry."""
    taxonomy = get_taxonomy(industry)
    if not taxonomy:
        return {
            "name": name,
            "occupation": occupation,
            "employer": employer,
            "target_industry": industry,
            "status": "error",
            "internal_reason": f"Unknown industry '{industry}'. Known industries: tech, consulting, banking, finance, healthcare, law, education, media, nonprofit, startups, venture_capital, private_equity.",
        }
    match = match_row_to_industry(occupation, employer, taxonomy, descriptor_text=descriptor_text)
    return {
        "name": name,
        "occupation": occupation,
        "employer": employer,
        "target_industry": taxonomy["industry"],
        "status": match["status"],
        "match_sources": match["match_sources"],
        "confidence": match["confidence"],
        "internal_reason": match["internal_reason"],
    }


def default_model_classifier(employer, occupation, taxonomy):
    """Classify an ambiguous employer with the configured LLM, if available.

    Returns the parsed model JSON ({"belongs_to_industry", "confidence",
    "classification", "reason"}) or None when no client is configured or the
    call/parse fails. Results are cached per (employer, industry).
    """
    if ai_service.client is None:
        return None
    industry = taxonomy.get("industry", "industry")
    cache_key = (_normalize(employer), industry)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    payload = {
        "employer": str(employer or ""),
        "occupation": str(occupation or ""),
        "target_industry": industry,
    }
    try:
        response = ai_service.client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            instructions=MODEL_CLASSIFIER_INSTRUCTIONS,
            input=json.dumps(payload),
            max_output_tokens=200,
            temperature=0,
            tools=[],
        )
        text = getattr(response, "output_text", "") or ""
        parsed = json.loads(text.strip())
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        parsed = None
    _model_cache[cache_key] = parsed
    return parsed


def budgeted_model_classifier(budget=20):
    """A per-operation model classifier limited to `budget` calls, or None when
    no model client is configured. Keeps a single request from fanning out into
    hundreds of classification calls on large datasets."""
    if ai_service.client is None:
        return None
    state = {"calls": 0}

    def classify(employer, occupation, taxonomy):
        if state["calls"] >= budget:
            return None
        state["calls"] += 1
        return default_model_classifier(employer, occupation, taxonomy)

    return classify


def _apply_model_classifier(model_classifier, employer, occupation, taxonomy):
    try:
        outcome = model_classifier(employer, occupation, taxonomy)
    except Exception:
        return None
    if not isinstance(outcome, dict):
        return None
    try:
        confidence = float(outcome.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    threshold = float(taxonomy.get("confidence_threshold", 0.8))
    belongs = bool(outcome.get("belongs_to_industry"))
    classification = str(outcome.get("classification") or "").lower()
    if classification == "direct_match":
        belongs = True
    elif classification in {"non_match", "adjacent"}:
        belongs = False
    reason = str(outcome.get("reason") or "Model classification.")
    if belongs and confidence >= threshold:
        return _employer_result("confirmed", "model_classification", confidence, f"Model classified employer as {taxonomy.get('industry')}: {reason}")
    if not belongs and confidence >= threshold:
        return _employer_result("excluded", "model_classification", confidence, f"Model classified employer as non-{taxonomy.get('industry')}: {reason}")
    return _employer_result("uncertain", "model_classification", confidence, f"Model was not confident about employer: {reason}")


def matched_term(text, terms):
    normalized_text = _normalize(text)
    if not normalized_text:
        return ""
    for term in terms:
        term_normalized = _normalize(term)
        if not term_normalized:
            continue
        if " " in term_normalized:
            if term_normalized in normalized_text:
                return term
        elif re.search(rf"\b{re.escape(term_normalized)}\b", normalized_text):
            return term
    return ""


def known_company_match(employer, known_companies):
    employer_norm = _normalize_company(employer)
    if not employer_norm:
        return ""
    for company in known_companies:
        company_norm = _normalize_company(company)
        if not company_norm:
            continue
        if company_norm == employer_norm or re.search(rf"\b{re.escape(company_norm)}\b", employer_norm):
            return company
    return ""


def _is_generic_business_role(occupation, taxonomy):
    roles = list(GENERIC_BUSINESS_ROLES) + list(taxonomy.get("generic_title_keywords") or [])
    return bool(matched_term(occupation, roles))


def _result(status, match_sources, confidence, internal_reason):
    return {
        "status": status,
        "match_sources": list(dict.fromkeys(match_sources)),
        "confidence": float(confidence),
        "internal_reason": str(internal_reason),
    }


def _employer_result(status, source, confidence, internal_reason):
    source_labels = {
        "known_company": "known_company",
        "strong_keyword": "employer_keyword",
        "exclusion": "exclusion_keyword",
        "model_classification": "model_classification",
    }
    sources = []
    if source in source_labels:
        sources.append(source_labels[source])
    elif status == "uncertain":
        sources.append("ambiguous_employer")
    return {
        "status": status,
        "source": source,
        "sources": sources,
        "confidence": float(confidence),
        "internal_reason": str(internal_reason),
    }


def _normalize(value):
    normalized = re.sub(r"[^a-z0-9&]+", " ", str(value or "").lower())
    return " ".join(normalized.split())


def _normalize_company(value):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    suffixes = {"inc", "incorporated", "llc", "l l c", "ltd", "limited", "corp", "corporation", "co", "company"}
    words = [word for word in normalized.split() if word not in suffixes]
    return " ".join(words)
