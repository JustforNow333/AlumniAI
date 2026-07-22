"""Typed exact-predicate gate between intent inference and execution.

Fuzzy concepts remain in ``intent.filters`` and continue through the taxonomy
and people-classifier path.  This module owns only mechanically interpretable
row predicates, with the original question taking precedence over model JSON.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from app.services.column_resolver import (
    CANONICAL_FIELD_ALIASES,
    resolve_all_semantic_columns,
)
from app.services.email_utils import domain_matches, extract_email_tokens, normalize_domain
from app.services.industry_taxonomies import classify_people_question


ALLOWED_PREDICATE_OPERATORS = {
    "exists",
    "missing",
    "equals",
    "not_equals",
    "contains",
    "not_contains",
    "email_domain_in",
    "email_domain_not_in",
}
ALLOWED_PREDICATE_LOGIC = {"and", "or"}
ALLOWED_QUANTIFIERS = {"any", "all", "none"}
PREDICATE_SOURCES = {"deterministic_explicit", "model_inferred", "heuristic_fallback"}
MAX_PREDICATES = 12
MAX_PREDICATE_VALUES = 20

_NEGATIVE_OPERATORS = {"missing", "not_equals", "not_contains", "email_domain_not_in"}
_POSITIVE_OPERATORS = {"exists", "equals", "contains", "email_domain_in"}


def apply_intent_filter(question: str, dataset_context: dict, inferred_intent: dict) -> tuple[dict, dict]:
    """Reconcile explicit source constraints with validated model predicates."""
    intent = copy.deepcopy(inferred_intent) if isinstance(inferred_intent, dict) else {}
    inferred_was_unknown = intent.get("intent") in {None, "", "unknown"}
    deterministic = extract_explicit_predicates(question)
    source_people_spec = classify_people_question(question)
    inferred_people_spec = intent.get("people_filter_spec") if isinstance(intent.get("people_filter_spec"), dict) else None
    attach_source_people_spec = bool(
        source_people_spec
        and (
            source_people_spec.get("capability")
            or inferred_was_unknown
            or not intent.get("filters")
            or intent.get("intent") != "people_filter"
        )
    )
    people_filter_repaired = bool(
        attach_source_people_spec
        and (
            not inferred_people_spec
            or inferred_people_spec.get("filter_type") != source_people_spec.get("filter_type")
            or inferred_people_spec.get("industry") != source_people_spec.get("industry")
            or inferred_people_spec.get("capability") != source_people_spec.get("capability")
        )
    )
    model_root, normalized_model_errors = normalize_row_predicates(
        intent.get("row_predicates"),
        dataset_context,
        question=question,
        default_source="model_inferred",
    )
    model_errors = list(intent.get("row_predicate_validation_errors") or []) + normalized_model_errors

    resolved_explicit = []
    resolution_errors = []
    for raw in deterministic["predicates"]:
        predicate, error = normalize_predicate(
            raw,
            dataset_context,
            question=question,
            default_source="deterministic_explicit",
        )
        if predicate:
            resolved_explicit.append(predicate)
        elif error:
            resolution_errors.append(error)

    retained = list(resolved_explicit)
    rejected = []
    conflicts_removed = 0
    deduplicated = 0
    model_assumptions_repaired = 0
    explicit_semantics = {predicate.get("semantic_column") for predicate in resolved_explicit}
    for candidate in model_root["predicates"]:
        if any(_equivalent_predicates(candidate, existing) for existing in retained):
            deduplicated += 1
            continue
        if resolved_explicit and candidate.get("semantic_column") not in explicit_semantics and not _semantic_is_mentioned(
            question,
            candidate,
        ):
            conflicts_removed += 1
            rejected.append(
                {
                    "operator": candidate.get("operator"),
                    "semantic_column": candidate.get("semantic_column"),
                    "reason": "model_predicate_not_anchored_in_source_question",
                }
            )
            continue
        conflict = next(
            (
                explicit
                for explicit in resolved_explicit
                if _predicates_conflict_or_add_unasked_scope(explicit, candidate)
            ),
            None,
        )
        if conflict:
            conflicts_removed += 1
            rejected.append(
                {
                    "operator": candidate.get("operator"),
                    "semantic_column": candidate.get("semantic_column"),
                    "reason": "conflicts_with_explicit_source_constraint",
                }
            )
            continue
        retained.append(candidate)

    # A model sometimes encodes an exact constraint as a fuzzy concept search.
    # Once the source question supplied a typed predicate for that semantic
    # field, the fuzzy representation is redundant at best and contradictory
    # at worst, so it cannot participate in execution.
    reconciled_filters = []
    for filter_spec in intent.get("filters") or []:
        applies_to = set(filter_spec.get("apply_to_semantic_columns") or []) if isinstance(filter_spec, dict) else set()
        overlap = explicit_semantics & applies_to
        if overlap and not _is_fuzzy_classification_filter(intent, filter_spec):
            conflicts_removed += 1
            rejected.append(
                {
                    "filter_concept": filter_spec.get("concept") if isinstance(filter_spec, dict) else "unknown",
                    "semantic_columns": sorted(overlap),
                    "reason": "exact_source_constraint_replaces_fuzzy_filter",
                }
            )
            continue
        reconciled_filters.append(filter_spec)
    intent["filters"] = reconciled_filters

    has_explicit = bool(resolved_explicit)
    logic = deterministic["logic"] if has_explicit else model_root["logic"]
    normalized_root = {"logic": logic, "predicates": retained}
    valid = not resolution_errors and bool(retained or not deterministic["predicates"])

    if attach_source_people_spec:
        intent["people_filter_spec"] = copy.deepcopy(source_people_spec)
        intent["intent"] = "people_filter"
        intent["target_entity"] = "rows"
        intent["user_goal"] = str(question or "").strip()
        intent["clarification_needed"] = False
        intent["clarifying_question"] = None

    if retained:
        intent["row_predicates"] = normalized_root
        # A high-confidence source predicate makes an otherwise-unknown offline
        # intent answerable without relying on a model paraphrase.
        if has_explicit and inferred_was_unknown:
            intent["intent"] = "people_filter" if source_people_spec else "find_records"
            intent["target_entity"] = "rows"
            intent["user_goal"] = str(question or "").strip()
            intent["desired_output"] = {
                "format": "table",
                "semantic_columns": ["first_name", "last_name", "email", "linkedin_url"],
                "limit": 500,
            }
        if has_explicit:
            intent["clarification_needed"] = False
            intent["clarifying_question"] = None
    else:
        intent.pop("row_predicates", None)

    if deterministic["predicates"] and not resolved_explicit:
        valid = False
        reason = resolution_errors[0] if resolution_errors else "The exact filter could not be resolved."
        intent["clarification_needed"] = True
        intent["clarifying_question"] = reason
    elif model_errors and not retained and not deterministic["predicates"]:
        valid = False
        intent["clarification_needed"] = True
        intent["clarifying_question"] = model_errors[0]

    resolved_columns = list(
        dict.fromkeys(
            column
            for predicate in retained
            for column in predicate.get("resolved_columns") or []
        )
    )
    sources = list(dict.fromkeys(predicate.get("source") for predicate in retained if predicate.get("source")))
    source_clauses, constraint_logic = _source_clause_audit(
        question,
        has_fuzzy=bool(source_people_spec),
        exact_predicate_count=len(resolved_explicit),
    )
    has_fuzzy = bool(source_people_spec or intent.get("people_filter_spec") or intent.get("filters"))
    has_exact = bool(retained)
    if has_fuzzy and has_exact:
        intent["logical_operator"] = constraint_logic
    elif has_exact:
        intent["logical_operator"] = logic
    else:
        intent["logical_operator"] = "and"

    if has_exact:
        original_assumptions = list(intent.get("assumptions") or [])
        assumptions = [
            item
            for item in original_assumptions
            if not _misassigns_alumni_email_to_employer(item)
        ]
        model_assumptions_repaired = len(original_assumptions) - len(assumptions)
        if any(predicate.get("semantic_column") == "email" for predicate in retained):
            assumptions.append(
                "Email predicates apply to the alumnus row's email-like columns, not to employer or recruiter contact information."
            )
        intent["assumptions"] = list(dict.fromkeys(assumptions))

    recognized_constraint_count = (1 if source_people_spec else 0) + len(resolved_explicit)
    represented_constraint_count = recognized_constraint_count
    fuzzy_clause_dropped = bool(source_people_spec and not (intent.get("people_filter_spec") or intent.get("filters")))
    if fuzzy_clause_dropped or represented_constraint_count < recognized_constraint_count:
        valid = False
        intent["clarification_needed"] = True
        intent["clarifying_question"] = "One or more recognized constraints could not be represented safely."
    trace = {
        "intent_filter_applied": bool(retained or deterministic["predicates"] or model_errors or source_people_spec),
        "intent_filter_source": sources[0] if len(sources) == 1 else ("reconciled" if sources else "none"),
        "intent_filter_predicate_count": len(retained),
        "intent_filter_valid": bool(valid and (not model_errors or has_explicit)),
        "intent_filter_repaired_model_output": bool(
            people_filter_repaired
            or (has_explicit and (model_errors or conflicts_removed or deduplicated or model_assumptions_repaired))
        ),
        "intent_filter_conflicts_removed": conflicts_removed,
        "intent_filter_deduplicated": deduplicated,
        "intent_filter_resolved_columns": resolved_columns,
        "intent_filter_operators": [predicate["operator"] for predicate in retained],
        "intent_filter_sources": sources,
        "intent_filter_rejected": rejected,
        "intent_filter_model_assumptions_repaired": model_assumptions_repaired,
        "intent_filter_people_filter_repaired": people_filter_repaired,
        "intent_filter_errors": list(dict.fromkeys(model_errors + resolution_errors)),
        "source_clauses": source_clauses,
        "has_fuzzy_people_filter": has_fuzzy,
        "has_exact_row_predicates": has_exact,
        "is_composite_filter": bool(has_fuzzy and has_exact),
        "constraint_logic": constraint_logic if has_fuzzy and has_exact else intent.get("logical_operator", "and"),
        "recognized_constraint_count": recognized_constraint_count,
        "planned_constraint_count": represented_constraint_count,
        "fuzzy_clause_dropped": fuzzy_clause_dropped,
    }
    return intent, trace


def extract_explicit_predicates(question: str) -> dict:
    """Extract a deliberately small set of high-confidence source constraints."""
    original = str(question or "")
    text = _normalized_sentence(original)
    predicates = []

    email_predicates = _extract_email_predicates(text)
    predicates.extend(email_predicates)

    # Missing/existence constraints for common alumni fields.  Domain-specific
    # email wording above takes priority over a generic "has email" predicate.
    field_terms = {
        "email": r"e-?mails?(?:\s+addresses?)?",
        "linkedin_url": r"linkedin(?:\s+urls?|\s+links?|\s+profiles?)?",
        "employer": r"(?:employer|company|organization|organisation)",
        "occupation": r"(?:title|occupation|job\s+title|role)",
    }
    for semantic, pattern in field_terms.items():
        if semantic == "email" and email_predicates:
            continue
        if re.search(rf"\bmissing\s+(?:an?\s+|their\s+)?{pattern}\b", text) or re.search(
            rf"\b(?:without|lacking)\s+(?:an?\s+|their\s+)?{pattern}\b", text
        ) or re.search(
            rf"\b(?:do(?:es)?\s+not|doesn't|don't|has\s+no|have\s+no)\s+(?:have\s+)?(?:an?\s+|their\s+)?{pattern}\b",
            text,
        ):
            predicates.append(_predicate(semantic, "missing", quantifier="all", require_valid_value=False))
        elif re.search(rf"\b(?:with|has|have|having)\s+(?:an?\s+|their\s+|any\s+)?{pattern}\b", text):
            predicates.append(_predicate(semantic, "exists", quantifier="any", require_valid_value=False))

    predicates.extend(_extract_string_predicates(original))
    return {
        "logic": "or" if len(predicates) > 1 and _explicit_or_relationship(text) else "and",
        "predicates": _dedupe_raw_predicates(predicates),
    }


def normalize_row_predicates(
    value: Any,
    dataset_context: dict,
    *,
    question: str = "",
    default_source: str = "model_inferred",
) -> tuple[dict, list[str]]:
    if value is None:
        return {"logic": "and", "predicates": []}, []
    if not isinstance(value, dict):
        return {"logic": "and", "predicates": []}, ["row_predicates must be an object."]
    logic = str(value.get("logic") or "and").strip().casefold()
    errors = []
    if logic not in ALLOWED_PREDICATE_LOGIC:
        errors.append(f"Unsupported predicate logic '{logic}'.")
        logic = "and"

    raw_predicates = value.get("predicates")
    if not isinstance(raw_predicates, list):
        return {"logic": logic, "predicates": []}, errors + ["row_predicates.predicates must be a list."]

    predicates = []
    for raw in raw_predicates[:MAX_PREDICATES]:
        predicate, error = normalize_predicate(
            raw,
            dataset_context,
            question=question,
            default_source=default_source,
        )
        if predicate:
            if not any(_equivalent_predicates(predicate, existing) for existing in predicates):
                predicates.append(predicate)
        elif error:
            errors.append(error)
    if len(raw_predicates) > MAX_PREDICATES:
        errors.append(f"Only {MAX_PREDICATES} row predicates are allowed.")
    return {"logic": logic, "predicates": predicates}, list(dict.fromkeys(errors))


def normalize_predicate(
    value: Any,
    dataset_context: dict,
    *,
    question: str = "",
    default_source: str = "model_inferred",
) -> tuple[dict | None, str]:
    if not isinstance(value, dict):
        return None, "Each row predicate must be an object."
    operator = str(value.get("operator") or "").strip().casefold()
    if operator not in ALLOWED_PREDICATE_OPERATORS:
        return None, f"Unsupported row predicate operator '{operator or 'missing'}'."
    semantic_column = _clean_key(
        value.get("semantic_column") or value.get("semantic") or value.get("column")
    )
    if not semantic_column:
        return None, "A semantic_column is required for each row predicate."

    source = str(value.get("source") or default_source).strip().casefold()
    if source not in PREDICATE_SOURCES:
        source = default_source
    quantifier = str(value.get("quantifier") or "any").strip().casefold()
    if quantifier not in ALLOWED_QUANTIFIERS:
        return None, f"Unsupported predicate quantifier '{quantifier}'."

    values = value.get("values")
    if values is None and value.get("value") is not None:
        values = [value.get("value")]
    if not isinstance(values, list):
        values = [] if values is None else [values]
    clean_values = []
    for item in values[:MAX_PREDICATE_VALUES]:
        text = str(item or "").strip()
        if not text:
            continue
        if operator in {"email_domain_in", "email_domain_not_in"}:
            text = normalize_domain(text)
        if semantic_column == "grad_year" and operator in {"equals", "not_equals"} and re.fullmatch(
            r"(?:19|20)\d{2}", text
        ):
            clean_values.append(int(text))
        else:
            clean_values.append(text)
    clean_values = list(dict.fromkeys(clean_values))
    if operator not in {"exists", "missing"} and not clean_values:
        return None, f"Predicate operator '{operator}' requires at least one value."

    requested_columns = value.get("resolved_columns") or value.get("columns")
    resolved_columns = _resolve_columns(
        semantic_column,
        dataset_context,
        question=question,
        requested=requested_columns,
    )
    if not resolved_columns:
        label = "email-like" if semantic_column == "email" else semantic_column.replace("_", " ")
        return None, f"I could not find a {label} column in this dataset."

    require_valid_default = operator in {
        "not_equals",
        "not_contains",
        "email_domain_in",
        "email_domain_not_in",
    }
    require_valid = bool(value.get("require_valid_value", require_valid_default))
    if operator in {"email_domain_in", "email_domain_not_in"}:
        require_valid = True
    predicate = {
        "target_entity": "alumni_row",
        "semantic_column": semantic_column,
        "resolved_columns": resolved_columns,
        "operator": operator,
        "values": clean_values,
        "include_subdomains": bool(value.get("include_subdomains", True)),
        "quantifier": quantifier,
        "require_valid_value": require_valid,
        "source": source,
    }
    return predicate, ""


def validate_filter_predicate_params(params: Any, available_columns: Any = None) -> tuple[bool, str]:
    """Validate execution params without accepting an expression language."""
    if not isinstance(params, dict):
        return False, "filter_predicates params must be an object."
    logic = str(params.get("logic") or "and").casefold()
    if logic not in ALLOWED_PREDICATE_LOGIC:
        return False, f"Unsupported predicate logic '{logic}'."
    predicates = params.get("predicates")
    if not isinstance(predicates, list) or not predicates:
        return False, "filter_predicates requires at least one predicate."
    if len(predicates) > MAX_PREDICATES:
        return False, f"Only {MAX_PREDICATES} row predicates are allowed."
    available = {str(column) for column in available_columns} if available_columns is not None else None
    for predicate in predicates:
        if not isinstance(predicate, dict):
            return False, "Each row predicate must be an object."
        operator = str(predicate.get("operator") or "").casefold()
        if operator not in ALLOWED_PREDICATE_OPERATORS:
            return False, f"Unsupported row predicate operator '{operator or 'missing'}'."
        quantifier = str(predicate.get("quantifier") or "any").casefold()
        if quantifier not in ALLOWED_QUANTIFIERS:
            return False, f"Unsupported predicate quantifier '{quantifier}'."
        columns = predicate.get("columns") or predicate.get("resolved_columns")
        if not isinstance(columns, list) or not columns:
            return False, "Each row predicate requires one or more resolved columns."
        if available is not None:
            missing = [str(column) for column in columns if str(column) not in available]
            if missing:
                return False, f"Predicate columns were not found: {', '.join(missing)}."
        values = predicate.get("values") or []
        if operator not in {"exists", "missing"} and (not isinstance(values, list) or not values):
            return False, f"Predicate operator '{operator}' requires at least one value."
    base_filter = params.get("base_filter")
    if base_filter is not None:
        if not isinstance(base_filter, dict) or base_filter.get("type") != "contains_any":
            return False, "The optional base_filter must be an approved contains_any operation."
        if not isinstance(base_filter.get("params"), dict):
            return False, "The optional base_filter params must be an object."
    return True, ""


def evaluate_predicate_row(row: Any, predicate: dict) -> bool:
    columns = predicate.get("columns") or predicate.get("resolved_columns") or []
    values = [_row_value(row, column) for column in columns]
    operator = predicate.get("operator")
    quantifier = predicate.get("quantifier") or "any"
    require_valid = bool(predicate.get("require_valid_value", False))

    if operator in {"email_domain_in", "email_domain_not_in"}:
        tokens = [token for value in values for token in extract_email_tokens(value)]
        if require_valid and not tokens:
            return False
        expected = predicate.get("values") or []
        include_subdomains = bool(predicate.get("include_subdomains", True))
        token_matches = [
            any(domain_matches(token["domain"], domain, include_subdomains=include_subdomains) for domain in expected)
            for token in tokens
        ]
        if operator == "email_domain_not_in":
            token_matches = [not matched for matched in token_matches]
        return _quantify(token_matches, quantifier, require_nonempty=require_valid)

    scalar_results = []
    for value in values:
        missing = _is_missing(value)
        text = "" if missing else str(value).strip()
        if operator == "exists":
            scalar_results.append(not missing)
        elif operator == "missing":
            scalar_results.append(missing)
        elif operator == "equals":
            scalar_results.append(any(text.casefold() == str(item).strip().casefold() for item in predicate.get("values") or []))
        elif operator == "not_equals":
            scalar_results.append(
                (not missing)
                and all(text.casefold() != str(item).strip().casefold() for item in predicate.get("values") or [])
            )
        elif operator == "contains":
            scalar_results.append(any(str(item).casefold() in text.casefold() for item in predicate.get("values") or []))
        elif operator == "not_contains":
            scalar_results.append(
                (not missing)
                and all(str(item).casefold() not in text.casefold() for item in predicate.get("values") or [])
            )
    if require_valid and not any(not _is_missing(value) for value in values):
        return False
    return _quantify(scalar_results, quantifier, require_nonempty=False)


def matching_email_addresses(row: Any, predicate: dict) -> list[str]:
    """Return only email addresses that visibly demonstrate the predicate."""
    operator = predicate.get("operator")
    columns = predicate.get("columns") or predicate.get("resolved_columns") or []
    tokens = [token for column in columns for token in extract_email_tokens(_row_value(row, column))]
    if operator not in {"email_domain_in", "email_domain_not_in"}:
        return [token["address"] for token in tokens]
    expected = predicate.get("values") or []
    include_subdomains = bool(predicate.get("include_subdomains", True))
    addresses = []
    for token in tokens:
        is_in = any(
            domain_matches(token["domain"], domain, include_subdomains=include_subdomains)
            for domain in expected
        )
        qualifies = is_in if operator == "email_domain_in" else not is_in
        if predicate.get("quantifier") == "none":
            qualifies = not is_in if operator == "email_domain_in" else is_in
        if qualifies and token["address"].casefold() not in {item.casefold() for item in addresses}:
            addresses.append(token["address"])
    return addresses


def row_satisfies_predicate_group(row: Any, logic: str, predicates: list[dict]) -> bool:
    outcomes = [evaluate_predicate_row(row, predicate) for predicate in predicates]
    if not outcomes:
        return True
    return any(outcomes) if logic == "or" else all(outcomes)


def _extract_email_predicates(text: str) -> list[dict]:
    cornell_domain = ["cornell.edu"]
    has_both = bool(
        re.search(r"\bboth\b.*\bcornell\b.*\b(?:non[- ]cornell|external|outside)\b", text)
        or re.search(r"\bboth\b.*\b(?:non[- ]cornell|external|outside)\b.*\bcornell\b", text)
    )
    if has_both:
        return [
            _predicate("email", "email_domain_in", values=cornell_domain, quantifier="any", require_valid_value=True),
            _predicate("email", "email_domain_not_in", values=cornell_domain, quantifier="any", require_valid_value=True),
        ]

    no_cornell = bool(
        re.search(r"\b(?:has|have|with)\s+no\s+cornell\s+e-?mail", text)
        or re.search(r"\bno\s+(?:listed\s+)?e-?mails?\s+(?:is|are|from|use)\s+(?:a\s+)?cornell", text)
    )
    if no_cornell:
        return [
            _predicate("email", "email_domain_in", values=cornell_domain, quantifier="none", require_valid_value=True)
        ]

    non_cornell = bool(
        re.search(r"\bnon[- ]cornell\s+e-?mail", text)
        or re.search(r"\be-?mail(?:\s+address)?\b.{0,45}\b(?:is\s+not|isn't|isnt|not)\s+(?:their\s+|a\s+)?cornell\s+e-?mail", text)
        or re.search(r"\bnot\s+@?cornell\.edu\b", text)
        or re.search(r"\bdoes\s+not\s+end\s+(?:in|with)\s+@?cornell\.edu\b", text)
        or re.search(r"\b(?:domain\s+is\s+)?outside\s+(?:of\s+)?(?:the\s+)?(?:cornell(?:\s+domain)?|cornell\.edu)\b", text)
        or re.search(r"\be-?mail(?:s|\s+addresses?)?\s+(?:other\s+than|besides)\s+(?:a\s+)?cornell", text)
        or re.search(r"\b(?:personal|external)\s+(?:or\s+(?:personal|external)\s+)?e-?mail", text)
    )
    if non_cornell:
        quantifier = "all" if re.search(
            r"\b(?:all|every|each)(?:\s+of)?(?:\s+their)?(?:\s+listed)?\s+e-?mails?\b",
            text,
        ) else "any"
        return [
            _predicate(
                "email",
                "email_domain_not_in",
                values=cornell_domain,
                quantifier=quantifier,
                require_valid_value=True,
            )
        ]

    only_domain = re.search(r"\bonly\s+(?:an?\s+)?([a-z0-9.-]+\.[a-z]{2,})\s+(?:e-?mail|domain)", text)
    only_gmail = re.search(r"\bonly\s+gmail(?:\s+e-?mail)?", text)
    if only_domain or only_gmail:
        domain = only_domain.group(1) if only_domain else "gmail.com"
        return [_predicate("email", "email_domain_in", values=[domain], quantifier="all", require_valid_value=True)]

    positive_cornell = bool(
        re.search(r"\be-?mail(?:\s+address|\s+domain)?\s+(?:is|uses|from|at)\s+(?:a\s+)?cornell", text)
        or re.search(r"\b(?:has|have|with)\s+(?:a\s+)?cornell\s+e-?mail", text)
    )
    if positive_cornell:
        return [_predicate("email", "email_domain_in", values=cornell_domain, quantifier="any", require_valid_value=True)]
    return []


def _extract_string_predicates(original: str) -> list[dict]:
    predicates = []
    patterns = [
        ("employer", "not_equals", r"\bemployer\s+(?:is|equals?)\s+not\s+([^,;?.]+)"),
        ("employer", "equals", r"\bemployer\s+(?:is|equals?)\s+(?!not\b)([^,;?.]+)"),
        ("occupation", "not_contains", r"\b(?:title|occupation|job\s+title)\s+does\s+not\s+contain\s+([^,;?.]+)"),
        ("occupation", "contains", r"\b(?:title|occupation|job\s+title)\s+contains?\s+([^,;?.]+)"),
    ]
    for semantic, operator, pattern in patterns:
        match = re.search(pattern, original, re.IGNORECASE)
        if not match:
            continue
        value = re.split(r"\s+and\s+|\s+or\s+", match.group(1).strip(), maxsplit=1, flags=re.IGNORECASE)[0]
        value = value.strip(" '\"")
        if value:
            predicates.append(
                _predicate(
                    semantic,
                    operator,
                    values=[value],
                    quantifier="any",
                    require_valid_value=operator in {"not_equals", "not_contains"},
                )
            )
    grad_year = re.search(
        r"\b(?:graduated|graduating|graduation\s+year|grad\s+year|class\s+of)\D{0,12}(19\d{2}|20\d{2})\b",
        original,
        re.IGNORECASE,
    )
    if grad_year:
        predicates.append(
            _predicate(
                "grad_year",
                "equals",
                values=[grad_year.group(1)],
                quantifier="any",
                require_valid_value=True,
            )
        )
    return predicates


def _source_clause_audit(question: str, *, has_fuzzy: bool, exact_predicate_count: int) -> tuple[list[dict], str]:
    """Record recognized source clauses and their top-level relationship.

    This is deliberately an audit representation, not a general expression
    parser.  High-confidence fuzzy and exact recognizers remain authoritative.
    """
    original = re.sub(r"\s+", " ", str(question or "")).strip()
    parts = re.split(r"\b(and|or)\b", original, flags=re.IGNORECASE)
    clause_parts = [parts[index].strip(" ,?.") for index in range(0, len(parts), 2) if parts[index].strip(" ,?.")]
    connectors = [parts[index].casefold() for index in range(1, len(parts), 2)]

    fuzzy_text = ""
    exact_texts = []
    for part in clause_parts:
        if not fuzzy_text and classify_people_question(part):
            fuzzy_text = part
        if extract_explicit_predicates(part).get("predicates"):
            exact_texts.append(part)
    if has_fuzzy and not fuzzy_text:
        fuzzy_text = original
    if exact_predicate_count and not exact_texts:
        exact_texts = [original]

    clauses = []
    if has_fuzzy:
        clauses.append(
            {
                "text": fuzzy_text or original,
                "type": "fuzzy_people_filter",
                "source": "deterministic_explicit",
                "preserved": True,
            }
        )
    for text in exact_texts[:exact_predicate_count]:
        clauses.append(
            {
                "text": text,
                "type": "exact_row_predicate",
                "source": "deterministic_explicit",
                "preserved": True,
            }
        )

    # When recognized clauses appear in separate source segments, the connector
    # between them is authoritative.  An implicit relationship defaults to AND.
    constraint_logic = "or" if len(clauses) > 1 and "or" in connectors and "and" not in connectors else "and"
    return clauses, constraint_logic


def _misassigns_alumni_email_to_employer(value: Any) -> bool:
    text = str(value or "").casefold()
    return bool(
        "email" in text
        and any(term in text for term in ["employer", "company", "recruiter"])
    )


def _predicate(
    semantic_column: str,
    operator: str,
    *,
    values: list[str] | None = None,
    quantifier: str = "any",
    require_valid_value: bool = False,
) -> dict:
    return {
        "target_entity": "alumni_row",
        "semantic_column": semantic_column,
        "operator": operator,
        "values": list(values or []),
        "include_subdomains": True,
        "quantifier": quantifier,
        "require_valid_value": require_valid_value,
        "source": "deterministic_explicit",
    }


def _resolve_columns(semantic_column, dataset_context, *, question, requested=None):
    actual_names = [
        str(column.get("name"))
        for column in (dataset_context.get("columns") or [])
        if isinstance(column, dict) and column.get("name") is not None
    ]
    if requested is not None:
        requested_items = requested if isinstance(requested, list) else [requested]
        resolved = []
        for item in requested_items:
            match = _match_actual_column(item, actual_names)
            if match and match not in resolved:
                resolved.append(match)
        if resolved:
            return resolved

    resolved = resolve_all_semantic_columns(semantic_column, dataset_context, question=question)
    if resolved:
        return resolved
    aliases = CANONICAL_FIELD_ALIASES.get(semantic_column) or [semantic_column]
    for alias in aliases:
        match = _match_actual_column(alias, actual_names)
        if match:
            return [match]
    # A model can name an exact source header as semantic_column.
    match = _match_actual_column(semantic_column, actual_names)
    return [match] if match else []


def _match_actual_column(requested, actual_names):
    text = str(requested or "").strip()
    for actual in actual_names:
        if text == actual:
            return actual
    for actual in actual_names:
        if text.casefold() == actual.casefold():
            return actual
    normalized = _clean_key(text).replace("_", "")
    for actual in actual_names:
        if _clean_key(actual).replace("_", "") == normalized:
            return actual
    return None


def _predicates_conflict_or_add_unasked_scope(explicit, candidate):
    if explicit.get("semantic_column") != candidate.get("semantic_column"):
        return False
    explicit_values = {str(value).casefold() for value in explicit.get("values") or []}
    candidate_values = {str(value).casefold() for value in candidate.get("values") or []}
    if candidate.get("operator") == "exists" and explicit.get("require_valid_value"):
        return False
    if explicit.get("operator") == candidate.get("operator"):
        # A different exact value on the same field is unasked model scope.
        return bool(explicit_values and candidate_values and explicit_values != candidate_values)
    opposite_pairs = {
        ("exists", "missing"),
        ("missing", "exists"),
        ("equals", "not_equals"),
        ("not_equals", "equals"),
        ("contains", "not_contains"),
        ("not_contains", "contains"),
        ("email_domain_in", "email_domain_not_in"),
        ("email_domain_not_in", "email_domain_in"),
    }
    pair = (explicit.get("operator"), candidate.get("operator"))
    if pair in opposite_pairs and (not explicit_values or not candidate_values or explicit_values & candidate_values):
        return True
    # Model string predicates must not reinterpret an explicit domain boundary.
    if explicit.get("operator", "").startswith("email_domain_") and candidate.get("operator") in {
        "equals",
        "not_equals",
        "contains",
        "not_contains",
    }:
        return True
    return False


def _is_fuzzy_classification_filter(intent, filter_spec):
    if not isinstance(filter_spec, dict):
        return False
    concept = str(filter_spec.get("concept") or "").casefold()
    if concept in {"tech_related", "software_engineer_role", "tech_company"}:
        return True
    people_spec = intent.get("people_filter_spec") if isinstance(intent.get("people_filter_spec"), dict) else {}
    return people_spec.get("filter_type") == "industry"


def _semantic_is_mentioned(question, predicate):
    text = _normalized_sentence(question)
    semantic = predicate.get("semantic_column")
    aliases = list(CANONICAL_FIELD_ALIASES.get(semantic) or []) + [str(semantic or "").replace("_", " ")]
    aliases.extend(predicate.get("resolved_columns") or [])
    return any(
        re.search(rf"\b{re.escape(str(alias).casefold().replace('_', ' '))}\b", text)
        for alias in aliases
        if str(alias or "").strip()
    )


def _equivalent_predicates(left, right):
    return (
        left.get("semantic_column") == right.get("semantic_column")
        and left.get("operator") == right.get("operator")
        and left.get("quantifier", "any") == right.get("quantifier", "any")
        and bool(left.get("include_subdomains", True)) == bool(right.get("include_subdomains", True))
        and {str(value).casefold() for value in left.get("values") or []}
        == {str(value).casefold() for value in right.get("values") or []}
        and set(left.get("resolved_columns") or []) == set(right.get("resolved_columns") or [])
    )


def _dedupe_raw_predicates(predicates):
    deduped = []
    seen = set()
    for predicate in predicates:
        key = (
            predicate.get("semantic_column"),
            predicate.get("operator"),
            tuple(str(value).casefold() for value in predicate.get("values") or []),
            predicate.get("quantifier"),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(predicate)
    return deduped


def _quantify(outcomes, quantifier, *, require_nonempty):
    if require_nonempty and not outcomes:
        return False
    if quantifier == "all":
        return bool(outcomes) and all(outcomes)
    if quantifier == "none":
        return (bool(outcomes) or not require_nonempty) and not any(outcomes)
    return any(outcomes)


def _row_value(row, column):
    try:
        return row[column]
    except (KeyError, TypeError, IndexError):
        return None


def _is_missing(value):
    if value is None:
        return True
    try:
        unequal_to_self = value != value
        if isinstance(unequal_to_self, bool) and unequal_to_self:
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold() in {"", "nan", "none", "null", "nat"}


def _clean_key(value):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold())).strip("_")


def _normalized_sentence(value):
    text = str(value or "").casefold().replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def _explicit_or_relationship(text):
    return bool(re.search(r"\b(?:either\b.*\bor\b|\bor\b)", text)) and not bool(
        re.search(r"\b(?:personal|external)\s+or\s+(?:personal|external)\s+e-?mail", text)
    )
