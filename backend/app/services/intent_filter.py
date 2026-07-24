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
from app.services.predicate_values import (
    normalize_membership_value,
    normalize_string_value,
    parse_date_value,
    parse_numeric_value,
    relative_date_boundary,
)


ALLOWED_PREDICATE_OPERATORS = {
    "exists",
    "missing",
    "equals",
    "not_equals",
    "contains",
    "not_contains",
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "between",
    "in",
    "not_in",
    "date_before",
    "date_after",
    "date_between",
    "starts_with",
    "ends_with",
    "email_domain_in",
    "email_domain_not_in",
}
ALLOWED_PREDICATE_LOGIC = {"and", "or"}
ALLOWED_QUANTIFIERS = {"any", "all", "none"}
PREDICATE_SOURCES = {"deterministic_explicit", "model_inferred", "heuristic_fallback"}
MAX_PREDICATES = 12
MAX_PREDICATE_VALUES = 20
MAX_PREDICATE_GROUP_DEPTH = 3
MAX_CLAUSES_PER_GROUP = 12

NO_VALUE_OPERATORS = {"exists", "missing"}
EXACTLY_ONE_VALUE_OPERATORS = {
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "date_before",
    "date_after",
    "starts_with",
    "ends_with",
}
EXACTLY_TWO_VALUE_OPERATORS = {"between", "date_between"}
ONE_OR_MORE_VALUE_OPERATORS = {
    "equals",
    "not_equals",
    "contains",
    "not_contains",
    "in",
    "not_in",
    "email_domain_in",
    "email_domain_not_in",
}
NUMERIC_OPERATORS = {
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "between",
}
DATE_OPERATORS = {"date_before", "date_after", "date_between"}
MEMBERSHIP_OPERATORS = {"in", "not_in"}
STRING_BOUNDARY_OPERATORS = {"starts_with", "ends_with"}

_NEGATIVE_OPERATORS = {
    "missing",
    "not_equals",
    "not_contains",
    "not_in",
    "email_domain_not_in",
}
_POSITIVE_OPERATORS = {
    "exists",
    "equals",
    "contains",
    "in",
    "email_domain_in",
}


def apply_intent_filter(
    question: str,
    dataset_context: dict,
    inferred_intent: dict,
    *,
    now: Any = None,
) -> tuple[dict, dict]:
    """Reconcile explicit source constraints with validated model predicates."""
    intent = copy.deepcopy(inferred_intent) if isinstance(inferred_intent, dict) else {}
    inferred_was_unknown = intent.get("intent") in {None, "", "unknown"}
    deterministic = extract_explicit_predicates(question, now=now)
    source_people_spec = classify_people_question(question)
    if _people_spec_is_redundant_with_exact(source_people_spec, deterministic, question):
        source_people_spec = None
        inferred_spec = intent.get("people_filter_spec")
        if isinstance(inferred_spec, dict) and inferred_spec.get("filter_type") in {"employer", "occupation"}:
            intent.pop("people_filter_spec", None)
            intent["filters"] = []
            if intent.get("intent") == "people_filter":
                intent["intent"] = "find_records"
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

    explicit_root, resolution_errors = normalize_row_predicates(
        deterministic,
        dataset_context,
        question=question,
        default_source="deterministic_explicit",
    )
    resolved_explicit = predicate_group_leaves(explicit_root)

    retained = list(resolved_explicit)
    rejected = []
    conflicts_removed = 0
    deduplicated = 0
    model_assumptions_repaired = 0
    explicit_semantics = {predicate.get("semantic_column") for predicate in resolved_explicit}
    accepted_model = []
    for candidate in predicate_group_leaves(model_root):
        if any(_equivalent_predicates(candidate, existing) for existing in retained):
            deduplicated += 1
            continue
        if candidate.get("semantic_column") not in explicit_semantics and not _predicate_is_explicit_in_question(
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
        accepted_model.append(candidate)

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
    if has_explicit:
        normalized_root = _merge_explicit_and_model_groups(explicit_root, accepted_model)
    else:
        normalized_root = model_root
    logic = normalized_root.get("logic") or "and"
    valid = not resolution_errors and bool(retained or not predicate_group_leaves(deterministic))

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

    if predicate_group_leaves(deterministic) and not resolved_explicit:
        valid = False
        reason = resolution_errors[0] if resolution_errors else "The exact filter could not be resolved."
        intent["clarification_needed"] = True
        intent["clarifying_question"] = reason
    elif model_errors and not retained and not predicate_group_leaves(deterministic):
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
    relative_dates_resolved = {
        str(predicate.get("relative_date_label")): str(predicate.get("values", [""])[0])
        for predicate in retained
        if predicate.get("relative_date_label") and predicate.get("values")
    }
    trace = {
        "intent_filter_applied": bool(
            retained or predicate_group_leaves(deterministic) or model_errors or source_people_spec
        ),
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
        "normalized_predicate_count": len(retained),
        "predicate_group_depth": predicate_group_depth(normalized_root),
        "relative_dates_resolved": relative_dates_resolved,
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


def extract_explicit_predicates(question: str, *, now: Any = None) -> dict:
    """Extract conservative source-anchored constraints and bounded grouping."""
    original = str(question or "")
    grouped = _extract_explicit_group(original, now=now)
    if grouped is not None:
        return grouped
    predicates = _extract_flat_explicit_predicates(original, now=now)
    logic = "or" if len(predicates) > 1 and _explicit_or_relationship(_normalized_sentence(original)) else "and"
    return _predicate_root(logic, [_predicate_clause(predicate) for predicate in predicates])


def _extract_flat_explicit_predicates(question: str, *, now: Any = None) -> list[dict]:
    original = str(question or "")
    text = _normalized_sentence(original)
    predicates = []

    email_predicates = _extract_email_predicates(text)
    predicates.extend(email_predicates)

    # Domain-specific email wording takes priority over generic existence.
    field_terms = {
        "email": r"e-?mails?(?:\s+addresses?)?",
        "linkedin_url": r"linkedin(?:\s+urls?|\s+links?|\s+profiles?)?",
        "employer": r"(?:employer|company|organization|organisation)",
        "occupation": r"(?:title|occupation|job\s+title|role)",
        "relationship_manager": (
            r"(?:(?:assigned|constituent|prospect|portfolio)\s+)?"
            r"(?:relationship|prospect|portfolio)\s+manager"
        ),
        "last_contact_date": r"(?:last\s+contact(?:\s+date)?|contact\s+date)",
    }
    for semantic, pattern in field_terms.items():
        if semantic == "email" and email_predicates:
            continue
        if re.search(rf"\bmissing\s+(?:an?\s+|their\s+)?{pattern}\b", text) or re.search(
            rf"\b(?:without|lacking)\s+(?:an?\s+|their\s+)?{pattern}\b", text
        ) or re.search(
            rf"\b(?:do(?:es)?\s+not|doesn't|don't|has\s+no|have\s+no)\s+"
            rf"(?:have\s+)?(?:an?\s+|their\s+|assigned\s+)?{pattern}\b",
            text,
        ):
            predicates.append(_predicate(semantic, "missing", quantifier="all", require_valid_value=False))
        elif re.search(rf"\b(?:with|has|have|having)\s+(?:an?\s+|their\s+|any\s+)?{pattern}\b", text):
            predicates.append(_predicate(semantic, "exists", quantifier="any", require_valid_value=False))

    predicates.extend(_extract_string_predicates(original))
    predicates.extend(_extract_numeric_predicates(original))
    predicates.extend(_extract_membership_predicates(original))
    predicates.extend(_extract_date_predicates(original, now=now))
    return _dedupe_raw_predicates(predicates)


def _extract_explicit_group(question: str, *, now: Any = None) -> dict | None:
    """Recognize ``A and either B or C`` without flattening the inner OR."""
    match = re.search(r"\beither\b(.+?)\bor\b(.+)$", question, re.IGNORECASE)
    if not match:
        return None
    prefix = question[: match.start()].strip(" ,")
    left = match.group(1).strip(" ,")
    right = match.group(2).strip(" ,?.")
    prefix_predicates = _extract_flat_explicit_predicates(prefix, now=now)
    left_predicates = _extract_flat_explicit_predicates(left, now=now)
    right_predicates = _extract_flat_explicit_predicates(right, now=now)
    if not prefix_predicates or not left_predicates or not right_predicates:
        return None
    inner = _predicate_root(
        "or",
        [
            *[_predicate_clause(predicate) for predicate in left_predicates],
            *[_predicate_clause(predicate) for predicate in right_predicates],
        ],
    )
    return _predicate_root(
        "and",
        [
            *[_predicate_clause(predicate) for predicate in prefix_predicates],
            {"type": "group", "logic": "or", "clauses": inner["clauses"]},
        ],
    )


def normalize_row_predicates(
    value: Any,
    dataset_context: dict,
    *,
    question: str = "",
    default_source: str = "model_inferred",
) -> tuple[dict, list[str]]:
    if value is None:
        return _predicate_root("and", []), []
    if not isinstance(value, dict):
        return _predicate_root("and", []), ["row_predicates must be an object."]

    state = {"leaf_count": 0}
    root, errors = _normalize_predicate_group(
        value,
        dataset_context,
        question=question,
        default_source=default_source,
        depth=1,
        state=state,
    )
    return root, list(dict.fromkeys(errors))


def _normalize_predicate_group(
    value: dict,
    dataset_context: dict,
    *,
    question: str,
    default_source: str,
    depth: int,
    state: dict,
) -> tuple[dict, list[str]]:
    errors = []
    logic = str(value.get("logic") or "and").strip().casefold()
    if logic not in ALLOWED_PREDICATE_LOGIC:
        errors.append(f"Unsupported predicate logic '{logic}'.")
        logic = "and"
    if depth > MAX_PREDICATE_GROUP_DEPTH:
        return _predicate_root(logic, []), [
            f"Predicate groups may be nested at most {MAX_PREDICATE_GROUP_DEPTH} levels."
        ]

    raw_clauses = value.get("clauses")
    if raw_clauses is None:
        raw_predicates = value.get("predicates")
        if not isinstance(raw_predicates, list):
            return _predicate_root(logic, []), errors + [
                "row_predicates requires a predicates or clauses list."
            ]
        raw_clauses = [_predicate_clause(raw) for raw in raw_predicates]
    if not isinstance(raw_clauses, list):
        return _predicate_root(logic, []), errors + ["Predicate group clauses must be a list."]
    if len(raw_clauses) > MAX_CLAUSES_PER_GROUP:
        errors.append(f"Only {MAX_CLAUSES_PER_GROUP} clauses are allowed per predicate group.")

    clauses = []
    for raw_clause in raw_clauses[:MAX_CLAUSES_PER_GROUP]:
        if not isinstance(raw_clause, dict):
            errors.append("Each predicate clause must be an object.")
            continue
        clause_type = str(raw_clause.get("type") or "").strip().casefold()
        if clause_type == "group":
            child, child_errors = _normalize_predicate_group(
                raw_clause,
                dataset_context,
                question=question,
                default_source=default_source,
                depth=depth + 1,
                state=state,
            )
            errors.extend(child_errors)
            if child["clauses"]:
                clauses.append(
                    {
                        "type": "group",
                        "logic": child["logic"],
                        "clauses": child["clauses"],
                    }
                )
            continue
        if clause_type != "predicate":
            errors.append(f"Unsupported predicate clause type '{clause_type or 'missing'}'.")
            continue
        if state["leaf_count"] >= MAX_PREDICATES:
            errors.append(f"Only {MAX_PREDICATES} row predicates are allowed.")
            continue
        predicate, error = normalize_predicate(
            raw_clause.get("predicate"),
            dataset_context,
            question=question,
            default_source=default_source,
        )
        if predicate:
            state["leaf_count"] += 1
            if not any(_equivalent_predicates(predicate, existing) for existing in predicate_group_leaves(
                _predicate_root(logic, clauses)
            )):
                clauses.append(_predicate_clause(predicate))
        elif error:
            errors.append(error)
    return _predicate_root(logic, clauses), errors


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
    clean_values, value_type, value_error = normalize_operator_values(
        operator,
        values[:MAX_PREDICATE_VALUES],
        semantic_column=semantic_column,
    )
    if value_error:
        return None, value_error
    if len(values) > MAX_PREDICATE_VALUES:
        return None, f"Only {MAX_PREDICATE_VALUES} values are allowed per predicate."

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
        *NUMERIC_OPERATORS,
        *DATE_OPERATORS,
        *MEMBERSHIP_OPERATORS,
        *STRING_BOUNDARY_OPERATORS,
        "email_domain_in",
        "email_domain_not_in",
    }
    require_valid = bool(value.get("require_valid_value", require_valid_default))
    if operator in (
        NUMERIC_OPERATORS
        | DATE_OPERATORS
        | MEMBERSHIP_OPERATORS
        | STRING_BOUNDARY_OPERATORS
        | {"email_domain_in", "email_domain_not_in"}
    ):
        require_valid = True
    predicate = {
        "target_entity": "alumni_row",
        "semantic_column": semantic_column,
        "resolved_columns": resolved_columns,
        "operator": operator,
        "values": clean_values,
        "value_type": value_type,
        "include_subdomains": bool(value.get("include_subdomains", True)),
        "quantifier": quantifier,
        "require_valid_value": require_valid,
        "source": source,
    }
    if operator in {"between", "date_between"}:
        predicate["lower_inclusive"] = bool(value.get("lower_inclusive", True))
        predicate["upper_inclusive"] = bool(value.get("upper_inclusive", True))
    if operator in DATE_OPERATORS and value.get("relative_date_label"):
        predicate["relative_date_label"] = str(value.get("relative_date_label"))[:80]
    return predicate, ""


def normalize_operator_values(
    operator: str,
    values: list[Any],
    *,
    semantic_column: str = "",
) -> tuple[list[Any], str, str]:
    """Validate operator arity and return stable typed values."""
    raw_values = list(values or [])
    if operator in NO_VALUE_OPERATORS:
        nonblank = [value for value in raw_values if normalize_string_value(value) is not None]
        if nonblank:
            return [], "text", f"Predicate operator '{operator}' does not accept values."
        return [], "text", ""

    if operator in EXACTLY_ONE_VALUE_OPERATORS and len(raw_values) != 1:
        return [], "text", f"Predicate operator '{operator}' requires exactly one value."
    if operator in EXACTLY_TWO_VALUE_OPERATORS and len(raw_values) != 2:
        return [], "text", f"Predicate operator '{operator}' requires exactly two values."
    if operator in ONE_OR_MORE_VALUE_OPERATORS and not raw_values:
        return [], "text", f"Predicate operator '{operator}' requires at least one value."

    if operator in NUMERIC_OPERATORS:
        parsed = [parse_numeric_value(value) for value in raw_values]
        if any(value is None for value in parsed):
            return [], "number", f"Predicate operator '{operator}' requires parseable numeric values."
        if operator == "between" and parsed[0] > parsed[1]:
            return [], "number", "The lower numeric bound must not exceed the upper bound."
        return parsed, "number", ""

    if operator in DATE_OPERATORS:
        parsed_dates = [parse_date_value(value) for value in raw_values]
        if any(value is None for value in parsed_dates):
            return [], "date", f"Predicate operator '{operator}' requires parseable date values."
        if operator == "date_between" and parsed_dates[0] > parsed_dates[1]:
            return [], "date", "The lower date bound must not exceed the upper bound."
        return [value.isoformat() for value in parsed_dates], "date", ""

    clean_values = []
    for item in raw_values:
        text = normalize_string_value(item)
        if text is None:
            continue
        if operator in {"email_domain_in", "email_domain_not_in"}:
            text = normalize_domain(text)
            if not text:
                continue
        if (
            semantic_column == "grad_year"
            and operator in {"equals", "not_equals", "in", "not_in"}
            and re.fullmatch(r"(?:19|20)\d{2}", text)
        ):
            clean_values.append(int(text))
        else:
            clean_values.append(text)
    clean_values = _dedupe_values(clean_values)
    if operator not in NO_VALUE_OPERATORS and not clean_values:
        return [], "text", f"Predicate operator '{operator}' requires at least one nonblank value."
    return clean_values, "text", ""


def validate_filter_predicate_params(params: Any, available_columns: Any = None) -> tuple[bool, str]:
    """Validate execution params without accepting an expression language."""
    if not isinstance(params, dict):
        return False, "filter_predicates params must be an object."
    available = {str(column) for column in available_columns} if available_columns is not None else None
    group = predicate_group_from_params(params)
    valid, error, leaf_count = _validate_execution_group(group, available, depth=1)
    if not valid:
        return False, error
    if leaf_count < 1:
        return False, "filter_predicates requires at least one predicate."
    base_filter = params.get("base_filter")
    if base_filter is not None:
        if not isinstance(base_filter, dict) or base_filter.get("type") != "contains_any":
            return False, "The optional base_filter must be an approved contains_any operation."
        if not isinstance(base_filter.get("params"), dict):
            return False, "The optional base_filter params must be an object."
    return True, ""


def _validate_execution_group(group: Any, available: set[str] | None, *, depth: int) -> tuple[bool, str, int]:
    if not isinstance(group, dict):
        return False, "A predicate group must be an object.", 0
    if depth > MAX_PREDICATE_GROUP_DEPTH:
        return False, f"Predicate groups may be nested at most {MAX_PREDICATE_GROUP_DEPTH} levels.", 0
    logic = str(group.get("logic") or "and").casefold()
    if logic not in ALLOWED_PREDICATE_LOGIC:
        return False, f"Unsupported predicate logic '{logic}'.", 0
    clauses = group.get("clauses")
    if not isinstance(clauses, list):
        return False, "Predicate group clauses must be a list.", 0
    if len(clauses) > MAX_CLAUSES_PER_GROUP:
        return False, f"Only {MAX_CLAUSES_PER_GROUP} clauses are allowed per predicate group.", 0

    leaf_count = 0
    for clause in clauses:
        if not isinstance(clause, dict):
            return False, "Each predicate clause must be an object.", leaf_count
        clause_type = str(clause.get("type") or "").casefold()
        if clause_type == "group":
            valid, error, child_count = _validate_execution_group(clause, available, depth=depth + 1)
            if not valid:
                return valid, error, leaf_count
            leaf_count += child_count
            continue
        if clause_type != "predicate":
            return False, f"Unsupported predicate clause type '{clause_type or 'missing'}'.", leaf_count
        predicate = clause.get("predicate")
        valid, error = _validate_execution_predicate(predicate, available)
        if not valid:
            return valid, error, leaf_count
        leaf_count += 1
        if leaf_count > MAX_PREDICATES:
            return False, f"Only {MAX_PREDICATES} row predicates are allowed.", leaf_count
    return True, "", leaf_count


def _validate_execution_predicate(predicate: Any, available: set[str] | None) -> tuple[bool, str]:
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
    if len(columns) > 12:
        return False, "Each row predicate may resolve to at most 12 columns."
    if available is not None:
        missing = [str(column) for column in columns if str(column) not in available]
        if missing:
            return False, f"Predicate columns were not found: {', '.join(missing)}."
    values = predicate.get("values")
    if values is None:
        values = []
    if not isinstance(values, list):
        return False, "Predicate values must be a list."
    _clean, _value_type, error = normalize_operator_values(
        operator,
        values,
        semantic_column=str(predicate.get("semantic_column") or ""),
    )
    if error:
        return False, error
    return True, ""


def evaluate_predicate_row(row: Any, predicate: dict) -> bool:
    columns = predicate.get("columns") or predicate.get("resolved_columns") or []
    values = [_row_value(row, column) for column in columns]
    operator = str(predicate.get("operator") or "").casefold()
    quantifier = str(predicate.get("quantifier") or "any").casefold()
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
    valid_value_count = 0
    expected_values = predicate.get("values") or []
    for value in values:
        missing = _is_missing(value)
        if operator == "exists":
            scalar_results.append(not missing)
        elif operator == "missing":
            scalar_results.append(missing)
        elif operator in NUMERIC_OPERATORS:
            parsed = parse_numeric_value(value)
            if parsed is None:
                continue
            valid_value_count += 1
            scalar_results.append(_evaluate_numeric(parsed, operator, expected_values, predicate))
        elif operator in DATE_OPERATORS:
            parsed = parse_date_value(value)
            if parsed is None:
                continue
            valid_value_count += 1
            scalar_results.append(_evaluate_date(parsed, operator, expected_values, predicate))
        else:
            text = normalize_string_value(value)
            if text is None:
                continue
            valid_value_count += 1
            folded = text.casefold()
            expected_folded = [
                normalized
                for item in expected_values
                if (normalized := normalize_membership_value(item)) is not None
            ]
            if operator == "equals":
                scalar_results.append(any(folded == expected for expected in expected_folded))
            elif operator == "not_equals":
                scalar_results.append(all(folded != expected for expected in expected_folded))
            elif operator == "contains":
                scalar_results.append(any(expected in folded for expected in expected_folded))
            elif operator == "not_contains":
                scalar_results.append(all(expected not in folded for expected in expected_folded))
            elif operator == "in":
                scalar_results.append(folded in expected_folded)
            elif operator == "not_in":
                scalar_results.append(folded not in expected_folded)
            elif operator == "starts_with":
                scalar_results.append(folded.startswith(expected_folded[0]))
            elif operator == "ends_with":
                scalar_results.append(folded.endswith(expected_folded[0]))
    if operator not in NO_VALUE_OPERATORS and require_valid and valid_value_count == 0:
        return False
    return _quantify(
        scalar_results,
        quantifier,
        require_nonempty=require_valid and operator not in NO_VALUE_OPERATORS,
    )


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


def row_satisfies_predicate_group(
    row: Any,
    logic_or_group: str | dict,
    predicates: list[dict] | None = None,
) -> bool:
    """Evaluate a recursive group while preserving the legacy flat signature."""
    if isinstance(logic_or_group, dict):
        group = logic_or_group
    else:
        group = predicate_group_from_params(
            {"logic": logic_or_group, "predicates": list(predicates or [])}
        )
    outcomes = []
    for clause in group.get("clauses") or []:
        if clause.get("type") == "group":
            outcomes.append(row_satisfies_predicate_group(row, clause))
        elif clause.get("type") == "predicate":
            outcomes.append(evaluate_predicate_row(row, clause.get("predicate") or {}))
    if not outcomes:
        return True
    return any(outcomes) if group.get("logic") == "or" else all(outcomes)


def predicate_parse_failure_counts(row: Any, group: dict) -> tuple[int, int]:
    numeric_failures = 0
    date_failures = 0
    for predicate in predicate_group_leaves(group):
        operator = predicate.get("operator")
        columns = predicate.get("columns") or predicate.get("resolved_columns") or []
        for column in columns:
            value = _row_value(row, column)
            if _is_missing(value):
                continue
            if operator in NUMERIC_OPERATORS and parse_numeric_value(value) is None:
                numeric_failures += 1
            elif operator in DATE_OPERATORS and parse_date_value(value) is None:
                date_failures += 1
    return numeric_failures, date_failures


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
        ("employer", "starts_with", r"\b(?:employer|company|organization|organisation)(?:\s+name)?\s+starts?\s+with\s+([^,;?.]+)"),
        ("occupation", "starts_with", r"\b(?:title|occupation|job\s+title|role)(?:\s+name)?\s+starts?\s+with\s+([^,;?.]+)"),
        ("employer", "ends_with", r"\b(?:employer|company|organization|organisation)(?:\s+name)?\s+ends?\s+with\s+([^,;?.]+)"),
        ("occupation", "ends_with", r"\b(?:title|occupation|job\s+title|role)(?:\s+name)?\s+ends?\s+with\s+([^,;?.]+)"),
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
    comparative_year = re.search(
        r"\b(?:graduated|graduating|graduation\s+year|grad\s+year|class\s+of)\b"
        r".{0,20}\b(?:between|from|after|before|over|under|at\s+least|at\s+most|more\s+than|less\s+than)\b",
        original,
        re.IGNORECASE,
    )
    if grad_year and not comparative_year:
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


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_NUMBER_TOKEN = (
    r"(?:\$\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
)
_COMPARISON_WORDS = (
    r"more\s+than|greater\s+than|over|after|"
    r"at\s+least|no\s+fewer\s+than|"
    r"less\s+than|under|fewer\s+than|before|"
    r"at\s+most|no\s+more\s+than"
)


def _extract_numeric_predicates(original: str) -> list[dict]:
    predicates = []
    field_patterns = {
        "grad_year": r"(?:graduat(?:ed|ing)|graduation\s+year|grad(?:uation)?\s+year|class\s+of)",
        "lifetime_giving": (
            r"(?:lifetime\s+(?:giving|gifts?|gift\s+amount)|total\s+giving|giving|"
            r"gift\s+amount|donat(?:ed|ions?)|(?:have\s+)?given)"
        ),
        "event_count": r"(?:event\s+count|events?\s+attended|attended|events?)",
    }
    for semantic, field_pattern in field_patterns.items():
        range_patterns = [
            rf"\b{field_pattern}\b.{{0,24}}?\b(?:between|from)\s+({_NUMBER_TOKEN})\s+(?:and|through|to)\s+({_NUMBER_TOKEN})",
            rf"\b(?:between|from)\s+({_NUMBER_TOKEN})\s+(?:and|through|to)\s+({_NUMBER_TOKEN}).{{0,24}}?\b{field_pattern}\b",
        ]
        range_match = next(
            (match for pattern in range_patterns if (match := re.search(pattern, original, re.IGNORECASE))),
            None,
        )
        if range_match:
            lower = _number_token_value(range_match.group(1))
            upper = _number_token_value(range_match.group(2))
            predicates.append(
                _predicate(
                    semantic,
                    "between",
                    values=[lower, upper],
                    require_valid_value=True,
                    lower_inclusive=True,
                    upper_inclusive=True,
                )
            )
            continue

        quantified = re.search(
            rf"\b({_NUMBER_TOKEN})\s+or\s+(more|fewer)\s+(?:in\s+|for\s+)?{field_pattern}\b",
            original,
            re.IGNORECASE,
        ) or re.search(
            rf"\b{field_pattern}\b.{{0,18}}?\b({_NUMBER_TOKEN})\s+or\s+(more|fewer)\b",
            original,
            re.IGNORECASE,
        )
        if quantified:
            predicates.append(
                _predicate(
                    semantic,
                    (
                        "greater_than_or_equal"
                        if quantified.group(2).casefold() == "more"
                        else "less_than_or_equal"
                    ),
                    values=[_number_token_value(quantified.group(1))],
                    require_valid_value=True,
                )
            )
            continue

        forward = re.search(
            rf"\b{field_pattern}\b.{{0,24}}?\b({_COMPARISON_WORDS})\s+({_NUMBER_TOKEN})\b",
            original,
            re.IGNORECASE,
        )
        reverse = re.search(
            rf"\b({_COMPARISON_WORDS})\s+({_NUMBER_TOKEN})\s+(?:in\s+|for\s+|total\s+)?{field_pattern}\b",
            original,
            re.IGNORECASE,
        )
        # Natural event wording places the count before the field:
        # "attended at least three events".
        event_reverse = None
        if semantic == "event_count":
            event_reverse = re.search(
                rf"\b(?:attended|attend)\s+({_COMPARISON_WORDS})\s+({_NUMBER_TOKEN})\s+events?\b",
                original,
                re.IGNORECASE,
            )
        match = forward or reverse or event_reverse
        if not match:
            continue
        wording = match.group(1)
        numeric_value = _number_token_value(match.group(2))
        predicates.append(
            _predicate(
                semantic,
                _comparison_operator(wording),
                values=[numeric_value],
                require_valid_value=True,
            )
        )
    return predicates


def _extract_membership_predicates(original: str) -> list[dict]:
    predicates = []
    patterns = [
        ("not_in", r"\b(?:outside|not\s+in|excluding)\s+([^?.;]+)"),
        (
            "in",
            r"\b(?:alumni|people|records|consultants?|engineers?|donors?)\s+"
            r"(?:who\s+(?:are|live|work)\s+)?(?:located\s+|based\s+)?(?:in|from)\s+([^?.;]+)",
        ),
        ("in", r"\b(?:city|cities|location|locations)\s+(?:is|are|in|one\s+of)\s+([^?.;]+)"),
    ]
    for operator, pattern in patterns:
        match = re.search(pattern, original, re.IGNORECASE)
        if not match:
            continue
        candidate = _trim_following_constraint(match.group(1))
        values = _split_membership_values(candidate)
        if len(values) < 2:
            continue
        predicates.append(
            _predicate(
                "city",
                operator,
                values=values,
                require_valid_value=True,
            )
        )
        break
    return predicates


_MONTH_PATTERN = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_DATE_TOKEN = (
    rf"(?:\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|"
    rf"{_MONTH_PATTERN}(?:\s+\d{{1,2}}(?:st|nd|rd|th)?)?(?:,)?\s+\d{{4}}|\d{{4}})"
)


def _extract_date_predicates(original: str, *, now: Any = None) -> list[dict]:
    predicates = []
    fields = {
        "last_contact_date": r"(?:last\s+contact(?:\s+date)?|contacted)",
        "updated_at": r"(?:records?\s+)?(?:updated|last\s+updated|modified)",
        "created_at": r"(?:records?\s+)?(?:created|creation\s+date)",
    }
    for semantic, field_pattern in fields.items():
        relative = re.search(
            rf"\b{field_pattern}\b.{{0,24}}?\bwithin\s+(?:the\s+)?past\s+(\d{{1,4}})\s+days?\b",
            original,
            re.IGNORECASE,
        )
        if relative:
            days = int(relative.group(1))
            try:
                boundary = relative_date_boundary(days, now=now)
            except ValueError:
                continue
            predicates.append(
                _predicate(
                    semantic,
                    "date_after",
                    values=[boundary.isoformat()],
                    require_valid_value=True,
                    relative_date_label=f"past_{days}_days",
                )
            )
            continue

        between = re.search(
            rf"\b{field_pattern}\b.{{0,24}}?\b(?:between|from)\s+({_DATE_TOKEN})\s+"
            rf"(?:and|through|to)\s+({_DATE_TOKEN})",
            original,
            re.IGNORECASE,
        )
        if between:
            predicates.append(
                _predicate(
                    semantic,
                    "date_between",
                    values=[_clean_ordinal_date(between.group(1)), _clean_ordinal_date(between.group(2))],
                    require_valid_value=True,
                    lower_inclusive=True,
                    upper_inclusive=True,
                )
            )
            continue
        comparison = re.search(
            rf"\b{field_pattern}\b.{{0,24}}?\b(before|after)\s+({_DATE_TOKEN})",
            original,
            re.IGNORECASE,
        )
        if comparison:
            predicates.append(
                _predicate(
                    semantic,
                    "date_before" if comparison.group(1).casefold() == "before" else "date_after",
                    values=[_clean_ordinal_date(comparison.group(2))],
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
    clauses_explicitly_separated = bool(fuzzy_text and exact_texts)
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
    constraint_logic = (
        "or"
        if clauses_explicitly_separated and len(clauses) > 1 and "or" in connectors and "and" not in connectors
        else "and"
    )
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
    values: list[Any] | None = None,
    quantifier: str = "any",
    require_valid_value: bool = False,
    **metadata: Any,
) -> dict:
    predicate = {
        "target_entity": "alumni_row",
        "semantic_column": semantic_column,
        "operator": operator,
        "values": list(values or []),
        "include_subdomains": True,
        "quantifier": quantifier,
        "require_valid_value": require_valid_value,
        "source": "deterministic_explicit",
    }
    for key in ("lower_inclusive", "upper_inclusive", "relative_date_label"):
        if key in metadata:
            predicate[key] = metadata[key]
    return predicate


def _predicate_clause(predicate: Any) -> dict:
    return {"type": "predicate", "predicate": predicate}


def _predicate_root(logic: str, clauses: list[dict]) -> dict:
    normalized_logic = logic if logic in ALLOWED_PREDICATE_LOGIC else "and"
    root = {
        "logic": normalized_logic,
        "clauses": list(clauses or []),
    }
    # Keep the historical flat list as a read-only compatibility view.
    root["predicates"] = predicate_group_leaves(root)
    return root


def predicate_group_leaves(group: Any) -> list[dict]:
    if not isinstance(group, dict):
        return []
    clauses = group.get("clauses")
    if not isinstance(clauses, list):
        predicates = group.get("predicates")
        return [predicate for predicate in predicates or [] if isinstance(predicate, dict)]
    leaves = []
    for clause in clauses:
        if not isinstance(clause, dict):
            continue
        if clause.get("type") == "predicate" and isinstance(clause.get("predicate"), dict):
            leaves.append(clause["predicate"])
        elif clause.get("type") == "group":
            leaves.extend(predicate_group_leaves(clause))
    return leaves


def predicate_group_depth(group: Any) -> int:
    if not isinstance(group, dict):
        return 0
    if not predicate_group_leaves(group):
        return 0
    clauses = group.get("clauses")
    if not isinstance(clauses, list):
        return 1 if group.get("predicates") else 0
    child_depths = [
        predicate_group_depth(clause)
        for clause in clauses
        if isinstance(clause, dict) and clause.get("type") == "group"
    ]
    return 1 + (max(child_depths) if child_depths else 0)


def predicate_group_from_params(params: Any) -> dict:
    if not isinstance(params, dict):
        return _predicate_root("and", [])
    explicit = params.get("predicate_group")
    if isinstance(explicit, dict):
        return explicit
    if isinstance(params.get("clauses"), list):
        return {
            "logic": str(params.get("logic") or "and").casefold(),
            "clauses": params["clauses"],
        }
    predicates = params.get("predicates")
    if not isinstance(predicates, list):
        predicates = []
    return _predicate_root(
        str(params.get("logic") or "and").casefold(),
        [_predicate_clause(predicate) for predicate in predicates],
    )


def predicate_group_for_execution(group: dict) -> dict:
    """Replace resolved_columns with execution columns recursively."""
    clauses = []
    for clause in group.get("clauses") or []:
        if clause.get("type") == "group":
            child = predicate_group_for_execution(clause)
            clauses.append({"type": "group", "logic": child["logic"], "clauses": child["clauses"]})
            continue
        predicate = dict(clause.get("predicate") or {})
        predicate["columns"] = list(predicate.get("columns") or predicate.get("resolved_columns") or [])
        predicate.pop("resolved_columns", None)
        clauses.append(_predicate_clause(predicate))
    return _predicate_root(str(group.get("logic") or "and").casefold(), clauses)


def _merge_explicit_and_model_groups(explicit_root: dict, model_predicates: list[dict]) -> dict:
    if not model_predicates:
        return explicit_root
    clauses = []
    explicit_clauses = explicit_root.get("clauses") or []
    if explicit_root.get("logic") == "and":
        clauses.extend(copy.deepcopy(explicit_clauses))
    else:
        clauses.append(
            {
                "type": "group",
                "logic": explicit_root.get("logic") or "and",
                "clauses": copy.deepcopy(explicit_clauses),
            }
        )
    clauses.extend(_predicate_clause(predicate) for predicate in model_predicates)
    return _predicate_root("and", clauses)


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
        return bool(
            (explicit_values and candidate_values and explicit_values != candidate_values)
            or bool(explicit.get("lower_inclusive", True)) != bool(candidate.get("lower_inclusive", True))
            or bool(explicit.get("upper_inclusive", True)) != bool(candidate.get("upper_inclusive", True))
        )
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
    # Comparisons, ranges, membership, dates, and string boundaries on the same
    # source-anchored field cannot be safely broadened or reversed by a model.
    if explicit.get("operator") in (
        NUMERIC_OPERATORS | DATE_OPERATORS | MEMBERSHIP_OPERATORS | STRING_BOUNDARY_OPERATORS
    ):
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


def _people_spec_is_redundant_with_exact(spec, predicate_root, question):
    if not isinstance(spec, dict) or not predicate_group_leaves(predicate_root):
        return False
    filter_type = spec.get("filter_type")
    semantics = {
        predicate.get("semantic_column")
        for predicate in predicate_group_leaves(predicate_root)
    }
    if filter_type == "occupation" and "occupation" in semantics:
        return True
    if filter_type != "employer":
        return False
    if "employer" in semantics:
        return True
    text = _normalized_sentence(question)
    return not bool(
        re.search(
            r"\b(?:works?|working)\s+(?:at|for)\b|\b(?:employer|company|organization|organisation)\b",
            text,
        )
    )


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


def _predicate_is_explicit_in_question(question, predicate):
    if not _semantic_is_mentioned(question, predicate):
        return False
    text = _normalized_sentence(question)
    operator = predicate.get("operator")
    patterns = {
        "exists": r"\b(?:with|has|have|having|exists?|present)\b",
        "missing": r"\b(?:missing|without|lacking|no|blank|empty|null)\b",
        "equals": r"\b(?:is|equals?|exactly)\b",
        "not_equals": r"\b(?:is not|isn't|not equal|does not equal)\b",
        "contains": r"\bcontains?\b",
        "not_contains": r"\b(?:does not contain|doesn't contain|not contain)\b",
        "greater_than": r"\b(?:more than|greater than|over|after)\b",
        "greater_than_or_equal": r"\b(?:at least|no fewer than|or more)\b",
        "less_than": r"\b(?:less than|under|fewer than|before)\b",
        "less_than_or_equal": r"\b(?:at most|no more than|or fewer)\b",
        "between": r"\b(?:between|from)\b",
        "in": r"\b(?:one of|in|among)\b",
        "not_in": r"\b(?:not in|outside|excluding)\b",
        "date_before": r"\bbefore\b",
        "date_after": r"\bafter\b|\bwithin\s+(?:the\s+)?past\b",
        "date_between": r"\b(?:between|from)\b",
        "starts_with": r"\bstarts?\s+with\b",
        "ends_with": r"\bends?\s+with\b",
        "email_domain_in": r"\b(?:domain|email|e-mail)\b.{0,50}\b(?:in|from|at|only|cornell|gmail)\b",
        "email_domain_not_in": r"\b(?:non[- ]|outside|not|external|personal)\b.{0,50}\b(?:email|domain|cornell)\b",
    }
    pattern = patterns.get(operator)
    return bool(pattern and re.search(pattern, text))


def _equivalent_predicates(left, right):
    return (
        left.get("semantic_column") == right.get("semantic_column")
        and left.get("operator") == right.get("operator")
        and left.get("quantifier", "any") == right.get("quantifier", "any")
        and bool(left.get("include_subdomains", True)) == bool(right.get("include_subdomains", True))
        and bool(left.get("lower_inclusive", True)) == bool(right.get("lower_inclusive", True))
        and bool(left.get("upper_inclusive", True)) == bool(right.get("upper_inclusive", True))
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


def _dedupe_values(values):
    deduped = []
    seen = set()
    for value in values:
        key = (type(value).__name__, str(value).casefold())
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def _evaluate_numeric(parsed, operator, expected_values, predicate):
    bounds = [parse_numeric_value(value) for value in expected_values]
    if any(value is None for value in bounds):
        return False
    if operator == "greater_than":
        return parsed > bounds[0]
    if operator == "greater_than_or_equal":
        return parsed >= bounds[0]
    if operator == "less_than":
        return parsed < bounds[0]
    if operator == "less_than_or_equal":
        return parsed <= bounds[0]
    if operator == "between":
        lower_ok = parsed >= bounds[0] if predicate.get("lower_inclusive", True) else parsed > bounds[0]
        upper_ok = parsed <= bounds[1] if predicate.get("upper_inclusive", True) else parsed < bounds[1]
        return lower_ok and upper_ok
    return False


def _evaluate_date(parsed, operator, expected_values, predicate):
    bounds = [parse_date_value(value) for value in expected_values]
    if any(value is None for value in bounds):
        return False
    if operator == "date_before":
        return parsed < bounds[0]
    if operator == "date_after":
        return parsed > bounds[0]
    if operator == "date_between":
        lower_ok = parsed >= bounds[0] if predicate.get("lower_inclusive", True) else parsed > bounds[0]
        upper_ok = parsed <= bounds[1] if predicate.get("upper_inclusive", True) else parsed < bounds[1]
        return lower_ok and upper_ok
    return False


def _number_token_value(value):
    text = str(value or "").strip().casefold()
    if text in _NUMBER_WORDS:
        return _NUMBER_WORDS[text]
    parsed = parse_numeric_value(text)
    return parsed if parsed is not None else text


def _comparison_operator(wording):
    text = _normalized_sentence(wording)
    if re.search(r"\b(?:at least|no fewer than)\b|\bor more\b", text):
        return "greater_than_or_equal"
    if re.search(r"\b(?:at most|no more than)\b|\bor fewer\b", text):
        return "less_than_or_equal"
    if re.search(r"\b(?:less than|under|fewer than|before)\b", text):
        return "less_than"
    return "greater_than"


def _split_membership_values(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip(" ,")
    if not text:
        return []
    parts = re.split(r"\s*,\s*|\s+(?:or|and)\s+", text, flags=re.IGNORECASE)
    values = []
    for part in parts:
        candidate = part.strip(" ,\"'")
        candidate = re.sub(r"^(?:or|and)\s+", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"^(?:the\s+)?(?:cities?\s+of\s+)", "", candidate, flags=re.IGNORECASE)
        if not candidate or len(candidate) > 80:
            continue
        # Reject obvious predicate prose accidentally captured after a location.
        if re.search(
            r"\b(?:have|has|with|whose|graduated|giving|events?|email|contacted|employer|title)\b",
            candidate,
            re.IGNORECASE,
        ):
            continue
        values.append(candidate)
    return _dedupe_values(values)[:MAX_PREDICATE_VALUES]


def _trim_following_constraint(value):
    text = str(value or "")
    return re.split(
        r"\s+and\s+(?=(?:have|has|with|whose|graduated|giving|attended|contacted|email|employer|title)\b)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


def _clean_ordinal_date(value):
    return re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", str(value or ""), flags=re.IGNORECASE)


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
