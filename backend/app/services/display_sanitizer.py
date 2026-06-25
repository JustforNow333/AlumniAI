"""Sanitize structured analysis data at user-visible response boundaries.

Analysis and classification code may use internal columns while selecting,
sorting, or debugging rows. This module removes those fields only after the
analysis is complete, without changing row selection or ordering.
"""

from __future__ import annotations

import copy
import re
from typing import Any


_ALWAYS_FORBIDDEN_KEYS = {
    "debug",
    "match_reason",
    "raw_match_reason",
    "matched_column",
    "matched_term",
    "matched_terms",
    "match_sources",
    "internal_reason",
    "classifier_reason",
    "classification_reason",
    "uncertainty_reason",
    "model_reason",
    "rationale",
    "classifier_rationale",
    "confidence",
    "classifier_confidence",
    "belongs_to_industry",
    "classification",
    "internal_score",
    "scorer_score",
    "row_index",
}

_OPTIONAL_COLUMN_GROUPS = {
    "major": {
        "major",
        "majors",
        "degree",
        "degrees",
        "field_of_study",
        "academic_background",
        "concentration",
    },
    "notes": {"note", "notes", "comment", "comments"},
    "email": {"email", "email_address", "e_mail"},
    "phone": {"phone", "phone_number", "mobile", "mobile_phone"},
    "grad_year": {
        "graduation_year",
        "graduation_yr",
        "grad_year",
        "grad_yr",
        "class_year",
        "class_yr",
    },
    "location": {
        "location",
        "city",
        "state",
        "province",
        "region",
        "country",
    },
    "preferred_name": {"preferred_name", "nickname", "nick_name"},
    "industry": {"industry", "sector"},
    "reason": {"reason"},
}


def normalize_display_key(value: Any) -> str:
    """Normalize spaces, underscores, and hyphens for case-insensitive checks."""
    text = str(value or "").strip().lower()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text)).strip("_")


def is_forbidden_display_column(column: Any) -> bool:
    """Return True for internal, debug, scorer, or eval-only fields."""
    raw = str(column or "").strip()
    key = normalize_display_key(raw)
    if not key:
        return False
    if raw.startswith("_"):
        return True
    if key.startswith("expected_") or key.startswith("eval_"):
        return True
    return key in _ALWAYS_FORBIDDEN_KEYS


def question_requests_major(question: Any) -> bool:
    """Detect an explicit request for academic major information.

    A bare phrase such as "major tech companies" intentionally does not match.
    """
    text = _normalized_question(question)
    if not text:
        return False
    patterns = [
        r"\bmajors\b",
        r"\bmajor(?:ed|ing)?\s+in\b",
        r"\b(?:their|the|include|show|list|with)\s+major\b",
        r"\bwhat\s+(?:did|do)\b.*\bmajor\b",
        r"\bfield(?:s)?\s+of\s+study\b",
        r"\bacademic\s+background\b",
        r"\bconcentrations?\b",
        r"\bdegrees?\b",
        r"\bwhat\b.*\b(?:study|studied)\b",
        r"\b(?:study|studied)\b.*\b(?:at|in)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def sanitize_display_columns(columns: Any, question: Any = "") -> list[str]:
    """Return safe user-visible columns while preserving their original order."""
    if not isinstance(columns, (list, tuple)):
        return []

    kept = []
    for column in columns:
        text = str(column or "").strip()
        if not text or is_forbidden_display_column(text):
            continue
        optional_group = _optional_column_group(text)
        if optional_group and not _question_requests_optional_group(question, optional_group):
            continue
        kept.append(text)
    return kept


def sanitize_display_rows(columns: Any, rows: Any, question: Any = "") -> tuple[list[str], list[Any]]:
    """Sanitize table columns and rows without changing row count or ordering."""
    source_columns = [str(column or "").strip() for column in columns or []]
    if not source_columns and isinstance(rows, list):
        source_columns = _columns_from_mapping_rows(rows)

    kept_columns = sanitize_display_columns(source_columns, question)
    kept_indices = [
        index
        for index, column in enumerate(source_columns)
        if column in kept_columns and source_columns.index(column) == index
    ]

    sanitized_rows = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            sanitized_rows.append(
                {
                    column: _mapping_value(row, column)
                    for column in kept_columns
                }
            )
        elif isinstance(row, (list, tuple)):
            sanitized_rows.append(
                [row[index] if index < len(row) else None for index in kept_indices]
            )
        else:
            sanitized_rows.append(
                [row] + [None] * max(len(kept_columns) - 1, 0)
                if kept_columns
                else []
            )
    return kept_columns, sanitized_rows


def sanitize_operation_result(result: Any, question: Any = "") -> Any:
    """Sanitize one structured operation result."""
    if not isinstance(result, dict):
        return copy.deepcopy(result)

    sanitized = _strip_internal_mapping_keys(result)
    columns = sanitized.get("columns")
    rows = sanitized.get("rows")
    if isinstance(rows, list):
        source_columns = columns
        if not isinstance(source_columns, list):
            source_columns = sanitized.get("visible_columns") or sanitized.get("display_columns") or []
        safe_columns, safe_rows = sanitize_display_rows(source_columns, rows, question)
        sanitized["columns"] = safe_columns
        sanitized["rows"] = safe_rows

    for row_key in ("direct_rows", "adjacent_rows", "uncertain_rows"):
        if isinstance(sanitized.get(row_key), list):
            source_columns = sanitized.get("visible_columns") or sanitized.get("display_columns") or sanitized.get("columns") or []
            _safe_columns, safe_rows = sanitize_display_rows(source_columns, sanitized[row_key], question)
            sanitized[row_key] = safe_rows

    if isinstance(sanitized.get("row_sections"), list):
        sanitized["row_sections"] = [
            _sanitize_row_section(section, question)
            for section in sanitized["row_sections"]
            if isinstance(section, dict)
        ]

    for key in ("visible_columns", "display_columns"):
        if isinstance(sanitized.get(key), list):
            sanitized[key] = sanitize_display_columns(sanitized[key], question)

    metrics = sanitized.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("display_columns"), list):
        metrics["display_columns"] = sanitize_display_columns(metrics["display_columns"], question)

    return sanitized


def _sanitize_row_section(section: dict[str, Any], question: Any = "") -> dict[str, Any]:
    clean_section = _strip_internal_mapping_keys(section)
    columns, rows = sanitize_display_rows(
        clean_section.get("columns") or [],
        clean_section.get("rows") or [],
        question,
    )
    clean_section["columns"] = columns
    clean_section["rows"] = rows
    return clean_section


def sanitize_operation_results(results: Any, question: Any = "") -> list[Any]:
    if not isinstance(results, list):
        return []
    return [sanitize_operation_result(result, question) for result in results]


def sanitize_answer(answer: Any, question: Any = "") -> Any:
    """Sanitize structured table and metric blocks in an answer object."""
    if not isinstance(answer, dict):
        return copy.deepcopy(answer)

    sanitized = _strip_internal_mapping_keys(answer)
    blocks = []
    for block in sanitized.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        clean_block = _strip_internal_mapping_keys(block)
        if clean_block.get("type") == "table":
            columns, rows = sanitize_display_rows(
                clean_block.get("columns") or [],
                clean_block.get("rows") or [],
                question,
            )
            clean_block["columns"] = columns
            clean_block["rows"] = rows
        elif clean_block.get("type") == "metrics":
            clean_block["items"] = [
                item
                for item in clean_block.get("items") or []
                if not (
                    isinstance(item, dict)
                    and is_forbidden_display_column(item.get("label"))
                )
            ]
        blocks.append(clean_block)
    sanitized["blocks"] = blocks
    return sanitized


def sanitize_response_payload(payload: Any, question: Any = None) -> Any:
    """Sanitize all structured user-visible tables in an API response snapshot."""
    if not isinstance(payload, dict):
        return copy.deepcopy(payload)

    effective_question = str(
        question if question is not None else payload.get("question") or ""
    )
    sanitized = _strip_internal_mapping_keys(payload)

    if isinstance(sanitized.get("operation_results"), list):
        sanitized["operation_results"] = sanitize_operation_results(
            sanitized["operation_results"],
            effective_question,
        )
    if isinstance(sanitized.get("result"), dict):
        sanitized["result"] = sanitize_operation_result(
            sanitized["result"],
            effective_question,
        )
    if isinstance(sanitized.get("answer"), dict):
        sanitized["answer"] = sanitize_answer(
            sanitized["answer"],
            effective_question,
        )
    if isinstance(sanitized.get("rows"), list):
        columns, rows = sanitize_display_rows(
            sanitized.get("columns") or [],
            sanitized["rows"],
            effective_question,
        )
        sanitized["columns"] = columns
        sanitized["rows"] = rows

    return sanitized


def _strip_internal_mapping_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_internal_mapping_keys(item)
            for key, item in value.items()
            if not is_forbidden_display_column(key)
        }
    if isinstance(value, list):
        return [_strip_internal_mapping_keys(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_internal_mapping_keys(item) for item in value]
    return copy.deepcopy(value)


def _optional_column_group(column: Any) -> str | None:
    key = normalize_display_key(column)
    for group, aliases in _OPTIONAL_COLUMN_GROUPS.items():
        if key in aliases:
            return group
    return None


def _question_requests_optional_group(question: Any, group: str) -> bool:
    text = _normalized_question(question)
    if group == "major":
        return question_requests_major(text)
    if group == "notes":
        return bool(re.search(r"\b(?:note|notes|comment|comments)\b", text))
    if group == "email":
        return bool(re.search(r"\b(?:email|emails|e-mail|contact info|contact information)\b", text))
    if group == "phone":
        return bool(re.search(r"\b(?:phone|phones|mobile|contact info|contact information)\b", text))
    if group == "grad_year":
        return bool(re.search(r"\b(?:graduation|graduated|grad year|class year|class of)\b", text))
    if group == "location":
        return bool(re.search(r"\b(?:location|locations|city|cities|state|country|where|based in|located in)\b", text))
    if group == "preferred_name":
        return bool(re.search(r"\b(?:preferred name|nickname|nick name)\b", text))
    if group == "industry":
        return bool(re.search(r"\b(?:industry|industries|sector|sectors)\b", text))
    if group == "reason":
        return bool(re.search(r"\b(?:reason|reasons|why)\b", text))
    return True


def _normalized_question(question: Any) -> str:
    return re.sub(r"\s+", " ", str(question or "").lower()).strip()


def _columns_from_mapping_rows(rows: list[Any]) -> list[str]:
    columns = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            text = str(key or "").strip()
            if text and text not in columns:
                columns.append(text)
    return columns


def _mapping_value(row: dict[Any, Any], column: str) -> Any:
    if column in row:
        return row[column]
    target = normalize_display_key(column)
    for key, value in row.items():
        if normalize_display_key(key) == target:
            return value
    return None
