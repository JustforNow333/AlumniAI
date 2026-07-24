"""Validation and normalization for persisted dataset schema profiles."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from app.services.canonical_schema import (
    CANONICAL_FIELD_REGISTRY,
    SCHEMA_PROFILE_VERSION,
    SCHEMA_STATUSES,
)


MAX_MAPPINGS = 64
MAX_SOURCE_COLUMNS_PER_FIELD = 10
MAX_IGNORED_COLUMNS = 500
MAX_COLUMN_NAME_LENGTH = 500
MAX_EVIDENCE_ITEMS = 5
MAX_EVIDENCE_LENGTH = 240
HIGH_CONFIDENCE_THRESHOLD = 0.90
MEDIUM_CONFIDENCE_THRESHOLD = 0.65


class SchemaProfileValidationError(ValueError):
    pass


def utc_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def confidence_label(confidence):
    score = _bounded_confidence(confidence)
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def schema_summary(profile):
    if not isinstance(profile, dict):
        return {
            "schema_status": "not_analyzed",
            "schema_mapped_count": 0,
            "schema_unmapped_count": 0,
            "schema_conflict_count": 0,
            "schema_version": None,
        }
    mappings = profile.get("mappings") if isinstance(profile.get("mappings"), dict) else {}
    return {
        "schema_status": (
            profile.get("status") if profile.get("status") in SCHEMA_STATUSES else "not_analyzed"
        ),
        "schema_mapped_count": len(
            [
                item
                for item in mappings.values()
                if isinstance(item, dict) and item.get("source_columns")
            ]
        ),
        "schema_unmapped_count": len(_clean_string_list(profile.get("unmapped_columns"), MAX_IGNORED_COLUMNS)),
        "schema_conflict_count": len(profile.get("conflicts") or [])
        if isinstance(profile.get("conflicts"), list)
        else 0,
        "schema_version": profile.get("version") or SCHEMA_PROFILE_VERSION,
    }


def canonical_to_source(profile, source_columns=None, *, include_inferred=True):
    """Return safe active mappings in priority order.

    User-confirmed mappings always participate. Unconfirmed automatic mappings
    participate only at high confidence, preserving legacy resolver behavior
    for reviewable medium/low suggestions.
    """
    if not isinstance(profile, dict):
        return {}
    allowed_sources = (
        {str(column) for column in source_columns}
        if source_columns is not None
        else None
    )
    if allowed_sources == set():
        allowed_sources = None
    mappings = profile.get("mappings") if isinstance(profile.get("mappings"), dict) else {}
    active = {}
    used_sources = set()
    ordered = sorted(
        mappings.items(),
        key=lambda pair: (
            not bool(pair[1].get("user_confirmed"))
            if isinstance(pair[1], dict)
            else True
        ),
    )
    for key, item in ordered:
        if key not in CANONICAL_FIELD_REGISTRY or not isinstance(item, dict):
            continue
        user_confirmed = bool(item.get("user_confirmed"))
        confidence = _bounded_confidence(item.get("confidence"))
        if not user_confirmed and (not include_inferred or confidence < HIGH_CONFIDENCE_THRESHOLD):
            continue
        sources = [
            source
            for source in _clean_string_list(
                item.get("source_columns"), MAX_SOURCE_COLUMNS_PER_FIELD
            )
            if (allowed_sources is None or source in allowed_sources)
            and source not in used_sources
        ]
        if sources:
            active[key] = sources
            used_sources.update(sources)
    return active


def normalize_profile_for_storage(profile, source_columns):
    """Tolerate older/malformed registry values without trusting them."""
    if not isinstance(profile, dict):
        return None
    source_names = [str(column) for column in source_columns]
    source_set = set(source_names)
    mappings = {}
    assigned_sources = set()
    raw_mappings = profile.get("mappings") if isinstance(profile.get("mappings"), dict) else {}
    for key, raw in list(raw_mappings.items())[:MAX_MAPPINGS]:
        if key not in CANONICAL_FIELD_REGISTRY or not isinstance(raw, dict):
            continue
        sources = [
            source
            for source in _clean_string_list(
                raw.get("source_columns"), MAX_SOURCE_COLUMNS_PER_FIELD
            )
            if source in source_set and source not in assigned_sources
        ]
        cardinality = CANONICAL_FIELD_REGISTRY[key]["cardinality"]
        if cardinality == "single":
            sources = sources[:1]
        if not sources:
            continue
        confidence = _bounded_confidence(raw.get("confidence"))
        mappings[key] = {
            "source_columns": sources,
            "confidence": confidence,
            "confidence_label": confidence_label(confidence),
            "method": _clean_text(raw.get("method"), 80) or "saved_profile",
            "user_confirmed": bool(raw.get("user_confirmed")),
            "evidence": _clean_evidence(raw.get("evidence")),
        }
        assigned_sources.update(sources)
    assigned = {source for item in mappings.values() for source in item["source_columns"]}
    ignored = [
        source
        for source in _clean_string_list(profile.get("ignored_columns"), MAX_IGNORED_COLUMNS)
        if source in source_set and source not in assigned
    ]
    unmapped = [source for source in source_names if source not in assigned and source not in ignored]
    status = profile.get("status") if profile.get("status") in SCHEMA_STATUSES else "unreviewed"
    return {
        "version": SCHEMA_PROFILE_VERSION,
        "status": status,
        "generated_at": profile.get("generated_at") or utc_timestamp(),
        "updated_at": profile.get("updated_at") or profile.get("generated_at") or utc_timestamp(),
        "confirmed_at": profile.get("confirmed_at") if status == "confirmed" else None,
        "mappings": mappings,
        "unmapped_columns": unmapped,
        "ignored_columns": ignored,
        "conflicts": _clean_records(profile.get("conflicts"), 100),
        "warnings": _clean_string_list(profile.get("warnings"), 50, max_length=MAX_EVIDENCE_LENGTH),
    }


def validate_schema_update(payload, source_columns, existing_profile=None):
    if not isinstance(payload, dict):
        raise SchemaProfileValidationError("Request body must be a JSON object.")
    raw_mappings = payload.get("mappings", {})
    if not isinstance(raw_mappings, dict):
        raise SchemaProfileValidationError("mappings must be an object.")
    if len(raw_mappings) > MAX_MAPPINGS:
        raise SchemaProfileValidationError(f"At most {MAX_MAPPINGS} mappings are allowed.")

    source_names = [str(column) for column in source_columns]
    source_set = set(source_names)
    existing_mappings = (
        existing_profile.get("mappings")
        if isinstance(existing_profile, dict) and isinstance(existing_profile.get("mappings"), dict)
        else {}
    )
    mappings = {}
    assigned_to = {}
    now = utc_timestamp()

    for key, raw in raw_mappings.items():
        if key not in CANONICAL_FIELD_REGISTRY:
            raise SchemaProfileValidationError(f"Unknown canonical field '{key}'.")
        if isinstance(raw, list):
            raw = {"source_columns": raw}
        if not isinstance(raw, dict):
            raise SchemaProfileValidationError(
                f"Mapping for '{key}' must be an object or source-column list."
            )
        raw_sources = raw.get("source_columns")
        if not isinstance(raw_sources, list):
            raise SchemaProfileValidationError(
                f"source_columns for '{key}' must be a list."
            )
        if len(raw_sources) > MAX_SOURCE_COLUMNS_PER_FIELD:
            raise SchemaProfileValidationError(
                f"Mapping '{key}' has too many source columns."
            )
        sources = _clean_string_list(raw_sources, MAX_SOURCE_COLUMNS_PER_FIELD)
        missing = [source for source in sources if source not in source_set]
        if missing:
            raise SchemaProfileValidationError(
                f"Source column '{missing[0]}' does not exist in this dataset."
            )
        if CANONICAL_FIELD_REGISTRY[key]["cardinality"] == "single" and len(sources) > 1:
            raise SchemaProfileValidationError(
                f"Canonical field '{key}' accepts only one source column."
            )
        for source in sources:
            if source in assigned_to and assigned_to[source] != key:
                raise SchemaProfileValidationError(
                    f"Source column '{source}' is assigned to both "
                    f"'{assigned_to[source]}' and '{key}'."
                )
            assigned_to[source] = key
        if not sources:
            continue
        previous = existing_mappings.get(key) if isinstance(existing_mappings.get(key), dict) else {}
        same_sources = list(previous.get("source_columns") or []) == sources
        confidence = _bounded_confidence(previous.get("confidence") if same_sources else 1.0)
        mappings[key] = {
            "source_columns": sources,
            "confidence": confidence,
            "confidence_label": confidence_label(confidence),
            "method": "user_confirmed",
            "user_confirmed": True,
            "evidence": ["Confirmed by the user during schema review."],
        }

    ignored = _validate_ignored_columns(payload.get("ignored_columns"), source_set)
    overlap = sorted(set(ignored) & set(assigned_to))
    if overlap:
        raise SchemaProfileValidationError(
            f"Mapped source column '{overlap[0]}' cannot also be ignored."
        )
    status = payload.get("status", "confirmed")
    if status not in SCHEMA_STATUSES:
        raise SchemaProfileValidationError(
            "status must be one of: confirmed, needs_review, unreviewed."
        )
    unmapped = [
        source for source in source_names if source not in assigned_to and source not in ignored
    ]
    generated_at = (
        existing_profile.get("generated_at")
        if isinstance(existing_profile, dict) and existing_profile.get("generated_at")
        else now
    )
    return {
        "version": SCHEMA_PROFILE_VERSION,
        "status": status,
        "generated_at": generated_at,
        "updated_at": now,
        "confirmed_at": now if status == "confirmed" else None,
        "mappings": mappings,
        "unmapped_columns": unmapped,
        "ignored_columns": ignored,
        "conflicts": [],
        "warnings": [],
    }


def copy_profile(profile):
    return deepcopy(profile) if isinstance(profile, dict) else None


def _validate_ignored_columns(value, source_set):
    if value is None:
        return []
    if not isinstance(value, list):
        raise SchemaProfileValidationError("ignored_columns must be a list.")
    if len(value) > MAX_IGNORED_COLUMNS:
        raise SchemaProfileValidationError(
            f"At most {MAX_IGNORED_COLUMNS} ignored columns are allowed."
        )
    ignored = _clean_string_list(value, MAX_IGNORED_COLUMNS)
    missing = [source for source in ignored if source not in source_set]
    if missing:
        raise SchemaProfileValidationError(
            f"Ignored source column '{missing[0]}' does not exist in this dataset."
        )
    return ignored


def _bounded_confidence(value):
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return 0.0


def _clean_text(value, limit):
    return str(value or "").replace("\x00", "").strip()[:limit]


def _clean_string_list(value, limit, *, max_length=MAX_COLUMN_NAME_LENGTH):
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value[:limit]:
        text = _clean_text(item, max_length)
        if text and text not in result:
            result.append(text)
    return result


def _clean_evidence(value):
    return _clean_string_list(
        value, MAX_EVIDENCE_ITEMS, max_length=MAX_EVIDENCE_LENGTH
    )


def _clean_records(value, limit):
    if not isinstance(value, list):
        return []
    cleaned = []
    for record in value[:limit]:
        if not isinstance(record, dict):
            continue
        item = {}
        for key, raw in list(record.items())[:10]:
            if isinstance(raw, list):
                item[_clean_text(key, 80)] = _clean_string_list(
                    raw, 20, max_length=MAX_EVIDENCE_LENGTH
                )
            elif isinstance(raw, (str, int, float, bool)) or raw is None:
                item[_clean_text(key, 80)] = (
                    _clean_text(raw, MAX_EVIDENCE_LENGTH)
                    if isinstance(raw, str)
                    else raw
                )
        if item:
            cleaned.append(item)
    return cleaned
