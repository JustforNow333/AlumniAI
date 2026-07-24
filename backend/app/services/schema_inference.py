"""Deterministic, bounded schema inference for uploaded alumni datasets."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

import pandas as pd

from app.services.canonical_schema import (
    CANONICAL_FIELD_REGISTRY,
    CANONICAL_FIELDS,
    SCHEMA_PROFILE_VERSION,
)
from app.services.email_utils import extract_email_tokens
from app.services.schema_profile import (
    MAX_SOURCE_COLUMNS_PER_FIELD,
    confidence_label,
    normalize_profile_for_storage,
    utc_timestamp,
)
from app.services.spreadsheet_service import to_json_safe
from app.utils.ai_helpers import extract_response_text, parse_json_response


MAX_INFERENCE_COLUMNS = 500
MAX_SAMPLE_VALUES = 3
MAX_SAMPLE_LENGTH = 100
MAX_MODEL_COLUMNS = 100
MAX_MODEL_OUTPUT_TOKENS = 800

FREE_TEXT_MARKERS = (
    "note",
    "notes",
    "comment",
    "comments",
    "description",
    "bio",
    "biography",
    "summary",
    "reason",
    "message",
    "memo",
)

# Layer-three rules are centralized here so institutional abbreviations are
# reviewable and independently testable. Rules are evaluated against compact
# headers (lowercase alphanumeric only).
HEADER_HEURISTICS = (
    ("first_name", re.compile(r"^(?:first|frst|fst|f)(?:name|nm)(?:\d+)?$"), 0.91, "Header uses a common first-name abbreviation."),
    ("last_name", re.compile(r"^(?:last|lst|l)(?:name|nm)(?:\d+)?$"), 0.91, "Header uses a common last-name abbreviation."),
    ("constituent_id", re.compile(r"^(?:record|constituent|person|alumni)(?:identifier|id|number|num|no|key)(?:\d+)?$"), 0.90, "Header uses institutional record-identifier terminology."),
    ("employer", re.compile(r"^(?:(?:primary|current)?(?:business|bus)(?:name|nm)|(?:current)?(?:company|employer|organization|organisation))(?:\d+)?$"), 0.86, "Header uses business or organization-name terminology."),
    ("occupation", re.compile(r"^(?:(?:primary|current)?(?:business|bus)?(?:position|title|role|occupation|jobtitle))(?:\d+)?$"), 0.88, "Header uses business-position or job-title terminology."),
    ("email", re.compile(r"^(?:(?:constituent|preferred|primary|alternate|personal|business|work|home)*email)(?:address)?(?:\d+)?$"), 0.90, "Header uses preferred or constituent email terminology."),
    ("linkedin_url", re.compile(r"^(?:linkedin)(?:profile)?(?:url|link)?(?:\d+)?$"), 0.94, "Header identifies a LinkedIn profile or URL."),
    ("grad_year", re.compile(r"^(?:(?:graduation|grad|classof|class)(?:year|yr|code|cd)?)(?:\d+)?$"), 0.82, "Header uses class or graduation-year terminology."),
    ("city", re.compile(r"^(?:(?:home|current|preferred|mailing)?city)(?:\d+)?$"), 0.86, "Header uses home or current-city terminology."),
    ("state", re.compile(r"^(?:(?:home|current|preferred|mailing)?(?:state|province|region))(?:\d+)?$"), 0.84, "Header uses state, province, or region terminology."),
    ("country", re.compile(r"^(?:(?:home|current|preferred|mailing)?country)(?:\d+)?$"), 0.84, "Header uses country terminology."),
    ("lifetime_giving", re.compile(r"^(?:(?:lifetime|lt)(?:giving|gift)(?:amount|amt|total)?|total(?:giving|gift)(?:amount|amt)?)(?:\d+)?$"), 0.90, "Header uses lifetime-gift amount terminology."),
    ("last_gift_date", re.compile(r"^(?:(?:last|latest|mostrecent)gift(?:date|dt))(?:\d+)?$"), 0.92, "Header identifies the most recent gift date."),
    ("last_contact_date", re.compile(r"^(?:(?:last|latest|mostrecent)contact(?:date|dt))(?:\d+)?$"), 0.92, "Header identifies the most recent contact date."),
    ("event_count", re.compile(r"^(?:(?:event|attendance)(?:count|cnt|number|num)|numberofevents)(?:\d+)?$"), 0.86, "Header uses event-attendance count terminology."),
    ("do_not_contact", re.compile(r"^(?:donotcontact|dnc|contactoptout|optout|nocontact)(?:flag|indicator|ind)?(?:\d+)?$"), 0.91, "Header uses contact opt-out terminology."),
    ("phone", re.compile(r"^(?:(?:preferred|primary|alternate|mobile|home|work)?(?:phone|telephone|tel))(?:number|num|no)?(?:\d+)?$"), 0.87, "Header uses phone-number terminology."),
)


MODEL_INSTRUCTIONS = """You suggest schema mappings for unresolved spreadsheet columns.
Return JSON only with this shape:
{"suggestions":[{"source_column":"...","canonical_field":"...","confidence":0.0,"evidence":"..."}]}
Choose source_column only from the provided unresolved columns and canonical_field
only from the provided canonical fields. Be conservative. Do not invent columns,
combine values, calculate results, or return code. At most one suggestion per
source column."""


def normalize_header(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def build_source_column_metadata(df, *, include_samples=True):
    metadata = []
    for column in list(df.columns)[:MAX_INFERENCE_COLUMNS]:
        series = df[column]
        non_missing = series[series.notna()]
        samples = []
        if include_samples:
            for value in non_missing.drop_duplicates().head(MAX_SAMPLE_VALUES).tolist():
                text = _sample_text(value)
                if text:
                    samples.append(_mask_sample(text))
        metadata.append(
            {
                "name": str(column),
                "type": _infer_type(series),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(non_missing.nunique(dropna=True)),
                "sample_values": samples,
            }
        )
    return to_json_safe(metadata)


def infer_schema_profile(
    df,
    *,
    existing_profile=None,
    reset_confirmed=False,
    use_model=False,
    ai_client=None,
):
    """Infer and merge a compact schema profile without modifying ``df``."""
    source_columns = [str(column) for column in list(df.columns)[:MAX_INFERENCE_COLUMNS]]
    candidates_by_source = {source: [] for source in source_columns}

    for source in source_columns:
        for candidate in _header_candidates(source):
            _add_candidate(candidates_by_source[source], candidate)
        for candidate in _sample_candidates(source, df[source]):
            _add_candidate(candidates_by_source[source], candidate)

    model_warning = None
    if use_model and ai_client is not None:
        unresolved = [
            source for source, candidates in candidates_by_source.items()
            if not candidates or max(item["confidence"] for item in candidates) < 0.90
        ]
        try:
            suggestions = _model_suggestions(df, unresolved, ai_client)
        except Exception:
            suggestions = []
            model_warning = "Optional model inference failed; deterministic suggestions were kept."
        for candidate in suggestions:
            _add_candidate(candidates_by_source[candidate["source_column"]], candidate)

    profile = _profile_from_candidates(source_columns, candidates_by_source)
    if model_warning:
        profile["warnings"].append(model_warning)

    existing = normalize_profile_for_storage(existing_profile, source_columns)
    if existing and not reset_confirmed:
        profile = _merge_confirmed_profile(profile, existing, source_columns)
    return profile


def _header_candidates(source):
    candidates = []
    source_text = str(source).strip()
    compact = normalize_header(source_text)
    for field in CANONICAL_FIELDS:
        aliases = field["aliases"]
        if source_text in aliases:
            candidates.append(
                _candidate(
                    field["key"],
                    0.99,
                    "exact_alias",
                    f"Column header exactly matched the known {field['label']} alias.",
                )
            )
            continue
        alias_compacts = {normalize_header(alias) for alias in aliases}
        if compact and compact in alias_compacts:
            candidates.append(
                _candidate(
                    field["key"],
                    0.96,
                    "normalized_alias",
                    f"Column header matched a known {field['label']} alias after ignoring case and punctuation.",
                )
            )
    for canonical_field, pattern, confidence, evidence in HEADER_HEURISTICS:
        if compact and pattern.fullmatch(compact):
            candidates.append(
                _candidate(canonical_field, confidence, "header_heuristic", evidence)
            )
    return candidates


def _sample_candidates(source, series):
    compact = normalize_header(source)
    if _is_free_text_header(compact):
        return []
    non_blank = [
        value
        for value in series.dropna().head(30).tolist()
        if str(value or "").strip()
    ]
    if not non_blank:
        return []

    candidates = []
    email_hits = sum(bool(extract_email_tokens(value)) for value in non_blank)
    if len(non_blank) >= 2 and email_hits / len(non_blank) >= 0.85:
        confidence = 0.88 if "email" in compact or "eml" in compact else 0.72
        candidates.append(
            _candidate(
                "email",
                confidence,
                "sample_inference",
                "Most sampled nonblank values are valid email addresses.",
            )
        )

    linkedin_hits = sum(
        "linkedin.com/" in str(value or "").casefold() for value in non_blank
    )
    if len(non_blank) >= 2 and linkedin_hits / len(non_blank) >= 0.80:
        candidates.append(
            _candidate(
                "linkedin_url",
                0.90 if "linkedin" in compact else 0.82,
                "sample_inference",
                "Most sampled values are LinkedIn profile URLs.",
            )
        )

    years = [_graduation_year(value) for value in non_blank]
    valid_years = [year for year in years if year is not None]
    if (
        len(non_blank) >= 2
        and len(valid_years) / len(non_blank) >= 0.85
        and any(marker in compact for marker in ("grad", "class", "year", "yr", "classcd"))
    ):
        candidates.append(
            _candidate(
                "grad_year",
                0.84,
                "sample_inference",
                "Sampled values are plausible four-digit graduation years.",
            )
        )

    if _boolean_like(non_blank) and any(
        marker in compact for marker in ("contact", "optout", "dnc", "donot")
    ):
        candidates.append(
            _candidate(
                "do_not_contact",
                0.82,
                "sample_inference",
                "Values are boolean-like and the header refers to contact preferences.",
            )
        )

    if any(marker in compact for marker in ("gift", "giving")) and _numeric_rate(non_blank) >= 0.85:
        candidates.append(
            _candidate(
                "lifetime_giving",
                0.76,
                "sample_inference",
                "Values are numeric or currency-like and the header refers to giving.",
            )
        )

    if any(marker in compact for marker in ("date", "dt")):
        date_rate = _date_rate(non_blank)
        if date_rate >= 0.80 and "contact" in compact:
            candidates.append(
                _candidate(
                    "last_contact_date",
                    0.79,
                    "sample_inference",
                    "Values are date-like and the header refers to contact.",
                )
            )
        elif date_rate >= 0.80 and "gift" in compact:
            candidates.append(
                _candidate(
                    "last_gift_date",
                    0.79,
                    "sample_inference",
                    "Values are date-like and the header refers to gifts.",
                )
            )
    return candidates


def _profile_from_candidates(source_columns, candidates_by_source):
    selected = {}
    conflicts = []
    for source in source_columns:
        candidates = sorted(
            candidates_by_source.get(source) or [],
            key=lambda item: (-item["confidence"], item["canonical_field"]),
        )
        if not candidates:
            continue
        best = candidates[0]
        competing = [
            item
            for item in candidates[1:]
            if item["canonical_field"] != best["canonical_field"]
            and item["confidence"] >= best["confidence"] - 0.03
        ]
        if competing:
            conflicts.append(
                {
                    "type": "source_column_conflict",
                    "source_column": source,
                    "candidate_fields": list(
                        dict.fromkeys(
                            [best["canonical_field"]]
                            + [item["canonical_field"] for item in competing]
                        )
                    ),
                }
            )
            continue
        selected[source] = best

    grouped = {}
    for source, candidate in selected.items():
        grouped.setdefault(candidate["canonical_field"], []).append((source, candidate))

    mappings = {}
    conflicted_sources = set()
    for canonical_field, assignments in grouped.items():
        field = CANONICAL_FIELD_REGISTRY[canonical_field]
        assignments.sort(key=lambda item: (-item[1]["confidence"], source_columns.index(item[0])))
        if field["cardinality"] == "single" and len(assignments) > 1:
            best_confidence = assignments[0][1]["confidence"]
            close = [
                item for item in assignments if item[1]["confidence"] >= best_confidence - 0.05
            ]
            if len(close) > 1:
                sources = [source for source, _candidate_item in assignments]
                conflicted_sources.update(sources)
                conflicts.append(
                    {
                        "type": "single_field_multiple_sources",
                        "canonical_field": canonical_field,
                        "source_columns": sources,
                    }
                )
                continue
            assignments = assignments[:1]
        elif field["cardinality"] == "multiple":
            assignments = assignments[:MAX_SOURCE_COLUMNS_PER_FIELD]

        sources = [source for source, _candidate_item in assignments]
        confidence = min(item["confidence"] for _source, item in assignments)
        methods = list(dict.fromkeys(item["method"] for _source, item in assignments))
        evidence = list(
            dict.fromkeys(
                item["evidence"] for _source, item in assignments if item.get("evidence")
            )
        )[:5]
        mappings[canonical_field] = {
            "source_columns": sources,
            "confidence": confidence,
            "confidence_label": confidence_label(confidence),
            "method": methods[0] if len(methods) == 1 else "combined_inference",
            "user_confirmed": False,
            "evidence": evidence,
        }

    assigned = {
        source for mapping in mappings.values() for source in mapping["source_columns"]
    }
    now = utc_timestamp()
    return {
        "version": SCHEMA_PROFILE_VERSION,
        "status": "unreviewed",
        "generated_at": now,
        "updated_at": now,
        "confirmed_at": None,
        "mappings": mappings,
        "unmapped_columns": [
            source for source in source_columns if source not in assigned
        ],
        "ignored_columns": [],
        "conflicts": conflicts,
        "warnings": [],
    }


def _merge_confirmed_profile(inferred, existing, source_columns):
    confirmed = {
        key: item
        for key, item in (existing.get("mappings") or {}).items()
        if isinstance(item, dict) and item.get("user_confirmed")
    }
    if not confirmed:
        return inferred
    confirmed_sources = {
        source for item in confirmed.values() for source in item.get("source_columns") or []
    }
    merged_mappings = {
        key: item
        for key, item in inferred["mappings"].items()
        if key not in confirmed
        and not confirmed_sources.intersection(item.get("source_columns") or [])
    }
    merged_mappings.update(confirmed)
    ignored = [
        source
        for source in existing.get("ignored_columns") or []
        if source in source_columns and source not in confirmed_sources
    ]
    assigned = {
        source for item in merged_mappings.values() for source in item.get("source_columns") or []
    }
    inferred["mappings"] = merged_mappings
    inferred["ignored_columns"] = ignored
    inferred["unmapped_columns"] = [
        source for source in source_columns if source not in assigned and source not in ignored
    ]
    inferred["conflicts"] = [
        conflict
        for conflict in inferred.get("conflicts") or []
        if conflict.get("source_column") not in confirmed_sources
        and not confirmed_sources.intersection(conflict.get("source_columns") or [])
        and conflict.get("canonical_field") not in confirmed
    ]
    if existing.get("status") == "confirmed":
        inferred["status"] = "confirmed"
        inferred["confirmed_at"] = existing.get("confirmed_at")
    return inferred


def _model_suggestions(df, unresolved, ai_client):
    unresolved_set = set(unresolved[:MAX_MODEL_COLUMNS])
    if not unresolved_set:
        return []
    column_metadata = [
        item
        for item in build_source_column_metadata(df, include_samples=True)
        if item["name"] in unresolved_set
    ]
    fields = [
        {
            "key": field["key"],
            "label": field["label"],
            "description": field["description"][:160],
            "category": field["category"],
            "cardinality": field["cardinality"],
            "expected_types": field["expected_types"],
            "aliases": field["aliases"][:12],
        }
        for field in CANONICAL_FIELDS
    ]
    response = ai_client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        instructions=MODEL_INSTRUCTIONS,
        input=json.dumps(
            {"unresolved_columns": column_metadata, "canonical_fields": fields},
            ensure_ascii=False,
        ),
        max_output_tokens=MAX_MODEL_OUTPUT_TOKENS,
        temperature=0,
        tools=[],
        text={
            "format": {
                "type": "json_schema",
                "name": "schema_mapping_suggestions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "suggestions": {
                            "type": "array",
                            "maxItems": MAX_MODEL_COLUMNS,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "source_column": {"type": "string"},
                                    "canonical_field": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "evidence": {"type": "string"},
                                },
                                "required": [
                                    "source_column",
                                    "canonical_field",
                                    "confidence",
                                    "evidence",
                                ],
                            },
                        }
                    },
                    "required": ["suggestions"],
                },
            }
        },
    )
    parsed = parse_json_response(extract_response_text(response))
    raw_suggestions = parsed.get("suggestions") if isinstance(parsed, dict) else None
    if not isinstance(raw_suggestions, list):
        return []
    suggestions = []
    seen = set()
    for raw in raw_suggestions[:MAX_MODEL_COLUMNS]:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source_column") or "")[:500]
        field = str(raw.get("canonical_field") or "")[:80]
        if source not in unresolved_set or field not in CANONICAL_FIELD_REGISTRY or source in seen:
            continue
        try:
            confidence = max(0.0, min(0.79, float(raw.get("confidence"))))
        except (TypeError, ValueError):
            confidence = 0.5
        if confidence < 0.50:
            continue
        evidence = str(raw.get("evidence") or "Suggested from compact column metadata.")[:240]
        suggestions.append(
            _candidate(field, confidence, "model_suggestion", evidence)
            | {"source_column": source}
        )
        seen.add(source)
    return suggestions


def _candidate(canonical_field, confidence, method, evidence):
    return {
        "canonical_field": canonical_field,
        "confidence": round(float(confidence), 4),
        "method": method,
        "evidence": str(evidence)[:240],
    }


def _add_candidate(candidates, candidate):
    field = candidate.get("canonical_field")
    if field not in CANONICAL_FIELD_REGISTRY:
        return
    existing = next(
        (item for item in candidates if item["canonical_field"] == field), None
    )
    if existing is None:
        candidates.append({key: value for key, value in candidate.items() if key != "source_column"})
    elif candidate["confidence"] > existing["confidence"]:
        existing.update({key: value for key, value in candidate.items() if key != "source_column"})


def _is_free_text_header(compact):
    return any(marker in compact for marker in FREE_TEXT_MARKERS)


def _sample_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value).replace("\x00", "").strip()[:MAX_SAMPLE_LENGTH]


def _mask_sample(text):
    emails = extract_email_tokens(text)
    if not emails:
        return text
    masked = text
    for token in emails:
        email = token.get("address") if isinstance(token, dict) else str(token)
        if not email or "@" not in email:
            continue
        local, domain = email.rsplit("@", 1)
        safe_local = (local[:1] + "***") if local else "***"
        original = token.get("original") if isinstance(token, dict) else email
        masked = masked.replace(original or email, f"{safe_local}@{domain}")
    return masked[:MAX_SAMPLE_LENGTH]


def _infer_type(series):
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "text"


def _graduation_year(value):
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{4}(?:\.0)?", text):
        return None
    year = int(float(text))
    current_year = datetime.now().year
    return year if 1900 <= year <= current_year + 10 else None


def _boolean_like(values):
    allowed = {"true", "false", "yes", "no", "y", "n", "1", "0"}
    normalized = [str(value).strip().casefold() for value in values]
    return bool(normalized) and sum(value in allowed for value in normalized) / len(normalized) >= 0.90


def _numeric_rate(values):
    parsed = 0
    for value in values:
        text = re.sub(r"[$,%\s]", "", str(value or ""))
        try:
            float(text)
            parsed += 1
        except (TypeError, ValueError):
            pass
    return parsed / len(values) if values else 0.0


def _date_rate(values):
    if not values:
        return 0.0
    parsed = pd.to_datetime(pd.Series(values), errors="coerce")
    return float(parsed.notna().mean())
