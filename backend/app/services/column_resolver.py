"""Centralized resolver from messy source column names to canonical person fields.

Canonical fields cover the alumni/person schema used by people-filter results.
Resolution order: exact match, case-insensitive match, compact-normalized match
(ignoring spaces/punctuation/case).
"""

import re

from app.services.canonical_schema import CANONICAL_FIELD_ALIASES
from app.services.email_utils import extract_email_tokens
from app.services.schema_profile import canonical_to_source

# Frontend-visible headers for alumni/person results, in display order.
PERSON_DISPLAY_HEADERS = {
    "first_name": "First Name",
    "last_name": "Last Name",
    "occupation": "Occupation",
    "employer": "Employer",
    "linkedin_url": "LinkedIn URL",
}


def resolve_canonical_column(df, canonical_field, schema_profile=None):
    """Resolve a canonical field name to an actual DataFrame column, or None."""
    mapped = resolve_canonical_columns(df, canonical_field, schema_profile=schema_profile)
    if mapped:
        return mapped[0]
    aliases = CANONICAL_FIELD_ALIASES.get(canonical_field)
    if not aliases:
        return None
    return resolve_by_aliases(df, aliases)


def resolve_canonical_columns(df, canonical_field, schema_profile=None):
    """Return every active dataset-mapped source for a canonical field.

    Multi-value mappings are preserved. For single-value callers,
    :func:`resolve_canonical_column` returns the first saved source.
    """
    profile = schema_profile
    if profile is None:
        attrs = getattr(df, "attrs", {})
        profile = attrs.get("schema_profile") if isinstance(attrs, dict) else None
    active = canonical_to_source(profile, getattr(df, "columns", []))
    return list(active.get(canonical_field) or [])


def resolve_by_aliases(df, aliases):
    for alias in aliases:
        alias_text = str(alias).strip()
        if alias_text in df.columns:
            return alias_text
    for alias in aliases:
        alias_text = str(alias).strip()
        for column in df.columns:
            if alias_text.casefold() == str(column).casefold():
                return str(column)
    normalized_aliases = {_normalize_compact(alias) for alias in aliases if _normalize_compact(alias)}
    for column in df.columns:
        if _normalize_compact(column) in normalized_aliases:
            return str(column)
    return None


def resolve_person_columns(df, schema_profile=None):
    """Map every resolvable canonical field to its actual column in df."""
    resolved = {}
    for canonical_field in CANONICAL_FIELD_ALIASES:
        column = resolve_canonical_column(
            df, canonical_field, schema_profile=schema_profile
        )
        if column:
            resolved[canonical_field] = column
    return resolved


def resolve_all_semantic_columns(semantic_key, dataset_context, *, question=""):
    """Resolve every relevant column for a semantic field in dataset order.

    Existing single-column callers keep using :func:`resolve_canonical_column`.
    The multi-column path is intentionally conservative and currently adds
    special handling for broad email questions.
    """
    column_contexts = _context_columns(dataset_context)
    actual_names = [str(column["name"]) for column in column_contexts if column.get("name") is not None]
    if not actual_names:
        return []

    explicit = _explicitly_named_column(question, actual_names, semantic_key)
    if explicit:
        return [explicit]

    schema_mapping = _context_schema_mapping(dataset_context)
    mapped = [
        source
        for source in schema_mapping.get(semantic_key, [])
        if source in actual_names
    ]
    if mapped:
        return list(dict.fromkeys(mapped))

    aliases = CANONICAL_FIELD_ALIASES.get(semantic_key, [semantic_key])
    alias_norms = {_normalize_compact(alias) for alias in aliases}
    matches = []
    for column in column_contexts:
        name = str(column.get("name") or "")
        compact = _normalize_compact(name)
        if semantic_key == "email":
            if (_is_email_column_name(compact) or compact in alias_norms) and not _is_unrelated_free_text_column(compact):
                matches.append(name)
                continue
            if not _is_unrelated_free_text_column(compact) and _samples_strongly_look_like_email(
                column.get("sample_values") or []
            ):
                matches.append(name)
        elif compact in alias_norms:
            matches.append(name)
    return list(dict.fromkeys(matches))


def _context_columns(dataset_context):
    if isinstance(dataset_context, dict):
        columns = dataset_context.get("columns")
        if isinstance(columns, list):
            return [column for column in columns if isinstance(column, dict)]
    columns = getattr(dataset_context, "columns", None)
    if columns is None:
        return []
    return [{"name": str(column), "sample_values": []} for column in columns]


def _context_schema_mapping(dataset_context):
    if not isinstance(dataset_context, dict):
        return {}
    compact = dataset_context.get("schema_mapping")
    if not isinstance(compact, dict):
        return {}
    mappings = compact.get("canonical_to_source")
    if not isinstance(mappings, dict):
        return {}
    normalized = {}
    for key, sources in mappings.items():
        if key not in CANONICAL_FIELD_ALIASES or not isinstance(sources, list):
            continue
        cleaned = [str(source) for source in sources if str(source).strip()]
        if cleaned:
            normalized[key] = cleaned
    return normalized


def _explicitly_named_column(question, actual_names, semantic_key):
    text = re.sub(r"\s+", " ", str(question or "").casefold()).strip()
    if not text:
        return None
    broad_email_scope = semantic_key == "email" and bool(
        re.search(r"\b(?:an|any|some)\s+emails?\b|\bemails?\s+in\s+(?:there|the\s+(?:file|database|sheet))\b", text)
    )
    if broad_email_scope:
        return None

    for actual in actual_names:
        actual_text = re.sub(r"\s+", " ", str(actual).casefold()).strip()
        if not actual_text or actual_text not in text:
            continue
        if re.search(
            rf"\b(?:column|field)\s+(?:named\s+)?['\"]?{re.escape(actual_text)}['\"]?\b"
            rf"|\b['\"]?{re.escape(actual_text)}['\"]?\s+(?:column|field)\b",
            text,
        ):
            return str(actual)
        # Qualified email headers such as Work Email are explicit when the
        # wording uses that full header.  "Cornell email" and "personal email"
        # remain broad concepts unless the user says column/field.
        if semantic_key == "email" and _normalize_compact(actual) not in {
            "email",
            "emailaddress",
            "cornellemail",
            "personalemail",
            "externalemail",
        }:
            return str(actual)
    return None


def _is_email_column_name(compact_name):
    if "email" not in compact_name:
        return False
    metadata_markers = {
        "emailoptout",
        "emailstatus",
        "emailconsent",
        "emailpermission",
        "emailverified",
        "emailvalid",
        "emailbounce",
        "emailpreference",
        "emailtype",
        "emaildomain",
    }
    return not any(marker in compact_name for marker in metadata_markers)


def _samples_strongly_look_like_email(samples):
    non_blank = [sample for sample in samples if str(sample or "").strip()]
    if len(non_blank) < 3:
        return False
    valid = sum(bool(extract_email_tokens(sample)) for sample in non_blank)
    return valid / len(non_blank) >= 0.8


def _is_unrelated_free_text_column(compact_name):
    return any(
        marker in compact_name
        for marker in ["note", "comment", "description", "bio", "summary", "reason", "message"]
    )


def _normalize_compact(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
