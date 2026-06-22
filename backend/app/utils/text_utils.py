"""Shared text normalization, matching, limit, and warning utilities."""

import re


MAX_LIMIT_VALUE = 500


def normalize_text(value):
    """Lowercase and collapse non-alphanumeric characters to single spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def contains_word_or_phrase(text, terms):
    """Return whether text contains any normalized word or phrase."""
    normalized = normalize_text(text)
    for term in terms:
        term_normalized = normalize_text(term)
        if not term_normalized:
            continue
        if " " in term_normalized:
            if term_normalized in normalized:
                return True
        elif re.search(rf"\b{re.escape(term_normalized)}\b", normalized):
            return True
    return False


def clamp_limit(value, default, max_value=MAX_LIMIT_VALUE):
    """Parse an integer limit, use the default when invalid, and cap it."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < 1:
        parsed = default
    return min(parsed, max_value)


def format_warning(warning):
    """Extract a displayable string from a warning dict or plain text."""
    if isinstance(warning, dict):
        return str(warning.get("message") or warning)
    return str(warning)


def dedupe_warnings(warnings):
    """Remove duplicate warnings while preserving order."""
    deduped = []
    seen = set()
    for warning in warnings or []:
        key = _warning_key(warning)
        if key not in seen:
            seen.add(key)
            deduped.append(warning)
    return deduped


def _warning_key(warning):
    if isinstance(warning, dict):
        return (
            warning.get("type"),
            warning.get("message"),
            str(warning.get("requested")),
            str(warning.get("resolved_to")),
        )
    return ("text", str(warning))
