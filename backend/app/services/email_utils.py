"""Deterministic email extraction and domain-boundary helpers.

The analysis pipeline uses these helpers for exact predicates.  They do not
perform DNS lookups and intentionally accept a narrower subset of email syntax
than RFC 5322 so arbitrary text containing ``@`` is not treated as an address.
"""

from __future__ import annotations

import re
from typing import Any


EMAIL_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9.!#$%&'*+/=?^_`{|}~-])"
    r"([A-Z0-9.!#$%&'*+/=?^_`{|}~-]+)@"
    r"((?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
    r"[A-Z]{2,63})"
    r"(?![A-Z0-9-])",
    re.IGNORECASE,
)


def extract_email_tokens(value: Any) -> list[dict[str, str]]:
    """Extract normalized email addresses from a scalar spreadsheet value.

    Common comma, semicolon, pipe, newline, and display-name forms work because
    extraction searches the whole scalar for address-shaped tokens.  Domains
    are normalized case-insensitively while the original matched token remains
    available for display.
    """
    if _is_nullish(value):
        return []
    text = str(value).strip().replace("|", " ")
    if not text:
        return []

    extracted = []
    seen = set()
    for match in EMAIL_TOKEN_RE.finditer(text):
        local_part = match.group(1)
        domain = match.group(2).rstrip(".").casefold()
        if not _valid_local_part(local_part) or not _valid_domain(domain):
            continue
        address = f"{local_part}@{domain}"
        key = address.casefold()
        if key in seen:
            continue
        seen.add(key)
        extracted.append(
            {
                "address": address,
                "local_part": local_part,
                "domain": domain,
                "original": match.group(0),
            }
        )
    return extracted


def domain_matches(domain: Any, expected: Any, *, include_subdomains: bool = True) -> bool:
    """Match a domain at DNS label boundaries, never by loose substring."""
    actual = normalize_domain(domain)
    target = normalize_domain(expected)
    if not actual or not target:
        return False
    if actual == target:
        return True
    return bool(include_subdomains and actual.endswith(f".{target}"))


def normalize_domain(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text.startswith("@"):
        text = text[1:]
    return text.rstrip(".")


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    try:
        unequal_to_self = value != value
        if isinstance(unequal_to_self, bool) and unequal_to_self:
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold() in {"", "nan", "none", "null", "nat"}


def _valid_local_part(value: str) -> bool:
    if not value or len(value) > 64:
        return False
    if value.startswith(".") or value.endswith(".") or ".." in value:
        return False
    return True


def _valid_domain(value: str) -> bool:
    if not value or len(value) > 253 or "." not in value:
        return False
    labels = value.split(".")
    return all(
        label
        and len(label) <= 63
        and not label.startswith("-")
        and not label.endswith("-")
        for label in labels
    )
