"""Centralized resolver from messy source column names to canonical person fields.

Canonical fields cover the alumni/person schema used by people-filter results.
Resolution order: exact match, case-insensitive match, compact-normalized match
(ignoring spaces/punctuation/case).
"""

import re


CANONICAL_FIELD_ALIASES = {
    "first_name": ["First Name", "first name", "first_name", "FirstName", "given name"],
    "last_name": ["Last Name", "LastName", "last name", "last_name", "surname", "family name"],
    "full_name": ["Full Name", "full_name", "First and last name", "Name", "alumni name"],
    "nickname": ["Nickname", "nickname", "preferred name"],
    "occupation": ["Occupation", "occupation", "Job Title", "job title", "Title", "title", "Role", "role", "position"],
    "employer": [
        "Employer",
        "employer",
        "Company",
        "company",
        "Organization",
        "organization",
        "organisation",
        "Firm",
        "firm",
        "workplace",
    ],
    "linkedin_url": [
        "LinkedIn URL",
        "LinkedinURL",
        "LinkedInURL",
        "LinkedIn",
        "Linkedin",
        "linkedin_url",
        "linkedin",
        "linked in",
        "linked in url",
    ],
    "email": ["Email", "Email 1", "Email1", "Email 2", "Email2", "email address", "e-mail"],
    "grad_year": ["Grad Yr", "Grad Year", "GradYear", "graduation year", "class year", "class yr"],
    "major": ["Major", "major", "degree", "field of study", "program"],
    "location": ["Location", "location"],
    "city": ["City", "city", "town"],
    "state": ["State", "state", "province", "region"],
    "country": ["Country", "country"],
}

# Frontend-visible headers for alumni/person results, in display order.
PERSON_DISPLAY_HEADERS = {
    "first_name": "First Name",
    "last_name": "Last Name",
    "occupation": "Occupation",
    "employer": "Employer",
    "linkedin_url": "LinkedIn URL",
}


def resolve_canonical_column(df, canonical_field):
    """Resolve a canonical field name to an actual DataFrame column, or None."""
    aliases = CANONICAL_FIELD_ALIASES.get(canonical_field)
    if not aliases:
        return None
    return resolve_by_aliases(df, aliases)


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


def resolve_person_columns(df):
    """Map every resolvable canonical field to its actual column in df."""
    resolved = {}
    for canonical_field in CANONICAL_FIELD_ALIASES:
        column = resolve_canonical_column(df, canonical_field)
        if column:
            resolved[canonical_field] = column
    return resolved


def _normalize_compact(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
