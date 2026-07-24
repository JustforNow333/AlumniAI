"""Central registry for AlumniAI's dataset-level canonical schema.

The registry is intentionally presentation-safe and is the single source for
both schema-onboarding choices and the legacy column resolver's aliases.
"""

from __future__ import annotations

from copy import deepcopy


SCHEMA_PROFILE_VERSION = "schema_mapping_v1"
SCHEMA_STATUSES = {"unreviewed", "needs_review", "confirmed"}
MULTI_VALUE_FIELDS = {"email", "phone"}


def _field(
    key,
    label,
    description,
    category,
    *,
    cardinality="single",
    expected_types=("text",),
    aliases=(),
):
    return {
        "key": key,
        "label": label,
        "description": description,
        "category": category,
        "cardinality": cardinality,
        "expected_types": list(expected_types),
        "aliases": list(dict.fromkeys([label, *aliases])),
    }


CANONICAL_FIELDS = (
    _field(
        "constituent_id",
        "Constituent ID",
        "A stable institutional identifier for the alumnus or constituent.",
        "Identity",
        aliases=("Record ID", "Record Identifier", "Constituent Number", "ID"),
    ),
    _field(
        "first_name",
        "First Name",
        "The alumnus's given or first name.",
        "Identity",
        aliases=("first name", "first_name", "FirstName", "Given Name"),
    ),
    _field(
        "last_name",
        "Last Name",
        "The alumnus's family or last name.",
        "Identity",
        aliases=("last name", "last_name", "LastName", "Surname", "Family Name"),
    ),
    _field(
        "full_name",
        "Full Name",
        "The alumnus's complete display name in one field.",
        "Identity",
        aliases=("full_name", "Name", "First and Last Name", "Alumni Name"),
    ),
    _field(
        "nickname",
        "Nickname",
        "A preferred, familiar, or nickname.",
        "Identity",
        aliases=("Preferred Name", "Familiar Name"),
    ),
    _field(
        "grad_year",
        "Graduation Year",
        "The class or graduation year.",
        "Education",
        expected_types=("integer", "text"),
        aliases=("Grad Yr", "Grad Year", "GradYear", "Class Year", "Class Yr", "Year Graduated"),
    ),
    _field(
        "major",
        "Major",
        "The alumnus's major, program, or field of study.",
        "Education",
        aliases=("Field of Study", "Program", "Academic Major"),
    ),
    _field(
        "degree",
        "Degree",
        "The degree or credential earned.",
        "Education",
        aliases=("Degree Name", "Credential", "Degree Type"),
    ),
    _field(
        "school",
        "School",
        "The school, college, faculty, or institution attended.",
        "Education",
        aliases=("College", "Faculty", "Institution", "School Name"),
    ),
    _field(
        "occupation",
        "Occupation",
        "The alumnus's current job title, role, or position.",
        "Employment",
        aliases=("Job Title", "Title", "Role", "Position", "Profession"),
    ),
    _field(
        "employer",
        "Employer",
        "The alumnus's current employer or organization.",
        "Employment",
        aliases=(
            "Company",
            "Organization",
            "Organisation",
            "Business Name",
            "Firm",
            "Workplace",
        ),
    ),
    _field(
        "industry",
        "Industry",
        "The employer or role's industry or sector.",
        "Employment",
        aliases=("Sector", "Business Sector", "Employer Industry"),
    ),
    _field(
        "email",
        "Email",
        "An email address associated with the alumnus.",
        "Contact",
        cardinality="multiple",
        aliases=(
            "Email Address",
            "E-mail",
            "Email 1",
            "Email1",
            "Email 2",
            "Email2",
            "Preferred Email",
            "Personal Email",
            "Work Email",
        ),
    ),
    _field(
        "phone",
        "Phone",
        "A phone or mobile number associated with the alumnus.",
        "Contact",
        cardinality="multiple",
        aliases=("Phone Number", "Mobile", "Mobile Phone", "Telephone", "Home Phone", "Work Phone"),
    ),
    _field(
        "linkedin_url",
        "LinkedIn URL",
        "A URL for the alumnus's LinkedIn profile.",
        "Contact",
        aliases=(
            "LinkedinURL",
            "LinkedInURL",
            "LinkedIn",
            "Linkedin",
            "linkedin_url",
            "Linked In",
            "Linked In URL",
        ),
    ),
    _field(
        "location",
        "Location",
        "A general current geographic location.",
        "Geography",
        aliases=("Current Location", "Address Location"),
    ),
    _field(
        "city",
        "City",
        "The alumnus's home or current city.",
        "Geography",
        aliases=("Town", "Home City", "Current City"),
    ),
    _field(
        "state",
        "State",
        "The alumnus's state, province, or region.",
        "Geography",
        aliases=("Province", "Region", "Home State", "Current State"),
    ),
    _field(
        "country",
        "Country",
        "The alumnus's country.",
        "Geography",
        aliases=("Nation", "Home Country", "Current Country"),
    ),
    _field(
        "lifetime_giving",
        "Lifetime Giving",
        "The total lifetime gift amount attributed to the alumnus.",
        "Engagement and development",
        expected_types=("number", "currency"),
        aliases=("Lifetime Gift Amount", "Lifetime Gifts", "Total Giving", "Lifetime Giving Amount"),
    ),
    _field(
        "last_gift_date",
        "Last Gift Date",
        "The date of the alumnus's most recent gift.",
        "Engagement and development",
        expected_types=("date", "text"),
        aliases=("Latest Gift Date", "Most Recent Gift Date"),
    ),
    _field(
        "last_contact_date",
        "Last Contact Date",
        "The date of the most recent recorded contact.",
        "Engagement and development",
        expected_types=("date", "text"),
        aliases=("Latest Contact Date", "Most Recent Contact Date"),
    ),
    _field(
        "event_count",
        "Event Count",
        "The number of events attended or recorded.",
        "Engagement and development",
        expected_types=("integer", "number"),
        aliases=("Events Attended", "Attendance Count", "Number of Events"),
    ),
    _field(
        "do_not_contact",
        "Do Not Contact",
        "Whether the alumnus has opted out of contact.",
        "Engagement and development",
        expected_types=("boolean", "text"),
        aliases=("DNC", "Contact Opt Out", "Opt Out", "No Contact"),
    ),
    _field(
        "relationship_manager",
        "Relationship Manager",
        "The staff member assigned to manage the constituent relationship.",
        "Engagement and development",
        aliases=(
            "Assigned Relationship Manager",
            "Relationship Manager Name",
            "Prospect Manager",
            "Portfolio Manager",
        ),
    ),
    _field(
        "updated_at",
        "Updated At",
        "The date when the source record was last updated.",
        "Engagement and development",
        expected_types=("date", "datetime", "text"),
        aliases=("Updated Date", "Last Updated", "Modified At", "Modified Date"),
    ),
    _field(
        "created_at",
        "Created At",
        "The date when the source record was created.",
        "Engagement and development",
        expected_types=("date", "datetime", "text"),
        aliases=("Created Date", "Record Created", "Creation Date"),
    ),
)

CANONICAL_FIELD_REGISTRY = {field["key"]: field for field in CANONICAL_FIELDS}
CANONICAL_FIELD_ALIASES = {
    key: list(field["aliases"]) for key, field in CANONICAL_FIELD_REGISTRY.items()
}


def canonical_field_definitions():
    """Return a copy suitable for API responses and compact model prompts."""
    return deepcopy(list(CANONICAL_FIELDS))


def get_canonical_field(key):
    field = CANONICAL_FIELD_REGISTRY.get(str(key or ""))
    return deepcopy(field) if field else None


def validate_canonical_registry():
    """Raise ``ValueError`` if a developer introduces an invalid definition."""
    keys = []
    for field in CANONICAL_FIELDS:
        key = field.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("Every canonical field requires a non-empty key.")
        keys.append(key)
        if field.get("cardinality") not in {"single", "multiple"}:
            raise ValueError(f"Invalid cardinality for canonical field '{key}'.")
        if not isinstance(field.get("aliases"), list) or not field["aliases"]:
            raise ValueError(f"Canonical field '{key}' requires aliases.")
        if not isinstance(field.get("expected_types"), list) or not field["expected_types"]:
            raise ValueError(f"Canonical field '{key}' requires expected types.")
    if len(keys) != len(set(keys)):
        raise ValueError("Canonical field keys must be unique.")
    return True


validate_canonical_registry()
