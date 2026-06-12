"""Query-aware multi-label classification of people rows.

Broad keyword retrieval finds *candidates*; this module decides which candidates
actually answer the user's question. Each candidate is profiled with independent
multi-label facets (employer industries, job functions, specialties) and then
classified against the query as one of:

    direct_match  — clear evidence the row answers the query; counted and displayed
    adjacent      — related business/strategy/finance language without direct
                    evidence; never counted unless the query asks for adjacent rows
    uncertain     — not enough information to decide; never counted by default
    non_match     — clearly unrelated

A keyword hit alone never makes a row a direct match. The classification label and
count_as_match flag decide inclusion; confidence is secondary and must not be used
to promote adjacent rows into the answer.

Consulting has a dedicated deterministic policy (the precision problems showed up
there first); every other industry wraps the layered taxonomy engine in
industry_matching so existing behavior is preserved. New industries can be added
by writing another policy function and registering it in _INDUSTRY_POLICIES.
"""

import re

from app.services.industry_matching import (
    known_company_match,
    match_row_to_industry,
    matched_term,
)
from app.services.industry_taxonomies import INDUSTRY_PRIORITY, get_taxonomy

CLASSIFICATION_VERSION = "multi_label_v1"

DIRECT_MATCH = "direct_match"
ADJACENT = "adjacent"
UNCERTAIN = "uncertain"
NON_MATCH = "non_match"

# direct_match must reach this confidence to count. Labels matter more than the
# number: adjacent/uncertain never count regardless of confidence.
DIRECT_MATCH_CONFIDENCE_FLOOR = 0.70

_RANK = {DIRECT_MATCH: 3, ADJACENT: 2, UNCERTAIN: 1, NON_MATCH: 0}

# Taxonomy name -> canonical employer-industry label. An employer can carry
# several labels (Spotify is technology and media).
EMPLOYER_INDUSTRY_LABELS = {
    "consulting": "consulting_professional_services",
    "banking": "financial_services",
    "finance": "financial_services",
    "venture_capital": "financial_services",
    "private_equity": "financial_services",
    "tech": "technology",
    "law": "law",
    "healthcare": "healthcare",
    "education": "education",
    "media": "media",
    "nonprofit": "nonprofit",
}

# Title phrases -> job function labels. A title can carry several functions.
JOB_FUNCTION_KEYWORDS = {
    "consulting_advisory": ["consultant", "consulting", "advisory"],
    "finance_investing": [
        "investment banking",
        "investment banker",
        "banker",
        "private equity",
        "portfolio manager",
        "trader",
        "trading",
        "equity research",
        "wealth management",
        "private wealth",
        "asset management",
        "hedge fund",
        "corporate finance",
        "capital markets",
        "sales and trading",
        "financial analyst",
        "financial advisor",
        "wealth advisor",
        "investment advisor",
        "investment analyst",
        "investment associate",
    ],
    "internal_strategy": [
        "strategy",
        "strategic planning",
        "chief of staff",
        "corporate development",
    ],
    "product": [
        "product manager",
        "product management",
        "product lead",
        "product owner",
        "head of product",
        "chief product officer",
        "product strategy",
    ],
    "operations": ["operations", "business operations"],
    "legal": ["attorney", "lawyer", "counsel", "law clerk", "paralegal", "judicial", "legal"],
    "engineering": [
        "software engineer",
        "engineer",
        "developer",
        "programmer",
        "devops",
        "site reliability",
        "cto",
    ],
    "data_analytics": ["data scientist", "data analyst", "data engineer", "analytics", "data science"],
    "sales_business_development": ["sales", "business development", "account executive", "partnerships"],
    "marketing_growth": ["marketing", "growth", "brand", "communications"],
    "education_teaching": ["teacher", "professor", "lecturer", "instructor", "educator"],
}

# Title phrases -> specialty/domain labels.
SPECIALTY_KEYWORDS = {
    "management_consulting": ["management consultant", "management consulting"],
    "strategy_consulting": ["strategy consultant", "strategy consulting"],
    "technology_consulting": ["technology consultant", "technology consulting", "it consultant"],
    "implementation": ["implementation consultant", "implementation"],
    "risk_consulting": ["risk consulting", "risk consultant", "risk advisory"],
    "transaction_advisory": ["transaction advisory", "transaction services"],
    "deal_advisory": ["deal advisory"],
    "valuation": ["valuation"],
    "restructuring": ["restructuring"],
    "financial_advisory": ["financial advisory"],
    "corporate_strategy": ["corporate strategy", "head of strategy", "director of strategy"],
    "product_strategy": ["product strategy"],
    "business_operations": ["business operations", "strategy & operations", "strategy and operations"],
    "investment_banking": ["investment banking", "investment banker"],
    "private_equity": ["private equity"],
    "asset_management": ["asset management"],
    "wealth_management": ["wealth management", "private wealth"],
    "legal": ["attorney", "lawyer", "law clerk", "paralegal", "counsel", "judicial"],
    "risk": ["risk"],
}

# Titles that clearly indicate professional/client advisory work. With a
# professional-services employer they are direct consulting matches; without one
# they still carry strong consulting signal.
CLIENT_ADVISORY_PHRASES = [
    "transaction advisory",
    "deal advisory",
    "valuation advisory",
    "restructuring advisory",
    "risk advisory",
    "technology advisory",
    "strategy advisory",
    "m&a advisory",
    "transaction services",
]

# Finance roles that are NOT consulting on their own (finance alone is not
# consulting; advisory phrases above are handled before this list applies).
FINANCE_NON_CONSULTING_PHRASES = [
    "investment banking",
    "investment banker",
    "banker",
    "private equity",
    "portfolio manager",
    "trader",
    "equity research",
    "wealth management",
    "private wealth",
    "hedge fund",
    "capital markets",
    "corporate finance",
    "sales and trading",
    "asset management",
    "financial analyst",
    "financial advisor",
    "wealth advisor",
]

# Business language that makes a row consulting-adjacent, never direct.
CONSULTING_ADJACENT_WORDS = ["strategy", "operations", "management", "manager", "transaction", "deal", "chief of staff"]

# Specialty labels that mark a row as finance-related for intersection queries
# like "finance consulting" (consulting + finance overlap is real).
INDUSTRY_RELATED_SPECIALTIES = {
    "finance": {
        "risk",
        "risk_consulting",
        "transaction_advisory",
        "deal_advisory",
        "valuation",
        "restructuring",
        "financial_advisory",
        "investment_banking",
        "private_equity",
        "asset_management",
        "wealth_management",
    },
    "banking": {"investment_banking", "transaction_advisory", "deal_advisory", "restructuring"},
    "tech": {"technology_consulting"},
}


def query_spec_from_filter(filter_spec, default_industry=None):
    """Build a normalized query spec from a people_filter spec produced by
    industry_taxonomies.classify_people_question (or the LLM intent)."""
    filter_spec = filter_spec if isinstance(filter_spec, dict) else {}
    industries = [str(item) for item in filter_spec.get("industries") or [] if str(item).strip()]
    if not industries:
        primary = filter_spec.get("industry") or default_industry
        industries = [str(primary)] if primary else []
    return {
        "industries": industries,
        "required_industries": [str(item) for item in filter_spec.get("required_industries") or [] if str(item).strip()],
        "include_functions": [str(item) for item in filter_spec.get("include_functions") or [] if str(item).strip()],
        "include_adjacent": bool(filter_spec.get("include_adjacent")),
    }


def profile_row(occupation, employer, descriptor_text=""):
    """Multi-label profile of one row, independent of any query."""
    occupation_text = _clean(occupation)
    employer_text = _clean(employer)
    return {
        "employer_industry": classify_employer_industries(employer_text, descriptor_text),
        "job_function": classify_job_functions(occupation_text),
        "specialties": classify_specialties(occupation_text),
    }


def classify_employer_industries(employer, descriptor_text=""):
    employer_text = _clean(employer)
    if not employer_text:
        return ["unknown"]
    labels = []
    for taxonomy_name, label in EMPLOYER_INDUSTRY_LABELS.items():
        taxonomy = get_taxonomy(taxonomy_name)
        if not taxonomy:
            continue
        if known_company_match(employer_text, taxonomy.get("known_companies") or []) or matched_term(
            employer_text, taxonomy.get("employer_keywords") or []
        ):
            if label not in labels:
                labels.append(label)
    return labels or ["unknown"]


def classify_job_functions(occupation):
    occupation_text = _clean(occupation)
    if not occupation_text:
        return ["unknown"]
    labels = []
    for label, phrases in JOB_FUNCTION_KEYWORDS.items():
        if matched_term(occupation_text, phrases):
            labels.append(label)
    # "Strategy Consultant" is consulting, not internal strategy: the strategy
    # wording belongs to the consulting role itself.
    if "consulting_advisory" in labels and "internal_strategy" in labels:
        labels.remove("internal_strategy")
    return labels or ["other"]


def classify_specialties(occupation):
    occupation_text = _clean(occupation)
    if not occupation_text:
        return ["unknown"]
    labels = [label for label, phrases in SPECIALTY_KEYWORDS.items() if matched_term(occupation_text, phrases)]
    return labels or ["unknown"]


def classify_candidate(occupation, employer, query_spec, descriptor_text="", model_classifier=None, row_id=None):
    """Classify one candidate row against the user's query.

    Returns the structured multi-label result:
        {"row_id", "classification", "count_as_match", "confidence",
         "employer_industry", "job_function", "specialties", "internal_reason"}
    """
    query_spec = query_spec if isinstance(query_spec, dict) else {}
    occupation_text = _clean(occupation)
    employer_text = _clean(employer)
    profile = profile_row(occupation_text, employer_text, descriptor_text)

    industries = [item for item in query_spec.get("industries") or [] if item]
    classification, confidence, reason = NON_MATCH, 0.0, "No target industry was specified for this query."

    for industry in industries:
        outcome = _classify_for_industry(
            industry, occupation_text, employer_text, profile, descriptor_text, model_classifier
        )
        if _RANK[outcome[0]] > _RANK[classification] or (outcome[0] == classification and outcome[1] > confidence):
            classification, confidence, reason = outcome

    # Union with explicitly requested job functions ("consulting or strategy"):
    # the query, not just the row, decides what counts.
    requested_functions = set(query_spec.get("include_functions") or [])
    if classification != DIRECT_MATCH and requested_functions.intersection(profile["job_function"]):
        matched = sorted(requested_functions.intersection(profile["job_function"]))
        classification, confidence = DIRECT_MATCH, max(confidence, 0.85)
        reason = f"Job function matches the requested function(s): {', '.join(matched)}."

    # Intersection queries ("finance consulting"): a direct match must also show
    # evidence for every required industry.
    if classification == DIRECT_MATCH:
        for required in query_spec.get("required_industries") or []:
            if not _has_industry_signal(occupation_text, employer_text, profile, required):
                classification, confidence = ADJACENT, min(confidence, 0.6)
                reason = f"Direct match for the primary criteria but no {required} evidence; the query requires the {required} intersection."
                break

    count_as_match = classification == DIRECT_MATCH and confidence >= DIRECT_MATCH_CONFIDENCE_FLOOR
    if classification == ADJACENT and query_spec.get("include_adjacent"):
        count_as_match = True

    result = {
        "classification": classification,
        "count_as_match": bool(count_as_match),
        "confidence": round(float(confidence), 2),
        "employer_industry": list(profile["employer_industry"]),
        "job_function": list(profile["job_function"]),
        "specialties": list(profile["specialties"]),
        "internal_reason": str(reason),
    }
    if row_id is not None:
        result["row_id"] = row_id
    return result


def _classify_for_industry(industry, occupation, employer, profile, descriptor_text, model_classifier):
    policy = _INDUSTRY_POLICIES.get(_taxonomy_key(industry))
    if policy:
        outcome = policy(occupation, employer, profile)
        # The model is consulted only for rows the deterministic rules could not
        # decide — never to promote adjacent/non-match rows.
        if outcome[0] == UNCERTAIN and model_classifier:
            modeled = _model_outcome(industry, occupation, employer, model_classifier)
            if modeled:
                return modeled
        return outcome
    return _classify_with_taxonomy(industry, occupation, employer, descriptor_text, model_classifier)


def _classify_with_taxonomy(industry, occupation, employer, descriptor_text, model_classifier):
    """Default policy: wrap the layered taxonomy engine for industries that do
    not yet have a dedicated policy."""
    taxonomy = get_taxonomy(industry)
    if not taxonomy:
        return NON_MATCH, 0.0, f"Unknown industry taxonomy: {industry}."
    match = match_row_to_industry(
        occupation, employer, taxonomy, descriptor_text=descriptor_text, model_classifier=model_classifier
    )
    status = match.get("status")
    if status == "confirmed":
        return DIRECT_MATCH, max(float(match.get("confidence", 0.8)), DIRECT_MATCH_CONFIDENCE_FLOOR), match.get("internal_reason", "")
    if status == "uncertain":
        return UNCERTAIN, float(match.get("confidence", 0.45)), match.get("internal_reason", "")
    return NON_MATCH, float(match.get("confidence", 0.0)), match.get("internal_reason", "")


def _classify_for_consulting(occupation, employer, profile):
    """Strict consulting policy: consulting/advisory/professional-services
    evidence is required; strategy/management/operations language alone is only
    adjacent; finance and legal roles without advisory context never match."""
    employer_is_ps = "consulting_professional_services" in profile["employer_industry"]
    employer_unknown = profile["employer_industry"] == ["unknown"]
    functions = set(profile["job_function"])

    if not occupation and not employer:
        return UNCERTAIN, 0.2, "No occupation or employer information is available."

    # 1. Explicit consulting role in the title confirms on its own.
    explicit = matched_term(occupation, ["consultant", "consulting"])
    if explicit:
        return DIRECT_MATCH, 0.95, f"Title explicitly contains a consulting role ('{explicit}')."

    # 2. Clear professional/client advisory work in the title.
    advisory = matched_term(occupation, CLIENT_ADVISORY_PHRASES)
    if advisory:
        if employer_is_ps:
            return DIRECT_MATCH, 0.9, f"Client advisory title ('{advisory}') at a professional-services firm ({employer})."
        if employer_unknown:
            return DIRECT_MATCH, 0.75, f"Client advisory title ('{advisory}'); employer is unknown but the title indicates client-service advisory work."
        if "financial_services" in profile["employer_industry"]:
            return (
                ADJACENT,
                0.6,
                f"Advisory title ('{advisory}') inside a financial-services employer ({employer}); likely finance, not consulting.",
            )
        return UNCERTAIN, 0.55, f"Advisory title ('{advisory}') at an employer with no professional-services evidence ({employer})."

    # 2b. "Financial Advisory" is consulting only at a professional-services firm;
    # elsewhere it is a finance role.
    if matched_term(occupation, ["financial advisory"]):
        if employer_is_ps:
            return DIRECT_MATCH, 0.85, f"Financial advisory at a professional-services firm ({employer})."
        return ADJACENT, 0.6, "Financial advisory without a professional-services employer; likely finance, not consulting."

    # 3. Clear non-consulting domains.
    if "legal" in functions:
        return NON_MATCH, 0.9, "Legal role (attorney/clerk/counsel) with no consulting or advisory evidence."
    finance_phrase = matched_term(occupation, FINANCE_NON_CONSULTING_PHRASES)
    if finance_phrase and not employer_is_ps:
        return NON_MATCH, 0.85, f"Finance role ('{finance_phrase}') with no consulting or advisory context; finance alone is not consulting."

    # 4. Recognized consulting/professional-services employer with a plausibly
    # client-serving title.
    if employer_is_ps:
        if functions.intersection({"engineering", "finance_investing", "education_teaching"}):
            return (
                UNCERTAIN,
                0.55,
                f"Employer is a professional-services firm ({employer}) but the title suggests a non-consulting function.",
            )
        return (
            DIRECT_MATCH,
            0.8,
            f"Recognized consulting/professional-services firm ({employer}) with a plausibly client-serving title ('{occupation or 'unspecified'}').",
        )

    # 5. Bare advisor/adviser titles outside professional services are ambiguous.
    if matched_term(occupation, ["advisor", "adviser"]):
        return UNCERTAIN, 0.5, "Title says advisor but neither the role nor the employer shows consulting/professional-services context."

    # 6. Business/strategy/operations language without consulting context is
    # adjacent, never direct.
    adjacent_word = matched_term(occupation, CONSULTING_ADJACENT_WORDS)
    if functions.intersection({"internal_strategy", "product", "operations"}) or adjacent_word:
        return (
            ADJACENT,
            0.6,
            "Business/strategy/operations language without consulting, advisory, or professional-services evidence (internal role at a non-consulting employer).",
        )

    # 7. Ambiguous employer wording (e.g. "... Partners") with no title signal.
    taxonomy = get_taxonomy("consulting") or {}
    if matched_term(employer, taxonomy.get("ambiguous_keywords") or []):
        return UNCERTAIN, 0.45, f"Employer wording is ambiguous for consulting: {employer}."

    return NON_MATCH, 0.2, "No consulting, advisory, or professional-services evidence in the title or employer."


_INDUSTRY_POLICIES = {
    "consulting": _classify_for_consulting,
}


def _has_industry_signal(occupation, employer, profile, industry):
    """True when the row shows evidence for `industry` (used for intersection
    queries; multi-label, so consulting rows can also be finance-related)."""
    key = _taxonomy_key(industry)
    label = EMPLOYER_INDUSTRY_LABELS.get(key)
    if label and label in profile["employer_industry"]:
        return True
    related = INDUSTRY_RELATED_SPECIALTIES.get(key) or set()
    if related.intersection(profile["specialties"]):
        return True
    taxonomy = get_taxonomy(key)
    if taxonomy:
        return match_row_to_industry(occupation, employer, taxonomy)["status"] == "confirmed"
    return False


def _model_outcome(industry, occupation, employer, model_classifier):
    taxonomy = get_taxonomy(industry)
    if not taxonomy or not str(employer or "").strip():
        return None
    try:
        outcome = model_classifier(employer, occupation, taxonomy)
    except Exception:
        return None
    if not isinstance(outcome, dict):
        return None
    try:
        confidence = float(outcome.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    threshold = float(taxonomy.get("confidence_threshold", 0.8))
    belongs = bool(outcome.get("belongs_to_industry"))
    reason = str(outcome.get("reason") or "Model classification.")
    if belongs and confidence >= threshold:
        return DIRECT_MATCH, confidence, f"Model classified the employer as {industry}: {reason}"
    if not belongs and confidence >= threshold:
        return NON_MATCH, confidence, f"Model classified the employer as non-{industry}: {reason}"
    return UNCERTAIN, confidence, f"Model was not confident about the employer: {reason}"


def _taxonomy_key(industry):
    taxonomy = get_taxonomy(industry)
    return taxonomy["industry"] if taxonomy else str(industry or "")


def _clean(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)
