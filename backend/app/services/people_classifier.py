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
    "investment_banking": "financial_services",
    "finance": "financial_services",
    "venture_capital": "financial_services",
    "private_equity": "financial_services",
    "tech": "technology",
    "marketing": "marketing",
    "operations": "operations",
    "law": "law",
    "government_legal": "government_legal",
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
        "m&a analyst",
        "mergers and acquisitions",
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
    "banking": [
        "corporate banking",
        "commercial banking",
        "credit analyst",
        "capital markets",
        "sales and trading",
        "equity research",
    ],
    "investment_banking": [
        "investment banking analyst",
        "investment banking associate",
        "investment banking",
        "investment banker",
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
    "operations": [
        "operations manager",
        "business operations",
        "strategy and operations",
        "strategy & operations",
        "supply chain",
        "logistics",
        "revops",
        "revenue operations",
        "sales operations",
        "people operations",
        "clinical operations",
        "program operations",
        "operations analyst",
        "operations associate",
        "director of operations",
        "head of operations",
        "chief operating officer",
        "coo",
    ],
    "legal": ["attorney", "lawyer", "counsel", "law clerk", "paralegal", "judicial", "legal"],
    "engineering": [
        "software engineer",
        "software engineering",
        "engineer",
        "founding engineer",
        "forward deployed engineer",
        "research engineer",
        "developer",
        "programmer",
        "devops",
        "site reliability",
        "infrastructure",
        "technical lead",
        "cto",
    ],
    "data_analytics": [
        "data scientist",
        "data analyst",
        "data engineer",
        "database",
        "database management systems",
        "analytics",
        "analytics engineering",
        "data science",
    ],
    "sales_business_development": ["sales", "business development", "account executive", "partnerships"],
    "marketing": [
        "marketing manager",
        "marketing analyst",
        "product marketing",
        "brand manager",
        "advertising",
        "performance marketing",
        "demand generation",
        "lifecycle marketing",
        "seo",
        "sem",
        "content marketing",
        "digital marketing",
        "marketing coordinator",
        "marketing director",
        "head of marketing",
        "chief marketing officer",
        "cmo",
    ],
    "marketing_growth": ["growth marketing", "growth"],
    "communications_pr": ["communications manager", "public relations", "pr manager"],
    "government_policy": [
        "policy analyst",
        "legislative aide",
        "government analyst",
        "public policy analyst",
        "city planner",
        "federal analyst",
        "campaign staff",
    ],
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
    "m_and_a": ["m&a", "mergers and acquisitions"],
    "capital_markets": ["capital markets"],
    "private_equity": ["private equity"],
    "asset_management": ["asset management"],
    "wealth_management": ["wealth management", "private wealth"],
    "legal": ["attorney", "lawyer", "law clerk", "paralegal", "counsel", "judicial"],
    "risk": ["risk"],
    "marketing": ["marketing", "brand manager", "advertising", "demand generation", "product marketing"],
    "growth": ["growth marketing", "head of growth"],
    "operations": ["operations", "business operations", "supply chain", "logistics", "revenue operations"],
    "government_policy": ["policy analyst", "legislative", "government analyst", "public policy"],
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

TECHNICAL_TITLE_PHRASES = [
    "software engineer",
    "software engineering",
    "swe",
    "software developer",
    "cloud software engineer",
    "staff software engineer",
    "senior software engineer",
    "backend engineer",
    "front end engineer",
    "frontend engineer",
    "full stack engineer",
    "founding engineer",
    "founder software engineer",
    "forward deployed engineer",
    "research engineer",
    "ai research engineer",
    "data scientist",
    "data engineer",
    "database engineer",
    "database administrator",
    "database management systems",
    "machine learning engineer",
    "ml engineer",
    "ai engineer",
    "cloud engineer",
    "cloud support engineer",
    "security engineer",
    "cybersecurity analyst",
    "site reliability engineer",
    "devops engineer",
    "systems engineer",
    "solutions engineer",
    "client solutions engineer",
    "blockchain engineer",
    "ux engineer",
    "technical program manager",
    "technical product manager",
    "technical lead",
    "technical services",
    "it director",
    "cto",
    "chief technology officer",
    "data platform analyst",
    "analytics engineer",
    "analytics engineering",
    "infrastructure engineer",
    "product engineer",
    "security analyst",
    "quantitative developer",
    "seo product analyst",
]

WEAK_TECH_TITLE_PHRASES = [
    "product manager",
    "product management",
    "product lead",
    "product owner",
    "product strategy",
    "analytics manager",
    "decision analytics",
    "growth analytics",
    "sales analytics",
    "client insights",
    "business systems analyst",
    "systems analyst",
    "revops",
    "revenue operations",
    "customer success",
    "customer operations",
    "professional services",
    "implementation",
    "technology consultant",
    "technical consultant",
    "project management",
    "head of growth",
    "growth",
    "digital",
    "platform",
    "product",
]

STRONG_TECH_EMPLOYER_KEYWORDS = [
    "technologies",
    "technology",
    "software",
    "ai",
    "artificial intelligence",
    "cloud",
    "cybersecurity",
    "fintech",
    "blockchain",
    "crypto",
    "saas",
    "semiconductor",
    "satellite",
    "space technology",
    "data platform",
    "developer tools",
    "analytics",
    "analytics platform",
]

INVESTMENT_BANKING_TITLE_PHRASES = [
    "investment banking analyst",
    "investment banking associate",
    "investment banking",
    "investment banker",
]

INVESTMENT_BANKING_CONTEXT_PHRASES = [
]

BANKING_TITLE_PHRASES = [
    "banker",
    "corporate banking",
    "commercial banking",
    "credit analyst",
    "capital markets",
    "sales and trading",
    "wealth management",
    "private wealth",
    "strategic advisory",
    "restructuring",
]

FINANCE_TITLE_PHRASES = [
    "financial analyst",
    "finance",
    "portfolio manager",
    "asset management",
    "wealth management",
    "hedge fund",
    "trader",
    "trading",
    "quant",
    "quantitative researcher",
    "investment analyst",
    "investment associate",
    "research analyst",
    "equity research",
    "investment management",
    "investor",
    "private equity",
    "venture capital",
    "quantitative developer",
    "risk analyst",
]

MARKETING_DIRECT_PHRASES = [
    "marketing manager",
    "marketing analyst",
    "growth marketing analyst",
    "growth marketing manager",
    "account strategist",
    "ads account strategist",
    "media planner",
    "brand strategist",
    "product marketing manager",
    "brand manager",
    "artist marketing manager",
    "communications manager",
    "advertising strategist",
    "performance marketing analyst",
    "demand generation manager",
    "lifecycle marketing manager",
    "seo manager",
    "sem manager",
    "content marketing manager",
    "digital marketing manager",
    "digital marketing associate",
    "digital marketing",
    "marketing coordinator",
    "marketing director",
    "head of marketing",
    "chief marketing officer",
    "cmo",
    "consumer insights analyst",
]

MARKETING_ADJACENT_PHRASES = [
    "head of growth",
    "growth",
    "strategy",
    "community",
    "partnerships",
    "sales",
    "product",
    "business development",
    "customer success",
    "content",
    "brand",
    "communications",
]

OPERATIONS_DIRECT_PHRASES = [
    "operations manager",
    "business operations",
    "strategy and operations",
    "strategy & operations",
    "marketplace operations analyst",
    "supply chain analyst",
    "supply chain manager",
    "logistics manager",
    "sales operations",
    "people operations",
    "clinical operations",
    "program operations",
    "operations analyst",
    "operations associate",
    "operations leadership associate",
    "director of operations",
    "head of operations",
    "manufacturing engineer",
    "logistics analyst",
    "revenue management analyst",
    "chief operating officer",
    "coo",
]

OPERATIONS_ADJACENT_PHRASES = [
    "strategy",
    "business analyst",
    "business",
    "management",
    "program manager",
    "project manager",
    "product operations",
    "customer success",
    "general manager",
    "revops",
    "revenue operations",
    "analyst",
    "associate",
]

GOVERNMENT_LEGAL_TITLE_PHRASES = [
    "attorney",
    "lawyer",
    "legal counsel",
    "associate attorney",
    "law clerk",
    "judicial clerk",
    "paralegal",
    "counsel",
    "litigation associate",
    "corporate counsel",
    "legal assistant",
    "policy analyst",
    "legislative aide",
    "government analyst",
    "public policy analyst",
    "city planner",
    "federal analyst",
]


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
        "excluded_industries": [str(item) for item in filter_spec.get("excluded_industries") or [] if str(item).strip()],
        "excluded_functions": [str(item) for item in filter_spec.get("excluded_functions") or [] if str(item).strip()],
        "include_functions": [str(item) for item in filter_spec.get("include_functions") or [] if str(item).strip()],
        "include_adjacent": bool(filter_spec.get("include_adjacent")),
        "query_scope": str(filter_spec.get("query_scope") or "industry"),
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
    selected_industry = ""

    for industry in industries:
        outcome = _classify_for_industry(
            industry, occupation_text, employer_text, profile, descriptor_text, model_classifier, query_spec
        )
        if _RANK[outcome[0]] > _RANK[classification] or (outcome[0] == classification and outcome[1] > confidence):
            classification, confidence, reason = outcome
            selected_industry = industry

    # Union with explicitly requested job functions ("consulting or strategy"):
    # the query, not just the row, decides what counts.
    requested_functions = set(query_spec.get("include_functions") or [])
    if classification != DIRECT_MATCH and requested_functions.intersection(profile["job_function"]):
        matched_functions = requested_functions.intersection(profile["job_function"])
        if "marketing_growth" in matched_functions and "technology" in profile["employer_industry"]:
            classification, confidence = NON_MATCH, max(confidence, 0.85)
            reason = "Growth role at a technology employer is not promoted by broad marketing/growth function matching."
        else:
            matched = sorted(matched_functions)
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

    if classification == DIRECT_MATCH:
        for excluded in query_spec.get("excluded_industries") or []:
            excluded_outcome = _classify_for_industry(
                excluded,
                occupation_text,
                employer_text,
                profile,
                descriptor_text,
                None,
                {"industries": [excluded], "query_scope": "industry"},
            )
            if excluded_outcome[0] == DIRECT_MATCH:
                classification, confidence = NON_MATCH, 0.9
                reason = f"Excluded by user-requested excluded industry '{excluded}': {excluded_outcome[2]}"
                break

    if classification == DIRECT_MATCH:
        excluded_functions = set(query_spec.get("excluded_functions") or [])
        if excluded_functions.intersection(profile["job_function"]):
            matched = sorted(excluded_functions.intersection(profile["job_function"]))
            classification, confidence = NON_MATCH, 0.9
            reason = f"Excluded by user-requested excluded function(s): {', '.join(matched)}."

    count_as_match = classification == DIRECT_MATCH and confidence >= DIRECT_MATCH_CONFIDENCE_FLOOR
    if classification == ADJACENT and query_spec.get("include_adjacent"):
        count_as_match = True

    result = {
        "classification": classification,
        "count_as_match": bool(count_as_match),
        "confidence": round(float(confidence), 2),
        "match_category": _match_category(classification),
        "match_confidence": _confidence_bucket(confidence),
        "role_signal": _role_signal(occupation_text),
        "employer_signal": _employer_signal(employer_text, profile),
        "match_reason_code": _match_reason_code(classification, selected_industry, occupation_text, employer_text, profile),
        "employer_industry": list(profile["employer_industry"]),
        "job_function": list(profile["job_function"]),
        "specialties": list(profile["specialties"]),
        "internal_reason": str(reason),
    }
    if row_id is not None:
        result["row_id"] = row_id
    return result


def _match_category(classification):
    return {
        DIRECT_MATCH: "direct",
        ADJACENT: "adjacent",
        UNCERTAIN: "uncertain",
        NON_MATCH: "excluded",
    }.get(classification, "excluded")


def _confidence_bucket(confidence):
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 0.0
    if value >= 0.85:
        return "high"
    if value >= 0.6:
        return "medium"
    return "low"


def _role_signal(occupation):
    if not occupation:
        return "unknown"
    if matched_term(occupation, TECHNICAL_TITLE_PHRASES):
        return "technical"
    if matched_term(occupation, ["product manager", "product management", "product lead", "product owner", "product strategy"]):
        return "product"
    if matched_term(occupation, ["analytics", "data analyst", "decision analytics", "growth analytics", "sales analytics"]):
        return "analytics"
    if matched_term(
        occupation,
        [
            "finance",
            "sales",
            "customer success",
            "business development",
            "account manager",
            "account executive",
            "strategy",
            "growth",
            "marketing",
            "operations",
            "legal",
            "counsel",
            "professional services",
        ],
    ):
        return "business"
    if matched_term(occupation, ["teacher", "physician", "doctor", "attorney", "law clerk"]):
        return "nontechnical"
    return "unknown"


def _employer_signal(employer, profile):
    if not employer:
        return "unknown"
    industries = set(profile.get("employer_industry") or [])
    if "technology" in industries:
        return "tech_company"
    if "financial_services" in industries:
        return "finance"
    if "consulting_professional_services" in industries:
        return "consulting"
    if "healthcare" in industries:
        return "healthcare"
    if "education" in industries:
        return "education"
    if matched_term(employer, ["laboratory", "lab", "research"]):
        return "research"
    taxonomy = get_taxonomy("tech") or {}
    if matched_term(employer, taxonomy.get("ambiguous_keywords") or []):
        return "ambiguous"
    return "unknown"


def _match_reason_code(classification, industry, occupation, employer, profile):
    if classification == DIRECT_MATCH:
        if matched_term(occupation, TECHNICAL_TITLE_PHRASES):
            if "financial_services" in profile.get("employer_industry", []):
                return "technical_role_in_finance"
            return "role_is_technical"
        if _taxonomy_key(industry) == "tech":
            taxonomy = get_taxonomy("tech") or {}
            if known_company_match(employer, taxonomy.get("known_companies") or []):
                role = _role_signal(occupation)
                if role == "product":
                    return "product_role_at_tech_company"
                if role == "business":
                    return "business_role_at_tech_company"
                return "employer_is_known_tech"
            if matched_term(employer, STRONG_TECH_EMPLOYER_KEYWORDS):
                return "employer_has_strong_tech_keyword"
        return "direct_match"
    if classification == ADJACENT:
        if _role_signal(occupation) == "product":
            return "product_at_nontech_or_ambiguous_employer"
        if _role_signal(occupation) == "analytics":
            return "analytics_adjacent"
        if _employer_signal(employer, profile) == "ambiguous":
            return "ambiguous_tech_company"
        return "tech_adjacent"
    if classification == UNCERTAIN:
        return "uncertain_signal"
    return "excluded_no_meaningful_signal"


def _classify_for_industry(industry, occupation, employer, profile, descriptor_text, model_classifier, query_spec=None):
    policy = _INDUSTRY_POLICIES.get(_taxonomy_key(industry))
    if policy:
        outcome = policy(occupation, employer, profile, query_spec or {})
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


def _classify_for_tech(occupation, employer, profile, query_spec=None):
    query_spec = query_spec or {}
    taxonomy = get_taxonomy("tech") or {}
    scope = query_spec.get("query_scope") or "industry"
    technical = matched_term(occupation, TECHNICAL_TITLE_PHRASES)
    known = known_company_match(employer, taxonomy.get("known_companies") or [])
    strong = matched_term(employer, STRONG_TECH_EMPLOYER_KEYWORDS)
    weak = matched_term(occupation, WEAK_TECH_TITLE_PHRASES)
    ambiguous = matched_term(employer, taxonomy.get("ambiguous_keywords") or [])
    exclusion = matched_term(" ".join([occupation, employer]), taxonomy.get("exclusion_keywords") or [])
    consulting_unit_employer = "consulting_professional_services" in profile.get("employer_industry", []) and matched_term(
        employer, ["consulting", "advisory", "professional services"]
    )

    if scope == "technical_role":
        if technical:
            return DIRECT_MATCH, 0.98, f"Explicit technical title ('{technical}')."
        if known or strong:
            return (
                ADJACENT,
                0.58,
                "Employer indicates the tech industry, but the query asks for software/technical roles and the title is not technical.",
            )
        if weak:
            return ADJACENT, 0.5, f"Title has tech-adjacent wording ('{weak}') but no explicit technical implementation role."
        if ambiguous:
            return UNCERTAIN, 0.45, f"Employer has ambiguous wording for tech ('{ambiguous}') without enough confirming context."
        return NON_MATCH, 0.2, "No explicit technical title was found."

    if scope == "tech_company":
        if consulting_unit_employer:
            return ADJACENT, 0.58, "Employer is a consulting/professional-services unit, not clearly a core tech-company role."
        if known:
            return DIRECT_MATCH, 0.95, f"Employer is a known technology company: {known}."
        if strong and not exclusion:
            return DIRECT_MATCH, 0.88, f"Employer has a strong technology-industry signal ('{strong}')."
        if technical:
            return ADJACENT, 0.62, f"Technical title ('{technical}') at an employer not classified as a tech company."
        if weak:
            return ADJACENT, 0.52, f"Title has tech-adjacent wording ('{weak}') but employer is not clearly a tech company."
        if ambiguous:
            return UNCERTAIN, 0.45, f"Employer has ambiguous wording for tech ('{ambiguous}') without enough confirming context."
        return NON_MATCH, 0.2, "No technology-company employer signal was found."

    if technical:
        if matched_term(occupation, ["clinical data analyst"]):
            return ADJACENT, 0.58, "Clinical data title has data signal but the healthcare context is not a direct tech match."
        return DIRECT_MATCH, 0.98, f"Explicit technical title ('{technical}')."

    if matched_term(occupation, ["marketplace operations"]):
        return ADJACENT, 0.52, "Marketplace operations is tech-adjacent but not a direct technical role or tech employer match."

    if consulting_unit_employer:
        return ADJACENT, 0.58, "Employer is a consulting/professional-services unit, not clearly a core tech-company role."

    if known:
        return DIRECT_MATCH, 0.95, f"Employer is a known technology company: {known}."
    if strong and not exclusion:
        return DIRECT_MATCH, 0.88, f"Employer has a strong technology-industry signal ('{strong}')."

    if weak:
        return ADJACENT, 0.55, f"Title has tech-adjacent wording ('{weak}') but no technical title or tech employer."

    if ambiguous:
        return UNCERTAIN, 0.45, f"Employer has ambiguous wording for tech ('{ambiguous}') without enough confirming context."

    return NON_MATCH, 0.2, "No technical title or technology-industry employer signal was found."


def _classify_for_investment_banking(occupation, employer, profile, query_spec=None):
    taxonomy = get_taxonomy("investment_banking") or {}
    explicit = matched_term(occupation, INVESTMENT_BANKING_TITLE_PHRASES)
    if explicit:
        return DIRECT_MATCH, 0.98, f"Title explicitly indicates investment banking ('{explicit}')."

    employer_is_ib = bool(known_company_match(employer, taxonomy.get("known_companies") or []))
    contextual = matched_term(occupation, INVESTMENT_BANKING_CONTEXT_PHRASES)
    if contextual and employer_is_ib:
        return DIRECT_MATCH, 0.9, f"Investment-bank employer with investment-banking context in title ('{contextual}')."

    if matched_term(occupation, ["corporate banking", "commercial banking", "risk analyst", "wealth management", "asset management"]):
        return NON_MATCH, 0.85, "Banking or finance role is not investment banking."

    if employer_is_ib and matched_term(occupation, ["analyst", "associate", "vice president"]):
        return UNCERTAIN, 0.45, "Known investment bank employer with a generic finance title; title does not prove investment banking."
    if employer_is_ib:
        return NON_MATCH, 0.75, "Known investment bank employer alone is not enough for investment banking."

    return NON_MATCH, 0.2, "No investment-banking title or context was found."


def _classify_for_banking(occupation, employer, profile, query_spec=None):
    taxonomy = get_taxonomy("banking") or {}
    if "government_legal" in profile["employer_industry"]:
        return NON_MATCH, 0.9, "Government/legal employer is not treated as banking despite weak bank wording."

    title = matched_term(occupation, BANKING_TITLE_PHRASES + INVESTMENT_BANKING_TITLE_PHRASES)
    if title:
        return DIRECT_MATCH, 0.92, f"Title indicates banking ('{title}')."

    employer_is_bank = bool(
        known_company_match(employer, taxonomy.get("known_companies") or [])
        or matched_term(employer, taxonomy.get("employer_keywords") or [])
    )
    if employer_is_bank:
        if matched_term(occupation, ["software engineer", "developer", "product manager", "marketing manager"]):
            return NON_MATCH, 0.8, "Employer is a bank, but the title is not a banking role."
        if not occupation:
            return DIRECT_MATCH, 0.78, "Known banking employer with no title; broad banking query treats employer as sufficient."
        if matched_term(occupation, ["analyst", "associate", "vice president", "risk", "credit", "finance"]):
            return DIRECT_MATCH, 0.78, "Banking employer with a plausible banking/finance title."
        return UNCERTAIN, 0.45, "Banking employer with an unclear title."

    return NON_MATCH, 0.2, "No banking title or employer signal was found."


def _classify_for_finance(occupation, employer, profile, query_spec=None):
    taxonomy = get_taxonomy("finance") or {}
    if "government_legal" in profile["employer_industry"]:
        return NON_MATCH, 0.9, "Government/legal employer (e.g. a development/world bank) is not treated as finance."
    banking_outcome = _classify_for_banking(occupation, employer, profile, query_spec or {})
    if banking_outcome[0] == DIRECT_MATCH:
        return NON_MATCH, 0.88, f"Banking is classified separately from finance: {banking_outcome[2]}"

    title = matched_term(occupation, FINANCE_TITLE_PHRASES + INVESTMENT_BANKING_TITLE_PHRASES)
    if title:
        return DIRECT_MATCH, 0.9, f"Title indicates finance ('{title}')."

    employer_is_finance = bool(
        known_company_match(employer, taxonomy.get("known_companies") or [])
        or matched_term(employer, taxonomy.get("employer_keywords") or [])
    )
    if employer_is_finance:
        return DIRECT_MATCH, 0.82, f"Employer indicates finance or financial services ({employer})."

    return NON_MATCH, 0.2, "No finance title or employer signal was found."


def _classify_for_marketing(occupation, employer, profile, query_spec=None):
    query_spec = query_spec or {}
    direct = matched_term(occupation, MARKETING_DIRECT_PHRASES)
    if direct:
        return DIRECT_MATCH, 0.95, f"Title explicitly indicates marketing ('{direct}')."

    if "marketing_growth" in set(query_spec.get("include_functions") or []) and matched_term(occupation, ["growth", "head of growth"]):
        if "technology" in profile["employer_industry"] and not matched_term(occupation, ["growth marketing"]):
            return NON_MATCH, 0.85, "Growth role at a technology employer is tracked as tech unless the title explicitly says growth marketing."
        return DIRECT_MATCH, 0.82, "The query explicitly includes growth roles and the title has growth responsibility."

    adjacent = matched_term(occupation, MARKETING_ADJACENT_PHRASES)
    if adjacent:
        return ADJACENT, 0.55, f"Title has marketing-adjacent wording ('{adjacent}') but no explicit marketing role."

    taxonomy = get_taxonomy("marketing") or {}
    employer_signal = known_company_match(employer, taxonomy.get("known_companies") or []) or matched_term(
        employer, taxonomy.get("employer_keywords") or []
    )
    if employer_signal and (not occupation or matched_term(occupation, ["manager", "analyst", "director", "coordinator", "strategist"])):
        return DIRECT_MATCH, 0.78, f"Marketing employer with a plausible marketing title ({employer_signal})."
    if employer_signal:
        return UNCERTAIN, 0.45, "Marketing/advertising employer with an unclear title."

    return NON_MATCH, 0.2, "No explicit marketing title or marketing employer signal was found."


def _classify_for_operations(occupation, employer, profile, query_spec=None):
    direct = matched_term(occupation, OPERATIONS_DIRECT_PHRASES)
    if direct:
        if matched_term(occupation, ["product operations", "curriculum operations", "clinical operations", "revops"]):
            return NON_MATCH, 0.85, f"Operations wording belongs to another labeled domain ('{direct}')."
        return DIRECT_MATCH, 0.95, f"Title explicitly indicates operations ('{direct}')."

    adjacent = matched_term(occupation, OPERATIONS_ADJACENT_PHRASES)
    if adjacent:
        return ADJACENT, 0.55, f"Title has operations-adjacent wording ('{adjacent}') but no explicit operations context."

    taxonomy = get_taxonomy("operations") or {}
    employer_signal = known_company_match(employer, taxonomy.get("known_companies") or []) or matched_term(
        employer, taxonomy.get("employer_keywords") or []
    )
    if employer_signal and matched_term(occupation, ["manager", "analyst", "associate", "director"]):
        return DIRECT_MATCH, 0.78, f"Operations/logistics employer with a plausible operations title ({employer_signal})."
    if employer_signal:
        return UNCERTAIN, 0.45, "Operations/logistics employer with an unclear title."

    return NON_MATCH, 0.2, "No explicit operations title or operations employer signal was found."


def _classify_for_government_legal(occupation, employer, profile, query_spec=None):
    title = matched_term(occupation, GOVERNMENT_LEGAL_TITLE_PHRASES)
    if title:
        return DIRECT_MATCH, 0.95, f"Title directly indicates legal, policy, or government work ('{title}')."

    taxonomy = get_taxonomy("government_legal") or {}
    employer_signal = known_company_match(employer, taxonomy.get("known_companies") or []) or matched_term(
        employer, taxonomy.get("employer_keywords") or []
    )
    if employer_signal:
        if matched_term(employer, ["department of education"]) and not matched_term(occupation, ["policy", "legal", "attorney", "law"]):
            return NON_MATCH, 0.85, "Education department context is treated as education unless the title is legal/policy."
        return DIRECT_MATCH, 0.88, f"Employer directly indicates government/legal work ({employer_signal})."

    weak = matched_term(" ".join([occupation, employer]), taxonomy.get("ambiguous_keywords") or [])
    if weak:
        return UNCERTAIN, 0.4, f"Weak public-sector wording ('{weak}') without direct government/legal evidence."

    return NON_MATCH, 0.2, "No direct legal, policy, or government signal was found."


def _classify_for_consulting(occupation, employer, profile, query_spec=None):
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
    "tech": _classify_for_tech,
    "investment_banking": _classify_for_investment_banking,
    "banking": _classify_for_banking,
    "finance": _classify_for_finance,
    "marketing": _classify_for_marketing,
    "operations": _classify_for_operations,
    "government_legal": _classify_for_government_legal,
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
        classification, _confidence, _reason = _classify_for_industry(
            key,
            occupation,
            employer,
            profile,
            "",
            None,
            {"industries": [key], "query_scope": "industry"},
        )
        return classification == DIRECT_MATCH
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
    classification = str(outcome.get("classification") or "").lower()
    if classification == DIRECT_MATCH:
        belongs = True
    elif classification in {NON_MATCH, ADJACENT}:
        belongs = False
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
