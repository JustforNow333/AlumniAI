"""Centralized industry taxonomy definitions and deterministic people-query classification.

Each taxonomy describes how to recognize one industry across messy alumni data:
- aliases:               natural-language phrases that map a question to this industry
- title_keywords:        occupation phrases that confirm membership on their own
- generic_title_keywords:roles too generic to confirm alone (count only with a matching employer)
- employer_keywords:     strong employer-name indicators
- known_companies:       employers that always confirm membership
- exclusion_keywords:    context that excludes a row unless the title is explicitly in-industry
- ambiguous_keywords:    employer wording that marks a row uncertain instead of excluded
- confidence_threshold:  minimum model confidence to confirm an ambiguous employer

The same row can belong to different industries depending on the query
(e.g. a Data Scientist at a hospital is tech by title and healthcare by employer).
"""

import json
import os
import re
from functools import lru_cache


KNOWN_TECH_COMPANIES_FILE = os.path.join(os.path.dirname(__file__), "known_tech_companies.json")

ANSWER_LABEL = "Alumni matching criteria"

# Roles that never define an industry by themselves; they count only when the
# employer matches the target industry.
GENERIC_BUSINESS_ROLES = [
    "founder",
    "co-founder",
    "cofounder",
    "ceo",
    "chief executive officer",
    "president",
    "vice president",
    "director",
    "manager",
    "partner",
    "principal",
    "head of growth",
    "growth",
    "strategy",
    "operations",
    "partnerships",
    "sales",
    "marketing",
    "business development",
    "chief of staff",
    "finance",
    "product",
    "product lead",
    "general manager",
    "analyst",
    "associate",
    "investor",
]

TAXONOMIES = {
    "tech": {
        "industry": "tech",
        "criteria_label": "working in tech or technical roles",
        "aliases": [
            "tech",
            "technology",
            "software",
            "software engineering",
            "computer science",
            "data",
            "ai",
            "artificial intelligence",
            "machine learning",
            "saas",
            "fintech",
            "internet",
            "tech company",
            "tech companies",
            "technology company",
            "technology companies",
            "technical role",
            "technical roles",
        ],
        "title_keywords": [
            "software engineer",
            "software developer",
            "developer",
            "programmer",
            "data scientist",
            "data engineer",
            "machine learning",
            "ml engineer",
            "ai",
            "ai engineer",
            "artificial intelligence",
            "product manager",
            "technical product manager",
            "engineering manager",
            "information technology",
            "it",
            "cybersecurity",
            "cloud",
            "systems engineer",
            "systems administrator",
            "systems analyst",
            "database",
            "analytics",
            "technical service",
            "technical services",
            "platform",
            "infrastructure",
            "solutions engineer",
            "sales engineer",
            "technical consultant",
            "software architect",
            "devops",
            "site reliability",
            "sre",
            "cto",
            "chief technology officer",
            "computer scientist",
            "full stack",
            "backend",
            "front end",
            "frontend",
            "security engineer",
            "mobile engineer",
            "ios engineer",
            "android engineer",
        ],
        "generic_title_keywords": [],
        "employer_keywords": [
            "technologies",
            "technology",
            "software",
            "ai",
            "data",
            "cloud",
            "systems",
            "labs",
            "platform",
            "digital",
            "analytics",
            "cybersecurity",
            "fintech",
            "blockchain",
            "crypto",
            "saas",
            "app",
            "internet",
            "network",
            "networks",
        ],
        # Merged with known_tech_companies.json in get_taxonomy().
        "known_companies": [],
        "exclusion_keywords": [
            "school",
            "middle school",
            "high school",
            "teacher",
            "department chair",
            "professor",
            "education",
            "hospital",
            "medical center",
            "healthcare",
            "health care",
            "oncology",
            "clinical",
            "surgery",
            "physician",
            "doctor",
            "law",
            "legal",
            "real estate",
            "insurance",
        ],
        "ambiguous_keywords": [
            "venture",
            "ventures",
            "innovation",
            "innovations",
            "dao",
            "capital",
            "partners",
        ],
        "confidence_threshold": 0.8,
    },
    "consulting": {
        "industry": "consulting",
        "criteria_label": "working in consulting",
        "aliases": [
            "consulting",
            "consultant",
            "consultants",
            "management consulting",
            "management consultant",
            "management consultants",
            "strategy consulting",
            "advisory",
        ],
        "title_keywords": [
            "consultant",
            "consulting",
            "associate consultant",
            "senior consultant",
            "management consultant",
            "strategy consultant",
            "implementation consultant",
            "engagement manager",
            "transaction advisory",
            "deal advisory",
            "valuation advisory",
            "restructuring advisory",
            "risk advisory",
            "technology advisory",
            "strategy advisory",
            "transaction services",
        ],
        "generic_title_keywords": [
            "analyst",
            "business analyst",
            "associate",
            "partner",
            "principal",
            "strategy",
        ],
        "employer_keywords": [
            "consulting",
            "consultants",
            "advisory",
            "advisors",
        ],
        "known_companies": [
            "McKinsey",
            "McKinsey & Company",
            "Bain",
            "Bain & Company",
            "Boston Consulting Group",
            "BCG",
            "Deloitte",
            "PwC",
            "PricewaterhouseCoopers",
            "EY",
            "Ernst & Young",
            "KPMG",
            "Accenture",
            "Oliver Wyman",
            "LEK",
            "L.E.K.",
            "Roland Berger",
            "Strategy&",
            "Booz Allen",
            "AlixPartners",
            "FTI Consulting",
            "Alvarez & Marsal",
            "A&M",
            "ZS",
            "Slalom",
            "Capgemini",
            "Cognizant",
            "Guidehouse",
            "Protiviti",
            "RSM",
            "Grant Thornton",
            "IBM Consulting",
            "Mercer",
            "Aon",
            "Willis Towers Watson",
            "WTW",
            "Navigant",
            "Huron",
            "Tata Consultancy Services",
            "TCS",
            "Infosys Consulting",
            "Wipro Consulting",
        ],
        "exclusion_keywords": [
            "attorney",
            "law clerk",
            "judicial",
            "paralegal",
            "investment banking",
            "private equity",
            "portfolio manager",
            "equity research",
            "wealth management",
        ],
        "ambiguous_keywords": ["strategy", "partners", "advisors"],
        # Broad candidate-retrieval terms: these find rows worth classifying,
        # they never confirm a match on their own.
        "retrieval_keywords": [
            "consultant",
            "consulting",
            "advisory",
            "advisor",
            "strategy",
            "operations",
            "management",
            "transaction",
            "deal",
            "valuation",
            "restructuring",
            "risk",
            "implementation",
            "transformation",
        ],
        "confidence_threshold": 0.8,
    },
    "banking": {
        "industry": "banking",
        "criteria_label": "working in banking or investment banking",
        "aliases": [
            "banking",
            "investment banking",
            "investment banker",
            "investment bankers",
            "ib",
            "banker",
            "bankers",
            "sales and trading",
            "capital markets",
            "equity research",
            "m&a",
        ],
        "title_keywords": [
            "investment banking analyst",
            "investment banking associate",
            "investment banking",
            "banker",
            "investment banker",
            "capital markets",
            "m&a",
            "mergers and acquisitions",
            "sales and trading",
            "s&t",
            "equity research",
            "credit analyst",
            "leveraged finance",
            "restructuring",
            "private wealth",
            "wealth management",
            "corporate banking",
            "commercial banking",
        ],
        "generic_title_keywords": ["analyst", "associate", "vice president"],
        "employer_keywords": [
            "bank",
            "banking",
            "capital markets",
            "securities",
        ],
        "known_companies": [
            "Goldman Sachs",
            "Morgan Stanley",
            "JPMorgan",
            "J.P. Morgan",
            "JP Morgan",
            "Bank of America",
            "BofA",
            "Citi",
            "Citigroup",
            "Barclays",
            "Evercore",
            "Lazard",
            "Moelis",
            "PJT",
            "Centerview",
            "RBC",
            "UBS",
            "Deutsche Bank",
            "Wells Fargo",
            "Jefferies",
            "Houlihan Lokey",
            "William Blair",
            "Guggenheim",
            "Rothschild",
            "Nomura",
            "HSBC",
        ],
        "exclusion_keywords": [],
        "ambiguous_keywords": ["capital", "partners"],
        "confidence_threshold": 0.8,
    },
    "finance": {
        "industry": "finance",
        "criteria_label": "working in finance or financial services",
        "aliases": [
            "finance",
            "financial services",
            "asset management",
            "wealth management",
            "hedge fund",
            "hedge funds",
            "trading",
            "quant",
            "quants",
            "private wealth",
        ],
        "title_keywords": [
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
        ],
        "generic_title_keywords": ["analyst", "associate", "risk"],
        "employer_keywords": [
            "capital",
            "investments",
            "investment management",
            "asset management",
            "wealth",
            "hedge fund",
            "securities",
            "financial",
            "advisors",
            "bank",
        ],
        "known_companies": [
            "BlackRock",
            "Vanguard",
            "Fidelity",
            "Bridgewater",
            "Citadel",
            "Jane Street",
            "Two Sigma",
            "D.E. Shaw",
            "Point72",
            "Millennium",
            "AQR",
            "PIMCO",
            "Apollo",
            "Blackstone",
            "KKR",
            "Carlyle",
            "T. Rowe Price",
            "Wellington Management",
            "State Street",
            "Charles Schwab",
            "Bloomberg",
            "Goldman Sachs",
            "Morgan Stanley",
            "JPMorgan",
        ],
        "exclusion_keywords": [],
        "ambiguous_keywords": ["partners"],
        "confidence_threshold": 0.8,
    },
    "healthcare": {
        "industry": "healthcare",
        "criteria_label": "working in healthcare or medicine",
        "aliases": [
            "healthcare",
            "health care",
            "medicine",
            "medical",
            "hospital",
            "hospitals",
            "biotech",
            "pharma",
            "clinical",
            "doctor",
            "doctors",
            "physician",
            "physicians",
            "nurse",
            "nurses",
        ],
        "title_keywords": [
            "physician",
            "doctor",
            "surgeon",
            "nurse",
            "clinician",
            "clinical",
            "medical",
            "healthcare",
            "health care",
            "therapist",
            "pharmacist",
            "dentist",
            "oncology",
            "radiology",
            "cardiology",
            "anesthesiology",
            "hospital administrator",
            "public health",
        ],
        "generic_title_keywords": ["resident", "fellow", "researcher"],
        "employer_keywords": [
            "hospital",
            "medical center",
            "clinic",
            "healthcare",
            "health",
            "pharma",
            "pharmaceutical",
            "biotech",
            "therapeutics",
            "life sciences",
        ],
        "known_companies": [
            "Mayo Clinic",
            "Cleveland Clinic",
            "Johns Hopkins",
            "Mass General",
            "Massachusetts General Hospital",
            "Hospital for Special Surgery",
            "HSS",
            "Pfizer",
            "Moderna",
            "Johnson & Johnson",
            "Merck",
            "Novartis",
            "Roche",
            "Genentech",
            "Amgen",
            "Gilead",
            "Regeneron",
            "Bristol Myers Squibb",
            "Eli Lilly",
            "UnitedHealth",
            "CVS Health",
            "Cigna",
            "Humana",
        ],
        "exclusion_keywords": [],
        "ambiguous_keywords": [],
        "confidence_threshold": 0.8,
    },
    "law": {
        "industry": "law",
        "criteria_label": "working in law or legal services",
        "aliases": [
            "law",
            "legal",
            "attorney",
            "attorneys",
            "lawyer",
            "lawyers",
            "law firm",
            "law firms",
        ],
        "title_keywords": [
            "lawyer",
            "attorney",
            "counsel",
            "legal counsel",
            "associate attorney",
            "law clerk",
            "paralegal",
            "litigation",
            "corporate law",
            "legal",
        ],
        "generic_title_keywords": ["partner", "associate"],
        "employer_keywords": [
            "law",
            "legal",
            "llp",
            "attorney",
            "attorneys",
        ],
        "known_companies": [
            "Skadden",
            "Cravath",
            "Wachtell",
            "Sullivan & Cromwell",
            "Davis Polk",
            "Latham & Watkins",
            "Kirkland & Ellis",
            "Simpson Thacher",
            "Paul Weiss",
            "Debevoise",
            "Weil",
            "Ropes & Gray",
            "Goodwin",
            "Cooley",
            "Fenwick",
            "Sidley",
            "Gibson Dunn",
            "White & Case",
        ],
        "exclusion_keywords": [],
        "ambiguous_keywords": [],
        "confidence_threshold": 0.8,
    },
    "education": {
        "industry": "education",
        "criteria_label": "working in education or academia",
        "aliases": [
            "education",
            "teacher",
            "teachers",
            "professor",
            "professors",
            "university",
            "universities",
            "academic",
            "academia",
            "school",
            "schools",
        ],
        "title_keywords": [
            "teacher",
            "professor",
            "lecturer",
            "instructor",
            "educator",
            "dean",
            "school counselor",
            "department chair",
            "postdoc",
            "phd student",
        ],
        "generic_title_keywords": ["principal", "researcher", "academic"],
        "employer_keywords": [
            "school",
            "university",
            "college",
            "academy",
            "education",
            "institute",
            "district",
        ],
        "known_companies": [
            "Cornell University",
            "Harvard University",
            "Stanford University",
            "MIT",
            "Yale University",
            "Princeton University",
            "Columbia University",
            "University of Chicago",
            "Latin School of Chicago",
        ],
        "exclusion_keywords": [],
        "ambiguous_keywords": [],
        "confidence_threshold": 0.8,
    },
    "media": {
        "industry": "media",
        "criteria_label": "working in media or entertainment",
        "aliases": [
            "media",
            "entertainment",
            "music",
            "streaming",
            "publishing",
            "journalism",
            "sports media",
        ],
        "title_keywords": [
            "producer",
            "editor",
            "journalist",
            "reporter",
            "media",
            "entertainment",
            "music",
            "streaming",
        ],
        "generic_title_keywords": [
            "writer",
            "content",
            "creative",
            "audience",
            "growth",
            "partnerships",
        ],
        "employer_keywords": [
            "media",
            "entertainment",
            "music",
            "streaming",
            "publishing",
            "news",
            "sports",
        ],
        "known_companies": [
            "Spotify",
            "Netflix",
            "Disney",
            "Hulu",
            "Warner Bros",
            "WarnerMedia",
            "NBCUniversal",
            "Paramount",
            "YouTube",
            "TikTok",
            "The New York Times",
            "Washington Post",
            "ESPN",
            "Vox",
            "Condé Nast",
            "Conde Nast",
        ],
        "exclusion_keywords": [],
        "ambiguous_keywords": [],
        "confidence_threshold": 0.8,
    },
    "nonprofit": {
        "industry": "nonprofit",
        "criteria_label": "working in nonprofit or social impact",
        "aliases": [
            "nonprofit",
            "nonprofits",
            "non-profit",
            "non-profits",
            "ngo",
            "ngos",
            "charity",
            "charities",
            "social impact",
        ],
        "title_keywords": [
            "program officer",
            "program director",
            "social worker",
            "volunteer coordinator",
            "development officer",
        ],
        "generic_title_keywords": ["director", "manager", "coordinator"],
        "employer_keywords": [
            "foundation",
            "nonprofit",
            "non-profit",
            "ngo",
            "charity",
            "charities",
        ],
        "known_companies": [
            "Red Cross",
            "United Way",
            "Gates Foundation",
            "Teach For America",
            "Peace Corps",
            "AmeriCorps",
            "Habitat for Humanity",
        ],
        "exclusion_keywords": [],
        "ambiguous_keywords": [],
        "confidence_threshold": 0.8,
    },
    "startups": {
        "industry": "startups",
        "criteria_label": "working at startups",
        "aliases": [
            "startup",
            "startups",
            "early stage",
            "early-stage",
        ],
        "title_keywords": [
            "founding engineer",
            "founding team",
            "startup",
        ],
        "generic_title_keywords": [
            "founder",
            "co-founder",
            "cofounder",
            "entrepreneur",
        ],
        "employer_keywords": [
            "labs",
            "ai",
            "technologies",
            "platform",
            "app",
            "software",
        ],
        "known_companies": [
            "FanAmp",
            "Cogni DAO",
            "Amass Insights",
            "Benchmrk",
            "Launch Potato",
            "Rune Technologies",
            "OpenAI",
            "Anthropic",
            "Databricks",
            "Stripe",
            "Ramp",
            "Scale AI",
            "Cursor",
            "Perplexity",
        ],
        "exclusion_keywords": [
            "school",
            "university",
            "hospital",
            "medical center",
            "government",
        ],
        "ambiguous_keywords": ["ventures", "venture"],
        "confidence_threshold": 0.8,
    },
    "venture_capital": {
        "industry": "venture_capital",
        "criteria_label": "working in venture capital",
        "aliases": [
            "venture capital",
            "vc",
            "venture partner",
            "venture investing",
        ],
        "title_keywords": [
            "venture capital",
            "vc",
            "venture partner",
            "investment partner",
        ],
        "generic_title_keywords": [
            "investor",
            "principal",
            "associate",
            "analyst",
            "partner",
        ],
        "employer_keywords": [
            "ventures",
            "venture capital",
            "venture partners",
        ],
        "known_companies": [
            "Andreessen Horowitz",
            "a16z",
            "Sequoia",
            "Benchmark",
            "Greylock",
            "Kleiner Perkins",
            "General Catalyst",
            "Lightspeed",
            "Accel",
            "NEA",
            "Bessemer",
            "Founders Fund",
            "Union Square Ventures",
            "First Round Capital",
            "Index Ventures",
            "Thrive Capital",
            "Coatue",
        ],
        "exclusion_keywords": [],
        "ambiguous_keywords": ["capital", "partners"],
        "confidence_threshold": 0.8,
    },
    "private_equity": {
        "industry": "private_equity",
        "criteria_label": "working in private equity",
        "aliases": [
            "private equity",
            "pe",
            "buyout",
            "buyouts",
            "growth equity",
        ],
        "title_keywords": [
            "private equity",
            "buyout",
            "growth equity",
        ],
        "generic_title_keywords": [
            "investment associate",
            "investment analyst",
            "vice president",
            "principal",
            "partner",
            "associate",
            "analyst",
        ],
        "employer_keywords": [
            "private equity",
            "equity partners",
        ],
        "known_companies": [
            "Blackstone",
            "KKR",
            "Apollo",
            "Carlyle",
            "TPG",
            "Warburg Pincus",
            "Silver Lake",
            "Vista Equity",
            "Thoma Bravo",
            "General Atlantic",
            "Bain Capital",
            "Hellman & Friedman",
            "Advent International",
            "EQT",
        ],
        "exclusion_keywords": [],
        "ambiguous_keywords": ["capital", "investments", "partners"],
        "confidence_threshold": 0.8,
    },
}

# Tie-break order when a question or company set matches several industries.
# More specific industries come first; tech is last because its lists are broad.
INDUSTRY_PRIORITY = [
    "banking",
    "venture_capital",
    "private_equity",
    "consulting",
    "law",
    "healthcare",
    "education",
    "media",
    "finance",
    "startups",
    "nonprofit",
    "tech",
]

# Explicit role queries ("Who are founders?", "Show me product managers") that
# should use occupation matching instead of an industry taxonomy.
OCCUPATION_QUERY_ROLES = {
    "founder": ["founder", "co-founder", "cofounder"],
    "co-founder": ["founder", "co-founder", "cofounder"],
    "cofounder": ["founder", "co-founder", "cofounder"],
    "ceo": ["ceo", "chief executive officer"],
    "chief executive officer": ["ceo", "chief executive officer"],
    "product manager": ["product manager"],
    "software engineer": ["software engineer", "software developer"],
    "software developer": ["software engineer", "software developer"],
    "data scientist": ["data scientist"],
    "data engineer": ["data engineer"],
    "engineer": ["engineer"],
}

PEOPLE_QUERY_HINTS = [
    "alumni",
    "alum",
    "alums",
    "who",
    "which",
    "people",
    "person",
    "anyone",
    "show me",
    "list",
    "how many",
    "find",
]


@lru_cache(maxsize=1)
def _known_tech_companies_from_config():
    try:
        with open(KNOWN_TECH_COMPANIES_FILE, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        loaded = []
    if not isinstance(loaded, list):
        return tuple()
    return tuple(str(item).strip() for item in loaded if str(item).strip())


def taxonomy_names():
    return list(TAXONOMIES)


def get_taxonomy(industry):
    base = TAXONOMIES.get(_normalize_industry_name(industry))
    if not base:
        return None
    taxonomy = dict(base)
    if taxonomy["industry"] == "tech":
        merged = list(taxonomy.get("known_companies") or [])
        for company in _known_tech_companies_from_config():
            if company not in merged:
                merged.append(company)
        taxonomy["known_companies"] = merged
    return taxonomy


def industry_for_question(question):
    """Map a natural-language question to a taxonomy name via aliases.

    The longest matching alias wins; ties go to INDUSTRY_PRIORITY order.
    Returns None when no alias matches.
    """
    normalized = _normalize(question)
    if not normalized:
        return None
    best_industry = None
    best_length = 0
    for industry in INDUSTRY_PRIORITY:
        taxonomy = TAXONOMIES[industry]
        for alias in taxonomy.get("aliases") or []:
            if _term_matches(normalized, alias) and len(alias) > best_length:
                best_industry = industry
                best_length = len(alias)
    return best_industry


def classify_people_question(question):
    """Deterministically classify a people/alumni question into a filter spec.

    Returns a dict like
        {"intent": "people_filter", "entity": "alumni", "filter_type": "industry",
         "industry": "consulting", "criteria_label": ..., "answer_label": ...}
    or None when the question does not look like a people filter. filter_type is
    one of industry, employer, occupation.
    """
    text = str(question or "").strip()
    normalized = _normalize(text)
    if not normalized:
        return None
    if not any(_term_matches(normalized, hint) for hint in PEOPLE_QUERY_HINTS):
        return None
    # Group-by/aggregation questions are handled by the aggregate planner.
    if " by " in f" {normalized} ":
        return None

    employers = _extract_employer_candidates(text)
    if employers:
        industries = [_industries_for_company(employer) for employer in employers]
        if len(employers) >= 2 and all(industries):
            shared = set(industries[0]).intersection(*industries[1:])
            if shared:
                industry = next(name for name in INDUSTRY_PRIORITY if name in shared)
                return _industry_spec(industry)
        return _employer_spec(employers)

    if re.search(r"\bstartups?\b", normalized):
        return _industry_spec("startups")

    occupation_terms = _occupation_terms_for_question(normalized)
    if occupation_terms:
        return _occupation_spec(occupation_terms)

    industry = industry_for_question(text)
    if industry:
        return _industry_spec(
            industry,
            include_adjacent=_question_requests_adjacent(normalized),
            include_functions=_requested_functions_for_question(industry, normalized),
            required_industries=_required_intersection_industries(industry, normalized),
        )

    fallback_employer = _lowercase_employer_fallback(normalized)
    if fallback_employer:
        return _employer_spec([fallback_employer])

    return None


def _industry_spec(industry, include_adjacent=False, include_functions=None, required_industries=None):
    taxonomy = get_taxonomy(industry)
    include_functions = list(include_functions or [])
    required_industries = list(required_industries or [])
    criteria_label = taxonomy["criteria_label"]
    if required_industries:
        criteria_label = f"{criteria_label} with {' and '.join(required_industries)} context"
    if include_functions:
        readable = ", ".join(label.replace("_", " ") for label in include_functions)
        criteria_label = f"{criteria_label} or in {readable} roles"
    if include_adjacent:
        criteria_label = f"{criteria_label} (including adjacent roles)"
    return {
        "intent": "people_filter",
        "entity": "alumni",
        "filter_type": "industry",
        "industry": industry,
        "industries": [industry],
        "required_industries": required_industries,
        "include_functions": include_functions,
        "include_adjacent": bool(include_adjacent),
        "criteria_label": criteria_label,
        "answer_label": ANSWER_LABEL,
    }


def _question_requests_adjacent(normalized):
    return bool(re.search(r"\badjacent\b", normalized))


def _requested_functions_for_question(industry, normalized):
    """Union queries like "consulting or strategy" pull in job functions the row
    classifier can match directly (the query, not the row, decides scope)."""
    functions = []
    if industry == "consulting" and re.search(
        r"\bconsult(?:ing|ants?)\s+(?:or|and)\s+strategy\b|\bstrategy\s+(?:or|and)\s+consult(?:ing|ants?)\b",
        normalized,
    ):
        functions.append("internal_strategy")
    return functions


def _required_intersection_industries(industry, normalized):
    """Intersection queries like "finance consulting": another industry's alias
    directly modifying the consulting noun narrows the answer to the overlap."""
    if industry != "consulting":
        return []
    required = []
    for other in INDUSTRY_PRIORITY:
        if other == industry:
            continue
        for alias in TAXONOMIES[other].get("aliases") or []:
            alias_norm = _normalize(alias)
            if alias_norm and re.search(rf"\b{re.escape(alias_norm)}\s+consult(?:ing|ants?)\b", normalized):
                if other not in required:
                    required.append(other)
                break
    return required


def _employer_spec(employers):
    employers = [str(item).strip() for item in employers if str(item).strip()]
    return {
        "intent": "people_filter",
        "entity": "alumni",
        "filter_type": "employer",
        "industry": None,
        "employer_terms": employers,
        "criteria_label": "working at " + " or ".join(employers),
        "answer_label": ANSWER_LABEL,
    }


def _occupation_spec(occupation_terms):
    return {
        "intent": "people_filter",
        "entity": "alumni",
        "filter_type": "occupation",
        "industry": None,
        "occupation_terms": list(occupation_terms),
        "criteria_label": "with occupation matching " + " or ".join(occupation_terms),
        "answer_label": ANSWER_LABEL,
    }


def _occupation_terms_for_question(normalized):
    for role in sorted(OCCUPATION_QUERY_ROLES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(role)}s?\b", normalized):
            return OCCUPATION_QUERY_ROLES[role]
    return None


def _extract_employer_candidates(question):
    """Extract capitalized employer names after "at"/"works for" from raw text."""
    text = str(question or "")
    match = re.search(
        r"\b(?:at|for)\s+((?:[A-Z][\w&.'’-]*|of|the|and|&|or|,)(?:\s+(?:[A-Z][\w&.'’-]*|of|the|and|&|or|,))*)",
        text,
    )
    if not match:
        return []
    tail = match.group(1).strip().rstrip("?.!,")
    parts = re.split(r"\s+or\s+|\s+and\s+|,", tail)
    employers = []
    for part in parts:
        cleaned = part.strip().strip("&").strip()
        cleaned = re.sub(r"^(?:the|of)\s+", "", cleaned)
        cleaned = re.sub(r"\s+(?:the|of)$", "", cleaned)
        if cleaned and cleaned[0].isupper():
            employers.append(cleaned)
    return employers


def _lowercase_employer_fallback(normalized):
    """Last resort: "who works at acme" with a lowercase, unrecognized employer."""
    match = re.search(r"\b(?:work(?:s|ing)? at|at)\s+([a-z][\w&.'-]*(?:\s+[\w&.'-]+){0,3})$", normalized)
    if not match:
        return None
    candidate = match.group(1).strip().rstrip("?.!")
    if not candidate or industry_for_question(candidate):
        return None
    return candidate


def _industries_for_company(company):
    industries = []
    company_norm = _normalize_company(company)
    if not company_norm:
        return industries
    for industry in INDUSTRY_PRIORITY:
        taxonomy = get_taxonomy(industry)
        for known in taxonomy.get("known_companies") or []:
            known_norm = _normalize_company(known)
            if known_norm and (known_norm == company_norm or re.search(rf"\b{re.escape(known_norm)}\b", company_norm)):
                industries.append(industry)
                break
    return industries


def _normalize_industry_name(value):
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "vc": "venture_capital",
        "venturecapital": "venture_capital",
        "venture_capital": "venture_capital",
        "pe": "private_equity",
        "privateequity": "private_equity",
        "private_equity": "private_equity",
        "startup": "startups",
        "technology": "tech",
    }
    return aliases.get(normalized, normalized)


def _term_matches(normalized_text, term):
    term_norm = _normalize(term)
    if not term_norm:
        return False
    if " " in term_norm:
        return term_norm in normalized_text
    return re.search(rf"\b{re.escape(term_norm)}\b", normalized_text) is not None


def _normalize(value):
    normalized = re.sub(r"[^a-z0-9&]+", " ", str(value or "").lower())
    return " ".join(normalized.split())


def _normalize_company(value):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    suffixes = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company"}
    words = [word for word in normalized.split() if word not in suffixes]
    return " ".join(words)
