import difflib
import json
import os
import re

from app.services import ai_service
from app.services.analysis_executor import MAX_OPERATIONS


INTENTS = {
    "find_records",
    "people_filter",
    "aggregate",
    "summarize_dataset",
    "inspect_missing",
    "rank_records",
    "compare_groups",
    "unknown",
}
TARGET_ENTITIES = {"rows", "columns", "groups", "dataset"}
OUTPUT_FORMATS = {"table", "metrics", "ranked_list", "markdown"}
FILTER_MATCH_MODES = {"contains_any", "contains_all", "equals"}
AGGREGATION_OPERATIONS = {"count", "sum", "average", "mean", "avg", "summary", "numeric_summary", "correlation", "date_summary"}

SEMANTIC_COLUMN_SYNONYMS = {
    "first_name": ["first name", "firstname", "first_name", "given name"],
    "last_name": ["last name", "lastname", "last_name", "surname", "family name"],
    "person_name": ["name", "full name", "nickname", "preferred name", "alumni name", "constituent name", "customer"],
    "occupation": ["occupation", "job title", "job", "role", "position", "profession", "title"],
    "employer": ["employer", "company", "organization", "organisation", "firm", "business", "corp", "corporation"],
    "industry": ["industry", "sector", "field", "category"],
    "major": ["major", "degree", "field of study", "program"],
    "date": ["date", "last contact", "created at", "updated at", "timestamp"],
    "numeric_value": ["value", "amount", "score", "revenue", "sales", "orders", "count"],
    "lifetime_giving": ["lifetime giving", "giving", "donation", "donor amount", "gift amount", "total giving"],
    "grad_year": ["grad year", "grad yr", "graduation year", "class year", "class yr"],
    "gpa": ["gpa", "grade point average"],
    "category": ["category", "type", "segment", "group"],
    "revenue": ["revenue", "sales", "income"],
    "orders": ["orders", "order count", "purchases"],
    "email": ["email", "email address", "e-mail"],
    "linkedin_url": ["linkedin url", "linkedinurl", "linkedin_url", "linkedin", "linked in", "linked in url"],
    "phone": ["phone", "phone number", "mobile"],
    "city": ["city", "location", "town"],
    "state": ["state", "province", "region"],
}

TECH_SEARCH_TERMS = [
    "software",
    "software engineer",
    "developer",
    "programmer",
    "data scientist",
    "data engineer",
    "analytics",
    "ai",
    "machine learning",
    "ml",
    "artificial intelligence",
    "product manager",
    "product",
    "technical",
    "technology",
    "tech",
    "platform",
    "cloud",
    "cybersecurity",
    "security",
    "cto",
    "information technology",
    "cybersecurity",
    "solutions engineer",
    "sales engineer",
    "technical consultant",
    "software architect",
    "devops",
    "site reliability",
    "sre",
]

TECH_KNOWN_ENTITIES = [
    "Google",
    "Meta",
    "Facebook",
    "Microsoft",
    "Amazon",
    "Apple",
    "OpenAI",
    "PayPal",
    "Adobe",
    "Epic",
    "HackerRank",
    "Salesforce",
    "Oracle",
    "IBM",
    "Intel",
    "Nvidia",
    "NVIDIA",
    "Uber",
    "Airbnb",
    "Stripe",
    "Datadog",
    "Snowflake",
    "Palantir",
    "MongoDB",
    "Atlassian",
    "ServiceNow",
    "Intuit",
    "FanAmp",
    "Cogni DAO",
    "Amass Insights",
    "Benchmrk",
    "Launch Potato",
    "Rune Technologies",
]

ALUMNI_TERMS = ["alumni", "alum", "alums"]
ALUMNI_TECH_FALLBACK_TERMS = [
    "working in tech",
    "work in tech",
    "tech company",
    "tech companies",
    "technology company",
    "technology companies",
    "software engineer",
    "software engineers",
    "software developer",
    "software developers",
    "technical role",
    "technical roles",
    "roles in tech",
    "in tech",
]

CONCEPT_LIBRARY = {
    "tech_related": {
        "definition": "Alumni whose occupation or employer suggests software, engineering, product, data, AI, technology, startup, or a known technology company.",
        "search_terms": TECH_SEARCH_TERMS,
        "known_entities": TECH_KNOWN_ENTITIES,
        "default_semantic_columns": ["occupation", "employer", "industry", "major"],
        "assumption": "I treated tech alumni as records whose occupation or employer matched software, engineering, product, data, AI, technology, startup, or known tech-company terms.",
    },
    "software_engineer_role": {
        "definition": "Alumni whose occupation suggests software engineering or a related technical role.",
        "search_terms": [
            "software engineer",
            "engineer",
            "developer",
            "programmer",
            "full stack",
            "backend",
            "frontend",
            "systems engineer",
            "application developer",
            "web developer",
            "mobile developer",
            "technical lead",
            "CTO",
            "data scientist",
            "data engineer",
            "information technology",
            "cybersecurity",
            "solutions engineer",
            "sales engineer",
            "technical consultant",
            "software architect",
            "devops",
            "site reliability",
            "SRE",
        ],
        "known_entities": [],
        "default_semantic_columns": ["occupation"],
        "assumption": "I treated software engineering roles as occupation text containing software engineer, engineer, developer, programmer, full stack, backend, frontend, systems engineer, application developer, web developer, mobile developer, technical lead, or CTO.",
    },
    "tech_company": {
        "definition": "Alumni whose employer suggests a technology company or technology-focused organization.",
        "search_terms": [
            "technology",
            "software",
            "platform",
            "cloud",
            "AI",
            "data",
            "startup",
            "technologies",
            "data",
            "digital",
            "analytics",
            "cybersecurity",
            "fintech",
            "blockchain",
            "crypto",
            "SaaS",
            "app",
            "internet",
        ],
        "known_entities": TECH_KNOWN_ENTITIES,
        "default_semantic_columns": ["employer"],
        "assumption": "I treated tech companies as employer text containing technology, software, platform, cloud, AI, data, startup, or known technology-company names.",
    },
}


INTENT_INSTRUCTIONS = """
You infer spreadsheet analysis intent for a safe pandas backend.
Return only valid JSON. Do not include markdown fences or prose.
Do not generate code. Do not compute final numeric answers.
Use the compact dataset context only to infer intent, semantic columns, fuzzy concepts,
search terms, output preference, and whether clarification is truly needed.
Do not require exact dataframe column names. Use semantic column names and aliases.
Ask for clarification only when the request cannot be answered with available columns or safe operations.

Return this JSON shape:
{
  "intent": "find_records | aggregate | summarize_dataset | inspect_missing | rank_records | compare_groups | unknown",
  "target_entity": "rows | columns | groups | dataset",
  "user_goal": "Plain English interpretation of the user request",
  "concepts": [
    {
      "name": "tech_related",
      "definition": "Plain English definition",
      "search_terms": ["software", "engineer", "developer", "data", "AI"],
      "known_entities": ["Google", "Microsoft", "Amazon"]
    }
  ],
  "semantic_columns": {
    "person_name": ["name", "full name", "nickname"],
    "occupation": ["occupation", "job title", "role", "position"],
    "employer": ["employer", "company", "organization"],
    "industry": ["industry", "sector"],
    "major": ["major", "degree", "field of study"],
    "date": ["date", "last contact", "created at"],
    "numeric_value": ["lifetime giving", "amount", "score"]
  },
  "filters": [
    {
      "concept": "tech_related",
      "apply_to_semantic_columns": ["occupation", "employer", "industry", "major"],
      "match_mode": "contains_any"
    }
  ],
  "sort": null,
  "aggregation": null,
  "desired_output": {
    "format": "table | metrics | ranked_list | markdown",
    "semantic_columns": ["first_name", "last_name", "occupation", "employer", "linkedin_url"],
    "limit": 100
  },
  "assumptions": [],
  "clarification_needed": false,
  "clarifying_question": null
}
""".strip()


def infer_analysis_intent(question, dataset_context):
    if ai_service.client is None:
        return heuristic_intent(question, dataset_context), True, ""

    payload = {
        "question": question,
        "dataset_context": dataset_context,
    }

    try:
        response = ai_service.client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            instructions=INTENT_INSTRUCTIONS,
            input=json.dumps(payload, indent=2),
            max_output_tokens=1200,
            temperature=0,
            tools=[],
        )
    except Exception as exc:
        intent = heuristic_intent(question, dataset_context)
        intent["assumptions"].append(f"Intent model unavailable; used deterministic inference. {exc}")
        return intent, True, ""

    try:
        parsed = _parse_json(_extract_response_text(response))
    except ValueError:
        intent = heuristic_intent(question, dataset_context)
        intent["assumptions"].append("Intent model returned invalid JSON; used deterministic inference.")
        return intent, True, ""

    intent, valid, error = validate_analysis_intent(parsed)
    if _should_use_alumni_tech_fallback(question, intent):
        fallback = alumni_tech_fallback_intent(question)
        fallback["assumptions"].extend(intent.get("assumptions") or [])
        if intent.get("clarification_needed"):
            fallback["assumptions"].append(
                "The intent model requested clarification, so I used the default alumni tech filter."
            )
        return fallback, True, ""
    return intent, valid, error


def validate_analysis_intent(value):
    if not isinstance(value, dict):
        return _unknown_intent("Intent response must be a JSON object."), False, "Intent response must be a JSON object."

    intent = _clean_choice(value.get("intent"), INTENTS, "unknown")
    target_entity = _clean_choice(value.get("target_entity"), TARGET_ENTITIES, "dataset")

    concepts = []
    raw_concepts = value.get("concepts") if isinstance(value.get("concepts"), list) else []
    for raw in raw_concepts[:8]:
        if not isinstance(raw, dict):
            continue
        name = _clean_key(raw.get("name"))
        if not name:
            continue
        concepts.append(
            {
                "name": name,
                "definition": _clean_text(raw.get("definition"), max_length=500),
                "search_terms": _clean_text_list(raw.get("search_terms"), limit=30),
                "known_entities": _clean_text_list(raw.get("known_entities"), limit=30),
            }
        )

    semantic_columns = {}
    raw_search_columns = value.get("search_columns")
    if isinstance(raw_search_columns, dict):
        for key, aliases in raw_search_columns.items():
            semantic_key = _clean_key(key)
            if semantic_key:
                semantic_columns[semantic_key] = _clean_text_list(aliases, limit=20)

    raw_semantic_columns = value.get("semantic_columns")
    if isinstance(raw_semantic_columns, dict):
        for key, aliases in raw_semantic_columns.items():
            semantic_key = _clean_key(key)
            if semantic_key:
                semantic_columns[semantic_key] = _clean_text_list(aliases, limit=20)

    filters = []
    raw_filters = value.get("filters") if isinstance(value.get("filters"), list) else []
    for raw in raw_filters[:5]:
        if not isinstance(raw, dict):
            continue
        concept = _clean_key(raw.get("concept"))
        apply_to = [_clean_key(item) for item in _as_list(raw.get("apply_to_semantic_columns"))]
        apply_to = [item for item in apply_to if item]
        match_mode = _clean_choice(raw.get("match_mode"), FILTER_MATCH_MODES, "contains_any")
        if concept and apply_to:
            filters.append({"concept": concept, "apply_to_semantic_columns": apply_to, "match_mode": match_mode})

    concepts = _expand_concepts(concepts, filters)

    sort = value.get("sort") if isinstance(value.get("sort"), dict) else None
    if sort:
        sort = {
            "semantic_column": _clean_key(sort.get("semantic_column") or sort.get("column")),
            "direction": _clean_choice(sort.get("direction"), {"asc", "ascending", "desc", "descending"}, "desc"),
        }

    aggregation = value.get("aggregation") if isinstance(value.get("aggregation"), dict) else None
    if aggregation:
        operation = _clean_choice(aggregation.get("operation"), AGGREGATION_OPERATIONS, "count")
        if operation == "avg":
            operation = "average"
        if operation == "numeric_summary":
            operation = "summary"
        aggregation = {
            "operation": operation,
            "group_by_semantic_column": _clean_key(
                aggregation.get("group_by_semantic_column") or aggregation.get("group_by")
            ),
            "value_semantic_column": _clean_key(
                aggregation.get("value_semantic_column") or aggregation.get("value_column")
            ),
        }

    desired_output = value.get("desired_output") if isinstance(value.get("desired_output"), dict) else {}
    if not desired_output and (value.get("display_columns") is not None or value.get("limit") is not None):
        desired_output = {
            "format": value.get("format") or "table",
            "semantic_columns": value.get("display_columns") or [],
            "limit": value.get("limit"),
        }
    desired_output = {
        "format": _clean_choice(desired_output.get("format"), OUTPUT_FORMATS, "markdown"),
        "semantic_columns": [
            _clean_key(item) for item in _as_list(desired_output.get("semantic_columns")) if _clean_key(item)
        ][:12],
        "limit": _limit(desired_output.get("limit"), default=100),
    }

    normalized = {
        "intent": intent,
        "target_entity": target_entity,
        "user_goal": _clean_text(value.get("user_goal"), max_length=800),
        "concepts": concepts,
        "semantic_columns": semantic_columns,
        "filters": filters,
        "sort": sort,
        "aggregation": aggregation,
        "desired_output": desired_output,
        "assumptions": _clean_text_list(value.get("assumptions"), limit=10),
        "clarification_needed": bool(value.get("clarification_needed", False)),
        "clarifying_question": _clean_text(value.get("clarifying_question"), max_length=500) or None,
    }

    return normalized, True, ""


def intent_to_analysis_plan(intent, dataset_context):
    intent = intent if isinstance(intent, dict) else _unknown_intent("Invalid analysis intent.")
    if intent.get("clarification_needed"):
        if _should_use_alumni_tech_fallback(intent.get("user_goal") or intent.get("clarifying_question"), intent):
            intent = alumni_tech_fallback_intent(intent.get("user_goal") or intent.get("clarifying_question"))
        else:
            reason = intent.get("clarifying_question") or "Clarification is needed before this analysis can run."
            return _empty_plan(reason)

    resolved = resolve_intent_semantic_columns(intent, dataset_context)
    assumptions = list(intent.get("assumptions") or [])
    assumptions.extend(_resolution_assumptions(resolved))

    if intent.get("intent") == "summarize_dataset":
        return _plan([{"type": "column_summary", "params": {"columns": None}}], "metrics", assumptions)

    if intent.get("intent") == "inspect_missing":
        columns = _resolved_columns_for_requested(intent, resolved, intent.get("desired_output", {}).get("semantic_columns"))
        return _plan([{"type": "missing_values", "params": {"columns": columns or None}}], "metrics", assumptions)

    if intent.get("intent") in {"find_records", "people_filter"}:
        return _plan_find_records(intent, resolved, assumptions)

    if intent.get("intent") == "rank_records":
        return _plan_rank_records(intent, resolved, assumptions)

    if intent.get("intent") in {"aggregate", "compare_groups"}:
        return _plan_aggregate(intent, resolved, assumptions)

    return _empty_plan("I could not map that question to the approved analysis operations.")


def resolve_intent_semantic_columns(intent, dataset_context):
    columns = dataset_context.get("columns") or []
    actual_names = [str(column.get("name")) for column in columns if column.get("name") is not None]
    semantic_aliases = dict(intent.get("semantic_columns") or {})
    concepts = intent.get("concepts") or []

    requested_keys = set(semantic_aliases)
    for filter_spec in intent.get("filters") or []:
        requested_keys.update(filter_spec.get("apply_to_semantic_columns") or [])
    desired = intent.get("desired_output") or {}
    requested_keys.update(item for item in desired.get("semantic_columns") or [] if item != "matched_reason")
    if "person_name" in requested_keys:
        requested_keys.update(["first_name", "last_name"])
    if any(_is_tech_concept(concept.get("name")) for concept in concepts):
        requested_keys.update(["first_name", "last_name", "occupation", "employer", "linkedin_url"])
    aggregation = intent.get("aggregation") or {}
    requested_keys.update(
        item
        for item in [aggregation.get("group_by_semantic_column"), aggregation.get("value_semantic_column")]
        if item
    )
    sort = intent.get("sort") or {}
    if sort.get("semantic_column"):
        requested_keys.add(sort["semantic_column"])

    known_entities = []
    for concept in concepts:
        known_entities.extend(concept.get("known_entities") or [])

    resolved = {}
    for semantic_key in sorted(requested_keys):
        aliases = []
        aliases.extend(semantic_aliases.get(semantic_key) or [])
        aliases.extend(SEMANTIC_COLUMN_SYNONYMS.get(semantic_key) or [])
        aliases.append(semantic_key)
        match = _resolve_semantic_column(semantic_key, aliases, columns, actual_names, known_entities)
        if match:
            resolved[semantic_key] = match

    return resolved


def heuristic_intent(question, dataset_context):
    question_lower = str(question or "").lower()
    columns = dataset_context.get("columns") or []
    column_names = [column["name"] for column in columns]
    text_semantics = ["occupation", "employer", "industry", "major"]

    if _is_mutation_request(question_lower):
        return _unknown_intent("Only read-only analysis is supported. The uploaded dataset was not modified.")

    if "gpa" in question_lower and not _resolve_semantic_column("gpa", SEMANTIC_COLUMN_SYNONYMS["gpa"], columns, column_names, []):
        return _clarification_intent("No GPA-like column is available in this dataset.")

    if any(term in question_lower for term in ["missing", "null", "blank", "empty"]):
        return _base_intent("inspect_missing", "columns", question, "metrics")

    if any(term in question_lower for term in ["summary", "summarize", "describe", "profile"]):
        return _base_intent("summarize_dataset", "dataset", question, "metrics")

    if _is_tech_related_question(question_lower):
        if _is_alumni_tech_fallback_question(question_lower):
            return alumni_tech_fallback_intent(question)

        intent = _base_intent("find_records", "rows", question, "table")
        intent["concepts"] = _heuristic_tech_concepts(question_lower)
        intent["semantic_columns"] = {
            "person_name": SEMANTIC_COLUMN_SYNONYMS["person_name"],
            "occupation": SEMANTIC_COLUMN_SYNONYMS["occupation"],
            "employer": SEMANTIC_COLUMN_SYNONYMS["employer"],
            "industry": SEMANTIC_COLUMN_SYNONYMS["industry"],
            "major": SEMANTIC_COLUMN_SYNONYMS["major"],
        }
        intent["filters"] = _heuristic_tech_filters(intent["concepts"], text_semantics)
        intent["desired_output"] = {
            "format": "table",
            "semantic_columns": _display_semantics_for_question(question_lower),
            "limit": _extract_requested_count(question_lower, 100),
        }
        intent["assumptions"] = _concept_assumptions(intent["concepts"])
        return intent

    if any(term in question_lower for term in ["top", "highest", "largest", "biggest", "donor", "donors"]):
        value_semantic = _semantic_for_mentioned_column(question_lower, dataset_context, numeric_only=True) or "lifetime_giving"
        intent = _base_intent("rank_records", "rows", question, "ranked_list")
        intent["semantic_columns"] = {
            value_semantic: SEMANTIC_COLUMN_SYNONYMS.get(value_semantic, [value_semantic]),
            "person_name": SEMANTIC_COLUMN_SYNONYMS["person_name"],
            "employer": SEMANTIC_COLUMN_SYNONYMS["employer"],
        }
        intent["sort"] = {"semantic_column": value_semantic, "direction": "desc"}
        intent["desired_output"] = {
            "format": "ranked_list",
            "semantic_columns": ["person_name", value_semantic, "employer"],
            "limit": _extract_requested_count(question_lower, 10),
        }
        return intent

    if any(term in question_lower for term in ["bottom", "lowest", "smallest"]):
        value_semantic = _semantic_for_mentioned_column(question_lower, dataset_context, numeric_only=True) or "numeric_value"
        intent = _base_intent("rank_records", "rows", question, "table")
        intent["semantic_columns"] = {value_semantic: SEMANTIC_COLUMN_SYNONYMS.get(value_semantic, [value_semantic])}
        intent["sort"] = {"semantic_column": value_semantic, "direction": "asc"}
        intent["desired_output"] = {
            "format": "table",
            "semantic_columns": ["person_name", value_semantic, "employer"],
            "limit": _extract_requested_count(question_lower, 10),
        }
        return intent

    if _contains_word_or_phrase(question_lower, ["correlation", "relationship", "related"]):
        intent = _base_intent("compare_groups", "columns", question, "ranked_list")
        intent["aggregation"] = {"operation": "correlation", "group_by_semantic_column": "", "value_semantic_column": ""}
        return intent

    if (" by " in question_lower) and any(term in question_lower for term in ["count", "how many", "number of"]):
        return _aggregate_intent(question, dataset_context, "count")

    if (" by " in question_lower) and _contains_word_or_phrase(question_lower, ["average", "mean", "avg"]):
        return _aggregate_intent(question, dataset_context, "average")

    if (" by " in question_lower) and _contains_word_or_phrase(question_lower, ["total", "sum"]):
        return _aggregate_intent(question, dataset_context, "sum")

    if _contains_word_or_phrase(question_lower, ["average", "mean", "avg", "total", "sum"]):
        value_semantic = _semantic_for_mentioned_column(question_lower, dataset_context, numeric_only=True) or "numeric_value"
        intent = _base_intent("aggregate", "columns", question, "metrics")
        intent["semantic_columns"] = {value_semantic: SEMANTIC_COLUMN_SYNONYMS.get(value_semantic, [value_semantic])}
        intent["aggregation"] = {"operation": "summary", "group_by_semantic_column": "", "value_semantic_column": value_semantic}
        return intent

    if any(term in question_lower for term in ["date", "month", "year"]):
        return _base_intent("aggregate", "columns", question, "metrics", aggregation={"operation": "date_summary"})

    return _unknown_intent("I could not map that question to the approved analysis operations.")


def _plan_find_records(intent, resolved, assumptions):
    groups = []
    all_columns = []
    all_terms = []
    skipped = []
    for filter_spec in intent.get("filters") or []:
        concept = _find_concept(intent, filter_spec.get("concept"))
        terms = _terms_for_concept(concept)
        columns = _resolved_columns_for_requested(intent, resolved, filter_spec.get("apply_to_semantic_columns"))
        if not columns:
            skipped.append(f"No matching columns for concept '{filter_spec.get('concept')}'.")
            continue
        if not terms:
            skipped.append(f"No search terms for concept '{filter_spec.get('concept')}'.")
            continue
        groups.append({"concept": filter_spec.get("concept"), "columns": columns, "terms": terms})
        for column in columns:
            if column not in all_columns:
                all_columns.append(column)
        all_terms.extend(terms)

    if not groups:
        if skipped:
            return _empty_plan("; ".join(skipped))
        return _empty_plan("No applicable filters were inferred for this row search.")

    return_columns = _return_columns_for_intent(intent, resolved, default=all_columns)
    operation_type = "contains_any"
    strict_people_filter = (
        intent.get("intent") == "people_filter"
        or any(_is_tech_concept(concept.get("name")) for concept in intent.get("concepts") or [])
    )
    params = {
        "columns": all_columns,
        "terms": list(dict.fromkeys(all_terms)),
        "column_term_groups": groups,
        "display_columns": return_columns,
        "question": intent.get("user_goal") or "",
        "limit": (intent.get("desired_output") or {}).get("limit", 100),
    }
    if strict_people_filter:
        params["filter_mode"] = "tech_people"
        params["people_filter"] = {
            "intent": "people_filter",
            "entity": "alumni",
            "criteria_label": "working in tech or technical roles",
            "answer_label": "Alumni matching criteria",
            "filters": {
                "include_explicit_technical_titles": True,
                "include_strong_tech_employer_names": True,
                "include_known_tech_companies": True,
                "include_high_confidence_model_classified_tech_companies": True,
                "exclude_low_confidence_ambiguous_companies": True,
            },
        }
    return _plan(
        [
            {
                "type": operation_type,
                "params": params,
            }
        ],
        (intent.get("desired_output") or {}).get("format", "table"),
        assumptions,
    )


def _plan_rank_records(intent, resolved, assumptions):
    sort = intent.get("sort") or {}
    sort_semantic = sort.get("semantic_column") or "numeric_value"
    sort_column = _first_resolved(resolved, sort_semantic)
    if not sort_column:
        return _empty_plan(f"No column matching '{sort_semantic}' is available for ranking.")
    ascending = sort.get("direction") in {"asc", "ascending"}
    operation_type = "bottom_n" if ascending else "top_n"
    return _plan(
        [
            {
                "type": operation_type,
                "params": {
                    "column": sort_column,
                    "n": (intent.get("desired_output") or {}).get("limit", 10),
                    "return_columns": _return_columns_for_intent(intent, resolved, default=[sort_column]),
                },
            }
        ],
        (intent.get("desired_output") or {}).get("format", "ranked_list"),
        assumptions,
    )


def _plan_aggregate(intent, resolved, assumptions):
    aggregation = intent.get("aggregation") or {}
    operation = aggregation.get("operation")

    if operation == "correlation":
        return _plan([{"type": "correlation", "params": {"columns": None, "limit": 20}}], "ranked_list", assumptions)

    if operation == "date_summary":
        return _plan([{"type": "date_summary", "params": {"columns": None}}], "metrics", assumptions)

    value_semantic = aggregation.get("value_semantic_column")
    group_semantic = aggregation.get("group_by_semantic_column")
    value_column = _first_resolved(resolved, value_semantic)
    group_column = _first_resolved(resolved, group_semantic)

    if operation == "count" and group_semantic:
        if not group_column:
            return _empty_plan(f"No column matching '{group_semantic}' is available for grouping.")
        return _plan(
            [{"type": "group_by_count", "params": {"group_by": group_column, "limit": 25}}],
            "table",
            assumptions,
        )

    if operation in {"sum", "average", "mean"} and group_semantic:
        if not group_column:
            return _empty_plan(f"No column matching '{group_semantic}' is available for grouping.")
        if not value_column:
            return _empty_plan(f"No column matching '{value_semantic}' is available for aggregation.")
        operation_type = "group_by_average" if operation in {"average", "mean"} else "group_by_sum"
        return _plan(
            [{"type": operation_type, "params": {"group_by": group_column, "value_column": value_column, "limit": 25}}],
            "table",
            assumptions,
        )

    if value_semantic:
        if not value_column:
            return _empty_plan(f"No column matching '{value_semantic}' is available for numeric summary.")
        return _plan([{"type": "numeric_summary", "params": {"columns": [value_column]}}], "metrics", assumptions)

    return _empty_plan("No supported aggregation could be inferred from the request.")


def _resolve_semantic_column(semantic_key, aliases, column_contexts, actual_names, known_entities):
    candidates = [semantic_key] + list(aliases or [])
    candidates = [str(candidate).strip() for candidate in candidates if str(candidate).strip()]

    for candidate in candidates:
        for actual in actual_names:
            if candidate == actual:
                return actual

    for candidate in candidates:
        for actual in actual_names:
            if candidate.casefold() == actual.casefold():
                return actual

    normalized_candidates = [_normalize(candidate) for candidate in candidates if _normalize(candidate)]
    for candidate_norm in normalized_candidates:
        for actual in actual_names:
            if candidate_norm == _normalize(actual):
                return actual

    for candidate_norm in normalized_candidates:
        for actual in actual_names:
            actual_norm = _normalize(actual)
            if len(candidate_norm) >= 4 and (candidate_norm in actual_norm or actual_norm in candidate_norm):
                return actual

    if semantic_key == "employer" and known_entities:
        entity_norms = [_normalize(entity) for entity in known_entities if _normalize(entity)]
        for column in column_contexts:
            samples = column.get("sample_values") or []
            sample_text = _normalize(" ".join(str(value) for value in samples))
            if any(entity_norm and entity_norm in sample_text for entity_norm in entity_norms):
                return str(column.get("name"))

    best = None
    best_score = 0.0
    for candidate_norm in normalized_candidates:
        for actual in actual_names:
            actual_norm = _normalize(actual)
            if not candidate_norm or not actual_norm:
                continue
            score = difflib.SequenceMatcher(None, candidate_norm, actual_norm).ratio()
            if score > best_score:
                best = actual
                best_score = score
    if best and best_score >= 0.88:
        return best

    return None


def _resolution_assumptions(resolved):
    if not resolved:
        return []
    pieces = []
    has_first_or_last = bool(resolved.get("first_name") or resolved.get("last_name"))
    for semantic, actual in sorted(resolved.items()):
        if semantic == "person_name" and has_first_or_last:
            continue
        pieces.append(f"{semantic} -> {actual}")
    if not pieces:
        return []
    return ["Semantic columns were resolved as: " + "; ".join(pieces) + "."]


def _resolved_columns_for_requested(intent, resolved, requested):
    columns = []
    for semantic_key in requested or []:
        if semantic_key == "person_name":
            first = _first_resolved(resolved, "first_name")
            last = _first_resolved(resolved, "last_name")
            if first or last:
                for column in [first, last]:
                    if column and column not in columns:
                        columns.append(column)
                continue
        column = _first_resolved(resolved, semantic_key)
        if column and column not in columns:
            columns.append(column)
    return columns


def _return_columns_for_intent(intent, resolved, default):
    desired = intent.get("desired_output") or {}
    columns = _resolved_columns_for_requested(intent, resolved, desired.get("semantic_columns"))
    if not columns:
        for column in default:
            if column not in columns:
                columns.append(column)
    return columns


def _display_semantics_for_question(question_lower):
    semantics = ["first_name", "last_name", "occupation", "employer"]
    requested = {
        "major": ["major", "majors", "degree", "degrees", "field of study"],
        "grad_year": ["graduation year", "graduation years", "class year", "class years", "grad year", "grad yr"],
        "email": ["email", "emails", "email address", "e-mail"],
        "phone": ["phone", "phone number", "mobile"],
        "city": ["city", "cities", "location", "locations"],
        "state": ["state", "states", "province", "region"],
    }
    for semantic, terms in requested.items():
        if _contains_word_or_phrase(question_lower, terms):
            insert_at = 1 if semantic == "grad_year" else len(semantics)
            semantics.insert(insert_at, semantic)
    semantics.append("linkedin_url")
    return list(dict.fromkeys(semantics))


def alumni_tech_fallback_intent(question):
    intent = _base_intent("people_filter", "rows", question, "table")
    intent["entity"] = "alumni"
    intent["criteria_label"] = "working in tech or technical roles"
    intent["answer_label"] = "Alumni matching criteria"
    intent["concepts"] = [_concept_from_library("software_engineer_role"), _concept_from_library("tech_company")]
    intent["semantic_columns"] = {
        "first_name": SEMANTIC_COLUMN_SYNONYMS["first_name"],
        "last_name": SEMANTIC_COLUMN_SYNONYMS["last_name"],
        "occupation": SEMANTIC_COLUMN_SYNONYMS["occupation"],
        "employer": SEMANTIC_COLUMN_SYNONYMS["employer"],
        "linkedin_url": SEMANTIC_COLUMN_SYNONYMS["linkedin_url"],
    }
    intent["filters"] = [
        {
            "concept": "software_engineer_role",
            "apply_to_semantic_columns": ["occupation"],
            "match_mode": "contains_any",
        },
        {
            "concept": "tech_company",
            "apply_to_semantic_columns": ["employer"],
            "match_mode": "contains_any",
        },
    ]
    intent["desired_output"] = {
        "format": "table",
        "semantic_columns": ["first_name", "last_name", "occupation", "employer", "linkedin_url"],
        "limit": _extract_requested_count(str(question or "").lower(), 100),
    }
    intent["assumptions"] = [
        "I used the default alumni tech filter: explicit technical titles, strong tech employer names, known technology companies, and high-confidence classified tech employers count as confirmed matches; uncertain matches are kept separate."
    ]
    return intent


def _should_use_alumni_tech_fallback(question, intent=None):
    question_lower = str(question or "").lower()
    if not _is_alumni_tech_fallback_question(question_lower):
        return False
    if not intent:
        return True
    if intent.get("clarification_needed"):
        return True
    if intent.get("intent") in {"unknown", ""}:
        return True
    return False


def _is_alumni_tech_fallback_question(question_lower):
    has_alumni = _contains_word_or_phrase(question_lower, ALUMNI_TERMS)
    has_tech = _contains_word_or_phrase(question_lower, ALUMNI_TECH_FALLBACK_TERMS)
    return has_alumni and has_tech


def _is_tech_concept(name):
    return _clean_key(name) in {"tech_related", "software_engineer_role", "tech_company"}


def _first_resolved(resolved, semantic_key):
    if not semantic_key:
        return None
    return resolved.get(semantic_key)


def _expand_concepts(concepts, filters):
    by_name = {concept.get("name"): dict(concept) for concept in concepts if concept.get("name")}
    for filter_spec in filters or []:
        name = _clean_key(filter_spec.get("concept"))
        if name and name not in by_name and name in CONCEPT_LIBRARY:
            by_name[name] = {"name": name, "definition": "", "search_terms": [], "known_entities": []}

    expanded = []
    for name, concept in by_name.items():
        library = CONCEPT_LIBRARY.get(name)
        if library:
            concept["definition"] = concept.get("definition") or library["definition"]
            concept["search_terms"] = _merge_lists(concept.get("search_terms"), library.get("search_terms"))
            concept["known_entities"] = _merge_lists(concept.get("known_entities"), library.get("known_entities"))
        expanded.append(concept)
    return expanded


def _terms_for_concept(concept):
    if not concept:
        return []
    name = concept.get("name")
    library = CONCEPT_LIBRARY.get(name)
    terms = []
    terms.extend(concept.get("search_terms") or [])
    terms.extend(concept.get("known_entities") or [])
    if library:
        terms.extend(library.get("search_terms") or [])
        terms.extend(library.get("known_entities") or [])
    return [term for term in dict.fromkeys(str(term).strip() for term in terms) if term]


def _merge_lists(*values):
    items = []
    for value in values:
        for item in _as_list(value):
            text = _clean_text(item, max_length=120)
            if text:
                items.append(text)
    return list(dict.fromkeys(items))


def _find_concept(intent, name):
    normalized = _clean_key(name)
    for concept in intent.get("concepts") or []:
        if concept.get("name") == normalized:
            return concept
    return None


def _aggregate_intent(question, dataset_context, operation):
    question_lower = str(question or "").lower()
    group_semantic = _semantic_after_by(question_lower, dataset_context) or "category"
    value_semantic = _semantic_for_mentioned_column(question_lower, dataset_context, numeric_only=True) or "numeric_value"
    intent = _base_intent("aggregate", "groups", question, "table")
    intent["semantic_columns"] = {
        group_semantic: SEMANTIC_COLUMN_SYNONYMS.get(group_semantic, [group_semantic]),
        value_semantic: SEMANTIC_COLUMN_SYNONYMS.get(value_semantic, [value_semantic]),
    }
    intent["aggregation"] = {
        "operation": operation,
        "group_by_semantic_column": group_semantic,
        "value_semantic_column": value_semantic if operation != "count" else "",
    }
    return intent


def _is_tech_related_question(question_lower):
    return _contains_word_or_phrase(
        question_lower,
        ["tech", "technology", "software", "software engineer", "engineer", "developer", "tech company", "technical role", "data", "ai"],
    )


def _heuristic_tech_concepts(question_lower):
    names = []
    if _contains_word_or_phrase(
        question_lower,
        ["software engineer", "software engineers", "developer", "developers", "programmer", "technical role", "technical lead", "cto"],
    ):
        names.append("software_engineer_role")
    if _contains_word_or_phrase(question_lower, ["tech company", "tech companies", "technology company", "startup"]):
        names.append("tech_company")
    if not names:
        names.append("tech_related")
    if "tech_related" not in names and _contains_word_or_phrase(question_lower, ["tech", "technology", "data", "ai", "product"]):
        names.append("tech_related")
    return [_concept_from_library(name) for name in names]


def _heuristic_tech_filters(concepts, fallback_semantics):
    filters = []
    for concept in concepts:
        name = concept.get("name")
        library = CONCEPT_LIBRARY.get(name) or {}
        filters.append(
            {
                "concept": name,
                "apply_to_semantic_columns": library.get("default_semantic_columns") or fallback_semantics,
                "match_mode": "contains_any",
            }
        )
    return filters


def _concept_from_library(name):
    library = CONCEPT_LIBRARY[name]
    return {
        "name": name,
        "definition": library["definition"],
        "search_terms": list(library.get("search_terms") or []),
        "known_entities": list(library.get("known_entities") or []),
    }


def _concept_assumptions(concepts):
    assumptions = []
    for concept in concepts:
        library = CONCEPT_LIBRARY.get(concept.get("name"))
        if library and library.get("assumption"):
            assumptions.append(library["assumption"])
    return list(dict.fromkeys(assumptions))


def _semantic_for_mentioned_column(question_lower, dataset_context, numeric_only=False):
    for column in dataset_context.get("columns") or []:
        if numeric_only and column.get("type") != "number":
            continue
        name = str(column.get("name") or "")
        if _normalize(name) and _normalize(name) in _normalize(question_lower):
            return _clean_key(name)
    if "giving" in question_lower or "donor" in question_lower:
        return "lifetime_giving"
    if "revenue" in question_lower or "sales" in question_lower:
        return "revenue"
    if "order" in question_lower:
        return "orders"
    if "amount" in question_lower or "score" in question_lower:
        return "numeric_value"
    return None


def _semantic_after_by(question_lower, dataset_context):
    if " by " not in question_lower:
        return None
    tail = question_lower.rsplit(" by ", 1)[-1]
    for column in dataset_context.get("columns") or []:
        name = str(column.get("name") or "")
        if _normalize(name) and _normalize(name) in _normalize(tail):
            return _clean_key(name)
    if "category" in tail:
        return "category"
    if "industry" in tail or "sector" in tail:
        return "industry"
    if "company" in tail or "employer" in tail:
        return "employer"
    return None


def _plan(operations, presentation_hint, assumptions):
    return {
        "operations": operations[:MAX_OPERATIONS],
        "presentation_hint": presentation_hint if presentation_hint in OUTPUT_FORMATS else "markdown",
        "assumptions": list(dict.fromkeys(assumptions or [])),
        "cannot_answer_reason": "",
    }


def _empty_plan(reason):
    return {
        "operations": [],
        "presentation_hint": "markdown",
        "assumptions": [],
        "cannot_answer_reason": str(reason or "No approved analysis operation matched the question."),
    }


def _base_intent(intent, target_entity, question, output_format, aggregation=None):
    return {
        "intent": intent,
        "target_entity": target_entity,
        "user_goal": str(question or "").strip(),
        "concepts": [],
        "semantic_columns": {},
        "filters": [],
        "sort": None,
        "aggregation": aggregation,
        "desired_output": {"format": output_format, "semantic_columns": [], "limit": 100},
        "assumptions": [],
        "clarification_needed": False,
        "clarifying_question": None,
    }


def _unknown_intent(reason):
    intent = _base_intent("unknown", "dataset", "", "markdown")
    intent["clarification_needed"] = True
    intent["clarifying_question"] = reason
    return intent


def _clarification_intent(reason):
    return _unknown_intent(reason)


def _parse_json(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as nested_exc:
                raise ValueError("Intent model returned invalid JSON.") from nested_exc
        raise ValueError("Intent model returned invalid JSON.") from exc


def _extract_response_text(response):
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                return text.strip()
    return ""


def _clean_choice(value, allowed, default):
    value = str(value or "").strip().lower()
    return value if value in allowed else default


def _clean_key(value):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())).strip("_")


def _clean_text(value, max_length=1000):
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>\n]*>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_length]


def _clean_text_list(value, limit):
    items = []
    for item in _as_list(value):
        text = _clean_text(item, max_length=120)
        if text:
            items.append(text)
    return list(dict.fromkeys(items))[:limit]


def _as_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _limit(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < 1:
        parsed = default
    return min(parsed, 500)


def _extract_requested_count(question_lower, default):
    for pattern in [
        r"\b(?:top|bottom|first|last)\s+(\d{1,3})\b",
        r"\bshow(?:\s+me)?\s+(?:the\s+)?(?:top|bottom|first|last)?\s*(\d{1,3})\b",
    ]:
        match = re.search(pattern, question_lower)
        if match:
            return _limit(match.group(1), default)
    return default


def _contains_word_or_phrase(text, terms):
    normalized = _normalize(text)
    for term in terms:
        term_normalized = _normalize(term)
        if not term_normalized:
            continue
        if " " in term_normalized:
            if term_normalized in normalized:
                return True
        elif re.search(rf"\b{re.escape(term_normalized)}\b", normalized):
            return True
    return False


def _is_mutation_request(question_lower):
    return _contains_word_or_phrase(
        question_lower,
        [
            "delete",
            "drop",
            "remove",
            "erase",
            "wipe",
            "truncate",
            "modify",
            "update",
            "insert",
            "append",
            "overwrite",
            "replace all",
        ],
    )


def _normalize(value):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return " ".join(normalized.split())
