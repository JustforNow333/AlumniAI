import json
import os
import re

from app.services import ai_service
from app.services.analysis_executor import MAX_OPERATIONS
from app.services.analysis_toolkit import ALLOWED_OPERATION_TYPES


PLANNER_INSTRUCTIONS = """
You are a planner for a spreadsheet analysis backend.
Return only valid JSON. Do not include markdown fences or prose.
You may choose only these operation types:
preview, select_columns, filter_equals, filter_contains, search_text, contains_any, contains_all,
sort_rows, top_n, bottom_n, group_by_count, group_by_sum, group_by_average,
value_counts, missing_values, column_summary, numeric_summary, correlation,
unique_values, duplicate_rows, date_summary, date_range_filter, group_by_month.

Use the full uploaded dataset through backend operations. Do not reason from sample rows alone.
Do not generate Python code.
Use exact column names from the dataset context.
Use at most 3 operations.
For fuzzy concepts such as tech-related, high value, lapsed, or engaged, define criteria
in assumptions and encode them in operation params using available columns.

Return this JSON shape:
{
  "operations": [{"type": "operation_name", "params": {}}],
  "presentation_hint": "table | metrics | ranked_list | markdown",
  "assumptions": ["Plain language assumptions when relevant"],
  "cannot_answer_reason": ""
}

If no allowed operation can answer the request, return operations as an empty list and
set cannot_answer_reason.
""".strip()


def plan_analysis(question, dataset_context):
    if ai_service.client is None:
        return heuristic_plan(question, dataset_context), True, ""

    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    payload = {
        "question": question,
        "dataset_context": dataset_context,
        "max_operations": MAX_OPERATIONS,
    }

    try:
        response = ai_service.client.responses.create(
            model=model,
            instructions=PLANNER_INSTRUCTIONS,
            input=json.dumps(payload, indent=2),
            max_output_tokens=900,
            temperature=0,
            tools=[],
        )
    except Exception as exc:
        return heuristic_plan(question, dataset_context), True, f"Planner model unavailable: {exc}"

    text = _extract_response_text(response)
    try:
        parsed = _parse_json(text)
    except ValueError as exc:
        return _empty_plan(str(exc)), False, str(exc)

    return validate_plan(parsed)


def validate_plan(plan):
    if not isinstance(plan, dict):
        return _empty_plan("Planner response must be a JSON object."), False, "Planner response must be a JSON object."

    operations = plan.get("operations")
    if not isinstance(operations, list):
        return _empty_plan("Planner operations must be a list."), False, "Planner operations must be a list."

    normalized_operations = []
    for operation in operations[:MAX_OPERATIONS]:
        if not isinstance(operation, dict):
            return _empty_plan("Each operation must be an object."), False, "Each operation must be an object."
        operation_type = operation.get("type")
        if operation_type not in ALLOWED_OPERATION_TYPES:
            return _empty_plan(f"Unknown operation type '{operation_type}'."), False, f"Unknown operation type '{operation_type}'."
        params = operation.get("params") if isinstance(operation.get("params"), dict) else {}
        normalized_operations.append({"type": operation_type, "params": params})

    presentation_hint = str(plan.get("presentation_hint") or "markdown").lower()
    if presentation_hint not in {"table", "metrics", "ranked_list", "markdown"}:
        presentation_hint = "markdown"

    assumptions = []
    if isinstance(plan.get("assumptions"), list):
        assumptions = [str(item).strip() for item in plan["assumptions"] if str(item).strip()]

    normalized = {
        "operations": normalized_operations,
        "presentation_hint": presentation_hint,
        "assumptions": assumptions,
        "cannot_answer_reason": str(plan.get("cannot_answer_reason") or "").strip(),
    }

    if len(operations) > MAX_OPERATIONS:
        normalized["assumptions"].append(f"Only the first {MAX_OPERATIONS} operations were used.")

    return normalized, True, ""


def heuristic_plan(question, dataset_context):
    question_lower = str(question or "").lower()
    columns = dataset_context.get("columns") or []
    column_names = [column["name"] for column in columns]
    text_columns = [column["name"] for column in columns if column.get("type") == "text"]
    numeric_columns = [column["name"] for column in columns if column.get("type") == "number"]
    date_columns = [column["name"] for column in columns if column.get("type") == "date"]

    if _is_mutation_request(question_lower):
        return {
            "operations": [],
            "presentation_hint": "markdown",
            "assumptions": [],
            "cannot_answer_reason": "Only read-only analysis is supported. The uploaded dataset was not modified.",
        }

    if any(term in question_lower for term in ["missing", "null", "blank", "empty"]):
        return {
            "operations": [{"type": "missing_values", "params": {"columns": None}}],
            "presentation_hint": "metrics",
            "assumptions": [],
            "cannot_answer_reason": "",
        }

    if "duplicate" in question_lower:
        return {
            "operations": [{"type": "duplicate_rows", "params": {"subset": None, "limit": 100}}],
            "presentation_hint": "table",
            "assumptions": [],
            "cannot_answer_reason": "",
        }

    if any(term in question_lower for term in ["correlation", "relationship", "related"]):
        return {
            "operations": [{"type": "correlation", "params": {"columns": None, "limit": 20}}],
            "presentation_hint": "ranked_list",
            "assumptions": [],
            "cannot_answer_reason": "",
        }

    tech_terms = ["tech", "technology", "software", "engineer", "developer", "data", "ai", "machine learning"]
    if _contains_word_or_phrase(question_lower, tech_terms):
        searchable = _preferred_columns(
            column_names,
            ["occupation", "job", "title", "employer", "company", "industry", "major"],
            fallback=text_columns,
        )
        return_columns = _preferred_columns(
            column_names,
            ["name", "customer", "alumni", "occupation", "job", "title", "employer", "company", "industry", "major"],
            fallback=(text_columns[:5] or column_names[:5]),
        )
        return {
            "operations": [
                {
                    "type": "contains_any",
                    "params": {
                        "columns": searchable,
                        "terms": ["software", "engineer", "developer", "data", "ai", "machine learning", "tech", "technology"],
                        "return_columns": return_columns,
                        "limit": 100,
                    },
                }
            ],
            "presentation_hint": "table",
            "assumptions": [
                "Tech-related means records containing software, engineer, developer, data, AI, machine learning, tech, or technology in available text fields."
            ],
            "cannot_answer_reason": "",
        }

    giving_column = _find_column(column_names, ["lifetime giving", "giving", "donation", "donor", "amount", "revenue", "sales"])
    industry_column = _find_column(column_names, ["industry", "sector", "field", "category"])

    if any(term in question_lower for term in ["top", "highest", "largest", "biggest", "donor", "donors"]):
        numeric = giving_column or _mentioned_column(question_lower, numeric_columns) or (numeric_columns[0] if numeric_columns else None)
        if numeric:
            return_columns = _preferred_columns(
                column_names,
                ["name", "customer", "alumni", numeric, "employer", "company", "industry"],
                fallback=column_names[:5],
            )
            if numeric not in return_columns:
                return_columns.insert(min(1, len(return_columns)), numeric)
            return {
                "operations": [{"type": "top_n", "params": {"column": numeric, "n": _extract_requested_count(question_lower, 10), "return_columns": return_columns}}],
                "presentation_hint": "ranked_list",
                "assumptions": [],
                "cannot_answer_reason": "",
            }

    if any(term in question_lower for term in ["bottom", "lowest", "smallest"]):
        numeric = _mentioned_column(question_lower, numeric_columns) or (numeric_columns[0] if numeric_columns else None)
        if numeric:
            return {
                "operations": [{"type": "bottom_n", "params": {"column": numeric, "n": _extract_requested_count(question_lower, 10), "return_columns": _preferred_columns(column_names, ["name", "customer", "alumni", numeric, "employer", "company"], fallback=column_names[:5])}}],
                "presentation_hint": "table",
                "assumptions": [],
                "cannot_answer_reason": "",
            }

    if _contains_word_or_phrase(question_lower, ["average", "mean", "avg"]) and " by " in question_lower:
        value_column = giving_column or _mentioned_column(question_lower, numeric_columns)
        group_column = industry_column or _column_after_by(question_lower, column_names) or (text_columns[0] if text_columns else None)
        if value_column and group_column:
            return {
                "operations": [
                    {
                        "type": "group_by_average",
                        "params": {"group_by": group_column, "value_column": value_column, "limit": 25},
                    }
                ],
                "presentation_hint": "table",
                "assumptions": [],
                "cannot_answer_reason": "",
            }

    if _contains_word_or_phrase(question_lower, ["average", "mean", "avg"]):
        value_column = giving_column or _mentioned_column(question_lower, numeric_columns) or (numeric_columns[0] if numeric_columns else None)
        if value_column:
            return {
                "operations": [{"type": "numeric_summary", "params": {"columns": [value_column]}}],
                "presentation_hint": "metrics",
                "assumptions": [],
                "cannot_answer_reason": "",
            }

    if _contains_word_or_phrase(question_lower, ["total", "sum"]) and " by " in question_lower:
        value_column = giving_column or _mentioned_column(question_lower, numeric_columns)
        group_column = industry_column or _column_after_by(question_lower, column_names) or (text_columns[0] if text_columns else None)
        if value_column and group_column:
            return {
                "operations": [
                    {
                        "type": "group_by_sum",
                        "params": {"group_by": group_column, "value_column": value_column, "limit": 25},
                    }
                ],
                "presentation_hint": "table",
                "assumptions": [],
                "cannot_answer_reason": "",
            }

    if _contains_word_or_phrase(question_lower, ["total", "sum"]):
        value_column = giving_column or _mentioned_column(question_lower, numeric_columns) or (numeric_columns[0] if numeric_columns else None)
        if value_column:
            return {
                "operations": [{"type": "numeric_summary", "params": {"columns": [value_column]}}],
                "presentation_hint": "metrics",
                "assumptions": [],
                "cannot_answer_reason": "",
            }

    if any(term in question_lower for term in ["count", "how many", "number of"]) and " by " in question_lower:
        group_column = _column_after_by(question_lower, column_names) or industry_column or (text_columns[0] if text_columns else None)
        if group_column:
            return {
                "operations": [{"type": "group_by_count", "params": {"group_by": group_column, "limit": 25}}],
                "presentation_hint": "table",
                "assumptions": [],
                "cannot_answer_reason": "",
            }

    if any(term in question_lower for term in ["date", "month", "year"]) and date_columns:
        return {
            "operations": [{"type": "date_summary", "params": {"columns": None}}],
            "presentation_hint": "metrics",
            "assumptions": [],
            "cannot_answer_reason": "",
        }

    if any(term in question_lower for term in ["summary", "summarize", "describe", "profile"]):
        return {
            "operations": [{"type": "column_summary", "params": {"columns": None}}],
            "presentation_hint": "metrics",
            "assumptions": [],
            "cannot_answer_reason": "",
        }

    return {
        "operations": [],
        "presentation_hint": "markdown",
        "assumptions": [],
        "cannot_answer_reason": "I could not map that question to the approved analysis operations.",
    }


def _empty_plan(reason):
    return {
        "operations": [],
        "presentation_hint": "markdown",
        "assumptions": [],
        "cannot_answer_reason": reason,
    }


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
                raise ValueError("Planner returned invalid JSON.") from nested_exc
        raise ValueError("Planner returned invalid JSON.") from exc


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


def _preferred_columns(column_names, keywords, fallback):
    selected = []
    for keyword in keywords:
        if keyword in column_names:
            match = keyword
        else:
            match = _find_column(column_names, [keyword])
        if match and match not in selected:
            selected.append(match)
    for column in fallback:
        if column not in selected:
            selected.append(column)
        if len(selected) >= 6:
            break
    return selected[:6]


def _find_column(column_names, keywords):
    for keyword in keywords:
        keyword_norm = _normalize(keyword)
        for column in column_names:
            if keyword_norm in _normalize(column):
                return column
    return None


def _mentioned_column(question_lower, column_names):
    for column in column_names:
        if _normalize(column) in _normalize(question_lower):
            return column
    return None


def _column_after_by(question_lower, column_names):
    if " by " not in question_lower:
        return None
    tail = question_lower.rsplit(" by ", 1)[-1]
    return _mentioned_column(tail, column_names)


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


def _extract_requested_count(question_lower, default):
    patterns = [
        r"\b(?:top|bottom|first|last)\s+(\d{1,3})\b",
        r"\bshow(?:\s+me)?\s+(?:the\s+)?(?:top|bottom|first|last)?\s*(\d{1,3})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            try:
                value = int(match.group(1))
            except ValueError:
                return default
            return max(1, min(value, 500))
    return default


def _is_mutation_request(question_lower):
    mutation_terms = [
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
    ]
    return _contains_word_or_phrase(question_lower, mutation_terms)


def _normalize(value):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return " ".join(normalized.split())
