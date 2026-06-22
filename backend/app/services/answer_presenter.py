import json
import logging
import os
import re

from app.services import ai_service
from app.services.answer_schema import (
    deterministic_answer_from_results,
    normalize_answer,
    plain_markdown_answer,
)
from app.services.spreadsheet_service import to_json_safe


PRESENTER_INSTRUCTIONS = """
You present spreadsheet analysis results.
Return only valid JSON. Do not include markdown fences or prose outside JSON.
Do not include HTML.
Use operation_results as the source of truth. Do not invent numbers.
Mention assumptions and warnings when relevant.
Keep the answer concise.
For text-search results, use display_columns/columns exactly as provided. Do not add extra
searched fields or debug fields to tables.
For people_filter alumni results, use answer_label with total_matches as the main metric, optionally
add Showing with displayed_count, and do not present display_limit as the answer.
Do not show match_reason, confidence, classification reason, uncertainty reason, or model rationale
in the main visible table.
Prefer tables for row-level results, metrics for counts and summaries, and ranked_list for recommendations.
If an operation failed, explain what went wrong and suggest a better question.

Return this JSON shape:
{
  "answer": {
    "title": "Optional title",
    "summary": "Short direct answer",
    "blocks": [
      {"type": "markdown", "content": "..."},
      {"type": "table", "title": "...", "columns": [], "rows": [], "caption": "..."},
      {"type": "metrics", "items": [{"label": "...", "value": "..."}]},
      {"type": "ranked_list", "title": "...", "items": [{"label": "...", "value": "...", "description": "..."}]}
    ],
    "followups": []
  }
}
""".strip()


def present_answer(question, plan, operation_results, dataset_context):
    fallback = deterministic_answer_from_results(question, plan, operation_results, dataset_context)

    if ai_service.client is None:
        return fallback

    payload = {
        "question": question,
        "analysis_plan": plan,
        "operation_results": operation_results,
        "dataset_metadata": {
            "dataset_id": dataset_context.get("dataset_id"),
            "filename": dataset_context.get("filename"),
            "row_count": dataset_context.get("row_count"),
            "column_count": dataset_context.get("column_count"),
            "columns": dataset_context.get("columns"),
        },
        "assumptions": plan.get("assumptions") if isinstance(plan, dict) else [],
        "warnings": _collect_warnings(operation_results),
    }

    try:
        response = ai_service.client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            instructions=PRESENTER_INSTRUCTIONS,
            input=json.dumps(to_json_safe(payload), indent=2),
            max_output_tokens=1400,
            temperature=0.2,
            tools=[],
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("Presenter model call failed, using deterministic fallback: %s", exc)
        return fallback

    try:
        parsed = _parse_json(_extract_response_text(response))
    except ValueError:
        return fallback

    answer, valid = normalize_answer(parsed, fallback_summary=fallback.get("summary"))
    if not valid:
        return fallback
    answer = _ensure_people_filter_blocks(answer, operation_results, fallback)
    return _ensure_notes(answer, plan, operation_results)


def planner_failure_answer(reason):
    summary = "The app could not create a valid analysis plan."
    if reason:
        summary = f"{summary} {reason}"
    return plain_markdown_answer(
        summary,
        title="Analysis Plan Error",
        followups=["Summarize this dataset", "Which columns have missing values?", "Show top rows by a numeric column"],
    )


def _collect_warnings(operation_results):
    warnings = []
    for result in operation_results:
        warnings.extend(result.get("warnings") or [])
    return _dedupe_warnings(warnings)


def _ensure_notes(answer, plan, operation_results):
    assumptions = []
    if isinstance(plan, dict):
        assumptions.extend(plan.get("assumptions") or [])
    warnings = _collect_warnings(operation_results)

    notes = []
    if assumptions:
        notes.append("Assumptions: " + "; ".join(dict.fromkeys(str(item) for item in assumptions if str(item).strip())))
    if warnings:
        notes.append("Warnings: " + "; ".join(_format_warning(warning) for warning in warnings))
    if not notes:
        return answer

    answer_text = json.dumps(answer, ensure_ascii=False)
    missing_notes = [note for note in notes if note not in answer_text]
    if not missing_notes:
        return answer

    updated = dict(answer)
    updated["blocks"] = list(updated.get("blocks") or [])
    updated["blocks"].append({"type": "markdown", "content": "\n".join(missing_notes)})
    normalized, valid = normalize_answer(updated, fallback_summary=answer.get("summary"))
    return normalized if valid else answer


def _ensure_people_filter_blocks(answer, operation_results, fallback):
    if not any(
        result.get("intent") == "people_filter" and result.get("entity") == "alumni"
        for result in operation_results
        if isinstance(result, dict)
    ):
        return answer

    source_blocks = [
        block
        for block in fallback.get("blocks", [])
        if isinstance(block, dict) and block.get("type") in {"metrics", "table"}
    ]
    if not source_blocks:
        return answer

    updated = dict(answer)
    other_blocks = [
        block
        for block in updated.get("blocks", [])
        if isinstance(block, dict) and block.get("type") not in {"metrics", "table"}
    ]
    updated["blocks"] = source_blocks + other_blocks
    normalized, valid = normalize_answer(updated, fallback_summary=answer.get("summary"))
    return normalized if valid else answer


def _format_warning(warning):
    if isinstance(warning, dict):
        return str(warning.get("message") or warning)
    return str(warning)


def _dedupe_warnings(warnings):
    deduped = []
    seen = set()
    for warning in warnings or []:
        text = _format_warning(warning).strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(warning)
    return deduped


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
                raise ValueError("Presenter returned invalid JSON.") from nested_exc
        raise ValueError("Presenter returned invalid JSON.") from exc


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
