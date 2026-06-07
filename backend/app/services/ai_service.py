import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from app.services.analysis_service import summarize_dataframe
from app.services.spreadsheet_service import (
    create_basic_summary,
    dataframe_preview,
    to_json_safe,
)


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key and OpenAI else None


AI_INSTRUCTIONS = """
You are an AI spreadsheet analyst.
Use only the provided spreadsheet data and computed analysis results.
Do not invent numbers.
If the provided data is insufficient, say what calculation or data is needed.
Be concise and clear.
When a calculation result is provided, explain it in plain English.
Mention caveats when relevant, such as missing values or limited preview data.
Return only valid JSON. Do not include markdown fences.
Do not include HTML.
The JSON must have this shape:
{
  "answer": {
    "title": "Optional short title",
    "summary": "Short natural language answer",
    "blocks": [
      {"type": "markdown", "content": "Text explanation"},
      {"type": "table", "title": "Title", "columns": ["Column"], "rows": [["Value"]], "caption": "Optional caption"},
      {"type": "metrics", "items": [{"label": "Rows analyzed", "value": "1,000"}]},
      {"type": "ranked_list", "title": "Title", "items": [{"label": "Item", "value": "Value", "description": "Why"}]}
    ],
    "followups": ["Short follow-up question"]
  }
}
Use tables for row-level results, metrics for compact numbers, ranked_list for ordered results, and markdown for explanation.
""".strip()


_HTML_TAG_RE = re.compile(r"<[^>\n]*>")
_MAX_BLOCKS = 8
_MAX_FOLLOWUPS = 4
_MAX_TABLE_COLUMNS = 12
_MAX_TABLE_ROWS = 20
_MAX_METRICS = 8
_MAX_RANKED_ITEMS = 10


def build_ai_context(df, metadata=None, dataset_id=None):
    basic_summary = create_basic_summary(df, include_preview=False)
    metadata = metadata or {}

    return to_json_safe(
        {
            "dataset_id": dataset_id or metadata.get("dataset_id"),
            "filename": metadata.get("original_filename") or metadata.get("filename"),
            "shape": {
                "rows": basic_summary["rows"],
                "columns": basic_summary["columns"],
            },
            "row_count": basic_summary["rows"],
            "column_count": basic_summary["columns"],
            "column_names": basic_summary["column_names"],
            "column_types": basic_summary["column_types"],
            "missing_values": basic_summary["missing_values"],
            "preview_first_10_rows": dataframe_preview(df, limit=10),
            "dataframe_summary": summarize_dataframe(df),
        }
    )


def generate_answer(question, dataset_context, operation=None, result=None):
    fallback = _fallback_structured_answer(operation, result)
    if client is None:
        return fallback

    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    prompt_payload = {
        "question": question,
        "dataset_context": dataset_context,
        "operation": operation,
        "result": result,
    }

    try:
        response = client.responses.create(
            model=model,
            instructions=AI_INSTRUCTIONS,
            input=json.dumps(to_json_safe(prompt_payload), indent=2),
            max_output_tokens=1200,
            temperature=0.2,
            tools=[],
        )
    except Exception as exc:
        return _with_markdown_notice(
            fallback, f"OpenAI narration was unavailable: {_clean_text(exc)}"
        )

    answer, valid = normalize_structured_answer(_extract_response_text(response))
    if valid:
        return answer

    return _with_markdown_notice(
        fallback,
        "The model response was not valid structured JSON, so this answer uses the backend computed result.",
    )


def normalize_structured_answer(value, fallback_summary=None):
    parsed = value
    if isinstance(value, str):
        try:
            parsed = _parse_model_json(value)
        except ValueError:
            summary = _clean_text(fallback_summary or value or "I could not format that response.")
            return _plain_markdown_answer(summary), False

    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), dict):
        parsed = parsed["answer"]

    if not isinstance(parsed, dict):
        summary = _clean_text(fallback_summary or "I could not format that response.")
        return _plain_markdown_answer(summary), False

    summary = _clean_text(parsed.get("summary") or fallback_summary or "", max_length=1200)
    title = _clean_text(parsed.get("title") or "", max_length=100)
    raw_blocks = parsed.get("blocks") if isinstance(parsed.get("blocks"), list) else []
    blocks = []

    for raw_block in raw_blocks[:_MAX_BLOCKS]:
        block = _normalize_block(raw_block)
        if block:
            blocks.append(block)

    if not blocks and summary:
        blocks.append({"type": "markdown", "content": summary})

    if not summary and not blocks:
        summary = _clean_text(fallback_summary or "I could not format that response.")
        return _plain_markdown_answer(summary), False

    followups = []
    raw_followups = parsed.get("followups")
    if isinstance(raw_followups, list):
        for item in raw_followups[:_MAX_FOLLOWUPS]:
            text = _clean_text(item, max_length=120)
            if text:
                followups.append(text)

    normalized = {
        "title": title,
        "summary": summary,
        "blocks": blocks,
        "followups": followups,
    }
    return normalized, True


def _fallback_answer(operation=None, result=None):
    return _fallback_structured_answer(operation, result)["summary"]


def _fallback_structured_answer(operation=None, result=None):
    operation_type = operation.get("type") if isinstance(operation, dict) else None

    if operation_type == "summarize_dataframe" and isinstance(result, dict):
        total_missing = result.get("total_missing_values", 0)
        rows = result.get("rows", 0)
        columns = result.get("columns", 0)
        missing_label = "value" if total_missing == 1 else "values"
        summary = (
            f"The dataset has {rows} rows, {columns} columns, and "
            f"{total_missing} missing {missing_label}."
        )
        blocks = [
            {
                "type": "metrics",
                "items": [
                    {"label": "Rows analyzed", "value": _format_value(rows)},
                    {"label": "Columns", "value": _format_value(columns)},
                    {"label": "Missing values", "value": _format_value(total_missing)},
                    {
                        "label": "Duplicate rows",
                        "value": _format_value(result.get("duplicate_row_count", 0)),
                    },
                ],
            }
        ]
        missing_rows = [
            [column, _format_value(count)]
            for column, count in (result.get("missing_values") or {}).items()
            if count
        ]
        if missing_rows:
            blocks.append(
                {
                    "type": "table",
                    "title": "Missing Values",
                    "columns": ["Column", "Missing"],
                    "rows": missing_rows,
                    "caption": "Only columns with missing values are shown.",
                }
            )
        return _answer(
            "Dataset Summary",
            summary,
            blocks,
            ["Summarize missing values", "Show top rows", "Which columns are numeric?"],
        )

    if operation_type == "group_by_aggregate" and isinstance(result, dict):
        aggregation = operation.get("aggregation") or "aggregate"
        group_col = operation.get("group_col") or "Group"
        value_col = operation.get("value_col") or "Value"
        if result:
            top_group, top_value = next(iter(result.items()))
            summary = (
                f"{top_group} is the top {group_col} by {aggregation} "
                f"{value_col}, with a value of {_format_value(top_value)}."
            )
        else:
            summary = f"I grouped {value_col} by {group_col}, but no result rows were returned."
        rows = [[key, _format_value(value)] for key, value in result.items()]
        return _answer(
            f"{aggregation.title()} {value_col} by {group_col}",
            summary,
            [
                {
                    "type": "table",
                    "title": "Grouped Results",
                    "columns": [group_col, f"{aggregation} {value_col}"],
                    "rows": rows,
                    "caption": "Sorted by the backend safe aggregation result.",
                },
                {
                    "type": "ranked_list",
                    "title": "Top Groups",
                    "items": [
                        {
                            "label": str(key),
                            "value": _format_value(value),
                            "description": f"{aggregation} {value_col}",
                        }
                        for key, value in list(result.items())[:_MAX_RANKED_ITEMS]
                    ],
                },
            ],
            [f"Show this grouped by another column", f"Summarize {value_col}", "Show missing values"],
        )

    if operation_type == "top_rows" and isinstance(result, dict):
        rows = result.get("rows") or []
        sort_col = result.get("sort_col") or operation.get("sort_col")
        if rows:
            summary = (
                f"The top row by {sort_col} has {sort_col} = "
                f"{_format_value(rows[0].get(sort_col))}."
            )
        else:
            summary = f"No rows were available to rank by {sort_col}."
        columns = _display_columns_for_rows(rows, preferred=sort_col)
        return _answer(
            f"Top Rows by {sort_col}",
            summary,
            [
                {
                    "type": "table",
                    "title": "Top Rows",
                    "columns": columns,
                    "rows": [[_format_value(row.get(column)) for column in columns] for row in rows],
                    "caption": "Sorted across all available rows by the backend.",
                }
            ],
            [f"Summarize {sort_col}", "Show missing values", "Group this by a category"],
        )

    if operation_type == "correlation" and isinstance(result, dict):
        coefficient = result.get("correlation")
        col1 = result.get("col1")
        col2 = result.get("col2")
        if coefficient is None:
            summary = f"There is not enough overlapping numeric data to correlate {col1} and {col2}."
            coefficient_value = "Not available"
        else:
            summary = f"{col1} and {col2} have a correlation of {coefficient:.2f}."
            coefficient_value = f"{coefficient:.2f}"
        return _answer(
            "Correlation",
            summary,
            [
                {
                    "type": "metrics",
                    "items": [
                        {"label": "Pearson r", "value": coefficient_value},
                        {"label": "Rows analyzed", "value": _format_value(result.get("rows_used", 0))},
                    ],
                },
                {"type": "markdown", "content": summary},
            ],
            [f"Show top rows by {col1}", f"Show top rows by {col2}", "Summarize missing values"],
        )

    if operation_type == "summarize_column" and isinstance(result, dict):
        column = result.get("column")
        result_type = result.get("type")
        if result_type == "numeric":
            summary = (
                f"{column} has {result.get('count')} values, an average of "
                f"{_format_value(result.get('mean'))}, and a median of {_format_value(result.get('median'))}."
            )
            return _answer(
                f"{column} Summary",
                summary,
                [
                    {
                        "type": "metrics",
                        "items": [
                            {"label": "Count", "value": _format_value(result.get("count"))},
                            {"label": "Mean", "value": _format_value(result.get("mean"))},
                            {"label": "Median", "value": _format_value(result.get("median"))},
                            {"label": "Min", "value": _format_value(result.get("min"))},
                            {"label": "Max", "value": _format_value(result.get("max"))},
                            {"label": "Missing", "value": _format_value(result.get("missing_count"))},
                        ],
                    }
                ],
                [f"Show top rows by {column}", f"Group average {column} by category", "Show missing values"],
            )
        if result_type == "date":
            summary = (
                f"{column} ranges from {result.get('earliest')} to "
                f"{result.get('latest')}."
            )
            return _answer(
                f"{column} Date Range",
                summary,
                [
                    {
                        "type": "metrics",
                        "items": [
                            {"label": "Earliest", "value": _format_value(result.get("earliest"))},
                            {"label": "Latest", "value": _format_value(result.get("latest"))},
                            {"label": "Missing", "value": _format_value(result.get("missing_count"))},
                        ],
                    }
                ],
                [f"Which records are missing {column}?", "Show top rows", "Summarize missing values"],
            )
        summary = (
            f"{column} has {result.get('unique_count')} unique values. "
            f"Top values are shown below."
        )
        top_values = result.get("top_values") or {}
        return _answer(
            f"{column} Summary",
            summary,
            [
                {
                    "type": "ranked_list",
                    "title": "Top Values",
                    "items": [
                        {
                            "label": str(label),
                            "value": _format_value(value),
                            "description": f"Records where {column} is {label}",
                        }
                        for label, value in list(top_values.items())[:_MAX_RANKED_ITEMS]
                    ],
                }
            ],
            [f"Group records by {column}", "Show missing values", "Show top rows"],
        )

    if operation_type == "analysis_error" and isinstance(result, dict):
        summary = result.get("error", "I could not run that analysis.")
        return _plain_markdown_answer(summary, title="Request Not Applied")

    summary = (
        "I could not map that question to a supported safe operation. Try asking for "
        "missing values, a column summary, top rows, a grouped total, or a correlation."
    )
    return _plain_markdown_answer(
        summary,
        title="Try a Data Question",
        followups=[
            "Summarize this dataset",
            "What values are missing?",
            "Show top rows by a numeric column",
            "Show a grouped total",
        ],
    )


def _answer(title, summary, blocks=None, followups=None):
    answer = {
        "title": title,
        "summary": summary,
        "blocks": blocks or [],
        "followups": followups or [],
    }
    normalized, _valid = normalize_structured_answer(answer, fallback_summary=summary)
    return normalized


def _plain_markdown_answer(summary, title="", followups=None):
    summary = _clean_text(summary)
    return _answer(
        title,
        summary,
        [{"type": "markdown", "content": summary}] if summary else [],
        followups or [],
    )


def _with_markdown_notice(answer, notice):
    answer = dict(answer or _plain_markdown_answer(""))
    blocks = list(answer.get("blocks") or [])
    blocks.append({"type": "markdown", "content": notice})
    answer["blocks"] = blocks
    normalized, _valid = normalize_structured_answer(answer, fallback_summary=answer.get("summary"))
    return normalized


def _parse_model_json(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError("Model response was not valid JSON.") from exc
        raise ValueError("Model response was not valid JSON.")


def _normalize_block(raw_block):
    if not isinstance(raw_block, dict):
        return None

    block_type = _clean_text(raw_block.get("type"), max_length=40).lower()

    if block_type == "markdown":
        content = _clean_text(raw_block.get("content"), max_length=4000)
        return {"type": "markdown", "content": content} if content else None

    if block_type == "table":
        columns = _normalize_string_list(raw_block.get("columns"), _MAX_TABLE_COLUMNS)
        if not columns:
            return None

        rows = []
        raw_rows = raw_block.get("rows") if isinstance(raw_block.get("rows"), list) else []
        for raw_row in raw_rows[:_MAX_TABLE_ROWS]:
            if isinstance(raw_row, dict):
                row = [_clean_text(raw_row.get(column), max_length=300) for column in columns]
            elif isinstance(raw_row, (list, tuple)):
                row = [_clean_text(value, max_length=300) for value in raw_row[: len(columns)]]
                row.extend([""] * (len(columns) - len(row)))
            else:
                row = [_clean_text(raw_row, max_length=300)]
                row.extend([""] * (len(columns) - 1))
            rows.append(row)

        return {
            "type": "table",
            "title": _clean_text(raw_block.get("title"), max_length=120),
            "columns": columns,
            "rows": rows,
            "caption": _clean_text(raw_block.get("caption"), max_length=240),
        }

    if block_type == "metrics":
        items = []
        raw_items = raw_block.get("items") if isinstance(raw_block.get("items"), list) else []
        for raw_item in raw_items[:_MAX_METRICS]:
            if not isinstance(raw_item, dict):
                continue
            label = _clean_text(raw_item.get("label"), max_length=80)
            value = _clean_text(raw_item.get("value"), max_length=120)
            if label or value:
                items.append({"label": label, "value": value})
        return {"type": "metrics", "items": items} if items else None

    if block_type == "ranked_list":
        items = []
        raw_items = raw_block.get("items") if isinstance(raw_block.get("items"), list) else []
        for raw_item in raw_items[:_MAX_RANKED_ITEMS]:
            if not isinstance(raw_item, dict):
                continue
            label = _clean_text(raw_item.get("label"), max_length=140)
            value = _clean_text(raw_item.get("value"), max_length=120)
            description = _clean_text(raw_item.get("description"), max_length=300)
            if label or value or description:
                items.append(
                    {
                        "label": label,
                        "value": value,
                        "description": description,
                    }
                )
        if not items:
            return None
        return {
            "type": "ranked_list",
            "title": _clean_text(raw_block.get("title"), max_length=120),
            "items": items,
        }

    return None


def _normalize_string_list(values, limit):
    if not isinstance(values, list):
        return []

    normalized = []
    for value in values[:limit]:
        text = _clean_text(value, max_length=120)
        if text:
            normalized.append(text)
    return normalized


def _clean_text(value, max_length=1000):
    if value is None:
        return ""

    text = str(value)
    text = _HTML_TAG_RE.sub("", text)
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > max_length:
        return text[: max_length - 1].rstrip() + "..."
    return text


def _format_value(value):
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return _clean_text(value, max_length=300)


def _display_columns_for_rows(rows, preferred=None):
    if not rows:
        return []

    first_row = rows[0]
    if not isinstance(first_row, dict):
        return []

    columns = list(first_row.keys())
    text_columns = [column for column in columns if "name" in str(column).lower()]
    selected = []
    if text_columns:
        selected.append(text_columns[0])

    for column in columns:
        if column not in selected and column != preferred:
            selected.append(column)
        if len(selected) >= 4:
            break

    if preferred and preferred not in selected and preferred in columns:
        selected.append(preferred)

    return selected[:_MAX_TABLE_COLUMNS]


def _extract_response_text(response):
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                return text.strip()

    return "I could not extract a text answer from the OpenAI response."
