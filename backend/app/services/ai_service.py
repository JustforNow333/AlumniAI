import json
import os
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
""".strip()


def build_ai_context(df):
    basic_summary = create_basic_summary(df, include_preview=False)

    return to_json_safe(
        {
            "shape": {
                "rows": basic_summary["rows"],
                "columns": basic_summary["columns"],
            },
            "column_names": basic_summary["column_names"],
            "column_types": basic_summary["column_types"],
            "missing_values": basic_summary["missing_values"],
            "preview_first_10_rows": dataframe_preview(df, limit=10),
            "dataframe_summary": summarize_dataframe(df),
        }
    )


def generate_answer(question, dataset_context, operation=None, result=None):
    if client is None:
        return _fallback_answer(operation, result)

    model = os.getenv("OPENAI_MODEL", "gpt-5.1")
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
            max_output_tokens=500,
            temperature=0.2,
            tools=[],
        )
    except Exception as exc:
        fallback = _fallback_answer(operation, result)
        return f"{fallback} OpenAI narration was unavailable: {exc}"

    return _extract_response_text(response)


def _fallback_answer(operation=None, result=None):
    operation_type = operation.get("type") if isinstance(operation, dict) else None

    if operation_type == "summarize_dataframe" and isinstance(result, dict):
        total_missing = result.get("total_missing_values", 0)
        rows = result.get("rows", 0)
        columns = result.get("columns", 0)
        missing_label = "value" if total_missing == 1 else "values"
        return (
            f"The dataset has {rows} rows, {columns} columns, and "
            f"{total_missing} missing {missing_label}."
        )

    if operation_type == "group_by_aggregate" and isinstance(result, dict):
        aggregation = operation.get("aggregation")
        group_col = operation.get("group_col")
        value_col = operation.get("value_col")
        if result:
            top_group, top_value = next(iter(result.items()))
            return (
                f"{top_group} is the top {group_col} by {aggregation} "
                f"{value_col}, with a value of {top_value}."
            )
        return f"I grouped {value_col} by {group_col}, but no result rows were returned."

    if operation_type == "top_rows" and isinstance(result, dict):
        rows = result.get("rows") or []
        sort_col = result.get("sort_col") or operation.get("sort_col")
        if rows:
            return f"The top row by {sort_col} has {sort_col} = {rows[0].get(sort_col)}."
        return f"No rows were available to rank by {sort_col}."

    if operation_type == "correlation" and isinstance(result, dict):
        coefficient = result.get("correlation")
        col1 = result.get("col1")
        col2 = result.get("col2")
        if coefficient is None:
            return f"There is not enough overlapping numeric data to correlate {col1} and {col2}."
        return f"{col1} and {col2} have a correlation of {coefficient:.2f}."

    if operation_type == "summarize_column" and isinstance(result, dict):
        column = result.get("column")
        result_type = result.get("type")
        if result_type == "numeric":
            return (
                f"{column} has {result.get('count')} values, an average of "
                f"{result.get('mean')}, and a median of {result.get('median')}."
            )
        if result_type == "date":
            return (
                f"{column} ranges from {result.get('earliest')} to "
                f"{result.get('latest')}."
            )
        return (
            f"{column} has {result.get('unique_count')} unique values. "
            f"Top values are shown below."
        )

    if operation_type == "analysis_error" and isinstance(result, dict):
        return result.get("error", "I could not run that analysis.")

    return (
        "I could not map that question to a supported safe operation. Try asking for "
        "missing values, a column summary, top rows, a grouped total, or a correlation."
    )


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
