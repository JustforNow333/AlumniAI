import re

from app.services.spreadsheet_service import to_json_safe


HTML_TAG_RE = re.compile(r"<[^>\n]*>")
MAX_BLOCKS = 8
MAX_FOLLOWUPS = 4
MAX_TABLE_COLUMNS = 12
MAX_TABLE_ROWS = 100
MAX_METRICS = 8
MAX_RANKED_ITEMS = 12


def normalize_answer(value, fallback_summary=None):
    parsed = value
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), dict):
        parsed = parsed["answer"]

    if not isinstance(parsed, dict):
        return plain_markdown_answer(fallback_summary or parsed or "I could not format that response."), False

    summary = clean_text(parsed.get("summary") or fallback_summary or "", max_length=1200)
    title = clean_text(parsed.get("title") or "", max_length=100)
    blocks = []

    raw_blocks = parsed.get("blocks") if isinstance(parsed.get("blocks"), list) else []
    for raw_block in raw_blocks[:MAX_BLOCKS]:
        block = normalize_block(raw_block)
        if block:
            blocks.append(block)

    if not blocks and summary:
        blocks.append({"type": "markdown", "content": summary})

    if not summary and not blocks:
        return plain_markdown_answer(fallback_summary or "I could not format that response."), False

    followups = []
    raw_followups = parsed.get("followups") if isinstance(parsed.get("followups"), list) else []
    for item in raw_followups[:MAX_FOLLOWUPS]:
        text = clean_text(item, max_length=120)
        if text:
            followups.append(text)

    return to_json_safe(
        {
            "title": title,
            "summary": summary,
            "blocks": blocks,
            "followups": followups,
        }
    ), True


def normalize_block(raw_block):
    if not isinstance(raw_block, dict):
        return None

    block_type = clean_text(raw_block.get("type"), max_length=40).lower()

    if block_type == "markdown":
        content = clean_text(raw_block.get("content"), max_length=4000)
        return {"type": "markdown", "content": content} if content else None

    if block_type == "table":
        columns = _clean_list(raw_block.get("columns"), MAX_TABLE_COLUMNS)
        if not columns:
            return None

        rows = []
        raw_rows = raw_block.get("rows") if isinstance(raw_block.get("rows"), list) else []
        for raw_row in raw_rows[:MAX_TABLE_ROWS]:
            if isinstance(raw_row, dict):
                row = [clean_text(raw_row.get(column), max_length=300) for column in columns]
            elif isinstance(raw_row, (list, tuple)):
                row = [clean_text(value, max_length=300) for value in raw_row[: len(columns)]]
                row.extend([""] * (len(columns) - len(row)))
            else:
                row = [clean_text(raw_row, max_length=300)]
                row.extend([""] * (len(columns) - 1))
            rows.append(row)

        return {
            "type": "table",
            "title": clean_text(raw_block.get("title"), max_length=120),
            "columns": columns,
            "rows": rows,
            "caption": clean_text(raw_block.get("caption"), max_length=240),
        }

    if block_type == "metrics":
        items = []
        raw_items = raw_block.get("items") if isinstance(raw_block.get("items"), list) else []
        for raw_item in raw_items[:MAX_METRICS]:
            if not isinstance(raw_item, dict):
                continue
            label = clean_text(raw_item.get("label"), max_length=80)
            value = clean_text(raw_item.get("value"), max_length=120)
            if label or value:
                items.append({"label": label, "value": value})
        return {"type": "metrics", "items": items} if items else None

    if block_type == "ranked_list":
        items = []
        raw_items = raw_block.get("items") if isinstance(raw_block.get("items"), list) else []
        for raw_item in raw_items[:MAX_RANKED_ITEMS]:
            if not isinstance(raw_item, dict):
                continue
            label = clean_text(raw_item.get("label"), max_length=140)
            value = clean_text(raw_item.get("value"), max_length=120)
            description = clean_text(raw_item.get("description"), max_length=300)
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
            "title": clean_text(raw_block.get("title"), max_length=120),
            "items": items,
        }

    return None


def plain_markdown_answer(summary, title="", followups=None):
    summary = clean_text(summary)
    answer = {
        "title": clean_text(title, max_length=100),
        "summary": summary,
        "blocks": [{"type": "markdown", "content": summary}] if summary else [],
        "followups": [clean_text(item, max_length=120) for item in (followups or []) if clean_text(item)],
    }
    normalized, _valid = normalize_answer(answer, fallback_summary=summary)
    return normalized


def deterministic_answer_from_results(question, plan, operation_results, dataset_context):
    ok_results = [result for result in operation_results if result.get("status") == "ok"]
    error_results = [result for result in operation_results if result.get("status") == "error"]
    assumptions = []
    warnings = []
    for result in operation_results:
        assumptions.extend(result.get("assumptions") or [])
        warnings.extend(result.get("warnings") or [])

    if not operation_results:
        reason = plan.get("cannot_answer_reason") if isinstance(plan, dict) else ""
        summary = reason or "I could not create a valid analysis plan for that question."
        return plain_markdown_answer(summary, title="Analysis Not Run")

    if error_results and not ok_results:
        summary = error_results[0].get("error") or "The requested analysis could not be completed."
        return plain_markdown_answer(
            summary,
            title="Analysis Error",
            followups=["Try a different column name", "Summarize this dataset", "Show missing values"],
        )

    blocks = []
    metric_items = []
    row_result = None
    people_row_sections = []
    ranked_result = None

    for result in ok_results:
        metrics = result.get("metrics") or {}
        is_people_filter = result.get("intent") == "people_filter" and result.get("entity") == "alumni"
        if is_people_filter:
            metric_items.extend(_people_filter_metric_items(result, metrics))
        elif result.get("is_filtered") and "matched_row_count" in metrics:
            metric_items.extend(_filtered_metric_items(metrics))
        else:
            for key, value in metrics.items():
                if key in {"total_rows", "rows_matched", "duplicate_row_count"}:
                    metric_items.append({"label": _labelize(key), "value": format_value(value)})

        if is_people_filter and isinstance(result.get("row_sections"), list) and result["row_sections"]:
            people_row_sections.extend(result["row_sections"])
        elif result.get("rows") and result.get("columns") and row_result is None:
            row_result = result

        if result.get("rows") and result.get("columns") and ranked_result is None:
            ranked_result = result

    if metric_items:
        deduped = []
        seen = set()
        for item in metric_items:
            key = (item["label"], item["value"])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        blocks.append({"type": "metrics", "items": deduped[:MAX_METRICS]})

    if people_row_sections:
        for section in people_row_sections:
            if not isinstance(section, dict) or not section.get("rows") or not section.get("columns"):
                continue
            blocks.append(
                {
                    "type": "table",
                    "title": section.get("title") or "Alumni",
                    "columns": section.get("columns") or [],
                    "rows": section.get("rows") or [],
                    "caption": section.get("caption") or "",
                }
            )
    elif row_result:
        blocks.append(
            {
                "type": "table",
                "title": row_result.get("summary") or "Analysis Results",
                "columns": row_result.get("columns") or [],
                "rows": row_result.get("rows") or [],
                "caption": _table_caption(row_result),
            }
        )

    hint = plan.get("presentation_hint") if isinstance(plan, dict) else ""
    if hint == "ranked_list" and ranked_result:
        columns = ranked_result.get("columns") or []
        rows = ranked_result.get("rows") or []
        items = []
        for row in rows[:MAX_RANKED_ITEMS]:
            label = row[0] if row else ""
            value = row[1] if len(row) > 1 else ""
            description = ", ".join(
                f"{columns[i]}: {format_value(row[i])}"
                for i in range(2, min(len(columns), len(row)))
            )
            items.append(
                {
                    "label": format_value(label),
                    "value": format_value(value),
                    "description": description,
                }
            )
        if items:
            blocks.append({"type": "ranked_list", "title": "Ranked Results", "items": items})

    notes = []
    if assumptions:
        notes.append("Assumptions: " + "; ".join(dict.fromkeys(assumptions)))
    if warnings:
        notes.append("Warnings: " + "; ".join(_dedupe_text(_format_warning(warning) for warning in warnings)))
    if notes:
        blocks.append({"type": "markdown", "content": "\n".join(notes)})

    first_summary = ok_results[0].get("summary") if ok_results else "Analysis complete."
    answer = {
        "title": "Analysis Result",
        "summary": first_summary,
        "blocks": blocks or [{"type": "markdown", "content": first_summary}],
        "followups": _default_followups(dataset_context),
    }
    normalized, _valid = normalize_answer(answer, fallback_summary=first_summary)
    return normalized


def clean_text(value, max_length=1000):
    if value is None:
        return ""
    text = HTML_TAG_RE.sub("", str(value))
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > max_length:
        return text[: max_length - 1].rstrip() + "..."
    return text


def format_value(value):
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
    return clean_text(value, max_length=300)


def _clean_list(values, limit):
    if not isinstance(values, list):
        return []
    cleaned = []
    for value in values[:limit]:
        text = clean_text(value, max_length=120)
        if text:
            cleaned.append(text)
    return cleaned


def _labelize(value):
    return clean_text(value).replace("_", " ").title()


def _filtered_metric_items(metrics):
    items = [
        {"label": "Unique alumni matched", "value": format_value(metrics.get("matched_row_count"))},
        {"label": "Rows shown", "value": format_value(metrics.get("returned_row_count"))},
        {"label": "Total dataset rows", "value": format_value(metrics.get("total_rows"))},
    ]
    raw = metrics.get("raw_match_count")
    matched = metrics.get("matched_row_count")
    if raw is not None and matched is not None and raw != matched:
        items.append({"label": "Raw keyword hits", "value": format_value(raw)})
    if metrics.get("returned_row_count", 0) < metrics.get("matched_row_count", 0):
        items.append({"label": "Display limit", "value": format_value(metrics.get("display_limit"))})
    return items


def _people_filter_metric_items(result, metrics):
    total_matches = result.get("total_matches", metrics.get("total_matches"))
    displayed_count = result.get("displayed_count", metrics.get("displayed_count"))
    answer_label = result.get("answer_label") or "Alumni matching criteria"
    items = [
        {"label": answer_label, "value": format_value(total_matches)},
    ]
    if displayed_count is not None and total_matches is not None and displayed_count != total_matches:
        items.append({"label": "Showing", "value": format_value(displayed_count)})
    uncertain_count = result.get("uncertain_count", metrics.get("uncertain_count"))
    if uncertain_count:
        items.append({"label": "Uncertain possible matches", "value": format_value(uncertain_count)})
    adjacent_count = result.get("adjacent_count", metrics.get("adjacent_count"))
    if adjacent_count:
        items.append({"label": "Adjacent tech-related matches", "value": format_value(adjacent_count)})
    adjacent_included = result.get("adjacent_included_count", metrics.get("adjacent_included_count"))
    if adjacent_included:
        items.append({"label": "Adjacent included", "value": format_value(adjacent_included)})
    return items


def _table_caption(result):
    if not result.get("is_filtered"):
        return "Computed from the full uploaded dataset."

    metrics = result.get("metrics") or {}
    pieces = []
    search_columns = metrics.get("search_columns") or result.get("search_columns") or metrics.get("searched_columns")
    if search_columns:
        pieces.append("Searched columns: " + ", ".join(str(column) for column in search_columns))
    raw = metrics.get("raw_match_count")
    matched = metrics.get("matched_row_count")
    if result.get("intent") == "people_filter" and result.get("entity") == "alumni":
        total = result.get("total_matches", matched)
        displayed = result.get("displayed_count", metrics.get("returned_row_count"))
        if displayed is not None and total is not None and displayed < total:
            pieces.append(f"Showing {format_value(displayed)} of {format_value(total)} matching alumni.")
        uncertain = result.get("uncertain_count", metrics.get("uncertain_count"))
        if uncertain:
            pieces.append(f"{format_value(uncertain)} uncertain possible matches were not counted.")
        adjacent = result.get("adjacent_count", metrics.get("adjacent_count"))
        if adjacent:
            pieces.append(
                f"{format_value(adjacent)} adjacent rows matched broad keywords but were not counted as direct matches."
            )
        return " ".join(pieces) or "Filtered from the full uploaded dataset."
    if raw is not None and matched is not None and raw != matched:
        pieces.append(f"{format_value(raw)} raw keyword hits were deduplicated to {format_value(matched)} matching rows.")
    returned = metrics.get("returned_row_count")
    limit = metrics.get("display_limit")
    if returned is not None and matched is not None and returned < matched:
        pieces.append(f"Showing {format_value(returned)} rows because the display limit is {format_value(limit)}.")
    return " ".join(pieces) or "Filtered from the full uploaded dataset."


def _format_warning(warning):
    if isinstance(warning, dict):
        return clean_text(warning.get("message") or warning)
    return clean_text(warning)


def _dedupe_text(values):
    items = []
    seen = set()
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            items.append(text)
    return items


def _default_followups(dataset_context):
    columns = dataset_context.get("columns") or []
    numeric = next((col["name"] for col in columns if col.get("type") == "number"), None)
    text = next((col["name"] for col in columns if col.get("type") == "text"), None)
    followups = ["Which columns have missing values?", "Summarize this dataset"]
    if numeric:
        followups.append(f"Show top rows by {numeric}")
    if numeric and text:
        followups.append(f"Average {numeric} by {text}")
    return followups[:MAX_FOLLOWUPS]
