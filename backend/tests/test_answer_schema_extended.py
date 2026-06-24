import pytest

from app.services.answer_schema import (
    MAX_BLOCKS,
    MAX_FOLLOWUPS,
    MAX_METRICS,
    MAX_RANKED_ITEMS,
    MAX_TABLE_COLUMNS,
    MAX_TABLE_ROWS,
    _default_followups,
    _filtered_metric_items,
    _people_filter_metric_items,
    _table_caption,
    clean_text,
    deterministic_answer_from_results,
    format_value,
    normalize_answer,
    normalize_block,
    plain_markdown_answer,
)


# --- normalize_answer ---

def test_normalize_answer_unwraps_nested():
    val = {"answer": {"summary": "hi", "blocks": [{"type": "markdown", "content": "hi"}]}}
    answer, valid = normalize_answer(val)
    assert valid
    assert answer["summary"] == "hi"


def test_normalize_answer_not_dict():
    answer, valid = normalize_answer("just a string")
    assert not valid
    assert "just a string" in answer["summary"] or "could not format" in answer["summary"].lower()


def test_normalize_answer_no_summary_no_blocks():
    answer, valid = normalize_answer({"summary": "", "blocks": []})
    assert not valid


def test_normalize_answer_uses_fallback_summary():
    answer, valid = normalize_answer({"blocks": []}, fallback_summary="fallback")
    assert "fallback" in answer["summary"]


def test_normalize_answer_truncates_followups():
    val = {
        "summary": "s",
        "blocks": [{"type": "markdown", "content": "s"}],
        "followups": [f"q{i}" for i in range(10)],
    }
    answer, valid = normalize_answer(val)
    assert valid
    assert len(answer["followups"]) <= MAX_FOLLOWUPS


def test_normalize_answer_empty_followup_items():
    val = {
        "summary": "s",
        "blocks": [{"type": "markdown", "content": "s"}],
        "followups": ["", None, "real"],
    }
    answer, valid = normalize_answer(val)
    assert answer["followups"] == ["real"]


# --- normalize_block ---

def test_normalize_block_not_dict():
    assert normalize_block("string") is None


def test_normalize_block_unknown_type():
    assert normalize_block({"type": "chart", "data": []}) is None


def test_normalize_block_markdown_empty():
    assert normalize_block({"type": "markdown", "content": ""}) is None


def test_normalize_block_table_no_columns():
    assert normalize_block({"type": "table", "columns": [], "rows": []}) is None


def test_normalize_block_table_with_dict_rows():
    block = normalize_block({
        "type": "table",
        "columns": ["A", "B"],
        "rows": [{"A": "1", "B": "2"}],
    })
    assert block is not None
    assert block["rows"] == [["1", "2"]]


def test_normalize_block_table_with_list_rows_shorter():
    block = normalize_block({
        "type": "table",
        "columns": ["A", "B", "C"],
        "rows": [["1"]],
    })
    assert block["rows"] == [["1", "", ""]]


def test_normalize_block_table_with_scalar_row():
    block = normalize_block({
        "type": "table",
        "columns": ["A", "B"],
        "rows": ["just-a-string"],
    })
    assert block["rows"] == [["just-a-string", ""]]


def test_normalize_block_metrics_empty():
    assert normalize_block({"type": "metrics", "items": []}) is None


def test_normalize_block_metrics_non_dict_items():
    block = normalize_block({
        "type": "metrics",
        "items": ["not-a-dict", {"label": "count", "value": "5"}],
    })
    assert block is not None
    assert len(block["items"]) == 1


def test_normalize_block_ranked_list_empty():
    assert normalize_block({"type": "ranked_list", "items": []}) is None


def test_normalize_block_ranked_list_non_dict_items():
    block = normalize_block({
        "type": "ranked_list",
        "items": ["bad", {"label": "top", "value": "1", "description": "first"}],
    })
    assert block is not None
    assert len(block["items"]) == 1


def test_normalize_block_ranked_list_with_description():
    block = normalize_block({
        "type": "ranked_list",
        "title": "Top Items",
        "items": [{"label": "A", "value": "1", "description": "desc"}],
    })
    assert block["title"] == "Top Items"
    assert block["items"][0]["description"] == "desc"


# --- clean_text ---

def test_clean_text_strips_html():
    assert clean_text("<b>bold</b>") == "bold"


def test_clean_text_none():
    assert clean_text(None) == ""


def test_clean_text_truncates():
    long = "a" * 2000
    result = clean_text(long, max_length=100)
    # max_length-1 chars + "..." = max_length+2 total
    assert len(result) <= max(100, 103)
    assert result.endswith("...")


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_collapses_newlines():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_clean_text_removes_null_bytes():
    assert clean_text("he\x00llo") == "hello"


# --- format_value ---

def test_format_value_none():
    assert format_value(None) == "Not available"


def test_format_value_bool_true():
    assert format_value(True) == "true"


def test_format_value_bool_false():
    assert format_value(False) == "false"


def test_format_value_int():
    assert format_value(1234) == "1,234"


def test_format_value_float_integer():
    assert format_value(5.0) == "5"


def test_format_value_float_decimal():
    assert format_value(3.14159) == "3.14"


def test_format_value_string():
    assert format_value("hello") == "hello"


# --- _filtered_metric_items ---

def test_filtered_metric_items_basic():
    metrics = {
        "matched_row_count": 10,
        "returned_row_count": 10,
        "total_rows": 100,
    }
    items = _filtered_metric_items(metrics)
    labels = [i["label"] for i in items]
    assert "Unique alumni matched" in labels
    assert "Rows shown" in labels
    assert "Total dataset rows" in labels


def test_filtered_metric_items_with_raw_mismatch():
    metrics = {
        "matched_row_count": 10,
        "returned_row_count": 10,
        "total_rows": 100,
        "raw_match_count": 15,
    }
    items = _filtered_metric_items(metrics)
    labels = [i["label"] for i in items]
    assert "Raw keyword hits" in labels


def test_filtered_metric_items_with_display_limit():
    metrics = {
        "matched_row_count": 20,
        "returned_row_count": 10,
        "total_rows": 100,
        "display_limit": 10,
    }
    items = _filtered_metric_items(metrics)
    labels = [i["label"] for i in items]
    assert "Display limit" in labels


# --- _people_filter_metric_items ---

def test_people_filter_metric_items_basic():
    result = {"total_matches": 5, "displayed_count": 5, "answer_label": "Alumni in tech"}
    items = _people_filter_metric_items(result, {})
    assert items[0]["label"] == "Alumni in tech"
    assert items[0]["value"] == "5"


def test_people_filter_metric_items_with_displayed_diff():
    result = {"total_matches": 10, "displayed_count": 5}
    items = _people_filter_metric_items(result, {})
    labels = [i["label"] for i in items]
    assert "Showing" in labels


def test_people_filter_metric_items_with_uncertain():
    result = {"total_matches": 10, "displayed_count": 10, "uncertain_count": 3}
    items = _people_filter_metric_items(result, {})
    labels = [i["label"] for i in items]
    assert "Uncertain not counted" in labels


def test_people_filter_metric_items_with_adjacent():
    result = {"total_matches": 10, "displayed_count": 10, "adjacent_count": 2}
    items = _people_filter_metric_items(result, {})
    labels = [i["label"] for i in items]
    assert "Adjacent not counted" in labels


def test_people_filter_metric_items_with_adjacent_included():
    result = {"total_matches": 10, "displayed_count": 10, "adjacent_included_count": 1}
    items = _people_filter_metric_items(result, {})
    labels = [i["label"] for i in items]
    assert "Adjacent included" in labels


def test_people_filter_default_label():
    result = {"total_matches": 5}
    items = _people_filter_metric_items(result, {})
    assert items[0]["label"] == "Alumni matching criteria"


# --- _table_caption ---

def test_table_caption_not_filtered():
    result = {"is_filtered": False}
    assert _table_caption(result) == "Computed from the full uploaded dataset."


def test_table_caption_people_filter_with_display_limit():
    result = {
        "is_filtered": True,
        "intent": "people_filter",
        "entity": "alumni",
        "total_matches": 20,
        "displayed_count": 10,
        "metrics": {"search_columns": ["Occupation"]},
    }
    caption = _table_caption(result)
    assert "Showing" in caption


def test_table_caption_people_filter_with_uncertain():
    result = {
        "is_filtered": True,
        "intent": "people_filter",
        "entity": "alumni",
        "uncertain_count": 5,
        "metrics": {},
    }
    caption = _table_caption(result)
    assert "uncertain" in caption.lower()


def test_table_caption_people_filter_with_adjacent():
    result = {
        "is_filtered": True,
        "intent": "people_filter",
        "entity": "alumni",
        "adjacent_count": 3,
        "metrics": {},
    }
    caption = _table_caption(result)
    assert "adjacent" in caption.lower()


def test_table_caption_non_people_dedup():
    result = {
        "is_filtered": True,
        "metrics": {
            "raw_match_count": 15,
            "matched_row_count": 10,
            "returned_row_count": 10,
        },
    }
    caption = _table_caption(result)
    assert "deduplicated" in caption.lower()


def test_table_caption_non_people_display_limit():
    result = {
        "is_filtered": True,
        "metrics": {
            "matched_row_count": 20,
            "returned_row_count": 10,
            "display_limit": 10,
        },
    }
    caption = _table_caption(result)
    assert "display limit" in caption.lower()


def test_table_caption_filtered_default():
    result = {"is_filtered": True, "metrics": {}}
    caption = _table_caption(result)
    assert "Filtered" in caption


# --- _default_followups ---

def test_default_followups_with_numeric_and_text():
    cols = [
        {"name": "Age", "type": "number"},
        {"name": "Name", "type": "text"},
    ]
    followups = _default_followups({"columns": cols})
    assert len(followups) <= MAX_FOLLOWUPS
    assert any("Age" in f for f in followups)


def test_default_followups_no_columns():
    followups = _default_followups({"columns": []})
    assert "Which columns have missing values?" in followups


# --- deterministic_answer_from_results ---

def test_deterministic_no_results():
    answer = deterministic_answer_from_results("q", {}, [], {})
    assert "could not create" in answer["summary"].lower() or "Not Run" in answer["title"]


def test_deterministic_no_results_with_reason():
    plan = {"cannot_answer_reason": "Ambiguous question."}
    answer = deterministic_answer_from_results("q", plan, [], {})
    assert "Ambiguous" in answer["summary"]


def test_deterministic_errors_only():
    results = [{"status": "error", "error": "Column X not found."}]
    answer = deterministic_answer_from_results("q", {}, results, {})
    assert "Column X" in answer["summary"]


def test_deterministic_ranked_list():
    results = [
        {
            "status": "ok",
            "summary": "Top items",
            "metrics": {},
            "columns": ["Name", "Score", "Category"],
            "rows": [["Alice", 95, "A"], ["Bob", 80, "B"]],
        }
    ]
    plan = {"presentation_hint": "ranked_list"}
    answer = deterministic_answer_from_results("q", plan, results, {"columns": []})
    block_types = [b["type"] for b in answer.get("blocks", [])]
    assert "ranked_list" in block_types


def test_deterministic_with_warnings_and_assumptions():
    results = [
        {
            "status": "ok",
            "summary": "Done",
            "metrics": {},
            "assumptions": ["Data is clean"],
            "warnings": ["Some nulls ignored"],
        }
    ]
    answer = deterministic_answer_from_results("q", {}, results, {"columns": []})
    block_texts = " ".join(b.get("content", "") for b in answer.get("blocks", []))
    assert "Data is clean" in block_texts or "Some nulls" in block_texts


# --- plain_markdown_answer ---

def test_plain_markdown_answer():
    answer = plain_markdown_answer("Summary text", title="Title", followups=["Q1"])
    assert answer["title"] == "Title"
    assert answer["summary"] == "Summary text"
    assert "Q1" in answer["followups"]
