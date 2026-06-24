import json

import pytest

from app.services.answer_presenter import (
    _collect_warnings,
    _dedupe_warnings,
    _ensure_notes,
    _ensure_people_filter_blocks,
    planner_failure_answer,
    present_answer,
)
from app.utils.ai_helpers import (
    extract_response_text as _extract_response_text,
    parse_json_response as _parse_json,
)
from app.utils.text_utils import format_warning as _format_warning
from app.services import ai_service


@pytest.fixture(autouse=True)
def disable_ai(monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)


# --- present_answer (deterministic fallback) ---

def test_present_answer_falls_back_without_ai_client():
    result = present_answer(
        question="How many rows?",
        plan={"operations": [{"type": "summary"}]},
        operation_results=[{"status": "ok", "summary": "3 rows", "metrics": {"total_rows": 3}}],
        dataset_context={"dataset_id": "abc", "columns": []},
    )
    assert "summary" in result
    assert "blocks" in result


# --- planner_failure_answer ---

def test_planner_failure_answer_with_reason():
    answer = planner_failure_answer("Column not found.")
    assert "Column not found." in answer["summary"]
    assert answer["title"] == "Analysis Plan Error"
    assert answer["followups"]


def test_planner_failure_answer_without_reason():
    answer = planner_failure_answer("")
    assert "could not create" in answer["summary"].lower()


# --- _collect_warnings ---

def test_collect_warnings():
    results = [
        {"warnings": ["w1", "w2"]},
        {"warnings": None},
        {"warnings": ["w2", "w3"]},
    ]
    warnings = _collect_warnings(results)
    assert "w1" in [_format_warning(w) for w in warnings]
    assert len(warnings) == 3  # w2 deduped


# --- _ensure_notes ---

def test_ensure_notes_adds_assumptions_and_warnings():
    answer = {"summary": "result", "blocks": [{"type": "markdown", "content": "result"}]}
    plan = {"assumptions": ["assume clean data"]}
    results = [{"warnings": ["check null values"]}]
    updated = _ensure_notes(answer, plan, results)
    block_texts = " ".join(b.get("content", "") for b in updated.get("blocks", []))
    assert "assume clean data" in block_texts or "check null values" in block_texts


def test_ensure_notes_no_notes():
    answer = {"summary": "result", "blocks": [{"type": "markdown", "content": "result"}]}
    result = _ensure_notes(answer, {}, [])
    assert result == answer


def test_ensure_notes_already_present():
    note_text = "Assumptions: already here"
    answer = {
        "summary": "result",
        "blocks": [{"type": "markdown", "content": note_text}],
    }
    plan = {"assumptions": ["already here"]}
    result = _ensure_notes(answer, plan, [])
    assert result == answer


def test_ensure_notes_plan_not_dict():
    answer = {"summary": "result", "blocks": []}
    result = _ensure_notes(answer, "bad", [{"warnings": ["w1"]}])
    block_texts = " ".join(b.get("content", "") for b in result.get("blocks", []))
    assert "w1" in block_texts


# --- _ensure_people_filter_blocks ---

def test_ensure_people_filter_blocks_replaces_metrics_and_table():
    answer = {
        "summary": "result",
        "blocks": [
            {"type": "metrics", "items": [{"label": "Wrong", "value": "0"}]},
            {"type": "table", "title": "Wrong", "columns": ["A"], "rows": [["1"]]},
            {"type": "markdown", "content": "keep this"},
        ],
    }
    fallback = {
        "summary": "result",
        "blocks": [
            {"type": "metrics", "items": [{"label": "Correct", "value": "5"}]},
            {"type": "table", "title": "Correct", "columns": ["B"], "rows": [["2"]]},
        ],
    }
    results = [{"intent": "people_filter", "entity": "alumni"}]
    updated = _ensure_people_filter_blocks(answer, results, fallback)
    types = [b["type"] for b in updated["blocks"]]
    assert "metrics" in types
    assert "table" in types
    assert "markdown" in types


def test_ensure_people_filter_blocks_no_match():
    answer = {"summary": "s", "blocks": []}
    result = _ensure_people_filter_blocks(answer, [{"intent": "other"}], {})
    assert result == answer


def test_ensure_people_filter_blocks_no_source_blocks():
    answer = {"summary": "s", "blocks": [{"type": "markdown", "content": "x"}]}
    results = [{"intent": "people_filter", "entity": "alumni"}]
    result = _ensure_people_filter_blocks(answer, results, {"blocks": []})
    assert result == answer


# --- _format_warning ---

def test_format_warning_dict():
    assert _format_warning({"message": "issue"}) == "issue"


def test_format_warning_dict_no_message():
    result = _format_warning({"other": "val"})
    assert "other" in result


def test_format_warning_string():
    assert _format_warning("plain warning") == "plain warning"


# --- _dedupe_warnings ---

def test_dedupe_warnings():
    warnings = ["w1", "w2", "w1", {"message": "w2"}, "w3"]
    deduped = _dedupe_warnings(warnings)
    texts = [_format_warning(w) for w in deduped]
    assert texts == ["w1", "w2", "w3"]


def test_dedupe_warnings_none():
    assert _dedupe_warnings(None) == []


def test_dedupe_warnings_empty_strings():
    assert _dedupe_warnings(["", "  ", "real"]) == ["real"]


# --- _parse_json ---

def test_parse_json_valid():
    result = _parse_json('{"a": 1}')
    assert result == {"a": 1}


def test_parse_json_with_markdown_fences():
    result = _parse_json('```json\n{"a": 1}\n```')
    assert result == {"a": 1}


def test_parse_json_embedded_in_text():
    result = _parse_json('Here is the JSON: {"a": 1} and some trailing text')
    assert result == {"a": 1}


def test_parse_json_invalid_raises():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_json("not json at all")


def test_parse_json_empty():
    with pytest.raises(ValueError):
        _parse_json("")


def test_parse_json_none():
    with pytest.raises(ValueError):
        _parse_json(None)


def test_parse_json_nested_invalid():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_json("prefix {bad json} suffix")


# --- _extract_response_text ---

def test_extract_response_text_from_output_text():
    class FakeResponse:
        output_text = "  hello  "
        output = []

    assert _extract_response_text(FakeResponse()) == "hello"


def test_extract_response_text_from_output_items():
    class FakeContent:
        text = "from-content"

    class FakeItem:
        content = [FakeContent()]

    class FakeResponse:
        output_text = None
        output = [FakeItem()]

    assert _extract_response_text(FakeResponse()) == "from-content"


def test_extract_response_text_empty():
    class FakeResponse:
        output_text = ""
        output = []

    assert _extract_response_text(FakeResponse()) == ""


def test_extract_response_text_no_text_in_content():
    class FakeContent:
        text = None

    class FakeItem:
        content = [FakeContent()]

    class FakeResponse:
        output_text = None
        output = [FakeItem()]

    assert _extract_response_text(FakeResponse()) == ""
