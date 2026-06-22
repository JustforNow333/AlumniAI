"""Shared helpers for parsing OpenAI model responses."""

import json
import re


def parse_json_response(text):
    """Parse a JSON value from model output, stripping markdown fences."""
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
                raise ValueError("Model returned invalid JSON.") from nested_exc
        raise ValueError("Model returned invalid JSON.") from exc


def extract_response_text(response):
    """Extract the first text content from an OpenAI Responses API result."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                return text.strip()
    return ""
