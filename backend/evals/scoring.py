from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


EVAL_ONLY_PREFIXES = ("expected_", "eval_")
EVAL_ONLY_COLUMNS = {"expected_industry"}
DEFAULT_FORBIDDEN_COLUMNS = {
    "expected_industry",
    "match reason",
    "matched_column",
    "matched_term",
    "internal_reason",
    "classification",
    "confidence",
}
DEFAULT_PRECISION_THRESHOLD = 0.90
DEFAULT_RECALL_THRESHOLD = 0.75
SCORED_FROM_OPERATION_RESULTS = "operation_results"
SCORED_FROM_ANSWER_TABLE = "answer_table"
SCORED_FROM_RESPONSE_PAYLOAD = "response_payload"
SCORED_FROM_TEXT_FALLBACK = "text_fallback"


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL eval cases and validate the minimal runner contract."""
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as case_file:
        for line_number, line in enumerate(case_file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(case, dict):
                raise ValueError(f"Case on line {line_number} must be a JSON object.")
            if not case.get("id"):
                raise ValueError(f"Case on line {line_number} is missing id.")
            if not case.get("question"):
                raise ValueError(f"Case {case.get('id')} is missing question.")
            cases.append(case)
    return cases


def load_gold_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def make_app_view_dataset(gold_df: pd.DataFrame) -> pd.DataFrame:
    """Return the CSV view uploaded to the app, with answer-key columns removed."""
    columns = [column for column in gold_df.columns if not is_eval_only_column(column)]
    return gold_df.loc[:, columns].copy()


def is_eval_only_column(column: Any) -> bool:
    normalized = str(column or "").strip().casefold()
    return normalized in EVAL_ONLY_COLUMNS or normalized.startswith(EVAL_ONLY_PREFIXES)


def get_full_name(row: Any) -> str:
    data = _row_mapping(row)
    first = _value_for_alias(data, {"first name", "first_name", "firstname", "given name"})
    last = _value_for_alias(data, {"last name", "last_name", "lastname", "surname", "family name"})
    if first and last:
        return f"{first} {last}".strip()
    full = _value_for_alias(data, {"full name", "name", "person name", "person_name", "alumni name"})
    if full:
        return full
    preferred = _value_for_alias(data, {"preferred name", "preferred_name", "nickname"})
    return preferred or ""


def normalize_name(name: Any) -> str:
    text = "" if name is None else str(name)
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def normalize_employer(value: Any) -> str:
    return _normalize_text(value)


def normalize_title(value: Any) -> str:
    return _normalize_text(value)


def extract_normalized_response(response_json: Any) -> dict[str, Any]:
    """Normalize current and legacy `/api/ask` responses into a stable shape.

    Current responses contain `operation_results` with structured `rows`,
    `columns`, and `metrics`. Older/fallback responses may only have answer
    blocks or prose; those paths are marked with a lower-confidence source.
    """
    raw = response_json if isinstance(response_json, dict) else {}
    answer = raw.get("answer") if isinstance(raw.get("answer"), dict) else {}
    answer_text = _collect_answer_text(raw, answer)

    table_source = _best_operation_result(raw)
    source = SCORED_FROM_OPERATION_RESULTS if table_source else ""
    if not table_source:
        table_source = _best_response_payload_result(raw)
        source = SCORED_FROM_RESPONSE_PAYLOAD if table_source else ""
    if not table_source:
        table_source = _best_answer_table(answer)
        source = SCORED_FROM_ANSWER_TABLE if table_source else SCORED_FROM_TEXT_FALLBACK

    rows: list[dict[str, Any]] = []
    displayed_columns: list[str] = []
    if table_source:
        displayed_columns = _extract_source_columns(table_source)
        rows = _rows_to_dicts(table_source.get("rows") or [], displayed_columns)

    metrics = _extract_metrics(raw, table_source or {}, answer)
    primary_count = _first_int(
        metrics,
        [
            "total_matches",
            "matched_row_count",
            "rows_matched",
            "count",
            "duplicate_row_count",
            "total_rows",
        ],
    )
    displayed_count = _first_int(metrics, ["displayed_count", "returned_row_count", "rows_returned"])
    if displayed_count is None and rows:
        displayed_count = len(rows)

    return {
        "answer_text": answer_text,
        "rows": rows,
        "displayed_columns": displayed_columns,
        "count": primary_count,
        "displayed_count": displayed_count,
        "metrics": metrics,
        "raw_response": raw,
        "extraction_source": source,
        "scored_from": source,
        "lower_confidence_extraction": source == SCORED_FROM_TEXT_FALLBACK,
    }


def extract_rows(normalized_response: dict[str, Any]) -> list[dict[str, Any]]:
    return list(normalized_response.get("rows") or [])


def extract_displayed_columns(normalized_response: dict[str, Any]) -> list[str]:
    return list(normalized_response.get("displayed_columns") or [])


def extract_answer_text(normalized_response: dict[str, Any]) -> str:
    return str(normalized_response.get("answer_text") or "")


def compute_expected_rows(case: dict[str, Any], gold_df: pd.DataFrame) -> pd.DataFrame:
    if case.get("expected_filter") is not None:
        mask = _filter_mask(gold_df, case["expected_filter"])
        return gold_df[mask].copy()

    if case.get("expected_count_mode") == "from_expected_industry" or case.get("expected_industry"):
        expected = case.get("expected_industry")
        if isinstance(expected, list):
            expected_values = {str(value).casefold() for value in expected}
            mask = gold_df["expected_industry"].astype("string").str.casefold().isin(expected_values)
        else:
            mask = gold_df["expected_industry"].astype("string").str.casefold().eq(str(expected).casefold())
        return gold_df[mask.fillna(False)].copy()

    if "expected_count" in case and int(case.get("expected_count") or 0) == 0:
        return gold_df.iloc[0:0].copy()

    return gold_df.iloc[0:0].copy()


def score_case(
    case: dict[str, Any],
    normalized_response: dict[str, Any],
    gold_df: pd.DataFrame,
) -> dict[str, Any]:
    expected_rows = compute_expected_rows(case, gold_df)
    expected_count = int(case.get("expected_count", len(expected_rows)))
    expected_names = _name_set(expected_rows.to_dict(orient="records"))
    known_names = _name_set(gold_df.to_dict(orient="records"))
    known_name_display = _name_display_map(gold_df.to_dict(orient="records"))

    rows = extract_rows(normalized_response)
    returned_names = [get_full_name(row) for row in rows]
    returned_names = [name for name in returned_names if normalize_name(name)]
    if not returned_names and normalized_response.get("lower_confidence_extraction"):
        returned_names = _extract_known_names_from_text(extract_answer_text(normalized_response), known_name_display)

    hallucinated_names = detect_hallucinated_names(returned_names, known_names)
    precision, recall = compute_precision_recall(returned_names, expected_names)
    returned_count = len(rows) if rows else len(returned_names)
    displayed_count = normalized_response.get("displayed_count")
    app_count = normalized_response.get("count")

    false_positive_names = sorted(
        _display_names(set(_normalize_names(returned_names)) - expected_names, known_name_display)
    )
    false_negative_names = sorted(
        _display_names(expected_names - set(_normalize_names(returned_names)), known_name_display)
    )

    failures: list[str] = []
    failure_categories: set[str] = set()
    precision_threshold = float(case.get("precision_threshold", DEFAULT_PRECISION_THRESHOLD))
    recall_threshold = float(case.get("recall_threshold", DEFAULT_RECALL_THRESHOLD))

    if hallucinated_names:
        _add_failure(
            failures,
            failure_categories,
            "hallucinated_names",
            f"Hallucinated names returned: {', '.join(hallucinated_names[:10])}.",
        )

    expect_rows = case.get("expect_rows")
    if expect_rows is None:
        expect_rows = bool(re.search(r"\b(show|find|which|who|list)\b", str(case.get("question", "")).casefold()))

    score_precision_recall = case.get(
        "score_precision_recall",
        _should_score_precision_recall(case) and (expect_rows or bool(rows) or bool(returned_names)),
    )
    if score_precision_recall:
        if precision < precision_threshold:
            _add_failure(
                failures,
                failure_categories,
                _selection_failure_category(case),
                f"Precision {precision:.2f} was below threshold {precision_threshold:.2f}.",
            )
        if recall < recall_threshold:
            _add_failure(
                failures,
                failure_categories,
                _selection_failure_category(case),
                f"Recall {recall:.2f} was below threshold {recall_threshold:.2f}.",
            )

    if case.get("exact_match"):
        observed_count = app_count if app_count is not None else returned_count
        if observed_count != expected_count:
            _add_failure(
                failures,
                failure_categories,
                "count_mismatch",
                f"Expected count {expected_count}, but app reported {observed_count}.",
            )
        if expect_rows and returned_count != expected_count:
            _add_failure(
                failures,
                failure_categories,
                "count_mismatch",
                f"Expected {expected_count} returned rows, but response included {returned_count}.",
            )
        if expect_rows and rows and set(_normalize_names(returned_names)) != expected_names:
            _add_failure(
                failures,
                failure_categories,
                "row_selection",
                "Returned row names did not exactly match the expected filtered rows.",
            )

    if rows and displayed_count is not None and int(displayed_count) != len(rows):
        _add_failure(
            failures,
            failure_categories,
            "count_mismatch",
            f"Displayed count was {displayed_count}, but the response included {len(rows)} rows.",
        )

    missing_columns = check_required_columns(
        extract_displayed_columns(normalized_response),
        case.get("required_columns") or [],
    )
    if missing_columns:
        _add_failure(
            failures,
            failure_categories,
            "response_parsing" if normalized_response.get("lower_confidence_extraction") else "forbidden_columns",
            f"Missing required displayed columns: {', '.join(missing_columns)}.",
        )

    forbidden_columns = list(DEFAULT_FORBIDDEN_COLUMNS)
    forbidden_columns.extend(case.get("forbidden_columns") or [])
    present_forbidden = check_forbidden_columns(
        extract_displayed_columns(normalized_response),
        forbidden_columns,
    )
    if present_forbidden:
        _add_failure(
            failures,
            failure_categories,
            "forbidden_columns",
            f"Forbidden displayed columns were present: {', '.join(present_forbidden)}.",
        )

    missing_phrases = check_required_phrases(
        extract_answer_text(normalized_response),
        case.get("required_phrases") or [],
    )
    if missing_phrases:
        _add_failure(
            failures,
            failure_categories,
            "response_parsing",
            f"Missing required answer phrases: {', '.join(missing_phrases)}.",
        )

    present_forbidden_phrases = check_forbidden_phrases(
        extract_answer_text(normalized_response),
        case.get("forbidden_phrases") or [],
    )
    if present_forbidden_phrases:
        _add_failure(
            failures,
            failure_categories,
            "response_parsing",
            f"Forbidden answer phrases were present: {', '.join(present_forbidden_phrases)}.",
        )

    missing_includes = [
        name
        for name in case.get("must_include_names") or []
        if normalize_name(name) not in set(_normalize_names(returned_names))
    ]
    if missing_includes:
        _add_failure(
            failures,
            failure_categories,
            _selection_failure_category(case),
            f"Required names were not returned: {', '.join(missing_includes)}.",
        )

    unexpected_names = [
        name
        for name in case.get("must_exclude_names") or []
        if normalize_name(name) in set(_normalize_names(returned_names))
    ]
    if unexpected_names:
        _add_failure(
            failures,
            failure_categories,
            _selection_failure_category(case),
            f"Excluded names were returned: {', '.join(unexpected_names)}.",
        )

    excluded_industries = {str(value).casefold() for value in case.get("must_exclude_industries") or []}
    excluded_industry_names = _returned_names_in_industries(rows, gold_df, excluded_industries)
    if excluded_industry_names:
        _add_failure(
            failures,
            failure_categories,
            "classification",
            "Returned names from excluded industries: "
            + ", ".join(sorted(excluded_industry_names)[:10])
            + ".",
        )

    if expected_count == 0:
        observed_count = app_count if app_count is not None else returned_count
        if observed_count != 0 or returned_count != 0:
            _add_failure(
                failures,
                failure_categories,
                "count_mismatch",
                f"Expected zero results, but app reported {observed_count} and returned {returned_count} rows."
            )
        if not _answer_says_zero_or_none(extract_answer_text(normalized_response)):
            _add_failure(
                failures,
                failure_categories,
                "response_parsing",
                "Expected a clear zero-result answer, but the answer text did not say none or zero.",
            )

    if normalized_response.get("lower_confidence_extraction"):
        failure_categories.add("response_parsing")

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "question": case.get("question"),
        "passed": not failures,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "precision_threshold": precision_threshold,
        "recall_threshold": recall_threshold,
        "expected_count": expected_count,
        "expected_unique_names": len(expected_names),
        "returned_count": returned_count,
        "displayed_count": displayed_count,
        "app_count": app_count,
        "failures": failures,
        "failure_categories": sorted(failure_categories),
        "false_positives": false_positive_names[:25],
        "false_negatives_sample": false_negative_names[:25],
        "hallucinated_names": hallucinated_names,
        "displayed_columns": extract_displayed_columns(normalized_response),
        "extraction_source": normalized_response.get("extraction_source"),
        "scored_from": normalized_response.get("scored_from") or normalized_response.get("extraction_source"),
        "lower_confidence_extraction": bool(normalized_response.get("lower_confidence_extraction")),
        "raw_answer_excerpt": _excerpt(extract_answer_text(normalized_response), 700),
    }


def apply_execution_expectations(case: dict[str, Any], scored: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    execution = case.get("execution") if isinstance(case.get("execution"), dict) else {}
    model_calls = str(execution.get("model_calls") or "allowed").lower()
    total_calls = int(trace.get("total_model_calls") or 0)
    classifier_calls = int(trace.get("llm_classifier_calls") or 0)
    final_calls = int(trace.get("final_model_synthesis_calls") or 0)

    failures = list(scored.get("failures") or [])
    categories = set(scored.get("failure_categories") or [])

    if model_calls == "disallowed" and total_calls:
        _add_failure(
            failures,
            categories,
            "execution_behavior",
            f"Model calls were disallowed for this case, but {total_calls} model call(s) occurred.",
        )
    elif model_calls == "required" and total_calls == 0:
        _add_failure(
            failures,
            categories,
            "execution_behavior",
            "Model calls were required for this case, but none occurred.",
        )

    classifier_policy = str(execution.get("llm_classifier") or "allowed").lower()
    if classifier_policy == "disallowed" and classifier_calls:
        _add_failure(
            failures,
            categories,
            "execution_behavior",
            f"LLM classifier calls were disallowed for this case, but {classifier_calls} occurred.",
        )
    elif classifier_policy == "required" and classifier_calls == 0:
        _add_failure(
            failures,
            categories,
            "execution_behavior",
            "LLM classifier calls were required for this case, but none occurred.",
        )

    final_policy = str(execution.get("final_model_synthesis") or "allowed").lower()
    if final_policy == "disallowed" and final_calls:
        _add_failure(
            failures,
            categories,
            "execution_behavior",
            f"Final model synthesis calls were disallowed for this case, but {final_calls} occurred.",
        )
    elif final_policy == "required" and final_calls == 0:
        _add_failure(
            failures,
            categories,
            "execution_behavior",
            "Final model synthesis calls were required for this case, but none occurred.",
        )

    expected_scored_from = execution.get("scored_from")
    if expected_scored_from and scored.get("scored_from") != expected_scored_from:
        _add_failure(
            failures,
            categories,
            "response_parsing",
            f"Expected scoring from {expected_scored_from}, but scored from {scored.get('scored_from')}.",
        )

    scored = dict(scored)
    scored["failures"] = failures
    scored["failure_categories"] = sorted(categories)
    scored["passed"] = not failures
    return scored


def detect_hallucinated_names(returned_names: list[str], known_names: set[str] | list[str]) -> list[str]:
    known_normalized = set(_normalize_names(known_names))
    hallucinations = []
    seen = set()
    for name in returned_names:
        normalized = normalize_name(name)
        if not normalized or normalized in known_normalized or normalized in seen:
            continue
        seen.add(normalized)
        hallucinations.append(str(name))
    return hallucinations


def compute_precision_recall(
    returned_names: list[str] | set[str],
    expected_names: list[str] | set[str],
) -> tuple[float, float]:
    returned = set(_normalize_names(returned_names))
    expected = set(_normalize_names(expected_names))

    if not returned:
        precision = 1.0
    else:
        precision = len(returned & expected) / len(returned)

    if not expected:
        recall = 1.0 if not returned else 0.0
    else:
        recall = len(returned & expected) / len(expected)

    return precision, recall


def check_required_columns(displayed_columns: list[str], required_columns: list[str]) -> list[str]:
    missing = []
    for required in required_columns:
        if not any(_columns_equivalent(column, required) for column in displayed_columns):
            missing.append(str(required))
    return missing


def check_forbidden_columns(displayed_columns: list[str], forbidden_columns: list[str]) -> list[str]:
    present = []
    for column in displayed_columns:
        if is_eval_only_column(column):
            present.append(str(column))
            continue
        for forbidden in forbidden_columns:
            if _columns_equivalent(column, forbidden):
                present.append(str(column))
                break
    return sorted(set(present))


def check_required_phrases(answer_text: str, required_phrases: list[str]) -> list[str]:
    haystack = str(answer_text or "").casefold()
    return [phrase for phrase in required_phrases if str(phrase).casefold() not in haystack]


def check_forbidden_phrases(answer_text: str, forbidden_phrases: list[str]) -> list[str]:
    haystack = str(answer_text or "").casefold()
    return [phrase for phrase in forbidden_phrases if str(phrase).casefold() in haystack]


def _add_failure(failures: list[str], categories: set[str], category: str, message: str) -> None:
    failures.append(message)
    categories.add(category)


def _selection_failure_category(case: dict[str, Any]) -> str:
    if case.get("category") in {"industry_classification", "messy_data"} or case.get("expected_industry"):
        return "classification"
    if case.get("eval_kind") == "direct_classifier":
        return "classification"
    return "row_selection"


def _best_operation_result(raw: dict[str, Any]) -> dict[str, Any] | None:
    candidates = raw.get("operation_results")
    if not isinstance(candidates, list):
        candidates = []
    if isinstance(raw.get("result"), dict):
        candidates = candidates + [raw["result"]]

    for result in candidates:
        if isinstance(result, dict) and result.get("rows") is not None and result.get("columns") is not None:
            return result
    return None


def _best_response_payload_result(raw: dict[str, Any]) -> dict[str, Any] | None:
    payload = raw.get("response_payload") if isinstance(raw.get("response_payload"), dict) else None
    if not payload:
        return None
    return _best_operation_result(payload) or _best_answer_table(
        payload.get("answer") if isinstance(payload.get("answer"), dict) else {}
    )


def _best_answer_table(answer: dict[str, Any]) -> dict[str, Any] | None:
    blocks = answer.get("blocks") if isinstance(answer.get("blocks"), list) else []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "table" and block.get("columns"):
            return block
    return None


def _extract_source_columns(source: dict[str, Any]) -> list[str]:
    for key in ("visible_columns", "display_columns", "columns"):
        value = source.get(key)
        if isinstance(value, list) and value:
            return [str(column) for column in value]
    return []


def _rows_to_dicts(rows: list[Any], columns: list[str]) -> list[dict[str, Any]]:
    normalized_rows = []
    for row in rows:
        if isinstance(row, dict):
            if columns:
                normalized_rows.append({column: row.get(column, "") for column in columns})
            else:
                normalized_rows.append(dict(row))
            continue
        if isinstance(row, (list, tuple)):
            normalized_rows.append({column: row[index] if index < len(row) else "" for index, column in enumerate(columns)})
    return normalized_rows


def _extract_metrics(raw: dict[str, Any], table_source: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for source in [table_source, raw.get("result") if isinstance(raw.get("result"), dict) else {}]:
        if not isinstance(source, dict):
            continue
        if isinstance(source.get("metrics"), dict):
            metrics.update(source["metrics"])
        for key in (
            "total_matches",
            "displayed_count",
            "matched_row_count",
            "returned_row_count",
            "rows_matched",
            "rows_returned",
            "display_limit",
        ):
            if key in source:
                metrics[key] = source[key]

    blocks = answer.get("blocks") if isinstance(answer.get("blocks"), list) else []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "metrics":
            continue
        for item in block.get("items") or []:
            if not isinstance(item, dict):
                continue
            label = _normalize_column(item.get("label"))
            if label and item.get("value") is not None:
                metrics.setdefault(label, item.get("value"))
    return metrics


def _collect_answer_text(raw: dict[str, Any], answer: dict[str, Any]) -> str:
    parts = []
    if raw.get("answer_text"):
        parts.append(str(raw.get("answer_text")))
    if answer.get("title"):
        parts.append(str(answer.get("title")))
    if answer.get("summary"):
        parts.append(str(answer.get("summary")))
    blocks = answer.get("blocks") if isinstance(answer.get("blocks"), list) else []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "markdown" and block.get("content"):
            parts.append(str(block.get("content")))
        elif block.get("type") == "metrics":
            for item in block.get("items") or []:
                if isinstance(item, dict):
                    parts.append(f"{item.get('label', '')}: {item.get('value', '')}")
        elif block.get("type") == "table":
            if block.get("caption"):
                parts.append(str(block.get("caption")))
    if not parts and isinstance(raw, dict):
        parts.append(json.dumps(raw, ensure_ascii=False)[:2000])
    return "\n".join(part for part in parts if part)


def _first_int(metrics: dict[str, Any], keys: list[str]) -> int | None:
    normalized_lookup = {_normalize_column(key): value for key, value in metrics.items()}
    for key in keys:
        for candidate in (key, _normalize_column(key)):
            if candidate in metrics:
                parsed = _parse_int(metrics[candidate])
                if parsed is not None:
                    return parsed
            normalized = _normalize_column(candidate)
            if normalized in normalized_lookup:
                parsed = _parse_int(normalized_lookup[normalized])
                if parsed is not None:
                    return parsed
    return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    match = re.search(r"-?\d[\d,]*", str(value))
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


def _filter_mask(df: pd.DataFrame, spec: Any) -> pd.Series:
    if not isinstance(spec, dict):
        raise ValueError("expected_filter must be an object.")
    if "all" in spec:
        masks = [_filter_mask(df, child) for child in spec["all"]]
        return _combine_masks(df, masks, all)
    if "any" in spec:
        masks = [_filter_mask(df, child) for child in spec["any"]]
        return _combine_masks(df, masks, any)
    if "not" in spec:
        return ~_filter_mask(df, spec["not"])

    column = _resolve_gold_column(df, spec.get("column"))
    op = str(spec.get("op") or "equals").casefold()
    value = spec.get("value")
    series = df[column]

    if op == "equals":
        if pd.api.types.is_numeric_dtype(series):
            return series.eq(value).fillna(False)
        return series.astype("string").str.strip().str.casefold().eq(str(value).strip().casefold()).fillna(False)
    if op == "not_equals":
        if pd.api.types.is_numeric_dtype(series):
            return (~series.eq(value)).fillna(False)
        return (~series.astype("string").str.strip().str.casefold().eq(str(value).strip().casefold())).fillna(False)
    if op == "contains":
        return series.astype("string").str.casefold().str.contains(str(value).casefold(), na=False, regex=False)
    if op == "in":
        values = {str(item).strip().casefold() for item in value or []}
        return series.astype("string").str.strip().str.casefold().isin(values).fillna(False)
    if op == "missing":
        return _missing_mask(series)
    if op == "not_missing":
        return ~_missing_mask(series)

    raise ValueError(f"Unsupported expected_filter op: {op}")


def _combine_masks(df: pd.DataFrame, masks: list[pd.Series], combiner: Any) -> pd.Series:
    if not masks:
        return pd.Series([True] * len(df), index=df.index)
    result = masks[0].copy()
    for mask in masks[1:]:
        result = result & mask if combiner is all else result | mask
    return result.fillna(False)


def _resolve_gold_column(df: pd.DataFrame, column: Any) -> str:
    requested = str(column or "")
    if requested in df.columns:
        return requested
    requested_normalized = _normalize_column(requested)
    for candidate in df.columns:
        if _normalize_column(candidate) == requested_normalized:
            return str(candidate)
    aliases = {
        "occupation": "title",
        "jobtitle": "title",
        "job": "title",
        "role": "title",
        "company": "employer",
        "organization": "employer",
        "linkedin": "linkedin_url",
        "linkedinurl": "linkedin_url",
        "gradyear": "graduation_year",
        "classyear": "graduation_year",
    }
    alias = aliases.get(requested_normalized)
    if alias and alias in df.columns:
        return alias
    raise KeyError(f"Column {column!r} was not found in gold dataset.")


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().isin(["", "nan", "none", "null"])


def _name_set(rows: list[dict[str, Any]]) -> set[str]:
    return {normalize_name(get_full_name(row)) for row in rows if normalize_name(get_full_name(row))}


def _name_display_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    mapping = {}
    for row in rows:
        name = get_full_name(row)
        normalized = normalize_name(name)
        if normalized and normalized not in mapping:
            mapping[normalized] = name
    return mapping


def _display_names(names: set[str], display_map: dict[str, str]) -> list[str]:
    return [display_map.get(name, name) for name in names if name]


def _normalize_names(names: list[str] | set[str]) -> list[str]:
    return [normalize_name(name) for name in names if normalize_name(name)]


def _extract_known_names_from_text(answer_text: str, known_name_display: dict[str, str]) -> list[str]:
    haystack = normalize_name(answer_text)
    found = []
    for normalized, display in known_name_display.items():
        if re.search(rf"\b{re.escape(normalized)}\b", haystack):
            found.append(display)
    return found


def _returned_names_in_industries(rows: list[dict[str, Any]], gold_df: pd.DataFrame, excluded_industries: set[str]) -> set[str]:
    if not excluded_industries:
        return set()
    names = set()
    for row in rows:
        matches = _match_gold_rows_for_returned_row(row, gold_df)
        if not matches.empty and matches["expected_industry"].astype("string").str.casefold().isin(excluded_industries).any():
            name = get_full_name(row)
            if name:
                names.add(name)
    return names


def _match_gold_rows_for_returned_row(row: dict[str, Any], gold_df: pd.DataFrame) -> pd.DataFrame:
    name = normalize_name(get_full_name(row))
    if not name:
        return gold_df.iloc[0:0]
    gold_names = gold_df.apply(lambda item: normalize_name(get_full_name(item)), axis=1)
    matches = gold_df[gold_names.eq(name)]
    if matches.empty:
        return matches

    employer = normalize_employer(_value_for_alias(row, {"employer", "company", "organization"}))
    if employer:
        employer_matches = matches[
            matches["employer"].fillna("").map(normalize_employer).eq(employer)
        ]
        if not employer_matches.empty:
            matches = employer_matches

    title = normalize_title(_value_for_alias(row, {"title", "occupation", "job title", "role", "position"}))
    if title:
        title_matches = matches[matches["title"].fillna("").map(normalize_title).eq(title)]
        if not title_matches.empty:
            matches = title_matches

    return matches


def _answer_says_zero_or_none(answer_text: str) -> bool:
    text = str(answer_text or "").casefold()
    return bool(re.search(r"\b0\b", text) or "no matching" in text or "no results" in text or "none" in text)


def _should_score_precision_recall(case: dict[str, Any]) -> bool:
    return bool(case.get("expected_industry") or case.get("expected_filter") or case.get("expected_count_mode"))


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    if isinstance(row, dict):
        return row
    return {}


def _value_for_alias(row: dict[str, Any], aliases: set[str]) -> str:
    normalized_aliases = {_normalize_column(alias) for alias in aliases}
    for key, value in row.items():
        if _normalize_column(key) in normalized_aliases:
            return _clean_value(value)
    return ""


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def _normalize_text(value: Any) -> str:
    text = _clean_value(value)
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def _normalize_column(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _columns_equivalent(actual: Any, expected: Any) -> bool:
    actual_norm = _normalize_column(actual)
    expected_norm = _normalize_column(expected)
    if actual_norm == expected_norm:
        return True
    groups = [
        {"title", "jobtitle", "occupation", "role", "position"},
        {"linkedin", "linkedinurl", "linkedinprofile", "linkedinprofileurl"},
        {"firstname", "first"},
        {"lastname", "last", "surname"},
        {"employer", "company", "organization", "organisation", "workplace"},
        {"graduationyear", "gradyear", "classyear", "graduationyr", "gradyr"},
        {"matchreason", "matchedreason", "internalreason", "classificationreason"},
    ]
    return any(actual_norm in group and expected_norm in group for group in groups)


def _excerpt(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."
