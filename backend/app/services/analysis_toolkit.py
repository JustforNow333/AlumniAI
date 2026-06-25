import logging
import re
import warnings

import numpy as np
import pandas as pd

from app.services.column_resolver import CANONICAL_FIELD_ALIASES, resolve_by_aliases
from app.services.industry_matching import (
    budgeted_model_classifier,
    classify_employer_status,
    is_strong_exclusion_context,
    is_title_match,
    known_company_match,
    match_row_to_industry,
    matched_term,
)
from app.services.industry_taxonomies import get_taxonomy
from app.services import people_classifier
from app.services.spreadsheet_service import to_json_safe
from app.utils.text_utils import (
    clamp_limit as _clamp_limit,
    contains_word_or_phrase as _shared_contains_word_or_phrase,
    dedupe_warnings as _shared_dedupe_warnings,
    normalize_text as _shared_normalize_text,
)


MAX_LIMIT = 500
DEFAULT_LIMIT = 100

ALLOWED_OPERATION_TYPES = {
    "preview",
    "select_columns",
    "filter_equals",
    "filter_missing",
    "filter_contains",
    "search_text",
    "contains_any",
    "contains_all",
    "sort_rows",
    "top_n",
    "bottom_n",
    "group_by_count",
    "group_by_sum",
    "group_by_average",
    "value_counts",
    "missing_values",
    "column_summary",
    "numeric_summary",
    "correlation",
    "unique_values",
    "duplicate_rows",
    "date_summary",
    "date_range_filter",
    "group_by_month",
}

SEARCHABLE_COLUMN_KEYWORDS = [
    "occupation",
    "job",
    "employer",
    "company",
    "organization",
    "organisation",
    "workplace",
    "industry",
    "major",
    "degree",
    "title",
    "role",
    "position",
    "field",
    "sector",
]

COLUMN_SYNONYM_GROUPS = [
    ["first name", "firstname", "first_name", "given name"],
    ["last name", "lastname", "last_name", "surname", "family name"],
    ["person name", "person_name", "name", "full name", "nickname", "preferred name", "alumni name"],
    ["occupation", "job", "title", "role", "position", "profession"],
    ["employer", "company", "organization", "organisation", "workplace", "firm", "business"],
    ["graduation year", "class year", "grad year", "grad yr", "graduation yr", "class yr", "grad_year"],
    ["name", "full name", "nickname", "preferred name", "alumni name"],
    ["major", "degree", "field of study", "program"],
    ["email", "email address", "e-mail"],
    ["linkedin url", "linkedinurl", "linkedin_url", "linkedin", "linked in", "linked in url"],
    ["phone", "phone number", "mobile"],
    ["city", "town", "location"],
    ["state", "province", "region"],
]

MATCH_REASON_COLUMN = "MATCH REASON"
TEXT_SEARCH_METADATA_COLUMNS = ["matched_column", "matched_term", MATCH_REASON_COLUMN]
DEFAULT_DISPLAY_SEMANTICS = ["first_name", "last_name", "person_name", "occupation", "employer", "linkedin_url"]
DISPLAY_REQUEST_KEYWORDS = {
    "major": ["major", "majors", "degree", "degrees", "field of study"],
    "grad_year": ["graduation year", "graduation years", "class year", "class years", "grad year", "grad yr"],
    "email": ["email", "emails", "email address", "e-mail"],
    "phone": ["phone", "phones", "phone number", "mobile"],
    "city": ["city", "cities", "location", "locations"],
    "state": ["state", "states", "province", "region"],
}
TECH_PEOPLE_FILTER_MODE = "tech_people"
PEOPLE_FILTER_MODE = "people"
PEOPLE_FILTER_MODES = {TECH_PEOPLE_FILTER_MODE, PEOPLE_FILTER_MODE}
PEOPLE_FILTER_INTENT = "people_filter"
PEOPLE_FILTER_ENTITY = "alumni"
PEOPLE_FILTER_CRITERIA_LABEL = "working in tech or technical roles"
PEOPLE_FILTER_ANSWER_LABEL = "Alumni matching criteria"

EMPLOYER_DESCRIPTOR_KEYWORDS = [
    "industry",
    "sector",
    "company description",
    "employer description",
    "organization description",
    "organisation description",
    "business description",
]


def build_dataset_context(df, metadata=None, sample_limit=5):
    metadata = metadata or {}
    columns = []

    for column in df.columns:
        series = df[column]
        missing_count = int(_missing_mask(series).sum())
        non_missing = series[~_missing_mask(series)]
        unique_count = int(non_missing.nunique(dropna=True))
        sample_values = to_json_safe(non_missing.head(sample_limit).tolist())
        column_context = {
            "name": str(column),
            "type": _infer_type(series),
            "missing_count": missing_count,
            "unique_count": unique_count,
            "sample_values": sample_values,
        }

        if 0 < unique_count <= 20:
            column_context["low_cardinality_values"] = to_json_safe(
                non_missing.drop_duplicates().head(20).tolist()
            )

        columns.append(column_context)

    return to_json_safe(
        {
            "dataset_id": metadata.get("dataset_id"),
            "filename": metadata.get("original_filename") or metadata.get("filename"),
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "columns": columns,
            "sample_rows": df.head(sample_limit).replace({np.nan: None}).to_dict(orient="records"),
        }
    )


def execute_operation(df, operation, assumptions=None):
    assumptions = list(assumptions or [])
    operation_type = operation.get("type") if isinstance(operation, dict) else None
    params = operation.get("params") if isinstance(operation, dict) and isinstance(operation.get("params"), dict) else {}

    if operation_type not in ALLOWED_OPERATION_TYPES:
        return _error_result(operation_type or "unknown", f"Unknown operation type '{operation_type}'.")

    try:
        if operation_type == "preview":
            return _op_preview(df, params, assumptions)
        if operation_type == "select_columns":
            return _op_select_columns(df, params, assumptions)
        if operation_type == "filter_equals":
            return _op_filter_equals(df, params, assumptions)
        if operation_type == "filter_missing":
            return _op_filter_missing(df, params, assumptions)
        if operation_type == "filter_contains":
            return _op_filter_contains(df, params, assumptions)
        if operation_type == "search_text":
            return _op_search_text(df, params, assumptions)
        if operation_type == "contains_any":
            return _op_contains_any(df, params, assumptions)
        if operation_type == "contains_all":
            return _op_contains_all(df, params, assumptions)
        if operation_type == "sort_rows":
            return _op_sort_rows(df, params, assumptions)
        if operation_type == "top_n":
            return _op_top_bottom_n(df, params, assumptions, top=True)
        if operation_type == "bottom_n":
            return _op_top_bottom_n(df, params, assumptions, top=False)
        if operation_type == "group_by_count":
            return _op_group_by_count(df, params, assumptions)
        if operation_type == "group_by_sum":
            return _op_group_by_numeric(df, params, assumptions, aggregation="sum")
        if operation_type == "group_by_average":
            return _op_group_by_numeric(df, params, assumptions, aggregation="average")
        if operation_type == "value_counts":
            return _op_value_counts(df, params, assumptions)
        if operation_type == "missing_values":
            return _op_missing_values(df, params, assumptions)
        if operation_type == "column_summary":
            return _op_column_summary(df, params, assumptions)
        if operation_type == "numeric_summary":
            return _op_numeric_summary(df, params, assumptions)
        if operation_type == "correlation":
            return _op_correlation(df, params, assumptions)
        if operation_type == "unique_values":
            return _op_unique_values(df, params, assumptions)
        if operation_type == "duplicate_rows":
            return _op_duplicate_rows(df, params, assumptions)
        if operation_type == "date_summary":
            return _op_date_summary(df, params, assumptions)
        if operation_type == "date_range_filter":
            return _op_date_range_filter(df, params, assumptions)
        if operation_type == "group_by_month":
            return _op_group_by_month(df, params, assumptions)
    except Exception as exc:
        logging.getLogger(__name__).warning("Operation '%s' raised an unexpected error: %s", operation_type, exc, exc_info=True)
        return _error_result(operation_type, f"Operation failed: {exc}")

    return _error_result(operation_type, f"Operation '{operation_type}' is not implemented.")


def validate_operation(operation):
    operation_type = operation.get("type") if isinstance(operation, dict) else None
    if operation_type not in ALLOWED_OPERATION_TYPES:
        return False, f"Unknown operation type '{operation_type}'."
    if not isinstance(operation.get("params", {}), dict):
        return False, "Operation params must be an object."
    return True, ""


def _op_preview(df, params, assumptions):
    limit = _limit(params.get("limit"), default=20)
    columns = list(df.columns)
    rows = _rows(df.head(limit), columns)
    return _ok_result(
        "preview",
        f"First {len(rows)} rows from the full dataset.",
        columns,
        rows,
        {"rows_matched": len(rows), "total_rows": int(df.shape[0])},
        assumptions,
        [],
        is_filtered=False,
    )


def _op_select_columns(df, params, assumptions):
    columns, warnings = _resolve_columns(df, params.get("columns"), required=True)
    if not columns:
        return _error_result("select_columns", "No valid columns were selected.", warnings)
    limit = _limit(params.get("limit"), default=DEFAULT_LIMIT)
    rows = _rows(df[columns].head(limit), columns)
    return _ok_result(
        "select_columns",
        f"Selected {len(columns)} columns across {len(rows)} rows.",
        columns,
        rows,
        {"rows_matched": len(rows), "total_rows": int(df.shape[0])},
        assumptions,
        warnings,
        is_filtered=False,
    )


def _op_filter_equals(df, params, assumptions):
    column, warning = _resolve_column(df, params.get("column"))
    if not column:
        return _error_result("filter_equals", f"Column '{params.get('column')}' was not found.", warning)

    value = params.get("value")
    case_sensitive = bool(params.get("case_sensitive", False))
    series = df[column]
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        left = series.astype("string").fillna("")
        right = "" if value is None else str(value)
        mask = left.eq(right) if case_sensitive else left.str.lower().eq(right.lower())
    else:
        mask = series.eq(value)
    return _filter_result(df, mask, params, "filter_equals", f"Rows where {column} equals {value}.", assumptions, warning)


def _op_filter_missing(df, params, assumptions):
    column, warning = _resolve_column(df, params.get("column"))
    if not column:
        return _error_result("filter_missing", f"Column '{params.get('column')}' was not found.", warning)

    mask = _missing_mask(df[column])
    return _filter_result(
        df,
        mask,
        params,
        "filter_missing",
        f"Rows where {column} is missing.",
        assumptions,
        warning,
    )


def _op_filter_contains(df, params, assumptions):
    terms = _search_terms_from_params(params)
    if not terms:
        return _error_result("filter_contains", "Search term is required.")

    requested_columns = params.get("columns") if params.get("columns") is not None else params.get("column")
    columns, warnings = _resolve_search_columns(df, requested_columns)
    if not columns:
        return _error_result("filter_contains", "No valid searchable columns were provided.", warnings)

    return _text_search_result(
        df,
        columns,
        terms,
        params,
        "filter_contains",
        "Rows where searchable text contains the requested term.",
        assumptions,
        warnings,
        require_all=False,
    )


def _op_search_text(df, params, assumptions):
    terms = _search_terms_from_params(params)
    if not terms:
        return _error_result("search_text", "At least one search term is required.")

    columns, warnings = _resolve_search_columns(df, params.get("columns"))
    if not columns:
        return _error_result("search_text", "No valid searchable columns were provided.", warnings)

    require_all = bool(params.get("require_all", False))
    return _text_search_result(
        df,
        columns,
        terms,
        params,
        "search_text",
        "Rows matching the requested text search.",
        assumptions,
        warnings,
        require_all=require_all,
    )


def _op_contains_any(df, params, assumptions):
    grouped = _resolve_column_term_groups(df, params)
    if grouped:
        if params.get("filter_mode") in PEOPLE_FILTER_MODES:
            return _people_filter_result(
                df,
                grouped["columns"],
                grouped["terms"],
                params,
                assumptions,
                grouped["warnings"],
                column_term_groups=grouped["groups"],
            )
        return _text_search_result(
            df,
            grouped["columns"],
            grouped["terms"],
            params,
            "contains_any",
            "Rows matching any requested term.",
            assumptions,
            grouped["warnings"],
            require_all=False,
            column_term_groups=grouped["groups"],
        )

    columns, warnings = _resolve_search_columns(df, params.get("columns"))
    terms = [str(term).strip() for term in params.get("terms", []) if str(term).strip()]
    if not columns:
        return _error_result("contains_any", "No valid searchable columns were provided.", warnings)
    if not terms:
        return _error_result("contains_any", "At least one search term is required.")

    if params.get("filter_mode") in PEOPLE_FILTER_MODES:
        return _people_filter_result(df, columns, terms, params, assumptions, warnings)

    return _text_search_result(
        df,
        columns,
        terms,
        params,
        "contains_any",
        "Rows matching any requested term.",
        assumptions,
        warnings,
        require_all=False,
    )


def _op_contains_all(df, params, assumptions):
    grouped = _resolve_column_term_groups(df, params)
    if grouped:
        return _text_search_result(
            df,
            grouped["columns"],
            grouped["terms"],
            params,
            "contains_all",
            "Rows matching all requested terms.",
            assumptions,
            grouped["warnings"],
            require_all=True,
            column_term_groups=grouped["groups"],
        )

    columns, warnings = _resolve_search_columns(df, params.get("columns"))
    terms = [str(term).strip() for term in params.get("terms", []) if str(term).strip()]
    if not columns:
        return _error_result("contains_all", "No valid searchable columns were provided.", warnings)
    if not terms:
        return _error_result("contains_all", "At least one search term is required.")

    return _text_search_result(
        df,
        columns,
        terms,
        params,
        "contains_all",
        "Rows matching all requested terms.",
        assumptions,
        warnings,
        require_all=True,
    )


def _op_sort_rows(df, params, assumptions):
    column, warning = _resolve_column(df, params.get("column"))
    if not column:
        return _error_result("sort_rows", f"Column '{params.get('column')}' was not found.", warning)
    ascending = bool(params.get("ascending", True))
    sorted_df = _sort_dataframe(df, column, ascending=ascending)
    return_columns, warnings = _return_columns(df, params.get("return_columns"), default=list(df.columns))
    warnings.extend(warning)
    limit = _limit(params.get("limit"), default=25)
    rows = _rows(sorted_df[return_columns].head(limit), return_columns)
    return _ok_result(
        "sort_rows",
        f"Sorted rows by {column}.",
        return_columns,
        rows,
        {"rows_matched": len(rows), "total_rows": int(df.shape[0])},
        assumptions,
        warnings,
    )


def _op_top_bottom_n(df, params, assumptions, top=True):
    operation_type = "top_n" if top else "bottom_n"
    column, warning = _resolve_column(df, params.get("column"))
    if not column:
        return _error_result(operation_type, f"Column '{params.get('column')}' was not found.", warning)
    n = _limit(params.get("n"), default=10)
    numeric = _numeric_series(df[column])
    if numeric.notna().sum() == 0:
        return _error_result(operation_type, f"Column '{column}' does not contain numeric values.")
    sorted_df = df.assign(_sort_value=numeric).sort_values(
        "_sort_value", ascending=not top, na_position="last", kind="mergesort"
    )
    return_columns, warnings = _return_columns(df, params.get("return_columns"), default=list(df.columns))
    warnings.extend(warning)
    rows = _rows(sorted_df[return_columns].head(n), return_columns)
    label = "Top" if top else "Bottom"
    return _ok_result(
        operation_type,
        f"{label} {len(rows)} rows by {column}.",
        return_columns,
        rows,
        {"rows_matched": len(rows), "total_rows": int(df.shape[0])},
        assumptions,
        warnings,
    )


def _op_group_by_count(df, params, assumptions):
    group_col, warning = _resolve_column(df, params.get("group_by"))
    if not group_col:
        return _error_result("group_by_count", f"Column '{params.get('group_by')}' was not found.", warning)
    limit = _limit(params.get("limit"), default=25)
    keys = _group_keys(df[group_col])
    grouped = keys.value_counts(dropna=False).head(limit)
    rows = [[_format_group_key(key), int(value)] for key, value in grouped.items()]
    return _ok_result(
        "group_by_count",
        f"Counts by {group_col}.",
        [group_col, "Count"],
        rows,
        {"rows_matched": len(rows), "total_rows": int(df.shape[0])},
        assumptions,
        warning,
    )


def _op_group_by_numeric(df, params, assumptions, aggregation):
    operation_type = "group_by_average" if aggregation == "average" else "group_by_sum"
    group_col, group_warning = _resolve_column(df, params.get("group_by"))
    value_col, value_warning = _resolve_column(df, params.get("value_column"))
    warnings = group_warning + value_warning
    if not group_col:
        return _error_result(operation_type, f"Column '{params.get('group_by')}' was not found.", warnings)
    if not value_col:
        return _error_result(operation_type, f"Column '{params.get('value_column')}' was not found.", warnings)

    numeric = _numeric_series(df[value_col])
    work = pd.DataFrame({"group": _group_keys(df[group_col]), "value": numeric})
    valid = work.dropna(subset=["value"])
    if aggregation == "average":
        grouped = valid.groupby("group", dropna=False)["value"].mean()
        value_label = f"Average {value_col}"
    else:
        grouped = valid.groupby("group", dropna=False)["value"].sum()
        value_label = f"Sum {value_col}"
    grouped = grouped.sort_values(ascending=False).head(_limit(params.get("limit"), default=25))
    rows = [[_format_group_key(key), to_json_safe(value)] for key, value in grouped.items()]
    return _ok_result(
        operation_type,
        f"{value_label} by {group_col}.",
        [group_col, value_label],
        rows,
        {
            "rows_matched": int(valid.shape[0]),
            "total_rows": int(df.shape[0]),
            "invalid_or_missing_numeric_values": int(numeric.isna().sum()),
        },
        assumptions,
        warnings,
    )


def _op_value_counts(df, params, assumptions):
    column, warning = _resolve_column(df, params.get("column"))
    if not column:
        return _error_result("value_counts", f"Column '{params.get('column')}' was not found.", warning)
    series = df[column][~_missing_mask(df[column])]
    counts = series.value_counts(dropna=True).head(_limit(params.get("limit"), default=25))
    rows = [[to_json_safe(key), int(value)] for key, value in counts.items()]
    return _ok_result(
        "value_counts",
        f"Most common values in {column}.",
        [column, "Count"],
        rows,
        {"rows_matched": int(series.shape[0]), "total_rows": int(df.shape[0])},
        assumptions,
        warning,
    )


def _op_missing_values(df, params, assumptions):
    columns, warnings = _resolve_columns(df, params.get("columns"), required=False)
    if not columns:
        columns = list(df.columns)
    rows = []
    total_rows = int(df.shape[0])
    for column in columns:
        missing_count = int(_missing_mask(df[column]).sum())
        missing_pct = (missing_count / total_rows * 100) if total_rows else 0
        rows.append([column, missing_count, round(missing_pct, 2)])
    rows.sort(key=lambda row: row[1], reverse=True)
    return _ok_result(
        "missing_values",
        "Missing value counts by column.",
        ["Column", "Missing Count", "Missing %"],
        rows,
        {"total_rows": total_rows, "columns_analyzed": len(columns)},
        assumptions,
        warnings,
    )


def _op_column_summary(df, params, assumptions):
    columns, warnings = _resolve_columns(df, params.get("columns"), required=False)
    if not columns:
        columns = list(df.columns)
    rows = []
    for column in columns:
        missing = int(_missing_mask(df[column]).sum())
        non_missing = df[column][~_missing_mask(df[column])]
        rows.append(
            [
                column,
                _infer_type(df[column]),
                int(non_missing.shape[0]),
                missing,
                int(non_missing.nunique(dropna=True)),
                ", ".join(str(value) for value in to_json_safe(non_missing.head(3).tolist())),
            ]
        )
    return _ok_result(
        "column_summary",
        "Column profile summary.",
        ["Column", "Type", "Non-null Count", "Missing Count", "Unique Count", "Sample Values"],
        rows,
        {"total_rows": int(df.shape[0]), "columns_analyzed": len(columns)},
        assumptions,
        warnings,
    )


def _op_numeric_summary(df, params, assumptions):
    columns, warnings = _resolve_columns(df, params.get("columns"), required=False)
    if not columns:
        columns = [column for column in df.columns if _numeric_series(df[column]).notna().sum() > 0]
    rows = []
    for column in columns:
        numeric = _numeric_series(df[column])
        valid = numeric.dropna()
        if valid.empty:
            warnings.append(f"Column '{column}' did not contain numeric values.")
            continue
        rows.append(
            [
                column,
                int(valid.count()),
                int(numeric.isna().sum()),
                to_json_safe(valid.sum()),
                to_json_safe(valid.mean()),
                to_json_safe(valid.median()),
                to_json_safe(valid.min()),
                to_json_safe(valid.max()),
                to_json_safe(valid.std()),
            ]
        )
    return _ok_result(
        "numeric_summary",
        "Numeric column summary.",
        ["Column", "Count", "Missing", "Sum", "Mean", "Median", "Min", "Max", "Standard Deviation"],
        rows,
        {"total_rows": int(df.shape[0]), "columns_analyzed": len(rows)},
        assumptions,
        warnings,
    )


def _op_correlation(df, params, assumptions):
    columns, warnings = _resolve_columns(df, params.get("columns"), required=False)
    if not columns:
        columns = [column for column in df.columns if _numeric_series(df[column]).notna().sum() >= 2]
    numeric_df = pd.DataFrame({column: _numeric_series(df[column]) for column in columns})
    corr = numeric_df.corr(numeric_only=True)
    rows = []
    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i + 1 :]:
            value = corr.loc[col1, col2]
            if pd.notna(value):
                rows.append([col1, col2, to_json_safe(value), to_json_safe(abs(value))])
    rows.sort(key=lambda row: row[3], reverse=True)
    rows = rows[: _limit(params.get("limit"), default=20)]
    return _ok_result(
        "correlation",
        "Strongest numeric correlations.",
        ["Column 1", "Column 2", "Correlation", "Absolute Correlation"],
        rows,
        {"columns_analyzed": len(columns), "total_rows": int(df.shape[0])},
        assumptions,
        warnings,
    )


def _op_unique_values(df, params, assumptions):
    column, warning = _resolve_column(df, params.get("column"))
    if not column:
        return _error_result("unique_values", f"Column '{params.get('column')}' was not found.", warning)
    series = df[column][~_missing_mask(df[column])]
    counts = series.value_counts(dropna=True).head(_limit(params.get("limit"), default=50))
    rows = [[to_json_safe(key), int(value)] for key, value in counts.items()]
    return _ok_result(
        "unique_values",
        f"Unique values in {column}.",
        [column, "Count"],
        rows,
        {"unique_count": int(series.nunique(dropna=True)), "total_rows": int(df.shape[0])},
        assumptions,
        warning,
    )


def _op_duplicate_rows(df, params, assumptions):
    subset, warnings = _resolve_columns(df, params.get("subset"), required=False)
    subset = subset or None
    duplicate_mask = df.duplicated(subset=subset, keep=False)
    duplicate_df = df[duplicate_mask]
    columns = subset or list(df.columns)
    limit = _limit(params.get("limit"), default=100)
    rows = _rows(duplicate_df[columns].head(limit), columns)
    return _ok_result(
        "duplicate_rows",
        f"Found {int(duplicate_mask.sum())} duplicate rows.",
        columns,
        rows,
        {
            "duplicate_row_count": int(duplicate_mask.sum()),
            "rows_returned": len(rows),
            "total_rows": int(df.shape[0]),
        },
        assumptions,
        warnings,
    )


def _op_date_summary(df, params, assumptions):
    columns, warnings = _resolve_columns(df, params.get("columns"), required=False)
    if not columns:
        columns = list(df.columns)
    rows = []
    for column in columns:
        parsed = _date_series(df[column])
        if parsed.notna().sum() == 0:
            continue
        rows.append(
            [
                column,
                to_json_safe(parsed.min()),
                to_json_safe(parsed.max()),
                int(_missing_mask(df[column]).sum()),
                int(parsed.notna().sum()),
            ]
        )
    return _ok_result(
        "date_summary",
        "Date-like column summary.",
        ["Column", "Min Date", "Max Date", "Missing Count", "Parsed Count"],
        rows,
        {"columns_analyzed": len(rows), "total_rows": int(df.shape[0])},
        assumptions,
        warnings,
    )


def _op_date_range_filter(df, params, assumptions):
    column, warning = _resolve_column(df, params.get("column"))
    if not column:
        return _error_result("date_range_filter", f"Column '{params.get('column')}' was not found.", warning)
    parsed = _date_series(df[column])
    start = pd.to_datetime(params.get("start"), errors="coerce") if params.get("start") else None
    end = pd.to_datetime(params.get("end"), errors="coerce") if params.get("end") else None
    valid_start = start is not None and pd.notna(start)
    valid_end = end is not None and pd.notna(end)
    if not valid_start and not valid_end:
        return _error_result("date_range_filter", "At least one valid start or end date is required.")
    mask = parsed.notna()
    if valid_start:
        mask &= parsed >= start
    if valid_end:
        mask &= parsed <= end
    return _filter_result(
        df,
        mask,
        params,
        "date_range_filter",
        f"Rows where {column} is inside the requested date range.",
        assumptions,
        warning,
    )


def _op_group_by_month(df, params, assumptions):
    date_col, date_warning = _resolve_column(df, params.get("date_column"))
    value_col = None
    value_warning = []
    if params.get("value_column"):
        value_col, value_warning = _resolve_column(df, params.get("value_column"))
    warnings = date_warning + value_warning
    if not date_col:
        return _error_result("group_by_month", f"Column '{params.get('date_column')}' was not found.", warnings)

    aggregation = str(params.get("aggregation") or "count").lower()
    if aggregation not in {"count", "sum", "average"}:
        return _error_result("group_by_month", "Aggregation must be count, sum, or average.")
    if aggregation != "count" and not value_col:
        return _error_result("group_by_month", "value_column is required for sum or average.")

    parsed = _date_series(df[date_col])
    month = parsed.dt.to_period("M").astype("string")
    work = pd.DataFrame({"month": month})
    if aggregation == "count":
        grouped = work[parsed.notna()].groupby("month").size()
        value_label = "Count"
    else:
        numeric = _numeric_series(df[value_col])
        work["value"] = numeric
        valid = work[parsed.notna() & numeric.notna()]
        grouped = valid.groupby("month")["value"].sum() if aggregation == "sum" else valid.groupby("month")["value"].mean()
        value_label = f"{aggregation.title()} {value_col}"
    grouped = grouped.sort_index().tail(_limit(params.get("limit"), default=24))
    rows = [[str(key), to_json_safe(value)] for key, value in grouped.items()]
    return _ok_result(
        "group_by_month",
        f"{value_label} by month.",
        ["Month", value_label],
        rows,
        {"rows_matched": int(len(rows)), "total_rows": int(df.shape[0])},
        assumptions,
        warnings,
    )


def _filter_result(df, mask, params, operation_type, summary, assumptions, warnings=None):
    warnings = list(warnings or [])
    return_columns, return_warnings = _return_columns(
        df,
        params.get("return_columns"),
        default=list(df.columns),
    )
    warnings.extend(return_warnings)
    limit = _limit(params.get("limit"), default=DEFAULT_LIMIT)
    matched = df[mask.fillna(False)]
    rows = _rows(matched[return_columns].head(limit), return_columns)
    return _ok_result(
        operation_type,
        summary,
        return_columns,
        rows,
        {"rows_matched": int(matched.shape[0]), "rows_returned": len(rows), "total_rows": int(df.shape[0])},
        assumptions,
        warnings,
        is_filtered=True,
    )


def _text_search_result(df, columns, terms, params, operation_type, summary, assumptions, warnings=None, require_all=False, column_term_groups=None):
    warnings = list(warnings or [])
    case_sensitive = bool(params.get("case_sensitive", False))
    search_groups = _prepared_search_groups(columns, terms, column_term_groups, case_sensitive)
    matched_columns = []
    matched_terms = []
    match_reasons = []
    mask_values = []
    raw_match_count = 0

    for _, row in df[columns].iterrows():
        row_matches = []
        for group in search_groups:
            for original_term, search_term in group["terms"]:
                term_column = ""
                term_value = ""
                for column in group["columns"]:
                    text = "" if _is_missing_value(row[column]) else str(row[column])
                    haystack = text if case_sensitive else text.lower()
                    if search_term in haystack:
                        term_column = column
                        term_value = text
                        break
                if term_column:
                    row_matches.append((term_column, original_term, term_value))

        # The same term can match in multiple groups, so compare distinct matched
        # terms (not raw match count) against the distinct required terms.
        if require_all:
            matched_term_names = set(match[1] for match in row_matches)
            is_match = matched_term_names == set(terms)
        else:
            is_match = bool(row_matches)
        if is_match:
            matched_columns.append(", ".join(dict.fromkeys(match[0] for match in row_matches)))
            matched_terms.append(", ".join(dict.fromkeys(match[1] for match in row_matches)))
            raw_match_count += len(row_matches)
            match_reasons.append("; ".join(_dedupe_match_reasons(row_matches)[:3]))
        else:
            matched_columns.append("")
            matched_terms.append("")
            match_reasons.append("")
        mask_values.append(is_match)

    working = df.copy()
    working["matched_column"] = matched_columns
    working["matched_term"] = matched_terms
    working[MATCH_REASON_COLUMN] = match_reasons
    working["Matched Column"] = matched_columns
    working["Matched Term"] = matched_terms
    return_columns, return_warnings = _display_columns_for_text_search(working, columns, params)
    warnings.extend(return_warnings)
    limit = _limit(params.get("limit"), default=DEFAULT_LIMIT)
    matched = working[pd.Series(mask_values, index=df.index)]
    rows = _rows(matched[return_columns].head(limit), return_columns)
    matched_row_count = int(sum(mask_values))
    returned_row_count = len(rows)
    capped = returned_row_count < matched_row_count
    if raw_match_count > matched_row_count:
        warnings.append(
            _warning(
                "deduplicated_text_matches",
                f"{raw_match_count} raw keyword hits were deduplicated to {matched_row_count} matching rows.",
                resolved_to={"raw_match_count": raw_match_count, "matched_row_count": matched_row_count},
            )
        )
    if capped:
        warnings.append(
            _warning(
                "display_limit_applied",
                f"Showing {returned_row_count} of {matched_row_count} matching rows because the display limit is {limit}.",
                resolved_to={"display_limit": limit, "matched_row_count": matched_row_count, "returned_row_count": returned_row_count},
            )
        )

    metrics = {
        "total_rows": int(df.shape[0]),
        "raw_match_count": raw_match_count,
        "matched_row_count": matched_row_count,
        "returned_row_count": returned_row_count,
        "display_limit": limit,
        "deduplicated": raw_match_count != matched_row_count,
        "search_columns": columns,
        "display_columns": return_columns,
        "search_terms": terms,
        # Backward-compatible aliases for older frontend/tests.
        "rows_matched": matched_row_count,
        "rows_returned": returned_row_count,
        "searched_columns": columns,
    }
    return _ok_result(
        operation_type,
        summary or f"Found {matched_row_count} filtered rows.",
        return_columns,
        rows,
        metrics,
        assumptions,
        warnings,
        is_filtered=True,
        extras={
            "total_rows": int(df.shape[0]),
            "raw_match_count": raw_match_count,
            "matched_row_count": matched_row_count,
            "returned_row_count": returned_row_count,
            "display_limit": limit,
            "deduplicated": raw_match_count != matched_row_count,
            "search_columns": columns,
            "display_columns": return_columns,
        },
    )


def _people_filter_result(df, columns, terms, params, assumptions, warnings=None, column_term_groups=None):
    """Generic people/alumni filter: classify each row with the layered industry
    matching engine (or employer/occupation matching), separate confirmed from
    uncertain, deduplicate people, and return the structured people_filter shape."""
    warnings = list(warnings or [])
    limit = _limit(params.get("limit"), default=DEFAULT_LIMIT)
    filter_spec = params.get("people_filter") if isinstance(params.get("people_filter"), dict) else {}
    filter_type = str(filter_spec.get("filter_type") or "industry")
    industry = filter_spec.get("industry") or ("tech" if filter_type == "industry" else None)
    criteria_label = filter_spec.get("criteria_label") or PEOPLE_FILTER_CRITERIA_LABEL
    answer_label = filter_spec.get("answer_label") or PEOPLE_FILTER_ANSWER_LABEL
    entity = filter_spec.get("entity") or PEOPLE_FILTER_ENTITY

    taxonomy = None
    if filter_type == "industry":
        taxonomy = get_taxonomy(industry) or get_taxonomy("tech")
        industry = taxonomy["industry"]

    display_specs = _people_display_column_specs(df, params)
    display_headers = [spec["header"] for spec in display_specs]

    extra_terms = []
    if taxonomy:
        extra_terms = list(taxonomy.get("known_companies") or []) + list(taxonomy.get("retrieval_keywords") or [])
    elif filter_type == "employer":
        extra_terms = filter_spec.get("employer_terms") or []
    elif filter_type == "occupation":
        extra_terms = filter_spec.get("occupation_terms") or []
    raw_terms = list(
        dict.fromkeys(
            [str(term).strip() for term in terms if str(term).strip()]
            + [str(term).strip() for term in extra_terms if str(term).strip()]
        )
    )
    raw_match_count = _raw_keyword_hit_count(df, columns, raw_terms)
    candidate_indices = _keyword_candidate_indices(df, columns, raw_terms)

    occupation_col = _resolved_display_column(df, "occupation")
    employer_col = _resolved_display_column(df, "employer")
    query_spec = people_classifier.query_spec_from_filter(filter_spec, default_industry=industry)
    classify_row = _people_row_classifier(filter_type, taxonomy, filter_spec, query_spec)

    counted_rows_by_key = {}
    direct_rows_by_key = {}
    adjacent_rows_by_key = {}
    uncertain_rows_by_key = {}
    debug_rows = []
    uncertain_keys = set()
    adjacent_keys = set()
    counted_adjacent_keys = set()
    non_match_candidate_count = 0

    for index, row in df.iterrows():
        occupation = _safe_row_text(row, occupation_col)
        employer = _safe_row_text(row, employer_col)
        descriptor_text = " ".join(_employer_descriptor_values(df, row))
        match = classify_row(occupation, employer, descriptor_text)
        classification = match.get("classification") or (
            "direct_match" if match["status"] == "confirmed" else ("uncertain" if match["status"] == "uncertain" else "non_match")
        )
        dedupe_key = _person_surface_key(df, row, index)
        display_row = {
            spec["header"]: _row_value_for_display(row, spec["source"])
            for spec in display_specs
        }
        debug_row = {
            "row_index": int(index) if isinstance(index, (int, np.integer)) else str(index),
            "status": match["status"],
            "match_category": match.get("match_category"),
            "match_confidence": match.get("match_confidence"),
            "role_signal": match.get("role_signal"),
            "employer_signal": match.get("employer_signal"),
            "match_reason_code": match.get("match_reason_code"),
            "match_reason": match.get("internal_reason", ""),
            "match_sources": match.get("match_sources") or [],
            "classification": classification,
            "confidence": match.get("confidence"),
            "employer_industry": match.get("employer_industry") or [],
            "job_function": match.get("job_function") or [],
            "specialties": match.get("specialties") or [],
        }

        is_counted = (
            bool(match.get("count_as_match"))
            if "count_as_match" in match
            else match["status"] == "confirmed"
        )
        if is_counted:
            counted_rows_by_key.setdefault(dedupe_key, display_row)
            debug_rows.append(debug_row)
            if classification == "adjacent":
                counted_adjacent_keys.add(dedupe_key)
                adjacent_keys.add(dedupe_key)
                if dedupe_key not in direct_rows_by_key:
                    adjacent_rows_by_key.setdefault(dedupe_key, display_row)
            else:
                direct_rows_by_key[dedupe_key] = display_row
                adjacent_rows_by_key.pop(dedupe_key, None)
                uncertain_rows_by_key.pop(dedupe_key, None)
            continue

        if classification == "adjacent" and dedupe_key not in counted_rows_by_key and dedupe_key not in direct_rows_by_key:
            adjacent_keys.add(dedupe_key)
            adjacent_rows_by_key.setdefault(dedupe_key, display_row)
            uncertain_rows_by_key.pop(dedupe_key, None)
            debug_rows.append(debug_row)
        elif (
            match["status"] == "uncertain"
            and dedupe_key not in counted_rows_by_key
            and dedupe_key not in direct_rows_by_key
            and dedupe_key not in adjacent_rows_by_key
        ):
            uncertain_keys.add(dedupe_key)
            uncertain_rows_by_key.setdefault(dedupe_key, display_row)
            debug_rows.append(debug_row)
        elif index in candidate_indices:
            non_match_candidate_count += 1

    total_matches = len(counted_rows_by_key)
    confirmed_rows = list(counted_rows_by_key.values())
    direct_rows = list(direct_rows_by_key.values())
    adjacent_rows = [
        row
        for key, row in adjacent_rows_by_key.items()
        if key not in counted_rows_by_key and key not in direct_rows_by_key
    ]
    uncertain_rows = [
        row
        for key, row in uncertain_rows_by_key.items()
        if key not in counted_rows_by_key and key not in direct_rows_by_key and key not in adjacent_rows_by_key
    ]
    scored_result_count = len(confirmed_rows)
    displayed_count = min(scored_result_count, limit)
    adjacent_included_count = len(counted_adjacent_keys)
    adjacent_count = len(adjacent_rows)
    uncertain_count = len(uncertain_rows)
    direct_count = len(direct_rows)
    surface_non_direct = (
        filter_type == "industry"
        and industry == "tech"
        and (query_spec.get("query_scope") or "industry") in {"industry", "tech_company", "technical_role"}
    )
    row_sections = []
    section_search_caption = "Searched columns: " + ", ".join(str(column) for column in columns) + "." if columns else ""
    if surface_non_direct:
        row_sections.append(
            {
                "category": "direct",
                "title": "Direct matches",
                "columns": display_headers,
                "rows": direct_rows,
                "count": direct_count,
                "caption": " ".join(
                    item for item in [section_search_caption, "Counted as matching the user's criteria."] if item
                ),
            }
        )
        if adjacent_rows:
            row_sections.append(
                {
                    "category": "adjacent",
                    "title": "Adjacent tech-related matches",
                    "columns": display_headers,
                    "rows": adjacent_rows,
                    "count": len(adjacent_rows),
                    "caption": " ".join(
                        item
                        for item in [
                            section_search_caption,
                            "Surfaced for networking and discovery, but not counted as direct matches.",
                        ]
                        if item
                    ),
                }
            )
        if uncertain_rows:
            row_sections.append(
                {
                    "category": "uncertain",
                    "title": "Uncertain possible matches",
                    "columns": display_headers,
                    "rows": uncertain_rows,
                    "count": len(uncertain_rows),
                    "caption": " ".join(
                        item
                        for item in [
                            section_search_caption,
                            "Possibly relevant rows with weaker or incomplete tech evidence.",
                        ]
                        if item
                    ),
                }
            )

    if displayed_count < total_matches:
        warnings.append(
            _warning(
                "display_limit_applied",
                f"Showing {displayed_count} of {total_matches} matching alumni because the display limit is {limit}.",
                resolved_to={
                    "display_limit": limit,
                    "total_matches": total_matches,
                    "displayed_count": displayed_count,
                },
            )
        )

    classification_counts = {
        "raw_candidate_count": len(candidate_indices),
        "direct_count": direct_count,
        "direct_match_count": direct_count,
        "adjacent_count": adjacent_count,
        "adjacent_not_counted_count": adjacent_count,
        "adjacent_included_count": adjacent_included_count,
        "uncertain_count": uncertain_count,
        "uncertain_not_counted_count": uncertain_count,
        "non_match_count": non_match_candidate_count,
        "excluded_count": non_match_candidate_count,
        "llm_ambiguous_candidate_count": uncertain_count,
        "classification_version": people_classifier.CLASSIFICATION_VERSION,
        "adjacent_included": bool(query_spec.get("include_adjacent")),
    }
    trace_fields = {
        "intent_type": PEOPLE_FILTER_INTENT,
        "query_scope": query_spec.get("query_scope") or "industry",
        "target_industries": list(query_spec.get("industries") or []),
        "excluded_industries": list(query_spec.get("excluded_industries") or []),
        "taxonomy_used": industry,
        "candidate_count": len(candidate_indices),
        "final_display_count": displayed_count,
        "scored_result_count": scored_result_count,
        "display_columns": display_headers,
    }

    metrics = {
        "total_dataset_rows": int(df.shape[0]),
        "total_keyword_hits": raw_match_count,
        "total_matches": total_matches,
        "direct_count": direct_count,
        "displayed_count": displayed_count,
        "display_limit": limit,
        "total_rows": int(df.shape[0]),
        "total_considered": int(df.shape[0]),
        "raw_match_count": raw_match_count,
        "matched_row_count": total_matches,
        "returned_row_count": scored_result_count,
        "scored_result_count": scored_result_count,
        "surfaced_count": direct_count + adjacent_count + uncertain_count,
        "deduplicated": True,
        "search_columns": columns,
        "display_columns": display_headers,
        "search_terms": raw_terms,
        # Backward-compatible aliases for older frontend/tests.
        "rows_matched": total_matches,
        "rows_returned": scored_result_count,
        "searched_columns": columns,
    }
    metrics.update(classification_counts)
    metrics.update(trace_fields)
    extras = {
        "intent": PEOPLE_FILTER_INTENT,
        "entity": entity,
        "filter_type": filter_type,
        "industry": industry,
        "query_scope": query_spec.get("query_scope") or "industry",
        "target_industries": list(query_spec.get("industries") or []),
        "excluded_industries": list(query_spec.get("excluded_industries") or []),
        "taxonomy_used": industry,
        "criteria_label": criteria_label,
        "answer_label": answer_label,
        "total_dataset_rows": int(df.shape[0]),
        "total_keyword_hits": raw_match_count,
        "total_matches": total_matches,
        "direct_count": direct_count,
        "displayed_count": displayed_count,
        "display_limit": limit,
        "total_considered": int(df.shape[0]),
        "scored_result_count": scored_result_count,
        "final_display_count": displayed_count,
        "surfaced_count": direct_count + adjacent_count + uncertain_count,
        "direct_rows": direct_rows,
        "adjacent_rows": adjacent_rows,
        "uncertain_rows": uncertain_rows,
        "row_sections": row_sections,
        "visible_columns": display_headers,
        "search_columns": columns,
        "display_columns": display_headers,
        "debug": {"rows": debug_rows},
    }
    extras.update(classification_counts)
    extras.update(trace_fields)

    summary = f"{answer_label}: {total_matches}"
    if filter_type == "industry":
        summary = f"{answer_label}: {total_matches} direct matches out of {int(df.shape[0])} alumni"
        if surface_non_direct and (adjacent_count or uncertain_count):
            summary = (
                f"Found {direct_count} direct matches out of {int(df.shape[0])} alumni. "
                f"Also showing {adjacent_count} adjacent tech-related matches and "
                f"{uncertain_count} uncertain possible matches for review."
            )
        if adjacent_included_count:
            summary = (
                f"{answer_label}: {total_matches} matches out of {int(df.shape[0])} alumni "
                f"(including {adjacent_included_count} adjacent, as requested)"
            )

    return _ok_result(
        "contains_any",
        summary,
        display_headers,
        confirmed_rows,
        metrics,
        assumptions,
        warnings,
        is_filtered=True,
        extras=extras,
    )


def _people_row_classifier(filter_type, taxonomy, filter_spec, query_spec=None):
    """Return classify(occupation, employer, descriptor_text) -> MatchResult for
    the requested filter type."""
    if filter_type == "employer":
        employer_terms = [str(term) for term in filter_spec.get("employer_terms") or [] if str(term).strip()]

        def classify_employer(occupation, employer, descriptor_text):
            match = known_company_match(employer, employer_terms)
            if match:
                return {
                    "status": "confirmed",
                    "classification": "direct_match",
                    "match_sources": ["employer_match"],
                    "confidence": 1.0,
                    "internal_reason": f"Employer matches requested employer: {match}",
                }
            return {
                "status": "excluded",
                "classification": "non_match",
                "match_sources": [],
                "confidence": 0.0,
                "internal_reason": "Employer does not match the requested employer.",
            }

        return classify_employer

    if filter_type == "occupation":
        occupation_terms = [str(term) for term in filter_spec.get("occupation_terms") or [] if str(term).strip()]

        def classify_occupation(occupation, employer, descriptor_text):
            match = matched_term(occupation, occupation_terms)
            if match:
                return {
                    "status": "confirmed",
                    "classification": "direct_match",
                    "match_sources": ["title_keyword"],
                    "confidence": 1.0,
                    "internal_reason": f"Occupation matches requested role: {match}",
                }
            return {
                "status": "excluded",
                "classification": "non_match",
                "match_sources": [],
                "confidence": 0.0,
                "internal_reason": "Occupation does not match the requested role.",
            }

        return classify_occupation

    # Industry filters run the query-aware multi-label classifier: broad keyword
    # hits only make a row a candidate; strict classification decides inclusion.
    model_classifier = budgeted_model_classifier()
    industry_query_spec = query_spec or people_classifier.query_spec_from_filter(
        filter_spec, default_industry=(taxonomy or {}).get("industry")
    )

    def classify_industry(occupation, employer, descriptor_text):
        outcome = people_classifier.classify_candidate(
            occupation,
            employer,
            industry_query_spec,
            descriptor_text=descriptor_text,
            model_classifier=model_classifier,
        )
        if outcome["count_as_match"]:
            status = "confirmed"
        elif outcome["classification"] == "uncertain":
            status = "uncertain"
        else:
            status = "excluded"
        return {
            "status": status,
            "classification": outcome["classification"],
            "count_as_match": outcome["count_as_match"],
            "match_category": outcome.get("match_category"),
            "match_confidence": outcome.get("match_confidence"),
            "role_signal": outcome.get("role_signal"),
            "employer_signal": outcome.get("employer_signal"),
            "match_reason_code": outcome.get("match_reason_code"),
            "match_sources": [outcome["classification"]],
            "confidence": outcome["confidence"],
            "internal_reason": outcome["internal_reason"],
            "employer_industry": outcome["employer_industry"],
            "job_function": outcome["job_function"],
            "specialties": outcome["specialties"],
        }

    return classify_industry


def _people_display_column_specs(df, params):
    first = _resolved_display_column(df, "first_name")
    last = _resolved_display_column(df, "last_name")
    occupation = _resolved_display_column(df, "occupation")
    employer = _resolved_display_column(df, "employer")
    linkedin = _resolved_display_column(df, "linkedin_url")
    requested = params.get("display_columns") or params.get("return_columns")
    requested_columns, _warnings = _resolve_columns(df, requested, required=False)
    restrict_to_requested = bool(requested_columns)

    def wants(column):
        return bool(column) and (not restrict_to_requested or column in requested_columns)

    specs = []
    if wants(first):
        specs.append({"header": "First Name", "source": first})
    if wants(last):
        specs.append({"header": "Last Name", "source": last})
    if not first and not last:
        fallback_name = _resolved_display_column(df, "person_name") or _resolved_display_column(df, "name")
        if wants(fallback_name):
            specs.append({"header": str(fallback_name), "source": fallback_name})
    if wants(occupation):
        specs.append({"header": "Occupation", "source": occupation})
    if wants(employer):
        specs.append({"header": "Employer", "source": employer})

    debug_sources = set(_debug_columns())
    existing_sources = {spec["source"] for spec in specs}
    linkedin_source = linkedin
    for column in requested_columns:
        if column in existing_sources or column == linkedin_source or _is_debug_column(column):
            continue
        header = _canonical_display_header(column)
        if _normalize_compact(header) in debug_sources:
            continue
        specs.append({"header": header, "source": column})
        existing_sources.add(column)

    if wants(linkedin):
        specs = [spec for spec in specs if spec["source"] != linkedin]
        specs.append({"header": "LinkedIn URL", "source": linkedin})

    return specs


def _resolved_display_column(df, semantic):
    if semantic in CANONICAL_FIELD_ALIASES:
        column = resolve_by_aliases(df, CANONICAL_FIELD_ALIASES[semantic])
        if column:
            return column
        if semantic in {"first_name", "last_name", "linkedin_url"}:
            return None
    column, _warnings = _resolve_column(df, semantic)
    return column


def _canonical_display_header(column):
    normalized = _normalize_compact(column)
    if normalized in {"firstname", "givenname"}:
        return "First Name"
    if normalized in {"lastname", "surname", "familyname"}:
        return "Last Name"
    if normalized in {"linkedinurl", "linkedin", "linkedinprofile", "linkedinprofileurl"}:
        return "LinkedIn URL"
    if normalized in {"occupation", "jobtitle", "job", "role", "position"}:
        return "Occupation"
    if normalized in {"employer", "company", "organization", "organisation", "workplace"}:
        return "Employer"
    return str(column)


def _row_value_for_display(row, column):
    if not column or column not in row.index:
        return ""
    value = row[column]
    if _is_missing_value(value):
        return ""
    return value


def is_explicit_technical_title(occupation):
    return is_title_match(occupation, get_taxonomy("tech"))


def classify_employer_tech_status(employer, occupation="", known_companies=None, descriptor_text=""):
    """Backward-compatible tech wrapper over the generic employer classifier."""
    taxonomy = get_taxonomy("tech")
    if known_companies is not None:
        taxonomy = dict(taxonomy)
        taxonomy["known_companies"] = list(known_companies)
    status = classify_employer_status(
        employer,
        taxonomy,
        occupation=occupation,
        descriptor_text=descriptor_text,
    )
    status_map = {"confirmed": "confirmed_tech", "excluded": "confirmed_non_tech", "uncertain": "uncertain"}
    source = "non_tech_exclusion" if status["source"] == "exclusion" else status["source"]
    return {
        "status": status_map[status["status"]],
        "source": source,
        "confidence": status["confidence"],
        "internal_reason": status["internal_reason"],
    }


def is_strong_non_tech_context(occupation, employer):
    return is_strong_exclusion_context(occupation, employer, get_taxonomy("tech"))


def _employer_descriptor_values(df, row):
    values = []
    for column in df.columns:
        column_norm = _normalize_name(column)
        if any(keyword in column_norm for keyword in EMPLOYER_DESCRIPTOR_KEYWORDS):
            text = _safe_row_text(row, column)
            if text:
                values.append(text)
    return values


def _raw_keyword_hit_count(df, columns, terms):
    if not columns or not terms:
        return 0
    hit_count = 0
    valid_columns = [column for column in columns if column in df.columns]
    for _, row in df[valid_columns].iterrows():
        for term in terms:
            for column in valid_columns:
                if _text_has_term(row[column], term):
                    hit_count += 1
                    break
    return hit_count


def _keyword_candidate_indices(df, columns, terms):
    """Row indices with at least one broad keyword hit. Candidates only — the
    classifier decides final inclusion; this exists for recall debugging."""
    if not columns or not terms:
        return set()
    valid_columns = [column for column in columns if column in df.columns]
    candidates = set()
    for index, row in df[valid_columns].iterrows():
        for column in valid_columns:
            if any(_text_has_term(row[column], term) for term in terms):
                candidates.add(index)
                break
    return candidates


def _text_has_term(text, term):
    normalized_text = _normalize_name(text)
    normalized_term = _normalize_name(term)
    if not normalized_text or not normalized_term:
        return False
    if " " in normalized_term:
        return normalized_term in normalized_text
    return re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text) is not None


def _person_dedupe_key(df, row, index):
    stable_columns = [
        "Cornell ID",
        "Constituent ID",
        "Alumni ID",
        "NetID",
        "Net ID",
        "Email",
        "Email Address",
    ]
    for requested in stable_columns:
        column, _warnings = _resolve_column(df, requested)
        value = _safe_row_text(row, column)
        if value:
            return ("stable", _normalize_compact(value))

    first = _safe_row_text(row, _resolved_display_column(df, "first_name"))
    last = _safe_row_text(row, _resolved_display_column(df, "last_name"))
    grad_year = _safe_row_text(row, _resolved_display_column(df, "grad_year"))
    employer = _safe_row_text(row, _resolved_display_column(df, "employer"))
    if first and last and grad_year:
        return ("name_grad", _normalize_compact(first), _normalize_compact(last), _normalize_compact(grad_year))
    if first and last and employer:
        return ("name_employer", _normalize_compact(first), _normalize_compact(last), _normalize_compact(employer))

    person_name = _safe_row_text(row, _resolved_display_column(df, "person_name"))
    if person_name and employer:
        return ("person_employer", _normalize_compact(person_name), _normalize_compact(employer))
    if person_name:
        return ("person", _normalize_compact(person_name), str(index))
    return ("row", str(index))


def _person_surface_key(df, row, index):
    first = _safe_row_text(row, _resolved_display_column(df, "first_name"))
    last = _safe_row_text(row, _resolved_display_column(df, "last_name"))
    if first and last:
        return ("name", _normalize_compact(first), _normalize_compact(last))

    person_name = _safe_row_text(row, _resolved_display_column(df, "person_name"))
    if person_name:
        return ("person", _normalize_compact(person_name))
    return _person_dedupe_key(df, row, index)


def _safe_row_text(row, column):
    if not column or column not in row.index:
        return ""
    value = row[column]
    if _is_missing_value(value):
        return ""
    return str(value).strip()


def _debug_columns():
    return {
        "matchreason",
        "rawmatchreason",
        "score",
        "internalscore",
        "matchedterms",
        "confidence",
        "classificationreason",
        "uncertaintyreason",
        "modelreason",
        "matchedcolumn",
        "matchedterm",
    }


def _is_debug_column(column):
    return _normalize_compact(column) in _debug_columns()


def _ok_result(operation_type, summary, columns, rows, metrics=None, assumptions=None, warnings=None, is_filtered=False, extras=None):
    payload = {
        "operation_type": operation_type,
        "status": "ok",
        "is_filtered": bool(is_filtered),
        "summary": summary,
        "columns": list(columns or []),
        "rows": rows or [],
        "metrics": metrics or {},
        "assumptions": list(assumptions or []),
        "warnings": list(warnings or []),
    }
    if extras:
        payload.update(extras)
    return to_json_safe(
        payload
    )


def _error_result(operation_type, error, warnings=None):
    return to_json_safe(
        {
            "operation_type": operation_type,
            "status": "error",
            "is_filtered": False,
            "error": str(error),
            "warnings": list(warnings or []),
        }
    )


def _rows(frame, columns):
    return to_json_safe(frame.loc[:, columns].replace({np.nan: None}).to_numpy().tolist())


def _limit(value, default=DEFAULT_LIMIT):
    return _clamp_limit(value, default, max_value=MAX_LIMIT)


def _resolve_column(df, requested):
    if requested is None:
        return None, []
    requested_text = str(requested).strip()
    if requested_text in df.columns:
        return requested_text, []
    for column in df.columns:
        if requested_text.casefold() == str(column).casefold():
            return str(column), []
    normalized = _normalize_compact(requested_text)
    for column in df.columns:
        if _normalize_compact(column) == normalized:
            return str(column), []

    for candidate in _synonym_candidates(requested_text):
        candidate_norm = _normalize_compact(candidate)
        for column in df.columns:
            column_norm = _normalize_compact(column)
            if candidate_norm == column_norm:
                return str(column), []
        for column in df.columns:
            column_norm = _normalize_compact(column)
            if len(candidate_norm) >= 4 and (candidate_norm in column_norm or column_norm in candidate_norm):
                return str(column), []

    warnings = []
    similar = _similar_columns(df, requested_text)
    if similar:
        warnings.append(
            _warning(
                "unresolved_column",
                f"Column '{requested_text}' was not found. Available similar columns: {', '.join(similar)}.",
                requested=requested_text,
                suggestions=similar,
            )
        )
    else:
        warnings.append(
            _warning(
                "unresolved_column",
                f"Column '{requested_text}' was not found.",
                requested=requested_text,
            )
        )
    return None, warnings


def _resolve_columns(df, requested, required=False):
    if requested is None:
        return [], []
    if not isinstance(requested, list):
        requested = [requested]
    columns = []
    warnings = []
    for item in requested:
        column, column_warnings = _resolve_column(df, item)
        if column:
            columns.append(column)
        else:
            warnings.extend(column_warnings or [_warning("unresolved_column", f"Column '{item}' was not found.", requested=item)])
    return list(dict.fromkeys(columns)), _dedupe_warnings(warnings)


def _resolve_search_columns(df, requested):
    columns, warnings = _resolve_columns(df, requested, required=False)
    if columns:
        return columns, warnings

    inferred = _infer_searchable_columns(df)
    if inferred:
        warnings.append(
            _warning(
                "inferred_search_columns",
                "Requested searchable columns could not be resolved; inferred default searchable columns.",
                requested=requested,
                resolved_to=inferred,
            )
        )
        return inferred, _dedupe_warnings(warnings)

    warnings.append(
        _warning(
            "no_searchable_columns",
            "No text-like searchable columns could be inferred from this dataframe.",
            requested=requested,
        )
    )
    return [], _dedupe_warnings(warnings)


def _return_columns(df, requested, default, extras=None):
    columns, warnings = _resolve_columns(df, requested, required=False)
    if not columns:
        columns = [column for column in default if column in df.columns]
    for extra in extras or []:
        if extra in df.columns and extra not in columns:
            columns.append(extra)
    return columns, warnings


def _resolve_column_term_groups(df, params):
    raw_groups = params.get("column_term_groups")
    if not isinstance(raw_groups, list):
        return None

    groups = []
    all_columns = []
    all_terms = []
    warnings = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            continue
        terms = [str(term).strip() for term in raw_group.get("terms", []) if str(term).strip()]
        if not terms:
            continue
        columns, column_warnings = _resolve_search_columns(df, raw_group.get("columns"))
        warnings.extend(column_warnings)
        if not columns:
            continue
        groups.append({"concept": raw_group.get("concept"), "columns": columns, "terms": terms})
        for column in columns:
            if column not in all_columns:
                all_columns.append(column)
        all_terms.extend(terms)

    if not groups:
        return None
    return {
        "groups": groups,
        "columns": all_columns,
        "terms": list(dict.fromkeys(all_terms)),
        "warnings": _dedupe_warnings(warnings),
    }


def _prepared_search_groups(columns, terms, column_term_groups, case_sensitive):
    if column_term_groups:
        prepared = []
        for group in column_term_groups:
            group_terms = []
            for term in group.get("terms", []):
                original = str(term)
                group_terms.append((original, original if case_sensitive else original.lower()))
            prepared.append({"columns": group.get("columns") or columns, "terms": group_terms})
        return prepared

    prepared_terms = []
    for term in terms:
        original = str(term)
        prepared_terms.append((original, original if case_sensitive else original.lower()))
    return [{"columns": columns, "terms": prepared_terms}]


def _display_columns_for_text_search(df, search_columns, params):
    requested = params.get("display_columns")
    if requested is None:
        requested = params.get("return_columns")
    if requested:
        columns, warnings = _resolve_columns(df, requested, required=False)
    else:
        columns, warnings = [], []

    if not columns:
        columns = _default_text_search_display_columns(df, params, search_columns)

    if params.get("include_match_reason", True):
        for required in [MATCH_REASON_COLUMN]:
            if required in df.columns and required not in columns:
                columns.append(required)

    return columns, warnings


def _default_text_search_display_columns(df, params, search_columns):
    question = str(params.get("question") or "")
    columns = []
    for semantic in DEFAULT_DISPLAY_SEMANTICS:
        column, _warnings = _resolve_column(df, semantic)
        if column and column not in columns:
            columns.append(column)

    for semantic, keywords in DISPLAY_REQUEST_KEYWORDS.items():
        if _contains_word_or_phrase(question, keywords):
            column, _warnings = _resolve_column(df, semantic)
            if column and column not in columns:
                insert_at = 1 if semantic == "grad_year" and columns else len(columns)
                columns.insert(insert_at, column)

    if not columns:
        columns = [column for column in search_columns if column in df.columns][:3]
    return columns


def _search_terms_from_params(params):
    terms = []
    raw_terms = params.get("terms")
    if isinstance(raw_terms, list):
        terms.extend(raw_terms)
    elif raw_terms:
        terms.append(raw_terms)
    for key in ["term", "query", "text"]:
        if params.get(key):
            terms.append(params.get(key))
    return [str(term).strip() for term in terms if str(term).strip()]


def _infer_searchable_columns(df):
    text_columns = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            text_columns.append(str(column))

    prioritized = []
    for keyword in SEARCHABLE_COLUMN_KEYWORDS:
        keyword_norm = _normalize_compact(keyword)
        for column in text_columns:
            column_norm = _normalize_compact(column)
            if keyword_norm in column_norm or column_norm in keyword_norm:
                if column not in prioritized:
                    prioritized.append(column)

    if prioritized:
        return prioritized[:8]
    return text_columns[:8]


def _missing_mask(series):
    string_values = series.astype("string")
    return series.isna() | string_values.fillna("").str.strip().eq("")


def _is_missing_value(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _numeric_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned.mask(cleaned.eq(""))
    return pd.to_numeric(cleaned, errors="coerce")


def _date_series(series):
    if not pd.api.types.is_datetime64_any_dtype(series):
        non_missing = series[~_missing_mask(series)]
        if non_missing.empty:
            return pd.to_datetime(series, errors="coerce")
        sample = " ".join(non_missing.astype(str).head(10).tolist()).lower()
        looks_date_like = re.search(
            r"(\d{4}[-/]\d{1,2})|(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)",
            sample,
        )
        if not looks_date_like:
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce")


def _sort_dataframe(df, column, ascending):
    numeric = _numeric_series(df[column])
    if numeric.notna().sum() > 0:
        return df.assign(_sort_value=numeric).sort_values(
            "_sort_value", ascending=ascending, na_position="last", kind="mergesort"
        )
    return df.sort_values(
        column,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
        key=lambda values: values.astype("string").str.lower(),
    )


def _group_keys(series):
    values = series.astype("string")
    return values.mask(_missing_mask(series), "Missing")


def _format_group_key(value):
    if _is_missing_value(value):
        return "Missing"
    return str(value)


def _format_match_reason(match):
    column, _term, value = match
    return f"Matched {column}: {value}"


def _dedupe_match_reasons(matches):
    reasons = []
    seen = set()
    for match in matches:
        column, _term, value = match
        key = (column, value)
        if key in seen:
            continue
        seen.add(key)
        reasons.append(_format_match_reason(match))
    return reasons


def _contains_word_or_phrase(text, terms):
    return _shared_contains_word_or_phrase(text, terms)


def _infer_type(series):
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    numeric = _numeric_series(series)
    non_missing = (~_missing_mask(series)).sum()
    if non_missing and numeric.notna().sum() / non_missing >= 0.8:
        return "number"
    parsed = _date_series(series)
    if non_missing and parsed.notna().sum() / non_missing >= 0.8:
        return "date"
    return "text"


def _warning(kind, message, requested=None, resolved_to=None, suggestions=None):
    warning = {
        "type": str(kind),
        "message": str(message),
    }
    if requested is not None:
        warning["requested"] = requested
    if resolved_to is not None:
        warning["resolved_to"] = resolved_to
    if suggestions:
        warning["suggestions"] = suggestions
    return warning


def _dedupe_warnings(warnings):
    return _shared_dedupe_warnings(warnings)


def _normalize_name(value):
    return _shared_normalize_text(value)


def _normalize_compact(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _synonym_candidates(requested):
    requested_norm = _normalize_compact(requested)
    candidates = []
    for group in COLUMN_SYNONYM_GROUPS:
        group_norms = [_normalize_compact(item) for item in group]
        if requested_norm in group_norms:
            candidates.extend(group)
            continue
        if any(len(item_norm) >= 4 and (item_norm in requested_norm or requested_norm in item_norm) for item_norm in group_norms):
            candidates.extend(group)
    return list(dict.fromkeys(candidates))


def _similar_columns(df, requested):
    requested_norm = _normalize_name(requested)
    if not requested_norm:
        return []
    matches = []
    for column in df.columns:
        column_norm = _normalize_name(column)
        if requested_norm in column_norm or column_norm in requested_norm:
            matches.append(str(column))
    return matches[:5]
