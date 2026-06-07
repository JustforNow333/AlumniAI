import re
import warnings

import numpy as np
import pandas as pd

from app.services.spreadsheet_service import to_json_safe


MAX_LIMIT = 500
DEFAULT_LIMIT = 100

ALLOWED_OPERATION_TYPES = {
    "preview",
    "select_columns",
    "filter_equals",
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
    ["person name", "person_name", "name", "full name", "nickname", "preferred name", "alumni name"],
    ["occupation", "job", "title", "role", "position", "profession"],
    ["employer", "company", "organization", "organisation", "workplace", "firm", "business"],
    ["graduation year", "class year", "grad year", "grad yr", "graduation yr", "class yr", "grad_year"],
    ["name", "full name", "nickname", "preferred name", "alumni name"],
    ["major", "degree", "field of study", "program"],
    ["email", "email address", "e-mail"],
    ["phone", "phone number", "mobile"],
    ["city", "town"],
    ["state", "province", "region"],
]

MATCH_REASON_COLUMN = "MATCH REASON"
TEXT_SEARCH_METADATA_COLUMNS = ["matched_column", "matched_term", MATCH_REASON_COLUMN]
DEFAULT_DISPLAY_SEMANTICS = ["person_name", "occupation", "employer"]
DISPLAY_REQUEST_KEYWORDS = {
    "major": ["major", "majors", "degree", "degrees", "field of study"],
    "grad_year": ["graduation year", "graduation years", "class year", "class years", "grad year", "grad yr"],
    "email": ["email", "emails", "email address", "e-mail"],
    "phone": ["phone", "phones", "phone number", "mobile"],
    "city": ["city", "cities", "location", "locations"],
    "state": ["state", "states", "province", "region"],
}


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

        is_match = len(row_matches) == len(terms) if require_all else bool(row_matches)
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
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    if value < 1:
        value = default
    return min(value, MAX_LIMIT)


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
    normalized = _normalize_name(text)
    for term in terms:
        term_normalized = _normalize_name(term)
        if not term_normalized:
            continue
        if " " in term_normalized:
            if term_normalized in normalized:
                return True
        elif re.search(rf"\b{re.escape(term_normalized)}\b", normalized):
            return True
    return False


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
    deduped = []
    seen = set()
    for warning in warnings or []:
        key = _warning_key(warning)
        if key not in seen:
            seen.add(key)
            deduped.append(warning)
    return deduped


def _warning_key(warning):
    if isinstance(warning, dict):
        return (
            warning.get("type"),
            warning.get("message"),
            str(warning.get("requested")),
            str(warning.get("resolved_to")),
        )
    return ("text", str(warning))


def _normalize_name(value):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return " ".join(normalized.split())


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
