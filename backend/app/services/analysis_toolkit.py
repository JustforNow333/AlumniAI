import json
import os
import re
import warnings
from functools import lru_cache

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
    ["city", "town"],
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
DISPLAY_COLUMN_ALIASES = {
    "first_name": ["First Name", "first name", "first_name", "FirstName", "given name"],
    "last_name": ["Last Name", "LastName", "last name", "last_name", "surname", "family name"],
    "occupation": ["Occupation", "occupation", "job title", "title", "role", "position"],
    "employer": ["Employer", "employer", "company", "organization", "organisation", "workplace"],
    "linkedin_url": [
        "LinkedIn URL",
        "LinkedinURL",
        "LinkedInURL",
        "LinkedIn",
        "Linkedin",
        "linkedin_url",
        "linkedin",
    ],
}

TECH_PEOPLE_FILTER_MODE = "tech_people"
PEOPLE_FILTER_INTENT = "people_filter"
PEOPLE_FILTER_ENTITY = "alumni"
PEOPLE_FILTER_CRITERIA_LABEL = "working in tech or technical roles"
PEOPLE_FILTER_ANSWER_LABEL = "Alumni matching criteria"
KNOWN_TECH_COMPANIES_FILE = os.path.join(os.path.dirname(__file__), "known_tech_companies.json")

TECHNICAL_TITLE_PATTERNS = [
    r"\bsoftware engineer\b",
    r"\bsoftware developer\b",
    r"\bdeveloper\b",
    r"\bprogrammer\b",
    r"\bdata scientist\b",
    r"\bdata engineer\b",
    r"\bmachine learning\b",
    r"\bml engineer\b",
    r"\bai\b",
    r"\bartificial intelligence\b",
    r"\bproduct manager\b",
    r"\btechnical product manager\b",
    r"\bengineering manager\b",
    r"\binformation technology\b",
    r"\bit\b",
    r"\bcybersecurity\b",
    r"\bcloud\b",
    r"\bsystems engineer\b",
    r"\bsystems administrator\b",
    r"\bsystems analyst\b",
    r"\bdatabase\b",
    r"\banalytics\b",
    r"\btechnical service\b",
    r"\btechnical services\b",
    r"\bplatform\b",
    r"\binfrastructure\b",
    r"\bsolutions engineer\b",
    r"\bsales engineer\b",
    r"\btechnical consultant\b",
    r"\bsoftware architect\b",
    r"\bdevops\b",
    r"\bsite reliability\b",
    r"\bsre\b",
    r"\bcto\b",
    r"\bchief technology officer\b",
    r"\bcomputer scientist\b",
    r"\bfull stack\b",
    r"\bbackend\b",
    r"\bfront end\b",
    r"\bfrontend\b",
]

STRONG_TECH_EMPLOYER_TERMS = [
    "technologies",
    "technology",
    "software",
    "ai",
    "data",
    "cloud",
    "systems",
    "labs",
    "platform",
    "digital",
    "analytics",
    "cybersecurity",
    "fintech",
    "blockchain",
    "crypto",
    "saas",
    "app",
    "internet",
]

EMPLOYER_DESCRIPTOR_KEYWORDS = [
    "industry",
    "sector",
    "company description",
    "employer description",
    "organization description",
    "organisation description",
    "business description",
]

NON_TECH_CONTEXT_TERMS = [
    "school",
    "middle school",
    "high school",
    "teacher",
    "department chair",
    "professor",
    "education",
    "hospital",
    "medical center",
    "healthcare",
    "health care",
    "oncology",
    "clinical",
    "surgery",
    "physician",
    "doctor",
    "law",
    "legal",
    "real estate",
    "insurance",
]

WEAK_AMBIGUOUS_EMPLOYER_TERMS = [
    "venture",
    "ventures",
    "innovation",
    "innovations",
    "dao",
    "capital",
    "partners",
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
        if params.get("filter_mode") == TECH_PEOPLE_FILTER_MODE:
            return _tech_people_filter_result(
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

    if params.get("filter_mode") == TECH_PEOPLE_FILTER_MODE:
        return _tech_people_filter_result(df, columns, terms, params, assumptions, warnings)

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


def _tech_people_filter_result(df, columns, terms, params, assumptions, warnings=None, column_term_groups=None):
    warnings = list(warnings or [])
    limit = _limit(params.get("limit"), default=DEFAULT_LIMIT)
    display_specs = _people_display_column_specs(df, params)
    display_headers = [spec["header"] for spec in display_specs]
    known_companies = _known_tech_companies()
    raw_terms = list(dict.fromkeys([str(term).strip() for term in terms if str(term).strip()] + known_companies))
    raw_match_count = _raw_keyword_hit_count(df, columns, raw_terms)

    confirmed_rows = []
    debug_rows = []
    seen_confirmed = set()
    uncertain_keys = set()

    for index, row in df.iterrows():
        classification = _classify_tech_people_row(df, row, known_companies)
        dedupe_key = _person_dedupe_key(df, row, index)

        if classification.get("is_match"):
            if dedupe_key in seen_confirmed:
                continue
            seen_confirmed.add(dedupe_key)
            debug_rows.append(
                {
                    "row_index": int(index) if isinstance(index, (int, np.integer)) else str(index),
                    "match_reason": classification.get("reason", ""),
                    "classification": classification.get("classification", ""),
                    "confidence": classification.get("confidence"),
                }
            )
            if len(confirmed_rows) < limit:
                confirmed_rows.append(
                    {
                        spec["header"]: _row_value_for_display(row, spec["source"])
                        for spec in display_specs
                    }
                )
            continue

        if classification.get("is_uncertain") and dedupe_key not in seen_confirmed:
            uncertain_keys.add(dedupe_key)

    total_matches = len(seen_confirmed)
    displayed_count = len(confirmed_rows)
    uncertain_count = len(uncertain_keys - seen_confirmed)

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

    metrics = {
        "total_dataset_rows": int(df.shape[0]),
        "total_keyword_hits": raw_match_count,
        "total_matches": total_matches,
        "displayed_count": displayed_count,
        "display_limit": limit,
        "uncertain_count": uncertain_count,
        "total_rows": int(df.shape[0]),
        "raw_match_count": raw_match_count,
        "matched_row_count": total_matches,
        "returned_row_count": displayed_count,
        "deduplicated": True,
        "search_columns": columns,
        "display_columns": display_headers,
        "search_terms": raw_terms,
        # Backward-compatible aliases for older frontend/tests.
        "rows_matched": total_matches,
        "rows_returned": displayed_count,
        "searched_columns": columns,
    }
    extras = {
        "intent": PEOPLE_FILTER_INTENT,
        "entity": PEOPLE_FILTER_ENTITY,
        "criteria_label": PEOPLE_FILTER_CRITERIA_LABEL,
        "answer_label": PEOPLE_FILTER_ANSWER_LABEL,
        "total_dataset_rows": int(df.shape[0]),
        "total_keyword_hits": raw_match_count,
        "total_matches": total_matches,
        "displayed_count": displayed_count,
        "display_limit": limit,
        "uncertain_count": uncertain_count,
        "visible_columns": display_headers,
        "search_columns": columns,
        "display_columns": display_headers,
        "debug": {"rows": debug_rows},
    }
    return _ok_result(
        "contains_any",
        f"{PEOPLE_FILTER_ANSWER_LABEL}: {total_matches}",
        display_headers,
        confirmed_rows,
        metrics,
        assumptions,
        warnings,
        is_filtered=True,
        extras=extras,
    )


def _people_display_column_specs(df, params):
    first = _resolved_display_column(df, "first_name")
    last = _resolved_display_column(df, "last_name")
    occupation = _resolved_display_column(df, "occupation")
    employer = _resolved_display_column(df, "employer")
    linkedin = _resolved_display_column(df, "linkedin_url")

    specs = []
    if first:
        specs.append({"header": "First Name", "source": first})
    if last:
        specs.append({"header": "Last Name", "source": last})
    if not first and not last:
        fallback_name = _resolved_display_column(df, "person_name") or _resolved_display_column(df, "name")
        if fallback_name:
            specs.append({"header": str(fallback_name), "source": fallback_name})
    if occupation:
        specs.append({"header": "Occupation", "source": occupation})
    if employer:
        specs.append({"header": "Employer", "source": employer})

    requested = params.get("display_columns") or params.get("return_columns")
    requested_columns, _warnings = _resolve_columns(df, requested, required=False)
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

    if linkedin:
        specs = [spec for spec in specs if spec["source"] != linkedin]
        specs.append({"header": "LinkedIn URL", "source": linkedin})

    return specs


def _resolved_display_column(df, semantic):
    if semantic in DISPLAY_COLUMN_ALIASES:
        column = _resolve_column_by_aliases(df, DISPLAY_COLUMN_ALIASES[semantic])
        if column:
            return column
        if semantic in {"first_name", "last_name", "linkedin_url"}:
            return None
    column, _warnings = _resolve_column(df, semantic)
    return column


def _resolve_column_by_aliases(df, aliases):
    for alias in aliases:
        alias_text = str(alias).strip()
        if alias_text in df.columns:
            return alias_text
    for alias in aliases:
        alias_text = str(alias).strip()
        for column in df.columns:
            if alias_text.casefold() == str(column).casefold():
                return str(column)
    normalized_aliases = {_normalize_compact(alias) for alias in aliases if _normalize_compact(alias)}
    for column in df.columns:
        if _normalize_compact(column) in normalized_aliases:
            return str(column)
    return None


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


def _classify_tech_people_row(df, row, known_companies):
    occupation_col = _resolved_display_column(df, "occupation")
    employer_col = _resolved_display_column(df, "employer")
    occupation = _safe_row_text(row, occupation_col)
    employer = _safe_row_text(row, employer_col)

    if is_explicit_technical_title(occupation):
        return {
            "is_match": True,
            "is_uncertain": False,
            "classification": "technical_title",
            "confidence": 1.0,
            "reason": f"Explicit technical title: {occupation}",
        }

    employer_classification = _classify_tech_employer(df, row, employer, known_companies, occupation=occupation)
    if employer_classification.get("is_tech_company"):
        return {
            "is_match": True,
            "is_uncertain": False,
            "classification": employer_classification.get("classification", "tech_company"),
            "confidence": employer_classification.get("confidence", 0.9),
            "reason": employer_classification.get("reason", ""),
        }

    if employer_classification.get("classification") == "uncertain":
        return {
            "is_match": False,
            "is_uncertain": True,
            "classification": "uncertain",
            "confidence": employer_classification.get("confidence", 0.5),
            "reason": employer_classification.get("reason", ""),
        }

    return {
        "is_match": False,
        "is_uncertain": False,
        "classification": "non_tech_company",
        "confidence": employer_classification.get("confidence", 0.0),
        "reason": employer_classification.get("reason", ""),
    }


def _classify_tech_employer(df, row, employer, known_companies, occupation=""):
    employer_text = str(employer or "")
    descriptor_text = " ".join(_employer_descriptor_values(df, row))
    combined_context = " ".join(item for item in [employer_text, descriptor_text] if item).strip()

    status = classify_employer_tech_status(
        employer_text,
        occupation=occupation,
        known_companies=known_companies,
        descriptor_text=descriptor_text,
    )
    classification = {
        "confirmed_tech": "tech_company",
        "confirmed_non_tech": "non_tech_company",
        "uncertain": "uncertain",
    }[status["status"]]
    return {
        "is_tech_company": status["status"] == "confirmed_tech",
        "confidence": status["confidence"],
        "classification": classification if status["source"] != "known_company" else "known_tech_company",
        "reason": status["internal_reason"],
    }


def is_explicit_technical_title(occupation):
    return _is_explicit_technical_title(occupation)


def classify_employer_tech_status(employer, occupation="", known_companies=None, descriptor_text=""):
    known_companies = list(known_companies or _known_tech_companies())
    employer_text = str(employer or "")
    descriptor_text = str(descriptor_text or "")
    combined_context = " ".join(item for item in [employer_text, descriptor_text] if item).strip()

    known_match = _known_company_match(employer_text, known_companies)
    if known_match:
        return _tech_classification(
            "confirmed_tech",
            "known_company",
            0.95,
            f"Employer matches known tech company list: {known_match}",
        )

    strong_match = _matched_term(employer_text, STRONG_TECH_EMPLOYER_TERMS)
    if strong_match:
        return _tech_classification(
            "confirmed_tech",
            "strong_keyword",
            0.9,
            f"Employer name contains strong tech indicator: {strong_match}",
        )

    descriptor_match = _matched_term(descriptor_text, STRONG_TECH_EMPLOYER_TERMS)
    if descriptor_match and not is_strong_non_tech_context(occupation, combined_context):
        return _tech_classification(
            "confirmed_tech",
            "strong_keyword",
            0.82,
            f"Employer descriptor contains strong tech indicator: {descriptor_match}",
        )

    if is_strong_non_tech_context(occupation, combined_context):
        return _tech_classification(
            "confirmed_non_tech",
            "non_tech_exclusion",
            0.9,
            "Employer or role context strongly indicates a non-tech domain.",
        )

    weak_match = _matched_term(employer_text, WEAK_AMBIGUOUS_EMPLOYER_TERMS)
    if weak_match:
        return _tech_classification(
            "uncertain",
            "none",
            0.45,
            f"Employer has ambiguous startup/company wording: {weak_match}",
        )

    return _tech_classification(
        "confirmed_non_tech",
        "none",
        0.0,
        "No strong technical title or tech-company signal was found.",
    )


def is_strong_non_tech_context(occupation, employer):
    if is_explicit_technical_title(occupation):
        return False
    return _contains_word_or_phrase(" ".join([str(occupation or ""), str(employer or "")]), NON_TECH_CONTEXT_TERMS)


def _tech_classification(status, source, confidence, internal_reason):
    return {
        "status": status,
        "source": source,
        "confidence": float(confidence),
        "internal_reason": internal_reason,
    }


def _is_explicit_technical_title(value):
    normalized = _normalize_name(value)
    if not normalized:
        return False
    for pattern in TECHNICAL_TITLE_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return True
    return False


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


def _matched_term(text, terms):
    for term in terms:
        if _text_has_term(text, term):
            return term
    return ""


def _text_has_term(text, term):
    normalized_text = _normalize_name(text)
    normalized_term = _normalize_name(term)
    if not normalized_text or not normalized_term:
        return False
    if " " in normalized_term:
        return normalized_term in normalized_text
    return re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text) is not None


def _known_company_match(employer, known_companies):
    employer_norm = _normalize_company_name(employer)
    if not employer_norm:
        return ""
    for company in known_companies:
        company_norm = _normalize_company_name(company)
        if not company_norm:
            continue
        if company_norm == employer_norm or re.search(rf"\b{re.escape(company_norm)}\b", employer_norm):
            return company
    return ""


def _normalize_company_name(value):
    normalized = _normalize_name(value)
    suffixes = {
        "inc",
        "incorporated",
        "llc",
        "l l c",
        "ltd",
        "limited",
        "corp",
        "corporation",
        "co",
        "company",
    }
    words = [word for word in normalized.split() if word not in suffixes]
    return " ".join(words)


@lru_cache(maxsize=1)
def _known_tech_companies():
    try:
        with open(KNOWN_TECH_COMPANIES_FILE, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        loaded = []
    if not isinstance(loaded, list):
        return []
    return [str(item).strip() for item in loaded if str(item).strip()]


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
