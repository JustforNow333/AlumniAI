import re

import pandas as pd

from app.services.spreadsheet_service import dataframe_preview, to_json_safe


class AnalysisError(ValueError):
    pass


def summarize_dataframe(df):
    missing_values = df.isna().sum()

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "column_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {
            column: int(count) for column, count in missing_values.to_dict().items()
        },
        "total_missing_values": int(missing_values.sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
        "numeric_summary": _numeric_summary(df),
        "categorical_summary": _categorical_summary(df),
        "date_summary": _date_summary(df),
    }

    return to_json_safe(summary)


def summarize_column(df, column):
    column = resolve_column(df, column)
    series = df[column]
    missing_count = int(series.isna().sum())
    date_series = _as_datetime_if_date_like(series)

    if date_series is not None:
        non_missing = date_series.dropna()
        return to_json_safe(
            {
                "column": column,
                "type": "date",
                "earliest": non_missing.min() if not non_missing.empty else None,
                "latest": non_missing.max() if not non_missing.empty else None,
                "missing_count": missing_count,
            }
        )

    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
        return to_json_safe(
            {
                "column": column,
                "type": "numeric",
                "count": int(numeric_series.count()),
                "mean": numeric_series.mean(),
                "median": numeric_series.median(),
                "min": numeric_series.min(),
                "max": numeric_series.max(),
                "sum": numeric_series.sum(),
                "standard_deviation": numeric_series.std(),
                "missing_count": missing_count,
            }
        )

    return to_json_safe(
        {
            "column": column,
            "type": "categorical",
            "unique_count": int(series.nunique(dropna=True)),
            "missing_count": missing_count,
            "top_values": series.value_counts(dropna=True).head(10).to_dict(),
        }
    )


def group_by_aggregate(df, group_col, value_col, operation):
    group_col = resolve_column(df, group_col)
    value_col = resolve_column(df, value_col)
    operation = str(operation).lower().strip()
    allowed_operations = {"sum", "mean", "count", "min", "max"}

    if operation not in allowed_operations:
        raise AnalysisError(
            f"Unsupported aggregation '{operation}'. Use sum, mean, count, min, or max."
        )

    if operation == "count":
        grouped = df.groupby(group_col, dropna=False).size()
    else:
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            raise AnalysisError(f"Column '{value_col}' must be numeric for {operation}.")
        work_df = df[[group_col, value_col]].copy()
        grouped = work_df.groupby(group_col, dropna=False)[value_col].agg(operation)

    ascending = operation == "min"
    grouped = grouped.sort_values(ascending=ascending).head(20)
    value_name = f"{operation}_{value_col}"
    result_map = {
        _format_group_key(group): to_json_safe(value) for group, value in grouped.items()
    }
    result_rows = [
        {group_col: _format_group_key(group), value_name: to_json_safe(value)}
        for group, value in grouped.items()
    ]

    return to_json_safe(
        {
            "group_col": group_col,
            "value_col": value_col,
            "operation": operation,
            "limit": 20,
            "results": result_map,
            "rows": result_rows,
        }
    )


def top_rows(df, sort_col, limit=10, ascending=False):
    sort_col = resolve_column(df, sort_col)
    limit = _clamp_limit(limit)

    try:
        sorted_df = df.sort_values(
            by=sort_col,
            ascending=ascending,
            na_position="last",
            kind="mergesort",
        )
    except TypeError:
        sorted_df = df.sort_values(
            by=sort_col,
            ascending=ascending,
            na_position="last",
            kind="mergesort",
            key=lambda values: values.astype("string").str.lower(),
        )

    return to_json_safe(
        {
            "sort_col": sort_col,
            "limit": limit,
            "ascending": bool(ascending),
            "rows": dataframe_preview(sorted_df, limit=limit),
        }
    )


def filter_rows_basic(df, column, operator, value):
    column = resolve_column(df, column)
    operator = _normalize_operator(operator)
    series = df[column]

    if operator == "equals":
        mask = _equals_mask(series, value)
    elif operator == "contains":
        mask = series.astype("string").str.contains(
            str(value), case=False, na=False, regex=False
        )
    elif operator in {"greater_than", "less_than"}:
        mask = _comparison_mask(series, operator, value)
    else:
        raise AnalysisError(
            "Unsupported filter operator. Use equals, contains, greater_than, or less_than."
        )

    filtered_df = df[mask.fillna(False)]

    return to_json_safe(
        {
            "column": column,
            "operator": operator,
            "value": value,
            "matching_row_count": int(filtered_df.shape[0]),
            "preview": dataframe_preview(filtered_df, limit=10),
        }
    )


def correlation(df, col1, col2):
    col1 = resolve_column(df, col1)
    col2 = resolve_column(df, col2)

    if not pd.api.types.is_numeric_dtype(df[col1]):
        raise AnalysisError(f"Column '{col1}' must be numeric for correlation.")

    if not pd.api.types.is_numeric_dtype(df[col2]):
        raise AnalysisError(f"Column '{col2}' must be numeric for correlation.")

    numeric_df = df[[col1, col2]].dropna()
    coefficient = numeric_df[col1].corr(numeric_df[col2]) if len(numeric_df) >= 2 else None

    return to_json_safe(
        {
            "col1": col1,
            "col2": col2,
            "correlation": coefficient,
            "rows_used": int(len(numeric_df)),
        }
    )


def run_safe_analysis_intent(df, question):
    question = str(question)
    question_lower = question.lower()
    mentioned_columns = _mentioned_columns(question, df.columns)

    try:
        if _asks_for_unsafe_mutation(question_lower):
            return (
                {"type": "analysis_error"},
                {"error": "Only read-only analysis is supported. The dataset was not modified."},
            )

        if _asks_for_dataframe_summary(question_lower):
            return {"type": "summarize_dataframe"}, summarize_dataframe(df)

        if any(word in question_lower for word in ["missing", "null", "blank"]):
            return {"type": "summarize_dataframe"}, summarize_dataframe(df)

        if _asks_for_correlation(question_lower):
            numeric_mentions = [
                column for column in mentioned_columns if pd.api.types.is_numeric_dtype(df[column])
            ]
            if len(numeric_mentions) >= 2:
                col1, col2 = numeric_mentions[:2]
                return {"type": "correlation", "col1": col1, "col2": col2}, correlation(
                    df, col1, col2
                )

        if _asks_for_column_summary(question_lower) and mentioned_columns:
            column = mentioned_columns[0]
            return {"type": "summarize_column", "column": column}, summarize_column(
                df, column
            )

        single_column_summary = _detect_single_column_summary(df, question_lower, mentioned_columns)
        if single_column_summary:
            return (
                {"type": "summarize_column", "column": single_column_summary},
                summarize_column(df, single_column_summary),
            )

        aggregate_intent = _detect_grouped_aggregate(df, question, mentioned_columns)
        if aggregate_intent:
            group_col, value_col, aggregation = aggregate_intent
            result = group_by_aggregate(df, group_col, value_col, aggregation)
            return (
                {
                    "type": "group_by_aggregate",
                    "group_col": group_col,
                    "value_col": value_col,
                    "aggregation": aggregation,
                },
                result["results"],
            )

        sort_intent = _detect_top_rows(df, question_lower, mentioned_columns)
        if sort_intent:
            sort_col, ascending, limit = sort_intent
            return (
                {
                    "type": "top_rows",
                    "sort_col": sort_col,
                    "limit": limit,
                    "ascending": ascending,
                },
                top_rows(df, sort_col, limit=limit, ascending=ascending),
            )
    except AnalysisError as exc:
        return {"type": "analysis_error"}, {"error": str(exc)}

    return None, None


def resolve_column(df, column):
    if column in df.columns:
        return column

    requested = str(column).strip()
    for existing_column in df.columns:
        if str(existing_column).strip().lower() == requested.lower():
            return existing_column

    normalized_requested = _normalize_for_match(requested)
    for existing_column in df.columns:
        if _normalize_for_match(existing_column) == normalized_requested:
            return existing_column

    raise AnalysisError(f"Column '{column}' was not found.")


def _numeric_summary(df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}

    return to_json_safe(numeric_df.describe().transpose().to_dict(orient="index"))


def _categorical_summary(df):
    categorical_columns = df.select_dtypes(include=["object", "str", "category", "bool"]).columns
    summary = {}

    for column in categorical_columns:
        series = df[column]
        summary[column] = {
            "unique_count": int(series.nunique(dropna=True)),
            "missing_count": int(series.isna().sum()),
            "top_values": to_json_safe(series.value_counts(dropna=True).head(10).to_dict()),
        }

    return summary


def _date_summary(df):
    summary = {}

    for column in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            continue

        series = df[column].dropna()
        summary[column] = {
            "earliest": series.min() if not series.empty else None,
            "latest": series.max() if not series.empty else None,
            "missing_count": int(df[column].isna().sum()),
        }

    return to_json_safe(summary)


def _as_datetime_if_date_like(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    is_categorical = isinstance(series.dtype, pd.CategoricalDtype)
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or is_categorical
    ):
        return None

    non_missing = series.notna()
    if not non_missing.any():
        return None

    parsed = pd.to_datetime(series, errors="coerce")
    parse_rate = parsed[non_missing].notna().mean()
    return parsed if parse_rate >= 0.8 else None


def _normalize_for_match(value):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return " ".join(normalized.split())


def _mentioned_columns(question, columns):
    normalized_question = f" {_normalize_for_match(question)} "
    matches = []

    for column in columns:
        normalized_column = _normalize_for_match(column)
        if not normalized_column:
            continue

        search_value = f" {normalized_column} "
        position = normalized_question.find(search_value)
        if position >= 0:
            matches.append((position, -len(normalized_column), column))

    return [column for _, _, column in sorted(matches)]


def _asks_for_correlation(question_lower):
    return any(
        phrase in question_lower
        for phrase in ["correlation", "relationship", "related", "relate"]
    )


def _asks_for_unsafe_mutation(question_lower):
    return any(
        phrase in question_lower
        for phrase in [
            "delete all rows",
            "delete rows",
            "drop the table",
            "drop table",
            "remove all rows",
            "truncate",
            "wipe",
        ]
    )


def _asks_for_dataframe_summary(question_lower):
    return any(
        phrase in question_lower
        for phrase in [
            "summarize this dataset",
            "summarise this dataset",
            "summarize the dataset",
            "summarise the dataset",
            "dataset summary",
            "describe this dataset",
            "describe the dataset",
        ]
    )


def _asks_for_column_summary(question_lower):
    return any(
        phrase in question_lower
        for phrase in ["summarize column", "summary of column", "describe column"]
    )


def _detect_single_column_summary(df, question_lower, mentioned_columns):
    if not mentioned_columns:
        return None

    if " by " in f" {question_lower} ":
        return None

    asks_for_metric = any(
        word in question_lower
        for word in ["average", "mean", "avg", "total", "sum", "minimum", "maximum", "min", "max"]
    )
    if not asks_for_metric:
        return None

    numeric_mentions = [
        column for column in mentioned_columns if pd.api.types.is_numeric_dtype(df[column])
    ]
    return numeric_mentions[0] if numeric_mentions else None


def _detect_grouped_aggregate(df, question, mentioned_columns):
    question_lower = question.lower()
    aggregation = None

    if any(word in question_lower for word in ["average", "mean"]):
        aggregation = "mean"
    elif any(word in question_lower for word in ["total", "sum"]):
        aggregation = "sum"
    elif any(phrase in question_lower for phrase in ["how many", "count", "number of"]):
        aggregation = "count"
    elif any(word in question_lower for word in ["maximum", "max"]):
        aggregation = "max"
    elif any(word in question_lower for word in ["minimum", "min", "lowest"]):
        aggregation = "min"

    if not aggregation:
        return None

    has_grouping_language = " by " in f" {question_lower} "
    has_ranked_total_language = (
        any(word in question_lower for word in ["highest", "largest", "biggest"])
        and aggregation == "sum"
    )

    if not has_grouping_language and not has_ranked_total_language:
        return None

    group_col = _column_after_keyword(question, "by", df.columns)
    numeric_mentions = [
        column for column in mentioned_columns if pd.api.types.is_numeric_dtype(df[column])
    ]
    non_numeric_mentions = [
        column for column in mentioned_columns if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if group_col is None and non_numeric_mentions:
        group_col = non_numeric_mentions[0]

    if group_col is None and mentioned_columns:
        group_col = mentioned_columns[-1]

    value_col = _choose_value_column(mentioned_columns, numeric_mentions, group_col)

    if aggregation == "count" and value_col is None:
        value_col = group_col

    if group_col and value_col:
        return group_col, value_col, aggregation

    return None


def _detect_top_rows(df, question_lower, mentioned_columns):
    if not mentioned_columns:
        return None

    wants_top_rows = bool(
        re.search(r"\b(top|highest|largest|biggest)\b", question_lower)
    )
    wants_lowest_rows = any(
        phrase in question_lower for phrase in ["lowest", "smallest", "bottom"]
    )

    if not wants_top_rows and not wants_lowest_rows:
        return None

    numeric_mentions = [
        column for column in mentioned_columns if pd.api.types.is_numeric_dtype(df[column])
    ]
    sort_col = numeric_mentions[0] if numeric_mentions else mentioned_columns[0]
    limit = _extract_limit(question_lower, default=10)
    return sort_col, bool(wants_lowest_rows), limit


def _extract_limit(question_lower, default=10):
    match = re.search(r"\b(?:top|bottom|first|last|show me the top)\s+(\d{1,3})\b", question_lower)
    if not match:
        match = re.search(r"\b(\d{1,3})\s+rows?\b", question_lower)

    if not match:
        return default

    return _clamp_limit(match.group(1))


def _column_after_keyword(question, keyword, columns):
    lowered = question.lower()
    marker = f" {keyword} "
    index = lowered.rfind(marker)
    if index < 0:
        return None

    tail = question[index + len(marker) :]
    matches = _mentioned_columns(tail, columns)
    return matches[0] if matches else None


def _choose_value_column(mentioned_columns, numeric_mentions, group_col):
    for column in numeric_mentions:
        if column != group_col:
            return column

    for column in mentioned_columns:
        if column != group_col:
            return column

    return None


def _format_group_key(value):
    safe_value = to_json_safe(value)
    return "Missing" if safe_value is None else str(safe_value)


def _clamp_limit(limit):
    try:
        parsed_limit = int(limit)
    except (TypeError, ValueError):
        parsed_limit = 10

    return max(1, min(parsed_limit, 100))


def _normalize_operator(operator):
    operator = str(operator).strip().lower()
    aliases = {
        "eq": "equals",
        "=": "equals",
        "==": "equals",
        "contains": "contains",
        ">": "greater_than",
        "gt": "greater_than",
        "greater than": "greater_than",
        "greater_than": "greater_than",
        "<": "less_than",
        "lt": "less_than",
        "less than": "less_than",
        "less_than": "less_than",
    }

    return aliases.get(operator, operator)


def _equals_mask(series, value):
    if pd.api.types.is_numeric_dtype(series):
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric_value):
            return pd.to_numeric(series, errors="coerce") == numeric_value

    return series.astype("string").str.lower() == str(value).lower()


def _comparison_mask(series, operator, value):
    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            raise AnalysisError(f"Value '{value}' must be numeric for {operator}.")
        if operator == "greater_than":
            return numeric_series > numeric_value
        return numeric_series < numeric_value

    date_series = _as_datetime_if_date_like(series)
    if date_series is not None:
        date_value = pd.to_datetime(value, errors="coerce")
        if pd.isna(date_value):
            raise AnalysisError(f"Value '{value}' must be date-like for {operator}.")
        if operator == "greater_than":
            return date_series > date_value
        return date_series < date_value

    raise AnalysisError(f"Column '{series.name}' must be numeric or date-like for {operator}.")
