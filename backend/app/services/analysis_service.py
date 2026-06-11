import pandas as pd

from app.services.spreadsheet_service import to_json_safe


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
