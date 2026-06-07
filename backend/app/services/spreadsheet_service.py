from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


DATASETS = {}


class SpreadsheetError(ValueError):
    pass


def read_spreadsheet(file_path):
    extension = Path(file_path).suffix.lower()

    try:
        if extension == ".csv":
            df = pd.read_csv(file_path)
        elif extension == ".xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            raise SpreadsheetError("Unsupported file type. Please upload a .csv or .xlsx file.")
    except pd.errors.EmptyDataError as exc:
        raise SpreadsheetError("Spreadsheet is empty or has no readable columns.") from exc
    except Exception as exc:
        raise SpreadsheetError(f"Could not read spreadsheet: {exc}") from exc

    if df.shape[1] == 0:
        raise SpreadsheetError("Spreadsheet has no readable columns.")

    return clean_dataframe(df)


def clean_dataframe(df):
    cleaned = df.copy()
    cleaned.columns = _make_unique_column_names(cleaned.columns)
    cleaned = _infer_obvious_datetime_columns(cleaned)
    return cleaned


def _make_unique_column_names(columns):
    counts = {}
    names = []

    for column in columns:
        base = str(column).strip()
        if not base or base.lower() == "nan":
            base = "Unnamed"

        counts[base] = counts.get(base, 0) + 1
        name = base if counts[base] == 1 else f"{base}_{counts[base]}"
        names.append(name)

    return names


def _infer_obvious_datetime_columns(df):
    for column in df.columns:
        if not pd.api.types.is_object_dtype(df[column]):
            continue

        column_name = str(column).lower()
        if "date" not in column_name and "time" not in column_name:
            continue

        non_missing = df[column].notna()
        if not non_missing.any():
            continue

        parsed = pd.to_datetime(df[column], errors="coerce")
        parse_rate = parsed[non_missing].notna().mean()
        if parse_rate >= 0.8:
            df[column] = parsed

    return df


def store_dataset(original_filename, saved_file_path, df):
    dataset_id = str(uuid4())
    metadata = create_basic_summary(df, include_preview=False)

    DATASETS[dataset_id] = {
        "original_filename": original_filename,
        "saved_file_path": saved_file_path,
        "dataframe": df,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
    }

    return dataset_id


def get_dataset(dataset_id):
    return DATASETS.get(dataset_id)


def create_basic_summary(df, include_preview=True):
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "column_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {
            column: int(count) for column, count in df.isna().sum().to_dict().items()
        },
    }

    if include_preview:
        summary["preview"] = dataframe_preview(df)

    return to_json_safe(summary)


def get_preview_payload(df, limit=10):
    missing_values = {
        column: int(count) for column, count in df.isna().sum().to_dict().items()
    }
    columns = list(df.columns)
    rows = dataframe_preview(df, limit=limit)

    return to_json_safe({
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_count": int(sum(missing_values.values())),
        "columns": columns,
        "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": missing_values,
        "rows": rows,
        "column_names": list(df.columns),
        "preview": rows,
    })


def dataframe_preview(df, limit=10):
    return to_json_safe(df.head(limit).to_dict(orient="records"))


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(to_json_safe(key)): to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]

    if _is_missing_scalar(value):
        return None

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    return value


def _is_missing_scalar(value):
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
