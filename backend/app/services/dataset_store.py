import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import current_app, has_app_context
from werkzeug.utils import secure_filename

from app.services.registry_store import (
    generate_unique_id,
    load_registry,
    save_registry,
)
from app.services.spreadsheet_service import clean_dataframe


class DatasetStoreError(Exception):
    status_code = 500


class DatasetValidationError(DatasetStoreError):
    status_code = 400


class DatasetNotFoundError(DatasetStoreError):
    status_code = 404


class DatasetFileMissingError(DatasetStoreError):
    status_code = 404


class DatasetRegistryError(DatasetStoreError):
    status_code = 500


class DatasetReadError(DatasetStoreError):
    status_code = 400


def get_storage_paths():
    backend_dir = Path(__file__).resolve().parents[2]

    if has_app_context():
        upload_folder = current_app.config.get("UPLOAD_FOLDER", backend_dir / "uploads")
        data_folder = current_app.config.get("DATA_FOLDER", backend_dir / "data")
        registry_path = current_app.config.get(
            "DATASET_REGISTRY_PATH", Path(data_folder) / "datasets.json"
        )
    else:
        upload_folder = backend_dir / "uploads"
        data_folder = backend_dir / "data"
        registry_path = data_folder / "datasets.json"

    return {
        "upload_folder": Path(upload_folder),
        "data_folder": Path(data_folder),
        "registry_path": Path(registry_path),
    }


def ensure_storage_dirs():
    paths = get_storage_paths()
    paths["upload_folder"].mkdir(parents=True, exist_ok=True)
    paths["data_folder"].mkdir(parents=True, exist_ok=True)

    registry_path = paths["registry_path"]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if not registry_path.exists():
        registry_path.write_text("{}\n", encoding="utf-8")


def load_dataset_registry():
    ensure_storage_dirs()
    return load_registry(
        get_storage_paths()["registry_path"], error_cls=DatasetRegistryError, label="Dataset"
    )


def save_dataset_registry(registry):
    ensure_storage_dirs()
    save_registry(
        registry, get_storage_paths()["registry_path"], error_cls=DatasetRegistryError, label="Dataset"
    )


def generate_dataset_id():
    return generate_unique_id({})


def get_file_type(filename):
    extension = Path(str(filename or "")).suffix.lower()
    if extension == ".csv":
        return "csv"
    if extension == ".xlsx":
        return "xlsx"
    return None


def build_stored_filename(dataset_id, original_filename):
    safe_filename = secure_filename(original_filename or "")
    file_type = get_file_type(safe_filename or original_filename)

    if not file_type:
        raise DatasetValidationError("Unsupported file type. Please upload a .csv or .xlsx file.")

    if not safe_filename:
        safe_filename = f"upload.{file_type}"

    return f"{dataset_id}_{safe_filename}"


def save_uploaded_file(file_storage, dataset_id):
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise DatasetValidationError("No file selected.")

    file_type = get_file_type(file_storage.filename)
    if not file_type:
        raise DatasetValidationError("Unsupported file type. Please upload a .csv or .xlsx file.")

    ensure_storage_dirs()
    paths = get_storage_paths()
    stored_filename = build_stored_filename(dataset_id, file_storage.filename)
    absolute_file_path = paths["upload_folder"] / stored_filename

    try:
        file_storage.save(absolute_file_path)
    except OSError as exc:
        raise DatasetStoreError(f"Could not save uploaded file: {exc}") from exc

    return {
        "original_filename": file_storage.filename,
        "stored_filename": stored_filename,
        "file_path": f"uploads/{stored_filename}",
        "absolute_file_path": absolute_file_path,
        "file_type": file_type,
    }


def read_dataframe_from_path(file_path, file_type):
    absolute_file_path = _resolve_dataset_file_path(file_path)

    if not absolute_file_path.exists():
        raise DatasetFileMissingError("Dataset file is missing from storage.")

    try:
        if file_type == "csv":
            df = pd.read_csv(absolute_file_path)
        elif file_type == "xlsx":
            df = pd.read_excel(absolute_file_path, engine="openpyxl")
        else:
            raise DatasetValidationError("Unsupported stored dataset file type.")
    except pd.errors.EmptyDataError as exc:
        raise DatasetReadError("Spreadsheet is empty or has no readable columns.") from exc
    except DatasetStoreError:
        raise
    except Exception as exc:
        raise DatasetReadError(f"Could not read dataset file: {exc}") from exc

    if df.shape[1] == 0:
        raise DatasetReadError("Spreadsheet has no readable columns.")

    return clean_dataframe(df)


def create_dataset_metadata(dataset_id, file_metadata, df):
    return {
        "dataset_id": dataset_id,
        "display_name": file_metadata["original_filename"],
        "original_filename": file_metadata["original_filename"],
        "stored_filename": file_metadata["stored_filename"],
        "file_path": file_metadata["file_path"],
        "file_type": file_metadata["file_type"],
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
    }


def register_uploaded_dataset(file_storage):
    registry = load_dataset_registry()
    dataset_id = generate_dataset_id()
    while dataset_id in registry:
        dataset_id = generate_dataset_id()

    file_metadata = save_uploaded_file(file_storage, dataset_id)

    try:
        df = read_dataframe_from_path(
            file_metadata["absolute_file_path"], file_metadata["file_type"]
        )
    except DatasetStoreError:
        Path(file_metadata["absolute_file_path"]).unlink(missing_ok=True)
        raise

    metadata = create_dataset_metadata(dataset_id, file_metadata, df)
    registry[dataset_id] = metadata
    save_dataset_registry(registry)
    return metadata


def get_dataset_metadata(dataset_id):
    registry = load_dataset_registry()
    return registry.get(str(dataset_id))


MAX_DISPLAY_NAME_LENGTH = 120


def dataset_public_metadata(metadata):
    """Shape one registry entry for the dataset library API: tolerate missing
    fields from older registries and report file availability without loading
    the DataFrame."""
    metadata = metadata if isinstance(metadata, dict) else {}
    file_path = metadata.get("file_path")
    status = "ready"
    try:
        if not file_path or not _resolve_dataset_file_path(file_path).exists():
            status = "missing"
    except OSError:
        status = "missing"
    original_filename = metadata.get("original_filename") or ""
    return {
        "dataset_id": metadata.get("dataset_id"),
        "display_name": metadata.get("display_name") or original_filename or "Untitled dataset",
        "original_filename": original_filename,
        "stored_filename": metadata.get("stored_filename"),
        "uploaded_at": metadata.get("uploaded_at"),
        "row_count": metadata.get("row_count"),
        "column_count": metadata.get("column_count"),
        "columns": metadata.get("columns") or [],
        "file_type": metadata.get("file_type"),
        "status": status,
    }


def list_datasets():
    """All saved datasets, newest first. Never loads DataFrame contents.

    uploaded_at has second resolution, so ties are broken by registry insertion
    order (later insertion = newer).
    """
    registry = load_dataset_registry()
    indexed = [
        (str(metadata.get("uploaded_at") or "") if isinstance(metadata, dict) else "", position, metadata)
        for position, metadata in enumerate(registry.values())
    ]
    indexed.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [dataset_public_metadata(metadata) for _uploaded_at, _position, metadata in indexed]


def rename_dataset(dataset_id, display_name):
    name = str(display_name or "").strip()
    if not name:
        raise DatasetValidationError("display_name must not be empty.")
    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        name = name[:MAX_DISPLAY_NAME_LENGTH].strip()

    registry = load_dataset_registry()
    metadata = registry.get(str(dataset_id))
    if metadata is None:
        raise DatasetNotFoundError("Dataset not found.")

    metadata["display_name"] = name
    save_dataset_registry(registry)
    return dataset_public_metadata(metadata)


def delete_dataset(dataset_id):
    registry = load_dataset_registry()
    metadata = registry.pop(str(dataset_id), None)
    if metadata is None:
        raise DatasetNotFoundError("Dataset not found.")

    file_path = metadata.get("file_path")
    if file_path:
        try:
            _resolve_dataset_file_path(file_path).unlink(missing_ok=True)
        except OSError:
            pass  # metadata removal still proceeds; the orphan file is harmless
    save_dataset_registry(registry)
    return dataset_public_metadata(metadata)


def load_dataset_dataframe(dataset_id):
    metadata = get_dataset_metadata(dataset_id)
    if metadata is None:
        raise DatasetNotFoundError("Dataset not found.")

    df = read_dataframe_from_path(metadata.get("file_path"), metadata.get("file_type"))
    return df, metadata


def _resolve_dataset_file_path(file_path):
    path = Path(str(file_path or ""))
    if path.is_absolute():
        return path

    paths = get_storage_paths()
    stored_filename = path.name

    if path.parts and path.parts[0] == "uploads":
        return paths["upload_folder"] / stored_filename

    if path.parts:
        candidate = paths["data_folder"].parent / path
        if candidate.exists():
            return candidate

    return paths["upload_folder"] / stored_filename
