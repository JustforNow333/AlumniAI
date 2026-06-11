import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from flask import current_app, has_app_context
from werkzeug.utils import secure_filename

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
    registry_path = get_storage_paths()["registry_path"]

    try:
        with registry_path.open("r", encoding="utf-8") as registry_file:
            registry = json.load(registry_file)
    except json.JSONDecodeError as exc:
        raise DatasetRegistryError("Dataset registry is invalid JSON.") from exc
    except OSError as exc:
        raise DatasetRegistryError(f"Could not read dataset registry: {exc}") from exc

    if not isinstance(registry, dict):
        raise DatasetRegistryError("Dataset registry must contain a JSON object.")

    return registry


def save_dataset_registry(registry):
    if not isinstance(registry, dict):
        raise DatasetRegistryError("Dataset registry must be a dictionary.")

    ensure_storage_dirs()
    registry_path = get_storage_paths()["registry_path"]
    temporary_path = registry_path.with_name(f"{registry_path.name}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as registry_file:
            json.dump(registry, registry_file, indent=2)
            registry_file.write("\n")
        os.replace(temporary_path, registry_path)
    except OSError as exc:
        raise DatasetRegistryError(f"Could not save dataset registry: {exc}") from exc


def generate_dataset_id():
    return str(uuid4())


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
