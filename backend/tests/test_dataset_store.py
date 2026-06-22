import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from app import create_app
from app.services import ai_service
from app.services.dataset_store import (
    DatasetFileMissingError,
    DatasetNotFoundError,
    DatasetReadError,
    DatasetRegistryError,
    DatasetStoreError,
    DatasetValidationError,
    _resolve_dataset_file_path,
    build_stored_filename,
    create_dataset_metadata,
    dataset_public_metadata,
    delete_dataset,
    ensure_storage_dirs,
    get_dataset_metadata,
    get_file_type,
    get_storage_paths,
    list_datasets,
    load_dataset_dataframe,
    load_dataset_registry,
    read_dataframe_from_path,
    rename_dataset,
    save_dataset_registry,
    save_uploaded_file,
)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_service, "client", None)
    app = create_app()
    app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        DATA_FOLDER=str(tmp_path / "data"),
        DATASET_REGISTRY_PATH=str(tmp_path / "data" / "datasets.json"),
    )
    yield app


@pytest.fixture
def ctx(app):
    with app.app_context():
        yield app


def make_csv_file(tmp_path, name="test.csv", content="A,B\n1,2\n3,4\n"):
    file_path = tmp_path / name
    file_path.write_text(content)
    return file_path


# --- Exception hierarchy ---

def test_exception_status_codes():
    assert DatasetStoreError("x").status_code == 500
    assert DatasetValidationError("x").status_code == 400
    assert DatasetNotFoundError("x").status_code == 404
    assert DatasetFileMissingError("x").status_code == 404
    assert DatasetRegistryError("x").status_code == 500
    assert DatasetReadError("x").status_code == 400


# --- get_file_type ---

def test_get_file_type_csv():
    assert get_file_type("data.csv") == "csv"


def test_get_file_type_xlsx():
    assert get_file_type("data.xlsx") == "xlsx"


def test_get_file_type_unsupported():
    assert get_file_type("data.txt") is None


def test_get_file_type_none():
    assert get_file_type(None) is None


def test_get_file_type_no_extension():
    assert get_file_type("noext") is None


# --- build_stored_filename ---

def test_build_stored_filename():
    result = build_stored_filename("abc123", "my_data.csv")
    assert result == "abc123_my_data.csv"


def test_build_stored_filename_unsupported():
    with pytest.raises(DatasetValidationError, match="Unsupported"):
        build_stored_filename("abc", "data.txt")


def test_build_stored_filename_empty_name():
    with pytest.raises(DatasetValidationError, match="Unsupported"):
        build_stored_filename("abc", "")


# --- get_storage_paths (outside app context) ---

def test_get_storage_paths_no_app_context():
    paths = get_storage_paths()
    assert "upload_folder" in paths
    assert "data_folder" in paths
    assert "registry_path" in paths


# --- get_storage_paths (with app context) ---

def test_get_storage_paths_with_context(ctx):
    paths = get_storage_paths()
    assert str(paths["upload_folder"]).endswith("uploads")


# --- ensure_storage_dirs ---

def test_ensure_storage_dirs_creates_directories(ctx, tmp_path):
    ensure_storage_dirs()
    paths = get_storage_paths()
    assert paths["upload_folder"].exists()
    assert paths["data_folder"].exists()
    assert paths["registry_path"].exists()


# --- load / save registry ---

def test_load_empty_registry(ctx):
    registry = load_dataset_registry()
    assert registry == {}


def test_save_and_load_registry(ctx):
    save_dataset_registry({"id1": {"dataset_id": "id1"}})
    registry = load_dataset_registry()
    assert "id1" in registry


def test_save_registry_non_dict(ctx):
    with pytest.raises(DatasetRegistryError, match="dictionary"):
        save_dataset_registry("not-a-dict")


def test_load_corrupt_registry(ctx):
    paths = get_storage_paths()
    ensure_storage_dirs()
    paths["registry_path"].write_text("NOT JSON", encoding="utf-8")
    with pytest.raises(DatasetRegistryError, match="invalid JSON"):
        load_dataset_registry()


def test_load_registry_not_object(ctx):
    paths = get_storage_paths()
    ensure_storage_dirs()
    paths["registry_path"].write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(DatasetRegistryError, match="JSON object"):
        load_dataset_registry()


# --- read_dataframe_from_path ---

def test_read_csv(ctx, tmp_path):
    csv_path = tmp_path / "uploads" / "test.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"A": [1], "B": [2]}).to_csv(csv_path, index=False)
    df = read_dataframe_from_path(str(csv_path), "csv")
    assert list(df.columns) == ["A", "B"]
    assert len(df) == 1


def test_read_xlsx(ctx, tmp_path):
    xlsx_path = tmp_path / "uploads" / "test.xlsx"
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"X": [10], "Y": [20]}).to_excel(xlsx_path, index=False)
    df = read_dataframe_from_path(str(xlsx_path), "xlsx")
    assert list(df.columns) == ["X", "Y"]


def test_read_unsupported_type(ctx, tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("data")
    with pytest.raises(DatasetValidationError, match="Unsupported"):
        read_dataframe_from_path(str(path), "txt")


def test_read_missing_file(ctx):
    with pytest.raises(DatasetFileMissingError, match="missing"):
        read_dataframe_from_path("/nonexistent/file.csv", "csv")


def test_read_empty_csv(ctx, tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")
    with pytest.raises(DatasetReadError):
        read_dataframe_from_path(str(csv_path), "csv")


# --- dataset_public_metadata ---

def test_dataset_public_metadata_full(ctx):
    meta = {
        "dataset_id": "abc",
        "display_name": "Test",
        "original_filename": "test.csv",
        "stored_filename": "abc_test.csv",
        "uploaded_at": "2026-01-01T00:00:00",
        "row_count": 5,
        "column_count": 2,
        "columns": ["A", "B"],
        "file_type": "csv",
        "file_path": "uploads/abc_test.csv",
    }
    result = dataset_public_metadata(meta)
    assert result["dataset_id"] == "abc"
    assert result["display_name"] == "Test"
    assert result["status"] == "missing"  # file doesn't actually exist


def test_dataset_public_metadata_none():
    result = dataset_public_metadata(None)
    assert result["display_name"] == "Untitled dataset"
    assert result["status"] == "missing"


def test_dataset_public_metadata_missing_display_name(ctx):
    meta = {"dataset_id": "x", "original_filename": "data.csv"}
    result = dataset_public_metadata(meta)
    assert result["display_name"] == "data.csv"


def test_dataset_public_metadata_no_filename_or_name(ctx):
    result = dataset_public_metadata({"dataset_id": "x"})
    assert result["display_name"] == "Untitled dataset"


# --- create_dataset_metadata ---

def test_create_dataset_metadata():
    df = pd.DataFrame({"Col1": [1, 2], "Col2": ["a", "b"]})
    file_meta = {
        "original_filename": "out.csv",
        "stored_filename": "id_out.csv",
        "file_path": "uploads/id_out.csv",
        "file_type": "csv",
    }
    result = create_dataset_metadata("some-id", file_meta, df)
    assert result["dataset_id"] == "some-id"
    assert result["row_count"] == 2
    assert result["column_count"] == 2
    assert result["columns"] == ["Col1", "Col2"]
    assert result["display_name"] == "out.csv"


# --- save_uploaded_file ---

def test_save_uploaded_file_none(ctx):
    with pytest.raises(DatasetValidationError, match="No file"):
        save_uploaded_file(None, "id1")


def test_save_uploaded_file_no_filename(ctx):
    class FakeFile:
        filename = ""
    with pytest.raises(DatasetValidationError, match="No file"):
        save_uploaded_file(FakeFile(), "id1")


def test_save_uploaded_file_bad_extension(ctx):
    class FakeFile:
        filename = "data.txt"
    with pytest.raises(DatasetValidationError, match="Unsupported"):
        save_uploaded_file(FakeFile(), "id1")


# --- rename_dataset ---

def test_rename_dataset_not_found(ctx):
    ensure_storage_dirs()
    with pytest.raises(DatasetNotFoundError):
        rename_dataset("nonexistent", "new name")


def test_rename_empty_name(ctx):
    ensure_storage_dirs()
    with pytest.raises(DatasetValidationError, match="empty"):
        rename_dataset("id", "")


def test_rename_truncates_long_name(ctx):
    ensure_storage_dirs()
    save_dataset_registry({"id1": {"dataset_id": "id1", "original_filename": "f.csv", "file_path": "uploads/f.csv"}})
    result = rename_dataset("id1", "A" * 200)
    assert len(result["display_name"]) <= 120


# --- delete_dataset ---

def test_delete_nonexistent(ctx):
    ensure_storage_dirs()
    with pytest.raises(DatasetNotFoundError):
        delete_dataset("nope")


def test_delete_without_file_path(ctx):
    ensure_storage_dirs()
    save_dataset_registry({"id1": {"dataset_id": "id1"}})
    result = delete_dataset("id1")
    assert result["dataset_id"] == "id1"


# --- load_dataset_dataframe ---

def test_load_dataset_dataframe_not_found(ctx):
    ensure_storage_dirs()
    with pytest.raises(DatasetNotFoundError):
        load_dataset_dataframe("missing-id")


# --- _resolve_dataset_file_path ---

def test_resolve_absolute_path(ctx):
    result = _resolve_dataset_file_path("/absolute/path/file.csv")
    assert result == Path("/absolute/path/file.csv")


def test_resolve_uploads_prefix(ctx):
    result = _resolve_dataset_file_path("uploads/abc_test.csv")
    assert result.name == "abc_test.csv"


def test_resolve_relative_non_uploads(ctx, tmp_path):
    paths = get_storage_paths()
    relative = "data/some_file.csv"
    result = _resolve_dataset_file_path(relative)
    assert "some_file.csv" in str(result)


def test_resolve_empty():
    result = _resolve_dataset_file_path("")
    assert isinstance(result, Path)
