"""Automatic analysis history persistence.

History is an append-only log of successful dataset questions. Unlike saved
insights, entries are created automatically by the ask flow and can outlive the
source dataset because each item stores the full response payload snapshot.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import current_app, has_app_context

from app.services.dataset_store import get_dataset_metadata, get_storage_paths
from app.services.insight_store import create_insight
from app.services.spreadsheet_service import to_json_safe


MAX_TITLE_LENGTH = 120
GENERATED_TITLE_LENGTH = 80
MAX_ANSWER_TEXT_LENGTH = 20000
MAX_RESPONSE_PAYLOAD_BYTES = 2_000_000


class HistoryStoreError(Exception):
    status_code = 500


class HistoryValidationError(HistoryStoreError):
    status_code = 400


class HistoryNotFoundError(HistoryStoreError):
    status_code = 404


class HistoryRegistryError(HistoryStoreError):
    status_code = 500


def get_history_registry_path():
    if has_app_context():
        configured = current_app.config.get("HISTORY_REGISTRY_PATH")
        if configured:
            return Path(configured)
    return get_storage_paths()["data_folder"] / "history.json"


def load_history_registry():
    registry_path = get_history_registry_path()
    if not registry_path.exists():
        return {}

    try:
        with registry_path.open("r", encoding="utf-8") as registry_file:
            registry = json.load(registry_file)
    except json.JSONDecodeError as exc:
        raise HistoryRegistryError("History registry is invalid JSON.") from exc
    except OSError as exc:
        raise HistoryRegistryError(f"Could not read history registry: {exc}") from exc

    if not isinstance(registry, dict):
        raise HistoryRegistryError("History registry must contain a JSON object.")

    return registry


def save_history_registry(registry):
    if not isinstance(registry, dict):
        raise HistoryRegistryError("History registry must be a dictionary.")

    registry_path = get_history_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = registry_path.with_name(f"{registry_path.name}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as registry_file:
            json.dump(registry, registry_file, indent=2)
            registry_file.write("\n")
        os.replace(temporary_path, registry_path)
    except OSError as exc:
        raise HistoryRegistryError(f"Could not save history registry: {exc}") from exc


def generate_title_from_question(question):
    text = " ".join(str(question or "").split()).strip().rstrip("?.!").strip()
    if not text:
        return "Analysis"
    if len(text) <= GENERATED_TITLE_LENGTH:
        return text
    clipped = text[:GENERATED_TITLE_LENGTH].rsplit(" ", 1)[0].rstrip()
    return f"{clipped or text[:GENERATED_TITLE_LENGTH]}..."


def history_public_metadata(entry, entry_id=None):
    entry = entry if isinstance(entry, dict) else {}
    history_id = entry.get("history_id") or entry.get("id") or entry_id
    dataset_id = entry.get("dataset_id") or ""
    dataset_status = "deleted"
    try:
        if dataset_id and get_dataset_metadata(dataset_id) is not None:
            dataset_status = "ready"
    except Exception:
        dataset_status = "deleted"

    response_payload = entry.get("response_payload")
    if not isinstance(response_payload, dict):
        response_payload = None

    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    answer_text = str(entry.get("answer_text") or entry.get("answer") or "")
    title = str(entry.get("title") or "").strip() or generate_title_from_question(entry.get("question"))

    return {
        "id": history_id,
        "history_id": history_id,
        "dataset_id": dataset_id,
        "dataset_filename": entry.get("dataset_filename")
        or entry.get("dataset_name_snapshot")
        or "Unknown dataset",
        "dataset_status": dataset_status,
        "title": title,
        "question": entry.get("question") or "",
        "answer_text": answer_text,
        "answer": answer_text,
        "response_payload": response_payload,
        "status": entry.get("status") or "success",
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "metadata": metadata,
    }


def create_history_item(
    dataset_id,
    question,
    answer_text,
    response_payload=None,
    title=None,
    status="success",
    metadata=None,
    dataset_metadata=None,
    dataset_filename=None,
):
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        raise HistoryValidationError("dataset_id is required.")

    question_text = str(question or "").strip()
    if not question_text:
        raise HistoryValidationError("question must not be empty.")

    answer_snapshot = str(answer_text or "").strip()
    if not answer_snapshot:
        raise HistoryValidationError("answer_text must not be empty.")
    answer_snapshot = answer_snapshot[:MAX_ANSWER_TEXT_LENGTH]

    cleaned_response_payload = _clean_response_payload(response_payload)

    if not isinstance(dataset_metadata, dict):
        try:
            dataset_metadata = get_dataset_metadata(dataset_id) or {}
        except Exception:
            dataset_metadata = {}

    filename = str(
        dataset_filename
        or dataset_metadata.get("display_name")
        or dataset_metadata.get("original_filename")
        or "Unknown dataset"
    ).strip() or "Unknown dataset"

    title_text = str(title or "").strip()[:MAX_TITLE_LENGTH].strip()
    if not title_text:
        title_text = generate_title_from_question(question_text)

    extra_metadata = {}
    if isinstance(metadata, dict):
        extra_metadata.update(to_json_safe(metadata))
    if dataset_metadata.get("row_count") is not None and "row_count" not in extra_metadata:
        extra_metadata["row_count"] = dataset_metadata.get("row_count")
    if dataset_metadata.get("column_count") is not None and "column_count" not in extra_metadata:
        extra_metadata["column_count"] = dataset_metadata.get("column_count")

    registry = load_history_registry()
    history_id = str(uuid4())
    while history_id in registry:
        history_id = str(uuid4())

    now = datetime.now().isoformat(timespec="seconds")
    entry = {
        "history_id": history_id,
        "dataset_id": dataset_id,
        "dataset_filename": filename,
        "title": title_text,
        "question": question_text,
        "answer_text": answer_snapshot,
        "status": str(status or "success").strip() or "success",
        "created_at": now,
        "updated_at": now,
        "metadata": extra_metadata,
    }
    if cleaned_response_payload is not None:
        entry["response_payload"] = cleaned_response_payload

    registry[history_id] = entry
    save_history_registry(registry)
    return history_public_metadata(entry)


def list_history():
    registry = load_history_registry()
    entries = [
        (entry_id, entry)
        for entry_id, entry in registry.items()
        if isinstance(entry, dict)
    ]
    indexed = [
        (str(entry.get("created_at") or ""), position, entry_id, entry)
        for position, (entry_id, entry) in enumerate(entries)
    ]
    indexed.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [
        history_public_metadata(entry, entry_id=entry_id)
        for _created_at, _position, entry_id, entry in indexed
    ]


def get_history_item(history_id):
    registry = load_history_registry()
    entry = registry.get(str(history_id))
    if entry is None:
        raise HistoryNotFoundError("History item not found.")
    return history_public_metadata(entry, entry_id=str(history_id))


def delete_history_item(history_id):
    registry = load_history_registry()
    entry = registry.pop(str(history_id), None)
    if entry is None:
        raise HistoryNotFoundError("History item not found.")
    save_history_registry(registry)
    return history_public_metadata(entry, entry_id=str(history_id))


def clear_history():
    save_history_registry({})
    return {"deleted": True, "count": 0}


def save_history_item_as_insight(history_id):
    item = get_history_item(history_id)
    return create_insight(
        dataset_id=item.get("dataset_id"),
        title=item.get("title"),
        question=item.get("question"),
        answer=item.get("answer_text"),
        response_payload=item.get("response_payload"),
        extra_metadata=item.get("metadata"),
    )


def _clean_response_payload(response_payload):
    if response_payload is None:
        return None
    if not isinstance(response_payload, dict) or isinstance(response_payload, list):
        raise HistoryValidationError("response_payload must be a JSON object.")

    safe_payload = to_json_safe(response_payload)
    try:
        encoded = json.dumps(safe_payload, ensure_ascii=False)
    except TypeError as exc:
        raise HistoryValidationError("response_payload must be JSON serializable.") from exc

    if len(encoded.encode("utf-8")) > MAX_RESPONSE_PAYLOAD_BYTES:
        raise HistoryValidationError("response_payload is too large.")

    return json.loads(encoded)
