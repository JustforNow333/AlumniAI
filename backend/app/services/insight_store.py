"""Saved insights persistence: manually saved AI answers tied to a dataset.

A saved insight is a snapshot of the answer at the time the user saved it — the
answer is never recomputed, and neither the DataFrame nor the uploaded file
contents are stored. Metadata-only JSON registry, same pattern as
dataset_store: a dict keyed by insight_id, written atomically via a tmp file.

This is distinct from (future) history: insights are saved manually by the
user; nothing in this module logs questions or answers automatically.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import current_app, has_app_context

from app.services.dataset_store import get_dataset_metadata, get_storage_paths


MAX_TITLE_LENGTH = 120
MAX_TAG_LENGTH = 40
MAX_TAGS = 12
GENERATED_TITLE_LENGTH = 80


class InsightStoreError(Exception):
    status_code = 500


class InsightValidationError(InsightStoreError):
    status_code = 400


class InsightNotFoundError(InsightStoreError):
    status_code = 404


class InsightRegistryError(InsightStoreError):
    status_code = 500


def get_insights_registry_path():
    if has_app_context():
        configured = current_app.config.get("INSIGHTS_REGISTRY_PATH")
        if configured:
            return Path(configured)
    # Default lives next to datasets.json so test fixtures that point
    # DATA_FOLDER at temp storage isolate insights automatically.
    return get_storage_paths()["data_folder"] / "saved_insights.json"


def load_insight_registry():
    registry_path = get_insights_registry_path()
    if not registry_path.exists():
        return {}

    try:
        with registry_path.open("r", encoding="utf-8") as registry_file:
            registry = json.load(registry_file)
    except json.JSONDecodeError as exc:
        raise InsightRegistryError("Saved insights registry is invalid JSON.") from exc
    except OSError as exc:
        raise InsightRegistryError(f"Could not read saved insights registry: {exc}") from exc

    if not isinstance(registry, dict):
        raise InsightRegistryError("Saved insights registry must contain a JSON object.")

    return registry


def save_insight_registry(registry):
    if not isinstance(registry, dict):
        raise InsightRegistryError("Saved insights registry must be a dictionary.")

    registry_path = get_insights_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = registry_path.with_name(f"{registry_path.name}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as registry_file:
            json.dump(registry, registry_file, indent=2)
            registry_file.write("\n")
        os.replace(temporary_path, registry_path)
    except OSError as exc:
        raise InsightRegistryError(f"Could not save saved insights registry: {exc}") from exc


def generate_title_from_question(question):
    """Short fallback title derived from the question text."""
    text = " ".join(str(question or "").split()).strip().rstrip("?.!").strip()
    if not text:
        return "Saved insight"
    if len(text) <= GENERATED_TITLE_LENGTH:
        return text
    clipped = text[:GENERATED_TITLE_LENGTH].rsplit(" ", 1)[0].rstrip()
    return f"{clipped or text[:GENERATED_TITLE_LENGTH]}…"


def insight_public_metadata(entry):
    """Shape one registry entry for the API: tolerate missing fields and report
    whether the referenced dataset still exists, without ever crashing."""
    entry = entry if isinstance(entry, dict) else {}
    dataset_id = entry.get("dataset_id")
    dataset_status = "deleted"
    try:
        if dataset_id and get_dataset_metadata(dataset_id) is not None:
            dataset_status = "ready"
    except Exception:
        dataset_status = "deleted"
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    return {
        "insight_id": entry.get("insight_id"),
        "dataset_id": dataset_id,
        "dataset_name_snapshot": entry.get("dataset_name_snapshot") or "Unknown dataset",
        "dataset_status": dataset_status,
        "title": entry.get("title") or generate_title_from_question(entry.get("question")),
        "question": entry.get("question") or "",
        "answer": entry.get("answer") or "",
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "tags": [str(tag) for tag in entry.get("tags") or [] if str(tag).strip()],
        "metadata": metadata,
    }


def create_insight(dataset_id, question, answer, title=None, tags=None, extra_metadata=None):
    """Validate and persist a manually saved insight. Returns the public shape."""
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        raise InsightValidationError("dataset_id is required to save an insight.")

    dataset_metadata = get_dataset_metadata(dataset_id)
    if dataset_metadata is None:
        raise InsightNotFoundError("Dataset not found. Insights must reference a saved dataset.")

    question_text = str(question or "").strip()
    if not question_text:
        raise InsightValidationError("question must not be empty.")

    answer_text = str(answer or "").strip()
    if not answer_text:
        raise InsightValidationError("answer must not be empty.")

    title_text = str(title or "").strip()[:MAX_TITLE_LENGTH].strip()
    if not title_text:
        title_text = generate_title_from_question(question_text)

    metadata = {}
    if isinstance(extra_metadata, dict):
        model = str(extra_metadata.get("model") or "").strip()
        if model:
            metadata["model"] = model[:80]
    if dataset_metadata.get("row_count") is not None:
        metadata["row_count"] = dataset_metadata.get("row_count")
    if dataset_metadata.get("column_count") is not None:
        metadata["column_count"] = dataset_metadata.get("column_count")

    registry = load_insight_registry()
    insight_id = str(uuid4())
    while insight_id in registry:
        insight_id = str(uuid4())

    now = datetime.now().isoformat(timespec="seconds")
    entry = {
        "insight_id": insight_id,
        "dataset_id": dataset_id,
        "dataset_name_snapshot": dataset_metadata.get("display_name")
        or dataset_metadata.get("original_filename")
        or "Untitled dataset",
        "title": title_text,
        "question": question_text,
        "answer": answer_text,
        "created_at": now,
        "updated_at": now,
        "tags": _clean_tags(tags),
        "metadata": metadata,
    }
    registry[insight_id] = entry
    save_insight_registry(registry)
    return insight_public_metadata(entry)


def list_insights(dataset_id=None):
    """All saved insights, newest first. created_at has second resolution, so
    ties break by registry insertion order (later insertion = newer)."""
    registry = load_insight_registry()
    entries = [entry for entry in registry.values() if isinstance(entry, dict)]
    if dataset_id:
        entries = [entry for entry in entries if entry.get("dataset_id") == str(dataset_id)]
    indexed = [
        (str(entry.get("created_at") or ""), position, entry)
        for position, entry in enumerate(entries)
    ]
    indexed.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [insight_public_metadata(entry) for _created_at, _position, entry in indexed]


def get_insight(insight_id):
    registry = load_insight_registry()
    entry = registry.get(str(insight_id))
    if entry is None:
        raise InsightNotFoundError("Saved insight not found.")
    return insight_public_metadata(entry)


def update_insight(insight_id, title=None, tags=None):
    """Edit the title and/or tags. dataset_id, question, and answer are
    immutable snapshots and cannot be changed."""
    registry = load_insight_registry()
    entry = registry.get(str(insight_id))
    if entry is None:
        raise InsightNotFoundError("Saved insight not found.")

    changed = False
    if title is not None:
        title_text = str(title or "").strip()[:MAX_TITLE_LENGTH].strip()
        if not title_text:
            raise InsightValidationError("title must not be empty.")
        entry["title"] = title_text
        changed = True
    if tags is not None:
        if not isinstance(tags, list):
            raise InsightValidationError("tags must be a list of strings.")
        entry["tags"] = _clean_tags(tags)
        changed = True

    if not changed:
        raise InsightValidationError("Provide a title or tags to update.")

    entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_insight_registry(registry)
    return insight_public_metadata(entry)


def delete_insight(insight_id):
    registry = load_insight_registry()
    entry = registry.pop(str(insight_id), None)
    if entry is None:
        raise InsightNotFoundError("Saved insight not found.")
    save_insight_registry(registry)
    return insight_public_metadata(entry)


def _clean_tags(tags):
    if not isinstance(tags, list):
        return []
    cleaned = []
    for tag in tags[:MAX_TAGS]:
        text = str(tag or "").strip()[:MAX_TAG_LENGTH].strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned
