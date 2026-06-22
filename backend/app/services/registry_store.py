"""Shared JSON-file registry persistence.

All three stores (dataset, insight, history) use the same pattern: a JSON
object keyed by ID, loaded/saved atomically via a tmp file. This module
extracts the common load/save/ensure logic so each store only declares its
own registry path, error class, and domain-specific validation.
"""

import json
import os
from pathlib import Path
from uuid import uuid4


class RegistryError(Exception):
    status_code = 500


def load_registry(registry_path, error_cls=RegistryError, label=None):
    registry_path = Path(registry_path)
    name = label or registry_path.stem.replace("_", " ").title()
    if not registry_path.exists():
        return {}

    try:
        with registry_path.open("r", encoding="utf-8") as registry_file:
            registry = json.load(registry_file)
    except json.JSONDecodeError as exc:
        raise error_cls(f"{name} registry is invalid JSON.") from exc
    except OSError as exc:
        raise error_cls(f"Could not read {name.lower()} registry: {exc}") from exc

    if not isinstance(registry, dict):
        raise error_cls(f"{name} registry must contain a JSON object.")

    return registry


def save_registry(registry, registry_path, error_cls=RegistryError, label=None):
    name = label or Path(registry_path).stem.replace("_", " ").title()
    if not isinstance(registry, dict):
        raise error_cls(f"{name} registry must be a dictionary.")

    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = registry_path.with_name(f"{registry_path.name}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as registry_file:
            json.dump(registry, registry_file, indent=2)
            registry_file.write("\n")
        os.replace(temporary_path, registry_path)
    except OSError as exc:
        raise error_cls(f"Could not save {name.lower()} registry: {exc}") from exc


def generate_unique_id(registry):
    new_id = str(uuid4())
    while new_id in registry:
        new_id = str(uuid4())
    return new_id


def list_registry_newest_first(registry, timestamp_key, transform_fn):
    entries = [
        (entry_id, entry)
        for entry_id, entry in registry.items()
        if isinstance(entry, dict)
    ]
    indexed = [
        (str(entry.get(timestamp_key) or ""), position, entry_id, entry)
        for position, (entry_id, entry) in enumerate(entries)
    ]
    indexed.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [
        transform_fn(entry, entry_id=entry_id)
        for _ts, _pos, entry_id, entry in indexed
    ]
