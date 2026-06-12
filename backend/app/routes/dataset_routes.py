from flask import Blueprint, jsonify, request

from app.services.analysis_service import summarize_dataframe
from app.services.dataset_store import (
    DatasetStoreError,
    delete_dataset,
    list_datasets,
    load_dataset_dataframe,
    rename_dataset,
)
from app.services.spreadsheet_service import get_preview_payload


dataset_bp = Blueprint("datasets", __name__, url_prefix="/api/datasets")


@dataset_bp.get("")
def list_all_datasets():
    try:
        datasets = list_datasets()
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify({"datasets": datasets, "count": len(datasets)})


@dataset_bp.patch("/<dataset_id>")
def rename_dataset_route(dataset_id):
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        metadata = rename_dataset(dataset_id, payload.get("display_name"))
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(metadata)


@dataset_bp.delete("/<dataset_id>")
def delete_dataset_route(dataset_id):
    try:
        metadata = delete_dataset(dataset_id)
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify({"deleted": True, "dataset_id": metadata.get("dataset_id")})


@dataset_bp.get("/<dataset_id>/preview")
def preview_dataset(dataset_id):
    try:
        df, metadata = load_dataset_dataframe(dataset_id)
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    payload = get_preview_payload(df)
    payload.update(
        {
            "dataset_id": metadata["dataset_id"],
            "filename": metadata["original_filename"],
        }
    )
    return jsonify(payload)


@dataset_bp.get("/<dataset_id>/summary")
def summarize_dataset(dataset_id):
    try:
        df, _metadata = load_dataset_dataframe(dataset_id)
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(summarize_dataframe(df))
