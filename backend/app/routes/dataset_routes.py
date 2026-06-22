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
from app.utils.response_helpers import get_json_payload, handle_store_errors


dataset_bp = Blueprint("datasets", __name__, url_prefix="/api/datasets")


@dataset_bp.get("")
@handle_store_errors(DatasetStoreError)
def list_all_datasets():
    datasets = list_datasets()
    return jsonify({"datasets": datasets, "count": len(datasets)})


@dataset_bp.patch("/<dataset_id>")
@handle_store_errors(DatasetStoreError)
def rename_dataset_route(dataset_id):
    payload, error = get_json_payload()
    if error:
        return error

    return jsonify(rename_dataset(dataset_id, payload.get("display_name")))


@dataset_bp.delete("/<dataset_id>")
@handle_store_errors(DatasetStoreError)
def delete_dataset_route(dataset_id):
    metadata = delete_dataset(dataset_id)
    return jsonify({"deleted": True, "dataset_id": metadata.get("dataset_id")})


@dataset_bp.get("/<dataset_id>/preview")
@handle_store_errors(DatasetStoreError)
def preview_dataset(dataset_id):
    df, metadata = load_dataset_dataframe(dataset_id)
    payload = get_preview_payload(df)
    payload.update(
        {
            "dataset_id": metadata["dataset_id"],
            "filename": metadata["original_filename"],
        }
    )
    return jsonify(payload)


@dataset_bp.get("/<dataset_id>/summary")
@handle_store_errors(DatasetStoreError)
def summarize_dataset(dataset_id):
    df, _metadata = load_dataset_dataframe(dataset_id)
    return jsonify(summarize_dataframe(df))
