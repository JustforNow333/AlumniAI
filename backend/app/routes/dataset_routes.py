from flask import Blueprint, jsonify

from app.services.analysis_service import summarize_dataframe
from app.services.dataset_store import DatasetStoreError, load_dataset_dataframe
from app.services.spreadsheet_service import get_preview_payload


dataset_bp = Blueprint("datasets", __name__, url_prefix="/api/datasets")


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
