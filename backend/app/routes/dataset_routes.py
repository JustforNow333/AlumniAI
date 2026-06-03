from flask import Blueprint, jsonify

from app.services.analysis_service import summarize_dataframe
from app.services.spreadsheet_service import get_dataset, get_preview_payload


dataset_bp = Blueprint("datasets", __name__, url_prefix="/api/datasets")


@dataset_bp.get("/<dataset_id>/preview")
def preview_dataset(dataset_id):
    dataset = get_dataset(dataset_id)
    if dataset is None:
        return jsonify({"error": "Dataset not found."}), 404

    return jsonify(get_preview_payload(dataset["dataframe"]))


@dataset_bp.get("/<dataset_id>/summary")
def summarize_dataset(dataset_id):
    dataset = get_dataset(dataset_id)
    if dataset is None:
        return jsonify({"error": "Dataset not found."}), 404

    return jsonify(summarize_dataframe(dataset["dataframe"]))
