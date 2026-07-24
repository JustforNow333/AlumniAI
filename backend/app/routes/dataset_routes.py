from flask import Blueprint, jsonify, request

from app.services.analysis_service import summarize_dataframe
from app.services.dataset_store import (
    DatasetStoreError,
    delete_dataset,
    list_datasets,
    load_dataset_dataframe,
    rename_dataset,
    save_dataset_schema_profile,
)
from app.services.canonical_schema import canonical_field_definitions
from app.services.schema_inference import (
    build_source_column_metadata,
    infer_schema_profile,
)
from app.services.schema_profile import (
    SchemaProfileValidationError,
    normalize_profile_for_storage,
    validate_schema_update,
)
from app.services.spreadsheet_service import get_preview_payload


dataset_bp = Blueprint("datasets", __name__, url_prefix="/api/datasets")
MAX_SCHEMA_REQUEST_BYTES = 256 * 1024


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


@dataset_bp.get("/<dataset_id>/schema")
def get_dataset_schema(dataset_id):
    try:
        df, metadata = load_dataset_dataframe(dataset_id)
        profile = normalize_profile_for_storage(
            metadata.get("schema_profile"), df.columns
        )
        if profile is None:
            profile = infer_schema_profile(df)
            profile = save_dataset_schema_profile(dataset_id, profile, df.columns)
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(_schema_response(dataset_id, df, profile))


@dataset_bp.put("/<dataset_id>/schema")
def put_dataset_schema(dataset_id):
    if request.content_length and request.content_length > MAX_SCHEMA_REQUEST_BYTES:
        return jsonify({"error": "Schema payload is too large."}), 413
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        df, metadata = load_dataset_dataframe(dataset_id)
        existing = normalize_profile_for_storage(
            metadata.get("schema_profile"), df.columns
        )
        profile = validate_schema_update(payload, df.columns, existing)
        profile = save_dataset_schema_profile(dataset_id, profile, df.columns)
    except SchemaProfileValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(_schema_response(dataset_id, df, profile))


@dataset_bp.post("/<dataset_id>/schema/infer")
def infer_dataset_schema(dataset_id):
    if request.content_length and request.content_length > MAX_SCHEMA_REQUEST_BYTES:
        return jsonify({"error": "Schema payload is too large."}), 413
    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    reset_confirmed = payload.get("reset_confirmed", False)
    use_model = payload.get("use_model", True)
    if not isinstance(reset_confirmed, bool) or not isinstance(use_model, bool):
        return jsonify({"error": "reset_confirmed and use_model must be booleans."}), 400

    try:
        df, metadata = load_dataset_dataframe(dataset_id)
        existing = normalize_profile_for_storage(
            metadata.get("schema_profile"), df.columns
        )
        if use_model:
            from app.services import ai_service

            ai_client = ai_service.client
        else:
            ai_client = None
        profile = infer_schema_profile(
            df,
            existing_profile=existing,
            reset_confirmed=reset_confirmed,
            use_model=use_model,
            ai_client=ai_client,
        )
        profile = save_dataset_schema_profile(dataset_id, profile, df.columns)
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(_schema_response(dataset_id, df, profile))


def _schema_response(dataset_id, df, profile):
    payload = {
        "dataset_id": str(dataset_id),
        **profile,
        "canonical_fields": canonical_field_definitions(),
        "source_columns": build_source_column_metadata(df),
    }
    # Keep a nested view for clients that prefer a profile object while the
    # top-level shape remains convenient for the no-build frontend.
    payload["profile"] = dict(profile)
    return payload
