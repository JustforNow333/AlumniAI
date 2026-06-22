from flask import Blueprint, jsonify, request

from app.services.dataset_store import DatasetStoreError, load_dataset_dataframe, register_uploaded_dataset
from app.services.spreadsheet_service import (
    create_basic_summary,
)
from app.utils.file_utils import (
    allowed_file,
    is_empty_upload,
    validate_file_content,
)


upload_bp = Blueprint("upload", __name__, url_prefix="/api")


@upload_bp.post("/upload")
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Missing file field. Upload a file under the key 'file'."}), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a .csv or .xlsx file."}), 400

    if is_empty_upload(uploaded_file):
        return jsonify({"error": "Uploaded file is empty."}), 400

    if not validate_file_content(uploaded_file, uploaded_file.filename):
        return jsonify({"error": "File content does not match expected format."}), 400

    try:
        metadata = register_uploaded_dataset(uploaded_file)
        df, _ = load_dataset_dataframe(metadata["dataset_id"])
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return (
        jsonify(
            {
                "dataset_id": metadata["dataset_id"],
                "filename": metadata["original_filename"],
                "metadata": metadata,
                "summary": create_basic_summary(df),
            }
        ),
        201,
    )
