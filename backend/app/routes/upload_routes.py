from flask import Blueprint, current_app, jsonify, request

from app.services.spreadsheet_service import (
    SpreadsheetError,
    create_basic_summary,
    read_spreadsheet,
    store_dataset,
)
from app.utils.file_utils import (
    allowed_file,
    is_empty_upload,
    save_uploaded_file,
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

    saved_path = save_uploaded_file(uploaded_file, current_app.config["UPLOAD_FOLDER"])

    try:
        df = read_spreadsheet(saved_path)
    except SpreadsheetError as exc:
        return jsonify({"error": str(exc)}), 400

    dataset_id = store_dataset(uploaded_file.filename, saved_path, df)

    return (
        jsonify(
            {
                "dataset_id": dataset_id,
                "filename": uploaded_file.filename,
                "summary": create_basic_summary(df),
            }
        ),
        201,
    )
