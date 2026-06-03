from pathlib import Path
from uuid import uuid4

from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


def ensure_upload_folder(upload_folder):
    Path(upload_folder).mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_empty_upload(uploaded_file):
    stream = uploaded_file.stream

    try:
        current_position = stream.tell()
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(current_position)
        return size == 0
    except (AttributeError, OSError):
        return uploaded_file.content_length == 0


def save_uploaded_file(uploaded_file, upload_folder):
    original_name = secure_filename(uploaded_file.filename)
    extension = Path(original_name).suffix.lower()
    stem = Path(original_name).stem or "upload"
    unique_filename = f"{uuid4()}_{stem}{extension}"
    saved_path = Path(upload_folder) / unique_filename

    uploaded_file.save(saved_path)
    return str(saved_path)
