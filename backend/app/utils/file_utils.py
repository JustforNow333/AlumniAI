from pathlib import Path


ALLOWED_EXTENSIONS = {".csv", ".xlsx"}

XLSX_MAGIC = b"PK\x03\x04"


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


def validate_file_content(uploaded_file, filename):
    """Check that an .xlsx upload starts with valid ZIP magic bytes.

    CSV files are plain text, so only .xlsx is validated here.
    Returns True when the content looks plausible for the claimed extension.
    """
    extension = Path(str(filename or "")).suffix.lower()
    if extension != ".xlsx":
        return True

    stream = uploaded_file.stream
    try:
        pos = stream.tell()
        header = stream.read(4)
        stream.seek(pos)
    except (AttributeError, OSError):
        return False

    return header == XLSX_MAGIC
