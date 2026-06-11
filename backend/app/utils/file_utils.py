from pathlib import Path


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
