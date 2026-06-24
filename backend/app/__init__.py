import logging
import os
import secrets
from pathlib import Path

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from app.routes.chat_routes import chat_bp
from app.routes.dataset_routes import dataset_bp
from app.routes.history_routes import history_bp
from app.routes.insight_routes import insight_bp
from app.routes.upload_routes import upload_bp
from app.utils.file_utils import ensure_upload_folder


def create_app():
    app = Flask(__name__)
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_dir = os.path.abspath(os.path.join(backend_dir, ".."))
    frontend_dir = os.path.join(project_dir, "frontend")
    upload_folder = os.path.join(backend_dir, "uploads")
    data_folder = os.path.join(backend_dir, "data")

    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
    app.config["UPLOAD_FOLDER"] = upload_folder
    app.config["DATA_FOLDER"] = data_folder
    app.config["DATASET_REGISTRY_PATH"] = os.path.join(data_folder, "datasets.json")
    app.config["HISTORY_REGISTRY_PATH"] = os.path.join(data_folder, "history.json")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    ensure_upload_folder(upload_folder)
    Path(data_folder).mkdir(parents=True, exist_ok=True)

    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": [
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://localhost:8000",
                    "http://127.0.0.1:8000",
                ]
            }
        },
    )

    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

    app.register_blueprint(upload_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(insight_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(chat_bp)

    @app.get("/api/health")
    def health_check():
        return jsonify({"status": "ok"})

    if os.path.isdir(frontend_dir):
        @app.get("/")
        def frontend_index():
            return send_from_directory(frontend_dir, "index.html")

        @app.get("/<path:filename>")
        def frontend_asset(filename):
            return send_from_directory(frontend_dir, filename)

    @app.errorhandler(413)
    def file_too_large(_error):
        return jsonify({"error": "File is too large. Maximum upload size is 16 MB."}), 413

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Route not found."}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logging.getLogger(__name__).exception("Unhandled server error: %s", error)
        return jsonify({"error": "An internal server error occurred."}), 500

    return app
