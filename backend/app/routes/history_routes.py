from flask import Blueprint, jsonify, request

from app.services.history_store import (
    HistoryStoreError,
    clear_history,
    create_history_item,
    delete_history_item,
    get_history_item,
    list_history,
    save_history_item_as_insight,
)


history_bp = Blueprint("history", __name__, url_prefix="/api/history")


@history_bp.get("")
def list_all_history():
    try:
        items = list_history()
    except HistoryStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify({"history": items, "count": len(items)})


@history_bp.post("")
def create_history_route():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        item = create_history_item(
            dataset_id=payload.get("dataset_id"),
            dataset_filename=payload.get("dataset_filename"),
            question=payload.get("question"),
            answer_text=payload.get("answer_text") if "answer_text" in payload else payload.get("answer"),
            title=payload.get("title"),
            status=payload.get("status") or "success",
            metadata=payload.get("metadata"),
            response_payload=payload.get("response_payload"),
        )
    except HistoryStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(item), 201


@history_bp.get("/<history_id>")
def get_history_route(history_id):
    try:
        item = get_history_item(history_id)
    except HistoryStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(item)


@history_bp.delete("/<history_id>")
def delete_history_route(history_id):
    try:
        item = delete_history_item(history_id)
    except HistoryStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify({"deleted": True, "history_id": item.get("history_id")})


@history_bp.delete("")
def clear_history_route():
    try:
        result = clear_history()
    except HistoryStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(result)


@history_bp.post("/<history_id>/save-insight")
def save_history_as_insight_route(history_id):
    try:
        insight = save_history_item_as_insight(history_id)
    except HistoryStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as exc:
        status_code = getattr(exc, "status_code", 500)
        return jsonify({"error": str(exc)}), status_code

    return jsonify(insight), 201
