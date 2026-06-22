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
from app.utils.response_helpers import get_json_payload, handle_store_errors


history_bp = Blueprint("history", __name__, url_prefix="/api/history")


@history_bp.get("")
@handle_store_errors(HistoryStoreError)
def list_all_history():
    items = list_history()
    return jsonify({"history": items, "count": len(items)})


@history_bp.post("")
@handle_store_errors(HistoryStoreError)
def create_history_route():
    payload, error = get_json_payload()
    if error:
        return error

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
    return jsonify(item), 201


@history_bp.get("/<history_id>")
@handle_store_errors(HistoryStoreError)
def get_history_route(history_id):
    return jsonify(get_history_item(history_id))


@history_bp.delete("/<history_id>")
@handle_store_errors(HistoryStoreError)
def delete_history_route(history_id):
    item = delete_history_item(history_id)
    return jsonify({"deleted": True, "history_id": item.get("history_id")})


@history_bp.delete("")
@handle_store_errors(HistoryStoreError)
def clear_history_route():
    return jsonify(clear_history())


@history_bp.post("/<history_id>/save-insight")
@handle_store_errors(HistoryStoreError)
def save_history_as_insight_route(history_id):
    try:
        insight = save_history_item_as_insight(history_id)
    except HistoryStoreError:
        raise
    except Exception as exc:
        status_code = getattr(exc, "status_code", 500)
        return jsonify({"error": str(exc)}), status_code

    return jsonify(insight), 201
