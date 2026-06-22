from flask import Blueprint, jsonify, request

from app.services.insight_store import (
    InsightStoreError,
    create_insight,
    delete_insight,
    get_insight,
    list_insights,
    update_insight,
)
from app.utils.response_helpers import get_json_payload, handle_store_errors


insight_bp = Blueprint("insights", __name__, url_prefix="/api/insights")


@insight_bp.get("")
@handle_store_errors(InsightStoreError)
def list_all_insights():
    dataset_id = request.args.get("dataset_id") or None
    insights = list_insights(dataset_id=dataset_id)
    return jsonify({"insights": insights, "count": len(insights)})


@insight_bp.post("")
@handle_store_errors(InsightStoreError)
def create_insight_route():
    payload, error = get_json_payload()
    if error:
        return error

    insight = create_insight(
        dataset_id=payload.get("dataset_id"),
        question=payload.get("question"),
        answer=payload.get("answer") if "answer" in payload else payload.get("answer_text"),
        title=payload.get("title"),
        tags=payload.get("tags"),
        extra_metadata=payload.get("metadata"),
        response_payload=payload.get("response_payload"),
    )
    return jsonify(insight), 201


@insight_bp.get("/<insight_id>")
@handle_store_errors(InsightStoreError)
def get_insight_route(insight_id):
    return jsonify(get_insight(insight_id))


@insight_bp.patch("/<insight_id>")
@handle_store_errors(InsightStoreError)
def update_insight_route(insight_id):
    payload, error = get_json_payload()
    if error:
        return error

    insight = update_insight(
        insight_id,
        title=payload.get("title") if "title" in payload else None,
        tags=payload.get("tags") if "tags" in payload else None,
    )
    return jsonify(insight)


@insight_bp.delete("/<insight_id>")
@handle_store_errors(InsightStoreError)
def delete_insight_route(insight_id):
    insight = delete_insight(insight_id)
    return jsonify({"deleted": True, "insight_id": insight.get("insight_id")})
