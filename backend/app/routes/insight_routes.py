from flask import Blueprint, jsonify, request

from app.services.insight_store import (
    InsightStoreError,
    create_insight,
    delete_insight,
    get_insight,
    list_insights,
    update_insight,
)


insight_bp = Blueprint("insights", __name__, url_prefix="/api/insights")


@insight_bp.get("")
def list_all_insights():
    dataset_id = request.args.get("dataset_id") or None
    try:
        insights = list_insights(dataset_id=dataset_id)
    except InsightStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify({"insights": insights, "count": len(insights)})


@insight_bp.post("")
def create_insight_route():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        insight = create_insight(
            dataset_id=payload.get("dataset_id"),
            question=payload.get("question"),
            answer=payload.get("answer"),
            title=payload.get("title"),
            tags=payload.get("tags"),
            extra_metadata=payload.get("metadata"),
        )
    except InsightStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(insight), 201


@insight_bp.get("/<insight_id>")
def get_insight_route(insight_id):
    try:
        insight = get_insight(insight_id)
    except InsightStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(insight)


@insight_bp.patch("/<insight_id>")
def update_insight_route(insight_id):
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        insight = update_insight(
            insight_id,
            title=payload.get("title") if "title" in payload else None,
            tags=payload.get("tags") if "tags" in payload else None,
        )
    except InsightStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify(insight)


@insight_bp.delete("/<insight_id>")
def delete_insight_route(insight_id):
    try:
        insight = delete_insight(insight_id)
    except InsightStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    return jsonify({"deleted": True, "insight_id": insight.get("insight_id")})
