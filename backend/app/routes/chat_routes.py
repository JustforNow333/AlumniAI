from flask import Blueprint, jsonify, request

from app.services.analysis_service import run_safe_analysis_intent
from app.services.ai_service import build_ai_context, generate_answer
from app.services.spreadsheet_service import get_dataset, to_json_safe


chat_bp = Blueprint("chat", __name__, url_prefix="/api")


@chat_bp.post("/ask")
def ask_dataset():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    dataset_id = payload.get("dataset_id")
    question = payload.get("question")

    if not dataset_id:
        return jsonify({"error": "dataset_id is required."}), 400

    if not question or not str(question).strip():
        return jsonify({"error": "question is required."}), 400

    dataset = get_dataset(dataset_id)
    if dataset is None:
        return jsonify({"error": "Dataset not found."}), 404

    df = dataset["dataframe"]
    context = build_ai_context(df)
    operation, result = run_safe_analysis_intent(df, str(question).strip())
    answer = generate_answer(str(question).strip(), context, operation=operation, result=result)

    return jsonify(
        to_json_safe(
            {
                "dataset_id": dataset_id,
                "question": str(question).strip(),
                "answer": answer,
                "operation": operation,
                "result": result,
            }
        )
    )
