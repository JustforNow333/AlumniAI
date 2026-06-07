from flask import Blueprint, jsonify, request

from app.services.analysis_executor import execute_analysis_plan
from app.services.analysis_intent import infer_analysis_intent, intent_to_analysis_plan
from app.services.analysis_toolkit import build_dataset_context
from app.services.answer_presenter import planner_failure_answer, present_answer
from app.services.dataset_store import DatasetStoreError, load_dataset_dataframe
from app.services.spreadsheet_service import to_json_safe


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

    try:
        df, metadata = load_dataset_dataframe(dataset_id)
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    question_text = str(question).strip()
    context = build_dataset_context(df, metadata=metadata)
    analysis_intent, intent_valid, intent_error = infer_analysis_intent(question_text, context)
    if intent_valid:
        plan = intent_to_analysis_plan(analysis_intent, context)
        plan_valid = True
        plan_error = ""
    else:
        plan = {"operations": [], "presentation_hint": "markdown", "assumptions": [], "cannot_answer_reason": intent_error}
        plan_valid = False
        plan_error = intent_error

    if plan_valid and plan.get("operations"):
        operation_results = execute_analysis_plan(df, plan)
        answer = present_answer(question_text, plan, operation_results, context)
    elif plan_valid:
        operation_results = []
        answer = planner_failure_answer(plan.get("cannot_answer_reason") or "No approved analysis operation matched the question.")
    else:
        operation_results = []
        answer = planner_failure_answer(plan_error)

    operation = plan["operations"][0] if plan.get("operations") else None
    result = operation_results[0] if operation_results else None

    return jsonify(
        to_json_safe(
            {
                "dataset_id": dataset_id,
                "question": question_text,
                "answer": answer,
                "answer_text": answer.get("summary", "") if isinstance(answer, dict) else str(answer),
                "operation": operation,
                "result": result,
                "analysis_intent": analysis_intent,
                "analysis_plan": plan,
                "operation_results": operation_results,
            }
        )
    )
