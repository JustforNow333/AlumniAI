from flask import Blueprint, jsonify, request

from app.services.analysis_executor import execute_analysis_plan
from app.services.analysis_intent import infer_analysis_intent, intent_to_analysis_plan
from app.services.analysis_toolkit import build_dataset_context
from app.services.answer_presenter import planner_failure_answer, present_answer
from app.services.column_resolver import resolve_person_columns
from app.services.dataset_store import DatasetStoreError, load_dataset_dataframe
from app.services.industry_matching import debug_classify_person
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


@chat_bp.get("/debug/classify-row")
def debug_classify_row():
    """Development-only helper: explain why a person row is included/excluded
    for an industry. Example:
    GET /api/debug/classify-row?dataset_id=...&name=Neil%20Wusu&industry=tech
    This debug info never appears in the normal user-facing table.
    """
    dataset_id = request.args.get("dataset_id")
    name_query = (request.args.get("name") or "").strip()
    industry = (request.args.get("industry") or "tech").strip()

    if not dataset_id:
        return jsonify({"error": "dataset_id is required."}), 400
    if not name_query:
        return jsonify({"error": "name is required."}), 400

    try:
        df, _metadata = load_dataset_dataframe(dataset_id)
    except DatasetStoreError as exc:
        return jsonify({"error": str(exc)}), exc.status_code

    resolved = resolve_person_columns(df)
    name_columns = [
        resolved.get(field)
        for field in ["first_name", "last_name", "full_name", "nickname"]
        if resolved.get(field)
    ]
    if not name_columns:
        return jsonify({"error": "No name-like columns could be resolved in this dataset."}), 422

    name_lower = name_query.casefold()
    matches = []
    for _index, row in df.iterrows():
        parts = [str(row[column]) for column in name_columns if row[column] is not None]
        combined = " ".join(parts).casefold()
        if all(token in combined for token in name_lower.split()):
            occupation = row[resolved["occupation"]] if resolved.get("occupation") else ""
            employer = row[resolved["employer"]] if resolved.get("employer") else ""
            matches.append(
                debug_classify_person(
                    occupation="" if occupation is None else str(occupation),
                    employer="" if employer is None else str(employer),
                    industry=industry,
                    name=" ".join(parts),
                )
            )
        if len(matches) >= 20:
            break

    return jsonify(
        to_json_safe(
            {
                "dataset_id": dataset_id,
                "name_query": name_query,
                "target_industry": industry,
                "resolved_columns": resolved,
                "match_count": len(matches),
                "matches": matches,
            }
        )
    )
