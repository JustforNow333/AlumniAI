from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app
from app.services import ai_service
from app.services import people_classifier
from app.services.industry_matching import budgeted_model_classifier
from evals.report import build_report, write_json_report, write_markdown_report
from evals.scoring import (
    apply_execution_expectations,
    extract_normalized_response,
    load_cases,
    load_gold_dataset,
    make_app_view_dataset,
    score_case,
)
from evals.tracing import ModelCallTrace, TracedAIClient


EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVALS_DIR / "datasets" / "synthetic_alumni_500.csv"
DEFAULT_APP_VIEW = EVALS_DIR / "generated" / "synthetic_alumni_500_app_view.csv"
DEFAULT_CASES = EVALS_DIR / "cases.jsonl"
DEFAULT_JSON_REPORT = EVALS_DIR / "results" / "latest.json"
DEFAULT_MARKDOWN_REPORT = EVALS_DIR / "results" / "latest.md"
MODES = {"offline", "hybrid", "classifier-live", "smoke-live"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    summary = report["summary"]
    print(
        "Ran {total} eval cases: {passed} passed, {failed} failed ({rate:.1f}% pass rate).".format(
            total=summary["total_cases"],
            passed=summary["passed"],
            failed=summary["failed"],
            rate=summary["pass_rate"] * 100,
        )
    )
    print(f"JSON report: {args.output}")
    print(f"Markdown report: {args.markdown}")
    if args.fail_on_failures and summary["failed"]:
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI Spreadsheet Analyst eval cases.")
    parser.add_argument("--case-id", action="append", help="Run one case id. Can be repeated.")
    parser.add_argument("--category", help="Run cases in one category.")
    parser.add_argument("--limit", type=int, help="Run at most N cases after filtering.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Gold dataset CSV path.")
    parser.add_argument("--app-view", default=str(DEFAULT_APP_VIEW), help="Generated app-facing CSV path.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL cases path.")
    parser.add_argument("--output", default=str(DEFAULT_JSON_REPORT), help="JSON report output path.")
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN_REPORT), help="Markdown report output path.")
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        default="offline",
        help=(
            "Eval execution mode. offline disables AI; hybrid runs product API cases with "
            "per-case AI policy; classifier-live runs direct classifier cases; smoke-live "
            "runs a small live smoke subset."
        ),
    )
    parser.add_argument(
        "--use-live-ai",
        action="store_true",
        help="Backward-compatible alias for --mode hybrid.",
    )
    parser.add_argument(
        "--fail-on-failures",
        action="store_true",
        help="Exit with status 1 when any case fails.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    app_view_path = Path(args.app_view)
    cases_path = Path(args.cases)
    output_path = Path(args.output)
    markdown_path = Path(args.markdown)

    gold_df = load_gold_dataset(dataset_path)
    app_view_df = make_app_view_dataset(gold_df)
    app_view_path.parent.mkdir(parents=True, exist_ok=True)
    app_view_df.to_csv(app_view_path, index=False)

    if args.use_live_ai and args.mode == "offline":
        args.mode = "hybrid"

    cases = _filter_cases(load_cases(cases_path), args)
    previous_client = ai_service.client

    try:
        case_results = _run_cases_with_test_client(
            cases,
            gold_df,
            app_view_path,
            mode=args.mode,
            live_client=previous_client,
        )
    finally:
        ai_service.client = previous_client

    report = build_report(
        case_results,
        metadata={
            "dataset": str(dataset_path),
            "app_view_dataset": str(app_view_path),
            "cases": str(cases_path),
            "mode": args.mode,
            "live_ai_enabled": any(result.get("ai_enabled") for result in case_results),
            "case_id": args.case_id or [],
            "category": args.category,
            "limit": args.limit,
        },
    )
    write_json_report(report, output_path)
    write_markdown_report(report, markdown_path)
    return report


def _filter_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filtered = cases
    if args.case_id:
        wanted = set(args.case_id)
        filtered = [case for case in filtered if case.get("id") in wanted]
    if args.category:
        filtered = [case for case in filtered if case.get("category") == args.category]
    filtered = [case for case in filtered if _case_runs_in_mode(case, args.mode)]
    if args.limit is not None:
        filtered = filtered[: max(args.limit, 0)]
    return filtered


def _case_runs_in_mode(case: dict[str, Any], mode: str) -> bool:
    modes = case.get("modes")
    if isinstance(modes, list) and modes:
        return mode in modes
    if case.get("eval_kind") == "direct_classifier":
        return mode in {"classifier-live", "smoke-live"}
    return mode in {"offline", "hybrid"}


def _run_cases_with_test_client(
    cases: list[dict[str, Any]],
    gold_df,
    app_view_path: Path,
    *,
    mode: str,
    live_client: Any,
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="alumni-evals-") as temp_dir:
        temp_path = Path(temp_dir)
        app = create_app()
        app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=str(temp_path / "uploads"),
            DATA_FOLDER=str(temp_path / "data"),
            DATASET_REGISTRY_PATH=str(temp_path / "data" / "datasets.json"),
            HISTORY_REGISTRY_PATH=str(temp_path / "data" / "history.json"),
            INSIGHTS_REGISTRY_PATH=str(temp_path / "data" / "saved_insights.json"),
        )
        client = app.test_client()
        dataset_id = _upload_app_view(client, app_view_path)

        results = []
        for case in cases:
            trace = ModelCallTrace()
            ai_enabled = _case_ai_enabled(case, mode, live_client)
            ai_service.client = TracedAIClient(live_client, trace) if ai_enabled else None
            started = time.perf_counter()
            if case.get("eval_kind") == "direct_classifier":
                scored = _run_direct_classifier_case(case, trace)
                response_json = {"_status_code": 200}
            else:
                response_json = _ask(client, dataset_id, case["question"])
                normalized = extract_normalized_response(response_json)
                scored = score_case(case, normalized, gold_df)
                scored["answer_source"] = _answer_source(response_json, trace)

            duration_ms = (time.perf_counter() - started) * 1000
            scored["duration_ms"] = round(duration_ms, 2)
            scored["status_code"] = response_json.get("_status_code")
            trace_fields = _trace_case_fields(trace, app.testing, ai_enabled)
            scored.update(trace_fields)
            scored = apply_execution_expectations(case, scored, trace_fields)
            scored["mode"] = mode
            results.append(scored)
        return results


def _case_ai_enabled(case: dict[str, Any], mode: str, live_client: Any) -> bool:
    if live_client is None:
        return False
    execution = case.get("execution") if isinstance(case.get("execution"), dict) else {}
    model_policy = str(execution.get("model_calls") or "allowed").lower()
    if mode == "offline" or model_policy == "disallowed":
        return False
    return mode in {"hybrid", "classifier-live", "smoke-live"}


def _trace_case_fields(trace: ModelCallTrace, backend_testing_mode: bool, ai_enabled: bool) -> dict[str, Any]:
    fields = trace.as_case_fields()
    fields.update(
        {
            "backend_testing_mode": bool(backend_testing_mode),
            "ai_enabled": bool(ai_enabled),
            "model_name": fields.get("model_name") or (os.getenv("OPENAI_MODEL", "gpt-5.4-mini") if ai_enabled else None),
        }
    )
    return fields


def _answer_source(response_json: dict[str, Any], trace: ModelCallTrace) -> str:
    answer = response_json.get("answer") if isinstance(response_json.get("answer"), dict) else {}
    title = str(answer.get("title") or "")
    if response_json.get("_status_code", 200) >= 400:
        return "http_error"
    if title == "Analysis Plan Error":
        return "planner_failure"
    if title in {"Analysis Error", "Analysis Not Run"}:
        return "analysis_error"
    if trace.used_final_model_synthesis:
        return "final_model_synthesis"
    return "deterministic_presenter"


def _run_direct_classifier_case(case: dict[str, Any], trace: ModelCallTrace) -> dict[str, Any]:
    classifier = case.get("classifier") if isinstance(case.get("classifier"), dict) else {}
    employer = classifier.get("employer", "")
    occupation = classifier.get("occupation", "")
    industry = classifier.get("industry", "tech")
    descriptor_text = classifier.get("descriptor_text", "")
    filter_spec = {"filter_type": "industry", "industry": industry, "industries": [industry]}
    if classifier.get("query_scope"):
        filter_spec["query_scope"] = classifier.get("query_scope")
    if classifier.get("required_industries"):
        filter_spec["required_industries"] = list(classifier.get("required_industries") or [])
    if classifier.get("excluded_industries"):
        filter_spec["excluded_industries"] = list(classifier.get("excluded_industries") or [])
    query_spec = people_classifier.query_spec_from_filter(
        filter_spec,
        default_industry=industry,
    )
    model_classifier = budgeted_model_classifier(budget=int(classifier.get("model_budget") or 1))
    outcome = people_classifier.classify_candidate(
        occupation,
        employer,
        query_spec,
        descriptor_text=descriptor_text,
        model_classifier=model_classifier,
    )

    expected_classification = classifier.get("expected_classification")
    expected_count_as_match = classifier.get("expected_count_as_match")
    failures = []
    categories = set()
    if expected_classification and outcome.get("classification") != expected_classification:
        failures.append(
            f"Expected classifier label {expected_classification}, got {outcome.get('classification')}."
        )
        categories.add("classification")
    if expected_count_as_match is not None and bool(outcome.get("count_as_match")) != bool(expected_count_as_match):
        failures.append(
            f"Expected count_as_match={bool(expected_count_as_match)}, got {bool(outcome.get('count_as_match'))}."
        )
        categories.add("classification")

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "question": case.get("question"),
        "passed": not failures,
        "precision": None,
        "recall": None,
        "expected_count": None,
        "returned_count": 1,
        "displayed_count": None,
        "app_count": None,
        "failures": failures,
        "failure_categories": sorted(categories),
        "false_positives": [],
        "false_negatives_sample": [],
        "hallucinated_names": [],
        "displayed_columns": [],
        "extraction_source": "direct_classifier",
        "scored_from": "direct_classifier",
        "lower_confidence_extraction": False,
        "answer_source": "direct_classifier",
        "classifier_input": {
            "industry": industry,
            "employer": employer,
            "occupation": occupation,
        },
        "classifier_result": outcome,
        "raw_answer_excerpt": (
            f"{employer} / {occupation}: {outcome.get('classification')} "
            f"count_as_match={outcome.get('count_as_match')} reason={outcome.get('internal_reason')}"
        ),
    }


def _upload_app_view(client, app_view_path: Path) -> str:
    payload = BytesIO(app_view_path.read_bytes())
    response = client.post(
        "/api/upload",
        data={"file": (payload, app_view_path.name)},
        content_type="multipart/form-data",
    )
    data = response.get_json(silent=True) or {}
    if response.status_code not in {200, 201} or not data.get("dataset_id"):
        raise RuntimeError(f"Upload failed with {response.status_code}: {data or response.get_data(as_text=True)}")
    return str(data["dataset_id"])


def _ask(client, dataset_id: str, question: str) -> dict[str, Any]:
    response = client.post("/api/ask", json={"dataset_id": dataset_id, "question": question})
    data = response.get_json(silent=True)
    if not isinstance(data, dict):
        data = {"answer_text": response.get_data(as_text=True)}
    data["_status_code"] = response.status_code
    if response.status_code >= 400:
        data.setdefault("answer_text", data.get("error") or response.get_data(as_text=True))
    return data


if __name__ == "__main__":
    raise SystemExit(main())
