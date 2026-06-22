from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_report(
    case_results: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for result in case_results if result.get("passed"))
    failed = total - passed
    precision_values = [float(result["precision"]) for result in case_results if result.get("precision") is not None]
    recall_values = [float(result["recall"]) for result in case_results if result.get("recall") is not None]
    total_hallucinations = sum(len(result.get("hallucinated_names") or []) for result in case_results)
    total_model_calls = sum(int(result.get("total_model_calls") or 0) for result in case_results)
    cases_with_model_calls = sum(1 for result in case_results if int(result.get("total_model_calls") or 0) > 0)
    answer_source_breakdown = Counter(str(result.get("answer_source") or "unknown") for result in case_results)
    llm_classifier_call_count = sum(int(result.get("llm_classifier_calls") or 0) for result in case_results)
    final_model_synthesis_call_count = sum(
        int(result.get("final_model_synthesis_calls") or 0) for result in case_results
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "summary": {
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "average_precision": round(sum(precision_values) / len(precision_values), 4) if precision_values else None,
            "average_recall": round(sum(recall_values) / len(recall_values), 4) if recall_values else None,
            "total_hallucinations": total_hallucinations,
            "total_model_calls": total_model_calls,
            "cases_with_model_calls": cases_with_model_calls,
            "cases_without_model_calls": total - cases_with_model_calls,
            "answer_source_breakdown": dict(sorted(answer_source_breakdown.items())),
            "llm_classifier_call_count": llm_classifier_call_count,
            "final_model_synthesis_call_count": final_model_synthesis_call_count,
        },
        "cases": case_results,
    }


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Eval Report",
        "",
        f"Total cases: {summary.get('total_cases', 0)}",
        f"Passed: {summary.get('passed', 0)}",
        f"Failed: {summary.get('failed', 0)}",
        f"Pass rate: {_format_pct(summary.get('pass_rate'))}",
        f"Average precision: {_format_float(summary.get('average_precision'))}",
        f"Average recall: {_format_float(summary.get('average_recall'))}",
        f"Total hallucinated names: {summary.get('total_hallucinations', 0)}",
        f"Total model calls: {summary.get('total_model_calls', 0)}",
        f"Cases with model calls: {summary.get('cases_with_model_calls', 0)}",
        f"Cases without model calls: {summary.get('cases_without_model_calls', 0)}",
        f"LLM classifier calls: {summary.get('llm_classifier_call_count', 0)}",
        f"Final model synthesis calls: {summary.get('final_model_synthesis_call_count', 0)}",
        "",
    ]

    metadata = report.get("metadata") or {}
    if metadata:
        lines.extend(
            [
                "## Run Metadata",
                "",
                f"- Dataset: `{metadata.get('dataset', '')}`",
                f"- App-facing CSV: `{metadata.get('app_view_dataset', '')}`",
                f"- Mode: `{metadata.get('mode', '')}`",
                f"- Any live AI enabled: `{metadata.get('live_ai_enabled', False)}`",
                "",
            ]
        )

    breakdown = summary.get("answer_source_breakdown") or {}
    if breakdown:
        lines.extend(["## Answer Sources", ""])
        for source, count in breakdown.items():
            lines.append(f"- `{source}`: {count}")
        lines.append("")

    failed_cases = [case for case in report.get("cases") or [] if not case.get("passed")]
    lines.extend(["## Failed Cases", ""])
    if not failed_cases:
        lines.extend(["No failed cases.", ""])
    else:
        for case in failed_cases:
            lines.extend(_failed_case_lines(case))

    lines.extend(["## Slowest Cases", ""])
    slowest = sorted(report.get("cases") or [], key=lambda case: case.get("duration_ms") or 0, reverse=True)[:5]
    if not slowest:
        lines.extend(["No cases were run.", ""])
    else:
        for case in slowest:
            lines.append(
                f"- `{case.get('id')}` ({case.get('duration_ms', 0):.0f} ms): {case.get('question')}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")


def _failed_case_lines(case: dict[str, Any]) -> list[str]:
    lines = [
        f"### `{case.get('id')}`",
        "",
        f"- Category: `{case.get('category')}`",
        f"- Question: {case.get('question')}",
        f"- Expected count: {case.get('expected_count')}",
        f"- Returned count: {case.get('returned_count')}",
        f"- Displayed count: {case.get('displayed_count')}",
        f"- App count: {case.get('app_count')}",
        f"- Precision / recall: {_format_float(case.get('precision'))} / {_format_float(case.get('recall'))}",
        f"- Failure categories: `{', '.join(case.get('failure_categories') or []) or 'uncategorized'}`",
        f"- Answer source: `{case.get('answer_source')}`",
        f"- Scored from: `{case.get('scored_from') or case.get('extraction_source')}`",
        f"- Model calls: {case.get('total_model_calls', 0)} total, {case.get('llm_classifier_calls', 0)} classifier, {case.get('final_model_synthesis_calls', 0)} final synthesis",
    ]

    failures = case.get("failures") or []
    if failures:
        lines.append("- Failure reasons:")
        for failure in failures:
            lines.append(f"  - {failure}")

    hallucinated = case.get("hallucinated_names") or []
    if hallucinated:
        lines.append(f"- Hallucinated names: {', '.join(hallucinated[:20])}")

    false_positives = case.get("false_positives") or []
    if false_positives:
        lines.append(f"- False positives: {', '.join(false_positives[:20])}")

    false_negatives = case.get("false_negatives_sample") or []
    if false_negatives:
        lines.append(f"- Sample false negatives: {', '.join(false_negatives[:20])}")

    excerpt = case.get("raw_answer_excerpt")
    if excerpt:
        lines.extend(["- Raw answer excerpt:", f"  > {excerpt}"])

    lines.append("")
    return lines


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
