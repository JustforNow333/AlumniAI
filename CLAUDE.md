# CLAUDE.md

Operational guidance for Claude Code in this repository. See `agents.md` for the current architecture and invariants; see `README.md` for the user quick start.

## Commands

```bash
# Run the app
./start_app.sh                     # Windows: start_app.bat

# Backend
./backend/venv/Scripts/python.exe -m pytest -q
./backend/venv/Scripts/python.exe -m pytest backend/tests/test_api.py -q

# Frontend
node --test frontend/tests/*.mjs

# Static validation
python3 -m compileall -q backend/app backend/evals backend/tests
git diff --check

# Product evals (from backend/)
./venv/Scripts/python.exe -m evals.run_evals --mode offline
./venv/Scripts/python.exe -m evals.run_evals --mode offline --category intent_filter
./venv/Scripts/python.exe -m evals.run_evals --mode offline --category filter_composition
./venv/Scripts/python.exe -m evals.run_evals --mode offline \
  --dataset evals/datasets/expanded_predicates.csv \
  --app-view evals/generated/expanded_predicates_app_view.csv \
  --cases evals/expanded_predicate_cases.jsonl \
  --category expanded_predicates
```

The checked-in environment is a Windows virtualenv at `backend/venv/Scripts/python.exe`; use the active Python interpreter if it is unavailable. Install dependencies from `backend/requirements.txt`.

`OPENAI_API_KEY` is optional. Tests and the offline product path must work with `ai_service.client = None`.

## Working rules

- After every user prompt, review and update `.gitignore`, `agents.md`, and `CLAUDE.md` with any new durable guidance, constraints, or generated artifacts. Keep them concise and avoid chronological logs.
- Preserve the no-build React/Babel frontend and route network access through `frontend/api.jsx`.
- Keep analysis on the whitelist in `analysis_toolkit.py`; never add generated-code execution, `eval`, `exec`, or arbitrary expressions.
- Models may infer intent and format computed results. They must not receive the full dataset, execute operations, or determine final counts.
- Validate operations, schema mappings, columns, and predicates on the backend. Do not rely on frontend validation.
- Do not rename DataFrame columns or mutate uploaded files to apply schema mappings.
- Preserve atomic JSON-registry writes, storage-root path containment, and compatibility with older records.
- Keep classifier, predicate, and model-debug fields out of normal user-facing tables.
- Counts must come from verified structured rows, not prose or display limits.
- Use `spreadsheet_service.to_json_safe` for API data containing pandas, NumPy, date, or missing-value types.
- Tests must use temporary storage and must not depend on files under `backend/uploads/` or `backend/data/`.

## Change checklist

For query behavior, trace the complete path:

```text
load dataset
→ build compact schema-aware context
→ infer intent
→ apply exact intent filter
→ create approved plan
→ execute whitelisted operations
→ verify rows and counts
→ sanitize and present
```

When changing semantic resolution, update all relevant resolver paths: intent planning, exact predicates, people classification/display, toolkit execution, and dataset context. Confirmed dataset mappings must outrank inferred mappings and generic aliases; multi-column canonical fields such as email must remain multi-column.

Run focused tests while editing, then the full backend and frontend suites for cross-cutting changes. Run the offline eval suite when modifying intent, filtering, classification, schema resolution, execution, counting, or presentation.

## Security-sensitive behavior

- `GET /api/debug/classify-row` must remain available only in `FLASK_DEBUG`.
- Keep dataset paths within configured storage roots.
- Preserve response security headers and runtime-generated/default-configured Flask secrets.
- Never persist raw schema-inference samples or expose filesystem paths.
- Explicit exact predicates must not be weakened into fuzzy searches.
- Preserve numeric/date boundary direction and inclusivity, recursive predicate grouping, and missing-value safety when repairing model intent.
