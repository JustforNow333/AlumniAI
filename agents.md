# AlumniAI Agent Guide

This is the current project reference. Historical increment notes were removed because the code, tests, and git history are the authoritative change record.

## Product and layout

AlumniAI is a Flask application for persistent CSV/XLSX alumni datasets. A no-build React 18 frontend is served from the same origin.

- `backend/app/routes/`: upload, dataset, chat, insight, history, and development-debug endpoints.
- `backend/app/services/`: persistence, schema mapping, intent, execution, classification, and presentation.
- `frontend/api.jsx`: normalized API/demo adapter; keep it free of JSX so Node tests can evaluate it.
- `frontend/app.jsx`: application state, libraries, schema review, chat, and structured-answer rendering.
- `backend/tests/`, `frontend/tests/`, `backend/evals/`: regression and product validation.

## Core query pipeline

`POST /api/ask` keeps a strict separation between language understanding and computation:

```text
load the persisted DataFrame
→ build compact dataset context
→ infer and validate semantic intent
→ reconcile deterministic exact predicates
→ resolve semantic columns
→ build a whitelisted analysis plan
→ execute pandas operations
→ verify selected rows and counts
→ sanitize and present structured results
→ store successful history
```

OpenAI is optional and is used only for bounded intent/presentation tasks. The deterministic offline path is a supported product mode. The full DataFrame is never sent to a model, and model output never becomes executable code.

Important pipeline files:

- `routes/chat_routes.py`
- `services/analysis_intent.py`
- `services/intent_filter.py`
- `services/analysis_executor.py`
- `services/analysis_toolkit.py`
- `services/answer_presenter.py`
- `services/answer_schema.py`

## Dataset persistence and schema mapping

`dataset_store.py` stores uploads under the configured upload directory and metadata in an atomically written JSON registry. DataFrames are reloaded per request. Dataset listing reads metadata and checks file existence without loading every DataFrame. Older registry entries and missing files must serialize cleanly.

The schema layer maps original source columns to a centralized canonical registry:

- `canonical_schema.py`: canonical fields, labels, aliases, cardinality, and public metadata.
- `schema_inference.py`: exact/normalized aliases, institutional-header heuristics, conservative sample inference, confidence, conflicts, and optional bounded model suggestions.
- `schema_profile.py`: normalization, validation, status, merge/re-inference, and persistence helpers.
- `column_resolver.py`: schema-aware semantic resolution used by the query pipeline.

Profiles use `schema_mapping_v1` and statuses `unreviewed`, `needs_review`, or `confirmed`. New uploads receive a proposal; older datasets infer lazily. Confirmed mappings outrank saved high-confidence inference, which outranks generic alias and heuristic resolution. Re-inference preserves confirmed mappings unless reset is explicit. Single-cardinality fields cannot silently accept multiple sources; multi-value fields such as email and phone can.

Schema endpoints:

- `GET /api/datasets/<dataset_id>/schema`
- `PUT /api/datasets/<dataset_id>/schema`
- `POST /api/datasets/<dataset_id>/schema/infer`

Mappings never rename DataFrame columns or modify uploaded files. Schema metadata may contain bounded evidence, but never persisted sample values.

## Exact predicates and people classification

Exact constraints are typed predicates owned by `intent_filter.py` and executed by `filter_predicates` or `composite_people_filter`. Allowed operators are:

`exists`, `missing`, `equals`, `not_equals`, `contains`, `not_contains`, `email_domain_in`, `email_domain_not_in`, `greater_than`, `greater_than_or_equal`, `less_than`, `less_than_or_equal`, `between`, `in`, `not_in`, `date_before`, `date_after`, `date_between`, `starts_with`, and `ends_with`.

Legacy flat `predicates` roots remain valid. New roots use bounded `and`/`or` clause groups with at most three levels, twelve leaves, and twelve clauses per group. Predicates support `any`, `all`, and `none` across resolved source columns. Comparisons require a parseable value; blanks never satisfy comparisons or negative membership. Ranges are inclusive unless their bounds explicitly say otherwise, and relative dates resolve once per request.

Deterministic constraints from the original question take precedence over model suggestions. Boundary direction, inclusivity, values, grouping, and fuzzy clauses must be repaired from the source question when a model changes or omits them. Unsupported predicates fail safely instead of becoming fuzzy text searches.

Fuzzy people questions use broad retrieval only to find candidates. `people_classifier.py`, `industry_taxonomies.py`, and `industry_matching.py` decide final inclusion through query-aware deterministic policies. Direct matches drive headline counts; adjacent and uncertain rows are separate review buckets. Internal classification reasons, confidence, temporary row IDs, and predicate traces must not appear in normal tables.

For combined fuzzy and exact questions, evaluate both clauses over the original DataFrame and combine row sets according to the requested Boolean logic. Reverify final rows and recompute every count after composition.

## Result and display invariants

- People results use canonical display labels such as `First Name`, `Last Name`, `Occupation`, `Employer`, and `LinkedIn URL`.
- Opaque source headers should not leak into ordinary people-result tables unless explicitly requested.
- Search columns and display columns are separate.
- `total_matches` is the direct, verified answer count; `displayed_count` and `display_limit` are presentation values.
- Structured answers support `markdown`, `table`, `metrics`, and `ranked_list` blocks.
- Sanitize model output, reject unknown block types, cap payload sizes, and fall back deterministically.
- Preserve JSON-safe API output and hide debug-only fields unless explicit debug mode is enabled.

## Dataset library, history, and insights

The dataset library supports list, select, rename, delete, missing-file status, schema status, and schema review. Renaming preserves schema metadata; deleting a dataset removes its registry entry and uploaded file.

History is created only for successful operation-backed answers and stores the response snapshot. Saved insights are explicit user-created snapshots; `/api/ask` does not create them automatically. Deleting a source dataset must not break existing history or insight rendering.

Relevant services:

- `dataset_store.py`
- `history_store.py`
- `insight_store.py`

## Frontend constraints

The frontend has no build step: `index.html` loads React and Babel, and scripts expose APIs on `window`. API mode uses relative `/api/*` URLs; demo mode must not make real schema or dataset-library requests.

Keep fetch logic in `api.jsx`. Schema review opens after upload when review is needed, remains skippable, preserves unsaved edits on errors, and can be reopened from the dataset library. The frontend should prevent obvious invalid duplicate selections, but backend validation remains authoritative.

## Safety invariants

- No arbitrary Python, generated code, `eval`, or `exec`.
- No complete dataset or uncapped samples sent to a model.
- No uploaded-file mutation during analysis or mapping.
- No filesystem paths in public schema responses.
- No registry writes without atomic replacement.
- No reads or deletes outside configured storage roots.
- No classifier or predicate internals in normal presentation.
- `GET /api/debug/classify-row` is development-only.

## Prompt maintenance

After every user prompt, review and update `.gitignore`, `agents.md`, and `CLAUDE.md` with any new durable repository guidance, operating constraints, or generated artifacts. Keep all three concise and avoid chronological logs.

## Validation

Run from the repository root unless noted:

```bash
./backend/venv/Scripts/python.exe -m pytest -q
node --test frontend/tests/*.mjs
python3 -m compileall -q backend/app backend/evals backend/tests
git diff --check
```

Run the offline product eval from `backend/`:

```bash
./venv/Scripts/python.exe -m evals.run_evals --mode offline
```

Focused eval categories include `industry_classification`, `intent_filter`, `filter_composition`, and `expanded_predicates`; schema cases are in `backend/evals/schema_cases.jsonl`. Tests and offline evals must not require an OpenAI key or local production data.
