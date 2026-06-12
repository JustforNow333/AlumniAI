# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AI Spreadsheet Analyst: upload a CSV/XLSX, preview it, and ask natural-language questions answered by a fixed whitelist of safe pandas operations. Flask backend (`backend/`) serves a no-build React 18 + Babel-standalone frontend (`frontend/`) from the same origin at `http://localhost:5000`; API routes live under `/api/*`.

## Commands

```bash
# Start the app (backend + frontend together)
./start_app.sh            # Windows: start_app.bat

# Backend tests (run from project root; pytest.ini sets testpaths/pythonpath)
python -m pytest -q
python -m pytest backend/tests/test_analysis_toolkit.py -q          # one file
python -m pytest backend/tests/test_api.py::test_health -q          # one test

# Frontend tests (dependency-free Node test runner)
node --test frontend/tests/api-preview.test.mjs
```

The backend virtualenv is a **Windows** venv at `backend/venv/Scripts/python.exe`. From WSL, run tests with `./backend/venv/Scripts/python.exe -m pytest -q`. Dependencies: `pip install -r backend/requirements.txt`.

`OPENAI_API_KEY` in `backend/.env` is optional — without it the app uses deterministic heuristic intent inference and deterministic answer formatting. Tests never need a key (they set `ai_service.client = None`).

## Architecture

### Ask pipeline (the core flow)

`POST /api/ask` in `backend/app/routes/chat_routes.py` runs a strict inference → plan → execute → present pipeline. The LLM never executes code or computes numbers; it only infers intent and formats results.

1. `dataset_store.load_dataset_dataframe` reloads the full DataFrame from disk by `dataset_id`.
2. `analysis_toolkit.build_dataset_context` builds a compact context (column names/types, missing/unique counts, sample values, 5 sample rows — never the full dataset).
3. `analysis_intent.infer_analysis_intent` produces a semantic intent JSON, via OpenAI when configured, else `heuristic_intent`. Intents use *semantic* column names (`occupation`, `employer`, …) resolved to actual DataFrame columns through exact/case-insensitive/normalized/synonym/fuzzy matching. A `CONCEPT_LIBRARY` (`tech_related`, `software_engineer_role`, `tech_company`) expands fuzzy concepts into search terms. Broad people questions never error out: `alumni_tech_fallback_intent` handles "alumni in tech" phrasing, and `industry_taxonomies.classify_people_question` + `people_filter_fallback_intent` handle every other answerable industry/employer/occupation query (both in `heuristic_intent` and when the LLM is unsure or asks for clarification).
4. `analysis_intent.intent_to_analysis_plan` converts the intent into whitelisted operations (max 3 per request, see `ALLOWED_OPERATION_TYPES` in `analysis_toolkit.py`).
5. `analysis_executor.execute_analysis_plan` validates and runs each operation via `analysis_toolkit.execute_operation`. No `eval`/`exec`/generated code is ever accepted. Unknown operations/columns return structured `{"status": "error"}` results.
6. `answer_presenter.present_answer` formats results into a structured answer (`title`, `summary`, `blocks`, `followups`; block types: `markdown`, `table`, `metrics`, `ranked_list`), falling back to `answer_schema.deterministic_answer_from_results` when the model is unavailable or returns invalid JSON. `answer_schema.normalize_answer` strips HTML, caps sizes, and rejects unknown block types.

### Industry taxonomy people filter

People/alumni queries run through a reusable taxonomy + classification system, not tech-specific code. **Core principle: a broad keyword hit only makes a row a candidate; strict query-aware classification decides final inclusion.** Raw keyword hits must never be presented as the answer count.

- `industry_taxonomies.py` defines 12 editable taxonomies (tech, consulting, banking, finance, healthcare, law, education, media, nonprofit, startups, venture_capital, private_equity), each with `aliases`, `title_keywords`, `generic_title_keywords` (roles that never confirm alone), `employer_keywords`, `known_companies`, `exclusion_keywords`, `ambiguous_keywords`, `confidence_threshold`, and (consulting) `retrieval_keywords` — broad candidate-generation terms (strategy, operations, management, transaction, …) that never confirm a match on their own. The tech taxonomy merges `known_tech_companies.json` (the editable known-company config). Companies may belong to several industries (Spotify is tech *and* media).
- `classify_people_question` maps a question to a filter spec with `filter_type: industry | employer | occupation`. Longest alias wins, `INDUSTRY_PRIORITY` breaks ties (banking > finance, media > tech). A single capitalized company after "at" → employer filter ("Who works at Spotify?"); multiple companies from one industry → that industry ("at McKinsey or BCG" → consulting). "Who are founders?" → occupation filter; "startup founders" → startups taxonomy. Industry specs also carry query modifiers: `industries`, `include_adjacent` ("show consulting-adjacent alumni too"), `include_functions` ("consulting or strategy" → `internal_strategy`), and `required_industries` ("finance consulting" → consulting ∩ finance intersection).
- `people_classifier.py` (`classification_version: multi_label_v1`) is the query-aware multi-label classifier used for all industry filters. Each candidate gets an independent profile — `employer_industry` (array: an employer can be technology *and* media), `job_function` (array: consulting_advisory, finance_investing, internal_strategy, product, operations, legal, engineering, …), `specialties` (risk_consulting, transaction_advisory, valuation, restructuring, investment_banking, wealth_management, …) — then a per-query verdict: `classification: direct_match | adjacent | uncertain | non_match` plus `count_as_match`, `confidence`, `internal_reason`. Only `count_as_match=true` rows are counted/displayed; direct matches need confidence ≥ 0.70, and confidence can never promote adjacent/uncertain rows. Consulting has a dedicated deterministic policy (explicit consultant/consulting titles confirm alone; client-advisory titles confirm at professional-services firms; finance-only/legal roles never match; strategy/product/operations language without consulting context is adjacent, never direct). Other industries wrap `match_row_to_industry`; add new industries by registering a policy in `_INDUSTRY_POLICIES`. The LLM is consulted only for rows deterministic rules leave `uncertain`.
- `industry_matching.match_row_to_industry(occupation, employer, taxonomy)` is the layered per-taxonomy engine: title keyword → known company → employer keyword → exclusion context → optional budgeted LLM employer classifier (confirms only at/above `confidence_threshold`; no-op without an API key) → ambiguous → excluded. Generic business roles (founder, head of growth, CEO, …) count only with a matching employer. Returns `{status: confirmed|uncertain|excluded, match_sources, confidence, internal_reason}`.
- `column_resolver.py` maps messy source columns (`LastName`, `LinkedinURL`, …) to canonical person fields; visible headers for people results are always `First Name`, `Last Name`, `Occupation`, `Employer`, `LinkedIn URL` (never `Nickname` when first/last exist).
- Execution: `contains_any` with `filter_mode: "people"` (new) or `"tech_people"` (legacy, routes through the tech taxonomy) runs `_people_filter_result` in `analysis_toolkit.py`: broad keyword retrieval finds candidates, the classifier decides inclusion, uncertain/adjacent stay out of `total_matches`, people are deduplicated, and the people_filter shape is returned (`intent`, `entity`, `filter_type`, `industry`, `criteria_label`, `answer_label`, `total_dataset_rows`, `total_keyword_hits`, `total_matches`, `displayed_count`, `display_limit`, `uncertain_count`, `visible_columns`, dict rows) plus classification metadata (`raw_candidate_count`, `direct_match_count`, `adjacent_count`, `adjacent_included_count`, `non_match_count`, `classification_version`, `adjacent_included`). `total_matches` is always the headline count (direct matches only, unless the query explicitly asked for adjacent rows). Match reasons/confidence/multi-label profiles live under a separate `debug` key and must stay out of visible tables.
- Even when the LLM confidently returns a generic `find_records` keyword plan, `_plan_find_records` re-checks `classify_people_question` and routes people/industry questions through the strict classifier (`filter_mode: "people"`).
- Dev-only `GET /api/debug/classify-row?dataset_id=…&name=…&industry=…` explains why a specific person was included/excluded.
- The tech wrappers `is_explicit_technical_title`, `classify_employer_tech_status`, and `is_strong_non_tech_context` in `analysis_toolkit.py` are kept for backward compatibility and delegate to the generic engine.

### Persistence & dataset library

Uploads are stored as `backend/uploads/<uuid>_<filename>`; metadata-only registry in `backend/data/datasets.json` (`dataset_store.py`). DataFrames are re-read from disk per request — there is no in-memory dataset cache. Flask config keys `UPLOAD_FOLDER`, `DATA_FOLDER`, `DATASET_REGISTRY_PATH` let tests point at temp storage.

Dataset library endpoints (`dataset_routes.py`): `GET /api/datasets` lists all saved datasets newest-first (ties broken by registry insertion order) without loading DataFrames — each entry has `dataset_id`, `display_name` (falls back to `original_filename`), counts, `columns`, `file_type`, and `status` (`"ready"` or `"missing"` when the file is gone from disk; never crashes). `PATCH /api/datasets/<id>` renames (`{"display_name": ...}`, non-empty, atomic via the registry tmp-file write). `DELETE /api/datasets/<id>` removes metadata + the stored file (clean JSON 404 when unknown). Registry entries now carry `display_name` (older entries without it are tolerated by `dataset_public_metadata`).

### Saved insights

Manually saved answer snapshots tied to a `dataset_id` (`insight_store.py`, registry at `backend/data/saved_insights.json`, config key `INSIGHTS_REGISTRY_PATH` defaulting under `DATA_FOLDER`). **Insights are saved only when the user clicks Save insight — `/api/ask` never writes them; automatic chat history is a separate future feature.** An insight is a snapshot (`insight_id`, `dataset_id`, `dataset_name_snapshot`, `title`, `question`, `answer`, `created_at`/`updated_at`, `tags`, `metadata` with row/column counts): the answer is never recomputed and no DataFrame/file contents are stored. Endpoints (`insight_routes.py`): `GET /api/insights[?dataset_id=…]` newest-first; `POST /api/insights` (validates dataset exists + non-empty question/answer; title auto-generated from the question when missing) → 201; `GET/PATCH/DELETE /api/insights/<id>` (PATCH edits title/tags only — dataset_id/question/answer are immutable; clean JSON 404s). Insights whose dataset was deleted are kept and reported with `dataset_status: "deleted"` (vs `"ready"`) without crashing.

### Frontend

No build step: `index.html` loads React/Babel from CDN and the `.jsx` files via in-browser transform; everything is exposed on `window` (no modules). `window.ALUMNI_CONFIG = { useApi, apiBase }` toggles backend mode vs. an in-browser CSV demo mode (`engine.jsx` + `sample-data.jsx`, demo only). `api.jsx` is the backend adapter: it normalizes upload/preview/ask responses plus the dataset library (`Alumni.datasets()/renameDataset()/deleteDataset()`, entries normalized by `normalizeDatasetEntry`) and saved insights (`Alumni.insights()/insight()/saveInsight()/renameInsight()/deleteInsight()`, normalized by `normalizeInsightEntry`; helpers `defaultInsightTitle` and `insightTextFromAnswer` flatten a structured answer into the plain-text snapshot), sanitizes structured answers (hides debug columns unless `cfg().debug`, canonicalizes name/LinkedIn columns, uses `total_matches` as the headline stat for people-filter results). `app.jsx` holds the UI and the structured-answer renderers. `frontend/tests/api-preview.test.mjs` tests the adapter by evaluating `api.jsx` in Node — keep `api.jsx` free of JSX/browser-only syntax.

Dataset library UI (`app.jsx`): `App` owns `datasets`, `view` (`"chat" | "datasets" | "insights"`), the active dataset (`ds`), and preview state. On boot in API mode it fetches `GET /api/datasets`, restores the active dataset from localStorage key `alumniActiveDatasetId` (else newest), and loads its preview; with no datasets it shows the upload empty state. The sidebar `Rail` navigates between Conversations, Datasets, and Saved insights (History stays a placeholder); the ACTIVE DATASET item reflects/links to the library. `DatasetLibrary` renders in the main panel: select (activates + loads preview), upload, rename (`window.prompt`), delete (`window.confirm`; switches to the next dataset or clears). `Workspace` is keyed by `dataset_id` + a chat sequence, so switching datasets or "New analysis" starts a fresh conversation; asks always send the active `dataset_id` and are blocked client-side when none is active.

Saved insights UI (`app.jsx`): `AiMsg` shows a Save insight button under completed answers (only with an originating `question`, an answer, an active `dataset_id`, and API mode); it prompts for a title (default generated from the question), shows Saving…/Saved ✓ states, and blocks duplicate saves. `Workspace.send` stamps `question` onto AI messages for this. `InsightsLibrary` renders the list (All insights / Current dataset only filter — current disabled without an active dataset) and `InsightDetail` the full snapshot with an "Open dataset" button that activates the insight's dataset when it still exists; deleted datasets show a "Dataset deleted" badge but never break the view. `App` owns `insights`, `insightsLoading`, `insightsError`, and `selectedInsightId` (App-level so the open detail survives the Workspace remount when switching datasets); the list loads when the view opens and updates in place on save/rename/delete.

### Result schema conventions

Operation results carry `is_filtered`, top-level counts (`total_rows`, `raw_match_count`, `matched_row_count`, `returned_row_count`, `display_limit`), and a `metrics` dict that duplicates them plus backward-compatible aliases (`rows_matched`, `rows_returned`, `searched_columns`). Search operations keep `search_columns` separate from `display_columns` — searched-only columns are not displayed unless the user asked for them. Display limits must never be presented as the answer count.

## Notes

- `agents.md` is a detailed increment-by-increment history of the project; useful background but CLAUDE.md is the operational reference.
- All JSON responses must pass through `spreadsheet_service.to_json_safe` (handles NaN/NaT/numpy scalars/timestamps).
- `backend/uploads/` and `backend/data/datasets.json` are local dev artifacts (gitignored); don't depend on their contents in tests.
