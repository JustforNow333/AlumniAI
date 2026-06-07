# AI Spreadsheet Analyst Project Summary

Built a Flask backend under `backend/` and connected the uploaded React/Babel frontend under `frontend/` for the AI Spreadsheet Analyst MVP.

Completed increments:

1. Upload and preview
   - Added `POST /api/upload` for `.csv` and `.xlsx` files.
   - Saves files into `backend/uploads/` with `secure_filename` and UUID prefixes.
   - Reads spreadsheets into pandas DataFrames.
   - Cleans column names, handles duplicate/blank names, and returns JSON-safe previews.

2. Dataset summary
   - Added `GET /api/datasets/<dataset_id>/preview`.
   - Added `GET /api/datasets/<dataset_id>/summary`.
   - Summary includes shape, columns, dtypes, missing counts, total missing values, numeric summaries, categorical summaries, date summaries, and duplicate row count.

3. Basic ask endpoint
   - Added `POST /api/ask`.
   - Reloads the DataFrame by `dataset_id`.
   - Builds OpenAI context from dataset ID, filename, row count, column count, column names, column types, first 10 rows, pandas summaries, and any computed safe backend result.
   - Uses the OpenAI Responses API when `OPENAI_API_KEY` is configured.
   - Returns a structured `answer` object plus `answer_text`, `operation`, and `result`.

4. Safe analysis functions
   - Added fixed pandas operations in `analysis_service.py`.
   - No `eval`, `exec`, arbitrary Python execution, vector database, or generated-code execution.
   - Implemented summary, column summary, group-by aggregation, top rows, basic filtering, and correlation.
   - Added a small rule-based intent layer for common spreadsheet questions before calling OpenAI.

5. Frontend connection
   - Uploaded frontend lives in `frontend/` and is a no-build React 18 + Babel static app.
   - `frontend/index.html` now defaults to API mode with `window.ALUMNI_CONFIG = { useApi: true, apiBase: "" }`, so Flask-served frontend requests use same-origin relative `/api/*` URLs.
   - `frontend/api.jsx` calls `POST /api/upload`, `GET /api/datasets/<dataset_id>/preview`, `GET /api/datasets/<dataset_id>/summary`, and `POST /api/ask` using the active `dataset_id`.
   - `frontend/api.jsx` normalizes structured backend answers and still safely adapts legacy plain-text answers into a markdown block.
   - Upload UI accepts `.csv` and `.xlsx` when API mode is enabled; local demo fallback only supports CSV-style text files.
   - Flask serves the frontend from `GET /` and static frontend assets by filename, so running `python backend/run.py` exposes the app and API from `http://localhost:5000`.
   - CORS still allows common separate frontend dev origins including `localhost:3000`, `localhost:5173`, and `localhost:8000`.

6. OpenAI fallback behavior
   - `OPENAI_API_KEY` is optional for backend startup.
   - When no key is configured, `/api/ask` still returns safe analysis results with deterministic structured answer blocks.
   - If OpenAI is configured but the API call fails, the backend still returns the computed safe operation/result and includes the failure as a safe markdown notice.
   - If OpenAI returns invalid JSON, the backend falls back to deterministic structured blocks instead of sending malformed model output.

7. Automated backend tests
   - Added pytest suite in `backend/tests/test_api.py`.
   - Tests create temporary CSV/XLSX uploads in memory and do not depend on files in `backend/uploads`.
   - Tests disable OpenAI by setting the AI client to `None`, so no real API key or internet access is required.
   - Coverage includes health, upload validation, preview, summary, ask validation, structured answer blocks, invalid model JSON fallback, summary/missing/group-by/average/top/correlation intents, unsupported prompts, dangerous read-only prompts, dataset isolation, persistence/restart behavior, missing persisted files, spaced column names, and a 1,000-row XLSX smoke test.
   - Pytest dependencies live in `backend/requirements.txt`.
   - Run from the project root with `pip install -r backend/requirements.txt && pytest`.
   - Added root `pytest.ini` so pytest discovers `backend/tests` and imports `backend/app` as top-level `app`.

8. Dataset persistence
   - Added local MVP persistence with uploaded files in `backend/uploads/` and metadata in `backend/data/datasets.json`.
   - Added `backend/app/services/dataset_store.py` for storage paths, registry load/save, upload registration, metadata lookup, dataframe reloads, missing-file errors, and optional cache clearing.
   - `datasets.json` stores metadata only: dataset ID, original/stored filenames, relative file path, file type, upload timestamp, row/column counts, and columns.
   - `POST /api/upload` writes the uploaded file and metadata, then returns the existing `dataset_id`, `filename`, and `summary` fields plus `metadata`.
   - `POST /api/ask`, `GET /api/datasets/<dataset_id>/preview`, and `GET /api/datasets/<dataset_id>/summary` reload CSV/XLSX data from disk using the persisted registry.
   - Flask config supports `UPLOAD_FOLDER`, `DATA_FOLDER`, and `DATASET_REGISTRY_PATH` so tests can use temporary isolated storage.
   - `.gitignore` keeps `backend/uploads/.gitkeep` and `backend/data/.gitkeep` while ignoring uploaded files and generated registry metadata.

9. Frontend uploaded dataset preview
   - After a successful upload, the frontend keeps the returned `dataset_id` on the dataset object and automatically calls `GET /api/datasets/<dataset_id>/preview`.
   - `frontend/api.jsx` uses same-origin relative URLs by default (`/api/upload`, `/api/datasets/<dataset_id>/preview`, `/api/ask`) to avoid localhost versus 127.0.0.1 CORS issues when Flask serves the UI.
   - Added `window.Alumni.preview(...)` in `frontend/api.jsx` to adapt both legacy (`column_names`, `preview`) and enriched (`columns`, `rows`, `filename`, counts, data types, missing values) preview response shapes.
   - The existing Claude-designed workspace dataset panel in `frontend/app.jsx` now merges the preview response into the active dataset state without changing the visual layout.
   - The right-side dataset panel updates from the uploaded file's real filename, row count, column count, missing value count, column names, inferred data types, and first 10 preview rows.
   - Upload failures and post-upload preview failures are surfaced separately (`Upload failed: ...` versus `Preview failed: ...`).
   - Added dependency-free Node tests in `frontend/tests/api-preview.test.mjs` for relative preview URL construction, response adaptation, and local fallback behavior.

10. Enriched dataset preview API
   - `GET /api/datasets/<dataset_id>/preview` now returns `dataset_id`, `filename`, `row_count`, `column_count`, `missing_count`, `columns`, `data_types`, `missing_values`, and `rows`.
   - The endpoint keeps the older `column_names` and `preview` fields for compatibility.
   - Backend tests now verify the enriched preview metadata and row payload.

11. Structured AI answer rendering
   - `backend/app/services/ai_service.py` now asks the model for structured JSON with `title`, `summary`, `blocks`, and `followups`.
   - Supported answer block types are `markdown`, `table`, `metrics`, and `ranked_list`.
   - Backend normalization strips model-provided HTML tags, ignores unknown block types, caps block sizes, and falls back to safe deterministic markdown/metric/table/list blocks when model JSON is invalid.
   - `POST /api/ask` returns the structured `answer` object and `answer_text` while preserving the existing `operation` and `result` payloads.
   - `frontend/app.jsx` adds `AnswerRenderer`, `AnswerCard`, `MarkdownBlock`, `TableBlock`, `MetricsBlock`, `RankedListBlock`, and `FollowupChips` inside the existing Claude-style chat layout.
   - Follow-up chips submit their text through the same ask flow and reuse the current uploaded dataset's `dataset_id`.
   - `sample-data.jsx` is no longer shown in normal API mode; it remains available only for explicit non-API demo mode.
   - Frontend adapter tests now cover structured ask rendering, legacy answer adaptation, summary URL construction, and local demo isolation.

12. Safe pandas-backed analysis toolkit
   - Added a planner/executor/presenter ask flow: planner JSON selects approved operations, pandas executes them on the full persisted DataFrame, and the presenter returns structured answer blocks.
   - Added `backend/app/services/analysis_toolkit.py` with whitelisted operations for preview, selection, filters, contains-any/all text search, sorting, top/bottom rows, group-by count/sum/average, value counts, missing values, column/numeric summaries, correlations, unique values, duplicates, date summaries, date range filters, and monthly grouping.
   - Added `backend/app/services/analysis_planner.py`, `analysis_executor.py`, `answer_presenter.py`, and `answer_schema.py` for strict JSON planning, operation validation, limit capping, deterministic fallback answers, and safe answer normalization.
   - Planner context is compact and includes dataset ID, filename, row/column counts, column names/types, missing counts, unique counts, sample values, low-cardinality values, and a small row sample. The full dataset is never sent to the model.
   - The ask route now reloads the full dataset by `dataset_id`, validates model-selected operations against a whitelist, executes only safe pandas operations, and returns `analysis_plan` plus normalized `operation_results`.
   - No arbitrary Python, generated code, `eval`, or `exec` is accepted from the model. Unknown operation types and unknown columns return structured errors.
   - Fallback planning handles common questions without OpenAI, including tech-related text search, top donors/top rows, average or total numeric summaries, group-by summaries, missing values, correlations, duplicates, date summaries, and read-only refusals for mutation requests.
   - Backend tests now cover toolkit operations, planner/presenter fallback behavior, full-dataset ask behavior beyond preview rows, normalized operation result schemas, and invalid model JSON safety.

13. Semantic analysis intent layer
   - Added an inference-first ask flow in `backend/app/services/analysis_intent.py`.
   - `/api/ask` now builds compact dataset context, infers analytical intent, validates the intent JSON, resolves semantic columns to actual dataframe columns, converts the intent to whitelisted pandas operations, executes only backend operations, and then lets the presenter format computed results.
   - The LLM is used only for semantic inference and presentation. It does not execute code and does not compute final numeric answers.
   - Intent JSON captures intent, target entity, user goal, concepts, search terms, known entities, semantic columns, filters, sorting, aggregation, desired output, assumptions, and clarification needs.
   - Semantic column resolution supports exact matches, case-insensitive matches, normalized matches, synonym maps, known-entity sample-value hints, and high-confidence fuzzy matches.
   - Common alumni semantics now map variants such as `occupation -> OCCUPATION`, `employer/company -> EMPLOYER`, `person_name -> NICKNAME or NAME`, `grad_year -> GRAD YR`, and `lifetime_giving/numeric_value -> LIFETIME GIVING` when matching columns exist.
   - Missing optional semantic columns no longer fail an analysis. For example, a tech-related search can run against `OCCUPATION` and `EMPLOYER` even when `Industry` and `Major` are absent.
   - If inferred filters or requested concepts cannot be applied to available columns, the backend returns a clarification/error plan instead of showing arbitrary unfiltered rows.
   - Deterministic intent fallback still handles common read-only requests when OpenAI is unavailable or returns invalid intent JSON.
   - Added backend tests for intent validation, uppercase semantic column resolution, tech-related operation mapping, optional missing semantic columns, unavailable concept clarification, and answer assumptions for fuzzy concepts.

14. Robust toolkit column resolution and safe text search fallback
   - Hardened `backend/app/services/analysis_toolkit.py` so toolkit operations resolve exact, case-insensitive, normalized, and synonym-based column names before execution.
   - Toolkit synonyms now map occupation/job/title/role/position, employer/company/organization/workplace, graduation/class/grad year variants, name/full name/nickname, and major/degree/field-of-study variants onto actual dataframe columns.
   - Added default searchable-column inference for `contains_any`, `contains_all`, `search_text`, and `filter_contains` when requested columns cannot be resolved.
   - Default text search prioritizes actual dataframe columns whose names look like occupation, employer/company, industry, major/degree, title, role, or position.
   - Text search remains case-insensitive by default, handles null/NaN/empty values safely, searches the full dataframe, and includes `matched_column` and `matched_term` metadata in filtered results.
   - Operation results now include `is_filtered` so presenters and clients can distinguish filtered result sets from non-filtered summaries or previews.
   - Failed search/filter operations return structured errors and do not include unfiltered rows as matches. Deterministic answer rendering does not create a result table when only failed filter operations are available.
   - Added backend tests for uppercase `OCCUPATION`/`EMPLOYER` resolution, `Company -> EMPLOYER`, inferred searchable columns, safe failed-filter answers, and the end-to-end tech-alumni ask route over persisted data.

15. Intent-aware text-search display and clearer metrics
   - Text-search operations now keep `search_columns` separate from `display_columns`.
   - Row-level lookup answers default to concise display columns: best available name, occupation/job title, employer/company, and `MATCH REASON`.
   - Search-only columns such as `MAJOR` are no longer displayed just because they were searched. They are shown only when the user asks for them, for example “tech alumni and their majors.”
   - Added a synthetic `MATCH REASON` column for fuzzy/text searches, such as `Matched OCCUPATION: Software Engineer` or `Matched EMPLOYER: Google`.
   - Text-search results now return explicit top-level counts: `total_rows`, `raw_match_count`, `matched_row_count`, `returned_row_count`, `display_limit`, and `deduplicated`.
   - Backward-compatible `metrics` still exist, but deterministic answer rendering now labels filtered metrics as `Unique alumni matched`, `Rows shown`, `Total dataset rows`, `Raw keyword hits`, and `Display limit`.
   - Table captions now state which columns were searched and explain deduplication or display limits when applicable.
   - Presenter instructions were tightened so live model presentation does not add extra searched fields to result tables and uses clear metric labels.
   - Added tests for default tech-alumni display columns, major inclusion only when requested, separate search/display columns, match reasons, raw keyword hits versus unique rows, clear metrics, and display-limit explanations.

16. Concept-library semantic inference
   - Expanded `backend/app/services/analysis_intent.py` with a built-in concept library for `tech_related`, `software_engineer_role`, and `tech_company`.
   - Known concepts now expand missing `search_terms` and `known_entities` before planning, so model output such as `software_engineer_role` with an empty term list no longer produces a “no search terms were inferred” failure.
   - Deterministic semantic inference recognizes tech, software engineer, developer, technical role, tech company, and startup language even when OpenAI is unavailable or under-infers concepts.
   - Tech/software company lookups now plan grouped text searches: role concepts search occupation-like columns, company concepts search employer-like columns, and broader tech concepts can search optional occupation/employer/industry/major columns.
   - Missing optional columns such as `Industry` or `Major` do not fail a tech search when actual `OCCUPATION` and `EMPLOYER` columns are available.
   - The exact query “Which alumni work in tech as either software engineers or any other role in a tech company?” now plans a valid `contains_any` operation over the full persisted dataset, returns concise display columns with `MATCH REASON`, and explains the inferred criteria.
   - Added backend tests for tech-company term inference, software-engineer role inference, concept-library term expansion, grouped occupation/employer search, missing optional columns, match reasons, minimal display columns, and clarification when no relevant searchable columns exist.

Important files:

- `backend/run.py`
- `backend/app/__init__.py`
- `backend/app/routes/upload_routes.py`
- `backend/app/routes/dataset_routes.py`
- `backend/app/routes/chat_routes.py`
- `backend/app/services/spreadsheet_service.py`
- `backend/app/services/dataset_store.py`
- `backend/app/services/analysis_service.py`
- `backend/app/services/analysis_toolkit.py`
- `backend/app/services/analysis_intent.py`
- `backend/app/services/analysis_planner.py`
- `backend/app/services/analysis_executor.py`
- `backend/app/services/answer_presenter.py`
- `backend/app/services/answer_schema.py`
- `backend/app/services/ai_service.py`
- `backend/app/utils/file_utils.py`
- `README.md`
- `backend/README.md`
- `backend/requirements.txt`
- `backend/tests/test_api.py`
- `backend/tests/test_analysis_toolkit.py`
- `backend/tests/test_analysis_intent.py`
- `backend/tests/test_ask_analysis_flow.py`
- `pytest.ini`
- `frontend/index.html`
- `frontend/app.jsx`
- `frontend/api.jsx`
- `frontend/engine.jsx`
- `frontend/kit.jsx`
- `frontend/sample-data.jsx`
- `frontend/theme.css`
- `frontend/README.md`
- `frontend/tests/api-preview.test.mjs`

Run notes:

- Start the full app from the project root with `./start_app.sh` or `start_app.bat`.
- Open `http://localhost:5000` to use the connected frontend.
- Flask serves the frontend from `/`, so one command starts the backend, frontend, and app experience together.
- The Flask-served frontend uses relative API URLs by default; if serving the frontend separately, set `apiBase` to the backend origin.
