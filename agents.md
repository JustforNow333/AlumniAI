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
   - `backend/app/services/answer_presenter.py` now asks the model for structured JSON with `title`, `summary`, `blocks`, and `followups`.
   - Supported answer block types are `markdown`, `table`, `metrics`, and `ranked_list`.
   - Backend normalization strips model-provided HTML tags, ignores unknown block types, caps block sizes, and falls back to safe deterministic markdown/metric/table/list blocks when model JSON is invalid.
   - `POST /api/ask` returns the structured `answer` object and `answer_text` while preserving the existing `operation` and `result` payloads.
   - `frontend/app.jsx` adds `AnswerRenderer`, `AnswerCard`, `MarkdownBlock`, `TableBlock`, `MetricsBlock`, `RankedListBlock`, and `FollowupChips` inside the existing Claude-style chat layout.
   - Follow-up chips submit their text through the same ask flow and reuse the current uploaded dataset's `dataset_id`.
   - `sample-data.jsx` is no longer shown in normal API mode; it remains available only for explicit non-API demo mode.
   - Frontend adapter tests now cover structured ask rendering, legacy answer adaptation, summary URL construction, and local demo isolation.

12. Safe pandas-backed analysis toolkit
   - Added an intent/executor/presenter ask flow: inferred intent selects approved operations, pandas executes them on the full persisted DataFrame, and the presenter returns structured answer blocks.
   - Added `backend/app/services/analysis_toolkit.py` with whitelisted operations for preview, selection, filters, contains-any/all text search, sorting, top/bottom rows, group-by count/sum/average, value counts, missing values, column/numeric summaries, correlations, unique values, duplicates, date summaries, date range filters, and monthly grouping.
   - Added `backend/app/services/analysis_executor.py`, `answer_presenter.py`, and `answer_schema.py` for operation validation, limit capping, deterministic fallback answers, and safe answer normalization.
   - Intent context is compact and includes dataset ID, filename, row/column counts, column names/types, missing counts, unique counts, sample values, low-cardinality values, and a small row sample. The full dataset is never sent to the model.
   - The ask route now reloads the full dataset by `dataset_id`, validates model-selected operations against a whitelist, executes only safe pandas operations, and returns `analysis_plan` plus normalized `operation_results`.
   - No arbitrary Python, generated code, `eval`, or `exec` is accepted from the model. Unknown operation types and unknown columns return structured errors.
   - Deterministic intent fallback handles common questions without OpenAI, including tech-related text search, top donors/top rows, average or total numeric summaries, group-by summaries, missing values, correlations, duplicates, date summaries, and read-only refusals for mutation requests.
   - Backend tests now cover toolkit operations, intent/presenter fallback behavior, full-dataset ask behavior beyond preview rows, normalized operation result schemas, and invalid model JSON safety.

13. Semantic analysis intent layer
   - Added an inference-first ask flow in `backend/app/services/analysis_intent.py`.
   - `/api/ask` now builds compact dataset context, infers analytical intent, validates the intent JSON, resolves semantic columns to actual dataframe columns, converts the intent to whitelisted pandas operations, executes only backend operations, and then lets the presenter format computed results.
   - The LLM is used only for semantic inference and presentation. It does not execute code and does not compute final numeric answers.
   - Intent JSON captures intent, target entity, user goal, concepts, search terms, known entities, semantic columns, filters, sorting, aggregation, desired output, assumptions, and clarification needs.
   - Semantic column resolution supports exact matches, case-insensitive matches, normalized matches, synonym maps, known-entity sample-value hints, and high-confidence fuzzy matches.
   - Common alumni semantics now map variants such as `first_name -> First Name`, `last_name -> LastName or Last Name`, `linkedin_url -> LinkedIn URL or LinkedinURL`, `occupation -> OCCUPATION`, `employer/company -> EMPLOYER`, `person_name -> First Name/Last Name or NAME/NICKNAME fallback`, `grad_year -> GRAD YR`, and `lifetime_giving/numeric_value -> LIFETIME GIVING` when matching columns exist.
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
   - Alumni/person row-level lookup answers default to clean identity columns when available: `First Name`, `Last Name`, `Occupation`, `Employer`, and `LinkedIn URL` as the final column.
   - `NICKNAME` is not used as the primary displayed identity when first/last name fields exist. It remains only a fallback when first/last name fields are missing.
   - Search-only columns such as `MAJOR` are no longer displayed just because they were searched. They are shown only when the user asks for them, for example “tech alumni and their majors.”
   - Generic fuzzy/text searches can still keep internal match metadata, but normal alumni/person result tables do not show `MATCH REASON`, confidence, scores, model rationale, or other debug fields.
   - Text-search results now return explicit top-level counts: `total_rows`, `raw_match_count`, `matched_row_count`, `returned_row_count`, `display_limit`, and `deduplicated`.
   - Backward-compatible `metrics` still exist for generic text searches, but alumni/person query-result views use `total_matches` as the main answer count with the label `Alumni matching criteria`.
   - Table captions now state which columns were searched and explain deduplication or display limits when applicable.
   - Presenter instructions were tightened so live model presentation does not add extra searched fields or debug fields to result tables and does not present display limits as answer counts.
   - Added tests for default tech-alumni display columns, major inclusion only when requested, separate search/display columns, hidden debug fields, raw keyword hits versus unique rows, clear metrics, and display-limit explanations.

16. Concept-library semantic inference
   - Expanded `backend/app/services/analysis_intent.py` with a built-in concept library for `tech_related`, `software_engineer_role`, and `tech_company`.
   - Known concepts now expand missing `search_terms` and `known_entities` before planning, so model output such as `software_engineer_role` with an empty term list no longer produces a “no search terms were inferred” failure.
   - Deterministic semantic inference recognizes tech, software engineer, developer, technical role, tech company, and startup language even when OpenAI is unavailable or under-infers concepts.
   - Tech/software company lookups now plan grouped text searches: role concepts search occupation-like columns, company concepts search employer-like columns, and broader tech concepts can search optional occupation/employer/industry/major columns.
   - Missing optional columns such as `Industry` or `Major` do not fail a tech search when actual `OCCUPATION` and `EMPLOYER` columns are available.
   - The exact query “Which alumni work in tech as either software engineers or any other role in a tech company?” now plans a valid strict `contains_any` operation with `filter_mode: tech_people` over the full persisted dataset, returns person-focused display columns, and explains the inferred criteria.
   - Added backend tests for tech-company term inference, software-engineer role inference, concept-library term expansion, grouped occupation/employer search, missing optional columns, hidden match reasons, minimal display columns, and clarification when no relevant searchable columns exist.

17. Strict tech-alumni filtering and clean result rendering
   - Added a strict tech-person filter path behind the existing whitelisted `contains_any` operation. It classifies tech alumni by explicit technical titles, strong employer tech indicators, and a configurable known-tech-company list.
   - Added `backend/app/services/known_tech_companies.json` for ambiguous tech/startup employers such as `FanAmp`, `Cogni DAO`, `Amass Insights`, `Benchmrk`, `Launch Potato`, and `Rune Technologies`.
   - Tech-person filtering no longer counts loose generic matches such as school mathematics department chairs, oncology directors, generic founders, or generic CEOs unless the title is explicitly technical or the employer is clearly/classifiably tech.
   - Explicit technical titles such as `Software Engineer`, `Data Scientist`, `IT Director`, and related roles are included regardless of employer.
   - Results are deduplicated by stable identifiers when available, then by first/last name plus grad year or employer.
   - Tech-person operation results now separate `total_dataset_rows`, `total_keyword_hits`, `total_matches`, `displayed_count`, `display_limit`, and `uncertain_count`.
   - Deterministic answer rendering shows `Alumni matching criteria: X` from `total_matches`, optionally `Showing: Y`, and does not use the display limit as the answer.
   - Frontend answer adaptation strips debug-only table columns unless debug mode is enabled, canonicalizes `LastName`/`last_name` to `Last Name`, canonicalizes LinkedIn columns to `LinkedIn URL`, and renders LinkedIn cells as clickable links.
   - Added backend and frontend tests for first/last name display, hidden nickname/match-reason/debug fields, LinkedIn column resolution, clickable LinkedIn URL helpers, result count separation, strict tech inclusion/exclusion, known ambiguous tech companies, uncertain matches, and frontend use of `total_matches`.

18. GPT-style alumni tech query fallback
   - Added a deterministic alumni-tech fallback intent for broad but answerable questions such as “How many alumni are working in tech either as software engineers or as other roles in a tech company?”
   - If the model asks for clarification about strict versus broad tech matching on this query type, `/api/ask` now uses the default `people_filter` plan instead of returning an `Analysis Plan Error`.
   - The fallback plan uses `contains_any` with `filter_mode: tech_people`, `intent: people_filter`, `entity: alumni`, `criteria_label: working in tech or technical roles`, and `answer_label: Alumni matching criteria`.
   - Added reusable classifier helpers in `analysis_toolkit.py`: `is_explicit_technical_title`, `classify_employer_tech_status`, and `is_strong_non_tech_context`.
   - Confirmed tech matches come from explicit technical titles, strong tech employer keywords, and configurable known tech companies; weak ambiguous employers are tracked as uncertain and are not included in `total_matches`.
   - People-filter operation rows now use user-facing keys such as `First Name`, `Last Name`, `Occupation`, `Employer`, and `LinkedIn URL`; internal match reasons, confidence, and classifications live under a separate `debug` structure.
   - Frontend answer adaptation also hides `internal_reason` and raw `classification` columns unless debug mode is enabled, and keeps `total_matches` as the main stat.
   - Added regression tests for the broad alumni-tech query variants, the model clarification fallback path, strict classifier examples, and frontend analysis-plan-error avoidance for normal people-filter results.

19. Generalized industry taxonomy matching system (replaces one-off tech filtering)
   - Pipeline inspection notes: `/api/ask` lives in `backend/app/routes/chat_routes.py` and runs `infer_analysis_intent` → `intent_to_analysis_plan` (`analysis_intent.py`) → `execute_analysis_plan` (`analysis_executor.py`) → `execute_operation` (`analysis_toolkit.py`) → `present_answer`/`deterministic_answer_from_results` (`answer_presenter.py`/`answer_schema.py`). The `Analysis Plan Error` path is `intent_to_analysis_plan` returning an empty plan → `planner_failure_answer`. Frontend tables/stats render in `frontend/app.jsx` (`TableBlock`/`MetricsBlock`) after sanitization in `frontend/api.jsx` (`sanitizeStructuredAnswer`).
   - Added `backend/app/services/industry_taxonomies.py`: centralized, editable taxonomies for tech, consulting, banking, finance, healthcare, law, education, media, nonprofit, startups, venture_capital, and private_equity. Each taxonomy has `aliases`, `title_keywords`, `generic_title_keywords` (roles that never confirm alone), `employer_keywords`, `known_companies`, `exclusion_keywords`, `ambiguous_keywords`, and `confidence_threshold`. The tech taxonomy merges `known_tech_companies.json` (now expanded with Spotify, Netflix, Shopify, Anthropic, Databricks, GitHub, etc.), so `Head of Growth at Spotify` counts for tech. Companies may appear in multiple taxonomies (Spotify is tech and media).
   - `classify_people_question(question)` deterministically classifies people queries into `filter_type: industry | employer | occupation`: longest-alias matching for industries with `INDUSTRY_PRIORITY` tie-breaks (banking beats finance, media beats tech), capitalized-employer extraction after "at"/"for" (a single company → employer filter; multiple companies from one industry → that industry), and explicit role queries ("Who are founders?", "Show me product managers") → occupation filter. "Startup founders" routes to the startups taxonomy while bare "founders" stays an occupation filter.
   - Added `backend/app/services/industry_matching.py`: layered `match_row_to_industry(occupation, employer, taxonomy)` returning `{status: confirmed|uncertain|excluded, match_sources, confidence, internal_reason}`. Layers: title keyword → known company → employer keyword (including descriptor columns) → exclusion context → optional model classifier (budgeted via `budgeted_model_classifier`, only confirms at/above `confidence_threshold`, no-op when no OpenAI client is configured) → ambiguous wording → excluded. Generic business roles (founder, head of growth, CEO, ...) only count with a matching employer and are reported via `generic_business_role_with_matching_employer` in `match_sources`.
   - Added `backend/app/services/column_resolver.py`: canonical person-field aliases (first/last/full name, nickname, occupation, employer, linkedin_url, email, grad_year, major, location/city/state/country) shared by the toolkit display logic and the debug endpoint. Frontend-visible headers stay `First Name`, `Last Name`, `Occupation`, `Employer`, `LinkedIn URL` regardless of source column naming.
   - `analysis_toolkit.py`: `_tech_people_filter_result` became the generic `_people_filter_result` driven by `params.people_filter` (`filter_type`, `industry`, labels, employer/occupation terms) with `filter_mode: "people"`; legacy `filter_mode: "tech_people"` still works and now routes through the tech taxonomy. The old tech helpers (`is_explicit_technical_title`, `classify_employer_tech_status`, `is_strong_non_tech_context`) remain as thin wrappers over the generic engine with unchanged signatures and behavior. People-filter results gained `filter_type` and `industry` fields.
   - `analysis_intent.py`: new `people_filter_fallback_intent(question, spec)` builds people_filter intents for any industry/employer/occupation spec; `heuristic_intent` falls back to it before giving up, and both the LLM path and `intent_to_analysis_plan` use it when the model is unsure or asks for clarification on an answerable people query. The alumni-tech fallback remains first for backward compatibility.
   - Added a development debug endpoint `GET /api/debug/classify-row?dataset_id=...&name=...&industry=...` returning per-person `status`, `match_sources`, `confidence`, and `internal_reason`; this info never renders in user-facing tables.
   - Frontend: no UI changes (the renderers were already generic); added a frontend test asserting non-tech industry people results render `visible_columns`, use `total_matches` as the main stat, keep `uncertain_count` separate, hide debug columns, and never show display_limit/total_keyword_hits as the answer.
   - Tests: `backend/tests/test_industry_taxonomies.py` (alias/spec classification for all 12 industries plus employer/occupation queries), `backend/tests/test_industry_matching.py` (tech examples including the Neil-Wusu-at-Spotify case, consulting/banking/finance/healthcare/law/education/media/startup examples, model-threshold behavior, debug helper), and new end-to-end flow tests (consulting, investment banking, employer filter, founders vs startup founders, healthcare/tech overlap, debug endpoint, no debug fields in rows). 167 backend + 15 frontend tests pass.
   - Manual QA against the real alumni spreadsheet: tech query returns 64 confirmed / 11 uncertain with Neil Wusu (Director, Premium Subscription Strategy at Spotify) included and the math department chair / oncology director excluded; consulting (20), banking (16), healthcare (31), `Who works at Spotify?` (employer filter, 1), `Who are founders?` (12), and `Who are startup founders?` (11) all answer without Analysis Plan Errors.

20. Datasets feature: persistent dataset library
   - Backend (`dataset_store.py`): registry entries now include `display_name` (defaults to the original filename on upload). Added `list_datasets()` (newest first; `uploaded_at` has second resolution so ties break by registry insertion order; never loads DataFrames; reports `status: ready|missing` by checking the stored file on disk without crashing), `rename_dataset()` (validates non-empty, trims to 120 chars, persists atomically through the existing tmp-file registry write), `delete_dataset()` (removes metadata and unlinks the uploaded file; clean 404 when unknown; there is no in-memory DataFrame cache to clear since frames are re-read per request), and `dataset_public_metadata()` (tolerates missing fields from older registries).
   - Routes (`dataset_routes.py`): `GET /api/datasets` → `{"datasets": [...], "count": N}`; `PATCH /api/datasets/<id>` with `{"display_name": ...}` → updated metadata (400 empty name, 404 unknown); `DELETE /api/datasets/<id>` → `{"deleted": true, "dataset_id": ...}`. Existing upload/preview/summary/ask routes untouched.
   - Frontend adapter (`api.jsx`, still JSX-free): `Alumni.datasets()`, `Alumni.renameDataset()`, `Alumni.deleteDataset()` with `normalizeDatasetEntry` guarding missing metadata fields; demo mode resolves an empty library.
   - Frontend UI (`app.jsx`, existing layout preserved): `Rail` now navigates — Conversations and Datasets are live views, Saved insights/History remain placeholders, and the ACTIVE DATASET item shows the selected dataset (truncated, full name in tooltip) and opens the library. New `DatasetLibrary` main-panel view lists saved datasets (name, rows/columns, upload date, `File missing` badge, Active chip) with Upload/Rename (`window.prompt`)/Delete (`window.confirm`) actions and an empty state.
   - State: `App` owns `datasets`, `view`, active `ds`, preview, and error/loading states. On boot (API mode) it fetches the library, restores the previously active dataset from localStorage (`alumniActiveDatasetId`) or selects the newest, and loads its preview; failures show a simple error. Selecting a dataset updates the sidebar, persists to localStorage, and refreshes the preview (missing files show a clear preview message). Upload adds + activates the new dataset and refreshes the list. Deleting the active dataset selects the next available or clears to the upload empty state. Renames update the list, the sidebar, and the workspace header; `mergeDatasetPreview` no longer lets the preview's filename overwrite a renamed display name.
   - `Workspace` is keyed by `dataset_id` plus a chat sequence: switching datasets or pressing "New analysis" starts a fresh conversation against the active dataset (New analysis no longer returns to the upload screen now that uploads live in the library). Asks always send the active `dataset_id` and are blocked with a clear chat message if none is selected in API mode.
   - Tests: `backend/tests/test_dataset_library.py` (list fields + newest-first ordering, empty registry, persistence across a simulated app restart, rename validation/persistence/404, delete removes metadata + file with clean repeat-404, missing file → `status: missing` without crashing plus clean preview error, and upload/preview/summary/ask regression). Frontend adapter tests cover list normalization/fallbacks, list failure, demo mode, PATCH body/encoding, rename validation errors, DELETE + 404, and missing-field tolerance. 174 backend + 22 frontend tests pass.
   - Live QA against the real registry (25 datasets): list newest-first with `ready` status, rename round-trip persisted, empty-name 400, unknown-id 404s, preview/summary/ask still working against the active dataset.

21. Strict multi-label people classification (broad retrieval for recall, strict classification for precision)
   - Problem fixed: "What alumni work in consulting?" was including keyword-adjacent rows (Head of Strategy at Hershey, Director of Premium Subscription Strategy at Spotify, Product Manager at Morgan Stanley, Attorney, Judicial Law Clerk). Two root causes: (a) when the intent model confidently returned a generic `find_records` keyword plan, execution ran plain `contains_any` where a keyword hit *was* the final match, and (b) the consulting taxonomy treated bare "advisory"/strategy-ish wording as confirming evidence.
   - Core principle now enforced end-to-end: a broad keyword hit only makes a row a *candidate*; strict query-aware classification decides final inclusion.
   - Added `backend/app/services/people_classifier.py` (`classification_version: multi_label_v1`): query-aware multi-label classifier. Each candidate gets an independent profile — `employer_industry` (array; Spotify is technology *and* media; Morgan Stanley is financial_services), `job_function` (array; consulting_advisory, finance_investing, internal_strategy, product, operations, legal, engineering, data_analytics, ...), and `specialties` (risk_consulting, transaction_advisory, deal_advisory, valuation, restructuring, financial_advisory, investment_banking, wealth_management, ...) — then a per-query verdict: `classification` (`direct_match | adjacent | uncertain | non_match`), `count_as_match`, `confidence`, `internal_reason`. Only `count_as_match=true` rows are counted/displayed; confidence is secondary (a floor of 0.70 for direct matches) and can never promote adjacent rows.
   - Consulting has a dedicated deterministic policy: explicit consultant/consulting titles confirm alone; client-advisory titles (transaction/deal/valuation/restructuring/risk/technology/strategy advisory, transaction services) confirm at professional-services firms (and with unknown employers); financial advisory confirms only at professional-services firms; plausibly client-serving titles at recognized consulting firms (Partner at McKinsey) confirm; legal roles and finance-only roles (IB, PE, portfolio manager, trader, equity research, wealth management) never match a consulting query; strategy/product/operations/management language without consulting context is `adjacent`, never direct. Other industries wrap the existing layered taxonomy engine, so their behavior is unchanged; new industries get policies by adding a function to `_INDUSTRY_POLICIES`.
   - Query-awareness (the query, not just the row, decides): plain "consulting" → direct matches only; "consulting or strategy" → union with `include_functions: [internal_strategy]`; "consulting-adjacent ... too" → `include_adjacent: true` (adjacent rows counted but still labeled adjacent and reported via `adjacent_included_count`); "finance consulting" → `required_industries: [finance]` intersection (Risk Consulting at EY and Transaction Advisory at KPMG count; Management Consultant without finance context and pure finance roles do not); plain "finance" still returns finance roles that are not consulting. `classify_people_question` emits the richer spec (`industries`, `required_industries`, `include_functions`, `include_adjacent`).
   - Consulting taxonomy hardened in `industry_taxonomies.py`: removed bare "advisory" as a self-confirming title keyword (explicit advisory phrases instead), added exclusion keywords (attorney, law clerk, judicial, IB, PE, wealth management, ...), expanded `known_companies` (Guidehouse, Protiviti, RSM, Grant Thornton, IBM Consulting, Mercer, Aon, WTW, Navigant, Huron, TCS, Infosys/Wipro Consulting, A&M, ...), and added a new `retrieval_keywords` list (consultant, consulting, advisory, advisor, strategy, operations, management, transaction, deal, valuation, restructuring, risk, implementation, transformation) used **only** for broad candidate retrieval, never for confirmation.
   - Execution (`analysis_toolkit.py` `_people_filter_result`): industry filters route through the classifier (legacy `tech_people` mode included; tech behavior unchanged via the default taxonomy policy). New debug metadata in metrics/extras — `raw_candidate_count`, `direct_match_count`, `adjacent_count` (excluded adjacents), `adjacent_included_count`, `uncertain_count`, `non_match_count` (within candidates), `classification_version`, `adjacent_included` — while `total_matches` remains the headline count (direct matches only). Per-row multi-label profiles live under the `debug` key only.
   - Intent plumbing (`analysis_intent.py`): even when the LLM confidently returns a broad `find_records` keyword plan, `_plan_find_records` re-checks `classify_people_question` and routes people/industry questions through `filter_mode: "people"` so raw keyword hits can never be presented as final matches.
   - Presentation: operation summaries now read "Alumni matching criteria: N direct matches out of M alumni"; deterministic captions/metrics add "Adjacent not counted" / "Adjacent included" and "N adjacent rows matched broad keywords but were not counted as direct matches." Frontend `api.jsx` mirrors the adjacent metric lines; `internal_reason`/classification fields stay out of visible tables (debug mode only). No UI redesign.
   - Tests: new `backend/tests/test_people_classifier.py` (58 tests: all required consulting inclusions/exclusions, weak-keyword non-counting, finance/consulting intersections, query-behavior specs, multi-label profiles, NaN/missing safety, model-only-for-ambiguous-rows). Extended `test_industry_taxonomies.py` (strict default spec, or-strategy, adjacent, finance-consulting intersection, retrieval-keyword hygiene) and `test_ask_analysis_flow.py` (end-to-end consulting precision dataset with the exact bad examples from the bug report, or-strategy/adjacent/finance/finance-consulting query behaviors, and a regression proving a confident model keyword plan still goes through the strict classifier). 244 backend + 22 frontend tests pass.

22. Saved insights: manually saved answer snapshots tied to a dataset
   - Scope guard: insights are saved **manually** by the user via a Save insight button. Nothing is logged automatically — full chat history is a separate, future feature (the History rail item stays a placeholder), and `/api/ask` never writes insights (locked in by a regression test).
   - Backend store (`backend/app/services/insight_store.py`): metadata-only JSON registry at `backend/data/saved_insights.json` (config key `INSIGHTS_REGISTRY_PATH`; defaults next to `datasets.json` under `DATA_FOLDER` so existing temp-storage test fixtures isolate automatically). Same patterns as `dataset_store`: dict keyed by `insight_id`, atomic tmp-file writes, error hierarchy (`InsightValidationError` 400, `InsightNotFoundError` 404, `InsightRegistryError` 500). Each entry: `insight_id`, `dataset_id`, `dataset_name_snapshot` (display name at save time), `title`, `question`, `answer`, `created_at`, `updated_at`, `tags`, `metadata` (row/column counts captured from the dataset, optional `model`). A saved insight is a snapshot — the answer is never recomputed, and no DataFrame or file contents are stored. `create_insight` validates the dataset exists and question/answer are non-empty, and generates a title from the question (80-char word-boundary clip) when missing; `update_insight` edits title/tags only (dataset_id/question/answer are immutable) and bumps `updated_at`; `insight_public_metadata` tolerates missing fields and reports `dataset_status: ready | deleted` without crashing when the referenced dataset is gone.
   - Routes (`backend/app/routes/insight_routes.py`, registered in `app/__init__.py`): `GET /api/insights` (newest first, ties by insertion order; optional `?dataset_id=` filter) → `{"insights": [...], "count": N}`; `POST /api/insights` → 201 + created insight; `GET /api/insights/<id>`; `PATCH /api/insights/<id>` (`{"title": ..., "tags": [...]}`, 400 on empty title or empty patch); `DELETE /api/insights/<id>` → `{"deleted": true, "insight_id": ...}`; clean JSON 404s throughout.
   - Frontend adapter (`api.jsx`, still JSX-free for the Node tests): `Alumni.insights(datasetId?)`, `Alumni.insight(id)`, `Alumni.saveInsight(payload)` (client-side guards: no dataset / no question / no answer → clear rejections; fills the default title), `Alumni.renameInsight(id, title)`, `Alumni.deleteInsight(id)`; demo mode resolves an empty list and rejects writes. New helpers `defaultInsightTitle(question)` and `insightTextFromAnswer(answer, fallback)` (flattens a structured answer's summary/markdown/metrics + a table row-count line into the plain-text snapshot; never debug fields), plus `normalizeInsightEntry` guarding missing fields.
   - Frontend UI (`app.jsx`, existing layout/sidebar preserved): the "Saved insights" rail item is now a live view (`view: "insights"`). `AiMsg` shows a Save insight button under completed answers only when the message has its originating question, an answer, an active `dataset_id`, and API mode; clicking prompts for a title (default generated from the question), shows Saving…/Saved ✓ states, blocks duplicate saves, and surfaces failures inline without breaking the chat. `Workspace.send` now stamps `question` onto AI messages so saves know their origin. New `InsightsLibrary` main-panel view: All insights / Current dataset only filter (current disabled without an active dataset), cards with title + dataset-name chip + "Dataset deleted" badge + italic question + ~140-char answer preview + saved date + Open/Rename(`window.prompt`)/Delete(`window.confirm`) actions, and an `InsightDetail` view (full question, full pre-wrapped saved answer, created/edited dates, tags, "Open dataset" button that activates the insight's dataset when it still exists). `App` owns `insights`, `insightsLoading`, `insightsError`, `selectedInsightId` (App-level so the open detail survives the Workspace remount when "Open dataset" switches datasets); the list loads when the view opens and updates in place on save/rename/delete; failed deletes keep the insight in the UI.
   - Tests: `backend/tests/test_saved_insights.py` (11 tests: create returns the full snapshot with dataset metadata, generated titles, clean validation errors incl. unknown dataset 404, newest-first listing, dataset_id filtering, single get + 404, PATCH title/tags immutability + updated_at, delete + repeat-404, persistence across a simulated app restart, deleted dataset → `dataset_status: "deleted"` without crashing, and ask-does-not-auto-save). Frontend adapter tests (11 new): list URL/filter/normalization, POST body shape, default title fill, save guards, PATCH/DELETE urls + error surfacing, single get, demo-mode behavior, entry normalization, title generation, answer flattening. Both `app.jsx` and `api.jsx` verified to compile with the same Babel standalone version the app loads (7.29.0). 255 backend + 33 frontend tests pass.

23. Saved insights full-response reopening
   - Backend insight snapshots now optionally persist `response_payload` alongside the existing plain answer text. `insight_public_metadata` returns backward-compatible aliases (`id`, `answer_text`, `dataset_filename`) plus `response_payload` when present, and returns `response_payload: null` for older insights. `POST /api/insights` accepts `answer_text` as an alias for `answer` and stores only JSON-safe response payload objects without recomputing the answer.
   - `frontend/api.jsx` now keeps a display-ready payload on every API ask message: sanitized structured `answer`, `answer_text`, `operation`, `result`, and any `analysis_intent`, `analysis_plan`, or `operation_results` returned by `/api/ask`. `Alumni.saveInsight` sends that payload with the manual saved insight; fetched insights normalize old entries safely and re-sanitize payload tables/metrics using the stored result metadata.
   - `frontend/app.jsx` now uses shared `DatasetResponseView` for both normal chat answers and saved-insight full responses. `InsightDetail` still shows the saved plain-text answer, keeps the existing dataset activation button, and adds an `Open full response` button directly under Saved answer only when a payload exists; clicking expands the rich answer view in-page without activating the dataset or re-querying `/api/ask`.
   - Validation: `./backend/venv/Scripts/python.exe -m pytest -q` passes 256 backend tests; `node --test frontend/tests/*.mjs` passes both frontend test files, covering payload save/fetch normalization and shared-renderer wiring.

Important files:

- `backend/run.py`
- `backend/app/__init__.py`
- `backend/app/routes/upload_routes.py`
- `backend/app/routes/dataset_routes.py`
- `backend/app/routes/insight_routes.py`
- `backend/app/routes/chat_routes.py`
- `backend/app/services/spreadsheet_service.py`
- `backend/app/services/dataset_store.py`
- `backend/app/services/insight_store.py`
- `backend/app/services/analysis_service.py`
- `backend/app/services/analysis_toolkit.py`
- `backend/app/services/analysis_intent.py`
- `backend/app/services/industry_taxonomies.py`
- `backend/app/services/industry_matching.py`
- `backend/app/services/people_classifier.py`
- `backend/app/services/column_resolver.py`
- `backend/app/services/known_tech_companies.json`
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
- `backend/tests/test_industry_taxonomies.py`
- `backend/tests/test_industry_matching.py`
- `backend/tests/test_people_classifier.py`
- `backend/tests/test_dataset_library.py`
- `backend/tests/test_saved_insights.py`
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
