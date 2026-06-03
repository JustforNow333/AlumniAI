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
   - Retrieves the in-memory DataFrame by `dataset_id`.
   - Builds OpenAI context from dataset metadata, first 10 rows, and pandas summaries.
   - Uses the OpenAI Responses API when `OPENAI_API_KEY` is configured.

4. Safe analysis functions
   - Added fixed pandas operations in `analysis_service.py`.
   - No `eval`, `exec`, arbitrary Python execution, vector database, or generated-code execution.
   - Implemented summary, column summary, group-by aggregation, top rows, basic filtering, and correlation.
   - Added a small rule-based intent layer for common spreadsheet questions before calling OpenAI.

5. Frontend connection
   - Uploaded frontend lives in `frontend/` and is a no-build React 18 + Babel static app.
   - `frontend/index.html` now defaults to API mode with `window.ALUMNI_CONFIG = { useApi: true, apiBase: "http://localhost:5000" }`.
   - `frontend/api.jsx` calls `POST /api/upload` and `POST /api/ask`, then adapts backend operation payloads to the UI result renderers.
   - Upload UI accepts `.csv` and `.xlsx` when API mode is enabled; local demo fallback only supports CSV-style text files.
   - Flask serves the frontend from `GET /` and static frontend assets by filename, so running `python backend/run.py` exposes the app and API from `http://localhost:5000`.
   - CORS still allows common separate frontend dev origins including `localhost:3000`, `localhost:5173`, and `localhost:8000`.

6. OpenAI fallback behavior
   - `OPENAI_API_KEY` is optional for backend startup.
   - When no key is configured, `/api/ask` still returns safe analysis results with a concise deterministic fallback answer.
   - If OpenAI is configured but the API call fails, the backend still returns the computed safe operation/result and includes the failure in the answer text.

Important files:

- `backend/run.py`
- `backend/app/__init__.py`
- `backend/app/routes/upload_routes.py`
- `backend/app/routes/dataset_routes.py`
- `backend/app/routes/chat_routes.py`
- `backend/app/services/spreadsheet_service.py`
- `backend/app/services/analysis_service.py`
- `backend/app/services/ai_service.py`
- `backend/app/utils/file_utils.py`
- `backend/README.md`
- `frontend/index.html`
- `frontend/app.jsx`
- `frontend/api.jsx`
- `frontend/engine.jsx`
- `frontend/kit.jsx`
- `frontend/sample-data.jsx`
- `frontend/theme.css`
- `frontend/README.md`

Run notes:

- Start the full app from the project root with `./start_app.sh` or `start_app.bat`.
- Open `http://localhost:5000` to use the connected frontend.
- Flask serves the frontend from `/`, so one command starts the backend, frontend, and app experience together.
- The frontend can also be served separately, but keep `apiBase` pointed at `http://localhost:5000`.
