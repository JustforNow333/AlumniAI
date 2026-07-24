# Alumni AI — frontend

A self-contained frontend for the AI Spreadsheet Analyst backend. No build step:
just open `index.html`, or serve the folder.

## Files
- `index.html` — entry point + backend config block
- `theme.css` — design tokens (light/dark) + all styles
- `kit.jsx` — icons, brand, and shared type pills
- `engine.jsx` — local CSV parse/profile/analysis (demo mode only)
- `sample-data.jsx` — the demo dataset
- `api.jsx` — talks to the Flask backend + maps responses to the UI
- `app.jsx` — the app (upload, workspace, chat, structured answer renderers)

## Run connected to Flask (default)
Start the backend:

```bash
cd backend
python run.py
```

Then open `http://localhost:5000`. Flask serves this frontend from `/`, and
the frontend posts uploads and questions to the same backend at
same-origin `/api/*` routes.

## Run as a standalone demo
In `index.html`, set:

```js
window.ALUMNI_CONFIG = { useApi: false, apiBase: "" };
```

Then open `index.html`. Demo mode runs in the browser and supports CSV-style
text files only.

## Serve the frontend separately
1. Point `apiBase` at the backend origin:
   ```js
   window.ALUMNI_CONFIG = { useApi: true, apiBase: "http://localhost:5000" };
   ```

2. Serve this folder over HTTP, for example `python3 -m http.server 8000` from
   `frontend/`. The backend CORS config allows common local dev origins.

## Test the API adapter
Run the dependency-free Node tests from the project root:

```bash
node --test frontend/tests/api-preview.test.mjs
```

## What the API layer expects
- `POST /api/upload` → `{ dataset_id, filename, summary }` where `summary` has
  `rows, columns, column_names, column_types, missing_values, preview`.
- `GET /api/datasets/<dataset_id>/preview` → `{ dataset_id, filename,
  row_count, column_count, missing_count, columns, data_types, missing_values,
  rows }`, plus legacy `{ column_names, preview }` compatibility fields.
- `GET /api/datasets/<dataset_id>/summary` → dataset summary metadata.
- `GET /api/datasets/<dataset_id>/schema` → canonical definitions, source
  metadata, inferred/saved mappings, conflicts, and review status.
- `PUT /api/datasets/<dataset_id>/schema` → validates and saves the complete
  reviewed mapping.
- `POST /api/datasets/<dataset_id>/schema/infer` → re-runs bounded inference
  while preserving confirmed mappings unless explicitly reset.
- `POST /api/ask` `{ dataset_id, question }` → `{ answer, answer_text,
  operation, result }`. `answer` is structured as `{ title, summary, blocks,
  followups }`; supported block types are `markdown`, `table`, `metrics`, and
  `ranked_list`.
- `api.jsx → adaptAnswer()` normalizes structured answers and safely wraps
  legacy plain-text answers in a markdown block.

After a new API-mode upload, `app.jsx` opens schema review when the profile is
not confirmed. Users can save, re-run detection, mark source columns ignored or
unmapped, or skip and continue chatting. Dataset-library badges show
`Schema ready`, `Needs schema review`, `Not analyzed`, or `File missing`, and
the library provides a **Review schema** action for returning later.

## Going to production
Drop the Babel CDN + in-browser transform by moving the `.jsx` into a Vite/CRA
project (`npm i react react-dom`, import components normally). `engine.jsx` and
`sample-data.jsx` are only used by explicit non-API demo mode before a real
upload.
