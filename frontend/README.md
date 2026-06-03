# Alumni AI — frontend

A self-contained frontend for the AI Spreadsheet Analyst backend. No build step:
just open `index.html`, or serve the folder.

## Files
- `index.html` — entry point + backend config block
- `theme.css` — design tokens (light/dark) + all styles
- `kit.jsx` — icons, brand, shared UI bits
- `engine.jsx` — local CSV parse/profile/analysis (demo mode only)
- `sample-data.jsx` — the demo dataset
- `api.jsx` — talks to the Flask backend + maps responses to the UI
- `app.jsx` — the app (upload, workspace, chat, result renderers)

## Run connected to Flask (default)
Start the backend:

```bash
cd backend
python run.py
```

Then open `http://localhost:5000`. Flask serves this frontend from `/`, and
the frontend posts uploads and questions to the same backend at
`http://localhost:5000/api/*`.

## Run as a standalone demo
In `index.html`, set:

```js
window.ALUMNI_CONFIG = { useApi: false, apiBase: "http://localhost:5000" };
```

Then open `index.html`. Demo mode runs in the browser and supports CSV-style
text files only.

## Serve the frontend separately
1. Keep `index.html` set to:
   ```js
   window.ALUMNI_CONFIG = { useApi: true, apiBase: "http://localhost:5000" };
   ```

2. Serve this folder over HTTP, for example `python3 -m http.server 8000` from
   `frontend/`. The backend CORS config allows common local dev origins.

## What the API layer expects
- `POST /api/upload` → `{ dataset_id, filename, summary }` where `summary` has
  `rows, columns, column_names, column_types, missing_values, preview`.
- `POST /api/ask` `{ dataset_id, question }` → `{ answer, operation, result }`.
  `api.jsx → adaptAnswer()` maps each `operation.type`
  (`group_by_aggregate`, `top_rows`, `correlation`, `summarize_column`,
  `summarize_dataframe`, `analysis_error`) to a result renderer. If your backend
  changes a result shape, that one function is the only place to update.

## Going to production
Drop the Babel CDN + in-browser transform by moving the `.jsx` into a Vite/CRA
project (`npm i react react-dom`, import components normally). `engine.jsx` and
`sample-data.jsx` can be removed once you rely on the backend.
