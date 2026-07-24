# AlumniAI

AlumniAI analyzes CSV and Excel alumni exports with safe, whitelisted pandas operations. It includes persistent datasets, schema mapping for unfamiliar headers, natural-language queries, structured answers, saved insights, and query history.

Exact filters cover missing values, equality, text and email-domain checks, numeric comparisons and ranges, set membership, dates, and string prefixes/suffixes. They can be grouped with bounded `and`/`or` logic and combined with fuzzy alumni classification.

## Run

From the project root:

```bash
./start_app.sh
```

On Windows, run `start_app.bat`. Open `http://localhost:5000`.

Install dependencies with:

```bash
pip install -r backend/requirements.txt
```

`OPENAI_API_KEY` in `backend/.env` is optional. Without it, deterministic intent inference and answer formatting remain available.

## Data and schema mapping

Uploads are stored in `backend/uploads/`; metadata is stored in `backend/data/datasets.json`. Both are local, gitignored development artifacts. Dataset IDs persist across restarts while these files remain available.

Each upload receives a deterministic proposal mapping its original columns to canonical AlumniAI fields. Users can confirm, correct, ignore, or postpone mappings. Confirmed mappings take priority during later queries without renaming or modifying the source file. Older datasets are inferred lazily when schema review is opened.

## Validation

Use the checked-in Windows virtual environment when available:

```bash
./backend/venv/Scripts/python.exe -m pytest -q
node --test frontend/tests/*.mjs
python3 -m compileall -q backend/app backend/evals backend/tests
cd backend && ./venv/Scripts/python.exe -m evals.run_evals --mode offline
```

An active Python environment may be substituted for `backend/venv/Scripts/python.exe`.
