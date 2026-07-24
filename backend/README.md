# AI Spreadsheet Analyst Backend

Flask API for uploading CSV or Excel spreadsheets, previewing/summarizing them with pandas, and asking natural-language questions grounded in safe pandas analysis results.

## Run Locally

1. Create and activate a virtual environment:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your environment file:

```bash
cp .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`.

4. Run the app from the project root:

```bash
./start_app.sh
```

The API and connected frontend run at `http://localhost:5000`. The frontend is
served from `/`, and API routes remain under `/api/*`.

On Windows, run this from the project root:

```bat
start_app.bat
```

## Test Upload

```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@/path/to/your/spreadsheet.csv"
```

The response includes a `dataset_id`, summary metadata, and the first 10 preview rows.

## Test Ask

```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"YOUR_DATASET_ID","question":"Which region had the highest total revenue?"}'
```

The response includes an English answer, the safe operation used when one was detected, and the computed result.

## Dataset Persistence

Uploaded CSV/XLSX files are stored locally in `backend/uploads/`. Dataset metadata is stored in `backend/data/datasets.json`.

These files are local development artifacts and are ignored by git. Existing `dataset_id`s survive backend restarts as long as the uploaded files and metadata JSON remain on disk.

### Schema profiles

Dataset registry entries can contain a compact `schema_profile` with version,
review status, canonical-to-source mappings, confidence/method/evidence,
ignored and unmapped columns, conflicts, and timestamps. Raw sample values are
not persisted in the registry. Confirmed mappings and high-confidence saved
inferences enter the existing intent/plan/whitelisted-operation pipeline
without renaming columns or mutating the uploaded file.

Schema endpoints:

```text
GET  /api/datasets/<dataset_id>/schema
PUT  /api/datasets/<dataset_id>/schema
POST /api/datasets/<dataset_id>/schema/infer
```

`GET` lazily infers and atomically persists a profile for older datasets.
`PUT` accepts a complete validated mapping update:

```json
{
  "status": "confirmed",
  "mappings": {
    "employer": {"source_columns": ["BUSINESS_NAME_1"]},
    "email": {
      "source_columns": ["CONSTITUENT_PREFERRED_EMAIL", "ALTERNATE_EMAIL"]
    }
  },
  "ignored_columns": ["INTERNAL_NOTES"]
}
```

`POST .../schema/infer` accepts `{"reset_confirmed": false, "use_model": true}`.
Confirmed mappings are preserved unless reset is explicitly requested. Model
suggestions are optional and receive only capped column metadata and safe sample
values; deterministic inference works without `OPENAI_API_KEY`.

## Exact predicates

Natural-language constraints are normalized into typed predicates and executed
only by whitelisted pandas operations. Supported operators are:

```text
exists, missing, equals, not_equals, contains, not_contains
email_domain_in, email_domain_not_in
greater_than, greater_than_or_equal, less_than, less_than_or_equal, between
in, not_in
date_before, date_after, date_between
starts_with, ends_with
```

Numeric parsing accepts native numbers and common currency forms such as
`5,000`, `$5,000`, and `$5,000.00`; invalid text and blanks are not zero.
Dates accept native pandas/Python values, ISO and common U.S. strings, and
Excel serial dates. They normalize to calendar dates before comparison.
`before` and `after` are strict. Numeric/date `between` includes both endpoints
unless explicit inclusivity flags say otherwise. Relative windows such as
“within the past 90 days” resolve once per request and record the ISO boundary
in the intent-filter trace.

Membership and prefix/suffix checks are case-insensitive and whitespace
normalized. Missing or unparseable values do not satisfy comparisons,
`not_in`, or other negative predicates by default.

Legacy `{"logic":"and","predicates":[...]}` roots remain accepted. Grouped
queries use recursive `clauses` with only `and`/`or`, at most three levels,
twelve leaf predicates, and twelve clauses per group. Exact predicates and
fuzzy people classification run independently over the original rows and are
then composed, so questions such as “Find tech alumni who graduated after
2020 and have a non-Cornell email” preserve every clause.

## Automated Tests

Install dependencies and run pytest:

```powershell
cd backend
pip install -r requirements.txt
python -m pytest -v
```

The test suite in `backend/tests/test_api.py` creates temporary CSV/XLSX files
in memory, uploads them through the Flask API, and verifies preview, summary,
ask, safety, and dataset-isolation behavior. It does not depend on existing
files in `backend/uploads` and does not require a real OpenAI API key.

## Running Evals

Run the larger answer-quality eval harness from `backend/`:

```bash
python -m evals.run_evals
```

The runner uploads a sanitized copy of `evals/datasets/synthetic_alumni_500.csv`
through the Flask test client, removes `expected_*` and `eval_*` gold-label
columns from the app-facing CSV, asks the cases in `evals/cases.jsonl`, and
writes `evals/results/latest.json` plus `evals/results/latest.md`.

Modes:

```bash
python -m evals.run_evals --mode offline
python -m evals.run_evals --mode hybrid
python -m evals.run_evals --mode classifier-live
python -m evals.run_evals --mode smoke-live
```

`offline` disables OpenAI for every case. Other modes still follow per-case
execution metadata, so deterministic cases can disallow model calls while broad
product or direct-classifier cases can allow or require them. Reports include
model-call tracing, answer source, scoring source, and failure categories.

Schema-aware eval cases may define `schema_mapping`. Run the focused opaque
header cases with:

```bash
python -m evals.run_evals --mode offline \
  --dataset evals/datasets/opaque_schema_alumni.csv \
  --app-view evals/generated/opaque_schema_alumni_app_view.csv \
  --cases evals/schema_cases.jsonl \
  --category schema_mapping
```

Run the expanded exact-predicate cases with:

```bash
python -m evals.run_evals --mode offline \
  --dataset evals/datasets/expanded_predicates.csv \
  --app-view evals/generated/expanded_predicates_app_view.csv \
  --cases evals/expanded_predicate_cases.jsonl \
  --category expanded_predicates
```
