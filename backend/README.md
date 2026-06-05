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
