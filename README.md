# AI Spreadsheet Analyst

Upload a CSV or Excel spreadsheet, preview it, and ask natural-language questions backed by safe pandas analysis operations.

## Start The App

From the project root:

```bash
./start_app.sh
```

On Windows, double-click `start_app.bat` or run:

```bat
start_app.bat
```

Then open `http://localhost:5000`.

The Flask backend serves the frontend from `/`, so one command starts the backend, frontend, and app experience together. API routes remain under `/api/*`.

## Setup

Install backend dependencies first if the virtualenv is not already set up:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`OPENAI_API_KEY` in `backend/.env` is optional. Without it, the app still runs safe built-in analysis operations and returns deterministic fallback answers.
