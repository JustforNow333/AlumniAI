#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"

if [[ -x "$BACKEND_DIR/venv/Scripts/python.exe" ]]; then
  PYTHON="$BACKEND_DIR/venv/Scripts/python.exe"
elif [[ -x "$BACKEND_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON="$BACKEND_DIR/.venv/Scripts/python.exe"
elif [[ -x "$BACKEND_DIR/.venv/bin/python" ]]; then
  PYTHON="$BACKEND_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "No Python interpreter found. Create a backend virtualenv and install backend/requirements.txt." >&2
  exit 1
fi

echo "Starting AI Spreadsheet Analyst at http://localhost:5000"
echo "Using Python: $PYTHON"
cd "$BACKEND_DIR"
export FLASK_DEBUG="${FLASK_DEBUG:-0}"
exec "$PYTHON" run.py
