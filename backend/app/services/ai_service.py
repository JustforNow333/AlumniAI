import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key and OpenAI else None
