from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env once when module is imported.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    ensure_openai_api_key()
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, max_retries=5)


def ensure_openai_api_key() -> str:
    """Return OPENAI_API_KEY from environment (loaded by python-dotenv)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    raise ValueError(
        "OPENAI_API_KEY is not set. Add OPENAI_API_KEY=... in .env (python-dotenv) or export it."
    )