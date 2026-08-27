"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LLM_PROVIDER = "openai"
OPENAI_API_KEY = ""
GEMINI_API_KEY = ""
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.0-flash"
ENV_FILE_PATH: Path | None = None

_PLACEHOLDER_KEYS = {
    "",
    "your_openai_key",
    "your_gemini_key",
    "your_openai_api_key",
    "your_gemini_api_key",
    "changeme",
    "xxx",
    "todo",
}


def _clean_env(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def _is_usable_key(value: str) -> bool:
    cleaned = _clean_env(value)
    return bool(cleaned) and cleaned.lower() not in _PLACEHOLDER_KEYS


def _dotenv_encoding(path: Path) -> str:
    raw = path.read_bytes()
    if not raw:
        return "utf-8"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if b"\x00" in raw[:100]:
        return "utf-16"
    return "utf-8"


def _find_env_file() -> Path | None:
    candidates = (
        PROJECT_ROOT / ".env",
        Path.cwd() / ".env",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _apply_streamlit_secrets() -> None:
    secrets_files = (
        PROJECT_ROOT / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    )
    if not any(path.is_file() for path in secrets_files):
        return
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        return
    names = (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "LLM_PROVIDER",
        "OPENAI_MODEL",
        "GEMINI_MODEL",
        "DATA_DIR",
        "OUTPUT_DIR",
    )
    for name in names:
        try:
            value = secrets.get(name)
        except Exception:
            continue
        if value and not _is_usable_key(os.getenv(name, "")):
            os.environ[name] = str(value)


def reload_env() -> Path | None:
    """Load `.env` from the project folder even if Streamlit was started elsewhere."""
    global DATA_DIR, OUTPUT_DIR, LLM_PROVIDER
    global OPENAI_API_KEY, GEMINI_API_KEY, OPENAI_MODEL, GEMINI_MODEL, ENV_FILE_PATH

    env_path = _find_env_file()
    if env_path is not None:
        try:
            load_dotenv(env_path, override=True, encoding=_dotenv_encoding(env_path))
        except UnicodeDecodeError:
            load_dotenv(env_path, override=True, encoding="utf-8")

    _apply_streamlit_secrets()

    DATA_DIR = Path(_clean_env(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))))
    OUTPUT_DIR = Path(_clean_env(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output"))))
    LLM_PROVIDER = _clean_env(os.getenv("LLM_PROVIDER", "openai")).lower()
    OPENAI_API_KEY = _clean_env(os.getenv("OPENAI_API_KEY", ""))
    GEMINI_API_KEY = _clean_env(os.getenv("GEMINI_API_KEY", ""))
    OPENAI_MODEL = _clean_env(os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    GEMINI_MODEL = _clean_env(os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    ENV_FILE_PATH = env_path
    return env_path


def openai_key_ready() -> bool:
    return _is_usable_key(OPENAI_API_KEY)


def gemini_key_ready() -> bool:
    return _is_usable_key(GEMINI_API_KEY)


reload_env()
