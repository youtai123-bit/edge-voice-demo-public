"""Language selection helpers for the multilingual SenseVoice demo."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


LANGUAGE_LABELS = {
    "auto": "自动识别",
    "zh": "中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "yue": "粤语",
}


def normalize_language(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("language must be a string")
    language = value.strip().lower()
    if language not in LANGUAGE_LABELS:
        choices = ", ".join(LANGUAGE_LABELS)
        raise ValueError(f"unsupported language {value!r}; choose one of: {choices}")
    return language


def load_language(path: Path, fallback: str = "auto") -> str:
    fallback = normalize_language(fallback)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return normalize_language(payload.get("language"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def save_language(path: Path, language: str) -> None:
    language = normalize_language(language)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"language": language}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
