"""Persistent user corrections and dynamic hotwords for edge ASR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any

from candidate_selection import normalize_text
from language_selection import normalize_language


MAX_ENTRIES = 200
MAX_TEXT_LENGTH = 300
MAX_HOTWORD_LENGTH = 64


@dataclass
class CorrectionEntry:
    id: str
    language: str
    observed: str
    corrected: str
    hotword: str
    created_at: str
    updates: int = 1


def _clean_text(value: Any, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    if len(cleaned) > max_length:
        raise ValueError(f"{field_name} is too long")
    if any(ord(character) < 32 and character not in "\t\n" for character in cleaned):
        raise ValueError(f"{field_name} contains control characters")
    return cleaned


class PersonalizationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.entries: list[CorrectionEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            loaded: list[CorrectionEntry] = []
            for item in payload.get("entries", []):
                loaded.append(
                    CorrectionEntry(
                        id=str(item["id"]),
                        language=normalize_language(item["language"]),
                        observed=str(item["observed"]),
                        corrected=str(item["corrected"]),
                        hotword=str(item.get("hotword", "")),
                        created_at=str(item["created_at"]),
                        updates=max(1, int(item.get("updates", 1))),
                    )
                )
            self.entries = loaded[-MAX_ENTRIES:]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.entries = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "version": 1,
            "entries": [asdict(entry) for entry in self.entries],
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)

    def add_correction(
        self,
        language: str,
        observed: str,
        corrected: str,
        hotword: str | None = None,
    ) -> CorrectionEntry:
        language = normalize_language(language)
        observed = _clean_text(observed, "observed", MAX_TEXT_LENGTH)
        corrected = _clean_text(corrected, "corrected", MAX_TEXT_LENGTH)
        if hotword is None:
            hotword = ""
        elif not isinstance(hotword, str):
            raise ValueError("hotword must be a string")
        else:
            hotword = hotword.strip()
        if len(hotword) > MAX_HOTWORD_LENGTH:
            raise ValueError("hotword is too long")
        if not hotword:
            normalized = normalize_text(corrected)
            hotword = corrected if 1 < len(normalized) <= 24 else ""

        observed_key = normalize_text(observed)
        if not observed_key:
            raise ValueError("observed has no recognizable text")
        with self.lock:
            for entry in reversed(self.entries):
                if (
                    entry.language == language
                    and normalize_text(entry.observed) == observed_key
                ):
                    entry.observed = observed
                    entry.corrected = corrected
                    entry.hotword = hotword
                    entry.updates += 1
                    self._save()
                    return CorrectionEntry(**asdict(entry))

            entry = CorrectionEntry(
                id=str(time.time_ns()),
                language=language,
                observed=observed,
                corrected=corrected,
                hotword=hotword,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            self.entries.append(entry)
            self.entries = self.entries[-MAX_ENTRIES:]
            self._save()
            return CorrectionEntry(**asdict(entry))

    def apply(self, language: str, text: str) -> tuple[str, CorrectionEntry | None]:
        language = normalize_language(language)
        key = normalize_text(text)
        if not key:
            return text, None
        with self.lock:
            for entry in reversed(self.entries):
                if entry.language != language:
                    continue
                if normalize_text(entry.corrected) == key:
                    return text, None
                if normalize_text(entry.observed) == key:
                    return entry.corrected, CorrectionEntry(**asdict(entry))
        return text, None

    def hotwords(self, language: str, limit: int = 32) -> list[str]:
        language = normalize_language(language)
        with self.lock:
            result: list[str] = []
            seen: set[str] = set()
            for entry in reversed(self.entries):
                if entry.language != language or not entry.hotword:
                    continue
                key = normalize_text(entry.hotword)
                if not key or key in seen:
                    continue
                seen.add(key)
                result.append(entry.hotword)
                if len(result) >= limit:
                    break
            return result

    def hotword_stream(self, language: str, limit: int = 32) -> str | None:
        encoded = []
        for hotword in self.hotwords(language, limit):
            normalized = normalize_text(hotword)
            if normalized:
                encoded.append(" ".join(normalized))
        return "/".join(encoded) or None

    def snapshot(self, language: str) -> dict[str, Any]:
        language = normalize_language(language)
        with self.lock:
            language_entries = [
                asdict(entry)
                for entry in reversed(self.entries)
                if entry.language == language
            ]
            recent = language_entries[:20]
        hotwords = self.hotwords(language)
        return {
            "language": language,
            "count": len(language_entries),
            "hotwords": hotwords,
            "recent": recent,
            "native_hotword_bias": language == "ja",
        }

    def clear(self, language: str) -> int:
        language = normalize_language(language)
        with self.lock:
            before = len(self.entries)
            self.entries = [entry for entry in self.entries if entry.language != language]
            removed = before - len(self.entries)
            if removed:
                self._save()
            return removed
