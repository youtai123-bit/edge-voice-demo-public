import tempfile
import unittest
from pathlib import Path

from language_selection import (
    LANGUAGE_LABELS,
    load_language,
    normalize_language,
    save_language,
)


class LanguageSelectionTest(unittest.TestCase):
    def test_all_documented_languages_are_accepted(self) -> None:
        for language in LANGUAGE_LABELS:
            self.assertEqual(normalize_language(language.upper()), language)

    def test_invalid_language_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_language("fr")
        with self.assertRaises(ValueError):
            normalize_language(None)

    def test_missing_or_corrupt_state_uses_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "language.json"
            self.assertEqual(load_language(path, "ja"), "ja")
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(load_language(path, "en"), "en")

    def test_saved_language_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "language.json"
            save_language(path, "YUE")
            self.assertEqual(load_language(path), "yue")


if __name__ == "__main__":
    unittest.main()
