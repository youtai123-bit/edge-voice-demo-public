from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from personalization import PersonalizationStore


class PersonalizationStoreTest(unittest.TestCase):
    def test_correction_is_persistent_and_language_scoped(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "personalization.json"
            store = PersonalizationStore(path)
            store.add_correction("ja", "南京大雪", "南京大学", "南京大学")
            self.assertEqual(store.apply("ja", "南京大雪")[0], "南京大学")
            self.assertEqual(store.apply("zh", "南京大雪")[0], "南京大雪")
            restored = PersonalizationStore(path)
            self.assertEqual(restored.apply("ja", "南京大雪")[0], "南京大学")

    def test_japanese_hotword_is_encoded_as_char_tokens(self) -> None:
        with TemporaryDirectory() as directory:
            store = PersonalizationStore(Path(directory) / "state.json")
            store.add_correction("ja", "東京大雪", "東京大学", "東京大学")
            store.add_correction("ja", "おーぷんえーあい", "オープンAI", "オープンAI")
            self.assertEqual(
                store.hotword_stream("ja"),
                "オ ー プ ン A I/東 京 大 学",
            )

    def test_duplicate_observation_updates_in_place(self) -> None:
        with TemporaryDirectory() as directory:
            store = PersonalizationStore(Path(directory) / "state.json")
            store.add_correction("en", "open eye", "OpenAI", "OpenAI")
            store.add_correction("en", "open eye", "OpenAI Codex", "Codex")
            snapshot = store.snapshot("en")
            self.assertEqual(snapshot["count"], 1)
            self.assertEqual(snapshot["hotwords"], ["Codex"])
            self.assertEqual(snapshot["recent"][0]["updates"], 2)

    def test_clear_only_removes_requested_language(self) -> None:
        with TemporaryDirectory() as directory:
            store = PersonalizationStore(Path(directory) / "state.json")
            store.add_correction("ja", "誤り", "正しい", "正しい")
            store.add_correction("zh", "南经大学", "南京大学", "南京大学")
            self.assertEqual(store.clear("ja"), 1)
            self.assertEqual(store.snapshot("ja")["count"], 0)
            self.assertEqual(store.snapshot("zh")["count"], 1)

    def test_rejects_empty_or_oversized_values(self) -> None:
        with TemporaryDirectory() as directory:
            store = PersonalizationStore(Path(directory) / "state.json")
            with self.assertRaises(ValueError):
                store.add_correction("ja", "", "正しい")
            with self.assertRaises(ValueError):
                store.add_correction("ja", "誤り", "正しい", "長" * 65)
            with self.assertRaises(ValueError):
                store.add_correction("ja", "誤り", "正しい", 123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
