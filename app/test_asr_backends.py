from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from asr_backends import (
    JAPANESE_BACKEND,
    SENSEVOICE_BACKEND,
    backend_label,
    find_reazonspeech_files,
)


class AsrBackendsTest(unittest.TestCase):
    def test_japanese_uses_dedicated_backend(self) -> None:
        self.assertEqual(backend_label("ja"), JAPANESE_BACKEND)
        for language in ("auto", "zh", "en", "ko", "yue"):
            self.assertEqual(backend_label(language), SENSEVOICE_BACKEND)

    def test_reazonspeech_file_set_is_validated(self) -> None:
        names = (
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        )
        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            with self.assertRaises(FileNotFoundError):
                find_reazonspeech_files(model_dir)
            for name in names:
                (model_dir / name).write_bytes(b"model")
            self.assertEqual(
                tuple(path.name for path in find_reazonspeech_files(model_dir)),
                names,
            )


if __name__ == "__main__":
    unittest.main()
