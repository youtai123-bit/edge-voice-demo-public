"""Language-aware ASR backend selection for the Raspberry Pi demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SENSEVOICE_BACKEND = "SenseVoice"
JAPANESE_BACKEND = "ReazonSpeech（日语专用）"


@dataclass(frozen=True)
class LoadedRecognizer:
    recognizer: Any
    backend: str
    model_files: tuple[Path, ...]


def backend_label(language: str) -> str:
    return JAPANESE_BACKEND if language == "ja" else SENSEVOICE_BACKEND


def find_reazonspeech_files(model_dir: Path) -> tuple[Path, Path, Path, Path]:
    files = (
        model_dir / "encoder-epoch-99-avg-1.int8.onnx",
        model_dir / "decoder-epoch-99-avg-1.int8.onnx",
        model_dir / "joiner-epoch-99-avg-1.int8.onnx",
        model_dir / "tokens.txt",
    )
    missing = [str(path) for path in files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("missing Japanese ASR model files: " + ", ".join(missing))
    return files


def load_language_recognizer(
    language: str,
    sensevoice_dir: Path,
    japanese_model_dir: Path,
    num_threads: int,
) -> LoadedRecognizer:
    if language == "ja":
        import sherpa_onnx

        encoder, decoder, joiner, tokens = find_reazonspeech_files(
            japanese_model_dir
        )
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="modified_beam_search",
            max_active_paths=4,
            hotwords_score=1.5,
            modeling_unit="cjkchar",
            provider="cpu",
        )
        return LoadedRecognizer(
            recognizer=recognizer,
            backend=JAPANESE_BACKEND,
            model_files=(encoder, decoder, joiner, tokens),
        )

    from voice_demo import make_recognizer

    recognizer, model, tokens = make_recognizer(
        sensevoice_dir, num_threads, language
    )
    return LoadedRecognizer(
        recognizer=recognizer,
        backend=SENSEVOICE_BACKEND,
        model_files=(model, tokens),
    )
