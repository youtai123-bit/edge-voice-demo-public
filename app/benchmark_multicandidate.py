#!/usr/bin/env python3
"""Run one saved WAV through all three candidate paths on the target device."""

from __future__ import annotations

import argparse
import json
import resource
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from asr_backends import load_language_recognizer
from fastenhancer_onnx import FastEnhancerOnnx
from candidate_selection import select_candidate
from language_selection import LANGUAGE_LABELS
from voice_demo import SAMPLE_RATE, make_denoiser
from voice_demo_multicandidate import (
    CANDIDATE_LABELS,
    CandidateWorker,
)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as file:
        if file.getsampwidth() != 2 or file.getnchannels() != 1:
            raise ValueError("benchmark WAV must be mono PCM16")
        if file.getframerate() != SAMPLE_RATE:
            raise ValueError(f"benchmark WAV must be {SAMPLE_RATE} Hz")
        return np.frombuffer(file.readframes(file.getnframes()), dtype="<i2").astype(
            np.float32
        ) / 32768.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtcrn", type=Path, required=True)
    parser.add_argument("--fastenhancer", type=Path, required=True)
    parser.add_argument("--sensevoice-dir", type=Path, required=True)
    parser.add_argument("--japanese-model-dir", type=Path, required=True)
    parser.add_argument("--language", choices=tuple(LANGUAGE_LABELS), default="auto")
    parser.add_argument("--wav", type=Path, required=True)
    args = parser.parse_args()
    samples = read_wav(args.wav)
    recognizers = {}
    backend = ""
    for name in CANDIDATE_LABELS:
        loaded = load_language_recognizer(
            args.language, args.sensevoice_dir, args.japanese_model_dir, 1
        )
        recognizers[name] = loaded.recognizer
        backend = loaded.backend
    del loaded
    worker = CandidateWorker(
        recognizers=recognizers,
        gtcrn=make_denoiser(args.gtcrn, 1),
        fastenhancer=FastEnhancerOnnx(args.fastenhancer, num_threads=1),
        work_queue=SimpleNamespace(),
        state=SimpleNamespace(),
        log_dir=Path("."),
        save_utterances=False,
        stop_event=SimpleNamespace(),
        sensevoice_dir=args.sensevoice_dir,
        japanese_model_dir=args.japanese_model_dir,
        asr_threads=1,
        language=args.language,
        asr_backend=backend,
        language_state_path=Path("language.json"),
    )
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(worker._run_one, name, samples): name
            for name in CANDIDATE_LABELS
        }
        runs = {}
        for future in as_completed(futures):
            run = future.result()
            runs[run.name] = run
    wall_s = time.perf_counter() - started
    decision = select_candidate(runs[name].evidence() for name in CANDIDATE_LABELS)
    report = {
        "wav": str(args.wav),
        "language": args.language,
        "asr_backend": backend,
        "audio_s": round(len(samples) / SAMPLE_RATE, 3),
        "parallel_wall_s": round(wall_s, 3),
        "parallel_rtf": round(wall_s * SAMPLE_RATE / len(samples), 4),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "candidates": {
            name: runs[name].public_dict() for name in CANDIDATE_LABELS
        },
        "decision": decision.to_dict(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
