#!/usr/bin/env python3
"""Raw/GTCRN/FastEnhancer candidate competition for Raspberry Pi 5.

The microphone is segmented on the raw INPUT 1 signal.  Each completed raw
utterance is then processed by three independent candidate paths and decoded by
SenseVoice.  The program never opens an ALSA playback device.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np

from asr_backends import JAPANESE_BACKEND, backend_label, load_language_recognizer
from candidate_selection import (
    CandidateEvidence,
    acoustic_safety_score,
    normalize_text,
    select_candidate,
)
from fastenhancer_onnx import FastEnhancerOnnx
from language_selection import (
    LANGUAGE_LABELS,
    load_language,
    normalize_language,
    save_language,
)
from personalization import PersonalizationStore
from partial_preview import PartialPreviewWorker, PrefixPreview, PreviewJob
from transcript_archive import recent_transcripts
from voice_demo import (
    CAPTURE_CHANNELS,
    SAMPLE_RATE,
    make_denoiser,
    make_vad,
    save_pcm16_wav,
    utc_timestamp,
)


CANDIDATE_LABELS = {
    "raw": "原始语音",
    "gtcrn": "GTCRN",
    "fastenhancer": "FastEnhancer-B",
}


def release_native_memory() -> None:
    """Release unreachable Python objects and return free glibc pages to Linux."""
    gc.collect()
    try:
        malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim")
    except (AttributeError, OSError):
        return
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


@dataclass
class MultiCandidateState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    partial_text: str = ""
    partial_active: bool = False
    partial_generation: int = 0
    partial_language: str = ""
    partial_decode_count: int = 0
    phase: str = "starting"
    message: str = "正在加载三路模型"
    speech_active: bool = False
    level_dbfs: float = -120.0
    latest_text: str = ""
    latest_latency_ms: int | None = None
    latest_route: str = ""
    latest_consensus_score: float | None = None
    latest_reliability: str = ""
    latest_reason: str = ""
    latest_candidates: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    utterance_count: int = 0
    dropped_utterances: int = 0
    error: str = ""
    language: str = "auto"
    language_requested: str = "auto"
    language_switching: bool = False
    asr_backend: str = "SenseVoice"
    personalization_count: int = 0
    personal_hotwords: list[str] = field(default_factory=list)
    native_hotword_bias: bool = False

    def update(self, **values: Any) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)

    def append_result(self, item: dict[str, Any]) -> None:
        with self.lock:
            self.latest_text = item["text"]
            self.latest_latency_ms = item["latency_ms"]
            self.latest_route = item["selected_name"]
            self.latest_consensus_score = item["consensus_score"]
            self.latest_reliability = item["reliability"]
            self.latest_reason = item["reason"]
            self.latest_candidates = item["candidates"]
            self.utterance_count += 1
            self.transcript.append(item)
            self.transcript = self.transcript[-100:]
            self.phase = "listening"
            self.message = "请继续说话"
            self.error = ""

    def begin_preview(self, language: str) -> int:
        with self.lock:
            self.partial_generation += 1
            self.partial_text, self.partial_active, self.partial_language = "", True, language
            return self.partial_generation

    def preview_is_current(self, job: PreviewJob) -> bool:
        with self.lock:
            return (self.partial_active and self.partial_generation == job.generation
                    and self.language == job.language and not self.language_switching)

    def publish_preview(self, job: PreviewJob, text: str) -> None:
        with self.lock:
            if (self.partial_active and self.partial_generation == job.generation
                    and self.language == job.language and not self.language_switching):
                self.partial_text = text

    def finish_preview(self, generation: int | None = None) -> None:
        with self.lock:
            if generation is None or generation == self.partial_generation:
                self.partial_text, self.partial_active = "", False

    def _correction_target_unlocked(
        self,
        utterance_index: int,
        language: str,
        observed: str,
    ) -> dict[str, Any]:
        target = next(
            (
                item
                for item in reversed(self.transcript)
                if int(item.get("index", -1)) == utterance_index
            ),
            None,
        )
        if target is None:
            raise ValueError("utterance is no longer available")
        if target.get("language") != language:
            raise ValueError("language changed before correction was saved")
        if normalize_text(str(target.get("text", ""))) != normalize_text(observed):
            raise ValueError("transcript changed before correction was saved")
        return target

    def validate_user_correction(
        self, utterance_index: int, language: str, observed: str
    ) -> None:
        with self.lock:
            self._correction_target_unlocked(utterance_index, language, observed)

    def apply_user_correction(
        self,
        utterance_index: int,
        language: str,
        observed: str,
        corrected: str,
        hotword: str,
    ) -> None:
        with self.lock:
            target = self._correction_target_unlocked(
                utterance_index, language, observed
            )
            target["asr_text_before_correction"] = target["text"]
            target["text"] = corrected
            target["user_corrected"] = True
            target["learned_hotword"] = hotword
            if self.transcript and self.transcript[-1] is target:
                self.latest_text = corrected
                self.latest_reason = (
                    f"用户已纠正并学习：{hotword}"
                    if hotword
                    else "用户已纠正，已写入个性化记忆"
                )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "phase": self.phase,
                "session_id": self.session_id,
                "ui_version": "partial-preview-v1",
                "partial_text": self.partial_text,
                "partial_active": self.partial_active,
                "partial_generation": self.partial_generation,
                "partial_language": self.partial_language,
                "partial_decode_count": self.partial_decode_count,
                "message": self.message,
                "speech_active": self.speech_active,
                "level_dbfs": round(self.level_dbfs, 1),
                "latest_text": self.latest_text,
                "latest_latency_ms": self.latest_latency_ms,
                "latest_route": self.latest_route,
                "latest_consensus_score": self.latest_consensus_score,
                "latest_reliability": self.latest_reliability,
                "latest_reason": self.latest_reason,
                "latest_candidates": list(self.latest_candidates),
                "transcript": list(self.transcript),
                "utterance_count": self.utterance_count,
                "dropped_utterances": self.dropped_utterances,
                "uptime_s": round(time.time() - self.started_at, 1),
                "error": self.error,
                "score_kind": "text_consensus_proxy",
                "language": self.language,
                "language_label": LANGUAGE_LABELS[self.language],
                "language_requested": self.language_requested,
                "language_requested_label": LANGUAGE_LABELS[
                    self.language_requested
                ],
                "language_switching": self.language_switching,
                "language_choices": dict(LANGUAGE_LABELS),
                "asr_backend": self.asr_backend,
                "personalization_count": self.personalization_count,
                "personal_hotwords": list(self.personal_hotwords),
                "native_hotword_bias": self.native_hotword_bias,
            }


@dataclass
class Utterance:
    index: int
    samples: np.ndarray
    vad_start_sample: int
    endpoint_monotonic: float
    preview_generation: int | None = None


@dataclass
class CandidateRun:
    name: str
    label: str
    text: str
    samples: np.ndarray
    process_ms: int
    decode_ms: int
    rms_dbfs: float
    clipping_ratio: float
    acoustic_score: float
    asr_text: str = ""
    personalization_applied: bool = False
    personalization_id: str = ""
    error: str = ""

    def evidence(self) -> CandidateEvidence:
        return CandidateEvidence(
            name=self.name,
            label=self.label,
            text=self.text,
            process_ms=self.process_ms,
            decode_ms=self.decode_ms,
            acoustic_score=self.acoustic_score,
            rms_dbfs=self.rms_dbfs,
            clipping_ratio=self.clipping_ratio,
        )

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("samples")
        result["total_ms"] = self.process_ms + self.decode_ms
        return result


def waveform_stats(samples: np.ndarray) -> tuple[float, float, float]:
    wav = np.asarray(samples, dtype=np.float32)
    finite = bool(np.all(np.isfinite(wav)))
    if not finite or wav.size == 0:
        return -120.0, 1.0, 0.0
    rms = float(np.sqrt(np.mean(wav * wav) + 1e-12))
    rms_dbfs = max(-120.0, 20.0 * np.log10(rms))
    clipping_ratio = float(np.mean(np.abs(wav) >= 0.999))
    safety = acoustic_safety_score(rms_dbfs, clipping_ratio, finite)
    return rms_dbfs, clipping_ratio, safety


def decode(
    recognizer: Any,
    samples: np.ndarray,
    hotwords: str | None = None,
) -> tuple[str, int]:
    started = time.perf_counter()
    stream = recognizer.create_stream(hotwords=hotwords) if hotwords else recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return stream.result.text.strip(), elapsed_ms


def enhance_gtcrn(denoiser: Any, samples: np.ndarray) -> np.ndarray:
    if hasattr(denoiser, "reset"):
        denoiser.reset()
    frame = int(denoiser.frame_shift_in_samples)
    outputs: list[np.ndarray] = []
    try:
        for position in range(0, len(samples), frame):
            chunk = samples[position : position + frame]
            if chunk.size < frame:
                chunk = np.pad(chunk, (0, frame - chunk.size))
            audio = denoiser.run(chunk, SAMPLE_RATE)
            output = np.asarray(audio.samples, dtype=np.float32)
            if output.size:
                outputs.append(output)
        tail = np.asarray(denoiser.flush().samples, dtype=np.float32)
        if tail.size:
            outputs.append(tail)
    except Exception:
        if hasattr(denoiser, "reset"):
            denoiser.reset()
        raise
    if not outputs:
        return np.zeros_like(samples, dtype=np.float32)
    enhanced = np.concatenate(outputs)[: len(samples)]
    if enhanced.size < len(samples):
        enhanced = np.pad(enhanced, (0, len(samples) - enhanced.size))
    return np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False)


class CandidateWorker(threading.Thread):
    def __init__(
        self,
        recognizers: dict[str, Any],
        gtcrn: Any,
        fastenhancer: FastEnhancerOnnx,
        work_queue: queue.Queue[Utterance],
        state: MultiCandidateState,
        log_dir: Path,
        save_utterances: bool,
        stop_event: threading.Event,
        sensevoice_dir: Path,
        japanese_model_dir: Path,
        asr_threads: int,
        language: str,
        asr_backend: str,
        language_state_path: Path,
        personalization: PersonalizationStore | None = None,
    ) -> None:
        super().__init__(name="candidate-orchestrator", daemon=True)
        self.recognizers = recognizers
        self.gtcrn = gtcrn
        self.fastenhancer = fastenhancer
        self.work_queue = work_queue
        self.state = state
        self.log_dir = log_dir
        self.save_utterances = save_utterances
        self.stop_event = stop_event
        self.sensevoice_dir = sensevoice_dir
        self.japanese_model_dir = japanese_model_dir
        self.asr_threads = asr_threads
        self.language = normalize_language(language)
        self.asr_backend = asr_backend
        self.language_state_path = language_state_path
        self.personalization = personalization
        self._language_lock = threading.Lock()
        self._requested_language = self.language
        self.final_busy = threading.Event()
        self._raw_asr_lock = threading.Lock()

    def decode_preview(self, job: PreviewJob) -> str | None:
        if self.final_busy.is_set() or not self.work_queue.empty() or not self.state.preview_is_current(job):
            return None
        if not self._raw_asr_lock.acquire(blocking=False):
            return None
        try:
            if self.final_busy.is_set() or self.language != job.language or not self.state.preview_is_current(job):
                return None
            hotwords = (self.personalization.hotword_stream(self.language)
                        if self.personalization is not None and self.language == "ja" else None)
            self.state.update(partial_decode_count=self.state.snapshot()["partial_decode_count"] + 1)
            return decode(self.recognizers["raw"], job.samples, hotwords)[0]
        finally:
            self._raw_asr_lock.release()

    def request_language(self, language: str) -> dict[str, Any]:
        requested = normalize_language(language)
        with self._language_lock:
            self._requested_language = requested
            switching = requested != self.language
        self.state.update(
            language_requested=requested,
            language_switching=switching,
            phase="switching" if switching else self.state.snapshot()["phase"],
            message=(
                f"正在切换到 {LANGUAGE_LABELS[requested]}…"
                if switching
                else f"当前语种：{LANGUAGE_LABELS[requested]}"
            ),
        )
        logging.info(
            "language switch requested: active=%s requested=%s",
            self.language,
            requested,
        )
        return self.state.snapshot()

    def _pending_language(self) -> str | None:
        with self._language_lock:
            if self._requested_language == self.language:
                return None
            return self._requested_language

    def _reload_language_if_needed(self) -> None:
        requested = self._pending_language()
        if requested is None:
            return
        self.final_busy.set()
        self.state.finish_preview()
        self.state.update(
            language_requested=requested,
            language_switching=True,
            phase="switching",
            message=f"正在加载 {LANGUAGE_LABELS[requested]} 识别器…",
            error="",
        )
        logging.info("loading three ASR recognizers for language=%s", requested)
        replacements: dict[str, Any] = {}
        replacement_backend = ""
        try:
            for name in CANDIDATE_LABELS:
                loaded = load_language_recognizer(
                    requested,
                    self.sensevoice_dir,
                    self.japanese_model_dir,
                    self.asr_threads,
                )
                replacements[name] = loaded.recognizer
                replacement_backend = loaded.backend
                logging.info(
                    "ASR[%s] language=%s backend=%s files=%s",
                    name,
                    requested,
                    loaded.backend,
                    ",".join(str(path) for path in loaded.model_files),
                )
            with self._raw_asr_lock:
                previous_recognizers = self.recognizers
                self.recognizers = replacements
                del previous_recognizers
            release_native_memory()
            self.language = requested
            self.asr_backend = replacement_backend
            save_language(self.language_state_path, requested)
        except Exception as exc:
            logging.exception("failed to switch SenseVoice language to %s", requested)
            self.final_busy.clear()
            with self._language_lock:
                self._requested_language = self.language
            self.state.update(
                language=self.language,
                language_requested=self.language,
                language_switching=False,
                phase="error",
                message=f"语种切换失败，仍使用 {LANGUAGE_LABELS[self.language]}",
                error=str(exc),
            )
            return

        with self._language_lock:
            switching_again = self._requested_language != self.language
            next_language = self._requested_language
        snapshot = self.state.snapshot()
        personalization = (
            self.personalization.snapshot(self.language)
            if self.personalization is not None
            else {"count": 0, "hotwords": [], "native_hotword_bias": False}
        )
        self.state.update(
            language=self.language,
            language_requested=next_language,
            language_switching=switching_again,
            asr_backend=self.asr_backend,
            personalization_count=personalization["count"],
            personal_hotwords=personalization["hotwords"],
            native_hotword_bias=personalization["native_hotword_bias"],
            phase="switching" if switching_again else (
                "speaking" if snapshot["speech_active"] else "listening"
            ),
            message=(
                f"正在切换到 {LANGUAGE_LABELS[next_language]}…"
                if switching_again
                else f"已切换到 {LANGUAGE_LABELS[self.language]}，请开始说话"
            ),
            error="",
        )
        logging.info("language switch completed: %s", self.language)
        self.final_busy.clear()

    def _run_one(
        self,
        name: str,
        raw: np.ndarray,
        hotwords: str | None = None,
    ) -> CandidateRun:
        process_started = time.perf_counter()
        try:
            if name == "raw":
                candidate = raw
            elif name == "gtcrn":
                candidate = enhance_gtcrn(self.gtcrn, raw)
            elif name == "fastenhancer":
                candidate = self.fastenhancer.enhance(raw)
            else:
                raise ValueError(f"unknown candidate: {name}")
            process_ms = round((time.perf_counter() - process_started) * 1000)
            if name == "raw":
                wait_started = time.perf_counter()
                with self._raw_asr_lock:
                    process_ms += round((time.perf_counter() - wait_started) * 1000)
                    asr_text, decode_ms = decode(self.recognizers[name], candidate, hotwords)
            else:
                asr_text, decode_ms = decode(self.recognizers[name], candidate, hotwords)
            text = asr_text
            personalization_entry = None
            if self.personalization is not None:
                text, personalization_entry = self.personalization.apply(
                    self.language, asr_text
                )
            rms_dbfs, clipping_ratio, safety = waveform_stats(candidate)
            return CandidateRun(
                name=name,
                label=CANDIDATE_LABELS[name],
                text=text,
                samples=candidate,
                process_ms=process_ms,
                decode_ms=decode_ms,
                rms_dbfs=round(rms_dbfs, 2),
                clipping_ratio=round(clipping_ratio, 6),
                acoustic_score=round(safety, 4),
                asr_text=asr_text,
                personalization_applied=personalization_entry is not None,
                personalization_id=(
                    personalization_entry.id if personalization_entry is not None else ""
                ),
            )
        except Exception as exc:
            logging.exception("candidate %s failed", name)
            return CandidateRun(
                name=name,
                label=CANDIDATE_LABELS[name],
                text="",
                samples=np.empty(0, dtype=np.float32),
                process_ms=round((time.perf_counter() - process_started) * 1000),
                decode_ms=0,
                rms_dbfs=-120.0,
                clipping_ratio=1.0,
                acoustic_score=0.0,
                error=str(exc),
            )

    def run(self) -> None:
        transcript_path = self.log_dir / "transcript_multicandidate.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="candidate"
        ) as executor:
            while not self.stop_event.is_set():
                self._reload_language_if_needed()
                try:
                    utterance = self.work_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                self.final_busy.set()
                self.state.update(
                    phase="comparing",
                    message="三路增强与识别正在竞争…",
                )
                try:
                    hotwords = None
                    if (
                        self.personalization is not None
                        and self.asr_backend == JAPANESE_BACKEND
                    ):
                        hotwords = self.personalization.hotword_stream(self.language)
                    futures = {
                        executor.submit(
                            self._run_one, name, utterance.samples, hotwords
                        ): name
                        for name in CANDIDATE_LABELS
                    }
                    runs_by_name: dict[str, CandidateRun] = {}
                    for future in as_completed(futures):
                        run = future.result()
                        runs_by_name[run.name] = run
                    runs = [runs_by_name[name] for name in CANDIDATE_LABELS]
                    decision = select_candidate(run.evidence() for run in runs)
                    decision_dict = decision.to_dict()
                    endpoint_ms = round(
                        (time.monotonic() - utterance.endpoint_monotonic) * 1000
                    )
                    public_runs = {run.name: run.public_dict() for run in runs}
                    # Merge measured branch data into the selection view.
                    candidates = []
                    for score in decision_dict["candidates"]:
                        score.update(public_runs[score["name"]])
                        candidates.append(score)
                    personalization_applied = any(
                        run.personalization_applied for run in runs
                    )
                    item = {
                        "index": utterance.index,
                        "session_id": self.state.session_id,
                        "record_id": f"{self.state.session_id}:{utterance.index}",
                        "preview_generation": utterance.preview_generation,
                        "time": utc_timestamp(),
                        "text": decision.text,
                        "selected_name": decision.selected_name,
                        "selected_label": CANDIDATE_LABELS.get(
                            decision.selected_name, "未选择"
                        ),
                        "consensus_score": decision.consensus_score,
                        "reliability": decision.reliability,
                        "reason": decision.reason + (
                            " · 已应用个性化纠错记忆"
                            if personalization_applied
                            else ""
                        ),
                        "duration_s": round(len(utterance.samples) / SAMPLE_RATE, 2),
                        "latency_ms": endpoint_ms,
                        "language": self.language,
                        "language_label": LANGUAGE_LABELS[self.language],
                        "asr_backend": self.asr_backend,
                        "personalization_applied": personalization_applied,
                        "hotword_bias_active": bool(hotwords),
                        "candidates": candidates,
                    }
                    if decision.text:
                        with transcript_path.open("a", encoding="utf-8") as file:
                            file.write(json.dumps(item, ensure_ascii=False) + "\n")
                        self.state.append_result(item)
                    else:
                        self.state.update(
                            phase="listening",
                            message="三路均未识别到文字，请继续说话",
                            latest_latency_ms=endpoint_ms,
                            latest_route="none",
                            latest_consensus_score=0.0,
                            latest_reliability="low",
                            latest_reason=decision.reason,
                            latest_candidates=candidates,
                        )
                    if self.save_utterances:
                        base = self.log_dir / "utterances" / f"utt_{utterance.index:05d}"
                        save_pcm16_wav(base.with_name(base.name + "_raw.wav"), utterance.samples)
                        for run in runs:
                            if run.name != "raw" and run.samples.size:
                                save_pcm16_wav(
                                    base.with_name(base.name + f"_{run.name}.wav"),
                                    run.samples,
                                )
                    logging.info(
                        "utterance=%d duration=%.2fs selected=%s consensus=%.3f "
                        "endpoint=%dms texts=%s",
                        utterance.index,
                        len(utterance.samples) / SAMPLE_RATE,
                        decision.selected_name,
                        decision.consensus_score,
                        endpoint_ms,
                        {run.name: run.text for run in runs},
                    )
                except Exception as exc:
                    logging.exception("candidate orchestration failed")
                    self.state.update(
                        phase="error",
                        message="候选比较失败，采集仍在继续",
                        error=str(exc),
                    )
                finally:
                    self.state.finish_preview(utterance.preview_generation)
                    self.final_busy.clear()
                    self.work_queue.task_done()


class AudioPipeline(threading.Thread):
    def __init__(
        self,
        args: argparse.Namespace,
        state: MultiCandidateState,
        work_queue: queue.Queue[Utterance],
        stop_event: threading.Event,
        preview_worker: PartialPreviewWorker | None = None,
    ) -> None:
        super().__init__(name="raw-vad-capture", daemon=True)
        self.args = args
        self.state = state
        self.work_queue = work_queue
        self.stop_event = stop_event
        self.process: subprocess.Popen[bytes] | None = None
        self.preview = (PrefixPreview(state.begin_preview, preview_worker.offer)
                        if preview_worker is not None else None)

    def _enqueue_segments(self, vad: Any, next_index: int) -> int:
        while not vad.empty():
            segment = vad.front
            samples = np.asarray(segment.samples, dtype=np.float32).copy()
            utterance = Utterance(
                index=next_index,
                samples=samples,
                vad_start_sample=int(segment.start),
                endpoint_monotonic=time.monotonic(),
                preview_generation=self.preview.finish() if self.preview is not None else None,
            )
            vad.pop()
            try:
                self.work_queue.put_nowait(utterance)
                next_index += 1
            except queue.Full:
                self.state.finish_preview(utterance.preview_generation)
                snap = self.state.snapshot()
                self.state.update(
                    dropped_utterances=snap["dropped_utterances"] + 1,
                    message="候选识别队列忙，已丢弃一段",
                )
                logging.warning("candidate queue full; dropped utterance")
        return next_index

    def run(self) -> None:
        vad = make_vad(
            self.args.vad_model,
            self.args.min_silence,
            self.args.vad_threshold,
        )
        capture_samples = 512
        frame_bytes = capture_samples * CAPTURE_CHANNELS * 2
        command = [
            "arecord",
            "-q",
            "-D",
            self.args.device,
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(SAMPLE_RATE),
            "-c",
            str(CAPTURE_CHANNELS),
            "--period-size",
            str(capture_samples),
            "--buffer-size",
            "2048",
        ]
        logging.info("capture command: %s", " ".join(command))
        next_index = max([int(item.get("index", 0)) for item in self.state.snapshot()["transcript"]] + [0]) + 1
        pending = bytearray()
        last_level_update = 0.0
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            assert self.process.stdout is not None
            self.state.update(phase="listening", message="请开始说话")
            while not self.stop_event.is_set():
                block = self.process.stdout.read(frame_bytes - len(pending))
                if not block:
                    if self.stop_event.is_set():
                        break
                    raise RuntimeError(
                        f"arecord stopped unexpectedly (code={self.process.poll()})"
                    )
                pending.extend(block)
                if len(pending) < frame_bytes:
                    continue
                frame_data = bytes(pending)
                pending.clear()
                stereo = np.frombuffer(frame_data, dtype="<i2").reshape(-1, 2)
                raw = stereo[:, 0].astype(np.float32) / 32768.0
                vad.accept_waveform(raw)
                next_index = self._enqueue_segments(vad, next_index)

                now = time.monotonic()
                speaking = bool(vad.is_speech_detected())
                if self.preview is not None:
                    preview_state = self.state.snapshot()
                    if not preview_state["language_switching"]:
                        self.preview.push(raw, speaking, preview_state["language"], now)
                    else:
                        self.preview.finish()
                if now - last_level_update >= 0.1:
                    rms = float(np.sqrt(np.mean(raw * raw) + 1e-12))
                    dbfs = max(-120.0, 20.0 * np.log10(rms))
                    speaking = bool(vad.is_speech_detected())
                    snapshot = self.state.snapshot()
                    self.state.update(
                        speech_active=speaking,
                        level_dbfs=dbfs,
                        phase="speaking" if speaking else snapshot["phase"],
                        message="正在聆听…" if speaking else snapshot["message"],
                    )
                    last_level_update = now
        except Exception as exc:
            logging.exception("audio pipeline failed")
            self.state.update(
                phase="error",
                message="麦克风采集失败",
                error=str(exc),
            )
            self.stop_event.set()
        finally:
            try:
                vad.flush()
                self._enqueue_segments(vad, next_index)
            except Exception:
                logging.exception("failed to flush raw VAD")
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.state.finish_preview()


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三路竞争语音识别</title>
<style>
:root{color-scheme:dark;--bg:#07111f;--card:#102137;--ink:#f2f7ff;--muted:#91a4bd;--cyan:#4ee6c1;--blue:#5aa7ff;--red:#ff6876;--gold:#ffd166}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 20% 0,#183957 0,transparent 42%),var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
main{width:min(1050px,93vw);margin:0 auto;padding:6vh 0 60px}.eyebrow{color:var(--cyan);letter-spacing:.14em;font-size:13px;font-weight:700}h1{font-size:clamp(32px,5vw,58px);line-height:1.05;margin:12px 0 22px}.card{border:1px solid #29435f;border-radius:22px;background:#102137dd;box-shadow:0 20px 70px #0007;padding:clamp(20px,3vw,34px)}
.status{display:flex;gap:13px;align-items:center;color:var(--muted)}.dot{width:12px;height:12px;border-radius:99px;background:var(--cyan);box-shadow:0 0 18px var(--cyan)}.speaking .dot,.comparing .dot,.switching .dot{animation:pulse .8s infinite alternate}.error .dot{background:var(--red)}@keyframes pulse{to{transform:scale(1.7);opacity:.45}}
#latest{min-height:2.3em;font-size:clamp(24px,4vw,40px);line-height:1.35;margin:26px 0 14px}.decision{color:var(--gold);min-height:1.5em}.meter{height:9px;background:#07111f;border-radius:99px;overflow:hidden;margin-top:22px}.meter>div{height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--cyan));transition:width .12s}.meta{display:flex;flex-wrap:wrap;gap:8px 24px;color:var(--muted);font-size:13px;margin-top:10px}
.language-control{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:20px 0 0;padding:12px 14px;border:1px solid #29435f;border-radius:14px;background:#09182a}.language-control label{display:flex;align-items:center;gap:10px;font-weight:650}.language-control select{min-width:150px;border:1px solid #3a5d7e;border-radius:9px;background:#102137;color:var(--ink);padding:8px 34px 8px 11px;font:inherit;cursor:pointer}.language-control select:disabled{cursor:wait;opacity:.6}.language-control span{color:var(--muted);font-size:13px;text-align:right}
.personalization{margin-top:18px;padding:16px;border:1px solid #29435f;border-radius:14px;background:#09182a}.learning-head{display:flex;justify-content:space-between;align-items:center;gap:12px}.learning-head b{color:var(--cyan)}.learning-head span{color:var(--muted);font-size:13px}.learn-grid{display:grid;grid-template-columns:1.4fr 1fr auto;gap:9px;margin-top:12px}.learn-grid input{min-width:0;border:1px solid #3a5d7e;border-radius:9px;background:#102137;color:var(--ink);padding:10px 11px;font:inherit}.learn-grid button,.clear-memory{border:0;border-radius:9px;background:var(--cyan);color:#06131c;padding:10px 14px;font-weight:750;cursor:pointer}.clear-memory{background:transparent;color:var(--muted);border:1px solid #35536f;padding:5px 9px;font-size:12px}.memory-row{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:10px}.tags{display:flex;flex-wrap:wrap;gap:6px}.tag{background:#17314a;color:#bfe9ff;border-radius:99px;padding:4px 9px;font-size:12px}.learn-result{min-height:1.4em;margin-top:9px;color:var(--gold);font-size:13px}
h2{margin:34px 0 13px;font-size:18px}.candidates{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.candidate{border:1px solid #29435f;border-radius:15px;background:#0c1a2b;padding:16px;min-height:145px}.candidate.selected{border-color:var(--cyan);box-shadow:0 0 0 1px var(--cyan) inset}.candidate b{color:var(--blue)}.candidate.selected b{color:var(--cyan)}.candidate .txt{font-size:18px;margin:12px 0;word-break:break-all}.candidate small{color:var(--muted)}.history{display:grid;gap:10px}.line{border-left:3px solid #355c82;padding:10px 14px;background:#0c1a2b99;border-radius:0 12px 12px 0}.line small{color:var(--muted);display:block;margin-top:5px}.empty{color:var(--muted)}
@media(max-width:760px){.candidates{grid-template-columns:1fr}.language-control{align-items:flex-start;flex-direction:column}.language-control span{text-align:left}.learn-grid{grid-template-columns:1fr}.memory-row{align-items:flex-start;flex-direction:column}}

#latest{height:300px;min-height:180px;overflow-y:auto;margin:10px 0;font-size:clamp(21px,3vw,29px);line-height:1.6;padding:4px 12px 4px 0;overflow-wrap:anywhere}.conversation-head{display:flex;justify-content:space-between;align-items:center;margin-top:22px;color:var(--cyan)}#clear-display{border:1px solid #35536f;border-radius:9px;background:transparent;color:var(--muted);padding:6px 12px;font:inherit;font-size:13px;cursor:pointer}.transcript-line{padding:12px 0;border-bottom:1px solid #29435f66;white-space:pre-wrap}.transcript-line small{display:block;font-size:12px;line-height:1.5;color:var(--muted);margin-top:4px}.transcript-line.partial{color:var(--gold)}.transcript-line.partial::after{content:' ▍';animation:pulse .8s infinite alternate}.archive-note{font-size:12px;color:var(--muted);line-height:1.6}.learn-grid input:disabled,.learn-grid button:disabled{opacity:.55;cursor:wait}
</style></head><body><main>
<div class="eyebrow">RAW · GTCRN · FASTENHANCER · SENSEVOICE</div><h1>三路竞争<br>语音理解</h1>
<section class="card" id="card"><div class="status"><span class="dot"></span><span id="message">正在连接树莓派…</span></div><div class="language-control"><label for="language">识别语种<select id="language"><option value="auto">自动识别</option><option value="zh">中文</option><option value="en">English</option><option value="ja">日本語</option><option value="ko">한국어</option><option value="yue">粤语</option></select></label><span id="language-status">正在读取语种设置…</span></div><div class="conversation-head"><b>实时转写</b><button id="clear-display" type="button" title="仅清空显示，不删除后台记录">清空显示</button></div><div id="latest" role="log" aria-live="polite"><div id="conversation-final"></div><div id="partial-line" class="transcript-line partial" hidden></div><div id="conversation-empty" class="empty">请开始说话，识别文字会在这里逐步显示。</div></div><div class="archive-note">页面保留最近500条，最终转写同时保存在设备后台；清空显示不删除记录或纠错记忆。</div><div class="decision" id="decision"></div><div class="meter"><div id="bar"></div></div><div class="meta"><span id="level">输入电平 -- dBFS</span><span id="route">尚未选择路径</span><span id="latency">等待首句</span></div><div class="personalization"><div class="learning-head"><b>个性化纠错学习</b><span id="memory-status">尚未学习专有词</span></div><div class="learn-grid"><input id="corrected-text" maxlength="300" aria-label="正确文字" placeholder="说完一句后，在这里修改正确文字"><input id="hotword-text" maxlength="64" aria-label="重点专有词" placeholder="重点专有词（可选）"><button id="learn-button" type="button">纠正并学习</button></div><div class="memory-row"><div class="tags" id="hotword-tags"></div><button class="clear-memory" id="clear-memory" type="button">清空本语种记忆</button></div><div class="learn-result" id="learn-result"></div></div></section>
<h2>候选对比</h2><div class="candidates" id="candidates"><div class="empty">完成第一句话后显示三路结果</div></div>

</main><script src="/transcript-view.js"></script><script>
const labels={raw:'原始语音',gtcrn:'GTCRN',fastenhancer:'FastEnhancer-B',none:'未选择'};
const languageNames={auto:'自动识别',zh:'中文',en:'English',ja:'日本語',ko:'한국어',yue:'粤语'};
const languageSelect=document.getElementById('language');
const languageStatus=document.getElementById('language-status');
const correctedInput=document.getElementById('corrected-text'),hotwordInput=document.getElementById('hotword-text');
const learnButton=document.getElementById('learn-button'),learnResult=document.getElementById('learn-result');
let lastCount=-1,languagePosting=false,currentState=null,correctionIndex=null,correctionObserved='';
let correctionPosting=false,conversationSignature='';
const displayBuffer=new TranscriptBuffer({getItem:key=>localStorage.getItem(key),setItem:(key,value)=>localStorage.setItem(key,value)});
function renderConversation(s){
  const view=displayBuffer.ingest(s),box=document.getElementById('latest');
  const stick=box.scrollTop+box.clientHeight>=box.scrollHeight-50;
  const signature=JSON.stringify(view.rows);
  if(signature!==conversationSignature){
    const fragment=document.createDocumentFragment();
    for(const row of view.rows){
      const line=document.createElement('div');line.className='transcript-line';
      const text=document.createElement('div');text.textContent=row.text;line.appendChild(text);
      const meta=document.createElement('small');meta.textContent=[row.language,row.latency_ms==null?'':`${row.latency_ms} ms`,row.user_corrected?'已纠正':''].filter(Boolean).join(' · ');line.appendChild(meta);fragment.appendChild(line);
    }
    document.getElementById('conversation-final').replaceChildren(fragment);conversationSignature=signature;
  }
  const draft=document.getElementById('partial-line');draft.hidden=!view.draft;
  draft.textContent=view.draft?view.draft.text:'';
  document.getElementById('conversation-empty').hidden=Boolean(view.rows.length||view.draft);
  if(stick)box.scrollTop=box.scrollHeight;
  const pending=Boolean(s.partial_active||s.speech_active||s.language_switching);
  learnButton.disabled=correctionPosting||pending||correctionIndex==null;
  correctedInput.disabled=pending;hotwordInput.disabled=pending;
}
document.getElementById('clear-display').addEventListener('click',()=>{
  if(!currentState)return;
  displayBuffer.clear(currentState);correctionIndex=null;correctedInput.value='';hotwordInput.value='';renderConversation(currentState);
});

function esc(s){const d=document.createElement('div');d.textContent=s??'';return d.innerHTML}
async function changeLanguage(){
  const language=languageSelect.value;languagePosting=true;languageSelect.disabled=true;
  languageStatus.textContent=`正在请求 ${languageNames[language]}…`;
  try{const r=await fetch('/api/language',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({language})});const result=await r.json();if(!r.ok)throw new Error(result.error||'语种切换失败')}
  catch(e){languageStatus.textContent=e.message}
  finally{languagePosting=false;refresh()}
}
languageSelect.addEventListener('change',changeLanguage);
async function learnCorrection(){
  const corrected=correctedInput.value.trim(),hotword=hotwordInput.value.trim();
  if(!currentState||correctionIndex==null){learnResult.textContent='请先说一句话';return}
  if(!corrected){learnResult.textContent='正确文字不能为空';return}
  correctionPosting=true;learnButton.disabled=true;learnResult.textContent='正在保存纠错并更新个人词库…';
  try{const r=await fetch('/api/correction',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({utterance_index:correctionIndex,language:currentState.language,observed:correctionObserved,corrected,hotword})});const result=await r.json();if(!r.ok)throw new Error(result.error||'学习失败');correctionObserved=corrected;hotwordInput.value='';lastCount=-1;learnResult.textContent=result.hotword?`已学习专有词：${result.hotword}`:'已保存纠错记忆';await refresh()}
  catch(e){learnResult.textContent=e.message}finally{correctionPosting=false;if(currentState)renderConversation(currentState)}
}
async function clearMemory(){
  if(!currentState||!confirm(`清空${languageNames[currentState.language]||currentState.language}的全部个性化记忆？`))return;
  try{const r=await fetch('/api/personalization/clear',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({language:currentState.language})});const result=await r.json();if(!r.ok)throw new Error(result.error||'清空失败');learnResult.textContent=`已清空 ${result.removed} 条记忆`;await refresh()}
  catch(e){learnResult.textContent=e.message}
}
learnButton.addEventListener('click',learnCorrection);document.getElementById('clear-memory').addEventListener('click',clearMemory);
async function refresh(){try{
  const r=await fetch('/api/state',{cache:'no-store'}),s=await r.json();
  currentState=s;
  document.getElementById('card').className='card '+s.phase;document.getElementById('message').textContent=s.message;
  const desired=s.language_requested||s.language||'auto';if(!languagePosting)languageSelect.value=desired;
  languageSelect.disabled=languagePosting||Boolean(s.language_switching);
  languageStatus.textContent=s.language_switching?`正在加载 ${languageNames[desired]||desired}…`:`当前：${s.language_label||languageNames[s.language]||s.language} · ${s.asr_backend||'ASR'}`;
  document.getElementById('memory-status').textContent=`已学习 ${s.personalization_count||0} 条 · ${s.native_hotword_bias?'模型热词立即生效':'精确纠错记忆'}`;
  document.getElementById('hotword-tags').innerHTML=s.personal_hotwords?.length?s.personal_hotwords.map(x=>`<span class="tag">${esc(x)}</span>`).join(''):'<span class="empty">暂无重点词</span>';
  document.getElementById('level').textContent=`输入电平 ${s.level_dbfs.toFixed(1)} dBFS`;document.getElementById('bar').style.width=`${Math.max(0,Math.min(100,(s.level_dbfs+70)*1.8))}%`;
  document.getElementById('latency').textContent=s.latest_latency_ms==null?'等待首句':`停顿后 ${s.latest_latency_ms} ms`;document.getElementById('route').textContent=s.latest_route?`胜出：${labels[s.latest_route]||s.latest_route}`:'尚未选择路径';
  document.getElementById('decision').textContent=s.partial_active?'识别中 · 临时文字可能修正':(s.latest_reason?(s.latest_reason+` · 共识分 ${Number(s.latest_consensus_score||0).toFixed(2)}`):'');
  if(s.latest_candidates?.length)document.getElementById('candidates').innerHTML=s.latest_candidates.map(x=>`<div class="candidate ${x.selected?'selected':''}"><b>${esc(x.label)}${x.selected?' · 胜出':''}</b><div class="txt">${esc(x.text)||'未识别到文字'}</div><small>共识 ${Number(x.consensus||0).toFixed(2)} · 音频安全 ${Number(x.acoustic_score||0).toFixed(2)} · ${x.total_ms} ms${x.personalization_applied?' · 已应用记忆':''}</small>${x.personalization_applied?`<small>模型原文：${esc(x.asr_text)}</small>`:''}${x.error?`<small> · ${esc(x.error)}</small>`:''}</div>`).join('');
  if(s.utterance_count!==lastCount){lastCount=s.utterance_count;const latest=s.transcript?.[s.transcript.length-1];correctionIndex=latest?.index??null;correctionObserved=latest?.text||'';correctedInput.value=latest?.text||'';hotwordInput.value=''}
  renderConversation(s);
}catch(e){document.getElementById('message').textContent='与树莓派断开连接，正在重试…'}}
setInterval(refresh,250);refresh();
</script></body></html>"""


class DemoHandler(BaseHTTPRequestHandler):
    state: MultiCandidateState
    language_callback: Callable[[str], dict[str, Any]]
    correction_callback: Callable[[dict[str, Any]], dict[str, Any]]
    clear_personalization_callback: Callable[[str], dict[str, Any]]

    @classmethod
    def configure_personalization_callbacks(
        cls,
        correction_callback: Callable[[dict[str, Any]], dict[str, Any]],
        clear_callback: Callable[[str], dict[str, Any]],
    ) -> None:
        # Plain functions stored on a class become bound methods when accessed
        # through a handler instance.  staticmethod preserves their API shape.
        cls.correction_callback = staticmethod(correction_callback)
        cls.clear_personalization_callback = staticmethod(clear_callback)

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route == "/":
            payload = INDEX_HTML.encode("utf-8")
            content_type = "text/html; charset=utf-8"
            status = 200
        elif route in ("/api/state", "/api/health"):
            payload = json.dumps(
                self.state.snapshot(), ensure_ascii=False
            ).encode("utf-8")
            content_type = "application/json; charset=utf-8"
            status = 200
        elif route == "/transcript-view.js":
            payload = Path(__file__).with_name("transcript_view.js").read_bytes()
            content_type = "application/javascript; charset=utf-8"
            status = 200
        else:
            payload = b"not found"
            content_type = "text/plain; charset=utf-8"
            status = 404
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route not in (
            "/api/language",
            "/api/correction",
            "/api/personalization/clear",
        ):
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= 4096:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request body must be an object")
            if route == "/api/language":
                snapshot = self.language_callback(request.get("language"))
                self._send_json(
                    202,
                    {
                        "ok": True,
                        "language": snapshot["language"],
                        "language_requested": snapshot["language_requested"],
                        "language_switching": snapshot["language_switching"],
                    },
                )
            elif route == "/api/correction":
                self._send_json(201, self.correction_callback(request))
            else:
                self._send_json(
                    200,
                    self.clear_personalization_callback(request.get("language")),
                )
        except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            logging.exception("HTTP API failed")
            self._send_json(500, {"ok": False, "error": str(exc)})

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.debug("http: " + fmt, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="plughw:U192k,0")
    parser.add_argument("--gtcrn", type=Path, required=True)
    parser.add_argument("--fastenhancer", type=Path, required=True)
    parser.add_argument("--vad-model", type=Path, required=True)
    parser.add_argument("--sensevoice-dir", type=Path, required=True)
    parser.add_argument(
        "--japanese-model-dir",
        type=Path,
        default=Path("models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--min-silence", type=float, default=0.45)
    parser.add_argument("--vad-threshold", type=float, default=0.50)
    parser.add_argument("--denoise-threads", type=int, default=1)
    parser.add_argument("--fastenhancer-threads", type=int, default=1)
    parser.add_argument("--asr-threads", type=int, default=1)
    parser.add_argument("--language", choices=tuple(LANGUAGE_LABELS), default=None)
    parser.add_argument("--save-utterances", action="store_true")
    parser.add_argument("--partial-preview", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                args.log_dir / "voice_demo_multicandidate.log", encoding="utf-8"
            ),
        ],
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    language_state_path = args.log_dir / "language.json"
    language = (
        normalize_language(args.language)
        if args.language is not None
        else load_language(language_state_path, "auto")
    )
    personalization = PersonalizationStore(args.log_dir / "personalization.json")
    personalization_snapshot = personalization.snapshot(language)
    state = MultiCandidateState(
        language=language,
        language_requested=language,
        asr_backend=backend_label(language),
        personalization_count=personalization_snapshot["count"],
        personal_hotwords=personalization_snapshot["hotwords"],
        native_hotword_bias=personalization_snapshot["native_hotword_bias"],
    )
    stop_event = threading.Event()
    state.transcript = recent_transcripts(args.log_dir / "transcript_multicandidate.jsonl")
    if state.transcript:
        state.latest_text = state.transcript[-1]["text"]
    work_queue: queue.Queue[Utterance] = queue.Queue(maxsize=2)

    logging.info("loading three independent ASR recognizers")
    recognizers: dict[str, Any] = {}
    for name in CANDIDATE_LABELS:
        loaded = load_language_recognizer(
            language,
            args.sensevoice_dir,
            args.japanese_model_dir,
            args.asr_threads,
        )
        recognizers[name] = loaded.recognizer
        logging.info(
            "ASR[%s] language=%s backend=%s files=%s",
            name,
            language,
            loaded.backend,
            ",".join(str(path) for path in loaded.model_files),
        )
    # Loop variables live until main() returns. Without this deletion the final
    # startup recognizer stays pinned after the first language change.
    del loaded
    gtcrn = make_denoiser(args.gtcrn, args.denoise_threads)
    fastenhancer = FastEnhancerOnnx(
        args.fastenhancer, num_threads=args.fastenhancer_threads
    )
    logging.info(
        "FastEnhancer model=%s hop=%d delay=%d",
        args.fastenhancer,
        fastenhancer.hop_size,
        fastenhancer.delay_samples,
    )

    worker = CandidateWorker(
        recognizers,
        gtcrn,
        fastenhancer,
        work_queue,
        state,
        args.log_dir,
        args.save_utterances,
        stop_event,
        args.sensevoice_dir,
        args.japanese_model_dir,
        args.asr_threads,
        language,
        backend_label(language),
        language_state_path,
        personalization,
    )
    # CandidateWorker owns the recognizer dictionary from this point. Keeping the
    # main-frame reference would pin the startup language after the first switch.
    del recognizers
    preview_worker = (PartialPreviewWorker(worker.decode_preview, state.publish_preview,
                      state.preview_is_current, worker.final_busy, stop_event)
                      if args.partial_preview else None)
    pipeline = AudioPipeline(args, state, work_queue, stop_event, preview_worker)
    if preview_worker is not None:
        preview_worker.start()
    worker.start()
    pipeline.start()

    DemoHandler.state = state
    DemoHandler.language_callback = worker.request_language

    def save_correction(request: dict[str, Any]) -> dict[str, Any]:
        correction_language = normalize_language(request.get("language"))
        try:
            utterance_index = int(request.get("utterance_index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("utterance_index must be an integer") from exc
        observed = request.get("observed")
        corrected = request.get("corrected")
        hotword = request.get("hotword")
        if not isinstance(observed, str):
            raise ValueError("observed must be a string")
        state.validate_user_correction(
            utterance_index, correction_language, observed
        )
        entry = personalization.add_correction(
            correction_language,
            observed,
            corrected,
            hotword,
        )
        state.apply_user_correction(
            utterance_index,
            correction_language,
            entry.observed,
            entry.corrected,
            entry.hotword,
        )
        summary = personalization.snapshot(correction_language)
        if state.snapshot()["language"] == correction_language:
            state.update(
                personalization_count=summary["count"],
                personal_hotwords=summary["hotwords"],
                native_hotword_bias=summary["native_hotword_bias"],
            )
        logging.info(
            "personalization learned language=%s observed=%r corrected=%r hotword=%r",
            correction_language,
            entry.observed,
            entry.corrected,
            entry.hotword,
        )
        return {
            "ok": True,
            "entry_id": entry.id,
            "corrected": entry.corrected,
            "hotword": entry.hotword,
            "personalization": summary,
        }

    def clear_personalization(language_value: str) -> dict[str, Any]:
        clear_language = normalize_language(language_value)
        removed = personalization.clear(clear_language)
        summary = personalization.snapshot(clear_language)
        if state.snapshot()["language"] == clear_language:
            state.update(
                personalization_count=0,
                personal_hotwords=[],
                native_hotword_bias=summary["native_hotword_bias"],
            )
        logging.info(
            "personalization cleared language=%s removed=%d",
            clear_language,
            removed,
        )
        return {"ok": True, "removed": removed, "personalization": summary}

    DemoHandler.configure_personalization_callbacks(
        save_correction,
        clear_personalization,
    )
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    server.daemon_threads = True

    def request_stop(_signum: int, _frame: Any) -> None:
        logging.info("shutdown requested")
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    logging.info("web UI listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        stop_event.set()
        server.server_close()
        pipeline.join(timeout=3)
        worker.join(timeout=3)
        if preview_worker is not None:
            preview_worker.join(timeout=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
