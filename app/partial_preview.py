"""Bounded prefix previews; final results alone enter the transcript archive."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import logging
import threading
from typing import Callable
import numpy as np

@dataclass(frozen=True)
class PreviewJob:
    generation: int
    language: str
    samples: np.ndarray

class PrefixPreview:
    def __init__(self, begin: Callable[[str], int], offer: Callable[[PreviewJob], None],
                 sample_rate: int = 16000, interval_s: float = 1.0, max_seconds: float = 21.0) -> None:
        self.begin, self.offer = begin, offer
        self.rate, self.interval = sample_rate, interval_s
        self.limit, self.pre_limit = int(sample_rate * max_seconds), int(sample_rate * .5)
        self.pre: deque[np.ndarray] = deque()
        self.pre_size = 0
        self.chunks: list[np.ndarray] = []
        self.size = 0
        self.generation: int | None = None
        self.language = ""
        self.next_offer = 0.0

    def finish(self) -> int | None:
        generation = self.generation
        self.generation = None
        self.chunks, self.size = [], 0
        self.pre.clear(); self.pre_size = 0
        return generation

    def push(self, frame: np.ndarray, speaking: bool, language: str, now: float) -> None:
        frame = np.asarray(frame, dtype=np.float32).copy()
        if self.generation is not None and language != self.language: self.finish()
        if speaking and self.generation is None:
            self.language, self.generation = language, self.begin(language)
            self.chunks, self.size = list(self.pre), self.pre_size
            self.next_offer = now + self.interval
        if self.generation is not None and self.size < self.limit:
            remaining = self.limit - self.size
            self.chunks.append(frame[:remaining]); self.size += min(len(frame), remaining)
            if speaking and now >= self.next_offer and self.size >= int(.8 * self.rate):
                self.offer(PreviewJob(self.generation, self.language, np.concatenate(self.chunks)))
                self.next_offer = now + self.interval
        self.pre.append(frame); self.pre_size += len(frame)
        while self.pre_size > self.pre_limit:
            excess, first = self.pre_size - self.pre_limit, self.pre.popleft()
            if len(first) > excess:
                self.pre.appendleft(first[excess:]); self.pre_size -= excess
            else: self.pre_size -= len(first)

class PartialPreviewWorker(threading.Thread):
    """One in-flight and one latest pending prefix; final decoding has priority."""
    def __init__(self, decode: Callable[[PreviewJob], str | None], publish: Callable[[PreviewJob, str], None],
                 is_current: Callable[[PreviewJob], bool], busy: threading.Event, stop: threading.Event) -> None:
        super().__init__(name="partial-preview", daemon=True)
        self.decode, self.publish, self.is_current = decode, publish, is_current
        self.busy, self.stop = busy, stop
        self.condition = threading.Condition()
        self.pending: PreviewJob | None = None
        self.decode_calls = self.failures = 0

    def offer(self, job: PreviewJob) -> None:
        with self.condition:
            self.pending = job; self.condition.notify()

    def run(self) -> None:
        while not self.stop.is_set():
            with self.condition:
                if self.pending is None: self.condition.wait(timeout=.2)
                if self.pending is None: continue
                if self.busy.is_set():
                    self.condition.wait(timeout=.05); continue
                job, self.pending = self.pending, None
            if not self.is_current(job): continue
            try:
                self.decode_calls += 1
                text = self.decode(job)
                if text is not None and self.is_current(job): self.publish(job, text)
            except Exception:
                self.failures += 1
                logging.exception("partial preview failed; final recognition remains available")
