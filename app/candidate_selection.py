#!/usr/bin/env python3
"""Deterministic selection for raw/GTCRN/FastEnhancer ASR candidates.

sherpa-onnx does not expose SenseVoice token posterior probabilities.  This
module therefore reports a *consensus score*, not an ASR confidence.  Text
agreement is the primary signal; acoustic checks only break ties and reject
obviously damaged waveforms.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Iterable


def normalize_text(text: str) -> str:
    """Normalize multilingual ASR output for consensus comparisons."""

    normalized = unicodedata.normalize("NFKC", text).upper()
    return "".join(character for character in normalized if character.isalnum())


def text_similarity(left: str, right: str) -> float:
    """Return a deterministic character similarity in ``[0, 1]``."""

    a = normalize_text(left)
    b = normalize_text(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b, autojunk=False).ratio())


@dataclass(frozen=True)
class CandidateEvidence:
    name: str
    label: str
    text: str
    process_ms: int
    decode_ms: int
    acoustic_score: float
    rms_dbfs: float
    clipping_ratio: float

    @property
    def total_ms(self) -> int:
        return self.process_ms + self.decode_ms


@dataclass(frozen=True)
class CandidateScore:
    name: str
    label: str
    text: str
    normalized_text: str
    consensus: float
    exact_votes: int
    acoustic_score: float
    total_ms: int
    score: float
    selected: bool = False


@dataclass(frozen=True)
class SelectionDecision:
    selected_name: str
    text: str
    consensus_score: float
    reliability: str
    reason: str
    candidates: tuple[CandidateScore, ...]

    def to_dict(self) -> dict:
        return {
            "selected_name": self.selected_name,
            "text": self.text,
            "consensus_score": round(self.consensus_score, 3),
            "reliability": self.reliability,
            "reason": self.reason,
            "candidates": [asdict(item) for item in self.candidates],
        }


def _reliability(value: float, exact_votes: int, nonempty_count: int) -> str:
    if nonempty_count >= 3 and exact_votes >= 3:
        return "high"
    if exact_votes >= 2 or value >= 0.72:
        return "medium"
    return "low"


def select_candidate(
    evidence: Iterable[CandidateEvidence],
) -> SelectionDecision:
    """Choose one ASR result using agreement first and safety/latency second.

    The returned ``consensus_score`` is deliberately named to avoid presenting
    it as a calibrated SenseVoice probability.
    """

    items = tuple(evidence)
    if not items:
        raise ValueError("at least one candidate is required")
    names = [item.name for item in items]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")

    normalized = {item.name: normalize_text(item.text) for item in items}
    nonempty = [item for item in items if normalized[item.name]]
    if not nonempty:
        scores = tuple(
            CandidateScore(
                name=item.name,
                label=item.label,
                text=item.text,
                normalized_text="",
                consensus=0.0,
                exact_votes=0,
                acoustic_score=float(item.acoustic_score),
                total_ms=item.total_ms,
                score=0.0,
                selected=False,
            )
            for item in items
        )
        return SelectionDecision(
            selected_name="none",
            text="",
            consensus_score=0.0,
            reliability="low",
            reason="三路均未识别到有效文字",
            candidates=scores,
        )

    max_latency = max(item.total_ms for item in nonempty)
    min_latency = min(item.total_ms for item in nonempty)
    latency_span = max(1, max_latency - min_latency)
    provisional: list[CandidateScore] = []
    for item in items:
        norm = normalized[item.name]
        if not norm:
            provisional.append(
                CandidateScore(
                    name=item.name,
                    label=item.label,
                    text=item.text,
                    normalized_text="",
                    consensus=0.0,
                    exact_votes=0,
                    acoustic_score=float(item.acoustic_score),
                    total_ms=item.total_ms,
                    score=-1.0,
                )
            )
            continue

        peers = [peer for peer in nonempty if peer.name != item.name]
        similarities = [text_similarity(item.text, peer.text) for peer in peers]
        consensus = sum(similarities) / len(similarities) if similarities else 0.5
        exact_votes = sum(
            normalized[peer.name] == norm for peer in nonempty
        )
        exact_ratio = exact_votes / len(nonempty)
        safe_acoustic = min(1.0, max(0.0, float(item.acoustic_score)))
        latency_bonus = (max_latency - item.total_ms) / latency_span
        # Agreement dominates. Acoustic validity and latency only decide close
        # calls; neither is a replacement for a real ASR posterior.
        score = (
            0.67 * consensus
            + 0.20 * exact_ratio
            + 0.10 * safe_acoustic
            + 0.03 * latency_bonus
        )
        provisional.append(
            CandidateScore(
                name=item.name,
                label=item.label,
                text=item.text,
                normalized_text=norm,
                consensus=round(consensus, 4),
                exact_votes=exact_votes,
                acoustic_score=round(safe_acoustic, 4),
                total_ms=item.total_ms,
                score=round(score, 4),
            )
        )

    # A literal majority is stronger evidence than small score differences.
    majority_size = max(score.exact_votes for score in provisional)
    if majority_size >= 2:
        pool = [score for score in provisional if score.exact_votes == majority_size]
        chosen = max(pool, key=lambda x: (x.acoustic_score, -x.total_ms, x.score))
        if majority_size == len(nonempty):
            reason = f"{majority_size} 路文字完全一致"
        else:
            reason = f"{majority_size} 路形成文字多数共识"
    else:
        chosen = max(provisional, key=lambda x: (x.score, -x.total_ms))
        reason = "三路文字不完全一致，按文本中心性与音频安全分择优"

    final_scores = tuple(
        CandidateScore(**{**asdict(score), "selected": score.name == chosen.name})
        for score in provisional
    )
    reliability = _reliability(chosen.consensus, chosen.exact_votes, len(nonempty))
    return SelectionDecision(
        selected_name=chosen.name,
        text=chosen.text,
        consensus_score=round(float(chosen.consensus), 3),
        reliability=reliability,
        reason=reason,
        candidates=final_scores,
    )


def acoustic_safety_score(
    rms_dbfs: float,
    clipping_ratio: float,
    finite: bool = True,
) -> float:
    """Score only obvious waveform failures; do not guess perceptual quality."""

    if not finite or not math.isfinite(rms_dbfs) or not math.isfinite(clipping_ratio):
        return 0.0
    score = 1.0
    if rms_dbfs < -65.0:
        score -= min(0.8, (-65.0 - rms_dbfs) / 35.0)
    if clipping_ratio > 0.0001:
        score -= min(0.9, clipping_ratio * 40.0)
    return min(1.0, max(0.0, score))
