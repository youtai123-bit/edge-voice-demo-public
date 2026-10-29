#!/usr/bin/env python3
"""Small, dependency-light FastEnhancer wav2wav ONNX adapter.

The framing and delay compensation follow the official FastEnhancer
``scripts/test_onnx.py`` implementation.  A fresh cache is used for every
utterance so two sentences cannot leak recurrent state into each other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class FastEnhancerOnnx:
    def __init__(
        self,
        model_path: Path | str,
        *,
        n_fft: int = 512,
        num_threads: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if n_fft <= 0 or n_fft % 2:
            raise ValueError("n_fft must be a positive even integer")

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        by_name = {node.name: node for node in inputs}
        if "wav_in" not in by_name:
            raise ValueError("FastEnhancer ONNX is missing the wav_in input")
        wav_shape = by_name["wav_in"].shape
        if len(wav_shape) != 2 or not isinstance(wav_shape[1], int):
            raise ValueError(f"unsupported wav_in shape: {wav_shape}")
        self.hop_size = int(wav_shape[1])
        self.n_fft = int(n_fft)
        if self.n_fft < self.hop_size:
            raise ValueError("n_fft must be at least the model hop size")
        self.delay_samples = self.n_fft - self.hop_size

        self.cache_inputs = sorted(
            (node for node in inputs if node.name.startswith("cache_in_")),
            key=lambda node: int(node.name.rsplit("_", 1)[1]),
        )
        self.cache_outputs = sorted(
            (node for node in outputs if node.name.startswith("cache_out_")),
            key=lambda node: int(node.name.rsplit("_", 1)[1]),
        )
        if len(self.cache_inputs) != len(self.cache_outputs):
            raise ValueError("FastEnhancer input/output cache counts do not match")
        for expected, node in enumerate(self.cache_inputs):
            if node.name != f"cache_in_{expected}":
                raise ValueError("FastEnhancer cache inputs must be densely numbered")

    @staticmethod
    def _concrete_shape(shape: list[int | str | None]) -> tuple[int, ...]:
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(f"cache has a dynamic or invalid shape: {shape}")
        return tuple(int(dim) for dim in shape)

    def _zero_caches(self) -> dict[str, np.ndarray]:
        return {
            node.name: np.zeros(self._concrete_shape(node.shape), dtype=np.float32)
            for node in self.cache_inputs
        }

    def enhance(self, samples: np.ndarray) -> np.ndarray:
        """Enhance one 16 kHz utterance and return an aligned float32 waveform."""

        wav = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(wav)):
            raise ValueError("input contains NaN or infinity")
        length = int(wav.size)
        if length == 0:
            return np.empty(0, dtype=np.float32)
        wav = np.clip(wav, -1.0, 1.0)
        # Official implementation pads n_fft zeros so the OLA tail is emitted.
        padded = np.pad(wav, (0, self.n_fft), mode="constant")
        inputs = self._zero_caches()
        outputs: list[np.ndarray] = []
        upper = length + self.n_fft - self.hop_size
        for index in range(0, upper, self.hop_size):
            chunk = padded[index : index + self.hop_size]
            if chunk.size < self.hop_size:
                chunk = np.pad(chunk, (0, self.hop_size - chunk.size))
            inputs["wav_in"] = chunk.reshape(1, -1).astype(np.float32, copy=False)
            result = self.session.run(None, inputs)
            outputs.append(np.asarray(result[0][0], dtype=np.float32))
            for cache_index, cache in enumerate(result[1:]):
                inputs[f"cache_in_{cache_index}"] = cache

        merged = np.concatenate(outputs)
        aligned = merged[self.delay_samples : self.delay_samples + length]
        if aligned.size < length:
            aligned = np.pad(aligned, (0, length - aligned.size))
        if not np.all(np.isfinite(aligned)):
            raise RuntimeError("FastEnhancer produced NaN or infinity")
        return np.clip(aligned, -1.0, 1.0).astype(np.float32, copy=False)
