#!/usr/bin/env python3

import argparse
import unittest
from pathlib import Path

import numpy as np

from fastenhancer_onnx import FastEnhancerOnnx


_HERE = Path(__file__).resolve().parent
_MODEL_CANDIDATES = (
    _HERE / "models" / "fastenhancer_b_dns16k.onnx",
    _HERE.parent / "models" / "fastenhancer_b_dns16k.onnx",
)
MODEL = next((path for path in _MODEL_CANDIDATES if path.is_file()), _MODEL_CANDIDATES[0])


@unittest.skipUnless(MODEL.is_file(), "FastEnhancer test model is not present")
class FastEnhancerOnnxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enhancer = FastEnhancerOnnx(MODEL)

    def test_official_interface_metadata(self):
        self.assertEqual(self.enhancer.hop_size, 256)
        self.assertEqual(self.enhancer.delay_samples, 256)

    def test_output_is_aligned_and_finite(self):
        rng = np.random.default_rng(7)
        samples = (rng.standard_normal(16_123) * 0.01).astype(np.float32)
        output = self.enhancer.enhance(samples)
        self.assertEqual(output.shape, samples.shape)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_per_utterance_cache_reset_is_deterministic(self):
        samples = np.zeros(4_321, dtype=np.float32)
        samples[100] = 0.25
        first = self.enhancer.enhance(samples)
        second = self.enhancer.enhance(samples)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)

    def test_matches_official_reference_loop(self):
        rng = np.random.default_rng(19)
        samples = (rng.standard_normal(7_777) * 0.02).astype(np.float32)
        n_fft = 512
        hop = 256
        padded = np.pad(samples, (0, n_fft))
        inputs = {
            node.name: np.zeros(tuple(node.shape), dtype=np.float32)
            for node in self.enhancer.session.get_inputs()
            if node.name.startswith("cache_in_")
        }
        chunks = []
        for index in range(0, len(samples) + n_fft - hop, hop):
            inputs["wav_in"] = padded[index : index + hop].reshape(1, -1)
            result = self.enhancer.session.run(None, inputs)
            chunks.append(result[0][0])
            for cache_index, cache in enumerate(result[1:]):
                inputs[f"cache_in_{cache_index}"] = cache
        reference = np.concatenate(chunks)[n_fft - hop : n_fft - hop + len(samples)]
        actual = self.enhancer.enhance(samples)
        np.testing.assert_allclose(actual, reference, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
