#!/usr/bin/env bash
set -euo pipefail
# This command documents a future deliberately scheduled reproduction.
# The accepted 2026-10-29 fixture attempt must not be automatically repeated.
DEVICE_ROOT="${DEVICE_ROOT:-${HOME}/voice_demo}"
PROJECT_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
exec nice -n 10 "$DEVICE_ROOT/venv/bin/python" "$PROJECT_ROOT/app/benchmark_multicandidate.py" \
  --gtcrn "$DEVICE_ROOT/models/gtcrn_simple.onnx" \
  --fastenhancer "$DEVICE_ROOT/models/fastenhancer_b_dns16k.onnx" \
  --sensevoice-dir "$DEVICE_ROOT/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09" \
  --japanese-model-dir "$DEVICE_ROOT/models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01" \
  --language en --wav "$DEVICE_ROOT/testdata/fastenhancer_official_p232_013.wav"
