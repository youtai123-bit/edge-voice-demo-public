#!/usr/bin/env bash
set -euo pipefail
# Manual alternate entry for a prepared Pi. Do not run beside an existing
# service using the same ALSA capture device or port. No auto-stop/restart.
PROJECT_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
DEVICE_ROOT="${DEVICE_ROOT:-${HOME}/voice_demo}"
exec "$DEVICE_ROOT/venv/bin/python" "$PROJECT_ROOT/app/voice_demo_multicandidate.py" \
  --device plughw:U192k,0 \
  --gtcrn "$DEVICE_ROOT/models/gtcrn_simple.onnx" \
  --fastenhancer "$DEVICE_ROOT/models/fastenhancer_b_dns16k.onnx" \
  --vad-model "$DEVICE_ROOT/models/silero_vad.onnx" \
  --sensevoice-dir "$DEVICE_ROOT/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09" \
  --japanese-model-dir "$DEVICE_ROOT/models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01" \
  --log-dir "$PROJECT_ROOT/logs" --min-silence 0.45 \
  --asr-threads 1 --denoise-threads 1 --fastenhancer-threads 1
