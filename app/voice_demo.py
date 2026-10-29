#!/usr/bin/env python3
"""Single-microphone GTCRN + Silero VAD + SenseVoice demo for Raspberry Pi 5.

Audio is captured from INPUT 1 (left/channel 0) of the UMC202HD.  This program
never opens an ALSA playback device and therefore cannot emit control audio.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx


SAMPLE_RATE = 16_000
CAPTURE_CHANNELS = 2


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    phase: str = "starting"
    message: str = "正在加载模型"
    speech_active: bool = False
    level_dbfs: float = -120.0
    latest_text: str = ""
    latest_latency_ms: int | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    utterance_count: int = 0
    dropped_utterances: int = 0
    error: str = ""

    def update(self, **values: Any) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)

    def append_result(self, item: dict[str, Any]) -> None:
        with self.lock:
            self.latest_text = item["text"]
            self.latest_latency_ms = item["latency_ms"]
            self.utterance_count += 1
            self.transcript.append(item)
            self.transcript = self.transcript[-100:]
            self.phase = "listening"
            self.message = "请继续说话"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "phase": self.phase,
                "message": self.message,
                "speech_active": self.speech_active,
                "level_dbfs": round(self.level_dbfs, 1),
                "latest_text": self.latest_text,
                "latest_latency_ms": self.latest_latency_ms,
                "transcript": list(self.transcript),
                "utterance_count": self.utterance_count,
                "dropped_utterances": self.dropped_utterances,
                "uptime_s": round(time.time() - self.started_at, 1),
                "error": self.error,
            }


@dataclass
class Utterance:
    index: int
    samples: np.ndarray
    vad_start_sample: int
    endpoint_monotonic: float


def find_sensevoice_files(model_dir: Path) -> tuple[Path, Path]:
    model_candidates = sorted(model_dir.rglob("*int8*.onnx"))
    if not model_candidates:
        model_candidates = sorted(model_dir.rglob("*.onnx"))
    token_candidates = sorted(model_dir.rglob("tokens.txt"))
    if not model_candidates or not token_candidates:
        raise FileNotFoundError(
            f"SenseVoice model/tokens not found below {model_dir}"
        )
    return model_candidates[0], token_candidates[0]


def make_denoiser(model_path: Path, num_threads: int) -> sherpa_onnx.OnlineSpeechDenoiser:
    gtcrn = sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
        model=str(model_path)
    )
    model = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
        gtcrn=gtcrn,
        num_threads=num_threads,
        provider="cpu",
        debug=False,
    )
    return sherpa_onnx.OnlineSpeechDenoiser(
        sherpa_onnx.OnlineSpeechDenoiserConfig(model=model)
    )


def make_vad(
    model_path: Path,
    min_silence_s: float,
    threshold: float,
) -> sherpa_onnx.VoiceActivityDetector:
    silero = sherpa_onnx.SileroVadModelConfig(
        model=str(model_path),
        threshold=threshold,
        min_silence_duration=min_silence_s,
        min_speech_duration=0.25,
        max_speech_duration=20.0,
        window_size=512,
    )
    config = sherpa_onnx.VadModelConfig(
        silero_vad=silero,
        sample_rate=SAMPLE_RATE,
        num_threads=1,
        provider="cpu",
        debug=False,
    )
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)


def make_recognizer(
    sensevoice_dir: Path,
    num_threads: int,
    language: str,
) -> tuple[sherpa_onnx.OfflineRecognizer, Path, Path]:
    model, tokens = find_sensevoice_files(sensevoice_dir)
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model),
        tokens=str(tokens),
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
        language=language,
        use_itn=True,
        debug=False,
    )
    return recognizer, model, tokens


def save_pcm16_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


class RecognizerWorker(threading.Thread):
    def __init__(
        self,
        recognizer: sherpa_onnx.OfflineRecognizer,
        work_queue: queue.Queue[Utterance],
        state: SharedState,
        log_dir: Path,
        save_utterances: bool,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="sensevoice-worker", daemon=True)
        self.recognizer = recognizer
        self.work_queue = work_queue
        self.state = state
        self.log_dir = log_dir
        self.save_utterances = save_utterances
        self.stop_event = stop_event

    def run(self) -> None:
        transcript_path = self.log_dir / "transcript.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        while not self.stop_event.is_set():
            try:
                utterance = self.work_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self.state.update(phase="recognizing", message="正在识别…")
            started = time.perf_counter()
            try:
                stream = self.recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, utterance.samples)
                self.recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                decode_ms = round((time.perf_counter() - started) * 1000)
                endpoint_ms = round(
                    (time.monotonic() - utterance.endpoint_monotonic) * 1000
                )
                item = {
                    "index": utterance.index,
                    "time": utc_timestamp(),
                    "text": text,
                    "duration_s": round(len(utterance.samples) / SAMPLE_RATE, 2),
                    "decode_ms": decode_ms,
                    "latency_ms": endpoint_ms,
                }
                if text:
                    self.state.append_result(item)
                    with transcript_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    self.state.update(
                        phase="listening",
                        message="未识别到文字，请继续说话",
                        latest_latency_ms=endpoint_ms,
                    )
                if self.save_utterances:
                    save_pcm16_wav(
                        self.log_dir / "utterances" / f"utt_{utterance.index:05d}.wav",
                        utterance.samples,
                    )
                logging.info(
                    "utterance=%d duration=%.2fs decode=%dms endpoint=%dms text=%r",
                    utterance.index,
                    len(utterance.samples) / SAMPLE_RATE,
                    decode_ms,
                    endpoint_ms,
                    text,
                )
            except Exception as exc:  # keep capture alive if one utterance fails
                logging.exception("SenseVoice decode failed")
                self.state.update(
                    phase="error",
                    message="识别失败，采集仍在继续",
                    error=str(exc),
                )
            finally:
                self.work_queue.task_done()


class AudioPipeline(threading.Thread):
    def __init__(
        self,
        args: argparse.Namespace,
        state: SharedState,
        work_queue: queue.Queue[Utterance],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="audio-pipeline", daemon=True)
        self.args = args
        self.state = state
        self.work_queue = work_queue
        self.stop_event = stop_event
        self.process: subprocess.Popen[bytes] | None = None

    def _enqueue_segments(
        self,
        vad: sherpa_onnx.VoiceActivityDetector,
        next_index: int,
    ) -> int:
        while not vad.empty():
            segment = vad.front
            samples = np.asarray(segment.samples, dtype=np.float32).copy()
            utterance = Utterance(
                index=next_index,
                samples=samples,
                vad_start_sample=int(segment.start),
                endpoint_monotonic=time.monotonic(),
            )
            vad.pop()
            try:
                self.work_queue.put_nowait(utterance)
                next_index += 1
            except queue.Full:
                snap = self.state.snapshot()
                self.state.update(
                    dropped_utterances=snap["dropped_utterances"] + 1,
                    message="识别队列忙，已丢弃一段",
                )
                logging.warning("ASR queue full; dropped utterance")
        return next_index

    def run(self) -> None:
        denoiser = make_denoiser(self.args.gtcrn, self.args.denoise_threads)
        vad = make_vad(
            self.args.vad_model,
            self.args.min_silence,
            self.args.vad_threshold,
        )
        frame_samples = int(denoiser.frame_shift_in_samples)
        frame_bytes = frame_samples * CAPTURE_CHANNELS * 2
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
            "512",
            "--buffer-size",
            "2048",
        ]
        logging.info("capture command: %s", " ".join(command))
        next_index = 1
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
                # Detach the immutable frame before resizing ``pending``;
                # np.frombuffer(bytearray) otherwise keeps an exported view.
                frame_data = bytes(pending)
                pending.clear()
                stereo = np.frombuffer(frame_data, dtype="<i2").reshape(-1, 2)
                raw = stereo[:, 0].astype(np.float32) / 32768.0

                denoised = denoiser.run(raw, SAMPLE_RATE)
                enhanced = np.asarray(denoised.samples, dtype=np.float32)
                if enhanced.size:
                    vad.accept_waveform(enhanced)
                    next_index = self._enqueue_segments(vad, next_index)

                now = time.monotonic()
                if now - last_level_update >= 0.1:
                    rms = float(np.sqrt(np.mean(raw * raw) + 1e-12))
                    dbfs = max(-120.0, 20.0 * np.log10(rms))
                    speaking = bool(vad.is_speech_detected())
                    self.state.update(
                        speech_active=speaking,
                        level_dbfs=dbfs,
                        phase="speaking" if speaking else self.state.snapshot()["phase"],
                        message="正在聆听…" if speaking else self.state.snapshot()["message"],
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
                tail = denoiser.flush()
                if len(tail.samples):
                    vad.accept_waveform(tail.samples)
                vad.flush()
                self._enqueue_segments(vad, next_index)
            except Exception:
                logging.exception("failed to flush audio pipeline")
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>语音增强转写演示</title>
  <style>
    :root { color-scheme: dark; --bg:#07111f; --card:#102137; --ink:#f2f7ff;
      --muted:#91a4bd; --cyan:#4ee6c1; --blue:#5aa7ff; --red:#ff6876; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; background:radial-gradient(circle at 20% 0,#183957 0,transparent 42%),var(--bg);
      color:var(--ink); font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
    main { width:min(920px,92vw); margin:0 auto; padding:7vh 0 60px; }
    .eyebrow { color:var(--cyan); letter-spacing:.16em; font-size:13px; font-weight:700; }
    h1 { font-size:clamp(34px,6vw,66px); line-height:1.05; margin:13px 0 22px; }
    .card { border:1px solid #29435f; border-radius:24px; background:#102137dd; box-shadow:0 20px 70px #0007; padding:clamp(22px,4vw,42px); }
    .status { display:flex; gap:13px; align-items:center; color:var(--muted); }
    .dot { width:13px; height:13px; border-radius:99px; background:var(--cyan); box-shadow:0 0 18px var(--cyan); }
    .speaking .dot { animation:pulse .8s infinite alternate; }
    .error .dot { background:var(--red); box-shadow:0 0 18px var(--red); }
    @keyframes pulse { to { transform:scale(1.7); opacity:.45; } }
    #latest { min-height:2.6em; font-size:clamp(25px,4vw,42px); line-height:1.35; margin:32px 0; }
    .meter { height:10px; background:#07111f; border-radius:99px; overflow:hidden; }
    .meter > div { height:100%; width:0; background:linear-gradient(90deg,var(--blue),var(--cyan)); transition:width .12s; }
    .meta { display:flex; justify-content:space-between; color:var(--muted); font-size:13px; margin-top:10px; }
    h2 { margin:38px 0 14px; font-size:18px; }
    .history { display:grid; gap:10px; }
    .line { border-left:3px solid #355c82; padding:10px 14px; background:#0c1a2b99; border-radius:0 12px 12px 0; }
    .line small { color:var(--muted); display:block; margin-top:5px; }
    .empty { color:var(--muted); }
  </style>
</head>
<body><main>
  <div class="eyebrow">GTCRN · SILERO VAD · SENSEVOICE</div>
  <h1>语音增强<br>实时转写</h1>
  <section class="card" id="card">
    <div class="status"><span class="dot"></span><span id="message">正在连接树莓派…</span></div>
    <div id="latest">说完一句话并稍作停顿，文字会出现在这里。</div>
    <div class="meter"><div id="bar"></div></div>
    <div class="meta"><span id="level">输入电平 -- dBFS</span><span id="latency">等待首句</span></div>
  </section>
  <h2>本次转写</h2>
  <div class="history" id="history"><div class="empty">还没有识别结果</div></div>
</main>
<script>
let lastCount=-1;
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
async function refresh(){
  try {
    const r=await fetch('/api/state',{cache:'no-store'}); const s=await r.json();
    document.getElementById('card').className='card '+s.phase;
    document.getElementById('message').textContent=s.message;
    document.getElementById('level').textContent=`输入电平 ${s.level_dbfs.toFixed(1)} dBFS`;
    document.getElementById('bar').style.width=`${Math.max(0,Math.min(100,(s.level_dbfs+70)*1.8))}%`;
    document.getElementById('latency').textContent=s.latest_latency_ms==null?'等待首句':`停顿后 ${s.latest_latency_ms} ms`;
    if(s.latest_text) document.getElementById('latest').textContent=s.latest_text;
    if(s.utterance_count!==lastCount){
      lastCount=s.utterance_count;
      document.getElementById('history').innerHTML=s.transcript.length?s.transcript.slice().reverse().map(x=>
        `<div class="line">${esc(x.text)}<small>${esc(x.time)} · ${x.duration_s}s 语音 · ${x.decode_ms}ms 识别</small></div>`
      ).join(''):'<div class="empty">还没有识别结果</div>';
    }
  } catch(e){ document.getElementById('message').textContent='与树莓派断开连接，正在重试…'; }
}
setInterval(refresh,250); refresh();
</script></body></html>"""


class DemoHandler(BaseHTTPRequestHandler):
    state: SharedState

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
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

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.debug("http: " + fmt, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="plughw:U192k,0")
    parser.add_argument("--gtcrn", type=Path, required=True)
    parser.add_argument("--vad-model", type=Path, required=True)
    parser.add_argument("--sensevoice-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--min-silence", type=float, default=0.45)
    parser.add_argument("--vad-threshold", type=float, default=0.50)
    parser.add_argument("--denoise-threads", type=int, default=1)
    parser.add_argument("--asr-threads", type=int, default=2)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--save-utterances", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_dir / "voice_demo.log", encoding="utf-8"),
        ],
    )
    os.environ.setdefault("OMP_NUM_THREADS", str(args.asr_threads))
    state = SharedState()
    stop_event = threading.Event()
    work_queue: queue.Queue[Utterance] = queue.Queue(maxsize=4)

    logging.info("loading SenseVoice from %s", args.sensevoice_dir)
    recognizer, asr_model, tokens = make_recognizer(
        args.sensevoice_dir, args.asr_threads, args.language
    )
    logging.info("SenseVoice model=%s tokens=%s", asr_model, tokens)

    worker = RecognizerWorker(
        recognizer,
        work_queue,
        state,
        args.log_dir,
        args.save_utterances,
        stop_event,
    )
    pipeline = AudioPipeline(args, state, work_queue, stop_event)
    worker.start()
    pipeline.start()

    DemoHandler.state = state
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
