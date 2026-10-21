"""Language ID only, served on its own port (default 8002) from the Whisper model already loaded.

Why: agent-studio's call worker needs P(language) for short clips within about 0.15 s. Loading its
own large-v3 for that would put a second copy of the same weights on the shared card, so it asks
this one instead and keeps a local Whisper-small as the fallback at its own deadline.

POST /v1/langid, multipart field `file`: 16 kHz mono 16-bit WAV, or `audio/pcm` / `audio/raw` as
16 kHz mono s16le (both read directly, fast), or any other format ffmpeg reads (~100 ms slower).
Only the first 30 s count. Detection is faster-whisper's `WhisperModel.detect_language(audio)` with
its defaults (no VAD, one 30 s segment, default threshold), the same call agent-studio's worker
makes locally, so local and remote probabilities are comparable. GS_STT_LANGUAGES never applies here.
Optional header X-Request-Id (up to 64 of A-Z a-z 0-9 . _ : -) is echoed as `request_id` and logged;
nothing else from the caller is logged, only the language, probability and timings.

200: {"language", "probability", "probabilities": {code: p, ...all languages},
      "model", "revision", "compute_type", "audio_seconds",
      "queue_ms", "decode_ms", "compute_ms", "server_ms"}   (server_ms: from the slot to the answer)
503 busy:        more than GS_LID_MAX_OUTSTANDING (default 2) requests already in progress.
503 not loaded:  the model is not in memory yet (warm-up). This listener never loads a model:
                 a load is a 3 GB allocation, which only the warm-up may make.
400/415:         empty or undecodable audio. 413: upload over GS_LID_MAX_BYTES (default 2 MiB).
401:             GS_LID_API_KEY is set and the Bearer token does not match.

No transcription happens here and no permit from gs_speech.ext is taken: language ID runs on the
second CT2 worker (WHISPER__NUM_WORKERS=2) beside at most one transcription job.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hmac
import logging
import os
import re
import threading
import time
from typing import TYPE_CHECKING, Any
import wave

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
import numpy as np
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

logger = logging.getLogger("gs_speech.langid")

_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")


class Slots:
    """A non-blocking cap: a request either gets a slot now or is turned away."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"GS_LID_MAX_OUTSTANDING must be at least 1, got {n}")
        self.n = n
        self.used = 0
        self._lock = threading.Lock()

    def try_take(self) -> bool:
        with self._lock:
            if self.used >= self.n:
                return False
            self.used += 1
            return True

    def give_back(self) -> None:
        with self._lock:
            self.used -= 1


class SpeachesWhisper:
    """The Whisper model Speaches has loaded, found without loading anything."""

    def __init__(self, model_id: str, compute_type: str) -> None:
        self.model_id = model_id
        self.compute_type = compute_type
        self._revision: str | None = None

    def _manager(self) -> Any:  # noqa: ANN401
        from speaches.dependencies import get_executor_registry

        executor = get_executor_registry().translation[0]
        assert executor.name == "whisper", executor.name
        return executor.model_manager

    @contextmanager
    def use(self) -> Iterator[Any | None]:
        """Yields the loaded WhisperModel, or None when it is not in memory."""
        sdm = self._manager().loaded_models.get(self.model_id)
        if sdm is None or sdm.model is None:
            yield None
            return
        with sdm as whisper:  # holds a reference, so an idle unload cannot happen mid-request
            yield whisper

    @property
    def revision(self) -> str:
        if self._revision is None:
            try:
                from huggingface_hub import try_to_load_from_cache

                path = try_to_load_from_cache(self.model_id, "model.bin")
                self._revision = os.path.basename(os.path.dirname(path)) if isinstance(path, str) else "unknown"
            except Exception:  # noqa: BLE001
                self._revision = "unknown"
        return self._revision


def decode(file: UploadFile) -> Any:  # noqa: ANN401
    """16 kHz mono 16-bit WAV is read directly; anything else goes through Speaches' decoder.

    Measured in the server: Speaches' decoder (PyAV) took ~117 ms for a 4 s WAV, more than the
    language detection itself (~75 ms). Raw `audio/pcm` was already a direct read in Speaches.
    """
    from speaches.audio import Audio
    from speaches.dependencies import audio_file_dependency

    if file.content_type not in ("audio/pcm", "audio/raw"):
        try:
            with wave.open(file.file, "rb") as w:
                if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) == (1, 2, 16000, "NONE"):
                    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
                    return Audio(pcm.astype(np.float32) / 32768.0, sample_rate=16000)
        except (wave.Error, EOFError):
            pass
        file.file.seek(0)
    return audio_file_dependency(file)


def make_app(
    env: Mapping[str, str] | None = None,
    model: Any | None = None,  # noqa: ANN401 - SpeachesWhisper or a test double
    decoder: Callable[[UploadFile], Any] = decode,
) -> FastAPI:
    env = os.environ if env is None else env
    if model is None:
        from speaches.dependencies import get_config

        cfg = get_config()
        model = SpeachesWhisper(env.get("GS_LID_MODEL") or env["STT_MODEL"], cfg.whisper.compute_type)
    slots = Slots(int(env.get("GS_LID_MAX_OUTSTANDING", "2")))
    max_bytes = int(env.get("GS_LID_MAX_BYTES", str(2 * 1024 * 1024)))
    api_key = env.get("GS_LID_API_KEY") or None
    # One thread per slot, separate from Starlette's shared pool, so queued transcriptions waiting
    # for their permit can never starve language ID of threads.
    pool = ThreadPoolExecutor(max_workers=slots.n, thread_name_prefix="langid")

    app = FastAPI(title="speech-stt language ID", openapi_url=None, docs_url=None, redoc_url=None)
    app.state.slots = slots

    def check_key(request: Request) -> None:
        if api_key is None:
            return
        got = request.headers.get("authorization", "")
        if not hmac.compare_digest(got.encode(), f"Bearer {api_key}".encode()):
            raise HTTPException(status_code=401, detail="Bearer token required.")

    def detect(file: UploadFile, queued_at: float, request_id: str | None) -> dict:
        started = time.perf_counter()
        audio = decoder(file)
        decoded = time.perf_counter()
        if audio.duration <= 0:
            raise HTTPException(status_code=400, detail="Empty audio.")
        with model.use() as whisper:
            if whisper is None:
                raise HTTPException(status_code=503, detail="Model not loaded yet.")
            t0 = time.perf_counter()
            language, probability, all_probs = whisper.detect_language(audio=audio.data)
            compute_ms = (time.perf_counter() - t0) * 1000
        result = {
            "language": language,
            "probability": probability,
            "probabilities": dict(all_probs),
            "model": model.model_id,
            "revision": model.revision,
            "compute_type": model.compute_type,
            "audio_seconds": round(audio.duration, 3),
            "queue_ms": round((started - queued_at) * 1000, 1),
            "decode_ms": round((decoded - started) * 1000, 1),
            "compute_ms": round(compute_ms, 1),
            "server_ms": round((time.perf_counter() - queued_at) * 1000, 1),
            "request_id": request_id,
        }
        logger.info(
            "langid%s: %s %.3f in %.0f ms (audio %.2f s, queued %.0f ms)",
            f" [{request_id}]" if request_id else "",
            language,
            probability,
            compute_ms,
            audio.duration,
            result["queue_ms"],
        )
        return result

    @app.get("/health")
    def health() -> JSONResponse:
        with model.use() as whisper:
            loaded = whisper is not None
        return JSONResponse({"status": "ok", "model_loaded": loaded, "outstanding": slots.used})

    @app.post("/v1/langid")
    async def langid(request: Request, file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
        check_key(request)
        if file.size is not None and file.size > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload over {max_bytes} bytes.")
        if not slots.try_take():
            return JSONResponse({"detail": "Busy.", "outstanding": slots.n}, status_code=503)
        # The slot is given back by the thread when the work ends, not here: a client that hangs
        # up cancels this await but not the GPU work, which must keep counting against the cap.
        try:
            rid = request.headers.get("x-request-id")
            rid = rid if rid and _REQUEST_ID.fullmatch(rid) else None
            future = asyncio.get_running_loop().run_in_executor(pool, run, file, time.perf_counter(), rid)
        except BaseException:
            slots.give_back()
            raise
        return JSONResponse(await future)

    def run(file: UploadFile, queued_at: float, request_id: str | None) -> dict:
        try:
            return detect(file, queued_at, request_id)
        finally:
            slots.give_back()

    return app
