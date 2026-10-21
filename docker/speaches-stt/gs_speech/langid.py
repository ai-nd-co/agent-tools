"""Language ID only, served on its own port (default 8002) from the Whisper model already loaded.

Why: agent-studio's call worker needs P(language) for short clips within about 0.15 s. Loading its
own large-v3 for that would put a second copy of the same weights on the shared card, so it asks
this one instead and keeps a local Whisper-small as the fallback at its own deadline.

POST /v1/langid, multipart with one field `file`: content-type audio/pcm (or audio/raw) as 16 kHz
mono s16le, or a 16 kHz mono 16-bit PCM WAV. Other formats get 415: decoding them would cost more
than the detection and could not be bounded cheaply. Only the first 30 s are read. Detection is
faster-whisper's `WhisperModel.detect_language(audio)` with its defaults (no VAD, one 30 s segment,
default threshold), the same call agent-studio's worker makes locally, so local and remote
probabilities are comparable. GS_STT_LANGUAGES never applies here.

200: {"language", "probability", "probabilities": {code: p, ...all languages},
      "model", "revision", "compute_type", "audio_seconds",
      "queue_ms", "decode_ms", "compute_ms", "server_ms", "request_id"}
      (server_ms: from taking the slot to the answer)
503 busy:        GS_LID_MAX_OUTSTANDING (default 2) detections already in progress. The upload
                 is read first (bounded, see 408/411/413) without holding a slot; there is no
                 queue, the caller's deadline decides.
503 not loaded:  the model is not in memory (warm-up). This listener never loads a model: a load
                 is a 3 GB allocation, which only the warm-up may make.
408/411/413:     an upload slower than GS_LID_BODY_TIMEOUT_S (default 5 s), no Content-Length,
                 or a body over GS_LID_MAX_BYTES (default 1 MiB; refused before reading it).
400/415:         empty, misaligned or malformed audio; unsupported format.
401:             GS_LID_API_KEY is set and the Bearer token does not match.

Optional header X-Request-Id (up to 64 of A-Z a-z 0-9 . _ : -) is echoed as `request_id` and
logged; nothing else from the caller is logged, only the language, probability and timings.

No transcription happens here and no admission permit is taken: language ID runs on the second CT2
worker (WHISPER__NUM_WORKERS=2) beside at most one transcription job.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hmac
import io
import logging
import os
import re
import threading
import time
from typing import TYPE_CHECKING, Any
import wave

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

logger = logging.getLogger("gs_speech.langid")

_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")
SAMPLE_RATE = 16000
MAX_SAMPLES = 30 * SAMPLE_RATE


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
    """The Whisper model Speaches has loaded, borrowed without ever loading it."""

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
        """Yields the loaded WhisperModel, or None when it is not in memory.

        The check and the reference are taken under the model's own lock, so an unload cannot slip
        in between them, and Speaches' lazy loader (SelfDisposingModel.__enter__) is never called.
        """
        sdm = self._manager().loaded_models.get(self.model_id)
        if sdm is None:
            yield None
            return
        with sdm.rlock:
            model = sdm.model
            if model is not None:
                sdm._increment_ref()  # noqa: SLF001
        if model is None:
            yield None
            return
        try:
            yield model
        finally:
            sdm._decrement_ref()  # noqa: SLF001

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


def decode(data: bytes, content_type: str | None) -> np.ndarray:
    """16 kHz mono s16le, raw or in a WAV, first 30 s only, as float32. Raises HTTPException."""
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype in ("audio/pcm", "audio/raw"):
        if len(data) % 2:
            raise HTTPException(status_code=400, detail="PCM length is not a whole number of 16-bit samples.")
        pcm = data[: MAX_SAMPLES * 2]
    else:
        try:
            with wave.open(io.BytesIO(data), "rb") as w:
                fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype())
                if fmt != (1, 2, SAMPLE_RATE, "NONE"):
                    raise HTTPException(status_code=415, detail="Send 16 kHz mono 16-bit PCM (WAV or audio/pcm).")
                want = min(w.getnframes(), MAX_SAMPLES)
                pcm = w.readframes(want)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 - wave raises wave.Error, EOFError, RuntimeError, struct.error
            raise HTTPException(status_code=415, detail="Not a readable WAV; send 16 kHz mono 16-bit PCM.") from e
        if len(pcm) != want * 2:
            raise HTTPException(status_code=400, detail="WAV is shorter than its header says.")
    if not pcm:
        raise HTTPException(status_code=400, detail="Empty audio.")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def make_app(env: Mapping[str, str] | None = None, model: Any | None = None) -> FastAPI:  # noqa: ANN401
    env = os.environ if env is None else env
    if model is None:
        from speaches.dependencies import get_config

        cfg = get_config()
        model = SpeachesWhisper(env.get("GS_LID_MODEL") or env["STT_MODEL"], cfg.whisper.compute_type)
    slots = Slots(int(env.get("GS_LID_MAX_OUTSTANDING", "2")))
    # 1 MiB keeps an upload under Starlette's in-memory spool limit (30 s of PCM is 960,000 bytes).
    max_bytes = int(env.get("GS_LID_MAX_BYTES", str(1024 * 1024)))
    api_key = env.get("GS_LID_API_KEY") or None
    body_timeout = float(env.get("GS_LID_BODY_TIMEOUT_S", "5"))
    # One thread per slot, separate from Starlette's shared pool, so queued transcriptions can never
    # starve language ID of threads.
    pool = ThreadPoolExecutor(max_workers=slots.n, thread_name_prefix="langid")

    app = FastAPI(title="speech-stt language ID", openapi_url=None, docs_url=None, redoc_url=None)
    app.state.slots = slots

    def check_key(request: Request) -> None:
        if api_key is None:
            return
        got = request.headers.get("authorization", "")
        if not hmac.compare_digest(got.encode(), f"Bearer {api_key}".encode()):
            raise HTTPException(status_code=401, detail="Bearer token required.")

    def detect(data: bytes, content_type: str | None, queued_at: float, request_id: str | None) -> dict:
        started = time.perf_counter()
        audio = decode(data, content_type)
        decoded = time.perf_counter()
        with model.use() as whisper:
            if whisper is None:
                raise HTTPException(status_code=503, detail="Model not loaded yet.")
            t0 = time.perf_counter()
            language, probability, all_probs = whisper.detect_language(audio=audio)
            compute_ms = (time.perf_counter() - t0) * 1000
        seconds = len(audio) / SAMPLE_RATE
        result = {
            "language": language,
            "probability": probability,
            "probabilities": dict(all_probs),
            "model": model.model_id,
            "revision": model.revision,
            "compute_type": model.compute_type,
            "audio_seconds": round(seconds, 3),
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
            seconds,
            result["queue_ms"],
        )
        return result

    @app.get("/health")
    def health() -> JSONResponse:
        with model.use() as whisper:
            loaded = whisper is not None
        return JSONResponse({"status": "ok", "model_loaded": loaded, "outstanding": slots.used})

    @app.post("/v1/langid")
    async def langid(request: Request) -> JSONResponse:
        check_key(request)
        length = request.headers.get("content-length")
        if length is None or not length.isdigit():
            raise HTTPException(status_code=411, detail="Content-Length required.")
        if int(length) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload over {max_bytes} bytes.")
        # The body first, bounded in size and time, holding no slot: a stalled upload cannot
        # occupy one. Starlette caches it, so the form below parses from memory.
        try:
            body = await asyncio.wait_for(request.body(), timeout=body_timeout)
        except TimeoutError as e:
            raise HTTPException(status_code=408, detail="Upload too slow.") from e
        if len(body) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload over {max_bytes} bytes.")
        try:
            form = await request.form(max_files=1, max_fields=1)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 - malformed multipart
            raise HTTPException(status_code=400, detail="Malformed multipart body.") from e
        try:
            part = form.get("file")
            if part is None or isinstance(part, str):
                raise HTTPException(status_code=400, detail="Multipart field `file` required.")
            data = await part.read()
            content_type = part.content_type
        finally:
            await form.close()
        rid = request.headers.get("x-request-id")
        rid = rid if rid and _REQUEST_ID.fullmatch(rid) else None
        if not slots.try_take():
            return JSONResponse({"detail": "Busy.", "outstanding": slots.n}, status_code=503)
        # The slot is given back exactly once, by the done-callback of the thread's future: after
        # the work, or when it is cancelled before starting.
        try:
            future = pool.submit(detect, data, content_type, time.perf_counter(), rid)
        except BaseException:
            slots.give_back()
            raise
        future.add_done_callback(lambda _f: slots.give_back())
        return JSONResponse(await asyncio.wrap_future(future))

    return app
