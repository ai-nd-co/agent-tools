"""Admission control for Whisper, a model allowlist, and a language set for auto-detection.

Why: on the shared 5060 Ti every extra Whisper job or model is extra GPU memory, and an
over-committed card gives a CUDA out-of-memory error to whichever process allocates last, which
may be another team's service. So:

- GS_WHISPER_PERMITS (default 1): how many transcription/translation requests run at once; the
  rest wait their turn, as they did behind one CT2 worker. With WHISPER__NUM_WORKERS=2 this leaves
  one CT2 worker free for language ID (gs_speech.langid), which does not take a permit.
  Admission (below) is an ASGI wrapper. It reads the whole request body first, holding nothing,
  so a slow upload from a phone never blocks other requests. Only then does it wait for a permit,
  in the event loop, so waiting requests use no worker thread. Requests without the right API key
  get 403 from Admission itself, by Speaches' own rule (Bearer, any case, exact token). In-process callers that
  bypass HTTP (Speaches' realtime loopback) are not admitted. This cluster routes no realtime
  path, and with this image's FastAPI that loopback fails before reaching Whisper anyway.
- GS_WHISPER_BATCH_SIZE: how many 30 s windows of one recording Whisper decodes at once (faster-
  whisper's default is 8). Measured at int8: one 26-minute file took the container from 2,706 to
  5,746 MiB for the 17 s of its decode at 8. On a shared card the transient is the budget, so the
  deployment sets a smaller value; only long recordings get slower. Unset means faster-whisper's.
- GS_MAX_BODY_BYTES (default 32 MiB), GS_MAX_PENDING (16), GS_UPLOAD_TIMEOUT_S (120),
  GS_SEND_TIMEOUT_S (60): the bounds Admission puts on what it buffers and waits for (see the
  class). 180 s of 16 kHz mono WAV is 5.8 MB.
- Model administration over HTTP (POST/PUT/PATCH/DELETE /api/ps/..., DELETE /v1/models/...) gets
  403. This service runs a fixed set of models, and upstream's unload removes a model from the
  registry before checking it is idle, so an unload during a request lets the next load make a
  second copy. GET /api/ps (the probe) and POST /v1/models/... (the warm-up's download,
  allowlisted) still work.
- GS_MODEL_ALLOWLIST: space- or comma-separated model ids that may be loaded or downloaded. Any
  other id gets 403 instead of a new model in GPU memory or on the volume. Unset or empty means no
  allowlist (upstream behaviour). Speaches' own VAD (silero_vad_v5) is always allowed.
- GS_STT_LANGUAGES: space- or comma-separated language codes. When a transcription request names
  no language, Whisper's detection still runs as before, but the language picked is the most
  probable one IN THIS SET. Unset or empty means any language (upstream behaviour). Why: at int8,
  one English phone cut of 134 came out as Latvian (lv 0.32, en 0.26, lt 0.21; float16 picked
  English) and every word of it was lost. A request that names a language is never changed, and
  the language-ID listener always reports the unrestricted probabilities.

Nothing here edits a Speaches file. Patch 433 rewrites whisper.py at start and refuses to run if
the file's hash changed, so these wrap the classes' methods after that has happened.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import functools
import hmac
import inspect
import logging
import os
import re
import threading
import time
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Mapping

logger = logging.getLogger("gs_speech")

# Speaches runs its own Silero VAD before every transcription; refusing it would refuse everything.
INTERNAL_MODELS = frozenset({"silero_vad_v5"})

_INSTALLED_MARK = "_gs_speech_installed"


def parse_allowlist(raw: str | None) -> frozenset[str] | None:
    if raw is None or not raw.strip():
        return None
    return frozenset(x for x in re.split(r"[\s,]+", raw.strip()) if x) | INTERNAL_MODELS


def resolve_alias(model_id: str) -> str:
    try:
        from speaches.model_aliases import resolve_model_id_alias

        return resolve_model_id_alias(model_id)
    except Exception:  # noqa: BLE001 - no alias file means no aliases
        return model_id


def parse_language_set(raw: str | None) -> frozenset[str] | None:
    if raw is None or not raw.strip():
        return None
    return frozenset(x.lower() for x in re.split(r"[\s,]+", raw.strip()) if x)


# Set only while a transcription runs on this thread, so the language-ID listener, which calls the
# same detect_language, always gets Whisper's own unrestricted answer.
_restrict = threading.local()


@contextmanager
def restricted_languages(langs: frozenset[str] | None) -> Iterator[None]:
    prev = getattr(_restrict, "langs", None)
    _restrict.langs = langs
    try:
        yield
    finally:
        _restrict.langs = prev


def pick_language(
    detected: tuple[str, float, list[tuple[str, float]]], langs: frozenset[str] | None
) -> tuple[str, float, list[tuple[str, float]]]:
    language, probability, all_probs = detected
    if not langs or language in langs:
        return detected
    allowed = [(code, p) for code, p in all_probs if code in langs]
    if not allowed:
        return detected
    code, p = max(allowed, key=lambda x: x[1])
    logger.info("language set %s: detected %s %.3f, using %s %.3f", sorted(langs), language, probability, code, p)
    return code, p, all_probs


def restricting_detect(orig: Callable) -> Callable:
    """Wraps WhisperModel.detect_language: unchanged unless a transcription set a language set."""

    @functools.wraps(orig)
    def detect_language(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        return pick_language(orig(self, *args, **kwargs), getattr(_restrict, "langs", None))

    return detect_language


def with_languages(fn: Callable, langs: frozenset[str]) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        with restricted_languages(langs):
            return fn(*args, **kwargs)

    return wrapper


def with_languages_generator(fn: Callable, langs: frozenset[str]) -> Callable:
    """Per item, because a streamed response may resume on a different thread each time."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Generator:  # noqa: ANN002, ANN003
        it = fn(*args, **kwargs)
        try:
            while True:
                with restricted_languages(langs):
                    try:
                        item = next(it)
                    except StopIteration:
                        return
                yield item
        finally:
            it.close()

    return wrapper


def with_batch_size(orig: Callable, batch_size: int) -> Callable:
    """Wraps BatchedInferencePipeline.transcribe: batch_size unless the caller gave one, by name or
    by position."""
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def transcribe(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if "batch_size" not in sig.bind_partial(self, *args, **kwargs).arguments:
            kwargs["batch_size"] = batch_size
        return orig(self, *args, **kwargs)

    return transcribe


def refuse_unless_allowed(allow: frozenset[str] | None, model_id: str) -> None:
    if allow is None:
        return
    if model_id in allow or resolve_alias(model_id) in allow:
        return
    logger.warning("model allowlist: refused %r", model_id)
    raise HTTPException(status_code=403, detail=f"Model '{model_id}' is not served here.")


ADMITTED_PATHS = ("/v1/audio/transcriptions", "/v1/audio/translations")


def is_model_admin(method: str, path: str) -> bool:
    if path.startswith("/api/ps/") and method in ("POST", "DELETE", "PUT", "PATCH"):
        return True
    return path.startswith("/v1/models/") and method == "DELETE"


async def _reply(send, status: int, detail: bytes, extra: list | None = None) -> None:  # noqa: ANN001
    headers = [(b"content-type", b"application/json"), *(extra or [])]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": b'{"detail":"' + detail + b'"}'})


REPLAY_CHUNK = 64 * 1024


class Admission:
    """ASGI wrapper for Speaches: body first, then a permit, for Whisper requests; the admin block.

    For POST /v1/audio/transcriptions and /translations with the right key:
    1. At most GS_MAX_PENDING (default 16) such requests are in progress, from upload to answer;
       more get 503 at once. With GS_MAX_BODY_BYTES (default 32 MiB) that bounds buffered bodies
       to 512 MiB in the worst case.
    2. The body is read whole, holding no permit, within GS_UPLOAD_TIMEOUT_S (default 120 s; 408).
    3. The permit is awaited in the event loop while watching the client: a request whose client
       has gone is dropped instead of being transcribed for nobody.
    4. Speaches gets the body back in 64 KiB pieces (one big piece would make Starlette's form
       parser copy it to disk synchronously, on the event loop), and every send must finish within
       GS_SEND_TIMEOUT_S (default 60 s), so a client that stops reading cannot hold the permit.
    The permit is released on every exit path (the `finally` blocks).
    """

    def __init__(  # noqa: PLR0913
        self,
        app,  # noqa: ANN001
        permits: int,
        api_key: str | None,
        max_body: int,
        max_pending: int = 16,
        upload_timeout: float = 120.0,
        send_timeout: float = 60.0,
    ) -> None:
        if permits < 1:
            raise ValueError(f"GS_WHISPER_PERMITS must be at least 1, got {permits}")
        self.app = app
        self.permits = permits
        self.max_body = max_body
        self.max_pending = max_pending
        self.upload_timeout = upload_timeout
        self.send_timeout = send_timeout
        self._key = api_key.encode() if api_key else None
        self._sem: asyncio.Semaphore | None = None
        self.pending = 0
        self.running = 0

    def _authorized(self, scope) -> bool:  # noqa: ANN001
        """Speaches' rule (FastAPI HTTPBearer): scheme `Bearer` in any case, then the exact token."""
        if self._key is None:
            return True
        got = next((v for k, v in scope.get("headers", []) if k == b"authorization"), b"")
        scheme, _, token = got.partition(b" ")
        return scheme.lower() == b"bearer" and hmac.compare_digest(token.strip(), self._key)

    async def _read_body(self, receive) -> bytes | None:  # noqa: ANN001
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return None
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_body:
                raise _TooLarge
            chunks.append(chunk)
            if not message.get("more_body", False):
                body = b"".join(chunks)
                chunks.clear()
                return body

    async def _acquire_unless_gone(self, receive) -> bool:  # noqa: ANN001
        """True with the permit held; False (and no permit) if the client disconnected first.

        A disconnect wins even when the permit arrived in the same instant. The cleanup has no
        await in it, so a cancellation anywhere here cannot strand a granted permit: a pending
        acquisition is cancelled (asyncio's Semaphore passes on a grant it loses that way), and a
        completed one that is not handed to the caller is released.
        """
        acquire = asyncio.ensure_future(self._sem.acquire())
        watch = asyncio.ensure_future(receive())
        owned = False
        try:
            await asyncio.wait({acquire, watch}, return_when=asyncio.FIRST_COMPLETED)
            gone = (
                watch.done()
                and not watch.cancelled()
                and watch.exception() is None
                and watch.result().get("type") == "http.disconnect"
            )
            owned = acquire.done() and not acquire.cancelled() and acquire.exception() is None and not gone
            return owned
        finally:
            watch.cancel()
            if not owned:
                if acquire.done():
                    if not acquire.cancelled() and acquire.exception() is None:
                        self._sem.release()
                else:
                    acquire.cancel()

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001, C901
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method, path = scope["method"], scope["path"]
        if is_model_admin(method, path):
            await _reply(send, 403, b"Model administration is disabled on this service.")
            return
        if method != "POST" or path not in ADMITTED_PATHS:
            await self.app(scope, receive, send)
            return
        if not self._authorized(scope):
            await _reply(send, 403, b"API key required or invalid.", [(b"www-authenticate", b"Bearer")])
            return
        if self.pending >= self.max_pending:
            await _reply(send, 503, b"Too many requests in progress.", [(b"retry-after", b"1")])
            return
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.permits)

        async def timed_send(message) -> None:  # noqa: ANN001
            await asyncio.wait_for(send(message), timeout=self.send_timeout)

        self.pending += 1
        try:
            try:
                body = await asyncio.wait_for(self._read_body(receive), timeout=self.upload_timeout)
            except _TooLarge:
                await _reply(timed_send, 413, b"Request body too large.")
                return
            except TimeoutError:
                await _reply(timed_send, 408, b"Upload too slow.")
                return
            if body is None:
                return
            pieces = [body[i : i + REPLAY_CHUNK] for i in range(0, len(body), REPLAY_CHUNK)] or [b""]
            del body

            async def replay():  # noqa: ANN202
                if pieces:
                    piece = pieces.pop(0)
                    return {"type": "http.request", "body": piece, "more_body": bool(pieces)}
                return await receive()  # afterwards the real channel, so a disconnect is still seen

            start = time.perf_counter()
            if not await self._acquire_unless_gone(receive):
                return
            self.running += 1
            try:
                waited = time.perf_counter() - start
                if waited >= 0.5:
                    logger.info("whisper admission: %s waited %.2f s", path, waited)
                await self.app(scope, replay, timed_send)
            finally:
                self.running -= 1
                self._sem.release()
        finally:
            self.pending -= 1


class _TooLarge(Exception):  # noqa: N818
    pass


class State:
    def __init__(self, permits: int, limits: dict, allow: frozenset[str] | None, langs: frozenset[str] | None) -> None:
        self.permits = permits
        self.limits = limits
        self.allow = allow
        self.langs = langs


STATE: State | None = None


def install(env: Mapping[str, str] | None = None) -> State:
    """Wrap Speaches' classes. Call once, before serving. A second call is a no-op."""
    global STATE  # noqa: PLW0603
    env = os.environ if env is None else env

    from speaches.executors.shared.base_model_manager import BaseModelManager
    from speaches.executors.shared.registry import ExecutorRegistry
    from speaches.executors.whisper import WhisperModelManager

    if getattr(WhisperModelManager, _INSTALLED_MARK, False):
        assert STATE is not None
        return STATE

    permits = int(env.get("GS_WHISPER_PERMITS", "1"))
    if permits < 1:
        raise ValueError(f"GS_WHISPER_PERMITS must be at least 1, got {permits}")
    limits = {
        "max_body": int(env.get("GS_MAX_BODY_BYTES", str(32 * 1024 * 1024))),
        "max_pending": int(env.get("GS_MAX_PENDING", "16")),
        "upload_timeout": float(env.get("GS_UPLOAD_TIMEOUT_S", "120")),
        "send_timeout": float(env.get("GS_SEND_TIMEOUT_S", "60")),
    }
    allow = parse_allowlist(env.get("GS_MODEL_ALLOWLIST"))
    langs = parse_language_set(env.get("GS_STT_LANGUAGES"))

    batch_size = int(env["GS_WHISPER_BATCH_SIZE"]) if env.get("GS_WHISPER_BATCH_SIZE") else None
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"GS_WHISPER_BATCH_SIZE must be at least 1, got {batch_size}")
        from faster_whisper import BatchedInferencePipeline

        BatchedInferencePipeline.transcribe = with_batch_size(BatchedInferencePipeline.transcribe, batch_size)

    m = WhisperModelManager
    if langs is not None:
        from faster_whisper import WhisperModel

        WhisperModel.detect_language = restricting_detect(WhisperModel.detect_language)
        m.handle_non_streaming_transcription_request = with_languages(m.handle_non_streaming_transcription_request, langs)
        m.handle_streaming_transcription_request = with_languages_generator(
            m.handle_streaming_transcription_request, langs
        )

    if allow is not None:
        # Before the stream starts, so a refused model gets a 403 rather than a broken 200 stream.
        orig_dispatch = m.handle_transcription_request

        @functools.wraps(orig_dispatch)
        def handle_transcription_request(self, request, **kwargs):  # noqa: ANN001, ANN003, ANN202
            refuse_unless_allowed(allow, request.model)
            return orig_dispatch(self, request, **kwargs)

        m.handle_transcription_request = handle_transcription_request

        orig_load = BaseModelManager.load_model
        orig_download = ExecutorRegistry.download_model_by_id

        @functools.wraps(orig_load)
        def load_model(self, model_id: str):  # noqa: ANN001, ANN202
            refuse_unless_allowed(allow, model_id)
            return orig_load(self, model_id)

        @functools.wraps(orig_download)
        def download_model_by_id(self, model_id: str) -> bool:  # noqa: ANN001
            refuse_unless_allowed(allow, model_id)
            return orig_download(self, model_id)

        BaseModelManager.load_model = load_model
        ExecutorRegistry.download_model_by_id = download_model_by_id

    setattr(WhisperModelManager, _INSTALLED_MARK, True)
    STATE = State(permits, limits, allow, langs)
    logger.info(
        "gs_speech: whisper permits %d; batch size %s; limits %s; model allowlist %s; transcription languages %s",
        permits,
        batch_size or "default",
        limits,
        "off" if allow is None else sorted(allow),
        "any" if langs is None else sorted(langs),
    )
    return STATE
