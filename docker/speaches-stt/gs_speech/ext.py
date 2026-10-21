"""Admission control for Whisper and a model allowlist.

Why: on the shared 5060 Ti every extra Whisper job or model is extra GPU memory, and an
over-committed card gives a CUDA out-of-memory error to whichever process allocates last, which
may be another team's service. So:

- GS_WHISPER_PERMITS (default 1): how many transcription/translation jobs run at once. Others wait
  their turn, as they did with one CT2 worker. With WHISPER__NUM_WORKERS=2 this leaves one CT2
  worker free for language ID (gs_speech.langid), which does not take a permit.
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

from contextlib import contextmanager
import functools
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


class Gate:
    """A counting semaphore that also reports how many jobs run and wait."""

    def __init__(self, permits: int) -> None:
        if permits < 1:
            raise ValueError(f"GS_WHISPER_PERMITS must be at least 1, got {permits}")
        self.permits = permits
        self._sem = threading.BoundedSemaphore(permits)
        self._lock = threading.Lock()
        self.running = 0
        self.waiting = 0

    @contextmanager
    def hold(self, what: str) -> Iterator[None]:
        start = time.perf_counter()
        with self._lock:
            self.waiting += 1
        try:
            self._sem.acquire()
        finally:
            with self._lock:
                self.waiting -= 1
        with self._lock:
            self.running += 1
        waited = time.perf_counter() - start
        if waited >= 0.5:
            logger.info("whisper admission: %s waited %.2f s for a permit", what, waited)
        try:
            yield
        finally:
            with self._lock:
                self.running -= 1
            self._sem.release()


def gated(fn: Callable, gate: Gate, what: str) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        with gate.hold(what):
            return fn(*args, **kwargs)

    return wrapper


def gated_generator(fn: Callable, gate: Gate, what: str) -> Callable:
    """The permit is taken at the first item and held until the stream ends or is closed."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Generator:  # noqa: ANN002, ANN003
        with gate.hold(what):
            yield from fn(*args, **kwargs)

    return wrapper


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


def refuse_unless_allowed(allow: frozenset[str] | None, model_id: str) -> None:
    if allow is None:
        return
    if model_id in allow or resolve_alias(model_id) in allow:
        return
    logger.warning("model allowlist: refused %r", model_id)
    raise HTTPException(status_code=403, detail=f"Model '{model_id}' is not served here.")


class State:
    def __init__(self, gate: Gate, allow: frozenset[str] | None, langs: frozenset[str] | None) -> None:
        self.gate = gate
        self.allow = allow
        self.langs = langs


STATE: State | None = None


def install(env: Mapping[str, str] | None = None) -> State:
    """Wrap Speaches' classes. Call once, before the app is created. A second call is a no-op."""
    global STATE  # noqa: PLW0603
    env = os.environ if env is None else env

    from speaches.executors.shared.base_model_manager import BaseModelManager
    from speaches.executors.shared.registry import ExecutorRegistry
    from speaches.executors.whisper import WhisperModelManager

    if getattr(WhisperModelManager, _INSTALLED_MARK, False):
        assert STATE is not None
        return STATE

    gate = Gate(int(env.get("GS_WHISPER_PERMITS", "1")))
    allow = parse_allowlist(env.get("GS_MODEL_ALLOWLIST"))
    langs = parse_language_set(env.get("GS_STT_LANGUAGES"))

    m = WhisperModelManager
    if langs is not None:
        from faster_whisper import WhisperModel

        WhisperModel.detect_language = restricting_detect(WhisperModel.detect_language)
        m.handle_non_streaming_transcription_request = with_languages(m.handle_non_streaming_transcription_request, langs)
        m.handle_streaming_transcription_request = with_languages_generator(
            m.handle_streaming_transcription_request, langs
        )
    # The permit is outermost: it is taken before any GPU work, language detection included.
    m.handle_non_streaming_transcription_request = gated(
        m.handle_non_streaming_transcription_request, gate, "transcription"
    )
    m.handle_streaming_transcription_request = gated_generator(
        m.handle_streaming_transcription_request, gate, "streaming transcription"
    )
    m.handle_translation_request = gated(m.handle_translation_request, gate, "translation")

    if allow is not None:
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
    STATE = State(gate, allow, langs)
    logger.info(
        "gs_speech: whisper permits %d; model allowlist %s; transcription languages %s",
        gate.permits,
        "off" if allow is None else sorted(allow),
        "any" if langs is None else sorted(langs),
    )
    return STATE
