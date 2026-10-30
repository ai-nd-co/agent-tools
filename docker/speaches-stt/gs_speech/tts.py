"""Kokoro text-to-speech on the CPU, so it takes no GPU memory.

Why: Speaches runs Kokoro through onnxruntime with the CUDA provider first, on the card it shares
with Whisper and two other services. Measured 2026-10-30: one five-sentence English paragraph grew
Kokoro's CUDA arena by about 3.6 GB and then failed to allocate a 1.1 GB buffer, and the arena
never shrinks. A gpu_mem_limit of 2 GiB broke paragraphs outright. On the CPU, with 4 intra-op
threads, the same 67-word paragraph (21.4 s of audio) took 3.2 s and a 4-word line 0.2 s. With
onnxruntime's default threads it took 22 s: the container has 4 CPUs on a 32-core host, and one
thread per host core gets throttled.

- GS_KOKORO_THREADS (default 4): onnxruntime intra-op threads for Kokoro's session, which runs on
  CPUExecutionProvider only, with one inter-op thread. 0 turns this off: Speaches' own providers
  (CUDA first on this image) and default threads.

How: Speaches builds Kokoro's session in KokoroModelManager._load_fn, calling the name
InferenceSession from speaches.executors.kokoro. install() rebinds that one module global, so
nothing else changes: Whisper is CTranslate2, and Silero VAD and the other ONNX executors use
their own modules' names. If that module is not shaped as expected, install() logs a warning and
leaves Speaches as it is; it never stops the service from starting. No Speaches file is edited.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

logger = logging.getLogger("gs_speech")

DEFAULT_THREADS = 4
CPU_PROVIDERS = ["CPUExecutionProvider"]
_INSTALLED_MARK = "_gs_speech_kokoro_cpu"


def parse_threads(raw: str | None) -> int:
    if raw is None or not raw.strip():
        return DEFAULT_THREADS
    threads = int(raw)
    if threads < 0:
        raise ValueError(f"GS_KOKORO_THREADS must be 0 (off) or more, got {threads}")
    return threads


def mismatch(module: Any, ort: Any) -> str | None:
    """Why install() cannot apply to this Speaches, or None when it can."""
    session = getattr(module, "InferenceSession", None)
    if session is None:
        return "speaches.executors.kokoro has no InferenceSession"
    if session is not ort.InferenceSession:
        return f"speaches.executors.kokoro.InferenceSession is {session!r}, not onnxruntime's"
    load_fn = getattr(getattr(module, "KokoroModelManager", None), "_load_fn", None)
    code = getattr(load_fn, "__code__", None)
    if code is None:
        return "speaches.executors.kokoro has no KokoroModelManager._load_fn"
    if "InferenceSession" not in code.co_names:
        return "KokoroModelManager._load_fn no longer calls InferenceSession"
    return None


def cpu_session(orig: Callable, session_options: Callable, threads: int) -> Callable:
    """Wraps InferenceSession: CPU provider only, `threads` intra-op threads, one inter-op."""

    @functools.wraps(orig)
    def make(path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):  # noqa: ANN001, ANN003, ANN202
        so = session_options() if sess_options is None else sess_options
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        logger.info(
            "kokoro: CPU session, %d threads (Speaches asked for %s)",
            threads,
            [p[0] if isinstance(p, tuple) else p for p in providers or []],
        )
        return orig(path_or_bytes, sess_options=so, providers=list(CPU_PROVIDERS), **kwargs)

    return make


def install(
    env: Mapping[str, str] | None = None, module: ModuleType | None = None, ort: Any = None
) -> bool:
    """Put Kokoro's session on the CPU. True if installed (now or before); False if left alone."""
    env = os.environ if env is None else env
    threads = parse_threads(env.get("GS_KOKORO_THREADS"))
    if threads == 0:
        logger.info("gs_speech: Kokoro on Speaches' own providers (GS_KOKORO_THREADS=0)")
        return False
    try:
        if ort is None:
            import onnxruntime as ort
        if module is None:
            from speaches.executors import kokoro as module
    except Exception as e:  # noqa: BLE001 - never stop the service over TTS placement
        logger.warning("gs_speech: Kokoro left on Speaches' providers: import failed: %r", e)
        return False
    if getattr(module, _INSTALLED_MARK, False):
        return True
    reason = mismatch(module, ort)
    if reason is not None:
        logger.warning("gs_speech: Kokoro left on Speaches' providers: %s", reason)
        return False
    module.InferenceSession = cpu_session(module.InferenceSession, ort.SessionOptions, threads)
    setattr(module, _INSTALLED_MARK, True)
    logger.info("gs_speech: Kokoro on CPUExecutionProvider, %d intra-op threads", threads)
    return True
