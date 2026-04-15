from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from agent_tools.audio import KOKORO_SAMPLE_RATE, concat_audio, wav_bytes
from agent_tools.cuda_runtime import resolve_torch_device

SUPPORTED_LANGUAGES = ("a", "b", "e", "f", "h", "i", "j", "p", "z")


@dataclass(frozen=True)
class TtsResult:
    wav: bytes
    sample_rate: int
    chunks: int
    requested_device: str = "auto"
    resolved_device: str = "cpu"
    device_fallback_reason: str | None = None
    metrics: TtsMetrics = field(default_factory=lambda: TtsMetrics())


@dataclass(frozen=True)
class TtsMetrics:
    device_probe_ms: float = 0.0
    import_ms: float = 0.0
    pipeline_init_ms: float = 0.0
    generation_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    text_chars: int = 0


def synthesize_wav(
    text: str,
    *,
    voice: str,
    language: str | None = None,
    speed: float = 1.0,
    device: str = "auto",
) -> TtsResult:
    total_started = perf_counter()
    if not text.strip():
        raise ValueError("Input text is empty.")
    lang_code = language or infer_language_from_voice(voice)
    if lang_code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported Kokoro language code: {lang_code}")

    import_started = perf_counter()
    from kokoro import KPipeline
    import_ms = (perf_counter() - import_started) * 1000.0

    device_resolution = resolve_torch_device(device)
    pipeline_device = device_resolution.resolved_device
    pipeline_started = perf_counter()
    pipeline = KPipeline(lang_code=lang_code, device=pipeline_device)
    pipeline_init_ms = (perf_counter() - pipeline_started) * 1000.0
    audio_chunks: list[np.ndarray] = []

    generation_started = perf_counter()
    for result in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        chunk = _extract_audio_chunk(result)
        if chunk is not None:
            audio_chunks.append(chunk)
    generation_ms = (perf_counter() - generation_started) * 1000.0

    if not audio_chunks:
        raise RuntimeError("Kokoro did not produce any audio chunks.")

    postprocess_started = perf_counter()
    merged = concat_audio(audio_chunks)
    wav = wav_bytes(merged, sample_rate=KOKORO_SAMPLE_RATE)
    postprocess_ms = (perf_counter() - postprocess_started) * 1000.0
    total_ms = (perf_counter() - total_started) * 1000.0
    return TtsResult(
        wav=wav,
        sample_rate=KOKORO_SAMPLE_RATE,
        chunks=len(audio_chunks),
        requested_device=device_resolution.requested_device,
        resolved_device=device_resolution.resolved_device,
        device_fallback_reason=device_resolution.fallback_reason,
        metrics=TtsMetrics(
            device_probe_ms=device_resolution.probe_ms,
            import_ms=import_ms,
            pipeline_init_ms=pipeline_init_ms,
            generation_ms=generation_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
            text_chars=len(text),
        ),
    )


def infer_language_from_voice(voice: str) -> str:
    normalized = voice.strip().lower()
    if not normalized:
        raise ValueError("Voice cannot be empty.")
    lang_code = normalized[0]
    if lang_code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Could not infer Kokoro language from voice: {voice}")
    return lang_code


def _extract_audio_chunk(result: Any) -> np.ndarray | None:
    audio = getattr(result, "audio", None)
    if audio is None and isinstance(result, tuple) and len(result) >= 3:
        audio = result[2]
    if audio is None:
        return None
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)
