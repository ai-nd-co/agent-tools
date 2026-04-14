from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agent_tools.audio import KOKORO_SAMPLE_RATE, concat_audio, wav_bytes

SUPPORTED_LANGUAGES = ("a", "b", "e", "f", "h", "i", "j", "p", "z")


@dataclass(frozen=True)
class TtsResult:
    wav: bytes
    sample_rate: int
    chunks: int


def synthesize_wav(
    text: str,
    *,
    voice: str,
    language: str | None = None,
    speed: float = 1.0,
    device: str = "auto",
) -> TtsResult:
    if not text.strip():
        raise ValueError("Input text is empty.")
    lang_code = language or infer_language_from_voice(voice)
    if lang_code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported Kokoro language code: {lang_code}")

    from kokoro import KPipeline

    pipeline_device = None if device == "auto" else device
    pipeline = KPipeline(lang_code=lang_code, device=pipeline_device)
    audio_chunks: list[np.ndarray] = []

    for result in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        chunk = _extract_audio_chunk(result)
        if chunk is not None:
            audio_chunks.append(chunk)

    if not audio_chunks:
        raise RuntimeError("Kokoro did not produce any audio chunks.")

    merged = concat_audio(audio_chunks)
    return TtsResult(
        wav=wav_bytes(merged, sample_rate=KOKORO_SAMPLE_RATE),
        sample_rate=KOKORO_SAMPLE_RATE,
        chunks=len(audio_chunks),
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
