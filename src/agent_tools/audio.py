from __future__ import annotations

import io
import wave

import numpy as np

KOKORO_SAMPLE_RATE = 24_000


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(mono, -1.0, 1.0)


def concat_audio(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    normalized = [normalize_audio(chunk) for chunk in chunks]
    return np.concatenate(normalized)


def pcm16_bytes(audio: np.ndarray) -> bytes:
    normalized = normalize_audio(audio)
    return (normalized * 32767.0).astype(np.int16).tobytes()


def wav_bytes(audio: np.ndarray, sample_rate: int = KOKORO_SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_bytes(audio))
    return buffer.getvalue()
