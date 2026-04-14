from __future__ import annotations

import io
import sys
import time
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


def wav_duration_ms(wav_data: bytes) -> int:
    with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    if rate <= 0:
        return 0
    return int((frames / rate) * 1000)


def play_wav_blocking(wav_data: bytes) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Playback mode is currently supported only on Windows.")

    import winsound

    duration_s = wav_duration_ms(wav_data) / 1000.0
    try:
        winsound.PlaySound(wav_data, winsound.SND_MEMORY | winsound.SND_ASYNC)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            time.sleep(0.05)
    except KeyboardInterrupt:
        winsound.PlaySound(None, 0)
        raise
    finally:
        winsound.PlaySound(None, 0)
