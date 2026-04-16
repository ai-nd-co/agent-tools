from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, cast

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


def play_wav_blocking(
    wav_data: bytes,
    *,
    backend: str,
    output_path: Path | None = None,
) -> None:
    if backend == "winsound":
        _play_wav_with_winsound(wav_data)
        return

    temp_path: Path | None = None
    audio_path = output_path
    if audio_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(wav_data)
            temp_path = Path(handle.name)
        audio_path = temp_path
    else:
        audio_path.write_bytes(wav_data)

    try:
        _play_audio_file_blocking(audio_path, backend=backend)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _play_wav_with_winsound(wav_data: bytes) -> None:
    import winsound

    winsound_api = cast(Any, winsound)
    winsound_api.PlaySound(wav_data, winsound_api.SND_MEMORY)


def _play_audio_file_blocking(audio_path: Path, *, backend: str) -> None:
    command = _player_command(audio_path, backend=backend)
    process = subprocess.Popen(command)
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=5)
        raise
    if process.returncode:
        raise RuntimeError(
            f"{backend} exited with status {process.returncode} while playing {audio_path}."
        )


def _player_command(audio_path: Path, *, backend: str) -> list[str]:
    if backend == "afplay":
        return ["afplay", str(audio_path)]
    if backend == "paplay":
        return ["paplay", str(audio_path)]
    if backend == "aplay":
        return ["aplay", str(audio_path)]
    if backend == "ffplay":
        return ["ffplay", "-nodisp", "-autoexit", str(audio_path)]
    raise RuntimeError(f"Unsupported playback backend: {backend}")
