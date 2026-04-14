from __future__ import annotations

import wave
from io import BytesIO

import numpy as np

from agent_tools.audio import KOKORO_SAMPLE_RATE, concat_audio, wav_bytes
from agent_tools.tts import infer_language_from_voice


def test_concat_audio_merges_chunks() -> None:
    merged = concat_audio([np.array([0.0, 0.1]), np.array([0.2], dtype=np.float32)])
    assert np.allclose(merged, np.array([0.0, 0.1, 0.2], dtype=np.float32))


def test_wav_bytes_produces_valid_wave_file() -> None:
    payload = wav_bytes(np.array([0.0, 0.2, -0.2], dtype=np.float32))
    with wave.open(BytesIO(payload), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == KOKORO_SAMPLE_RATE
        assert wav_file.getsampwidth() == 2


def test_infer_language_from_voice() -> None:
    assert infer_language_from_voice("af_heart") == "a"
