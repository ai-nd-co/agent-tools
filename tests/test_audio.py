from __future__ import annotations

import sys
import types
import wave
from io import BytesIO

import numpy as np

from agent_tools.audio import KOKORO_SAMPLE_RATE, concat_audio, wav_bytes
from agent_tools.cuda_runtime import DeviceResolution
from agent_tools.tts import infer_language_from_voice, synthesize_wav


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


def test_synthesize_wav_uses_resolved_device(monkeypatch: object) -> None:
    import agent_tools.tts as tts_module

    captured: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, *, lang_code: str, device: str) -> None:
            captured["lang_code"] = lang_code
            captured["device"] = device

        def __call__(self, _text: str, *, voice: str, speed: float, split_pattern: str):
            captured["voice"] = voice
            captured["speed"] = speed
            captured["split_pattern"] = split_pattern
            yield (None, None, np.array([0.0, 0.1], dtype=np.float32))

    monkeypatch.setattr(
        tts_module,
        "resolve_torch_device",
        lambda requested_device: DeviceResolution(
            requested_device=requested_device,
            resolved_device="cpu",
            probe_ms=1.25,
            fallback_reason="cuda probe failed",
        ),
    )
    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KPipeline=FakePipeline))

    result = synthesize_wav("hello there", voice="af_heart", device="auto")

    assert captured["lang_code"] == "a"
    assert captured["device"] == "cpu"
    assert captured["voice"] == "af_heart"
    assert result.requested_device == "auto"
    assert result.resolved_device == "cpu"
    assert result.device_fallback_reason == "cuda probe failed"
    assert result.metrics.device_probe_ms == 1.25
