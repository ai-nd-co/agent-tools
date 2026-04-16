from __future__ import annotations

import builtins
import types
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from agent_tools.audio import KOKORO_SAMPLE_RATE, concat_audio, play_wav_blocking, wav_bytes
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


def test_synthesize_wav_reports_actionable_kokoro_import_error(
    monkeypatch: object,
) -> None:
    import agent_tools.tts as tts_module

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "kokoro":
            raise ModuleNotFoundError("broken kokoro import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="Re-run `agent-tools install-cuda`"):
        tts_module._load_kokoro_pipeline()


def test_play_wav_blocking_uses_winsound_backend(monkeypatch: object) -> None:
    import agent_tools.audio as audio_module

    captured: dict[str, object] = {}

    class FakeWinSound:
        SND_MEMORY = 1

        @staticmethod
        def PlaySound(data: bytes, flags: int) -> None:
            captured["data"] = data
            captured["flags"] = flags

    monkeypatch.setitem(__import__("sys").modules, "winsound", FakeWinSound)

    play_wav_blocking(b"WAV", backend="winsound")

    assert captured == {"data": b"WAV", "flags": 1}


def test_play_wav_blocking_uses_external_player_for_unix_backend(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.audio as audio_module

    output_path = tmp_path / "out.wav"
    captured: dict[str, object] = {}

    class FakeProcess:
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            captured["wait_timeout"] = timeout
            return 0

    def fake_popen(command: list[str]) -> FakeProcess:
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(audio_module.subprocess, "Popen", fake_popen)

    play_wav_blocking(b"WAV", backend="ffplay", output_path=output_path)

    assert output_path.read_bytes() == b"WAV"
    assert captured["command"] == ["ffplay", "-nodisp", "-autoexit", str(output_path)]


def test_play_wav_blocking_raises_when_player_fails(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.audio as audio_module

    class FakeProcess:
        returncode = 1

        def wait(self, timeout: float | None = None) -> int:
            return 1

    monkeypatch.setattr(audio_module.subprocess, "Popen", lambda _command: FakeProcess())

    with pytest.raises(RuntimeError, match="ffplay exited with status 1"):
        play_wav_blocking(b"WAV", backend="ffplay", output_path=tmp_path / "out.wav")
