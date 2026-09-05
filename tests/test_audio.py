from __future__ import annotations

import builtins
import sys
import types
import wave
from io import BytesIO

import numpy as np
import pytest

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
    monkeypatch.setattr(tts_module, "_create_kokoro_pipeline", lambda cls, **kwargs: cls(**kwargs))

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


def test_synthesize_wav_resolves_cpu_before_importing_kokoro(monkeypatch: object) -> None:
    import agent_tools.tts as tts_module

    events: list[str] = []

    def fake_resolve(_requested_device: str) -> DeviceResolution:
        events.append("resolve")
        return DeviceResolution(requested_device="cpu", resolved_device="cpu")

    def fake_load() -> object:
        events.append("import")
        raise RuntimeError("stop after ordering proof")

    monkeypatch.setattr(tts_module, "resolve_torch_device", fake_resolve)
    monkeypatch.setattr(tts_module, "_load_kokoro_pipeline", fake_load)

    with pytest.raises(RuntimeError, match="ordering proof"):
        synthesize_wav("hello", voice="af_heart", device="cpu")

    assert events == ["resolve", "import"]


def test_create_kokoro_pipeline_uses_offline_english_fallback(
    monkeypatch: object,
) -> None:
    import agent_tools.tts as tts_module

    captured: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, *, lang_code: str, repo_id: str, device: str) -> None:
            captured.update(lang_code=lang_code, repo_id=repo_id, device=device)
            self.g2p = None

    class FakeEspeakG2P:
        def __init__(self, *, language: str) -> None:
            captured["fallback_language"] = language

    monkeypatch.setattr(tts_module, "_has_spacy_english_model", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "misaki.espeak",
        types.SimpleNamespace(EspeakG2P=FakeEspeakG2P),
    )

    pipeline = tts_module._create_kokoro_pipeline(
        FakePipeline,
        lang_code="a",
        device="cpu",
    )

    assert captured == {
        "lang_code": "e",
        "repo_id": "hexgrad/Kokoro-82M",
        "device": "cpu",
        "fallback_language": "en-us",
    }
    assert isinstance(pipeline.g2p, FakeEspeakG2P)


def test_spacy_model_detection_uses_distribution_not_importable_module(
    monkeypatch: object,
) -> None:
    import agent_tools.tts as tts_module

    monkeypatch.setitem(sys.modules, "en_core_web_sm", types.SimpleNamespace())
    monkeypatch.setattr(
        tts_module.importlib_metadata,
        "distribution",
        lambda _name: (_ for _ in ()).throw(
            tts_module.importlib_metadata.PackageNotFoundError("en-core-web-sm")
        ),
    )

    assert tts_module._has_spacy_english_model() is False
