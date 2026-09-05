from __future__ import annotations

import hashlib
import io
import wave
from contextlib import nullcontext
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest

from agent_tools.cuda_runtime import DeviceResolution
from agent_tools.tts import resolve_tts_options


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _use_small_model_identity(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    import agent_tools.silero_tts as silero_module

    monkeypatch.setattr(silero_module, "SILERO_MODEL_SIZE", len(payload))
    monkeypatch.setattr(
        silero_module,
        "SILERO_MODEL_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )


def test_model_download_is_atomic_and_integrity_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_tools.silero_tts as silero_module

    payload = b"official-silero-model"
    _use_small_model_identity(monkeypatch, payload)
    monkeypatch.setattr(
        silero_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(payload),
    )

    model_path = silero_module.ensure_silero_model(cache_dir=tmp_path)

    assert model_path.read_bytes() == payload
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_valid_cached_model_works_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_tools.silero_tts as silero_module

    payload = b"cached-model"
    _use_small_model_identity(monkeypatch, payload)
    model_path = tmp_path / silero_module.SILERO_MODEL_FILENAME
    model_path.write_bytes(payload)
    monkeypatch.setattr(
        silero_module,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("valid cache must not access the network"),
    )

    assert silero_module.ensure_silero_model(cache_dir=tmp_path) == model_path


def test_concurrent_download_reuses_valid_model_that_won_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_tools.silero_tts as silero_module

    payload = b"official-silero-model"
    _use_small_model_identity(monkeypatch, payload)
    monkeypatch.setattr(
        silero_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(payload),
    )
    target_path = tmp_path / silero_module.SILERO_MODEL_FILENAME

    def lose_replace_race(_source: Path, _target: Path) -> None:
        target_path.write_bytes(payload)
        raise PermissionError("target is loaded by the winning process")

    monkeypatch.setattr(silero_module.os, "replace", lose_replace_race)

    assert silero_module.ensure_silero_model(cache_dir=tmp_path) == target_path
    assert target_path.read_bytes() == payload
    assert list(tmp_path.glob(".*.tmp")) == []


def test_corrupt_cached_model_has_actionable_error(tmp_path: Path) -> None:
    import agent_tools.silero_tts as silero_module

    (tmp_path / silero_module.SILERO_MODEL_FILENAME).write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="integrity check failed.*Remove that file"):
        silero_module.ensure_silero_model(cache_dir=tmp_path)


def test_download_failure_leaves_no_partial_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_tools.silero_tts as silero_module

    monkeypatch.setattr(
        silero_module,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")),
    )

    with pytest.raises(RuntimeError, match="official Silero v5_5_ru"):
        silero_module.ensure_silero_model(cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_silero_synthesis_reports_truthful_pcm_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_tools.silero_tts as silero_module

    captured: dict[str, object] = {}

    class FakeModel:
        def apply_tts(self, **kwargs: object) -> np.ndarray:
            captured.update(kwargs)
            return np.array([0.0, 0.25, -0.25], dtype=np.float32)

    class FakeTorch:
        @staticmethod
        def inference_mode() -> nullcontext[None]:
            return nullcontext()

    model_path = tmp_path / "v5_5_ru.pt"
    model_path.write_bytes(b"model")
    monkeypatch.setattr(silero_module, "_load_torch", lambda: FakeTorch())
    monkeypatch.setattr(silero_module, "ensure_silero_model", lambda: model_path)
    monkeypatch.setattr(
        silero_module,
        "_load_silero_model_cached",
        lambda _path, _device: FakeModel(),
    )
    options = resolve_tts_options("Привет, мир!")

    result = silero_module.synthesize_silero_wav(
        "Привет, мир!",
        options=options,
        device_resolution=DeviceResolution(
            requested_device="cpu",
            resolved_device="cpu",
        ),
    )

    assert result.engine == "silero"
    assert result.model == "snakers4/silero-models:v5_5_ru"
    assert result.voice == "xenia"
    assert result.language == "ru"
    assert result.sample_rate == 24_000
    assert result.chunks == 1
    assert captured["put_accent"] is True
    assert captured["put_stress_homo"] is True
    with wave.open(io.BytesIO(result.wav), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 24_000
        assert wav_file.getnframes() == 3


def test_silero_synthesis_chunks_long_text_and_wraps_model_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_tools.silero_tts as silero_module

    captured_texts: list[str] = []

    class FakeModel:
        def apply_tts(self, **kwargs: object) -> np.ndarray:
            text = str(kwargs["text"])
            captured_texts.append(text)
            if len(text) > 500:
                raise Exception("text is too long")
            return np.array([0.0, 0.25], dtype=np.float32)

    class FakeTorch:
        @staticmethod
        def inference_mode() -> nullcontext[None]:
            return nullcontext()

    model_path = tmp_path / "v5_5_ru.pt"
    model_path.write_bytes(b"model")
    monkeypatch.setattr(silero_module, "_load_torch", lambda: FakeTorch())
    monkeypatch.setattr(silero_module, "ensure_silero_model", lambda: model_path)
    monkeypatch.setattr(
        silero_module,
        "_load_silero_model_cached",
        lambda _path, _device: FakeModel(),
    )
    text = "Это длинное предложение для проверки разбиения. " * 30

    result = silero_module.synthesize_silero_wav(
        text,
        options=resolve_tts_options(text),
        device_resolution=DeviceResolution(
            requested_device="cpu",
            resolved_device="cpu",
        ),
    )

    assert result.chunks == len(captured_texts)
    assert result.chunks > 1
    assert all(0 < len(chunk) <= 500 for chunk in captured_texts)

    class BrokenModel:
        def apply_tts(self, **_kwargs: object) -> np.ndarray:
            raise Exception("upstream base exception")

    monkeypatch.setattr(
        silero_module,
        "_load_silero_model_cached",
        lambda _path, _device: BrokenModel(),
    )
    with pytest.raises(RuntimeError, match="chunk 1 of .*upstream base exception"):
        silero_module.synthesize_silero_wav(
            "Короткий русский текст.",
            options=resolve_tts_options("Короткий русский текст."),
            device_resolution=DeviceResolution(
                requested_device="cpu",
                resolved_device="cpu",
            ),
        )


def test_silero_rejects_non_finite_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_tools.silero_tts as silero_module

    class FakeModel:
        def apply_tts(self, **_kwargs: object) -> np.ndarray:
            return np.array([0.0, np.nan], dtype=np.float32)

    class FakeTorch:
        @staticmethod
        def inference_mode() -> nullcontext[None]:
            return nullcontext()

    model_path = tmp_path / "v5_5_ru.pt"
    model_path.write_bytes(b"model")
    monkeypatch.setattr(silero_module, "_load_torch", lambda: FakeTorch())
    monkeypatch.setattr(silero_module, "ensure_silero_model", lambda: model_path)
    monkeypatch.setattr(
        silero_module,
        "_load_silero_model_cached",
        lambda _path, _device: FakeModel(),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        silero_module.synthesize_silero_wav(
            "Привет, мир!",
            options=resolve_tts_options("Привет, мир!"),
            device_resolution=DeviceResolution(
                requested_device="cpu",
                resolved_device="cpu",
            ),
        )
