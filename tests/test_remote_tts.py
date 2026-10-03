from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from typing import Any
from urllib import error

import pytest

from agent_tools import remote_tts, tts
from agent_tools.remote_tts import (
    DEFAULT_REMOTE_EN_MODEL,
    DEFAULT_REMOTE_RU_MODEL,
    DEFAULT_REMOTE_RU_VOICE,
    DEFAULT_SPEECH_BASE_URL,
    RemoteTtsError,
    RemoteTtsSettings,
    claude_tools_secrets_path,
    load_remote_tts_settings,
    normalize_tts_backend,
    remote_model_and_voice,
    synthesize_remote_wav,
)
from agent_tools.tts import TtsResult, resolve_tts_options, synthesize_wav

KEY = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"


def _wav(sample_rate: int = 24_000, frames: int = 2_400, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x01" * frames * channels)
    return buffer.getvalue()


@pytest.fixture
def secrets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CLAUDE_TOOLS_CONFIG_DIR", str(tmp_path))
    for name in (
        "SPEECH_API_KEY",
        "SPEECH_BASE_URL",
        "AGENT_TOOLS_TTS_BACKEND",
        "AGENT_TOOLS_REMOTE_TTS_EN_MODEL",
        "AGENT_TOOLS_REMOTE_TTS_RU_MODEL",
        "AGENT_TOOLS_REMOTE_TTS_RU_VOICE",
        "AGENT_TOOLS_REMOTE_TTS_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _write_secrets(directory: Path, speech: dict[str, Any] | None) -> None:
    payload: dict[str, Any] = {"groq": {"apiKey": "gsk_x"}}
    if speech is not None:
        payload["speech"] = speech
    (directory / "secrets.json").write_text(json.dumps(payload), encoding="utf-8")


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._offset = 0

    def read(self, limit: int | None = None) -> bytes:
        end = len(self._body) if limit is None else min(len(self._body), self._offset + limit)
        chunk = self._body[self._offset : end]
        self._offset = end
        return chunk

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _capture_urlopen(monkeypatch: pytest.MonkeyPatch, body: bytes = b"") -> list[Any]:
    calls: list[Any] = []

    def fake_urlopen(req: Any, *, timeout: float | None = None) -> _FakeResponse:
        calls.append((req, timeout))
        return _FakeResponse(body or _wav())

    monkeypatch.setattr(remote_tts, "_open", fake_urlopen)
    return calls


# --- settings ---------------------------------------------------------------------


def test_secrets_path_follows_claude_tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_TOOLS_CONFIG_DIR", str(tmp_path))
    assert claude_tools_secrets_path() == tmp_path / "secrets.json"


def test_settings_absent_without_a_key(secrets_dir: Path) -> None:
    _write_secrets(secrets_dir, None)
    assert load_remote_tts_settings() is None


def test_settings_from_the_file_with_cluster_defaults(secrets_dir: Path) -> None:
    _write_secrets(secrets_dir, {"apiKey": KEY})
    settings = load_remote_tts_settings()
    assert settings is not None
    assert settings.api_key == KEY
    assert settings.base_url == DEFAULT_SPEECH_BASE_URL
    assert settings.en_model == DEFAULT_REMOTE_EN_MODEL
    assert settings.ru_model == DEFAULT_REMOTE_RU_MODEL == "silero-v5-ru"
    assert settings.ru_voice == DEFAULT_REMOTE_RU_VOICE == "eugene"


def test_settings_env_wins_and_url_is_canonical(
    secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_secrets(secrets_dir, {"apiKey": "x" * 40, "baseUrl": "https://file.example/v1"})
    monkeypatch.setenv("SPEECH_API_KEY", KEY)
    monkeypatch.setenv("SPEECH_BASE_URL", "http://127.0.0.1:18000/v1/")
    monkeypatch.setenv("AGENT_TOOLS_REMOTE_TTS_RU_VOICE", "ruslan")
    settings = load_remote_tts_settings()
    assert settings is not None
    assert settings.api_key == KEY
    assert settings.base_url == "http://127.0.0.1:18000/v1"
    assert settings.ru_voice == "ruslan"


def test_short_key_counts_as_absent(secrets_dir: Path) -> None:
    _write_secrets(secrets_dir, {"apiKey": "short"})
    assert load_remote_tts_settings() is None


@pytest.mark.parametrize(
    "url", ["speech.example/v1", "https://speech.example", "https://speech.example/v1?x=1"]
)
def test_bad_base_url_is_refused(secrets_dir: Path, url: str) -> None:
    _write_secrets(secrets_dir, {"apiKey": KEY, "baseUrl": url})
    with pytest.raises(RemoteTtsError, match="ending in /v1"):
        load_remote_tts_settings()


def test_backend_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_TOOLS_TTS_BACKEND", raising=False)
    assert normalize_tts_backend(None) == "auto"
    monkeypatch.setenv("AGENT_TOOLS_TTS_BACKEND", "local")
    assert normalize_tts_backend(None) == "local"
    assert normalize_tts_backend("REMOTE") == "remote"
    with pytest.raises(ValueError, match="Unsupported TTS backend"):
        normalize_tts_backend("cloud")


# --- model mapping --------------------------------------------------------------------


def test_russian_goes_to_silero_route_and_english_keeps_its_kokoro_voice() -> None:
    settings = RemoteTtsSettings(
        base_url="https://s/v1",
        api_key=KEY,
        en_model="k",
        ru_model="silero-v5-ru",
        ru_voice="eugene",
        timeout_seconds=1,
    )
    ru = resolve_tts_options("Привет, это тест.", engine="auto")
    assert remote_model_and_voice(ru, settings) == ("silero-v5-ru", "eugene")
    assert remote_tts.remote_path(ru) == "/audio/speech-ru"
    # An explicit Silero voice is honoured on the cluster: same engine there.
    ru_voice = resolve_tts_options("Привет, это тест.", engine="auto", voice="aidar")
    assert remote_model_and_voice(ru_voice, settings, explicit_voice="aidar") == (
        "silero-v5-ru",
        "aidar",
    )
    en = resolve_tts_options("Hello there.", engine="auto", voice="am_adam")
    assert remote_model_and_voice(en, settings) == ("k", "am_adam")
    assert remote_tts.remote_path(en) == "/audio/speech"


# --- the request ------------------------------------------------------------------------


def test_request_shape_and_result(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_urlopen(monkeypatch, _wav(sample_rate=22_050, frames=1_000))
    settings = RemoteTtsSettings(
        base_url="https://s/v1",
        api_key=KEY,
        en_model="k",
        ru_model="p",
        ru_voice="dmitri",
        timeout_seconds=7,
    )
    options = resolve_tts_options("Привет.", engine="auto")

    result = synthesize_remote_wav("Привет.", options=options, settings=settings)

    req, timeout = calls[0]
    assert req.full_url == "https://s/v1/audio/speech-ru"  # Russian goes to the Silero container
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == f"Bearer {KEY}"
    assert json.loads(req.data) == {
        "model": "p",
        "voice": "dmitri",
        "input": "Привет.",
        "response_format": "wav",
        "speed": 1.0,
    }
    assert timeout == 7
    assert result.backend == "remote"
    assert result.resolved_device == "remote"
    assert result.engine == "silero"
    assert result.model == "p"
    assert result.sample_rate == 22_050
    assert result.wav.startswith(b"RIFF")


def test_http_errors_never_carry_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RemoteTtsSettings(
        base_url="https://s/v1",
        api_key=KEY,
        en_model="k",
        ru_model="p",
        ru_voice="dmitri",
        timeout_seconds=1,
    )
    options = resolve_tts_options("Hello.", engine="auto")

    def forbidden(req: Any, *, timeout: float | None = None) -> _FakeResponse:
        raise error.HTTPError(
            req.full_url,
            403,
            "Forbidden",
            {},
            io.BytesIO(json.dumps({"detail": f"bad {KEY}"}).encode()),
        )

    monkeypatch.setattr(remote_tts, "_open", forbidden)
    with pytest.raises(RemoteTtsError) as caught:
        synthesize_remote_wav("Hello.", options=options, settings=settings)
    assert "rejected the API key" in str(caught.value)
    assert KEY not in str(caught.value)

    def down(req: Any, *, timeout: float | None = None) -> _FakeResponse:
        raise error.URLError(f"connection refused {KEY}")

    monkeypatch.setattr(remote_tts, "_open", down)
    with pytest.raises(RemoteTtsError, match="unreachable") as caught:
        synthesize_remote_wav("Hello.", options=options, settings=settings)
    assert KEY not in str(caught.value)


def test_non_wav_and_empty_audio_are_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RemoteTtsSettings(
        base_url="https://s/v1",
        api_key=KEY,
        en_model="k",
        ru_model="p",
        ru_voice="dmitri",
        timeout_seconds=1,
    )
    options = resolve_tts_options("Hello.", engine="auto")
    _capture_urlopen(monkeypatch, b"<html>not audio</html>")
    with pytest.raises(RemoteTtsError, match="not a WAV"):
        synthesize_remote_wav("Hello.", options=options, settings=settings)
    _capture_urlopen(monkeypatch, _wav(frames=0))
    with pytest.raises(RemoteTtsError, match="empty"):
        synthesize_remote_wav("Hello.", options=options, settings=settings)


# --- synthesize_wav routing ----------------------------------------------------------------


def _fake_local(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    class FakeEngine:
        def synthesize(self, text: str, *, options: Any, device_resolution: Any) -> TtsResult:
            seen.append(options.engine)
            return TtsResult(
                wav=_wav(), sample_rate=24_000, chunks=1, engine=options.engine, model=options.model
            )

    monkeypatch.setattr(tts, "_SYNTHESIS_ENGINES", {"kokoro": FakeEngine(), "silero": FakeEngine()})
    monkeypatch.setattr(tts, "resolve_torch_device", lambda device: object())
    return seen


def test_auto_uses_the_cluster_when_configured(
    secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_secrets(secrets_dir, {"apiKey": KEY})
    local = _fake_local(monkeypatch)
    calls = _capture_urlopen(monkeypatch)

    result = synthesize_wav("Hello there.")

    assert result.backend == "remote"
    assert result.backend_fallback_reason is None
    assert len(calls) == 1
    assert local == [], "no local model must be loaded when the cluster answered"


def test_auto_without_a_key_is_plain_local(
    secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_secrets(secrets_dir, None)
    local = _fake_local(monkeypatch)
    calls = _capture_urlopen(monkeypatch)

    result = synthesize_wav("Hello there.")

    assert result.backend == "local"
    assert (
        result.backend_fallback_reason is not None
        and "not configured" in result.backend_fallback_reason
    )
    assert calls == []
    assert local == ["kokoro"]


def test_auto_falls_back_to_local_and_says_why(
    secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_secrets(secrets_dir, {"apiKey": KEY})
    local = _fake_local(monkeypatch)

    def down(req: Any, *, timeout: float | None = None) -> _FakeResponse:
        raise error.URLError("connection refused")

    monkeypatch.setattr(remote_tts, "_open", down)
    result = synthesize_wav("Привет, это тест.")

    assert result.backend == "local"
    assert result.engine == "silero"
    assert (
        result.backend_fallback_reason is not None
        and "unreachable" in result.backend_fallback_reason
    )
    assert local == ["silero"]


def test_remote_only_raises_instead_of_falling_back(
    secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_secrets(secrets_dir, {"apiKey": KEY})
    local = _fake_local(monkeypatch)

    def down(req: Any, *, timeout: float | None = None) -> _FakeResponse:
        raise error.URLError("connection refused")

    monkeypatch.setattr(remote_tts, "_open", down)
    with pytest.raises(RemoteTtsError, match="unreachable"):
        synthesize_wav("Hello there.", backend="remote")
    assert local == []


def test_local_only_never_touches_the_network(
    secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_secrets(secrets_dir, {"apiKey": KEY})
    local = _fake_local(monkeypatch)
    calls = _capture_urlopen(monkeypatch)

    result = synthesize_wav("Hello there.", backend="local")

    assert result.backend == "local"
    assert result.backend_fallback_reason is None
    assert calls == []
    assert local == ["kokoro"]


# --- round 1 audit regressions -------------------------------------------------------------


def test_env_settings_do_not_need_a_readable_file(
    secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (secrets_dir / "secrets.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("SPEECH_API_KEY", KEY)
    monkeypatch.setenv("SPEECH_BASE_URL", "https://s/v1")
    settings = load_remote_tts_settings()
    assert settings is not None and settings.api_key == KEY

    monkeypatch.delenv("SPEECH_BASE_URL")
    with pytest.raises(RemoteTtsError, match="Could not read the secrets file"):
        load_remote_tts_settings()


def test_a_key_that_cannot_travel_in_a_header_is_refused_without_quoting_it(
    secrets_dir: Path,
) -> None:
    bad = KEY[:40] + "\n" + KEY[40:]
    _write_secrets(secrets_dir, {"apiKey": bad})
    with pytest.raises(RemoteTtsError) as caught:
        load_remote_tts_settings()
    assert "cannot be sent in an HTTP header" in str(caught.value)
    assert KEY[:40] not in str(caught.value)


def test_redirects_are_refused() -> None:
    handler = remote_tts._NoRedirects()
    assert (
        handler.redirect_request(None, None, 302, "Found", {}, "http://elsewhere/v1/audio/speech")
        is None
    )


def test_redirect_answer_is_an_error_not_a_fallback_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RemoteTtsSettings(
        base_url="https://s/v1",
        api_key=KEY,
        en_model="k",
        ru_model="p",
        ru_voice="dmitri",
        timeout_seconds=1,
    )
    options = resolve_tts_options("Hello.", engine="auto")

    def redirected(req: Any, *, timeout: float | None = None) -> _FakeResponse:
        raise error.HTTPError(req.full_url, 302, "Found", {}, io.BytesIO(b""))

    monkeypatch.setattr(remote_tts, "_open", redirected)
    with pytest.raises(RemoteTtsError, match="redirect"):
        synthesize_remote_wav("Hello.", options=options, settings=settings)


def test_truncated_wav_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RemoteTtsSettings(
        base_url="https://s/v1",
        api_key=KEY,
        en_model="k",
        ru_model="p",
        ru_voice="dmitri",
        timeout_seconds=1,
    )
    options = resolve_tts_options("Hello.", engine="auto")
    whole = _wav(frames=2_400)
    _capture_urlopen(monkeypatch, whole[: len(whole) // 2])
    with pytest.raises(RemoteTtsError, match="truncated|not a WAV"):
        synthesize_remote_wav("Hello.", options=options, settings=settings)


def test_protocol_failures_fall_under_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client

    settings = RemoteTtsSettings(
        base_url="https://s/v1",
        api_key=KEY,
        en_model="k",
        ru_model="p",
        ru_voice="dmitri",
        timeout_seconds=1,
    )
    options = resolve_tts_options("Hello.", engine="auto")

    def broken(req: Any, *, timeout: float | None = None) -> _FakeResponse:
        raise http.client.BadStatusLine(f"garbage {KEY}")

    monkeypatch.setattr(remote_tts, "_open", broken)
    with pytest.raises(RemoteTtsError, match="unreachable") as caught:
        synthesize_remote_wav("Hello.", options=options, settings=settings)
    assert KEY not in str(caught.value)


def test_a_trickling_body_hits_the_overall_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RemoteTtsSettings(
        base_url="https://s/v1",
        api_key=KEY,
        en_model="k",
        ru_model="p",
        ru_voice="dmitri",
        timeout_seconds=5,
    )
    options = resolve_tts_options("Hello.", engine="auto")
    clock = iter([0.0, 0.0, 1.0, 2.0, 6.0, 7.0, 8.0])
    monkeypatch.setattr(remote_tts, "perf_counter", lambda: next(clock))

    class Trickle:
        def read1(self, limit: int | None = None) -> bytes:
            return b"x"

        def read(self, limit: int | None = None) -> bytes:  # pragma: no cover - read1 must win
            raise AssertionError("read(n) blocks across socket reads; the reader must use read1")

        def __enter__(self) -> Trickle:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(remote_tts, "_open", lambda req, *, timeout=None: Trickle())
    with pytest.raises(RemoteTtsError, match="unreachable: timed out"):
        synthesize_remote_wav("Hello.", options=options, settings=settings)


def test_reader_prefers_read1_over_read() -> None:
    """``read(n)`` would block until n bytes arrived; ``read1`` returns after one socket read."""

    class Both:
        calls: list[str] = []

        def read1(self, limit: int | None = None) -> bytes:
            self.calls.append("read1")
            return b"" if len(self.calls) > 1 else b"abc"

        def read(self, limit: int | None = None) -> bytes:
            self.calls.append("read")
            return b""

    response = Both()
    assert remote_tts._read_with_deadline(response, started=0.0, deadline_seconds=10**9) == b"abc"
    assert response.calls == ["read1", "read1"]


def test_an_explicit_eugene_stays_eugene_when_the_configured_voice_differs(
    secrets_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_secrets(secrets_dir, {"apiKey": KEY})
    monkeypatch.setenv("AGENT_TOOLS_REMOTE_TTS_RU_VOICE", "aidar")
    _fake_local(monkeypatch)
    calls = _capture_urlopen(monkeypatch, _wav(sample_rate=48_000))

    synthesize_wav("Привет, это тест.", voice="eugene")
    assert json.loads(calls[0][0].data)["voice"] == "eugene"

    synthesize_wav("Привет, это тест.")
    assert json.loads(calls[1][0].data)["voice"] == "aidar", "unnamed → the configured voice"


def _streamed_wav(frames: int, *, cut: int = 0) -> bytes:
    """A WAV as Speaches streams it: 0xFFFFFFFF in both size fields, data to the end."""
    whole = _wav(frames=frames)
    body = bytearray(whole)
    body[4:8] = b"\xff\xff\xff\xff"
    data_at = whole.index(b"data")
    body[data_at + 4 : data_at + 8] = b"\xff\xff\xff\xff"
    return bytes(body[: len(body) - cut] if cut else body)


def test_streamed_wav_with_unknown_length_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RemoteTtsSettings(
        base_url="https://s/v1",
        api_key=KEY,
        en_model="k",
        ru_model="p",
        ru_voice="eugene",
        timeout_seconds=1,
    )
    options = resolve_tts_options("Hello.", engine="auto")
    _capture_urlopen(monkeypatch, _streamed_wav(2_400))
    result = synthesize_remote_wav("Hello.", options=options, settings=settings)
    assert result.sample_rate == 24_000 and result.backend == "remote"

    # A body cut mid-frame is still refused.
    _capture_urlopen(monkeypatch, _streamed_wav(2_400, cut=1))
    with pytest.raises(RemoteTtsError, match="truncated"):
        synthesize_remote_wav("Hello.", options=options, settings=settings)
