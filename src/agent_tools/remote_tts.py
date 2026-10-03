"""Text-to-speech on the office speech cluster, through the OpenAI speech API.

The cluster (k3s-gpu-cluster docs/32-speech.md) serves ``POST /v1/audio/speech`` next to the
speech-to-text routes, with the same Bearer key claude-tools already keeps in the owner's secrets
file. This module reads that file, maps the local engine choice (Kokoro for English, Silero for
Russian) onto the models the cluster hosts, and returns the same ``TtsResult`` the local engines do,
so nothing downstream can tell where the audio came from except by the ``backend`` fields.

Nothing here imports torch or the local engines: a remote synthesis must stay cheap on a machine
that has no GPU and no models.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import re
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib import error, request

from agent_tools.codex_config import read_string_env
from agent_tools.tts import DEFAULT_SILERO_VOICE, ResolvedTtsOptions, TtsMetrics, TtsResult

ENV_TTS_BACKEND = "AGENT_TOOLS_TTS_BACKEND"
ENV_SPEECH_API_KEY = "SPEECH_API_KEY"
ENV_SPEECH_BASE_URL = "SPEECH_BASE_URL"
ENV_REMOTE_TTS_EN_MODEL = "AGENT_TOOLS_REMOTE_TTS_EN_MODEL"
ENV_REMOTE_TTS_RU_MODEL = "AGENT_TOOLS_REMOTE_TTS_RU_MODEL"
ENV_REMOTE_TTS_RU_VOICE = "AGENT_TOOLS_REMOTE_TTS_RU_VOICE"
ENV_REMOTE_TTS_TIMEOUT = "AGENT_TOOLS_REMOTE_TTS_TIMEOUT_SECONDS"

SUPPORTED_TTS_BACKENDS = ("auto", "remote", "local")
DEFAULT_SPEECH_BASE_URL = "https://speech.k8s.tele-agent.site/v1"
# What the cluster hosts. English is Kokoro in Speaches at /audio/speech, with the same voice
# names as the local engine, so the resolved Kokoro voice is sent as-is. Russian is Silero in
# agent-tools' own tts-server container at /audio/speech-ru (model id `silero-v5-ru`), the same
# engine as locally, so the resolved Silero voice is forwarded too; the owner's pick is eugene.
DEFAULT_REMOTE_EN_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
DEFAULT_REMOTE_RU_MODEL = "silero-v5-ru"
DEFAULT_REMOTE_RU_VOICE = "eugene"
REMOTE_EN_PATH = "/audio/speech"
REMOTE_RU_PATH = "/audio/speech-ru"
DEFAULT_TIMEOUT_SECONDS = 120.0
MIN_API_KEY_CHARACTERS = 32
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
# What may travel in an Authorization header. Anything else (a newline above all) would make
# http.client raise with the header's bytes in the message, which no redaction catches.
_HEADER_SAFE_KEY = re.compile(r"[A-Za-z0-9._~+/=-]+\Z")


class RemoteTtsError(RuntimeError):
    """The cluster could not be used for this request. The caller may fall back to local."""


class RemoteTtsNotConfigured(RemoteTtsError):
    """No key in the secrets file or the environment."""


@dataclass(frozen=True)
class RemoteTtsSettings:
    base_url: str
    api_key: str
    en_model: str
    ru_model: str
    ru_voice: str
    timeout_seconds: float


def claude_tools_secrets_path() -> Path:
    """The same file claude-tools reads (``src/config/paths.ts``): one credential, two tools."""
    override = os.environ.get("CLAUDE_TOOLS_CONFIG_DIR")
    if override:
        return Path(override) / "secrets.json"
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "claude-tools" / "secrets.json"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "claude-tools" / "secrets.json"


def _read_speech_secrets() -> dict[str, object]:
    path = claude_tools_secrets_path()
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise RemoteTtsError(
            f"Could not read the secrets file {path}: {exc.__class__.__name__}"
        ) from exc
    speech = data.get("speech") if isinstance(data, dict) else None
    return speech if isinstance(speech, dict) else {}


def normalize_tts_backend(value: str | None) -> str:
    backend = (value or read_string_env(ENV_TTS_BACKEND) or "auto").strip().lower()
    if backend not in SUPPORTED_TTS_BACKENDS:
        raise ValueError(
            f"Unsupported TTS backend {value!r}. Expected one of {SUPPORTED_TTS_BACKENDS}."
        )
    return backend


def _canonical_base_url(value: str) -> str:
    from urllib.parse import urlsplit

    parts = urlsplit(value)
    path = parts.path.rstrip("/")
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
        raise RemoteTtsError("speech.baseUrl must be an http(s) URL ending in /v1")
    if not path.endswith("/v1"):
        raise RemoteTtsError("speech.baseUrl must be an http(s) URL ending in /v1")
    return f"{parts.scheme}://{parts.netloc}{path}"


def load_remote_tts_settings() -> RemoteTtsSettings | None:
    """The cluster settings, or ``None`` when no key is configured anywhere.

    Environment wins over the file, as in claude-tools. A key shorter than the server's own
    minimum is treated as absent rather than sent: it cannot be the real one.
    """
    env_key = read_string_env(ENV_SPEECH_API_KEY)
    env_url = read_string_env(ENV_SPEECH_BASE_URL)
    # The file is opened only for what the environment did not supply, so a broken file cannot
    # take the cluster away from a shell that configured it fully.
    secrets = _read_speech_secrets() if not (env_key and env_url) else {}
    api_key = (env_key or _as_str(secrets.get("apiKey")) or "").strip()
    if len(api_key) < MIN_API_KEY_CHARACTERS:
        return None
    if not _HEADER_SAFE_KEY.match(api_key):
        raise RemoteTtsError(
            "speech.apiKey contains characters that cannot be sent in an HTTP header"
        )
    base_url = _canonical_base_url(
        env_url or _as_str(secrets.get("baseUrl")) or DEFAULT_SPEECH_BASE_URL
    )
    timeout_raw = read_string_env(ENV_REMOTE_TTS_TIMEOUT)
    try:
        timeout_seconds = float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT_SECONDS
    except ValueError as exc:
        raise ValueError(f"{ENV_REMOTE_TTS_TIMEOUT} must be a number of seconds.") from exc
    if timeout_seconds <= 0:
        raise ValueError(f"{ENV_REMOTE_TTS_TIMEOUT} must be positive.")
    return RemoteTtsSettings(
        base_url=base_url,
        api_key=api_key,
        en_model=read_string_env(ENV_REMOTE_TTS_EN_MODEL) or DEFAULT_REMOTE_EN_MODEL,
        ru_model=read_string_env(ENV_REMOTE_TTS_RU_MODEL) or DEFAULT_REMOTE_RU_MODEL,
        ru_voice=read_string_env(ENV_REMOTE_TTS_RU_VOICE) or DEFAULT_REMOTE_RU_VOICE,
        timeout_seconds=timeout_seconds,
    )


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def remote_model_and_voice(
    options: ResolvedTtsOptions, settings: RemoteTtsSettings
) -> tuple[str, str]:
    """The cluster model and voice for a locally resolved engine choice.

    A Silero voice chosen locally is honoured remotely (same engine); the configured Russian
    voice is the default when the request named none, which is how the owner's pick applies.
    """
    if options.engine == "silero":
        voice = options.voice if options.voice != DEFAULT_SILERO_VOICE else settings.ru_voice
        return settings.ru_model, voice
    return settings.en_model, options.voice


def remote_path(options: ResolvedTtsOptions) -> str:
    """Russian and English live on different containers behind the same host."""
    return REMOTE_RU_PATH if options.engine == "silero" else REMOTE_EN_PATH


def _error_detail(exc: error.HTTPError) -> str:
    """The server's own words for a failure, best effort: FastAPI puts them under ``detail``."""
    try:
        raw = exc.read(2048).decode("utf-8", "replace")
    except (OSError, http.client.HTTPException):
        return ""
    finally:
        exc.close()
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(parsed, dict) and "detail" in parsed:
        detail = parsed["detail"]
        return detail if isinstance(detail, str) else json.dumps(detail)
    return raw


def _redact(text: str, api_key: str) -> str:
    return text.replace(api_key, "***") if api_key else text


class _NoRedirects(request.HTTPRedirectHandler):
    """A redirect would re-send the Authorization header to wherever the server points, another
    host or plain http included. The cluster never redirects; anything that does is refused."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


_OPENER = request.build_opener(_NoRedirects)


def _open(req: request.Request, *, timeout: float) -> Any:
    """The one network call, kept separate so tests can stand in for it."""
    return _OPENER.open(req, timeout=timeout)  # noqa: S310 - https to a configured host


def _read_with_deadline(response: Any, *, started: float, deadline_seconds: float) -> bytes:
    """Read the body against an overall deadline.

    The socket timeout only bounds one blocking read, and ``read(n)`` keeps reading until it has
    ``n`` bytes, so a server trickling bytes could hold the call for as long as it liked. ``read1``
    returns after at most ONE underlying read; checked between calls, the whole body is bounded by
    the deadline plus a single socket timeout.
    """
    read_once = getattr(response, "read1", None) or response.read
    chunks: list[bytes] = []
    total = 0
    while True:
        if perf_counter() - started > deadline_seconds:
            raise TimeoutError("timed out")
        chunk = read_once(_READ_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise RemoteTtsError("the speech cluster returned more audio than AgentTools accepts")
        chunks.append(chunk)


def synthesize_remote_wav(
    text: str,
    *,
    options: ResolvedTtsOptions,
    settings: RemoteTtsSettings,
) -> TtsResult:
    """One ``POST /v1/audio/speech``; a validated WAV back, or ``RemoteTtsError``.

    Any failure is reported without the key: the error text is the only thing that reaches a
    terminal or a log, and an upstream body is arbitrary text.
    """
    started = perf_counter()
    model, voice = remote_model_and_voice(options, settings)
    body = json.dumps(
        {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": "wav",
            "speed": options.speed,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{settings.base_url}{remote_path(options)}",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        },
    )
    try:
        with _open(req, timeout=settings.timeout_seconds) as response:
            wav = _read_with_deadline(
                response, started=started, deadline_seconds=settings.timeout_seconds
            )
    except RemoteTtsError:
        raise
    except error.HTTPError as exc:
        detail = _error_detail(exc)
        if exc.code in (401, 403):
            raise RemoteTtsError(
                _redact(
                    f"the speech cluster rejected the API key ({exc.code}): {detail}",
                    settings.api_key,
                )
            ) from None
        if 300 <= exc.code < 400:
            raise RemoteTtsError(
                f"the speech cluster answered a redirect ({exc.code}), which is refused"
            ) from None
        raise RemoteTtsError(
            _redact(
                f"the speech cluster answered {exc.code} for {model}: {detail}", settings.api_key
            )
        ) from None
    except (error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        # URLError carries the cause in `reason`; the others are their own message. Any of them
        # may quote request bytes, so the key is scrubbed from whatever text comes out.
        reason = (
            getattr(exc, "reason", None)
            or (str(exc) if isinstance(exc, TimeoutError) and str(exc) else None)
            or exc.__class__.__name__ + (f": {exc}" if str(exc) else "")
        )
        raise RemoteTtsError(
            _redact(f"the speech cluster is unreachable: {reason}", settings.api_key)
        ) from None
    except ValueError as exc:
        # http.client raises ValueError for a header or URL it will not send. Its message can
        # quote the header, so the exception text is not repeated.
        raise RemoteTtsError(
            f"the request to the speech cluster could not be built ({exc.__class__.__name__})"
        ) from None

    generation_ms = (perf_counter() - started) * 1000.0
    try:
        with wave.open(io.BytesIO(wav)) as handle:
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            # The header declares the frame count; only reading them proves the bytes exist.
            payload = handle.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise RemoteTtsError(
            f"the speech cluster returned something that is not a WAV: {exc}"
        ) from None
    if frames == 0 or channels != 1 or sample_rate <= 0:
        raise RemoteTtsError("the speech cluster returned empty or non-mono audio")
    if len(payload) != frames * channels * sample_width:
        raise RemoteTtsError("the speech cluster returned a truncated WAV")

    total_ms = (perf_counter() - started) * 1000.0
    return TtsResult(
        wav=wav,
        sample_rate=sample_rate,
        chunks=1,
        requested_device="remote",
        resolved_device="remote",
        requested_engine=options.requested_engine,
        engine=options.engine,
        model=model,
        voice=voice,
        language=options.language,
        backend="remote",
        metrics=TtsMetrics(
            generation_ms=generation_ms,
            total_ms=total_ms,
            text_chars=len(text),
        ),
    )
