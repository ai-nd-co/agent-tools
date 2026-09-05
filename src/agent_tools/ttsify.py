from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from time import perf_counter

from agent_tools.codex_config import (
    DEFAULT_CLAUDE_CODE_EFFORT,
    DEFAULT_CLAUDE_CODE_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TTSIFY_MODEL,
    DEFAULT_TTSIFY_REASONING_EFFORT,
    ENV_CLAUDE_CODE_BARE,
    ENV_CLAUDE_CODE_EFFORT,
    ENV_CLAUDE_CODE_MODEL,
    ENV_CODEX_MODEL,
    ENV_CODEX_REASONING_EFFORT,
    ENV_KOKORO_DEVICE,
    ENV_KOKORO_LANGUAGE,
    ENV_KOKORO_SPEED,
    ENV_KOKORO_VOICE,
    ENV_OLLAMA_MODEL,
    ENV_SILERO_VOICE,
    ENV_TTS_ENGINE,
    normalize_claude_code_model,
    read_bool_env,
    read_float_env,
    read_preferred_tts_speed,
    read_string_env,
)
from agent_tools.codex_private_api import TransformResult
from agent_tools.transformer import (
    TransformOptions,
    TransformProvider,
    resolve_effective_transform_provider,
    transform_text,
)
from agent_tools.tts import (
    DEFAULT_KOKORO_VOICE,
    DEFAULT_SILERO_VOICE,
    TtsResult,
    resolve_tts_engine,
    synthesize_wav,
)

SUPPORTED_TTSIFY_DEVICES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class TtsifyOptions:
    provider: TransformProvider | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    fast: bool = False
    tts_engine: str | None = None
    voice: str | None = None
    language: str | None = None
    speed: float | None = None
    device: str | None = None
    codex_home: Path | None = None
    base_url: str | None = None
    originator: str | None = None
    claude_model: str | None = None
    claude_effort: str | None = None
    claude_bare: bool | None = None
    timeout_seconds: float | None = None
    no_transform: bool = False


@dataclass(frozen=True)
class TtsifyResult:
    transformed_text: str
    transform_result: TransformResult
    tts_result: TtsResult
    model: str
    reasoning_effort: str | None
    voice: str
    language: str | None
    speed: float
    device: str
    resolved_device: str = "cpu"
    device_fallback_reason: str | None = None
    tts_engine: str = "kokoro"
    tts_model: str | None = None
    metrics: TtsifyMetrics = field(default_factory=lambda: TtsifyMetrics())


@dataclass(frozen=True)
class TtsifyMetrics:
    transform_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0


def ttsify_text(input_text: str, options: TtsifyOptions) -> TtsifyResult:
    total_started = perf_counter()
    device = options.device or read_string_env(ENV_KOKORO_DEVICE) or "auto"
    if device not in SUPPORTED_TTSIFY_DEVICES:
        raise ValueError(
            f"Unsupported TTS device {device!r}. Expected one of {SUPPORTED_TTSIFY_DEVICES}."
        )
    if options.no_transform:
        _validate_no_transform_options(options)
        transform_started = perf_counter()
        transform_result = TransformResult(
            text=input_text,
            response_id=None,
            usage=None,
            session_id=f"local-{uuid.uuid4()}",
        )
        transform_ms = (perf_counter() - transform_started) * 1000.0
        effective_model = "passthrough"
        effective_reasoning: str | None = None
    else:
        timeout_seconds = (
            options.timeout_seconds if options.timeout_seconds is not None else 120.0
        )
        prompt_text = load_ttsify_prompt()
        provider = resolve_effective_transform_provider(
            options.provider,
            codex_home=options.codex_home,
            base_url=options.base_url,
        )
        codex_model = options.model or read_string_env(ENV_CODEX_MODEL) or DEFAULT_TTSIFY_MODEL
        ollama_model = options.model or read_string_env(ENV_OLLAMA_MODEL) or DEFAULT_OLLAMA_MODEL
        codex_reasoning = (
            options.reasoning_effort
            or read_string_env(ENV_CODEX_REASONING_EFFORT)
            or DEFAULT_TTSIFY_REASONING_EFFORT
        )
        raw_claude_model = (
            options.claude_model
            or options.model
            or read_string_env(ENV_CLAUDE_CODE_MODEL)
        )
        claude_model = raw_claude_model or DEFAULT_CLAUDE_CODE_MODEL
        if provider == "claude-code":
            claude_model = normalize_claude_code_model(claude_model)
        claude_effort = options.claude_effort or read_string_env(ENV_CLAUDE_CODE_EFFORT)
        if claude_effort is None:
            claude_effort = _map_reasoning_effort_to_claude_effort(options.reasoning_effort)
        if claude_effort is None:
            claude_effort = DEFAULT_CLAUDE_CODE_EFFORT
        claude_bare = (
            options.claude_bare
            if options.claude_bare is not None
            else read_bool_env(ENV_CLAUDE_CODE_BARE) or False
        )
        transform_options = TransformOptions(
            system_prompt_text=prompt_text,
            provider=provider,
            model=codex_model,
            reasoning_effort=codex_reasoning,
            fast=options.fast,
            codex_home=options.codex_home,
            base_url=options.base_url,
            originator=options.originator,
            claude_model=claude_model,
            claude_effort=claude_effort,
            claude_bare=claude_bare,
            timeout_seconds=timeout_seconds,
        )
        transform_started = perf_counter()
        transform_result = transform_text(input_text, transform_options)
        transform_ms = (perf_counter() - transform_started) * 1000.0
        effective_model = (
            claude_model
            if provider == "claude-code"
            else ollama_model if provider == "ollama" else codex_model
        )
        effective_reasoning = (
            claude_effort
            if provider == "claude-code"
            else None if provider == "ollama" else codex_reasoning
        )
    requested_tts_engine = options.tts_engine or read_string_env(ENV_TTS_ENGINE) or "auto"
    resolved_tts_engine = resolve_tts_engine(
        transform_result.text,
        engine=requested_tts_engine,
        language=options.language,
    )
    language = options.language
    if language is None and resolved_tts_engine == "kokoro":
        language = read_string_env(ENV_KOKORO_LANGUAGE)
    if options.voice is not None:
        voice = options.voice
    elif resolved_tts_engine == "silero":
        voice = read_string_env(ENV_SILERO_VOICE) or DEFAULT_SILERO_VOICE
    else:
        voice = read_string_env(ENV_KOKORO_VOICE) or DEFAULT_KOKORO_VOICE
    if options.speed is not None:
        speed = options.speed
    elif resolved_tts_engine == "kokoro":
        speed = read_float_env(ENV_KOKORO_SPEED) or read_preferred_tts_speed() or 1.0
    else:
        speed = 1.0
    tts_started = perf_counter()
    tts_result = synthesize_wav(
        transform_result.text,
        voice=voice,
        language=language,
        speed=speed,
        device=device,
        engine=requested_tts_engine,
    )
    tts_ms = (perf_counter() - tts_started) * 1000.0
    total_ms = (perf_counter() - total_started) * 1000.0
    return TtsifyResult(
        transformed_text=transform_result.text,
        transform_result=transform_result,
        tts_result=tts_result,
        model=effective_model,
        reasoning_effort=effective_reasoning,
        voice=tts_result.voice or voice,
        language=tts_result.language if tts_result.language is not None else language,
        speed=speed,
        device=device,
        resolved_device=tts_result.resolved_device,
        device_fallback_reason=tts_result.device_fallback_reason,
        tts_engine=tts_result.engine,
        tts_model=tts_result.model,
        metrics=TtsifyMetrics(
            transform_ms=transform_ms,
            tts_ms=tts_ms,
            total_ms=total_ms,
        ),
    )


def load_ttsify_prompt() -> str:
    return resources.files("agent_tools.prompts").joinpath("ttsify.md").read_text(
        encoding="utf-8"
    )


def _validate_no_transform_options(options: TtsifyOptions) -> None:
    if any(
        value is not None
        for value in (
            options.provider,
            options.model,
            options.reasoning_effort,
            options.codex_home,
            options.base_url,
            options.originator,
            options.claude_model,
            options.claude_effort,
            options.timeout_seconds,
        )
    ) or options.fast or options.claude_bare:
        raise ValueError(
            "--no-transform cannot be combined with transform provider, model, auth, or "
            "reasoning options."
        )


def _map_reasoning_effort_to_claude_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    mapping = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
        "none": None,
    }
    return mapping.get(reasoning_effort)
