from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from agent_tools.codex_config import (
    DEFAULT_TTSIFY_MODEL,
    DEFAULT_TTSIFY_VOICE,
    ENV_CODEX_MODEL,
    ENV_CODEX_REASONING_EFFORT,
    ENV_KOKORO_DEVICE,
    ENV_KOKORO_LANGUAGE,
    ENV_KOKORO_SPEED,
    ENV_KOKORO_VOICE,
    read_float_env,
    read_string_env,
)
from agent_tools.transformer import TransformOptions, TransformResult, transform_text
from agent_tools.tts import TtsResult, synthesize_wav

SUPPORTED_TTSIFY_DEVICES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class TtsifyOptions:
    model: str | None = None
    reasoning_effort: str | None = None
    fast: bool = False
    voice: str | None = None
    language: str | None = None
    speed: float | None = None
    device: str | None = None
    codex_home: Path | None = None
    base_url: str | None = None
    originator: str | None = None
    timeout_seconds: float = 120.0


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


def ttsify_text(input_text: str, options: TtsifyOptions) -> TtsifyResult:
    prompt_text = load_ttsify_prompt()
    voice = options.voice or read_string_env(ENV_KOKORO_VOICE) or DEFAULT_TTSIFY_VOICE
    language = options.language or read_string_env(ENV_KOKORO_LANGUAGE)
    speed = options.speed if options.speed is not None else read_float_env(ENV_KOKORO_SPEED) or 1.0
    device = options.device or read_string_env(ENV_KOKORO_DEVICE) or "auto"
    if device not in SUPPORTED_TTSIFY_DEVICES:
        raise ValueError(
            f"Unsupported Kokoro device {device!r}. Expected one of {SUPPORTED_TTSIFY_DEVICES}."
        )
    transform_options = TransformOptions(
        system_prompt_text=prompt_text,
        model=options.model or read_string_env(ENV_CODEX_MODEL) or DEFAULT_TTSIFY_MODEL,
        reasoning_effort=options.reasoning_effort or read_string_env(ENV_CODEX_REASONING_EFFORT),
        fast=options.fast,
        codex_home=options.codex_home,
        base_url=options.base_url,
        originator=options.originator,
        timeout_seconds=options.timeout_seconds,
    )
    transform_result = transform_text(input_text, transform_options)
    tts_result = synthesize_wav(
        transform_result.text,
        voice=voice,
        language=language,
        speed=speed,
        device=device,
    )
    return TtsifyResult(
        transformed_text=transform_result.text,
        transform_result=transform_result,
        tts_result=tts_result,
        model=transform_options.model or DEFAULT_TTSIFY_MODEL,
        reasoning_effort=transform_options.reasoning_effort,
        voice=voice,
        language=language,
        speed=speed,
        device=device,
    )


def load_ttsify_prompt() -> str:
    return resources.files("agent_tools.prompts").joinpath("ttsify.md").read_text(
        encoding="utf-8"
    )
