from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_tools.claude_code_transform import (
    ClaudeCodeTransformOptions,
    transform_with_claude_code,
)
from agent_tools.codex_auth import load_auth_state
from agent_tools.codex_config import (
    DEFAULT_CLAUDE_CODE_EFFORT,
    DEFAULT_CLAUDE_CODE_MODEL,
    DEFAULT_MODEL,
    DEFAULT_TRANSFORM_PROVIDER,
    ENV_TRANSFORM_PROVIDER,
    CodexDefaults,
    load_codex_defaults,
    normalize_claude_code_model,
    read_preferred_transform_provider,
    read_string_env,
)
from agent_tools.codex_private_api import ClientSettings, CodexPrivateClient, TransformResult

TransformProvider = Literal["codex", "claude-code"]


@dataclass(frozen=True)
class TransformOptions:
    system_prompt_file: Path | None = None
    system_prompt_text: str | None = None
    provider: TransformProvider | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    fast: bool = False
    codex_home: Path | None = None
    base_url: str | None = None
    originator: str | None = None
    claude_model: str | None = None
    claude_effort: str | None = None
    claude_bare: bool = False
    timeout_seconds: float = 120.0


def transform_text(input_text: str, options: TransformOptions) -> TransformResult:
    provider = resolve_transform_provider(options.provider)
    if provider == "claude-code":
        system_prompt = _read_system_prompt(options)
        return transform_with_claude_code(
            ClaudeCodeTransformOptions(
                system_prompt=system_prompt,
                input_text=input_text,
                model=_resolve_claude_model(options),
                effort=_resolve_claude_effort(options),
                bare=options.claude_bare,
                timeout_seconds=options.timeout_seconds,
            )
        )

    defaults = load_codex_defaults(options.codex_home)
    auth_state = load_auth_state(defaults.codex_home)
    system_prompt = _read_system_prompt(options)
    effective_model = options.model or defaults.model or DEFAULT_MODEL
    effective_reasoning = _resolve_reasoning_effort(options.reasoning_effort, defaults)
    settings = ClientSettings(
        base_url=options.base_url or defaults.base_url,
        originator=options.originator or defaults.originator,
        version=defaults.version,
        timeout_seconds=options.timeout_seconds,
    )
    client = CodexPrivateClient(settings=settings)
    return client.transform(
        auth_state=auth_state,
        system_prompt=system_prompt,
        user_text=input_text,
        model=effective_model,
        reasoning_effort=effective_reasoning,
        fast=options.fast,
    )


def resolve_transform_provider(value: TransformProvider | None) -> TransformProvider:
    provider = (
        value
        or read_preferred_transform_provider()
        or read_string_env(ENV_TRANSFORM_PROVIDER)
        or DEFAULT_TRANSFORM_PROVIDER
    )
    if provider not in {"codex", "claude-code"}:
        raise ValueError(f"Unsupported transform provider {provider!r}.")
    return provider


def _resolve_reasoning_effort(value: str | None, defaults: CodexDefaults) -> str | None:
    if value == "none":
        return None
    if value is not None:
        return value
    return defaults.reasoning_effort


def _resolve_claude_model(options: TransformOptions) -> str:
    return normalize_claude_code_model(
        options.claude_model or options.model or DEFAULT_CLAUDE_CODE_MODEL
    )


def _resolve_claude_effort(options: TransformOptions) -> str | None:
    if options.claude_effort is not None:
        return options.claude_effort
    reasoning = options.reasoning_effort
    if reasoning in (None, "none"):
        return DEFAULT_CLAUDE_CODE_EFFORT
    mapped = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
    }
    return mapped.get(reasoning, DEFAULT_CLAUDE_CODE_EFFORT)


def _read_system_prompt(options: TransformOptions) -> str:
    if options.system_prompt_text is not None and options.system_prompt_file is not None:
        raise ValueError("Specify either system_prompt_text or system_prompt_file, not both.")
    if options.system_prompt_text is not None:
        return options.system_prompt_text
    if options.system_prompt_file is not None:
        return options.system_prompt_file.read_text(encoding="utf-8")
    raise ValueError("A system prompt is required.")
