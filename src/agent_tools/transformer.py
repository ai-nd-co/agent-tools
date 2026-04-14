from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_tools.codex_auth import load_auth_state
from agent_tools.codex_config import DEFAULT_MODEL, CodexDefaults, load_codex_defaults
from agent_tools.codex_private_api import ClientSettings, CodexPrivateClient, TransformResult


@dataclass(frozen=True)
class TransformOptions:
    system_prompt_file: Path | None = None
    system_prompt_text: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    fast: bool = False
    codex_home: Path | None = None
    base_url: str | None = None
    originator: str | None = None
    timeout_seconds: float = 120.0


def transform_text(input_text: str, options: TransformOptions) -> TransformResult:
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


def _resolve_reasoning_effort(value: str | None, defaults: CodexDefaults) -> str | None:
    if value == "none":
        return None
    if value is not None:
        return value
    return defaults.reasoning_effort


def _read_system_prompt(options: TransformOptions) -> str:
    if options.system_prompt_text is not None and options.system_prompt_file is not None:
        raise ValueError("Specify either system_prompt_text or system_prompt_file, not both.")
    if options.system_prompt_text is not None:
        return options.system_prompt_text
    if options.system_prompt_file is not None:
        return options.system_prompt_file.read_text(encoding="utf-8")
    raise ValueError("A system prompt is required.")
