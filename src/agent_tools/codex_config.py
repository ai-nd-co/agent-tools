from __future__ import annotations

import json
import os
import platform
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_tools.runtime import load_preferences

DEFAULT_CODEX_HOME = Path.home() / ".codex"
DEFAULT_CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_ORIGINATOR = "codex_cli_rs"
DEFAULT_MODEL = "gpt-5"
DEFAULT_TTSIFY_MODEL = "gpt-5.4-mini"
DEFAULT_TTSIFY_VOICE = "af_heart"
DEFAULT_TRANSFORM_PROVIDER = "codex"
DEFAULT_CLAUDE_CODE_MODEL = "haiku"
DEFAULT_CLAUDE_CODE_EFFORT = "low"
ALLOWED_CLAUDE_CODE_MODELS = ("haiku", "sonnet")

ENV_CODEX_MODEL = "AGENT_TOOLS_CODEX_MODEL"
ENV_CODEX_REASONING_EFFORT = "AGENT_TOOLS_CODEX_REASONING_EFFORT"
ENV_KOKORO_VOICE = "AGENT_TOOLS_KOKORO_VOICE"
ENV_KOKORO_LANGUAGE = "AGENT_TOOLS_KOKORO_LANGUAGE"
ENV_KOKORO_SPEED = "AGENT_TOOLS_KOKORO_SPEED"
ENV_KOKORO_DEVICE = "AGENT_TOOLS_KOKORO_DEVICE"
ENV_SOURCE = "AGENT_TOOLS_SOURCE"
ENV_TRANSFORM_PROVIDER = "AGENT_TOOLS_TRANSFORM_PROVIDER"
ENV_CLAUDE_CODE_MODEL = "AGENT_TOOLS_CLAUDE_CODE_MODEL"
ENV_CLAUDE_CODE_EFFORT = "AGENT_TOOLS_CLAUDE_CODE_EFFORT"
ENV_CLAUDE_CODE_BARE = "AGENT_TOOLS_CLAUDE_CODE_BARE"
DEFAULT_PREFERRED_TTS_SPEED = 1.0
PREFERENCE_TRANSFORM_PROVIDER = "preferred_transform_provider"


@dataclass(frozen=True)
class CodexDefaults:
    codex_home: Path
    model: str | None
    reasoning_effort: str | None
    version: str
    originator: str
    base_url: str


def resolve_codex_home(codex_home: Path | None = None) -> Path:
    if codex_home is not None:
        return codex_home
    env_home = os.environ.get("CODEX_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return DEFAULT_CODEX_HOME


def load_codex_defaults(codex_home: Path | None = None) -> CodexDefaults:
    home = resolve_codex_home(codex_home)
    config = _load_toml_if_present(home / "config.toml")
    version = _load_version(home / "version.json")
    originator = os.environ.get("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", DEFAULT_ORIGINATOR)
    return CodexDefaults(
        codex_home=home,
        model=_coerce_str(config.get("model")),
        reasoning_effort=_coerce_str(config.get("model_reasoning_effort")),
        version=version,
        originator=originator,
        base_url=DEFAULT_CHATGPT_CODEX_BASE_URL,
    )


def build_user_agent(originator: str, version: str) -> str:
    system = platform.system() or "UnknownOS"
    release = platform.release() or "unknown"
    machine = platform.machine() or "unknown"
    return (
        f"{originator}/{version} "
        f"({system} {release}; {machine}) "
        f"Python/{platform.python_version()}"
    )


def _load_toml_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _load_version(path: Path) -> str:
    if not path.exists():
        return "0.0.0"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "0.0.0"
    return _coerce_str(payload.get("latest_version")) or "0.0.0"


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def read_string_env(name: str) -> str | None:
    return _coerce_str(os.environ.get(name))


def read_float_env(name: str) -> float | None:
    value = read_string_env(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {value!r}") from exc


def read_bool_env(name: str) -> bool | None:
    value = read_string_env(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable {name} must be a boolean-like value, got {value!r}"
    )


def read_preferred_tts_speed() -> float | None:
    value = load_preferences().get("preferred_tts_speed")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def read_preferred_transform_provider() -> str | None:
    value = load_preferences().get(PREFERENCE_TRANSFORM_PROVIDER)
    return _coerce_str(value)


def normalize_claude_code_model(value: str | None) -> str:
    normalized = _coerce_str(value) or DEFAULT_CLAUDE_CODE_MODEL
    lowered = normalized.lower()
    if lowered in ALLOWED_CLAUDE_CODE_MODELS:
        return lowered
    raise ValueError(
        "Claude Code transform model must be 'haiku' or 'sonnet'; Opus is not allowed."
    )
