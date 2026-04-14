from __future__ import annotations

import json
import os
import platform
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CODEX_HOME = Path.home() / ".codex"
DEFAULT_CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_ORIGINATOR = "codex_cli_rs"
DEFAULT_MODEL = "gpt-5"


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
