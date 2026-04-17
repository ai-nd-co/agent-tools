from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

APP_DIR_NAME = "AgentTools"
DEFAULT_CONTROLLER_HOST = "127.0.0.1"
# Stay below Windows' default dynamic/excluded port ranges (typically 49152+).
DEFAULT_CONTROLLER_PORT = 38173


def _controller_port_from_env(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw_value = env.get("AGENT_TOOLS_CONTROLLER_PORT")
    if raw_value is None or raw_value == "":
        return DEFAULT_CONTROLLER_PORT
    try:
        port = int(raw_value)
    except ValueError:
        return DEFAULT_CONTROLLER_PORT
    return port if 1 <= port <= 65535 else DEFAULT_CONTROLLER_PORT


CONTROLLER_HOST = os.environ.get("AGENT_TOOLS_CONTROLLER_HOST", DEFAULT_CONTROLLER_HOST)
CONTROLLER_PORT = _controller_port_from_env()


def app_root() -> Path:
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / APP_DIR_NAME
    return Path.home() / ".local" / "share" / "agent-tools"


def state_dir() -> Path:
    return app_root() / "state"


def audio_cache_dir() -> Path:
    return app_root() / "audio"


def queue_db_path() -> Path:
    return state_dir() / "queue.sqlite3"


def preferences_path() -> Path:
    return state_dir() / "preferences.json"


def ensure_runtime_dirs() -> None:
    state_dir().mkdir(parents=True, exist_ok=True)
    audio_cache_dir().mkdir(parents=True, exist_ok=True)


def load_preferences() -> dict[str, object]:
    path = preferences_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


def save_preferences(preferences: dict[str, object]) -> None:
    ensure_runtime_dirs()
    preferences_path().write_text(
        json.dumps(preferences, indent=2, sort_keys=True),
        encoding="utf-8",
    )
