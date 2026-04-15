from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_DIR_NAME = "AgentTools"
CONTROLLER_HOST = "127.0.0.1"
CONTROLLER_PORT = 51173


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
