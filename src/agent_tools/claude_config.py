from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CLAUDE_HOME = Path.home() / ".claude"


def resolve_claude_home(claude_home: Path | None = None) -> Path:
    if claude_home is not None:
        return claude_home
    env_home = os.environ.get("CLAUDE_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return DEFAULT_CLAUDE_HOME
