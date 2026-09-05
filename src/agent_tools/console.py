from __future__ import annotations

import sys
from typing import Any


def configure_windows_utf8(
    *,
    stdout: Any = None,
    stderr: Any = None,
    platform_name: str | None = None,
) -> None:
    """Use UTF-8 for AgentTools text output without requiring shell configuration."""
    if (platform_name or sys.platform) != "win32":
        return
    streams = (
        sys.stdout if stdout is None else stdout,
        sys.stderr if stderr is None else stderr,
    )
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="strict")
        except (AttributeError, OSError, TypeError, ValueError):
            continue
