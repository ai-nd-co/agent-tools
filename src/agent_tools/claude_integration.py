from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_tools.claude_config import resolve_claude_home
from agent_tools.codex_integration import load_codex_integration_enabled

ENV_CLAUDE_INTEGRATION_TRIGGERED = "AGENT_TOOLS_CLAUDE_INTEGRATION_TRIGGERED"
CLAUDE_STOP_HOOK_COMMAND = 'bash -lc \'"$HOME/.claude/agent-tools/stop_tts.sh"\''

ClaudeIntegrationMode = Literal["stop-hook"]
ClaudeIntegrationInstallState = Literal["installed", "missing", "broken"]


@dataclass(frozen=True)
class ClaudeIntegrationStatus:
    mode: ClaudeIntegrationMode
    claude_home: Path
    settings_path: Path
    enabled: bool
    available: bool
    install_state: ClaudeIntegrationInstallState
    hook_script_path: Path | None = None
    availability_issues: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def effective_enabled(self) -> bool:
        return self.install_state == "installed" and self.enabled


def is_claude_integration_triggered() -> bool:
    return os.environ.get(ENV_CLAUDE_INTEGRATION_TRIGGERED) == "1"


def load_claude_integration_status(
    claude_home: Path | None = None,
) -> ClaudeIntegrationStatus:
    home = resolve_claude_home(claude_home)
    settings_path = home / "settings.json"
    hook_script_path = home / "agent-tools" / "stop_tts.sh"
    available, availability_issues = _detect_claude_backend_availability()
    install_state, issues = _detect_claude_stop_hook_install_state(
        settings_path=settings_path,
        hook_script_path=hook_script_path,
    )
    return ClaudeIntegrationStatus(
        mode="stop-hook",
        claude_home=home,
        settings_path=settings_path,
        enabled=load_codex_integration_enabled(),
        available=available,
        install_state=install_state,
        hook_script_path=hook_script_path,
        availability_issues=availability_issues,
        issues=issues,
    )


def _detect_claude_stop_hook_install_state(
    *,
    settings_path: Path,
    hook_script_path: Path,
) -> tuple[ClaudeIntegrationInstallState, tuple[str, ...]]:
    settings_exists = settings_path.exists()
    script_exists = hook_script_path.exists()

    payload: dict[str, Any] | None = None
    if settings_exists:
        try:
            parsed = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "broken", ("settings-json-invalid",)
        if not isinstance(parsed, dict):
            return "broken", ("settings-json-invalid",)
        payload = parsed

    has_stop_entry = _settings_payload_has_agent_tools_stop_entry(payload)

    if has_stop_entry and script_exists:
        return "installed", ()
    if not settings_exists and not script_exists:
        return "missing", ()
    if not has_stop_entry and not script_exists:
        return "missing", ()

    issues: list[str] = []
    if not has_stop_entry:
        issues.append("stop-hook-missing")
    if not script_exists and (has_stop_entry or settings_exists):
        issues.append("hook-script-missing")
    if not issues:
        issues.append("claude-stop-hook-incomplete")
    return "broken", tuple(issues)


def _settings_payload_has_agent_tools_stop_entry(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    hooks = payload.get("hooks")
    if not isinstance(hooks, dict):
        return False
    stop_entries = hooks.get("Stop")
    if not isinstance(stop_entries, list):
        return False
    for entry in stop_entries:
        if not isinstance(entry, dict):
            continue
        nested_hooks = entry.get("hooks")
        if not isinstance(nested_hooks, list):
            continue
        for hook in nested_hooks:
            if not isinstance(hook, dict):
                continue
            command = hook.get("command")
            if isinstance(command, str) and "agent-tools/stop_tts.sh" in command:
                return True
            if command == CLAUDE_STOP_HOOK_COMMAND:
                return True
    return False


def _detect_claude_backend_availability() -> tuple[bool, tuple[str, ...]]:
    if shutil.which("claude") is None:
        return False, ("cli-unavailable",)
    return True, ()
