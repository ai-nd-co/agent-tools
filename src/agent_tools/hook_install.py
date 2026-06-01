from __future__ import annotations

import json
import re
import stat
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from agent_tools.claude_config import resolve_claude_home
from agent_tools.codex_config import resolve_codex_home
from agent_tools.codex_integration import set_codex_integration_enabled

STOP_HOOK_COMMAND = 'bash -lc \'"$HOME/.codex/hooks/stop_tts.sh"\''
WINDOWS_NOTIFY_COMMAND = "codex-notify-dispatch"
WINDOWS_NOTIFY_LAUNCHER = "agent-tools"
CLAUDE_STOP_HOOK_COMMAND = 'bash -lc \'"$HOME/.claude/agent-tools/stop_tts.sh"\'' 
STOP_HOOK_ENTRY = {
    "hooks": [
        {
            "type": "command",
            "command": STOP_HOOK_COMMAND,
            "timeout": 30,
            "statusMessage": "Queueing final assistant message for TTS",
        }
    ]
}
CLAUDE_STOP_HOOK_ENTRY = {
    "hooks": [
        {
            "type": "command",
            "command": CLAUDE_STOP_HOOK_COMMAND,
            "timeout": 30,
        }
    ]
}


@dataclass(frozen=True)
class InstallCodexIntegrationResult:
    mode: Literal["notify", "stop-hook"]
    codex_home: Path
    config_path: Path
    backups: tuple[Path, ...]
    hooks_json_path: Path | None = None
    hook_script_path: Path | None = None
    notify_command: tuple[str, ...] | None = None


@dataclass(frozen=True)
class InstallClaudeIntegrationResult:
    claude_home: Path
    settings_path: Path
    hook_script_path: Path
    backups: tuple[Path, ...]


@dataclass(frozen=True)
class InstallAgentIntegrationsResult:
    codex: InstallCodexIntegrationResult
    claude: InstallClaudeIntegrationResult


def install_codex_integration(
    codex_home: Path | None = None,
    *,
    platform_name: str | None = None,
) -> InstallCodexIntegrationResult:
    current_platform = platform_name or sys.platform
    if current_platform == "win32":
        return install_windows_notify_integration(codex_home)
    return install_codex_stop_hook(codex_home)


def install_windows_notify_integration(
    codex_home: Path | None = None,
) -> InstallCodexIntegrationResult:
    home = resolve_codex_home(codex_home)
    config_path = home / "config.toml"
    notify_command = (
        WINDOWS_NOTIFY_LAUNCHER,
        WINDOWS_NOTIFY_COMMAND,
    )

    backups: list[Path] = []
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    updated_text = ensure_notify_command(config_text, notify_command)
    updated_text = ensure_feature_assignment(updated_text, "codex_hooks", "false")
    backup = _write_text_if_changed(config_path, updated_text)
    if backup is not None:
        backups.append(backup)
    set_codex_integration_enabled(True)

    return InstallCodexIntegrationResult(
        mode="notify",
        codex_home=home,
        config_path=config_path,
        backups=tuple(backups),
        notify_command=notify_command,
    )


def install_codex_stop_hook(codex_home: Path | None = None) -> InstallCodexIntegrationResult:
    home = resolve_codex_home(codex_home)
    hook_dir = home / "hooks"
    hook_dir.mkdir(parents=True, exist_ok=True)

    backups: list[Path] = []
    config_path = home / "config.toml"
    hooks_json_path = home / "hooks.json"
    hook_script_path = hook_dir / "stop_tts.sh"

    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    updated_text = ensure_feature_assignment(config_text, "codex_hooks", "true")
    backup = _write_text_if_changed(config_path, updated_text)
    if backup is not None:
        backups.append(backup)

    new_hooks_payload = build_updated_hooks_payload(hooks_json_path)
    backup = _write_text_if_changed(
        hooks_json_path,
        json.dumps(new_hooks_payload, indent=2) + "\n",
    )
    if backup is not None:
        backups.append(backup)

    script_text = load_packaged_hook_script()
    backup = _write_text_if_changed(hook_script_path, script_text)
    if backup is not None:
        backups.append(backup)
    _ensure_executable(hook_script_path)
    set_codex_integration_enabled(True)

    return InstallCodexIntegrationResult(
        mode="stop-hook",
        codex_home=home,
        config_path=config_path,
        backups=tuple(backups),
        hooks_json_path=hooks_json_path,
        hook_script_path=hook_script_path,
    )


def install_claude_integration(
    claude_home: Path | None = None,
) -> InstallClaudeIntegrationResult:
    home = resolve_claude_home(claude_home)
    script_dir = home / "agent-tools"
    script_dir.mkdir(parents=True, exist_ok=True)

    backups: list[Path] = []
    settings_path = home / "settings.json"
    hook_script_path = script_dir / "stop_tts.sh"

    updated_payload = build_updated_claude_settings_payload(settings_path)
    backup = _write_text_if_changed(
        settings_path,
        json.dumps(updated_payload, indent=2) + "\n",
    )
    if backup is not None:
        backups.append(backup)

    script_text = load_packaged_claude_hook_script()
    backup = _write_text_if_changed(hook_script_path, script_text)
    if backup is not None:
        backups.append(backup)
    _ensure_executable(hook_script_path)
    set_codex_integration_enabled(True)

    return InstallClaudeIntegrationResult(
        claude_home=home,
        settings_path=settings_path,
        hook_script_path=hook_script_path,
        backups=tuple(backups),
    )


def install_agent_integrations(
    codex_home: Path | None = None,
    claude_home: Path | None = None,
    *,
    platform_name: str | None = None,
) -> InstallAgentIntegrationsResult:
    return InstallAgentIntegrationsResult(
        codex=install_codex_integration(codex_home, platform_name=platform_name),
        claude=install_claude_integration(claude_home),
    )


def load_packaged_hook_script() -> str:
    return resources.files("agent_tools.codex_hooks").joinpath("stop_tts.sh").read_text(
        encoding="utf-8"
    )


def load_packaged_claude_hook_script() -> str:
    return resources.files("agent_tools.claude_hooks").joinpath("stop_tts.sh").read_text(
        encoding="utf-8"
    )


def build_updated_hooks_payload(hooks_json_path: Path) -> dict[str, Any]:
    payload = _load_existing_hooks_payload(hooks_json_path)
    hooks = payload.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError(f"{hooks_json_path} must contain a JSON object under 'hooks'.")
    stop_entries = hooks.setdefault("Stop", [])
    if not isinstance(stop_entries, list):
        raise ValueError(f"{hooks_json_path} must contain a JSON list under hooks.Stop.")

    replaced = False
    for index, entry in enumerate(stop_entries):
        if _is_agent_tools_stop_entry(entry):
            stop_entries[index] = STOP_HOOK_ENTRY
            replaced = True
            break
    if not replaced:
        stop_entries.append(STOP_HOOK_ENTRY)

    return payload


def build_updated_claude_settings_payload(settings_path: Path) -> dict[str, Any]:
    payload = _load_existing_settings_payload(settings_path)
    hooks = payload.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError(f"{settings_path} must contain a JSON object under 'hooks'.")
    stop_entries = hooks.setdefault("Stop", [])
    if not isinstance(stop_entries, list):
        raise ValueError(f"{settings_path} must contain a JSON list under hooks.Stop.")

    replaced = False
    for index, entry in enumerate(stop_entries):
        if _is_agent_tools_claude_stop_entry(entry):
            stop_entries[index] = CLAUDE_STOP_HOOK_ENTRY
            replaced = True
            break
    if not replaced:
        stop_entries.append(CLAUDE_STOP_HOOK_ENTRY)

    return payload


def ensure_notify_command(config_text: str, notify_command: tuple[str, ...]) -> str:
    notify_line = f"notify = {json.dumps(list(notify_command))}"
    lines = config_text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^\s*notify\s*=", line):
            lines[index] = notify_line
            return "\n".join(lines) + "\n"

    section_start = _first_section_index(lines)
    lines.insert(section_start, notify_line)
    if section_start < len(lines) - 1 and lines[section_start + 1].strip():
        lines.insert(section_start + 1, "")
    return "\n".join(lines).rstrip() + "\n"


def ensure_feature_assignment(config_text: str, key: str, rendered_value: str) -> str:
    lines = config_text.splitlines()
    features_index = _find_section_header(lines, "features")
    assignment = f"{key} = {rendered_value}"

    if features_index is None:
        suffix = "" if not config_text.strip() else "\n\n"
        return f"{config_text.rstrip()}{suffix}[features]\n{assignment}\n"

    section_end = _find_next_section_index(lines, features_index + 1)
    for index in range(features_index + 1, section_end):
        if re.match(rf"^\s*{re.escape(key)}\s*=", lines[index]):
            lines[index] = assignment
            return "\n".join(lines) + "\n"

    lines.insert(section_end, assignment)
    return "\n".join(lines) + "\n"


def _find_section_header(lines: list[str], section_name: str) -> int | None:
    pattern = re.compile(rf"^\s*\[{re.escape(section_name)}\]\s*$")
    for index, line in enumerate(lines):
        if pattern.match(line):
            return index
    return None


def _first_section_index(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if re.match(r"^\s*\[[^\]]+\]\s*$", line):
            return index
    return len(lines)


def _find_next_section_index(lines: list[str], start_index: int) -> int:
    for index in range(start_index, len(lines)):
        if re.match(r"^\s*\[[^\]]+\]\s*$", lines[index]):
            return index
    return len(lines)


def _load_existing_hooks_payload(hooks_json_path: Path) -> dict[str, Any]:
    if not hooks_json_path.exists():
        return {"hooks": {}}

    return _load_existing_settings_payload(hooks_json_path)


def _load_existing_settings_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object.")
    return payload


def _is_agent_tools_stop_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    hooks = entry.get("hooks")
    if not isinstance(hooks, list):
        return False
    for hook in hooks:
        if not isinstance(hook, dict):
            continue
        command = hook.get("command")
        if isinstance(command, str) and "stop_tts.sh" in command:
            return True
    return False


def _is_agent_tools_claude_stop_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    hooks = entry.get("hooks")
    if not isinstance(hooks, list):
        return False
    for hook in hooks:
        if not isinstance(hook, dict):
            continue
        command = hook.get("command")
        if command == CLAUDE_STOP_HOOK_COMMAND:
            return True
        if isinstance(command, str) and "agent-tools/stop_tts.sh" in command:
            return True
    return False


def _write_text_if_changed(path: Path, new_text: str) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing_text = path.read_text(encoding="utf-8")
        if existing_text == new_text:
            return None
        backup_path = path.with_name(f"{path.name}.bak-{_timestamp_suffix()}")
        backup_path.write_text(existing_text, encoding="utf-8")
    else:
        backup_path = None
    path.write_text(new_text, encoding="utf-8")
    return backup_path


def _ensure_executable(path: Path) -> None:
    try:
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        return


def _timestamp_suffix() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
