from __future__ import annotations

import json
import os
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_tools.codex_auth import AuthError, load_auth_state
from agent_tools.codex_config import resolve_codex_home
from agent_tools.runtime import load_preferences, save_preferences

PREFERENCE_CODEX_INTEGRATION_ENABLED = "codex_integration_enabled"
ENV_CODEX_INTEGRATION_TRIGGERED = "AGENT_TOOLS_CODEX_INTEGRATION_TRIGGERED"
WINDOWS_NOTIFY_COMMAND = "codex-notify-dispatch"
WINDOWS_NOTIFY_LAUNCHER = "agent-tools"

CodexIntegrationMode = Literal["notify", "stop-hook"]
CodexIntegrationInstallState = Literal["installed", "missing", "broken"]


@dataclass(frozen=True)
class CodexIntegrationStatus:
    mode: CodexIntegrationMode
    codex_home: Path
    config_path: Path
    enabled: bool
    available: bool
    install_state: CodexIntegrationInstallState
    hooks_json_path: Path | None = None
    hook_script_path: Path | None = None
    notify_command: tuple[str, ...] | None = None
    availability_issues: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def effective_enabled(self) -> bool:
        return self.install_state == "installed" and self.enabled


def load_codex_integration_enabled() -> bool:
    value = load_preferences().get(PREFERENCE_CODEX_INTEGRATION_ENABLED)
    if isinstance(value, bool):
        return value
    return True


def set_codex_integration_enabled(enabled: bool) -> None:
    preferences = load_preferences()
    preferences[PREFERENCE_CODEX_INTEGRATION_ENABLED] = enabled
    save_preferences(preferences)


def is_codex_integration_triggered() -> bool:
    return os.environ.get(ENV_CODEX_INTEGRATION_TRIGGERED) == "1"


def codex_integration_mode(platform_name: str | None = None) -> CodexIntegrationMode:
    current_platform = platform_name or sys.platform
    return "notify" if current_platform == "win32" else "stop-hook"


def load_codex_integration_status(
    codex_home: Path | None = None,
    *,
    platform_name: str | None = None,
) -> CodexIntegrationStatus:
    home = resolve_codex_home(codex_home)
    mode = codex_integration_mode(platform_name)
    enabled = load_codex_integration_enabled()
    config_path = home / "config.toml"
    available, availability_issues = _detect_codex_backend_availability(home)

    if mode == "notify":
        notify_command = _load_notify_command(config_path)
        install_state, issues = _detect_windows_notify_install_state(
            config_path,
            notify_command=notify_command,
        )
        return CodexIntegrationStatus(
            mode=mode,
            codex_home=home,
            config_path=config_path,
            enabled=enabled,
            available=available,
            install_state=install_state,
            notify_command=notify_command,
            availability_issues=availability_issues,
            issues=issues,
        )

    hooks_json_path = home / "hooks.json"
    hook_script_path = home / "hooks" / "stop_tts.sh"
    install_state, issues = _detect_stop_hook_install_state(
        config_path=config_path,
        hooks_json_path=hooks_json_path,
        hook_script_path=hook_script_path,
    )
    return CodexIntegrationStatus(
        mode=mode,
        codex_home=home,
        config_path=config_path,
        enabled=enabled,
        available=available,
        install_state=install_state,
        hooks_json_path=hooks_json_path,
        hook_script_path=hook_script_path,
        availability_issues=availability_issues,
        issues=issues,
    )


def codex_integration_status_text(status: CodexIntegrationStatus) -> str:
    if status.install_state == "installed":
        if status.enabled:
            mode_label = "Notify mode" if status.mode == "notify" else "Stop hook mode"
            return f"On - {mode_label}"
        return "Off - soft disabled"
    if status.install_state == "broken":
        return "Off - repair needed"
    return "Off - not installed"


def codex_integration_toggle_checked(status: CodexIntegrationStatus) -> bool:
    return status.effective_enabled


def _detect_windows_notify_install_state(
    config_path: Path,
    *,
    notify_command: tuple[str, ...] | None = None,
) -> tuple[CodexIntegrationInstallState, tuple[str, ...]]:
    if not config_path.exists():
        return "missing", ()

    config_text = config_path.read_text(encoding="utf-8")
    try:
        config = tomllib.loads(config_text)
    except tomllib.TOMLDecodeError:
        return "broken", ("config-invalid",)

    if not isinstance(config, dict):
        return "broken", ("config-invalid",)

    notify_matches = _matches_notify_command(config.get("notify"))
    features_value = _read_codex_hooks_feature(config)
    has_agenttools_trace = "codex-notify-dispatch" in config_text

    issues: list[str] = []
    if notify_matches and notify_command is not None:
        executable_issues = _validate_notify_executable(notify_command)
        if executable_issues:
            issues.extend(executable_issues)
    if notify_matches and features_value is False:
        if not issues:
            return "installed", ()
    if not notify_matches and not has_agenttools_trace and features_value is None:
        return "missing", ()

    if not notify_matches and has_agenttools_trace:
        issues.append("notify-missing")
    if notify_matches and features_value is not False:
        issues.append("features-inconsistent")
    if features_value not in (None, False):
        issues.append("features-inconsistent")
    if not issues:
        issues.append("notify-incomplete")
    return "broken", tuple(issues)


def _load_notify_command(config_path: Path) -> tuple[str, ...] | None:
    if not config_path.exists():
        return None
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return None
    if not isinstance(config, dict):
        return None
    value = config.get("notify")
    if not isinstance(value, list) or not all(isinstance(part, str) for part in value):
        return None
    return tuple(value)


def _validate_notify_executable(notify_command: tuple[str, ...]) -> tuple[str, ...]:
    if not notify_command:
        return ()
    issues: list[str] = []
    executable = notify_command[0]
    if _is_agent_tools_launcher(executable):
        if shutil.which(executable) is None:
            issues.append("notify-launcher-missing")
        return tuple(issues)
    if _is_python_command(executable) and not _is_path_like_command(executable):
        if shutil.which(executable) is None:
            issues.append("notify-executable-missing")
        return tuple(issues)

    executable_path = Path(executable).expanduser()
    if not executable_path.exists():
        issues.append("notify-executable-missing")
        return tuple(issues)
    if _is_python_command(executable):
        current_executable = Path(sys.executable).resolve()
        configured_executable = executable_path.resolve()
        if configured_executable != current_executable:
            issues.append("notify-python-mismatch")
    return tuple(issues)


def _detect_stop_hook_install_state(
    *,
    config_path: Path,
    hooks_json_path: Path,
    hook_script_path: Path,
) -> tuple[CodexIntegrationInstallState, tuple[str, ...]]:
    config_exists = config_path.exists()
    hooks_exists = hooks_json_path.exists()
    script_exists = hook_script_path.exists()

    config: dict[str, Any] | None = None
    if config_exists:
        try:
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            return "broken", ("config-invalid",)

    codex_hooks_enabled = _read_codex_hooks_feature(config) if config is not None else None

    hooks_payload: dict[str, Any] | None = None
    if hooks_exists:
        try:
            parsed = json.loads(hooks_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "broken", ("hooks-json-invalid",)
        if not isinstance(parsed, dict):
            return "broken", ("hooks-json-invalid",)
        hooks_payload = parsed

    has_stop_entry = _hooks_payload_has_agent_tools_stop_entry(hooks_payload)

    if codex_hooks_enabled is True and has_stop_entry and script_exists:
        return "installed", ()
    if not config_exists and not hooks_exists and not script_exists:
        return "missing", ()
    if codex_hooks_enabled is None and not has_stop_entry and not script_exists:
        return "missing", ()

    issues: list[str] = []
    if codex_hooks_enabled is not True:
        issues.append("feature-disabled")
    if not has_stop_entry and (hooks_exists or script_exists or codex_hooks_enabled is True):
        issues.append("stop-hook-missing")
    if not script_exists and (hooks_exists or codex_hooks_enabled is True):
        issues.append("hook-script-missing")
    if not issues:
        issues.append("stop-hook-incomplete")
    return "broken", tuple(issues)


def _detect_codex_backend_availability(codex_home: Path) -> tuple[bool, tuple[str, ...]]:
    try:
        load_auth_state(codex_home)
    except AuthError:
        return False, ("auth-unavailable",)
    return True, ()


def _read_codex_hooks_feature(config: dict[str, Any] | None) -> bool | None:
    if not isinstance(config, dict):
        return None
    features = config.get("features")
    if not isinstance(features, dict):
        return None
    value = features.get("codex_hooks")
    if isinstance(value, bool):
        return value
    return None


def _matches_notify_command(value: object) -> bool:
    if not isinstance(value, list):
        return False
    command = [str(part) for part in value if isinstance(part, str)]
    if len(command) != len(value):
        return False
    if len(command) == 2:
        return _is_agent_tools_launcher(command[0]) and command[1] == WINDOWS_NOTIFY_COMMAND
    return (
        len(command) >= 4
        and command[-1] == "codex-notify-dispatch"
        and command[-3] == "-m"
        and command[-2] == "agent_tools"
        and _is_python_command(command[0])
    )


def _is_agent_tools_launcher(value: str) -> bool:
    return Path(value).name.lower() in {WINDOWS_NOTIFY_LAUNCHER, f"{WINDOWS_NOTIFY_LAUNCHER}.exe"}


def _is_python_command(value: str) -> bool:
    lowered = Path(value).name.lower()
    return lowered in {"python", "python.exe", "py", "py.exe"} or lowered.startswith("python")


def _is_path_like_command(value: str) -> bool:
    return any(separator in value for separator in ("/", "\\", ":"))


def _hooks_payload_has_agent_tools_stop_entry(payload: dict[str, Any] | None) -> bool:
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
            if isinstance(command, str) and "stop_tts.sh" in command:
                return True
    return False
