from __future__ import annotations

import json
from pathlib import Path

from agent_tools.codex_integration import load_codex_integration_enabled
from agent_tools.hook_install import (
    CLAUDE_STOP_HOOK_COMMAND,
    STOP_HOOK_COMMAND,
    WINDOWS_NOTIFY_COMMAND,
    build_updated_claude_settings_payload,
    build_updated_hooks_payload,
    ensure_feature_assignment,
    ensure_notify_command,
    install_claude_integration,
    install_codex_integration,
    install_codex_stop_hook,
)


def test_ensure_notify_command_appends_before_first_section() -> None:
    updated = ensure_notify_command(
        '[features]\njs_repl = true\n',
        ("C:\\Python312\\python.exe", "-m", "agent_tools", WINDOWS_NOTIFY_COMMAND),
    )
    expected_prefix = (
        'notify = ["C:\\\\Python312\\\\python.exe", "-m", "agent_tools", '
        '"codex-notify-dispatch"]\n\n[features]'
    )
    assert updated.startswith(expected_prefix)


def test_ensure_notify_command_replaces_existing_notify() -> None:
    updated = ensure_notify_command(
        'notify = ["old.exe"]\nmodel = "gpt-5"\n',
        ("C:\\Python312\\python.exe", "-m", "agent_tools", WINDOWS_NOTIFY_COMMAND),
    )
    assert 'notify = ["old.exe"]' not in updated
    assert 'codex-notify-dispatch' in updated


def test_ensure_feature_assignment_updates_existing_section() -> None:
    updated = ensure_feature_assignment(
        'model = "gpt-5"\n\n[features]\njs_repl = true\n',
        "codex_hooks",
        "false",
    )
    assert "[features]" in updated
    assert "js_repl = true" in updated
    assert "codex_hooks = false" in updated


def test_build_updated_hooks_payload_preserves_existing_stop_hooks(tmp_path: Path) -> None:
    hooks_json_path = tmp_path / "hooks.json"
    hooks_json_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "bash -lc 'echo existing-stop-hook'",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    payload = build_updated_hooks_payload(hooks_json_path)
    stop_entries = payload["hooks"]["Stop"]

    assert len(stop_entries) == 2
    assert stop_entries[0]["hooks"][0]["command"] == "bash -lc 'echo existing-stop-hook'"
    assert stop_entries[1]["hooks"][0]["command"] == STOP_HOOK_COMMAND


def test_build_updated_claude_settings_payload_preserves_existing_stop_hooks(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "bash -lc 'echo existing-claude-hook'",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    payload = build_updated_claude_settings_payload(settings_path)
    stop_entries = payload["hooks"]["Stop"]

    assert len(stop_entries) == 2
    assert stop_entries[0]["hooks"][0]["command"] == "bash -lc 'echo existing-claude-hook'"
    assert stop_entries[1]["hooks"][0]["command"] == CLAUDE_STOP_HOOK_COMMAND


def test_install_codex_integration_writes_windows_notify_config(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    codex_home = tmp_path / ".codex"

    result = install_codex_integration(codex_home, platform_name="win32")

    assert result.mode == "notify"
    assert result.config_path.exists()
    assert result.notify_command is not None
    config_text = result.config_path.read_text(encoding="utf-8")
    assert "codex-notify-dispatch" in config_text
    assert "codex_hooks = false" in config_text
    assert result.hooks_json_path is None
    assert result.hook_script_path is None


def test_install_codex_stop_hook_writes_files(monkeypatch: object, tmp_path: Path) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    codex_home = tmp_path / ".codex"

    result = install_codex_stop_hook(codex_home)

    assert result.mode == "stop-hook"
    assert result.config_path.exists()
    assert result.hooks_json_path is not None
    assert result.hook_script_path is not None
    assert "codex_hooks = true" in result.config_path.read_text(encoding="utf-8")
    hooks_payload = json.loads(result.hooks_json_path.read_text(encoding="utf-8"))
    assert hooks_payload["hooks"]["Stop"][0]["hooks"][0]["command"] == STOP_HOOK_COMMAND
    script_text = result.hook_script_path.read_text(encoding="utf-8")
    assert "stop_tts.log" in script_text
    assert "hook_start" in script_text


def test_install_codex_integration_marks_preference_enabled(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    codex_home = tmp_path / ".codex"

    install_codex_integration(codex_home, platform_name="win32")

    assert load_codex_integration_enabled() is True


def test_install_claude_integration_writes_settings_and_script(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    claude_home = tmp_path / ".claude"

    result = install_claude_integration(claude_home)

    assert result.settings_path.exists()
    assert result.hook_script_path.exists()
    settings_payload = json.loads(result.settings_path.read_text(encoding="utf-8"))
    assert settings_payload["hooks"]["Stop"][0]["hooks"][0]["command"] == CLAUDE_STOP_HOOK_COMMAND
    script_text = result.hook_script_path.read_text(encoding="utf-8")
    assert "AGENT_TOOLS_CLAUDE_INTEGRATION_TRIGGERED=1" in script_text
    assert "stop_tts.log" in script_text
