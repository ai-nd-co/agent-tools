from __future__ import annotations

import json
from pathlib import Path

from agent_tools.claude_integration import (
    CLAUDE_STOP_HOOK_COMMAND,
    ENV_CLAUDE_INTEGRATION_TRIGGERED,
    is_claude_integration_triggered,
    load_claude_integration_status,
)


def test_claude_integration_trigger_env_constant_is_detected(monkeypatch: object) -> None:
    monkeypatch.setenv(ENV_CLAUDE_INTEGRATION_TRIGGERED, "1")
    assert is_claude_integration_triggered() is True


def test_load_claude_integration_status_detects_install(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    claude_home = tmp_path / ".claude"
    script_path = claude_home / "agent-tools" / "stop_tts.sh"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (claude_home / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": CLAUDE_STOP_HOOK_COMMAND,
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    status = load_claude_integration_status(claude_home)

    assert status.install_state == "installed"
    assert status.effective_enabled is True


def test_load_claude_integration_status_detects_missing_script(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    claude_home = tmp_path / ".claude"
    claude_home.mkdir()
    (claude_home / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": CLAUDE_STOP_HOOK_COMMAND,
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    status = load_claude_integration_status(claude_home)

    assert status.install_state == "broken"
    assert "hook-script-missing" in status.issues


def test_load_claude_integration_status_detects_missing_install(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")

    status = load_claude_integration_status(tmp_path / ".claude")

    assert status.install_state == "missing"
