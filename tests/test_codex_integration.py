from __future__ import annotations

import json
from pathlib import Path

from agent_tools.codex_integration import (
    ENV_CODEX_INTEGRATION_TRIGGERED,
    codex_integration_mode,
    load_codex_integration_enabled,
    load_codex_integration_status,
    set_codex_integration_enabled,
)


def test_codex_integration_mode_uses_platform() -> None:
    assert codex_integration_mode("win32") == "notify"
    assert codex_integration_mode("linux") == "stop-hook"


def test_load_codex_integration_enabled_defaults_true(monkeypatch: object, tmp_path: Path) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")

    assert load_codex_integration_enabled() is True


def test_set_codex_integration_enabled_persists_preference(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")

    set_codex_integration_enabled(False)

    assert load_codex_integration_enabled() is False


def test_load_codex_integration_status_detects_windows_notify_install(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "C:/Scripts/agent-tools.exe" if name == "agent-tools" else None,
    )
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        (
            'notify = ["agent-tools", "codex-notify-dispatch"]\n\n'
            "[features]\n"
            "codex_hooks = false\n"
        ),
        encoding="utf-8",
    )

    status = load_codex_integration_status(codex_home, platform_name="win32")

    assert status.mode == "notify"
    assert status.install_state == "installed"
    assert status.enabled is True
    assert status.effective_enabled is True


def test_load_codex_integration_status_detects_windows_broken_partial_install(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        'notify = ["C:/Python312/python.exe", "-m", "agent_tools", "codex-notify-dispatch"]\n',
        encoding="utf-8",
    )

    status = load_codex_integration_status(codex_home, platform_name="win32")

    assert status.install_state == "broken"
    assert "features-inconsistent" in status.issues


def test_load_codex_integration_status_detects_missing_agent_tools_launcher(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    monkeypatch.setattr("shutil.which", lambda name: None)
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        (
            'notify = ["agent-tools", "codex-notify-dispatch"]\n\n'
            "[features]\n"
            "codex_hooks = false\n"
        ),
        encoding="utf-8",
    )

    status = load_codex_integration_status(codex_home, platform_name="win32")

    assert status.install_state == "broken"
    assert "notify-launcher-missing" in status.issues


def test_load_codex_integration_status_detects_windows_notify_python_mismatch(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.codex_integration as codex_integration_module
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    configured_python = tmp_path / "OtherPython" / "python.exe"
    current_python = tmp_path / "Python312" / "python.exe"
    configured_python.parent.mkdir(parents=True, exist_ok=True)
    current_python.parent.mkdir(parents=True, exist_ok=True)
    configured_python.write_text("", encoding="utf-8")
    current_python.write_text("", encoding="utf-8")
    config_text = (
        f'notify = ["{configured_python.as_posix()}", "-m", '
        '"agent_tools", "codex-notify-dispatch"]\n\n'
        "[features]\n"
        "codex_hooks = false\n"
    )
    (codex_home / "config.toml").write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(
        codex_integration_module.sys,
        "executable",
        str(current_python.resolve()),
    )

    status = load_codex_integration_status(codex_home, platform_name="win32")

    assert status.install_state == "broken"
    assert "notify-python-mismatch" in status.issues
    assert status.notify_command is not None


def test_load_codex_integration_status_detects_stop_hook_install(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    codex_home = tmp_path / ".codex"
    hook_dir = codex_home / "hooks"
    hook_dir.mkdir(parents=True)
    (codex_home / "config.toml").write_text(
        "[features]\n"
        "codex_hooks = true\n",
        encoding="utf-8",
    )
    (codex_home / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "bash -lc '$HOME/.codex/hooks/stop_tts.sh'",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (hook_dir / "stop_tts.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    status = load_codex_integration_status(codex_home, platform_name="linux")

    assert status.mode == "stop-hook"
    assert status.install_state == "installed"
    assert status.hooks_json_path == codex_home / "hooks.json"
    assert status.hook_script_path == hook_dir / "stop_tts.sh"


def test_load_codex_integration_status_detects_stop_hook_soft_disabled(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    set_codex_integration_enabled(False)

    codex_home = tmp_path / ".codex"
    hook_dir = codex_home / "hooks"
    hook_dir.mkdir(parents=True)
    (codex_home / "config.toml").write_text(
        "[features]\n"
        "codex_hooks = true\n",
        encoding="utf-8",
    )
    (codex_home / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "bash -lc '$HOME/.codex/hooks/stop_tts.sh'",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (hook_dir / "stop_tts.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    status = load_codex_integration_status(codex_home, platform_name="linux")

    assert status.install_state == "installed"
    assert status.enabled is False
    assert status.effective_enabled is False


def test_load_codex_integration_status_detects_missing_stop_hook_script(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "[features]\n"
        "codex_hooks = true\n",
        encoding="utf-8",
    )
    (codex_home / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "bash -lc '$HOME/.codex/hooks/stop_tts.sh'",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    status = load_codex_integration_status(codex_home, platform_name="linux")

    assert status.install_state == "broken"
    assert "hook-script-missing" in status.issues


def test_codex_integration_trigger_env_constant_is_stable() -> None:
    assert ENV_CODEX_INTEGRATION_TRIGGERED == "AGENT_TOOLS_CODEX_INTEGRATION_TRIGGERED"
