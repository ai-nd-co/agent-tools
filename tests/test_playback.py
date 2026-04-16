from __future__ import annotations

import builtins

import pytest

from agent_tools.playback import detect_playback_capabilities, plan_playback


def test_plan_playback_prefers_controller_queue_on_windows(monkeypatch: object) -> None:
    import agent_tools.playback as playback_module

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name.startswith("PySide6"):
            return real_import("types", *args, **kwargs)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(playback_module.importlib, "import_module", fake_import)

    plan = plan_playback(platform_name="win32")

    assert plan.available is True
    assert plan.strategy == "controller-queue"
    assert plan.backend == "pyside6-qmediaplayer"


def test_detect_playback_capabilities_reports_actionable_linux_player_error(
    monkeypatch: object,
) -> None:
    import agent_tools.playback as playback_module

    monkeypatch.setattr(playback_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        playback_module.importlib,
        "import_module",
        lambda _name: object(),
    )

    capabilities = detect_playback_capabilities(platform_name="linux")

    assert capabilities.direct.available is False
    assert capabilities.direct.error_message is not None
    assert "paplay" in capabilities.direct.error_message
    assert "aplay" in capabilities.direct.error_message
    assert "ffplay" in capabilities.direct.error_message
    assert capabilities.controller_queue.available is True


def test_detect_playback_capabilities_uses_first_linux_player_on_path(
    monkeypatch: object,
) -> None:
    import agent_tools.playback as playback_module

    monkeypatch.setattr(
        playback_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == "paplay" else None,
    )
    monkeypatch.setattr(
        playback_module.importlib,
        "import_module",
        lambda _name: object(),
    )

    capabilities = detect_playback_capabilities(platform_name="linux")

    assert capabilities.direct.available is True
    assert capabilities.direct.backend == "paplay"


def test_plan_playback_uses_direct_playback_on_macos(monkeypatch: object) -> None:
    import agent_tools.playback as playback_module

    monkeypatch.setattr(
        playback_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == "afplay" else None,
    )
    monkeypatch.setattr(
        playback_module.importlib,
        "import_module",
        lambda _name: object(),
    )

    plan = plan_playback(platform_name="darwin")

    assert plan.available is True
    assert plan.strategy == "direct"
    assert plan.backend == "afplay"


def test_plan_playback_requires_controller_runtime_on_windows(monkeypatch: object) -> None:
    import agent_tools.playback as playback_module

    def fake_import(_name: str) -> object:
        raise ImportError("missing pyside6")

    monkeypatch.setattr(playback_module.importlib, "import_module", fake_import)

    plan = plan_playback(platform_name="win32")

    assert plan.available is False
    assert plan.error_message is not None
    assert "PySide6" in plan.error_message


def test_plan_playback_rejects_unknown_platform() -> None:
    plan = plan_playback(platform_name="freebsd")

    assert plan.available is False
    assert plan.error_message == "Playback is supported only on Windows, macOS, and Linux."


def test_detect_playback_capabilities_reports_actionable_macos_error(
    monkeypatch: object,
) -> None:
    import agent_tools.playback as playback_module

    monkeypatch.setattr(playback_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        playback_module.importlib,
        "import_module",
        lambda _name: object(),
    )

    capabilities = detect_playback_capabilities(platform_name="darwin")

    assert capabilities.direct.available is False
    assert capabilities.direct.error_message is not None
    assert "afplay" in capabilities.direct.error_message
    assert "ffplay" in capabilities.direct.error_message
