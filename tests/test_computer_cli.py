from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_tools import cli as root_cli
from agent_tools.computer import cli as computer_cli
from agent_tools.computer.models import SCHEMA_VERSION, ComputerError
from agent_tools.computer.win32_backend import load_win32_modules


def test_computer_group_help_distinguishes_reads_from_guarded_actions(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        root_cli.main(["computer", "--help"])

    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "inspection is read-only" in output
    assert "guarded mutations" in output


def _sample_info(*, background: bool) -> dict:
    data = {
        "identity": {
            "available": True,
            "host": "host",
            "user": "user",
            "session_id": 1,
            "session_name": "Console",
            "interactive_desktop_access": True,
            "lock_state": "unlocked",
            "elevated": False,
            "integrity": "medium",
        },
        "clock": {
            "available": True,
            "local_iso": "2026-08-29T20:00:00+05:00",
            "timezone": "UZT",
            "utc_offset_seconds": 18_000,
        },
        "system": {
            "available": True,
            "windows": {"major": 10, "minor": 0, "build": 26200},
            "architecture": "AMD64",
            "uptime_seconds": 100,
            "power": {"ac": "online", "battery_percent": None},
        },
        "display": {
            "available": True,
            "virtual_bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
            "primary_bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
            "monitor_count": 1,
            "scaling_percent": 100,
        },
        "focus": {"available": True, "active": False, "window": None},
        "visible_apps": {"available": True, "count": 0, "truncated": False, "items": []},
        "media": {"available": True, "active": False, "session": None},
        "network": {
            "available": True,
            "connectivity": "internet",
            "interface": "Wi-Fi",
            "interface_type": "802.11",
            "primary_address": "192.0.2.2",
            "wifi": {"available": True, "connected": False},
            "tailscale": {"available": True, "status": "running", "addresses": []},
        },
        "readiness": {
            "available": True,
            "agent_tools_version": "0.7.1",
            "capabilities": ["info", "windows", "focused", "screenshot"],
            "backends": {
                "win32": {"available": True},
                "uia_winapp": {"available": False, "status": "not_installed"},
            },
            "action_lock": "locked",
            "actions_available": True,
            "actions_disabled": False,
            "desktop_usable": True,
            "action_blocked_reason": None,
        },
    }
    if background:
        data["background"] = {
            "available": True,
            "count": 1,
            "truncated": False,
            "items": [{"pid": 5, "name": "worker.exe", "session_id": 1, "state": "running"}],
        }
    return data


def test_info_human_output_states_background_policy(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        computer_cli,
        "collect_info",
        lambda *, background, backend: _sample_info(background=background),
    )

    assert root_cli.main(["computer", "info"]) == 0

    output = capsys.readouterr().out
    assert "Background processes: hidden" in output
    assert "use --background" in output
    assert "rg or jq" in output
    assert "uia_winapp:not_installed" in output
    assert "actions=available" in output
    assert "desktop=yes" in output
    assert "blocked=-" in output
    assert "lock=locked" in output


def test_info_json_has_versioned_envelope_and_explicit_background(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        computer_cli,
        "collect_info",
        lambda *, background, backend: _sample_info(background=background),
    )

    assert root_cli.main(["computer", "info", "--json"]) == 0
    default_payload = json.loads(capsys.readouterr().out)
    assert default_payload["schema_version"] == SCHEMA_VERSION
    assert default_payload["command"] == "computer.info"
    assert default_payload["ok"] is True
    assert "background" not in default_payload["data"]

    assert root_cli.main(["computer", "info", "--background", "--json"]) == 0
    background_payload = json.loads(capsys.readouterr().out)
    assert "background" in background_payload["data"]
    assert set(background_payload["data"]) - set(default_payload["data"]) == {"background"}


def test_screenshot_error_is_stable_json(monkeypatch, capsys, tmp_path: Path) -> None:
    def fail_capture(**_kwargs):
        raise ComputerError("stale_window", "The window changed before capture.")

    monkeypatch.setattr(computer_cli, "capture_screenshot", fail_capture)

    result = root_cli.main(
        [
            "computer",
            "screenshot",
            "--hwnd",
            "123",
            "--output",
            str(tmp_path / "capture.png"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "stale_window"
    assert payload["data"] is None


def test_semantic_capability_failure_renders_guarded_physical_fallback_in_json_and_human(
    monkeypatch,
    capsys,
) -> None:
    fallback = {
        "kind": "guarded_physical_input",
        "status": "available",
        "reason": "requires_fresh_capture_and_explicit_allow_physical",
    }

    def fail_invoke(**_kwargs):
        raise ComputerError(
            "capability_missing",
            "The semantic action provider is unavailable.",
            exit_code=2,
            details={"outcome": "rejected", "fallback": fallback},
        )

    monkeypatch.setattr(computer_cli, "invoke_element", fail_invoke)
    arguments = [
        "computer",
        "invoke",
        "--hwnd",
        "123",
        "--element",
        "fixture-control",
    ]

    assert root_cli.main([*arguments, "--json"]) == 2
    json_output = capsys.readouterr()
    payload = json.loads(json_output.out)
    assert json_output.err == ""
    assert payload["error"]["details"] == {
        "outcome": "rejected",
        "fallback": fallback,
    }

    assert root_cli.main(arguments) == 2
    human_output = capsys.readouterr()
    assert human_output.out == ""
    assert "Fallback: guarded_physical_input" in human_output.err
    assert "status=available" in human_output.err
    assert "requires_fresh_capture_and_explicit_allow_physical" in human_output.err


def test_parser_failure_is_stable_json(capsys, tmp_path: Path) -> None:
    result = root_cli.main(
        [
            "computer",
            "screenshot",
            "--hwnd",
            "not-an-integer",
            "--output",
            str(tmp_path / "capture.png"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert captured.err == ""
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["command"] == "computer.screenshot"
    assert payload["error"]["code"] == "invalid_arguments"


def test_set_value_forwards_opt_in_keyboard_focus_requirement(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def set_value(**kwargs):
        captured.update(kwargs)
        return {"method": "uia.ValuePattern", "outcome": "postcondition_verified"}

    monkeypatch.setattr(computer_cli, "set_element_value", set_value)

    assert root_cli.main(
        [
            "computer",
            "set-value",
            "--hwnd",
            "123",
            "--element",
            "fixture-edit",
            "--value",
            "fixture",
            "--require-keyboard-focus",
            "--json",
        ]
    ) == 0

    capsys.readouterr()
    assert captured["require_keyboard_focus"] is True


def test_screenshot_profiles_default_to_standard_and_native_is_explicit() -> None:
    parser = root_cli.build_parser()

    default = parser.parse_args(
        ["computer", "screenshot", "--hwnd", "1", "--output", "capture.png"]
    )
    compact = parser.parse_args(
        [
            "computer",
            "screenshot",
            "--hwnd",
            "1",
            "--output",
            "capture.png",
            "--profile",
            "compact",
        ]
    )
    native = parser.parse_args(
        [
            "computer",
            "screenshot",
            "--hwnd",
            "1",
            "--output",
            "capture.png",
            "--profile",
            "native",
        ]
    )

    assert default.profile == "standard"
    assert compact.profile == "compact"
    assert native.profile == "native"


def test_deep_semantic_inspection_is_explicit_and_bounded() -> None:
    parser = root_cli.build_parser()

    parsed = parser.parse_args(
        ["computer", "inspect", "--hwnd", "1", "--depth", "12", "--json"]
    )

    assert parsed.depth == 12


def test_semantic_integer_bounds_reject_oversized_values_without_overflow() -> None:
    parser = root_cli.build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(
            ["computer", "inspect", "--hwnd", "1", "--depth", "1" + "0" * 400]
        )

    assert raised.value.code == 2


def test_non_windows_command_fails_without_import_breakage(monkeypatch, capsys) -> None:
    load_win32_modules.cache_clear()
    monkeypatch.setattr(sys, "platform", "linux")

    assert root_cli.main(["computer", "windows", "--json"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "unsupported_platform"
    load_win32_modules.cache_clear()


def test_non_windows_cold_import_does_not_load_windows_backends() -> None:
    source_root = Path(__file__).parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    probe = """
import builtins
import sys

real_import = builtins.__import__

def guarded(name, *args, **kwargs):
    if name.startswith(("win32", "PIL")):
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded
from agent_tools import cli
parsed = cli.build_parser().parse_args(
    ["transform", "--system-prompt-file", "prompt.md"]
)
print(parsed.command)
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "transform"


def test_non_windows_parser_keeps_unrelated_commands_available(monkeypatch) -> None:
    load_win32_modules.cache_clear()
    monkeypatch.setattr(sys, "platform", "linux")

    parser = root_cli.build_parser()
    parsed = parser.parse_args(
        ["transform", "--system-prompt-file", "prompt.md", "--input-file", "input.txt"]
    )

    assert parsed.command == "transform"
    load_win32_modules.cache_clear()


def test_parser_exposes_only_read_only_computer_commands() -> None:
    parser = root_cli.build_parser()
    for command in (
        "info",
        "windows",
        "focused",
        "screenshot",
        "inspect",
        "read",
        "scroll-areas",
        "capabilities",
    ):
        args = ["computer", command]
        if command == "screenshot":
            args += ["--hwnd", "1", "--output", "capture.png"]
        elif command == "inspect":
            args += ["--hwnd", "1"]
        elif command == "read":
            args += ["--hwnd", "1", "--element", "doc-fixture-a1b2"]
        elif command == "scroll-areas":
            args += ["--hwnd", "1"]
        parsed = parser.parse_args(args)
        assert parsed.computer_command == command


def test_physical_parser_keeps_staging_and_enter_separate() -> None:
    parser = root_cli.build_parser()
    capture_id = "cap_" + "a" * 32
    staged_id = "stage_" + "b" * 32

    staged = parser.parse_args(
        [
            "computer",
            "type",
            "--hwnd",
            "123",
            "--from-capture",
            capture_id,
            "--image-at",
            "10,20",
            "--text",
            "draft without submission",
            "--allow-physical",
            "--visible-tab-verified",
        ]
    )
    enter = parser.parse_args(
        [
            "computer",
            "key",
            "--hwnd",
            "123",
            "--from-capture",
            capture_id,
            "--image-at",
            "10,20",
            "--key",
            "ENTER",
            "--staged-input",
            staged_id,
            "--draft-evidence",
            "collapsed_visible_match",
            "--allow-physical",
            "--visible-tab-verified",
        ]
    )

    assert staged.computer_command == "type"
    assert staged.text == "draft without submission"
    assert not hasattr(staged, "staged_input")
    assert enter.computer_command == "key"
    assert enter.key == "ENTER"
    assert enter.staged_input == staged_id
    assert enter.draft_evidence == "collapsed_visible_match"


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("move-pointer", []),
        ("click", ["--double"]),
        ("wheel", ["--horizontal", "-3"]),
    ],
)
def test_physical_pointer_and_wheel_parser_require_capture_authority(
    command: str,
    extra: list[str],
) -> None:
    parser = root_cli.build_parser()
    arguments = [
        "computer",
        command,
        "--hwnd",
        "123",
        "--from-capture",
        "cap_" + "c" * 32,
        "--image-at",
        "10,20",
        "--allow-physical",
        *extra,
    ]

    parsed = parser.parse_args(arguments)

    assert parsed.from_capture.startswith("cap_")
    assert parsed.image_at == "10,20"
    assert parsed.allow_physical is True
