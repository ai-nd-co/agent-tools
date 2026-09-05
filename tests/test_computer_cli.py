from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_tools import cli as root_cli
from agent_tools.computer import cli as computer_cli
from agent_tools.computer import uia_winapp
from agent_tools.computer.command_registry import all_command_names
from agent_tools.computer.models import SCHEMA_VERSION, ComputerError
from agent_tools.computer.rendering import render_human_error
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
    assert payload["error"]["code"] == "invalid_argument_value"
    assert payload["error"]["details"]["argument"] == "--hwnd"
    assert payload["error"]["details"]["retryable"] is False
    assert "not-an-integer" not in captured.out


def test_argument_errors_are_distinct_structured_and_sanitized(capsys) -> None:
    """Regression (validation 2026-09-06 backlog item 3).

    Depth 0, max-elements 999999, unknown --grep, missing --hwnd, and a
    non-integer --hwnd previously yielded the identical opaque invalid_arguments
    envelope; each class must now carry its own code, sanitized details, and a
    next action, without echoing caller argument values.
    """
    cases = [
        (
            ["computer", "inspect", "--hwnd", "1", "--depth", "0", "--json"],
            "argument_out_of_range",
            {"argument": "--depth", "minimum": 1, "maximum": 12},
            ["0"],
        ),
        (
            ["computer", "inspect", "--hwnd", "1", "--max-elements", "999999", "--json"],
            "argument_out_of_range",
            {"argument": "--max-elements", "minimum": 1, "maximum": 200},
            ["999999"],
        ),
        (
            ["computer", "inspect", "--hwnd", "1", "--grep", "Button", "--json"],
            "unknown_argument",
            {"unknown_arguments": ["--grep"]},
            ["Button"],
        ),
        (
            ["computer", "inspect", "--json"],
            "missing_required_argument",
            {"required_arguments": ["--hwnd"]},
            [],
        ),
        (
            ["computer", "read", "--hwnd", "not-an-integer", "--element", "x", "--json"],
            "invalid_argument_value",
            {"argument": "--hwnd", "expected_type": "int"},
            ["not-an-integer"],
        ),
        (
            ["computer", "scroll", "--hwnd", "1", "--percent", "200", "--json"],
            "argument_out_of_range",
            {"argument": "--percent", "minimum": 0, "maximum": 100},
            ["200"],
        ),
    ]
    for argv, code, expected_details, forbidden_fragments in cases:
        result = root_cli.main(argv)
        assert result == 2
        captured = capsys.readouterr()
        assert captured.err == ""
        payload = json.loads(captured.out)
        assert payload["ok"] is False
        assert payload["data"] is None
        assert payload["error"]["code"] == code, argv
        details = payload["error"]["details"]
        assert details["retryable"] is False
        assert isinstance(details["next_action"], str) and details["next_action"]
        for key, value in expected_details.items():
            assert details[key] == value, (argv, key)
        for fragment in forbidden_fragments:
            assert fragment not in captured.out, (argv, fragment)


def test_argument_error_human_output_includes_next_action(capsys) -> None:
    result = root_cli.main(["computer", "inspect", "--hwnd", "1", "--depth", "0"])

    assert result == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error [argument_out_of_range]:" in captured.err
    assert "Retryable: no" in captured.err
    assert "Next:" in captured.err
    assert "between 1 and 12" in captured.err


def test_unknown_argument_sanitizes_attached_values_and_dash_prefixed_values(capsys) -> None:
    """Audit 2026-09-06 finding 1.

    Unknown-argument sanitization must strip attached values (`--token=SECRET`),
    drop dash-prefixed values copied from the unrecognized fragment, and bound
    the reported option-name shape; caller values must never reach the message
    or details in either form.
    """
    result = root_cli.main(
        ["computer", "inspect", "--hwnd", "1", "--token=TESTSECRET", "--json"]
    )
    assert result == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["error"]["code"] == "unknown_argument"
    assert payload["error"]["details"]["unknown_arguments"] == ["--token"]
    assert "TESTSECRET" not in captured.out

    result = root_cli.main(
        ["computer", "inspect", "--hwnd", "1", "--grep", "-TESTSECRET", "--json"]
    )
    assert result == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["error"]["code"] == "unknown_argument"
    assert payload["error"]["details"]["unknown_arguments"] == ["--grep"]
    assert "TESTSECRET" not in captured.out


def test_multiline_unknown_argument_cannot_inject_argparse_error_text(capsys) -> None:
    """Audit 2026-09-06 finding 1.

    Recovered stderr text is untrusted: a newline inside an unknown token must
    not be able to fabricate the last argparse error line and leak text through
    the choice classifier.
    """
    injected = "evil\nx: error: invalid choice: x (choose from TESTSECRET)"
    result = root_cli.main(["computer", "inspect", "--hwnd", "1", injected, "--json"])

    assert result == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["error"]["code"] == "invalid_arguments"
    assert payload["error"]["details"]["retryable"] is False
    assert "TESTSECRET" not in captured.out


def test_recovered_choice_text_is_validated_against_registered_actions(
    capsys, tmp_path: Path
) -> None:
    """Audit 2026-09-06 finding 1.

    An injected argument-prefixed choice line must not be treated as parser
    metadata: allowed choices may only be emitted when they match the actions
    the parser actually registered.
    """
    injected = "evil\nx: error: argument --profile: invalid choice: 'y' (choose from TESTSECRET)"
    result = root_cli.main(
        [
            "computer",
            "screenshot",
            "--hwnd",
            "1",
            "--output",
            str(tmp_path / "capture.png"),
            injected,
            "--json",
        ]
    )

    assert result == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["error"]["code"] != "invalid_argument_choice"
    assert "TESTSECRET" not in captured.out


def test_required_flag_names_drop_argparse_list_commas(capsys) -> None:
    """Audit 2026-09-06 finding 1 hardening.

    argparse joins multi-argument required lists with ", "; the sanitized flag
    names must not carry that punctuation into messages or details.
    """
    result = root_cli.main(["computer", "key", "--hwnd", "1", "--json"])

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "missing_required_argument"
    assert payload["error"]["details"]["required_arguments"] == [
        "--from-capture",
        "--image-at",
        "--key",
        "--staged-input",
        "--draft-evidence",
    ]


def test_human_root_parser_failures_are_sanitized(capsys) -> None:
    """Audit 2026-09-06 finding 2.

    Parent/root parser failures in human mode must route through the same
    classified, sanitized renderer as JSON mode: no raw usage, no raw argparse
    message, and the promised retry/next-action fields.
    """
    result = root_cli.main(["computer", "inspect", "--hwnd", "1", "--grep", "TESTSECRET"])

    assert result == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" not in captured.err
    assert "unrecognized arguments" not in captured.err
    assert "Error [unknown_argument]:" in captured.err
    assert "Retryable: no" in captured.err
    assert "Next:" in captured.err
    assert "TESTSECRET" not in captured.err

    result = root_cli.main(["computer", "TESTSECRET"])

    assert result == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" not in captured.err
    assert "choose from" not in captured.err
    assert "Error [invalid_argument_choice]:" in captured.err
    assert "Retryable: no" in captured.err
    assert "Next:" in captured.err
    assert "TESTSECRET" not in captured.err


def test_invalid_computer_command_envelope_never_echoes_the_token(capsys) -> None:
    """Audit 2026-09-06 findings 3 and 4.

    The complete serialized envelope must use a command name only after
    registry validation, and an invalid subcommand must classify as
    invalid_argument_choice with the parser-registered choices, never the
    submitted token.
    """
    result = root_cli.main(["computer", "TESTSECRET", "--json"])

    assert result == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["command"] == "computer.unknown"
    assert payload["ok"] is False
    assert payload["data"] is None
    assert payload["error"]["code"] == "invalid_argument_choice"
    details = payload["error"]["details"]
    assert details["argument"] == "computer_command"
    assert details["choices"] == all_command_names()
    assert details["retryable"] is False
    assert isinstance(details["next_action"], str) and details["next_action"]
    assert "TESTSECRET" not in captured.out


def test_invalid_argument_choice_reports_registered_choices_without_value(
    capsys, tmp_path: Path
) -> None:
    """Audit 2026-09-06 finding 4.

    Real parser-entry choice failures must reach the distinct
    invalid_argument_choice classifier carrying the argument identity and the
    allowed choices, without the submitted value.
    """
    result = root_cli.main(
        [
            "computer",
            "screenshot",
            "--hwnd",
            "1",
            "--output",
            str(tmp_path / "capture.png"),
            "--profile",
            "TESTSECRET",
            "--json",
        ]
    )

    assert result == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["command"] == "computer.screenshot"
    assert payload["error"]["code"] == "invalid_argument_choice"
    details = payload["error"]["details"]
    assert details["argument"] == "--profile"
    assert details["choices"] == ["compact", "standard", "native"]
    assert details["retryable"] is False
    assert isinstance(details["next_action"], str) and details["next_action"]
    assert "TESTSECRET" not in captured.out


def test_identity_errors_carry_recovery_next_actions() -> None:
    identity_error = uia_winapp._uia_identity_unavailable_error(
        "The element's exact UI Automation identity could not be verified."
    )
    assert identity_error.code == "uia_identity_unavailable"
    assert identity_error.details["retryable"] is True
    assert "inspect" in identity_error.details["next_action"].casefold()

    not_found = uia_winapp._element_not_found_error()
    assert not_found.code == "element_not_found"
    assert not_found.details["retryable"] is True
    assert "inspect" in not_found.details["next_action"].casefold()

    rendered = render_human_error(identity_error)
    assert "Retryable: yes" in rendered
    assert "Next:" in rendered


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
    oversized = "1" + "0" * 400

    with pytest.raises(ComputerError) as raised:
        parser.parse_args(["computer", "inspect", "--hwnd", "1", "--depth", oversized])

    assert raised.value.code == "argument_out_of_range"
    assert raised.value.exit_code == 2
    assert oversized not in raised.value.message


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
