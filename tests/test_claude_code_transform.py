from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent_tools.claude_code_transform import (
    ClaudeCodeTransformOptions,
    transform_with_claude_code,
)


def test_transform_with_claude_code_uses_constrained_command(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.claude_code_transform as transform_module
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")

    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = args[0]
        captured["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(
                {
                    "result": "spoken text",
                    "session_id": "session-1",
                    "usage": {"output_tokens": 12},
                    "is_error": False,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(transform_module.subprocess, "run", fake_run)

    result = transform_with_claude_code(
        ClaudeCodeTransformOptions(
            system_prompt="rewrite this",
            input_text="raw text",
            model="haiku",
            effort="low",
        )
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert "--max-turns" in command
    assert "--agent" in command
    assert "--tools" in command
    assert "--no-session-persistence" in command
    assert "--output-format" in command
    assert "--system-prompt" in command
    assert "--setting-sources" in command
    assert "--no-chrome" in command
    assert "--strict-mcp-config" in command
    assert "--mcp-config" in command
    assert "--disable-slash-commands" in command
    assert result.text == "spoken text"
    assert result.session_id == "session-1"
    assert Path(captured["cwd"]).name == "claude-transform"


def test_transform_with_claude_code_parses_event_list(
    monkeypatch: object,
) -> None:
    import agent_tools.claude_code_transform as transform_module

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args[0],
            1,
            stdout=json.dumps(
                [
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "Not logged in"}]},
                    },
                    {
                        "type": "result",
                        "result": "Not logged in",
                        "session_id": "session-2",
                        "is_error": True,
                    },
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(transform_module.subprocess, "run", fake_run)

    try:
        transform_with_claude_code(
            ClaudeCodeTransformOptions(system_prompt="prompt", input_text="raw")
        )
    except RuntimeError as exc:
        assert "Not logged in" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_transform_with_claude_code_rejects_opus() -> None:
    with pytest.raises(ValueError, match="Opus is not allowed"):
        transform_with_claude_code(
            ClaudeCodeTransformOptions(
                system_prompt="prompt",
                input_text="raw",
                model="opus",
            )
        )
