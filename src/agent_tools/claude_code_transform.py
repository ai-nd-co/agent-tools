from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any

from agent_tools.codex_config import normalize_claude_code_model
from agent_tools.codex_private_api import TransformResult
from agent_tools.runtime import app_root

DEFAULT_CLAUDE_CODE_MODEL = "haiku"
DEFAULT_CLAUDE_CODE_EFFORT = "low"
EMPTY_MCP_CONFIG = '{"mcpServers":{}}'


@dataclass(frozen=True)
class ClaudeCodeTransformOptions:
    system_prompt: str
    input_text: str
    model: str = DEFAULT_CLAUDE_CODE_MODEL
    effort: str | None = DEFAULT_CLAUDE_CODE_EFFORT
    bare: bool = False
    claude_command: str = "claude"
    timeout_seconds: float = 120.0


def transform_with_claude_code(options: ClaudeCodeTransformOptions) -> TransformResult:
    working_dir = app_root() / "claude-transform"
    working_dir.mkdir(parents=True, exist_ok=True)
    model = normalize_claude_code_model(options.model)

    command = [
        options.claude_command,
        "-p",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--max-turns",
        "1",
        "--agent",
        "general-purpose",
        "--tools",
        "",
        "--permission-mode",
        "default",
        "--model",
        model,
        "--system-prompt",
        options.system_prompt,
        "--setting-sources",
        "local",
        "--no-chrome",
        "--strict-mcp-config",
        "--mcp-config",
        EMPTY_MCP_CONFIG,
        "--disable-slash-commands",
    ]
    if options.effort:
        command.extend(["--effort", options.effort])
    if options.bare:
        command.append("--bare")

    env = os.environ.copy()
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")

    try:
        completed = subprocess.run(
            command,
            input=options.input_text,
            text=True,
            capture_output=True,
            cwd=working_dir,
            env=env,
            timeout=options.timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Claude Code CLI was not found on PATH.") from exc

    parsed = _parse_claude_json_output(completed.stdout)
    if parsed["is_error"]:
        raise RuntimeError(f"Claude Code transform failed: {parsed['text']}")
    if completed.returncode != 0 and not parsed["text"]:
        stderr = completed.stderr.strip()
        raise RuntimeError(stderr or "Claude Code transform failed.")

    return TransformResult(
        text=parsed["text"],
        response_id=None,
        usage=parsed["usage"],
        session_id=parsed["session_id"] or str(uuid.uuid4()),
    )


def _parse_claude_json_output(stdout: str) -> dict[str, Any]:
    payload = stdout.strip()
    if not payload:
        return {"text": "", "usage": None, "session_id": None, "is_error": True}

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {"text": payload, "usage": None, "session_id": None, "is_error": False}

    if isinstance(parsed, dict):
        result_text = parsed.get("result")
        session_id = parsed.get("session_id")
        return {
            "text": result_text if isinstance(result_text, str) else "",
            "usage": parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None,
            "session_id": session_id if isinstance(session_id, str) else None,
            "is_error": bool(parsed.get("is_error")),
        }

    if isinstance(parsed, list):
        result_event = _find_result_event(parsed)
        assistant_text = _find_assistant_text(parsed)
        result_value = result_event.get("result")
        text = result_value if isinstance(result_value, str) else assistant_text
        usage = result_event.get("usage") if isinstance(result_event.get("usage"), dict) else None
        result_session_id = result_event.get("session_id")
        session_id = result_session_id if isinstance(result_session_id, str) else None
        return {
            "text": text,
            "usage": usage,
            "session_id": session_id,
            "is_error": bool(result_event.get("is_error")),
        }

    return {"text": "", "usage": None, "session_id": None, "is_error": True}


def _find_result_event(events: list[object]) -> dict[str, Any]:
    for event in reversed(events):
        if isinstance(event, dict) and event.get("type") == "result":
            return event
    return {}


def _find_assistant_text(events: list[object]) -> str:
    for event in reversed(events):
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)
    return ""
