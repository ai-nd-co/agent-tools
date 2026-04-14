from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agent_tools.codex_auth import load_auth_state
from agent_tools.codex_private_api import (
    ClientSettings,
    CodexPrivateClient,
    CodexTransportError,
    build_request_headers,
    build_transform_request,
    consume_response_stream,
    extract_assistant_text,
)


def test_build_transform_request_includes_reasoning_and_fast() -> None:
    request = build_transform_request(
        model="gpt-5",
        system_prompt="rewrite",
        user_text="hello",
        session_id="session-1",
        reasoning_effort="medium",
        fast=True,
    )

    assert request["instructions"] == "rewrite"
    assert request["input"][0]["role"] == "user"
    assert request["reasoning"]["effort"] == "medium"
    assert request["service_tier"] == "priority"
    assert request["prompt_cache_key"] == "session-1"


def test_build_request_headers_sets_codex_fields() -> None:
    headers = build_request_headers(
        access_token="access",
        account_id="account",
        originator="codex_cli_rs",
        version="0.120.0",
        session_id="session-1",
    )

    assert headers["authorization"] == "Bearer access"
    assert headers["chatgpt-account-id"] == "account"
    assert headers["originator"] == "codex_cli_rs"
    assert headers["session_id"] == "session-1"


def test_consume_response_stream_accumulates_text_deltas() -> None:
    lines = [
        'data: {"type":"response.output_text.delta","delta":"hello "}',
        "",
        'data: {"type":"response.output_text.delta","delta":"world"}',
        "",
        'data: {"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":1}}}',
        "",
    ]
    result = consume_response_stream(lines, session_id="session-1")
    assert result.text == "hello world"
    assert result.response_id == "resp_1"
    assert result.usage == {"input_tokens": 1}


def test_consume_response_stream_falls_back_to_output_item() -> None:
    lines = [
        (
            'data: {"type":"response.output_item.done","item":{"type":"message",'
            '"role":"assistant","content":[{"type":"output_text","text":"fallback"}]}}'
        ),
        "",
        'data: {"type":"response.completed","response":{"id":"resp_1"}}',
        "",
    ]
    result = consume_response_stream(lines, session_id="session-1")
    assert result.text == "fallback"


def test_extract_assistant_text_returns_empty_for_non_assistant_items() -> None:
    assert extract_assistant_text([{"type": "message", "role": "user", "content": []}]) == ""


def test_consume_response_stream_raises_on_failed_event() -> None:
    lines = [
        'data: {"type":"response.failed","response":{"error":{"message":"nope"}}}',
        "",
    ]
    with pytest.raises(CodexTransportError, match="nope"):
        consume_response_stream(lines, session_id="session-1")


def test_private_client_reloads_or_refreshes_after_401(tmp_path: Path) -> None:
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "expired-token",
                    "refresh_token": "refresh-token",
                    "account_id": "account-1",
                },
            }
        ),
        encoding="utf-8",
    )

    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        if request.url == httpx.URL("https://auth.openai.com/oauth/token"):
            return httpx.Response(
                200,
                json={"access_token": "fresh-token", "refresh_token": "fresh-refresh"},
            )
        auth_header = request.headers.get("authorization")
        if auth_header == "Bearer expired-token":
            return httpx.Response(401, text="unauthorized")
        if auth_header == "Bearer fresh-token":
            content = (
                'data: {"type":"response.output_text.delta","delta":"done"}\n\n'
                'data: {"type":"response.completed","response":{"id":"resp_1"}}\n\n'
            )
            return httpx.Response(
                200,
                text=content,
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected auth header: {auth_header}")

    client = CodexPrivateClient(
        settings=ClientSettings(
            base_url="https://chatgpt.com/backend-api/codex",
            originator="codex_cli_rs",
            version="0.120.0",
        ),
        transport=httpx.MockTransport(handler),
    )
    auth_state = load_auth_state(codex_home)
    result = client.transform(
        auth_state=auth_state,
        system_prompt="rewrite",
        user_text="hello",
        model="gpt-5",
        reasoning_effort=None,
        fast=False,
    )

    assert result.text == "done"
    assert any(req.url == httpx.URL("https://auth.openai.com/oauth/token") for req in requests_seen)
    stored = json.loads((codex_home / "auth.json").read_text(encoding="utf-8"))
    assert stored["tokens"]["access_token"] == "fresh-token"
