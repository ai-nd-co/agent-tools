from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from agent_tools.codex_auth import (
    AuthError,
    AuthState,
    PermanentRefreshError,
    refresh_chatgpt_tokens,
    reload_auth_state,
)
from agent_tools.codex_config import build_user_agent

ReasoningEffort = str
PROMPT_CACHE_NAMESPACE = uuid.UUID("75f0f17d-8430-5a44-a76d-678d5aee9c5c")


class CodexTransportError(RuntimeError):
    pass


class UnauthorizedResponse(CodexTransportError):
    pass


@dataclass(frozen=True)
class TransformResult:
    text: str
    response_id: str | None
    usage: dict[str, Any] | None
    session_id: str


@dataclass(frozen=True)
class ClientSettings:
    base_url: str
    originator: str
    version: str
    timeout_seconds: float = 120.0


class CodexPrivateClient:
    def __init__(
        self,
        settings: ClientSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport

    def transform(
        self,
        *,
        auth_state: AuthState,
        system_prompt: str,
        user_text: str,
        model: str,
        reasoning_effort: ReasoningEffort | None,
        fast: bool,
    ) -> TransformResult:
        session_id = str(uuid.uuid4())
        request_body = build_transform_request(
            model=model,
            system_prompt=system_prompt,
            user_text=user_text,
            session_id=session_id,
            reasoning_effort=reasoning_effort,
            fast=fast,
        )
        current_auth = auth_state
        reloaded_once = False
        refreshed_once = False

        while True:
            try:
                return self._send_once(current_auth, request_body, session_id=session_id)
            except UnauthorizedResponse as exc:
                if not reloaded_once:
                    reloaded_once = True
                    reloaded, changed = reload_auth_state(current_auth)
                    if changed:
                        if reloaded.tokens.account_id != current_auth.tokens.account_id:
                            raise AuthError(
                                "Codex auth account changed on disk during retry."
                            ) from exc
                        current_auth = reloaded
                        continue
                if not refreshed_once:
                    refreshed_once = True
                    current_auth = refresh_chatgpt_tokens(
                        current_auth,
                        originator=self.settings.originator,
                        version=self.settings.version,
                        timeout_seconds=self.settings.timeout_seconds,
                        transport=self._transport,
                    )
                    continue
                raise
            except PermanentRefreshError:
                raise

    def _send_once(
        self,
        auth_state: AuthState,
        request_body: dict[str, Any],
        *,
        session_id: str,
    ) -> TransformResult:
        url = f"{self.settings.base_url.rstrip('/')}/responses"
        headers = build_request_headers(
            access_token=auth_state.tokens.access_token,
            account_id=auth_state.tokens.account_id,
            originator=self.settings.originator,
            version=self.settings.version,
            session_id=session_id,
        )
        timeout = httpx.Timeout(self.settings.timeout_seconds, read=None)
        with httpx.Client(
            timeout=timeout,
            transport=self._transport,
            follow_redirects=True,
        ) as client:
            with client.stream("POST", url, headers=headers, json=request_body) as response:
                if response.status_code == 401:
                    raise UnauthorizedResponse(_safe_response_text(response))
                if response.status_code >= 400:
                    message = _safe_response_text(response)
                    raise CodexTransportError(
                        f"Codex backend request failed with {response.status_code}: {message}"
                    )
                return consume_response_stream(response.iter_lines(), session_id=session_id)


def build_transform_request(
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    session_id: str,
    reasoning_effort: ReasoningEffort | None,
    fast: bool,
) -> dict[str, Any]:
    prompt_cache_key = str(uuid.uuid5(PROMPT_CACHE_NAMESPACE, system_prompt))
    request: dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            }
        ],
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "stream": True,
        "include": [],
        "prompt_cache_key": prompt_cache_key,
    }
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        request["include"] = ["reasoning.encrypted_content"]
    if fast:
        request["service_tier"] = "priority"
    return request


def build_request_headers(
    *,
    access_token: str,
    account_id: str,
    originator: str,
    version: str,
    session_id: str,
) -> dict[str, str]:
    return {
        "authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "content-type": "application/json",
        "accept": "text/event-stream",
        "originator": originator,
        "user-agent": build_user_agent(originator, version),
        "session_id": session_id,
        "version": version,
    }


def iter_sse_payloads(lines: Iterable[str]) -> Iterator[str]:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def consume_response_stream(lines: Iterable[str], *, session_id: str) -> TransformResult:
    output_text: list[str] = []
    assistant_items: list[dict[str, Any]] = []
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    completed = False

    for payload in iter_sse_payloads(lines):
        if payload == "[DONE]":
            continue
        event = json.loads(payload)
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                output_text.append(delta)
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                assistant_items.append(item)
        elif event_type == "response.completed":
            response = event.get("response")
            if isinstance(response, dict):
                raw_response_id = response.get("id")
                if isinstance(raw_response_id, str):
                    response_id = raw_response_id
                raw_usage = response.get("usage")
                if isinstance(raw_usage, dict):
                    usage = raw_usage
            completed = True
        elif event_type == "response.failed":
            raise CodexTransportError(_build_failed_response_message(event))
        elif event_type == "response.incomplete":
            raise CodexTransportError("Codex backend returned response.incomplete.")

    if not completed:
        raise CodexTransportError("Codex stream closed before response.completed.")

    text = "".join(output_text)
    if not text:
        text = extract_assistant_text(assistant_items)
    return TransformResult(text=text, response_id=response_id, usage=usage, session_id=session_id)


def extract_assistant_text(items: list[dict[str, Any]]) -> str:
    for item in reversed(items):
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        parts: list[str] = []
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") != "output_text":
                continue
            text = content_item.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)
    return ""


def _build_failed_response_message(event: dict[str, Any]) -> str:
    response = event.get("response")
    if isinstance(response, dict):
        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
    return "Codex backend returned response.failed."


def _safe_response_text(response: httpx.Response) -> str:
    try:
        text = response.text
    except httpx.ResponseNotRead:
        text = response.read().decode("utf-8", "replace")
    redacted = text.replace("Bearer ", "Bearer [REDACTED]")
    return redacted.strip() or f"HTTP {response.status_code}"
