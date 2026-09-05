from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from agent_tools.codex_config import DEFAULT_OLLAMA_BASE_URL, ENV_OLLAMA_BASE_URL, read_string_env
from agent_tools.codex_private_api import TransformResult


class OllamaTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaTransformOptions:
    system_prompt: str
    input_text: str
    model: str
    base_url: str | None = None
    timeout_seconds: float = 120.0


def resolve_ollama_base_url(base_url: str | None = None) -> str:
    raw_value = base_url or read_string_env(ENV_OLLAMA_BASE_URL) or DEFAULT_OLLAMA_BASE_URL
    normalized = raw_value.rstrip("/")
    for suffix in ("/api", "/v1"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def ollama_available(
    *,
    base_url: str | None = None,
    timeout_seconds: float = 2.0,
    transport: httpx.BaseTransport | None = None,
) -> tuple[bool, str | None]:
    url = f"{resolve_ollama_base_url(base_url)}/api/version"
    timeout = httpx.Timeout(timeout_seconds)
    try:
        with httpx.Client(timeout=timeout, transport=transport, follow_redirects=True) as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        return False, str(exc)
    if response.status_code >= 400:
        return False, _safe_error_text(response)
    return True, None


def transform_with_ollama(
    options: OllamaTransformOptions,
    *,
    transport: httpx.BaseTransport | None = None,
) -> TransformResult:
    base_url = resolve_ollama_base_url(options.base_url)
    url = f"{base_url}/api/chat"
    payload = {
        "model": options.model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": options.system_prompt},
            {"role": "user", "content": options.input_text},
        ],
    }
    timeout = httpx.Timeout(options.timeout_seconds)
    try:
        with httpx.Client(timeout=timeout, transport=transport, follow_redirects=True) as client:
            response = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise OllamaTransportError(f"Ollama request failed: {exc}") from exc
    if response.status_code >= 400:
        raise OllamaTransportError(
            f"Ollama request failed with {response.status_code}: {_safe_error_text(response)}"
        )

    body = response.json()
    message = body.get("message")
    if not isinstance(message, dict):
        raise OllamaTransportError("Ollama response did not include a message object.")
    content = message.get("content")
    if not isinstance(content, str):
        raise OllamaTransportError("Ollama response did not include message content.")

    prompt_eval_count = _coerce_int(body.get("prompt_eval_count"))
    eval_count = _coerce_int(body.get("eval_count"))
    total_tokens = None
    if prompt_eval_count is not None or eval_count is not None:
        total_tokens = (prompt_eval_count or 0) + (eval_count or 0)

    usage: dict[str, Any] = {
        "provider": "ollama",
        "model": body.get("model") if isinstance(body.get("model"), str) else options.model,
        "total_duration": _coerce_int(body.get("total_duration")),
        "load_duration": _coerce_int(body.get("load_duration")),
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": _coerce_int(body.get("prompt_eval_duration")),
        "eval_count": eval_count,
        "eval_duration": _coerce_int(body.get("eval_duration")),
        "total_tokens": total_tokens,
    }

    return TransformResult(
        text=content,
        response_id=None,
        usage=usage,
        session_id=str(uuid.uuid4()),
    )


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _safe_error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or "<empty response body>"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
    text = response.text.strip()
    return text or "<empty response body>"
