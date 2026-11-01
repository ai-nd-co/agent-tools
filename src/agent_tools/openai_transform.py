from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from agent_tools.codex_config import (
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_BASE_URL,
    ENV_OPENAI_MODEL,
    ENV_OPENAI_TEMPERATURE,
    ENV_OPENAI_TOP_P,
    read_float_env,
    read_string_env,
)
from agent_tools.codex_private_api import TransformResult

MAX_ERROR_TEXT_CHARS = 500


class OpenAITransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAITransformOptions:
    system_prompt: str
    input_text: str
    model: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 120.0
    temperature: float | None = None
    top_p: float | None = None


def resolve_openai_base_url(base_url: str | None = None) -> str | None:
    """Return the server root ending in /v1, or None when no server is configured."""
    raw_value = base_url or read_string_env(ENV_OPENAI_BASE_URL)
    if raw_value is None:
        return None
    normalized = raw_value.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def resolve_openai_model(model: str | None = None) -> str | None:
    return model or read_string_env(ENV_OPENAI_MODEL)


def transform_with_openai(
    options: OpenAITransformOptions,
    *,
    transport: httpx.BaseTransport | None = None,
) -> TransformResult:
    base_url = resolve_openai_base_url(options.base_url)
    if base_url is None:
        raise OpenAITransportError(
            f"No OpenAI-compatible server configured. Pass --base-url or set {ENV_OPENAI_BASE_URL}."
        )
    model = resolve_openai_model(options.model)
    payload: dict[str, Any] = {
        "stream": False,
        "messages": [
            {"role": "system", "content": options.system_prompt},
            {"role": "user", "content": options.input_text},
        ],
    }
    # A single-model server such as llama-server answers without a model name.
    if model is not None:
        payload["model"] = model
    # Unset means the server's own sampling defaults.
    temperature = _first_set(options.temperature, read_float_env(ENV_OPENAI_TEMPERATURE))
    if temperature is not None:
        payload["temperature"] = temperature
    top_p = _first_set(options.top_p, read_float_env(ENV_OPENAI_TOP_P))
    if top_p is not None:
        payload["top_p"] = top_p
    headers: dict[str, str] = {}
    api_key = read_string_env(ENV_OPENAI_API_KEY)
    if api_key is not None:
        # h11 would quote a malformed value back in its error; refuse it without echoing it.
        if any(not "\x21" <= char <= "\x7e" for char in api_key):
            raise OpenAITransportError(
                f"{ENV_OPENAI_API_KEY} contains characters not allowed in an HTTP header."
            )
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = httpx.Timeout(options.timeout_seconds)
    try:
        with httpx.Client(timeout=timeout, transport=transport, follow_redirects=True) as client:
            response = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
    except httpx.HTTPError as exc:
        # No chaining: an invalid header makes h11 quote the Authorization value in its message.
        message = _redact(str(exc), api_key)
        raise OpenAITransportError(f"OpenAI-compatible request failed: {message}") from None
    if response.status_code >= 400:
        raise OpenAITransportError(
            f"OpenAI-compatible request failed with {response.status_code}: "
            f"{_redact(_safe_error_text(response), api_key)[:MAX_ERROR_TEXT_CHARS]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise OpenAITransportError("OpenAI-compatible response was not JSON.") from exc
    content = _first_choice_content(body)
    if content is None:
        raise OpenAITransportError("OpenAI-compatible response did not include message content.")

    raw_usage = body.get("usage") if isinstance(body, dict) else None
    raw_usage = raw_usage if isinstance(raw_usage, dict) else {}
    usage: dict[str, Any] = {
        "provider": "openai",
        "model": body.get("model") if isinstance(body.get("model"), str) else model,
        "prompt_tokens": _coerce_int(raw_usage.get("prompt_tokens")),
        "completion_tokens": _coerce_int(raw_usage.get("completion_tokens")),
        "total_tokens": _coerce_int(raw_usage.get("total_tokens")),
    }
    response_id = body.get("id") if isinstance(body.get("id"), str) else None
    return TransformResult(
        text=content,
        response_id=response_id,
        usage=usage,
        session_id=str(uuid.uuid4()),
    )


def _first_choice_content(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _first_set(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _redact(text: str, secret: str | None) -> str:
    if secret:
        text = text.replace(secret, "***")
    return text


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _safe_error_text(response: httpx.Response) -> str:
    text = response.text.strip()
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            text = error["message"].strip()
        elif isinstance(error, str) and error.strip():
            text = error.strip()
    return text or "<empty response body>"
