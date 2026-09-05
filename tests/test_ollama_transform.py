from __future__ import annotations

import httpx
import pytest

from agent_tools.ollama_transform import (
    OllamaTransformOptions,
    OllamaTransportError,
    ollama_available,
    resolve_ollama_base_url,
    transform_with_ollama,
)


def test_resolve_ollama_base_url_strips_api_suffix() -> None:
    assert resolve_ollama_base_url("http://localhost:11434/api") == "http://localhost:11434"
    assert resolve_ollama_base_url("http://localhost:11434/v1") == "http://localhost:11434"


def test_ollama_available_checks_version_endpoint() -> None:
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(200, json={"version": "0.21.0"})

    available, issue = ollama_available(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(handler),
    )

    assert available is True
    assert issue is None
    assert requests_seen[0].url == httpx.URL("http://localhost:11434/api/version")


def test_transform_with_ollama_reads_message_and_usage() -> None:
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "gemma4:e4b",
                "message": {"role": "assistant", "content": "rewritten"},
                "prompt_eval_count": 12,
                "eval_count": 5,
                "total_duration": 50,
                "load_duration": 10,
                "prompt_eval_duration": 20,
                "eval_duration": 20,
                "done": True,
            },
        )

    result = transform_with_ollama(
        OllamaTransformOptions(
            system_prompt="rewrite",
            input_text="hello",
            model="gemma4:e4b",
            base_url="http://localhost:11434",
        ),
        transport=httpx.MockTransport(handler),
    )

    assert requests_seen[0].url == httpx.URL("http://localhost:11434/api/chat")
    assert result.text == "rewritten"
    assert result.usage is not None
    assert result.usage["total_tokens"] == 17
    assert result.usage["model"] == "gemma4:e4b"


def test_transform_with_ollama_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    with pytest.raises(OllamaTransportError, match="boom"):
        transform_with_ollama(
            OllamaTransformOptions(
                system_prompt="rewrite",
                input_text="hello",
                model="gemma4:e4b",
            ),
            transport=httpx.MockTransport(handler),
        )
