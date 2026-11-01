from __future__ import annotations

import json
import traceback

import httpx
import pytest

from agent_tools.codex_private_api import TransformResult
from agent_tools.openai_transform import (
    OpenAITransformOptions,
    OpenAITransportError,
    resolve_openai_base_url,
    transform_with_openai,
)
from agent_tools.transformer import TransformOptions
from agent_tools.tts import TtsResult
from agent_tools.ttsify import TtsifyOptions, ttsify_text

ENV_NAMES = (
    "AGENT_TOOLS_OPENAI_BASE_URL",
    "AGENT_TOOLS_OPENAI_MODEL",
    "AGENT_TOOLS_OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _completion(content: str = "rewritten") -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "model": "gemma-4-31b-it-ud-iq2_m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


def test_resolve_openai_base_url_normalizes_to_v1() -> None:
    assert resolve_openai_base_url("http://llm:8080") == "http://llm:8080/v1"
    assert resolve_openai_base_url("http://llm:8080/v1/") == "http://llm:8080/v1"
    assert resolve_openai_base_url("http://llm:8080/v1/chat/completions") == "http://llm:8080/v1"
    assert resolve_openai_base_url(None) is None


def test_resolve_openai_base_url_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TOOLS_OPENAI_BASE_URL", "http://env-llm:8080/v1")
    assert resolve_openai_base_url() == "http://env-llm:8080/v1"
    assert resolve_openai_base_url("http://flag:1") == "http://flag:1/v1"


def test_transform_with_openai_sends_prompt_and_reads_usage() -> None:
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(200, json=_completion())

    result = transform_with_openai(
        OpenAITransformOptions(
            system_prompt="rewrite",
            input_text="hello",
            model="gemma-4-31b-it-ud-iq2_m",
            base_url="http://llm:8080/v1",
        ),
        transport=httpx.MockTransport(handler),
    )

    request = requests_seen[0]
    assert request.url == httpx.URL("http://llm:8080/v1/chat/completions")
    assert "authorization" not in request.headers
    payload = json.loads(request.content)
    assert payload["model"] == "gemma-4-31b-it-ud-iq2_m"
    assert payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "rewrite"},
        {"role": "user", "content": "hello"},
    ]
    assert result.text == "rewritten"
    assert result.response_id == "chatcmpl-1"
    assert result.usage is not None
    assert result.usage["provider"] == "openai"
    assert result.usage["total_tokens"] == 17


def test_transform_with_openai_omits_model_and_sends_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_TOOLS_OPENAI_BASE_URL", "http://env-llm:8080")
    monkeypatch.setenv("AGENT_TOOLS_OPENAI_API_KEY", "test-key")
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(200, json=_completion())

    transform_with_openai(
        OpenAITransformOptions(system_prompt="rewrite", input_text="hello"),
        transport=httpx.MockTransport(handler),
    )

    request = requests_seen[0]
    assert request.url == httpx.URL("http://env-llm:8080/v1/chat/completions")
    assert request.headers["authorization"] == "Bearer test-key"
    assert "model" not in json.loads(request.content)


def test_transform_with_openai_reads_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TOOLS_OPENAI_MODEL", "env-model")
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(200, json=_completion())

    transform_with_openai(
        OpenAITransformOptions(system_prompt="p", input_text="t", base_url="http://llm"),
        transport=httpx.MockTransport(handler),
    )

    assert json.loads(requests_seen[0].content)["model"] == "env-model"


def test_transform_with_openai_requires_a_base_url() -> None:
    with pytest.raises(OpenAITransportError, match="AGENT_TOOLS_OPENAI_BASE_URL"):
        transform_with_openai(OpenAITransformOptions(system_prompt="p", input_text="t"))


def test_transform_with_openai_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "Loading model", "code": 503}})

    with pytest.raises(OpenAITransportError, match="503: Loading model"):
        transform_with_openai(
            OpenAITransformOptions(system_prompt="p", input_text="t", base_url="http://llm"),
            transport=httpx.MockTransport(handler),
        )


def test_transform_with_openai_raises_on_missing_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(OpenAITransportError, match="message content"):
        transform_with_openai(
            OpenAITransformOptions(system_prompt="p", input_text="t", base_url="http://llm"),
            transport=httpx.MockTransport(handler),
        )


def test_transform_with_openai_wraps_transport_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OpenAITransportError, match="refused"):
        transform_with_openai(
            OpenAITransformOptions(system_prompt="p", input_text="t", base_url="http://llm"),
            transport=httpx.MockTransport(handler),
        )


def test_explicit_openai_provider_needs_a_base_url() -> None:
    from agent_tools.transformer import resolve_effective_transform_provider

    with pytest.raises(RuntimeError, match="openai is not available"):
        resolve_effective_transform_provider("openai")
    assert resolve_effective_transform_provider("openai", base_url="http://llm") == "openai"


def test_transform_text_routes_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_tools.transformer as transformer_module

    captured: dict[str, object] = {}

    def fake_transform_with_openai(options: OpenAITransformOptions) -> TransformResult:
        captured["options"] = options
        return TransformResult(text="spoken", response_id=None, usage=None, session_id="s")

    monkeypatch.setattr(transformer_module, "transform_with_openai", fake_transform_with_openai)

    result = transformer_module.transform_text(
        "raw",
        TransformOptions(
            system_prompt_text="prompt",
            provider="openai",
            model="m",
            base_url="http://llm",
            timeout_seconds=30.0,
        ),
    )

    options = captured["options"]
    assert isinstance(options, OpenAITransformOptions)
    assert (options.system_prompt, options.input_text, options.model) == ("prompt", "raw", "m")
    assert (options.base_url, options.timeout_seconds) == ("http://llm", 30.0)
    assert result.text == "spoken"


def test_ttsify_openai_provider_does_not_inherit_the_codex_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_tools.ttsify as ttsify_module

    monkeypatch.setenv("AGENT_TOOLS_CODEX_MODEL", "gpt-5.4-mini")
    captured: dict[str, object] = {}

    def fake_transform_text(_input_text: str, options: TransformOptions) -> TransformResult:
        captured["provider"] = options.provider
        captured["model"] = options.model
        return TransformResult(text="spoken", response_id=None, usage=None, session_id="s")

    def fake_synthesize_wav(_text: str, **_kwargs: object) -> TtsResult:
        return TtsResult(wav=b"WAV", sample_rate=24_000, chunks=1)

    monkeypatch.setattr(ttsify_module, "transform_text", fake_transform_text)
    monkeypatch.setattr(ttsify_module, "synthesize_wav", fake_synthesize_wav)
    monkeypatch.setattr(
        ttsify_module,
        "resolve_effective_transform_provider",
        lambda provider, *, codex_home=None, base_url=None: provider or "codex",
    )

    result = ttsify_text("raw", TtsifyOptions(provider="openai", base_url="http://llm"))
    assert captured == {"provider": "openai", "model": None}
    assert result.model == "server-default"
    assert result.reasoning_effort is None

    monkeypatch.setenv("AGENT_TOOLS_OPENAI_MODEL", "gemma")
    result = ttsify_text("raw", TtsifyOptions(provider="openai", base_url="http://llm"))
    assert captured["model"] == "gemma"
    assert result.model == "gemma"


def test_transform_with_openai_redacts_an_echoed_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TOOLS_OPENAI_API_KEY", "sk-test-secret")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key sk-test-secret"}})

    with pytest.raises(OpenAITransportError) as excinfo:
        transform_with_openai(
            OpenAITransformOptions(system_prompt="p", input_text="t", base_url="http://llm"),
            transport=httpx.MockTransport(handler),
        )

    assert "sk-test-secret" not in str(excinfo.value)
    assert "bad key ***" in str(excinfo.value)


def test_transport_error_hides_the_key_and_its_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TOOLS_OPENAI_API_KEY", "sk-test-secret")

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["authorization"]
        raise httpx.LocalProtocolError(f"Illegal header value {auth!r}")

    with pytest.raises(OpenAITransportError) as excinfo:
        transform_with_openai(
            OpenAITransformOptions(system_prompt="p", input_text="t", base_url="http://llm"),
            transport=httpx.MockTransport(handler),
        )

    assert "sk-test-secret" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


@pytest.mark.parametrize("key", ["sk-first\nsecond", "sk-tab\tkey", "sk-ключ"])
def test_transform_with_openai_rejects_a_key_unfit_for_a_header(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.setenv("AGENT_TOOLS_OPENAI_API_KEY", key)

    def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("a malformed key must not reach the transport")

    with pytest.raises(OpenAITransportError) as excinfo:
        transform_with_openai(
            OpenAITransformOptions(system_prompt="p", input_text="t", base_url="http://llm"),
            transport=httpx.MockTransport(handler),
        )

    rendered = "".join(traceback.format_exception(excinfo.value))
    for fragment in ("sk-first", "second", "sk-tab", "sk-ключ"):
        assert fragment not in rendered
    assert "AGENT_TOOLS_OPENAI_API_KEY" in str(excinfo.value)
