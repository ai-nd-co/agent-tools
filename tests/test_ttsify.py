from __future__ import annotations

from pathlib import Path

from agent_tools.codex_private_api import TransformResult
from agent_tools.transformer import TransformOptions
from agent_tools.tts import TtsResult
from agent_tools.ttsify import TtsifyOptions, load_ttsify_prompt, ttsify_text


def test_load_ttsify_prompt_has_expected_contract() -> None:
    prompt = load_ttsify_prompt()
    assert "Table policy:" in prompt
    assert "File path policy:" in prompt
    assert "Rewrite any path to the shortest meaningful spoken label." in prompt
    assert "Do not speak literal slash characters" in prompt
    assert "do not read every cell verbatim" in prompt.lower()
    assert "Convert paths into the most meaningful user-facing file name" in prompt
    assert "Time policy:" in prompt
    assert "Do not use AM or PM." in prompt
    assert "01:16 UTC" in prompt
    assert "Numbers policy:" in prompt
    assert "Logs and errors policy:" in prompt
    assert "Code policy:" in prompt
    assert "Reference policy:" in prompt
    assert "Acronyms policy:" in prompt
    assert "Quotes and parentheses policy:" in prompt
    assert "Boilerplate policy:" in prompt
    assert "Core narration rule:" in prompt
    assert "Output only the final TTS-ready text." in prompt


def test_ttsify_uses_repo_defaults_and_env(monkeypatch: object) -> None:
    import agent_tools.ttsify as ttsify_module

    captured: dict[str, object] = {}

    def fake_transform_text(input_text: str, options: TransformOptions) -> TransformResult:
        captured["transform_input"] = input_text
        captured["model"] = options.model
        captured["reasoning_effort"] = options.reasoning_effort
        captured["prompt_text"] = options.system_prompt_text
        return TransformResult(
            text="spoken text",
            response_id="resp_1",
            usage={"input_tokens": 1},
            session_id="session-1",
        )

    def fake_synthesize_wav(
        text: str,
        *,
        voice: str,
        language: str | None,
        speed: float,
        device: str,
    ) -> TtsResult:
        captured["tts_text"] = text
        captured["voice"] = voice
        captured["language"] = language
        captured["speed"] = speed
        captured["device"] = device
        return TtsResult(wav=b"WAV", sample_rate=24_000, chunks=1)

    monkeypatch.setattr(ttsify_module, "transform_text", fake_transform_text)
    monkeypatch.setattr(ttsify_module, "synthesize_wav", fake_synthesize_wav)
    monkeypatch.setenv("AGENT_TOOLS_CODEX_REASONING_EFFORT", "medium")
    monkeypatch.setenv("AGENT_TOOLS_KOKORO_VOICE", "bf_emma")
    monkeypatch.setenv("AGENT_TOOLS_KOKORO_LANGUAGE", "b")
    monkeypatch.setenv("AGENT_TOOLS_KOKORO_SPEED", "1.25")
    monkeypatch.setenv("AGENT_TOOLS_KOKORO_DEVICE", "cpu")

    result = ttsify_text("raw text", TtsifyOptions())

    assert result.transformed_text == "spoken text"
    assert captured["transform_input"] == "raw text"
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["reasoning_effort"] == "medium"
    assert isinstance(captured["prompt_text"], str)
    assert captured["tts_text"] == "spoken text"
    assert captured["voice"] == "bf_emma"
    assert captured["language"] == "b"
    assert captured["speed"] == 1.25
    assert captured["device"] == "cpu"


def test_ttsify_cli_values_override_env(monkeypatch: object) -> None:
    import agent_tools.ttsify as ttsify_module

    captured: dict[str, object] = {}

    def fake_transform_text(_input_text: str, options: TransformOptions) -> TransformResult:
        captured["model"] = options.model
        captured["reasoning_effort"] = options.reasoning_effort
        return TransformResult(
            text="cli override text",
            response_id="resp_1",
            usage=None,
            session_id="session-1",
        )

    def fake_synthesize_wav(
        _text: str,
        *,
        voice: str,
        language: str | None,
        speed: float,
        device: str,
    ) -> TtsResult:
        captured["voice"] = voice
        captured["language"] = language
        captured["speed"] = speed
        captured["device"] = device
        return TtsResult(wav=b"WAV", sample_rate=24_000, chunks=1)

    monkeypatch.setattr(ttsify_module, "transform_text", fake_transform_text)
    monkeypatch.setattr(ttsify_module, "synthesize_wav", fake_synthesize_wav)
    monkeypatch.setenv("AGENT_TOOLS_CODEX_MODEL", "env-model")
    monkeypatch.setenv("AGENT_TOOLS_KOKORO_VOICE", "env-voice")

    ttsify_text(
        "raw text",
        TtsifyOptions(
            model="cli-model",
            reasoning_effort="high",
            voice="af_heart",
            language="a",
            speed=1.5,
            device="cuda",
        ),
    )

    assert captured["model"] == "cli-model"
    assert captured["reasoning_effort"] == "high"
    assert captured["voice"] == "af_heart"
    assert captured["language"] == "a"
    assert captured["speed"] == 1.5
    assert captured["device"] == "cuda"


def test_ttsify_rejects_invalid_env_device(monkeypatch: object) -> None:
    import pytest

    monkeypatch.setenv("AGENT_TOOLS_KOKORO_DEVICE", "neural-engine")

    with pytest.raises(ValueError, match="Unsupported Kokoro device"):
        ttsify_text("raw text", TtsifyOptions())


def test_transformer_accepts_system_prompt_text(tmp_path: Path, monkeypatch: object) -> None:
    import agent_tools.transformer as transformer_module

    class FakeClient:
        def __init__(self, settings: object) -> None:
            self.settings = settings

        def transform(self, **kwargs: object) -> TransformResult:
            assert kwargs["system_prompt"] == "prompt from memory"
            return TransformResult(
                text="done",
                response_id="resp_1",
                usage=None,
                session_id="session-1",
            )

    monkeypatch.setattr(
        transformer_module,
        "load_codex_defaults",
        lambda _codex_home=None: transformer_module.CodexDefaults(
            codex_home=tmp_path,
            model=None,
            reasoning_effort=None,
            version="0.0.0",
            originator="codex_cli_rs",
            base_url="https://chatgpt.com/backend-api/codex",
        ),
    )
    monkeypatch.setattr(transformer_module, "load_auth_state", lambda _codex_home=None: object())
    monkeypatch.setattr(transformer_module, "CodexPrivateClient", FakeClient)

    result = transformer_module.transform_text(
        "hello",
        transformer_module.TransformOptions(system_prompt_text="prompt from memory"),
    )

    assert result.text == "done"
