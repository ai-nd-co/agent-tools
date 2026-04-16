from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from agent_tools.codex_private_api import TransformResult
from agent_tools.tts import TtsResult
from agent_tools.ttsify import TtsifyResult


def test_run_ttsify_short_circuits_for_disabled_codex_integration(
    monkeypatch: object,
    capsys: object,
) -> None:
    import agent_tools.cli as cli_module

    monkeypatch.setattr(cli_module, "is_codex_integration_triggered", lambda: True)
    monkeypatch.setattr(cli_module, "is_claude_integration_triggered", lambda: False)
    monkeypatch.setattr(cli_module, "load_codex_integration_enabled", lambda: False)
    monkeypatch.setattr(
        cli_module,
        "ttsify_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ttsify_text should not run when integration is disabled")
        ),
    )

    result = cli_module._run_ttsify(
        argparse.Namespace(
            model=None,
            reasoning_effort=None,
            fast=False,
            voice=None,
            language=None,
            speed=None,
            device=None,
            input_file=None,
            output_file="-",
            output_mode="write",
            source=None,
            codex_home=None,
            base_url=None,
            originator=None,
            timeout_seconds=120.0,
        )
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "AgentTools integration disabled" in captured.err


def test_run_ttsify_keeps_manual_path_unchanged(monkeypatch: object, tmp_path: Path) -> None:
    import agent_tools.cli as cli_module

    output_path = tmp_path / "out.wav"
    input_path = tmp_path / "input.txt"
    input_path.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(cli_module, "is_codex_integration_triggered", lambda: False)
    monkeypatch.setattr(cli_module, "is_claude_integration_triggered", lambda: False)
    monkeypatch.setattr(cli_module, "load_codex_integration_enabled", lambda: False)
    monkeypatch.setattr(
        cli_module,
        "ttsify_text",
        lambda input_text, _options: TtsifyResult(
            transformed_text=input_text.upper(),
            transform_result=TransformResult(
                text=input_text.upper(),
                response_id="resp_1",
                usage=None,
                session_id="session-1",
            ),
            tts_result=TtsResult(wav=b"WAV", sample_rate=24_000, chunks=1),
            model="gpt-5.4-mini",
            reasoning_effort=None,
            voice="af_heart",
            language="a",
            speed=1.0,
            device="auto",
            resolved_device="cpu",
        ),
    )

    result = cli_module._run_ttsify(
        argparse.Namespace(
            model=None,
            reasoning_effort=None,
            fast=False,
            voice=None,
            language=None,
            speed=None,
            device=None,
            input_file=input_path,
            output_file=str(output_path),
            output_mode="write",
            source=None,
            codex_home=None,
            base_url=None,
            originator=None,
            timeout_seconds=120.0,
        )
    )

    assert result == 0
    assert output_path.read_bytes() == b"WAV"


def test_cli_rejects_opus_for_claude_model() -> None:
    import agent_tools.cli as cli_module

    parser = cli_module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "transform",
                "--system-prompt-file",
                "prompt.md",
                "--provider",
                "claude-code",
                "--claude-model",
                "opus",
            ]
        )
