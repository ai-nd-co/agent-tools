from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from agent_tools.codex_private_api import TransformResult
from agent_tools.tts import TtsMetrics, TtsResult
from agent_tools.ttsify import TtsifyMetrics, TtsifyResult


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
    perf_events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        cli_module,
        "append_perf_event",
        lambda event, **fields: perf_events.append((event, fields)),
    )
    monkeypatch.setattr(
        cli_module,
        "resolve_effective_transform_provider",
        lambda provider, *, codex_home=None: provider or "codex",
    )
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
            metrics=TtsifyMetrics(transform_ms=12.0, tts_ms=34.0, total_ms=46.0),
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
    assert perf_events and perf_events[0][0] == "ttsify_completed"
    assert perf_events[0][1]["provider"] == "codex"
    assert perf_events[0][1]["transform_ms"] == 12.0
    assert perf_events[0][1]["tts_ms"] == 34.0
    assert perf_events[0][1]["output_mode"] == "write"


def test_run_transform_logs_perf_entry(monkeypatch: object, tmp_path: Path) -> None:
    import agent_tools.cli as cli_module

    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    prompt_path = tmp_path / "prompt.txt"
    input_path.write_text("hello", encoding="utf-8")
    prompt_path.write_text("prompt", encoding="utf-8")

    perf_events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        cli_module,
        "append_perf_event",
        lambda event, **fields: perf_events.append((event, fields)),
    )
    monkeypatch.setattr(
        cli_module,
        "resolve_effective_transform_provider",
        lambda provider, *, codex_home=None: provider or "codex",
    )
    monkeypatch.setattr(
        cli_module,
        "transform_text",
        lambda input_text, _options: TransformResult(
            text=input_text.upper(),
            response_id="resp_1",
            usage=None,
            session_id="session-1",
        ),
    )

    result = cli_module._run_transform(
        argparse.Namespace(
            system_prompt_file=prompt_path,
            provider=None,
            model=None,
            reasoning_effort=None,
            fast=False,
            claude_model=None,
            claude_effort=None,
            claude_bare=False,
            input_file=input_path,
            output_file=str(output_path),
            codex_home=None,
            base_url=None,
            originator=None,
            timeout_seconds=120.0,
        )
    )

    assert result == 0
    assert output_path.read_text(encoding="utf-8") == "HELLO"
    assert perf_events and perf_events[0][0] == "transform_completed"
    assert perf_events[0][1]["provider"] == "codex"
    assert perf_events[0][1]["input_chars"] == 5
    assert perf_events[0][1]["output_chars"] == 5


def test_run_tts_logs_perf_entry(monkeypatch: object) -> None:
    import agent_tools.cli as cli_module

    perf_events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        cli_module,
        "append_perf_event",
        lambda event, **fields: perf_events.append((event, fields)),
    )
    monkeypatch.setattr(
        cli_module,
        "_read_text_input",
        lambda _input_file=None: "hello there",
    )
    monkeypatch.setattr(
        cli_module,
        "synthesize_wav",
        lambda *_args, **_kwargs: TtsResult(
            wav=b"WAV",
            sample_rate=24_000,
            chunks=2,
            resolved_device="cpu",
            metrics=TtsMetrics(
                total_ms=55.0,
                device_probe_ms=1.0,
                import_ms=2.0,
                pipeline_init_ms=3.0,
                generation_ms=40.0,
                postprocess_ms=9.0,
            ),
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_handle_audio_output",
        lambda **_kwargs: cli_module.AudioOutputMetrics(
            output_mode="write",
            bytes_written=3,
            total_ms=4.0,
            file_write_ms=4.0,
            output_target="stdout",
        ),
    )

    result = cli_module._run_tts(
        argparse.Namespace(
            voice="af_heart",
            language="a",
            speed=1.0,
            device="cpu",
            input_file=None,
            output_file="-",
            output_mode="write",
            source="manual",
        )
    )

    assert result == 0
    assert perf_events and perf_events[0][0] == "tts_completed"
    assert perf_events[0][1]["tts_total_ms"] == 55.0
    assert perf_events[0][1]["output_total_ms"] == 4.0


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
