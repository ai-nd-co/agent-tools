from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from agent_tools.playback import PlaybackPlan
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


def test_run_tts_play_mode_queues_on_windows(monkeypatch: object, tmp_path: Path) -> None:
    import agent_tools.cli as cli_module

    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "out.wav"
    input_path.write_text("hello", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_enqueue(request: object) -> object:
        captured["source_label"] = request.source_label
        captured["wav_data"] = request.wav_data
        return object()

    monkeypatch.setattr(
        cli_module,
        "plan_playback",
        lambda: PlaybackPlan(
            available=True,
            strategy="controller-queue",
            backend="pyside6-qmediaplayer",
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "synthesize_wav",
        lambda *_args, **_kwargs: TtsResult(wav=b"WAV", sample_rate=24_000, chunks=1),
    )
    monkeypatch.setattr(cli_module, "enqueue_for_playback", fake_enqueue)

    result = cli_module._run_tts(
        argparse.Namespace(
            voice="af_heart",
            language=None,
            speed=1.0,
            device="auto",
            input_file=input_path,
            output_file=str(output_path),
            output_mode="play",
            source="agent-a",
        )
    )

    assert result == 0
    assert output_path.read_bytes() == b"WAV"
    assert captured["source_label"] == "agent-a"
    assert captured["wav_data"] == b"WAV"


def test_run_tts_play_mode_uses_direct_playback_on_linux(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.cli as cli_module

    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "out.wav"
    input_path.write_text("hello", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "plan_playback",
        lambda: PlaybackPlan(
            available=True,
            strategy="direct",
            backend="paplay",
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "synthesize_wav",
        lambda *_args, **_kwargs: TtsResult(wav=b"WAV", sample_rate=24_000, chunks=1),
    )
    monkeypatch.setattr(
        cli_module,
        "play_direct_wav",
        lambda data, *, backend, output_path=None: captured.update(
            {"data": data, "backend": backend, "output_path": output_path}
        ),
    )

    result = cli_module._run_tts(
        argparse.Namespace(
            voice="af_heart",
            language=None,
            speed=1.0,
            device="auto",
            input_file=input_path,
            output_file=str(output_path),
            output_mode="play",
            source="agent-a",
        )
    )

    assert result == 0
    assert captured["data"] == b"WAV"
    assert captured["backend"] == "paplay"
    assert captured["output_path"] == output_path


def test_run_ttsify_play_mode_uses_direct_playback_on_macos(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.cli as cli_module

    input_path = tmp_path / "input.txt"
    input_path.write_text("hello", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli_module, "is_codex_integration_triggered", lambda: False)
    monkeypatch.setattr(cli_module, "is_claude_integration_triggered", lambda: False)
    monkeypatch.setattr(
        cli_module,
        "plan_playback",
        lambda: PlaybackPlan(
            available=True,
            strategy="direct",
            backend="afplay",
        ),
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
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "play_direct_wav",
        lambda data, *, backend, output_path=None: captured.update(
            {"data": data, "backend": backend, "output_path": output_path}
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
            output_file="-",
            output_mode="play",
            source="agent-a",
            codex_home=None,
            base_url=None,
            originator=None,
            claude_model=None,
            claude_effort=None,
            claude_bare=False,
            timeout_seconds=120.0,
        )
    )

    assert result == 0
    assert captured["data"] == b"WAV"
    assert captured["backend"] == "afplay"
    assert captured["output_path"] is None


def test_run_tts_play_mode_raises_actionable_error_when_backend_missing(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.cli as cli_module

    input_path = tmp_path / "input.txt"
    input_path.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(
        cli_module,
        "plan_playback",
        lambda: PlaybackPlan(
            available=False,
            error_message="Linux playback requires one of `paplay`, `aplay`, or `ffplay` on PATH.",
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "synthesize_wav",
        lambda *_args, **_kwargs: TtsResult(wav=b"WAV", sample_rate=24_000, chunks=1),
    )

    with pytest.raises(RuntimeError, match="paplay"):
        cli_module._run_tts(
            argparse.Namespace(
                voice="af_heart",
                language=None,
                speed=1.0,
                device="auto",
                input_file=input_path,
                output_file="-",
                output_mode="play",
                source=None,
            )
        )
