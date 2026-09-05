from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from agent_tools.codex_private_api import TransformResult
from agent_tools.tts import TtsMetrics, TtsResult
from agent_tools.ttsify import TtsifyMetrics, TtsifyResult
from agent_tools.worklog import RepoHours, WeeklyHours, WorklogReport


def test_main_dispatches_zcode_arguments_without_root_parser_reinterpretation(
    monkeypatch: object,
) -> None:
    import agent_tools.cli as cli_module

    captured: list[list[str]] = []
    monkeypatch.setattr(
        cli_module,
        "run_zcode",
        lambda arguments: captured.append(list(arguments)) or 0,
    )

    result = cli_module.main(["zcode", "bridge", "status", "--json"])

    assert result == 0
    assert captured == [["bridge", "status", "--json"]]


def test_main_renders_runtime_failure_without_traceback(
    monkeypatch: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_tools.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "_run_tts",
        lambda _args: (_ for _ in ()).throw(RuntimeError("CPU runtime unavailable")),
    )

    result = cli_module.main(["tts", "--device", "cpu"])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "error: CPU runtime unavailable\n"
    assert "Traceback" not in captured.err


def test_main_bounds_multiline_runtime_failure(
    monkeypatch: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_tools.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "_run_tts",
        lambda _args: (_ for _ in ()).throw(RuntimeError("detail\r\n" * 4_000)),
    )

    result = cli_module.main(["tts", "--device", "cpu"])

    error = capsys.readouterr().err
    assert result == 1
    assert error.startswith("error: detail detail")
    assert error.endswith("...[truncated]\n")
    assert error.count("\n") == 1
    payload = error.removeprefix("error: ").removesuffix("\n")
    assert len(payload) == cli_module.CLI_ERROR_TEXT_LIMIT


def test_ttsify_no_transform_rejects_explicit_timeout(
    monkeypatch: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_tools.cli as cli_module

    monkeypatch.setattr(cli_module, "_read_text_input", lambda _path: "ready text")

    result = cli_module.main(
        ["ttsify", "--no-transform", "--device", "cpu", "--timeout-seconds", "5"]
    )

    error = capsys.readouterr().err
    assert result == 1
    assert "--no-transform cannot be combined" in error


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
            playback_target=None,
            adb_serial=None,
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
        lambda provider, *, codex_home=None, base_url=None: provider or "codex",
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
            playback_target=None,
            adb_serial=None,
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
        lambda provider, *, codex_home=None, base_url=None: provider or "codex",
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
            engine="kokoro",
            model="hexgrad/Kokoro-82M",
            voice="af_heart",
            language="a",
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
            playback_target=None,
            adb_serial=None,
        )
    )

    assert result == 0
    assert perf_events and perf_events[0][0] == "tts_completed"
    assert perf_events[0][1]["tts_total_ms"] == 55.0
    assert perf_events[0][1]["output_total_ms"] == 4.0
    assert perf_events[0][1]["tts_engine"] == "kokoro"
    assert perf_events[0][1]["tts_model"] == "hexgrad/Kokoro-82M"


def test_tts_cli_accepts_explicit_silero_engine_and_russian_language() -> None:
    import agent_tools.cli as cli_module

    args = cli_module.build_parser().parse_args(
        [
            "tts",
            "--tts-engine",
            "silero",
            "--language",
            "ru",
            "--voice",
            "baya",
        ]
    )

    assert args.tts_engine == "silero"
    assert args.language == "ru"
    assert args.voice == "baya"


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


def test_run_worklog_json_writes_output(monkeypatch: object, tmp_path: Path) -> None:
    import agent_tools.cli as cli_module

    output_path = tmp_path / "worklog.json"
    report = WorklogReport(
        time_zone="Asia/Tashkent",
        generated_at=datetime(2026, 4, 23, 0, 0, tzinfo=UTC),
        since=None,
        until=None,
        total_hours=1.5,
        turn_count=2,
        merged_interval_count=1,
        weeks=(),
        repos=(),
        intervals=(),
        turns=(),
        warnings=(),
    )
    monkeypatch.setattr(cli_module, "generate_worklog_report", lambda **_kwargs: report)

    result = cli_module._run_worklog(
        argparse.Namespace(
            since=None,
            until=None,
            codex_home=None,
            claude_home=None,
            format="json",
            output=str(output_path),
        )
    )

    assert result == 0
    payload = output_path.read_text(encoding="utf-8")
    assert '"total_hours": 1.5' in payload


def test_run_worklog_rejects_inverted_date_range() -> None:
    import agent_tools.cli as cli_module

    with pytest.raises(SystemExit):
        cli_module._run_worklog(
            argparse.Namespace(
                since=date(2026, 4, 24),
                until=date(2026, 4, 23),
                codex_home=None,
                claude_home=None,
                format="table",
                output="-",
            )
        )


def test_run_worklog_table_is_repo_focused(monkeypatch: object, tmp_path: Path) -> None:
    import agent_tools.cli as cli_module

    output_path = tmp_path / "worklog.txt"
    report = WorklogReport(
        time_zone="Asia/Tashkent",
        generated_at=datetime(2026, 4, 23, 0, 0, tzinfo=UTC),
        since=date(2026, 4, 13),
        until=date(2026, 4, 19),
        total_hours=3.0,
        turn_count=6,
        merged_interval_count=2,
        weeks=(WeeklyHours(date(2026, 4, 13), date(2026, 4, 19), 10800.0),),
        repos=(
            RepoHours("ai-nd-co/pdd-test", "pdd-test", 7200.0, 1, 4),
            RepoHours("ai-nd-co/agent-tools", "agent-tools", 3600.0, 1, 2),
        ),
        intervals=(),
        turns=(),
        warnings=(),
    )
    monkeypatch.setattr(cli_module, "generate_worklog_report", lambda **_kwargs: report)

    result = cli_module._run_worklog(
        argparse.Namespace(
            since=date(2026, 4, 13),
            until=date(2026, 4, 19),
            codex_home=None,
            claude_home=None,
            format="table",
            output=str(output_path),
        )
    )

    assert result == 0
    payload = output_path.read_text(encoding="utf-8")
    assert "Worklog" in payload
    assert "Week: 2026-04-13 to 2026-04-19 | Hours: 3.00" in payload
    assert "Repo" in payload and "Hours" in payload and "Share" in payload
    assert "pdd-test" in payload
    assert "66.7%" in payload
