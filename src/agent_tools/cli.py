from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter

from agent_tools.claude_integration import is_claude_integration_triggered
from agent_tools.codex_config import (
    ALLOWED_CLAUDE_CODE_MODELS,
    ENV_ADB_SERIAL,
    ENV_KOKORO_SPEED,
    ENV_KOKORO_VOICE,
    ENV_PLAYBACK_TARGET,
    ENV_SILERO_VOICE,
    ENV_SOURCE,
    ENV_TTS_ENGINE,
    read_float_env,
    read_preferred_tts_speed,
    read_string_env,
)
from agent_tools.codex_integration import (
    is_codex_integration_triggered,
    load_codex_integration_enabled,
)
from agent_tools.codex_notify import dispatch_codex_notify
from agent_tools.computer.cli import (
    add_computer_parser,
    extract_argparse_error,
    render_computer_argument_error,
    render_computer_parse_error,
    run_computer,
)
from agent_tools.computer.models import ComputerError
from agent_tools.console import configure_windows_utf8
from agent_tools.controller_client import (
    ProcessingNotice,
    send_controller_command,
    start_processing_notice,
)
from agent_tools.cuda_install import install_cuda
from agent_tools.cuda_runtime import SUPPORTED_CUDA_TRACKS, probe_tts_runtime
from agent_tools.hook_install import (
    install_agent_integrations,
    install_claude_integration,
    install_codex_integration,
)
from agent_tools.perf_log import append_perf_event
from agent_tools.playback_queue import QueuePlaybackRequest, enqueue_for_playback
from agent_tools.transformer import (
    TransformOptions,
    resolve_effective_transform_provider,
    transform_text,
)
from agent_tools.tts import (
    DEFAULT_KOKORO_VOICE,
    DEFAULT_SILERO_VOICE,
    SUPPORTED_LANGUAGES,
    SUPPORTED_TTS_ENGINES,
    resolve_tts_engine,
    synthesize_wav,
)
from agent_tools.tts_server import (
    DEFAULT_TTS_SERVER_HOST,
    DEFAULT_TTS_SERVER_PORT,
    run_tts_server,
)
from agent_tools.tts_startup import main as run_tts_startup
from agent_tools.ttsify import SUPPORTED_TTSIFY_DEVICES, TtsifyOptions, ttsify_text
from agent_tools.ui_app import run_ui
from agent_tools.voice_onboarding import main as run_voice
from agent_tools.voxflow_adb import (
    PLAYBACK_TARGET_CHOICES,
    PLAYBACK_TARGET_VOXFLOW_ADB,
    PLAYBACK_TARGET_WINDOWS,
    VoxFlowAdbDispatchRequest,
    dispatch_to_voxflow_adb,
)
from agent_tools.worklog import generate_worklog_report, render_worklog_table
from agent_tools.zcode_bridge import main as run_zcode

REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high", "xhigh", "none")
OUTPUT_MODE_CHOICES = ("write", "play")
CLI_ERROR_TEXT_LIMIT = 1024


@dataclass(frozen=True)
class AudioOutputMetrics:
    output_mode: str
    bytes_written: int
    total_ms: float
    file_write_ms: float = 0.0
    enqueue_ms: float = 0.0
    queue_id: int | None = None
    playback_target: str = PLAYBACK_TARGET_WINDOWS
    output_target: str | None = None
    dispatch_ms: float = 0.0
    remote_item_id: str | None = None
    adb_serial: str | None = None


def main(argv: list[str] | None = None) -> int:
    configure_windows_utf8()
    try:
        return _main(argv)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {_bounded_cli_error(exc)}", file=sys.stderr)
        return 1


def _main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    command_line = list(sys.argv[1:] if argv is None else argv)
    if command_line[:1] == ["voice"]:
        return run_voice(command_line[1:])
    if command_line[:1] == ["zcode"]:
        return run_zcode(command_line[1:])
    if command_line[:1] == ["computer"]:
        # Every computer parse failure — leaf, parent, or root — routes through
        # the structured sanitizer in both output modes. argparse writes raw
        # usage/error text (which can embed caller values) to stderr before
        # exiting, so stderr is suppressed and only the recovered message,
        # validated against parser actions, is reclassified.
        suppressed_stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(suppressed_stderr):
                args = parser.parse_args(command_line)
        except ComputerError as exc:
            return render_computer_parse_error(command_line, exc)
        except SystemExit as exc:
            if exc.code == 2:
                return render_computer_argument_error(
                    command_line,
                    parser_message=extract_argparse_error(suppressed_stderr.getvalue()),
                    parser=parser,
                )
            raise
    else:
        try:
            args = parser.parse_args(command_line)
        except ComputerError as exc:
            return render_computer_parse_error(command_line, exc)

    if args.command == "transform":
        return _run_transform(args)
    if args.command == "tts":
        return _run_tts(args)
    if args.command == "ttsify":
        return _run_ttsify(args)
    if args.command == "tts-server":
        return _run_tts_server(args)
    if args.command == "tts-startup":
        return run_tts_startup(args.startup_args)
    if args.command == "ui":
        return _run_ui(args)
    if args.command == "install-integrations":
        return _run_install_integrations(args)
    if args.command == "install-codex-integration":
        return _run_install_codex_integration(args)
    if args.command == "install-claude-integration":
        return _run_install_claude_integration(args)
    if args.command == "install-codex-stop-hook":
        return _run_install_codex_integration(args)
    if args.command == "install-cuda":
        return _run_install_cuda(args)
    if args.command == "codex-notify-dispatch":
        return _run_codex_notify_dispatch(args)
    if args.command == "cuda-self-check":
        return _run_cuda_self_check(args)
    if args.command == "worklog":
        return _run_worklog(args)
    if args.command == "computer":
        return run_computer(args)

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-tools")
    subparsers = parser.add_subparsers(dest="command")
    add_computer_parser(subparsers)

    transform_parser = subparsers.add_parser(
        "transform",
        help="Transform text through Codex or Claude Code.",
    )
    transform_parser.add_argument("--system-prompt-file", required=True, type=Path)
    transform_parser.add_argument("--provider", choices=("codex", "claude-code", "ollama"))
    transform_parser.add_argument("--model")
    transform_parser.add_argument("--reasoning-effort", choices=REASONING_EFFORT_CHOICES)
    transform_parser.add_argument("--fast", action="store_true")
    transform_parser.add_argument("--claude-model", choices=ALLOWED_CLAUDE_CODE_MODELS)
    transform_parser.add_argument("--claude-effort", choices=("low", "medium", "high", "max"))
    transform_parser.add_argument("--claude-bare", action="store_true")
    transform_parser.add_argument("--input-file", type=Path)
    transform_parser.add_argument("--output-file", default="-")
    transform_parser.add_argument("--codex-home", type=Path)
    transform_parser.add_argument("--base-url")
    transform_parser.add_argument("--originator")
    transform_parser.add_argument("--timeout-seconds", type=float, default=120.0)

    tts_parser = subparsers.add_parser("tts", help="Synthesize text with Kokoro or Silero.")
    tts_parser.add_argument("--tts-engine", choices=SUPPORTED_TTS_ENGINES)
    tts_parser.add_argument("--voice")
    tts_parser.add_argument("--language", choices=SUPPORTED_LANGUAGES)
    tts_parser.add_argument("--speed", type=float)
    tts_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    tts_parser.add_argument("--input-file", type=Path)
    tts_parser.add_argument("--output-file", default="-")
    tts_parser.add_argument("--output-mode", choices=OUTPUT_MODE_CHOICES, default="write")
    tts_parser.add_argument("--source")
    tts_parser.add_argument("--playback-target", choices=PLAYBACK_TARGET_CHOICES)
    tts_parser.add_argument("--adb-serial")

    ttsify_parser = subparsers.add_parser(
        "ttsify",
        help="Transform raw text into TTS-friendly text and synthesize it in one step.",
    )
    ttsify_parser.add_argument("--provider", choices=("codex", "claude-code", "ollama"))
    ttsify_parser.add_argument("--model")
    ttsify_parser.add_argument("--reasoning-effort", choices=REASONING_EFFORT_CHOICES)
    ttsify_parser.add_argument("--fast", action="store_true")
    ttsify_parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Synthesize the input verbatim without calling an LLM transform provider.",
    )
    ttsify_parser.add_argument("--claude-model", choices=ALLOWED_CLAUDE_CODE_MODELS)
    ttsify_parser.add_argument("--claude-effort", choices=("low", "medium", "high", "max"))
    ttsify_parser.add_argument("--claude-bare", action="store_true")
    ttsify_parser.add_argument("--tts-engine", choices=SUPPORTED_TTS_ENGINES)
    ttsify_parser.add_argument("--voice")
    ttsify_parser.add_argument("--language", choices=SUPPORTED_LANGUAGES)
    ttsify_parser.add_argument("--speed", type=float)
    ttsify_parser.add_argument("--device", choices=SUPPORTED_TTSIFY_DEVICES)
    ttsify_parser.add_argument("--input-file", type=Path)
    ttsify_parser.add_argument("--output-file", default="-")
    ttsify_parser.add_argument("--output-mode", choices=OUTPUT_MODE_CHOICES, default="write")
    ttsify_parser.add_argument("--source")
    ttsify_parser.add_argument("--playback-target", choices=PLAYBACK_TARGET_CHOICES)
    ttsify_parser.add_argument("--adb-serial")
    ttsify_parser.add_argument("--codex-home", type=Path)
    ttsify_parser.add_argument("--base-url")
    ttsify_parser.add_argument("--originator")
    ttsify_parser.add_argument("--timeout-seconds", type=float)

    tts_server_parser = subparsers.add_parser(
        "tts-server",
        help="Serve authenticated local Kokoro and Silero WAV synthesis.",
    )
    tts_server_parser.add_argument("--host", default=DEFAULT_TTS_SERVER_HOST)
    tts_server_parser.add_argument("--port", type=int, default=DEFAULT_TTS_SERVER_PORT)
    tts_server_parser.add_argument("--token-file", type=Path, required=True)
    tts_server_parser.add_argument(
        "--device",
        choices=SUPPORTED_TTSIFY_DEVICES,
        default="cpu",
    )

    tts_startup_parser = subparsers.add_parser(
        "tts-startup",
        help="Manage the exact owner-logon lifecycle for the optional TTS service.",
    )
    tts_startup_parser.add_argument("startup_args", nargs=argparse.REMAINDER)

    voice_parser = subparsers.add_parser(
        "voice",
        help="Set up, check, repair, or pair the Vox voice client.",
    )
    voice_parser.add_argument("voice_args", nargs=argparse.REMAINDER)

    zcode_parser = subparsers.add_parser(
        "zcode",
        help="Control the running ZCode Desktop harness through a private bridge.",
    )
    zcode_parser.add_argument("zcode_args", nargs=argparse.REMAINDER)

    ui_parser = subparsers.add_parser("ui", help="Run or focus the desktop queue controller.")
    ui_parser.add_argument("--hidden", action="store_true", help=argparse.SUPPRESS)
    ui_parser.add_argument("--quit", action="store_true", help="Stop the desktop queue controller.")

    install_integrations_parser = subparsers.add_parser(
        "install-integrations",
        help="Install the supported Codex and Claude Code audio integrations.",
    )
    install_integrations_parser.add_argument("--codex-home", type=Path)
    install_integrations_parser.add_argument("--claude-home", type=Path)

    install_integration_parser = subparsers.add_parser(
        "install-codex-integration",
        help="Install the supported Codex audio integration for the current platform.",
    )
    install_integration_parser.add_argument("--codex-home", type=Path)

    install_claude_parser = subparsers.add_parser(
        "install-claude-integration",
        help="Install the Claude Code Stop-hook audio integration.",
    )
    install_claude_parser.add_argument("--claude-home", type=Path)

    install_hook_parser = subparsers.add_parser(
        "install-codex-stop-hook",
        help="Install the Codex audio integration (compatibility alias).",
    )
    install_hook_parser.add_argument("--codex-home", type=Path)

    install_cuda_parser = subparsers.add_parser(
        "install-cuda",
        help="Install a CUDA-enabled PyTorch stack into this Python environment.",
    )
    install_cuda_parser.add_argument(
        "--cuda-track",
        choices=("auto",) + SUPPORTED_CUDA_TRACKS,
        default="auto",
    )
    install_cuda_parser.add_argument("--no-validate", action="store_true")

    notify_dispatch_parser = subparsers.add_parser(
        "codex-notify-dispatch",
        help=argparse.SUPPRESS,
    )
    notify_dispatch_parser.add_argument("payload_json")
    notify_dispatch_parser.add_argument("--codex-home", type=Path)

    cuda_self_check_parser = subparsers.add_parser(
        "cuda-self-check",
        help=argparse.SUPPRESS,
    )
    cuda_self_check_parser.add_argument("--json", action="store_true")

    worklog_parser = subparsers.add_parser(
        "worklog",
        help="Analyze local Codex and Claude Code logs into weekly work totals.",
    )
    worklog_parser.add_argument("--since", type=_parse_iso_date)
    worklog_parser.add_argument("--until", type=_parse_iso_date)
    worklog_parser.add_argument("--codex-home", type=Path)
    worklog_parser.add_argument("--claude-home", type=Path)
    worklog_parser.add_argument("--format", choices=("table", "json"), default="table")
    worklog_parser.add_argument("--output", default="-")

    return parser


def _run_transform(args: argparse.Namespace) -> int:
    input_text = _read_text_input(args.input_file)
    trace_id = str(uuid.uuid4())
    options = TransformOptions(
        system_prompt_file=args.system_prompt_file,
        provider=getattr(args, "provider", None),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        fast=args.fast,
        codex_home=args.codex_home,
        base_url=args.base_url,
        originator=args.originator,
        claude_model=getattr(args, "claude_model", None),
        claude_effort=getattr(args, "claude_effort", None),
        claude_bare=getattr(args, "claude_bare", False),
        timeout_seconds=args.timeout_seconds,
    )
    started = perf_counter()
    result = transform_text(input_text, options)
    transform_ms = (perf_counter() - started) * 1000.0
    _write_text_output(args.output_file, result.text)
    append_perf_event(
        "transform_completed",
        trace_id=trace_id,
        command="transform",
        provider=resolve_effective_transform_provider(
            getattr(args, "provider", None),
            codex_home=args.codex_home,
            base_url=args.base_url,
        ),
        requested_provider=getattr(args, "provider", None),
        requested_model=args.model,
        requested_reasoning_effort=args.reasoning_effort,
        claude_model=getattr(args, "claude_model", None),
        claude_effort=getattr(args, "claude_effort", None),
        input_chars=len(input_text),
        output_chars=len(result.text),
        transform_ms=transform_ms,
        response_id=result.response_id,
        session_id=result.session_id,
    )
    return 0


def _run_tts(args: argparse.Namespace) -> int:
    input_text = _read_text_input(args.input_file)
    trace_id = str(uuid.uuid4())
    requested_tts_engine = (
        getattr(args, "tts_engine", None) or read_string_env(ENV_TTS_ENGINE) or "auto"
    )
    resolved_tts_engine = resolve_tts_engine(
        input_text,
        engine=requested_tts_engine,
        language=args.language,
    )
    if args.voice is not None:
        voice = args.voice
    elif resolved_tts_engine == "silero":
        voice = read_string_env(ENV_SILERO_VOICE) or DEFAULT_SILERO_VOICE
    else:
        voice = read_string_env(ENV_KOKORO_VOICE) or DEFAULT_KOKORO_VOICE
    if args.speed is not None:
        speed = args.speed
    elif resolved_tts_engine == "kokoro":
        speed = read_float_env(ENV_KOKORO_SPEED) or read_preferred_tts_speed() or 1.0
    else:
        speed = 1.0
    source_label = args.source or read_string_env(ENV_SOURCE)
    processing_notice = _maybe_start_processing_notice(
        output_mode=args.output_mode,
        source_label=source_label,
        preview_text=input_text,
        stage="Synthesizing audio",
    )
    try:
        result = synthesize_wav(
            input_text,
            voice=voice,
            language=args.language,
            speed=speed,
            device=args.device,
            engine=requested_tts_engine,
        )
        output_metrics = _handle_audio_output(
            output_mode=args.output_mode,
            output_file=args.output_file,
            data=result.wav,
            raw_text=input_text,
            tts_text=input_text,
            source_label=source_label,
            voice=result.voice,
            language=result.language,
            speed=speed,
            model=None,
            reasoning_effort=None,
            tts_engine=result.engine,
            tts_model=result.model,
            playback_target=args.playback_target,
            adb_serial=args.adb_serial,
        )
        append_perf_event(
            "tts_completed",
            trace_id=trace_id,
            command="tts",
            source_label=source_label,
            requested_engine=requested_tts_engine,
            tts_engine=result.engine,
            tts_model=result.model,
            voice=result.voice,
            language=result.language,
            speed=speed,
            requested_device=args.device,
            resolved_device=result.resolved_device,
            device_fallback_reason=result.device_fallback_reason,
            text_chars=len(input_text),
            chunks=result.chunks,
            tts_total_ms=result.metrics.total_ms,
            tts_device_probe_ms=result.metrics.device_probe_ms,
            tts_import_ms=result.metrics.import_ms,
            tts_pipeline_init_ms=result.metrics.pipeline_init_ms,
            tts_generation_ms=result.metrics.generation_ms,
            tts_postprocess_ms=result.metrics.postprocess_ms,
            output_mode=output_metrics.output_mode,
            playback_target=output_metrics.playback_target,
            output_target=output_metrics.output_target,
            output_total_ms=output_metrics.total_ms,
            output_file_write_ms=output_metrics.file_write_ms,
            output_enqueue_ms=output_metrics.enqueue_ms,
            output_dispatch_ms=output_metrics.dispatch_ms,
            queue_id=output_metrics.queue_id,
            remote_item_id=output_metrics.remote_item_id,
            adb_serial=output_metrics.adb_serial,
            bytes_written=output_metrics.bytes_written,
        )
    finally:
        processing_notice.finish()
    return 0


def _run_ttsify(args: argparse.Namespace) -> int:
    if (
        is_codex_integration_triggered() or is_claude_integration_triggered()
    ) and not load_codex_integration_enabled():
        print("AgentTools integration disabled; skipping audio generation.", file=sys.stderr)
        return 0

    input_text = _read_text_input(args.input_file)
    trace_id = str(uuid.uuid4())
    source_label = args.source or read_string_env(ENV_SOURCE)
    processing_notice = _maybe_start_processing_notice(
        output_mode=args.output_mode,
        source_label=source_label,
        preview_text=input_text,
        stage="Transforming for speech",
    )
    try:
        result = ttsify_text(
            input_text,
            TtsifyOptions(
                provider=getattr(args, "provider", None),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                fast=args.fast,
                tts_engine=getattr(args, "tts_engine", None),
                voice=args.voice,
                language=args.language,
                speed=args.speed,
                device=args.device,
                codex_home=args.codex_home,
                base_url=args.base_url,
                originator=args.originator,
                claude_model=getattr(args, "claude_model", None),
                claude_effort=getattr(args, "claude_effort", None),
                claude_bare=getattr(args, "claude_bare", False),
                timeout_seconds=args.timeout_seconds,
                no_transform=getattr(args, "no_transform", False),
            ),
        )
        output_metrics = _handle_audio_output(
            output_mode=args.output_mode,
            output_file=args.output_file,
            data=result.tts_result.wav,
            raw_text=input_text,
            tts_text=result.transformed_text,
            source_label=source_label,
            voice=result.voice,
            language=result.language,
            speed=result.speed,
            model=result.model,
            reasoning_effort=result.reasoning_effort,
            tts_engine=result.tts_engine,
            tts_model=result.tts_model,
            playback_target=args.playback_target,
            adb_serial=args.adb_serial,
        )
        append_perf_event(
            "ttsify_completed",
            trace_id=trace_id,
            command="ttsify",
            provider=(
                "passthrough"
                if getattr(args, "no_transform", False)
                else resolve_effective_transform_provider(
                    getattr(args, "provider", None),
                    codex_home=args.codex_home,
                    base_url=args.base_url,
                )
            ),
            requested_provider=getattr(args, "provider", None),
            model=result.model,
            reasoning_effort=result.reasoning_effort,
            requested_engine=getattr(args, "tts_engine", None)
            or read_string_env(ENV_TTS_ENGINE)
            or "auto",
            tts_engine=result.tts_engine,
            tts_model=result.tts_model,
            source_label=source_label,
            input_chars=len(input_text),
            transformed_chars=len(result.transformed_text),
            voice=result.voice,
            language=result.language,
            speed=result.speed,
            requested_device=result.device,
            resolved_device=result.resolved_device,
            device_fallback_reason=result.device_fallback_reason,
            total_ms=result.metrics.total_ms,
            transform_ms=result.metrics.transform_ms,
            tts_ms=result.metrics.tts_ms,
            tts_device_probe_ms=result.tts_result.metrics.device_probe_ms,
            tts_import_ms=result.tts_result.metrics.import_ms,
            tts_pipeline_init_ms=result.tts_result.metrics.pipeline_init_ms,
            tts_generation_ms=result.tts_result.metrics.generation_ms,
            tts_postprocess_ms=result.tts_result.metrics.postprocess_ms,
            output_mode=output_metrics.output_mode,
            playback_target=output_metrics.playback_target,
            output_target=output_metrics.output_target,
            output_total_ms=output_metrics.total_ms,
            output_file_write_ms=output_metrics.file_write_ms,
            output_enqueue_ms=output_metrics.enqueue_ms,
            output_dispatch_ms=output_metrics.dispatch_ms,
            queue_id=output_metrics.queue_id,
            remote_item_id=output_metrics.remote_item_id,
            adb_serial=output_metrics.adb_serial,
            bytes_written=output_metrics.bytes_written,
            response_id=result.transform_result.response_id,
            session_id=result.transform_result.session_id,
        )
    finally:
        processing_notice.finish()
    return 0


def _run_tts_server(args: argparse.Namespace) -> int:
    return run_tts_server(
        host=args.host,
        port=args.port,
        token_file=args.token_file,
        device=args.device,
    )


def _run_ui(args: argparse.Namespace) -> int:
    action = "shutdown" if args.quit else "refresh" if args.hidden else "show"
    if send_controller_command(action):
        return 0
    if args.quit:
        return 0
    return run_ui(hidden=args.hidden)


def _run_install_codex_integration(args: argparse.Namespace) -> int:
    result = install_codex_integration(args.codex_home)
    print(f"Installed Codex integration in {result.mode} mode.")
    print(f"  config: {result.config_path}")
    if result.notify_command is not None:
        print(f"  notify command: {list(result.notify_command)}")
        print(f"  notify log: {result.codex_home / 'notify_tts.log'}")
        print(f"  child log: {result.codex_home / 'notify_tts_agent_tools.log'}")
    if result.hooks_json_path is not None:
        print(f"  hooks.json: {result.hooks_json_path}")
    if result.hook_script_path is not None:
        print(f"  script: {result.hook_script_path}")
        print(f"  hook log: {result.codex_home / 'hooks' / 'stop_tts.log'}")
        print(f"  child log: {result.codex_home / 'hooks' / 'stop_tts_agent_tools.log'}")
    if result.backups:
        print("  backups:")
        for backup in result.backups:
            print(f"    - {backup}")
    return 0


def _run_install_claude_integration(args: argparse.Namespace) -> int:
    result = install_claude_integration(args.claude_home)
    print("Installed Claude Code integration.")
    print(f"  settings: {result.settings_path}")
    print(f"  script: {result.hook_script_path}")
    print(f"  hook log: {result.claude_home / 'agent-tools' / 'stop_tts.log'}")
    print(f"  child log: {result.claude_home / 'agent-tools' / 'stop_tts_agent_tools.log'}")
    if result.backups:
        print("  backups:")
        for backup in result.backups:
            print(f"    - {backup}")
    return 0


def _run_install_integrations(args: argparse.Namespace) -> int:
    result = install_agent_integrations(
        codex_home=args.codex_home,
        claude_home=args.claude_home,
    )
    print("Installed AgentTools integrations.")
    print(f"  codex config: {result.codex.config_path}")
    print(f"  claude settings: {result.claude.settings_path}")
    print(f"  claude script: {result.claude.hook_script_path}")
    return 0


def _run_install_cuda(args: argparse.Namespace) -> int:
    result = install_cuda(
        cuda_track=args.cuda_track,
        validate=not args.no_validate,
    )
    print("Installed CUDA-enabled PyTorch stack for this Python environment.")
    print(f"  python: {result.python_executable}")
    print(f"  python_version: {result.python_version}")
    print(f"  platform: {result.platform}")
    print(f"  machine: {result.machine}")
    if result.detected_cuda_version is not None:
        print(f"  detected runtime: {result.detected_cuda_version}")
    print(f"  selected track: {result.selected_track}")
    print(f"  install command: {list(result.install_command)}")
    if result.validation_payload is not None:
        print("  validation:")
        for key in (
            "ok",
            "cuda_ok",
            "tts_stack_ok",
            "python_executable",
            "python_version",
            "platform",
            "machine",
            "torch_version",
            "torch_cuda_version",
            "torchvision_version",
            "torchaudio_version",
            "transformers_version",
            "kokoro_version",
            "device_count",
            "device_name",
            "reason",
            "warning_text",
        ):
            value = result.validation_payload.get(key)
            if value not in (None, ""):
                print(f"    {key}: {value}")
    return 0


def _run_codex_notify_dispatch(args: argparse.Namespace) -> int:
    result = dispatch_codex_notify(args.payload_json, codex_home=args.codex_home)
    success_statuses = {
        "queued",
        "dispatched",
        "duplicate",
        "ignored-event",
        "ignored-blank-message",
        "disabled",
    }
    return 0 if result.status in success_statuses else 1


def _run_cuda_self_check(args: argparse.Namespace) -> int:
    result = probe_tts_runtime()
    if args.json:
        sys.stdout.write(result.to_json())
        sys.stdout.write("\n")
    else:
        sys.stdout.write(
            json.dumps(
                {
                    "ok": result.ok,
                    "cuda_ok": result.cuda_ok,
                    "tts_stack_ok": result.tts_stack_ok,
                    "python_executable": result.python_executable,
                    "python_version": result.python_version,
                    "platform": result.platform,
                    "machine": result.machine,
                    "torch_version": result.torch_version,
                    "torch_cuda_version": result.torch_cuda_version,
                    "torchvision_version": result.torchvision_version,
                    "torchaudio_version": result.torchaudio_version,
                    "transformers_version": result.transformers_version,
                    "kokoro_version": result.kokoro_version,
                    "device_count": result.device_count,
                    "device_name": result.device_name,
                    "reason": result.reason,
                    "warning_text": result.warning_text,
                },
                indent=2,
                sort_keys=True,
            )
        )
        sys.stdout.write("\n")
    return 0 if result.ok else 1


def _run_worklog(args: argparse.Namespace) -> int:
    if args.since is not None and args.until is not None and args.since > args.until:
        raise SystemExit("--since must be on or before --until")
    report = generate_worklog_report(
        since=args.since,
        until=args.until,
        codex_home=args.codex_home,
        claude_home=args.claude_home,
    )
    if args.format == "json":
        output_text = json.dumps(report.to_jsonable(), indent=2, sort_keys=True)
    else:
        output_text = render_worklog_table(report)
    if not output_text.endswith("\n"):
        output_text += "\n"
    _write_text_output(args.output, output_text)
    return 0


def _read_text_input(input_file: Path | None) -> str:
    if input_file is not None:
        return input_file.read_text(encoding="utf-8")
    return sys.stdin.read()


def _write_text_output(output_file: str, text: str) -> None:
    if output_file == "-":
        sys.stdout.write(text)
        return
    Path(output_file).write_text(text, encoding="utf-8")


def _write_binary_output(output_file: str, data: bytes) -> None:
    if output_file == "-":
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return
    Path(output_file).write_bytes(data)


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc


def _handle_audio_output(
    *,
    output_mode: str,
    output_file: str,
    data: bytes,
    raw_text: str,
    tts_text: str,
    source_label: str | None,
    voice: str | None,
    language: str | None,
    speed: float,
    model: str | None,
    reasoning_effort: str | None,
    tts_engine: str,
    tts_model: str | None,
    playback_target: str | None,
    adb_serial: str | None,
) -> AudioOutputMetrics:
    file_write_ms = 0.0
    if output_mode == "write":
        started = perf_counter()
        _write_binary_output(output_file, data)
        file_write_ms = (perf_counter() - started) * 1000.0
        return AudioOutputMetrics(
            output_mode=output_mode,
            bytes_written=len(data),
            total_ms=file_write_ms,
            file_write_ms=file_write_ms,
            output_target="stdout" if output_file == "-" else output_file,
        )

    resolved_playback_target = _resolve_playback_target(playback_target)

    if resolved_playback_target == PLAYBACK_TARGET_WINDOWS and sys.platform != "win32":
        raise RuntimeError("Windows playback mode is currently supported only on Windows.")

    if output_file != "-":
        file_write_started = perf_counter()
        Path(output_file).write_bytes(data)
        file_write_ms = (perf_counter() - file_write_started) * 1000.0

    if resolved_playback_target == PLAYBACK_TARGET_VOXFLOW_ADB:
        dispatch_started = perf_counter()
        dispatch_result = dispatch_to_voxflow_adb(
            VoxFlowAdbDispatchRequest(
                raw_text=raw_text,
                tts_text=tts_text,
                wav_data=data,
                source_label=source_label,
                voice=voice or "unknown",
                language=language,
                speed=speed,
                model=model,
                reasoning_effort=reasoning_effort,
                adb_serial=adb_serial or read_string_env(ENV_ADB_SERIAL),
            )
        )
        dispatch_ms = (perf_counter() - dispatch_started) * 1000.0
        return AudioOutputMetrics(
            output_mode=output_mode,
            bytes_written=len(data),
            total_ms=file_write_ms + dispatch_ms,
            file_write_ms=file_write_ms,
            playback_target=resolved_playback_target,
            output_target=dispatch_result.remote_audio_path,
            dispatch_ms=dispatch_ms,
            remote_item_id=dispatch_result.remote_item_id,
            adb_serial=dispatch_result.adb_serial,
        )

    enqueue_started = perf_counter()
    queue_item = enqueue_for_playback(
        QueuePlaybackRequest(
            raw_text=raw_text,
            tts_text=tts_text,
            wav_data=data,
            source_label=source_label,
            voice=voice or "unknown",
            language=language,
            speed=speed,
            model=model,
            reasoning_effort=reasoning_effort,
            tts_engine=tts_engine,
            tts_model=tts_model,
        )
    )
    enqueue_ms = (perf_counter() - enqueue_started) * 1000.0
    return AudioOutputMetrics(
        output_mode=output_mode,
        bytes_written=len(data),
        total_ms=file_write_ms + enqueue_ms,
        file_write_ms=file_write_ms,
        enqueue_ms=enqueue_ms,
        queue_id=queue_item.queue_id,
        playback_target=resolved_playback_target,
        output_target="queue" if output_file == "-" else output_file,
    )


def _maybe_start_processing_notice(
    *,
    output_mode: str,
    source_label: str | None,
    preview_text: str,
    stage: str,
) -> ProcessingNotice:
    if output_mode != "play" or sys.platform != "win32":
        return ProcessingNotice(progress_id="", available=False)
    return start_processing_notice(
        source_label=source_label,
        preview_text=preview_text,
        detail_text=preview_text,
        stage=stage,
    )


def _resolve_playback_target(playback_target: str | None) -> str:
    configured = playback_target or read_string_env(ENV_PLAYBACK_TARGET) or PLAYBACK_TARGET_WINDOWS
    if configured not in PLAYBACK_TARGET_CHOICES:
        raise ValueError(
            f"Unsupported playback target {configured!r}. "
            f"Expected one of {PLAYBACK_TARGET_CHOICES}."
        )
    return configured


def _bounded_cli_error(exc: Exception) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    if len(text) <= CLI_ERROR_TEXT_LIMIT:
        return text
    suffix = "...[truncated]"
    return f"{text[: CLI_ERROR_TEXT_LIMIT - len(suffix)]}{suffix}"
