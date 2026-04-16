from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_tools.claude_integration import is_claude_integration_triggered
from agent_tools.codex_config import (
    ALLOWED_CLAUDE_CODE_MODELS,
    ENV_KOKORO_SPEED,
    ENV_SOURCE,
    read_float_env,
    read_preferred_tts_speed,
    read_string_env,
)
from agent_tools.codex_integration import (
    is_codex_integration_triggered,
    load_codex_integration_enabled,
)
from agent_tools.codex_notify import dispatch_codex_notify
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
from agent_tools.playback_queue import QueuePlaybackRequest, enqueue_for_playback
from agent_tools.transformer import TransformOptions, transform_text
from agent_tools.tts import SUPPORTED_LANGUAGES, synthesize_wav
from agent_tools.ttsify import SUPPORTED_TTSIFY_DEVICES, TtsifyOptions, ttsify_text
from agent_tools.ui_app import run_ui

REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high", "xhigh", "none")
OUTPUT_MODE_CHOICES = ("write", "play")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "transform":
        return _run_transform(args)
    if args.command == "tts":
        return _run_tts(args)
    if args.command == "ttsify":
        return _run_ttsify(args)
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

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-tools")
    subparsers = parser.add_subparsers(dest="command")

    transform_parser = subparsers.add_parser(
        "transform",
        help="Transform text through Codex or Claude Code.",
    )
    transform_parser.add_argument("--system-prompt-file", required=True, type=Path)
    transform_parser.add_argument("--provider", choices=("codex", "claude-code"))
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

    tts_parser = subparsers.add_parser("tts", help="Synthesize text with Kokoro.")
    tts_parser.add_argument("--voice", default="af_heart")
    tts_parser.add_argument("--language", choices=SUPPORTED_LANGUAGES)
    tts_parser.add_argument("--speed", type=float)
    tts_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    tts_parser.add_argument("--input-file", type=Path)
    tts_parser.add_argument("--output-file", default="-")
    tts_parser.add_argument("--output-mode", choices=OUTPUT_MODE_CHOICES, default="write")
    tts_parser.add_argument("--source")

    ttsify_parser = subparsers.add_parser(
        "ttsify",
        help="Transform raw text into TTS-friendly text and synthesize it in one step.",
    )
    ttsify_parser.add_argument("--provider", choices=("codex", "claude-code"))
    ttsify_parser.add_argument("--model")
    ttsify_parser.add_argument("--reasoning-effort", choices=REASONING_EFFORT_CHOICES)
    ttsify_parser.add_argument("--fast", action="store_true")
    ttsify_parser.add_argument("--claude-model", choices=ALLOWED_CLAUDE_CODE_MODELS)
    ttsify_parser.add_argument("--claude-effort", choices=("low", "medium", "high", "max"))
    ttsify_parser.add_argument("--claude-bare", action="store_true")
    ttsify_parser.add_argument("--voice")
    ttsify_parser.add_argument("--language", choices=SUPPORTED_LANGUAGES)
    ttsify_parser.add_argument("--speed", type=float)
    ttsify_parser.add_argument("--device", choices=SUPPORTED_TTSIFY_DEVICES)
    ttsify_parser.add_argument("--input-file", type=Path)
    ttsify_parser.add_argument("--output-file", default="-")
    ttsify_parser.add_argument("--output-mode", choices=OUTPUT_MODE_CHOICES, default="write")
    ttsify_parser.add_argument("--source")
    ttsify_parser.add_argument("--codex-home", type=Path)
    ttsify_parser.add_argument("--base-url")
    ttsify_parser.add_argument("--originator")
    ttsify_parser.add_argument("--timeout-seconds", type=float, default=120.0)

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

    return parser


def _run_transform(args: argparse.Namespace) -> int:
    input_text = _read_text_input(args.input_file)
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
    result = transform_text(input_text, options)
    _write_text_output(args.output_file, result.text)
    return 0


def _run_tts(args: argparse.Namespace) -> int:
    input_text = _read_text_input(args.input_file)
    speed = args.speed if args.speed is not None else read_float_env(ENV_KOKORO_SPEED)
    if speed is None:
        speed = read_preferred_tts_speed() or 1.0
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
            voice=args.voice,
            language=args.language,
            speed=speed,
            device=args.device,
        )
        _handle_audio_output(
            output_mode=args.output_mode,
            output_file=args.output_file,
            data=result.wav,
            raw_text=input_text,
            tts_text=input_text,
            source_label=source_label,
            voice=args.voice,
            language=args.language,
            speed=speed,
            model=None,
            reasoning_effort=None,
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
            ),
        )
        _handle_audio_output(
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
        )
    finally:
        processing_notice.finish()
    return 0


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
) -> None:
    if output_mode == "write":
        _write_binary_output(output_file, data)
        return

    if sys.platform != "win32":
        raise RuntimeError("Playback mode is currently supported only on Windows.")

    if output_file != "-":
        Path(output_file).write_bytes(data)

    enqueue_for_playback(
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
        )
    )


def _maybe_start_processing_notice(
    *,
    output_mode: str,
    source_label: str | None,
    preview_text: str,
    stage: str,
)-> ProcessingNotice:
    if output_mode != "play" or sys.platform != "win32":
        return ProcessingNotice(progress_id="", available=False)
    return start_processing_notice(
        source_label=source_label,
        preview_text=preview_text,
        detail_text=preview_text,
        stage=stage,
    )
