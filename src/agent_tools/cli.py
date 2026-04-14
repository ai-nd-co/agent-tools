from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_tools.codex_config import ENV_SOURCE, read_string_env
from agent_tools.controller_client import send_controller_command
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

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-tools")
    subparsers = parser.add_subparsers(dest="command")

    transform_parser = subparsers.add_parser("transform", help="Transform text through Codex.")
    transform_parser.add_argument("--system-prompt-file", required=True, type=Path)
    transform_parser.add_argument("--model")
    transform_parser.add_argument("--reasoning-effort", choices=REASONING_EFFORT_CHOICES)
    transform_parser.add_argument("--fast", action="store_true")
    transform_parser.add_argument("--input-file", type=Path)
    transform_parser.add_argument("--output-file", default="-")
    transform_parser.add_argument("--codex-home", type=Path)
    transform_parser.add_argument("--base-url")
    transform_parser.add_argument("--originator")
    transform_parser.add_argument("--timeout-seconds", type=float, default=120.0)

    tts_parser = subparsers.add_parser("tts", help="Synthesize text with Kokoro.")
    tts_parser.add_argument("--voice", default="af_heart")
    tts_parser.add_argument("--language", choices=SUPPORTED_LANGUAGES)
    tts_parser.add_argument("--speed", type=float, default=1.0)
    tts_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    tts_parser.add_argument("--input-file", type=Path)
    tts_parser.add_argument("--output-file", default="-")
    tts_parser.add_argument("--output-mode", choices=OUTPUT_MODE_CHOICES, default="write")
    tts_parser.add_argument("--source")

    ttsify_parser = subparsers.add_parser(
        "ttsify",
        help="Transform raw text into TTS-friendly text and synthesize it in one step.",
    )
    ttsify_parser.add_argument("--model")
    ttsify_parser.add_argument("--reasoning-effort", choices=REASONING_EFFORT_CHOICES)
    ttsify_parser.add_argument("--fast", action="store_true")
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

    return parser


def _run_transform(args: argparse.Namespace) -> int:
    input_text = _read_text_input(args.input_file)
    options = TransformOptions(
        system_prompt_file=args.system_prompt_file,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        fast=args.fast,
        codex_home=args.codex_home,
        base_url=args.base_url,
        originator=args.originator,
        timeout_seconds=args.timeout_seconds,
    )
    result = transform_text(input_text, options)
    _write_text_output(args.output_file, result.text)
    return 0


def _run_tts(args: argparse.Namespace) -> int:
    input_text = _read_text_input(args.input_file)
    result = synthesize_wav(
        input_text,
        voice=args.voice,
        language=args.language,
        speed=args.speed,
        device=args.device,
    )
    source_label = args.source or read_string_env(ENV_SOURCE)
    _handle_audio_output(
        output_mode=args.output_mode,
        output_file=args.output_file,
        data=result.wav,
        raw_text=input_text,
        tts_text=input_text,
        source_label=source_label,
        voice=args.voice,
        language=args.language,
        speed=args.speed,
        model=None,
        reasoning_effort=None,
    )
    return 0


def _run_ttsify(args: argparse.Namespace) -> int:
    input_text = _read_text_input(args.input_file)
    result = ttsify_text(
        input_text,
        TtsifyOptions(
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
            timeout_seconds=args.timeout_seconds,
        ),
    )
    source_label = args.source or read_string_env(ENV_SOURCE)
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
    return 0


def _run_ui(args: argparse.Namespace) -> int:
    action = "refresh" if args.hidden else "show"
    if send_controller_command(action):
        return 0
    return run_ui(hidden=args.hidden)


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
