from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_tools.transformer import TransformOptions, transform_text
from agent_tools.tts import SUPPORTED_LANGUAGES, synthesize_wav

REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high", "xhigh", "none")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "transform":
        return _run_transform(args)
    if args.command == "tts":
        return _run_tts(args)

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
    _write_binary_output(args.output_file, result.wav)
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
