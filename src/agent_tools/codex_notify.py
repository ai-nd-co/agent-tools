from __future__ import annotations

import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from agent_tools.codex_config import ENV_KOKORO_DEVICE, read_string_env, resolve_codex_home
from agent_tools.codex_integration import load_codex_integration_enabled
from agent_tools.controller_client import start_processing_notice
from agent_tools.perf_log import append_perf_event
from agent_tools.playback_queue import QueuePlaybackRequest, enqueue_for_playback
from agent_tools.ttsify import TtsifyOptions, ttsify_text


@dataclass(frozen=True)
class CodexNotifyPayload:
    event_type: str
    thread_id: str
    turn_id: str
    cwd: str
    input_messages: tuple[str, ...]
    last_assistant_message: str | None


@dataclass(frozen=True)
class CodexNotifyDispatchResult:
    status: str
    codex_home: Path
    log_path: Path
    child_log_path: Path
    marker_path: Path | None = None
    queue_id: int | None = None


def dispatch_codex_notify(
    payload_json: str,
    *,
    codex_home: Path | None = None,
) -> CodexNotifyDispatchResult:
    dispatch_started = perf_counter()
    home = resolve_codex_home(codex_home)
    log_path = home / "notify_tts.log"
    child_log_path = home / "notify_tts_agent_tools.log"
    marker_dir = home / "notify-state"
    marker_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_markers(marker_dir)

    _log(log_path, f"dispatch_start payload_bytes={len(payload_json)}")
    if not load_codex_integration_enabled():
        _log(log_path, "integration_disabled_exit")
        return CodexNotifyDispatchResult(
            status="disabled",
            codex_home=home,
            log_path=log_path,
            child_log_path=child_log_path,
        )
    try:
        payload = parse_codex_notify_payload(payload_json)
    except Exception as exc:
        _log(log_path, f"payload_parse_failed error={exc}")
        _append_child_log(child_log_path, traceback.format_exc())
        return CodexNotifyDispatchResult(
            status="invalid-payload",
            codex_home=home,
            log_path=log_path,
            child_log_path=child_log_path,
        )

    _log(
        log_path,
        (
            f"payload_parsed type={payload.event_type} "
            f"thread_id={payload.thread_id or '<empty>'} "
            f"turn_id={payload.turn_id or '<empty>'} "
            f"message_bytes={len(payload.last_assistant_message or '')}"
        ),
    )
    if payload.event_type != "agent-turn-complete":
        _log(log_path, "ignored_non_agent_turn_complete")
        return CodexNotifyDispatchResult(
            status="ignored-event",
            codex_home=home,
            log_path=log_path,
            child_log_path=child_log_path,
        )

    message = payload.last_assistant_message or ""
    if not message.strip():
        _log(log_path, "ignored_blank_message")
        return CodexNotifyDispatchResult(
            status="ignored-blank-message",
            codex_home=home,
            log_path=log_path,
            child_log_path=child_log_path,
        )

    marker_path = marker_dir / f"{payload.turn_id}.done"
    if payload.turn_id and marker_path.exists():
        _log(log_path, f"duplicate_turn turn_id={payload.turn_id}")
        return CodexNotifyDispatchResult(
            status="duplicate",
            codex_home=home,
            log_path=log_path,
            child_log_path=child_log_path,
            marker_path=marker_path,
        )

    desired_device = read_string_env(ENV_KOKORO_DEVICE) or "auto"
    trace_id = payload.turn_id or f"codex-notify-{int(dispatch_started * 1000)}"
    source_label = f"codex-notify:{payload.thread_id}" if payload.thread_id else "codex-notify"
    processing_notice = start_processing_notice(
        source_label=source_label,
        preview_text=message,
        detail_text=message,
        stage="Processing Codex reply",
    )
    try:
        with child_log_path.open("a", encoding="utf-8") as child_log:
            child_log.write(
                f"\n=== {datetime.now(tz=UTC).isoformat()} turn={payload.turn_id} ===\n"
            )
            child_log.flush()
            with redirect_stdout(child_log), redirect_stderr(child_log):
                ttsify_started = perf_counter()
                ttsify_result = ttsify_text(
                    message,
                    TtsifyOptions(
                        device=desired_device,
                        codex_home=home,
                    ),
                )
                ttsify_wall_ms = (perf_counter() - ttsify_started) * 1000.0
        enqueue_started = perf_counter()
        queue_item = enqueue_for_playback(
            QueuePlaybackRequest(
                raw_text=message,
                tts_text=ttsify_result.transformed_text,
                wav_data=ttsify_result.tts_result.wav,
                source_label=source_label,
                voice=ttsify_result.voice,
                language=ttsify_result.language,
                speed=ttsify_result.speed,
                model=ttsify_result.model,
                reasoning_effort=ttsify_result.reasoning_effort,
            )
        )
        enqueue_ms = (perf_counter() - enqueue_started) * 1000.0
    except Exception:
        total_failed_ms = (perf_counter() - dispatch_started) * 1000.0
        processing_notice.finish()
        _log(
            log_path,
            f"dispatch_failed turn_id={payload.turn_id} total_ms={_fmt_ms(total_failed_ms)}",
        )
        _append_child_log(child_log_path, traceback.format_exc())
        return CodexNotifyDispatchResult(
            status="failed",
            codex_home=home,
            log_path=log_path,
            child_log_path=child_log_path,
            marker_path=marker_path if payload.turn_id else None,
        )

    if payload.turn_id:
        marker_path.write_text(queue_item.item_id, encoding="utf-8")
    total_dispatch_ms = (perf_counter() - dispatch_started) * 1000.0
    transform_ms = ttsify_result.metrics.transform_ms
    tts_wall_ms = ttsify_wall_ms
    tts_internal_ms = ttsify_result.tts_result.metrics.total_ms
    tts_import_ms = ttsify_result.tts_result.metrics.import_ms
    tts_pipeline_init_ms = ttsify_result.tts_result.metrics.pipeline_init_ms
    tts_generation_ms = ttsify_result.tts_result.metrics.generation_ms
    tts_postprocess_ms = ttsify_result.tts_result.metrics.postprocess_ms
    dominant_stage = max(
        (
            ("transform", transform_ms),
            ("tts_import", tts_import_ms),
            ("tts_pipeline_init", tts_pipeline_init_ms),
            ("tts_generation", tts_generation_ms),
            ("tts_postprocess", tts_postprocess_ms),
            ("enqueue", enqueue_ms),
        ),
        key=lambda item: item[1],
    )[0]
    _log(
        log_path,
        (
            f"dispatch_completed turn_id={payload.turn_id or '<empty>'} "
            f"queue_id={queue_item.queue_id} source={source_label} "
            f"requested_device={desired_device} resolved_device={ttsify_result.resolved_device} "
            f"total_ms={_fmt_ms(total_dispatch_ms)} transform_ms={_fmt_ms(transform_ms)} "
            f"tts_wall_ms={_fmt_ms(tts_wall_ms)} "
            f"tts_internal_ms={_fmt_ms(tts_internal_ms)} "
            f"tts_device_probe_ms={_fmt_ms(ttsify_result.tts_result.metrics.device_probe_ms)} "
            f"tts_import_ms={_fmt_ms(tts_import_ms)} "
            f"tts_pipeline_init_ms={_fmt_ms(tts_pipeline_init_ms)} "
            f"tts_generation_ms={_fmt_ms(tts_generation_ms)} "
            f"tts_postprocess_ms={_fmt_ms(tts_postprocess_ms)} "
            f"enqueue_ms={_fmt_ms(enqueue_ms)} dominant_stage={dominant_stage} "
            f"device_fallback_reason={_render_log_value(ttsify_result.device_fallback_reason)}"
        ),
    )
    append_perf_event(
        "codex_notify_completed",
        trace_id=trace_id,
        command="codex-notify-dispatch",
        turn_id=payload.turn_id,
        thread_id=payload.thread_id,
        source_label=source_label,
        queue_id=queue_item.queue_id,
        requested_device=desired_device,
        resolved_device=ttsify_result.resolved_device,
        device_fallback_reason=ttsify_result.device_fallback_reason,
        total_ms=total_dispatch_ms,
        transform_ms=transform_ms,
        tts_wall_ms=tts_wall_ms,
        tts_internal_ms=tts_internal_ms,
        tts_device_probe_ms=ttsify_result.tts_result.metrics.device_probe_ms,
        tts_import_ms=tts_import_ms,
        tts_pipeline_init_ms=tts_pipeline_init_ms,
        tts_generation_ms=tts_generation_ms,
        tts_postprocess_ms=tts_postprocess_ms,
        enqueue_ms=enqueue_ms,
        dominant_stage=dominant_stage,
        input_chars=len(message),
        transformed_chars=len(ttsify_result.transformed_text),
        model=ttsify_result.model,
        reasoning_effort=ttsify_result.reasoning_effort,
    )
    processing_notice.finish()
    return CodexNotifyDispatchResult(
        status="queued",
        codex_home=home,
        log_path=log_path,
        child_log_path=child_log_path,
        marker_path=marker_path if payload.turn_id else None,
        queue_id=queue_item.queue_id,
    )


def parse_codex_notify_payload(payload_json: str) -> CodexNotifyPayload:
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):
        raise ValueError("Codex notify payload must be a JSON object.")

    return CodexNotifyPayload(
        event_type=_coerce_str(payload.get("type")),
        thread_id=_coerce_str(payload.get("thread-id") or payload.get("thread_id")),
        turn_id=_coerce_str(payload.get("turn-id") or payload.get("turn_id")),
        cwd=_coerce_str(payload.get("cwd")),
        input_messages=_coerce_messages(
            payload.get("input-messages") or payload.get("input_messages"),
        ),
        last_assistant_message=_coerce_optional_str(
            payload.get("last-assistant-message") or payload.get("last_assistant_message"),
        ),
    )


def _cleanup_old_markers(marker_dir: Path) -> None:
    cutoff = datetime.now(tz=UTC).timestamp() - 7 * 24 * 60 * 60
    for marker in marker_dir.glob("*.done"):
        try:
            if marker.stat().st_mtime < cutoff:
                marker.unlink()
        except OSError:
            continue


def _log(path: Path, message: str) -> None:
    timestamp = datetime.now(tz=UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def _append_child_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _coerce_optional_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _coerce_messages(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _fmt_ms(value: float) -> str:
    return f"{value:.1f}"


def _render_log_value(value: str | None) -> str:
    if value is None:
        return "<none>"
    collapsed = " ".join(value.split())
    return collapsed if collapsed else "<empty>"
