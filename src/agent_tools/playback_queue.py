from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from agent_tools.audio import wav_duration_ms
from agent_tools.controller_client import ensure_controller_running
from agent_tools.queue_db import EnqueueRequest, connect, enqueue_item
from agent_tools.runtime import audio_cache_dir, ensure_runtime_dirs
from agent_tools.voxflow_adb import (
    PLAYBACK_TARGET_VOXFLOW_ADB,
    PLAYBACK_TARGET_WINDOWS,
    VoxFlowAdbDispatchRequest,
    VoxFlowAdbDispatchResult,
    dispatch_to_voxflow_adb,
)


@dataclass(frozen=True)
class QueuePlaybackRequest:
    raw_text: str
    tts_text: str
    wav_data: bytes
    source_label: str | None
    voice: str
    language: str | None
    speed: float
    model: str | None
    reasoning_effort: str | None
    tts_engine: str = "kokoro"
    tts_model: str | None = None
    playback_target: str | None = PLAYBACK_TARGET_WINDOWS
    adb_serial: str | None = None


@dataclass(frozen=True)
class PlaybackDispatchResult:
    playback_target: str
    queue_id: int | None = None
    item_id: str | None = None
    remote_item_id: str | None = None
    remote_audio_path: str | None = None
    adb_serial: str | None = None
    adb_result: VoxFlowAdbDispatchResult | None = None


def enqueue_for_playback(request: QueuePlaybackRequest) -> PlaybackDispatchResult:
    playback_target = request.playback_target or PLAYBACK_TARGET_WINDOWS
    if playback_target == PLAYBACK_TARGET_VOXFLOW_ADB:
        adb_result = dispatch_to_voxflow_adb(
            VoxFlowAdbDispatchRequest(
                raw_text=request.raw_text,
                tts_text=request.tts_text,
                wav_data=request.wav_data,
                source_label=request.source_label,
                voice=request.voice,
                language=request.language,
                speed=request.speed,
                model=request.model,
                reasoning_effort=request.reasoning_effort,
                adb_serial=request.adb_serial,
            )
        )
        return PlaybackDispatchResult(
            playback_target=playback_target,
            remote_item_id=adb_result.remote_item_id,
            remote_audio_path=adb_result.remote_audio_path,
            adb_serial=adb_result.adb_serial,
            adb_result=adb_result,
        )

    ensure_runtime_dirs()
    audio_path = audio_cache_dir() / f"{uuid4()}.wav"
    with connect() as conn:
        try:
            audio_path.write_bytes(request.wav_data)
            duration_ms = wav_duration_ms(request.wav_data)
            item = enqueue_item(
                conn,
                EnqueueRequest(
                    source_label=request.source_label,
                    raw_text=request.raw_text,
                    tts_text=request.tts_text,
                    audio_path=audio_path,
                    duration_ms=duration_ms,
                    voice=request.voice,
                    language=request.language,
                    speed=request.speed,
                    model=request.model,
                    reasoning_effort=request.reasoning_effort,
                    tts_engine=request.tts_engine,
                    tts_model=request.tts_model,
                ),
            )
        except Exception:
            audio_path.unlink(missing_ok=True)
            raise
    ensure_controller_running(show_window=False, detached=True)
    return PlaybackDispatchResult(
        playback_target=playback_target,
        queue_id=item.queue_id,
        item_id=item.item_id,
    )
