from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from agent_tools.audio import wav_duration_ms
from agent_tools.controller_client import ensure_controller_running
from agent_tools.queue_db import EnqueueRequest, QueueItem, connect, enqueue_item
from agent_tools.runtime import audio_cache_dir, ensure_runtime_dirs


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


def enqueue_for_playback(request: QueuePlaybackRequest) -> QueueItem:
    ensure_runtime_dirs()
    if not ensure_controller_running(show_window=False, detached=True):
        raise RuntimeError(
            "AgentTools controller playback is unavailable. "
            "Install with: pip install \"ai-nd-co-agent-tools[ui]\" "
            "and ensure Qt multimedia audio support is available."
        )
    audio_path = audio_cache_dir() / f"{uuid4()}.wav"
    audio_path.write_bytes(request.wav_data)
    duration_ms = wav_duration_ms(request.wav_data)
    with connect() as conn:
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
            ),
        )
    return item
