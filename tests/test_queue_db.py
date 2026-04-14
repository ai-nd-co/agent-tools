from __future__ import annotations

from pathlib import Path

import numpy as np

from agent_tools.audio import wav_bytes
from agent_tools.playback_queue import QueuePlaybackRequest, enqueue_for_playback
from agent_tools.queue_db import STATUS_QUEUED, connect, get_next_queued_item


def test_enqueue_for_playback_persists_audio_and_metadata(
    tmp_path: Path, monkeypatch: object
) -> None:
    import agent_tools.playback_queue as playback_queue

    audio_dir = tmp_path / "audio"
    db_path = tmp_path / "queue.sqlite3"
    audio_dir.mkdir()

    monkeypatch.setattr(playback_queue, "audio_cache_dir", lambda: audio_dir)
    monkeypatch.setattr(playback_queue, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(playback_queue, "connect", lambda: connect(db_path))
    monkeypatch.setattr(
        playback_queue,
        "ensure_controller_running",
        lambda **_kwargs: True,
    )

    item = enqueue_for_playback(
        QueuePlaybackRequest(
            raw_text="raw text",
            tts_text="spoken text",
            wav_data=wav_bytes(np.array([0.0, 0.1, -0.1], dtype=np.float32)),
            source_label="agent-a",
            voice="af_heart",
            language="a",
            speed=1.0,
            model="gpt-5.4-mini",
            reasoning_effort="medium",
        )
    )

    assert item.status == STATUS_QUEUED
    assert Path(item.audio_path).exists()
    assert Path(item.audio_path).parent == audio_dir

    with connect(db_path) as conn:
        queued = get_next_queued_item(conn)
    assert queued is not None
    assert queued.source_label == "agent-a"
    assert queued.tts_text == "spoken text"
    assert queued.model == "gpt-5.4-mini"
    assert queued.voice == "af_heart"
