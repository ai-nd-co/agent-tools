from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from agent_tools.audio import wav_bytes
from agent_tools.playback_queue import QueuePlaybackRequest, enqueue_for_playback
from agent_tools.queue_db import (
    STATUS_QUEUED,
    connect,
    get_next_queued_item,
    get_next_queued_item_after,
    init_db,
)


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

    dispatch = enqueue_for_playback(
        QueuePlaybackRequest(
            raw_text="raw text",
            tts_text="spoken text",
            wav_data=wav_bytes(np.array([0.0, 0.1, -0.1], dtype=np.float32)),
            source_label="agent-a",
            voice="xenia",
            language="ru",
            speed=1.0,
            model="gpt-5.4-mini",
            reasoning_effort="medium",
            tts_engine="silero",
            tts_model="snakers4/silero-models:v5_5_ru",
        )
    )

    with connect(db_path) as conn:
        queued = get_next_queued_item(conn)
    assert queued is not None
    assert dispatch.queue_id == queued.queue_id
    assert queued.source_label == "agent-a"
    assert queued.tts_text == "spoken text"
    assert queued.model == "gpt-5.4-mini"
    assert queued.voice == "xenia"
    assert queued.tts_engine == "silero"
    assert queued.tts_model == "snakers4/silero-models:v5_5_ru"
    assert queued.status == STATUS_QUEUED
    assert Path(queued.audio_path).exists()
    assert Path(queued.audio_path).parent == audio_dir


def test_get_next_queued_item_after_returns_following_item(
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

    first = enqueue_for_playback(
        QueuePlaybackRequest(
            raw_text="first",
            tts_text="first",
            wav_data=wav_bytes(np.array([0.0, 0.1], dtype=np.float32)),
            source_label="agent-a",
            voice="af_heart",
            language="a",
            speed=1.0,
            model="gpt-5.4-mini",
            reasoning_effort=None,
        )
    )
    second = enqueue_for_playback(
        QueuePlaybackRequest(
            raw_text="second",
            tts_text="second",
            wav_data=wav_bytes(np.array([0.0, 0.1], dtype=np.float32)),
            source_label="agent-a",
            voice="af_heart",
            language="a",
            speed=1.0,
            model="gpt-5.4-mini",
            reasoning_effort=None,
        )
    )

    with connect(db_path) as conn:
        next_item = get_next_queued_item_after(conn, after_queue_id=first.queue_id or 0)

    assert next_item is not None
    assert next_item.item_id == second.item_id


def test_init_db_migrates_existing_queue_with_tts_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE queue_items (
            queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source_label TEXT,
            raw_text TEXT NOT NULL,
            tts_text TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            status TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            error_message TEXT,
            voice TEXT NOT NULL,
            language TEXT,
            speed REAL NOT NULL,
            model TEXT,
            reasoning_effort TEXT
        )
        """
    )

    init_db(conn)

    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(queue_items)")}
    conn.close()
    assert {"tts_engine", "tts_model"}.issubset(columns)


def test_init_db_serializes_concurrent_legacy_migrations(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-concurrent.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE queue_items (
            queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source_label TEXT,
            raw_text TEXT NOT NULL,
            tts_text TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            status TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            error_message TEXT,
            voice TEXT NOT NULL,
            language TEXT,
            speed REAL NOT NULL,
            model TEXT,
            reasoning_effort TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    def migrate() -> None:
        thread_conn = sqlite3.connect(db_path, timeout=5)
        try:
            init_db(thread_conn)
        finally:
            thread_conn.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _index: migrate(), range(8)))

    conn = sqlite3.connect(db_path)
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(queue_items)")}
    conn.close()
    assert {"tts_engine", "tts_model"}.issubset(columns)


def test_enqueue_failure_does_not_leave_orphaned_audio(
    tmp_path: Path, monkeypatch: object
) -> None:
    import agent_tools.playback_queue as playback_queue
    import agent_tools.queue_db as queue_db

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    monkeypatch.setattr(playback_queue, "audio_cache_dir", lambda: audio_dir)
    monkeypatch.setattr(playback_queue, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(
        queue_db,
        "get_item_by_item_id",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("readback failed")),
    )

    with sqlite3.connect(tmp_path / "queue.sqlite3") as raw_conn:
        init_db(raw_conn)
    monkeypatch.setattr(
        playback_queue,
        "connect",
        lambda: connect(tmp_path / "queue.sqlite3"),
    )

    request = QueuePlaybackRequest(
        raw_text="raw",
        tts_text="spoken",
        wav_data=wav_bytes(np.array([0.0, 0.1], dtype=np.float32)),
        source_label=None,
        voice="xenia",
        language="ru",
        speed=1.0,
        model=None,
        reasoning_effort=None,
        tts_engine="silero",
    )
    with pytest.raises(RuntimeError, match="readback failed"):
        enqueue_for_playback(request)

    assert list(audio_dir.iterdir()) == []
    raw_conn = sqlite3.connect(tmp_path / "queue.sqlite3")
    queued_rows = int(raw_conn.execute("SELECT COUNT(*) FROM queue_items").fetchone()[0])
    raw_conn.close()
    assert queued_rows == 0
