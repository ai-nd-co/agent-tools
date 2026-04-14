from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from agent_tools.runtime import ensure_runtime_dirs, queue_db_path

STATUS_QUEUED: Final = "queued"
STATUS_PLAYING: Final = "playing"
STATUS_PAUSED: Final = "paused"
STATUS_COMPLETED: Final = "completed"
STATUS_STOPPED: Final = "stopped"
STATUS_FAILED: Final = "failed"

QUEUE_ACTIVE_STATUSES: Final[tuple[str, ...]] = (STATUS_QUEUED, STATUS_PLAYING, STATUS_PAUSED)
QUEUE_HISTORY_STATUSES: Final[tuple[str, ...]] = (
    STATUS_COMPLETED,
    STATUS_STOPPED,
    STATUS_FAILED,
)


@dataclass(frozen=True)
class QueueItem:
    queue_id: int
    item_id: str
    created_at: str
    updated_at: str
    source_label: str | None
    raw_text: str
    tts_text: str
    audio_path: str
    status: str
    duration_ms: int
    error_message: str | None
    voice: str
    language: str | None
    speed: float
    model: str | None
    reasoning_effort: str | None


@dataclass(frozen=True)
class EnqueueRequest:
    source_label: str | None
    raw_text: str
    tts_text: str
    audio_path: Path
    duration_ms: int
    voice: str
    language: str | None
    speed: float
    model: str | None
    reasoning_effort: str | None


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    ensure_runtime_dirs()
    path = db_path or queue_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queue_items (
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


def normalize_inflight_items(conn: sqlite3.Connection) -> None:
    now = utc_now()
    conn.execute(
        """
        UPDATE queue_items
        SET status = ?, updated_at = ?
        WHERE status IN (?, ?)
        """,
        (STATUS_QUEUED, now, STATUS_PLAYING, STATUS_PAUSED),
    )
    conn.commit()


def enqueue_item(conn: sqlite3.Connection, request: EnqueueRequest) -> QueueItem:
    now = utc_now()
    item_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO queue_items (
            item_id, created_at, updated_at, source_label, raw_text, tts_text,
            audio_path, status, duration_ms, error_message, voice, language,
            speed, model, reasoning_effort
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item_id,
            now,
            now,
            request.source_label,
            request.raw_text,
            request.tts_text,
            str(request.audio_path),
            STATUS_QUEUED,
            request.duration_ms,
            None,
            request.voice,
            request.language,
            request.speed,
            request.model,
            request.reasoning_effort,
        ),
    )
    conn.commit()
    return get_item_by_item_id(conn, item_id)


def get_item_by_item_id(conn: sqlite3.Connection, item_id: str) -> QueueItem:
    row = conn.execute(
        "SELECT * FROM queue_items WHERE item_id = ?",
        (item_id,),
    ).fetchone()
    if row is None:
        raise KeyError(item_id)
    return row_to_item(row)


def get_current_item(conn: sqlite3.Connection) -> QueueItem | None:
    row = conn.execute(
        """
        SELECT * FROM queue_items
        WHERE status IN (?, ?)
        ORDER BY queue_id ASC
        LIMIT 1
        """,
        (STATUS_PLAYING, STATUS_PAUSED),
    ).fetchone()
    return None if row is None else row_to_item(row)


def get_next_queued_item(conn: sqlite3.Connection) -> QueueItem | None:
    row = conn.execute(
        """
        SELECT * FROM queue_items
        WHERE status = ?
        ORDER BY queue_id ASC
        LIMIT 1
        """,
        (STATUS_QUEUED,),
    ).fetchone()
    return None if row is None else row_to_item(row)


def list_queue_items(conn: sqlite3.Connection, limit: int = 50) -> list[QueueItem]:
    rows = conn.execute(
        """
        SELECT * FROM queue_items
        WHERE status IN (?, ?, ?)
        ORDER BY queue_id ASC
        LIMIT ?
        """,
        (STATUS_QUEUED, STATUS_PLAYING, STATUS_PAUSED, limit),
    ).fetchall()
    return [row_to_item(row) for row in rows]


def list_history_items(conn: sqlite3.Connection, limit: int = 50) -> list[QueueItem]:
    rows = conn.execute(
        """
        SELECT * FROM queue_items
        WHERE status IN (?, ?, ?)
        ORDER BY queue_id DESC
        LIMIT ?
        """,
        (STATUS_COMPLETED, STATUS_STOPPED, STATUS_FAILED, limit),
    ).fetchall()
    return [row_to_item(row) for row in rows]


def update_status(
    conn: sqlite3.Connection,
    item_id: str,
    status: str,
    *,
    error_message: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE queue_items
        SET status = ?, updated_at = ?, error_message = ?
        WHERE item_id = ?
        """,
        (status, utc_now(), error_message, item_id),
    )
    conn.commit()


def clear_queued_items(conn: sqlite3.Connection) -> None:
    conn.execute(
        "DELETE FROM queue_items WHERE status = ?",
        (STATUS_QUEUED,),
    )
    conn.commit()


def delete_item(conn: sqlite3.Connection, item_id: str) -> None:
    conn.execute("DELETE FROM queue_items WHERE item_id = ?", (item_id,))
    conn.commit()


def row_to_item(row: sqlite3.Row) -> QueueItem:
    return QueueItem(
        queue_id=int(row["queue_id"]),
        item_id=str(row["item_id"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        source_label=row["source_label"],
        raw_text=str(row["raw_text"]),
        tts_text=str(row["tts_text"]),
        audio_path=str(row["audio_path"]),
        status=str(row["status"]),
        duration_ms=int(row["duration_ms"]),
        error_message=row["error_message"],
        voice=str(row["voice"]),
        language=row["language"],
        speed=float(row["speed"]),
        model=row["model"],
        reasoning_effort=row["reasoning_effort"],
    )


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()

