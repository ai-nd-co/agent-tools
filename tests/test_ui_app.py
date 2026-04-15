from __future__ import annotations

from agent_tools.queue_db import STATUS_COMPLETED, STATUS_QUEUED, STATUS_STOPPED, QueueItem
from agent_tools.ui_app import (
    ProcessingItem,
    clamp_playback_rate,
    interrupted_status_for_switch,
    merged_feed_entries,
    playback_rate_label,
    processing_stage_label,
    restored_scroll_value,
)


def test_interrupted_status_for_switch_preserves_history_statuses() -> None:
    assert interrupted_status_for_switch(STATUS_COMPLETED) == STATUS_COMPLETED
    assert interrupted_status_for_switch(STATUS_STOPPED) == STATUS_STOPPED


def test_interrupted_status_for_switch_requeues_queued_items() -> None:
    assert interrupted_status_for_switch(STATUS_QUEUED) == STATUS_QUEUED


def test_processing_stage_label_falls_back_to_default() -> None:
    assert processing_stage_label(None) == "Processing audio"
    assert processing_stage_label("   ") == "Processing audio"


def test_merged_feed_entries_places_processing_rows_first() -> None:
    processing_items = [
        ProcessingItem(
            progress_id="progress-2",
            source_label="codex",
            preview_text="second",
            detail_text="second detail",
            stage="Synthesizing audio",
            order=2,
        ),
        ProcessingItem(
            progress_id="progress-1",
            source_label="codex",
            preview_text="first",
            detail_text="first detail",
            stage="Transforming for speech",
            order=1,
        ),
    ]
    queue_items = [
        QueueItem(
            queue_id=7,
            item_id="item-7",
            created_at="2026-04-15T00:00:00Z",
            updated_at="2026-04-15T00:00:00Z",
            source_label="codex",
            raw_text="raw",
            tts_text="tts",
            audio_path="C:/tmp/a.wav",
            status=STATUS_QUEUED,
            duration_ms=1000,
            error_message=None,
            voice="af_heart",
            language=None,
            speed=1.0,
            model="gpt-5.4-mini",
            reasoning_effort=None,
        )
    ]

    entries = merged_feed_entries(processing_items, queue_items)

    assert [entry.kind for entry in entries] == ["processing", "processing", "queue"]
    assert entries[0].processing_item is not None
    assert entries[0].processing_item.progress_id == "progress-2"


def test_restored_scroll_value_keeps_top_anchor() -> None:
    assert restored_scroll_value(old_value=0, old_max=200, new_max=260) == 0


def test_restored_scroll_value_keeps_bottom_anchor() -> None:
    assert restored_scroll_value(old_value=200, old_max=200, new_max=260) == 260


def test_restored_scroll_value_preserves_distance_from_bottom() -> None:
    assert restored_scroll_value(old_value=120, old_max=200, new_max=260) == 180


def test_clamp_playback_rate_respects_bounds() -> None:
    assert clamp_playback_rate(0.1) == 0.5
    assert clamp_playback_rate(1.37) == 1.37
    assert clamp_playback_rate(3.0) == 2.0


def test_playback_rate_label_formats_single_decimal() -> None:
    assert playback_rate_label(1.0) == "1.0x"
    assert playback_rate_label(1.25) == "1.2x"
