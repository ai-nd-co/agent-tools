from __future__ import annotations

from pathlib import Path

import pytest

from agent_tools.agent_integration import AgentIntegrationStatus
from agent_tools.claude_integration import ClaudeIntegrationStatus
from agent_tools.codex_integration import CodexIntegrationStatus
from agent_tools.queue_db import STATUS_COMPLETED, STATUS_QUEUED, STATUS_STOPPED, QueueItem
from agent_tools.ui_app import (
    ProcessingItem,
    _ensure_audio_outputs_available,
    _load_preferred_transform_provider,
    _save_preferred_transform_provider,
    clamp_tts_speed,
    codex_integration_install_action_text,
    codex_integration_install_body,
    codex_integration_install_title,
    codex_integration_status_text,
    codex_integration_toggle_checked,
    interrupted_status_for_switch,
    merged_feed_entries,
    processing_stage_label,
    restored_scroll_value,
    should_show_codex_install_panel,
    tts_speed_label,
)


def _make_agent_status(
    *,
    enabled: bool,
    install_state: str,
    integration_state: str | None = None,
    codex_available: bool = True,
    claude_available: bool = True,
    codex_mode: str = "notify",
    codex_install_state: str = "installed",
    claude_install_state: str = "installed",
    availability_issues: tuple[str, ...] = (),
    issues: tuple[str, ...] = (),
) -> AgentIntegrationStatus:
    available_providers: list[str] = []
    if codex_available:
        available_providers.append("codex")
    if claude_available:
        available_providers.append("claude-code")
    return AgentIntegrationStatus(
        enabled=enabled,
        install_state=install_state,
        integration_state=integration_state or install_state,
        available_providers=tuple(available_providers),
        codex=CodexIntegrationStatus(
            mode=codex_mode,
            codex_home=Path("/tmp/.codex"),
            config_path=Path("/tmp/.codex/config.toml"),
            enabled=enabled,
            available=codex_available,
            install_state=codex_install_state,
            availability_issues=availability_issues,
        ),
        claude=ClaudeIntegrationStatus(
            mode="stop-hook",
            claude_home=Path("/tmp/.claude"),
            settings_path=Path("/tmp/.claude/settings.json"),
            enabled=enabled,
            available=claude_available,
            install_state=claude_install_state,
            hook_script_path=Path("/tmp/.claude/agent-tools/stop_tts.sh"),
        ),
        availability_issues=availability_issues,
        issues=issues,
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


def test_clamp_tts_speed_respects_bounds() -> None:
    assert clamp_tts_speed(0.1) == 0.7
    assert clamp_tts_speed(1.12) == 1.12
    assert clamp_tts_speed(3.0) == 1.3


def test_tts_speed_label_formats_two_decimals() -> None:
    assert tts_speed_label(1.0) == "TTS 1.00x"
    assert tts_speed_label(1.25) == "TTS 1.25x"


def test_transform_provider_preference_round_trip(monkeypatch: object, tmp_path: Path) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")

    assert _load_preferred_transform_provider() == "codex"

    _save_preferred_transform_provider("claude-code")

    assert _load_preferred_transform_provider() == "claude-code"


def test_codex_integration_status_text_for_enabled_notify() -> None:
    status = _make_agent_status(enabled=True, install_state="installed", codex_mode="notify")

    assert codex_integration_status_text(status) == "On - Codex and Claude Code available"
    assert codex_integration_toggle_checked(status) is True


def test_codex_integration_status_text_for_soft_disabled() -> None:
    status = _make_agent_status(
        enabled=False,
        install_state="installed",
        integration_state="installed",
        codex_mode="stop-hook",
    )

    assert codex_integration_status_text(
        status
    ) == "Off - soft disabled (Codex and Claude Code available)"
    assert codex_integration_toggle_checked(status) is False


def test_codex_integration_status_text_for_missing_or_broken() -> None:
    missing = _make_agent_status(
        enabled=False,
        install_state="missing",
        integration_state="missing",
        codex_available=False,
        claude_available=False,
        codex_install_state="missing",
        claude_install_state="missing",
        availability_issues=("codex:auth-unavailable", "claude:cli-unavailable"),
    )
    broken = _make_agent_status(
        enabled=True,
        install_state="installed",
        integration_state="broken",
        claude_available=False,
        issues=("codex:config-invalid",),
    )

    assert codex_integration_status_text(missing) == "Install Codex or Claude Code first"
    assert (
        codex_integration_status_text(broken)
        == "Codex available - auto-TTS integration needs repair"
    )
    assert codex_integration_toggle_checked(missing) is False


def test_codex_install_panel_helpers_for_missing_state() -> None:
    status = _make_agent_status(
        enabled=False,
        install_state="missing",
        integration_state="missing",
        codex_available=False,
        claude_available=False,
        codex_install_state="missing",
        claude_install_state="missing",
    )

    assert should_show_codex_install_panel(status) is True
    assert codex_integration_install_title(status) == "Install Codex or Claude Code first"
    assert codex_integration_install_action_text(status) == ""
    assert (
        "AgentTools can work with either Codex or Claude Code"
        in codex_integration_install_body(status)
    )


def test_codex_install_panel_helpers_for_broken_state() -> None:
    status = _make_agent_status(
        enabled=True,
        install_state="installed",
        integration_state="broken",
        issues=("claude:settings-json-invalid",),
    )

    assert should_show_codex_install_panel(status) is False
    assert codex_integration_install_title(status) == "Install Codex or Claude Code first"
    assert codex_integration_install_action_text(status) == "Repair"
    assert (
        "AgentTools can work with either Codex or Claude Code"
        in codex_integration_install_body(status)
    )


def test_codex_install_panel_hidden_when_installed() -> None:
    status = _make_agent_status(enabled=True, install_state="installed")

    assert should_show_codex_install_panel(status) is False


def test_ensure_audio_outputs_available_requires_backend() -> None:
    with pytest.raises(RuntimeError, match="Qt audio output backend"):
        _ensure_audio_outputs_available(0)
