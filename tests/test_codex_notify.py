from __future__ import annotations

import json
from pathlib import Path

from agent_tools.codex_notify import dispatch_codex_notify, parse_codex_notify_payload
from agent_tools.codex_private_api import TransformResult
from agent_tools.tts import TtsResult
from agent_tools.ttsify import TtsifyResult


def test_parse_codex_notify_payload_reads_hyphenated_keys() -> None:
    payload = parse_codex_notify_payload(
        json.dumps(
            {
                "type": "agent-turn-complete",
                "thread-id": "thread-1",
                "turn-id": "turn-1",
                "cwd": "C:\\repo",
                "input-messages": ["hello"],
                "last-assistant-message": "done",
            }
        )
    )

    assert payload.event_type == "agent-turn-complete"
    assert payload.thread_id == "thread-1"
    assert payload.turn_id == "turn-1"
    assert payload.input_messages == ("hello",)
    assert payload.last_assistant_message == "done"


def test_dispatch_codex_notify_skips_blank_message(tmp_path: Path) -> None:
    codex_home = tmp_path / ".codex"
    result = dispatch_codex_notify(
        json.dumps(
            {
                "type": "agent-turn-complete",
                "thread-id": "thread-1",
                "turn-id": "turn-1",
                "cwd": "C:\\repo",
                "input-messages": ["hello"],
                "last-assistant-message": "   ",
            }
        ),
        codex_home=codex_home,
    )

    assert result.status == "ignored-blank-message"
    assert result.log_path.exists()


def test_dispatch_codex_notify_defaults_to_auto_and_dedupes(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    import agent_tools.codex_notify as notify_module

    captured: dict[str, object] = {}
    notice_events: list[tuple[str, str]] = []

    def fake_ttsify_text(input_text: str, options: object) -> TtsifyResult:
        captured["input_text"] = input_text
        assert isinstance(options, object)
        captured["device"] = options.device
        return TtsifyResult(
            transformed_text="spoken text",
            transform_result=TransformResult(
                text="spoken text",
                response_id="resp_1",
                usage=None,
                session_id="session-1",
            ),
            tts_result=TtsResult(wav=b"WAV", sample_rate=24_000, chunks=1),
            model="gpt-5.4-mini",
            reasoning_effort="medium",
            voice="af_heart",
            language="a",
            speed=1.0,
            device="auto",
            resolved_device="cpu",
            device_fallback_reason="torch cuda unavailable",
        )

    class FakeQueueItem:
        queue_id = 42
        item_id = "item-42"

    def fake_enqueue_for_playback(request: object) -> FakeQueueItem:
        assert isinstance(request, object)
        captured["source_label"] = request.source_label
        captured["raw_text"] = request.raw_text
        return FakeQueueItem()

    class FakeNotice:
        def __init__(self, progress_id: str) -> None:
            self.progress_id = progress_id
            self.available = True

        def finish(self) -> bool:
            notice_events.append(("finish", self.progress_id))
            return True

    def fake_start_processing_notice(**kwargs: object) -> FakeNotice:
        notice_events.append(("start", str(kwargs["stage"])))
        return FakeNotice(progress_id="notice-1")

    monkeypatch.setattr(notify_module, "ttsify_text", fake_ttsify_text)
    monkeypatch.setattr(notify_module, "enqueue_for_playback", fake_enqueue_for_playback)
    monkeypatch.setattr(notify_module, "start_processing_notice", fake_start_processing_notice)

    payload = json.dumps(
        {
            "type": "agent-turn-complete",
            "thread-id": "thread-1",
            "turn-id": "turn-1",
            "cwd": "C:\\repo",
            "input-messages": ["hello"],
            "last-assistant-message": "assistant message",
        }
    )

    result = dispatch_codex_notify(payload, codex_home=tmp_path / ".codex")
    duplicate = dispatch_codex_notify(payload, codex_home=tmp_path / ".codex")

    assert result.status == "queued"
    assert result.queue_id == 42
    assert notice_events == [("start", "Processing Codex reply"), ("finish", "notice-1")]
    assert captured["input_text"] == "assistant message"
    assert captured["device"] == "auto"
    assert captured["source_label"] == "codex-notify:thread-1"
    assert captured["raw_text"] == "assistant message"
    assert duplicate.status == "duplicate"
    assert result.marker_path is not None and result.marker_path.exists()
    log_text = result.log_path.read_text(encoding="utf-8")
    assert "dispatch_completed" in log_text
    assert "requested_device=auto" in log_text
    assert "resolved_device=cpu" in log_text
    assert "transform_ms=" in log_text
    assert "tts_device_probe_ms=" in log_text
    assert "tts_pipeline_init_ms=" in log_text
    assert "device_fallback_reason=torch cuda unavailable" in log_text
    assert "dominant_stage=" in log_text


def test_dispatch_codex_notify_respects_device_env(tmp_path: Path, monkeypatch: object) -> None:
    import agent_tools.codex_notify as notify_module

    captured: dict[str, object] = {}

    def fake_ttsify_text(_input_text: str, options: object) -> TtsifyResult:
        assert isinstance(options, object)
        captured["device"] = options.device
        return TtsifyResult(
            transformed_text="spoken text",
            transform_result=TransformResult(
                text="spoken text",
                response_id="resp_1",
                usage=None,
                session_id="session-1",
            ),
            tts_result=TtsResult(wav=b"WAV", sample_rate=24_000, chunks=1),
            model="gpt-5.4-mini",
            reasoning_effort=None,
            voice="af_heart",
            language="a",
            speed=1.0,
            device="cuda",
            resolved_device="cuda",
        )

    class FakeQueueItem:
        queue_id = 7
        item_id = "item-7"

    class FakeNotice:
        def __init__(self) -> None:
            self.available = True

        def finish(self) -> bool:
            return True

    monkeypatch.setattr(notify_module, "ttsify_text", fake_ttsify_text)
    monkeypatch.setattr(notify_module, "enqueue_for_playback", lambda _request: FakeQueueItem())
    monkeypatch.setattr(
        notify_module,
        "start_processing_notice",
        lambda **_kwargs: FakeNotice(),
    )
    monkeypatch.setenv("AGENT_TOOLS_KOKORO_DEVICE", "cuda")

    result = dispatch_codex_notify(
        json.dumps(
            {
                "type": "agent-turn-complete",
                "thread-id": "thread-2",
                "turn-id": "turn-2",
                "cwd": "C:\\repo",
                "input-messages": [],
                "last-assistant-message": "assistant message",
            }
        ),
        codex_home=tmp_path / ".codex",
    )

    assert result.status == "queued"
    assert captured["device"] == "cuda"
