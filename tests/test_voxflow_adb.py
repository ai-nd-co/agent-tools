from __future__ import annotations

import subprocess

import pytest

from agent_tools.voxflow_adb import (
    VoxFlowAdbDispatchRequest,
    _broadcast_args,
    resolve_single_connected_device,
)


def test_resolve_single_connected_device_returns_only_device(monkeypatch: object) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout="List of devices attached\nserial-1\tdevice\n",
            stderr="",
        ),
    )

    assert resolve_single_connected_device() == "serial-1"


def test_resolve_single_connected_device_rejects_multiple(monkeypatch: object) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout="List of devices attached\none\tdevice\ntwo\tdevice\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="Multiple ADB devices"):
        resolve_single_connected_device()


def test_broadcast_args_include_metadata() -> None:
    args = _broadcast_args(
        remote_item_id="item-1",
        created_at_ms=123,
        remote_audio_path="/sdcard/Android/data/com.ai_nd.co.voxflow/files/adb_inbox/item-1.wav",
        duration_ms=456,
        request=VoxFlowAdbDispatchRequest(
            raw_text="raw",
            tts_text="tts",
            wav_data=b"WAV",
            source_label="codex-notify:thread-1",
            voice="af_heart",
            language="a",
            speed=1.1,
            model="gpt-5.4-mini",
            reasoning_effort="medium",
            adb_serial=None,
        ),
    )

    assert "--es" in args
    assert "item_id" in args
    assert "tts_text" in args
    assert "source_label" in args
    assert "reasoning_effort" in args
    assert "456" in args
