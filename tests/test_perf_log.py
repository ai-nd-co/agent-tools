from __future__ import annotations

import json
from pathlib import Path


def test_append_perf_event_writes_jsonl(monkeypatch: object, tmp_path: Path) -> None:
    import agent_tools.perf_log as perf_log_module
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")

    path = perf_log_module.append_perf_event(
        "ttsify_completed",
        trace_id="trace-1",
        total_ms=12.5,
        output_file=tmp_path / "out.wav",
    )

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["event"] == "ttsify_completed"
    assert payload["trace_id"] == "trace-1"
    assert payload["total_ms"] == 12.5
    assert payload["output_file"] == str(tmp_path / "out.wav")
