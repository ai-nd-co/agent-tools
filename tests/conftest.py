from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _tts_stays_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches the office speech cluster by accident.

    The developer's own secrets file may carry the cluster key; with the default ``auto`` backend
    any test that synthesizes would then make a network call. Tests of the remote backend set the
    variables they need themselves.
    """
    monkeypatch.setenv("AGENT_TOOLS_TTS_BACKEND", "local")
