from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_tools.zcode_startup import (
    Observation,
    StartupError,
    _high_owner_session,
    main,
    render_task_xml,
    supervise,
)

GENERATION = (123, 456, "episode")


class Clock:
    now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Host:
    def __init__(self, observations):
        self.observations = iter(observations)
        self.current = Observation("stopped")
        self.calls = []
        self.producer_failure = None

    def producer(self, *, start):
        self.calls.append(("producer", start))
        if self.producer_failure:
            raise StartupError(self.producer_failure)

    def observe(self):
        self.current = next(self.observations, self.current)
        return self.current

    def start(self):
        self.calls.append(("start",))

    def stop(self):
        self.calls.append(("stop",))

    def retire(self):
        self.calls.append(("retire",))


def run(host, stop=lambda: False, clock=None):
    clock = clock or Clock()
    return supervise(host, stop, clock=clock.time, sleep=clock.sleep)


@pytest.mark.parametrize("code", [
    "producer_not_high_owner_session", "producer_child_token_unproved",
    "producer_ambiguous", "producer_identity_drift", "producer_readiness_timeout",
])
def test_producer_failure_never_starts_or_stops_bridge(code):
    host = Host([])
    host.producer_failure = code
    with pytest.raises(StartupError, match=code):
        run(host)
    assert host.calls == [("producer", True)]


def test_preexisting_medium_bridge_is_not_elevated_by_wrapper():
    host = Host([Observation("ready", GENERATION, False)])
    with pytest.raises(StartupError, match="bridge_not_high_owner_session"):
        run(host)
    assert host.calls == [("producer", True)]


def test_stop_before_start_has_no_effect():
    host = Host([])
    assert run(host, lambda: True) == "stopped"
    assert not host.calls


def test_intentional_child_stop_does_not_respawn():
    host = Host([
        Observation("stopped"), Observation("ready", GENERATION, True), Observation("stopped")
    ])
    assert run(host) == "stopped"
    assert host.calls.count(("start",)) == 1
    assert ("retire",) not in host.calls


def test_stop_during_backoff_never_retires_or_restarts():
    clock = Clock()
    host = Host([Observation("stale_state", GENERATION, last_code="ready")])
    assert run(host, lambda: clock.now >= 2, clock) == "stopped"
    assert host.calls == [("producer", True)]
    assert clock.now == 2


def test_three_crash_retries_then_exhaustion():
    clock = Clock()
    host = Host([Observation("stale_state", GENERATION, last_code="ready")])
    with pytest.raises(StartupError, match="bridge_retry_exhausted"):
        run(host, clock=clock)
    assert clock.now == 100
    assert host.calls.count(("retire",)) == 3


@pytest.mark.parametrize("code", [
    "auth_failed", "workspace_identity_drift", "relay_reconnect_exhausted"
])
def test_fatal_stale_generation_never_retries(code):
    host = Host([Observation("stale_state", GENERATION, last_code=code)])
    with pytest.raises(StartupError, match="bridge_fatal_state"):
        run(host)
    assert host.calls == [("producer", True)]


def test_foreign_generation_change_is_not_stopped():
    host = Host([
        Observation("ready", GENERATION, True), Observation("ready", (789, 999, "other"), True)
    ])
    with pytest.raises(StartupError, match="bridge_generation_changed"):
        run(host)
    assert ("stop",) not in host.calls


def test_reconnecting_uses_bridge_budget_and_bounded_readiness():
    clock = Clock()
    host = Host([Observation("reconnecting", GENERATION, True)])
    with pytest.raises(StartupError, match="bridge_readiness_timeout"):
        run(host, clock=clock)
    assert clock.now == 60
    assert ("start",) not in host.calls
    assert ("retire",) not in host.calls


def test_stop_uses_existing_bridge_contract_once():
    clock = Clock()
    host = Host([Observation("ready", GENERATION, True)])
    assert run(host, lambda: clock.now >= 1, clock) == "stopped"
    assert host.calls.count(("stop",)) == 1


def test_prepared_task_is_disabled_high_interactive_without_scheduler_retry(tmp_path):
    command = tmp_path / "python.exe"
    root = ET.fromstring(render_task_xml(
        "S-1-5-21-1-2-3-1001", command, ["-I", "space and & value"], tmp_path
    ))
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    assert root.findtext("t:Principals/t:Principal/t:RunLevel", namespaces=ns) == "HighestAvailable"
    assert root.findtext(
        "t:Principals/t:Principal/t:LogonType", namespaces=ns
    ) == "InteractiveToken"
    assert root.findtext("t:Settings/t:Enabled", namespaces=ns) == "false"
    assert root.findtext("t:Settings/t:AllowHardTerminate", namespaces=ns) == "false"
    assert root.find("t:Settings/t:RestartOnFailure", ns) is None
    assert root.findtext("t:Actions/t:Exec/t:Arguments", namespaces=ns) == '-I "space and & value"'
    assert root.findtext(
        "t:Triggers/t:LogonTrigger/t:UserId", namespaces=ns
    ) == "S-1-5-21-1-2-3-1001"


def test_five_minutes_stable_resets_crash_budget():
    clock = Clock()
    stale = Observation("stale_state", GENERATION, last_code="ready")
    ready = Observation("ready", GENERATION, True)
    host = Host([stale] * 3 + [ready] * 1202 + [stale] * 4)
    with pytest.raises(StartupError, match="bridge_retry_exhausted"):
        run(host, clock=clock)
    assert host.calls.count(("retire",)) == 6


@pytest.mark.skipif(sys.platform != "win32", reason="real PowerShell helper/token fixture")
@pytest.mark.parametrize(("scenario", "code", "launches"), [
    ("Absent", "producer_absent", 0),
    ("Medium", "producer_not_high_owner_session", 0),
    ("Foreign", "producer_identity_unproved", 0),
    ("Multiple", "producer_ambiguous", 0),
    ("High", "producer_ready", 0),
    ("Launch", "producer_ready", 1),
])
def test_actual_producer_helper_with_disposable_process_observations(
    scenario, code, launches, tmp_path
):
    from agent_tools.tts_startup import _windows_system_directory

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        str(_windows_system_directory() / "WindowsPowerShell/v1.0/powershell.exe"),
        "-NoProfile", "-File", str(root / "tests/fixtures/zcode_producer_startup.ps1"),
        "-Helper", str(root / "src/agent_tools/zcode_producer_startup.ps1"),
        "-Scenario", scenario, "-FixtureRoot", str(tmp_path),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert lines[0]["code"] == code
    assert lines[1]["launches"] == launches


@pytest.mark.skipif(sys.platform != "win32", reason="real Windows process tokens")
def test_actual_token_matches_current_owner_and_rejects_other_owner():
    import win32api
    import win32con
    import win32security
    import win32ts

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        owner = win32security.ConvertSidToStringSid(
            win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        )
        high = bool(win32security.GetTokenInformation(token, win32security.TokenElevation))
        high = high and win32ts.ProcessIdToSessionId(os.getpid()) != 0
    finally:
        token.Close()
    assert _high_owner_session(os.getpid(), owner) == high
    assert not _high_owner_session(os.getpid(), "S-1-5-21-1-2-3-9999")
    assert not _high_owner_session(0, owner)


@pytest.mark.parametrize(("scenario", "expected", "written"), [
    ("released", "stopped", True),
    ("foreign", "bridge_release_unproved", True),
    ("stale", "wrapper_stale", False),
    ("wrong_config", "wrapper_state_invalid", False),
])
def test_stop_receipt_binds_episode_config_and_exact_runner(
    scenario, expected, written, tmp_path, monkeypatch, capsys
):
    from agent_tools import tts_startup, voice_onboarding, zcode_bridge

    state = {
        "schema": 1, "episode": "a" * 32, "config": "b" * 64,
        "runner_pid": 123, "runner_created": 456,
    }
    if scenario == "wrong_config":
        state["config"] = "c" * 64
    state_path = tmp_path / "startup-wrapper.json"
    stop_path = tmp_path / "startup-wrapper-stop.json"
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(
        voice_onboarding, "_read_private_bytes", lambda path, maximum: path.read_bytes()
    )
    monkeypatch.setattr(voice_onboarding, "_write_private_json", lambda path, payload, **kw:
                        path.write_text(json.dumps(payload)))
    monkeypatch.setattr(zcode_bridge, "load_bridge_config", lambda path:
                        SimpleNamespace(root=tmp_path, config_sha256="b" * 64))
    monkeypatch.setattr(tts_startup, "_inspect_windows_process", lambda pid:
                        None if stop_path.exists() or scenario == "stale"
                        else SimpleNamespace(creation_time_ns=456))
    monkeypatch.setattr(zcode_bridge, "inspect_bridge", lambda config: SimpleNamespace(
        code="foreign_listener" if scenario == "foreign" else "stopped", process_alive=False,
    ))
    result = main(["stop", "--root", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["code"] == expected
    assert result == (0 if scenario == "released" else 2)
    assert stop_path.exists() == written
    if written:
        assert json.loads(stop_path.read_text()) == state
