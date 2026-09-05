from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import agent_tools.tts_startup as startup
from agent_tools.tts_startup import (
    FileIdentity,
    LifecycleStatus,
    ListenerIdentity,
    ProcessIdentity,
    RunnerPolicy,
    ScheduledTaskDefinition,
    ScheduledTaskSnapshot,
    TtsStartupConfig,
    build_task_definition,
    execute_lifecycle,
    inspect_lifecycle,
    load_startup_config,
    plan_lifecycle,
    render_scheduled_task_xml,
    run_owned_runner,
)

TOKEN = "task319-owner-only-token-0123456789abcdef"
OWNER_SID = "S-1-5-21-319"
EXECUTABLE_PATH = str(Path(sys.executable).resolve())
EXECUTABLE_HASH = hashlib.sha256(Path(EXECUTABLE_PATH).read_bytes()).hexdigest()


def _random_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _identity(seed: int) -> FileIdentity:
    return FileIdentity(device=1, inode=seed, size=64, mtime_ns=seed * 100)


def _config(tmp_path: Path, *, credential: FileIdentity | None = None) -> TtsStartupConfig:
    token_file = tmp_path / "credential"
    token_file.write_text(TOKEN, encoding="ascii")
    state_directory = tmp_path / "state"
    log_directory = tmp_path / "logs"
    module_root = tmp_path / "src"
    bootstrap_path = module_root / "agent_tools" / "tts_startup_bootstrap.pyw"
    state_directory.mkdir()
    log_directory.mkdir()
    bootstrap_path.parent.mkdir(parents=True)
    bootstrap_path.write_text("bootstrap", encoding="ascii")
    config_path = tmp_path / "startup.json"
    config_path.write_text("{}", encoding="utf-8")
    return TtsStartupConfig(
        path=config_path.resolve(),
        config_sha256=hashlib.sha256(b"{}").hexdigest(),
        config_file_identity=startup._file_identity(config_path.stat()),
        owner_sid=OWNER_SID,
        python_executable=Path(sys.executable).resolve(),
        python_executable_sha256=EXECUTABLE_HASH,
        module_root=module_root.resolve(),
        bootstrap_path=bootstrap_path.resolve(),
        bootstrap_sha256=hashlib.sha256(b"bootstrap").hexdigest(),
        bootstrap_source="bootstrap",
        working_directory=tmp_path.resolve(),
        token_file=token_file.resolve(),
        credential_identity=credential or startup._file_identity(token_file.stat()),
        state_directory=state_directory.resolve(),
        log_directory=log_directory.resolve(),
        host="127.0.0.1",
        port=_random_loopback_port(),
        device="cpu",
    )


def _state(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    *,
    runner_generation: int = 100,
    child_generation: int = 200,
    config_sha256: str | None = None,
    credential_fingerprint: str | None = None,
    phase: str = "ready",
) -> startup.RuntimeState:
    return startup.RuntimeState(
        schema_version=1,
        config_sha256=config_sha256 or config.config_sha256,
        config_file_fingerprint=config.config_file_identity.fingerprint,
        task_fingerprint=definition.fingerprint,
        credential_fingerprint=(credential_fingerprint or config.credential_identity.fingerprint),
        runner_pid=1001,
        runner_creation_time_ns=runner_generation,
        child_pid=1002,
        child_creation_time_ns=child_generation,
        phase=phase,
        restart_count=0,
        authenticated_ready=phase == "ready",
    )


@dataclass
class Scenario:
    health: bool = True
    crash_before_ready: bool = False
    crash_after_ready: bool = False
    foreign_listener: bool = False
    release_delay_sleeps: int = 0


class FakeChild:
    def __init__(self, runtime: FakeRuntime, identity: ProcessIdentity, scenario: Scenario) -> None:
        self.runtime = runtime
        self._identity = identity
        self.scenario = scenario
        self.polls = 0
        self.health_seen = False
        self.exited = False

    @property
    def identity(self) -> ProcessIdentity:
        return self._identity

    def poll(self) -> int | None:
        self.polls += 1
        if self.scenario.crash_before_ready and self.polls >= 1:
            self._exit(17)
        elif self.scenario.crash_after_ready and self.health_seen and self.polls >= 2:
            self._exit(18)
        return 17 if self.exited else None

    def terminate(self) -> None:
        self._exit(0)

    def _exit(self, _code: int) -> None:
        if self.exited:
            return
        self.exited = True
        self.runtime.processes.pop(self.identity.pid, None)
        self.runtime.release_countdown = self.scenario.release_delay_sleeps


class FakeRuntime:
    def __init__(
        self,
        scenarios: Sequence[Scenario] = (),
        *,
        initial_listeners: Sequence[ListenerIdentity] = (),
    ) -> None:
        self.clock = 0.0
        self.owner_sid = OWNER_SID
        self.runner = ProcessIdentity(900, 90, EXECUTABLE_PATH, EXECUTABLE_HASH)
        self.processes: dict[int, ProcessIdentity] = {self.runner.pid: self.runner}
        self.scenarios = list(scenarios)
        self.children: list[FakeChild] = []
        self.initial_listeners = tuple(initial_listeners)
        self.release_countdown = 0
        self.on_sleep: Callable[[FakeRuntime], None] | None = None
        self.sleep_count = 0
        self.health_calls = 0

    def current_owner_sid(self) -> str:
        return self.owner_sid

    def current_process(self) -> ProcessIdentity:
        return self.runner

    def inspect_process(self, pid: int) -> ProcessIdentity | None:
        return self.processes.get(pid)

    def listeners(self, _host: str, _port: int) -> tuple[ListenerIdentity, ...]:
        if not self.children:
            return self.initial_listeners
        child = self.children[-1]
        if child.exited:
            if self.release_countdown > 0:
                return (ListenerIdentity(child.identity.pid, child.identity.creation_time_ns),)
            return ()
        if child.scenario.crash_before_ready:
            return ()
        if child.scenario.foreign_listener:
            return (ListenerIdentity(8888, 8888),)
        return (ListenerIdentity(child.identity.pid, child.identity.creation_time_ns),)

    def spawn_child(
        self,
        _command: Sequence[str],
        *,
        expected_executable: Path,
        expected_executable_sha256: str,
        working_directory: Path,
        log_directory: Path,
    ) -> FakeChild:
        assert expected_executable == Path(EXECUTABLE_PATH)
        assert expected_executable_sha256 == EXECUTABLE_HASH
        assert working_directory.is_absolute()
        assert log_directory.is_absolute()
        scenario = self.scenarios.pop(0) if self.scenarios else Scenario()
        identity = ProcessIdentity(
            1000 + len(self.children),
            2000 + len(self.children),
            EXECUTABLE_PATH,
            EXECUTABLE_HASH,
        )
        child = FakeChild(self, identity, scenario)
        self.children.append(child)
        self.processes[identity.pid] = identity
        return child

    def authenticated_health(
        self,
        _host: str,
        _port: int,
        token: str,
        *,
        expected_process: ProcessIdentity,
        timeout_seconds: float,
    ) -> bool:
        assert token == TOKEN
        assert timeout_seconds > 0
        self.health_calls += 1
        if not self.children:
            return True
        child = self.children[-1]
        assert expected_process == child.identity
        child.health_seen = True
        return child.scenario.health

    def now(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.clock += seconds
        self.sleep_count += 1
        if self.release_countdown > 0:
            self.release_countdown -= 1
        if self.on_sleep is not None:
            self.on_sleep(self)


class FakeScheduler:
    def __init__(self, snapshot: ScheduledTaskSnapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[str] = []

    def query(self, _definition: ScheduledTaskDefinition) -> ScheduledTaskSnapshot:
        self.calls.append("query")
        return self.snapshot

    def install(
        self,
        _definition: ScheduledTaskDefinition,
        xml: str,
        xml_path: Path,
    ) -> bool:
        assert startup.TASK_DESCRIPTION_PREFIX in xml
        assert xml_path.read_text(encoding="utf-8") == xml
        self.calls.append("install")
        return True

    def start(self, _definition: ScheduledTaskDefinition) -> bool:
        self.calls.append("start")
        self.snapshot = ScheduledTaskSnapshot(True, True, True, True, "Running", True)
        return True

    def disable(self, _definition: ScheduledTaskDefinition) -> bool:
        self.calls.append("disable")
        self.snapshot = ScheduledTaskSnapshot(
            self.snapshot.present,
            self.snapshot.exact,
            False,
            self.snapshot.running,
            self.snapshot.state,
            self.snapshot.owned,
        )
        return True

    def stop(self, _definition: ScheduledTaskDefinition) -> bool:
        self.calls.append("stop")
        self.snapshot = ScheduledTaskSnapshot(
            self.snapshot.present,
            self.snapshot.exact,
            self.snapshot.enabled,
            False,
            "Ready",
            self.snapshot.owned,
        )
        return True

    def uninstall(self, _definition: ScheduledTaskDefinition) -> bool:
        self.calls.append("uninstall")
        self.snapshot = ScheduledTaskSnapshot(False)
        return True


@pytest.fixture(autouse=True)
def _fixture_owner_only_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(startup, "_verify_owner_only_permissions", lambda *_args: None)
    monkeypatch.setattr(startup, "load_owner_only_bearer_token", lambda _path: TOKEN)


def test_config_load_pins_exact_executable_bootstrap_and_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "pythonw.exe"
    executable.write_bytes(b"python")
    module_root = tmp_path / "src"
    bootstrap = module_root / "agent_tools" / "tts_startup_bootstrap.pyw"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_bytes(b"bootstrap")
    working = tmp_path / "work"
    state = tmp_path / "state"
    logs = tmp_path / "logs"
    for directory in (working, state, logs):
        directory.mkdir()
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN, encoding="ascii")
    payload = {
        "schemaVersion": 1,
        "taskName": startup.TASK_NAME,
        "ownerSid": OWNER_SID,
        "pythonExecutable": str(executable.resolve()),
        "pythonExecutableSha256": hashlib.sha256(b"python").hexdigest(),
        "moduleRoot": str(module_root.resolve()),
        "bootstrapSha256": hashlib.sha256(b"bootstrap").hexdigest(),
        "workingDirectory": str(working.resolve()),
        "tokenFile": str(token_file.resolve()),
        "stateDirectory": str(state.resolve()),
        "logDirectory": str(logs.resolve()),
        "host": "127.0.0.1",
        "port": _random_loopback_port(),
        "device": "cpu",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_startup_config(config_path, current_owner_sid=OWNER_SID)

    assert config.python_executable_sha256 == hashlib.sha256(b"python").hexdigest()
    assert config.bootstrap_sha256 == hashlib.sha256(b"bootstrap").hexdigest()
    assert config.credential_identity.fingerprint
    assert config.host == "127.0.0.1"
    assert config.port != 4223

    payload["pythonExecutableSha256"] = "0" * 64
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(startup.TtsStartupError) as error:
        load_startup_config(config_path, current_owner_sid=OWNER_SID)
    assert error.value.code == "executable_drift"


def test_config_rejects_protected_file_collision_with_reserved_artifact(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "pythonw.exe"
    executable.write_bytes(b"python")
    module_root = tmp_path / "src"
    bootstrap = module_root / "agent_tools" / "tts_startup_bootstrap.pyw"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_bytes(b"bootstrap")
    working = tmp_path / "work"
    state = tmp_path / "state"
    logs = tmp_path / "logs"
    for directory in (working, state, logs):
        directory.mkdir()
    reserved_token = state / startup.STATE_FILENAME
    reserved_token.write_text(TOKEN, encoding="ascii")
    payload = {
        "schemaVersion": 1,
        "taskName": startup.TASK_NAME,
        "ownerSid": OWNER_SID,
        "pythonExecutable": str(executable.resolve()),
        "pythonExecutableSha256": hashlib.sha256(b"python").hexdigest(),
        "moduleRoot": str(module_root.resolve()),
        "bootstrapSha256": hashlib.sha256(b"bootstrap").hexdigest(),
        "workingDirectory": str(working.resolve()),
        "tokenFile": str(reserved_token.resolve()),
        "stateDirectory": str(state.resolve()),
        "logDirectory": str(logs.resolve()),
        "host": "127.0.0.1",
        "port": _random_loopback_port(),
        "device": "cpu",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(startup.TtsStartupError) as error:
        load_startup_config(config_path, current_owner_sid=OWNER_SID)

    assert error.value.code == "protected_path_collision"
    assert reserved_token.read_text(encoding="ascii") == TOKEN

    normal_token = tmp_path / "normal-token"
    normal_token.write_text(TOKEN, encoding="ascii")
    payload["tokenFile"] = str(normal_token.resolve())
    reserved_config = state / startup.TASK_XML_FILENAME
    reserved_config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(startup.TtsStartupError) as error:
        load_startup_config(reserved_config, current_owner_sid=OWNER_SID)
    assert error.value.code == "protected_path_collision"

    reserved_backup = logs / f"{startup.EVENT_LOG_FILENAME}.1"
    reserved_backup.write_text(TOKEN, encoding="ascii")
    payload["tokenFile"] = str(reserved_backup.resolve())
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(startup.TtsStartupError) as error:
        load_startup_config(config_path, current_owner_sid=OWNER_SID)
    assert error.value.code == "protected_path_collision"


def test_task_definition_is_exact_hidden_current_user_logon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_system_root = tmp_path / "untrusted-system-root"
    monkeypatch.setenv("SystemRoot", str(fake_system_root))
    config = _config(tmp_path)
    definition = build_task_definition(config)
    xml = render_scheduled_task_xml(definition)

    assert startup.TASK_NAME == definition.task_name
    assert config.owner_sid in xml
    assert "InteractiveToken" in xml
    assert "LeastPrivilege" in xml
    assert "<Hidden>true</Hidden>" in xml
    assert "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>" in xml
    assert "<Interval>PT1M</Interval>" in xml
    assert "<Count>3</Count>" in xml
    assert "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in xml
    assert "<AllowHardTerminate>true</AllowHardTerminate>" in xml
    assert str(config.bootstrap_path) not in definition.arguments
    assert config.bootstrap_source not in definition.arguments
    assert definition.command.casefold().endswith("windowspowershell\\v1.0\\powershell.exe")
    encoded_launcher = definition.arguments.rsplit(" ", 1)[-1].strip('"')
    launcher_source = base64.b64decode(encoded_launcher).decode("utf-16le")
    assert config.bootstrap_source in launcher_source
    assert config.config_sha256 in launcher_source
    assert config.credential_identity.fingerprint in launcher_source
    assert "FILE_FLAG_OPEN_REPARSE_POINT" in launcher_source
    assert "FILE_ATTRIBUTE_REPARSE_POINT" in launcher_source
    assert config.python_executable_sha256 in launcher_source
    assert os.path.normcase(str(fake_system_root)) not in os.path.normcase(definition.command)
    assert TOKEN not in xml


@pytest.mark.skipif(sys.platform != "win32", reason="Windows trusted launcher")
def test_trusted_task_launcher_hash_locks_and_runs_exact_fixture(tmp_path: Path) -> None:
    config = _config(tmp_path)
    python_arguments = startup.subprocess.list2cmdline(("-I", "-c", "import sys; sys.exit(0)"))
    command, arguments = startup._trusted_task_launcher(config, python_arguments)
    encoded = arguments.rsplit(" ", 1)[-1].strip('"')

    completed = startup.subprocess.run(
        (
            command,
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-EncodedCommand",
            encoded,
        ),
        check=False,
        capture_output=True,
        timeout=15,
    )

    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""


def _status_obj(code: str, **values: bool) -> LifecycleStatus:
    defaults = {
        "task_present": True,
        "task_exact": True,
        "task_enabled": False,
        "task_running": False,
        "config_exact": True,
        "credential_exact": True,
        "runner_alive": False,
        "child_alive": False,
        "listener_owned": False,
        "authenticated_ready": False,
    }
    defaults.update(values)
    return LifecycleStatus(code=code, **defaults)


@pytest.mark.parametrize(
    ("action", "status", "expected_code", "allowed", "changed"),
    [
        (
            "install",
            _status_obj("task_absent", task_present=False, task_exact=False),
            "install",
            True,
            True,
        ),
        (
            "install",
            _status_obj("task_drift", task_exact=False),
            "task_drift",
            False,
            False,
        ),
        (
            "stop",
            _status_obj("task_absent", task_present=False, task_exact=False),
            "already_stopped",
            True,
            False,
        ),
        (
            "uninstall",
            _status_obj("task_absent", task_present=False, task_exact=False),
            "already_uninstalled",
            True,
            False,
        ),
        ("start", _status_obj("stopped"), "start", True, True),
        (
            "start",
            _status_obj(
                "ready",
                task_enabled=True,
                task_running=True,
                runner_alive=True,
                child_alive=True,
                listener_owned=True,
                authenticated_ready=True,
            ),
            "already_started",
            True,
            False,
        ),
        (
            "start",
            _status_obj("foreign_listener", task_enabled=True),
            "foreign_listener",
            False,
            False,
        ),
        (
            "stop",
            _status_obj(
                "ready",
                task_enabled=True,
                task_running=True,
                runner_alive=True,
                child_alive=True,
                listener_owned=True,
                authenticated_ready=True,
            ),
            "stop",
            True,
            True,
        ),
        ("uninstall", _status_obj("stopped"), "uninstall", True, True),
        (
            "uninstall",
            _status_obj("ready", runner_alive=True),
            "not_stopped",
            False,
            False,
        ),
    ],
)
def test_lifecycle_plans_are_idempotent_and_fail_closed(
    action: str,
    status: LifecycleStatus,
    expected_code: str,
    allowed: bool,
    changed: bool,
) -> None:
    plan = plan_lifecycle(action, status)
    assert (plan.code, plan.allowed, plan.changed) == (expected_code, allowed, changed)


def test_install_execution_writes_exact_definition_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    scheduler = FakeScheduler(ScheduledTaskSnapshot(False))
    runtime = FakeRuntime()

    plan = execute_lifecycle("install", config, definition, scheduler, runtime)

    assert plan.code == "install"
    assert scheduler.calls == ["query", "install"]
    assert (config.state_directory / startup.TASK_XML_FILENAME).exists()


def test_duplicate_install_and_start_do_not_mutate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime()
    scheduler = FakeScheduler(ScheduledTaskSnapshot(True, True, True, False, "Ready"))

    install_plan = execute_lifecycle("install", config, definition, scheduler, runtime)
    assert install_plan.code == "already_installed"
    assert scheduler.calls == ["query"]

    runtime.processes[1001] = ProcessIdentity(1001, 100, EXECUTABLE_PATH, EXECUTABLE_HASH)
    runtime.processes[1002] = ProcessIdentity(1002, 200, EXECUTABLE_PATH, EXECUTABLE_HASH)
    startup._publish_state(
        config,
        definition,
        runtime.runner,
        runtime.processes[1002],
        "ready",
        0,
        True,
    )
    runtime.initial_listeners = (ListenerIdentity(1002, 200),)
    scheduler.snapshot = ScheduledTaskSnapshot(True, True, True, True, "Running")
    start_plan = execute_lifecycle("start", config, definition, scheduler, runtime)
    assert start_plan.code == "already_started"
    assert scheduler.calls == ["query", "query"]


def test_stop_terminates_exact_task_during_pre_state_startup_window(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime()
    scheduler = FakeScheduler(ScheduledTaskSnapshot(True, True, True, True, "Running", True))

    plan = execute_lifecycle(
        "stop",
        config,
        definition,
        scheduler,
        runtime,
        policy=_runner_policy(),
    )

    assert plan.code == "stop_startup_window"
    assert scheduler.calls == ["query", "disable", "stop", "query"]

    restart = execute_lifecycle("start", config, definition, scheduler, runtime)
    assert restart.code == "start"
    assert scheduler.calls == ["query", "disable", "stop", "query", "query", "start"]


@pytest.mark.parametrize("state_kind", ["invalid", "stale"])
def test_stop_never_reports_stopped_for_running_task_with_untrusted_state(
    tmp_path: Path,
    state_kind: str,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    state_path = config.state_directory / startup.STATE_FILENAME
    if state_kind == "invalid":
        state_path.write_text("invalid", encoding="ascii")
    else:
        startup._write_private_json(state_path, startup.asdict(_state(config, definition)))
    runtime = FakeRuntime()
    scheduler = FakeScheduler(ScheduledTaskSnapshot(True, True, True, True, "Running", True))

    plan = execute_lifecycle(
        "stop",
        config,
        definition,
        scheduler,
        runtime,
        policy=_runner_policy(),
    )

    assert plan.code == "stop_startup_window"
    assert "stop" in scheduler.calls


def test_stop_release_proof_rejects_live_recorded_runner_generation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime()
    runtime.processes[1001] = ProcessIdentity(1001, 100, EXECUTABLE_PATH, EXECUTABLE_HASH)
    startup._publish_state(
        config,
        definition,
        runtime.processes[1001],
        None,
        "stopped",
        0,
        False,
    )
    scheduler = FakeScheduler(ScheduledTaskSnapshot(True, True, False, False, "Ready", True))

    with pytest.raises(startup.TtsStartupError) as error:
        startup._wait_for_external_stop(config, definition, scheduler, runtime, _runner_policy())

    assert error.value.code == "stop_timeout"


def test_status_refuses_pid_reuse_and_foreign_listener(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    state = _state(config, definition)
    startup._write_private_json(
        config.state_directory / startup.STATE_FILENAME,
        startup.asdict(state),
    )
    runtime = FakeRuntime(initial_listeners=(ListenerIdentity(1002, 201),))
    runtime.processes[1001] = ProcessIdentity(1001, 101, EXECUTABLE_PATH, EXECUTABLE_HASH)
    runtime.processes[1002] = ProcessIdentity(1002, 201, EXECUTABLE_PATH, EXECUTABLE_HASH)
    task = ScheduledTaskSnapshot(True, True, True, True, "Running")

    status = inspect_lifecycle(config, definition, task, runtime)

    assert status.code == "foreign_listener"
    assert status.runner_alive is False
    assert status.child_alive is False
    assert status.listener_owned is False


def test_status_distinguishes_stale_config_and_credential_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    task = ScheduledTaskSnapshot(True, True, True, False, "Ready")
    runtime = FakeRuntime()

    startup._write_private_json(
        config.state_directory / startup.STATE_FILENAME,
        startup.asdict(_state(config, definition, config_sha256="c" * 64)),
    )
    assert inspect_lifecycle(config, definition, task, runtime).code == "config_drift"

    startup._write_private_json(
        config.state_directory / startup.STATE_FILENAME,
        startup.asdict(_state(config, definition, credential_fingerprint="d" * 64)),
    )
    assert inspect_lifecycle(config, definition, task, runtime).code == "credential_drift"

    startup._write_private_json(
        config.state_directory / startup.STATE_FILENAME,
        startup.asdict(_state(config, definition)),
    )
    assert inspect_lifecycle(config, definition, task, runtime).code == "stale_state"


def test_status_distinguishes_invalid_state_and_owned_task_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime()
    state_path = config.state_directory / startup.STATE_FILENAME
    state_path.write_text("not-json", encoding="utf-8")

    exact_task = ScheduledTaskSnapshot(True, True, True, False, "Ready", True)
    assert inspect_lifecycle(config, definition, exact_task, runtime).code == "state_invalid"

    state_path.unlink()
    owned_drift = ScheduledTaskSnapshot(True, False, True, False, "Ready", True)
    assert inspect_lifecycle(config, definition, owned_drift, runtime).code == "task_drift"
    foreign = ScheduledTaskSnapshot(True, False, True, False, "Ready", False)
    assert inspect_lifecycle(config, definition, foreign, runtime).code == "foreign_task"


def test_invalid_state_cannot_hide_existing_listener_as_stopped(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    state_path = config.state_directory / startup.STATE_FILENAME
    state_path.write_text("invalid", encoding="ascii")
    listener = ListenerIdentity(7777, 8888)
    runtime = FakeRuntime(initial_listeners=(listener,))
    task = ScheduledTaskSnapshot(True, True, False, False, "Disabled", True)

    status = inspect_lifecycle(config, definition, task, runtime)
    plan = plan_lifecycle("stop", status)

    assert status.code == "foreign_listener"
    assert plan.code == "foreign_listener_preserved"
    assert plan.code != "already_stopped"
    assert plan.changed is False
    assert runtime.listeners(config.host, config.port) == (listener,)


def test_status_recognizes_running_task_before_state_publication(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime()
    running = ScheduledTaskSnapshot(True, True, True, True, "Running", True)

    status = inspect_lifecycle(config, definition, running, runtime)

    assert status.code == "starting"
    assert status.task_running is True


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"runner_pid": True},
        {"runner_pid": startup.MAX_WINDOWS_PID + 1},
        {"runner_creation_time_ns": startup.MAX_WINDOWS_TIMESTAMP + 1},
        {"phase": "unexpected"},
        {"authenticated_ready": True, "phase": "starting"},
        {"authenticated_ready": False, "phase": "ready"},
        {
            "authenticated_ready": True,
            "phase": "ready",
            "child_pid": None,
            "child_creation_time_ns": None,
        },
        {"child_pid": 1002, "child_creation_time_ns": None},
        {"child_pid": startup.MAX_WINDOWS_PID + 1},
    ],
)
def test_runtime_state_schema_is_strict(tmp_path: Path, mutation: dict[str, object]) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    payload = startup.asdict(_state(config, definition))
    payload.update(mutation)
    path = config.state_directory / startup.STATE_FILENAME
    startup._write_private_json(path, payload)

    assert startup._read_runtime_state(path) is None


def _runner_policy(**values: Any) -> RunnerPolicy:
    defaults: dict[str, Any] = {
        "ready_timeout_seconds": 0.05,
        "release_timeout_seconds": 0.05,
        "restart_delay_seconds": 0.02,
        "restart_reset_seconds": 1.0,
        "max_restarts": 3,
        "poll_seconds": 0.01,
    }
    defaults.update(values)
    return RunnerPolicy(**defaults)


def _request_stop_after_ready(
    runtime: FakeRuntime,
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
) -> None:
    if runtime.children and runtime.children[-1].health_seen:
        startup._write_stop_request(config, definition)
        runtime.on_sleep = None


def test_runner_starts_exact_child_and_intentional_stop_does_not_respawn(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime([Scenario()])
    runtime.on_sleep = lambda current: _request_stop_after_ready(current, config, definition)

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(),
    )

    assert result == 0
    assert len(runtime.children) == 1
    assert runtime.children[0].exited is True
    state = startup._read_runtime_state(config.state_directory / startup.STATE_FILENAME)
    assert state is not None and state.phase == "stopped"


def test_runner_restarts_one_crash_then_becomes_ready(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime([Scenario(crash_before_ready=True), Scenario()])
    runtime.on_sleep = lambda current: _request_stop_after_ready(current, config, definition)

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(),
    )

    assert result == 0
    assert len(runtime.children) == 2


def test_runner_crash_recovery_is_bounded(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime([Scenario(crash_before_ready=True)] * 3)

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(max_restarts=2),
    )

    assert result == 22
    assert len(runtime.children) == 3
    state = startup._read_runtime_state(config.state_directory / startup.STATE_FILENAME)
    assert state is not None and state.phase == "restart_exhausted"


def test_runner_intentional_stop_interrupts_restart_delay(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime([Scenario(crash_before_ready=True), Scenario()])

    def request_during_delay(current: FakeRuntime) -> None:
        state = startup._read_runtime_state(config.state_directory / startup.STATE_FILENAME)
        if state is not None and state.phase == "restart_wait":
            startup._write_stop_request(config, definition)
            current.on_sleep = None

    runtime.on_sleep = request_during_delay
    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(),
    )

    assert result == 0
    assert len(runtime.children) == 1


def test_runner_refuses_foreign_listener_before_spawning(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime(initial_listeners=(ListenerIdentity(777, 888),))

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(),
    )

    assert result == 20
    assert runtime.children == []


def test_runner_rechecks_listener_generation_before_bearer_use(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)

    class SwapRuntime(FakeRuntime):
        def listeners(self, host: str, port: int) -> tuple[ListenerIdentity, ...]:
            listeners = super().listeners(host, port)
            if self.children and not self.children[-1].exited and self.sleep_count == 0:
                self.sleep_count = 1
                return listeners
            if self.children and not self.children[-1].exited:
                return (ListenerIdentity(8080, 9090),)
            return listeners

    runtime = SwapRuntime([Scenario()])

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(max_restarts=0),
    )

    assert result == 20
    assert runtime.health_calls == 0


def test_runner_readiness_failure_terminates_and_exhausts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime([Scenario(health=False)])

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(max_restarts=0),
    )

    assert result == 22
    assert runtime.children[0].exited is True


class _HangingHealthRuntime(FakeRuntime):
    """Ready child keeps its PID and listener but serves failing health responses."""

    def __init__(self, scenarios: Sequence[Scenario], *, failing_probes: int) -> None:
        super().__init__(scenarios)
        self.failing_probes = failing_probes
        self.failed_probes = 0

    def authenticated_health(
        self,
        host: str,
        port: int,
        token: str,
        *,
        expected_process: ProcessIdentity,
        timeout_seconds: float,
    ) -> bool:
        if self.failing_probes > 0 and self.children and self.children[-1].health_seen:
            self.failing_probes -= 1
            self.failed_probes += 1
            return False
        return super().authenticated_health(
            host, port, token, expected_process=expected_process, timeout_seconds=timeout_seconds
        )


def test_runner_recovers_ready_child_whose_authenticated_health_hangs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = _HangingHealthRuntime([Scenario(), Scenario()], failing_probes=3)
    published: list[tuple[str, bool]] = []
    original_publish = startup._publish_state

    def spy_publish(
        spy_config: TtsStartupConfig,
        spy_definition: ScheduledTaskDefinition,
        spy_runner: ProcessIdentity,
        spy_child: ProcessIdentity | None,
        phase: str,
        restart_count: int,
        authenticated_ready: bool,
    ) -> None:
        published.append((phase, authenticated_ready))
        original_publish(
            spy_config,
            spy_definition,
            spy_runner,
            spy_child,
            phase,
            restart_count,
            authenticated_ready,
        )

    monkeypatch.setattr(startup, "_publish_state", spy_publish)

    def hang_then_stop(_current: FakeRuntime) -> None:
        if runtime.children and len(runtime.children) == 2 and runtime.children[1].health_seen:
            startup._write_stop_request(config, definition)
            runtime.on_sleep = None

    runtime.on_sleep = hang_then_stop

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(health_reprobe_seconds=0.02),
    )

    assert result == 0
    assert len(runtime.children) == 2
    assert runtime.children[0].exited is True
    assert runtime.failed_probes == 3
    assert ("health_failed", False) in published


def test_runner_tolerates_single_transient_health_failure_in_ready_monitor(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = _HangingHealthRuntime([Scenario()], failing_probes=1)

    def stop_after_monitor_reprobe(_current: FakeRuntime) -> None:
        if runtime.failed_probes >= 1 and runtime.health_calls >= 4:
            startup._write_stop_request(config, definition)
            runtime.on_sleep = None

    runtime.on_sleep = stop_after_monitor_reprobe

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(health_reprobe_seconds=0.02),
    )

    assert result == 0
    assert len(runtime.children) == 1
    assert runtime.failed_probes == 1
    assert runtime.health_calls >= 4


def test_runner_hung_ready_child_recovery_is_bounded(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = _HangingHealthRuntime([Scenario()], failing_probes=99)

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(
            max_restarts=0,
            restart_reset_seconds=1000.0,
            health_reprobe_seconds=0.02,
        ),
    )

    assert result == 22
    assert len(runtime.children) == 1
    assert runtime.children[0].exited is True
    state = startup._read_runtime_state(config.state_directory / startup.STATE_FILENAME)
    assert state is not None and state.phase == "restart_exhausted"


class _ScriptedMonitorHealthRuntime(FakeRuntime):
    """Ready-monitor probes after the first healthy probe return scripted outcomes."""

    def __init__(
        self,
        scenarios: Sequence[Scenario],
        *,
        monitor_results: Sequence[bool],
    ) -> None:
        super().__init__(scenarios)
        self._monitor_results = list(monitor_results)
        self.monitor_probe_calls = 0

    def authenticated_health(
        self,
        host: str,
        port: int,
        token: str,
        *,
        expected_process: ProcessIdentity,
        timeout_seconds: float,
    ) -> bool:
        child = self.children[-1]
        if self._monitor_results and child.health_seen:
            self.monitor_probe_calls += 1
            return self._monitor_results.pop(0)
        return super().authenticated_health(
            host, port, token, expected_process=expected_process, timeout_seconds=timeout_seconds
        )


def test_runner_resets_health_failure_counter_on_interleaved_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = _ScriptedMonitorHealthRuntime(
        [Scenario(), Scenario()],
        monitor_results=[False, True, False, False, False],
    )
    published: list[tuple[str, bool]] = []
    original_publish = startup._publish_state

    def spy_publish(
        spy_config: TtsStartupConfig,
        spy_definition: ScheduledTaskDefinition,
        spy_runner: ProcessIdentity,
        spy_child: ProcessIdentity | None,
        phase: str,
        restart_count: int,
        authenticated_ready: bool,
    ) -> None:
        published.append((phase, authenticated_ready))
        original_publish(
            spy_config,
            spy_definition,
            spy_runner,
            spy_child,
            phase,
            restart_count,
            authenticated_ready,
        )

    monkeypatch.setattr(startup, "_publish_state", spy_publish)

    def stop_after_replacement_ready(_current: FakeRuntime) -> None:
        if runtime.children and len(runtime.children) == 2 and runtime.children[1].health_seen:
            startup._write_stop_request(config, definition)
            runtime.on_sleep = None

    runtime.on_sleep = stop_after_replacement_ready

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(health_reprobe_seconds=0.02),
    )

    assert result == 0
    assert len(runtime.children) == 2
    assert runtime.children[0].exited is True
    assert runtime.monitor_probe_calls == 5
    assert ("health_failed", False) in published


def test_runner_policy_defaults_reprobe_health_with_consecutive_failure_budget() -> None:
    policy = RunnerPolicy()

    assert policy.health_reprobe_seconds == 30.0
    assert policy.health_failure_threshold == 3


@pytest.mark.parametrize(
    ("target", "expected_result", "expected_phase"),
    [
        ("config", 23, "config_drift"),
        ("credential", 24, "credential_drift"),
        ("bootstrap", 26, "bootstrap_drift"),
    ],
)
def test_runner_stops_owned_child_on_live_input_drift(
    tmp_path: Path,
    target: str,
    expected_result: int,
    expected_phase: str,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime([Scenario()])

    def replace_input(current: FakeRuntime) -> None:
        if not current.children or not current.children[-1].health_seen:
            return
        if target == "config":
            path = config.path
        elif target == "credential":
            path = config.token_file
        else:
            path = config.bootstrap_path
        path.write_text("replacement", encoding="utf-8")
        current.on_sleep = None

    runtime.on_sleep = replace_input
    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(),
    )

    assert result == expected_result
    assert len(runtime.children) == 1
    assert runtime.children[0].exited is True
    state = startup._read_runtime_state(config.state_directory / startup.STATE_FILENAME)
    assert state is not None and state.phase == expected_phase


def test_runner_rejects_wrong_child_executable_before_bearer_use(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)

    class WrongChildRuntime(FakeRuntime):
        def spawn_child(
            self,
            command: Sequence[str],
            *,
            expected_executable: Path,
            expected_executable_sha256: str,
            working_directory: Path,
            log_directory: Path,
        ) -> FakeChild:
            child = super().spawn_child(
                command,
                expected_executable=expected_executable,
                expected_executable_sha256=expected_executable_sha256,
                working_directory=working_directory,
                log_directory=log_directory,
            )
            child._identity = ProcessIdentity(
                child.identity.pid,
                child.identity.creation_time_ns,
                str(tmp_path / "foreign.exe"),
                "f" * 64,
            )
            self.processes[child.identity.pid] = child.identity
            return child

    runtime = WrongChildRuntime([Scenario()])

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(),
    )

    assert result == 25
    assert runtime.children[0].exited is True
    assert runtime.health_calls == 0


def test_runner_terminates_exact_child_when_post_spawn_publication_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime([Scenario()])
    real_publish = startup._publish_state
    calls = 0

    def fail_after_spawn(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        real_publish(*args, **kwargs)

    monkeypatch.setattr(startup, "_publish_state", fail_after_spawn)

    with pytest.raises(OSError, match="injected publication failure"):
        run_owned_runner(
            config,
            definition,
            runtime,
            expected_config_sha256=config.config_sha256,
            expected_config_file_fingerprint=config.config_file_identity.fingerprint,
            expected_credential_fingerprint=config.credential_identity.fingerprint,
            policy=_runner_policy(),
        )

    assert runtime.children[0].exited is True
    assert not (config.state_directory / startup.LOCK_FILENAME).exists()


def test_runner_waits_for_independent_listener_release(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime([Scenario(crash_before_ready=True, release_delay_sleeps=2), Scenario()])
    runtime.on_sleep = lambda current: _request_stop_after_ready(current, config, definition)

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(),
    )

    assert result == 0
    assert runtime.clock >= 0.02
    assert len(runtime.children) == 2


def test_runner_refuses_restart_when_listener_release_exceeds_budget(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime([Scenario(crash_before_ready=True, release_delay_sleeps=100)])

    result = run_owned_runner(
        config,
        definition,
        runtime,
        expected_config_sha256=config.config_sha256,
        expected_config_file_fingerprint=config.config_file_identity.fingerprint,
        expected_credential_fingerprint=config.credential_identity.fingerprint,
        policy=_runner_policy(release_timeout_seconds=0.02),
    )

    assert result == 21
    assert len(runtime.children) == 1


@pytest.mark.parametrize(
    ("config_hash", "config_file", "credential", "expected_code"),
    [
        ("0" * 64, None, None, "config_drift"),
        (None, "0" * 64, None, "config_drift"),
        (None, None, "0" * 64, "credential_drift"),
    ],
)
def test_runner_refuses_config_and_credential_drift_before_spawn(
    tmp_path: Path,
    config_hash: str | None,
    config_file: str | None,
    credential: str | None,
    expected_code: str,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime()

    with pytest.raises(startup.TtsStartupError) as error:
        run_owned_runner(
            config,
            definition,
            runtime,
            expected_config_sha256=config_hash or config.config_sha256,
            expected_config_file_fingerprint=(
                config_file or config.config_file_identity.fingerprint
            ),
            expected_credential_fingerprint=(credential or config.credential_identity.fingerprint),
            policy=_runner_policy(),
        )

    assert error.value.code == expected_code
    assert runtime.children == []


def test_runner_lock_rejects_live_generation_and_recovers_exact_stale_lock(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime([Scenario()])
    lock_path = config.state_directory / startup.LOCK_FILENAME
    live_lock = {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "config_file_fingerprint": config.config_file_identity.fingerprint,
        "task_fingerprint": definition.fingerprint,
        "runner_pid": runtime.runner.pid,
        "runner_creation_time_ns": runtime.runner.creation_time_ns,
    }
    startup._write_new_private_json(lock_path, live_lock)
    with pytest.raises(startup.TtsStartupError) as error:
        run_owned_runner(
            config,
            definition,
            runtime,
            expected_config_sha256=config.config_sha256,
            expected_config_file_fingerprint=config.config_file_identity.fingerprint,
            expected_credential_fingerprint=config.credential_identity.fingerprint,
            policy=_runner_policy(),
        )
    assert error.value.code == "runner_already_active"
    lock_path.unlink()

    stale_lock = dict(live_lock, runner_pid=12345, runner_creation_time_ns=54321)
    startup._write_new_private_json(lock_path, stale_lock)
    startup._write_new_private_json(
        lock_path.with_name(lock_path.name + ".takeover"),
        dict(live_lock, runner_pid=54321, runner_creation_time_ns=12345),
    )
    runtime.on_sleep = lambda current: _request_stop_after_ready(current, config, definition)
    assert (
        run_owned_runner(
            config,
            definition,
            runtime,
            expected_config_sha256=config.config_sha256,
            expected_config_file_fingerprint=config.config_file_identity.fingerprint,
            expected_credential_fingerprint=config.credential_identity.fingerprint,
            policy=_runner_policy(),
        )
        == 0
    )
    assert not lock_path.exists()
    assert not lock_path.with_name(lock_path.name + ".takeover").exists()


def test_uninstall_removes_only_validated_owned_definition_and_stale_locks(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime()
    scheduler = FakeScheduler(ScheduledTaskSnapshot(True, True, False, False, "Ready", True))
    startup._write_private_text(
        config.state_directory / startup.TASK_XML_FILENAME,
        render_scheduled_task_xml(definition),
    )
    stale = {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "config_file_fingerprint": config.config_file_identity.fingerprint,
        "task_fingerprint": definition.fingerprint,
        "runner_pid": 7777,
        "runner_creation_time_ns": 8888,
    }
    startup._write_new_private_json(config.state_directory / startup.LOCK_FILENAME, stale)
    startup._write_new_private_json(
        config.state_directory / f"{startup.LOCK_FILENAME}.takeover",
        dict(stale, runner_pid=7778),
    )

    plan = execute_lifecycle("uninstall", config, definition, scheduler, runtime)

    assert plan.code == "uninstall"
    assert scheduler.calls == ["query", "disable", "uninstall"]
    assert not (config.state_directory / startup.TASK_XML_FILENAME).exists()
    assert not (config.state_directory / startup.LOCK_FILENAME).exists()
    assert not (config.state_directory / f"{startup.LOCK_FILENAME}.takeover").exists()

    second = execute_lifecycle("uninstall", config, definition, scheduler, runtime)
    assert second.code == "already_uninstalled"
    assert second.changed is False
    assert scheduler.calls == ["query", "disable", "uninstall", "query"]


def test_uninstall_recovers_owned_artifacts_after_task_already_absent(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime()
    scheduler = FakeScheduler(ScheduledTaskSnapshot(False))
    xml_path = config.state_directory / startup.TASK_XML_FILENAME
    startup._write_private_text(xml_path, render_scheduled_task_xml(definition))

    plan = execute_lifecycle("uninstall", config, definition, scheduler, runtime)

    assert plan.code == "remove_owned_artifacts"
    assert scheduler.calls == ["query"]
    assert not xml_path.exists()


def test_start_refuses_to_remove_foreign_stop_request(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    stop_path = config.state_directory / startup.STOP_FILENAME
    startup._write_new_private_json(stop_path, {"foreign": True})

    with pytest.raises(startup.TtsStartupError) as error:
        startup._remove_owned_state_artifacts(
            config,
            definition,
            allow_runtime_state=True,
        )

    assert error.value.code == "stop_request_invalid"
    assert stop_path.exists()


def test_stop_request_rejects_boolean_schema_and_out_of_range_pid(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runner = FakeRuntime().runner
    payload = {
        "schema_version": True,
        "config_sha256": config.config_sha256,
        "config_file_fingerprint": config.config_file_identity.fingerprint,
        "task_fingerprint": definition.fingerprint,
        "runner_pid": startup.MAX_WINDOWS_PID + 1,
        "runner_creation_time_ns": runner.creation_time_ns,
    }
    stop_path = config.state_directory / startup.STOP_FILENAME
    startup._write_new_private_json(stop_path, payload)

    assert startup._stop_requested(config, definition, runner) is False
    with pytest.raises(startup.TtsStartupError) as error:
        startup._remove_owned_state_artifacts(config, definition, allow_runtime_state=True)
    assert error.value.code == "stop_request_invalid"


def test_write_stop_request_rejects_equal_looking_invalid_control_types(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    definition = build_task_definition(config)
    runtime = FakeRuntime()
    startup._publish_state(config, definition, runtime.runner, None, "starting", 0, False)
    invalid = {
        "schema_version": True,
        "config_sha256": config.config_sha256,
        "config_file_fingerprint": config.config_file_identity.fingerprint,
        "task_fingerprint": definition.fingerprint,
        "runner_pid": float(runtime.runner.pid),
        "runner_creation_time_ns": float(runtime.runner.creation_time_ns),
    }
    startup._write_new_private_json(config.state_directory / startup.STOP_FILENAME, invalid)

    with pytest.raises(startup.TtsStartupError) as error:
        startup._write_stop_request(config, definition)

    assert error.value.code == "stop_request_invalid"


def test_public_status_and_cli_error_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_path = tmp_path / "private-token-path"
    private_path.write_text(TOKEN, encoding="ascii")
    config_path = tmp_path / "bad-config.json"
    config_path.write_text("{}", encoding="utf-8")

    result = startup.main(
        ["status", "--config", str(config_path)],
        runtime=FakeRuntime(),
    )

    captured = capsys.readouterr()
    assert result == 1
    assert json.loads(captured.err) == {"error": {"code": "config_invalid"}}
    assert TOKEN not in captured.err
    assert str(private_path) not in captured.err
    status = LifecycleStatus("ready", True, True, True, True, True, True, True, True, True, True)
    rendered = json.dumps(plan_lifecycle("status", status).to_public_dict())
    assert TOKEN not in rendered
    assert str(private_path) not in rendered


def test_scheduled_task_backend_uses_fixed_script_and_shell_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    definition = build_task_definition(_config(tmp_path))
    observed: dict[str, object] = {}

    def fake_run(command: Sequence[str], **kwargs: object) -> Any:
        observed.update(command=list(command), kwargs=kwargs)
        return type("Completed", (), {"returncode": 0, "stdout": b'{"present":false}'})()

    monkeypatch.setattr(startup.sys, "platform", "win32")
    monkeypatch.setattr(startup.subprocess, "run", fake_run)
    backend = startup.PowerShellScheduledTaskBackend(tmp_path / "fixed.ps1")

    snapshot = backend.query(definition)

    assert snapshot.present is False
    assert observed["kwargs"]["shell"] is False  # type: ignore[index]
    command = observed["command"]
    assert isinstance(command, list)
    assert "-File" in command
    assert "-Command" not in command
    assert TOKEN not in " ".join(command)


def test_windows_child_output_is_discarded_and_event_log_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    class FakePopen:
        pid = 4319

        def __init__(self, _command: Sequence[str], **kwargs: object) -> None:
            observed.update(kwargs)

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            return None

    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    log_path = log_directory / startup.EVENT_LOG_FILENAME
    log_path.write_bytes(b"x" * startup.MAX_EVENT_LOG_BYTES)
    monkeypatch.setattr(startup.subprocess, "Popen", FakePopen)
    runtime = startup.WindowsRuntimeBackend()
    identity = ProcessIdentity(4319, 999, EXECUTABLE_PATH, EXECUTABLE_HASH)
    monkeypatch.setattr(runtime, "inspect_process", lambda _pid: identity)

    runtime.spawn_child(
        (EXECUTABLE_PATH, "private-command"),
        expected_executable=Path(EXECUTABLE_PATH),
        expected_executable_sha256=EXECUTABLE_HASH,
        working_directory=tmp_path,
        log_directory=log_directory,
    )

    assert observed["stdout"] is startup.subprocess.DEVNULL
    assert observed["stderr"] is startup.subprocess.DEVNULL
    assert log_path.stat().st_size <= startup.MAX_EVENT_LOG_BYTES
    assert json.loads(log_path.read_text(encoding="ascii")) == {"event": "child_spawn"}
    assert TOKEN not in log_path.read_text(encoding="ascii")
    assert "private-command" not in log_path.read_text(encoding="ascii")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows executable share lock")
def test_windows_child_executable_is_hash_checked_while_replacement_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "pythonw.exe"
    executable.write_bytes(b"exact executable")
    executable_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    replacement_succeeded = False

    class FakePopen:
        pid = 4320

        def __init__(self, _command: Sequence[str], **_kwargs: object) -> None:
            nonlocal replacement_succeeded
            try:
                executable.write_bytes(b"replacement")
                replacement_succeeded = True
            except OSError:
                replacement_succeeded = False

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            return None

    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    monkeypatch.setattr(startup.subprocess, "Popen", FakePopen)
    runtime = startup.WindowsRuntimeBackend()
    identity = ProcessIdentity(4320, 1000, str(executable), executable_hash)
    monkeypatch.setattr(runtime, "inspect_process", lambda _pid: identity)

    runtime.spawn_child(
        (str(executable), "-c", "pass"),
        expected_executable=executable.resolve(),
        expected_executable_sha256=executable_hash,
        working_directory=tmp_path,
        log_directory=log_directory,
    )

    assert replacement_succeeded is False
    assert executable.read_bytes() == b"exact executable"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows reparse-point identity")
def test_windows_execution_lock_rejects_reparse_path(tmp_path: Path) -> None:
    target = tmp_path / "target.exe"
    target.write_bytes(b"target")
    link = tmp_path / "pythonw.exe"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlink unavailable: {exc}")

    with pytest.raises(startup.TtsStartupError) as error:
        with startup._windows_execution_lock(link):
            raise AssertionError("reparse path must not be yielded")

    assert error.value.code == "executable_lock_failed"


def test_windows_listener_detection_includes_wildcard_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = ProcessIdentity(4444, 5555, EXECUTABLE_PATH, EXECUTABLE_HASH)
    row = startup._TcpOwnerRow(2, "0.0.0.0", 43190, "0.0.0.0", 0, process.pid)
    monkeypatch.setattr(startup, "_windows_tcp_owner_rows", lambda _table: (row,))

    listeners = startup._windows_tcp_listeners(
        "127.0.0.1", 43190, lambda pid: process if pid == process.pid else None
    )

    assert listeners == (ListenerIdentity(process.pid, process.creation_time_ns),)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows TCP ownership proof")
@pytest.mark.parametrize("expected_owner", [True, False])
def test_windows_health_sends_bearer_only_to_exact_established_owner(
    expected_owner: bool,
) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()[:2]
    received: list[bytes] = []

    def serve() -> None:
        connection, _ = listener.accept()
        connection.settimeout(2)
        try:
            try:
                data = connection.recv(4096)
            except TimeoutError:
                data = b""
            received.append(data)
            if data:
                body = b'{"status":"ready","engines":["kokoro","silero"]}'
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
        finally:
            connection.close()
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    runtime = startup.WindowsRuntimeBackend()
    owner = runtime.current_process()
    expected = (
        owner
        if expected_owner
        else ProcessIdentity(
            owner.pid,
            owner.creation_time_ns + 1,
            owner.executable_path,
            owner.executable_sha256,
        )
    )

    result = runtime.authenticated_health(
        str(host),
        int(port),
        TOKEN,
        expected_process=expected,
        timeout_seconds=2,
    )
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert result is expected_owner
    if expected_owner:
        assert f"Authorization: Bearer {TOKEN}".encode() in received[0]
    else:
        assert TOKEN.encode() not in received[0]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows TCP ownership proof")
def test_windows_health_probe_is_bounded_by_a_total_deadline() -> None:
    body = b'{"status":"ready","engines":["kokoro","silero"]}'
    response = (
        b"HTTP/1.1 200 OK\r\nContent-Length: "
        + str(len(body)).encode("ascii")
        + b"\r\nConnection: close\r\n\r\n"
        + body
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()[:2]

    def serve() -> None:
        connection, _ = listener.accept()
        try:
            connection.settimeout(2)
            try:
                connection.recv(4096)
            except OSError:
                pass
            for byte in response:
                try:
                    connection.sendall(bytes((byte,)))
                except OSError:
                    break
                time.sleep(0.04)
        finally:
            connection.close()
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    runtime = startup.WindowsRuntimeBackend()

    started = time.monotonic()
    result = runtime.authenticated_health(
        str(host),
        int(port),
        TOKEN,
        expected_process=runtime.current_process(),
        timeout_seconds=0.1,
    )
    elapsed = time.monotonic() - started
    thread.join(timeout=5)

    assert not thread.is_alive()
    # A drip-feeding peer keeps every individual socket read under the nominal
    # timeout, so only a total deadline can bound the probe: expiry is an
    # unsuccessful probe within the bound instead of a multi-second stall.
    assert result is False
    assert elapsed < 1.0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows TCP ownership proof")
def test_windows_health_probe_counts_http_protocol_errors_as_unhealthy() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()[:2]

    def serve() -> None:
        connection, _ = listener.accept()
        try:
            connection.settimeout(2)
            try:
                connection.recv(4096)
            except OSError:
                pass
            connection.sendall(b"broken status\r\n\r\n")
        finally:
            connection.close()
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    runtime = startup.WindowsRuntimeBackend()

    result = runtime.authenticated_health(
        str(host),
        int(port),
        TOKEN,
        expected_process=runtime.current_process(),
        timeout_seconds=2,
    )
    thread.join(timeout=3)

    assert not thread.is_alive()
    # A malformed status line raises http.client.BadStatusLine; the probe must
    # count it as unhealthy instead of letting the protocol error escape.
    assert result is False


def test_cli_forwards_tts_startup_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_tools.cli as cli

    captured: list[str] = []
    monkeypatch.setattr(cli, "run_tts_startup", lambda args: captured.extend(args) or 0)

    result = cli.main(["tts-startup", "status", "--config", "fixture.json"])

    assert result == 0
    assert captured == ["status", "--config", "fixture.json"]


def test_task_script_has_no_generic_kill_or_network_mutation() -> None:
    source = Path(startup.__file__).with_name("tts_startup_task.ps1").read_text(encoding="utf-8")
    folded = source.casefold()
    assert "taskkill" not in folded
    assert "stop-process" not in folded
    assert "get-nettcpconnection" not in folded
    assert "tailscale" not in folded
    assert "restart-service" not in folded
    assert "-command" not in folded
    assert 'logontype -eq "interactive"' in folded
    exact_block = folded.split("$exact = (", 1)[1].split("\n    )", 1)[0]
    assert "$task.settings.enabled" not in exact_block
    assert "enabled = [bool]$task.settings.enabled" in folded
    assert "$task.settings.allowhardterminate -eq $true" in folded
    assert "stopscheduledtask" not in folded
    assert "stop-scheduledtask" in folded
    for setting in (
        "disallowstartifonbatteries",
        "stopifgoingonbatteries",
        "allowhardterminate",
        "runonlyifnetworkavailable",
    ):
        assert setting in folded


def test_task_script_strips_xml_declaration_before_registration() -> None:
    source = Path(startup.__file__).with_name("tts_startup_task.ps1").read_text(encoding="utf-8")
    folded = source.casefold()
    # Task Scheduler's XML parser rejects a declared encoding when the document is handed
    # over as an already-decoded string ("unable to switch the encoding"), so the exact
    # persisted XML must have its declaration stripped before Register-ScheduledTask.
    # The producer writes BOM-less UTF-8, and the helper runs under Windows PowerShell
    # 5.1 where Get-Content without -Encoding decodes BOM-less input with the legacy
    # ANSI code page, so the file must instead be read strictly as UTF-8 with a
    # throwing UTF8Encoding (which also tolerates a BOM), and only a syntactically
    # valid XML declaration (whitespace after the "xml" target) may be stripped.
    assert "-xml (get-content -raw -literalpath $xmlpath)" not in folded
    assert "get-content" not in folded
    assert (
        "[system.io.file]::readalltext($xmlpath,"
        " [system.text.utf8encoding]::new($false, $true))" in folded
    )
    assert '-replace "^\\s*<\\?xml\\s+[^>]*\\?>\\s*", \"\"' in folded
    assert "-xml $xmltext" in folded


def _windows_powershell_51() -> Path:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    return Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"


def _task_script_xml_read_line() -> str:
    source = Path(startup.__file__).with_name("tts_startup_task.ps1").read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*(\$xmlText = .+)$", source)
    assert match is not None, "task script XML read line not found"
    return match.group(1)


def _run_task_script_xml_read(xml_path: Path, tmp_path: Path) -> tuple[int, str]:
    # Executes the exact XML read/strip statement copied out of the shipped task
    # script inside real Windows PowerShell 5.1 (the interpreter the scheduler
    # action launches) so the encoding behavior is exercised, not just asserted.
    harness = tmp_path / "xml-read-harness.ps1"
    harness.write_text(
        "\n".join(
            (
                "$ErrorActionPreference = 'Stop'",
                f"$XmlPath = '{xml_path}'",
                _task_script_xml_read_line(),
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8",
                "Write-Output $xmlText",
            )
        )
        + "\n",
        encoding="utf-8-sig",
    )
    completed = subprocess.run(
        [
            str(_windows_powershell_51()),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        shell=False,
        timeout=60,
    )
    return (
        completed.returncode,
        completed.stdout.decode("utf-8-sig", errors="replace").strip(),
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell 5.1 task helper")
def test_task_script_decodes_bomless_utf8_task_xml_before_registration(tmp_path: Path) -> None:
    directory = tmp_path / "ünïcode-örder"
    directory.mkdir()
    xml_path = directory / "scheduled-task.xml"
    arguments = "-Pfad mit Ünicode öffnen"
    document = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f"<Task><Arguments>{arguments}</Arguments></Task>"
    )
    xml_path.write_bytes(document.encode("utf-8"))

    code, output = _run_task_script_xml_read(xml_path, tmp_path)

    assert code == 0
    assert output == f"<Task><Arguments>{arguments}</Arguments></Task>"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell 5.1 task helper")
def test_task_script_strip_keeps_leading_xml_stylesheet_pi(tmp_path: Path) -> None:
    stylesheet = "<?xml-stylesheet type='text/xsl' href='style.xsl'?>"
    body = "<Task><Arguments>launch</Arguments></Task>"
    declaration = "<?xml version='1.0' encoding='utf-8'?>\n"
    # A processing instruction whose target merely starts with "xml" must never
    # be treated as an XML declaration, whether it leads the document or
    # follows a real declaration.
    leading = tmp_path / "leading.xml"
    leading.write_bytes(f"{stylesheet}\n{body}".encode())
    after_declaration = tmp_path / "after-declaration.xml"
    after_declaration.write_bytes(f"{declaration}{stylesheet}\n{body}".encode())

    code_leading, output_leading = _run_task_script_xml_read(leading, tmp_path)
    code_after, output_after = _run_task_script_xml_read(after_declaration, tmp_path)

    assert code_leading == 0
    assert output_leading == f"{stylesheet}\n{body}"
    assert code_after == 0
    assert output_after == f"{stylesheet}\n{body}"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell 5.1 task helper")
def test_task_script_rejects_invalid_utf8_task_xml_before_registration(tmp_path: Path) -> None:
    xml_path = tmp_path / "scheduled-task.xml"
    xml_path.write_bytes(
        b"<?xml version='1.0'?>\n<Task><Arguments>bad \xff\xfe bytes</Arguments></Task>"
    )

    code, output = _run_task_script_xml_read(xml_path, tmp_path)

    assert code != 0
    assert output == ""


def test_task_script_compares_owner_sids_after_scheduler_name_normalization() -> None:
    source = Path(startup.__file__).with_name("tts_startup_task.ps1").read_text(encoding="utf-8")
    folded = source.casefold()
    # Register-ScheduledTask resolves the SID principal/trigger identities in the XML into
    # account names, so the exact-definition check must resolve them back to SIDs instead
    # of comparing the stored strings directly.
    assert "function resolve-ownersid" in folded
    assert "(resolve-ownersid ([string]$task.principal.userid)) -eq $ownersid" in folded
    assert "(resolve-ownersid ([string]$trigger[0].userid)) -eq $ownersid" in folded
    assert "$task.principal.userid -eq $ownersid" not in folded
    assert "$trigger[0].userid -eq $ownersid" not in folded


def test_file_identity_normalizes_windows_volume_serial_across_python_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Live Task 319 evidence: os.stat().st_dev on Windows is the volume serial
    # number, reported by Python 3.11 as the classic 32-bit serial
    # (BY_HANDLE_FILE_INFORMATION.dwVolumeSerialNumber, e.g. 0x9ab16d67 for
    # volume C:) and by Python 3.13 as the full 64-bit serial
    # (FILE_ID_INFO.VolumeSerialNumber, e.g. 0x6a9ab1a49ab16d67 for the same
    # volume). Without normalization, install-time pins computed by one build
    # never match runner-computed pins under the other build.
    monkeypatch.setattr(sys, "platform", "win32")
    classic_serial = 0x9AB16D67
    full_serial = 0x6A9AB1A49AB16D67
    inode = 0x30000006D27A3
    metadata_python_3_11 = os.stat_result(
        (0, inode, classic_serial, 1, 0, 0, 156044, 0, 1, 1)
    )
    metadata_python_3_13 = os.stat_result((0, inode, full_serial, 1, 0, 0, 156044, 0, 1, 1))

    identity_python_3_11 = startup._file_identity(metadata_python_3_11)
    identity_python_3_13 = startup._file_identity(metadata_python_3_13)

    assert identity_python_3_11 == identity_python_3_13
    assert identity_python_3_11.fingerprint == identity_python_3_13.fingerprint
    assert identity_python_3_11.device == classic_serial

    monkeypatch.setattr(sys, "platform", "linux")
    assert startup._file_identity(metadata_python_3_13).device == full_serial


_RAW_STAT_CODE = (
    "import json, os, sys; st = os.stat(sys.argv[1]); "
    "print(json.dumps({'device': st.st_dev, 'inode': st.st_ino, "
    "'size': st.st_size, 'mtime_ns': st.st_mtime_ns}))"
)


def _discovered_interpreters() -> list[str]:
    candidates = [sys.executable]
    uv_python_root = Path.home() / "AppData" / "Roaming" / "uv" / "python"
    if uv_python_root.is_dir():
        candidates.extend(
            str(path) for path in sorted(uv_python_root.glob("cpython-*/python.exe"))
        )
    discovered: list[str] = []
    for candidate in candidates:
        resolved = str(Path(candidate).resolve())
        if resolved not in discovered and os.path.isfile(resolved):
            discovered.append(resolved)
    return discovered


@pytest.mark.skipif(sys.platform != "win32", reason="Windows volume serial normalization")
def test_file_identity_fingerprint_is_stable_across_discovered_interpreters(
    tmp_path: Path,
) -> None:
    # Stats one pinned file through every discoverable CPython build (the repo
    # venv plus any uv-managed interpreters) and proves the normalized identity
    # is identical across them, reproducing the 3.11-vs-3.13 live drift.
    pinned = tmp_path / "pinned-runtime.exe"
    pinned.write_bytes(b"pinned image")
    fingerprints: dict[str, str] = {}
    for interpreter in _discovered_interpreters():
        probe = subprocess.run(
            [interpreter, "-c", "import sys; print(sys.version_info[:2])"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
            timeout=60,
        )
        if probe.returncode != 0:
            continue
        try:
            version = ast.literal_eval(probe.stdout.decode("ascii", errors="replace"))
        except (SyntaxError, ValueError):
            continue
        if not isinstance(version, tuple) or len(version) < 2:
            continue
        major, minor = version[0], version[1]
        if (major, minor) < (3, 11):
            continue
        raw = subprocess.run(
            [interpreter, "-c", _RAW_STAT_CODE, str(pinned)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
            timeout=60,
        )
        if raw.returncode != 0:
            continue
        payload = json.loads(raw.stdout.decode("utf-8"))
        identity = FileIdentity(
            device=startup._stable_device(int(payload["device"])),
            inode=int(payload["inode"]),
            size=int(payload["size"]),
            mtime_ns=int(payload["mtime_ns"]),
        )
        fingerprints[f"{major}.{minor}"] = identity.fingerprint

    assert fingerprints
    assert len(set(fingerprints.values())) == 1, fingerprints


def test_cli_unexpected_error_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "private-config.json"
    config_path.write_text(TOKEN, encoding="ascii")
    monkeypatch.setattr(
        startup,
        "load_startup_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError(TOKEN)),
    )

    result = startup.main(["status", "--config", str(config_path)], runtime=FakeRuntime())

    captured = capsys.readouterr()
    assert result == 1
    assert json.loads(captured.err) == {"error": {"code": "internal_error"}}
    assert TOKEN not in captured.err
    assert str(config_path) not in captured.err
