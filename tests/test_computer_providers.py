from __future__ import annotations

import ctypes
import io
import subprocess
import sys
import threading
import time
import venv
from pathlib import Path

import pytest

from agent_tools.computer import providers
from agent_tools.computer.models import ComputerError, available_section
from agent_tools.computer.providers import ProviderSpec


class FakeBackend:
    def identity_session(self):
        return available_section(
            host="host",
            session_id=1,
            interactive_desktop_access=True,
            lock_state="unlocked",
            lock_ui_foreground=False,
            desktop_name="Default",
        )

    def system_info(self):
        return available_section(build=1)

    def display_info(self):
        return available_section(monitor_count=1)

    def focused_window(self):
        return available_section(active=False, window=None)

    def list_windows(self):
        return available_section(count=0, truncated=False, items=[])

    def background_processes(self):
        return available_section(count=0, truncated=False, items=[])

    def probe(self):
        return {"available": True, "capabilities": []}


def test_provider_failures_and_timeouts_are_isolated() -> None:
    blocker = threading.Event()

    def fail() -> dict:
        raise ComputerError("provider_failed", "failure")

    started = time.monotonic()
    result = providers._run_providers(
        [
            ProviderSpec("ready", 0.1, lambda: available_section(value=1)),
            ProviderSpec("failed", 0.1, fail),
            ProviderSpec("slow", 0.02, lambda: blocker.wait(1.0) or available_section()),
        ]
    )

    assert time.monotonic() - started < 0.5
    assert result["ready"] == {"available": True, "value": 1}
    assert result["failed"]["error_code"] == "provider_failed"
    assert result["slow"]["error_code"] == "slow_timeout"


def test_collect_info_adds_background_only_when_requested(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(providers, "_media_provider", lambda: available_section(active=False))
    monkeypatch.setattr(providers, "_network_provider", lambda: available_section(connected=True))
    monkeypatch.setattr(
        providers,
        "_readiness_provider",
        lambda _backend: available_section(action_lock="locked"),
    )

    default = providers.collect_info(background=False, backend=backend)
    with_background = providers.collect_info(background=True, backend=backend)

    required = {
        "identity",
        "clock",
        "system",
        "display",
        "focus",
        "visible_apps",
        "media",
        "network",
        "readiness",
    }
    assert set(default) == required
    assert set(with_background) == required | {"background"}
    assert all("available" in section for section in with_background.values())


def test_readiness_provider_deadline_includes_winapp_cleanup_slack(monkeypatch) -> None:
    captured: list[ProviderSpec] = []

    def collect(specs: list[ProviderSpec]) -> dict:
        captured.extend(specs)
        return {}

    monkeypatch.setattr(providers, "_run_providers", collect)

    providers.collect_info(background=False, backend=FakeBackend())

    readiness = next(spec for spec in captured if spec.name == "readiness")
    assert readiness.timeout_seconds > providers.WINAPP_PROBE_MAX_SECONDS


def test_missing_media_backend_is_normal_unavailable_data(monkeypatch) -> None:
    monkeypatch.setattr(providers.sys, "platform", "win32")
    monkeypatch.setattr(providers.importlib.util, "find_spec", lambda _name: None)

    assert providers._media_provider() == {
        "available": False,
        "error_code": "media_backend_unavailable",
        "active": False,
    }


def test_readiness_reports_dynamic_winapp_capability(monkeypatch) -> None:
    class FakeWinAppAdapter:
        def __init__(self, _backend) -> None:
            pass

        def probe(self):
            return {
                "available": True,
                "status": "ready",
                "version": "0.6.0",
                "capabilities": ["inspect", "read", "scroll-areas"],
            }

    monkeypatch.setattr(providers, "WinAppAdapter", FakeWinAppAdapter)

    section = providers._readiness_provider(FakeBackend())

    assert section["read_only"] is False
    assert section["action_lock"].endswith("computer-mutation.v1")
    assert section["actions_available"] is True
    assert section["desktop_usable"] is True
    assert section["action_blocked_reason"] is None
    assert section["physical_input"] is False
    assert section["backends"]["uia_winapp"]["version"] == "0.6.0"
    assert section["capabilities"] == [
        "info",
        "windows",
        "focused",
        "screenshot",
        "capabilities",
        "inspect",
        "read",
        "scroll-areas",
        "invoke",
        "set-value",
        "scroll",
        "focus",
        "resize",
        "notify",
    ]


def test_readiness_fails_closed_for_foreground_lock_ui(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(
        backend,
        "identity_session",
        lambda: available_section(
            session_id=3,
            interactive_desktop_access=True,
            lock_state="locked",
            lock_ui_foreground=True,
            desktop_name="Default",
        ),
    )
    monkeypatch.setattr(
        providers,
        "WinAppAdapter",
        lambda _backend: type("Probe", (), {"probe": lambda self: {"available": True}})(),
    )

    section = providers._readiness_provider(backend)

    assert section["desktop_usable"] is False
    assert section["actions_available"] is False
    assert section["action_blocked_reason"] == "secure_desktop_unavailable"
    assert section["lock_state"] == "locked"
    assert section["lock_ui_foreground"] is True


def test_readiness_fails_closed_when_owner_session_is_unresolved(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(
        backend,
        "identity_session",
        lambda: available_section(
            session_id=None,
            interactive_desktop_access=True,
            lock_state="unlocked",
            lock_ui_foreground=False,
            desktop_name="Default",
        ),
    )
    monkeypatch.setattr(
        providers,
        "WinAppAdapter",
        lambda _backend: type("Probe", (), {"probe": lambda self: {"available": True}})(),
    )

    section = providers._readiness_provider(backend)

    assert section["desktop_usable"] is False
    assert section["actions_available"] is False
    assert section["action_blocked_reason"] == "secure_desktop_unavailable"


def test_external_output_is_stopped_at_streaming_limit() -> None:
    result = providers._run_external(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; sys.stdout.write('x' * 70000); sys.stdout.flush()",
        ],
        2.0,
    )

    assert result.error_code == "external_output_too_large"
    assert result.stdout == ""


def test_external_reader_path_does_not_spawn_threads(monkeypatch) -> None:
    monkeypatch.setattr(
        providers.threading,
        "Thread",
        lambda *args, **kwargs: pytest.fail("external reader thread was created"),
    )

    result = providers._run_external(
        [sys.executable, "-c", "print('bounded')"], 2.0
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "bounded"


def test_external_kill_failure_still_attempts_bounded_reap() -> None:
    class FakeProcess:
        wait_called = False

        def kill(self) -> None:
            raise OSError("termination denied")

        def wait(self, timeout: float) -> None:
            self.wait_called = True
            raise subprocess.TimeoutExpired("fixture", timeout)

        def poll(self) -> None:
            return None

    process = FakeProcess()

    reaped = providers._terminate_external_process(
        process, time.monotonic() + 0.01  # type: ignore[arg-type]
    )

    assert reaped is False
    assert process.wait_called is True


def test_tailscale_timeout_is_unavailable_not_stopped(monkeypatch) -> None:
    monkeypatch.setattr(providers, "_find_tailscale", lambda: Path("tailscale.exe"))
    monkeypatch.setattr(
        providers,
        "_run_external",
        lambda _args, _timeout: providers.ExternalResult(
            None, "", "external_command_timeout"
        ),
    )

    result = providers._tailscale_details()

    assert result == {
        "available": False,
        "error_code": "external_command_timeout",
        "present": True,
        "status": "unknown",
        "addresses": [],
    }


def test_wifi_parser_selects_connected_interface(monkeypatch) -> None:
    response = """
Name : Wi-Fi A
State : connected
SSID : Owner Network
Signal : 88%

Name : Wi-Fi B
State : disconnected
SSID :
Signal : 0%
"""
    monkeypatch.setattr(
        providers, "_system_executable", lambda *_parts: Path("netsh.exe")
    )
    monkeypatch.setattr(
        providers,
        "_run_external",
        lambda _args, _timeout: providers.ExternalResult(0, response),
    )

    result = providers._wifi_details()

    assert result["available"] is True
    assert result["connected"] is True
    assert result["ssid"] == "Owner Network"
    assert result["signal_percent"] == 88


def test_localized_wifi_response_degrades_instead_of_claiming_disconnected(
    monkeypatch,
) -> None:
    response = "Имя : Wi-Fi\nСостояние : подключено\nСигнал : 88%\n"
    monkeypatch.setattr(
        providers, "_system_executable", lambda *_parts: Path("netsh.exe")
    )
    monkeypatch.setattr(
        providers,
        "_run_external",
        lambda _args, _timeout: providers.ExternalResult(0, response),
    )

    result = providers._wifi_details()

    assert result["available"] is False
    assert result["error_code"] == "wifi_response_unrecognized"


def test_nonzero_wifi_query_degrades_instead_of_claiming_disconnected(monkeypatch) -> None:
    monkeypatch.setattr(
        providers, "_system_executable", lambda *_parts: Path("netsh.exe")
    )
    monkeypatch.setattr(
        providers,
        "_run_external",
        lambda _args, _timeout: providers.ExternalResult(1, "Access is denied."),
    )

    result = providers._wifi_details()

    assert result["available"] is False
    assert result["error_code"] == "wifi_query_failed"
    assert result["connected"] is False


def test_system_executable_does_not_search_caller_directory(
    monkeypatch, tmp_path: Path
) -> None:
    hostile = tmp_path / "powershell.exe"
    hostile.write_bytes(b"hostile")
    windows = tmp_path / "Windows"
    trusted = windows / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    trusted.parent.mkdir(parents=True)
    trusted.write_bytes(b"trusted")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(providers, "_windows_directory", lambda: windows)

    resolved = providers._system_executable(
        "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
    )

    assert resolved == trusted
    assert resolved != hostile


def test_network_internal_deadlines_fit_inside_provider_timeout(monkeypatch) -> None:
    timeouts: list[float] = []
    monkeypatch.setattr(
        providers, "_system_executable", lambda *_parts: Path("system-tool.exe")
    )
    monkeypatch.setattr(providers, "_find_tailscale", lambda: Path("tailscale.exe"))

    def timed_out(_args: list[str], timeout_seconds: float) -> providers.ExternalResult:
        timeouts.append(timeout_seconds)
        return providers.ExternalResult(None, "", "external_command_timeout")

    monkeypatch.setattr(providers, "_run_external", timed_out)

    result = providers._network_provider()

    assert result["available"] is True
    assert len(timeouts) == 3
    assert sum(timeouts) <= providers.NETWORK_COMMAND_BUDGET_SECONDS
    assert max(timeouts) <= 3.25


def test_network_stops_launching_tools_after_shared_deadline(monkeypatch) -> None:
    clock = [100.0]
    calls: list[list[str]] = []
    monkeypatch.setattr(providers.sys, "platform", "win32")
    monkeypatch.setattr(providers.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        providers, "_system_executable", lambda *_parts: Path("system-tool.exe")
    )
    monkeypatch.setattr(providers, "_find_tailscale", lambda: Path("tailscale.exe"))
    monkeypatch.setattr(providers, "_primary_address", lambda: None)

    def consume_budget(
        args: list[str], _timeout_seconds: float
    ) -> providers.ExternalResult:
        calls.append(args)
        clock[0] += providers.NETWORK_COMMAND_BUDGET_SECONDS + 0.1
        return providers.ExternalResult(None, "", "external_command_timeout")

    monkeypatch.setattr(providers, "_run_external", consume_budget)

    result = providers._network_provider()

    assert len(calls) == 1
    assert result["wifi"]["error_code"] == "network_deadline_exceeded"
    assert result["tailscale"]["error_code"] == "network_deadline_exceeded"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process termination probe")
def test_external_timeout_does_not_leave_child_running(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    code = (
        "import os,time,pathlib;"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )

    result = providers._run_external([sys.executable, "-c", code], 0.25)

    assert result.error_code == "external_command_timeout"
    pid = int(pid_file.read_text(encoding="utf-8"))
    assert _windows_process_has_exited(pid)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process termination probe")
def test_external_timeout_reaps_descendant_tree_within_cleanup_budget(tmp_path: Path) -> None:
    descendant_pid_file = tmp_path / "descendant.pid"
    descendant_code = (
        "import os,time,pathlib;"
        f"pathlib.Path({str(descendant_pid_file)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )
    launcher_code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{descendant_code!r}]);"
        "time.sleep(30)"
    )
    started = time.monotonic()

    result = providers._run_external([sys.executable, "-c", launcher_code], 0.5)

    elapsed = time.monotonic() - started
    assert result.error_code == "external_command_timeout"
    descendant_pid = int(descendant_pid_file.read_text(encoding="utf-8"))
    try:
        assert _windows_process_has_exited(descendant_pid)
        assert elapsed < 0.5 + providers.EXTERNAL_CLEANUP_BUDGET_SECONDS
    finally:
        _terminate_task_owned_windows_process(descendant_pid)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process termination probe")
def test_external_timeout_reaps_venv_launcher_and_its_descendant(tmp_path: Path) -> None:
    venv_directory = tmp_path / "fixture-venv"
    venv.EnvBuilder(with_pip=False).create(venv_directory)
    venv_python = venv_directory / "Scripts" / "python.exe"
    payload_pid_file = tmp_path / "venv-payload.pid"
    descendant_pid_file = tmp_path / "venv-descendant.pid"
    descendant_code = (
        "import os,time,pathlib;"
        f"pathlib.Path({str(descendant_pid_file)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )
    payload_code = (
        "import os,pathlib,subprocess,sys,time;"
        f"pathlib.Path({str(payload_pid_file)!r}).write_text(str(os.getpid()));"
        f"subprocess.Popen([sys.executable,'-c',{descendant_code!r}]);"
        "time.sleep(30)"
    )

    result = providers._run_external([str(venv_python), "-c", payload_code], 0.5)

    assert result.error_code == "external_command_timeout"
    payload_pid = int(payload_pid_file.read_text(encoding="utf-8"))
    descendant_pid = int(descendant_pid_file.read_text(encoding="utf-8"))
    try:
        assert _windows_process_has_exited(payload_pid)
        assert _windows_process_has_exited(descendant_pid)
    finally:
        _terminate_task_owned_windows_process(payload_pid)
        _terminate_task_owned_windows_process(descendant_pid)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process termination probe")
def test_external_timeout_does_not_terminate_unrelated_process() -> None:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    try:
        result = providers._run_external(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            0.25,
        )

        assert result.error_code == "external_command_timeout"
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_external_job_accepts_already_exited_termination_race() -> None:
    kernel32 = _FakeJobKernel(active_counts=[1, 0], terminate_result=False)
    job = providers._WindowsProcessJob(handle=42, kernel32=kernel32)

    assert job.terminate_and_wait(time.monotonic() + 0.1) is True
    assert kernel32.terminate_calls == 1


def test_external_job_cleanup_return_is_bounded_by_deadline() -> None:
    kernel32 = _FakeJobKernel(active_counts=[1], terminate_result=True)
    job = providers._WindowsProcessJob(handle=42, kernel32=kernel32)
    started = time.monotonic()

    assert job.terminate_and_wait(started + 0.02) is False
    assert time.monotonic() - started < 0.2


def test_external_descendant_cleanup_finds_late_child_of_exited_held_parent(
    monkeypatch,
) -> None:
    kernel32 = _FakeLateDescendantKernel()
    tree = providers._WindowsDescendantTree(
        kernel32=kernel32,
        root_pid=1,
        root_created=100,
        handles={
            2: providers._ExactDescendantHandle(
                pid=2,
                parent_pid=1,
                created=120,
                depth=1,
                handle=200,
            )
        },
    )
    snapshots = iter([{}, {3: 2}, {3: 2}, {3: 2}])
    monkeypatch.setattr(providers, "_system_filetime", lambda _kernel32: 200)
    monkeypatch.setattr(
        providers,
        "_process_parent_snapshot",
        lambda _kernel32: next(snapshots, {3: 2}),
    )
    monkeypatch.setattr(
        providers,
        "_process_created_filetime",
        lambda _kernel32, handle: 150 if handle == 300 else None,
    )

    assert tree.terminate_and_wait(time.monotonic() + 0.2) is True
    assert kernel32.opened_pids == [3]
    assert kernel32.terminated_handles == [300]
    assert tree.handles[3].parent_pid == 2
    assert tree.handles[3].depth == 2


def test_external_descendant_exit_during_terminate_handle_open_is_success(
    monkeypatch,
) -> None:
    kernel32 = _FakeExitedDuringOpenKernel()
    tree = providers._WindowsDescendantTree(
        kernel32=kernel32,
        root_pid=1,
        root_created=100,
        handles={},
    )
    monkeypatch.setattr(providers, "_system_filetime", lambda _kernel32: 200)
    monkeypatch.setattr(
        providers,
        "_process_parent_snapshot",
        lambda _kernel32: {2: 1},
    )
    monkeypatch.setattr(
        providers,
        "_process_created_filetime",
        lambda _kernel32, handle: 150 if handle == 300 else None,
    )

    assert tree.terminate_and_wait(time.monotonic() + 0.2) is True
    assert kernel32.requested_access == [
        providers.PROCESS_TERMINATE
        | providers.PROCESS_QUERY_LIMITED_INFORMATION
        | providers.SYNCHRONIZE,
        providers.PROCESS_QUERY_LIMITED_INFORMATION | providers.SYNCHRONIZE,
    ]
    assert tree.handles[2].can_terminate is False
    assert kernel32.terminated_handles == []


def test_external_job_launch_is_suspended_until_assignment_and_resume(monkeypatch) -> None:
    events: list[str] = []
    process = _FakeLaunchProcess()
    job = _FakeLaunchJob(events)

    def fake_popen(*_args, **kwargs):
        assert kwargs["creationflags"] & getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
        events.append("popen_suspended")
        return process

    monkeypatch.setattr(providers.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        providers,
        "_resume_external_process",
        lambda received: events.append("resume") or received is process,
    )
    monkeypatch.setattr(providers, "_read_pipe_available", lambda _stream: (b"", True))

    result = providers._run_external_in_job(
        ["fixture.exe"],
        1.0,
        job,  # type: ignore[arg-type]
    )

    assert result.returncode == 0
    assert result.error_code is None
    assert events == ["popen_suspended", "assign", "resume", "terminate_and_wait"]


def test_external_job_finalizer_retries_close_and_reports_permanent_failure() -> None:
    recovered = _FakeFinalizerJob(close_results=[False, True])
    failed = _FakeFinalizerJob(close_results=[False, False])

    assert providers._finalize_external_process_job(recovered) is True  # type: ignore[arg-type]
    assert recovered.close_calls == 2
    assert providers._finalize_external_process_job(failed) is False  # type: ignore[arg-type]
    assert failed.close_calls == 2


def test_external_job_finalizer_runs_on_unexpected_unwind(monkeypatch) -> None:
    job = _FakeFinalizerJob(close_results=[True])
    monkeypatch.setattr(providers, "_create_external_process_job", lambda: job)
    monkeypatch.setattr(
        providers,
        "_run_external_in_job",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        providers._run_external(["fixture.exe"], 1.0)

    assert job.terminate_calls == 1
    assert job.close_calls == 1


class _FakeJobKernel:
    def __init__(self, *, active_counts: list[int], terminate_result: bool) -> None:
        self.active_counts = active_counts
        self.terminate_result = terminate_result
        self.terminate_calls = 0

    def QueryInformationJobObject(
        self,
        _handle,
        _information_class,
        information,
        _information_size,
        _return_length,
    ) -> bool:
        active = self.active_counts.pop(0) if len(self.active_counts) > 1 else self.active_counts[0]
        accounting = ctypes.cast(
            information,
            ctypes.POINTER(providers._JobObjectBasicAccountingInformation),
        ).contents
        accounting.ActiveProcesses = active
        return True

    def TerminateJobObject(self, _handle, _exit_code) -> bool:
        self.terminate_calls += 1
        return self.terminate_result

    def CloseHandle(self, _handle) -> bool:
        return True


class _FakeLateDescendantKernel:
    def __init__(self) -> None:
        self.opened_pids: list[int] = []
        self.terminated_handles: list[int] = []

    def OpenProcess(self, _access, _inherit, pid) -> int:
        self.opened_pids.append(pid)
        return 300

    def WaitForSingleObject(self, handle, _timeout) -> int:
        value = int(handle.value)
        if value == 200 or value in self.terminated_handles:
            return 0
        return 258

    def TerminateProcess(self, handle, _exit_code) -> bool:
        self.terminated_handles.append(int(handle.value))
        return True

    def CloseHandle(self, _handle) -> bool:
        return True


class _FakeExitedDuringOpenKernel:
    def __init__(self) -> None:
        self.requested_access: list[int] = []
        self.terminated_handles: list[int] = []

    def OpenProcess(self, access, _inherit, _pid) -> int:
        self.requested_access.append(access)
        return 0 if access & providers.PROCESS_TERMINATE else 300

    def WaitForSingleObject(self, _handle, _timeout) -> int:
        return 0

    def TerminateProcess(self, handle, _exit_code) -> bool:
        self.terminated_handles.append(int(handle.value))
        return True

    def CloseHandle(self, _handle) -> bool:
        return True


class _FakeLaunchProcess:
    def __init__(self) -> None:
        self.stdout = io.BytesIO()
        self.stderr = io.BytesIO()
        self.returncode = 0
        self.pid = 123
        self._handle = 456

    def poll(self) -> int:
        return 0

    def wait(self, timeout: float) -> int:
        assert timeout > 0
        return 0


class _FakeLaunchJob:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def assign(self, process: _FakeLaunchProcess) -> bool:
        assert process._handle == 456
        self.events.append("assign")
        return True

    def terminate_and_wait(self, deadline: float) -> bool:
        assert deadline > time.monotonic()
        self.events.append("terminate_and_wait")
        return True


class _FakeFinalizerJob:
    def __init__(self, *, close_results: list[bool]) -> None:
        self.close_results = close_results
        self.terminate_calls = 0
        self.close_calls = 0

    def terminate_and_wait(self, deadline: float) -> bool:
        assert deadline > time.monotonic()
        self.terminate_calls += 1
        return True

    def close(self) -> bool:
        self.close_calls += 1
        return self.close_results.pop(0)


def _windows_process_has_exited(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel32.OpenProcess(0x00100000, False, pid)
    if not handle:
        return True
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0
    finally:
        kernel32.CloseHandle(handle)


def _terminate_task_owned_windows_process(pid: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    kernel32.TerminateProcess.restype = ctypes.c_int
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel32.OpenProcess(0x00100001, False, pid)
    if not handle:
        return
    try:
        if kernel32.WaitForSingleObject(handle, 0) != 0:
            kernel32.TerminateProcess(handle, 1)
            kernel32.WaitForSingleObject(handle, 5000)
    finally:
        kernel32.CloseHandle(handle)
