from __future__ import annotations

import ctypes
import gc
import inspect
import io
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_tools import cli as root_cli
from agent_tools.computer import process_io, screenshot, win32_backend
from agent_tools.computer.capture_records import CaptureRecordStore
from agent_tools.computer.models import Bounds, ComputerError, WindowIdentity
from agent_tools.computer.win32_backend import (
    Win32Backend,
    _classify_lock_state,
    _rectangle_covered_by_monitors,
)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only integration")


def test_preloaded_process_stdin_is_unbuffered_and_delete_on_close() -> None:
    payload = b"task297-preloaded-input" * 8_192
    stream = process_io.preloaded_process_stdin(payload)
    path = Path(stream.name)
    try:
        assert isinstance(stream.file, io.FileIO)
        assert stream.read() == payload
        assert path.exists()
    finally:
        stream.close()

    assert not path.exists()


def test_pipe_cleanup_closes_crt_owner_once_after_cancellation_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    process.wait(timeout=5)
    assert isinstance(process.stdout, io.FileIO)
    stdout = process.stdout
    original_fd = stdout.fileno()
    monkeypatch.setattr(
        process_io,
        "_cancel_stream_io",
        lambda _stream: "CancelIoEx:5",
    )

    assert process_io.close_stream_os_enforced(stdout) == "CancelIoEx:5"
    assert stdout.closed is True
    assert "CloseHandle" not in inspect.getsource(
        process_io.close_stream_os_enforced
    )

    process.stdout = None
    replacement_fd: int | None = None
    opened_fds: list[int] = []
    for index in range(64):
        candidate_fd = os.open(
            tmp_path / f"replacement-handle-{index}.bin",
            os.O_CREAT | os.O_RDWR | os.O_TRUNC | getattr(os, "O_BINARY", 0),
            0o600,
        )
        opened_fds.append(candidate_fd)
        if candidate_fd == original_fd:
            replacement_fd = candidate_fd
            break
    try:
        assert replacement_fd is not None
        del stdout
        gc.collect()
        assert os.write(replacement_fd, b"still-owned") == len(b"still-owned")
        os.lseek(replacement_fd, 0, os.SEEK_SET)
        assert os.read(replacement_fd, 11) == b"still-owned"
    finally:
        for opened_fd in opened_fds:
            os.close(opened_fd)
        if process.stderr is not None:
            process_io.close_stream_os_enforced(process.stderr)


class _DpiCall:
    def __init__(self, implementation) -> None:
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


def _fake_dpi_user32(*, failure: str | None = None):
    previous = 1234
    state = {"current": previous, "setter_calls": 0, "getter_calls": 0, "compare_calls": 0}

    def value(handle) -> int:
        return int(getattr(handle, "value", handle))

    def setter(handle):
        state["setter_calls"] += 1
        if state["setter_calls"] == 2 and failure == "restore_setter_false":
            return 0
        old = state["current"]
        state["current"] = value(handle)
        return old

    def getter():
        state["getter_calls"] += 1
        if state["getter_calls"] == 1 and failure == "active_getter_raises":
            raise OSError("fixture active getter failure")
        if state["getter_calls"] == 2 and failure == "restore_getter_raises":
            raise OSError("fixture restore getter failure")
        return state["current"]

    def comparer(left, right):
        state["compare_calls"] += 1
        if state["compare_calls"] == 1 and failure == "active_comparer_raises":
            raise OSError("fixture active comparer failure")
        if state["compare_calls"] == 2 and failure == "restore_comparer_raises":
            raise OSError("fixture restore comparer failure")
        if state["compare_calls"] == 2 and failure == "restore_comparer_false":
            return False
        return value(left) == value(right)

    return (
        SimpleNamespace(
            SetThreadDpiAwarenessContext=_DpiCall(setter),
            GetThreadDpiAwarenessContext=_DpiCall(getter),
            AreDpiAwarenessContextsEqual=_DpiCall(comparer),
        ),
        state,
        previous,
    )


@pytest.mark.parametrize("failure", ["active_getter_raises", "active_comparer_raises"])
def test_native_pixel_context_restores_after_active_verification_failure(
    monkeypatch,
    failure: str,
) -> None:
    user32, state, previous = _fake_dpi_user32(failure=failure)
    monkeypatch.setattr(win32_backend, "_windll", lambda _name: user32)

    with pytest.raises(ComputerError) as raised:
        with Win32Backend().native_pixel_context():
            pytest.fail("context body must not run")

    assert raised.value.code == "dpi_awareness_unavailable"
    assert state["setter_calls"] == 2
    assert state["current"] == previous


@pytest.mark.parametrize(
    "failure",
    [
        "restore_setter_false",
        "restore_getter_raises",
        "restore_comparer_raises",
        "restore_comparer_false",
    ],
)
def test_native_pixel_context_reports_every_restoration_failure(
    monkeypatch,
    failure: str,
) -> None:
    user32, _state, _previous = _fake_dpi_user32(failure=failure)
    monkeypatch.setattr(win32_backend, "_windll", lambda _name: user32)

    with pytest.raises(ComputerError) as raised:
        with Win32Backend().native_pixel_context():
            pass

    assert raised.value.code == "dpi_awareness_restore_failed"


def test_native_pixel_context_restores_before_propagating_body_failure(monkeypatch) -> None:
    user32, state, previous = _fake_dpi_user32()
    monkeypatch.setattr(win32_backend, "_windll", lambda _name: user32)

    with pytest.raises(RuntimeError, match="fixture body failure"):
        with Win32Backend().native_pixel_context():
            raise RuntimeError("fixture body failure")

    assert state["setter_calls"] == 2
    assert state["current"] == previous


def _action_identity(
    *, process: str | None = "fixture.exe", process_started: int | None = 999
) -> WindowIdentity:
    return WindowIdentity(
        hwnd=123,
        pid=456,
        thread_id=789,
        process=process,
        title="Task-owned fixture",
        class_name="FixtureClass",
        bounds=Bounds(10, 20, 400, 300),
        process_started=process_started,
    )


def test_action_security_rejects_locked_protected_and_higher_integrity_targets(
    monkeypatch,
) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)
    monkeypatch.setattr(
        backend,
        "_desktop_state",
        lambda _session_id=None: (False, "Winlogon", "locked", False),
    )
    with pytest.raises(ComputerError) as locked:
        backend.assert_action_allowed(identity)
    assert locked.value.code == "secure_desktop_unavailable"

    monkeypatch.setattr(
        backend,
        "_desktop_state",
        lambda _session_id=None: (True, "Default", "locked", True),
    )
    with pytest.raises(ComputerError) as lock_ui:
        backend.assert_action_allowed(identity)
    assert lock_ui.value.code == "secure_desktop_unavailable"
    assert lock_ui.value.details == {
        "lock_state": "locked",
        "lock_ui_foreground": True,
        "interactive_desktop_access": True,
        "desktop_name": "Default",
    }

    monkeypatch.setattr(
        backend,
        "_desktop_state",
        lambda _session_id=None: (True, "Default", "unlocked", False),
    )
    with pytest.raises(ComputerError) as protected:
        backend.assert_action_allowed(_action_identity(process="LogonUI.exe"))
    assert protected.value.code == "protected_target"

    monkeypatch.setattr(backend, "_security_state", lambda: (False, "medium"))
    monkeypatch.setattr(backend, "_process_security_state", lambda _pid: (True, "high"))
    with pytest.raises(ComputerError) as integrity:
        backend.assert_action_allowed(identity)
    assert integrity.value.code == "target_integrity_denied"


@pytest.mark.parametrize("process_name", ["LockApp.exe", "LogonUI.exe"])
def test_foreground_lock_ui_uses_same_session_process_identity_not_title(
    monkeypatch,
    process_name: str,
) -> None:
    backend = Win32Backend()
    title_reads: list[int] = []
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=lambda: 777,
            GetWindowText=lambda hwnd: title_reads.append(hwnd) or "volatile title",
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (12, 456),
        ),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "_process_session_id", lambda _pid: 3)
    monkeypatch.setattr(backend, "_process_snapshot", lambda: {456: process_name})

    assert backend._foreground_lock_ui(3) is True
    assert title_reads == []


def test_foreground_non_lock_ui_and_cross_session_do_not_report_locked(monkeypatch) -> None:
    backend = Win32Backend()
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(GetForegroundWindow=lambda: 777),
        win32process=SimpleNamespace(GetWindowThreadProcessId=lambda _hwnd: (12, 456)),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "_process_snapshot", lambda: {456: "explorer.exe"})
    monkeypatch.setattr(backend, "_process_session_id", lambda _pid: 3)
    assert backend._foreground_lock_ui(3) is False

    monkeypatch.setattr(backend, "_process_snapshot", lambda: {456: "LockApp.exe"})
    assert backend._foreground_lock_ui(4) is False


def test_foreground_lock_ui_fails_closed_when_foreground_is_unresolved(monkeypatch) -> None:
    backend = Win32Backend()
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(GetForegroundWindow=lambda: 0),
        win32process=SimpleNamespace(),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)

    assert backend._foreground_lock_ui(3) is None


@pytest.mark.parametrize(
    ("foregrounds", "pids"),
    [
        ([777, 888], [456]),
        ([777, 777], [456, 654]),
    ],
)
def test_foreground_lock_ui_fails_closed_when_identity_changes_during_probe(
    monkeypatch,
    foregrounds: list[int],
    pids: list[int],
) -> None:
    backend = Win32Backend()
    foreground = iter(foregrounds)
    process_ids = iter(pids)
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(GetForegroundWindow=lambda: next(foreground)),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (12, next(process_ids)),
        ),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "_process_session_id", lambda _pid: 3)
    monkeypatch.setattr(backend, "_process_snapshot", lambda: {456: "explorer.exe"})

    assert backend._foreground_lock_ui(3) is None


def test_foreground_lock_ui_fails_closed_for_pywin32_disappearance_error(
    monkeypatch,
) -> None:
    class FixturePyWin32Error(Exception):
        pass

    backend = Win32Backend()
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(GetForegroundWindow=lambda: 777),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (_ for _ in ()).throw(
                FixturePyWin32Error("window disappeared")
            ),
        ),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(
        win32_backend,
        "_is_pywin32_error",
        lambda error: isinstance(error, FixturePyWin32Error),
    )

    assert backend._foreground_lock_ui(3) is None


def test_foreground_lock_ui_requires_resolved_owner_session(monkeypatch) -> None:
    backend = Win32Backend()
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(GetForegroundWindow=lambda: 777),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (12, 456),
        ),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)

    assert backend._foreground_lock_ui(None) is None


def test_lock_state_is_fail_closed_and_title_independent() -> None:
    assert _classify_lock_state("Default", True) == "locked"
    assert _classify_lock_state("Default", False) == "unlocked"
    assert _classify_lock_state("Default", None) == "unknown"
    assert _classify_lock_state("Winlogon", False) == "locked"


def test_focus_reports_failure_when_foreground_is_stolen(monkeypatch) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    foreground = iter([999, 888])
    attachments: list[tuple[int, int, bool]] = []
    win32gui = SimpleNamespace(
        GetForegroundWindow=lambda: next(foreground),
        IsIconic=lambda _hwnd: False,
        ShowWindow=lambda *_args: True,
        SetForegroundWindow=lambda _hwnd: True,
        error=RuntimeError,
    )
    win32process = SimpleNamespace(
        GetWindowThreadProcessId=lambda _hwnd: (50, 1),
        AttachThreadInput=lambda current, target, attach: attachments.append(
            (current, target, attach)
        ),
        error=RuntimeError,
    )
    modules = SimpleNamespace(
        win32gui=win32gui,
        win32process=win32process,
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: 40),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "assert_action_allowed", lambda expected: expected)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity)

    assert raised.value.code == "focus_postcondition_failed"
    assert raised.value.details["state_before"] == "normal"
    assert raised.value.details["state_after"] == "normal"
    assert raised.value.details["restore_performed"] is False
    assert raised.value.details["restored_to_normal"] is True
    assert raised.value.details["foreground_hwnd"] == 888
    assert raised.value.details["input_attachments"] == {
        "attached_count": 2,
        "detached_count": 2,
        "failed_count": 0,
    }
    assert attachments == [(40, 50, True), (40, 789, True), (40, 789, False), (40, 50, False)]


def test_focus_initial_foreground_observation_failure_is_pre_delivery(monkeypatch) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    deliveries: list[str] = []

    def fail_foreground() -> int:
        raise OSError("fixture initial foreground failure")

    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=fail_foreground,
            IsIconic=lambda _hwnd: False,
            ShowWindow=lambda *_args: deliveries.append("show"),
            SetForegroundWindow=lambda _hwnd: deliveries.append("foreground"),
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (50, 1),
            AttachThreadInput=lambda *_args: deliveries.append("attach"),
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: 40),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "assert_action_allowed", lambda expected: expected)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)
    monkeypatch.setattr(backend, "_is_zoomed", lambda _hwnd: False)

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity)

    assert deliveries == []
    assert raised.value.code == "win32_action_failed"
    assert raised.value.details == {
        "outcome": "failed",
        "method": "win32.GetForegroundWindow",
        "state_before": "normal",
        "postcondition_verified": False,
    }


def test_focus_final_foreground_observation_failure_preserves_delivery_metadata(
    monkeypatch,
) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    foreground_reads = 0
    deliveries: list[object] = []

    def foreground() -> int:
        nonlocal foreground_reads
        foreground_reads += 1
        if foreground_reads == 2:
            raise OSError("fixture final foreground failure")
        return 0

    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=foreground,
            IsIconic=lambda _hwnd: False,
            ShowWindow=lambda _hwnd, command: deliveries.append(command),
            SetForegroundWindow=lambda _hwnd: deliveries.append("foreground"),
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (identity.thread_id, identity.pid),
            AttachThreadInput=lambda *_args: deliveries.append("attach"),
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: identity.thread_id),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "assert_action_allowed", lambda expected: expected)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)
    monkeypatch.setattr(backend, "_is_zoomed", lambda _hwnd: False)

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity)

    assert deliveries == [win32_backend.SW_SHOW, "foreground"]
    assert raised.value.code == "win32_action_failed"
    assert raised.value.details == {
        "outcome": "delivery_only",
        "method": "win32.SetForegroundWindow",
        "state_before": "normal",
        "state_after": "normal",
        "restore_performed": False,
        "restored_to_normal": True,
        "foreground_hwnd": 0,
        "input_attachments": {
            "attached_count": 0,
            "detached_count": 0,
            "failed_count": 0,
        },
        "verification_scope": "immediate",
        "postcondition_verified": False,
    }


def test_focus_identity_drift_after_delivery_is_not_reported_as_rejection(
    monkeypatch,
) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    delivered = False

    def set_foreground(_hwnd):
        nonlocal delivered
        delivered = True

    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=lambda: 999,
            IsIconic=lambda _hwnd: False,
            ShowWindow=lambda *_args: True,
            SetForegroundWindow=set_foreground,
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (50, 1),
            AttachThreadInput=lambda *_args: None,
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: 40),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "assert_action_allowed", lambda expected: expected)

    def revalidate(expected):
        if delivered:
            raise ComputerError("stale_window", "fixture drift after delivery")
        return expected

    monkeypatch.setattr(backend, "revalidate_identity", revalidate)

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity)

    assert raised.value.code == "stale_window"
    assert raised.value.details == {
        "outcome": "delivery_only",
        "method": "win32.SetForegroundWindow",
        "state_before": "normal",
        "state_after": "unavailable",
        "restore_performed": False,
        "restored_to_normal": None,
        "foreground_hwnd": 999,
        "input_attachments": {
            "attached_count": 2,
            "detached_count": 2,
            "failed_count": 0,
        },
        "verification_scope": "immediate",
        "postcondition_verified": False,
    }


def test_focus_rechecks_full_security_before_input_attachment(monkeypatch) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    security_checks = 0
    attachments: list[tuple[int, int, bool]] = []
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=lambda: 999,
            IsIconic=lambda _hwnd: False,
            ShowWindow=lambda *_args: True,
            SetForegroundWindow=lambda _hwnd: True,
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (50, 1),
            AttachThreadInput=lambda *args: attachments.append(args),
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: 40),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)

    def security(expected):
        nonlocal security_checks
        security_checks += 1
        if security_checks == 2:
            raise ComputerError(
                "secure_desktop_unavailable", "fixture desktop transitioned"
            )
        return expected

    monkeypatch.setattr(backend, "assert_action_allowed", security)

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity)

    assert raised.value.code == "secure_desktop_unavailable"
    assert attachments == []


def test_focus_rechecks_identity_after_guard_before_first_win32_delivery(
    monkeypatch,
) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    target_changed = False
    deliveries: list[str] = []
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=lambda: 999,
            IsIconic=lambda _hwnd: False,
            ShowWindow=lambda *_args: deliveries.append("show"),
            SetForegroundWindow=lambda _hwnd: deliveries.append("foreground"),
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (50, 1),
            AttachThreadInput=lambda *_args: deliveries.append("attach"),
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: 40),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)

    def security(expected):
        if target_changed:
            raise ComputerError("stale_window", "fixture changed during guard")
        return expected

    def change_target() -> None:
        nonlocal target_changed
        target_changed = True

    monkeypatch.setattr(backend, "assert_action_allowed", security)

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity, mutation_guard=change_target)

    assert raised.value.code == "stale_window"
    assert deliveries == []


def test_focus_reports_restore_only_when_restore_call_was_attempted(monkeypatch) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    iconic = True
    show_calls: list[int] = []
    foreground = iter([0, identity.hwnd])

    def restore_during_guard() -> None:
        nonlocal iconic
        iconic = False

    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=lambda: next(foreground),
            IsIconic=lambda _hwnd: iconic,
            ShowWindow=lambda _hwnd, command: show_calls.append(command),
            SetForegroundWindow=lambda _hwnd: True,
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (identity.thread_id, identity.pid),
            AttachThreadInput=lambda *_args: None,
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: identity.thread_id),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "assert_action_allowed", lambda expected: expected)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)
    monkeypatch.setattr(backend, "_is_zoomed", lambda _hwnd: False)

    result = backend.focus_window(identity, mutation_guard=restore_during_guard)

    assert show_calls == [win32_backend.SW_SHOW]
    assert result["state_before"] == "minimized"
    assert result["state_after"] == "normal"
    assert result["restore_performed"] is False
    assert result["restored_to_normal"] is True


def test_focus_restore_failure_preserves_known_state_and_restore_metadata(
    monkeypatch,
) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    show_calls: list[int] = []
    clock = iter((100.0, 103.0))
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=lambda: 0,
            IsIconic=lambda _hwnd: True,
            ShowWindow=lambda _hwnd, command: show_calls.append(command),
            SetForegroundWindow=lambda _hwnd: True,
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (identity.thread_id, identity.pid),
            AttachThreadInput=lambda *_args: None,
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: identity.thread_id),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "assert_action_allowed", lambda expected: expected)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)
    monkeypatch.setattr(win32_backend.time, "monotonic", lambda: next(clock))

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity)

    assert raised.value.code == "focus_restore_failed"
    assert raised.value.details == {
        "outcome": "failed",
        "method": "win32.ShowWindow",
        "state_before": "minimized",
        "state_after": "minimized",
        "restore_performed": True,
        "restored_to_normal": False,
        "foreground_hwnd": 0,
        "input_attachments": {
            "attached_count": 0,
            "detached_count": 0,
            "failed_count": 0,
        },
        "verification_scope": "immediate",
        "postcondition_verified": False,
    }
    assert show_calls == [win32_backend.SW_RESTORE]


def test_focus_failure_after_restore_reports_partial_delivery(monkeypatch) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    iconic = True
    show_calls: list[int] = []

    def is_iconic(_hwnd):
        return iconic

    def show_window(_hwnd, command):
        nonlocal iconic
        show_calls.append(command)
        iconic = False
        return True

    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=lambda: 0,
            IsIconic=is_iconic,
            ShowWindow=show_window,
            SetForegroundWindow=lambda _hwnd: True,
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (50, 1),
            AttachThreadInput=lambda *_args: None,
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: identity.thread_id),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)

    def security(expected):
        if show_calls:
            raise ComputerError("stale_window", "fixture drift after restore")
        return expected

    monkeypatch.setattr(backend, "assert_action_allowed", security)

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity)

    assert show_calls
    assert raised.value.code == "stale_window"
    assert raised.value.details == {
        "outcome": "delivery_only",
        "method": "win32.ShowWindow",
        "state_before": "minimized",
        "state_after": "normal",
        "restore_performed": True,
        "restored_to_normal": True,
        "foreground_hwnd": 0,
        "input_attachments": {
            "attached_count": 0,
            "detached_count": 0,
            "failed_count": 0,
        },
        "verification_scope": "immediate",
        "postcondition_verified": False,
    }


def test_focus_primary_delivery_error_survives_detach_cleanup_failure(
    monkeypatch,
) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    iconic = True

    def attach(_current, _target, enabled):
        if not enabled:
            raise RuntimeError("fixture detach failure")

    def show_window(_hwnd, _command):
        nonlocal iconic
        iconic = False
        return True

    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=lambda: 999,
            IsIconic=lambda _hwnd: iconic,
            ShowWindow=show_window,
            SetForegroundWindow=lambda _hwnd: True,
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (50, 1),
            AttachThreadInput=attach,
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: 40),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)

    def security(expected):
        if not iconic:
            raise ComputerError("stale_window", "fixture drift after restore")
        return expected

    monkeypatch.setattr(backend, "assert_action_allowed", security)

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity)

    assert raised.value.code == "stale_window"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["method"] == "win32.ShowWindow"
    assert raised.value.details["warnings"] == ["input_attachment_cleanup_failed"]
    assert raised.value.details["input_attachment_cleanup"]["status"] == "failed"


def test_focus_native_error_survives_detach_cleanup_failure(monkeypatch) -> None:
    backend = Win32Backend()
    identity = _action_identity()

    def attach(_current, _target, enabled):
        if not enabled:
            raise RuntimeError("fixture detach failure")

    def show_window(_hwnd, _command):
        raise OSError("fixture native focus failure")

    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=lambda: 999,
            IsIconic=lambda _hwnd: False,
            ShowWindow=show_window,
            SetForegroundWindow=lambda _hwnd: True,
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (50, 1),
            AttachThreadInput=attach,
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: 40),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "assert_action_allowed", lambda expected: expected)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity)

    assert raised.value.code == "win32_action_failed"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["method"] == "win32.ShowWindow"
    assert raised.value.details["warnings"] == ["input_attachment_cleanup_failed"]
    assert raised.value.details["input_attachment_cleanup"] == {
        "status": "failed",
        "count": 2,
    }


def test_focus_guard_rejection_after_attachment_reports_unresolved_input_mutation(
    monkeypatch,
) -> None:
    backend = Win32Backend()
    identity = _action_identity()
    attachment_calls: list[bool] = []
    show_calls: list[int] = []
    guard_calls = 0

    def attach(_current, _target, enabled):
        attachment_calls.append(enabled)
        if not enabled:
            raise RuntimeError("fixture detach failure")

    def guard() -> None:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 2:
            raise ComputerError("computer_actions_disabled", "fixture stop")

    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetForegroundWindow=lambda: 999,
            IsIconic=lambda _hwnd: False,
            ShowWindow=lambda _hwnd, command: show_calls.append(command),
            SetForegroundWindow=lambda _hwnd: True,
            error=RuntimeError,
        ),
        win32process=SimpleNamespace(
            GetWindowThreadProcessId=lambda _hwnd: (identity.thread_id, identity.pid),
            AttachThreadInput=attach,
            error=RuntimeError,
        ),
        win32api=SimpleNamespace(GetCurrentThreadId=lambda: 40),
    )
    monkeypatch.setattr(backend, "modules", lambda: modules)
    monkeypatch.setattr(backend, "assert_action_allowed", lambda expected: expected)
    monkeypatch.setattr(backend, "revalidate_identity", lambda expected: expected)

    with pytest.raises(ComputerError) as raised:
        backend.focus_window(identity, mutation_guard=guard)

    assert raised.value.code == "computer_actions_disabled"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["method"] == "win32.AttachThreadInput"
    assert raised.value.details["partial_mutation"] == "input_attachment"
    assert raised.value.details["focus_failure"] == {
        "code": "computer_actions_disabled",
        "outcome": "rejected",
        "method": None,
    }
    assert raised.value.details["input_attachment_cleanup"] == {
        "status": "failed",
        "count": 1,
    }
    assert raised.value.details["input_attachments"] == {
        "attached_count": 1,
        "detached_count": 0,
        "failed_count": 0,
    }
    assert attachment_calls == [True, False]
    assert show_calls == []


def test_mutation_identity_requires_process_name_and_creation_time() -> None:
    backend = Win32Backend()

    with pytest.raises(ComputerError) as missing_name:
        backend.revalidate_identity(_action_identity(process=None))
    assert missing_name.value.code == "target_identity_unavailable"

    incomplete = _action_identity(process_started=None)
    with pytest.raises(ComputerError) as missing_started:
        backend.revalidate_identity(incomplete)
    assert missing_started.value.code == "target_identity_unavailable"


def test_monitor_coverage_rejects_virtual_desktop_gaps() -> None:
    adjacent = [Bounds(0, 0, 100, 100), Bounds(100, 0, 100, 100)]
    gapped = [Bounds(0, 0, 100, 100), Bounds(150, 0, 100, 100)]

    assert _rectangle_covered_by_monitors(Bounds(50, 10, 100, 80), adjacent) is True
    assert _rectangle_covered_by_monitors(Bounds(50, 10, 150, 80), gapped) is False


@pytest.fixture
def disposable_window():
    import win32con
    import win32gui
    import win32process

    title = f"AgentToolsReadOnlyTest-{uuid.uuid4().hex}"
    script = Path(__file__).parents[1] / "scripts" / "windows_control_poc.py"
    process = subprocess.Popen(
        [sys.executable, str(script), "_target-process", "--title", title],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    hwnd = None
    actual_pid = 0
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and hwnd is None:
        matches: list[int] = []

        def callback(
            candidate: int,
            _extra: object,
            found: list[int] = matches,
        ) -> bool:
            if win32gui.GetWindowText(candidate) == title:
                found.append(candidate)
            return True

        win32gui.EnumWindows(callback, None)
        if len(matches) > 1:
            process.terminate()
            process.wait(timeout=5)
            pytest.fail("Disposable target title was ambiguous")
        if matches:
            candidate = matches[0]
            _thread_id, pid = win32process.GetWindowThreadProcessId(candidate)
            if win32gui.IsWindowVisible(candidate):
                hwnd = candidate
                actual_pid = int(pid)
                break
        time.sleep(0.05)
    if hwnd is None:
        process.terminate()
        process.wait(timeout=5)
        pytest.fail("Disposable target window did not start")

    try:
        yield {
            "hwnd": hwnd,
            "title": title,
            "process": process,
            "pid": actual_pid,
        }
    finally:
        if win32gui.IsWindow(hwnd):
            _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == actual_pid and win32gui.GetWindowText(hwnd) == title:
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        assert process.poll() is not None
        assert not win32gui.IsWindow(hwnd)


def test_real_windows_and_focused_match_direct_win32_reads() -> None:
    import win32gui
    import win32process

    backend = Win32Backend()
    windows = backend.list_windows()
    assert windows["available"] is True
    assert windows["count"] >= 1
    for item in windows["items"]:
        assert win32gui.IsWindow(item["hwnd"])
        _thread_id, pid = win32process.GetWindowThreadProcessId(item["hwnd"])
        assert item["pid"] == pid
        assert item["title"] == win32gui.GetWindowText(item["hwnd"])[:240]

    for _attempt in range(20):
        before = int(win32gui.GetForegroundWindow() or 0)
        focused = backend.focused_window()
        after = int(win32gui.GetForegroundWindow() or 0)
        if before == after:
            assert focused["available"] is True
            assert focused["active"] is bool(before)
            if before:
                assert focused["window"]["hwnd"] == before
            break
    else:
        pytest.fail("Foreground window did not remain stable for a direct comparison")


def test_background_rows_are_bounded_and_private() -> None:
    section = Win32Backend().background_processes()

    assert section["available"] is True
    assert len(section["items"]) <= 250
    assert section["truncated"] is (section["count"] > 250)
    for item in section["items"]:
        assert set(item) == {"pid", "name", "session_id", "state"}
        assert "\\" not in item["name"]
        assert "/" not in item["name"]


def test_real_cli_screenshot_full_region_cleanup_and_input_invariants(
    disposable_window, tmp_path: Path, capsys
) -> None:
    import win32api
    import win32gui
    from PIL import Image

    hwnd = disposable_window["hwnd"]
    backend = Win32Backend()
    identity = backend.capture_identity(hwnd)

    with pytest.raises(ComputerError) as forced:
        screenshot._capture_full(
            backend,
            identity,
            force_failure_after_selection=True,
        )
    assert forced.value.code == "forced_capture_failure"

    full_output = tmp_path / "full.png"
    foreground_before = int(win32gui.GetForegroundWindow() or 0)
    cursor_before = win32api.GetCursorPos()
    assert root_cli.main(
        [
            "computer",
            "screenshot",
            "--hwnd",
            str(hwnd),
            "--output",
            str(full_output),
            "--json",
        ]
    ) == 0
    cursor_after = win32api.GetCursorPos()
    foreground_after = int(win32gui.GetForegroundWindow() or 0)
    full_payload = json.loads(capsys.readouterr().out)

    assert cursor_after == cursor_before
    assert foreground_after == foreground_before
    result = full_payload["data"]["screenshot"]
    assert result["cleanup_verified"] is True
    assert result["content_non_uniform"] is True
    assert (result["width"], result["height"]) == (
        identity.bounds.width,
        identity.bounds.height,
    )
    with Image.open(full_output) as image:
        assert image.size == (identity.bounds.width, identity.bounds.height)
        assert image.convert("L").getextrema()[0] != image.convert("L").getextrema()[1]

    region_output = tmp_path / "region.png"
    assert root_cli.main(
        [
            "computer",
            "screenshot",
            "--hwnd",
            str(hwnd),
            "--output",
            str(region_output),
            "--region",
            "10,10,200,150",
            "--json",
        ]
    ) == 0
    region_payload = json.loads(capsys.readouterr().out)["data"]["screenshot"]
    assert (region_payload["width"], region_payload["height"]) == (200, 150)
    assert region_payload["content_non_uniform"] is True
    with Image.open(region_output) as image:
        assert image.size == (200, 150)


def test_real_disposable_capture_profiles_metadata_mapping_and_geometry_drift(
    disposable_window, tmp_path: Path, capsys
) -> None:
    import win32gui
    from PIL import Image

    hwnd = disposable_window["hwnd"]
    backend = Win32Backend()
    initial = backend.capture_native_geometry(hwnd)
    win32gui.MoveWindow(
        hwnd,
        initial.identity.bounds.x,
        initial.identity.bounds.y,
        1400,
        900,
        True,
    )
    geometry = backend.capture_native_geometry(hwnd)
    assert geometry.identity.bounds.width == 1400
    assert geometry.identity.bounds.height == 900
    assert geometry.capture_thread_awareness == "per_monitor_aware_v2"
    assert geometry.window_dpi > 0
    assert geometry.system_dpi > 0
    assert geometry.client_bounds.width > 0
    assert geometry.client_bounds.height > 0

    expected_sizes = {
        "compact": (480, 309),
        "standard": (720, 463),
        "native": (1400, 900),
    }
    payloads = {}
    for profile, expected in expected_sizes.items():
        output = tmp_path / f"real-{profile}.png"
        assert root_cli.main(
            [
                "computer",
                "screenshot",
                "--hwnd",
                str(hwnd),
                "--output",
                str(output),
                "--profile",
                profile,
                "--json",
            ]
        ) == 0
        payload = json.loads(capsys.readouterr().out)["data"]["screenshot"]
        payloads[profile] = payload
        assert (payload["width"], payload["height"]) == expected
        assert payload["presentation"]["profile"] == profile
        assert payload["native_pixel_size"] == {"width": 1400, "height": 900}
        assert payload["geometry"]["window_bounds"] == geometry.identity.bounds.to_jsonable()
        assert payload["geometry"]["client_bounds"] == geometry.client_bounds.to_jsonable()
        assert payload["geometry"]["dpi"]["window"] == geometry.window_dpi
        with Image.open(output) as image:
            assert image.size == expected

    standard = payloads["standard"]
    mapped = screenshot.resolve_capture_point(
        capture_id=standard["capture_id"],
        image_x=standard["width"] - 1,
        image_y=standard["height"] - 1,
        backend=backend,
    )
    assert mapped["native_window_point"] == {"x": 1399, "y": 899}
    assert mapped["native_screen_point"] == {
        "x": geometry.identity.bounds.x + 1399,
        "y": geometry.identity.bounds.y + 899,
    }

    win32gui.MoveWindow(
        hwnd,
        geometry.identity.bounds.x + 1,
        geometry.identity.bounds.y,
        1400,
        900,
        True,
    )
    with pytest.raises(ComputerError) as drifted:
        screenshot.resolve_capture_point(
            capture_id=standard["capture_id"],
            image_x=0,
            image_y=0,
            backend=backend,
        )
    assert drifted.value.code == "capture_geometry_changed"


def test_invalid_and_out_of_bounds_targets_fail_closed(
    disposable_window, tmp_path: Path
) -> None:
    backend = Win32Backend()
    with pytest.raises(ComputerError) as invalid:
        screenshot.capture_screenshot(
            hwnd=0,
            output=tmp_path / "invalid.png",
            region=None,
            backend=backend,
        )
    assert invalid.value.code == "invalid_window"

    with pytest.raises(ComputerError) as outside:
        screenshot.capture_screenshot(
            hwnd=disposable_window["hwnd"],
            output=tmp_path / "outside.png",
            region=screenshot.parse_region("0,0,99999,10"),
            backend=backend,
        )
    assert outside.value.code == "region_out_of_bounds"
    assert not (tmp_path / "outside.png").exists()


def test_identity_drift_during_capture_returns_stale_window(
    disposable_window, tmp_path: Path
) -> None:
    import win32con
    import win32gui

    class ClosingBackend(Win32Backend):
        closed = False

        def revalidate(self, expected):
            if not self.closed:
                self.closed = True
                win32gui.PostMessage(expected.hwnd, win32con.WM_CLOSE, 0, 0)
                deadline = time.monotonic() + 3.0
                while win32gui.IsWindow(expected.hwnd) and time.monotonic() < deadline:
                    time.sleep(0.02)
            return super().revalidate(expected)

    with pytest.raises(ComputerError) as stale:
        screenshot.capture_screenshot(
            hwnd=disposable_window["hwnd"],
            output=tmp_path / "stale.png",
            region=None,
            backend=ClosingBackend(),
        )

    assert stale.value.code == "stale_window"
    assert not (tmp_path / "stale.png").exists()


def test_unresponsive_target_capture_times_out_and_cleans_worker(
    disposable_window, tmp_path: Path, capsys
) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    ntdll.NtSuspendProcess.argtypes = (ctypes.c_void_p,)
    ntdll.NtSuspendProcess.restype = ctypes.c_long
    ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)
    ntdll.NtResumeProcess.restype = ctypes.c_long
    process_handle = kernel32.OpenProcess(0x0800, False, disposable_window["pid"])
    assert process_handle
    output = tmp_path / "hung.png"
    try:
        assert ntdll.NtSuspendProcess(process_handle) == 0
        started = time.monotonic()
        result = root_cli.main(
            [
                "computer",
                "screenshot",
                "--hwnd",
                str(disposable_window["hwnd"]),
                "--output",
                str(output),
                "--json",
            ]
        )
        elapsed = time.monotonic() - started
    finally:
        ntdll.NtResumeProcess(process_handle)
        kernel32.CloseHandle(process_handle)

    payload = json.loads(capsys.readouterr().out)
    assert result == 1
    assert payload["error"]["code"] == "capture_timeout"
    assert elapsed < screenshot.CAPTURE_TIMEOUT_SECONDS + 3.0
    assert not output.exists()
    assert not list(tmp_path.glob(".agent-tools-capture-worker.*"))


def test_capture_worker_ignores_hostile_caller_package(
    disposable_window, tmp_path: Path, monkeypatch, capsys
) -> None:
    hostile_package = tmp_path / "agent_tools" / "computer"
    hostile_package.mkdir(parents=True)
    (tmp_path / "agent_tools" / "__init__.py").write_text("", encoding="utf-8")
    (hostile_package / "__init__.py").write_text("", encoding="utf-8")
    marker = tmp_path / "hostile-worker-ran"
    (hostile_package / "capture_worker.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "trusted.png"

    result = root_cli.main(
        [
            "computer",
            "screenshot",
            "--hwnd",
            str(disposable_window["hwnd"]),
            "--output",
            str(output),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["ok"] is True
    assert output.is_file()
    assert not marker.exists()


def test_real_animated_title_capture_is_strong_identity_safe(tmp_path: Path) -> None:
    import win32con
    import win32gui
    import win32process

    title = f"Task297-AnimatedTitle-{uuid.uuid4().hex}"
    script = Path(__file__).parents[1] / "scripts" / "windows_control_poc.py"
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "_target-process",
            "--title",
            title,
            "--title-spinner-ms",
            "20",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    hwnd: int | None = None
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and hwnd is None:
        matches: list[int] = []

        def callback(
            candidate: int,
            _extra: object,
            found: list[int] = matches,
        ) -> bool:
            _thread_id, pid = win32process.GetWindowThreadProcessId(candidate)
            if (
                pid == process.pid
                and win32gui.IsWindowVisible(candidate)
                and win32gui.GetWindowText(candidate).startswith(title)
            ):
                found.append(candidate)
            return True

        win32gui.EnumWindows(callback, None)
        if len(matches) == 1:
            hwnd = matches[0]
            break
        time.sleep(0.02)
    if hwnd is None:
        process.terminate()
        process.wait(timeout=5)
        pytest.fail("Disposable animated-title window did not start")

    backend = Win32Backend()
    identity_before = backend.capture_identity(hwnd)
    foreground_before = int(win32gui.GetForegroundWindow() or 0)
    try:
        result = screenshot.capture_screenshot(
            hwnd=hwnd,
            output=tmp_path / "animated-title.png",
            region=None,
            backend=backend,
            record_store=CaptureRecordStore(tmp_path / "animated-records"),
        )
        identity_after = backend.capture_identity(hwnd)
        foreground_after_capture = int(win32gui.GetForegroundWindow() or 0)
    finally:
        if win32gui.IsWindow(hwnd):
            _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == process.pid and win32gui.GetWindowText(hwnd).startswith(title):
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        process.wait(timeout=5)

    assert result["strong_identity_verified"] is True
    assert result["identity_mismatch_fields"] == []
    assert result["title_changed_during_operation"] is True
    assert result["title_before_hmac_sha256"] != result["title_after_hmac_sha256"]
    assert backend.screenshot_identity_mismatches(
        identity_before,
        identity_after,
    ) == []
    assert foreground_after_capture == foreground_before


def test_real_geometry_drift_recaptures_exact_final_provenance(
    disposable_window,
    tmp_path: Path,
) -> None:
    import win32con
    import win32gui

    class MovingBackend(Win32Backend):
        moved = False

        def revalidate(self, expected):
            if not self.moved:
                left, top, _right, _bottom = win32gui.GetWindowRect(expected.hwnd)
                win32gui.SetWindowPos(
                    expected.hwnd,
                    0,
                    left + 37,
                    top + 19,
                    0,
                    0,
                    win32con.SWP_NOSIZE
                    | win32con.SWP_NOZORDER
                    | win32con.SWP_NOACTIVATE,
                )
                self.moved = True
            return super().revalidate(expected)

    hwnd = disposable_window["hwnd"]
    foreground_before = int(win32gui.GetForegroundWindow() or 0)
    store = CaptureRecordStore(tmp_path / "geometry-records")
    result = screenshot.capture_screenshot(
        hwnd=hwnd,
        output=tmp_path / "geometry-recapture.png",
        region=None,
        backend=MovingBackend(),
        record_store=store,
    )
    actual = Win32Backend().capture_native_geometry(hwnd)
    mapped = screenshot.resolve_capture_point(
        capture_id=result["capture_id"],
        image_x=0,
        image_y=0,
        backend=Win32Backend(),
        record_store=store,
    )

    assert result["geometry_recapture_count"] == 1
    assert result["geometry"]["window_bounds"] == actual.identity.bounds.to_jsonable()
    assert mapped["native_screen_point"] == {
        "x": actual.identity.bounds.x,
        "y": actual.identity.bounds.y,
    }
    assert int(win32gui.GetForegroundWindow() or 0) == foreground_before


def test_zzz_task297_leaves_no_fixture_windows_or_helpers() -> None:
    import win32com.client
    import win32gui
    import win32process

    task_windows: list[dict[str, object]] = []

    def collect_window(hwnd: int, _extra: object) -> bool:
        title = win32gui.GetWindowText(hwnd)
        if "task297" not in title.casefold():
            return True
        _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
        task_windows.append(
            {
                "hwnd": int(hwnd),
                "pid": int(pid),
                "title": title,
                "class": win32gui.GetClassName(hwnd),
            }
        )
        return True

    win32gui.EnumWindows(collect_window, None)
    task_helpers: list[dict[str, object]] = []
    service = win32com.client.GetObject("winmgmts:")
    for process in service.ExecQuery(
        "SELECT ProcessId, Name, CommandLine FROM Win32_Process"
    ):
        command_line = str(process.CommandLine or "").casefold()
        is_task_fixture = (
            "scripts/windows_control_poc.py" in command_line
            and "_target-process" in command_line
            and "task297" in command_line
        )
        is_task_scratch = ".tmp/task297" in command_line
        is_capture_worker = "agent_tools.computer.capture_worker" in command_line
        if is_task_fixture or is_task_scratch or is_capture_worker:
            task_helpers.append(
                {
                    "pid": int(process.ProcessId),
                    "name": str(process.Name),
                }
            )

    assert task_windows == []
    assert task_helpers == []
