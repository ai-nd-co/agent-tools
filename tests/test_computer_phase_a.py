from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_tools import cli as root_cli
from agent_tools.computer import uia_winapp, win32_backend
from agent_tools.computer.element_refs import ElementReferenceStore
from agent_tools.computer.models import Bounds, ComputerError, WindowIdentity
from agent_tools.computer.uia_winapp import WinAppAdapter, WinAppBinding
from agent_tools.computer.win32_backend import (
    SW_MAXIMIZE,
    SW_MINIMIZE,
    SW_RESTORE,
    Win32Backend,
)


def _identity(**changes: object) -> WindowIdentity:
    identity = WindowIdentity(
        hwnd=123,
        pid=456,
        thread_id=789,
        process="Taskmgr.exe",
        title="Task Manager",
        class_name="TaskManagerWindow",
        bounds=Bounds(10, 20, 800, 600),
        process_started=999,
        session_id=1,
        desktop_name="Default",
    )
    return replace(identity, **changes)


def _element() -> dict[str, Any]:
    ancestor = "sha256:" + "a" * 24
    return {
        "element": "btn-restore-a1b2",
        "provider": "winapp",
        "automation_id": "Restore",
        "control_type": "Button",
        "class": "Button",
        "bounds": {"x": 20, "y": 40, "width": 80, "height": 24},
        "enabled": True,
        "read_only": False,
        "password": False,
        "locator": {
            "provider": "winapp",
            "strategy": "selector_v1",
            "selector": "btn-restore-a1b2",
            "runtime_id": "42,1",
            "automation_id": "Restore",
            "ancestor_signature": ancestor,
        },
        "hierarchy": {"ancestor_signature": ancestor},
    }


class MutableClock:
    def __init__(self, value: int = 1_000_000) -> None:
        self.value = value

    def now_unix_ms(self) -> int:
        return self.value


@pytest.mark.skipif(sys.platform != "win32", reason="Windows owner-private store")
def test_element_reference_is_title_tolerant_but_strongly_bound_and_expires(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    store = ElementReferenceStore(tmp_path / "refs", clock=clock)
    issued = store.issue(identity=_identity(), element=_element())

    changed_title = _identity(title="Task Manager (2)")
    resolved = store.resolve(issued["element_ref"], identity=changed_title)
    assert resolved["element"]["runtime_id"] == "42,1"

    with pytest.raises(ComputerError) as strict:
        store.resolve(
            issued["element_ref"],
            identity=changed_title,
            require_title_match=True,
        )
    assert strict.value.code == "stale_window"
    assert strict.value.details["identity_mismatch_fields"] == ["title"]

    with pytest.raises(ComputerError) as drift:
        store.resolve(
            issued["element_ref"],
            identity=_identity(session_id=2),
        )
    assert drift.value.code == "stale_window"
    assert drift.value.details["identity_mismatch_fields"] == ["session_id"]

    clock.value += 15_000
    with pytest.raises(ComputerError) as expired:
        store.resolve(issued["element_ref"], identity=_identity())
    assert expired.value.code == "element_ref_expired"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows owner-private store")
def test_element_reference_tamper_is_rejected(tmp_path: Path) -> None:
    store = ElementReferenceStore(tmp_path / "refs", clock=MutableClock())
    issued = store.issue(identity=_identity(), element=_element())
    path = store.root / f"{issued['element_ref']}.json"
    envelope = json.loads(path.read_text(encoding="ascii"))
    envelope["payload"]["element"]["automation_id"] = "Terminate"
    path.write_text(json.dumps(envelope), encoding="ascii")

    with pytest.raises(ComputerError) as altered:
        store.resolve(issued["element_ref"], identity=_identity())
    assert altered.value.code == "element_ref_altered"


@pytest.mark.parametrize(
    "missing",
    ["automation_id", "control_type", "class", "ancestor_signature"],
)
def test_winapp_reference_requires_complete_stable_identity(
    tmp_path: Path,
    missing: str,
) -> None:
    element = _element()
    if missing == "automation_id":
        element["automation_id"] = None
        element["locator"]["automation_id"] = None
    elif missing == "ancestor_signature":
        element["hierarchy"]["ancestor_signature"] = None
        element["locator"]["ancestor_signature"] = None
    else:
        element[missing] = None

    with pytest.raises(ComputerError) as raised:
        ElementReferenceStore(tmp_path / "refs").issue(
            identity=_identity(),
            element=element,
        )

    assert raised.value.code == "element_ref_unavailable"
    assert not (tmp_path / "refs").exists()


@pytest.mark.parametrize(
    ("field", "length"),
    [
        ("provider", 65),
        ("selector", 241),
        ("automation_id", 161),
        ("control_type", 65),
        ("class", 129),
        ("ancestor_signature", 97),
    ],
)
def test_winapp_reference_rejects_overlong_identity_without_truncation(
    tmp_path: Path,
    field: str,
    length: int,
) -> None:
    element = _element()
    value = "x" * length
    if field == "provider":
        element["provider"] = value
        element["locator"]["provider"] = value
    elif field == "selector":
        element["element"] = value
        element["locator"][field] = value
    elif field == "automation_id":
        element[field] = value
        element["locator"][field] = value
    elif field == "ancestor_signature":
        element["hierarchy"][field] = value
        element["locator"][field] = value
    else:
        element[field] = value

    with pytest.raises(ComputerError) as raised:
        ElementReferenceStore(tmp_path / "refs").issue(
            identity=_identity(),
            element=element,
        )

    assert raised.value.code == "element_ref_unavailable"
    assert not (tmp_path / "refs").exists()


def test_title_change_is_descriptive_but_strict_mode_can_reject(monkeypatch) -> None:
    backend = Win32Backend()
    expected = _identity()
    changed = _identity(title="Task Manager - Processes")
    monkeypatch.setattr(backend, "capture_identity", lambda _hwnd: changed)

    assert backend.revalidate_identity(expected) == changed
    monkeypatch.setattr(backend, "capture_identity", lambda _hwnd: _identity(title=""))
    assert backend.revalidate_identity(expected).title == ""
    monkeypatch.setattr(backend, "capture_identity", lambda _hwnd: changed)
    with pytest.raises(ComputerError) as strict:
        backend.revalidate_identity(expected, require_title_match=True)
    assert strict.value.code == "stale_window"
    assert strict.value.details == {
        "identity_mismatch_fields": ["title"],
        "title_changed": True,
        "title_change_was_authoritative": True,
    }


class StateBackend(Win32Backend):
    def __init__(self, state: str) -> None:
        self.state = state
        self.identity = _identity()
        self.commands: list[int] = []
        self.modules_value = SimpleNamespace(
            win32gui=SimpleNamespace(ShowWindow=self._show_window),
        )

    def modules(self):
        return self.modules_value

    def _show_window(self, _hwnd: int, command: int) -> None:
        self.commands.append(command)
        self.state = {
            SW_RESTORE: "normal",
            SW_MINIMIZE: "minimized",
            SW_MAXIMIZE: "maximized",
        }[command]

    def window_state(self, _hwnd: int) -> str:
        return self.state

    def _assert_action_target_allowed(
        self,
        expected: WindowIdentity,
        require_title_match: bool,
    ) -> WindowIdentity:
        return expected

    def _revalidate_action_target(
        self,
        expected: WindowIdentity,
        require_title_match: bool,
    ) -> WindowIdentity:
        return expected


class ResizeBackend(Win32Backend):
    def __init__(self, state: str, layouts: list[list[tuple[int, int, int, int]]]) -> None:
        self.state = state
        self.identity = _identity()
        self.layouts = layouts
        self.layout_calls = 0
        self.commands: list[int] = []
        self.moves: list[tuple[int, int, int, int]] = []
        self.modules_value = SimpleNamespace(
            win32gui=SimpleNamespace(
                ShowWindow=self._show_window,
                IsIconic=lambda _hwnd: self.state == "minimized",
                MoveWindow=self._move_window,
            ),
            win32api=SimpleNamespace(EnumDisplayMonitors=self._enum_monitors),
        )

    def modules(self):
        return self.modules_value

    def _show_window(self, _hwnd: int, command: int) -> None:
        self.commands.append(command)
        self.state = {
            SW_RESTORE: "normal",
            SW_MINIMIZE: "minimized",
            SW_MAXIMIZE: "maximized",
        }[command]

    def _move_window(
        self,
        _hwnd: int,
        x: int,
        y: int,
        width: int,
        height: int,
        _repaint: bool,
    ) -> None:
        self.moves.append((x, y, width, height))
        self.identity = replace(self.identity, bounds=Bounds(x, y, width, height))

    def _enum_monitors(self):
        index = min(self.layout_calls, len(self.layouts) - 1)
        self.layout_calls += 1
        return [(None, None, bounds) for bounds in self.layouts[index]]

    def window_state(self, _hwnd: int) -> str:
        return self.state

    def _is_zoomed(self, _hwnd: int) -> bool:
        return self.state == "maximized"

    def _assert_action_target_allowed(
        self,
        expected: WindowIdentity,
        require_title_match: bool,
    ) -> WindowIdentity:
        return expected

    def _revalidate_action_target(
        self,
        expected: WindowIdentity,
        require_title_match: bool,
    ) -> WindowIdentity:
        return self.identity


@pytest.mark.parametrize(
    ("requested", "command"),
    [
        ("normal", SW_RESTORE),
        ("minimized", SW_MINIMIZE),
        ("maximized", SW_MAXIMIZE),
    ],
)
def test_window_state_commands_report_truthful_final_state(
    requested: str,
    command: int,
) -> None:
    backend = StateBackend("maximized" if requested == "normal" else "normal")
    result = backend.set_window_state(backend.identity, state=requested)

    assert result["state_after"] == requested
    assert result["requested_state"] == requested
    assert result["postcondition_verified"] is True
    assert backend.commands == [command]


def test_window_state_rechecks_strict_identity_after_guard_before_delivery() -> None:
    backend = StateBackend("normal")
    title_changed = False

    def strict_identity(
        expected: WindowIdentity,
        require_title_match: bool,
    ) -> WindowIdentity:
        assert require_title_match is True
        if title_changed:
            raise ComputerError("stale_window", "fixture strict title changed")
        return expected

    def change_title() -> None:
        nonlocal title_changed
        title_changed = True

    backend._assert_action_target_allowed = strict_identity  # type: ignore[method-assign]

    with pytest.raises(ComputerError) as raised:
        backend.set_window_state(
            backend.identity,
            state="maximized",
            mutation_guard=change_title,
            require_title_match=True,
        )

    assert raised.value.code == "stale_window"
    assert backend.commands == []


def test_window_state_failure_exposes_requested_and_observed_state(monkeypatch) -> None:
    backend = StateBackend("normal")

    def ignore_state_change(_hwnd: int, command: int) -> None:
        backend.commands.append(command)

    backend.modules_value.win32gui.ShowWindow = ignore_state_change
    moments = iter([0.0, 3.0])
    monkeypatch.setattr(win32_backend.time, "monotonic", lambda: next(moments))

    with pytest.raises(ComputerError) as raised:
        backend.set_window_state(backend.identity, state="maximized")

    assert raised.value.code == "window_state_postcondition_failed"
    assert raised.value.details == {
        "outcome": "failed",
        "method": "win32.ShowWindow",
        "requested_state": "maximized",
        "state_before": "normal",
        "state_after": "normal",
        "postcondition_verified": False,
    }


def test_resize_rejects_invalid_geometry_before_restore_mutation() -> None:
    backend = ResizeBackend("maximized", [[(0, 0, 100, 100)]])

    with pytest.raises(ComputerError) as raised:
        backend.resize_window(
            backend.identity,
            x=200,
            y=200,
            width=50,
            height=50,
            restore_first=True,
        )

    assert raised.value.code == "geometry_offscreen"
    assert backend.commands == []
    assert backend.moves == []
    assert backend.state == "maximized"


def test_resize_rechecks_topology_after_restore_and_reports_partial_mutation() -> None:
    backend = ResizeBackend(
        "maximized",
        [
            [(0, 0, 500, 500)],
            [(0, 0, 100, 100)],
        ],
    )

    with pytest.raises(ComputerError) as raised:
        backend.resize_window(
            backend.identity,
            x=10,
            y=10,
            width=200,
            height=200,
            restore_first=True,
        )

    assert raised.value.code == "geometry_offscreen"
    assert raised.value.details == {
        "outcome": "delivery_only",
        "method": "win32.ShowWindow",
        "restore_method": "win32.ShowWindow(SW_RESTORE)",
        "restore_performed": True,
        "restore_first": True,
        "restored_first": True,
        "state_after": "normal",
        "requested_bounds": {"x": 10, "y": 10, "width": 200, "height": 200},
        "actual_bounds": backend.identity.bounds.to_jsonable(),
        "postcondition_verified": False,
    }
    assert backend.commands == [SW_RESTORE]
    assert backend.moves == []
    assert backend.layout_calls == 2


def test_resize_wraps_restore_delivery_observation_failure_with_resize_context() -> None:
    backend = ResizeBackend("maximized", [[(0, 0, 1000, 1000)]])

    def fail_after_restore(
        expected: WindowIdentity,
        _require_title_match: bool,
    ) -> WindowIdentity:
        if backend.commands:
            raise ComputerError("stale_window", "fixture drift after restore delivery")
        return expected

    backend._revalidate_action_target = fail_after_restore  # type: ignore[method-assign]

    with pytest.raises(ComputerError) as raised:
        backend.resize_window(
            backend.identity,
            x=100,
            y=100,
            width=520,
            height=360,
            restore_first=True,
        )

    assert backend.commands == [SW_RESTORE]
    assert backend.moves == []
    assert raised.value.code == "stale_window"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["method"] == "win32.ShowWindow"
    assert raised.value.details["requested_bounds"] == {
        "x": 100,
        "y": 100,
        "width": 520,
        "height": 360,
    }
    assert raised.value.details["actual_bounds"] is None
    assert raised.value.details["restore_first"] is True
    assert raised.value.details["restored_first"] is None
    assert raised.value.details["restore_performed"] is None
    assert raised.value.details["restore_method"] == "win32.ShowWindow(SW_RESTORE)"
    assert raised.value.details["restore_status"] == "unknown"
    assert raised.value.details["state_after"] == "unavailable"
    assert raised.value.details["postcondition_verified"] is False


def test_resize_preserves_nested_restore_guard_rejection_before_delivery() -> None:
    backend = ResizeBackend("maximized", [[(0, 0, 1000, 1000)]])

    def reject_restore() -> None:
        raise ComputerError(
            "computer_actions_disabled",
            "fixture emergency stop",
        )

    with pytest.raises(ComputerError) as raised:
        backend.resize_window(
            backend.identity,
            x=100,
            y=100,
            width=520,
            height=360,
            restore_first=True,
            mutation_guard=reject_restore,
        )

    assert backend.commands == []
    assert backend.moves == []
    assert raised.value.code == "computer_actions_disabled"
    assert raised.value.details["outcome"] == "rejected"
    assert raised.value.details["method"] is None
    assert raised.value.details["restore_performed"] is False
    assert raised.value.details["restored_first"] is False
    assert raised.value.details["restore_method"] is None
    assert raised.value.details["restore_status"] == "not_attempted"
    assert raised.value.details["state_after"] == "maximized"
    assert raised.value.details["postcondition_verified"] is False


def test_resize_preserves_nested_restore_identity_rejection_before_delivery() -> None:
    backend = ResizeBackend("maximized", [[(0, 0, 1000, 1000)]])
    checks = 0

    def reject_nested_identity(
        expected: WindowIdentity,
        _require_title_match: bool,
    ) -> WindowIdentity:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ComputerError("stale_window", "fixture identity drift")
        return expected

    backend._assert_action_target_allowed = reject_nested_identity  # type: ignore[method-assign]

    with pytest.raises(ComputerError) as raised:
        backend.resize_window(
            backend.identity,
            x=100,
            y=100,
            width=520,
            height=360,
            restore_first=True,
        )

    assert checks == 2
    assert backend.commands == []
    assert backend.moves == []
    assert raised.value.code == "stale_window"
    assert raised.value.details["outcome"] == "rejected"
    assert raised.value.details["method"] is None
    assert raised.value.details["restore_performed"] is False
    assert raised.value.details["restored_first"] is False
    assert raised.value.details["restore_method"] is None
    assert raised.value.details["restore_status"] == "not_attempted"
    assert raised.value.details["state_after"] == "maximized"
    assert raised.value.details["postcondition_verified"] is False


def test_resize_restore_rechecks_identity_after_guard_before_show_window() -> None:
    backend = ResizeBackend("maximized", [[(0, 0, 1000, 1000)]])
    target_changed = False

    def exact_identity(
        expected: WindowIdentity,
        _require_title_match: bool,
    ) -> WindowIdentity:
        if target_changed:
            raise ComputerError("stale_window", "fixture changed during restore guard")
        return expected

    def change_target() -> None:
        nonlocal target_changed
        target_changed = True

    backend._assert_action_target_allowed = exact_identity  # type: ignore[method-assign]

    with pytest.raises(ComputerError) as raised:
        backend.resize_window(
            backend.identity,
            x=100,
            y=100,
            width=520,
            height=360,
            restore_first=True,
            mutation_guard=change_target,
        )

    assert raised.value.code == "stale_window"
    assert backend.commands == []
    assert backend.moves == []
    assert raised.value.details["outcome"] == "rejected"
    assert raised.value.details["restore_performed"] is False
    assert raised.value.details["restore_method"] is None
    assert raised.value.details["restore_status"] == "not_attempted"


def test_resize_rechecks_identity_state_and_topology_after_guard_before_move() -> None:
    identity_backend = ResizeBackend("normal", [[(0, 0, 1000, 1000)]])
    target_changed = False

    def exact_identity(
        expected: WindowIdentity,
        require_title_match: bool,
    ) -> WindowIdentity:
        assert require_title_match is True
        if target_changed:
            raise ComputerError("stale_window", "fixture changed during resize guard")
        return expected

    def change_target() -> None:
        nonlocal target_changed
        target_changed = True

    identity_backend._assert_action_target_allowed = exact_identity  # type: ignore[method-assign]

    with pytest.raises(ComputerError) as identity:
        identity_backend.resize_window(
            identity_backend.identity,
            x=100,
            y=100,
            width=520,
            height=360,
            mutation_guard=change_target,
            require_title_match=True,
        )

    assert identity.value.code == "stale_window"
    assert identity_backend.moves == []

    backend = ResizeBackend(
        "normal",
        [
            [(0, 0, 1000, 1000)],
            [(0, 0, 1000, 1000)],
            [(0, 0, 100, 100)],
        ],
    )

    with pytest.raises(ComputerError) as topology:
        backend.resize_window(
            backend.identity,
            x=100,
            y=100,
            width=520,
            height=360,
            mutation_guard=lambda: None,
        )

    assert topology.value.code == "geometry_offscreen"
    assert backend.layout_calls == 3
    assert backend.moves == []

    state_backend = ResizeBackend("normal", [[(0, 0, 1000, 1000)]])

    def maximize_during_guard() -> None:
        state_backend.state = "maximized"

    with pytest.raises(ComputerError) as state:
        state_backend.resize_window(
            state_backend.identity,
            x=100,
            y=100,
            width=520,
            height=360,
            mutation_guard=maximize_during_guard,
        )

    assert state.value.code == "window_maximized"
    assert state_backend.moves == []


def test_resize_normalizes_unexpected_post_restore_failure_as_partial() -> None:
    backend = ResizeBackend("maximized", [[(0, 0, 500, 500)]])

    def fail_is_iconic(_hwnd: int) -> bool:
        raise RuntimeError("fixture IsIconic failure")

    backend.modules_value.win32gui.IsIconic = fail_is_iconic

    with pytest.raises(ComputerError) as raised:
        backend.resize_window(
            backend.identity,
            x=10,
            y=10,
            width=200,
            height=200,
            restore_first=True,
        )

    assert raised.value.code == "win32_action_failed"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["restore_performed"] is True
    assert raised.value.details["restore_method"] == "win32.ShowWindow(SW_RESTORE)"
    assert raised.value.details["state_after"] == "normal"
    assert backend.commands == [SW_RESTORE]
    assert backend.moves == []


def test_failure_state_is_unavailable_when_strong_identity_cannot_be_revalidated() -> None:
    backend = StateBackend("maximized")

    def drift(_expected: WindowIdentity, _require_title_match: bool):
        raise ComputerError("stale_window", "fixture identity drift")

    backend._revalidate_action_target = drift  # type: ignore[method-assign]

    assert (
        win32_backend._safe_window_state(backend, backend.identity, False)
        == "unavailable"
    )


def test_window_state_delivery_exception_preserves_partial_metadata() -> None:
    backend = StateBackend("normal")

    def fail_show(_hwnd: int, _command: int) -> None:
        raise RuntimeError("fixture ShowWindow failure")

    backend.modules_value.win32gui.ShowWindow = fail_show

    with pytest.raises(ComputerError) as raised:
        backend.set_window_state(backend.identity, state="maximized")

    assert raised.value.code == "win32_action_failed"
    assert raised.value.details == {
        "outcome": "delivery_only",
        "method": "win32.ShowWindow",
        "requested_state": "maximized",
        "state_before": "normal",
        "state_after": "normal",
        "postcondition_verified": False,
    }


def test_resize_post_delivery_identity_failure_preserves_partial_metadata() -> None:
    backend = ResizeBackend("normal", [[(0, 0, 1000, 1000)]])

    def drift_after_move(
        expected: WindowIdentity,
        _require_title_match: bool,
    ) -> WindowIdentity:
        if backend.moves:
            raise ComputerError("stale_window", "fixture identity drift")
        return expected

    backend._revalidate_action_target = drift_after_move  # type: ignore[method-assign]

    with pytest.raises(ComputerError) as raised:
        backend.resize_window(
            backend.identity,
            x=100,
            y=100,
            width=520,
            height=360,
        )

    assert raised.value.code == "stale_window"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["method"] == "win32.MoveWindow"
    assert raised.value.details["requested_bounds"] == {
        "x": 100,
        "y": 100,
        "width": 520,
        "height": 360,
    }
    assert raised.value.details["actual_bounds"] is None
    assert raised.value.details["restore_first"] is False
    assert raised.value.details["restored_first"] is False
    assert raised.value.details["restore_performed"] is False
    assert raised.value.details["restore_method"] is None
    assert raised.value.details["state_after"] == "unavailable"
    assert raised.value.details["postcondition_verified"] is False


def test_resize_final_state_exception_preserves_partial_metadata() -> None:
    backend = ResizeBackend("normal", [[(0, 0, 1000, 1000)]])
    original_is_iconic = backend.modules_value.win32gui.IsIconic

    def fail_after_move(hwnd: int) -> bool:
        if backend.moves:
            raise RuntimeError("fixture final IsIconic failure")
        return bool(original_is_iconic(hwnd))

    backend.modules_value.win32gui.IsIconic = fail_after_move

    with pytest.raises(ComputerError) as raised:
        backend.resize_window(
            backend.identity,
            x=100,
            y=100,
            width=520,
            height=360,
        )

    assert raised.value.code == "win32_action_failed"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["method"] == "win32.MoveWindow"
    assert raised.value.details["requested_bounds"] == {
        "x": 100,
        "y": 100,
        "width": 520,
        "height": 360,
    }
    assert raised.value.details["actual_bounds"] == {
        "x": 100,
        "y": 100,
        "width": 520,
        "height": 360,
    }
    assert raised.value.details["state_after"] == "normal"
    assert raised.value.details["postcondition_verified"] is False


def test_resize_optional_dpi_failure_preserves_verified_result() -> None:
    backend = ResizeBackend("maximized", [[(0, 0, 1000, 1000)]])

    def fail_dpi(_hwnd: int) -> int | None:
        raise RuntimeError("fixture optional DPI failure")

    backend.window_dpi = fail_dpi  # type: ignore[method-assign]

    result = backend.resize_window(
        backend.identity,
        x=100,
        y=100,
        width=520,
        height=360,
        restore_first=True,
    )

    assert backend.commands == [SW_RESTORE]
    assert backend.moves == [(100, 100, 520, 360)]
    assert result["outcome"] == "postcondition_verified"
    assert result["postcondition_verified"] is True
    assert result["method"] == "win32.MoveWindow"
    assert result["requested_bounds"] == {
        "x": 100,
        "y": 100,
        "width": 520,
        "height": 360,
    }
    assert result["actual_bounds"] == result["requested_bounds"]
    assert result["restore_first"] is True
    assert result["restored_first"] is True
    assert result["dpi"] is None


def test_resize_failure_exposes_requested_and_actual_bounds(monkeypatch) -> None:
    backend = ResizeBackend("normal", [[(0, 0, 1000, 1000)]])

    def ignore_move(
        _hwnd: int,
        x: int,
        y: int,
        width: int,
        height: int,
        _repaint: bool,
    ) -> None:
        backend.moves.append((x, y, width, height))

    backend.modules_value.win32gui.MoveWindow = ignore_move
    moments = iter([0.0, 3.0])
    monkeypatch.setattr(win32_backend.time, "monotonic", lambda: next(moments))

    with pytest.raises(ComputerError) as raised:
        backend.resize_window(
            backend.identity,
            x=100,
            y=100,
            width=520,
            height=360,
        )

    assert raised.value.code == "resize_postcondition_failed"
    assert raised.value.details["requested_bounds"] == {
        "x": 100,
        "y": 100,
        "width": 520,
        "height": 360,
    }
    assert raised.value.details["actual_bounds"] == backend.identity.bounds.to_jsonable()
    assert raised.value.details["state_after"] == "normal"
    assert raised.value.details["postcondition_verified"] is False


def _binding() -> WinAppBinding:
    return WinAppBinding(
        path=Path("winapp.exe"),
        source="fixture",
        version=uia_winapp.WINAPP_VERSION,
        sha256=uia_winapp.WINAPP_EXE_SHA256,
    )


def _properties(element_id: str, toggle_state: str) -> dict[str, object]:
    return {
        "elementId": element_id,
        "properties": {
            "AutomationId": "Restore",
            "ControlType": "Button",
            "ClassName": "Button",
            "IsPassword": "False",
            "IsEnabled": "True",
            "IsOffscreen": "False",
            "IsInvokePatternAvailable": "False",
            "IsTogglePatternAvailable": "True",
            "ToggleState": toggle_state,
        },
    }


def _native_state(runtime_id: str) -> dict[str, object]:
    return {
        "automation_id": "Restore",
        "runtime_id": runtime_id,
        "is_password": False,
        "enabled": True,
        "control_type": "button",
        "class": "Button",
        "ancestor_signature": "sha256:" + "a" * 24,
        "patterns": ["uia.TogglePattern"],
        "range_value_bits": None,
        "value_pattern_status": "absent",
        "range_value_pattern_status": "absent",
        "legacy_value_read_only": None,
        "legacy_value_protected": None,
    }


def test_task_manager_runtime_and_provider_element_change_use_unique_stable_metadata(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(SimpleNamespace())
    properties = iter(
        [
            _properties("taskmgr-restore-old", "Off"),
            _properties("taskmgr-restore-new", "On"),
        ]
    )
    states = iter([_native_state("42,old"), _native_state("42,new")])
    native_reference_calls: list[dict[str, object] | None] = []
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (_binding(), next(properties)),
    )
    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **kwargs: (
            native_reference_calls.append(kwargs.get("element_reference"))
            or next(states)
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda *_args, **_kwargs: {
            "count": 1,
            "automationId": "Restore",
            "runtimeId": "42,new",
            "method": "uia.TogglePattern",
            "delivered": True,
            "resolutionStage": "stable_metadata_unique",
        },
    )
    reference = {
        **_element()["locator"],
        "control_type": "Button",
        "class": "Button",
        "enabled": True,
        "read_only": False,
        "password": False,
        "expires_unix_ms": 2_000_000,
    }

    _resolved_binding, result = adapter.invoke(
        _identity(),
        selector="btn-restore-a1b2",
        element_reference=reference,
    )

    assert result["outcome"] == "postcondition_verified"
    assert result["resolution_stage"] == "stable_metadata_unique"
    assert native_reference_calls == [reference, reference]


def test_native_worker_contains_atomic_focus_freshness_and_stable_resolution_guards() -> None:
    script = uia_winapp._native_uia_script(123)

    assert "GetForegroundWindow() -ne $handle" in script
    assert "element_ref_expired" in script
    assert "stable_metadata_unique" in script
    assert "requireTitleMatch" in script
    assert "process.SessionId" in script
    assert "InputDesktopName" in script
    assert "Ancestor-Signature" in script
    assert "function Resolve-TargetElement" in script
    assert script.count("Resolve-TargetElement") >= 4
    assert "Assert-FinalWindowIdentity;\n        Assert-ReferenceFresh;" in script
    assert (
        "Assert-OwnerAlive;\n        Assert-ActionsEnabled;\n        "
        "if([AgentToolsNativeSafety]::GetForegroundWindow() -ne $handle)"
    ) in script
    assert (
        "Assert-TargetSafe $pattern; $deliveryAttempted=$true;\n            "
        "$pattern.Invoke()"
    ) in script


def test_phase_a_cli_exposes_state_strict_title_and_reference_targets() -> None:
    parser = root_cli.build_parser()
    for command in ("restore", "minimize", "maximize"):
        parsed = parser.parse_args(
            ["computer", command, "--hwnd", "123", "--require-title-match"]
        )
        assert parsed.computer_command == command
        assert parsed.require_title_match is True

    resized = parser.parse_args(
        [
            "computer",
            "resize",
            "--hwnd",
            "123",
            "--x",
            "10",
            "--y",
            "20",
            "--width",
            "800",
            "--height",
            "600",
            "--restore-first",
        ]
    )
    assert resized.restore_first is True

    invoked = parser.parse_args(
        ["computer", "invoke", "--hwnd", "123", "--element-ref", "eref_" + "a" * 32]
    )
    assert invoked.element is None
    assert invoked.element_ref == "eref_" + "a" * 32

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "computer",
                "invoke",
                "--hwnd",
                "123",
                "--element",
                "button-a1b2",
                "--element-ref",
                "eref_" + "a" * 32,
            ]
        )
