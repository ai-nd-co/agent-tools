from __future__ import annotations

import ctypes
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_tools.computer import actions
from agent_tools.computer import physical_input as physical_input_module
from agent_tools.computer.actions import ActionCoordinator, ActionExecution, NativeMutexLease
from agent_tools.computer.consequence import (
    ConsequenceObserver,
    ConsequenceSnapshot,
    compare_snapshots,
    delivery_from_result,
)
from agent_tools.computer.models import Bounds, ComputerError, WindowIdentity
from agent_tools.computer.physical_input import (
    PhysicalCaptureTarget,
    StagedInputStore,
    Win32PhysicalInput,
    parse_shortcut,
    terminal_targeting_contract,
)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only physical input")


def _identity(**changes: object) -> WindowIdentity:
    identity = WindowIdentity(
        hwnd=123,
        pid=456,
        thread_id=789,
        process="fixture.exe",
        title="Task fixture",
        class_name="FixtureClass",
        bounds=Bounds(10, 20, 400, 300),
        process_started=999,
        session_id=1,
        desktop_name="Default",
    )
    return replace(identity, **changes)


def _snapshot(
    *,
    identity: WindowIdentity | None = None,
    raw_title: str | None = "Task fixture",
    state: str | None = "normal",
    foreground: int | None = 123,
    cursor: dict[str, int] | None = None,
    topology: dict[str, tuple[int, int | None, int | None, str]] | None = None,
) -> ConsequenceSnapshot:
    resolved = identity
    target: dict[str, Any] = {
        "exists": resolved is not None,
        "observation_error": None if resolved is not None else "invalid_window",
    }
    if resolved is not None:
        target.update(
            {
                "strong_identity": {
                    "hwnd": resolved.hwnd,
                    "pid": resolved.pid,
                    "thread_id": resolved.thread_id,
                    "process_started": resolved.process_started,
                    "session_id": resolved.session_id,
                    "process_hmac_sha256": "p" * 64,
                    "class_hmac_sha256": "c" * 64,
                    "desktop_hmac_sha256": "d" * 64,
                    "fingerprint": "f" * 64,
                },
                "state": state,
                "bounds": resolved.bounds.to_jsonable(),
                "monitor": {"x": 0, "y": 0, "width": 1920, "height": 1080},
                "dpi": 96,
                "foreground": foreground == resolved.hwnd,
                "z_order": {"visible_index": 0, "visible_count": 1},
                "title_hmac_sha256": "t" * 64 if raw_title == "Task fixture" else "u" * 64,
            }
        )
    public: dict[str, Any] = {
        "target": target,
        "foreground": {"hwnd": foreground},
        "topology": {"count": 1, "truncated": False, "digest": "z" * 64},
        "input_generation": 7,
        "capture_provenance": None,
    }
    if cursor is not None:
        public["cursor"] = cursor
    return ConsequenceSnapshot(
        public=public,
        raw_title=raw_title,
        target_identity=resolved,
        topology=topology or {"123": (0, 456, 999, "FixtureClass")},
        topology_truncated=False,
    )


@pytest.mark.parametrize(
    ("after", "expected_field"),
    [
        (_snapshot(identity=_identity(bounds=Bounds(30, 40, 400, 300))), "bounds"),
        (_snapshot(identity=_identity(bounds=Bounds(10, 20, 500, 350))), "bounds"),
        (_snapshot(identity=_identity(), state="minimized"), "state"),
        (_snapshot(identity=_identity(), state="maximized"), "state"),
        (_snapshot(identity=None, raw_title=None, state=None, foreground=None), "exists"),
        (_snapshot(identity=_identity(process_started=1000)), "strong_identity"),
    ],
)
def test_consequence_detects_required_window_deltas(
    after: ConsequenceSnapshot,
    expected_field: str,
) -> None:
    result = compare_snapshots(
        _snapshot(identity=_identity()),
        after,
        operation="fixture",
        expectation=None,
    )

    assert result["observed_effect"] == "changed"
    assert expected_field in result["target_delta"]


def test_consequence_detects_monitor_and_dpi_drift() -> None:
    before = _snapshot(identity=_identity())
    after = _snapshot(identity=_identity())
    after.public["target"]["monitor"] = {
        "x": 1920,
        "y": 0,
        "width": 2560,
        "height": 1440,
    }
    after.public["target"]["dpi"] = 144

    result = compare_snapshots(before, after, operation="move", expectation=None)

    assert result["observed_effect"] == "changed"
    assert result["target_delta"]["monitor"]["after"]["x"] == 1920
    assert result["target_delta"]["dpi"] == {"before": 96, "after": 144}


def test_consequence_distinguishes_spinner_churn_from_meaningful_rename() -> None:
    spinner = compare_snapshots(
        _snapshot(identity=_identity(), raw_title="| Task fixture"),
        _snapshot(identity=_identity(), raw_title="/ Task fixture"),
        operation="focus",
        expectation=None,
    )
    renamed = compare_snapshots(
        _snapshot(identity=_identity(), raw_title="Task fixture"),
        _snapshot(identity=_identity(), raw_title="Build complete"),
        operation="focus",
        expectation=None,
    )

    assert spinner["target_delta"]["title"]["raw_changed"] is True
    assert spinner["target_delta"]["title"]["classification"] == "volatile_only"
    assert renamed["target_delta"]["title"]["classification"] == "meaningful"
    serialized = json.dumps([spinner, renamed])
    assert "Task fixture" not in serialized
    assert "Build complete" not in serialized


def test_consequence_separates_unrelated_window_churn_and_noop() -> None:
    before = _snapshot(identity=_identity())
    after = _snapshot(
        identity=_identity(),
        topology={
            "123": (0, 456, 999, "FixtureClass"),
            "999": (1, 1000, 1001, "OtherClass"),
        },
    )
    churn = compare_snapshots(before, after, operation="invoke", expectation=None)
    noop = compare_snapshots(before, before, operation="invoke", expectation=None)

    assert churn["observed_effect"] == "no_observable_change"
    assert churn["unrelated_window_churn"]["created_count"] == 1
    assert noop["observed_effect"] == "no_observable_change"
    assert noop["target_delta"] == {}


def test_consequence_reports_input_generation_drift() -> None:
    before = _snapshot(identity=_identity())
    after = _snapshot(identity=_identity())
    after.public["input_generation"] = 8

    result = compare_snapshots(before, after, operation="type", expectation=None)

    assert result["observed_effect"] == "changed"
    assert result["target_delta"]["input_generation"] == {
        "before": 7,
        "after": 8,
    }


def test_consequence_classifies_focus_theft_and_semantic_state() -> None:
    focus_theft = compare_snapshots(
        _snapshot(identity=_identity()),
        _snapshot(identity=_identity(), foreground=777),
        operation="focus",
        expectation={"kind": "foreground", "hwnd": 123, "required": True},
    )
    semantic = compare_snapshots(
        _snapshot(identity=_identity()),
        _snapshot(identity=_identity()),
        operation="invoke",
        expectation={"kind": "semantic", "required": True},
        semantic_postcondition={
            "verified": True,
            "changed": True,
            "selected_before": False,
            "selected_after": True,
        },
    )

    assert focus_theft["expectation"] == "unexpected_effect"
    assert focus_theft["target_delta"]["foreground"]["after"] is False
    assert semantic["expectation"] == "expected_effect_verified"
    assert semantic["semantic_postcondition"]["selected_after"] is True


def test_noop_focus_keeps_delivery_separate_from_observed_effect() -> None:
    snapshot = _snapshot(identity=_identity())
    consequence = compare_snapshots(
        snapshot,
        snapshot,
        operation="focus",
        expectation={"kind": "foreground", "hwnd": 123, "required": True},
        semantic_postcondition={"verified": True, "changed": False},
    )
    delivery = delivery_from_result(
        {
            "outcome": "postcondition_verified",
            "method": "win32.SetForegroundWindow",
            "postcondition_verified": True,
        }
    )

    assert consequence["before"] == consequence["after"]
    assert consequence["target_delta"] == {}
    assert consequence["semantic_postcondition"] == {
        "changed": False,
        "verified": True,
    }
    assert consequence["observed_effect"] == "no_observable_change"
    assert consequence["expectation"] == "unexpected_effect"
    assert delivery == {
        "status": "delivered",
        "method": "win32.SetForegroundWindow",
    }


def test_verified_idempotent_semantic_postcondition_is_not_mislabeled() -> None:
    snapshot = _snapshot(identity=_identity())

    consequence = compare_snapshots(
        snapshot,
        snapshot,
        operation="set-value",
        expectation={"kind": "semantic", "required": True},
        semantic_postcondition={
            "verified": True,
            "changed": False,
            "change_required": False,
            "value_char_count": 7,
            "readback_char_count": 7,
        },
    )

    assert consequence["observed_effect"] == "no_observable_change"
    assert consequence["expectation"] == "expected_effect_verified"
    assert consequence["semantic_postcondition"]["change_required"] is False


@pytest.mark.parametrize(
    ("kind", "expectation"),
    [
        ("state", {"kind": "state", "state": "normal", "required": True}),
        (
            "bounds",
            {
                "kind": "bounds",
                "bounds": _identity().bounds.to_jsonable(),
                "required": True,
            },
        ),
        ("semantic", {"kind": "semantic", "required": True}),
    ],
)
def test_already_satisfied_postcondition_is_not_an_observed_effect(
    kind: str,
    expectation: dict[str, Any],
) -> None:
    snapshot = _snapshot(identity=_identity())
    result = compare_snapshots(
        snapshot,
        snapshot,
        operation=kind,
        expectation=expectation,
        semantic_postcondition=(
            {"verified": True, "changed": False} if kind == "semantic" else None
        ),
    )

    assert result["observed_effect"] == "no_observable_change"
    assert result["expectation"] == "unexpected_effect"


def test_matching_changed_focus_is_expected_effect_verified() -> None:
    result = compare_snapshots(
        _snapshot(identity=_identity(), foreground=777),
        _snapshot(identity=_identity(), foreground=123),
        operation="focus",
        expectation={"kind": "foreground", "hwnd": 123, "required": True},
        semantic_postcondition={"verified": True, "changed": False},
    )

    assert result["observed_effect"] == "changed"
    assert result["target_delta"]["foreground"] == {
        "before": False,
        "after": True,
    }
    assert result["expectation"] == "expected_effect_verified"


def test_consequence_classifies_expected_close_replacement_and_ambiguous_snapshot() -> None:
    closed = compare_snapshots(
        _snapshot(identity=_identity()),
        _snapshot(identity=None, raw_title=None, state=None, foreground=None),
        operation="close",
        expectation={"kind": "exists", "exists": False, "required": True},
    )
    replacement = compare_snapshots(
        _snapshot(identity=_identity()),
        _snapshot(identity=_identity(process_started=1_001)),
        operation="replace",
        expectation=None,
    )
    ambiguous_before = _snapshot(identity=_identity())
    ambiguous_before.public["target"]["observation_error"] = "snapshot_failed"
    ambiguous = compare_snapshots(
        ambiguous_before,
        _snapshot(identity=_identity()),
        operation="focus",
        expectation={"kind": "foreground", "hwnd": 123, "required": True},
    )

    assert closed["expectation"] == "expected_effect_verified"
    assert replacement["target_delta"]["strong_identity"]["after"] is not None
    assert ambiguous["observed_effect"] == "ambiguous"
    assert ambiguous["expectation"] == "ambiguous_effect"


@pytest.mark.parametrize(
    ("terminal_effect", "expected_effect", "expected_expectation"),
    [
        (
            {"character_delivery": "delivered"},
            "no_observable_change",
            "no_declared_expected_effect",
        ),
        (
            {"enter_delivery": "delivered", "submission": "not_observed"},
            "no_observable_change",
            "unexpected_effect",
        ),
        (
            {"enter_delivery": "delivered", "submission": "observed"},
            "changed",
            "expected_effect_verified",
        ),
        (
            {"enter_delivery": "delivered", "inserted_newline": "inserted"},
            "changed",
            "unexpected_effect",
        ),
        (
            {"composer_change": "changed"},
            "changed",
            "unexpected_effect",
        ),
        (
            {"focus_change": "changed"},
            "changed",
            "unexpected_effect",
        ),
        (
            {"tab_change": "changed"},
            "changed",
            "unexpected_effect",
        ),
        (
            {
                "character_delivery": "not_attempted",
                "enter_delivery": "not_attempted",
                "submission": "not_observed",
                "inserted_newline": "not_observed",
                "composer_change": "not_observed",
                "focus_change": "not_observed",
                "tab_change": "not_observed",
            },
            "no_observable_change",
            "unexpected_effect",
        ),
    ],
)
def test_terminal_consequence_categories_are_distinct(
    terminal_effect: dict[str, Any],
    expected_effect: str,
    expected_expectation: str,
) -> None:
    expects_submission = terminal_effect.get("character_delivery") != "delivered"
    result = compare_snapshots(
        _snapshot(identity=_identity()),
        _snapshot(identity=_identity()),
        operation="key" if expects_submission else "type",
        expectation=(
            {"kind": "terminal_submission", "required": True}
            if expects_submission
            else None
        ),
        terminal_effect=terminal_effect,
    )

    assert result["observed_effect"] == expected_effect
    assert result["expectation"] == expected_expectation
    assert result["terminal_effect"] == terminal_effect


def test_consequence_snapshot_has_privacy_safe_foreground_identity_and_active_monitor() -> None:
    backend = _TimedBackend()
    backend.identity = replace(backend.identity, title="Private fixture title")

    snapshot = ConsequenceObserver(backend=backend, hwnd=123, settle_seconds=0).capture()
    serialized = json.dumps(snapshot.public, sort_keys=True)

    assert snapshot.public["foreground"]["hwnd"] == 123
    assert len(snapshot.public["foreground"]["strong_identity"]["fingerprint"]) == 64
    assert snapshot.public["active_monitor"] == {
        "x": 0,
        "y": 0,
        "width": 1920,
        "height": 1080,
    }
    assert "Private fixture title" not in serialized
    assert "screenshot" not in serialized
    assert "process_list" not in serialized
    assert "ui_tree" not in serialized
    assert "contents" not in serialized


class _TimedBackend:
    def __init__(self) -> None:
        self.identity = _identity()

    def capture_identity(self, _hwnd: int) -> WindowIdentity:
        return self.identity

    def focused_window(self) -> dict[str, Any]:
        return {"window": {"hwnd": self.identity.hwnd}}

    def window_state(self, _hwnd: int) -> str:
        return "normal"

    def window_dpi(self, _hwnd: int) -> int:
        return 96

    def capture_native_geometry(self, _hwnd: int) -> Any:
        return SimpleNamespace(monitor_bounds=Bounds(0, 0, 1920, 1080))


def test_exact_two_second_observation_includes_inside_delay_not_beyond() -> None:
    backend = _TimedBackend()
    sleeps: list[float] = []

    def mutate_inside(seconds: float) -> None:
        sleeps.append(seconds)
        backend.identity = replace(backend.identity, bounds=Bounds(20, 30, 400, 300))

    observer = ConsequenceObserver(
        backend=backend,
        hwnd=123,
        settle_seconds=2.0,
        sleeper=mutate_inside,
    )
    before = observer.capture()
    inside = observer.settle_and_compare(before, operation="resize", expectation=None)
    backend.identity = _identity()
    observer = ConsequenceObserver(
        backend=backend,
        hwnd=123,
        settle_seconds=2.0,
        sleeper=lambda seconds: sleeps.append(seconds),
    )
    before = observer.capture()
    beyond = observer.settle_and_compare(before, operation="resize", expectation=None)
    backend.identity = replace(backend.identity, bounds=Bounds(99, 99, 400, 300))

    assert sleeps == [2.0, 2.0]
    assert inside["observed_effect"] == "changed"
    assert beyond["observed_effect"] == "no_observable_change"
    assert set(inside["phase_timings_ms"]) == {
        "settle_wait",
        "after_capture",
        "compare",
        "total",
    }
    assert all(value >= 0 for value in inside["phase_timings_ms"].values())


class _StageClock:
    def __init__(self, value: int = 1_000_000) -> None:
        self.value = value

    def now_unix_ms(self) -> int:
        return self.value


def _capture_target(identity: WindowIdentity | None = None) -> PhysicalCaptureTarget:
    resolved = identity or _identity()
    return PhysicalCaptureTarget(
        capture_id="cap_" + "a" * 32,
        identity=resolved,
        record={
            "geometry": {"window_bounds": resolved.bounds.to_jsonable()},
            "window_identity": {"hwnd": resolved.hwnd},
        },
        image_point={"x": 10, "y": 20},
        screen_point={"x": 20, "y": 40},
        provenance={"capture_id": "cap_" + "a" * 32},
    )


def test_physical_preflight_reports_only_bounded_non_content_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    target = _capture_target(identity)
    fresh = replace(
        target,
        provenance={
            "capture_id": target.capture_id,
            "image_sha256": "a" * 64,
            "profile": "compact",
            "capture_age_ms": 125,
            "capture_ttl_remaining_ms": 29_875,
            "geometry_fingerprint": "b" * 64,
        },
    )

    class Backend:
        @staticmethod
        def focus_window(_identity, **_kwargs):
            return {"postcondition_verified": True, "outcome": "postcondition_verified"}

    class Physical:
        @staticmethod
        def assert_no_held_input() -> None:
            return None

        @staticmethod
        def assert_foreground_identity(_backend, _identity) -> None:
            return None

    execution = ActionExecution("fixture", "click", None, input_generation=9)
    monkeypatch.setattr(execution, "assert_input_generation_current", lambda: None)
    monkeypatch.setattr(actions, "resolve_physical_capture", lambda **_kwargs: fresh)

    resolved = actions._physical_delivery_preflight(
        execution,
        Backend(),  # type: ignore[arg-type]
        object(),
        Physical(),  # type: ignore[arg-type]
        target,
        require_point=True,
    )

    assert resolved is fresh
    assert execution.physical_preflight == {
        **fresh.provenance,
        "target": {
            "hwnd": 123,
            "pid": 456,
            "thread_id": 789,
            "process_started": 999,
            "session_id": 1,
        },
        "point_mapping": {
            "required": True,
            "verified": True,
            "image_point": {"x": 10, "y": 20},
            "native_screen_point": {"x": 20, "y": 40},
        },
        "foreground_identity_verified": True,
        "input_generation": 9,
    }
    serialized = json.dumps(execution.physical_preflight)
    assert identity.title not in serialized
    assert identity.process not in serialized
    assert set(execution.phase_timings_ms) == {"focus", "capture_revalidation"}


def test_staged_input_is_signed_generation_bound_and_one_shot(tmp_path: Path) -> None:
    clock = _StageClock()
    store = StagedInputStore(tmp_path / "stages", clock=clock)
    identity = _identity()
    issued = store.issue(
        identity=identity,
        capture=_capture_target(identity),
        input_generation=10,
        text="full draft",
        terminal_targeting={"status": "not_applicable"},
    )
    evidence = _capture_target(identity)
    consumed = store.consume_for_enter(
        issued["staged_input"],
        identity=identity,
        current_input_generation=11,
        evidence_capture=evidence,
        draft_evidence="exact",
    )

    assert issued["ttl_seconds"] == 120
    assert issued["expires_at"] == "1970-01-01T00:18:40.000Z"
    assert consumed["text_sha256"] == hashlib_sha256("full draft")
    assert consumed["draft_evidence"] == "exact"
    with pytest.raises(ComputerError) as replay:
        store.consume_for_enter(
            issued["staged_input"],
            identity=identity,
            current_input_generation=11,
            evidence_capture=evidence,
            draft_evidence="exact",
        )
    assert replay.value.code == "staged_input_missing"


def test_staged_input_rejects_generation_geometry_and_identity_drift(tmp_path: Path) -> None:
    store = StagedInputStore(tmp_path / "stages", clock=_StageClock())

    def issue() -> str:
        return str(
            store.issue(
                identity=_identity(),
                capture=_capture_target(),
                input_generation=20,
                text="draft",
                terminal_targeting={"status": "not_applicable"},
            )["staged_input"]
        )

    with pytest.raises(ComputerError) as generation:
        store.consume_for_enter(
            issue(),
            identity=_identity(),
            current_input_generation=22,
            evidence_capture=_capture_target(),
            draft_evidence="exact",
        )
    assert generation.value.code == "staged_input_generation_changed"

    moved = _capture_target(_identity(bounds=Bounds(30, 40, 400, 300)))
    with pytest.raises(ComputerError) as geometry:
        store.consume_for_enter(
            issue(),
            identity=_identity(),
            current_input_generation=21,
            evidence_capture=moved,
            draft_evidence="exact",
        )
    assert geometry.value.code == "staged_input_geometry_changed"

    with pytest.raises(ComputerError) as identity:
        store.consume_for_enter(
            issue(),
            identity=_identity(process_started=1000),
            current_input_generation=21,
            evidence_capture=_capture_target(),
            draft_evidence="exact",
        )
    assert identity.value.code == "staged_input_target_changed"


def test_terminal_targeting_truthfully_requires_visible_content_verification() -> None:
    terminal = _identity(
        process="WindowsTerminal.exe",
        class_name="CASCADIA_HOSTING_WINDOW_CLASS",
    )
    with pytest.raises(ComputerError) as missing:
        terminal_targeting_contract(terminal, visible_tab_verified=False)
    accepted = terminal_targeting_contract(terminal, visible_tab_verified=True)

    assert missing.value.code == "terminal_tab_verification_required"
    assert accepted == {
        "status": "limited",
        "reason": "multi_tab_same_hwnd_not_semantically_bound",
        "visible_tab_verification_required": True,
        "caller_visible_tab_verified": True,
    }


class _PhysicalModules:
    def __init__(self, owner: _PhysicalBackend) -> None:
        self.win32gui = SimpleNamespace(GetForegroundWindow=lambda: owner.foreground)
        self.win32api = SimpleNamespace(GetCursorPos=lambda: owner.cursor)


class _PhysicalBackend:
    def __init__(self, identity: WindowIdentity) -> None:
        self.identity = identity
        self.foreground = identity.hwnd
        self.cursor = (100, 200)
        self._modules = _PhysicalModules(self)

    def modules(self) -> Any:
        return self._modules

    def revalidate_identity(self, identity: WindowIdentity) -> WindowIdentity:
        assert identity == self.identity
        return identity

    def display_info(self) -> dict[str, Any]:
        return {"virtual_bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080}}


def _current_thread_identity() -> WindowIdentity:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentThreadId.restype = wintypes_dword()
    return _identity(thread_id=int(kernel32.GetCurrentThreadId()))


def wintypes_dword() -> Any:
    from ctypes import wintypes

    return wintypes.DWORD


def _fake_user32(mapping: dict[str, int]) -> Any:
    return SimpleNamespace(
        GetKeyboardLayout=lambda _thread_id: 1,
        VkKeyScanW=lambda character: mapping.get(character, -1),
        GetAsyncKeyState=lambda _virtual_key: 0,
    )


def test_virtual_key_typing_uses_shift_and_never_unicode_packets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    backend = _PhysicalBackend(identity)
    batches: list[list[dict[str, int]]] = []
    monkeypatch.setattr(
        physical_input_module,
        "_user32",
        lambda: _fake_user32({"a": 0x41, "A": 0x0141}),
    )

    def sender(events: list[dict[str, int]]) -> int:
        batches.append(events)
        return len(events)

    result = Win32PhysicalInput(event_sender=sender).type_text(
        backend,  # type: ignore[arg-type]
        identity,
        40,
        50,
        "aA",
        mutation_guard=lambda: None,
    )

    keyboard = [event for batch in batches for event in batch if event["type"] == 1]
    assert result["delivered_character_count"] == 2
    assert result["first_undelivered_position"] is None
    assert all("vk" in event and "scan" not in event for event in keyboard)
    assert any(event["vk"] == 0x10 and event["flags"] == 0 for event in keyboard)


def test_layout_mapped_unicode_uses_ordinary_virtual_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    backend = _PhysicalBackend(identity)
    batches: list[list[dict[str, int]]] = []
    monkeypatch.setattr(
        physical_input_module,
        "_user32",
        lambda: _fake_user32({"Ж": 0x0146}),
    )

    result = Win32PhysicalInput(
        event_sender=lambda events: batches.append(events) or len(events)
    ).type_text(
        backend,  # type: ignore[arg-type]
        identity,
        40,
        50,
        "Ж",
        mutation_guard=lambda: None,
    )

    keyboard = [event for batch in batches for event in batch if event["type"] == 1]
    assert result["delivered_character_count"] == 1
    assert [event["vk"] for event in keyboard] == [0x10, 0x46, 0x46, 0x10]
    assert all(set(event) <= {"type", "vk", "flags"} for event in keyboard)


def test_focus_drift_after_one_character_reports_exact_partial_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    backend = _PhysicalBackend(identity)
    calls = 0
    monkeypatch.setattr(
        physical_input_module,
        "_user32",
        lambda: _fake_user32({"a": 0x41, "b": 0x42}),
    )

    def sender(events: list[dict[str, int]]) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            backend.foreground = 999
        return len(events)

    with pytest.raises(ComputerError) as partial:
        Win32PhysicalInput(event_sender=sender).type_text(
            backend,  # type: ignore[arg-type]
            identity,
            40,
            50,
            "ab",
            mutation_guard=lambda: None,
        )

    assert partial.value.code == "physical_text_partial"
    assert partial.value.details["reason"] == "physical_input_focus_changed"
    assert partial.value.details["delivered_character_count"] == 1
    assert partial.value.details["first_undelivered_position"] == 1
    assert partial.value.details["focus_click_delivered"] is True


def test_generation_drift_after_one_character_reports_exact_partial_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    backend = _PhysicalBackend(identity)
    guard_calls = 0
    monkeypatch.setattr(
        physical_input_module,
        "_user32",
        lambda: _fake_user32({"a": 0x41, "b": 0x42}),
    )

    def generation_guard() -> None:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 2:
            raise ComputerError(
                "computer_input_generation_changed",
                "fixture generation drift",
            )

    with pytest.raises(ComputerError) as partial:
        Win32PhysicalInput(event_sender=lambda events: len(events)).type_text(
            backend,  # type: ignore[arg-type]
            identity,
            40,
            50,
            "ab",
            mutation_guard=generation_guard,
        )

    assert partial.value.code == "physical_text_partial"
    assert partial.value.details["reason"] == "computer_input_generation_changed"
    assert partial.value.details["delivered_character_count"] == 1
    assert partial.value.details["first_undelivered_position"] == 1
    assert partial.value.details["automatic_cleanup_performed"] is False


def test_focus_click_without_character_delivery_is_still_partial_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    backend = _PhysicalBackend(identity)
    monkeypatch.setattr(
        physical_input_module,
        "_user32",
        lambda: _fake_user32({"a": 0x41}),
    )

    with pytest.raises(ComputerError) as partial:
        Win32PhysicalInput(event_sender=lambda events: len(events)).type_text(
            backend,  # type: ignore[arg-type]
            identity,
            40,
            50,
            "a",
            mutation_guard=lambda: (_ for _ in ()).throw(
                ComputerError("computer_input_generation_changed", "fixture drift")
            ),
        )

    assert partial.value.details["delivered_character_count"] == 0
    assert partial.value.details["first_undelivered_position"] == 0
    assert partial.value.details["focus_click_delivered"] is True
    assert partial.value.details["partial_delivery"] is True
    assert partial.value.details["outcome"] == "delivery_only"


def test_partial_click_releases_injected_pointer_button() -> None:
    identity = _identity()
    backend = _PhysicalBackend(identity)
    batches: list[list[dict[str, int]]] = []

    def sender(events: list[dict[str, int]]) -> int:
        batches.append(events)
        return 2 if len(batches) == 1 else len(events)

    with pytest.raises(ComputerError) as partial:
        Win32PhysicalInput(event_sender=sender).click(
            backend,  # type: ignore[arg-type]
            identity,
            40,
            50,
            double=False,
        )

    assert partial.value.code == "physical_input_partial"
    assert batches[1] == [{"type": 0, "flags": 0x0004, "data": 0}]


def test_pointer_click_and_wheel_restore_cursor_and_report_delivery_only() -> None:
    identity = _identity()

    for operation in ("click", "double-click", "vertical-wheel", "horizontal-wheel"):
        backend = _PhysicalBackend(identity)
        batches: list[list[dict[str, int]]] = []

        def sender(
            events: list[dict[str, int]],
            *,
            _batches: list[list[dict[str, int]]] = batches,
            _backend: _PhysicalBackend = backend,
        ) -> int:
            _batches.append(events)
            _backend.cursor = (40, 50) if len(_batches) == 1 else (100, 200)
            return len(events)

        physical = Win32PhysicalInput(event_sender=sender)
        if operation in {"click", "double-click"}:
            result = physical.click(
                backend,  # type: ignore[arg-type]
                identity,
                40,
                50,
                double=operation == "double-click",
            )
            assert result["events_intended"] == (5 if operation == "double-click" else 3)
        else:
            result = physical.wheel(
                backend,  # type: ignore[arg-type]
                identity,
                40,
                50,
                vertical=2 if operation == "vertical-wheel" else None,
                horizontal=-3 if operation == "horizontal-wheel" else None,
            )
            assert result["axis"] == operation.removesuffix("-wheel")
        assert result["outcome"] == "delivery_only"
        assert result["verification_required"] is True
        assert backend.cursor == (100, 200)


@pytest.mark.parametrize(
    ("operation", "method"),
    [
        ("click", "win32.SendInput.mouse_click"),
        ("wheel", "win32.SendInput.wheel"),
        ("text", "win32.SendInput.VkKeyScanW"),
    ],
)
def test_cursor_restore_failure_preserves_full_primary_delivery_and_consumes_once(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    method: str,
) -> None:
    identity = _identity()
    backend = _PhysicalBackend(identity)
    calls = 0
    restore_call = 3 if operation == "text" else 2
    monkeypatch.setattr(
        physical_input_module,
        "_user32",
        lambda: _fake_user32({"a": 0x41}),
    )

    def sender(events: list[dict[str, int]]) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            backend.cursor = (40, 50)
        return 0 if calls == restore_call else len(events)

    physical = Win32PhysicalInput(event_sender=sender)
    with pytest.raises(ComputerError) as failed:
        if operation == "click":
            physical.click(
                backend,  # type: ignore[arg-type]
                identity,
                40,
                50,
                double=False,
            )
        elif operation == "wheel":
            physical.wheel(
                backend,  # type: ignore[arg-type]
                identity,
                40,
                50,
                vertical=1,
                horizontal=None,
            )
        else:
            physical.type_text(
                backend,  # type: ignore[arg-type]
                identity,
                40,
                50,
                "a",
                mutation_guard=lambda: None,
            )

    assert failed.value.code == "physical_cursor_restore_failed"
    assert failed.value.details["method"] == method
    assert failed.value.details["cursor_restored"] is False
    assert failed.value.details["cursor_cleanup"] == {
        "status": "failed",
        "error_code": "physical_cursor_restore_failed",
        "method": "win32.SendInput.mouse_move",
    }
    if operation == "text":
        assert failed.value.details["intended_character_count"] == 1
        assert failed.value.details["delivered_character_count"] == 1
    else:
        assert failed.value.details["events_intended"] in {2, 3}
        assert failed.value.details["events_delivered"] == failed.value.details["events_intended"]

    class RecordStore:
        calls = 0

        def delete(self, _capture_id: str) -> None:
            self.calls += 1

    records = RecordStore()
    actions._consume_physical_capture(
        records,
        "cap_" + "c" * 32,
        delivered=actions._error_has_physical_delivery(failed.value),
        delivery_error=failed.value,
    )
    assert records.calls == 1


@pytest.mark.parametrize(
    ("operation", "expected_code", "primary_calls", "restore_call"),
    [
        ("click", "physical_input_partial", [2, 1], 3),
        ("wheel", "physical_input_partial", [1], 2),
        ("text", "physical_text_partial", [3, 0], 3),
    ],
)
def test_cursor_restore_failure_does_not_override_partial_primary_delivery(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    expected_code: str,
    primary_calls: list[int],
    restore_call: int,
) -> None:
    identity = _identity()
    backend = _PhysicalBackend(identity)
    calls = 0
    monkeypatch.setattr(
        physical_input_module,
        "_user32",
        lambda: _fake_user32({"a": 0x41}),
    )

    def sender(events: list[dict[str, int]]) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            backend.cursor = (40, 50)
        if calls == restore_call:
            return 0
        if calls <= len(primary_calls):
            return primary_calls[calls - 1]
        return len(events)

    physical = Win32PhysicalInput(event_sender=sender)
    with pytest.raises(ComputerError) as failed:
        if operation == "click":
            physical.click(
                backend,  # type: ignore[arg-type]
                identity,
                40,
                50,
                double=False,
            )
        elif operation == "wheel":
            physical.wheel(
                backend,  # type: ignore[arg-type]
                identity,
                40,
                50,
                vertical=1,
                horizontal=None,
            )
        else:
            physical.type_text(
                backend,  # type: ignore[arg-type]
                identity,
                40,
                50,
                "a",
                mutation_guard=lambda: None,
            )

    assert failed.value.code == expected_code
    assert failed.value.details["partial_delivery"] is True
    assert failed.value.details["cursor_restored"] is False
    assert failed.value.details["cursor_cleanup"]["status"] == "failed"
    if operation == "text":
        assert failed.value.details["delivered_character_count"] == 0
        assert failed.value.details["first_undelivered_position"] == 0
        assert failed.value.details["focus_click_delivered"] is True
    else:
        assert failed.value.details["events_intended"] in {2, 3}
        assert failed.value.details["events_delivered"] == primary_calls[0]


def test_move_pointer_intentionally_leaves_verified_target_position() -> None:
    identity = _identity()
    backend = _PhysicalBackend(identity)

    def sender(events: list[dict[str, int]]) -> int:
        backend.cursor = (40, 50)
        return len(events)

    result = Win32PhysicalInput(event_sender=sender).move_pointer(
        backend,  # type: ignore[arg-type]
        identity,
        40,
        50,
    )

    assert result["outcome"] == "postcondition_verified"
    assert backend.cursor == (40, 50)


def test_partial_typing_reports_exact_position_and_does_not_submit_or_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    backend = _PhysicalBackend(identity)
    calls = 0
    monkeypatch.setattr(
        physical_input_module,
        "_user32",
        lambda: _fake_user32({"a": 0x41, "b": 0x42}),
    )

    def sender(events: list[dict[str, int]]) -> int:
        nonlocal calls
        calls += 1
        if calls == 3:
            return 0
        return len(events)

    with pytest.raises(ComputerError) as partial:
        Win32PhysicalInput(event_sender=sender).type_text(
            backend,  # type: ignore[arg-type]
            identity,
            40,
            50,
            "ab",
            mutation_guard=lambda: None,
        )

    assert partial.value.code == "physical_text_partial"
    assert partial.value.details["delivered_character_count"] == 1
    assert partial.value.details["first_undelivered_position"] == 1
    assert partial.value.details["automatic_cleanup_performed"] is False
    assert partial.value.details["stage_valid"] is False


def test_enter_is_a_separate_two_event_delivery() -> None:
    identity = _current_thread_identity()
    backend = _PhysicalBackend(identity)
    batches: list[list[dict[str, int]]] = []
    result = Win32PhysicalInput(
        event_sender=lambda events: batches.append(events) or len(events)
    ).press_enter(backend, identity)  # type: ignore[arg-type]

    assert result["events_intended"] == 2
    assert [event["vk"] for event in batches[0]] == [0x0D, 0x0D]
    assert [event["flags"] for event in batches[0]] == [0, 0x0002]


@pytest.mark.parametrize("shortcut", ["ALT+F4", "CTRL+ALT+DELETE", "CTRL+SHIFT+ESC", "WIN+L"])
def test_system_global_shortcuts_are_rejected(shortcut: str) -> None:
    with pytest.raises(ComputerError) as error:
        parse_shortcut(shortcut)
    assert error.value.code in {"forbidden_shortcut", "invalid_shortcut"}


@pytest.mark.parametrize(
    "shortcut",
    ["ENTER", "CTRL+ENTER", "SHIFT+ENTER", "ALT+ENTER", "ctrl+shift+enter"],
)
def test_shortcut_rejects_every_enter_bearing_request(shortcut: str) -> None:
    with pytest.raises(ComputerError) as error:
        parse_shortcut(shortcut)

    assert error.value.code == "forbidden_shortcut"
    assert "staged-input key transaction" in error.value.message


def test_allowed_shortcut_has_one_non_modifier() -> None:
    modifiers, virtual_key, normalized = parse_shortcut("ctrl+shift+k")
    assert modifiers == ["CTRL", "SHIFT"]
    assert virtual_key == ord("K")
    assert normalized == "CTRL+SHIFT+K"


class _FakeMutex:
    def try_acquire(self) -> NativeMutexLease:
        return NativeMutexLease(1, True)

    def release(self, _lease: NativeMutexLease) -> None:
        return None


def test_action_coordinator_returns_delivery_and_consequence_on_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(actions.STATE_DIR_ENV, str(tmp_path / "state"))
    coordinator = ActionCoordinator(_FakeMutex())  # type: ignore[arg-type]

    with pytest.raises(ComputerError) as error:
        coordinator.run(
            operation="fixture",
            target_ids={"hwnd": 123},
            short_explanation=None,
            perform=lambda _execution: (_ for _ in ()).throw(
                ComputerError("fixture_rejected", "No delivery", details={"outcome": "rejected"})
            ),
        )

    assert error.value.details["delivery"]["status"] == "rejected"
    assert error.value.details["consequence"]["observed_effect"] == "ambiguous"
    assert error.value.details["phase_timings_ms"]["total"] >= 0


def test_capture_consumption_failure_is_attempted_once_and_preserves_delivery() -> None:
    class RecordStore:
        calls = 0

        def delete(self, _capture_id: str) -> None:
            self.calls += 1
            raise ComputerError("capture_store_unavailable", "fixture failure")

    store = RecordStore()
    delivery_error = ComputerError(
        "physical_text_partial",
        "fixture partial delivery",
        details={
            "outcome": "delivery_only",
            "method": "win32.SendInput.VkKeyScanW",
            "partial_delivery": True,
            "intended_character_count": 4,
            "delivered_character_count": 2,
            "first_undelivered_position": 2,
        },
    )

    with pytest.raises(ComputerError) as failed:
        actions._consume_physical_capture(
            store,
            "cap_" + "a" * 32,
            delivered=True,
            delivery_error=delivery_error,
        )

    assert store.calls == 1
    assert failed.value.code == "physical_capture_consumption_failed"
    assert failed.value.details["capture_consumed"] is False
    assert failed.value.details["delivery_error_code"] == "physical_text_partial"
    assert failed.value.details["delivered_character_count"] == 2


def test_stage_store_failure_after_delivery_preserves_full_delivery_evidence() -> None:
    stage_error = ComputerError(
        "staged_input_store_unavailable",
        "fixture stage-store failure",
        details={"outcome": "rejected"},
    )
    delivery_result = {
        "outcome": "delivery_only",
        "method": "win32.SendInput.VkKeyScanW",
        "events_intended": 8,
        "events_delivered": 8,
        "intended_character_count": 4,
        "delivered_character_count": 4,
    }

    error = actions._error_after_physical_delivery(stage_error, delivery_result)

    assert error.code == "staged_input_store_unavailable"
    assert error.details["outcome"] == "delivery_only"
    assert error.details["method"] == "win32.SendInput.VkKeyScanW"
    assert error.details["events_delivered"] == 8
    assert error.details["delivered_character_count"] == 4
    assert actions._error_has_physical_delivery(error) is True


def hashlib_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()
