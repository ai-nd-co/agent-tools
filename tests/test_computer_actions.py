from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_tools import cli as root_cli
from agent_tools.computer import actions, notification_worker
from agent_tools.computer import cli as computer_cli
from agent_tools.computer.actions import (
    ActionCoordinator,
    NativeMutexLease,
    focus_window,
    invoke_element,
    notify_user,
    resize_window,
    scroll_element,
    set_element_value,
    set_window_state,
    validate_short_explanation,
)
from agent_tools.computer.models import Bounds, ComputerError, WindowIdentity
from agent_tools.computer.rendering import render_action, render_human_error, render_json_error


def _identity() -> WindowIdentity:
    return WindowIdentity(
        hwnd=123,
        pid=456,
        thread_id=789,
        process="fixture.exe",
        title="Task-owned fixture",
        class_name="FixtureClass",
        bounds=Bounds(10, 20, 400, 300),
        process_started=999,
        session_id=1,
        desktop_name="Default",
    )


class FakeMutex:
    def __init__(self, *, acquired: bool = True, abandoned: bool = False) -> None:
        self.lease = NativeMutexLease(42 if acquired else 0, acquired, abandoned)
        self.released = False

    def try_acquire(self) -> NativeMutexLease:
        return self.lease

    def release(self, lease: NativeMutexLease) -> None:
        assert lease is self.lease
        self.released = True


class FakeActionBackend:
    def __init__(self) -> None:
        self.identity = _identity()
        self.events: list[str] = []

    def capture_identity(self, hwnd: int) -> WindowIdentity:
        assert hwnd == self.identity.hwnd
        self.events.append("capture")
        return self.identity

    def assert_action_allowed(self, identity: WindowIdentity) -> WindowIdentity:
        assert identity is self.identity
        self.events.append("security")
        return identity

    def revalidate_identity(self, identity: WindowIdentity) -> WindowIdentity:
        assert identity is self.identity
        self.events.append("revalidate")
        return identity

    def focus_window(self, identity: WindowIdentity, **_kwargs) -> dict[str, object]:
        assert identity is self.identity
        self.events.append("focus")
        return {
            "window": identity.public(),
            "method": "win32.SetForegroundWindow",
            "outcome": "postcondition_verified",
            "postcondition_verified": True,
        }


@pytest.fixture
def action_state(monkeypatch, tmp_path: Path) -> Path:
    state = tmp_path / "state"
    monkeypatch.setenv(actions.STATE_DIR_ENV, str(state))
    monkeypatch.delenv(actions.ACTIONS_DISABLED_ENV, raising=False)
    return state


def test_coordinator_journal_is_redacted_and_releases_lock(action_state: Path) -> None:
    mutex = FakeMutex()
    coordinator = ActionCoordinator(mutex)  # type: ignore[arg-type]

    result = coordinator.run(
        operation="set-value",
        target_ids={"hwnd": 123, "element": "edit-secret-a1b2", "value": "never"},
        short_explanation="Update the disposable fixture",
        perform=lambda _execution: {
            "method": "uia.ValuePattern",
            "outcome": "postcondition_verified",
            "value_char_count": 18,
            "resolved_element": "edit-secret-a1b2",
        },
    )

    assert result["outcome"] == "postcondition_verified"
    assert result["short_explanation"] == "Update the disposable fixture"
    assert set(result["phase_timings_ms"]) == {"total", "unattributed"}
    assert result["phase_timings_ms"]["total"] >= 0
    assert result["phase_timings_ms"]["unattributed"] == result["phase_timings_ms"]["total"]
    assert mutex.released is True
    rows = [
        json.loads(line)
        for line in (action_state / actions.JOURNAL_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["outcome"] for row in rows] == ["started", "postcondition_verified"]
    serialized = json.dumps(rows)
    assert "Update the disposable fixture" not in serialized
    assert "never" not in serialized
    assert "value" not in rows[-1]["target_ids"]
    assert "element" not in rows[-1]["target_ids"]
    assert len(rows[-1]["target_ids"]["element_digest"]) == 24
    assert list(action_state.glob(f"{actions.OWNER_FILE}.*.json")) == []


def test_before_input_generation_failure_preserves_generation_and_skips_action(
    monkeypatch,
    action_state: Path,
) -> None:
    actions._ensure_state_ready()
    generation_before = actions._read_input_generation()
    performed = False

    def reject_before_generation(_execution) -> None:
        raise ComputerError("computer_action_worker_busy", "fixture busy")

    def perform(_execution):
        nonlocal performed
        performed = True
        return {"method": "fixture", "outcome": "postcondition_verified"}

    with pytest.raises(ComputerError) as raised:
        ActionCoordinator(FakeMutex(), FakeMutex()).run(  # type: ignore[arg-type]
            operation="key",
            target_ids={"hwnd": 123},
            short_explanation="Submit fixture text",
            perform=perform,
            before_input_generation=reject_before_generation,
        )

    assert raised.value.code == "computer_action_worker_busy"
    assert performed is False
    assert actions._read_input_generation() == generation_before


def test_staged_enter_notification_busy_preserves_authority(
    monkeypatch,
    action_state: Path,
) -> None:
    actions._ensure_state_ready()
    generation_before = actions._read_input_generation()
    monkeypatch.setattr(
        actions,
        "show_notification",
        lambda **_kwargs: {"requested": True, "status": "worker_busy"},
    )

    class StageStore:
        @staticmethod
        def consume_for_enter(*_args, **_kwargs):
            pytest.fail("staged authority was consumed after notification-busy")

    with pytest.raises(ComputerError) as raised:
        actions.press_staged_enter(
            hwnd=123,
            capture_id="cap_" + "a" * 32,
            image_x=1,
            image_y=1,
            staged_input="stage_" + "b" * 32,
            draft_evidence="exact",
            allow_physical=True,
            visible_tab_verified=True,
            short_explanation="Submit fixture text",
            backend=object(),  # type: ignore[arg-type]
            coordinator=ActionCoordinator(FakeMutex(), FakeMutex()),  # type: ignore[arg-type]
            record_store=object(),
            stage_store=StageStore(),  # type: ignore[arg-type]
            physical=object(),  # type: ignore[arg-type]
        )

    assert raised.value.code == "computer_action_worker_busy"
    assert actions._read_input_generation() == generation_before


def test_phase_timings_are_additive_and_exclude_nested_totals(monkeypatch) -> None:
    execution = actions.ActionExecution("fixture", "invoke", None)
    execution.phase_timings_ms["focus"] = 10.0
    nested_before = actions._nested_delivery_phase_total(execution)
    execution.phase_timings_ms["focus"] = 25.0
    monkeypatch.setattr(actions.time, "perf_counter", lambda: 1.030)

    actions._record_exclusive_delivery_phase(execution, 1.0, nested_before)
    actions._record_consequence_timings(
        execution,
        {
            "phase_timings_ms": {
                "settle_wait": 2_000.0,
                "after_capture": 5.0,
                "compare": 2.0,
                "total": 2_007.0,
            }
        },
    )
    public = actions._public_phase_timings(execution.phase_timings_ms, total=2_050.0)

    assert public["delivery"] == 15.0
    assert "consequence_total" not in public
    leaf_sum = sum(value for key, value in public.items() if key != "total")
    assert abs(leaf_sum - public["total"]) <= 0.001


def test_release_failure_preserves_verified_result(action_state: Path) -> None:
    class ReleaseFailMutex(FakeMutex):
        def release(self, lease: NativeMutexLease) -> None:
            super().release(lease)
            raise ComputerError(
                "computer_action_lock_release_failed",
                "fixture release failure",
            )

    result = ActionCoordinator(ReleaseFailMutex()).run(  # type: ignore[arg-type]
        operation="focus",
        target_ids={"hwnd": 123},
        short_explanation=None,
        perform=lambda _execution: {
            "method": "win32.SetForegroundWindow",
            "outcome": "postcondition_verified",
        },
    )

    assert result["outcome"] == "postcondition_verified"
    assert result["lock"] == {"status": "release_failed"}
    assert "computer_action_lock_release_failed" in result["warnings"]


def test_final_cleanup_warnings_remain_bounded_and_prioritized_on_success(
    monkeypatch,
    action_state: Path,
) -> None:
    class ReleaseFailMutex(FakeMutex):
        def release(self, lease: NativeMutexLease) -> None:
            super().release(lease)
            raise ComputerError(
                "computer_action_lock_release_failed",
                "fixture release failure",
            )

    monkeypatch.setattr(actions, "_clear_owner_metadata", lambda _operation_id: False)

    def succeed(execution):
        execution.warnings.extend(f"fixture_warning_{index:02d}" for index in range(16))
        return {
            "method": "win32.SetForegroundWindow",
            "outcome": "postcondition_verified",
        }

    result = ActionCoordinator(ReleaseFailMutex()).run(  # type: ignore[arg-type]
        operation="focus",
        target_ids={"hwnd": 123},
        short_explanation=None,
        perform=succeed,
    )

    assert len(result["warnings"]) == actions.MAX_ACTION_WARNING_COUNT
    assert "computer_action_lock_release_failed" in result["warnings"]
    assert "computer_action_owner_cleanup_failed" in result["warnings"]
    assert result["lock"] == {"status": "release_failed"}


def test_final_cleanup_warnings_remain_bounded_and_prioritized_on_error(
    monkeypatch,
    action_state: Path,
) -> None:
    class ReleaseFailMutex(FakeMutex):
        def release(self, lease: NativeMutexLease) -> None:
            super().release(lease)
            raise ComputerError(
                "computer_action_lock_release_failed",
                "fixture release failure",
            )

    monkeypatch.setattr(actions, "_clear_owner_metadata", lambda _operation_id: False)
    backend_warnings = [f"fixture_warning_{index:02d}" for index in range(16)]

    with pytest.raises(ComputerError) as raised:
        ActionCoordinator(ReleaseFailMutex()).run(  # type: ignore[arg-type]
            operation="focus",
            target_ids={"hwnd": 123},
            short_explanation=None,
            perform=lambda _execution: (_ for _ in ()).throw(
                ComputerError(
                    "stale_window",
                    "fixture drift",
                    details={"warnings": backend_warnings},
                )
            ),
        )

    public_warnings = raised.value.details["warnings"]
    assert len(public_warnings) == actions.MAX_ACTION_WARNING_COUNT
    assert "computer_action_lock_release_failed" in public_warnings
    assert "computer_action_owner_cleanup_failed" in public_warnings
    assert raised.value.details["lock"] == {"status": "release_failed"}


def test_success_journal_warning_is_bounded_and_backend_cannot_replace_envelope(
    monkeypatch,
    action_state: Path,
) -> None:
    journal_calls = 0

    def append_journal(**_kwargs) -> None:
        nonlocal journal_calls
        journal_calls += 1
        if journal_calls == 2:
            raise ComputerError(
                "computer_action_journal_failed",
                "fixture completion journal failure",
            )

    monkeypatch.setattr(actions, "_append_journal", append_journal)

    def succeed(execution):
        execution.warnings.extend(f"fixture_warning_{index:02d}" for index in range(16))
        return {
            "method": "win32.SetForegroundWindow",
            "outcome": "postcondition_verified",
            "warnings": [
                "backend_warning",
                "x" * (actions.MAX_ACTION_WARNING_CHARS + 1),
            ],
            "operation_id": "backend-controlled-id",
            "operation": "backend-controlled-operation",
            "notification": {"status": "backend-controlled"},
            "duration_ms": -1,
            "journal": {"status": "backend-controlled"},
        }

    result = ActionCoordinator(FakeMutex()).run(  # type: ignore[arg-type]
        operation="focus",
        target_ids={"hwnd": 123},
        short_explanation=None,
        perform=succeed,
    )

    assert journal_calls == 2
    assert len(result["warnings"]) == actions.MAX_ACTION_WARNING_COUNT
    assert "journal_completion_write_failed" in result["warnings"]
    assert all(len(warning) <= actions.MAX_ACTION_WARNING_CHARS for warning in result["warnings"])
    assert result["operation_id"] != "backend-controlled-id"
    assert result["operation"] == "focus"
    assert result["notification"] == {"requested": False, "status": "not_requested"}
    assert result["duration_ms"] >= 0
    assert result["journal"] == {"status": "completion_write_failed"}


def test_unexpected_backend_exception_has_safe_action_envelope(action_state: Path) -> None:
    def fail(_execution):
        raise ValueError("private backend detail")

    with pytest.raises(ComputerError) as raised:
        ActionCoordinator(FakeMutex()).run(  # type: ignore[arg-type]
            operation="invoke",
            target_ids={"hwnd": 123},
            short_explanation=None,
            perform=fail,
        )

    assert raised.value.code == "action_backend_failed"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["method"] == "unknown"
    assert "private backend detail" not in raised.value.message


def test_busy_rejects_without_queue_and_reports_bounded_owner(
    monkeypatch, action_state: Path
) -> None:
    mutex = FakeMutex(acquired=False)
    monkeypatch.setattr(
        actions,
        "_read_owner_metadata",
        lambda: {
            "owner": "worker-a",
            "conversation_alias": "terminal-one",
            "operation": "focus",
            "pid": 100,
        },
    )

    with pytest.raises(ComputerError) as raised:
        ActionCoordinator(mutex).run(  # type: ignore[arg-type]
            operation="resize",
            target_ids={"hwnd": 123},
            short_explanation=None,
            perform=lambda _execution: pytest.fail("busy actions must not execute"),
        )

    assert raised.value.code == "computer_action_busy"
    assert raised.value.details["outcome"] == "rejected"
    assert raised.value.details["current_owner"]["conversation_alias"] == "terminal-one"
    assert (action_state / actions.JOURNAL_FILE).is_file()


def test_busy_owner_retry_selects_newest_publication(monkeypatch) -> None:
    records = iter(
        [
            {
                "owner": "abandoned-owner",
                "conversation_alias": "old-terminal",
                "operation": "old-action",
                "pid": 100,
                "_published_at_unix_ns": 1,
            },
            {
                "owner": "current-owner",
                "conversation_alias": "new-terminal",
                "operation": "current-action",
                "pid": 200,
                "_published_at_unix_ns": 2,
            },
        ]
    )
    latest: dict[str, object] | None = None

    def read():
        nonlocal latest
        try:
            latest = next(records)
        except StopIteration:
            pass
        return latest

    monkeypatch.setattr(actions, "_read_owner_metadata", read)
    owner = actions._read_owner_metadata_with_retry(timeout_seconds=0.03)

    assert owner is not None
    assert owner["owner"] == "current-owner"
    assert "_published_at_unix_ns" not in owner


def test_dead_owner_process_identity_is_never_reported(action_state: Path) -> None:
    action_state.mkdir(parents=True)
    actions._owner_path("dead-operation").write_text(
        json.dumps(
            {
                "operation_id": "dead-operation",
                "operation": "invoke",
                "owner": "dead-owner",
                "conversation_alias": "dead-terminal",
                "pid": 2_147_000_000,
                "process_started": 1,
                "published_at_unix_ns": 1,
            }
        ),
        encoding="ascii",
    )

    assert actions._read_owner_metadata() is None


def test_failed_owner_publication_removes_temporary_record(monkeypatch, action_state: Path) -> None:
    action_state.mkdir(parents=True)

    def fail_replace(*_args):
        raise OSError("fixture publication failure")

    monkeypatch.setattr(actions.os, "replace", fail_replace)

    with pytest.raises(ComputerError) as raised:
        actions._write_owner_metadata("failed-operation", "focus")

    assert raised.value.code == "computer_action_state_unavailable"
    assert list(action_state.glob(f".{actions.OWNER_FILE}*.tmp")) == []


def test_environment_and_config_marker_emergency_disable_are_fail_closed(
    monkeypatch, action_state: Path
) -> None:
    monkeypatch.setenv(actions.ACTIONS_DISABLED_ENV, "true")
    assert actions.action_capabilities()["disabled"] is True
    assert "environment" in actions.action_capabilities()["disable_sources"]
    with pytest.raises(ComputerError) as environment_error:
        ActionCoordinator(FakeMutex()).run(  # type: ignore[arg-type]
            operation="focus",
            target_ids={"hwnd": 123},
            short_explanation=None,
            perform=lambda _execution: pytest.fail("disabled action executed"),
        )
    assert environment_error.value.code == "computer_actions_disabled"

    monkeypatch.delenv(actions.ACTIONS_DISABLED_ENV)
    action_state.mkdir(parents=True)
    (action_state / actions.DISABLE_MARKER).write_text("disabled\n", encoding="ascii")
    assert actions.action_capabilities()["disable_sources"] == ["config_marker"]


def test_emergency_disable_after_notification_rejects_before_backend_mutation(
    monkeypatch, action_state: Path
) -> None:
    backend = FakeActionBackend()

    def disable_during_notification(**_kwargs):
        (action_state / actions.DISABLE_MARKER).write_text("disabled\n", encoding="ascii")
        return {"requested": True, "status": "queued"}

    monkeypatch.setattr(actions, "show_notification", disable_during_notification)

    with pytest.raises(ComputerError) as raised:
        focus_window(
            hwnd=123,
            short_explanation="Focus the disposable fixture",
            backend=backend,  # type: ignore[arg-type]
            coordinator=ActionCoordinator(FakeMutex()),  # type: ignore[arg-type]
        )

    assert raised.value.code == "computer_actions_disabled"
    assert raised.value.details["outcome"] == "rejected"
    assert backend.events == ["capture", "security"]


def test_stale_window_rejects_before_backend_mutation(action_state: Path) -> None:
    backend = FakeActionBackend()

    def stale(_identity: WindowIdentity) -> WindowIdentity:
        backend.events.append("stale")
        raise ComputerError("stale_window", "fixture changed")

    backend.assert_action_allowed = stale  # type: ignore[method-assign]

    with pytest.raises(ComputerError) as raised:
        focus_window(
            hwnd=123,
            short_explanation=None,
            backend=backend,  # type: ignore[arg-type]
            coordinator=ActionCoordinator(FakeMutex()),  # type: ignore[arg-type]
        )

    assert raised.value.code == "stale_window"
    assert raised.value.details["outcome"] == "rejected"
    assert backend.events == ["capture", "stale"]


def test_public_focus_preserves_backend_cleanup_warning_and_renders_status(
    monkeypatch,
    action_state: Path,
) -> None:
    class CleanupFailureBackend(FakeActionBackend):
        def focus_window(self, identity: WindowIdentity, **_kwargs) -> dict[str, object]:
            assert identity is self.identity
            self.events.append("focus")
            raise ComputerError(
                "computer_actions_disabled",
                "fixture stop after input attachment",
                details={
                    "outcome": "delivery_only",
                    "method": "win32.AttachThreadInput",
                    "partial_mutation": "input_attachment",
                    "warnings": [
                        "input_attachment_cleanup_failed",
                        "x" * (actions.MAX_ACTION_WARNING_CHARS + 1),
                    ],
                    "input_attachment_cleanup": {"status": "failed", "count": 1},
                },
            )

    backend = CleanupFailureBackend()
    monkeypatch.setattr(
        actions,
        "show_notification",
        lambda **_kwargs: {"requested": True, "status": "unavailable"},
    )

    with pytest.raises(ComputerError) as raised:
        focus_window(
            hwnd=123,
            short_explanation="Focus the disposable fixture",
            backend=backend,  # type: ignore[arg-type]
            coordinator=ActionCoordinator(FakeMutex()),  # type: ignore[arg-type]
        )

    assert raised.value.details["warnings"] == [
        "input_attachment_cleanup_failed",
        "notification_unavailable",
    ]
    assert raised.value.details["input_attachment_cleanup"] == {
        "status": "failed",
        "count": 1,
    }
    human = render_human_error(raised.value)
    assert "warnings=input_attachment_cleanup_failed,notification_unavailable" in human
    assert "Input attachment cleanup: status=failed count=1" in human


def test_ambiguous_element_pre_delivery_maps_to_rejected(action_state: Path) -> None:
    with pytest.raises(ComputerError) as raised:
        ActionCoordinator(FakeMutex()).run(  # type: ignore[arg-type]
            operation="invoke",
            target_ids={"hwnd": 123},
            short_explanation=None,
            perform=lambda _execution: (_ for _ in ()).throw(
                ComputerError("ambiguous_element", "fixture duplicate")
            ),
        )

    assert raised.value.code == "ambiguous_element"
    assert raised.value.details["outcome"] == "rejected"


def test_standalone_notification_reports_truthful_delivery_only(
    monkeypatch, action_state: Path
) -> None:
    desktop_checks: list[bool] = []
    monkeypatch.setattr(
        actions.Win32Backend,
        "assert_desktop_action_allowed",
        lambda _self: desktop_checks.append(True),
    )
    monkeypatch.setattr(
        actions,
        "show_notification",
        lambda **_kwargs: {"requested": True, "status": "queued"},
    )

    result = notify_user(
        short_explanation="Validate the disposable fixture",
        title="AgentTools test",
        coordinator=ActionCoordinator(FakeMutex()),  # type: ignore[arg-type]
    )

    assert result["outcome"] == "delivery_only"
    assert result["notification"] == {"requested": True, "status": "queued"}
    assert result["method"] == "windows.Shell_NotifyIcon"
    assert desktop_checks == [True]


def test_standalone_notification_rejects_unusable_desktop_before_delivery(
    monkeypatch,
    action_state: Path,
) -> None:
    deliveries: list[bool] = []

    def reject_desktop(_self) -> None:
        raise ComputerError(
            "secure_desktop_unavailable",
            "Computer actions require the unlocked default interactive desktop.",
        )

    monkeypatch.setattr(
        actions.Win32Backend,
        "assert_desktop_action_allowed",
        reject_desktop,
    )
    monkeypatch.setattr(
        actions,
        "show_notification",
        lambda **_kwargs: deliveries.append(True) or {"requested": True, "status": "queued"},
    )

    with pytest.raises(ComputerError) as raised:
        notify_user(
            short_explanation="Validate the disposable fixture",
            title="AgentTools test",
            coordinator=ActionCoordinator(FakeMutex()),  # type: ignore[arg-type]
        )

    assert raised.value.code == "secure_desktop_unavailable"
    assert deliveries == []


def test_verified_action_survives_completion_journal_failure(
    monkeypatch, action_state: Path
) -> None:
    real_append = actions._append_journal
    writes = 0

    def fail_completion(**kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise ComputerError("computer_action_journal_failed", "fixture failure")
        return real_append(**kwargs)

    monkeypatch.setattr(actions, "_append_journal", fail_completion)
    result = ActionCoordinator(FakeMutex()).run(  # type: ignore[arg-type]
        operation="focus",
        target_ids={"hwnd": 123},
        short_explanation=None,
        perform=lambda _execution: {
            "method": "win32.SetForegroundWindow",
            "outcome": "postcondition_verified",
            "postcondition_verified": True,
        },
    )

    assert result["outcome"] == "postcondition_verified"
    assert result["postcondition_verified"] is True
    assert result["journal"] == {"status": "completion_write_failed"}
    assert "journal_completion_write_failed" in result["warnings"]


def test_pre_delivery_element_failures_are_rejected(action_state: Path) -> None:
    with pytest.raises(ComputerError) as raised:
        ActionCoordinator(FakeMutex()).run(  # type: ignore[arg-type]
            operation="invoke",
            target_ids={"hwnd": 123},
            short_explanation=None,
            perform=lambda _execution: (_ for _ in ()).throw(
                ComputerError("stale_element", "fixture changed")
            ),
        )

    assert raised.value.details["outcome"] == "rejected"


@pytest.mark.parametrize(
    "code",
    [
        "invalid_element_ref",
        "element_ref_missing",
        "element_ref_expired",
        "element_ref_altered",
        "element_ref_invalid",
        "semantic_provider_action_unsupported",
    ],
)
def test_pre_delivery_reference_and_provider_failures_are_rejected(
    code: str,
    action_state: Path,
) -> None:
    with pytest.raises(ComputerError) as raised:
        ActionCoordinator(FakeMutex()).run(  # type: ignore[arg-type]
            operation="invoke",
            target_ids={"hwnd": 123},
            short_explanation=None,
            perform=lambda _execution: (_ for _ in ()).throw(
                ComputerError(code, "fixture rejected")
            ),
        )

    assert raised.value.details["outcome"] == "rejected"


def test_human_action_error_preserves_operation_and_method_metadata() -> None:
    output = render_human_error(
        ComputerError(
            "focus_postcondition_failed",
            "fixture did not focus",
            details={
                "operation_id": "op-fixture",
                "operation": "focus",
                "outcome": "failed",
                "method": "win32.SetForegroundWindow",
                "notification": {"status": "queued"},
                "warnings": ["fixture-warning"],
            },
        )
    )

    assert "id=op-fixture" in output
    assert "outcome=failed" in output
    assert "method=win32.SetForegroundWindow" in output
    assert "Notification: queued" in output


def test_human_action_and_error_render_phase_a_contract_fields() -> None:
    action_output = render_action(
        {
            "action": {
                "operation": "resize",
                "operation_id": "op-phase-a",
                "outcome": "postcondition_verified",
                "method": "win32.SetWindowPos",
                "control_tier": "windows_api",
                "requested_bounds": {"x": 10, "y": 20, "width": 800, "height": 600},
                "actual_bounds": {"x": 10, "y": 20, "width": 800, "height": 600},
                "restore_first": True,
                "restored_first": True,
                "postcondition_verified": True,
                "resolution_stage": "stable_metadata_unique",
                "fallback_used": False,
            }
        }
    )
    assert "control tier: windows_api" in action_output
    assert "restore-first=yes" in action_output
    assert "restored-first=yes" in action_output
    assert "resolution stage: stable_metadata_unique" in action_output
    assert "fallback used: no" in action_output

    error_output = render_human_error(
        ComputerError(
            "stale_window",
            "fixture changed",
            details={
                "identity_mismatch_fields": ["session_id"],
                "resolution_stage": "element_ref_window_identity",
                "requested_state": "normal",
                "state_before": "maximized",
                "state_after": "normal",
                "requested_bounds": {"x": 10, "y": 20, "width": 800, "height": 600},
                "restore_performed": True,
                "restore_method": "win32.ShowWindow(SW_RESTORE)",
                "postcondition_verified": False,
                "fallback": {
                    "kind": "guarded_physical_input",
                    "status": "available",
                    "reason": "requires_fresh_capture_and_explicit_allow_physical",
                },
            },
        )
    )
    assert "Mismatch fields: session_id" in error_output
    assert "Resolution stage: element_ref_window_identity" in error_output
    assert "State: requested=normal before=maximized after=normal verified=no" in error_output
    assert "Bounds: requested=10,20 800x600 actual=- verified=no" in error_output
    assert "Restore: performed=yes" in error_output
    assert "status=available" in error_output


def test_human_scroll_capability_error_renders_guarded_physical_fallback() -> None:
    output = render_human_error(
        ComputerError(
            "scroll_axis_unavailable",
            "fixture axis unavailable",
            details={
                "fallback": {
                    "kind": "guarded_physical_input",
                    "status": "available",
                    "reason": "requires_fresh_capture_and_explicit_allow_physical",
                }
            },
        )
    )

    assert "Fallback: guarded_physical_input" in output
    assert "status=available" in output
    assert "requires_fresh_capture_and_explicit_allow_physical" in output


def test_human_focus_output_renders_optional_state_and_restore_result() -> None:
    output = render_action(
        {
            "action": {
                "operation": "focus",
                "operation_id": "op-focus",
                "outcome": "postcondition_verified",
                "method": "win32.SetForegroundWindow",
                "control_tier": "windows_api",
                "state_before": "minimized",
                "state_after": "normal",
                "restore_performed": False,
                "restored_to_normal": True,
                "postcondition_verified": True,
            }
        }
    )

    assert "state: requested=- before=minimized after=normal verified=yes" in output
    assert "restore: performed=no normal=yes" in output


def test_human_semantic_output_renders_nested_focus_and_partial_error() -> None:
    focus = {
        "outcome": "postcondition_verified",
        "method": "win32.SetForegroundWindow",
        "foreground_hwnd": 123,
        "state_before": "minimized",
        "state_after": "normal",
        "restore_performed": True,
        "restored_to_normal": True,
        "postcondition_verified": True,
    }
    action_output = render_action(
        {
            "action": {
                "operation": "invoke",
                "operation_id": "op-semantic-focus",
                "outcome": "delivery_only",
                "method": "uia.InvokePattern",
                "control_tier": "uia_accessibility_provider",
                "focus": focus,
            }
        }
    )
    error_output = render_human_error(
        ComputerError(
            "stale_element",
            "fixture element changed",
            details={
                "operation_id": "op-semantic-focus",
                "operation": "invoke",
                "outcome": "delivery_only",
                "method": "win32.SetForegroundWindow",
                "focus": focus,
                "semantic_outcome": "rejected",
                "semantic_method": "uia.InvokePattern",
            },
        )
    )

    assert (
        "focus: outcome=postcondition_verified method=win32.SetForegroundWindow "
        "foreground=123 state=minimized->normal verified=yes" in action_output
    )
    assert "focus restore: performed=yes normal=yes" in action_output
    assert "Focus side effect: outcome=postcondition_verified" in error_output
    assert "Focus side effect restore: performed=yes normal=yes" in error_output
    assert "Semantic action: outcome=rejected method=uia.InvokePattern" in error_output

    direct_focus_error = render_human_error(
        ComputerError(
            "focus_postcondition_failed",
            "fixture foreground stolen",
            details={
                "state_before": "minimized",
                "state_after": "normal",
                "restore_performed": True,
                "restored_to_normal": True,
                "postcondition_verified": False,
            },
        )
    )
    assert "Focus restore: performed=yes normal=yes" in direct_focus_error
    assert "restored-first" not in direct_focus_error


def test_notification_failure_is_warning_not_action_permission_gate(
    monkeypatch, action_state: Path
) -> None:
    backend = FakeActionBackend()
    mutex = FakeMutex()
    monkeypatch.setattr(
        actions,
        "show_notification",
        lambda **_kwargs: {
            "requested": True,
            "status": "unavailable",
            "warning": "fixture",
        },
    )

    result = focus_window(
        hwnd=123,
        short_explanation="Focus the disposable fixture",
        backend=backend,  # type: ignore[arg-type]
        coordinator=ActionCoordinator(mutex),  # type: ignore[arg-type]
    )

    assert result["outcome"] == "postcondition_verified"
    assert result["warnings"] == ["notification_unavailable"]
    assert backend.events == [
        "capture",
        "security",
        "security",
        "capture",  # bounded consequence snapshot before delivery
        "focus",
        "revalidate",
        "capture",  # bounded consequence snapshot after delivery
    ]


def test_semantic_action_focus_and_native_delivery_share_one_atomic_operation(
    action_state: Path,
) -> None:
    backend = FakeActionBackend()

    class ReferenceStore:
        def resolve(self, _reference, *, identity, require_title_match=False):
            assert identity == backend.identity
            assert require_title_match is False
            backend.events.append("resolve-ref")
            return {
                "element_ref": "eref_" + "a" * 32,
                "expires_unix_ms": 2_000_000,
                "expires_at": "1970-01-01T00:33:20Z",
                "element": {
                    "provider": "winapp",
                    "selector": "button-fixture-a1b2",
                    "runtime_id": "42,1",
                    "automation_id": "FixtureButton",
                    "control_type": "Button",
                    "class": "Button",
                    "ancestor_signature": "sha256:" + "a" * 24,
                    "bounds": {"x": 10, "y": 20, "width": 80, "height": 24},
                    "enabled": True,
                    "read_only": False,
                    "password": False,
                },
            }

    class Adapter:
        def invoke(self, identity, **kwargs):
            assert identity == backend.identity
            assert kwargs["element_reference"]["expires_unix_ms"] == 2_000_000
            backend.events.append("native-action")
            binding = SimpleNamespace(public=lambda: {"name": "winapp"})
            return binding, {
                "method": "uia.InvokePattern",
                "outcome": "postcondition_verified",
                "postcondition_verified": True,
                "resolved_element": "button-fixture-a1b2",
            }

    result = invoke_element(
        hwnd=123,
        element=None,
        element_ref="eref_" + "a" * 32,
        short_explanation=None,
        backend=backend,  # type: ignore[arg-type]
        adapter=Adapter(),  # type: ignore[arg-type]
        coordinator=ActionCoordinator(FakeMutex()),  # type: ignore[arg-type]
        element_ref_store=ReferenceStore(),  # type: ignore[arg-type]
    )

    assert result["outcome"] == "postcondition_verified"
    assert backend.events == [
        "capture",
        "security",
        "resolve-ref",
        "capture",  # bounded consequence snapshot before delivery
        "security",
        "focus",
        "security",
        "native-action",
        "revalidate",
        "capture",  # bounded consequence snapshot after delivery
    ]


def test_native_navigation_reference_skips_winapp_preflight_and_keeps_atomic_focus(
    action_state: Path,
) -> None:
    backend = FakeActionBackend()

    class ReferenceStore:
        def resolve(self, _reference, *, identity, require_title_match=False):
            assert identity == backend.identity
            assert require_title_match is False
            backend.events.append("resolve-ref")
            return {
                "element_ref": "eref_" + "b" * 32,
                "expires_unix_ms": 2_000_000_000_000,
                "expires_at": "2033-05-18T03:33:20Z",
                "element": {
                    "provider": "native_uia_raw",
                    "selector": "nuia-" + "c" * 24,
                    "runtime_id": "42,1",
                    "automation_id": "TaskNavigation",
                    "control_type": "ListItem",
                    "class": "ListViewItem",
                    "ancestor_signature": "sha256:" + "d" * 24,
                    "bounds": {"x": 10, "y": 20, "width": 80, "height": 24},
                    "enabled": True,
                    "read_only": False,
                    "password": False,
                },
            }

    class Adapter:
        def require(self):
            raise AssertionError("native navigation must not require WinApp")

        def invoke(self, identity, **kwargs):
            assert identity == backend.identity
            assert kwargs["element_reference"]["provider"] == "native_uia_raw"
            backend.events.append("native-selection")
            binding = SimpleNamespace(
                public=lambda: {"name": "native_uia_raw", "actions_exposed": True}
            )
            return binding, {
                "method": "uia.SelectionItemPattern",
                "outcome": "postcondition_verified",
                "postcondition_verified": True,
                "resolved_element": "nuia-" + "c" * 24,
            }

    result = invoke_element(
        hwnd=123,
        element=None,
        element_ref="eref_" + "b" * 32,
        short_explanation=None,
        backend=backend,  # type: ignore[arg-type]
        adapter=Adapter(),  # type: ignore[arg-type]
        coordinator=ActionCoordinator(FakeMutex()),  # type: ignore[arg-type]
        element_ref_store=ReferenceStore(),  # type: ignore[arg-type]
    )

    assert result["outcome"] == "postcondition_verified"
    assert result["backend"]["name"] == "native_uia_raw"
    assert backend.events == [
        "capture",
        "security",
        "resolve-ref",
        "capture",  # bounded consequence snapshot before delivery
        "security",
        "focus",
        "security",
        "native-selection",
        "revalidate",
        "capture",  # bounded consequence snapshot after delivery
    ]


@pytest.mark.parametrize("operation", ["invoke", "set-value", "scroll"])
def test_semantic_rejection_after_focus_reports_verified_partial_mutation(
    action_state: Path,
    operation: str,
) -> None:
    class Backend(FakeActionBackend):
        def focus_window(self, identity: WindowIdentity, **_kwargs) -> dict[str, object]:
            result = super().focus_window(identity)
            return {
                **result,
                "foreground_hwnd": identity.hwnd,
                "state_before": "minimized",
                "state_after": "normal",
                "restore_performed": True,
                "restored_to_normal": True,
                "input_attachments": {
                    "attached_count": 1,
                    "detached_count": 1,
                    "failed_count": 0,
                },
            }

    class Adapter:
        def invoke(self, _identity, **_kwargs):
            raise self._error()

        def set_value(self, _identity, **_kwargs):
            raise self._error()

        def scroll(self, _identity, **_kwargs):
            raise self._error()

        @staticmethod
        def _error() -> ComputerError:
            return ComputerError(
                "stale_element",
                "fixture element changed",
                details={
                    "outcome": "rejected",
                    "method": "uia.fixture",
                    "resolution_stage": "provider_metadata",
                },
            )

    backend = Backend()
    common = {
        "hwnd": 123,
        "element": "fixture-control",
        "short_explanation": None,
        "backend": backend,
        "adapter": Adapter(),
        "coordinator": ActionCoordinator(FakeMutex()),
    }

    with pytest.raises(ComputerError) as raised:
        if operation == "invoke":
            invoke_element(**common)  # type: ignore[arg-type]
        elif operation == "set-value":
            set_element_value(value="fixture", **common)  # type: ignore[arg-type]
        else:
            scroll_element(
                direction="down",
                to=None,
                percent=None,
                **common,  # type: ignore[arg-type]
            )

    details = raised.value.details
    assert raised.value.code == "stale_element"
    assert details["outcome"] == "delivery_only"
    assert details["method"] == "win32.SetForegroundWindow"
    assert details["partial_mutation"] == "focus"
    assert details["semantic_outcome"] == "rejected"
    assert details["semantic_method"] == "uia.fixture"
    assert details["focus"]["state_before"] == "minimized"
    assert details["focus"]["state_after"] == "normal"
    assert details["focus"]["restore_performed"] is True
    assert details["focus"]["restored_to_normal"] is True


def test_semantic_element_focus_side_effect_is_preserved_when_window_focus_is_noop() -> None:
    error = actions._semantic_error_after_focus(
        ComputerError(
            "stale_element",
            "fixture replaced on focus",
            details={
                "outcome": "delivery_only",
                "method": "uia.ValuePattern",
                "partial_mutation": "element_focus",
                "focus_delivery": "delivered",
                "value_delivery": "not_attempted",
            },
        ),
        {
            "method": "win32.SetForegroundWindow",
            "outcome": "postcondition_verified",
            "postcondition_verified": True,
            "foreground_hwnd": 123,
            "changed": False,
        },
    )

    assert error.details["outcome"] == "delivery_only"
    assert error.details["method"] == "uia.ValuePattern"
    assert error.details["partial_mutation"] == "element_focus"
    assert error.details["focus_delivery"] == "delivered"
    assert error.details["value_delivery"] == "not_attempted"
    assert error.details["semantic_outcome"] == "delivery_only"
    assert error.details["focus"]["changed"] is False


@pytest.mark.parametrize("operation", ["invoke", "set-value", "scroll"])
def test_semantic_backend_preflight_rejects_before_focus_with_fallback(
    action_state: Path,
    operation: str,
) -> None:
    backend = FakeActionBackend()

    class Adapter:
        def require(self):
            raise ComputerError("capability_missing", "fixture backend missing")

        def invoke(self, _identity, **_kwargs):
            pytest.fail("semantic delivery must not run")

    common = {
        "hwnd": 123,
        "element": "fixture-control",
        "short_explanation": None,
        "backend": backend,
        "adapter": Adapter(),
        "coordinator": ActionCoordinator(FakeMutex()),
    }
    with pytest.raises(ComputerError) as raised:
        if operation == "invoke":
            invoke_element(**common)  # type: ignore[arg-type]
        elif operation == "set-value":
            set_element_value(value="fixture", **common)  # type: ignore[arg-type]
        else:
            scroll_element(
                direction="down",
                to=None,
                percent=None,
                **common,  # type: ignore[arg-type]
            )

    assert raised.value.code == "capability_missing"
    assert raised.value.details["outcome"] == "rejected"
    assert raised.value.details["fallback"] == {
        "kind": "guarded_physical_input",
        "status": "available",
        "reason": "requires_fresh_capture_and_explicit_allow_physical",
    }
    assert "requires_fresh_capture_and_explicit_allow_physical" in render_json_error(
        f"computer {operation}", raised.value
    )
    assert "Fallback: guarded_physical_input" in render_human_error(raised.value)
    assert backend.events == ["capture", "security"]


@pytest.mark.parametrize("operation", ["invoke", "set-value", "scroll"])
def test_post_focus_security_rejection_preserves_verified_focus_side_effect(
    action_state: Path,
    operation: str,
) -> None:
    class Backend(FakeActionBackend):
        def __init__(self) -> None:
            super().__init__()
            self.security_checks = 0

        def assert_action_allowed(self, identity: WindowIdentity) -> WindowIdentity:
            self.security_checks += 1
            if self.security_checks == 3:
                raise ComputerError("stale_window", "fixture changed after focus")
            return super().assert_action_allowed(identity)

        def focus_window(self, identity: WindowIdentity, **_kwargs) -> dict[str, object]:
            result = super().focus_window(identity)
            return {
                **result,
                "foreground_hwnd": identity.hwnd,
                "state_before": "minimized",
                "state_after": "normal",
                "restore_performed": True,
                "restored_to_normal": True,
            }

    class Adapter:
        def invoke(self, _identity, **_kwargs):
            pytest.fail("semantic delivery must not run")

        def set_value(self, _identity, **_kwargs):
            pytest.fail("semantic delivery must not run")

        def scroll(self, _identity, **_kwargs):
            pytest.fail("semantic delivery must not run")

    backend = Backend()
    common = {
        "hwnd": 123,
        "element": "fixture-control",
        "short_explanation": None,
        "backend": backend,
        "adapter": Adapter(),
        "coordinator": ActionCoordinator(FakeMutex()),
    }

    with pytest.raises(ComputerError) as raised:
        if operation == "invoke":
            invoke_element(**common)  # type: ignore[arg-type]
        elif operation == "set-value":
            set_element_value(value="fixture", **common)  # type: ignore[arg-type]
        else:
            scroll_element(
                direction="down",
                to=None,
                percent=None,
                **common,  # type: ignore[arg-type]
            )

    assert raised.value.code == "stale_window"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["partial_mutation"] == "focus"
    assert raised.value.details["semantic_outcome"] == "rejected"
    assert raised.value.details["focus"]["restore_performed"] is True


def test_successful_semantic_action_preserves_complete_focus_summary(
    action_state: Path,
) -> None:
    class Backend(FakeActionBackend):
        def focus_window(self, identity: WindowIdentity, **_kwargs) -> dict[str, object]:
            result = super().focus_window(identity)
            return {
                **result,
                "foreground_hwnd": identity.hwnd,
                "state_before": "minimized",
                "state_after": "normal",
                "restore_performed": True,
                "restored_to_normal": True,
                "input_attachments": {
                    "attached_count": 1,
                    "detached_count": 1,
                    "failed_count": 0,
                },
            }

    class Adapter:
        def invoke(self, _identity, **_kwargs):
            return SimpleNamespace(public=lambda: {"name": "winapp"}), {
                "method": "uia.InvokePattern",
                "outcome": "delivery_only",
                "postcondition_verified": False,
            }

    result = invoke_element(
        hwnd=123,
        element="fixture-control",
        short_explanation=None,
        backend=Backend(),  # type: ignore[arg-type]
        adapter=Adapter(),  # type: ignore[arg-type]
        coordinator=ActionCoordinator(FakeMutex()),  # type: ignore[arg-type]
    )

    assert result["focus"] == {
        "method": "win32.SetForegroundWindow",
        "outcome": "postcondition_verified",
        "postcondition_verified": True,
        "foreground_hwnd": 123,
        "state_before": "minimized",
        "state_after": "normal",
        "restore_performed": True,
        "restored_to_normal": True,
        "input_attachments": {
            "attached_count": 1,
            "detached_count": 1,
            "failed_count": 0,
        },
        "verification_scope": "atomic_action",
    }


@pytest.mark.parametrize(
    ("operation", "method"),
    [
        ("focus", "win32.SetForegroundWindow"),
        ("resize", "win32.MoveWindow"),
        ("state", "win32.ShowWindow"),
        ("invoke", "uia.InvokePattern"),
        ("set-value", "uia.ValuePattern"),
        ("scroll", "uia.ScrollPattern"),
    ],
)
def test_wrapper_identity_drift_after_success_is_delivery_only(
    action_state: Path,
    operation: str,
    method: str,
) -> None:
    class Backend(FakeActionBackend):
        def revalidate_identity(self, identity: WindowIdentity) -> WindowIdentity:
            assert identity is self.identity
            raise ComputerError("stale_window", "fixture drift after delivery")

        def resize_window(self, identity: WindowIdentity, **_kwargs):
            assert identity is self.identity
            return {
                "window": identity.public(),
                "requested_bounds": {"x": 1, "y": 2, "width": 300, "height": 200},
                "actual_bounds": {"x": 1, "y": 2, "width": 300, "height": 200},
                "method": "win32.MoveWindow",
                "outcome": "postcondition_verified",
                "postcondition_verified": True,
            }

        def set_window_state(self, identity: WindowIdentity, **_kwargs):
            assert identity is self.identity
            return {
                "window": identity.public(),
                "requested_state": "maximized",
                "state_before": "normal",
                "state_after": "maximized",
                "method": "win32.ShowWindow",
                "outcome": "postcondition_verified",
                "postcondition_verified": True,
            }

    class Adapter:
        binding = SimpleNamespace(public=lambda: {"name": "winapp"})

        def invoke(self, _identity, **_kwargs):
            return self.binding, self._result("uia.InvokePattern")

        def set_value(self, _identity, **_kwargs):
            return self.binding, self._result("uia.ValuePattern")

        def scroll(self, _identity, **_kwargs):
            return self.binding, self._result("uia.ScrollPattern")

        @staticmethod
        def _result(result_method: str) -> dict[str, object]:
            return {
                "method": result_method,
                "outcome": "postcondition_verified",
                "postcondition_verified": True,
                "resolved_element": "fixture-control",
            }

    backend = Backend()
    coordinator = ActionCoordinator(FakeMutex())  # type: ignore[arg-type]
    adapter = Adapter()

    with pytest.raises(ComputerError) as raised:
        if operation == "focus":
            focus_window(
                hwnd=123,
                short_explanation=None,
                backend=backend,  # type: ignore[arg-type]
                coordinator=coordinator,
            )
        elif operation == "resize":
            resize_window(
                hwnd=123,
                x=1,
                y=2,
                width=300,
                height=200,
                short_explanation=None,
                backend=backend,  # type: ignore[arg-type]
                coordinator=coordinator,
            )
        elif operation == "state":
            set_window_state(
                hwnd=123,
                state="maximized",
                short_explanation=None,
                backend=backend,  # type: ignore[arg-type]
                coordinator=coordinator,
            )
        elif operation == "invoke":
            invoke_element(
                hwnd=123,
                element="fixture-control",
                short_explanation=None,
                backend=backend,  # type: ignore[arg-type]
                adapter=adapter,  # type: ignore[arg-type]
                coordinator=coordinator,
            )
        elif operation == "set-value":
            set_element_value(
                hwnd=123,
                element="fixture-control",
                value="fixture",
                short_explanation=None,
                backend=backend,  # type: ignore[arg-type]
                adapter=adapter,  # type: ignore[arg-type]
                coordinator=coordinator,
            )
        else:
            scroll_element(
                hwnd=123,
                element="fixture-control",
                direction="down",
                to=None,
                percent=None,
                short_explanation=None,
                backend=backend,  # type: ignore[arg-type]
                adapter=adapter,  # type: ignore[arg-type]
                coordinator=coordinator,
            )

    assert raised.value.code == "stale_window"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["method"] == method
    assert raised.value.details["backend_outcome"] == "postcondition_verified"
    assert raised.value.details["backend_postcondition_verified"] is True
    assert raised.value.details["postcondition_verified"] is False


@pytest.mark.parametrize(
    "value",
    [
        "",
        "x" * 161,
        "line\nbreak",
        "tab\tbreak",
        "delete\x7f",
        "next-line\u0085spoof",
        "spoof\u202efile",
        "😀" * 81,
    ],
)
def test_short_explanation_rejects_empty_long_and_control_text(value: str) -> None:
    with pytest.raises(ComputerError) as raised:
        validate_short_explanation(value, required=True)
    assert raised.value.code == "invalid_explanation"


def test_notification_worker_rechecks_emergency_disable_marker(tmp_path: Path) -> None:
    marker = tmp_path / actions.DISABLE_MARKER
    payload = {"disableMarker": str(marker)}
    assert notification_worker._actions_disabled(payload) is False
    marker.write_text("disabled\n", encoding="ascii")
    assert notification_worker._actions_disabled(payload) is True


def test_notification_worker_emits_explicit_busy_status(monkeypatch, capsys) -> None:
    monkeypatch.setattr(notification_worker, "_try_acquire_worker_mutex", lambda: 0)

    assert notification_worker.main() == 7
    assert capsys.readouterr().out == "BUSY\n"


def test_notification_bounds_are_utf16_and_control_safe() -> None:
    assert notification_worker._bounded("😀" * 24, 48) == "😀" * 24
    assert notification_worker._bounded("😀" * 25, 48) is None
    assert notification_worker._bounded("unsafe\u0085text", 48) is None
    with pytest.raises(ComputerError):
        actions.validate_notification_title("😀" * 25)


def test_cli_action_json_and_stdin_value_never_echo_value(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_set(**kwargs):
        captured.update(kwargs)
        return {
            "operation_id": "op-1",
            "operation": "set-value",
            "method": "uia.ValuePattern",
            "outcome": "postcondition_verified",
            "short_explanation": "Update fixture",
            "notification": {"requested": True, "status": "queued"},
            "warnings": [],
            "duration_ms": 1.0,
            "value_char_count": len(str(kwargs["value"])),
        }

    monkeypatch.setattr(computer_cli, "set_element_value", fake_set)
    monkeypatch.setattr(computer_cli.sys, "stdin", io.StringIO("sensitive-value"))

    assert (
        root_cli.main(
            [
                "computer",
                "set-value",
                "--hwnd",
                "123",
                "--element",
                "edit-a1b2",
                "--stdin",
                "--short-explanation",
                "Update fixture",
                "--json",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert captured["value"] == "sensitive-value"
    assert payload["data"]["action"]["value_char_count"] == 15
    assert "sensitive-value" not in output


def test_action_parser_rejects_non_finite_percent(capsys) -> None:
    assert (
        root_cli.main(
            [
                "computer",
                "scroll",
                "--hwnd",
                "123",
                "--element",
                "scroll-a1b2",
                "--percent",
                "nan",
                "--json",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "invalid_arguments"


@pytest.mark.parametrize(
    ("arguments", "command"),
    [
        (["focus", "--hwnd", "123"], "focus"),
        (
            [
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
            ],
            "resize",
        ),
        (["invoke", "--hwnd", "123", "--element", "button-a1b2"], "invoke"),
        (
            [
                "set-value",
                "--hwnd",
                "123",
                "--element",
                "edit-a1b2",
                "--stdin",
            ],
            "set-value",
        ),
        (
            [
                "scroll",
                "--hwnd",
                "123",
                "--element",
                "scroll-a1b2",
                "--percent",
                "50",
            ],
            "scroll",
        ),
        (["notify", "--short-explanation", "Fixture notification"], "notify"),
    ],
)
def test_required_semantic_action_parser_surface(arguments: list[str], command: str) -> None:
    parsed = root_cli.build_parser().parse_args(["computer", *arguments, "--json"])

    assert parsed.computer_command == command
    assert parsed.json is True
