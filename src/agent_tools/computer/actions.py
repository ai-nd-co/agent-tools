from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import time
import unicodedata
import uuid
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from agent_tools.computer import command_registry
from agent_tools.computer.capture_records import _atomic_private_write
from agent_tools.computer.consequence import (
    ConsequenceObserver,
    delivery_from_error,
    delivery_from_result,
    unavailable_consequence,
)
from agent_tools.computer.element_refs import ElementReferenceStore
from agent_tools.computer.models import ComputerError, WindowIdentity, bounded_text
from agent_tools.computer.physical_input import (
    PHYSICAL_CAPTURE_MAX_AGE_SECONDS,
    STAGED_INPUT_TTL_SECONDS,
    PhysicalCaptureTarget,
    StagedInputStore,
    Win32PhysicalInput,
    parse_shortcut,
    resolve_physical_capture,
    terminal_targeting_contract,
    validate_wheel_request,
)
from agent_tools.computer.uia_winapp import WinAppAdapter
from agent_tools.computer.win32_backend import Win32Backend

ACTIONS_DISABLED_ENV = "AGENT_TOOLS_COMPUTER_ACTIONS_DISABLED"
STATE_DIR_ENV = "AGENT_TOOLS_COMPUTER_STATE_DIR"
OWNER_ENV = "AGENT_TOOLS_COMPUTER_OWNER"
CONVERSATION_ALIAS_ENV = "AGENT_TOOLS_COMPUTER_CONVERSATION_ALIAS"
MUTEX_NAME = r"Local\ai-nd-co.agent-tools.computer-mutation.v1"
WORKER_MUTEX_NAME = r"Local\ai-nd-co.agent-tools.computer-worker.v1"
NOTIFICATION_MUTEX_NAME = r"Local\ai-nd-co.agent-tools.computer-notification.v1"
DISABLE_MARKER = "computer-actions.disabled"
OWNER_FILE = "computer-action-owner"
JOURNAL_FILE = "computer-actions.jsonl"
INPUT_GENERATION_FILE = "computer-input-generation.json"
MAX_EXPLANATION_CHARS = 160
MAX_VALUE_CHARS = 16_000
MAX_OWNER_CHARS = 64
MAX_NOTIFICATION_TITLE_CHARS = 48
MAX_ACTION_WARNING_COUNT = 16
MAX_ACTION_WARNING_CHARS = 96
MAX_PHASE_TIMING_MS = 86_400_000.0
PHASE_TIMING_KEYS = frozenset(
    {
        "notification_ready",
        "focus",
        "capture_revalidation",
        "consequence_before_capture",
        "delivery",
        "consequence_settle_wait",
        "consequence_after_capture",
        "consequence_compare",
        "unattributed",
        "total",
    }
)
NOTIFICATION_READY_TIMEOUT_SECONDS = 1.5
SEMANTIC_BACKEND_PREFLIGHT_FALLBACK_ERRORS = frozenset(
    {"backend_version_mismatch", "capability_missing"}
)
ACTION_ENVELOPE_FIELDS = frozenset(
    {
        "operation_id",
        "operation",
        "method",
        "outcome",
        "short_explanation",
        "notification",
        "warnings",
        "duration_ms",
        "phase_timings_ms",
        "physical_preflight",
        "journal",
        "lock",
    }
)

WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
WAIT_TIMEOUT = 0x00000102


@dataclass
class ActionExecution:
    operation_id: str
    operation: str
    short_explanation: str | None
    input_generation: int = 0
    notification: dict[str, Any] = field(
        default_factory=lambda: {"requested": False, "status": "not_requested"}
    )
    warnings: list[str] = field(default_factory=list)
    phase_timings_ms: dict[str, float] = field(default_factory=dict)
    physical_preflight: dict[str, Any] | None = None

    def record_phase(self, phase: str, started: float) -> None:
        if (
            phase.isascii()
            and 0 < len(phase) <= 48
            and all(character.isalnum() or character == "_" for character in phase)
        ):
            self.phase_timings_ms[phase] = round(
                max(0.0, (time.perf_counter() - started) * 1000.0),
                3,
            )

    def notify(self, action_label: str) -> None:
        if self.short_explanation is None:
            return
        self.assert_enabled()
        started = time.perf_counter()
        try:
            self.notification = show_notification(
                title="AgentTools computer action",
                message=f"{action_label}: {self.short_explanation}",
            )
        finally:
            self.record_phase("notification_ready", started)
        if self.notification["status"] == "disabled":
            raise ComputerError(
                "computer_actions_disabled",
                "Computer mutations are disabled by the emergency stop.",
            )
        if self.notification["status"] == "worker_busy":
            raise ComputerError(
                "computer_action_worker_busy",
                "Another computer mutation worker is still active.",
            )
        if self.notification["status"] != "queued":
            self.warnings.append("notification_unavailable")

    def assert_enabled(self) -> None:
        if actions_disabled():
            raise ComputerError(
                "computer_actions_disabled",
                "Computer mutations are disabled by the emergency stop.",
            )

    def assert_input_generation_current(self) -> None:
        self.assert_enabled()
        _assert_input_generation(self.input_generation)


@dataclass
class NativeMutexLease:
    handle: int
    acquired: bool
    abandoned: bool = False


class NamedComputerMutationMutex:
    def __init__(self, name: str = MUTEX_NAME) -> None:
        self.name = name

    def try_acquire(self) -> NativeMutexLease:
        if sys.platform != "win32":
            raise ComputerError(
                "unsupported_platform",
                "Computer actions require Windows.",
                exit_code=2,
            )
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        )
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        handle = int(kernel32.CreateMutexW(None, False, self.name) or 0)
        if not handle:
            raise ComputerError(
                "computer_action_lock_unavailable",
                "The global computer action lock could not be opened.",
            )
        wait_result = int(kernel32.WaitForSingleObject(handle, 0))
        if wait_result == WAIT_OBJECT_0:
            return NativeMutexLease(handle=handle, acquired=True)
        if wait_result == WAIT_ABANDONED:
            return NativeMutexLease(handle=handle, acquired=True, abandoned=True)
        if wait_result == WAIT_TIMEOUT:
            kernel32.CloseHandle(handle)
            return NativeMutexLease(handle=0, acquired=False)
        kernel32.CloseHandle(handle)
        raise ComputerError(
            "computer_action_lock_unavailable",
            "The global computer action lock failed.",
        )

    def release(self, lease: NativeMutexLease) -> None:
        if not lease.acquired or not lease.handle:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        try:
            if not kernel32.ReleaseMutex(lease.handle):
                raise ComputerError(
                    "computer_action_lock_release_failed",
                    "The global computer action lock could not be released safely.",
                )
        finally:
            kernel32.CloseHandle(lease.handle)


class ActionCoordinator:
    def __init__(
        self,
        mutex: NamedComputerMutationMutex | None = None,
        worker_mutex: NamedComputerMutationMutex | None = None,
    ) -> None:
        self.mutex = mutex or NamedComputerMutationMutex()
        self.worker_mutex = worker_mutex or NamedComputerMutationMutex(WORKER_MUTEX_NAME)

    def run(
        self,
        *,
        operation: str,
        target_ids: dict[str, Any],
        short_explanation: str | None,
        perform: Callable[[ActionExecution], dict[str, Any]],
        before_input_generation: Callable[[ActionExecution], None] | None = None,
    ) -> dict[str, Any]:
        explanation = validate_short_explanation(short_explanation, required=False)
        operation_id = uuid.uuid4().hex
        if actions_disabled():
            raise _action_error(
                ComputerError(
                    "computer_actions_disabled",
                    "Computer mutations are disabled by the emergency stop.",
                ),
                operation_id=operation_id,
                operation=operation,
                outcome="rejected",
                explanation=explanation,
            )
        _ensure_state_ready()
        lease = self.mutex.try_acquire()
        if not lease.acquired:
            details: dict[str, Any] = {
                "retry": "Retry after the current computer action completes.",
            }
            owner = _read_owner_metadata_with_retry()
            if owner:
                details["current_owner"] = owner
            raise _action_error(
                ComputerError(
                    "computer_action_busy",
                    "Another computer mutation is in progress; this action was not queued.",
                    details=details,
                ),
                operation_id=operation_id,
                operation=operation,
                outcome="rejected",
                explanation=explanation,
            )

        execution = ActionExecution(operation_id, operation, explanation)
        started = time.perf_counter()
        owner_written = False
        completed_output: dict[str, Any] | None = None
        action_error: ComputerError | None = None
        try:
            worker_probe = self.worker_mutex.try_acquire()
            if not worker_probe.acquired:
                raise ComputerError(
                    "computer_action_worker_busy",
                    "A computer mutation worker is still active; this action was not queued.",
                    details={
                        "outcome": "rejected",
                        "retry": "Retry after the active computer worker exits.",
                    },
                )
            self.worker_mutex.release(worker_probe)
            if lease.abandoned:
                raise ComputerError(
                    "computer_action_abandoned_recovery",
                    "The prior coordinator ended unexpectedly; retry after worker recovery.",
                    details={
                        "outcome": "rejected",
                        "retry": "Retry once more after orphan-worker recovery.",
                    },
                )
            _prune_owner_records()
            if actions_disabled():
                raise ComputerError(
                    "computer_actions_disabled",
                    "Computer mutations are disabled by the emergency stop.",
                )
            _write_owner_metadata(operation_id, operation)
            owner_written = True
            if before_input_generation is not None:
                before_input_generation(execution)
            execution.input_generation = _advance_input_generation()
            _append_journal(
                operation_id=operation_id,
                operation=operation,
                target_ids=target_ids,
                method=None,
                outcome="started",
                duration_ms=0.0,
            )
            result = perform(execution)
            outcome = str(result.get("outcome") or "failed")
            if outcome not in {
                "postcondition_verified",
                "delivery_only",
                "rejected",
                "failed",
            }:
                raise ComputerError(
                    "invalid_action_result",
                    "The action backend returned an invalid outcome.",
                )
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            completed_output = {
                "operation_id": operation_id,
                "operation": operation,
                "method": result.get("method"),
                "outcome": outcome,
                "short_explanation": explanation,
                "notification": execution.notification,
                "warnings": _bounded_action_warnings(
                    execution.warnings,
                    result.get("warnings"),
                ),
                "duration_ms": duration_ms,
                "phase_timings_ms": _public_phase_timings(
                    execution.phase_timings_ms,
                    total=duration_ms,
                ),
                "physical_preflight": execution.physical_preflight,
                **{
                    key: value for key, value in result.items() if key not in ACTION_ENVELOPE_FIELDS
                },
            }
            completed_output.setdefault("delivery", delivery_from_result(result))
            completed_output.setdefault(
                "consequence",
                unavailable_consequence("consequence_observation_not_started"),
            )
            try:
                _append_journal(
                    operation_id=operation_id,
                    operation=operation,
                    target_ids=target_ids,
                    element_id=result.get("resolved_element"),
                    method=str(result.get("method") or "unknown"),
                    outcome=outcome,
                    duration_ms=duration_ms,
                )
                completed_output["journal"] = {"status": "recorded"}
            except ComputerError:
                completed_output["warnings"] = _bounded_action_warnings(
                    completed_output["warnings"],
                    ["journal_completion_write_failed"],
                    priority=["journal_completion_write_failed"],
                )
                completed_output["journal"] = {"status": "completion_write_failed"}
            return completed_output
        except Exception as raw_exc:
            if isinstance(raw_exc, ComputerError):
                exc = raw_exc
            else:
                exc = ComputerError(
                    "action_backend_failed",
                    "The action backend failed; delivery state is unknown.",
                    details={"outcome": "delivery_only", "method": "unknown"},
                )
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            outcome = str(exc.details.get("outcome") or _failure_outcome(exc.code))
            if exc.code != "computer_action_journal_failed":
                try:
                    _append_journal(
                        operation_id=operation_id,
                        operation=operation,
                        target_ids=target_ids,
                        method=str(exc.details.get("method") or "none"),
                        outcome=outcome,
                        duration_ms=duration_ms,
                    )
                except ComputerError as journal_error:
                    execution.warnings.append("journal_completion_write_failed")
                    exc = ComputerError(
                        exc.code,
                        exc.message,
                        exit_code=exc.exit_code,
                        details={
                            **exc.details,
                            "journal_status": "completion_write_failed",
                            "journal_error_code": journal_error.code,
                        },
                    )
            action_error = _action_error(
                exc,
                operation_id=operation_id,
                operation=operation,
                outcome=outcome,
                explanation=explanation,
                notification=execution.notification,
                warnings=execution.warnings,
                phase_timings_ms=_public_phase_timings(
                    execution.phase_timings_ms,
                    total=duration_ms,
                ),
                physical_preflight=execution.physical_preflight,
            )
            action_error.details.setdefault("delivery", delivery_from_error(exc))
            action_error.details.setdefault(
                "consequence",
                unavailable_consequence("consequence_observation_not_started"),
            )
            raise action_error from raw_exc
        finally:
            release_warning: str | None = None
            try:
                self.mutex.release(lease)
            except ComputerError:
                release_warning = "computer_action_lock_release_failed"
            cleanup_warning: str | None = None
            if owner_written and not _clear_owner_metadata(operation_id):
                cleanup_warning = "computer_action_owner_cleanup_failed"
            cleanup_warnings = [
                warning for warning in (release_warning, cleanup_warning) if warning is not None
            ]
            if cleanup_warnings and completed_output is not None:
                completed_output["warnings"] = _bounded_action_warnings(
                    completed_output["warnings"],
                    cleanup_warnings,
                    priority=cleanup_warnings,
                )
                completed_output["lock"] = {
                    "status": ("release_failed" if release_warning is not None else "released")
                }
            if cleanup_warnings and action_error is not None:
                action_error.details["warnings"] = _bounded_action_warnings(
                    action_error.details.get("warnings"),
                    cleanup_warnings,
                    priority=cleanup_warnings,
                )
                if release_warning is not None:
                    action_error.details["lock"] = {"status": "release_failed"}


def focus_window(
    *,
    hwnd: int,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    coordinator: ActionCoordinator | None = None,
    require_title_match: bool = False,
) -> dict[str, Any]:
    resolved_backend = backend or Win32Backend()

    def perform(execution: ActionExecution) -> dict[str, Any]:
        identity = resolved_backend.capture_identity(hwnd)
        _assert_action_allowed(resolved_backend, identity, require_title_match)
        execution.notify("Focusing a window")
        execution.assert_enabled()
        _assert_action_allowed(resolved_backend, identity, require_title_match)

        def deliver() -> dict[str, Any]:
            focus_kwargs: dict[str, Any] = {"mutation_guard": execution.assert_enabled}
            if require_title_match:
                focus_kwargs["require_title_match"] = True
            result = resolved_backend.focus_window(identity, **focus_kwargs)
            actual = _revalidate_after_delivery(
                resolved_backend,
                identity,
                require_title_match,
                result,
            )
            return {
                **result,
                **_identity_transaction_metadata(identity, actual),
                "verification_scope": "immediate",
                "control_tier": "windows_api",
            }

        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation="focus",
            expectation={"kind": "foreground", "hwnd": hwnd, "required": True},
            deliver=deliver,
        )

    return (coordinator or ActionCoordinator()).run(
        operation="focus",
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
    )


def resize_window(
    *,
    hwnd: int,
    x: int,
    y: int,
    width: int,
    height: int,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    coordinator: ActionCoordinator | None = None,
    restore_first: bool = False,
    require_title_match: bool = False,
) -> dict[str, Any]:
    resolved_backend = backend or Win32Backend()

    def perform(execution: ActionExecution) -> dict[str, Any]:
        identity = resolved_backend.capture_identity(hwnd)
        _assert_action_allowed(resolved_backend, identity, require_title_match)
        execution.notify("Resizing a window")
        execution.assert_enabled()
        _assert_action_allowed(resolved_backend, identity, require_title_match)

        def deliver() -> dict[str, Any]:
            resize_kwargs: dict[str, Any] = {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "mutation_guard": execution.assert_enabled,
            }
            if restore_first:
                resize_kwargs["restore_first"] = True
            if require_title_match:
                resize_kwargs["require_title_match"] = True
            result = resolved_backend.resize_window(identity, **resize_kwargs)
            actual = _revalidate_after_delivery(
                resolved_backend,
                identity,
                require_title_match,
                result,
            )
            return {
                **result,
                **_identity_transaction_metadata(identity, actual),
                "control_tier": "windows_api",
            }

        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation="resize",
            expectation={
                "kind": "bounds",
                "bounds": {"x": x, "y": y, "width": width, "height": height},
                "required": True,
            },
            deliver=deliver,
        )

    return (coordinator or ActionCoordinator()).run(
        operation="resize",
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
    )


def set_window_state(
    *,
    hwnd: int,
    state: str,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    coordinator: ActionCoordinator | None = None,
    require_title_match: bool = False,
) -> dict[str, Any]:
    resolved_backend = backend or Win32Backend()
    operation = {
        "normal": "restore",
        "minimized": "minimize",
        "maximized": "maximize",
    }.get(state)
    if operation is None:
        raise ComputerError("invalid_window_state", "The requested window state is invalid.")

    def perform(execution: ActionExecution) -> dict[str, Any]:
        identity = resolved_backend.capture_identity(hwnd)
        _assert_action_allowed(resolved_backend, identity, require_title_match)
        execution.notify(f"{operation.capitalize()} a window")
        execution.assert_enabled()

        def deliver() -> dict[str, Any]:
            result = resolved_backend.set_window_state(
                identity,
                state=state,
                mutation_guard=execution.assert_enabled,
                require_title_match=require_title_match,
            )
            actual = _revalidate_after_delivery(
                resolved_backend,
                identity,
                require_title_match,
                result,
            )
            return {
                **result,
                **_identity_transaction_metadata(identity, actual),
                "control_tier": "windows_api",
            }

        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation=operation,
            expectation={"kind": "state", "state": state, "required": True},
            deliver=deliver,
        )

    return (coordinator or ActionCoordinator()).run(
        operation=operation,
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
    )


def invoke_element(
    *,
    hwnd: int,
    element: str | None,
    element_ref: str | None = None,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    adapter: WinAppAdapter | None = None,
    coordinator: ActionCoordinator | None = None,
    element_ref_store: ElementReferenceStore | None = None,
    require_title_match: bool = False,
) -> dict[str, Any]:
    _validate_element_target(element, element_ref)
    resolved_backend = backend or Win32Backend()
    resolved_adapter = adapter or WinAppAdapter(resolved_backend)

    def perform(execution: ActionExecution) -> dict[str, Any]:
        identity = resolved_backend.capture_identity(hwnd)
        _assert_action_allowed(resolved_backend, identity, require_title_match)
        selector, reference = _resolve_element_target(
            identity=identity,
            element=element,
            element_ref=element_ref,
            store=element_ref_store,
            require_title_match=require_title_match,
            allow_native_navigation=True,
        )
        if reference is None or reference["element"].get("provider") == "winapp":
            _preflight_semantic_backend(resolved_adapter)
        execution.notify("Invoking a control")
        execution.assert_enabled()
        invoke_kwargs: dict[str, Any] = {
            "selector": selector,
            "mutation_guard": execution.assert_enabled,
            "disable_marker": _state_dir() / DISABLE_MARKER,
        }
        if reference is not None:
            invoke_kwargs["element_reference"] = _native_element_reference(reference)
        if require_title_match:
            invoke_kwargs["require_title_match"] = True

        def deliver() -> dict[str, Any]:
            focus = _focus_for_semantic_action(
                resolved_backend,
                identity,
                execution,
                require_title_match=require_title_match,
            )
            try:
                _assert_action_allowed(resolved_backend, identity, require_title_match)
                binding, result = resolved_adapter.invoke(identity, **invoke_kwargs)
            except Exception as raw_exc:
                raise _semantic_error_after_focus(raw_exc, focus) from raw_exc
            delivered_result = {
                "backend": binding.public(),
                **result,
                "focus": _atomic_focus_summary(focus),
                "element_ref": element_ref,
                "element_ref_expires_at": reference.get("expires_at") if reference else None,
                "control_tier": "uia_accessibility_provider",
                "fallback_used": False,
            }
            actual = _revalidate_after_delivery(
                resolved_backend,
                identity,
                require_title_match,
                delivered_result,
            )
            return {
                "window": actual.public(),
                **delivered_result,
                **_identity_transaction_metadata(identity, actual),
            }

        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation="invoke",
            expectation={"kind": "semantic", "required": False},
            deliver=deliver,
        )

    return (coordinator or ActionCoordinator()).run(
        operation="invoke",
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
    )


def set_element_value(
    *,
    hwnd: int,
    element: str | None,
    element_ref: str | None = None,
    value: str,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    adapter: WinAppAdapter | None = None,
    coordinator: ActionCoordinator | None = None,
    element_ref_store: ElementReferenceStore | None = None,
    require_title_match: bool = False,
    require_keyboard_focus: bool = False,
) -> dict[str, Any]:
    _validate_element_target(element, element_ref)
    if len(value) > MAX_VALUE_CHARS or "\x00" in value:
        raise ComputerError(
            "invalid_value",
            f"The value must contain at most {MAX_VALUE_CHARS} characters and no NUL bytes.",
            exit_code=2,
        )
    resolved_backend = backend or Win32Backend()
    resolved_adapter = adapter or WinAppAdapter(resolved_backend)

    def perform(execution: ActionExecution) -> dict[str, Any]:
        identity = resolved_backend.capture_identity(hwnd)
        _assert_action_allowed(resolved_backend, identity, require_title_match)
        selector, reference = _resolve_element_target(
            identity=identity,
            element=element,
            element_ref=element_ref,
            store=element_ref_store,
            require_title_match=require_title_match,
        )
        _preflight_semantic_backend(resolved_adapter)
        execution.notify("Setting a control value")
        execution.assert_enabled()
        set_kwargs: dict[str, Any] = {
            "selector": selector,
            "value": value,
            "mutation_guard": execution.assert_enabled,
            "disable_marker": _state_dir() / DISABLE_MARKER,
            "require_keyboard_focus": require_keyboard_focus,
        }
        if reference is not None:
            set_kwargs["element_reference"] = _native_element_reference(reference)
        if require_title_match:
            set_kwargs["require_title_match"] = True

        def deliver() -> dict[str, Any]:
            focus = _focus_for_semantic_action(
                resolved_backend,
                identity,
                execution,
                require_title_match=require_title_match,
            )
            try:
                _assert_action_allowed(resolved_backend, identity, require_title_match)
                binding, result = resolved_adapter.set_value(identity, **set_kwargs)
            except Exception as raw_exc:
                raise _semantic_error_after_focus(raw_exc, focus) from raw_exc
            delivered_result = {
                "backend": binding.public(),
                **result,
                "focus": _atomic_focus_summary(focus),
                "element_ref": element_ref,
                "element_ref_expires_at": reference.get("expires_at") if reference else None,
                "control_tier": "uia_accessibility_provider",
                "fallback_used": False,
            }
            actual = _revalidate_after_delivery(
                resolved_backend,
                identity,
                require_title_match,
                delivered_result,
            )
            return {
                "window": actual.public(),
                **delivered_result,
                **_identity_transaction_metadata(identity, actual),
            }

        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation="set-value",
            expectation={"kind": "semantic", "required": True},
            deliver=deliver,
        )

    return (coordinator or ActionCoordinator()).run(
        operation="set-value",
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
    )


def scroll_element(
    *,
    hwnd: int,
    element: str | None,
    element_ref: str | None = None,
    direction: str | None,
    to: str | None,
    percent: float | None,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    adapter: WinAppAdapter | None = None,
    coordinator: ActionCoordinator | None = None,
    element_ref_store: ElementReferenceStore | None = None,
    require_title_match: bool = False,
) -> dict[str, Any]:
    _validate_element_target(element, element_ref)
    resolved_backend = backend or Win32Backend()
    resolved_adapter = adapter or WinAppAdapter(resolved_backend)

    def perform(execution: ActionExecution) -> dict[str, Any]:
        identity = resolved_backend.capture_identity(hwnd)
        _assert_action_allowed(resolved_backend, identity, require_title_match)
        selector, reference = _resolve_element_target(
            identity=identity,
            element=element,
            element_ref=element_ref,
            store=element_ref_store,
            require_title_match=require_title_match,
        )
        _preflight_semantic_backend(resolved_adapter)
        execution.notify("Scrolling a control")
        execution.assert_enabled()
        scroll_kwargs: dict[str, Any] = {
            "selector": selector,
            "direction": direction,
            "to": to,
            "percent": percent,
            "mutation_guard": execution.assert_enabled,
            "disable_marker": _state_dir() / DISABLE_MARKER,
        }
        if reference is not None:
            scroll_kwargs["element_reference"] = _native_element_reference(reference)
        if require_title_match:
            scroll_kwargs["require_title_match"] = True

        def deliver() -> dict[str, Any]:
            focus = _focus_for_semantic_action(
                resolved_backend,
                identity,
                execution,
                require_title_match=require_title_match,
            )
            try:
                _assert_action_allowed(resolved_backend, identity, require_title_match)
                binding, result = resolved_adapter.scroll(identity, **scroll_kwargs)
            except Exception as raw_exc:
                raise _semantic_error_after_focus(raw_exc, focus) from raw_exc
            delivered_result = {
                "backend": binding.public(),
                **result,
                "focus": _atomic_focus_summary(focus),
                "element_ref": element_ref,
                "element_ref_expires_at": reference.get("expires_at") if reference else None,
                "control_tier": "uia_accessibility_provider",
                "fallback_used": False,
            }
            actual = _revalidate_after_delivery(
                resolved_backend,
                identity,
                require_title_match,
                delivered_result,
            )
            return {
                "window": actual.public(),
                **delivered_result,
                **_identity_transaction_metadata(identity, actual),
            }

        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation="scroll",
            expectation={"kind": "semantic", "required": True},
            deliver=deliver,
        )

    return (coordinator or ActionCoordinator()).run(
        operation="scroll",
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
    )


def notify_user(
    *,
    short_explanation: str,
    title: str | None,
    coordinator: ActionCoordinator | None = None,
) -> dict[str, Any]:
    explanation = validate_short_explanation(short_explanation, required=True)
    resolved_title = validate_notification_title(title)
    backend = Win32Backend()

    def perform(execution: ActionExecution) -> dict[str, Any]:
        def deliver() -> dict[str, Any]:
            execution.assert_enabled()
            backend.assert_desktop_action_allowed()
            execution.notification = show_notification(
                title=resolved_title,
                message=explanation or "AgentTools action",
            )
            if execution.notification["status"] == "disabled":
                raise ComputerError(
                    "computer_actions_disabled",
                    "Computer mutations are disabled by the emergency stop.",
                    details={
                        "outcome": "rejected",
                        "method": "windows.notification.worker",
                    },
                )
            if execution.notification["status"] == "worker_busy":
                raise ComputerError(
                    "computer_action_worker_busy",
                    "Another computer mutation worker is still active.",
                    details={"outcome": "rejected"},
                )
            if execution.notification["status"] != "queued":
                execution.warnings.append("notification_unavailable")
                raise ComputerError(
                    "notification_unavailable",
                    "The Windows notification could not be displayed.",
                    details={
                        "outcome": "failed",
                        "method": "windows.notification.worker",
                    },
                )
            return {
                "method": "windows.Shell_NotifyIcon",
                "outcome": "delivery_only",
            }

        return execute_with_consequence(
            execution=execution,
            backend=backend,
            hwnd=None,
            operation="notify",
            expectation=None,
            deliver=deliver,
        )

    return (coordinator or ActionCoordinator()).run(
        operation="notify",
        target_ids={},
        short_explanation=explanation,
        perform=perform,
    )


def move_pointer(
    *,
    hwnd: int,
    capture_id: str,
    image_x: int,
    image_y: int,
    allow_physical: bool,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    coordinator: ActionCoordinator | None = None,
    record_store: Any = None,
    physical: Win32PhysicalInput | None = None,
) -> dict[str, Any]:
    return _physical_pointer_action(
        operation="move-pointer",
        hwnd=hwnd,
        capture_id=capture_id,
        image_x=image_x,
        image_y=image_y,
        allow_physical=allow_physical,
        short_explanation=short_explanation,
        backend=backend,
        coordinator=coordinator,
        record_store=record_store,
        physical=physical,
    )


def click_pointer(
    *,
    hwnd: int,
    capture_id: str,
    image_x: int,
    image_y: int,
    double: bool,
    allow_physical: bool,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    coordinator: ActionCoordinator | None = None,
    record_store: Any = None,
    physical: Win32PhysicalInput | None = None,
) -> dict[str, Any]:
    return _physical_pointer_action(
        operation="double-click" if double else "click",
        hwnd=hwnd,
        capture_id=capture_id,
        image_x=image_x,
        image_y=image_y,
        double=double,
        allow_physical=allow_physical,
        short_explanation=short_explanation,
        backend=backend,
        coordinator=coordinator,
        record_store=record_store,
        physical=physical,
    )


def stage_physical_text(
    *,
    hwnd: int,
    capture_id: str,
    image_x: int,
    image_y: int,
    text: str,
    allow_physical: bool,
    visible_tab_verified: bool,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    coordinator: ActionCoordinator | None = None,
    record_store: Any = None,
    stage_store: StagedInputStore | None = None,
    physical: Win32PhysicalInput | None = None,
) -> dict[str, Any]:
    from agent_tools.computer.capture_records import CaptureRecordStore

    resolved_backend = backend or Win32Backend()
    resolved_records = record_store or CaptureRecordStore()
    resolved_stages = stage_store or StagedInputStore()
    resolved_physical = physical or Win32PhysicalInput()

    def perform(execution: ActionExecution) -> dict[str, Any]:
        _require_physical_allowed(allow_physical)
        target = resolve_physical_capture(
            hwnd=hwnd,
            capture_id=capture_id,
            image_x=image_x,
            image_y=image_y,
            backend=resolved_backend,
            record_store=resolved_records,
            require_point=True,
        )
        terminal = terminal_targeting_contract(
            target.identity,
            visible_tab_verified=visible_tab_verified,
        )
        resolved_physical.validate_text_mapping(target.identity, text)
        execution.notify("Staging physical text")
        terminal_effect = {
            "character_delivery": "not_attempted",
            "enter_delivery": "not_attempted",
            "submission": "not_observed",
            "inserted_newline": "not_observed",
            "composer_change": "not_observed",
            "focus_change": "not_observed",
            "tab_change": "not_observed",
        }

        def deliver() -> dict[str, Any]:
            fresh = _physical_delivery_preflight(
                execution,
                resolved_backend,
                resolved_records,
                resolved_physical,
                target,
                require_point=True,
            )
            assert fresh.screen_point is not None
            try:
                result = resolved_physical.type_text(
                    resolved_backend,
                    fresh.identity,
                    fresh.screen_point["x"],
                    fresh.screen_point["y"],
                    text,
                    mutation_guard=execution.assert_input_generation_current,
                )
                _assert_generation_after_delivery(execution, result)
                terminal_effect["character_delivery"] = "delivered"
                try:
                    staged = resolved_stages.issue(
                        identity=fresh.identity,
                        capture=fresh,
                        input_generation=execution.input_generation,
                        text=text,
                        terminal_targeting=terminal,
                    )
                except ComputerError as stage_error:
                    raise _error_after_physical_delivery(stage_error, result) from stage_error
            except ComputerError as exc:
                delivered = _error_has_physical_delivery(exc)
                if delivered:
                    terminal_effect["character_delivery"] = "partial"
                _consume_physical_capture(
                    resolved_records,
                    capture_id,
                    delivered=delivered,
                    delivery_error=exc,
                )
                raise _with_capture_consumption(exc, delivered) from exc
            _consume_physical_capture(
                resolved_records,
                capture_id,
                delivered=True,
                delivery_result=result,
            )
            return {
                **result,
                **staged,
                "stage_valid": True,
                "automatic_cleanup_performed": False,
                "enter_delivered": False,
                "terminal_targeting": terminal,
                "capture_consumed": True,
                "control_tier": "guarded_physical_input",
                "fallback_used": True,
                "fallback_reason": "semantic_action_unavailable_or_caller_selected_last_resort",
            }

        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation="type",
            expectation=None,
            deliver=deliver,
            capture_provenance=target.provenance,
            cursor_relevant=True,
            terminal_effect=terminal_effect,
        )

    return (coordinator or ActionCoordinator()).run(
        operation="type",
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
    )


def press_staged_enter(
    *,
    hwnd: int,
    capture_id: str,
    image_x: int,
    image_y: int,
    staged_input: str,
    draft_evidence: str,
    allow_physical: bool,
    visible_tab_verified: bool,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    coordinator: ActionCoordinator | None = None,
    record_store: Any = None,
    stage_store: StagedInputStore | None = None,
    physical: Win32PhysicalInput | None = None,
) -> dict[str, Any]:
    from agent_tools.computer.capture_records import CaptureRecordStore

    resolved_backend = backend or Win32Backend()
    resolved_records = record_store or CaptureRecordStore()
    resolved_stages = stage_store or StagedInputStore()
    resolved_physical = physical or Win32PhysicalInput()

    def perform(execution: ActionExecution) -> dict[str, Any]:
        _require_physical_allowed(allow_physical)
        target = resolve_physical_capture(
            hwnd=hwnd,
            capture_id=capture_id,
            image_x=image_x,
            image_y=image_y,
            backend=resolved_backend,
            record_store=resolved_records,
            require_point=True,
        )
        terminal = terminal_targeting_contract(
            target.identity,
            visible_tab_verified=visible_tab_verified,
        )
        stage = resolved_stages.consume_for_enter(
            staged_input,
            identity=target.identity,
            current_input_generation=execution.input_generation,
            evidence_capture=target,
            draft_evidence=draft_evidence,
        )
        terminal_effect = {
            "character_delivery": "staged_previously",
            "enter_delivery": "not_attempted",
            "submission": "not_observed",
            "inserted_newline": "not_observed",
            "composer_change": "not_observed",
            "focus_change": "not_observed",
            "tab_change": "not_observed",
        }

        def deliver() -> dict[str, Any]:
            fresh = _physical_delivery_preflight(
                execution,
                resolved_backend,
                resolved_records,
                resolved_physical,
                target,
                require_point=True,
            )
            try:
                result = resolved_physical.press_enter(resolved_backend, fresh.identity)
                _assert_generation_after_delivery(execution, result)
                terminal_effect["enter_delivery"] = "delivered"
            except ComputerError as exc:
                delivered = _error_has_physical_delivery(exc)
                if delivered:
                    terminal_effect["enter_delivery"] = "partial"
                _consume_physical_capture(
                    resolved_records,
                    capture_id,
                    delivered=delivered,
                    delivery_error=exc,
                )
                raise _with_capture_consumption(exc, delivered) from exc
            _consume_physical_capture(
                resolved_records,
                capture_id,
                delivered=True,
                delivery_result=result,
            )
            return {
                **result,
                **stage,
                "staged_input_consumed": True,
                "terminal_targeting": terminal,
                "submission_observed": False,
                "inserted_newline_observed": False,
                "capture_consumed": True,
                "control_tier": "guarded_physical_input",
                "fallback_used": True,
                "fallback_reason": "semantic_submission_unavailable",
            }

        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation="key",
            expectation={"kind": "terminal_submission", "required": False},
            deliver=deliver,
            capture_provenance=target.provenance,
            cursor_relevant=False,
            terminal_effect=terminal_effect,
        )

    return (coordinator or ActionCoordinator()).run(
        operation="key",
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
        before_input_generation=lambda execution: execution.notify(
            "Submitting staged physical text"
        ),
    )


def physical_shortcut(
    *,
    hwnd: int,
    capture_id: str,
    keys: str,
    allow_physical: bool,
    visible_tab_verified: bool,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    coordinator: ActionCoordinator | None = None,
    record_store: Any = None,
    physical: Win32PhysicalInput | None = None,
) -> dict[str, Any]:
    from agent_tools.computer.capture_records import CaptureRecordStore

    resolved_backend = backend or Win32Backend()
    resolved_records = record_store or CaptureRecordStore()
    resolved_physical = physical or Win32PhysicalInput()
    parse_shortcut(keys)

    def perform(execution: ActionExecution) -> dict[str, Any]:
        _require_physical_allowed(allow_physical)
        target = resolve_physical_capture(
            hwnd=hwnd,
            capture_id=capture_id,
            image_x=None,
            image_y=None,
            backend=resolved_backend,
            record_store=resolved_records,
            require_point=False,
        )
        terminal = terminal_targeting_contract(
            target.identity,
            visible_tab_verified=visible_tab_verified,
        )
        execution.notify("Sending a guarded shortcut")

        def deliver() -> dict[str, Any]:
            fresh = _physical_delivery_preflight(
                execution,
                resolved_backend,
                resolved_records,
                resolved_physical,
                target,
                require_point=False,
            )
            try:
                result = resolved_physical.shortcut(resolved_backend, fresh.identity, keys)
                _assert_generation_after_delivery(execution, result)
            except ComputerError as exc:
                delivered = _error_has_physical_delivery(exc)
                _consume_physical_capture(
                    resolved_records,
                    capture_id,
                    delivered=delivered,
                    delivery_error=exc,
                )
                raise _with_capture_consumption(exc, delivered) from exc
            _consume_physical_capture(
                resolved_records,
                capture_id,
                delivered=True,
                delivery_result=result,
            )
            return {
                **result,
                "terminal_targeting": terminal,
                "capture_consumed": True,
                "control_tier": "guarded_physical_input",
                "fallback_used": True,
                "fallback_reason": "semantic_shortcut_unavailable",
            }

        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation="shortcut",
            expectation=None,
            deliver=deliver,
            capture_provenance=target.provenance,
        )

    return (coordinator or ActionCoordinator()).run(
        operation="shortcut",
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
    )


def physical_wheel(
    *,
    hwnd: int,
    capture_id: str,
    image_x: int,
    image_y: int,
    vertical: int | None,
    horizontal: int | None,
    allow_physical: bool,
    short_explanation: str | None,
    backend: Win32Backend | None = None,
    coordinator: ActionCoordinator | None = None,
    record_store: Any = None,
    physical: Win32PhysicalInput | None = None,
) -> dict[str, Any]:
    from agent_tools.computer.capture_records import CaptureRecordStore

    resolved_backend = backend or Win32Backend()
    resolved_records = record_store or CaptureRecordStore()
    resolved_physical = physical or Win32PhysicalInput()
    validate_wheel_request(vertical=vertical, horizontal=horizontal)

    def perform(execution: ActionExecution) -> dict[str, Any]:
        _require_physical_allowed(allow_physical)
        target = resolve_physical_capture(
            hwnd=hwnd,
            capture_id=capture_id,
            image_x=image_x,
            image_y=image_y,
            backend=resolved_backend,
            record_store=resolved_records,
            require_point=True,
        )
        execution.notify("Sending guarded physical wheel input")

        def deliver() -> dict[str, Any]:
            fresh = _physical_delivery_preflight(
                execution,
                resolved_backend,
                resolved_records,
                resolved_physical,
                target,
                require_point=True,
            )
            assert fresh.screen_point is not None
            try:
                result = resolved_physical.wheel(
                    resolved_backend,
                    fresh.identity,
                    fresh.screen_point["x"],
                    fresh.screen_point["y"],
                    vertical=vertical,
                    horizontal=horizontal,
                )
                _assert_generation_after_delivery(execution, result)
            except ComputerError as exc:
                delivered = _error_has_physical_delivery(exc)
                _consume_physical_capture(
                    resolved_records,
                    capture_id,
                    delivered=delivered,
                    delivery_error=exc,
                )
                raise _with_capture_consumption(exc, delivered) from exc
            _consume_physical_capture(
                resolved_records,
                capture_id,
                delivered=True,
                delivery_result=result,
            )
            return {
                **result,
                "capture_consumed": True,
                "control_tier": "guarded_physical_input",
                "fallback_used": True,
                "fallback_reason": "semantic_scroll_unavailable",
            }

        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation="wheel",
            expectation=None,
            deliver=deliver,
            capture_provenance=target.provenance,
            cursor_relevant=True,
        )

    return (coordinator or ActionCoordinator()).run(
        operation="wheel",
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
    )


def _physical_pointer_action(
    *,
    operation: str,
    hwnd: int,
    capture_id: str,
    image_x: int,
    image_y: int,
    allow_physical: bool,
    short_explanation: str | None,
    double: bool = False,
    backend: Win32Backend | None,
    coordinator: ActionCoordinator | None,
    record_store: Any,
    physical: Win32PhysicalInput | None,
) -> dict[str, Any]:
    from agent_tools.computer.capture_records import CaptureRecordStore

    resolved_backend = backend or Win32Backend()
    resolved_records = record_store or CaptureRecordStore()
    resolved_physical = physical or Win32PhysicalInput()

    def perform(execution: ActionExecution) -> dict[str, Any]:
        _require_physical_allowed(allow_physical)
        target = resolve_physical_capture(
            hwnd=hwnd,
            capture_id=capture_id,
            image_x=image_x,
            image_y=image_y,
            backend=resolved_backend,
            record_store=resolved_records,
            require_point=True,
        )
        execution.notify(
            "Moving the pointer" if operation == "move-pointer" else "Clicking a target"
        )

        def deliver() -> dict[str, Any]:
            fresh = _physical_delivery_preflight(
                execution,
                resolved_backend,
                resolved_records,
                resolved_physical,
                target,
                require_point=True,
            )
            assert fresh.screen_point is not None
            try:
                if operation == "move-pointer":
                    result = resolved_physical.move_pointer(
                        resolved_backend,
                        fresh.identity,
                        fresh.screen_point["x"],
                        fresh.screen_point["y"],
                    )
                else:
                    result = resolved_physical.click(
                        resolved_backend,
                        fresh.identity,
                        fresh.screen_point["x"],
                        fresh.screen_point["y"],
                        double=double,
                    )
                _assert_generation_after_delivery(execution, result)
            except ComputerError as exc:
                delivered = _error_has_physical_delivery(exc)
                _consume_physical_capture(
                    resolved_records,
                    capture_id,
                    delivered=delivered,
                    delivery_error=exc,
                )
                raise _with_capture_consumption(exc, delivered) from exc
            _consume_physical_capture(
                resolved_records,
                capture_id,
                delivered=True,
                delivery_result=result,
            )
            return {
                **result,
                "capture_consumed": True,
                "control_tier": "guarded_physical_input",
                "fallback_used": True,
                "fallback_reason": "semantic_pointer_action_unavailable",
            }

        expectation = (
            {
                "kind": "cursor",
                "cursor": target.screen_point,
                "required": True,
            }
            if operation == "move-pointer"
            else None
        )
        return execute_with_consequence(
            execution=execution,
            backend=resolved_backend,
            hwnd=hwnd,
            operation=operation,
            expectation=expectation,
            deliver=deliver,
            capture_provenance=target.provenance,
            cursor_relevant=True,
        )

    return (coordinator or ActionCoordinator()).run(
        operation=operation,
        target_ids={"hwnd": hwnd},
        short_explanation=short_explanation,
        perform=perform,
    )


def _require_physical_allowed(allow_physical: bool) -> None:
    if not allow_physical:
        raise ComputerError(
            "physical_input_not_authorized",
            "Physical input requires the explicit --allow-physical flag.",
            exit_code=2,
            details={"outcome": "rejected"},
        )


def _physical_delivery_preflight(
    execution: ActionExecution,
    backend: Win32Backend,
    record_store: Any,
    physical: Win32PhysicalInput,
    target: PhysicalCaptureTarget,
    *,
    require_point: bool,
) -> PhysicalCaptureTarget:
    execution.assert_input_generation_current()
    physical.assert_no_held_input()
    focus_started = time.perf_counter()
    try:
        focus = backend.focus_window(
            target.identity,
            mutation_guard=execution.assert_enabled,
        )
    finally:
        execution.record_phase("focus", focus_started)
    if focus.get("postcondition_verified") is not True:
        raise ComputerError(
            "physical_input_focus_failed",
            "The exact target could not be focused before physical input.",
            details={
                **focus,
                "outcome": str(focus.get("outcome") or "failed"),
                "partial_mutation": "focus",
            },
        )
    physical.assert_foreground_identity(backend, target.identity)
    capture_started = time.perf_counter()
    try:
        fresh = resolve_physical_capture(
            hwnd=target.identity.hwnd,
            capture_id=target.capture_id,
            image_x=(target.image_point or {}).get("x"),
            image_y=(target.image_point or {}).get("y"),
            backend=backend,
            record_store=record_store,
            require_point=require_point,
        )
    finally:
        execution.record_phase("capture_revalidation", capture_started)
    physical.assert_foreground_identity(backend, fresh.identity)
    physical.assert_no_held_input()
    execution.assert_input_generation_current()
    execution.physical_preflight = {
        **fresh.provenance,
        "target": {
            "hwnd": fresh.identity.hwnd,
            "pid": fresh.identity.pid,
            "thread_id": fresh.identity.thread_id,
            "process_started": fresh.identity.process_started,
            "session_id": fresh.identity.session_id,
        },
        "point_mapping": {
            "required": require_point,
            "verified": fresh.screen_point is not None if require_point else True,
            "image_point": fresh.image_point,
            "native_screen_point": fresh.screen_point,
        },
        "foreground_identity_verified": True,
        "input_generation": execution.input_generation,
    }
    return fresh


def _consume_physical_capture(
    record_store: Any,
    capture_id: str,
    *,
    delivered: bool,
    delivery_error: ComputerError | None = None,
    delivery_result: dict[str, Any] | None = None,
) -> None:
    if not delivered:
        return
    try:
        record_store.delete(capture_id)
    except ComputerError as exc:
        evidence = delivery_error.details if delivery_error is not None else delivery_result or {}
        raise ComputerError(
            "physical_capture_consumption_failed",
            "Physical input was delivered but its capture authority could not be consumed.",
            details={
                **exc.details,
                "outcome": "delivery_only",
                "method": (evidence.get("method") or "capture_record.consume"),
                "capture_consumed": False,
                "delivery_error_code": (
                    delivery_error.code if delivery_error is not None else None
                ),
                **(
                    {
                        key: evidence[key]
                        for key in (
                            "events_intended",
                            "events_delivered",
                            "intended_character_count",
                            "delivered_character_count",
                            "first_undelivered_position",
                            "partial_delivery",
                        )
                        if key in evidence
                    }
                    if evidence
                    else {}
                ),
            },
        ) from exc


def _error_after_physical_delivery(
    error: ComputerError,
    result: dict[str, Any],
) -> ComputerError:
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details={
            **error.details,
            "outcome": "delivery_only",
            "method": result.get("method"),
            **{
                key: result[key]
                for key in (
                    "events_intended",
                    "events_delivered",
                    "intended_character_count",
                    "delivered_character_count",
                )
                if key in result
            },
        },
    )


def _assert_generation_after_delivery(
    execution: ActionExecution,
    result: dict[str, Any],
) -> None:
    try:
        execution.assert_input_generation_current()
    except ComputerError as exc:
        raise ComputerError(
            "physical_input_generation_changed",
            "The physical input generation changed during delivery.",
            details={
                **exc.details,
                "outcome": "delivery_only",
                "method": result.get("method"),
                **{
                    key: result[key]
                    for key in (
                        "events_intended",
                        "events_delivered",
                        "intended_character_count",
                        "delivered_character_count",
                    )
                    if key in result
                },
            },
        ) from exc


def _error_has_physical_delivery(error: ComputerError) -> bool:
    return bool(
        error.details.get("partial_delivery")
        or error.details.get("events_delivered")
        or error.details.get("delivered_character_count")
        or error.details.get("outcome") == "delivery_only"
    )


def _with_capture_consumption(error: ComputerError, delivered: bool) -> ComputerError:
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details={
            **error.details,
            "capture_consumed": delivered,
        },
    )


def action_capabilities() -> dict[str, Any]:
    disabled = actions_disabled()
    return {
        "available": sys.platform == "win32" and not disabled,
        "disabled": disabled,
        "disable_sources": _disable_sources(),
        "mutex": MUTEX_NAME,
        "busy_behavior": "reject_without_queue",
        "commands": command_registry.mutation_command_names(),
        "physical_input": True,
        "physical_input_policy": "explicit_capture_bound_last_resort",
        "physical_capture_max_age_seconds": PHYSICAL_CAPTURE_MAX_AGE_SECONDS,
        "staged_input_ttl_seconds": STAGED_INPUT_TTL_SECONDS,
        "terminal_targeting": (
            "visible_tab_verification_required_when_multi_tab_identity_is_ambiguous"
        ),
    }


def actions_disabled() -> bool:
    return bool(_disable_sources())


def execute_with_consequence(
    *,
    execution: ActionExecution,
    backend: Win32Backend,
    hwnd: int | None,
    operation: str,
    expectation: dict[str, Any] | None,
    deliver: Callable[[], dict[str, Any]],
    capture_provenance: dict[str, Any] | None = None,
    cursor_relevant: bool = False,
    terminal_effect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observer = ConsequenceObserver(
        backend=backend,
        hwnd=hwnd,
        input_generation=execution.input_generation,
        input_generation_reader=_read_input_generation,
        capture_provenance=capture_provenance,
        cursor_relevant=cursor_relevant,
    )
    before_started = time.perf_counter()
    try:
        before = observer.capture()
    finally:
        execution.record_phase("consequence_before_capture", before_started)
    delivery_started = time.perf_counter()
    nested_delivery_before = _nested_delivery_phase_total(execution)
    try:
        result = deliver()
    except Exception as raw_exc:
        _record_exclusive_delivery_phase(
            execution,
            delivery_started,
            nested_delivery_before,
        )
        exc = (
            raw_exc
            if isinstance(raw_exc, ComputerError)
            else ComputerError(
                "action_backend_failed",
                "The action backend failed; delivery state is unknown.",
            )
        )
        semantic = _semantic_consequence_from_payload(exc.details)
        consequence = observer.settle_and_compare(
            before,
            operation=operation,
            expectation=expectation,
            semantic_postcondition=semantic,
            terminal_effect=terminal_effect,
        )
        _record_consequence_timings(execution, consequence)
        raise ComputerError(
            exc.code,
            exc.message,
            exit_code=exc.exit_code,
            details={
                **exc.details,
                "delivery": delivery_from_error(exc),
                "consequence": consequence,
            },
        ) from raw_exc
    _record_exclusive_delivery_phase(
        execution,
        delivery_started,
        nested_delivery_before,
    )
    consequence = observer.settle_and_compare(
        before,
        operation=operation,
        expectation=expectation,
        semantic_postcondition=_semantic_consequence_from_payload(result),
        terminal_effect=terminal_effect,
    )
    _record_consequence_timings(execution, consequence)
    return {
        **result,
        "delivery": delivery_from_result(result),
        "consequence": consequence,
    }


def _record_consequence_timings(
    execution: ActionExecution,
    consequence: dict[str, Any],
) -> None:
    timings = consequence.get("phase_timings_ms")
    if not isinstance(timings, dict):
        return
    for name, value in timings.items():
        if name != "total" and isinstance(name, str) and type(value) in {int, float} and value >= 0:
            execution.phase_timings_ms[f"consequence_{name}"] = round(float(value), 3)


def _nested_delivery_phase_total(execution: ActionExecution) -> float:
    return sum(
        execution.phase_timings_ms.get(name, 0.0) for name in ("focus", "capture_revalidation")
    )


def _record_exclusive_delivery_phase(
    execution: ActionExecution,
    started: float,
    nested_before: float,
) -> None:
    elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
    nested_after = _nested_delivery_phase_total(execution)
    nested_elapsed = max(0.0, nested_after - nested_before)
    execution.phase_timings_ms["delivery"] = round(
        max(0.0, elapsed_ms - nested_elapsed),
        3,
    )


def _semantic_consequence_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not any(
        key in payload
        for key in (
            "postcondition_verified",
            "before",
            "after",
            "changed",
            "change_required",
            "value_char_count",
            "readback_char_count",
        )
    ):
        return None
    before_value = payload.get("before")
    after_value = payload.get("after")
    before: dict[str, Any] = before_value if isinstance(before_value, dict) else {}
    after: dict[str, Any] = after_value if isinstance(after_value, dict) else {}
    result: dict[str, Any] = {
        "verified": payload.get("postcondition_verified") is True,
        "changed": payload.get("changed") is True,
        "change_required": payload.get("change_required") is not False,
    }
    for public, source in (
        ("selected_before", before.get("selected")),
        ("selected_after", after.get("selected")),
        ("expanded_before", before.get("expanded")),
        ("expanded_after", after.get("expanded")),
        ("checked_before", before.get("checked")),
        ("checked_after", after.get("checked")),
        ("enabled_before", before.get("enabled")),
        ("enabled_after", after.get("enabled")),
        ("scroll_before", before.get("vertical") or before.get("horizontal")),
        ("scroll_after", after.get("vertical") or after.get("horizontal")),
        ("value_char_count", payload.get("value_char_count")),
        ("readback_char_count", payload.get("readback_char_count")),
    ):
        if source is not None:
            result[public] = source
    return result


def validate_short_explanation(value: str | None, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ComputerError(
                "invalid_explanation",
                "A short explanation is required.",
                exit_code=2,
            )
        return None
    if not value or _utf16_units(value) > MAX_EXPLANATION_CHARS:
        raise ComputerError(
            "invalid_explanation",
            f"The short explanation must contain 1 through {MAX_EXPLANATION_CHARS} characters.",
            exit_code=2,
        )
    if any(unicodedata.category(character) in {"Cc", "Cf"} for character in value):
        raise ComputerError(
            "invalid_explanation",
            "The short explanation cannot contain control characters.",
            exit_code=2,
        )
    return value


def validate_notification_title(value: str | None) -> str:
    title = value or "AgentTools"
    if (
        not title
        or _utf16_units(title) > MAX_NOTIFICATION_TITLE_CHARS
        or any(unicodedata.category(character) in {"Cc", "Cf"} for character in title)
    ):
        raise ComputerError(
            "invalid_notification_title",
            f"The notification title must contain 1 through {MAX_NOTIFICATION_TITLE_CHARS} "
            "characters without controls.",
            exit_code=2,
        )
    return title


def show_notification(*, title: str, message: str) -> dict[str, Any]:
    if sys.platform != "win32":
        return {"requested": True, "status": "unavailable", "warning": "unsupported_platform"}
    worker = Path(__file__).with_name("notification_worker.py").resolve()
    process_started = Win32Backend().process_started(os.getpid())
    if process_started is None:
        return {
            "requested": True,
            "status": "unavailable",
            "warning": "owner_identity_unavailable",
        }
    payload = json.dumps(
        {
            "title": title,
            "message": message,
            "disableMarker": str(_state_dir() / DISABLE_MARKER),
            "owner": {
                "pid": os.getpid(),
                "processStarted": process_started,
            },
        },
        ensure_ascii=True,
    ).encode("ascii")
    creationflags = (
        getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "DETACHED_PROCESS", 0)
    )
    try:
        process = subprocess.Popen(
            [sys.executable, "-P", str(worker)],
            cwd=worker.parent,
            env=_notification_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            creationflags=creationflags,
        )
    except OSError:
        return {"requested": True, "status": "unavailable", "warning": "launch_failed"}
    if process.stdin is None or process.stdout is None:
        _terminate_notification_process(process)
        return {"requested": True, "status": "unavailable", "warning": "pipe_failed"}
    deadline = time.monotonic() + NOTIFICATION_READY_TIMEOUT_SECONDS
    reported_status: str | None = None
    try:
        process.stdin.write(payload)
        process.stdin.close()
        ready = bytearray()
        while time.monotonic() < deadline:
            chunk, closed = _read_notification_pipe(process.stdout)
            ready.extend(chunk)
            normalized = ready.replace(b"\r\n", b"\n")
            if b"READY\n" in normalized:
                process.stdout.close()
                return {"requested": True, "status": "queued"}
            for token, status in (
                (b"DISABLED\n", "disabled"),
                (b"BUSY\n", "worker_busy"),
                (b"OWNER_GONE\n", "owner_gone"),
            ):
                if token in normalized:
                    reported_status = status
                    break
            if reported_status is not None:
                break
            if closed or process.poll() is not None:
                break
            time.sleep(0.01)
    except OSError:
        reported_status = None
    if process.stdout is not None:
        process.stdout.close()
    remaining = max(0.0, deadline - time.monotonic())
    returncode: int | None
    try:
        returncode = process.wait(timeout=remaining)
    except (OSError, subprocess.TimeoutExpired):
        returncode = process.poll()
    if reported_status == "disabled" or returncode == 6:
        return {"requested": True, "status": "disabled"}
    if reported_status == "worker_busy" or returncode == 7:
        return {"requested": True, "status": "worker_busy"}
    if reported_status == "owner_gone" or returncode == 8:
        return {
            "requested": True,
            "status": "unavailable",
            "warning": "owner_gone",
        }
    _terminate_notification_process(process)
    return {"requested": True, "status": "unavailable", "warning": "worker_failed"}


def _identity_transaction_metadata(
    before: WindowIdentity,
    after: WindowIdentity,
) -> dict[str, Any]:
    key = secrets.token_bytes(32)

    def title_fingerprint(value: str) -> str:
        return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()

    return {
        "strong_identity_verified": True,
        "identity_mismatch_fields": [],
        "title_changed_during_operation": before.title != after.title,
        "title_before_hmac_sha256": title_fingerprint(before.title),
        "title_after_hmac_sha256": title_fingerprint(after.title),
    }


def _validate_element_target(element: str | None, element_ref: str | None) -> None:
    if (element is None) == (element_ref is None):
        raise ComputerError(
            "invalid_element_target",
            "Choose exactly one inspect element slug or element reference.",
            exit_code=2,
        )


def _resolve_element_target(
    *,
    identity: WindowIdentity,
    element: str | None,
    element_ref: str | None,
    store: ElementReferenceStore | None,
    require_title_match: bool,
    allow_native_navigation: bool = False,
) -> tuple[str, dict[str, Any] | None]:
    if element_ref is None:
        assert element is not None
        return element, None
    reference = (store or ElementReferenceStore()).resolve(
        element_ref,
        identity=identity,
        require_title_match=require_title_match,
    )
    selector = reference["element"].get("selector")
    if not isinstance(selector, str) or not selector:
        raise ComputerError(
            "element_ref_invalid",
            "The element reference has no valid provider selector.",
            details={"resolution_stage": "provider_selector"},
        )
    provider = reference["element"].get("provider")
    native_navigation = bool(
        allow_native_navigation
        and provider == "native_uia_raw"
        and reference["element"].get("control_type") in {"TreeItem", "ListItem"}
    )
    if provider != "winapp" and not native_navigation:
        raise ComputerError(
            "semantic_provider_action_unsupported",
            "The referenced inspection provider has no safe semantic action path.",
            details={
                "resolution_stage": "provider_capability",
                "fallback": {
                    "kind": "guarded_physical_input",
                    "status": "available",
                    "reason": "requires_fresh_capture_and_explicit_allow_physical",
                },
            },
        )
    return selector, reference


def _focus_for_semantic_action(
    backend: Win32Backend,
    identity: WindowIdentity,
    execution: ActionExecution,
    *,
    require_title_match: bool,
) -> dict[str, Any]:
    _assert_action_allowed(backend, identity, require_title_match)
    focus_kwargs: dict[str, Any] = {"mutation_guard": execution.assert_enabled}
    if require_title_match:
        focus_kwargs["require_title_match"] = True
    started = time.perf_counter()
    try:
        return backend.focus_window(identity, **focus_kwargs)
    finally:
        execution.record_phase("focus", started)


def _native_element_reference(reference: dict[str, Any]) -> dict[str, Any]:
    element = reference.get("element")
    if not isinstance(element, dict) or type(reference.get("expires_unix_ms")) is not int:
        raise ComputerError(
            "element_ref_invalid",
            "The element reference has invalid action metadata.",
            details={"resolution_stage": "reference_validation"},
        )
    return {
        **element,
        "expires_unix_ms": reference["expires_unix_ms"],
    }


def _preflight_semantic_backend(adapter: WinAppAdapter) -> None:
    require = getattr(adapter, "require", None)
    if callable(require):
        try:
            require()
        except ComputerError as exc:
            if exc.code not in SEMANTIC_BACKEND_PREFLIGHT_FALLBACK_ERRORS:
                raise
            raise ComputerError(
                exc.code,
                exc.message,
                exit_code=exc.exit_code,
                details={
                    **exc.details,
                    "outcome": "rejected",
                    "fallback": {
                        "kind": "guarded_physical_input",
                        "status": "available",
                        "reason": "requires_fresh_capture_and_explicit_allow_physical",
                    },
                },
            ) from exc


def _atomic_focus_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "method": result.get("method"),
        "outcome": result.get("outcome"),
        "postcondition_verified": result.get("postcondition_verified") is True,
        "foreground_hwnd": result.get("foreground_hwnd"),
        "state_before": result.get("state_before"),
        "state_after": result.get("state_after"),
        "restore_performed": result.get("restore_performed"),
        "restored_to_normal": result.get("restored_to_normal"),
        "input_attachments": result.get("input_attachments"),
        "verification_scope": "atomic_action",
    }
    for key in ("foreground_before", "changed"):
        if key in result:
            summary[key] = result[key]
    return summary


def _semantic_error_after_focus(
    raw_exc: Exception,
    focus: dict[str, Any],
) -> ComputerError:
    exc = (
        raw_exc
        if isinstance(raw_exc, ComputerError)
        else ComputerError(
            "action_backend_failed",
            "The semantic action backend failed after the target was focused.",
        )
    )
    semantic_outcome = str(exc.details.get("outcome") or _failure_outcome(exc.code))
    semantic_method = exc.details.get("method")
    focus_summary = _atomic_focus_summary(focus)
    if semantic_outcome == "delivery_only" or focus_summary.get("changed") is False:
        return ComputerError(
            exc.code,
            exc.message,
            exit_code=exc.exit_code,
            details={
                **exc.details,
                "outcome": semantic_outcome,
                "method": semantic_method,
                "focus": focus_summary,
                "semantic_outcome": semantic_outcome,
                "semantic_method": semantic_method,
                "semantic_postcondition_verified": (
                    exc.details.get("postcondition_verified") is True
                ),
            },
        )
    return ComputerError(
        exc.code,
        exc.message,
        exit_code=exc.exit_code,
        details={
            **exc.details,
            "outcome": "delivery_only",
            "method": focus_summary.get("method"),
            "postcondition_verified": False,
            "partial_mutation": "focus",
            "focus": focus_summary,
            "semantic_outcome": semantic_outcome,
            "semantic_method": semantic_method,
            "semantic_postcondition_verified": (exc.details.get("postcondition_verified") is True),
        },
    )


def _assert_action_allowed(
    backend: Win32Backend,
    identity: WindowIdentity,
    require_title_match: bool,
) -> WindowIdentity:
    if require_title_match:
        return backend.assert_action_allowed(identity, require_title_match=True)
    return backend.assert_action_allowed(identity)


def _revalidate_action_identity(
    backend: Win32Backend,
    identity: WindowIdentity,
    require_title_match: bool,
) -> WindowIdentity:
    if require_title_match:
        return backend.revalidate_identity(identity, require_title_match=True)
    return backend.revalidate_identity(identity)


def _revalidate_after_delivery(
    backend: Win32Backend,
    identity: WindowIdentity,
    require_title_match: bool,
    delivered_result: dict[str, Any],
) -> WindowIdentity:
    try:
        return _revalidate_action_identity(
            backend,
            identity,
            require_title_match,
        )
    except Exception as raw_exc:
        exc = (
            raw_exc
            if isinstance(raw_exc, ComputerError)
            else ComputerError(
                "action_backend_failed",
                "The action was delivered, but the final window identity could not be verified.",
            )
        )
        preserved = {
            key: value
            for key, value in delivered_result.items()
            if key not in {"outcome", "postcondition_verified"}
        }
        raise ComputerError(
            exc.code,
            exc.message,
            exit_code=exc.exit_code,
            details={
                **preserved,
                **exc.details,
                "outcome": "delivery_only",
                "method": str(delivered_result.get("method") or "unknown"),
                "backend_outcome": delivered_result.get("outcome"),
                "backend_postcondition_verified": (
                    delivered_result.get("postcondition_verified") is True
                ),
                "postcondition_verified": False,
            },
        ) from raw_exc


def _failure_outcome(code: str) -> str:
    if code.startswith(("capture_", "physical_capture_", "staged_input_")) or code in {
        "computer_actions_disabled",
        "computer_action_busy",
        "computer_action_worker_busy",
        "computer_action_abandoned_recovery",
        "invalid_window",
        "ambiguous_window_target",
        "window_not_visible",
        "window_untitled",
        "stale_window",
        "secure_desktop_unavailable",
        "protected_target",
        "target_integrity_unavailable",
        "target_integrity_denied",
        "password_control",
        "control_read_only",
        "invalid_element",
        "invalid_geometry",
        "geometry_offscreen",
        "window_minimized",
        "window_maximized",
        "target_identity_unavailable",
        "stale_element",
        "ambiguous_element",
        "element_not_found",
        "control_disabled",
        "invoke_pattern_unavailable",
        "set_value_pattern_unavailable",
        "scroll_pattern_unavailable",
        "scroll_axis_unavailable",
        "scroll_percent_unavailable",
        "native_uia_unavailable",
        "uia_identity_unavailable",
        "capability_missing",
        "backend_version_mismatch",
        "uia_provider_unavailable",
        "semantic_provider_action_unsupported",
        "invalid_element_ref",
        "element_ref_missing",
        "element_ref_expired",
        "element_ref_altered",
        "element_ref_invalid",
        "element_ref_store_unavailable",
        "element_ref_clock_invalid",
        "physical_input_not_authorized",
        "physical_input_user_state_held",
        "physical_input_focus_changed",
        "physical_target_occluded",
        "physical_point_offscreen",
        "physical_virtual_screen_unavailable",
        "physical_image_point_required",
        "keyboard_layout_mismatch",
        "physical_character_unmappable",
        "physical_character_requires_unsafe_modifier",
        "physical_text_control_rejected",
        "invalid_physical_text",
        "invalid_shortcut",
        "invalid_shortcut_key",
        "forbidden_shortcut",
        "invalid_wheel_request",
        "invalid_wheel_count",
        "invalid_draft_evidence",
        "terminal_tab_verification_required",
        "computer_input_generation_changed",
    }:
        return "rejected"
    return "failed"


def _action_error(
    error: ComputerError,
    *,
    operation_id: str,
    operation: str,
    outcome: str,
    explanation: str | None,
    notification: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    phase_timings_ms: dict[str, float] | None = None,
    physical_preflight: dict[str, Any] | None = None,
) -> ComputerError:
    details = {
        **error.details,
        "operation_id": operation_id,
        "operation": operation,
        "outcome": outcome,
        "short_explanation": explanation,
        "notification": notification or {"requested": False, "status": "not_requested"},
        "warnings": _bounded_action_warnings(error.details.get("warnings"), warnings),
        "phase_timings_ms": _public_phase_timings(phase_timings_ms or {}),
        "physical_preflight": physical_preflight,
    }
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details=details,
    )


def _public_phase_timings(
    values: dict[str, float],
    *,
    total: float | None = None,
) -> dict[str, float]:
    candidates: dict[str, object] = dict(values)
    if total is not None:
        candidates["total"] = total
    result: dict[str, float] = {}
    for key in sorted(PHASE_TIMING_KEYS & set(candidates)):
        value = candidates[key]
        if type(value) not in {int, float}:
            continue
        number = float(cast(int | float, value))
        if number != number or not 0 <= number <= MAX_PHASE_TIMING_MS:
            continue
        result[key] = round(number, 3)
    if total is not None and "total" in result:
        leaf_sum = sum(
            value for key, value in result.items() if key not in {"total", "unattributed"}
        )
        result["unattributed"] = round(
            max(0.0, result["total"] - leaf_sum),
            3,
        )
        result = dict(sorted(result.items()))
    return result


def _bounded_action_warnings(
    *groups: object,
    priority: object = None,
) -> list[str]:
    def safe_codes(group: object) -> set[str]:
        codes: set[str] = set()
        if not isinstance(group, (list, tuple, set, frozenset)):
            return codes
        for item in group:
            if (
                isinstance(item, str)
                and 0 < len(item) <= MAX_ACTION_WARNING_CHARS
                and all(
                    character.isascii() and (character.isalnum() or character in "._-")
                    for character in item
                )
            ):
                codes.add(item)
        return codes

    warning_codes = safe_codes(priority)
    priority_codes = set(warning_codes)
    for group in groups:
        warning_codes.update(safe_codes(group))
    selected = set(sorted(priority_codes)[:MAX_ACTION_WARNING_COUNT])
    for warning in sorted(warning_codes - selected):
        if len(selected) >= MAX_ACTION_WARNING_COUNT:
            break
        selected.add(warning)
    return sorted(selected)


def _disable_sources() -> list[str]:
    sources: list[str] = []
    if os.environ.get(ACTIONS_DISABLED_ENV, "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        sources.append("environment")
    try:
        if (_state_dir() / DISABLE_MARKER).is_file():
            sources.append("config_marker")
    except OSError:
        sources.append("state_unavailable")
    return sources


def _state_dir() -> Path:
    explicit = os.environ.get(STATE_DIR_ENV)
    if explicit:
        return Path(explicit).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "AgentTools" / "state"
    return Path.home() / ".local" / "share" / "agent-tools" / "state"


def _ensure_state_ready() -> None:
    try:
        state = _state_dir()
        state.mkdir(parents=True, exist_ok=True)
        with (state / JOURNAL_FILE).open("a", encoding="utf-8"):
            pass
    except OSError as exc:
        raise ComputerError(
            "computer_action_state_unavailable",
            "The computer action audit state is unavailable; no mutation was attempted.",
        ) from exc


def _advance_input_generation() -> int:
    path = _state_dir() / INPUT_GENERATION_FILE
    try:
        current = _read_input_generation()
        generation = current + 1
        _atomic_private_write(
            path,
            json.dumps(
                {"generation": generation},
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii"),
        )
        return generation
    except ComputerError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComputerError(
            "computer_input_generation_unavailable",
            "The physical input generation state is unavailable.",
        ) from exc


def _assert_input_generation(expected: int) -> None:
    current = _read_input_generation()
    if current != expected:
        raise ComputerError(
            "computer_input_generation_changed",
            "Another computer mutation changed the physical input generation.",
            details={
                "outcome": "rejected",
                "expected_input_generation": expected,
                "current_input_generation": current,
            },
        )


def _read_input_generation() -> int:
    path = _state_dir() / INPUT_GENERATION_FILE
    try:
        if not path.exists():
            return 0
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComputerError(
            "computer_input_generation_unavailable",
            "The physical input generation state is unavailable.",
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"generation"}
        or type(payload.get("generation")) is not int
        or not 0 <= int(payload["generation"]) < (2**63 - 1)
    ):
        raise ComputerError(
            "computer_input_generation_invalid",
            "The physical input generation state is invalid.",
        )
    return int(payload["generation"])


def _write_owner_metadata(operation_id: str, operation: str) -> None:
    owner = _bounded_metadata(os.environ.get(OWNER_ENV)) or f"pid-{os.getpid()}"
    alias = _bounded_metadata(os.environ.get(CONVERSATION_ALIAS_ENV))
    process_started = Win32Backend().process_started(os.getpid())
    if process_started is None:
        raise ComputerError(
            "computer_action_state_unavailable",
            "The computer action owner process identity could not be recorded.",
        )
    payload = {
        "operation_id": operation_id,
        "operation": operation,
        "owner": owner,
        "conversation_alias": alias,
        "pid": os.getpid(),
        "process_started": process_started,
        "published_at_unix_ns": time.time_ns(),
    }
    path = _owner_path(operation_id)
    temporary = path.with_name(f".{path.name}.{operation_id}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=True), encoding="ascii")
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ComputerError(
            "computer_action_state_unavailable",
            "The computer action owner state could not be recorded.",
        ) from exc


def _clear_owner_metadata(operation_id: str) -> bool:
    path = _owner_path(operation_id)
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
        if isinstance(payload, dict) and payload.get("operation_id") == operation_id:
            path.unlink(missing_ok=True)
        return True
    except FileNotFoundError:
        return True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _read_owner_metadata() -> dict[str, Any] | None:
    newest: dict[str, Any] | None = None
    newest_published = -1
    try:
        paths = sorted(
            _state_dir().glob(f"{OWNER_FILE}.*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )[:32]
    except OSError:
        return None
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        published = payload.get("published_at_unix_ns")
        pid = payload.get("pid")
        process_started = payload.get("process_started")
        if (
            type(published) is not int
            or type(pid) is not int
            or type(process_started) is not int
            or published <= newest_published
            or Win32Backend().process_started(pid) != process_started
        ):
            continue
        newest_published = published
        newest = {
            "owner": _bounded_metadata(payload.get("owner")),
            "conversation_alias": _bounded_metadata(payload.get("conversation_alias")),
            "operation": _bounded_metadata(payload.get("operation")),
            "pid": pid,
            "operation_id": _bounded_metadata(payload.get("operation_id")),
            "_published_at_unix_ns": published,
        }
    return newest


def _read_owner_metadata_with_retry(*, timeout_seconds: float = 0.15) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_seconds
    newest: dict[str, Any] | None = None
    newest_published = -1
    while True:
        owner = _read_owner_metadata()
        if owner is not None:
            published = owner.get("_published_at_unix_ns")
            if type(published) is not int:
                newest = owner
            elif published > newest_published:
                newest = owner
                newest_published = published
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if newest is None:
                return None
            return {key: value for key, value in newest.items() if not key.startswith("_")}
        time.sleep(min(0.01, remaining))


def _owner_path(operation_id: str) -> Path:
    digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:24]
    return _state_dir() / f"{OWNER_FILE}.{digest}.json"


def _prune_owner_records() -> None:
    try:
        for pattern in (f"{OWNER_FILE}*", f".{OWNER_FILE}*"):
            for path in _state_dir().glob(pattern):
                if path.is_file():
                    path.unlink()
    except OSError as exc:
        raise ComputerError(
            "computer_action_state_unavailable",
            "Abandoned computer action owner metadata could not be removed.",
        ) from exc


def _append_journal(
    *,
    operation_id: str,
    operation: str,
    target_ids: dict[str, Any],
    element_id: object = None,
    method: str | None,
    outcome: str,
    duration_ms: float,
) -> None:
    safe_target_ids: dict[str, Any] = {
        key: value
        for key, value in target_ids.items()
        if key in {"hwnd", "pid"} and isinstance(value, int)
    }
    if isinstance(element_id, str) and element_id:
        safe_target_ids["element_digest"] = hashlib.sha256(element_id.encode("utf-8")).hexdigest()[
            :24
        ]
    row = {
        "schema_version": "agent-tools.computer-action-journal/v1",
        "operation_id": operation_id,
        "operation": operation,
        "target_ids": safe_target_ids,
        "method": bounded_text(method, 96) if method else None,
        "outcome": outcome,
        "duration_ms": duration_ms,
        "timestamp_unix_ms": int(time.time() * 1000),
    }
    line = (json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")
    try:
        descriptor = os.open(
            _state_dir() / JOURNAL_FILE,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ComputerError(
            "computer_action_journal_failed",
            "The computer action journal could not be written.",
        ) from exc


def _bounded_metadata(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    safe = "".join(character for character in value if 32 <= ord(character) < 127)
    return bounded_text(safe, MAX_OWNER_CHARS) or None


def _utf16_units(value: str) -> int:
    try:
        return len(value.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        return MAX_EXPLANATION_CHARS + 1


def _notification_environment() -> dict[str, str]:
    environment: dict[str, str] = {"PYTHONIOENCODING": "utf-8"}
    for name in ("SystemRoot", "WINDIR", "TEMP", "TMP"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _read_notification_pipe(stream: Any) -> tuple[bytes, bool]:
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.PeekNamedPipe.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPDWORD,
        wintypes.LPDWORD,
        wintypes.LPDWORD,
    )
    kernel32.PeekNamedPipe.restype = wintypes.BOOL
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
    available = wintypes.DWORD()
    if not kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
        error = ctypes.get_last_error()
        if error in {109, 232}:
            return b"", True
        raise OSError(error, "PeekNamedPipe failed")
    if available.value == 0:
        return b"", False
    chunk = os.read(stream.fileno(), min(4096, available.value))
    return chunk, not chunk


def _terminate_notification_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass
