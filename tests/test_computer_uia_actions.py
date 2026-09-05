from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_tools.computer import uia_winapp
from agent_tools.computer.models import Bounds, ComputerError, WindowIdentity
from agent_tools.computer.uia_winapp import (
    ProcessResult,
    WinAppAdapter,
    WinAppBinding,
)

_REAL_NATIVE_UIA_ACTION = WinAppAdapter._native_uia_action
_REAL_NATIVE_ELEMENT_STATE = WinAppAdapter._native_element_state


def _identity() -> WindowIdentity:
    return WindowIdentity(
        hwnd=123,
        pid=456,
        thread_id=789,
        process="fixture.exe",
        title="Task-owned fixture",
        class_name="FixtureClass",
        bounds=Bounds(10, 20, 800, 600),
        process_started=999,
    )


def _binding() -> WinAppBinding:
    return WinAppBinding(
        path=Path("winapp.exe"),
        source="fixture",
        version=uia_winapp.WINAPP_VERSION,
        sha256=uia_winapp.WINAPP_EXE_SHA256,
    )


def _properties(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "AutomationId": "TaskFixture",
        "IsPassword": "False",
        "IsEnabled": "True",
        "IsOffscreen": "False",
        "IsInvokePatternAvailable": "True",
        "IsValuePatternAvailable": "True",
        "ValueIsReadOnly": "False",
        "HorizontallyScrollable": "False",
        "VerticallyScrollable": "False",
    }
    values.update(overrides)
    return {"elementId": "fixture-element-a1b2", "properties": values}


@pytest.fixture(autouse=True)
def native_uia_fixture(monkeypatch) -> None:
    def state(_self, _identity, *, properties):
        patterns: list[str] = []
        mapping = (
            ("IsInvokePatternAvailable", "uia.InvokePattern"),
            ("IsTogglePatternAvailable", "uia.TogglePattern"),
            ("IsSelectionItemPatternAvailable", "uia.SelectionItemPattern"),
            ("IsExpandCollapsePatternAvailable", "uia.ExpandCollapsePattern"),
            ("IsValuePatternAvailable", "uia.ValuePattern"),
            ("IsRangeValuePatternAvailable", "uia.RangeValuePattern"),
            ("IsLegacyIAccessiblePatternAvailable", "uia.LegacyIAccessible"),
        )
        for key, pattern in mapping:
            if str(properties.get(key)).casefold() == "true":
                patterns.append(pattern)
        if (
            str(properties.get("HorizontallyScrollable")).casefold() == "true"
            or str(properties.get("VerticallyScrollable")).casefold() == "true"
        ):
            patterns.append("uia.ScrollPattern")
        return {
            "automation_id": properties.get("AutomationId", "TaskFixture"),
            "runtime_id": "runtime-1",
            "is_password": str(properties.get("IsPassword")).casefold() == "true",
            "enabled": str(properties.get("IsEnabled", "True")).casefold() == "true",
            "patterns": patterns,
            "range_value_bits": properties.get("RangeValueBits"),
            "value_pattern_status": ("available" if "uia.ValuePattern" in patterns else "absent"),
            "range_value_pattern_status": (
                "available" if "uia.RangeValuePattern" in patterns else "absent"
            ),
            "legacy_value_read_only": properties.get("LegacyValueIsReadOnly"),
            "legacy_value_protected": properties.get("LegacyValueProtected"),
            "has_keyboard_focus": (
                str(properties.get("HasKeyboardFocus", "False")).casefold() == "true"
            ),
        }

    def action(_self, _identity, *, properties, method, **kwargs):
        request = kwargs.get("request")
        focus_required = bool(
            isinstance(request, dict) and request.get("requireKeyboardFocus") is True
        )
        return {
            "count": 1,
            "automationId": properties["automation_id"],
            "runtimeId": properties["runtime_id"],
            "method": method,
            "delivered": True,
            "keyboardFocusVerified": True if focus_required else None,
        }

    monkeypatch.setattr(WinAppAdapter, "_native_element_state", state)
    monkeypatch.setattr(WinAppAdapter, "_native_uia_action", action)


class ActionBackend:
    def __init__(self) -> None:
        self.identity = _identity()
        self.revalidations = 0
        self.security_checks = 0

    def capture_identity(self, hwnd: int) -> WindowIdentity:
        assert hwnd == self.identity.hwnd
        self.revalidations += 1
        return self.identity

    def assert_action_allowed(self, identity: WindowIdentity) -> WindowIdentity:
        assert identity == self.identity
        self.security_checks += 1
        return identity

    def revalidate_identity(self, identity: WindowIdentity) -> WindowIdentity:
        assert identity == self.identity
        self.revalidations += 1
        return identity


def test_mutating_target_call_checks_security_and_identity_at_process_boundary(monkeypatch) -> None:
    backend = ActionBackend()
    adapter = WinAppAdapter(backend)  # type: ignore[arg-type]
    monkeypatch.setattr(adapter, "require", _binding)
    calls: list[list[str]] = []

    def run(_path, arguments, **_kwargs):
        calls.append(arguments)
        return ProcessResult(0, b'{"ok":true}', b"")

    monkeypatch.setattr(uia_winapp, "_run_process", run)

    _resolved_binding, payload = adapter._target_json(
        backend.identity,
        ["ui", "invoke", "fixture-element-a1b2"],
        mutating=True,
        action_method="uia.InvokePattern",
    )

    assert payload == {"ok": True}
    assert backend.security_checks == 1
    assert backend.revalidations == 2
    assert calls[0][-3:] == ["--window", "123", "--json"]


def test_invoke_reports_pattern_and_only_verifies_observable_change(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    responses = iter(
        [
            _properties(
                ToggleState="Off",
                IsInvokePatternAvailable="False",
                IsTogglePatternAvailable="True",
                IsRangeValuePatternAvailable="True",
                RangeValueBits="4029000000000000",
            ),
            _properties(
                ToggleState="On",
                IsInvokePatternAvailable="False",
                IsTogglePatternAvailable="True",
                IsRangeValuePatternAvailable="True",
                RangeValueBits="4042A00000000000",
            ),
        ]
    )
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (_binding(), next(responses)),
    )

    _resolved_binding, result = adapter.invoke(_identity(), selector="fixture-element-a1b2")

    assert result["pattern"] == "uia.TogglePattern"
    assert result["before"]["toggle_state"] == "Off"
    assert result["after"]["toggle_state"] == "On"
    assert "range_value_bits" not in result["before"]
    assert "range_value_bits" not in result["after"]
    assert "value_pattern_status" not in result["before"]
    assert "range_value_pattern_status" not in result["after"]
    assert result["outcome"] == "postcondition_verified"


def test_invoke_without_observable_change_is_truthful_delivery_only(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    responses = iter(
        [
            _properties(),
            _properties(),
        ]
    )
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (_binding(), next(responses)),
    )

    _resolved_binding, result = adapter.invoke(_identity(), selector="fixture-element-a1b2")

    assert result["outcome"] == "delivery_only"
    assert result["postcondition_verified"] is False


def test_native_navigation_reference_selects_with_fresh_pattern_and_state(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    selector = "nuia-" + "a" * 24
    reference = {
        "provider": "native_uia_raw",
        "selector": selector,
        "runtime_id": "runtime-old",
        "automation_id": "TaskNavigation",
        "control_type": "ListItem",
        "class": "ListViewItem",
        "ancestor_signature": "sha256:" + "b" * 24,
        "enabled": True,
        "read_only": False,
        "password": False,
        "expires_unix_ms": 2_000_000_000_000,
    }
    states = iter(
        [
            {
                "automation_id": "TaskNavigation",
                "runtime_id": "runtime-old",
                "is_password": False,
                "enabled": True,
                "patterns": ["uia.SelectionItemPattern"],
                "control_type": "ListItem",
                "class": "ListViewItem",
                "ancestor_signature": "sha256:" + "b" * 24,
                "selected": False,
            },
            {
                "automation_id": "TaskNavigation",
                "runtime_id": "runtime-new",
                "is_password": False,
                "enabled": True,
                "patterns": ["uia.SelectionItemPattern"],
                "control_type": "ListItem",
                "class": "ListViewItem",
                "ancestor_signature": "sha256:" + "b" * 24,
                "selected": True,
            },
        ]
    )
    action_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **_kwargs: next(states),
    )
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda _identity, **kwargs: (
            action_calls.append(kwargs)
            or {
                "count": 1,
                "automationId": "TaskNavigation",
                "runtimeId": "runtime-new",
                "method": "uia.SelectionItemPattern",
                "delivered": True,
                "resolutionStage": "stable_metadata_unique",
            }
        ),
    )

    binding, result = adapter.invoke(
        _identity(),
        selector=selector,
        element_reference=reference,
    )

    assert binding.public()["name"] == "native_uia_raw"
    assert action_calls[0]["method"] == "uia.SelectionItemPattern"
    assert result["outcome"] == "postcondition_verified"
    assert result["before"]["selected"] is False
    assert result["after"]["selected"] is True
    assert result["resolution_stage"] == "stable_metadata_unique"


def test_native_navigation_without_selection_pattern_rejects_before_delivery(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    selector = "nuia-" + "c" * 24
    reference = {
        "provider": "native_uia_raw",
        "selector": selector,
        "runtime_id": "runtime-1",
        "automation_id": "TaskNavigation",
        "control_type": "TreeItem",
        "class": "NavigationViewItem",
        "ancestor_signature": "sha256:" + "d" * 24,
        "enabled": True,
        "read_only": False,
        "password": False,
        "expires_unix_ms": 2_000_000_000_000,
    }
    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **_kwargs: {
            "automation_id": "TaskNavigation",
            "runtime_id": "runtime-1",
            "is_password": False,
            "enabled": True,
            "patterns": ["uia.ExpandCollapsePattern"],
            "control_type": "TreeItem",
            "class": "NavigationViewItem",
            "ancestor_signature": "sha256:" + "d" * 24,
            "selected": None,
        },
    )
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda *_args, **_kwargs: pytest.fail("unsupported pattern reached delivery"),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.invoke(
            _identity(),
            selector=selector,
            element_reference=reference,
        )

    assert raised.value.code == "invoke_pattern_unavailable"


def test_native_navigation_ambiguity_fails_closed_before_delivery(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    selector = "nuia-" + "e" * 24
    reference = {
        "provider": "native_uia_raw",
        "selector": selector,
        "runtime_id": "runtime-old",
        "automation_id": "TaskNavigation",
        "control_type": "TreeItem",
        "class": "NavigationViewItem",
        "ancestor_signature": "sha256:" + "f" * 24,
        "enabled": True,
        "read_only": False,
        "password": False,
        "expires_unix_ms": 2_000_000_000_000,
    }
    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ComputerError(
                "ambiguous_element",
                "fixture duplicates",
                details={
                    "resolution_stage": "stable_metadata_unique",
                    "candidate_count": 2,
                },
            )
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda *_args, **_kwargs: pytest.fail("ambiguous target reached delivery"),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.invoke(
            _identity(),
            selector=selector,
            element_reference=reference,
        )

    assert raised.value.code == "ambiguous_element"
    assert raised.value.details["candidate_count"] == 2


def test_native_state_reads_selection_postcondition_from_trusted_worker(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    monkeypatch.setattr(
        adapter,
        "_native_uia_command",
        lambda *_args, **_kwargs: {
            "count": 1,
            "automationId": "TaskNavigation",
            "runtimeId": "runtime-1",
            "resolutionStage": "runtime_id_exact",
            "isPassword": False,
            "enabled": True,
            "controlType": "listitem",
            "className": "ListViewItem",
            "ancestorSignature": "sha256:" + "a" * 24,
            "patterns": ["uia.SelectionItemPattern"],
            "selectionItemIsSelected": True,
            "rangeBits": None,
            "valuePatternStatus": "absent",
            "rangeValuePatternStatus": "absent",
            "legacyValueReadOnly": None,
            "legacyValueProtected": None,
        },
    )

    state = _REAL_NATIVE_ELEMENT_STATE(
        adapter,
        _identity(),
        properties={"AutomationId": "TaskNavigation"},
    )

    assert state is not None
    assert state["selected"] is True
    assert state["patterns"] == ["uia.SelectionItemPattern"]


@pytest.mark.parametrize(
    ("method", "overrides", "kwargs"),
    [
        (
            "invoke",
            {"IsInvokePatternAvailable": "False"},
            {"selector": "fixture-element-a1b2"},
        ),
        (
            "set_value",
            {
                "IsValuePatternAvailable": "False",
                "IsRangeValuePatternAvailable": "False",
                "IsLegacyIAccessiblePatternAvailable": "False",
            },
            {"selector": "fixture-element-a1b2", "value": "fixture"},
        ),
        (
            "scroll",
            {},
            {
                "selector": "fixture-element-a1b2",
                "direction": "down",
                "to": None,
                "percent": None,
            },
        ),
    ],
)
def test_pattern_unavailable_reports_phase_a_physical_fallback(
    monkeypatch,
    method: str,
    overrides: dict[str, object],
    kwargs: dict[str, object],
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (_binding(), _properties(**overrides)),
    )

    with pytest.raises(ComputerError) as raised:
        getattr(adapter, method)(_identity(), **kwargs)

    assert raised.value.details["fallback"] == {
        "kind": "guarded_physical_input",
        "status": "available",
        "reason": "requires_fresh_capture_and_explicit_allow_physical",
    }


@pytest.mark.parametrize(
    ("method", "error_code", "kwargs"),
    [
        ("invoke", "capability_missing", {"selector": "fixture-element-a1b2"}),
        (
            "set_value",
            "backend_version_mismatch",
            {"selector": "fixture-element-a1b2", "value": "fixture"},
        ),
        (
            "scroll",
            "uia_provider_unavailable",
            {
                "selector": "fixture-element-a1b2",
                "direction": "down",
                "to": None,
                "percent": None,
            },
        ),
        ("invoke", "uia_identity_unavailable", {"selector": "fixture-element-a1b2"}),
        (
            "set_value",
            "native_uia_unavailable",
            {"selector": "fixture-element-a1b2", "value": "fixture"},
        ),
    ],
)
def test_action_preflight_capability_failure_is_rejected_with_phase_a_fallback(
    monkeypatch,
    method: str,
    error_code: str,
    kwargs: dict[str, object],
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    deliveries = 0

    def fail_preflight(*_args, **_kwargs):
        raise ComputerError(error_code, "fixture capability unavailable", exit_code=2)

    def unexpected_delivery(*_args, **_kwargs):
        nonlocal deliveries
        deliveries += 1
        pytest.fail("semantic delivery was attempted after failed preflight")

    monkeypatch.setattr(adapter, "_element_properties", fail_preflight)
    monkeypatch.setattr(adapter, "_native_uia_action", unexpected_delivery)

    with pytest.raises(ComputerError) as raised:
        getattr(adapter, method)(_identity(), **kwargs)

    assert deliveries == 0
    assert raised.value.code == error_code
    assert raised.value.exit_code == 2
    assert raised.value.details == {
        "fallback": {
            "kind": "guarded_physical_input",
            "status": "available",
            "reason": "requires_fresh_capture_and_explicit_allow_physical",
        },
        "outcome": "rejected",
    }


@pytest.mark.parametrize(
    ("method", "property_overrides", "kwargs"),
    [
        ("invoke", {}, {"selector": "fixture-element-a1b2"}),
        (
            "set_value",
            {},
            {"selector": "fixture-element-a1b2", "value": "fixture"},
        ),
        (
            "scroll",
            {"VerticallyScrollable": "True"},
            {
                "selector": "fixture-element-a1b2",
                "direction": "down",
                "to": None,
                "percent": None,
            },
        ),
    ],
)
def test_native_worker_start_loss_after_preflight_has_fallback_without_process_delivery(
    monkeypatch,
    method: str,
    property_overrides: dict[str, object],
    kwargs: dict[str, object],
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    process_starts = 0

    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (_binding(), _properties(**property_overrides)),
    )
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda identity, **action_kwargs: _REAL_NATIVE_UIA_ACTION(
            adapter,
            identity,
            **action_kwargs,
        ),
    )
    monkeypatch.setattr(uia_winapp, "_trusted_powershell", lambda: None)

    def unexpected_process(*_args, **_kwargs):
        nonlocal process_starts
        process_starts += 1
        pytest.fail("native worker process started after availability check failed")

    monkeypatch.setattr(uia_winapp, "_run_process", unexpected_process)

    with pytest.raises(ComputerError) as raised:
        getattr(adapter, method)(_identity(), **kwargs)

    assert process_starts == 0
    assert raised.value.code == "native_uia_unavailable"
    assert raised.value.details == {
        "fallback": {
            "kind": "guarded_physical_input",
            "status": "available",
            "reason": "requires_fresh_capture_and_explicit_allow_physical",
        },
        "outcome": "rejected",
    }


def test_incomplete_reference_is_rejected_before_provider_access(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    provider_accessed = False

    def target(*_args, **_kwargs):
        nonlocal provider_accessed
        provider_accessed = True
        pytest.fail("incomplete reference reached provider")

    monkeypatch.setattr(adapter, "_target_json", target)

    with pytest.raises(ComputerError) as raised:
        adapter._element_properties(
            _identity(),
            "fixture-element-a1b2",
            element_reference={
                "selector": "fixture-element-a1b2",
                "automation_id": "TaskFixture",
                "control_type": "edit",
                "class": None,
                "ancestor_signature": "sha256:" + "a" * 24,
            },
        )

    assert raised.value.code == "element_ref_invalid"
    assert raised.value.details["resolution_stage"] == "reference_metadata"
    assert provider_accessed is False


def test_invoke_element_identity_change_is_truthful_delivery_only(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    responses = iter(
        [
            _properties(),
            {
                "elementId": "replacement-element-c3d4",
                "properties": {
                    "AutomationId": "TaskFixture",
                    "IsPassword": "False",
                    "IsEnabled": "True",
                    "IsInvokePatternAvailable": "True",
                },
            },
        ]
    )
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (_binding(), next(responses)),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.invoke(_identity(), selector="fixture-element-a1b2")

    assert raised.value.code == "stale_element"
    assert raised.value.details["outcome"] == "delivery_only"
    assert raised.value.details["method"] == "uia.InvokePattern"


def test_set_value_reads_back_without_returning_the_value(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    responses = iter(
        [
            _properties(),
            _properties(),
            _properties(),
        ]
    )
    calls: list[list[str]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return _binding(), next(responses)

    monkeypatch.setattr(adapter, "_target_json", target)
    monkeypatch.setattr(
        adapter,
        "_native_method_value_fingerprint",
        lambda *_args, **_kwargs: (
            *uia_winapp._value_fingerprint("secret-fixture"),
            "runtime-1",
            "automation_id_unique",
        ),
    )

    _resolved_binding, result = adapter.set_value(
        _identity(),
        selector="fixture-element-a1b2",
        value="secret-fixture",
    )

    assert result["outcome"] == "postcondition_verified"
    assert result["value_char_count"] == 14
    assert result["readback_char_count"] == 14
    assert "secret-fixture" not in str(result)
    assert all(call[1] != "set-value" for call in calls)
    assert all(call[1] != "get-value" for call in calls)
    assert result["keyboard_focus_status"] == "not_requested"
    assert result["submission_ready"] is False
    assert result["change_required"] is False


def test_set_value_can_require_exact_keyboard_focus(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    responses = iter(
        [
            _properties(HasKeyboardFocus="False"),
            _properties(HasKeyboardFocus="True"),
        ]
    )
    native_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (_binding(), next(responses)),
    )
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda _identity, **kwargs: (
            native_calls.append(kwargs)
            or {
                "count": 1,
                "automationId": kwargs["properties"]["automation_id"],
                "runtimeId": kwargs["properties"]["runtime_id"],
                "method": kwargs["method"],
                "delivered": True,
                "keyboardFocusVerified": True,
            }
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_native_method_value_fingerprint",
        lambda *_args, **_kwargs: (
            *uia_winapp._value_fingerprint("fixture"),
            "runtime-1",
            "automation_id_unique",
        ),
    )

    _resolved_binding, result = adapter.set_value(
        _identity(),
        selector="fixture-element-a1b2",
        value="fixture",
        require_keyboard_focus=True,
    )

    assert native_calls[0]["request"]["requireKeyboardFocus"] is True
    assert result["keyboard_focus_verified"] is True
    assert result["keyboard_focus_status"] == "verified"
    assert result["submission_ready"] is True


def test_set_value_focus_requirement_fails_closed_after_exact_readback(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    responses = iter(
        [
            _properties(HasKeyboardFocus="False"),
            _properties(HasKeyboardFocus="False"),
        ]
    )
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (_binding(), next(responses)),
    )
    monkeypatch.setattr(
        adapter,
        "_native_method_value_fingerprint",
        lambda *_args, **_kwargs: (
            *uia_winapp._value_fingerprint("fixture"),
            "runtime-1",
            "automation_id_unique",
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.set_value(
            _identity(),
            selector="fixture-element-a1b2",
            value="fixture",
            require_keyboard_focus=True,
        )

    assert raised.value.code == "keyboard_focus_postcondition_failed"
    assert raised.value.details["postcondition_verified"] is False
    assert raised.value.details["value_postcondition_verified"] is True
    assert raised.value.details["outcome"] == "delivery_only"


def test_range_set_value_verifies_through_exact_native_range_bits(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    before = _properties(
        IsValuePatternAvailable="False",
        IsRangeValuePatternAvailable="True",
        RangeValueIsReadOnly="False",
        RangeValueBits="4029000000000000",
    )
    after = _properties(
        IsValuePatternAvailable="False",
        IsRangeValuePatternAvailable="True",
        RangeValueIsReadOnly="False",
        RangeValueBits="4042A00000000000",
    )
    responses = iter([before, after, after])
    calls: list[list[str]] = []
    native_calls: list[dict[str, object]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return _binding(), next(responses)

    monkeypatch.setattr(adapter, "_target_json", target)
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda _identity, **kwargs: (
            native_calls.append(kwargs)
            or {
                "count": 1,
                "automationId": kwargs["properties"]["automation_id"],
                "runtimeId": kwargs["properties"]["runtime_id"],
                "method": kwargs["method"],
                "delivered": True,
            }
        ),
    )

    _resolved_binding, result = adapter.set_value(
        _identity(),
        selector="fixture-element-a1b2",
        value="37.25",
    )

    assert result["postcondition_verified"] is True
    assert result["method"] == "uia.RangeValuePattern"
    assert result["readback_char_count"] == 5
    assert [call[1] for call in calls] == ["get-property", "get-property"]
    assert native_calls[0]["request"] == {
        "action": "set-value",
        "rangeBits": "4042A00000000000",
    }


def test_range_set_value_never_falls_back_when_effective_pattern_changes(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    before = _properties(
        IsValuePatternAvailable="False",
        IsRangeValuePatternAvailable="True",
        RangeValueIsReadOnly="False",
        RangeValueBits="4029000000000000",
    )
    changed = _properties(
        IsValuePatternAvailable="True",
        IsRangeValuePatternAvailable="False",
        ValueIsReadOnly="False",
    )
    responses = iter([before, changed])
    calls: list[list[str]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return _binding(), next(responses)

    monkeypatch.setattr(adapter, "_target_json", target)

    with pytest.raises(ComputerError) as raised:
        adapter.set_value(_identity(), selector="fixture-element-a1b2", value="37.25")

    assert raised.value.code == "uia_pattern_changed"
    assert raised.value.details == {
        "outcome": "delivery_only",
        "method": "uia.RangeValuePattern",
    }
    assert [call[1] for call in calls] == ["get-property", "get-property"]


@pytest.mark.parametrize(
    ("value_read_only", "range_read_only", "expected_method"),
    [
        ("False", "True", "uia.ValuePattern"),
        ("True", "False", "uia.RangeValuePattern"),
        ("Unknown", "False", "uia.RangeValuePattern"),
    ],
)
def test_set_value_selects_a_writable_method_without_cross_pattern_rejection(
    monkeypatch,
    value_read_only: str,
    range_read_only: str,
    expected_method: str,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    properties = _properties(
        IsValuePatternAvailable="True",
        IsRangeValuePatternAvailable="True",
        ValueIsReadOnly=value_read_only,
        RangeValueIsReadOnly=range_read_only,
        RangeValueBits="4042A00000000000",
    )
    responses = iter(
        [
            properties,
            properties,
            properties,
        ]
        if expected_method == "uia.ValuePattern"
        else [properties, properties, properties]
    )
    native_methods: list[str] = []

    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (_binding(), next(responses)),
    )
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda _identity, **kwargs: (
            native_methods.append(kwargs["method"])
            or {
                "count": 1,
                "automationId": kwargs["properties"]["automation_id"],
                "runtimeId": kwargs["properties"]["runtime_id"],
                "method": kwargs["method"],
                "delivered": True,
            }
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_native_method_value_fingerprint",
        lambda *_args, **_kwargs: (
            *uia_winapp._value_fingerprint("37.25"),
            "runtime-1",
            "automation_id_unique",
        ),
    )

    _binding_value, result = adapter.set_value(
        _identity(), selector="fixture-element-a1b2", value="37.25"
    )

    assert result["method"] == expected_method
    assert native_methods == [expected_method]


def test_set_value_rejects_legacy_fallback_after_inconclusive_pattern_query(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (
            _binding(),
            _properties(
                IsValuePatternAvailable="False",
                IsRangeValuePatternAvailable="False",
                IsLegacyIAccessiblePatternAvailable="True",
            ),
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **_kwargs: {
            "automation_id": "TaskFixture",
            "runtime_id": "runtime-1",
            "is_password": False,
            "enabled": True,
            "patterns": ["uia.LegacyIAccessible"],
            "range_value_bits": None,
            "value_pattern_status": "failed",
            "range_value_pattern_status": "absent",
            "legacy_value_read_only": False,
            "legacy_value_protected": False,
        },
    )
    deliveries: list[str] = []
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda *_args, **_kwargs: deliveries.append("delivered"),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.set_value(_identity(), selector="fixture-element-a1b2", value="fixture")

    assert raised.value.code == "uia_pattern_query_failed"
    assert deliveries == []


def test_set_value_does_not_claim_read_only_when_an_alternative_query_failed(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (
            _binding(),
            _properties(
                IsValuePatternAvailable="True",
                IsRangeValuePatternAvailable="False",
                ValueIsReadOnly="True",
            ),
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **_kwargs: {
            "automation_id": "TaskFixture",
            "runtime_id": "runtime-1",
            "is_password": False,
            "enabled": True,
            "patterns": ["uia.ValuePattern"],
            "range_value_bits": None,
            "value_pattern_status": "available",
            "range_value_pattern_status": "failed",
            "legacy_value_read_only": None,
        },
    )

    with pytest.raises(ComputerError) as raised:
        adapter.set_value(_identity(), selector="fixture-element-a1b2", value="fixture")

    assert raised.value.code == "uia_pattern_query_failed"


def test_set_value_uses_available_unknown_state_before_alternative_query_failure() -> None:
    properties = {
        "value_available": True,
        "range_value_available": False,
        "value_read_only": None,
        "range_value_read_only": None,
        "value_pattern_status": "available",
        "range_value_pattern_status": "failed",
        "legacy_value_available": False,
    }

    assert uia_winapp._set_value_pattern(properties) == "uia.ValuePattern"


def test_legacy_mutation_is_unavailable_even_when_legacy_state_is_writable() -> None:
    properties = {
        "value_available": False,
        "range_value_available": False,
        "value_pattern_status": "absent",
        "range_value_pattern_status": "absent",
        "legacy_value_available": True,
        "value_read_only": True,
        "legacy_value_read_only": False,
    }

    assert uia_winapp._set_value_pattern(properties) is None


def test_native_legacy_mutation_rejects_before_worker_delivery(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    deliveries: list[str] = []
    monkeypatch.setattr(
        adapter,
        "_native_uia_command",
        lambda *_args, **_kwargs: deliveries.append("delivered"),
    )

    with pytest.raises(ComputerError) as raised:
        _REAL_NATIVE_UIA_ACTION(
            adapter,
            _identity(),
            properties={
                "automation_id": "TaskFixture",
                "runtime_id": "runtime-1",
            },
            request={"action": "set-value", "value": "fixture"},
            method="uia.LegacyIAccessible",
        )

    assert raised.value.code == "set_value_pattern_unavailable"
    assert raised.value.details == {
        "fallback": {
            "kind": "guarded_physical_input",
            "status": "available",
            "reason": "requires_fresh_capture_and_explicit_allow_physical",
        },
        "outcome": "rejected",
        "method": "uia.LegacyIAccessible",
    }
    assert deliveries == []


def test_set_value_rejects_writable_legacy_control_without_delivery(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    deliveries: list[str] = []
    monkeypatch.setattr(
        adapter,
        "_action_properties",
        lambda *_args, **_kwargs: (
            _binding(),
            "fixture-element-a1b2",
            {
                "password": False,
                "enabled": True,
                "runtime_id": "runtime-1",
                "value_available": False,
                "range_value_available": False,
                "value_pattern_status": "absent",
                "range_value_pattern_status": "absent",
                "legacy_value_available": True,
                "legacy_value_read_only": False,
            },
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda *_args, **_kwargs: deliveries.append("delivered"),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.set_value(_identity(), selector="fixture-element-a1b2", value="fixture")

    assert raised.value.code == "set_value_pattern_unavailable"
    assert "fallback" in raised.value.details
    assert deliveries == []


def test_legacy_mutation_treats_unknown_protected_state_as_sensitive(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (
            _binding(),
            _properties(
                IsValuePatternAvailable="False",
                IsRangeValuePatternAvailable="False",
                IsLegacyIAccessiblePatternAvailable="True",
                LegacyValueIsReadOnly=False,
            ),
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.set_value(_identity(), selector="fixture-element-a1b2", value="fixture")

    assert raised.value.code == "password_control"


@pytest.mark.parametrize(
    "value",
    ["1,000", " 1", "1 ", "NaN", "Infinity", "-Infinity", "1e309"],
)
def test_range_set_value_rejects_non_invariant_or_non_finite_input_before_delivery(
    monkeypatch, value: str
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (
            _binding(),
            _properties(
                IsValuePatternAvailable="False",
                IsRangeValuePatternAvailable="True",
                RangeValueIsReadOnly="False",
                RangeValueBits="4029000000000000",
            ),
        ),
    )
    deliveries: list[str] = []
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda *_args, **_kwargs: deliveries.append("delivered"),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.set_value(_identity(), selector="fixture-element-a1b2", value=value)

    assert raised.value.code == "invalid_range_value"
    assert deliveries == []


def test_password_and_read_only_controls_reject_before_mutation(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    calls: list[list[str]] = []

    def password_target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return _binding(), _properties(IsPassword="True")

    monkeypatch.setattr(adapter, "_target_json", password_target)
    with pytest.raises(ComputerError) as password:
        adapter.set_value(_identity(), selector="fixture-element-a1b2", value="never")
    assert password.value.code == "password_control"
    assert len(calls) == 1

    calls.clear()
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda _identity, arguments, **_kwargs: (
            calls.append(arguments) or _binding(),
            _properties(ValueIsReadOnly="True"),
        ),
    )
    with pytest.raises(ComputerError) as read_only:
        adapter.set_value(_identity(), selector="fixture-element-a1b2", value="never")
    assert read_only.value.code == "control_read_only"
    assert len(calls) == 1


def test_semantic_scroll_direction_verifies_percent_and_never_uses_wheel(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    responses = iter(
        [
            _properties(
                VerticallyScrollable="True",
                ScrollVerticalPercent="10",
                VerticalViewSize="20",
            ),
            _properties(
                VerticallyScrollable="True",
                ScrollVerticalPercent="20",
                VerticalViewSize="20",
            ),
        ]
    )
    calls: list[list[str]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return _binding(), next(responses)

    monkeypatch.setattr(adapter, "_target_json", target)

    _resolved_binding, result = adapter.scroll(
        _identity(),
        selector="fixture-element-a1b2",
        direction="down",
        to=None,
        percent=None,
    )

    assert result["before"]["vertical"]["percent"] == 10.0
    assert result["after"]["vertical"]["percent"] == 20.0
    assert result["outcome"] == "postcondition_verified"
    assert all("--wheel" not in call for call in calls)


def test_semantic_scroll_percent_uses_one_direct_uia_call(monkeypatch) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    percentages = iter([10, 49.5])
    calls: list[list[str]] = []
    native_calls: list[dict[str, object]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        if arguments[1] == "get-property":
            percent = next(percentages)
            return _binding(), _properties(
                VerticallyScrollable="True",
                ScrollVerticalPercent=str(percent),
                VerticalViewSize="20",
            )
        raise AssertionError("semantic scroll must not use a WinApp mutation command")

    monkeypatch.setattr(adapter, "_target_json", target)
    monkeypatch.setattr(
        adapter,
        "_native_uia_action",
        lambda _identity, **kwargs: (
            native_calls.append(kwargs)
            or {
                "count": 1,
                "automationId": kwargs["properties"]["automation_id"],
                "runtimeId": kwargs["properties"]["runtime_id"],
                "method": kwargs["method"],
                "delivered": True,
            }
        ),
    )

    _resolved_binding, result = adapter.scroll(
        _identity(),
        selector="fixture-element-a1b2",
        direction=None,
        to=None,
        percent=50,
    )

    assert result["after"]["vertical"]["percent"] == 49.5
    assert result["steps"] == 1
    assert result["request"]["tolerance"] == 1.0
    assert result["method"] == "uia.ScrollPattern.SetScrollPercent"
    assert len(native_calls) == 1
    assert native_calls[0]["request"] == {
        "action": "scroll",
        "mode": "percent",
        "axis": "vertical",
        "percent": 50,
    }
    assert native_calls[0]["method"] == "uia.ScrollPattern.SetScrollPercent"
    assert all("--wheel" not in call for call in calls)


@pytest.mark.parametrize(
    ("direction", "percent", "vertical", "expected_code"),
    [
        (
            "down",
            None,
            {"scrollable": False, "percent": None},
            "scroll_axis_unavailable",
        ),
        (
            None,
            50.0,
            {"scrollable": True, "percent": None},
            "scroll_percent_unavailable",
        ),
    ],
)
def test_unsupported_scroll_details_include_phase_a_fallback(
    monkeypatch,
    direction: str | None,
    percent: float | None,
    vertical: dict[str, object],
    expected_code: str,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    snapshot = {
        "horizontal": {"scrollable": False, "percent": None},
        "vertical": vertical,
    }
    monkeypatch.setattr(
        adapter,
        "_scroll_snapshot",
        lambda *_args, **_kwargs: (
            _binding(),
            "fixture-element-a1b2",
            snapshot,
            {"scroll_available": True},
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.scroll(
            _identity(),
            selector="fixture-element-a1b2",
            direction=direction,
            to=None,
            percent=percent,
        )

    assert raised.value.code == expected_code
    assert raised.value.details["fallback"] == {
        "kind": "guarded_physical_input",
        "status": "available",
        "reason": "requires_fresh_capture_and_explicit_allow_physical",
    }
    assert "guarded_physical_input" in json.dumps(raised.value.details)


def test_text_values_require_exact_readback_and_range_values_allow_numeric_equivalence() -> None:
    assert uia_winapp._values_match("01", "1", method="uia.ValuePattern") is False
    assert uia_winapp._values_match("01", "1", method="uia.RangeValuePattern") is True
    assert (
        uia_winapp._values_match(
            "100000000000",
            "100000000001",
            method="uia.RangeValuePattern",
        )
        is False
    )
    assert uia_winapp._values_match("-0", "0", method="uia.RangeValuePattern") is False


@pytest.mark.parametrize("value", ["+1", ".5", "1.", "1e2", "-1.5E-2"])
def test_invariant_range_value_parser_accepts_dot_decimal_forms(value: str) -> None:
    assert uia_winapp._parse_invariant_range_value(value) is not None


@pytest.mark.parametrize("value", ["١٢", "۱۲", "１２", "1e٢"])
def test_invariant_range_value_parser_rejects_unicode_decimal_digits(value: str) -> None:
    assert uia_winapp._parse_invariant_range_value(value) is None


def test_range_value_verification_uses_observed_bits_for_negative_zero() -> None:
    assert (
        uia_winapp._values_match(
            "-0",
            "0",
            method="uia.RangeValuePattern",
            observed_range_bits="8000000000000000",
        )
        is True
    )
    assert (
        uia_winapp._values_match(
            "-0",
            "0",
            method="uia.RangeValuePattern",
            observed_range_bits="0000000000000000",
        )
        is False
    )


def test_value_fingerprint_is_bounded_for_maximum_escaped_unicode_input() -> None:
    value = '😀"\\\n\t' * 3_199
    utf8_bytes, utf16_chars, digest = uia_winapp._value_fingerprint(value)

    assert len(value) == 15_995
    assert utf8_bytes == len(value.encode("utf-8"))
    assert utf16_chars == len(value.encode("utf-16-le")) // 2
    assert digest == hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_value_fingerprint_rejects_overlength_input() -> None:
    with pytest.raises(ComputerError) as raised:
        uia_winapp._value_fingerprint("x" * 16_001)

    assert raised.value.code == "invalid_value"


def test_value_fingerprint_rejects_unpaired_surrogate_input() -> None:
    with pytest.raises(ComputerError) as raised:
        uia_winapp._value_fingerprint("\ud800")

    assert raised.value.code == "invalid_value"


def test_native_uia_helper_reports_emergency_disable_without_delivery(
    monkeypatch, tmp_path: Path
) -> None:
    backend = ActionBackend()
    adapter = WinAppAdapter(backend)  # type: ignore[arg-type]
    marker = tmp_path / "computer-actions.disabled"
    requests: list[dict[str, object]] = []
    guards: list[str] = []
    monkeypatch.setattr(uia_winapp, "_trusted_powershell", lambda: Path("powershell.exe"))

    def run(_path, _arguments, **kwargs):
        requests.append(json.loads(kwargs["input_bytes"]))
        return ProcessResult(7, b'{"error":"computer_actions_disabled"}', b"")

    monkeypatch.setattr(uia_winapp, "_run_process", run)

    with pytest.raises(ComputerError) as disabled:
        adapter._native_uia_command(
            backend.identity,
            request={
                "action": "invoke",
                "automationId": "TaskFixture",
                "runtimeId": "runtime-1",
                "method": "uia.InvokePattern",
                "disableMarker": str(marker),
            },
            mutating=True,
            method="uia.InvokePattern",
            mutation_guard=lambda: guards.append("checked"),
        )

    assert disabled.value.code == "computer_actions_disabled"
    assert disabled.value.details == {
        "outcome": "rejected",
        "method": "uia.InvokePattern",
    }
    assert guards == ["checked"]
    assert requests[0]["disableMarker"] == str(marker)
    assert requests[0]["owner"]["pid"] > 0
    assert requests[0]["owner"]["processStarted"] > 0
    script = uia_winapp._native_uia_script(123)
    assert "Assert-OwnerAlive" in script
    assert "Assert-OwnerAlive;\n        Assert-ActionsEnabled;" in script
    assert (
        "Assert-TargetSafe $pattern; $deliveryAttempted=$true;\n            $pattern.Invoke()"
    ) in script
    assert "$pattern.SetValue([string]$request.value)" in script
    assert "$pattern.Scroll($horizontal,$vertical)" in script
    assert "Assert-TargetSafe $pattern" in script
    assert "[System.Windows.Automation.RangeValuePattern]::ValueProperty,$false" in script
    assert "patterns=$patterns;rangeBits=$rangeBits" in script
    assert "$selectionItemIsSelected=$null" in script
    assert "$selectionItemIsSelected=$selectionPattern.Current.IsSelected" in script
    assert "selectionItemIsSelected=$selectionItemIsSelected" in script
    assert "$valuePatternStatus='failed';$rangeValuePatternStatus='failed'" in script
    assert "$valuePatternStatus='absent'" in script
    assert "$rangeValuePatternStatus='absent'" in script
    assert "function Is-PatternUnavailable" in script
    assert "$current=$current.InnerException" in script
    assert "valuePatternStatus=$valuePatternStatus" in script
    assert "rangeValuePatternStatus=$rangeValuePatternStatus" in script
    assert "$action -ne 'inspect' -and $action -ne 'read-value'" in script
    assert "if($action -eq 'read-value')" in script
    assert "$readPattern.Current.Value" in script
    assert "legacyValueReadOnly=$legacyValueReadOnly" in script
    assert "legacyValueProtected=$legacyValueProtected" in script
    assert "-band 0x20000000" in script
    assert "function Assert-LegacyUnprotected" in script
    assert "public static class AgentToolsLegacyUia" in script
    assert "const int LegacyPattern = 10018" in script
    assert "[AgentToolsLegacyUia]::State(" in script
    assert "[AgentToolsLegacyUia]::Value(" in script
    assert "[System.Windows.Automation.LegacyIAccessiblePattern]" not in script
    assert "[AgentToolsLegacyUia]::SetValue(" not in script
    assert "public sealed class AgentToolsDeliveryState" not in script
    assert "delivery.Started = true;" not in script
    assert "function Emit-Legacy-Safety" in script
    legacy_privacy_guard = script.split("function Assert-LegacyUnprotected", 1)[1].split(
        "if($action -eq 'inspect')", 1
    )[0]
    assert legacy_privacy_guard.index("Emit-Legacy-Safety") < legacy_privacy_guard.index(
        "password_control"
    )
    legacy_state_method = uia_winapp._NATIVE_SAFETY_CSHARP.split("public static long State", 1)[
        1
    ].split("public static string Value", 1)[0]
    assert legacy_state_method.index("catch (COMException error)") < (
        legacy_state_method.index("return pattern.get_CurrentState();")
    )
    assert "$legacyValueProtected -ne $false" in script
    assert "[System.Text.UTF8Encoding]::new($false,$true)" in script
    assert "$readValue.Length -gt 32000" in script
    assert "$readBytes.Length -gt 64000" in script
    assert "valueSha256=$readHash" in script
    assert "value=$readValue" not in script
    assert "[string]$request.rangeBits" in script
    assert "[BitConverter]::ToDouble($bytes,0)" in script
    assert "[double]::Parse" not in script
    assert script.index("$passwordBefore=$element.Current.IsPassword") < script.index(
        "[System.Windows.Automation.RangeValuePattern]::ValueProperty"
    )
    assert "$passwordAfter=$element.Current.IsPassword" in script
    assert "$runtimeBefore -cne $runtimeAfter" in script
    assert "Emit @{error='stale_element';runtimeId=$runtimeAfter} 4" in script
    assert "resolutionStage='reference_metadata'" in script
    assert "(-not $expectedType)" not in script
    assert "$elementMatches=$root.FindAll(" in script
    assert "$readTarget=Resolve-TargetElement" in script
    assert "$runtimeId=$runtimeBefore" in script
    refresh = script.split("function Refresh-TargetElement", 1)[1].split(
        "$deliveryAttempted=$false", 1
    )[0]
    assert "$freshRuntimeId" in refresh
    assert "$freshRuntimeId -cne $runtimeId" in refresh
    assert "$freshElement,$element" in refresh
    assert "$script:runtimeId=" not in refresh
    assert "focusAttempted=$elementFocusAttempted" in refresh
    assert "focusDelivered=$elementFocusDelivered" in refresh
    assert refresh.index("Emit @{error='stale_element'") < refresh.index(
        "$script:element=$freshElement"
    )
    assert "-not $request.elementRef -and $request.runtimeId" in script
    assert script.index("$runtimeId=$runtimeBefore") < script.index(
        "Assert-LegacyUnprotected;\n    try{"
    )
    assert "$legacyTargetState -eq -1" not in script
    assert "if(Is-PatternUnavailable $_){" in script
    assert "function Is-ElementUnavailable" in script
    assert "public sealed class AgentToolsLegacySafetyException" in script
    assert "function Legacy-Safety-Code" in script
    assert 'AgentToolsLegacySafetyException("ambiguous_element")' in script
    assert 'AgentToolsLegacySafetyException("stale_element")' in script
    assert 'AgentToolsLegacySafetyException("stale_window")' in script
    assert 'AgentToolsLegacySafetyException("target_safety_unavailable")' in script
    assert "$current.HResult -eq -2147220991" in script
    target_safety = script.split("function Assert-TargetSafe", 1)[1].split(
        "function Refresh-TargetElement", 1
    )[0]
    broad_safety_catch = target_safety.rsplit("} catch {", 1)[1]
    assert "Is-ElementUnavailable" in broad_safety_catch
    assert "Is-PatternUnavailable" not in broad_safety_catch
    assert "try{$patternReadOnly=$pattern.Current.IsReadOnly}catch{" in target_safety
    action_catch = script.rsplit("} catch {", 1)[1]
    assert action_catch.index("Emit-Legacy-Safety") < action_catch.index("Is-PatternUnavailable")
    assert action_catch.index("Is-ElementUnavailable") < action_catch.index("Is-PatternUnavailable")
    assert "$pattern.Current.HorizontalScrollPercent" not in script
    assert "$pattern.Current.VerticalScrollPercent" not in script
    assert "$horizontal=if($axis -eq 'horizontal'){$target}\n                else{$no}" in script
    assert "$vertical=if($axis -eq 'vertical'){$target}\n                else{$no}" in script
    assert "if(-not $deliveryAttempted -and (Is-PatternUnavailable $_))" in script


@pytest.mark.parametrize(
    ("returncode", "error_code"),
    [
        (3, "ambiguous_element"),
        (4, "stale_element"),
        (8, "stale_window"),
        (8, "target_safety_unavailable"),
        (11, "computer_action_owner_gone"),
        (7, "computer_actions_disabled"),
        (8, "element_ref_expired"),
        (8, "target_not_foreground"),
    ],
)
def test_native_legacy_safety_failure_never_reports_pattern_fallback(
    monkeypatch,
    returncode: int,
    error_code: str,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    monkeypatch.setattr(uia_winapp, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(
            returncode,
            json.dumps(
                {
                    "error": error_code,
                    "method": "uia.LegacyIAccessible",
                    "delivered": False,
                    "resolutionStage": "runtime_id_exact",
                }
            ).encode(),
            b"",
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter._native_uia_command(
            _identity(),
            request={
                "action": "set-value",
                "automationId": "TaskFixture",
                "runtimeId": "runtime-1",
                "method": "uia.LegacyIAccessible",
            },
            mutating=True,
            method="uia.LegacyIAccessible",
        )

    assert raised.value.code == error_code
    assert raised.value.details.get("outcome") in {None, "rejected"}
    assert "fallback" not in raised.value.details


@pytest.mark.parametrize(
    ("method", "expected_code"),
    [
        ("uia.InvokePattern", "invoke_pattern_unavailable"),
        ("uia.ValuePattern", "set_value_pattern_unavailable"),
        ("uia.ScrollPattern.SetScrollPercent", "scroll_pattern_unavailable"),
    ],
)
def test_native_final_pattern_loss_reports_phase_a_fallback(
    monkeypatch,
    method: str,
    expected_code: str,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    monkeypatch.setattr(uia_winapp, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(
            5,
            json.dumps(
                {
                    "error": "pattern_unavailable",
                    "method": method,
                    "delivered": False,
                    "resolutionStage": "runtime_id_exact",
                }
            ).encode(),
            b"",
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter._native_uia_command(
            _identity(),
            request={
                "action": "invoke",
                "automationId": "TaskFixture",
                "runtimeId": "runtime-1",
                "method": method,
            },
            mutating=True,
            method=method,
        )

    assert raised.value.code == expected_code
    assert raised.value.details["outcome"] == "rejected"
    assert raised.value.details["method"] == method
    assert raised.value.details["resolution_stage"] == "runtime_id_exact"
    assert raised.value.details["fallback"] == {
        "kind": "guarded_physical_input",
        "status": "available",
        "reason": "requires_fresh_capture_and_explicit_allow_physical",
    }


@pytest.mark.parametrize("count", [True, 1.0])
def test_native_action_payload_rejects_non_integer_count(count: object) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    properties = {"automation_id": "TaskFixture", "runtime_id": "runtime-1"}

    with pytest.raises(ComputerError) as raised:
        adapter._validate_native_action_payload(
            {
                "count": count,
                "automationId": "TaskFixture",
                "runtimeId": "runtime-1",
                "method": "uia.ValuePattern",
                "delivered": True,
            },
            properties=properties,
            method="uia.ValuePattern",
        )

    assert raised.value.code == "uia_delivery_unverified"


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected_code"),
    [
        (9, b'{"error":"password_control"}', "password_control"),
        (10, b'{"error":"computer_action_worker_busy"}', "computer_action_worker_busy"),
    ],
)
def test_nonmutating_native_read_errors_do_not_claim_rejected_outcome(
    monkeypatch,
    returncode: int,
    stdout: bytes,
    expected_code: str,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    monkeypatch.setattr(uia_winapp, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(returncode, stdout, b""),
    )

    with pytest.raises(ComputerError) as raised:
        adapter._native_uia_command(
            _identity(),
            request={"action": "read-value", "automationId": "TaskFixture"},
            mutating=False,
            method="uia.ValuePattern",
        )

    assert raised.value.code == expected_code
    assert "outcome" not in raised.value.details
    marked = uia_winapp._mark_delivery_error(raised.value, "uia.ValuePattern")
    assert marked.details["outcome"] == "delivery_only"


@pytest.mark.parametrize(
    "stdout",
    [
        b"[" * 2_000 + b"0" + b"]" * 2_000,
        b'{"count":' + b"9" * 5_000 + b"}",
    ],
)
def test_nonmutating_native_read_normalizes_pathological_json(
    monkeypatch,
    stdout: bytes,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    monkeypatch.setattr(uia_winapp, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(0, stdout, b""),
    )

    with pytest.raises(ComputerError) as raised:
        adapter._native_uia_command(
            _identity(),
            request={"action": "read-value", "automationId": "TaskFixture"},
            mutating=False,
            method="uia.ValuePattern",
        )

    assert raised.value.code == "uia_invalid_response"


@pytest.mark.parametrize(
    "overrides",
    [
        {"valueUtf8Bytes": True},
        {"valueUtf8Bytes": -1},
        {"valueUtf8Bytes": 64_001},
        {"valueUtf16Chars": 1.0},
        {"valueUtf16Chars": -1},
        {"valueUtf16Chars": 32_001},
        {"valueSha256": []},
        {"valueSha256": "not-a-sha256"},
    ],
)
def test_native_method_value_fingerprint_rejects_malformed_payload(
    monkeypatch,
    overrides: dict[str, object],
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    payload: dict[str, object] = {
        "count": 1,
        "automationId": "TaskFixture",
        "runtimeId": "runtime-1",
        "method": "uia.ValuePattern",
        "isPassword": False,
        "valueUtf8Bytes": 7,
        "valueUtf16Chars": 7,
        "valueSha256": hashlib.sha256(b"fixture").hexdigest(),
    }
    payload.update(overrides)
    monkeypatch.setattr(
        adapter,
        "_native_uia_command",
        lambda *_args, **_kwargs: payload,
    )

    with pytest.raises(ComputerError) as raised:
        adapter._native_method_value_fingerprint(
            _identity(),
            properties={
                "automation_id": "TaskFixture",
                "runtime_id": "runtime-1",
            },
            method="uia.ValuePattern",
        )

    assert raised.value.code == "uia_invalid_response"


@pytest.mark.parametrize("method", ["uia.ValuePattern", "uia.LegacyIAccessible"])
def test_referenced_value_readback_threads_reference_and_accepts_stable_recovery(
    monkeypatch,
    method: str,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    reference = {
        "runtime_id": "runtime-old",
        "automation_id": "TaskFixture",
        "control_type": "edit",
        "class": "TextBox",
        "ancestor_signature": "sha256:" + "a" * 24,
    }
    requests: list[dict[str, object]] = []

    def native_command(*_args, **kwargs):
        requests.append(kwargs["request"])
        return {
            "count": 1,
            "automationId": "TaskFixture",
            "runtimeId": "runtime-new",
            "method": method,
            "isPassword": False,
            "valueUtf8Bytes": 7,
            "valueUtf16Chars": 7,
            "valueSha256": hashlib.sha256(b"fixture").hexdigest(),
            "resolutionStage": "stable_metadata_unique",
        }

    monkeypatch.setattr(adapter, "_native_uia_command", native_command)

    assert adapter._native_method_value_fingerprint(
        _identity(),
        properties={
            "automation_id": "TaskFixture",
            "runtime_id": "runtime-old",
            "element_reference": reference,
        },
        method=method,
    ) == (
        7,
        7,
        hashlib.sha256(b"fixture").hexdigest(),
        "runtime-new",
        "stable_metadata_unique",
    )
    assert requests == [
        {
            "action": "read-value",
            "automationId": "TaskFixture",
            "runtimeId": "runtime-old",
            "method": method,
            "elementRef": reference,
        }
    ]


def test_reference_value_verification_reads_final_native_resolution_last(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(ActionBackend())  # type: ignore[arg-type]
    reference = {
        "selector": "fixture-element-a1b2",
        "runtime_id": "runtime-a",
        "automation_id": "TaskFixture",
        "control_type": "edit",
        "class": "TextBox",
        "ancestor_signature": "sha256:" + "a" * 24,
    }
    events: list[str] = []
    final = {
        "automation_id": "TaskFixture",
        "runtime_id": "runtime-a",
        "control_type": "edit",
        "class": "TextBox",
        "ancestor_signature": "sha256:" + "a" * 24,
        "value_available": True,
        "element_reference": reference,
    }

    def action_properties(*_args, **_kwargs):
        events.append("reacquire")
        return _binding(), "provider-a", final

    def fingerprint(*_args, **kwargs):
        events.append("native-read-final")
        assert kwargs["properties"] is final
        return (
            *uia_winapp._value_fingerprint("wrong value on rebuilt B"),
            "runtime-b",
            "stable_metadata_unique",
        )

    monkeypatch.setattr(adapter, "_action_properties", action_properties)
    monkeypatch.setattr(adapter, "_native_method_value_fingerprint", fingerprint)

    result = adapter._read_value_for_verification(
        _identity(),
        "fixture-element-a1b2",
        method="uia.ValuePattern",
        requested_value="requested value",
        expected_value_fingerprint=uia_winapp._value_fingerprint("requested value"),
        element_reference=reference,
    )

    assert events == ["reacquire", "native-read-final"]
    assert result[0] == "provider-a"
    assert result[1] == "runtime-b"
    assert result[3] is False
    assert result[4] == "stable_metadata_unique"


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell interop semantics")
def test_legacy_safety_codes_survive_managed_worker_invocation() -> None:
    powershell = uia_winapp._trusted_powershell()
    if powershell is None:
        pytest.skip("trusted PowerShell unavailable")
    csharp = base64.b64encode(
        (
            uia_winapp._NATIVE_SAFETY_CSHARP
            + "\npublic static class AgentToolsLegacySafetyProbe {"
            + "public static void Throw(string code) {"
            + "throw new AgentToolsLegacySafetyException(code);}}"
        ).encode()
    ).decode()
    script = (
        f"$csharp=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{csharp}'));"
        "Add-Type -TypeDefinition $csharp -ErrorAction Stop;"
        "$codes=@('stale_window','target_safety_unavailable',"
        "'ambiguous_element','stale_element');"
        "foreach($expected in $codes){"
        "try{[AgentToolsLegacySafetyProbe]::Throw($expected);exit 2}catch{"
        "$current=$_.Exception;$actual=$null;"
        "while($null -ne $current){"
        "if($current -is [AgentToolsLegacySafetyException]){"
        "$actual=[string]$current.Code;break};$current=$current.InnerException};"
        "if($actual -cne $expected){exit 3}}}"
    )

    completed = subprocess.run(
        [
            str(powershell),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            uia_winapp._compressed_powershell_command(script),
        ],
        check=False,
        capture_output=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell bridge validation")
def test_native_worker_script_parses_and_embedded_csharp_compiles() -> None:
    powershell = uia_winapp._trusted_powershell()
    if powershell is None:
        pytest.skip("trusted PowerShell unavailable")
    native = base64.b64encode(uia_winapp._native_uia_script(123).encode()).decode()
    csharp = base64.b64encode(uia_winapp._NATIVE_SAFETY_CSHARP.encode()).decode()
    script = (
        f"$native=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{native}'));"
        f"$csharp=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{csharp}'));"
        "$tokens=$null;$errors=$null;"
        "[void][System.Management.Automation.Language.Parser]::ParseInput("
        "$native,[ref]$tokens,[ref]$errors);"
        "if($errors.Count -ne 0){$errors|ForEach-Object{Write-Error $_};exit 1};"
        "Add-Type -TypeDefinition $csharp -ErrorAction Stop"
    )

    completed = subprocess.run(
        [
            str(powershell),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            uia_winapp._compressed_powershell_command(script),
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")


def test_native_uia_zero_match_is_stale_not_ambiguous(monkeypatch) -> None:
    backend = ActionBackend()
    adapter = WinAppAdapter(backend)  # type: ignore[arg-type]
    monkeypatch.setattr(uia_winapp, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(
            3,
            b'{"error":"element_not_unique","count":0}',
            b"",
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter._native_uia_command(
            backend.identity,
            request={"action": "inspect", "automationId": "missing"},
            mutating=False,
            method="uia.inspect",
        )

    assert raised.value.code == "stale_element"


def test_native_uia_launch_failure_is_definitely_not_delivered(monkeypatch) -> None:
    backend = ActionBackend()
    adapter = WinAppAdapter(backend)  # type: ignore[arg-type]
    monkeypatch.setattr(uia_winapp, "_trusted_powershell", lambda: Path("powershell.exe"))

    def fail_start(*_args, **_kwargs):
        raise ComputerError(
            "uia_backend_start_failed",
            "The WinApp backend could not be started.",
        )

    monkeypatch.setattr(uia_winapp, "_run_process", fail_start)

    with pytest.raises(ComputerError) as raised:
        adapter._native_uia_command(
            backend.identity,
            request={"action": "invoke"},
            mutating=True,
            method="uia.InvokePattern",
        )

    assert raised.value.code == "uia_backend_start_failed"
    assert raised.value.details == {"outcome": "failed"}


@pytest.mark.parametrize(
    ("process_started", "expected_outcome"),
    [(False, "failed"), (True, "delivery_only")],
)
def test_native_uia_cleanup_failure_uses_process_lifecycle_for_delivery_state(
    monkeypatch,
    process_started: bool,
    expected_outcome: str,
) -> None:
    backend = ActionBackend()
    adapter = WinAppAdapter(backend)  # type: ignore[arg-type]
    monkeypatch.setattr(uia_winapp, "_trusted_powershell", lambda: Path("powershell.exe"))

    def fail_cleanup(*_args, **_kwargs):
        raise ComputerError(
            "uia_cleanup_failed",
            "fixture cleanup failure",
            details={
                "process_started": process_started,
                "lifecycle_stage": (
                    "post_spawn_cleanup" if process_started else "pre_spawn_cleanup"
                ),
            },
        )

    monkeypatch.setattr(uia_winapp, "_run_process", fail_cleanup)

    with pytest.raises(ComputerError) as raised:
        adapter._native_uia_command(
            backend.identity,
            request={"action": "invoke"},
            mutating=True,
            method="uia.InvokePattern",
        )

    assert raised.value.code == "uia_cleanup_failed"
    assert raised.value.details["process_started"] is process_started
    assert raised.value.details["outcome"] == expected_outcome


def test_native_uia_post_spawn_pipe_failure_is_delivery_only(monkeypatch) -> None:
    backend = ActionBackend()
    adapter = WinAppAdapter(backend)  # type: ignore[arg-type]
    monkeypatch.setattr(uia_winapp, "_trusted_powershell", lambda: Path("powershell.exe"))

    def fail_pipe(*_args, **_kwargs):
        raise ComputerError(
            "uia_pipe_unavailable",
            "fixture pipe unavailable",
            details={
                "process_started": True,
                "lifecycle_stage": "post_spawn_pre_io",
            },
        )

    monkeypatch.setattr(uia_winapp, "_run_process", fail_pipe)

    with pytest.raises(ComputerError) as raised:
        adapter._native_uia_command(
            backend.identity,
            request={"action": "invoke"},
            mutating=True,
            method="uia.InvokePattern",
        )

    assert raised.value.code == "uia_pipe_unavailable"
    assert raised.value.details["process_started"] is True
    assert raised.value.details["outcome"] == "delivery_only"
