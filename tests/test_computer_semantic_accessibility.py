from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_tools.computer import semantic_accessibility as semantic
from agent_tools.computer import uia_winapp
from agent_tools.computer.element_refs import ElementReferenceStore, attach_element_references
from agent_tools.computer.models import Bounds, ComputerError, WindowIdentity
from agent_tools.computer.rendering import render_inspect, render_read
from agent_tools.computer.uia_winapp import ProcessResult, WinAppAdapter, WinAppBinding


def _identity() -> WindowIdentity:
    return WindowIdentity(
        hwnd=123,
        pid=456,
        thread_id=789,
        process="fixture.exe",
        title="Semantic fixture",
        class_name="FixtureClass",
        bounds=Bounds(10, 20, 800, 600),
        process_started=999,
        session_id=1,
        desktop_name="Default",
    )


def _binding() -> WinAppBinding:
    return WinAppBinding(
        path=Path("winapp.exe"),
        source="fixture",
        version="0.6.0",
        sha256="a" * 64,
    )


class FakeBackend:
    def __init__(self) -> None:
        self.identity = _identity()
        self.revalidations = 0

    def capture_identity(self, _hwnd: int) -> WindowIdentity:
        return self.identity

    def revalidate_identity(self, identity: WindowIdentity) -> WindowIdentity:
        self.revalidations += 1
        assert identity == self.identity
        return identity


class FakeWinApp:
    def __init__(
        self,
        elements: list[dict[str, Any]],
        *,
        truncated: bool = False,
    ) -> None:
        self.elements = elements
        self.truncated = truncated
        self.interactive_requests: list[bool] = []

    def inspect(self, _identity: WindowIdentity, **kwargs):
        self.interactive_requests.append(kwargs["interactive"])
        return _binding(), {
            "depth": kwargs["depth"],
            "interactive": kwargs["interactive"],
            "count": len(self.elements),
            "truncated": self.truncated,
            "values_omitted": True,
            "elements": self.elements,
        }


class FakeNative:
    def __init__(
        self,
        inspections: dict[str, dict[str, Any] | ComputerError],
    ) -> None:
        self.inspections = inspections
        self.views: list[str] = []

    def inspect(self, _identity: WindowIdentity, *, view: str, **_kwargs):
        self.views.append(view)
        outcome = self.inspections[view]
        if isinstance(outcome, ComputerError):
            raise outcome
        return semantic.NativeUiaBinding(view), outcome


class FailingWinApp:
    def __init__(self, error: ComputerError) -> None:
        self.error = error

    def inspect(self, _identity: WindowIdentity, **_kwargs):
        raise self.error


def _empty_document() -> dict[str, Any]:
    return {
        "element": "document-a1",
        "control_type": "Document",
        "name": None,
        "depth": 1,
        "parent": None,
        "actionable": False,
        "bounds": {"x": 0, "y": 0, "width": 600, "height": 400},
    }


def _native_element(
    runtime_id: str,
    *,
    parent: str | None = None,
    name: str | None = None,
    control_type: str = "Pane",
    patterns: list[str] | None = None,
    depth: int = 0,
    password: bool | None = False,
    value: str | None = None,
    enabled: bool | None = True,
    offscreen: bool | None = False,
    automation_id: str | None = None,
    class_name: str = "Fixture",
) -> dict[str, Any]:
    normalized_patterns = patterns or []
    return {
        "runtime_id": runtime_id,
        "parent_runtime_id": parent,
        "automation_id": automation_id,
        "name": name,
        "control_type": control_type,
        "class": class_name,
        "framework": "Fixture",
        "value": value,
        "text": None,
        "bounds": {"x": 1, "y": 2, "width": 100, "height": 20},
        "depth": depth,
        "enabled": enabled,
        "offscreen": offscreen,
        "password": password,
        "read_only": False if "value" in normalized_patterns else None,
        "range_read_only": False if "range_value" in normalized_patterns else None,
        "patterns": normalized_patterns,
        "scroll": None,
    }


def test_native_navigation_support_requires_provider_proven_selection_and_identity() -> None:
    normalized = semantic._normalize_native_elements(
        [
            _native_element(
                "nav-1",
                control_type="TreeItem",
                patterns=["selection_item", "expand_collapse"],
                automation_id="TaskNavigationOne",
                class_name="NavigationViewItem",
            ),
            _native_element(
                "nav-2",
                control_type="ListItem",
                patterns=["selection_item"],
                automation_id="TaskNavigationTwo",
                class_name="ListViewItem",
            ),
            _native_element(
                "nav-3",
                control_type="TreeItem",
                patterns=["expand_collapse"],
                automation_id="TaskNavigationThree",
                class_name="NavigationViewItem",
            ),
            _native_element(
                "nav-4",
                control_type="Button",
                patterns=["selection_item"],
                automation_id="TaskNotNavigation",
                class_name="Button",
            ),
            _native_element(
                "nav-5",
                control_type="TreeItem",
                patterns=["selection_item"],
                class_name="NavigationViewItem",
            ),
            _native_element(
                "nav-6",
                control_type="ListItem",
                patterns=["selection_item"],
                automation_id="TaskDisabledNavigation",
                class_name="ListViewItem",
                enabled=False,
            ),
        ],
        provider="native_uia_raw",
    )

    assert [item["action_supported"] for item in normalized] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert all(item["provider"] == "native_uia_raw" for item in normalized)


@pytest.mark.parametrize("view", ["control", "content"])
def test_native_diagnostic_views_never_advertise_navigation_actions(view: str) -> None:
    normalized = semantic._normalize_native_elements(
        [
            _native_element(
                "nav-1",
                control_type="TreeItem",
                patterns=["selection_item"],
                automation_id="TaskNavigation",
                class_name="NavigationViewItem",
            )
        ],
        provider=f"native_uia_{view}",
    )

    assert semantic.NativeUiaBinding(view).public()["actions_exposed"] is False
    assert normalized[0]["action_supported"] is False


def _native_inspection(
    elements: list[dict[str, Any]],
    *,
    view: str,
) -> dict[str, Any]:
    normalized = semantic._normalize_native_elements(
        elements,
        provider=f"native_uia_{view}",
    )
    return {
        "depth": 12,
        "count": len(normalized),
        "truncated": False,
        "values_omitted": True,
        "elements": normalized,
    }


def test_default_inspection_stays_compact_and_does_not_probe_fallback() -> None:
    backend = FakeBackend()
    native = FakeNative({})

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=3,
        interactive=False,
        max_elements=50,
        backend=backend,
        winapp_adapter=FakeWinApp([_empty_document()]),
        native_adapter=native,
    )

    assert result["backend"]["name"] == "winapp"
    assert result["semantic_surface_shallow"] is None
    assert result["semantic_surface_inconclusive"] == {
        "code": "semantic_surface_inconclusive",
        "provider": "winapp",
        "element_count": 1,
        "maximum_depth": 1,
        "actionable_count": 0,
        "text_bearing_count": 0,
        "requested_depth": 3,
        "reason": "deeper_provider_diagnostics_required",
        "candidate_reason": "no_actionable_or_text_bearing_elements",
    }
    assert result["provider_attempts"][0]["status"] == "inconclusive_selected"
    assert "fallbacks" not in result
    assert native.views == []


def test_native_deadline_failure_after_winapp_success_does_not_restart_fallback(
    monkeypatch,
) -> None:
    native = FakeNative(
        {
            "raw": _native_inspection([], view="raw"),
            "control": _native_inspection([], view="control"),
            "content": _native_inspection([], view="content"),
        }
    )
    monkeypatch.setattr(
        semantic,
        "_require_deadline",
        lambda _deadline: (_ for _ in ()).throw(
            ComputerError("uia_timeout", "native deadline exhausted")
        ),
    )

    with pytest.raises(ComputerError) as raised:
        semantic.inspect_semantic_window(
            hwnd=123,
            depth=12,
            interactive=False,
            max_elements=50,
            backend=FakeBackend(),
            winapp_adapter=FakeWinApp([_empty_document()]),
            native_adapter=native,
        )

    assert raised.value.code == "uia_timeout"
    assert native.views == ["raw", "control", "content"]


def test_winapp_interactive_diagnostics_use_the_unfiltered_tree() -> None:
    winapp = FakeWinApp(
        [
            {
                "element": "status-a1",
                "control_type": "Text",
                "name": "Ready",
                "depth": 1,
                "parent": None,
                "actionable": False,
                "bounds": {"x": 0, "y": 0, "width": 100, "height": 20},
            }
        ]
    )

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=3,
        interactive=True,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=winapp,
        native_adapter=FakeNative({}),
    )

    assert winapp.interactive_requests == [True]
    assert result["semantic_surface_shallow"] is None
    assert result["count"] == 0
    assert result["interactive"] is True


def test_deep_inspection_uses_fixed_view_order_and_selects_useful_raw() -> None:
    raw = _native_inspection(
        [
            _native_element("1", name="Fixture"),
            _native_element(
                "2",
                parent="1",
                name="Editor",
                control_type="Edit",
                patterns=["value"],
                depth=1,
            ),
            _native_element(
                "3",
                parent="1",
                name="Save",
                control_type="Button",
                patterns=["invoke"],
                depth=1,
            ),
        ],
        view="raw",
    )
    shallow = _native_inspection([_native_element("1")], view="control")
    native = FakeNative({"raw": raw, "control": shallow, "content": shallow})

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp([_empty_document()]),
        native_adapter=native,
    )

    assert native.views == ["raw", "control", "content"]
    assert result["backend"]["name"] == "native_uia_raw"
    assert result["semantic_surface_shallow"] is None
    assert result["provider_attempts"][1]["status"] == "selected"
    editor = next(item for item in result["elements"] if item["name"] == "Editor")
    assert editor["provider"] == "native_uia_raw"
    assert editor["role"] == "Edit"
    assert editor["read_only"] is False
    assert editor["password"] is False
    assert editor["actionable"] is True
    assert editor["action_supported"] is False
    assert editor["hierarchy"]["ancestor_signature"].startswith("sha256:")
    assert editor["locator"]["strategy"] == "runtime_ancestor_v1"


def test_first_native_action_replaces_a_named_shallow_winapp_container() -> None:
    raw = _native_inspection(
        [
            _native_element(
                "51",
                name="Save",
                control_type="Button",
                patterns=["invoke"],
            )
        ],
        view="raw",
    )
    other = _native_inspection([], view="control")

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp(
            [
                {
                    "element": "window-a1",
                    "control_type": "Window",
                    "name": "Application",
                    "depth": 0,
                    "parent": None,
                    "actionable": False,
                    "bounds": {"x": 0, "y": 0, "width": 600, "height": 400},
                }
            ]
        ),
        native_adapter=FakeNative({"raw": raw, "control": other, "content": other}),
    )

    assert result["backend"]["name"] == "native_uia_raw"
    assert result["elements"][0]["name"] == "Save"
    assert result["elements"][0]["actionable"] is True
    assert result["semantic_surface_shallow"] is None


def test_disabled_native_action_does_not_replace_named_shallow_winapp_container() -> None:
    raw = _native_inspection(
        [
            _native_element(
                "52",
                name="Save",
                control_type="Button",
                patterns=["invoke"],
                enabled=False,
            )
        ],
        view="raw",
    )
    other = _native_inspection([], view="control")

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp(
            [
                {
                    "element": "window-a2",
                    "control_type": "Window",
                    "name": "Application",
                    "depth": 0,
                    "parent": None,
                    "actionable": False,
                    "enabled": True,
                    "offscreen": False,
                    "bounds": {"x": 0, "y": 0, "width": 600, "height": 400},
                }
            ]
        ),
        native_adapter=FakeNative({"raw": raw, "control": other, "content": other}),
    )

    assert result["backend"]["name"] == "winapp"
    assert result["elements"][0]["name"] == "Application"
    assert result["semantic_surface_shallow"]["reason"] == "container_only_surface"


def test_disabled_winapp_action_does_not_block_enabled_native_first_action() -> None:
    raw = _native_inspection(
        [
            _native_element(
                "53",
                name="Save",
                control_type="Button",
                patterns=["invoke"],
            )
        ],
        view="raw",
    )
    other = _native_inspection([], view="control")

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp(
            [
                {
                    "element": "window-a3",
                    "control_type": "Window",
                    "name": "Application",
                    "depth": 0,
                    "parent": None,
                    "actionable": True,
                    "enabled": False,
                    "offscreen": False,
                    "bounds": {"x": 0, "y": 0, "width": 600, "height": 400},
                }
            ]
        ),
        native_adapter=FakeNative({"raw": raw, "control": other, "content": other}),
    )

    assert result["backend"]["name"] == "native_uia_raw"
    assert result["elements"][0]["name"] == "Save"
    assert result["elements"][0]["enabled"] is True
    assert result["elements"][0]["offscreen"] is False
    assert result["elements"][0]["locator"]["strategy"] == "runtime_ancestor_v1"
    assert result["semantic_surface_shallow"] is None


def test_conclusive_raw_replaces_inconclusive_winapp_prefix() -> None:
    raw = _native_inspection(
        [
            _native_element("1", name="Fixture"),
            _native_element(
                "2",
                parent="1",
                name="Editor",
                control_type="Edit",
                patterns=["value"],
                depth=1,
            ),
        ],
        view="raw",
    )
    shallow = _native_inspection([_native_element("1")], view="control")

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp([_empty_document()], truncated=True),
        native_adapter=FakeNative({"raw": raw, "control": shallow, "content": shallow}),
    )

    assert result["backend"]["name"] == "native_uia_raw"
    assert result["semantic_surface_shallow"] is None
    assert result["semantic_surface_inconclusive"] is None
    assert result["provider_attempts"][0]["status"] == "not_selected"
    assert result["provider_attempts"][1]["status"] == "selected"


def test_deep_provider_consensus_marks_missing_document_surface_shallow() -> None:
    winapp_elements = [
        {
            "element": "address",
            "control_type": "Edit",
            "name": "Address",
            "depth": 2,
            "parent": "toolbar",
            "actionable": True,
            "bounds": {"x": 0, "y": 0, "width": 300, "height": 20},
        }
    ]
    raw = _native_inspection(
        [
            _native_element("1", name="Window"),
            _native_element("2", parent="1", control_type="Document", depth=1),
        ],
        view="raw",
    )
    other = _native_inspection([_native_element("1", name="Window")], view="control")
    native = FakeNative({"raw": raw, "control": other, "content": other})

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp(winapp_elements),
        native_adapter=native,
    )

    assert result["backend"]["name"] == "winapp"
    assert result["semantic_surface_shallow"]["reason"] == (
        "provider_consensus_empty_document_fragment"
    )
    assert result["provider_attempts"][0]["status"] == "shallow_selected"


def test_truncated_raw_empty_document_with_complete_shell_views_is_shallow() -> None:
    winapp_elements = [
        {
            "element": f"winapp-shell-{index}",
            "control_type": "Button" if index < 13 else "Pane",
            "name": f"Browser shell {index}" if index < 17 else None,
            "depth": min(index // 4, 9),
            "parent": None,
            "actionable": index < 13,
            "enabled": True,
            "offscreen": False,
            "bounds": {"x": 0, "y": 0, "width": 300, "height": 20},
        }
        for index in range(39)
    ]
    raw_elements = [
        _native_element("raw-root", name="Browser window"),
        _native_element(
            "raw-document",
            parent="raw-root",
            control_type="Document",
            depth=1,
        ),
        *[
            _native_element(
                f"raw-shell-{index}",
                parent="raw-root",
                name=f"Browser shell {index}" if index < 42 else None,
                control_type="Button",
                patterns=["invoke"] if index < 35 else [],
                depth=min((index % 10) + 1, 10),
            )
            for index in range(195)
        ],
    ]
    raw = _native_inspection(raw_elements, view="raw")
    raw["truncated"] = True
    complete_shell = [
        _native_element(
            f"shell-{index}",
            name=f"Browser shell {index}" if index < 16 else None,
            control_type="Button",
            patterns=["invoke"] if index < 13 else [],
            depth=min(index // 4, 9),
        )
        for index in range(37)
    ]

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=200,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp(winapp_elements),
        native_adapter=FakeNative(
            {
                "raw": raw,
                "control": _native_inspection(complete_shell, view="control"),
                "content": _native_inspection(complete_shell, view="content"),
            }
        ),
    )

    assert result["backend"]["name"] == "winapp"
    assert result["count"] == 39
    assert result["semantic_surface_shallow"]["reason"] == (
        "provider_consensus_empty_document_fragment"
    )
    assert result["semantic_surface_inconclusive"] is None
    assert result["fallbacks"] == {
        "observation": {
            "kind": "targeted_screenshot",
            "status": "conditional",
            "when": "only_if_missing_state_is_genuinely_visual",
        }
    }
    assert "semantic_action_unavailable" not in result
    assert result["provider_attempts"][0]["status"] == "shallow_selected"
    assert result["provider_attempts"][0]["reason"] == (
        "provider_consensus_empty_document_fragment"
    )
    assert result["provider_attempts"][1]["status"] == "inconclusive"
    assert result["provider_attempts"][1]["reason"] == (
        "truncated_before_surface_classification"
    )
    assert result["provider_attempts"][1]["element_count"] == 197
    assert result["provider_attempts"][1]["maximum_depth"] == 10
    assert result["provider_attempts"][1]["actionable_count"] == 35
    assert result["provider_attempts"][1]["text_bearing_count"] == 43
    assert result["provider_attempts"][2]["status"] == "diagnostic_only"
    assert result["provider_attempts"][3]["status"] == "diagnostic_only"
    for attempt in result["provider_attempts"][2:]:
        assert attempt["element_count"] == 37
        assert attempt["maximum_depth"] == 9
        assert attempt["actionable_count"] == 13
        assert attempt["text_bearing_count"] == 16


def test_winapp_unavailable_selects_native_at_compact_depth() -> None:
    raw = _native_inspection(
        [
            _native_element("1", name="Fixture"),
            _native_element(
                "2",
                parent="1",
                name="Editor",
                control_type="Edit",
                patterns=["value"],
                depth=1,
            ),
        ],
        view="raw",
    )
    other = _native_inspection([_native_element("1")], view="control")
    native = FakeNative({"raw": raw, "control": other, "content": other})

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=3,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FailingWinApp(
            ComputerError("uia_provider_unavailable", "fixture unavailable")
        ),
        native_adapter=native,
    )

    assert result["backend"]["name"] == "native_uia_raw"
    assert result["semantic_surface_shallow"] is None
    assert result["provider_attempts"][0]["error_code"] == "uia_provider_unavailable"
    assert result["provider_attempts"][1]["status"] == "selected"
    assert result["semantic_action_unavailable"]["reason"] == "provider_read_only"
    assert "observation" not in result["fallbacks"]
    assert result["fallbacks"]["action"] == {
        "kind": "guarded_physical_input",
        "status": "available",
        "reason": "requires_fresh_capture_and_explicit_allow_physical",
    }


def test_winapp_unavailable_exposes_only_proven_raw_navigation_action() -> None:
    raw = _native_inspection(
        [
            _native_element("1", name="Fixture"),
            _native_element(
                "2",
                parent="1",
                control_type="TreeItem",
                patterns=["selection_item"],
                depth=1,
                automation_id="TaskNavigation",
                class_name="NavigationViewItem",
            ),
        ],
        view="raw",
    )
    other = _native_inspection([_native_element("1")], view="control")

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=3,
        interactive=True,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FailingWinApp(
            ComputerError("uia_provider_unavailable", "fixture unavailable")
        ),
        native_adapter=FakeNative({"raw": raw, "control": other, "content": other}),
    )

    assert result["backend"]["name"] == "native_uia_raw"
    assert result["backend"]["actions_exposed"] is True
    assert result["count"] == 1
    assert result["elements"][0]["action_supported"] is True
    assert "semantic_action_unavailable" not in result
    assert "fallbacks" not in result


@pytest.mark.parametrize("code", ["uia_backend_unreadable", "uia_process_failed"])
def test_safe_winapp_execution_failures_select_native(code: str) -> None:
    raw = _native_inspection(
        [
            _native_element("1", name="Fixture"),
            _native_element(
                "2",
                parent="1",
                name="Editor",
                control_type="Edit",
                patterns=["value"],
                depth=1,
            ),
        ],
        view="raw",
    )
    other = _native_inspection([_native_element("1")], view="control")

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=3,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FailingWinApp(ComputerError(code, "fixture unavailable")),
        native_adapter=FakeNative({"raw": raw, "control": other, "content": other}),
    )

    assert result["backend"]["name"] == "native_uia_raw"
    assert result["provider_attempts"][0]["error_code"] == code


def test_native_interactive_fallback_filters_non_actionable_elements() -> None:
    raw = _native_inspection(
        [
            _native_element("1", name="Fixture"),
            _native_element(
                "2",
                parent="1",
                name="Save",
                control_type="Button",
                patterns=["invoke"],
                depth=1,
            ),
            _native_element(
                "3",
                parent="1",
                name="Cancel",
                control_type="Button",
                patterns=["invoke"],
                depth=1,
            ),
        ],
        view="raw",
    )
    shallow = _native_inspection([_native_element("1")], view="control")
    native = FakeNative({"raw": raw, "control": shallow, "content": shallow})

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=True,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp([_empty_document()]),
        native_adapter=native,
    )

    assert result["backend"]["name"] == "native_uia_raw"
    assert result["count"] == 2
    assert [item["name"] for item in result["elements"]] == ["Save", "Cancel"]
    assert result["interactive"] is True


def test_native_interactive_filter_requires_enabled_visible_actions() -> None:
    inspection = _native_inspection(
        [
            _native_element("1", name="Fixture"),
            _native_element("2", patterns=["invoke"], enabled=False, depth=1),
            _native_element("3", patterns=["invoke"], offscreen=True, depth=1),
            _native_element("4", patterns=["invoke"], offscreen=None, depth=1),
            _native_element("5", patterns=["invoke"], depth=1),
        ],
        view="raw",
    )

    filtered = semantic._filter_native_inspection(inspection, interactive=True)

    assert filtered["count"] == 1
    assert filtered["elements"][0]["locator"]["runtime_id"] == "5"


def test_native_actionability_requires_effective_mutation_state() -> None:
    no_scroll = {
        "horizontal_scrollable": False,
        "vertical_scrollable": False,
        "horizontal_percent": -1,
        "vertical_percent": -1,
    }
    vertical_scroll = {
        "horizontal_scrollable": False,
        "vertical_scrollable": True,
        "horizontal_percent": -1,
        "vertical_percent": 50,
    }
    elements = semantic._normalize_native_elements(
        [
            {**_native_element("31", patterns=["value"]), "read_only": True},
            {
                **_native_element("32", patterns=["range_value"]),
                "range_read_only": True,
            },
            {**_native_element("33", patterns=["scroll"]), "scroll": no_scroll},
            _native_element("34", patterns=["value"]),
            _native_element("35", patterns=["range_value"]),
            {
                **_native_element("36", patterns=["scroll"]),
                "scroll": vertical_scroll,
            },
            {
                **_native_element("37", patterns=["value", "range_value"]),
                "read_only": True,
                "range_read_only": False,
            },
        ],
        provider="native_uia_raw",
    )

    assert [item["actionable"] for item in elements] == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    combined = elements[-1]
    assert combined["read_only"] is False
    assert combined["value_read_only"] is True
    assert combined["range_read_only"] is False


def test_interactive_filter_does_not_change_native_surface_diagnostic() -> None:
    raw = _native_inspection(
        [
            _native_element("1", name="Fixture"),
            _native_element(
                "2",
                parent="1",
                name="Status",
                control_type="Text",
                depth=1,
            ),
        ],
        view="raw",
    )
    other = _native_inspection([_native_element("1")], view="control")

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=True,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FailingWinApp(
            ComputerError("uia_provider_unavailable", "fixture unavailable")
        ),
        native_adapter=FakeNative({"raw": raw, "control": other, "content": other}),
    )

    assert result["count"] == 0
    assert result["semantic_surface_shallow"] is None


def test_native_control_or_content_view_is_never_selected() -> None:
    raw = _native_inspection([_native_element("1")], view="raw")
    useful = _native_inspection(
        [
            _native_element("1", name="Fixture"),
            _native_element("2", parent="1", name="Save", patterns=["invoke"], depth=1),
        ],
        view="control",
    )

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp([_empty_document()]),
        native_adapter=FakeNative({"raw": raw, "control": useful, "content": useful}),
    )

    assert result["backend"]["name"] == "winapp"
    assert result["semantic_surface_shallow"] is not None
    assert result["provider_attempts"][2]["status"] == "diagnostic_only"


def test_all_semantic_providers_unavailable_returns_bounded_error() -> None:
    errors = {
        view: ComputerError("native_uia_unavailable", f"fixture {view}")
        for view in ("raw", "control", "content")
    }

    with pytest.raises(ComputerError) as raised:
        semantic.inspect_semantic_window(
            hwnd=123,
            depth=3,
            interactive=False,
            max_elements=50,
            backend=FakeBackend(),
            winapp_adapter=FailingWinApp(
                ComputerError("uia_provider_unavailable", "fixture unavailable")
            ),
            native_adapter=FakeNative(errors),
        )

    assert raised.value.code == "semantic_providers_unavailable"
    attempts = raised.value.details["provider_attempts"]
    assert len(attempts) == 4
    assert all(item["status"] == "unavailable" for item in attempts)


def test_deep_shallow_surface_reports_every_failed_provider_and_separate_fallbacks() -> None:
    errors = {
        "raw": ComputerError("uia_timeout", "fixture timeout"),
        "control": ComputerError("native_uia_query_failed", "fixture crash"),
        "content": ComputerError("native_uia_unavailable", "fixture unavailable"),
    }

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp([_empty_document()]),
        native_adapter=FakeNative(errors),
    )

    assert result["semantic_surface_shallow"]["code"] == "semantic_surface_shallow"
    assert [item["provider"] for item in result["provider_attempts"]] == [
        "winapp",
        "native_uia_raw",
        "native_uia_control",
        "native_uia_content",
    ]
    assert [item.get("error_code") for item in result["provider_attempts"][1:]] == [
        "uia_timeout",
        "native_uia_query_failed",
        "native_uia_unavailable",
    ]
    assert result["fallbacks"]["observation"]["when"] == (
        "only_if_missing_state_is_genuinely_visual"
    )
    assert "action" not in result["fallbacks"]
    assert "semantic_action_unavailable" not in result


def test_native_raw_inconclusive_suppresses_observation_fallback() -> None:
    raw = _native_inspection([_native_element("1")], view="raw")
    raw["truncated"] = True
    shallow = _native_inspection([_native_element("1")], view="control")

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp([_empty_document()]),
        native_adapter=FakeNative({"raw": raw, "control": shallow, "content": shallow}),
    )

    assert result["semantic_surface_shallow"] is None
    assert result["semantic_surface_inconclusive"] == {
        "code": "semantic_surface_inconclusive",
        "provider": "winapp",
        "element_count": 1,
        "maximum_depth": 1,
        "actionable_count": 0,
        "text_bearing_count": 0,
        "requested_depth": 12,
        "reason": "native_raw_surface_inconclusive",
        "candidate_reason": "no_actionable_or_text_bearing_elements",
        "diagnostic_reason": "truncated_before_surface_classification",
    }
    assert result["provider_attempts"][0]["status"] == "inconclusive_selected"
    assert result["provider_attempts"][1]["status"] == "inconclusive"
    assert "fallbacks" not in result


def test_truncated_rich_native_raw_is_not_selected_as_complete() -> None:
    raw = _native_inspection(
        [
            _native_element("1", name="Fixture"),
            _native_element(
                "2",
                parent="1",
                name="Editor",
                control_type="Edit",
                patterns=["value"],
                depth=1,
            ),
        ],
        view="raw",
    )
    raw["truncated"] = True
    shallow = _native_inspection([_native_element("1")], view="control")

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp([_empty_document()]),
        native_adapter=FakeNative({"raw": raw, "control": shallow, "content": shallow}),
    )

    assert result["backend"]["name"] == "winapp"
    assert result["semantic_surface_shallow"] is None
    assert result["semantic_surface_inconclusive"]["reason"] == (
        "native_raw_surface_inconclusive"
    )
    assert result["provider_attempts"][1]["status"] == "inconclusive"
    assert result["provider_attempts"][1]["reason"] == (
        "truncated_before_surface_classification"
    )
    assert "fallbacks" not in result


def test_safe_native_execution_failures_do_not_abort_diagnostics() -> None:
    errors = {
        "raw": ComputerError("uia_process_failed", "fixture process failure"),
        "control": ComputerError("uia_pipe_unavailable", "fixture pipe failure"),
        "content": _native_inspection([], view="content"),
    }

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=12,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp([_empty_document()]),
        native_adapter=FakeNative(errors),
    )

    assert result["backend"]["name"] == "winapp"
    assert [item.get("error_code") for item in result["provider_attempts"][1:3]] == [
        "uia_process_failed",
        "uia_pipe_unavailable",
    ]


def test_truncated_shallow_prefix_is_inconclusive_without_observation_fallback() -> None:
    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=3,
        interactive=False,
        max_elements=1,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp([_empty_document()], truncated=True),
        native_adapter=FakeNative({}),
    )

    assert result["semantic_surface_shallow"] is None
    assert result["semantic_surface_inconclusive"] == {
        "code": "semantic_surface_inconclusive",
        "provider": "winapp",
        "element_count": 1,
        "maximum_depth": 1,
        "actionable_count": 0,
        "text_bearing_count": 0,
        "requested_depth": 3,
        "reason": "truncated_before_surface_classification",
        "candidate_reason": "no_actionable_or_text_bearing_elements",
    }
    assert result["provider_attempts"][0]["status"] == "inconclusive_selected"
    assert "fallbacks" not in result


def test_depth_limited_shallow_prefix_is_inconclusive_without_observation_fallback() -> None:
    depth_limited = {**_empty_document(), "depth": 3}

    result = semantic.inspect_semantic_window(
        hwnd=123,
        depth=3,
        interactive=False,
        max_elements=50,
        backend=FakeBackend(),
        winapp_adapter=FakeWinApp([depth_limited]),
        native_adapter=FakeNative({}),
    )

    assert result["semantic_surface_shallow"] is None
    assert result["semantic_surface_inconclusive"] == {
        "code": "semantic_surface_inconclusive",
        "provider": "winapp",
        "element_count": 1,
        "maximum_depth": 3,
        "actionable_count": 0,
        "text_bearing_count": 0,
        "requested_depth": 3,
        "reason": "depth_limit_reached_before_surface_classification",
        "candidate_reason": "no_actionable_or_text_bearing_elements",
    }
    assert result["provider_attempts"][0]["status"] == "inconclusive_selected"
    assert "fallbacks" not in result


def test_cleanup_failure_from_native_provider_is_not_degraded() -> None:
    native = FakeNative(
        {
            "raw": ComputerError("uia_cleanup_failed", "fixture cleanup failure"),
            "control": _native_inspection([], view="control"),
            "content": _native_inspection([], view="content"),
        }
    )

    with pytest.raises(ComputerError, match="cleanup failure") as raised:
        semantic.inspect_semantic_window(
            hwnd=123,
            depth=12,
            interactive=False,
            max_elements=50,
            backend=FakeBackend(),
            winapp_adapter=FakeWinApp([_empty_document()]),
            native_adapter=native,
        )

    assert raised.value.code == "uia_cleanup_failed"


@pytest.mark.parametrize("code", ["stale_window", "invalid_window", "invalid_depth"])
def test_identity_and_input_errors_never_trigger_provider_fallback(code: str) -> None:
    backend = FakeBackend()

    with pytest.raises(ComputerError) as raised:
        semantic.inspect_semantic_window(
            hwnd=123,
            depth=12,
            interactive=False,
            max_elements=50,
            backend=backend,
            winapp_adapter=FailingWinApp(ComputerError(code, "fixture failure")),
            native_adapter=FakeNative({}),
        )

    assert raised.value.code == code


def test_native_identity_failure_is_not_degraded() -> None:
    native = FakeNative(
        {
            "raw": ComputerError("stale_window", "fixture drift"),
            "control": _native_inspection([], view="control"),
            "content": _native_inspection([], view="content"),
        }
    )

    with pytest.raises(ComputerError) as raised:
        semantic.inspect_semantic_window(
            hwnd=123,
            depth=12,
            interactive=False,
            max_elements=50,
            backend=FakeBackend(),
            winapp_adapter=FakeWinApp([_empty_document()]),
            native_adapter=native,
        )

    assert raised.value.code == "stale_window"


@pytest.mark.parametrize(
    ("elements", "expected_code"),
    [
        ([], "stale_element"),
        (
            [
                {"element": "nuia-" + "a" * 24, "password": False},
                {"element": "nuia-" + "a" * 24, "password": False},
            ],
            "ambiguous_element",
        ),
    ],
)
def test_native_read_rejects_stale_or_ambiguous_locator(
    monkeypatch,
    elements: list[dict[str, Any]],
    expected_code: str,
) -> None:
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    monkeypatch.setattr(
        adapter,
        "inspect",
        lambda *_args, **_kwargs: (
            semantic.NativeUiaBinding("raw"),
            {"elements": elements},
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.read(_identity(), element="nuia-" + "a" * 24, max_chars=100)

    assert raised.value.code == expected_code


def test_native_read_rejects_identity_ambiguous_inspection() -> None:
    locator = "nuia-" + "6" * 24

    with pytest.raises(ComputerError) as raised:
        semantic._unique_native_match(
            {
                "truncated": False,
                "identity_ambiguous": True,
                "elements": [{"element": locator}],
            },
            locator,
        )

    assert raised.value.code == "ambiguous_element"


@pytest.mark.parametrize("password", [True, None])
def test_native_read_fails_closed_when_password_state_is_not_false(
    monkeypatch,
    password: bool | None,
) -> None:
    locator = "nuia-" + "b" * 24
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    monkeypatch.setattr(
        adapter,
        "inspect",
        lambda *_args, **_kwargs: (
            semantic.NativeUiaBinding("raw"),
            {
                "elements": [
                    {
                        "element": locator,
                        "password": password,
                        "value": "must-not-leak",
                    }
                ]
            },
        ),
    )

    result = adapter.read(_identity(), element=locator, max_chars=100)[1]

    assert result["text"] is None
    assert result["redacted"] is True
    assert result["redaction_reason"] in {
        "password_control",
        "uia_sensitive_state_unavailable",
    }


@pytest.mark.parametrize(
    ("value_status", "read_text", "max_chars", "expected_text"),
    [
        ("read", "visible", 7, "visible"),
        ("range_read", "12.5", 3, "12."),
    ],
)
def test_native_read_fetches_only_unique_target_and_rechecks_password(
    monkeypatch,
    value_status: str,
    read_text: str,
    max_chars: int,
    expected_text: str,
) -> None:
    locator = "nuia-" + "d" * 24
    runtime_id = "42,7"
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    inspections = iter(
        [
            {
                "truncated": False,
                "elements": [
                    {
                        "element": locator,
                        "password": False,
                        "name": "Editor",
                        "locator": {"runtime_id": runtime_id},
                    }
                ],
            },
            {
                "truncated": False,
                "elements": [
                    {
                        "element": locator,
                        "password": False,
                        "locator": {"runtime_id": runtime_id},
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(
        adapter,
        "inspect",
        lambda *_args, **_kwargs: (semantic.NativeUiaBinding("raw"), next(inspections)),
    )
    calls: list[dict[str, object]] = []

    def read_value(_identity, **kwargs):
        calls.append(kwargs)
        range_read = value_status == "range_read"
        return {
            "match_count": 1,
            "runtime_id_before": runtime_id,
            "runtime_id_after": runtime_id,
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": value_status,
            "text": None if range_read else read_text,
            "char_count": 0 if range_read else semantic._utf16_code_units(read_text),
            "truncated": False,
            "range_bits": "4029000000000000" if range_read else None,
            "tree_truncated": False,
        }

    monkeypatch.setattr(adapter, "_read_runtime_value", read_value)

    result = adapter.read(_identity(), element=locator, max_chars=max_chars)[1]

    assert result["text"] == expected_text
    assert result["char_count"] == semantic._utf16_code_units(read_text)
    assert result["truncated"] is (max_chars < len(read_text))
    assert calls[0]["runtime_id"] == runtime_id
    assert calls[0]["max_chars"] == max_chars


@pytest.mark.parametrize(
    ("value_status", "name", "max_chars", "expected_text", "expected_code"),
    [
        ("name_fallback", "Editor", 100, "Editor", None),
        ("name_fallback", "\U0001f642", 1, "", None),
        ("read_failed", "Editor", 100, None, "native_uia_value_read_failed"),
    ],
)
def test_native_read_falls_back_only_for_conclusively_unsupported_value_pattern(
    monkeypatch,
    value_status: str,
    name: str,
    max_chars: int,
    expected_text: str | None,
    expected_code: str | None,
) -> None:
    locator = "nuia-" + "8" * 24
    runtime_id = "42,18"
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    inspections = iter(
        [
            {
                "truncated": False,
                "elements": [
                    {
                        "element": locator,
                        "password": False,
                        "name": name,
                        "locator": {"runtime_id": runtime_id},
                    }
                ],
            },
            {
                "truncated": False,
                "elements": [
                    {
                        "element": locator,
                        "password": False,
                        "locator": {"runtime_id": runtime_id},
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(
        adapter,
        "inspect",
        lambda *_args, **_kwargs: (semantic.NativeUiaBinding("raw"), next(inspections)),
    )
    fallback_text, fallback_truncated = semantic._truncate_utf16(name, max_chars)
    monkeypatch.setattr(
        adapter,
        "_read_runtime_value",
        lambda *_args, **_kwargs: {
            "match_count": 1,
            "runtime_id_before": runtime_id,
            "runtime_id_after": runtime_id,
            "password_before": False,
            "password_after": False,
            "value_supported": False,
            "value_status": value_status,
            "text": fallback_text if value_status == "name_fallback" else None,
            "char_count": (
                semantic._utf16_code_units(name)
                if value_status == "name_fallback"
                else 0
            ),
            "truncated": fallback_truncated if value_status == "name_fallback" else False,
            "tree_truncated": False,
        },
    )

    if expected_code is not None:
        with pytest.raises(ComputerError) as raised:
            adapter.read(_identity(), element=locator, max_chars=max_chars)
        assert raised.value.code == expected_code
    else:
        result = adapter.read(_identity(), element=locator, max_chars=max_chars)[1]
        assert result["text"] == expected_text
        assert result["char_count"] == semantic._utf16_code_units(name)
        assert result["truncated"] is (semantic._utf16_code_units(name) > max_chars)


def test_native_read_rechecks_deadline_after_final_redaction(monkeypatch) -> None:
    locator = "nuia-" + "7" * 24
    runtime_id = "42,25"
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    inspections = iter(
        [
            {
                "truncated": False,
                "elements": [
                    {
                        "element": locator,
                        "password": False,
                        "locator": {"runtime_id": runtime_id},
                    }
                ],
            },
            {
                "truncated": False,
                "elements": [
                    {
                        "element": locator,
                        "password": False,
                        "locator": {"runtime_id": runtime_id},
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(
        adapter,
        "inspect",
        lambda *_args, **_kwargs: (semantic.NativeUiaBinding("raw"), next(inspections)),
    )
    monkeypatch.setattr(
        adapter,
        "_read_runtime_value",
        lambda *_args, **_kwargs: {
            "match_count": 1,
            "runtime_id_before": runtime_id,
            "runtime_id_after": runtime_id,
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "read",
            "text": "value",
            "char_count": 5,
            "truncated": False,
            "tree_truncated": False,
        },
    )
    monkeypatch.setattr(
        semantic,
        "_require_deadline",
        lambda _deadline: (_ for _ in ()).throw(
            ComputerError("uia_timeout", "deadline exhausted")
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.read(_identity(), element=locator, max_chars=100)

    assert raised.value.code == "uia_timeout"


@pytest.mark.parametrize(
    (
        "raw_text",
        "char_count",
        "transport_truncated",
        "max_chars",
        "forbidden",
        "expected_truncated",
    ),
    [
        ("password=x", 10, False, 10, "x", True),
        ("Bearer abc", 100, True, 10, "abc", True),
        (
            "authorization: Basic secret" + "x" * 23,
            100,
            True,
            50,
            "secret",
            True,
        ),
        ("Basic dXNlcjpwYXNz", 18, False, 100, "dXNlcjpwYXNz", False),
        ("Bearer abc", 10, False, 100, "abc", False),
    ],
)
def test_native_read_caps_the_final_redacted_representation(
    monkeypatch,
    raw_text: str,
    char_count: int,
    transport_truncated: bool,
    max_chars: int,
    forbidden: str,
    expected_truncated: bool,
) -> None:
    locator = "nuia-" + "9" * 24
    runtime_id = "42,9"
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    inspections = iter(
        [
            {
                "truncated": False,
                "elements": [
                    {
                        "element": locator,
                        "password": False,
                        "locator": {"runtime_id": runtime_id},
                    }
                ],
            },
            {
                "truncated": False,
                "elements": [
                    {
                        "element": locator,
                        "password": False,
                        "locator": {"runtime_id": runtime_id},
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(
        adapter,
        "inspect",
        lambda *_args, **_kwargs: (semantic.NativeUiaBinding("raw"), next(inspections)),
    )
    monkeypatch.setattr(
        adapter,
        "_read_runtime_value",
        lambda *_args, **_kwargs: {
            "match_count": 1,
            "runtime_id_before": runtime_id,
            "runtime_id_after": runtime_id,
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "read",
            "text": raw_text,
            "char_count": char_count,
            "truncated": transport_truncated,
            "tree_truncated": False,
        },
    )

    result = adapter.read(_identity(), element=locator, max_chars=max_chars)[1]

    assert semantic._utf16_code_units(result["text"]) <= max_chars
    assert result["truncated"] is expected_truncated
    assert result["redacted"] is True
    assert forbidden not in result["text"]


@pytest.mark.parametrize("password", [True, None])
def test_native_read_discards_value_when_post_read_password_is_not_false(
    monkeypatch,
    password: bool | None,
) -> None:
    locator = "nuia-" + "e" * 24
    runtime_id = "42,8"
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    inspections = iter(
        [
            {
                "truncated": False,
                "elements": [
                    {
                        "element": locator,
                        "password": False,
                        "locator": {"runtime_id": runtime_id},
                    }
                ],
            },
            {
                "truncated": False,
                "elements": [
                    {
                        "element": locator,
                        "password": password,
                        "locator": {"runtime_id": runtime_id},
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(
        adapter,
        "inspect",
        lambda *_args, **_kwargs: (semantic.NativeUiaBinding("raw"), next(inspections)),
    )
    monkeypatch.setattr(
        adapter,
        "_read_runtime_value",
        lambda *_args, **_kwargs: {
            "match_count": 1,
            "runtime_id_before": runtime_id,
            "runtime_id_after": runtime_id,
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "read",
            "text": "must-not-leak",
            "char_count": 13,
            "truncated": False,
            "tree_truncated": False,
        },
    )

    result = adapter.read(_identity(), element=locator, max_chars=100)[1]

    assert result["text"] is None
    assert result["redacted"] is True


def test_native_read_rejects_truncated_metadata_before_value_fetch(monkeypatch) -> None:
    locator = "nuia-" + "f" * 24
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    monkeypatch.setattr(
        adapter,
        "inspect",
        lambda *_args, **_kwargs: (
            semantic.NativeUiaBinding("raw"),
            {
                "truncated": True,
                "elements": [{"element": locator, "password": False}],
            },
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_read_runtime_value",
        lambda *_args, **_kwargs: pytest.fail("value fetch must not run"),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.read(_identity(), element=locator, max_chars=100)

    assert raised.value.code == "ambiguous_element"


def test_native_inspection_rejects_response_larger_than_requested_cap(monkeypatch) -> None:
    backend = FakeBackend()
    adapter = semantic.NativeUiaAdapter(backend)
    elements = [_native_element(str(index)) for index in range(3)]
    monkeypatch.setattr(semantic, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(semantic, "_compressed_powershell_command", lambda script: script)
    monkeypatch.setattr(
        semantic,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(
            0,
            json.dumps(
                {"count": 3, "truncated": True, "elements": elements}
            ).encode(),
            b"",
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.inspect(_identity(), view="raw", depth=12, max_elements=2)

    assert raised.value.code == "native_uia_invalid_response"
    assert backend.revalidations == 2


@pytest.mark.parametrize("operation", ["inspect", "value"])
@pytest.mark.parametrize("failure", [ValueError("oversized integer"), RecursionError()])
def test_native_json_parser_failures_are_bounded(
    monkeypatch,
    operation: str,
    failure: Exception,
) -> None:
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    monkeypatch.setattr(semantic, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(semantic, "_compressed_powershell_command", lambda script: script)
    monkeypatch.setattr(
        semantic,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(0, b"{}", b""),
    )

    def fail_json_loads(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(semantic.json, "loads", fail_json_loads)

    with pytest.raises(ComputerError) as raised:
        if operation == "inspect":
            adapter.inspect(_identity(), view="raw", depth=12, max_elements=3)
        else:
            adapter._read_runtime_value(
                _identity(),
                runtime_id="42,19",
                max_chars=100,
                deadline=float("inf"),
            )

    assert raised.value.code == "native_uia_invalid_response"


@pytest.mark.parametrize(
    ("row", "requested_depth"),
    [
        (_native_element("42,21", depth=3), 2),
        ({**_native_element("42,22", patterns=["value"]), "read_only": None}, 12),
        (_native_element("42,23", patterns=["scroll"]), 12),
    ],
)
def test_native_inspection_rejects_request_envelope_or_incomplete_pattern_metadata(
    monkeypatch,
    row: dict[str, Any],
    requested_depth: int,
) -> None:
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    monkeypatch.setattr(semantic, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(semantic, "_compressed_powershell_command", lambda script: script)
    monkeypatch.setattr(
        semantic,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(
            0,
            json.dumps({"count": 1, "truncated": False, "elements": [row]}).encode(),
            b"",
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.inspect(
            _identity(),
            view="raw",
            depth=requested_depth,
            max_elements=3,
        )

    assert raised.value.code == "native_uia_invalid_response"


@pytest.mark.parametrize("operation", ["inspect", "value"])
def test_native_success_rechecks_deadline_after_response_processing(
    monkeypatch,
    operation: str,
) -> None:
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    monkeypatch.setattr(semantic, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(semantic, "_compressed_powershell_command", lambda script: script)
    payload = (
        {"count": 1, "truncated": False, "elements": [_native_element("42,24")]}
        if operation == "inspect"
        else {
            "match_count": 1,
            "runtime_id_before": "42,24",
            "runtime_id_after": "42,24",
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "read",
            "text": "value",
            "char_count": 5,
            "truncated": False,
            "tree_truncated": False,
        }
    )
    monkeypatch.setattr(
        semantic,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(0, json.dumps(payload).encode(), b""),
    )
    calls = 0

    def require_deadline(_deadline):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ComputerError("uia_timeout", "deadline exhausted")

    monkeypatch.setattr(semantic, "_require_deadline", require_deadline)

    with pytest.raises(ComputerError) as raised:
        if operation == "inspect":
            adapter.inspect(_identity(), view="raw", depth=12, max_elements=3)
        else:
            adapter._read_runtime_value(
                _identity(),
                runtime_id="42,24",
                max_chars=100,
                deadline=float("inf"),
            )

    assert raised.value.code == "uia_timeout"
    assert calls == 3


def test_native_provider_loop_rechecks_deadline_before_success(monkeypatch) -> None:
    native = FakeNative(
        {
            "raw": _native_inspection([_native_element("42,26")], view="raw"),
            "control": _native_inspection([], view="control"),
            "content": _native_inspection([], view="content"),
        }
    )
    monkeypatch.setattr(
        semantic,
        "_require_deadline",
        lambda _deadline: (_ for _ in ()).throw(
            ComputerError("uia_timeout", "deadline exhausted")
        ),
    )

    with pytest.raises(ComputerError) as raised:
        semantic._inspect_native_only(
            identity=_identity(),
            depth=12,
            interactive=False,
            max_elements=50,
            backend=FakeBackend(),
            native_adapter=native,
            winapp_error=ComputerError("uia_timeout", "winapp unavailable"),
        )

    assert raised.value.code == "uia_timeout"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "count": 2,
            "truncated": False,
            "elements": [_native_element("1")],
        },
        {"count": 1, "truncated": False, "elements": ["not-an-element"]},
        {
            "count": 1,
            "truncated": False,
            "elements": [_native_element("1", value="must-not-cross")],
        },
        {
            "count": 1,
            "truncated": False,
            "elements": [{**_native_element("1"), "text": "must-not-cross"}],
        },
        {
            "count": 1,
            "truncated": False,
            "elements": [_native_element("1", name="\ud800")],
        },
        {
            "count": 1,
            "truncated": False,
            "elements": [{**_native_element("1"), "patterns": 1}],
        },
        {
            "count": 1,
            "truncated": False,
            "elements": [{**_native_element("1"), "enabled": "true"}],
        },
        {
            "count": 1,
            "truncated": False,
            "elements": [{**_native_element("1"), "depth": -1}],
        },
        {
            "count": 1,
            "truncated": False,
            "elements": [
                {
                    **_native_element("1"),
                    "bounds": {"x": 1, "y": 2, "width": 3},
                }
            ],
        },
        {
            "count": 1,
            "truncated": False,
            "elements": [{**_native_element("1"), "unexpected": True}],
        },
    ],
)
def test_native_inspection_rejects_partial_or_non_object_rows(
    monkeypatch,
    payload: dict[str, Any],
) -> None:
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    monkeypatch.setattr(semantic, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(semantic, "_compressed_powershell_command", lambda script: script)
    monkeypatch.setattr(
        semantic,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(0, json.dumps(payload).encode(), b""),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.inspect(_identity(), view="raw", depth=12, max_elements=3)

    assert raised.value.code == "native_uia_invalid_response"


def test_native_inspection_returns_only_explicitly_visible_elements(monkeypatch) -> None:
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    elements = [
        _native_element("1", offscreen=False),
        _native_element("2", offscreen=True),
        _native_element("3", offscreen=None),
    ]
    monkeypatch.setattr(semantic, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(semantic, "_compressed_powershell_command", lambda script: script)
    monkeypatch.setattr(
        semantic,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(
            0,
            json.dumps({"count": 3, "truncated": False, "elements": elements}).encode(),
            b"",
        ),
    )

    _binding_value, result = adapter.inspect(
        _identity(), view="raw", depth=12, max_elements=3
    )

    assert result["provider_count"] == 3
    assert result["count"] == 1
    assert result["elements"][0]["locator"]["runtime_id"] == "1"


def test_duplicate_native_runtime_ids_are_excluded_and_inconclusive(monkeypatch) -> None:
    adapter = semantic.NativeUiaAdapter(FakeBackend())
    elements = [
        _native_element("42,20", name="Duplicate A", patterns=["invoke"]),
        _native_element("42,20", name="Duplicate B", patterns=["invoke"]),
    ]
    monkeypatch.setattr(semantic, "_trusted_powershell", lambda: Path("powershell.exe"))
    monkeypatch.setattr(semantic, "_compressed_powershell_command", lambda script: script)
    monkeypatch.setattr(
        semantic,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(
            0,
            json.dumps({"count": 2, "truncated": False, "elements": elements}).encode(),
            b"",
        ),
    )

    _binding_value, result = adapter.inspect(
        _identity(), view="raw", depth=12, max_elements=3
    )
    diagnostic = semantic._surface_inconclusive_diagnostic(
        provider="native_uia_raw",
        inspection=result,
        requested_depth=12,
    )

    assert result["identity_ambiguous"] is True
    assert result["ambiguous_runtime_id_count"] == 1
    assert result["elements"] == []
    assert semantic._surface_stats(result["elements"])["actionable_count"] == 0
    assert diagnostic is not None
    assert diagnostic["reason"] == "ambiguous_runtime_ids"


def test_native_normalization_hard_omits_provider_value_fields() -> None:
    normalized = semantic._normalize_native_elements(
        [{**_native_element("1", value="must-not-cross"), "text": "must-not-cross"}],
        provider="native_uia_raw",
    )

    assert normalized[0]["value"] is None
    assert normalized[0]["text"] is None


def test_native_interactive_filter_excludes_unaddressable_actions() -> None:
    inspection = _native_inspection(
        [_native_element("", patterns=["invoke"])],
        view="raw",
    )

    assert inspection["elements"][0]["element"] is None
    assert inspection["elements"][0]["locator"] is None
    assert inspection["elements"][0]["actionable"] is False
    assert semantic._filter_native_inspection(inspection, interactive=True)["count"] == 0


def test_native_scripts_omit_bulk_values_and_bound_the_queue() -> None:
    inspection_script = semantic._native_inspection_script(123, "raw")
    value_script = semantic._native_value_script(123)

    assert ".Current.Value" not in inspection_script
    assert "($items.Count+$queue.Count)" in inspection_script
    assert "AutomationElement]::NotSupported" in inspection_script
    assert "catch{$script:truncated=$true;return $null}" in inspection_script
    assert "GetSupportedPatterns()" in inspection_script
    assert "catch{$truncated=$true;$child=$null}" in inspection_script
    assert "catch{$truncated=$true}" in inspection_script
    assert inspection_script.count("catch{$truncated=$true}}") == 3
    assert "[char]::IsHighSurrogate($text[$take-1])" in inspection_script
    assert "[char]::IsLowSurrogate($text[$take])" in inspection_script
    assert "function SafeIdentityText" in inspection_script
    assert "runtime_id=SafeIdentityText $runtime 512" in inspection_script
    assert "parent_runtime_id=SafeIdentityText $entry.parent 512" in inspection_script
    assert "automation_id=SafeIdentityText (SafeProperty $element" in inspection_script
    assert "runtime_id=SafeText $runtime 512" not in inspection_script
    assert "automation_id=SafeText (SafeProperty $element" not in inspection_script
    assert "[Windows.Automation.RangeValuePattern]::Pattern" in inspection_script
    assert "range_read_only=$rangeReadOnly" in inspection_script
    assert value_script.count(".Current.Value") == 1
    assert "$raw=[string]$pattern.Current.Value;$valueSupported=$true" in value_script
    assert "$valueStatus='name_fallback'" in value_script
    assert "IsRangeValuePatternAvailableProperty" in value_script
    assert "function PatternAvailability" in value_script
    assert "known=$false;value=$null" in value_script
    assert "$valueAvailability.known" in value_script
    assert "$rangeAvailability.known" in value_script
    assert "[Windows.Automation.RangeValuePattern]::ValueProperty,$false" in value_script
    assert "$number=[double]$rangeValue" in value_script
    assert "$range.Current.Value" not in value_script
    assert "$valueStatus='range_read'" in value_script
    assert "[BitConverter]::GetBytes($number)" in value_script
    assert "[BitConverter]::IsLittleEndian" in value_script
    assert "[Array]::Reverse($bytes)" in value_script
    assert "range_bits=$rangeBits" in value_script
    assert "range_value_exceeds_limit" not in value_script
    assert ".ToString('G17'," not in value_script
    assert ".ToString('R'," not in value_script
    assert "AutomationElement]::NameProperty,$true" in value_script
    assert "function SetBoundedText" in value_script
    assert "$valueStatus='read_failed'" in value_script
    assert "value_status=$valueStatus" in value_script
    assert "[char]::IsHighSurrogate($raw[$take-1])" in value_script
    assert "[char]::IsLowSurrogate($raw[$take])" in value_script
    assert "($visited+$queue.Count)" in value_script
    assert "AutomationElement]::NotSupported" in value_script
    assert "$boundaryChild=$walker.GetFirstChild($element)" in value_script
    assert "if($null -ne $boundaryChild){$treeTruncated=$true}" in value_script
    assert "catch{$script:treeTruncated=$true;return $null}" in value_script
    assert "catch{$treeTruncated=$true;$child=$null}" in value_script
    assert "$passwordBefore -isnot [bool]" in value_script
    assert "$runtimeAfter -ne [string]$request.runtimeId" in value_script
    assert "$valueSupported=$false;$text=$null;$charCount=0" in value_script


def test_cleared_unknown_password_payload_is_valid_for_truthful_redaction() -> None:
    assert semantic._valid_native_value_payload(
        {
            "match_count": 1,
            "runtime_id_before": "42,10",
            "runtime_id_after": "42,10",
            "password_before": False,
            "password_after": None,
            "value_supported": False,
            "value_status": "discarded_unsafe",
            "text": None,
            "char_count": 0,
            "truncated": False,
            "tree_truncated": False,
        },
        runtime_id="42,10",
        limit=100,
    )


def test_value_supported_payload_requires_a_string_text_field() -> None:
    assert not semantic._valid_native_value_payload(
        {
            "match_count": 1,
            "runtime_id_before": "42,11",
            "runtime_id_after": "42,11",
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "read",
            "text": None,
            "char_count": 0,
            "truncated": False,
            "tree_truncated": False,
        },
        runtime_id="42,11",
        limit=100,
    )


@pytest.mark.parametrize(
    "override",
    [
        {
            "value_supported": False,
            "value_status": "name_fallback",
            "text": "must-not-cross",
            "char_count": 14,
            "truncated": False,
        },
        {
            "value_supported": False,
            "value_status": "name_fallback",
            "text": "",
            "char_count": 1,
            "truncated": False,
        },
        {
            "value_supported": True,
            "value_status": "read",
            "text": "abc",
            "char_count": 4,
            "truncated": False,
        },
        {
            "value_supported": True,
            "value_status": "read",
            "text": "abc",
            "char_count": 3,
            "truncated": True,
        },
        {
            "value_supported": True,
            "value_status": "read",
            "text": "abc",
            "char_count": 2**31,
            "truncated": True,
        },
    ],
)
def test_native_value_payload_rejects_uncleared_or_inconsistent_metadata(
    override: dict[str, Any],
) -> None:
    payload = {
        "match_count": 1,
        "runtime_id_before": "42,12",
        "runtime_id_after": "42,12",
        "password_before": False,
        "password_after": False,
        "tree_truncated": False,
        **override,
    }

    assert not semantic._valid_native_value_payload(
        payload,
        runtime_id="42,12",
        limit=3,
    )


@pytest.mark.parametrize(
    "text,char_count,truncated,limit",
    [
        ("abc", 3, False, 3),
        ("abc", 4, True, 3),
        ("\U0001f642", 2, False, 2),
    ],
)
def test_native_value_payload_accepts_exact_supported_metadata(
    text: str,
    char_count: int,
    truncated: bool,
    limit: int,
) -> None:
    assert semantic._valid_native_value_payload(
        {
            "match_count": 1,
            "runtime_id_before": "42,13",
            "runtime_id_after": "42,13",
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "read",
            "text": text,
            "char_count": char_count,
            "truncated": truncated,
            "tree_truncated": False,
        },
        runtime_id="42,13",
        limit=limit,
    )


def test_native_value_payload_accepts_exact_name_fallback_metadata() -> None:
    assert semantic._valid_native_value_payload(
        {
            "match_count": 1,
            "runtime_id_before": "42,15",
            "runtime_id_after": "42,15",
            "password_before": False,
            "password_after": False,
            "value_supported": False,
            "value_status": "name_fallback",
            "text": "Editor",
            "char_count": 6,
            "truncated": False,
            "tree_truncated": False,
        },
        runtime_id="42,15",
        limit=100,
    )


def test_native_value_payload_accepts_exact_range_value_metadata() -> None:
    assert semantic._valid_native_value_payload(
        {
            "match_count": 1,
            "runtime_id_before": "42,28",
            "runtime_id_after": "42,28",
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "range_read",
            "text": None,
            "char_count": 0,
            "truncated": False,
            "range_bits": "4029000000000000",
            "tree_truncated": False,
        },
        runtime_id="42,28",
        limit=100,
    )


@pytest.mark.parametrize(
    "range_bits",
    [
        None,
        "",
        "402900000000000",
        "40290000000000000",
        "402900000000000g",
        "402900000000000a",
        "7FF0000000000000",
        "FFF0000000000000",
        "7FF8000000000000",
    ],
)
def test_native_value_payload_rejects_invalid_range_bits(range_bits: object) -> None:
    assert not semantic._valid_native_value_payload(
        {
            "match_count": 1,
            "runtime_id_before": "42,29",
            "runtime_id_after": "42,29",
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "range_read",
            "text": None,
            "char_count": 0,
            "truncated": False,
            "range_bits": range_bits,
            "tree_truncated": False,
        },
        runtime_id="42,29",
        limit=100,
    )


def test_native_value_payload_rejects_range_bits_on_text_status() -> None:
    assert not semantic._valid_native_value_payload(
        {
            "match_count": 1,
            "runtime_id_before": "42,29",
            "runtime_id_after": "42,29",
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "read",
            "text": "12.5",
            "char_count": 4,
            "truncated": False,
            "range_bits": "4029000000000000",
            "tree_truncated": False,
        },
        runtime_id="42,29",
        limit=100,
    )


@pytest.mark.parametrize(
    ("range_bits", "expected_text"),
    [
        ("4029000000000000", "12.5"),
        ("FFEFFFFFFFFFFFFF", "-1.7976931348623157E+308"),
        ("8000000000000001", "-5E-324"),
        ("8000000000000000", "0"),
    ],
)
def test_native_value_bits_produce_finite_canonical_text(
    range_bits: str,
    expected_text: str,
) -> None:
    assert semantic._valid_native_value_payload(
        {
            "match_count": 1,
            "runtime_id_before": "42,30",
            "runtime_id_after": "42,30",
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "range_read",
            "text": None,
            "char_count": 0,
            "truncated": False,
            "range_bits": range_bits,
            "tree_truncated": False,
        },
        runtime_id="42,30",
        limit=100,
    )
    assert semantic._canonical_invariant_double(
        semantic._range_value_from_bits(range_bits)
    ) == expected_text
    assert len(expected_text) <= semantic._MAX_INVARIANT_DOUBLE_CHARS


@pytest.mark.parametrize("value_status", [[], {}])
def test_native_value_payload_rejects_non_string_status(value_status: object) -> None:
    assert not semantic._valid_native_value_payload(
        {
            "match_count": 1,
            "runtime_id_before": "42,27",
            "runtime_id_after": "42,27",
            "password_before": False,
            "password_after": False,
            "value_supported": False,
            "value_status": value_status,
            "text": None,
            "char_count": 0,
            "truncated": False,
            "tree_truncated": False,
        },
        runtime_id="42,27",
        limit=100,
    )


@pytest.mark.parametrize(
    ("scrollable", "percent", "expected"),
    [
        (False, -1, True),
        (True, 0, True),
        (True, 50.5, True),
        (True, 100, True),
        (False, 50, False),
        (False, -0.5, False),
        (True, -1, False),
        (True, float("nan"), False),
    ],
)
def test_native_scroll_metadata_requires_consistent_axis_state(
    scrollable: bool,
    percent: float,
    expected: bool,
) -> None:
    assert (
        semantic._valid_native_scroll(
            {
                "horizontal_scrollable": scrollable,
                "vertical_scrollable": False,
                "horizontal_percent": percent,
                "vertical_percent": -1,
            }
        )
        is expected
    )


@pytest.mark.parametrize("text", ["\ud800", "\udc00", "ok\ud800"])
def test_native_value_payload_rejects_unpaired_surrogates(text: str) -> None:
    assert not semantic._valid_native_value_payload(
        {
            "match_count": 1,
            "runtime_id_before": "42,14",
            "runtime_id_after": "42,14",
            "password_before": False,
            "password_after": False,
            "value_supported": True,
            "value_status": "read",
            "text": text,
            "char_count": semantic._utf16_code_units(text),
            "truncated": False,
            "tree_truncated": False,
        },
        runtime_id="42,14",
        limit=100,
    )


@pytest.mark.parametrize("field", ["runtime_id_before", "runtime_id_after"])
def test_native_value_payload_rejects_unpaired_runtime_ids(field: str) -> None:
    payload = {
        "match_count": 1,
        "runtime_id_before": "42,16",
        "runtime_id_after": "42,16",
        "password_before": False,
        "password_after": False,
        "value_supported": False,
        "value_status": "discarded_unsafe",
        "text": None,
        "char_count": 0,
        "truncated": False,
        "tree_truncated": False,
    }
    payload[field] = "42,\ud800"

    assert not semantic._valid_native_value_payload(
        payload,
        runtime_id="42,16",
        limit=100,
    )


@pytest.mark.parametrize("method", ["invoke", "set_value", "scroll"])
def test_native_diagnostic_locator_rejects_semantic_actions(method: str) -> None:
    adapter = WinAppAdapter(FakeBackend())
    kwargs: dict[str, Any] = {"selector": "nuia-" + "c" * 24}
    if method == "set_value":
        kwargs["value"] = "fixture"
    if method == "scroll":
        kwargs.update(direction="down", to=None, percent=None)

    with pytest.raises(ComputerError) as raised:
        getattr(adapter, method)(_identity(), **kwargs)

    assert raised.value.code == "semantic_provider_action_unsupported"


@pytest.mark.parametrize("method", ["invoke", "set_value", "scroll"])
def test_malformed_native_prefix_remains_invalid_element(method: str) -> None:
    adapter = WinAppAdapter(FakeBackend())
    kwargs: dict[str, Any] = {"selector": "nuia-bad"}
    if method == "set_value":
        kwargs["value"] = "fixture"
    if method == "scroll":
        kwargs.update(direction="down", to=None, percent=None)

    with pytest.raises(ComputerError) as raised:
        getattr(adapter, method)(_identity(), **kwargs)

    assert raised.value.code == "invalid_element"


def test_invalid_scroll_request_precedes_native_action_rejection() -> None:
    adapter = WinAppAdapter(FakeBackend())

    with pytest.raises(ComputerError) as raised:
        adapter.scroll(
            _identity(),
            selector="nuia-" + "a" * 24,
            direction=None,
            to=None,
            percent=None,
        )

    assert raised.value.code == "invalid_scroll_request"


def test_malformed_native_prefix_is_rejected_before_read_provider_routing() -> None:
    with pytest.raises(ComputerError) as raised:
        semantic.read_semantic_element(
            hwnd=123,
            element="nuia-bad",
            max_chars=100,
            backend=FakeBackend(),
            winapp_adapter=FakeWinApp([]),
        )

    assert raised.value.code == "invalid_element"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows owner-private store")
def test_read_reacquires_safe_locator_and_issues_short_lived_reference(
    tmp_path: Path,
) -> None:
    element = {
        "element": "edit-fixture-a1b2",
        "automation_id": "TaskValue",
        "control_type": "Edit",
        "class": "TextBox",
        "name": "Fixture input",
        "bounds": {"x": 20, "y": 40, "width": 200, "height": 30},
        "enabled": True,
        "offscreen": False,
        "actionable": True,
        "depth": 0,
        "parent": None,
    }

    class ReadWinApp(FakeWinApp):
        def read(self, _identity: WindowIdentity, **_kwargs):
            return _binding(), {
                "element": "edit-fixture-a1b2",
                "resolved_element": "provider-element-old",
                "text": "fixture value",
                "char_count": 13,
                "truncated": False,
                "redacted": False,
                "redaction_reason": None,
                "_element_reference_identity": {
                    "automation_id": "TaskValue",
                    "runtime_id": "42,old",
                    "control_type": "Edit",
                    "class": "TextBox",
                    "ancestor_signature": "sha256:" + "a" * 24,
                    "enabled": True,
                    "read_only": False,
                    "password": False,
                },
            }

        def _element_properties(self, _identity, _element):
            return _binding(), "provider-element-new", {
                "automation_id": "TaskValue",
                "runtime_id": "42,new",
                "control_type": "edit",
                "class": "TextBox",
                "ancestor_signature": "sha256:" + "a" * 24,
                "enabled": True,
                "value_read_only": False,
                "range_value_read_only": None,
                "password": False,
            }

    result = semantic.read_semantic_element(
        hwnd=123,
        element="edit-fixture-a1b2",
        max_chars=100,
        backend=FakeBackend(),
        winapp_adapter=ReadWinApp([element]),
        element_ref_store=ElementReferenceStore(tmp_path / "refs"),
    )

    assert result["element_ref_status"] == "available"
    assert result["element_ref"].startswith("eref_")
    assert result["expires_at"].endswith("Z")
    assert result["observation_tier"] == "uia_accessibility_provider"
    output = render_read({"reading": result})
    assert "Tier: uia_accessibility_provider" in output
    assert f"ref={result['element_ref']}" in output


@pytest.mark.skipif(sys.platform != "win32", reason="Windows owner-private store")
def test_read_does_not_issue_reference_when_selector_now_names_replacement_control(
    tmp_path: Path,
) -> None:
    element = {
        "element": "edit-fixture-a1b2",
        "automation_id": "ReplacementValue",
        "control_type": "Edit",
        "class": "ReplacementTextBox",
        "name": "Replacement input",
        "bounds": {"x": 20, "y": 40, "width": 200, "height": 30},
        "enabled": True,
        "offscreen": False,
        "actionable": True,
        "depth": 0,
        "parent": None,
    }

    class ReplacedReadWinApp(FakeWinApp):
        def read(self, _identity: WindowIdentity, **_kwargs):
            return _binding(), {
                "element": "edit-fixture-a1b2",
                "resolved_element": "provider-element-old",
                "text": "original value",
                "char_count": 14,
                "truncated": False,
                "redacted": False,
                "redaction_reason": None,
                "_element_reference_identity": {
                    "automation_id": "TaskValue",
                    "runtime_id": "42,old",
                    "control_type": "Edit",
                    "class": "TextBox",
                    "ancestor_signature": "sha256:" + "a" * 24,
                    "enabled": True,
                    "read_only": False,
                    "password": False,
                },
            }

        def _element_properties(self, _identity, _element):
            return _binding(), "provider-element-new", {
                "automation_id": "ReplacementValue",
                "runtime_id": "42,new",
                "control_type": "edit",
                "class": "ReplacementTextBox",
                "ancestor_signature": "sha256:" + "b" * 24,
                "enabled": True,
                "value_read_only": False,
                "range_value_read_only": None,
                "password": False,
            }

    store_root = tmp_path / "refs"
    result = semantic.read_semantic_element(
        hwnd=123,
        element="edit-fixture-a1b2",
        max_chars=100,
        backend=FakeBackend(),
        winapp_adapter=ReplacedReadWinApp([element]),
        element_ref_store=ElementReferenceStore(store_root),
    )

    assert result["text"] == "original value"
    assert result["element_ref"] is None
    assert result["element_ref_status"] == "unavailable"
    assert result["element_ref_reason"] == "post_read_stable_identity_changed"
    assert not store_root.exists()


def test_native_action_unsupported_reports_phase_a_fallback() -> None:
    adapter = WinAppAdapter(FakeBackend())

    with pytest.raises(ComputerError) as raised:
        adapter.invoke(_identity(), selector="nuia-" + "c" * 24)

    assert raised.value.details["fallback"] == {
        "kind": "guarded_physical_input",
        "status": "available",
        "reason": "requires_fresh_capture_and_explicit_allow_physical",
    }


def test_native_document_hierarchy_detects_an_empty_fragment() -> None:
    elements = semantic._normalize_native_elements(
        [
            _native_element("1", name="Browser"),
            _native_element("2", parent="1", control_type="Document", depth=1),
        ],
        provider="native_uia_raw",
    )

    diagnostic = semantic._shallow_diagnostic(
        provider="native_uia_raw",
        inspection={"elements": elements},
        requested_depth=12,
    )

    assert diagnostic is not None
    assert diagnostic["reason"] == "empty_document_fragment"


def test_whitespace_and_control_only_text_do_not_make_a_semantic_surface_useful() -> None:
    document = _native_element("41", control_type="Document", name=" \t")
    child = _native_element("42", parent="41", name="\n\x00", depth=1)
    elements = semantic._normalize_native_elements(
        [document, child],
        provider="native_uia_raw",
    )

    assert semantic._surface_stats(elements)["text_bearing_count"] == 0
    assert semantic._has_empty_document_fragment(elements) is True
    assert semantic._has_meaningful_text(" Ready ") is True


def test_single_named_container_is_shallow() -> None:
    diagnostic = semantic._shallow_diagnostic(
        provider="winapp",
        inspection={
            "elements": [
                {
                    "element": "window-a1",
                    "control_type": "Window",
                    "name": "Fixture",
                    "depth": 0,
                    "actionable": False,
                }
            ]
        },
        requested_depth=12,
    )

    assert diagnostic is not None
    assert diagnostic["reason"] == "container_only_surface"


def test_winapp_interactive_filter_requires_safe_action_metadata() -> None:
    normalized = semantic._normalize_winapp_elements(
        [
            {
                "element": "button-good",
                "automation_id": "SaveButton",
                "control_type": "Button",
                "class": "Button",
                "actionable": True,
                "enabled": True,
                "offscreen": False,
                "depth": 1,
            },
            {
                "element": "button-disabled",
                "control_type": "Button",
                "actionable": True,
                "enabled": False,
                "offscreen": False,
                "depth": 1,
            },
            {
                "element": "button-offscreen",
                "control_type": "Button",
                "actionable": True,
                "enabled": True,
                "offscreen": True,
                "depth": 1,
            },
            {
                "element": "button-unknown",
                "control_type": "Button",
                "actionable": True,
                "enabled": True,
                "offscreen": None,
                "depth": 1,
            },
            {
                "element": None,
                "control_type": "Button",
                "actionable": True,
                "enabled": True,
                "offscreen": False,
                "depth": 1,
            },
        ]
    )

    filtered = semantic._filter_winapp_inspection(
        {"count": len(normalized), "elements": normalized},
        interactive=True,
    )

    assert [item["element"] for item in filtered["elements"]] == ["button-good"]
    assert [item["action_supported"] for item in normalized] == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_winapp_reference_is_unavailable_when_returned_ancestry_is_incomplete() -> None:
    normalized = semantic._normalize_winapp_elements(
        [
            {
                "element": "button-child",
                "parent": "missing-parent",
                "automation_id": "TaskButton",
                "control_type": "Button",
                "class": "Button",
                "actionable": True,
                "enabled": True,
                "offscreen": False,
                "depth": 2,
            }
        ]
    )

    assert normalized[0]["locator"] is None
    assert normalized[0]["action_supported"] is False
    assert normalized[0]["hierarchy"]["ancestor_signature"] is None


def test_winapp_interactive_flattening_preserves_path_but_withholds_unsafe_reference(
    tmp_path: Path,
) -> None:
    raw, traversal_truncated = uia_winapp._flatten_elements(
        [
            {
                "selector": "TaskValue",
                "automationId": "TaskValue",
                "type": "Edit",
                "className": "TextBox",
                "x": 20,
                "y": 40,
                "width": 200,
                "height": 30,
                "isEnabled": True,
                "isOffscreen": False,
                "ancestorPath": ["Window", "Pane"],
            }
        ],
        max_elements=10,
    )

    assert traversal_truncated is False
    assert raw[0]["depth"] == 2
    assert raw[0]["parent"] is None
    assert raw[0]["_ancestor_signature"] is None
    assert raw[0]["_ancestor_identity_exact"] is False
    normalized = semantic._normalize_winapp_elements(raw)
    assert normalized[0]["hierarchy"] == {
        "depth": 2,
        "parent": None,
        "ancestor_signature": None,
        "provider_ancestor_path": ["Window", "Pane"],
        "identity_complete": False,
    }
    assert normalized[0]["locator"] is None
    assert normalized[0]["action_supported"] is False
    assert normalized[0]["element_ref_reason"] == "provider_filtered_hierarchy_inexact"

    filtered = semantic._filter_winapp_inspection(
        {
            "provider_filter_applied": True,
            "count": len(normalized),
            "elements": normalized,
        },
        interactive=True,
    )
    assert filtered["count"] == 1
    assert filtered["elements"][0]["action_supported"] is False

    identity = WindowIdentity(
        hwnd=123,
        pid=456,
        thread_id=789,
        process="fixture.exe",
        title="Fixture",
        class_name="FixtureWindow",
        bounds=Bounds(0, 0, 400, 300),
        process_started=999,
        session_id=1,
        desktop_name="Default",
    )
    attached = attach_element_references(
        {"elements": normalized},
        identity=identity,
        store=ElementReferenceStore(tmp_path / "refs"),
    )

    assert attached["elements"][0]["element_ref"] is None
    assert attached["elements"][0]["element_ref_status"] == "unavailable"
    assert (
        attached["elements"][0]["element_ref_reason"]
        == "provider_filtered_hierarchy_inexact"
    )
    assert not (tmp_path / "refs").exists()

    missing_path, _truncated = uia_winapp._flatten_elements(
        [
            {
                "selector": "TaskValue",
                "automationId": "TaskValue",
                "type": "Edit",
                "className": "TextBox",
                "x": 20,
                "y": 40,
                "width": 200,
                "height": 30,
                "isEnabled": True,
                "isOffscreen": False,
            }
        ],
        max_elements=10,
        provider_filtered_hierarchy=True,
    )
    assert missing_path[0]["depth"] is None
    assert missing_path[0]["_ancestor_signature"] is None
    assert missing_path[0]["_ancestor_identity_exact"] is False


def test_duplicate_winapp_selectors_cannot_receive_reconstructed_references(
    tmp_path: Path,
) -> None:
    ancestor = "sha256:" + "a" * 24
    raw = [
        {
            "element": "duplicate-selector",
            "automation_id": automation_id,
            "control_type": "Button",
            "class": "Button",
            "actionable": True,
            "enabled": True,
            "offscreen": False,
            "bounds": {"x": index * 100, "y": 0, "width": 80, "height": 24},
            "_ancestor_signature": ancestor,
        }
        for index, automation_id in enumerate(("FirstButton", "SecondButton"))
    ]
    normalized = semantic._normalize_winapp_elements(raw)
    identity = WindowIdentity(
        hwnd=123,
        pid=456,
        thread_id=789,
        process="fixture.exe",
        title="Fixture",
        class_name="FixtureWindow",
        bounds=Bounds(0, 0, 400, 300),
        process_started=999,
        session_id=1,
        desktop_name="Default",
    )

    attached = attach_element_references(
        {"elements": normalized},
        identity=identity,
        store=ElementReferenceStore(tmp_path / "refs"),
    )

    assert [item["locator"] for item in normalized] == [None, None]
    assert [item["action_supported"] for item in normalized] == [False, False]
    assert [item["element_ref_status"] for item in attached["elements"]] == [
        "unavailable",
        "unavailable",
    ]
    assert not (tmp_path / "refs").exists()


def test_winapp_overlong_identity_cannot_issue_reference_for_shortened_replacement(
    tmp_path: Path,
) -> None:
    raw, traversal_truncated = uia_winapp._flatten_elements(
        [
            {
                "selector": "parent",
                "automationId": "A" * 161,
                "type": "Pane",
                "className": "Panel",
                "x": 0,
                "y": 0,
                "width": 400,
                "height": 300,
                "isEnabled": True,
                "isOffscreen": False,
                "children": [
                    {
                        "selector": "child",
                        "automationId": "ChildButton",
                        "type": "Button",
                        "className": "Button",
                        "x": 10,
                        "y": 10,
                        "width": 80,
                        "height": 24,
                        "isInvokable": True,
                        "isEnabled": True,
                        "isOffscreen": False,
                    }
                ],
            }
        ],
        max_elements=10,
    )
    assert traversal_truncated is False
    normalized = semantic._normalize_winapp_elements(raw)

    assert normalized[0]["automation_id"] is None
    assert normalized[0]["locator"] is None
    assert normalized[0]["action_supported"] is False
    assert normalized[1]["locator"] is None
    assert normalized[1]["action_supported"] is False
    identity = WindowIdentity(
        hwnd=123,
        pid=456,
        thread_id=789,
        process="fixture.exe",
        title="Fixture",
        class_name="FixtureWindow",
        bounds=Bounds(0, 0, 400, 300),
        process_started=999,
        session_id=1,
        desktop_name="Default",
    )
    attached = attach_element_references(
        {"elements": normalized},
        identity=identity,
        store=ElementReferenceStore(tmp_path / "refs"),
    )

    assert [item["element_ref"] for item in attached["elements"]] == [None, None]
    assert [item["element_ref_status"] for item in attached["elements"]] == [
        "unavailable",
        "unavailable",
    ]
    assert "A" * 157 + "..." not in str(attached)
    assert not (tmp_path / "refs").exists()


def test_human_inspect_renders_read_only_action_and_separate_fallbacks() -> None:
    output = render_inspect(
        {
            "inspection": {
                "window": {"hwnd": 123},
                "backend": {
                    "name": "native_uia_raw",
                    "version": "fixture",
                    "actions_exposed": False,
                },
                "count": 1,
                "elements": [
                    {
                        "element": "nuia-" + "a" * 24,
                        "control_type": "Button",
                        "name": "Save",
                        "bounds": None,
                        "actionable": True,
                        "action_supported": False,
                    }
                ],
                "semantic_action_unavailable": {
                    "provider": "native_uia_raw",
                    "reason": "provider_read_only",
                },
                "semantic_surface_inconclusive": {
                    "provider": "native_uia_raw",
                    "reason": "truncated_before_surface_classification",
                    "candidate_reason": "empty_document_fragment",
                    "element_count": 1,
                },
                "fallbacks": {
                    "observation": {
                        "kind": "targeted_screenshot",
                        "status": "conditional",
                        "when": "visual_only",
                    },
                    "action": {
                        "kind": "guarded_physical_input",
                        "status": "conditional",
                        "when": "no_semantic_path",
                    },
                },
            }
        }
    )

    assert "semantic-read-only action-unsupported" in output
    assert "Semantic actions unavailable: provider_read_only" in output
    assert "Inconclusive semantic surface: truncated_before_surface_classification" in output
    assert "Observation fallback: targeted_screenshot" in output
    assert "Action fallback: guarded_physical_input" in output


def test_human_inspect_renders_effective_winapp_action_limitations() -> None:
    output = render_inspect(
        {
            "inspection": {
                "window": {"hwnd": 123},
                "backend": {
                    "name": "winapp",
                    "version": "fixture",
                    "actions_exposed": True,
                },
                "count": 5,
                "elements": [
                    {
                        "element": "button-good",
                        "control_type": "Button",
                        "name": "Save",
                        "actionable": True,
                        "enabled": True,
                        "offscreen": False,
                        "locator": {"strategy": "selector_v1"},
                    },
                    {
                        "element": "button-disabled",
                        "control_type": "Button",
                        "name": "Delete",
                        "actionable": True,
                        "enabled": False,
                        "offscreen": False,
                        "locator": {"strategy": "selector_v1"},
                    },
                    {
                        "element": "button-offscreen",
                        "control_type": "Button",
                        "name": "Hidden",
                        "actionable": True,
                        "enabled": True,
                        "offscreen": True,
                        "locator": {"strategy": "selector_v1"},
                    },
                    {
                        "element": "button-unknown",
                        "control_type": "Button",
                        "name": "Unknown",
                        "actionable": True,
                        "enabled": None,
                        "offscreen": None,
                        "locator": {"strategy": "selector_v1"},
                    },
                    {
                        "element": "",
                        "control_type": "Button",
                        "name": "Lost",
                        "actionable": True,
                        "enabled": True,
                        "offscreen": False,
                        "locator": None,
                    },
                ],
            }
        }
    )

    assert "button-good Button Save - [actionable]" in output
    assert "button-disabled Button Delete - [disabled]" in output
    assert "button-offscreen Button Hidden - [offscreen]" in output
    assert (
        "button-unknown Button Unknown - [enabled-unknown visibility-unknown]"
        in output
    )
    assert "- Button Lost - [unaddressable]" in output
    assert "semantic-read-only" not in output
