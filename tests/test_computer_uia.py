from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pytest

from agent_tools import cli as root_cli
from agent_tools.computer import cli as computer_cli
from agent_tools.computer import process_io, semantic_accessibility, uia_winapp
from agent_tools.computer.models import Bounds, ComputerError, WindowIdentity
from agent_tools.computer.uia_winapp import (
    MAX_INSPECT_ELEMENTS,
    MAX_READ_CHARS,
    MAX_SCROLL_AREAS,
    ProcessResult,
    WinAppAdapter,
    WinAppBinding,
)


def _identity() -> WindowIdentity:
    return WindowIdentity(
        hwnd=123,
        pid=456,
        thread_id=789,
        process="fixture.exe",
        title="AgentTools fixture",
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


def _trusted_winapp_read_properties(
    *,
    runtime_id: str = "42,7",
    password: bool = False,
) -> dict[str, object]:
    return {
        "password": password,
        "enabled": True,
        "runtime_id": runtime_id,
        "value_available": False,
        "range_value_available": False,
        "legacy_value_available": False,
        "value_pattern_status": "absent",
        "range_value_pattern_status": "absent",
    }


class FakeBackend:
    def __init__(self) -> None:
        self.identity = _identity()
        self.captures = 0
        self.fail_on_capture: int | None = None
        self.next_identity: WindowIdentity | None = None

    def capture_identity(self, _hwnd: int) -> WindowIdentity:
        self.captures += 1
        if self.fail_on_capture == self.captures:
            raise ComputerError("stale_window", "fixture identity drift")
        return self.next_identity or self.identity

    def revalidate_identity(self, expected: WindowIdentity) -> WindowIdentity:
        try:
            actual = self.capture_identity(expected.hwnd)
        except ComputerError as exc:
            raise ComputerError(
                "stale_window", "The fixture window changed or closed."
            ) from exc
        if (
            actual.hwnd,
            actual.pid,
            actual.thread_id,
            actual.process,
            actual.process_started,
            actual.title,
            actual.class_name,
        ) != (
            expected.hwnd,
            expected.pid,
            expected.thread_id,
            expected.process,
            expected.process_started,
            expected.title,
            expected.class_name,
        ):
            raise ComputerError("stale_window", "The fixture window changed or closed.")
        return actual

    def revalidate(self, expected: WindowIdentity) -> WindowIdentity:
        actual = self.capture_identity(expected.hwnd)
        if (
            actual.hwnd,
            actual.pid,
            actual.process_started,
            actual.class_name,
        ) != (
            expected.hwnd,
            expected.pid,
            expected.process_started,
            expected.class_name,
        ):
            raise ComputerError("stale_window", "The fixture window changed or closed.")
        return actual


def test_discovery_prefers_explicit_environment_path(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "explicit" / "winapp.exe"
    explicit.parent.mkdir()
    explicit.write_bytes(b"fixture")
    monkeypatch.setenv(uia_winapp.WINAPP_ENV, str(explicit))

    candidate, source = uia_winapp._discover_candidate()

    assert candidate == explicit.resolve()
    assert source == "environment"


def test_probe_reports_version_mismatch_without_path_leak(monkeypatch, tmp_path: Path) -> None:
    candidate = tmp_path / "winapp.exe"
    candidate.write_bytes(b"fixture")
    monkeypatch.setattr(uia_winapp.sys, "platform", "win32")
    monkeypatch.setenv(uia_winapp.WINAPP_ENV, str(candidate))
    monkeypatch.setattr(uia_winapp, "_sha256", lambda _path: uia_winapp.WINAPP_EXE_SHA256)
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(0, b"0.5.0\n", b""),
    )

    probe = WinAppAdapter(FakeBackend()).probe()

    assert probe["available"] is False
    assert probe["error_code"] == "backend_version_mismatch"
    assert str(candidate) not in json.dumps(probe)


def test_probe_reports_operational_failure_as_error(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())

    def fail(**_kwargs) -> WinAppBinding:
        raise ComputerError("uia_timeout", "fixture timeout")

    monkeypatch.setattr(adapter, "require", fail)

    probe = adapter.probe()

    assert probe["status"] == "error"
    assert probe["error_code"] == "uia_timeout"


def test_probe_never_triggers_first_use_backend_install(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    observed: dict[str, bool] = {}

    def missing(*, auto_install: bool = True, **_kwargs) -> WinAppBinding:
        observed["auto_install"] = auto_install
        raise ComputerError("capability_missing", "fixture backend is absent")

    monkeypatch.setattr(adapter, "require", missing)

    probe = adapter.probe()

    assert probe["status"] == "missing"
    assert observed == {"auto_install": False}


def test_require_installs_pinned_backend_on_first_semantic_use(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "winapp.exe"
    candidate.write_bytes(b"fixture executable")
    candidate.with_name(uia_winapp.WINAPP_COMPANION).write_bytes(b"fixture companion")
    installs: list[bool] = []
    monkeypatch.setattr(uia_winapp.sys, "platform", "win32")
    monkeypatch.setattr(uia_winapp, "_discover_candidate", lambda: (None, "missing"))
    monkeypatch.setattr(
        uia_winapp,
        "_install_pinned_winapp",
        lambda: installs.append(True) or candidate,
    )
    monkeypatch.setattr(
        uia_winapp,
        "_sha256",
        lambda path, **_kwargs: (
            uia_winapp.WINAPP_COMPANION_SHA256
            if Path(path).name == uia_winapp.WINAPP_COMPANION
            else uia_winapp.WINAPP_EXE_SHA256
        ),
    )
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(0, b"0.6.0\n", b""),
    )

    binding = WinAppAdapter(FakeBackend()).require()

    assert installs == [True]
    assert binding.source == "managed_cache"
    assert binding.path == candidate


def test_pinned_backend_extraction_ignores_unneeded_debug_symbols(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "winapp.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("winapp.exe", b"fixture executable")
        bundle.writestr(uia_winapp.WINAPP_COMPANION, b"fixture companion")
        bundle.writestr("winapp.pdb", b"large debug symbols are not installed")
    executable = tmp_path / "staged.exe"
    companion = tmp_path / "staged.dll"

    uia_winapp._extract_winapp_runtime(
        archive,
        executable=executable,
        companion=companion,
    )

    assert executable.read_bytes() == b"fixture executable"
    assert companion.read_bytes() == b"fixture companion"
    assert not (tmp_path / "winapp.pdb").exists()


def test_screenshot_uses_exact_window_without_focus_and_validates_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    monkeypatch.setattr(adapter, "require", _binding)
    output = tmp_path / "capture.png"
    observed: list[str] = []

    def run(_path, arguments, **_kwargs):
        observed.extend(arguments)
        return ProcessResult(
            0,
            json.dumps(
                {
                    "filePath": str(output.resolve()),
                    "width": 800,
                    "height": 600,
                    "processId": 456,
                    "windowTitle": "ignored output title",
                    "hwnd": 123,
                }
            ).encode(),
            b"",
        )

    monkeypatch.setattr(uia_winapp, "_run_process", run)

    binding, metadata = adapter.screenshot(_identity(), output=output)

    assert binding == _binding()
    assert metadata["effective_capture_api"] == "unreported_by_winapp_0.6.0"
    assert observed == [
        "ui",
        "screenshot",
        "--output",
        str(output),
        "--window",
        "123",
        "--json",
    ]
    assert "--focus" not in observed
    assert "--capture-screen" not in observed


def test_screenshot_revalidation_allows_mutable_title(monkeypatch, tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.next_identity = WindowIdentity(
        **{
            **_identity().__dict__,
            "title": "AgentTools fixture - animated frame",
        }
    )
    adapter = WinAppAdapter(backend)
    monkeypatch.setattr(adapter, "require", _binding)
    output = tmp_path / "animated-title.png"
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(
            0,
            json.dumps(
                {
                    "filePath": str(output.resolve()),
                    "width": 800,
                    "height": 600,
                    "processId": 456,
                    "hwnd": 123,
                }
            ).encode(),
            b"",
        ),
    )

    adapter.screenshot(_identity(), output=output)

    assert backend.captures == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hwnd", 124),
        ("processId", 457),
        ("width", "800"),
        ("height", 0),
        ("filePath", "different.png"),
    ],
)
def test_screenshot_rejects_untrusted_target_or_file_metadata(
    monkeypatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    monkeypatch.setattr(adapter, "require", _binding)
    output = tmp_path / "capture.png"
    payload = {
        "filePath": str(output.resolve()),
        "width": 800,
        "height": 600,
        "processId": 456,
        "hwnd": 123,
    }
    payload[field] = value
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(
            0, json.dumps(payload).encode(), b""
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.screenshot(_identity(), output=output)

    assert raised.value.code == "uia_invalid_response"


def test_probe_reports_nonzero_version_process_as_operational_error(
    monkeypatch, tmp_path: Path
) -> None:
    candidate = tmp_path / "winapp.exe"
    candidate.write_bytes(b"fixture")
    candidate.with_name(uia_winapp.WINAPP_COMPANION).write_bytes(b"companion")
    monkeypatch.setattr(uia_winapp.sys, "platform", "win32")
    monkeypatch.setenv(uia_winapp.WINAPP_ENV, str(candidate))
    monkeypatch.setattr(
        uia_winapp,
        "_sha256",
        lambda path, **_kwargs: (
            uia_winapp.WINAPP_COMPANION_SHA256
            if Path(path).name == uia_winapp.WINAPP_COMPANION
            else uia_winapp.WINAPP_EXE_SHA256
        ),
    )
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(1, b"", b"loader failure"),
    )

    probe = WinAppAdapter(FakeBackend()).probe()

    assert probe["status"] == "error"
    assert probe["error_code"] == "uia_version_probe_failed"
    assert str(candidate) not in json.dumps(probe)


def test_process_start_failure_is_operational_not_missing(monkeypatch) -> None:
    def fail_start(*_args, **_kwargs):
        raise OSError("access denied")

    monkeypatch.setattr(uia_winapp.subprocess, "Popen", fail_start)

    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path("winapp.exe"),
            ["--version"],
            timeout_seconds=0.5,
            max_output_bytes=1_024,
        )

    assert raised.value.code == "uia_backend_start_failed"
    assert raised.value.details == {
        "process_started": False,
        "lifecycle_stage": "pre_spawn",
    }
    assert "access denied" not in raised.value.message


def test_trusted_powershell_ignores_fake_systemroot(monkeypatch, tmp_path: Path) -> None:
    fake = tmp_path / "fake-windows"
    fake_powershell = (
        fake / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    fake_powershell.parent.mkdir(parents=True)
    fake_powershell.write_bytes(b"not a trusted executable")
    monkeypatch.setenv("SystemRoot", str(fake))
    monkeypatch.setenv("WINDIR", str(fake))

    trusted = uia_winapp._trusted_powershell()

    assert trusted is not None
    assert trusted != fake_powershell.resolve()
    assert "System32" in trusted.parts


def test_native_identity_preserves_ambiguous_element_error(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())

    def ambiguous(*_args, **_kwargs):
        raise ComputerError("ambiguous_element", "fixture duplicate AutomationId")

    monkeypatch.setattr(adapter, "_native_uia_command", ambiguous)

    with pytest.raises(ComputerError) as raised:
        adapter._native_element_state(
            _identity(),
            properties={"AutomationId": "DuplicatedFixture"},
        )

    assert raised.value.code == "ambiguous_element"


def test_scroll_areas_binds_provider_candidate_before_native_identity(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    responses = iter(
        [
            (
                _binding(),
                {
                    "matchCount": 3,
                    "hasMore": False,
                    "matches": [
                        {
                            "selector": "hidden-scrollbar",
                            "automationId": "VerticalScrollBar",
                            "type": "ScrollBar",
                            "className": "ScrollBar",
                            "isOffscreen": True,
                            "x": 0,
                            "y": 0,
                            "width": 0,
                            "height": 0,
                        },
                        {
                            "selector": "TaskScroll",
                            "automationId": "TaskScroll",
                            "type": "Pane",
                            "className": "ScrollViewer",
                            "x": 60,
                            "y": 241,
                            "width": 684,
                            "height": 150,
                        },
                        {
                            "selector": "visible-scrollbar",
                            "automationId": "VerticalScrollBar",
                            "type": "ScrollBar",
                            "className": "ScrollBar",
                            "isOffscreen": False,
                            "x": 727,
                            "y": 241,
                            "width": 17,
                            "height": 150,
                        },
                    ],
                },
            ),
            (
                _binding(),
                {
                    "properties": {
                        "AutomationId": "TaskScroll",
                        "ControlType": "Pane",
                        "ClassName": "ScrollViewer",
                        "HorizontallyScrollable": "False",
                        "VerticallyScrollable": "True",
                        "ScrollVerticalPercent": "25",
                    }
                },
            ),
            (
                _binding(),
                {
                    "properties": {
                        "AutomationId": "VerticalScrollBar",
                        "ControlType": "ScrollBar",
                        "ClassName": "ScrollBar",
                        "IsOffscreen": "False",
                    }
                },
            ),
        ]
    )
    monkeypatch.setattr(adapter, "_target_json", lambda *_args, **_kwargs: next(responses))
    captured: dict[str, object] = {}
    native_calls = 0

    def native(_identity, *, request, **_kwargs):
        nonlocal native_calls
        native_calls += 1
        captured.update(request)
        return {
            "count": 1,
            "automationId": "TaskScroll",
            "runtimeId": "42,scroll",
            "resolutionStage": "provider_identity_unique",
            "isPassword": False,
            "enabled": True,
            "controlType": "pane",
            "className": "ScrollViewer",
            "ancestorSignature": "sha256:" + "a" * 24,
            "patterns": ["uia.ScrollPattern"],
            "rangeBits": None,
            "valuePatternStatus": "absent",
            "rangeValuePatternStatus": "absent",
        }

    monkeypatch.setattr(adapter, "_native_uia_command", native)

    _binding_value, result = adapter.scroll_areas(_identity(), max_elements=20)

    assert captured["automationId"] == "TaskScroll"
    assert captured["providerIdentity"] == {
        "controlType": "Pane",
        "className": "ScrollViewer",
        "bounds": {"x": 60, "y": 241, "width": 684, "height": 150},
    }
    assert native_calls == 1
    assert result["count"] == 1
    assert result["areas"][0]["locator"]["runtime_id"] == "42,scroll"


def test_scroll_areas_preserves_genuine_provider_identity_ambiguity(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    responses = iter(
        [
            (
                _binding(),
                {
                    "matchCount": 1,
                    "hasMore": False,
                    "matches": [
                        {
                            "selector": "TaskScroll",
                            "automationId": "TaskScroll",
                            "type": "Pane",
                            "className": "ScrollViewer",
                            "x": 60,
                            "y": 241,
                            "width": 684,
                            "height": 150,
                        }
                    ],
                },
            ),
            (
                _binding(),
                {
                    "properties": {
                        "AutomationId": "TaskScroll",
                        "ControlType": "Pane",
                        "ClassName": "ScrollViewer",
                        "VerticallyScrollable": "True",
                    }
                },
            ),
        ]
    )
    monkeypatch.setattr(adapter, "_target_json", lambda *_args, **_kwargs: next(responses))

    def ambiguous(*_args, **_kwargs):
        raise ComputerError(
            "ambiguous_element",
            "fixture exact provider identity is duplicated",
            details={
                "resolution_stage": "provider_identity_unique",
                "candidate_count": 2,
            },
        )

    monkeypatch.setattr(adapter, "_native_uia_command", ambiguous)

    with pytest.raises(ComputerError) as raised:
        adapter.scroll_areas(_identity(), max_elements=20)

    assert raised.value.code == "ambiguous_element"
    assert raised.value.details == {
        "resolution_stage": "provider_identity_unique",
        "candidate_count": 2,
    }


def test_native_identity_rejects_unverified_provider_resolution_stage(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    monkeypatch.setattr(
        adapter,
        "_native_uia_command",
        lambda *_args, **_kwargs: {
            "count": 1,
            "automationId": "TaskScroll",
            "runtimeId": "42,scroll",
            "resolutionStage": "automation_id_unique",
            "isPassword": False,
            "enabled": True,
            "patterns": ["uia.ScrollPattern"],
            "rangeBits": None,
            "valuePatternStatus": "absent",
            "rangeValuePatternStatus": "absent",
        },
    )

    with pytest.raises(ComputerError) as raised:
        adapter._native_element_state(
            _identity(),
            properties={"AutomationId": "TaskScroll"},
            provider_identity={
                "controlType": "Pane",
                "className": "ScrollViewer",
                "bounds": {"x": 60, "y": 241, "width": 684, "height": 150},
            },
        )

    assert raised.value.code == "uia_invalid_response"


def test_native_script_never_selects_first_broad_provider_candidate() -> None:
    script = uia_winapp._native_uia_script(123)

    assert "$providerMatches+=$candidate" in script
    assert "$providerMatches.Count -gt 1" in script
    assert "$resolvedElement=$providerMatches[0]" in script
    assert "resolutionStage='provider_identity_unique'" in script


def test_scroll_provider_identity_accepts_classless_exact_bounds() -> None:
    identity = uia_winapp._scroll_provider_identity(
        {
            "type": "Pane",
            "x": 60,
            "y": 241,
            "width": 684,
            "height": 150,
        },
        {"ControlType": "Pane"},
    )

    assert identity == {
        "controlType": "Pane",
        "className": None,
        "bounds": {"x": 60, "y": 241, "width": 684, "height": 150},
    }


def test_geometry_resolution_accepts_only_ancestor_drift() -> None:
    reference = {
        "selector": "button-save",
        "automation_id": "SaveButton",
        "control_type": "Button",
        "class": "Button",
        "ancestor_signature": "sha256:" + "a" * 24,
        "password": False,
        "enabled": True,
    }
    properties = {
        **reference,
        "ancestor_signature": "sha256:" + "b" * 24,
        "resolution_stage": "stable_geometry_unique",
        "password": False,
        "enabled": True,
    }

    uia_winapp._validate_element_reference_properties(
        "button-save",
        properties,
        reference,
    )

    properties["class"] = "DifferentButton"
    with pytest.raises(ComputerError) as raised:
        uia_winapp._validate_element_reference_properties(
            "button-save",
            properties,
            reference,
        )

    assert raised.value.code == "stale_element"
    assert raised.value.details["element_mismatch_fields"] == ["class", "ancestor_signature"]


def test_native_script_geometry_fallback_is_unique_and_exact() -> None:
    script = uia_winapp._native_uia_script(123)

    assert "$geometryCandidates.Count -gt 4096" in script
    assert "[System.Windows.Automation.Condition]::TrueCondition" in script
    assert "function Compatible-Control-Type" in script
    assert "$class -clike 'windowsforms10.edit.*'" in script
    assert "$geometryMatches.Count -gt 1" in script
    assert "$stage='stable_geometry_unique'" in script
    assert "[int][Math]::Round($candidateRect.X)" in script
    assert "[int]$expectedBounds.width" in script
    assert "nativeAutomationId=[string]$element.Current.AutomationId" in script
    assert "nativeControlType=$nativeControlType" in script


def test_geometry_resolution_error_is_reported_without_stage_loss() -> None:
    assert uia_winapp._resolution_details(
        {
            "resolutionStage": "stable_geometry_unique",
            "count": 0,
        }
    ) == {
        "resolution_stage": "stable_geometry_unique",
        "candidate_count": 0,
    }


@pytest.mark.parametrize(
    "code",
    [
        "password_control",
        "uia_invalid_response",
        "uia_timeout",
        "uia_backend_start_failed",
    ],
)
def test_native_identity_propagates_sensitive_and_query_failures(
    monkeypatch,
    code: str,
) -> None:
    adapter = WinAppAdapter(FakeBackend())

    def failed(*_args, **_kwargs):
        raise ComputerError(code, "fixture native failure")

    monkeypatch.setattr(adapter, "_native_uia_command", failed)

    with pytest.raises(ComputerError) as raised:
        adapter._native_element_state(
            _identity(),
            properties={"AutomationId": "TaskFixture"},
        )

    assert raised.value.code == code


def test_native_identity_degrades_only_when_bridge_is_unavailable(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())

    def unavailable(*_args, **_kwargs):
        raise ComputerError("native_uia_unavailable", "fixture bridge unavailable")

    monkeypatch.setattr(adapter, "_native_uia_command", unavailable)

    assert (
        adapter._native_element_state(
            _identity(),
            properties={"AutomationId": "TaskFixture"},
        )
        is None
    )


@pytest.mark.parametrize("automation_id", [None, "", "x" * 161])
def test_native_identity_rejects_missing_or_unbounded_automation_id(
    automation_id: object,
) -> None:
    adapter = WinAppAdapter(FakeBackend())

    with pytest.raises(ComputerError) as raised:
        adapter._native_element_state(
            _identity(),
            properties={"AutomationId": automation_id},
        )

    assert raised.value.code == "uia_identity_unavailable"


@pytest.mark.parametrize(
    "overrides",
    [
        {"isPassword": True},
        {"patterns": []},
    ],
)
def test_native_identity_rejects_range_bits_with_inconsistent_privacy_or_pattern(
    monkeypatch,
    overrides: dict[str, object],
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    payload: dict[str, object] = {
        "count": 1,
        "automationId": "TaskRange",
        "runtimeId": "42,7",
        "isPassword": False,
        "enabled": True,
        "patterns": ["uia.RangeValuePattern"],
        "rangeBits": "4029000000000000",
        "valuePatternStatus": "absent",
        "rangeValuePatternStatus": "available",
    }
    payload.update(overrides)
    monkeypatch.setattr(adapter, "_native_uia_command", lambda *_args, **_kwargs: payload)

    with pytest.raises(ComputerError) as raised:
        adapter._native_element_state(
            _identity(),
            properties={"AutomationId": "TaskRange"},
        )

    assert raised.value.code == "uia_invalid_response"


@pytest.mark.parametrize(
    ("patterns", "value_status", "range_status"),
    [
        ([], "available", "absent"),
        (["uia.ValuePattern"], "absent", "absent"),
        ([], "absent", "available"),
        (["uia.RangeValuePattern"], "absent", "failed"),
        ([], "unknown", "absent"),
    ],
)
def test_native_identity_rejects_inconsistent_pattern_query_statuses(
    monkeypatch,
    patterns: list[str],
    value_status: str,
    range_status: str,
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    payload: dict[str, object] = {
        "count": 1,
        "automationId": "TaskPattern",
        "runtimeId": "42,7",
        "isPassword": False,
        "enabled": True,
        "patterns": patterns,
        "rangeBits": None,
        "valuePatternStatus": value_status,
        "rangeValuePatternStatus": range_status,
    }
    monkeypatch.setattr(adapter, "_native_uia_command", lambda *_args, **_kwargs: payload)

    with pytest.raises(ComputerError) as raised:
        adapter._native_element_state(
            _identity(),
            properties={"AutomationId": "TaskPattern"},
        )

    assert raised.value.code == "uia_invalid_response"


def test_native_identity_preserves_conclusive_and_failed_pattern_statuses(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    payload: dict[str, object] = {
        "count": 1,
        "automationId": "TaskPattern",
        "runtimeId": "42,7",
        "isPassword": False,
        "enabled": True,
        "patterns": [],
        "rangeBits": None,
        "valuePatternStatus": "failed",
        "rangeValuePatternStatus": "absent",
    }
    monkeypatch.setattr(adapter, "_native_uia_command", lambda *_args, **_kwargs: payload)

    result = adapter._native_element_state(
        _identity(),
        properties={"AutomationId": "TaskPattern"},
    )

    assert result is not None
    assert result["value_pattern_status"] == "failed"
    assert result["range_value_pattern_status"] == "absent"


def test_native_identity_preserves_legacy_protected_state(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    payload: dict[str, object] = {
        "count": 1,
        "automationId": "TaskLegacy",
        "runtimeId": "42,7",
        "isPassword": False,
        "enabled": True,
        "patterns": ["uia.LegacyIAccessible"],
        "rangeBits": None,
        "valuePatternStatus": "absent",
        "rangeValuePatternStatus": "absent",
        "legacyValueReadOnly": False,
        "legacyValueProtected": True,
    }
    monkeypatch.setattr(adapter, "_native_uia_command", lambda *_args, **_kwargs: payload)

    result = adapter._native_element_state(
        _identity(),
        properties={"AutomationId": "TaskLegacy"},
    )

    assert result is not None
    assert result["legacy_value_protected"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"count": True},
        {"count": 1.0},
        {"valuePatternStatus": []},
        {"rangeValuePatternStatus": {}},
        {"legacyValueProtected": True},
    ],
)
def test_native_identity_type_rejects_malformed_count_and_pattern_statuses(
    monkeypatch,
    overrides: dict[str, object],
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    payload: dict[str, object] = {
        "count": 1,
        "automationId": "TaskPattern",
        "runtimeId": "42,7",
        "isPassword": False,
        "enabled": True,
        "patterns": [],
        "rangeBits": None,
        "valuePatternStatus": "absent",
        "rangeValuePatternStatus": "absent",
    }
    payload.update(overrides)
    monkeypatch.setattr(adapter, "_native_uia_command", lambda *_args, **_kwargs: payload)

    with pytest.raises(ComputerError) as raised:
        adapter._native_element_state(
            _identity(),
            properties={"AutomationId": "TaskPattern"},
        )

    assert raised.value.code == "uia_invalid_response"


def test_process_timeout_with_large_preloaded_stdin_is_deadline_bounded() -> None:
    started = time.monotonic()
    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path(sys.executable),
            ["-c", "import time; time.sleep(5)"],
            timeout_seconds=0.1,
            max_output_bytes=1_024,
            input_bytes=b"x" * (1024 * 1024),
        )

    assert raised.value.code == "uia_timeout"
    assert time.monotonic() - started < 1.0


def test_process_rechecks_cleanup_budget_after_input_staging(monkeypatch) -> None:
    clock = {"now": 100.0}
    staged = io.BytesIO()

    def stage(_payload: bytes):
        clock["now"] = 100.1
        return staged

    monkeypatch.setattr(uia_winapp.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(uia_winapp, "preloaded_process_stdin", stage)
    monkeypatch.setattr(
        uia_winapp.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("WinApp must not start"),
    )

    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path("winapp.exe"),
            ["--version"],
            timeout_seconds=1.0,
            max_output_bytes=1_024,
            input_bytes=b"fixture",
            deadline=100.3,
        )

    assert raised.value.code == "uia_timeout"
    assert staged.closed is True


def test_process_pre_spawn_staged_close_failure_is_fatal(monkeypatch) -> None:
    class FaultyClose(io.BytesIO):
        def close(self) -> None:
            raise OSError("fixture staged close failure")

    monkeypatch.setattr(
        uia_winapp,
        "preloaded_process_stdin",
        lambda _payload: FaultyClose(),
    )
    monkeypatch.setattr(
        uia_winapp.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("WinApp must not start"),
    )

    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path("winapp.exe"),
            ["--version"],
            timeout_seconds=0.1,
            max_output_bytes=1_024,
            input_bytes=b"fixture",
        )

    assert raised.value.code == "uia_cleanup_failed"
    assert raised.value.details == {
        "steps": ["staged_stdin:OSError"],
        "primary_error": "uia_timeout",
        "process_started": False,
        "lifecycle_stage": "pre_spawn_cleanup",
    }


@pytest.mark.parametrize("failing_operation", ["write", "seek"])
def test_process_staging_and_internal_close_failure_is_fatal(
    monkeypatch,
    failing_operation: str,
) -> None:
    class StagedOwner:
        def __init__(self) -> None:
            self.file = io.BytesIO()
            self.close_calls = 0

        @property
        def closed(self) -> bool:
            return self.file.closed

        def write(self, payload) -> int:
            if failing_operation == "write":
                raise OSError("fixture stage write failure")
            return self.file.write(payload)

        def seek(self, offset: int) -> int:
            if failing_operation == "seek":
                raise OSError("fixture stage seek failure")
            return self.file.seek(offset)

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("fixture wrapper close failure")

    owner = StagedOwner()
    monkeypatch.setattr(
        process_io.tempfile,
        "TemporaryFile",
        lambda **_kwargs: owner,
    )
    monkeypatch.setattr(
        uia_winapp.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("WinApp must not start"),
    )

    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path("winapp.exe"),
            ["--version"],
            timeout_seconds=1.0,
            max_output_bytes=1_024,
            input_bytes=b"fixture",
        )

    assert raised.value.code == "uia_cleanup_failed"
    assert raised.value.details == {
        "steps": ["staged_stdin:OSError"],
        "secondary_errors": ["staging:OSError"],
        "process_started": False,
        "lifecycle_stage": "pre_spawn",
    }
    assert owner.close_calls == 1
    assert owner.file.closed is True


def test_process_post_spawn_staged_close_failure_is_supervised(
    monkeypatch,
) -> None:
    class FaultyClose(io.BytesIO):
        def close(self) -> None:
            raise OSError("fixture staged close failure")

    class Process:
        stdin = None

        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    process = Process()

    def close_stream(stream):
        stream.close()
        return None

    monkeypatch.setattr(
        uia_winapp,
        "preloaded_process_stdin",
        lambda _payload: FaultyClose(),
    )
    monkeypatch.setattr(
        uia_winapp.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(uia_winapp, "close_stream_os_enforced", close_stream)

    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path("winapp.exe"),
            ["--version"],
            timeout_seconds=1.0,
            max_output_bytes=1_024,
            input_bytes=b"fixture",
        )

    assert raised.value.code == "uia_cleanup_failed"
    assert raised.value.details == {
        "steps": ["staged_stdin:OSError"],
        "process_started": True,
        "lifecycle_stage": "post_spawn_cleanup",
    }
    assert process.killed is True
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_process_pipe_termination_failure_keeps_post_spawn_lifecycle(monkeypatch) -> None:
    class Process:
        stdin = None
        stdout = None
        stderr = None

    monkeypatch.setattr(uia_winapp.subprocess, "Popen", lambda *_a, **_k: Process())
    monkeypatch.setattr(
        uia_winapp,
        "_terminate_process",
        lambda *_args: (_ for _ in ()).throw(
            ComputerError("uia_termination_failed", "fixture termination failed")
        ),
    )

    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path("winapp.exe"),
            ["--version"],
            timeout_seconds=0.5,
            max_output_bytes=1_024,
        )

    assert raised.value.code == "uia_termination_failed"
    assert raised.value.details == {
        "process_started": True,
        "lifecycle_stage": "post_spawn_termination",
    }


def test_process_staged_close_with_termination_failure_keeps_termination_primary(
    monkeypatch,
) -> None:
    class FaultyClose(io.BytesIO):
        def close(self) -> None:
            raise OSError("fixture staged close failure")

    class Process:
        stdin = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode = None

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            raise OSError("fixture termination failure")

    def close_stream(stream):
        stream.close()
        return None

    monkeypatch.setattr(
        uia_winapp,
        "preloaded_process_stdin",
        lambda _payload: FaultyClose(),
    )
    monkeypatch.setattr(
        uia_winapp.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Process(),
    )
    monkeypatch.setattr(uia_winapp, "close_stream_os_enforced", close_stream)

    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path("winapp.exe"),
            ["--version"],
            timeout_seconds=1.0,
            max_output_bytes=1_024,
            input_bytes=b"fixture",
        )

    assert raised.value.code == "uia_termination_failed"
    assert raised.value.details["secondary_cleanup_errors"] == [
        "primary_cleanup:staged_stdin:OSError",
        "primary:uia_cleanup_failed",
    ]


def test_process_pipe_cleanup_failure_is_fatal(monkeypatch) -> None:
    class Stream(io.BytesIO):
        def __init__(self, *, fail_close: bool = False) -> None:
            super().__init__()
            self.fail_close = fail_close

        def close(self) -> None:
            if self.fail_close:
                raise OSError("fixture close failure")
            super().close()

    class Process:
        stdin = None
        stdout = Stream(fail_close=True)
        stderr = Stream()
        returncode = 0

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(timeout=None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(uia_winapp.subprocess, "Popen", lambda *_a, **_k: Process())
    monkeypatch.setattr(
        uia_winapp,
        "_read_pipe_available",
        lambda _stream: (b"", True),
    )
    monkeypatch.setattr(
        uia_winapp,
        "close_stream_os_enforced",
        lambda stream: "OSError" if getattr(stream, "fail_close", False) else None,
    )

    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path("winapp.exe"),
            ["--version"],
            timeout_seconds=0.5,
            max_output_bytes=1_024,
        )

    assert raised.value.code == "uia_cleanup_failed"
    assert raised.value.details["steps"] == ["stdout:OSError"]
    assert raised.value.details["process_started"] is True
    assert raised.value.details["lifecycle_stage"] == "post_spawn_cleanup"


def test_duplicate_scroll_selectors_are_all_excluded_from_action_candidates() -> None:
    duplicated = {
        "selector": "scroll-duplicate",
        "automationId": "TaskScroll",
    }
    unique = {"selector": "scroll-unique", "automationId": "TaskUnique"}

    result = uia_winapp._deduplicate_candidates(
        [duplicated, {**duplicated}, unique]
    )

    assert result == [unique]


def test_process_termination_failure_precedes_pipe_cleanup(monkeypatch) -> None:
    class Stream(io.BytesIO):
        pass

    class Process:
        stdin = None
        stdout = Stream()
        stderr = Stream()
        returncode = None

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            raise OSError("fixture termination failure")

    monkeypatch.setattr(uia_winapp.subprocess, "Popen", lambda *_a, **_k: Process())
    monkeypatch.setattr(
        uia_winapp,
        "_read_pipe_available",
        lambda _stream: (b"", False),
    )
    monkeypatch.setattr(
        uia_winapp,
        "close_stream_os_enforced",
        lambda _stream: "fixture_cleanup",
    )

    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path("winapp.exe"),
            ["--version"],
            timeout_seconds=0.5,
            max_output_bytes=1_024,
        )

    assert raised.value.code == "uia_termination_failed"
    assert raised.value.details["secondary_cleanup_errors"] == [
        "stdout:fixture_cleanup",
        "stderr:fixture_cleanup",
        "primary:uia_timeout",
    ]


def test_winapp_fatal_process_error_survives_post_revalidation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    backend.fail_on_capture = 2
    adapter = WinAppAdapter(backend)
    monkeypatch.setattr(adapter, "require", _binding)
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ComputerError("uia_termination_failed", "fixture leaked helper")
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.screenshot(_identity(), output=tmp_path / "never-written.png")

    assert raised.value.code == "uia_termination_failed"
    assert raised.value.details["secondary_errors"] == ["stale_window"]


def test_sanitized_environment_drops_secrets_and_disables_network_features(monkeypatch) -> None:
    monkeypatch.setenv("AGENTTOOLS_TEST_SECRET", "never-forward")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")

    environment = uia_winapp._sanitized_environment()

    assert "AGENTTOOLS_TEST_SECRET" not in environment
    assert environment["SystemRoot"] == r"C:\Windows"
    assert environment["WINAPP_CLI_UPDATE_CHECK"] == "0"
    assert environment["WINAPP_CLI_TELEMETRY_OPTOUT"] == "1"


def test_target_call_revalidates_identity_before_and_after(monkeypatch) -> None:
    backend = FakeBackend()
    adapter = WinAppAdapter(backend)
    monkeypatch.setattr(adapter, "require", _binding)
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(0, b'{"value": 1}', b""),
    )

    binding, payload = adapter._target_json(backend.identity, ["ui", "status"])

    assert binding.version == "0.6.0"
    assert payload == {"value": 1}
    assert backend.captures == 2


def test_target_call_prioritizes_post_call_identity_drift(monkeypatch) -> None:
    backend = FakeBackend()
    backend.fail_on_capture = 2
    adapter = WinAppAdapter(backend)
    monkeypatch.setattr(adapter, "require", _binding)
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(0, b'{"value": 1}', b""),
    )

    with pytest.raises(ComputerError, match="changed or closed") as raised:
        adapter._target_json(backend.identity, ["ui", "status"])

    assert raised.value.code == "stale_window"


def test_target_identity_allows_geometry_change(monkeypatch) -> None:
    backend = FakeBackend()
    backend.next_identity = WindowIdentity(
        **{
            **backend.identity.__dict__,
            "bounds": Bounds(30, 40, 1024, 768),
        }
    )
    adapter = WinAppAdapter(backend)
    monkeypatch.setattr(adapter, "require", _binding)
    monkeypatch.setattr(
        uia_winapp,
        "_run_process",
        lambda *_args, **_kwargs: ProcessResult(0, b'{"value": 1}', b""),
    )

    _binding_value, payload = adapter._target_json(backend.identity, ["ui", "status"])

    assert payload == {"value": 1}
    assert backend.captures == 2


def test_target_call_never_accepts_success_after_absolute_deadline(monkeypatch) -> None:
    backend = FakeBackend()
    adapter = WinAppAdapter(backend)
    clock = [100.0]
    monkeypatch.setattr(uia_winapp.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(adapter, "require", lambda *, deadline=None: _binding())

    def run(_path, _arguments, *, timeout_seconds, **_kwargs):
        assert 0 < timeout_seconds <= 1.0
        clock[0] = 101.1
        return ProcessResult(0, b'{"value": 1}', b"")

    monkeypatch.setattr(uia_winapp, "_run_process", run)

    with pytest.raises(ComputerError) as raised:
        adapter._target_json(
            backend.identity,
            ["ui", "status"],
            timeout_seconds=8.0,
            deadline=101.0,
        )

    assert raised.value.code == "uia_timeout"


def test_require_charges_hash_and_version_probe_to_absolute_deadline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "winapp.exe"
    candidate.write_bytes(b"fixture")
    clock = [100.0]
    observed: dict[str, object] = {}
    monkeypatch.setattr(uia_winapp.sys, "platform", "win32")
    monkeypatch.setattr(uia_winapp.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        uia_winapp,
        "_discover_candidate",
        lambda: (candidate, "fixture"),
    )

    def sha256(path, *, deadline=None):
        observed["hash_deadline"] = deadline
        clock[0] = 100.4
        return (
            uia_winapp.WINAPP_COMPANION_SHA256
            if Path(path).name == uia_winapp.WINAPP_COMPANION
            else uia_winapp.WINAPP_EXE_SHA256
        )

    def run(_path, _arguments, *, timeout_seconds, **_kwargs):
        observed["version_timeout"] = timeout_seconds
        clock[0] = 100.8
        return ProcessResult(0, b"0.6.0\n", b"")

    monkeypatch.setattr(uia_winapp, "_sha256", sha256)
    monkeypatch.setattr(uia_winapp, "_run_process", run)
    candidate.with_name(uia_winapp.WINAPP_COMPANION).write_bytes(b"companion")

    binding = WinAppAdapter(FakeBackend()).require(deadline=101.0)

    assert binding.version == "0.6.0"
    assert observed["hash_deadline"] == 101.0
    assert observed["version_timeout"] == pytest.approx(0.6)


def test_inspect_normalizes_and_omits_backend_values(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    payload = {
        "windows": [
            {
                "hwnd": 123,
                "elements": [
                    {
                        "type": "Window",
                        "name": "fixture",
                        "x": 0,
                        "y": 0,
                        "width": 800,
                        "height": 600,
                        "value": "password=should-not-survive",
                        "children": [
                            {
                                "type": "Button",
                                "name": "Go",
                                "selector": "btn-go-a1b2",
                                "isInvokable": True,
                                "x": 10,
                                "y": 10,
                                "width": 50,
                                "height": 20,
                            }
                        ],
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr(adapter, "_target_json", lambda *_args, **_kwargs: (_binding(), payload))

    _binding_value, result = adapter.inspect(
        _identity(), depth=3, interactive=False, max_elements=1
    )

    assert result["truncated"] is True
    assert result["values_omitted"] is True
    assert result["elements"][0]["element"] == "btn-go-a1b2"
    assert "value" not in result["elements"][0]
    assert "should-not-survive" not in json.dumps(result)
    normalized = semantic_accessibility._normalize_winapp_elements(result["elements"])
    expected_ancestor = semantic_accessibility._ancestor_signature(["|window|"])
    assert normalized[0]["locator"]["ancestor_signature"] == expected_ancestor
    assert normalized[0]["hierarchy"]["ancestor_signature"] == expected_ancestor
    assert "_ancestor_signature" not in normalized[0]


def test_inspect_bounds_provider_traversal_before_response_serialization(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    payload = {
        "windows": [
            {
                "hwnd": 123,
                "elements": [
                    {
                        "type": "Button",
                        "name": f"fixture-{index}",
                        "selector": f"btn-fixture-{index}",
                        "isInvokable": True,
                        "x": index,
                        "y": 1,
                        "width": 10,
                        "height": 10,
                    }
                    for index in range(1_000)
                ],
            }
        ]
    }
    calls: list[list[str]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return _binding(), payload

    monkeypatch.setattr(adapter, "_target_json", target)

    _binding_value, result = adapter.inspect(
        _identity(), depth=12, interactive=True, max_elements=2
    )

    assert "--interactive" not in calls[0]
    assert result["count"] == 2
    assert result["truncated"] is True
    assert result["interactive"] is True
    assert result["provider_filter_applied"] is False
    assert result["provider_bound"] == {
        "max_elements": 2,
        "interactive_filter_forwarded": False,
        "interactive_filter_applied": "agenttools_local_after_bounded_traversal",
        "upstream_native_max_supported": False,
        "enforced_before_agenttools_response_serialization": True,
        "external_output_guard": "byte_cap_and_process_termination",
    }
    assert [item["name"] for item in result["elements"]] == [
        "fixture-0",
        "fixture-1",
    ]


def test_password_read_never_invokes_get_value(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    calls: list[list[str]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return _binding(), {
            "elementId": "txt-password-a1b2",
            "properties": {"IsPassword": "True"},
        }

    monkeypatch.setattr(adapter, "_target_json", target)

    _binding_value, result = adapter.read(
        _identity(), selector="txt-password-a1b2", max_chars=100
    )

    assert len(calls) == 2
    assert [call[1] for call in calls] == ["get-property", "get-property"]
    assert all(call[1] != "get-value" for call in calls)
    assert result["text"] is None
    assert result["redaction_reason"] == "password_control"


def test_read_native_privacy_metadata_overrides_winapp_false_positive(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    responses = iter(
        [
            (
                _binding(),
                {"elementId": "txt-safe-a1b2", "properties": {"IsPassword": "True"}},
            ),
            (_binding(), {"elementId": "txt-safe-a1b2", "text": "safe fixture"}),
        ]
    )
    exact = {
        "password": False,
        "value_available": True,
        "range_value_available": False,
        "legacy_value_available": True,
        "value_pattern_status": "available",
        "range_value_pattern_status": "absent",
        "runtime_id": "42,7",
    }
    monkeypatch.setattr(adapter, "_target_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(
        adapter,
        "_element_properties",
        lambda *_args, **_kwargs: (_binding(), "txt-safe-a1b2", exact),
    )

    _binding_value, result = adapter.read(
        _identity(),
        selector="txt-safe-a1b2",
        max_chars=100,
    )

    assert result["text"] == "safe fixture"
    assert result["redacted"] is False
    assert result["redaction_reason"] is None


def test_read_bounds_and_redacts_credential_like_text(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    responses = iter(
        [
            (_binding(), {"elementId": "doc-a1b2", "properties": {"IsPassword": "False"}}),
            (
                _binding(),
                {
                    "elementId": "doc-a1b2",
                    "text": "safe password=hunter2 Bearer abcdefghijklmnop tail",
                },
            ),
        ]
    )
    monkeypatch.setattr(adapter, "_target_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(
        adapter,
        "_element_properties",
        lambda *_args, **_kwargs: (
            _binding(),
            "doc-a1b2",
            _trusted_winapp_read_properties(),
        ),
    )

    _binding_value, result = adapter.read(_identity(), selector="doc-a1b2", max_chars=40)

    assert result["redacted"] is True
    assert "hunter2" not in str(result["text"])
    assert "abcdefghijklmnop" not in str(result["text"])
    assert result["truncated"] is True


def test_read_does_not_fallback_after_native_privacy_failure(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    calls: list[list[str]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return (
            _binding(),
            {"elementId": "doc-a1b2", "properties": {"IsPassword": "False"}},
        )

    def protected(*_args, **_kwargs):
        raise ComputerError("password_control", "fixture protected state unknown")

    monkeypatch.setattr(adapter, "_target_json", target)
    monkeypatch.setattr(adapter, "_element_properties", protected)

    with pytest.raises(ComputerError) as raised:
        adapter.read(_identity(), selector="doc-a1b2", max_chars=100)

    assert raised.value.code == "password_control"
    assert [call[1] for call in calls] == ["get-property"]


def test_read_fails_closed_when_trusted_native_privacy_check_is_unavailable(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    calls: list[list[str]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return (
            _binding(),
            {"elementId": "doc-a1b2", "properties": {"IsPassword": "False"}},
        )

    def unavailable(*_args, **_kwargs):
        raise ComputerError("native_uia_unavailable", "fixture bridge unavailable")

    monkeypatch.setattr(adapter, "_target_json", target)
    monkeypatch.setattr(adapter, "_element_properties", unavailable)

    with pytest.raises(ComputerError) as raised:
        adapter.read(
            _identity(),
            selector="doc-a1b2",
            max_chars=100,
        )

    assert raised.value.code == "uia_sensitive_state_unavailable"
    assert [call[1] for call in calls] == ["get-property"]


def test_read_always_uses_exact_native_range_value_without_winapp_text(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    full_properties = {
        "elementId": "sld-range-a1b2",
        "properties": {
            "AutomationId": "TaskRange",
            "IsPassword": "False",
            "IsEnabled": "True",
            "IsValuePatternAvailable": "False",
            "IsRangeValuePatternAvailable": "True",
            "RangeValueIsReadOnly": "False",
        },
    }
    responses = iter(
        [
            (
                _binding(),
                {"elementId": "sld-range-a1b2", "properties": {"IsPassword": "False"}},
            ),
            (_binding(), full_properties),
            (_binding(), full_properties),
        ]
    )
    calls: list[list[str]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return next(responses)

    monkeypatch.setattr(adapter, "_target_json", target)
    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **_kwargs: {
            "automation_id": "TaskRange",
            "runtime_id": "42,7",
            "is_password": False,
            "enabled": True,
            "patterns": ["uia.RangeValuePattern"],
            "range_value_bits": "4029000000000000",
            "value_pattern_status": "absent",
            "range_value_pattern_status": "available",
            "legacy_value_read_only": None,
        },
    )

    _binding_value, result = adapter.read(
        _identity(), selector="sld-range-a1b2", max_chars=100
    )

    assert result["text"] == "12.5"
    assert result["char_count"] == 4
    assert result["resolved_element"] == "sld-range-a1b2"
    assert [call[1] for call in calls] == [
        "get-property",
        "get-property",
        "get-property",
    ]


def test_read_fails_closed_when_value_pattern_queries_are_inconclusive(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    responses = iter(
        [
            (
                _binding(),
                {"elementId": "txt-pattern-a1b2", "properties": {"IsPassword": "False"}},
            ),
            (
                _binding(),
                {
                    "elementId": "txt-pattern-a1b2",
                    "properties": {
                        "AutomationId": "TaskPattern",
                        "IsPassword": "False",
                        "IsEnabled": "True",
                    },
                },
            ),
        ]
    )
    calls: list[list[str]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return next(responses)

    monkeypatch.setattr(adapter, "_target_json", target)
    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **_kwargs: {
            "automation_id": "TaskPattern",
            "runtime_id": "42,7",
            "is_password": False,
            "enabled": True,
            "patterns": [],
            "range_value_bits": None,
            "value_pattern_status": "failed",
            "range_value_pattern_status": "absent",
            "legacy_value_read_only": None,
        },
    )

    with pytest.raises(ComputerError) as raised:
        adapter.read(_identity(), selector="txt-pattern-a1b2", max_chars=100)

    assert raised.value.code == "uia_pattern_query_failed"
    assert [call[1] for call in calls] == ["get-property", "get-property"]


def test_read_rejects_effective_pattern_drift_after_winapp_value_read(
    monkeypatch,
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    properties = {
        "elementId": "txt-pattern-a1b2",
        "properties": {
            "AutomationId": "TaskPattern",
            "IsPassword": "False",
            "IsEnabled": "True",
        },
    }
    responses = iter(
        [
            (
                _binding(),
                {"elementId": "txt-pattern-a1b2", "properties": {"IsPassword": "False"}},
            ),
            (_binding(), properties),
            (_binding(), {"elementId": "txt-pattern-a1b2", "text": "stale text"}),
            (_binding(), properties),
        ]
    )
    native_states = iter(
        [
            {
                "automation_id": "TaskPattern",
                "runtime_id": "42,7",
                "is_password": False,
                "enabled": True,
                "patterns": ["uia.ValuePattern"],
                "range_value_bits": None,
                "value_pattern_status": "available",
                "range_value_pattern_status": "absent",
                "legacy_value_read_only": None,
            },
            {
                "automation_id": "TaskPattern",
                "runtime_id": "42,7",
                "is_password": False,
                "enabled": True,
                "patterns": ["uia.RangeValuePattern"],
                "range_value_bits": "4029000000000000",
                "value_pattern_status": "absent",
                "range_value_pattern_status": "available",
                "legacy_value_read_only": None,
            },
        ]
    )
    calls: list[list[str]] = []

    def target(_identity, arguments, **_kwargs):
        calls.append(arguments)
        return next(responses)

    monkeypatch.setattr(adapter, "_target_json", target)
    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **_kwargs: next(native_states),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.read(_identity(), selector="txt-pattern-a1b2", max_chars=100)

    assert raised.value.code == "uia_pattern_changed"
    assert [call[1] for call in calls] == [
        "get-property",
        "get-property",
        "get-value",
        "get-property",
    ]


@pytest.mark.parametrize("value_payload", [{"text": 37}, {}])
def test_read_rejects_missing_or_non_string_winapp_value_payload(
    monkeypatch,
    value_payload: dict[str, object],
) -> None:
    adapter = WinAppAdapter(FakeBackend())
    properties = {
        "elementId": "txt-value-a1b2",
        "properties": {
            "AutomationId": "TaskValue",
            "IsPassword": "False",
            "IsEnabled": "True",
        },
    }
    responses = iter(
        [
            (
                _binding(),
                {"elementId": "txt-value-a1b2", "properties": {"IsPassword": "False"}},
            ),
            (_binding(), properties),
            (_binding(), {"elementId": "txt-value-a1b2", **value_payload}),
        ]
    )
    monkeypatch.setattr(adapter, "_target_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **_kwargs: {
            "automation_id": "TaskValue",
            "runtime_id": "42,7",
            "is_password": False,
            "enabled": True,
            "patterns": ["uia.ValuePattern"],
            "range_value_bits": None,
            "value_pattern_status": "available",
            "range_value_pattern_status": "absent",
            "legacy_value_read_only": None,
        },
    )

    with pytest.raises(ComputerError) as raised:
        adapter.read(_identity(), selector="txt-value-a1b2", max_chars=100)

    assert raised.value.code == "uia_invalid_response"


def test_scroll_areas_normalize_percent_viewport_and_remaining_directions(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    responses = iter(
        [
            (
                _binding(),
                {
                    "matchCount": 1,
                    "hasMore": False,
                    "matches": [
                            {
                                "selector": "pn-scroll-a1b2",
                                "type": "Pane",
                                "className": "ScrollFixture",
                                "name": "Viewport",
                            "x": 10,
                            "y": 20,
                            "width": 300,
                            "height": 200,
                        }
                    ]
                },
            ),
            (
                _binding(),
                {
                    "properties": {
                        "HorizontallyScrollable": "0x0",
                        "VerticallyScrollable": "0x1",
                        "ScrollHorizontalPercent": "-1",
                        "ScrollVerticalPercent": "25.5",
                    }
                },
            ),
        ]
    )
    monkeypatch.setattr(adapter, "_target_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(
        adapter,
        "_native_element_state",
        lambda *_args, **_kwargs: {
            "automation_id": "Scroller",
            "runtime_id": "42,scroll",
            "is_password": False,
            "enabled": True,
            "control_type": "pane",
            "class": "ScrollFixture",
            "ancestor_signature": "sha256:" + "a" * 24,
            "patterns": ["uia.ScrollPattern"],
        },
    )

    _binding_value, result = adapter.scroll_areas(_identity(), max_elements=10)

    area = result["areas"][0]
    assert area["hwnd"] == 123
    assert area["pid"] == 456
    assert area["viewport"] == {"width": 300, "height": 200}
    assert area["horizontal"]["percent_status"] == "no_scroll"
    assert area["vertical"]["percent"] == 25.5
    assert area["vertical"]["moreBefore"] is True
    assert area["vertical"]["moreAfter"] is True
    assert area["warning"] == "view_size_percent_unavailable"
    assert area["automation_id"] == "Scroller"
    assert area["locator"]["automation_id"] == "Scroller"


def test_scroll_areas_never_downgrade_target_drift(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    responses = iter(
        [
            (
                _binding(),
                {
                    "matchCount": 1,
                    "hasMore": False,
                    "matches": [
                        {
                            "selector": "pn-scroll-a1b2",
                            "type": "Pane",
                            "x": 0,
                            "y": 0,
                            "width": 10,
                            "height": 10,
                        }
                    ]
                },
            ),
            ComputerError("stale_window", "target changed"),
        ]
    )

    def target(*_args, **_kwargs):
        response = next(responses)
        if isinstance(response, ComputerError):
            raise response
        return response

    monkeypatch.setattr(adapter, "_target_json", target)

    with pytest.raises(ComputerError) as raised:
        adapter.scroll_areas(_identity(), max_elements=10)

    assert raised.value.code == "stale_window"


def test_password_state_missing_fails_closed_before_get_value(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    calls = 0

    def target(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _binding(), {"elementId": "doc-a1b2", "properties": {}}

    monkeypatch.setattr(adapter, "_target_json", target)

    with pytest.raises(ComputerError) as raised:
        adapter.read(_identity(), selector="doc-a1b2", max_chars=100)

    assert raised.value.code == "uia_sensitive_state_unavailable"
    assert calls == 1


def test_read_rejects_element_replacement_between_password_check_and_value(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    responses = iter(
        [
            (_binding(), {"elementId": "doc-a1b2", "properties": {"IsPassword": "False"}}),
            (_binding(), {"elementId": "doc-replaced-c3d4", "text": "must-not-return"}),
        ]
    )
    monkeypatch.setattr(adapter, "_target_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(
        adapter,
        "_element_properties",
        lambda *_args, **_kwargs: (
            _binding(),
            "doc-a1b2",
            _trusted_winapp_read_properties(),
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.read(_identity(), selector="doc-a1b2", max_chars=100)

    assert raised.value.code == "stale_element"


def test_read_compares_long_element_ids_without_lossy_normalization(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    shared_prefix = "doc-" + ("a" * 300)
    responses = iter(
        [
            (
                _binding(),
                {
                    "elementId": shared_prefix + "-before",
                    "properties": {"IsPassword": "False"},
                },
            ),
            (_binding(), {"elementId": shared_prefix + "-after", "text": "must-not-return"}),
        ]
    )
    monkeypatch.setattr(adapter, "_target_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(
        adapter,
        "_element_properties",
        lambda *_args, **_kwargs: (
            _binding(),
            shared_prefix + "-before",
            _trusted_winapp_read_properties(),
        ),
    )

    with pytest.raises(ComputerError) as raised:
        adapter.read(_identity(), selector="doc-a1b2", max_chars=100)

    assert raised.value.code == "stale_element"


def test_read_rechecks_password_state_before_returning_text(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    responses = iter(
        [
            (_binding(), {"elementId": "doc-a1b2", "properties": {"IsPassword": "False"}}),
            (_binding(), {"elementId": "doc-a1b2", "text": "must-not-return"}),
        ]
    )
    monkeypatch.setattr(adapter, "_target_json", lambda *_args, **_kwargs: next(responses))
    exact_states = iter(
        [
            _trusted_winapp_read_properties(),
            _trusted_winapp_read_properties(password=True),
        ]
    )
    monkeypatch.setattr(
        adapter,
        "_element_properties",
        lambda *_args, **_kwargs: (_binding(), "doc-a1b2", next(exact_states)),
    )

    _binding_value, result = adapter.read(_identity(), selector="doc-a1b2", max_chars=100)

    assert result["text"] is None
    assert result["redaction_reason"] == "password_control"


def test_malformed_backend_boolean_is_unavailable_not_exception() -> None:
    assert uia_winapp._parse_bool({"unexpected": True}) is None


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_scroll_percent_is_unavailable(value: str) -> None:
    assert uia_winapp._parse_percent(value) is None


def test_scroll_areas_preserve_backend_truncation_truth(monkeypatch) -> None:
    adapter = WinAppAdapter(FakeBackend())
    monkeypatch.setattr(
        adapter,
        "_target_json",
        lambda *_args, **_kwargs: (
            _binding(),
            {"matchCount": 75, "hasMore": True, "matches": []},
        ),
    )

    _binding_value, result = adapter.scroll_areas(_identity(), max_elements=10)

    assert result["truncated"] is True
    assert result["searched_candidates"] == 75


def test_backend_errors_are_stable_and_do_not_leak_raw_text() -> None:
    raw = b'{"error":{"code":"internal_error","message":"secret provider exploded"}}'

    error = uia_winapp._map_backend_error(raw)

    assert error.code == "uia_backend_error"
    assert "secret" not in error.message


def test_provider_unavailable_error_is_stable() -> None:
    raw = (
        b'{"error":{"code":"internal_error","message":"Window is not accessible.",'
        b'"details":"AppNotFoundException"}}'
    )

    error = uia_winapp._map_backend_error(raw)

    assert error.code == "uia_provider_unavailable"


@pytest.mark.parametrize(
    ("source", "forbidden"),
    [
        ('{"password":"example"}', "example"),
        ("password='multi word value'", "multi word value"),
        (r'{"password":"prefix\"SECRET_SUFFIX"}', "SECRET_SUFFIX"),
        ("password='first line\nSECRET_SECOND_LINE'", "SECRET_SECOND_LINE"),
        ("authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ('{"api_key":"abc123xyz"}', "abc123xyz"),
    ],
)
def test_common_credential_formats_are_fully_redacted(source: str, forbidden: str) -> None:
    output, redacted = uia_winapp.redact_credential_like_text(source)

    assert redacted is True
    assert forbidden not in output
    assert "[REDACTED]" in output


def test_unterminated_quoted_credential_redaction_is_linear_and_fail_closed() -> None:
    source = 'password="' + ("\\" * 34) + "SECRET_SUFFIX"

    started = time.perf_counter()
    output, redacted = uia_winapp.redact_credential_like_text(source)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.25
    assert redacted is True
    assert "SECRET_SUFFIX" not in output
    assert output == 'password="[REDACTED]'


def test_ambiguous_selector_error_is_stable() -> None:
    raw = b'{"error":{"code":"ambiguous","message":"Multiple matches: secret names"}}'

    error = uia_winapp._map_backend_error(raw)

    assert error.code == "ambiguous_element"
    assert "secret" not in error.message


def test_stale_selector_error_is_stable() -> None:
    raw = b'{"error":{"code":"stale","message":"Element may have changed. Re-run inspect."}}'

    error = uia_winapp._map_backend_error(raw)

    assert error.code == "stale_element"


def test_process_output_cap_terminates_child() -> None:
    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path(sys.executable),
            ["-u", "-c", "import sys; sys.stdout.write('x' * 100000); sys.stdout.flush()"],
            timeout_seconds=5.0,
            max_output_bytes=4_096,
        )

    assert raised.value.code == "uia_output_too_large"


def test_process_timeout_terminates_child() -> None:
    with pytest.raises(ComputerError) as raised:
        uia_winapp._run_process(
            Path(sys.executable),
            ["-c", "import time; time.sleep(5)"],
            timeout_seconds=0.1,
            max_output_bytes=4_096,
        )

    assert raised.value.code == "uia_timeout"


def test_cli_commands_emit_stable_versioned_json(monkeypatch, capsys) -> None:
    sample_window = _identity().public()
    monkeypatch.setattr(
        computer_cli,
        "inspect_semantic_window",
        lambda **_kwargs: {
            "window": sample_window,
            "backend": _binding().public(),
            "count": 0,
            "truncated": False,
            "elements": [],
        },
    )
    monkeypatch.setattr(
        computer_cli,
        "read_semantic_element",
        lambda **_kwargs: {
            "window": sample_window,
            "backend": _binding().public(),
            "element": "doc-a1b2",
            "text": "known",
            "char_count": 5,
            "truncated": False,
            "redacted": False,
        },
    )
    monkeypatch.setattr(
        computer_cli,
        "list_scroll_areas",
        lambda **_kwargs: {
            "window": sample_window,
            "backend": _binding().public(),
            "count": 0,
            "truncated": False,
            "areas": [],
        },
    )

    commands = [
        ["computer", "inspect", "--hwnd", "123", "--json"],
        ["computer", "read", "--hwnd", "123", "--element", "doc-a1b2", "--json"],
        ["computer", "scroll-areas", "--hwnd", "123", "--json"],
    ]
    for arguments in commands:
        assert root_cli.main(arguments) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == "agent-tools.computer/v1"
        assert payload["ok"] is True


@pytest.mark.parametrize(
    ("arguments", "argument", "minimum", "maximum"),
    [
        (["computer", "inspect", "--hwnd", "1", "--depth", "0", "--json"], "--depth", 1, 12),
        (["computer", "inspect", "--hwnd", "1", "--depth", "13", "--json"], "--depth", 1, 12),
        (
            ["computer", "inspect", "--hwnd", "1", "--max-elements", "201", "--json"],
            "--max-elements",
            1,
            MAX_INSPECT_ELEMENTS,
        ),
        (
            ["computer", "read", "--hwnd", "1", "--element", "doc", "--max-chars", "0", "--json"],
            "--max-chars",
            1,
            MAX_READ_CHARS,
        ),
        (
            ["computer", "scroll-areas", "--hwnd", "1", "--max-elements", "51", "--json"],
            "--max-elements",
            1,
            MAX_SCROLL_AREAS,
        ),
    ],
)
def test_cli_rejects_unbounded_uia_arguments(
    arguments: list[str], argument: str, minimum: int, maximum: int, capsys
) -> None:
    assert root_cli.main(arguments) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "argument_out_of_range"
    details = payload["error"]["details"]
    assert details["argument"] == argument
    assert details["minimum"] == minimum
    assert details["maximum"] == maximum
    assert details["retryable"] is False
    assert isinstance(details["next_action"], str) and details["next_action"]


def test_non_windows_capabilities_degrade_without_import_failure(monkeypatch) -> None:
    monkeypatch.setattr(uia_winapp.sys, "platform", "linux")

    result = uia_winapp.capabilities(adapter=WinAppAdapter(FakeBackend()))

    assert result["uia_winapp"]["available"] is False
    assert result["uia_winapp"]["error_code"] == "capability_missing"
    assert result["pin"]["runtime_auto_download"] is False


def test_no_physical_or_semantic_mutation_command_is_public() -> None:
    parser = root_cli.build_parser()
    with pytest.raises(ComputerError) as scroll_error:
        parser.parse_args(["computer", "scroll", "--hwnd", "123"])
    assert scroll_error.value.code == "missing_required_argument"
    with pytest.raises(ComputerError) as click_error:
        parser.parse_args(["computer", "click", "--hwnd", "123"])
    assert click_error.value.code == "missing_required_argument"
