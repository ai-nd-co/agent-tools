from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from agent_tools.computer import actions, uia_winapp
from agent_tools.computer.actions import (
    ActionCoordinator,
    focus_window,
    invoke_element,
    resize_window,
    scroll_element,
    set_element_value,
    set_window_state,
)
from agent_tools.computer.element_refs import ElementReferenceStore
from agent_tools.computer.models import ComputerError
from agent_tools.computer.semantic_accessibility import (
    inspect_semantic_window,
    read_semantic_element,
)
from agent_tools.computer.uia_winapp import (
    WinAppAdapter,
    inspect_window,
    list_scroll_areas,
    read_element,
)
from agent_tools.computer.win32_backend import Win32Backend

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only integration")

REPO = Path(__file__).parents[1]
POC = REPO / "scripts" / "windows_control_poc.py"
NOTEPAD = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "notepad.exe"


def _visible_windows() -> dict[int, tuple[int, str]]:
    import win32gui
    import win32process

    rows: dict[int, tuple[int, str]] = {}

    def visit(hwnd: int, _extra: object) -> bool:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
                rows[hwnd] = (int(pid), title)
        return True

    win32gui.EnumWindows(visit, None)
    return rows


def _find_new_window(before: set[int], marker: str, timeout: float = 12.0) -> tuple[int, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = [
            (hwnd, pid)
            for hwnd, (pid, title) in _visible_windows().items()
            if hwnd not in before and marker.casefold() in title.casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            pytest.fail(f"Ambiguous task-owned fixture windows: {matches}")
        time.sleep(0.05)
    pytest.fail(f"Task-owned fixture did not appear: {marker}")


def _close_exact(hwnd: int, pid: int, marker: str) -> None:
    import win32con
    import win32gui
    import win32process

    if not win32gui.IsWindow(hwnd):
        return
    _thread_id, actual_pid = win32process.GetWindowThreadProcessId(hwnd)
    title = win32gui.GetWindowText(hwnd)
    assert int(actual_pid) == pid and marker.casefold() in title.casefold()
    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
    deadline = time.monotonic() + 6.0
    while win32gui.IsWindow(hwnd) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not win32gui.IsWindow(hwnd)


@contextmanager
def _classic_fixture() -> Iterator[tuple[int, int, str]]:
    marker = f"AgentTools-273-Classic-{uuid.uuid4().hex}"
    before = set(_visible_windows())
    process = subprocess.Popen(
        [sys.executable, str(POC), "_target-process", "--title", marker],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    hwnd = 0
    pid = 0
    try:
        hwnd, pid = _find_new_window(before, marker, timeout=8.0)
        yield hwnd, pid, marker
    finally:
        if hwnd:
            _close_exact(hwnd, pid, marker)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@contextmanager
def _notepad_fixture(tmp_path: Path) -> Iterator[tuple[int, int, str]]:
    marker = f"agenttools-273-notepad-{uuid.uuid4().hex}"
    document = tmp_path / f"{marker}.txt"
    document.write_text(
        "\r\n".join(f"TASK_273_KNOWN_{index:04d}" for index in range(1_200)),
        encoding="utf-8",
    )
    before = set(_visible_windows())
    launcher = subprocess.Popen(
        [str(NOTEPAD), str(document)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    hwnd = 0
    pid = 0
    try:
        hwnd, pid = _find_new_window(before, marker)
        yield hwnd, pid, marker
    finally:
        if hwnd:
            _close_exact(hwnd, pid, marker)
        if launcher.poll() is None and launcher.pid == pid:
            launcher.terminate()
        try:
            launcher.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if launcher.pid == pid:
                launcher.kill()
                launcher.wait(timeout=5)


@contextmanager
def _wpf_fixture(*, vanishing_pattern_path: Path | None = None) -> Iterator[tuple[int, int, str]]:
    marker = f"AgentTools-273-WPF-{uuid.uuid4().hex}"
    custom_type_setup = ""
    custom_control_setup = ""
    custom_control_add = ""
    if vanishing_pattern_path is not None:
        type_source = r"""
using System;
using System.IO;
using System.Windows.Automation;
using System.Windows.Automation.Peers;
using System.Windows.Automation.Provider;
using System.Windows.Controls;

public sealed class AgentToolsVanishingValue : TextBox {
    public string BlockPath { get; set; }

    protected override AutomationPeer OnCreateAutomationPeer() {
        return new AgentToolsVanishingValuePeer(this);
    }
}

public sealed class AgentToolsVanishingValuePeer : TextBoxAutomationPeer {
    private readonly AgentToolsVanishingValueProvider provider;

    public AgentToolsVanishingValuePeer(AgentToolsVanishingValue owner) : base(owner) {
        provider = new AgentToolsVanishingValueProvider(owner);
    }

    private AgentToolsVanishingValue Target {
        get { return (AgentToolsVanishingValue)Owner; }
    }

        public override object GetPattern(PatternInterface patternInterface) {
            if (patternInterface == PatternInterface.Value) {
                if (File.Exists(Target.BlockPath)) {
                    var mode = File.ReadAllText(Target.BlockPath).Trim();
                    if (mode == "pattern") return null;
                }
                return provider;
            }
        return base.GetPattern(patternInterface);
    }

    protected override string GetAutomationIdCore() { return Target.Name; }
    protected override string GetClassNameCore() { return "AgentToolsVanishingValue"; }
    protected override string GetNameCore() { return "Vanishing value fixture"; }
    protected override AutomationControlType GetAutomationControlTypeCore() {
        return AutomationControlType.Edit;
    }
}

public sealed class AgentToolsVanishingValueProvider : IValueProvider {
    private readonly AgentToolsVanishingValue target;

    public AgentToolsVanishingValueProvider(AgentToolsVanishingValue target) {
        this.target = target;
    }

        public bool IsReadOnly {
            get {
                if (File.Exists(target.BlockPath) &&
                    File.ReadAllText(target.BlockPath).Trim() == "stale") {
                    throw new ElementNotAvailableException();
                }
                return false;
            }
        }

    public string Value {
        get { return target.Dispatcher.Invoke(() => target.Text ?? ""); }
    }

    public void SetValue(string value) {
        target.Dispatcher.Invoke(() => target.Text = value);
    }
}
"""
        encoded_type = base64.b64encode(type_source.encode("utf-8")).decode("ascii")
        escaped_path = str(vanishing_pattern_path).replace("'", "''")
        custom_type_setup = f"""
$typeSource = [Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String('{encoded_type}'))
Add-Type -AssemblyName PresentationCore
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationProvider
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Xaml
$typeReferences = @(
    [System.Windows.Controls.TextBox].Assembly.Location,
    [System.Windows.Automation.Peers.AutomationPeer].Assembly.Location,
    [System.Windows.Automation.ElementNotAvailableException].Assembly.Location,
    [System.Windows.Automation.Provider.IValueProvider].Assembly.Location,
    [System.Windows.DependencyObject].Assembly.Location,
    [System.Xaml.AttachablePropertyServices].Assembly.Location) |
    Select-Object -Unique
Add-Type -TypeDefinition $typeSource -ReferencedAssemblies $typeReferences -ErrorAction Stop
"""
        custom_control_setup = f"""
$vanishing = New-Object AgentToolsVanishingValue
$vanishing.Name = 'TaskVanishingValue'
$vanishing.BlockPath = '{escaped_path}'
$vanishing.Text = 'Initial vanishing fixture value'
$vanishing.Height = 24
"""
        custom_control_add = f"""
[void]$panel.Children.Add($vanishing)
$vanishingTimer = New-Object System.Windows.Threading.DispatcherTimer
$vanishingTimer.Interval = [TimeSpan]::FromMilliseconds(25)
$vanishingTimer.Add_Tick({{
    if ((Test-Path -LiteralPath '{escaped_path}') -and
        ((Get-Content -LiteralPath '{escaped_path}' -Raw).Trim() -eq 'stale')) {{
        $index = $panel.Children.IndexOf($vanishing)
        if ($index -ge 0) {{ [void]$panel.Children.RemoveAt($index) }}
        $vanishingTimer.Stop()
    }}
}})
$window.Add_Closed({{ $vanishingTimer.Stop() }})
$vanishingTimer.Start()
"""
    script = f"""
Add-Type -AssemblyName PresentationFramework
{custom_type_setup}
$window = New-Object System.Windows.Window
$window.Title = '{marker}'
$window.Width = 520
$window.Height = 360
$panel = New-Object System.Windows.Controls.StackPanel
$text = New-Object System.Windows.Controls.TextBox
$text.Name = 'TaskValue'
$text.Text = 'Initial fixture value'
$password = New-Object System.Windows.Controls.PasswordBox
$password.Name = 'TaskPassword'
$password.Password = 'fixture-password'
$button = New-Object System.Windows.Controls.Button
$button.Name = 'TaskInvoke'
$button.Content = 'Click 0'
$button.Add_Click({{ $button.Content = 'Click 1' }})
$toggle = New-Object System.Windows.Controls.CheckBox
$toggle.Name = 'TaskToggle'
$toggle.Content = 'Toggle fixture'
$list = New-Object System.Windows.Controls.ListBox
$list.Name = 'TaskList'
$item = New-Object System.Windows.Controls.ListBoxItem
$item.Name = 'TaskSelect'
$item.Content = 'Select fixture'
[void]$list.Items.Add($item)
$replaceText = New-Object System.Windows.Controls.TextBox
$replaceText.Name = 'TaskReplaceValue'
$replaceText.Text = 'Original replacement fixture'
$replaceButton = New-Object System.Windows.Controls.Button
$replaceButton.Name = 'TaskReplaceButton'
$replaceButton.Content = 'Replace control'
{custom_control_setup}
$window.Tag = $replaceText
$replaceButton.Add_Click({{
    $old = [System.Windows.Controls.TextBox]$window.Tag
    $index = $panel.Children.IndexOf($old)
    [void]$panel.Children.Remove($old)
    $new = New-Object System.Windows.Controls.TextBox
    $new.Name = 'TaskReplaceValue'
    $new.Text = 'Fresh replacement fixture'
    [void]$panel.Children.Insert($index, $new)
    $window.Tag = $new
}})
$focusReplaceText = New-Object System.Windows.Controls.TextBox
$focusReplaceText.Name = 'TaskFocusReplaceValue'
$focusReplaceText.Text = 'Original focus replacement fixture'
$focusReplaceText.Tag = 'disarmed'
$focusReplaceText.Add_GotKeyboardFocus({{
    if ($focusReplaceText.Tag -eq 'armed') {{
        $focusReplaceText.Tag = 'done'
        $index = $panel.Children.IndexOf($focusReplaceText)
        [void]$panel.Children.Remove($focusReplaceText)
        $new = New-Object System.Windows.Controls.TextBox
        $new.Name = 'TaskFocusReplaceValue'
        $new.Text = 'Fresh focus replacement fixture'
        [void]$panel.Children.Insert($index, $new)
    }}
}})
[void]$panel.Children.Add($text)
[void]$panel.Children.Add($password)
[void]$panel.Children.Add($button)
[void]$panel.Children.Add($toggle)
[void]$panel.Children.Add($list)
[void]$panel.Children.Add($replaceText)
[void]$panel.Children.Add($replaceButton)
[void]$panel.Children.Add($focusReplaceText)
{custom_control_add}
$window.Content = $panel
$window.Add_ContentRendered({{ $focusReplaceText.Tag = 'armed' }})
[void]$window.ShowDialog()
"""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    powershell = (
        Path(os.environ.get("WINDIR", r"C:\Windows"))
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    before = set(_visible_windows())
    process = subprocess.Popen(
        [
            str(powershell),
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-EncodedCommand",
            encoded,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=(subprocess.PIPE if vanishing_pattern_path is not None else subprocess.DEVNULL),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    hwnd = 0
    try:
        try:
            hwnd, pid = _find_new_window(before, marker)
        except BaseException:
            if process.poll() is not None and process.stderr is not None:
                stderr = process.stderr.read().decode(errors="replace")
                pytest.fail(f"WPF fixture failed before opening: {stderr}")
            raise
        assert pid == process.pid
        yield hwnd, pid, marker
    finally:
        if hwnd:
            _close_exact(hwnd, process.pid, marker)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if vanishing_pattern_path is not None:
            vanishing_pattern_path.unlink(missing_ok=True)


def test_real_disposable_focus_resize_set_value_and_invoke(monkeypatch, tmp_path: Path) -> None:
    import win32con
    import win32gui

    monkeypatch.setenv(actions.STATE_DIR_ENV, str(tmp_path / "state"))
    backend = Win32Backend()
    adapter = WinAppAdapter(backend)
    with _classic_fixture() as (hwnd, _pid, _marker):
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        minimized_bounds = tuple(win32gui.GetWindowRect(hwnd))
        with pytest.raises(ComputerError) as minimized:
            resize_window(
                hwnd=hwnd,
                x=80,
                y=80,
                width=520,
                height=360,
                short_explanation=None,
                backend=backend,
            )
        assert minimized.value.code == "window_minimized"
        assert tuple(win32gui.GetWindowRect(hwnd)) == minimized_bounds
        try:
            focused = focus_window(
                hwnd=hwnd,
                short_explanation=None,
                backend=backend,
            )
        except ComputerError as exc:
            assert exc.code == "focus_postcondition_failed"
            assert exc.details["outcome"] == "failed"
            assert not win32gui.IsIconic(hwnd)
        else:
            assert focused["outcome"] == "postcondition_verified"
            assert focused["foreground_hwnd"] == hwnd

        virtual = backend.display_info()["virtual_bounds"]
        before_offscreen = tuple(win32gui.GetWindowRect(hwnd))
        with pytest.raises(ComputerError) as offscreen:
            resize_window(
                hwnd=hwnd,
                x=int(virtual["x"]) + int(virtual["width"]) - 100,
                y=int(virtual["y"]),
                width=200,
                height=200,
                short_explanation=None,
                backend=backend,
            )
        assert offscreen.value.code == "geometry_offscreen"
        assert tuple(win32gui.GetWindowRect(hwnd)) == before_offscreen

        class MinimizeAtBoundaryBackend(Win32Backend):
            security_checks = 0

            def assert_action_allowed(self, expected):
                result = super().assert_action_allowed(expected)
                self.security_checks += 1
                if self.security_checks == 2:
                    win32gui.ShowWindow(expected.hwnd, win32con.SW_MINIMIZE)
                return result

        with pytest.raises(ComputerError) as moved_state:
            resize_window(
                hwnd=hwnd,
                x=int(virtual["x"]) + 80,
                y=int(virtual["y"]) + 80,
                width=520,
                height=360,
                short_explanation=None,
                backend=MinimizeAtBoundaryBackend(),
            )
        assert moved_state.value.code == "window_minimized"
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

        resized = resize_window(
            hwnd=hwnd,
            x=int(virtual["x"]) + 80,
            y=int(virtual["y"]) + 80,
            width=520,
            height=360,
            short_explanation=None,
            backend=backend,
        )
        assert resized["actual_bounds"] == resized["requested_bounds"]
        assert resized["postcondition_verified"] is True

    with _wpf_fixture() as (hwnd, _pid, _marker):
        inspection = inspect_window(
            hwnd=hwnd,
            depth=5,
            interactive=False,
            max_elements=50,
            backend=backend,
            adapter=adapter,
        )
        edit = next(
            item for item in inspection["elements"] if item.get("automation_id") == "TaskValue"
        )
        button = next(
            item for item in inspection["elements"] if item.get("automation_id") == "TaskInvoke"
        )
        password = next(
            item for item in inspection["elements"] if item.get("automation_id") == "TaskPassword"
        )
        toggle = next(
            item for item in inspection["elements"] if item.get("automation_id") == "TaskToggle"
        )
        selection = next(
            item for item in inspection["elements"] if item.get("automation_id") == "TaskSelect"
        )
        replace_value = next(
            item
            for item in inspection["elements"]
            if item.get("automation_id") == "TaskReplaceValue"
        )
        replace_button = next(
            item
            for item in inspection["elements"]
            if item.get("automation_id") == "TaskReplaceButton"
        )
        with pytest.raises(ComputerError) as password_error:
            set_element_value(
                hwnd=hwnd,
                element=str(password["element"]),
                value="must-not-be-written",
                short_explanation=None,
                backend=backend,
                adapter=adapter,
            )
        assert password_error.value.code == "password_control"
        if password_error.value.details["outcome"] == "delivery_only":
            assert password_error.value.details["partial_mutation"] == "focus"
            assert password_error.value.details["focus"]["changed"] is True
        else:
            assert password_error.value.details["outcome"] == "rejected"
            assert password_error.value.details["focus"]["changed"] is False
        set_result = set_element_value(
            hwnd=hwnd,
            element=str(edit["element"]),
            value="Task 273 known value",
            short_explanation=None,
            backend=backend,
            adapter=adapter,
        )
        assert set_result["outcome"] == "postcondition_verified"
        assert set_result["method"] == "uia.ValuePattern"
        assert "Task 273 known value" not in str(set_result)
        _binding, raw_readback = adapter._target_json(
            backend.capture_identity(hwnd),
            ["ui", "get-value", str(edit["element"])],
        )
        assert raw_readback["text"] == "Task 273 known value"

        marker = tmp_path / "state" / actions.DISABLE_MARKER
        original_run_process = uia_winapp._run_process

        def disable_at_helper_boundary(executable, arguments, **kwargs):
            input_bytes = kwargs.get("input_bytes")
            if input_bytes is not None:
                request = json.loads(input_bytes.decode("ascii"))
                if request.get("action") == "set-value":
                    marker.write_text("disabled\n", encoding="ascii")
            return original_run_process(executable, arguments, **kwargs)

        with monkeypatch.context() as boundary_patch:
            boundary_patch.setattr(uia_winapp, "_run_process", disable_at_helper_boundary)
            with pytest.raises(ComputerError) as disabled_at_boundary:
                set_element_value(
                    hwnd=hwnd,
                    element=str(edit["element"]),
                    value="must-not-cross-emergency-boundary",
                    short_explanation=None,
                    backend=backend,
                    adapter=adapter,
                )
        assert disabled_at_boundary.value.code == "computer_actions_disabled"
        marker.unlink()
        _binding, unchanged_readback = adapter._target_json(
            backend.capture_identity(hwnd),
            ["ui", "get-value", str(edit["element"])],
        )
        assert unchanged_readback["text"] == "Task 273 known value"

        original_title = win32gui.GetWindowText(hwnd)

        def drift_window_at_helper_boundary(executable, arguments, **kwargs):
            input_bytes = kwargs.get("input_bytes")
            if input_bytes is not None:
                request = json.loads(input_bytes.decode("ascii"))
                if request.get("action") == "set-value":
                    win32gui.SetWindowText(hwnd, f"{original_title}-drifted")
            return original_run_process(executable, arguments, **kwargs)

        try:
            with monkeypatch.context() as boundary_patch:
                boundary_patch.setattr(uia_winapp, "_run_process", drift_window_at_helper_boundary)
                title_tolerant = set_element_value(
                    hwnd=hwnd,
                    element=str(edit["element"]),
                    value="Task 285 title-tolerant value",
                    short_explanation=None,
                    backend=backend,
                    adapter=adapter,
                )
            assert title_tolerant["outcome"] == "postcondition_verified"
            assert title_tolerant["title_changed_during_operation"] is True
        finally:
            win32gui.SetWindowText(hwnd, original_title)
        _binding, changed_after_title_drift = adapter._target_json(
            backend.capture_identity(hwnd),
            ["ui", "get-value", str(edit["element"])],
        )
        assert changed_after_title_drift["text"] == "Task 285 title-tolerant value"

        try:
            with monkeypatch.context() as boundary_patch:
                boundary_patch.setattr(uia_winapp, "_run_process", drift_window_at_helper_boundary)
                with pytest.raises(ComputerError) as strict_title:
                    set_element_value(
                        hwnd=hwnd,
                        element=str(edit["element"]),
                        value="must-not-cross-strict-title-boundary",
                        short_explanation=None,
                        backend=backend,
                        adapter=adapter,
                        require_title_match=True,
                    )
            assert strict_title.value.code == "stale_window"
            assert strict_title.value.details["outcome"] == "rejected"
            assert strict_title.value.details["identity_mismatch_fields"] == ["title"]
        finally:
            win32gui.SetWindowText(hwnd, original_title)

        invoked = invoke_element(
            hwnd=hwnd,
            element=str(button["element"]),
            short_explanation=None,
            backend=backend,
            adapter=adapter,
        )
        assert invoked["outcome"] == "delivery_only"
        assert invoked["method"] == "uia.InvokePattern"
        button_read = read_element(
            hwnd=hwnd,
            element=str(button["element"]),
            max_chars=100,
            backend=backend,
            adapter=adapter,
        )
        assert button_read["text"] == "Click 1"

        toggled = invoke_element(
            hwnd=hwnd,
            element=str(toggle["element"]),
            short_explanation=None,
            backend=backend,
            adapter=adapter,
        )
        assert toggled["outcome"] == "postcondition_verified"
        assert toggled["pattern"] == "uia.TogglePattern"
        assert toggled["before"]["toggle_state"] != toggled["after"]["toggle_state"]

        selected = invoke_element(
            hwnd=hwnd,
            element=str(selection["element"]),
            short_explanation=None,
            backend=backend,
            adapter=adapter,
        )
        assert selected["outcome"] == "postcondition_verified"
        assert selected["pattern"] == "uia.SelectionItemPattern"
        assert selected["after"]["selected"] is True

        identity = backend.capture_identity(hwnd)
        old_binding, old_element_id, old_properties = adapter._action_properties(
            identity, str(replace_value["element"])
        )
        replaced = invoke_element(
            hwnd=hwnd,
            element=str(replace_button["element"]),
            short_explanation=None,
            backend=backend,
            adapter=adapter,
        )
        assert replaced["outcome"] == "delivery_only"
        real_action_properties = adapter._action_properties
        stale_returned = False

        def stale_once(action_identity, selector, **kwargs):
            nonlocal stale_returned
            if not stale_returned and selector == str(replace_value["element"]):
                stale_returned = True
                return old_binding, old_element_id, old_properties
            return real_action_properties(action_identity, selector, **kwargs)

        monkeypatch.setattr(adapter, "_action_properties", stale_once)
        with pytest.raises(ComputerError) as stale_element:
            set_element_value(
                hwnd=hwnd,
                element=str(replace_value["element"]),
                value="must-not-reach-replacement",
                short_explanation=None,
                backend=backend,
                adapter=adapter,
            )
        assert stale_element.value.code == "stale_element"
        monkeypatch.setattr(adapter, "_action_properties", real_action_properties)
        refreshed = inspect_window(
            hwnd=hwnd,
            depth=5,
            interactive=False,
            max_elements=80,
            backend=backend,
            adapter=adapter,
        )
        fresh_value = next(
            item
            for item in refreshed["elements"]
            if item.get("automation_id") == "TaskReplaceValue"
        )
        _binding, fresh_readback = adapter._target_json(
            backend.capture_identity(hwnd),
            ["ui", "get-value", str(fresh_value["element"])],
        )
        assert fresh_readback["text"] == "Fresh replacement fixture"


def test_real_required_focus_replacement_fails_before_value_delivery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(actions.STATE_DIR_ENV, str(tmp_path / "state"))
    backend = Win32Backend()
    adapter = WinAppAdapter(backend)
    with _wpf_fixture() as (hwnd, _pid, _marker):
        inspection = inspect_window(
            hwnd=hwnd,
            depth=5,
            interactive=False,
            max_elements=50,
            backend=backend,
            adapter=adapter,
        )
        focus_replace_value = next(
            item
            for item in inspection["elements"]
            if item.get("automation_id") == "TaskFocusReplaceValue"
        )
        final_error: ComputerError | None = None
        for _attempt in range(3):
            try:
                prefocused = backend.focus_window(backend.capture_identity(hwnd))
                assert prefocused["postcondition_verified"] is True
                set_element_value(
                    hwnd=hwnd,
                    element=str(focus_replace_value["element"]),
                    value="must-not-reach-focus-replacement",
                    short_explanation=None,
                    backend=backend,
                    adapter=adapter,
                    require_keyboard_focus=True,
                )
            except ComputerError as exc:
                final_error = exc
                if exc.code in {
                    "focus_postcondition_failed",
                    "secure_desktop_unavailable",
                    "target_not_foreground",
                }:
                    continue
                break
            pytest.fail("replacement-on-focus value mutation unexpectedly succeeded")
        assert final_error is not None
        assert final_error.code == "stale_element"
        assert final_error.details["outcome"] == "delivery_only"
        assert final_error.details["method"] == "uia.ValuePattern"
        assert final_error.details["partial_mutation"] == "element_focus"
        assert final_error.details["focus_delivery"] == "delivered"
        assert final_error.details["value_delivery"] == "not_attempted"
        assert final_error.details["delivery"]["status"] == "partial"
        assert final_error.details["semantic_outcome"] == "delivery_only"

        refreshed = inspect_window(
            hwnd=hwnd,
            depth=5,
            interactive=False,
            max_elements=50,
            backend=backend,
            adapter=adapter,
        )
        fresh_focus_value = next(
            item
            for item in refreshed["elements"]
            if item.get("automation_id") == "TaskFocusReplaceValue"
        )
        _binding, readback = adapter._target_json(
            backend.capture_identity(hwnd),
            ["ui", "get-value", str(fresh_focus_value["element"])],
        )
        assert readback["text"] == "Fresh focus replacement fixture"


def test_real_disposable_notepad_scroll_direction_percent_and_limits(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(actions.STATE_DIR_ENV, str(tmp_path / "state"))
    backend = Win32Backend()
    adapter = WinAppAdapter(backend)
    with _notepad_fixture(tmp_path) as (hwnd, _pid, _marker):
        areas = list_scroll_areas(
            hwnd=hwnd,
            max_elements=20,
            backend=backend,
            adapter=adapter,
        )
        area = next(item for item in areas["areas"] if item["vertical"]["scrollable"])
        element = str(area["element"])
        down = scroll_element(
            hwnd=hwnd,
            element=element,
            direction="down",
            to=None,
            percent=None,
            short_explanation=None,
            backend=backend,
            adapter=adapter,
        )
        assert down["after"]["vertical"]["percent"] > down["before"]["vertical"]["percent"]

        bottom = scroll_element(
            hwnd=hwnd,
            element=element,
            direction=None,
            to="bottom",
            percent=None,
            short_explanation=None,
            backend=backend,
            adapter=adapter,
        )
        assert bottom["after"]["vertical"]["moreAfter"] is False

        middle = scroll_element(
            hwnd=hwnd,
            element=element,
            direction=None,
            to=None,
            percent=50,
            short_explanation=None,
            backend=backend,
            adapter=adapter,
        )
        assert abs(middle["after"]["vertical"]["percent"] - 50) <= 1

        top = scroll_element(
            hwnd=hwnd,
            element=element,
            direction=None,
            to="top",
            percent=None,
            short_explanation=None,
            backend=backend,
            adapter=adapter,
        )
        assert top["after"]["vertical"]["moreBefore"] is False

    leftovers = [
        title
        for _pid, title in _visible_windows().values()
        if "agenttools-273-" in title.casefold()
    ]
    assert leftovers == []


def test_real_phase_a_window_state_and_element_reference_recovery(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(actions.STATE_DIR_ENV, str(tmp_path / "state"))
    backend = Win32Backend()
    adapter = WinAppAdapter(backend)

    with _classic_fixture() as (hwnd, _pid, _marker):
        maximized = set_window_state(
            hwnd=hwnd,
            state="maximized",
            short_explanation=None,
            backend=backend,
        )
        assert maximized["state_after"] == "maximized"

        resized = resize_window(
            hwnd=hwnd,
            x=100,
            y=100,
            width=520,
            height=360,
            restore_first=True,
            short_explanation=None,
            backend=backend,
        )
        assert backend.window_state(hwnd) == "normal"
        assert resized["postcondition_verified"] is True

        minimized = set_window_state(
            hwnd=hwnd,
            state="minimized",
            short_explanation=None,
            backend=backend,
        )
        assert minimized["state_after"] == "minimized"

        restored = set_window_state(
            hwnd=hwnd,
            state="normal",
            short_explanation=None,
            backend=backend,
        )
        assert restored["state_after"] == "normal"

    with _wpf_fixture() as (hwnd, _pid, _marker):
        store = ElementReferenceStore(tmp_path / "refs")
        inspection = inspect_semantic_window(
            hwnd=hwnd,
            depth=5,
            interactive=False,
            max_elements=80,
            backend=backend,
            winapp_adapter=adapter,
            element_ref_store=store,
        )
        replace_value = next(
            item
            for item in inspection["elements"]
            if item.get("automation_id") == "TaskReplaceValue"
        )
        replace_button = next(
            item
            for item in inspection["elements"]
            if item.get("automation_id") == "TaskReplaceButton"
        )
        assert replace_value["element_ref"].startswith("eref_")
        assert replace_button["element_ref"].startswith("eref_")

        replaced = invoke_element(
            hwnd=hwnd,
            element=None,
            element_ref=str(replace_button["element_ref"]),
            short_explanation=None,
            backend=backend,
            adapter=adapter,
            element_ref_store=store,
        )
        assert replaced["resolution_stage"] == "runtime_id_exact"

        recovered = set_element_value(
            hwnd=hwnd,
            element=None,
            element_ref=str(replace_value["element_ref"]),
            value="Task 285 stable reference proof",
            short_explanation=None,
            backend=backend,
            adapter=adapter,
            element_ref_store=store,
        )
        assert recovered["outcome"] == "postcondition_verified"
        assert recovered["resolution_stage"] in {
            "runtime_id_exact",
            "stable_metadata_unique",
        }
        assert recovered["fallback_used"] is False

        refreshed = inspect_semantic_window(
            hwnd=hwnd,
            depth=5,
            interactive=False,
            max_elements=80,
            backend=backend,
            winapp_adapter=adapter,
            element_ref_store=store,
        )
        read_target = next(
            item for item in refreshed["elements"] if item.get("automation_id") == "TaskValue"
        )
        reading = read_semantic_element(
            hwnd=hwnd,
            element=str(read_target["element"]),
            max_chars=100,
            backend=backend,
            winapp_adapter=adapter,
            element_ref_store=store,
        )
        assert reading["text"] == "Initial fixture value", reading
        assert reading["element_ref_status"] == "available"
        assert reading["element_ref"].startswith("eref_")


def test_real_generated_worker_reports_final_value_pattern_loss(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(actions.STATE_DIR_ENV, str(tmp_path / "state"))
    backend = Win32Backend()
    adapter = WinAppAdapter(backend)
    block_path = tmp_path / "value-pattern.blocked"

    with _wpf_fixture(vanishing_pattern_path=block_path) as (hwnd, _pid, _marker):
        inspection = inspect_window(
            hwnd=hwnd,
            depth=5,
            interactive=False,
            max_elements=80,
            backend=backend,
            adapter=adapter,
        )
        target = next(
            item
            for item in inspection["elements"]
            if item.get("automation_id") == "TaskVanishingValue"
        )
        identity = backend.capture_identity(hwnd)
        _binding, _element_id, properties = adapter._action_properties(
            identity,
            str(target["element"]),
        )
        assert properties["value_available"] is True
        focus_window(hwnd=hwnd, short_explanation=None, backend=backend)
        delivered = adapter._native_uia_action(
            identity,
            properties=properties,
            request={"action": "set-value", "value": "pre-block-probe"},
            method="uia.ValuePattern",
        )
        assert delivered["delivered"] is True

        block_path.write_text("pattern\n", encoding="ascii")
        with pytest.raises(ComputerError) as raised:
            adapter._native_uia_action(
                identity,
                properties=properties,
                request={"action": "set-value", "value": "must-not-deliver"},
                method="uia.ValuePattern",
            )

        assert raised.value.code == "set_value_pattern_unavailable"
        assert raised.value.details["outcome"] == "rejected"
        assert raised.value.details["method"] == "uia.ValuePattern"
        assert raised.value.details["fallback"] == {
            "kind": "guarded_physical_input",
            "status": "available",
            "reason": "requires_fresh_capture_and_explicit_allow_physical",
        }

        block_path.write_text("stale\n", encoding="ascii")
        removed = False
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                adapter._action_properties(identity, str(target["element"]))
            except ComputerError as exc:
                if exc.code in {"element_not_found", "stale_element"}:
                    removed = True
                    break
                raise
            time.sleep(0.025)
        assert removed, "The task-owned fixture did not remove its stale element"
        with pytest.raises(ComputerError) as stale:
            adapter._native_uia_action(
                identity,
                properties=properties,
                request={"action": "set-value", "value": "must-not-deliver"},
                method="uia.ValuePattern",
            )

        assert stale.value.code == "stale_element", (
            stale.value.code,
            stale.value.details,
        )
        assert "fallback" not in stale.value.details


def test_real_cross_process_mutex_rejects_without_blocking_read_only(
    monkeypatch, tmp_path: Path
) -> None:
    state = tmp_path / "mutex-state"
    monkeypatch.setenv(actions.STATE_DIR_ENV, str(state))
    environment = os.environ.copy()
    environment[actions.STATE_DIR_ENV] = str(state)
    environment[actions.OWNER_ENV] = "task-273-holder"
    environment[actions.CONVERSATION_ALIAS_ENV] = "fixture-terminal"
    child_code = (
        "import time\n"
        "from agent_tools.computer.actions import (NamedComputerMutationMutex,"
        "_clear_owner_metadata,_ensure_state_ready,_write_owner_metadata)\n"
        "_ensure_state_ready()\n"
        "mutex=NamedComputerMutationMutex()\n"
        "lease=mutex.try_acquire()\n"
        "assert lease.acquired\n"
        "print('LOCKED',flush=True)\n"
        "time.sleep(0.05)\n"
        "_write_owner_metadata('fixture-operation','fixture-hold')\n"
        "time.sleep(2.5)\n"
        "mutex.release(lease)\n"
        "_clear_owner_metadata('fixture-operation')\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code],
        cwd=REPO,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "LOCKED"

        started = time.monotonic()
        with pytest.raises(ComputerError) as busy:
            ActionCoordinator().run(
                operation="fixture-contender",
                target_ids={},
                short_explanation=None,
                perform=lambda _execution: pytest.fail("busy action was queued"),
            )
        assert time.monotonic() - started < 0.75
        assert busy.value.code == "computer_action_busy"
        assert busy.value.details["current_owner"]["owner"] == "task-273-holder"
        assert busy.value.details["current_owner"]["conversation_alias"] == "fixture-terminal"

        windows = Win32Backend().list_windows(max_items=5)
        assert windows["available"] is True
        assert child.poll() is None
    finally:
        stdout, stderr = child.communicate(timeout=6)
    assert child.returncode == 0, (stdout, stderr)


def test_real_abandoned_owner_record_is_pruned_on_next_lock_owner(
    monkeypatch, tmp_path: Path
) -> None:
    state = tmp_path / "abandoned-owner-state"
    monkeypatch.setenv(actions.STATE_DIR_ENV, str(state))
    environment = os.environ.copy()
    environment[actions.STATE_DIR_ENV] = str(state)
    environment[actions.OWNER_ENV] = "crashed-task-273-owner"
    child_code = (
        "import os\n"
        "from agent_tools.computer.actions import _ensure_state_ready,_write_owner_metadata\n"
        "_ensure_state_ready()\n"
        "_write_owner_metadata('crashed-operation','fixture-crash')\n"
        "os._exit(0)\n"
    )
    child = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=REPO,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        timeout=10,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert list(state.glob(f"{actions.OWNER_FILE}.*.json"))

    result = ActionCoordinator().run(
        operation="fixture-recovery",
        target_ids={},
        short_explanation=None,
        perform=lambda _execution: {
            "method": "fixture.noop",
            "outcome": "postcondition_verified",
        },
    )

    assert result["outcome"] == "postcondition_verified"
    assert list(state.glob(f"{actions.OWNER_FILE}.*.json")) == []


def test_real_orphan_worker_blocks_abandoned_mutex_recovery(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "orphan-worker-state"
    monkeypatch.setenv(actions.STATE_DIR_ENV, str(state))
    worker_code = (
        "import time\n"
        "from agent_tools.computer.actions import ("
        "NamedComputerMutationMutex,WORKER_MUTEX_NAME)\n"
        "mutex=NamedComputerMutationMutex(WORKER_MUTEX_NAME)\n"
        "lease=mutex.try_acquire()\n"
        "assert lease.acquired\n"
        "print('WORKER_LOCKED',flush=True)\n"
        "time.sleep(2)\n"
        "mutex.release(lease)\n"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert worker.stdout is not None
    assert worker.stdout.readline().strip() == "WORKER_LOCKED"

    crasher_code = (
        "import os\n"
        "from agent_tools.computer.actions import NamedComputerMutationMutex\n"
        "mutex=NamedComputerMutationMutex()\n"
        "lease=mutex.try_acquire()\n"
        "assert lease.acquired\n"
        "os._exit(0)\n"
    )
    crasher = subprocess.run(
        [sys.executable, "-c", crasher_code],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        timeout=10,
        check=False,
    )
    assert crasher.returncode == 0, crasher.stderr

    with pytest.raises(ComputerError) as blocked:
        ActionCoordinator().run(
            operation="must-not-overlap-orphan",
            target_ids={},
            short_explanation=None,
            perform=lambda _execution: pytest.fail("orphan worker overlap"),
        )
    assert blocked.value.code == "computer_action_worker_busy"

    stdout, stderr = worker.communicate(timeout=6)
    assert worker.returncode == 0, (stdout, stderr)
    recovered = ActionCoordinator().run(
        operation="post-orphan-recovery",
        target_ids={},
        short_explanation=None,
        perform=lambda _execution: {
            "method": "fixture.noop",
            "outcome": "postcondition_verified",
        },
    )
    assert recovered["outcome"] == "postcondition_verified"


def test_real_notification_worker_reports_ready_and_cleans_up() -> None:
    import win32gui

    worker = Path(actions.__file__).with_name("notification_worker.py")
    owner_started = Win32Backend().process_started(os.getpid())
    assert owner_started is not None
    process = subprocess.Popen(
        [sys.executable, "-P", str(worker)],
        cwd=worker.parent,
        env=actions._notification_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdin is not None
    assert process.stdout is not None
    payload = json.dumps(
        {
            "title": "AgentTools fixture",
            "message": "Task 273 disposable notification validation",
            "owner": {
                "pid": os.getpid(),
                "processStarted": owner_started,
            },
        }
    ).encode("ascii")
    process.stdin.write(payload)
    process.stdin.close()
    assert process.stdout.readline().replace(b"\r\n", b"\n") == b"READY\n"
    process.stdout.close()
    assert win32gui.FindWindow(None, "AgentToolsNotificationWorker")
    action_worker_mutex = actions.NamedComputerMutationMutex(actions.WORKER_MUTEX_NAME)
    action_worker_probe = action_worker_mutex.try_acquire()
    assert action_worker_probe.acquired is True
    action_worker_mutex.release(action_worker_probe)
    second = actions.show_notification(
        title="AgentTools fixture",
        message="Overlapping notification must be rejected",
    )
    assert second == {"requested": True, "status": "worker_busy"}
    assert process.wait(timeout=8) == 0
    assert not win32gui.FindWindow(None, "AgentToolsNotificationWorker")


def test_delayed_notification_worker_rejects_dead_owner_before_mutation() -> None:
    import ctypes

    import win32gui

    worker_path = Path(actions.__file__).with_name("notification_worker.py")
    owner = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    delayed_worker: subprocess.Popen[bytes] | None = None
    try:
        owner_started = Win32Backend().process_started(owner.pid)
        assert owner_started is not None
        delayed_worker = subprocess.Popen(
            [sys.executable, "-P", str(worker_path)],
            cwd=worker_path.parent,
            env=actions._notification_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_SUSPENDED", 0)
            ),
        )
        assert delayed_worker.stdin is not None
        delayed_worker.stdin.write(
            json.dumps(
                {
                    "title": "AgentTools delayed orphan fixture",
                    "message": "This notification must never be queued",
                    "owner": {
                        "pid": owner.pid,
                        "processStarted": owner_started,
                    },
                }
            ).encode("ascii")
        )
        delayed_worker.stdin.close()
        owner.terminate()
        owner.wait(timeout=5)
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)
        ntdll.NtResumeProcess.restype = ctypes.c_long
        assert ntdll.NtResumeProcess(int(delayed_worker._handle)) == 0
        assert delayed_worker.wait(timeout=5) == 8
        assert not win32gui.FindWindow(None, "AgentToolsNotificationWorker")
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if delayed_worker is not None and delayed_worker.poll() is None:
            delayed_worker.kill()
            delayed_worker.wait(timeout=5)


def test_delayed_native_uia_worker_rejects_dead_owner_before_target_access() -> None:
    import ctypes

    powershell = uia_winapp._trusted_powershell()
    assert powershell is not None
    owner = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    delayed_worker: subprocess.Popen[bytes] | None = None
    try:
        owner_started = Win32Backend().process_started(owner.pid)
        assert owner_started is not None
        encoded_script = uia_winapp._compressed_powershell_command(uia_winapp._native_uia_script(0))
        delayed_worker = subprocess.Popen(
            [
                str(powershell),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-STA",
                "-Command",
                encoded_script,
            ],
            cwd=powershell.parent,
            env=uia_winapp._sanitized_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_SUSPENDED", 0)
            ),
        )
        assert delayed_worker.stdin is not None
        delayed_worker.stdin.write(
            json.dumps(
                {
                    "action": "invoke",
                    "owner": {
                        "pid": owner.pid,
                        "processStarted": owner_started,
                    },
                }
            ).encode("ascii")
        )
        delayed_worker.stdin.close()
        owner.terminate()
        owner.wait(timeout=5)
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)
        ntdll.NtResumeProcess.restype = ctypes.c_long
        assert ntdll.NtResumeProcess(int(delayed_worker._handle)) == 0
        assert delayed_worker.stdout is not None
        output = delayed_worker.stdout.read()
        assert delayed_worker.wait(timeout=8) == 11
        assert json.loads(output)["error"] == "computer_action_owner_gone"
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if delayed_worker is not None and delayed_worker.poll() is None:
            delayed_worker.kill()
            delayed_worker.wait(timeout=5)


def test_real_notification_launcher_accepts_windows_ready_line() -> None:
    import win32gui

    worker_title = "AgentToolsNotificationWorker"
    deadline = time.monotonic() + 7.0
    while win32gui.FindWindow(None, worker_title) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not win32gui.FindWindow(None, worker_title)

    started = time.perf_counter()
    result = actions.show_notification(
        title="AgentTools fixture",
        message="Task 273 launcher validation",
    )
    elapsed = time.perf_counter() - started

    assert result == {"requested": True, "status": "queued"}
    assert elapsed < 1.5
    assert win32gui.FindWindow(None, worker_title)
    deadline = time.monotonic() + 7.0
    while win32gui.FindWindow(None, worker_title) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not win32gui.FindWindow(None, worker_title)


def test_real_notification_launcher_surfaces_emergency_disable(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "notification-state"
    state.mkdir()
    monkeypatch.setenv(actions.STATE_DIR_ENV, str(state))
    (state / actions.DISABLE_MARKER).write_text("disabled\n", encoding="ascii")

    result = actions.show_notification(
        title="AgentTools fixture",
        message="Task 273 disabled notification validation",
    )

    assert result == {"requested": True, "status": "disabled"}
