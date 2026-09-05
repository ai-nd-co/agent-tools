#!/usr/bin/env python
"""Standalone Windows control/capture POC CLI.

Feasibility spike for task 2026-08-29-266. Proves that native Python on
Windows can enumerate windows, inspect the foreground window, and -- against
a disposable self-created target only -- focus/restore, resize, capture, set
text, and invoke a button through direct Win32 APIs without moving the real
mouse cursor or touching any pre-existing window.

This is a standalone CLI modeled after the `$telegram` skill pattern: verb-
first subcommands, stable JSON on stdout for success, human-readable errors
on stderr with a non-zero exit code. It has no MCP server and no Codex/Claude
harness integration -- it is a local proof of primitives only.

Commands (subcommand form, or the `--verb` alias shown in parentheses):
    list-windows   (--list-windows)   List visible titled top-level windows.
    foreground     (--foreground)     Inspect the actual foreground window.
    self-test      (--self-test)      Spawn a disposable target window and
                                       exercise/verify every control primitive
                                       against it only.

Known limitations (see the worker report for the full matrix):
    - SetForegroundWindow is subject to Windows focus-stealing prevention;
      this POC uses a bounded AttachThreadInput technique and honestly
      records whether activation succeeded instead of assuming success.
    - Only classic Win32 (HWND + WM_* messages) controls are addressed here.
      Modern UWP/XAML surfaces and many Electron/Chromium apps expose little
      through this layer and would need UI Automation instead.
    - Windows Terminal and most console apps do not expose a reliable HWND
      text/control surface; a production tool should prefer the hosted
      process's own stdio/PTY API over window-message tricks for those.
    - No UAC/integrity-level elevation and no secure-desktop interaction are
      attempted; a lower-integrity caller cannot message a higher-integrity
      window (UIPI), and this script does not try to bypass that.
    - SendInput is intentionally never used; only PostMessage/SendMessage to
      specific control HWNDs. That means this cannot exercise UI that only
      responds to real hardware input (custom-drawn/owner-drawn controls
      with no message-level API).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
import uuid
from ctypes import wintypes
from pathlib import Path

if sys.platform != "win32":
    sys.stderr.write(
        json.dumps(
            {
                "error": "unsupported_platform",
                "message": "windows_control_poc.py requires Windows (sys.platform == 'win32').",
            }
        )
        + "\n"
    )
    sys.exit(1)

try:
    import win32api
    import win32con
    import win32gui
    import win32process
    import win32ui
except ImportError as exc:  # pragma: no cover - environment guard
    sys.stderr.write(
        json.dumps(
            {
                "error": "missing_dependency",
                "message": f"pywin32 is required but failed to import: {exc}",
            }
        )
        + "\n"
    )
    sys.exit(1)

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment guard
    sys.stderr.write(
        json.dumps(
            {
                "error": "missing_dependency",
                "message": f"Pillow is required but failed to import: {exc}",
            }
        )
        + "\n"
    )
    sys.exit(1)


USER32 = ctypes.WinDLL("user32", use_last_error=True)
GDI32 = ctypes.WinDLL("gdi32", use_last_error=True)
USER32.PrintWindow.argtypes = (wintypes.HWND, wintypes.HDC, wintypes.UINT)
USER32.PrintWindow.restype = wintypes.BOOL
USER32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
USER32.ReleaseDC.restype = ctypes.c_int
USER32.SetTimer.argtypes = (
    wintypes.HWND,
    ctypes.c_size_t,
    wintypes.UINT,
    wintypes.LPVOID,
)
USER32.SetTimer.restype = ctypes.c_size_t
GDI32.DeleteDC.argtypes = (wintypes.HDC,)
GDI32.DeleteDC.restype = wintypes.BOOL
GDI32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
GDI32.DeleteObject.restype = wintypes.BOOL


# PW_RENDERFULLCONTENT (0x2) was tried and rejected during validation: on this
# classic GDI target window it left the newly-resized area solid black even
# after a forced repaint. Plain PrintWindow (flag 0) captured correctly.
PW_CAPTURE_FLAGS = 0x00000000
TARGET_WINDOW_SIZE = (420, 300)
TARGET_RESIZE_TO = (560, 380)
EDIT_CONTROL_ID = 1001
BUTTON_CONTROL_ID = 1002
TITLE_SPINNER_TIMER_ID = 2971
TARGET_SAMPLE_TEXT = "agent-tools poc 2026-08-29"
CORE_SELF_TEST_REQUIRED_CHECKS = (
    "identity_verified",
    "stale_identity_rejected",
    "minimize_verified",
    "restore_verified",
    "foreground_activation_succeeded",
    "input_attachments_detached",
    "resize_verified",
    "full_capture_valid",
    "capture_gdi_cleanup_verified",
    "capture_gdi_cleanup_on_error_verified",
    "region_capture_valid",
    "text_set_verified",
    "button_click_verified",
    "cursor_unchanged",
    "cursor_unchanged_after_cleanup",
    "cursor_post_cleanup_sample_verified",
    "cleanup_verified",
    "identity_revalidated_before_actions",
)


def _fail(message: str, code: str = "error") -> None:
    sys.stderr.write(json.dumps({"error": code, "message": message}) + "\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Shared window metadata
# ---------------------------------------------------------------------------


def _window_state(hwnd: int) -> str:
    placement = win32gui.GetWindowPlacement(hwnd)
    show_cmd = placement[1]
    if show_cmd == win32con.SW_SHOWMINIMIZED:
        return "minimized"
    if show_cmd == win32con.SW_SHOWMAXIMIZED:
        return "maximized"
    return "normal"


def _window_info(hwnd: int) -> dict:
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
    return {
        "hwnd": int(hwnd),
        "pid": int(pid),
        "title": win32gui.GetWindowText(hwnd),
        "class_name": win32gui.GetClassName(hwnd),
        "visible": bool(win32gui.IsWindowVisible(hwnd)),
        "state": _window_state(hwnd),
        "rect": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": right - left,
            "height": bottom - top,
        },
    }


def _enum_visible_titled_top_level_windows() -> list[dict]:
    results: list[dict] = []

    def _callback(hwnd: int, _extra) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return True
        try:
            results.append(_window_info(hwnd))
        except win32gui.error:
            # Window closed between enumeration and inspection; skip it.
            pass
        return True

    win32gui.EnumWindows(_callback, None)
    return results


# ---------------------------------------------------------------------------
# list-windows / foreground commands
# ---------------------------------------------------------------------------


def cmd_list_windows(_args: argparse.Namespace) -> int:
    windows = _enum_visible_titled_top_level_windows()
    print(json.dumps(windows, indent=2))
    return 0


def cmd_foreground(_args: argparse.Namespace) -> int:
    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        print(json.dumps({"available": False, "hwnd": None}, indent=2))
        return 0
    info = _window_info(hwnd)
    info["available"] = True
    print(json.dumps(info, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Disposable target process (spawned by self-test, run via `_target-process`)
# ---------------------------------------------------------------------------


def _run_target_process(title: str, title_spinner_ms: int = 0) -> int:
    """Create a native Win32 window with an Edit + Button child control and
    pump its message loop until the controller closes it (WM_CLOSE)."""

    ctypes.windll.user32.SetProcessDPIAware()

    state = {
        "click_count": 0,
        "edit_hwnd": None,
        "button_hwnd": None,
        "title_tick": 0,
    }

    def _on_paint(hwnd, _msg, _wparam, _lparam):
        hdc, paint_struct = win32gui.BeginPaint(hwnd)
        rect = win32gui.GetClientRect(hwnd)
        brush = win32gui.GetStockObject(win32con.WHITE_BRUSH)
        win32gui.FillRect(hdc, rect, brush)
        red_brush = win32gui.CreateSolidBrush(win32api.RGB(200, 40, 40))
        marker_rect = (20, 20, rect[2] - 20, 70)
        win32gui.FillRect(hdc, marker_rect, red_brush)
        win32gui.DeleteObject(red_brush)
        win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
        win32gui.DrawText(
            hdc,
            "AgentTools POC target -- disposable window",
            -1,
            (24, 30, rect[2] - 24, 60),
            win32con.DT_LEFT | win32con.DT_SINGLELINE,
        )
        win32gui.EndPaint(hwnd, paint_struct)
        return 0

    def _on_command(hwnd, _msg, wparam, _lparam):
        control_id = wparam & 0xFFFF
        notify_code = (wparam >> 16) & 0xFFFF
        if control_id == BUTTON_CONTROL_ID and notify_code == win32con.BN_CLICKED:
            state["click_count"] += 1
            win32gui.SetWindowText(
                state["button_hwnd"], f"Click Me ({state['click_count']})"
            )
        return 0

    def _on_close(hwnd, _msg, _wparam, _lparam):
        win32gui.DestroyWindow(hwnd)
        return 0

    def _on_destroy(_hwnd, _msg, _wparam, _lparam):
        win32gui.PostQuitMessage(0)
        return 0

    def _on_timer(hwnd, _msg, wparam, _lparam):
        if int(wparam) == TITLE_SPINNER_TIMER_ID:
            state["title_tick"] += 1
            win32gui.SetWindowText(hwnd, f"{title} [{state['title_tick']}]")
        return 0

    message_map = {
        win32con.WM_PAINT: _on_paint,
        win32con.WM_COMMAND: _on_command,
        win32con.WM_CLOSE: _on_close,
        win32con.WM_DESTROY: _on_destroy,
        win32con.WM_TIMER: _on_timer,
    }

    wc = win32gui.WNDCLASS()
    wc.hInstance = win32api.GetModuleHandle(None)
    wc.lpszClassName = f"AgentToolsPocTargetClass-{uuid.uuid4().hex[:8]}"
    wc.hbrBackground = win32con.COLOR_WINDOW + 1
    wc.lpfnWndProc = message_map
    class_atom = win32gui.RegisterClass(wc)

    width, height = TARGET_WINDOW_SIZE
    hwnd = win32gui.CreateWindow(
        class_atom,
        title,
        win32con.WS_OVERLAPPEDWINDOW,
        100,
        100,
        width,
        height,
        0,
        0,
        wc.hInstance,
        None,
    )

    edit_hwnd = win32gui.CreateWindowEx(
        win32con.WS_EX_CLIENTEDGE,
        "EDIT",
        "",
        win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.ES_LEFT,
        24,
        90,
        width - 60,
        26,
        hwnd,
        EDIT_CONTROL_ID,
        wc.hInstance,
        None,
    )
    button_hwnd = win32gui.CreateWindowEx(
        0,
        "BUTTON",
        "Click Me (0)",
        win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.BS_PUSHBUTTON,
        24,
        140,
        160,
        30,
        hwnd,
        BUTTON_CONTROL_ID,
        wc.hInstance,
        None,
    )
    state["edit_hwnd"] = edit_hwnd
    state["button_hwnd"] = button_hwnd

    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.UpdateWindow(hwnd)
    if title_spinner_ms > 0:
        USER32.SetTimer(hwnd, TITLE_SPINNER_TIMER_ID, title_spinner_ms, None)

    win32gui.PumpMessages()
    return 0


# ---------------------------------------------------------------------------
# self-test controller
# ---------------------------------------------------------------------------


def _find_target_window(title: str, expected_pid: int, timeout_s: float = 8.0) -> int:
    """Locate the disposable target by exact title, then verify its PID
    matches the process we spawned. Fails closed on any ambiguity."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        matches: list[int] = []

        def _callback(hwnd: int, _extra, found: list[int] = matches) -> bool:
            if win32gui.GetWindowText(hwnd) == title:
                found.append(hwnd)
            return True

        win32gui.EnumWindows(_callback, None)

        if len(matches) > 1:
            raise RuntimeError(
                f"ambiguous target identity: {len(matches)} windows share title {title!r}"
            )
        if len(matches) == 1:
            hwnd = matches[0]
            _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid != expected_pid:
                raise RuntimeError(
                    f"identity mismatch: window titled {title!r} belongs to pid {pid}, "
                    f"expected spawned pid {expected_pid}"
                )
            return hwnd
        time.sleep(0.1)

    raise RuntimeError(f"target window titled {title!r} did not appear within {timeout_s}s")


def _assert_target_identity(hwnd: int, expected_title: str, expected_pid: int) -> None:
    """Fail closed unless HWND, PID, and globally unique exact title still match."""

    matches: list[int] = []

    def _callback(candidate_hwnd: int, _extra) -> bool:
        try:
            if win32gui.GetWindowText(candidate_hwnd) == expected_title:
                matches.append(candidate_hwnd)
        except win32gui.error:
            pass
        return True

    win32gui.EnumWindows(_callback, None)

    if matches != [hwnd]:
        raise RuntimeError(
            "target identity drift: expected the exact title to identify only HWND "
            f"{hwnd}, found {matches}"
        )
    if not win32gui.IsWindow(hwnd):
        raise RuntimeError(f"target identity drift: HWND {hwnd} is no longer valid")

    actual_title = win32gui.GetWindowText(hwnd)
    _tid, actual_pid = win32process.GetWindowThreadProcessId(hwnd)
    if actual_title != expected_title or actual_pid != expected_pid:
        raise RuntimeError(
            "target identity drift: exact HWND/title/PID tuple no longer matches "
            f"({hwnd}, {actual_title!r}, {actual_pid})"
        )


def _guarded_target_action(
    action_name: str,
    hwnd: int,
    expected_title: str,
    expected_pid: int,
    revalidation_log: list[str],
    action,
):
    """Revalidate immediately before an HWND action and record successful use."""

    _assert_target_identity(hwnd, expected_title, expected_pid)
    result = action()
    if action_name not in revalidation_log:
        revalidation_log.append(action_name)
    return result


def _poll_until(predicate, timeout_s: float = 3.0, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def _find_child_by_class(parent_hwnd: int, class_name: str) -> int | None:
    found: list[int] = []

    def _callback(hwnd: int, _extra) -> bool:
        if win32gui.GetClassName(hwnd) == class_name:
            found.append(hwnd)
        return True

    win32gui.EnumChildWindows(parent_hwnd, _callback, None)
    return found[0] if found else None


def _assert_child_identity(
    parent_hwnd: int,
    child_hwnd: int,
    expected_class: str,
    expected_control_id: int,
    expected_title: str,
    expected_pid: int,
) -> None:
    _assert_target_identity(parent_hwnd, expected_title, expected_pid)
    if not win32gui.IsWindow(child_hwnd) or not win32gui.IsChild(parent_hwnd, child_hwnd):
        raise RuntimeError(
            f"child identity drift: HWND {child_hwnd} is no longer a child of {parent_hwnd}"
        )
    _tid, actual_pid = win32process.GetWindowThreadProcessId(child_hwnd)
    actual_class = win32gui.GetClassName(child_hwnd)
    actual_control_id = win32gui.GetDlgCtrlID(child_hwnd)
    if (
        actual_pid != expected_pid
        or actual_class != expected_class
        or actual_control_id != expected_control_id
    ):
        raise RuntimeError(
            "child identity drift: HWND/class/control-id/PID tuple no longer matches "
            f"({child_hwnd}, {actual_class!r}, {actual_control_id}, {actual_pid})"
        )


def _guarded_child_action(
    action_name: str,
    parent_hwnd: int,
    child_hwnd: int,
    expected_class: str,
    expected_control_id: int,
    expected_title: str,
    expected_pid: int,
    revalidation_log: list[str],
    action,
):
    _assert_child_identity(
        parent_hwnd,
        child_hwnd,
        expected_class,
        expected_control_id,
        expected_title,
        expected_pid,
    )
    result = action()
    if action_name not in revalidation_log:
        revalidation_log.append(action_name)
    return result


def _try_set_foreground(
    hwnd: int,
    expected_title: str,
    expected_pid: int,
    revalidation_log: list[str],
    attachment_details: dict,
) -> bool:
    """Bounded activation attempt using AttachThreadInput. Never uses
    SendInput and never moves the cursor. Returns whether it actually
    worked, verified by reading GetForegroundWindow() afterward."""

    current_tid = win32api.GetCurrentThreadId()
    fg_hwnd = win32gui.GetForegroundWindow()
    fg_tid = win32process.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else 0
    target_tid = _guarded_target_action(
        "foreground_target_thread",
        hwnd,
        expected_title,
        expected_pid,
        revalidation_log,
        lambda: win32process.GetWindowThreadProcessId(hwnd)[0],
    )

    thread_ids_to_attach: list[int] = []
    for thread_id in (fg_tid, target_tid):
        if thread_id and thread_id != current_tid and thread_id not in thread_ids_to_attach:
            thread_ids_to_attach.append(thread_id)

    attached_thread_ids: list[int] = []
    detached_thread_ids: list[int] = []
    detach_errors: list[str] = []
    try:
        for thread_id in thread_ids_to_attach:
            # pywin32 returns None on success and raises pywintypes.error on
            # failure, so successful completion -- not truthiness -- is the
            # condition that obligates a matching detach.
            if thread_id == target_tid:
                _guarded_target_action(
                    "foreground_attach_target",
                    hwnd,
                    expected_title,
                    expected_pid,
                    revalidation_log,
                    lambda target_tid=thread_id: win32process.AttachThreadInput(
                        current_tid, target_tid, True
                    ),
                )
            else:
                win32process.AttachThreadInput(current_tid, thread_id, True)
            attached_thread_ids.append(thread_id)

        _guarded_target_action(
            "foreground_show",
            hwnd,
            expected_title,
            expected_pid,
            revalidation_log,
            lambda: win32gui.ShowWindow(hwnd, win32con.SW_SHOW),
        )
        try:
            _guarded_target_action(
                "foreground_set",
                hwnd,
                expected_title,
                expected_pid,
                revalidation_log,
                lambda: win32gui.SetForegroundWindow(hwnd),
            )
        except win32gui.error:
            pass
    finally:
        for thread_id in reversed(attached_thread_ids):
            try:
                win32process.AttachThreadInput(current_tid, thread_id, False)
                detached_thread_ids.append(thread_id)
            except win32process.error as exc:
                detach_errors.append(f"thread {thread_id}: {exc}")

        attachment_details.update(
            {
                "attached_thread_ids": attached_thread_ids,
                "detached_thread_ids": detached_thread_ids,
                "detach_errors": detach_errors,
            }
        )
        if detach_errors:
            raise RuntimeError(
                "failed to detach one or more successfully attached input threads: "
                + "; ".join(detach_errors)
            )

    return win32gui.GetForegroundWindow() == hwnd


def _capture_window(
    hwnd: int,
    expected_title: str,
    expected_pid: int,
    revalidation_log: list[str],
    capture_details: dict,
    force_failure_after_selection: bool = False,
) -> tuple[Image.Image, bool, int, int]:
    # Force a fresh paint before capture rather than assuming the message
    # loop already caught up with a prior resize/state change.
    _guarded_target_action(
        "capture_invalidate",
        hwnd,
        expected_title,
        expected_pid,
        revalidation_log,
        lambda: win32gui.InvalidateRect(hwnd, None, True),
    )
    _guarded_target_action(
        "capture_update",
        hwnd,
        expected_title,
        expected_pid,
        revalidation_log,
        lambda: win32gui.UpdateWindow(hwnd),
    )

    left, top, right, bottom = _guarded_target_action(
        "capture_get_rect",
        hwnd,
        expected_title,
        expected_pid,
        revalidation_log,
        lambda: win32gui.GetWindowRect(hwnd),
    )
    width, height = right - left, bottom - top

    cleanup_steps: list[tuple[str, object]] = []
    cleanup_completed: list[str] = []
    cleanup_errors: list[str] = []
    bitmap_owner_detached = False
    try:
        hwnd_dc = _guarded_target_action(
            "capture_get_dc",
            hwnd,
            expected_title,
            expected_pid,
            revalidation_log,
            lambda: win32gui.GetWindowDC(hwnd),
        )
        def _release_window_dc() -> None:
            if not USER32.ReleaseDC(int(hwnd), int(hwnd_dc)):
                raise RuntimeError("ReleaseDC returned failure")

        cleanup_steps.append(("window_dc_released", _release_window_dc))
        compatible_dc = win32gui.CreateCompatibleDC(hwnd_dc)

        def _delete_compatible_dc() -> None:
            if not GDI32.DeleteDC(int(compatible_dc)):
                raise ctypes.WinError()

        cleanup_steps.append(("compatible_dc_deleted", _delete_compatible_dc))
        bitmap_owner = win32gui.CreateCompatibleBitmap(hwnd_dc, width, height)
        bitmap_handle = bitmap_owner.Detach()
        bitmap_owner_detached = True

        def _delete_bitmap() -> None:
            if not GDI32.DeleteObject(int(bitmap_handle)):
                raise ctypes.WinError()

        cleanup_steps.append(("bitmap_deleted", _delete_bitmap))
        previous_bitmap = win32gui.SelectObject(compatible_dc, bitmap_handle)
        cleanup_steps.append(
            (
                "previous_bitmap_restored",
                lambda: win32gui.SelectObject(compatible_dc, previous_bitmap),
            )
        )
        if force_failure_after_selection:
            raise RuntimeError("forced capture failure after bitmap selection")

        result = _guarded_target_action(
            "capture_print_window",
            hwnd,
            expected_title,
            expected_pid,
            revalidation_log,
            lambda: USER32.PrintWindow(
                int(hwnd), int(compatible_dc), PW_CAPTURE_FLAGS
            ),
        )

        save_bitmap = win32ui.CreateBitmapFromHandle(bitmap_handle)
        bmp_info = save_bitmap.GetInfo()
        bmp_bits = save_bitmap.GetBitmapBits(True)
        image = Image.frombuffer(
            "RGB",
            (bmp_info["bmWidth"], bmp_info["bmHeight"]),
            bmp_bits,
            "raw",
            "BGRX",
            0,
            1,
        )
    finally:
        for label, cleanup in reversed(cleanup_steps):
            try:
                cleanup()
                cleanup_completed.append(label)
            except Exception as exc:  # noqa: BLE001 - report every cleanup failure
                cleanup_errors.append(f"{label}: {exc}")
        capture_details.update(
            {
                "gdi_cleanup_completed": cleanup_completed,
                "gdi_cleanup_errors": cleanup_errors,
                "bitmap_owner_detached": bitmap_owner_detached,
            }
        )
        if cleanup_errors:
            raise RuntimeError(
                "capture GDI cleanup failed: " + "; ".join(cleanup_errors)
            )

    return image, bool(result), bmp_info["bmWidth"], bmp_info["bmHeight"]


def _image_non_uniform(image: Image.Image) -> bool:
    extrema = image.convert("L").getextrema()
    return extrema[0] != extrema[1]


def _core_self_test_succeeds(checks: dict, errors: list[str]) -> bool:
    return not errors and all(
        checks.get(name) is True for name in CORE_SELF_TEST_REQUIRED_CHECKS
    )


def cmd_self_test(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    title = f"AgentToolsPOCTarget-{uuid.uuid4().hex}"
    checks: dict = {}
    warnings: list[str] = []
    errors: list[str] = []
    artifacts: dict = {}
    target_info: dict = {}
    hwnd = None
    proc = None
    cursor_before = None
    cursor_after_cleanup = None
    identity_revalidation_actions: list[str] = []
    attachment_details: dict = {}
    capture_details: dict = {}
    failed_capture_details: dict = {}
    required_identity_actions = {
        "minimize",
        "minimize_verify",
        "restore",
        "restore_verify",
        "foreground_show",
        "foreground_set",
        "foreground_target_thread",
        "foreground_attach_target",
        "resize_read_before",
        "resize",
        "resize_read_after",
        "capture_invalidate",
        "capture_update",
        "capture_get_rect",
        "capture_get_dc",
        "capture_print_window",
        "edit_find",
        "edit_set_text",
        "edit_get_text_length",
        "edit_get_text",
        "button_find",
        "button_get_text_before",
        "button_click",
        "button_poll_text",
        "button_get_text_after",
        "cleanup_close",
    }

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_target-process",
                "--title",
                title,
            ]
        )
        cursor_before = win32api.GetCursorPos()
        hwnd = _find_target_window(title, proc.pid, timeout_s=8.0)
        _assert_target_identity(hwnd, title, proc.pid)
        checks["identity_verified"] = True
        target_info = _window_info(hwnd)
        target_info["spawned_pid"] = proc.pid

        identity_rejection_results: dict[str, bool] = {}
        for label, probe_hwnd, probe_title, probe_pid in (
            ("invalid_hwnd", 0, title, proc.pid),
            ("wrong_title", hwnd, f"{title}-stale", proc.pid),
            ("wrong_pid", hwnd, title, proc.pid + 1),
        ):
            try:
                _assert_target_identity(probe_hwnd, probe_title, probe_pid)
            except RuntimeError:
                identity_rejection_results[label] = True
            else:
                identity_rejection_results[label] = False
        checks["stale_identity_rejected"] = all(identity_rejection_results.values())
        artifacts["identity_rejection_results"] = identity_rejection_results

        # --- minimize / restore, verified via IsIconic reads ---------------
        _guarded_target_action(
            "minimize",
            hwnd,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE),
        )
        checks["minimize_verified"] = _poll_until(
            lambda: _guarded_target_action(
                "minimize_verify",
                hwnd,
                title,
                proc.pid,
                identity_revalidation_actions,
                lambda: win32gui.IsIconic(hwnd),
            )
        )

        _guarded_target_action(
            "restore",
            hwnd,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: win32gui.ShowWindow(hwnd, win32con.SW_RESTORE),
        )
        checks["restore_verified"] = _poll_until(
            lambda: _guarded_target_action(
                "restore_verify",
                hwnd,
                title,
                proc.pid,
                identity_revalidation_actions,
                lambda: (not win32gui.IsIconic(hwnd))
                and win32gui.IsWindowVisible(hwnd),
            )
        )

        # --- bounded foreground activation, honestly recorded --------------
        checks["foreground_activation_attempted"] = True
        checks["foreground_activation_succeeded"] = _try_set_foreground(
            hwnd,
            title,
            proc.pid,
            identity_revalidation_actions,
            attachment_details,
        )
        checks["input_attachments_detached"] = bool(
            attachment_details.get("attached_thread_ids")
            and sorted(attachment_details["attached_thread_ids"])
            == sorted(attachment_details.get("detached_thread_ids", []))
            and not attachment_details.get("detach_errors")
        )
        artifacts["foreground_input_attachments"] = attachment_details
        if not checks["foreground_activation_succeeded"]:
            warnings.append(
                "SetForegroundWindow did not take effect; Windows focus-stealing "
                "prevention applies even with AttachThreadInput in some session states; "
                "the self-test therefore fails."
            )

        # --- resize, verified via GetWindowRect post-condition read --------
        new_w, new_h = TARGET_RESIZE_TO
        rect_before_resize = _guarded_target_action(
            "resize_read_before",
            hwnd,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: win32gui.GetWindowRect(hwnd),
        )
        _guarded_target_action(
            "resize",
            hwnd,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: win32gui.MoveWindow(
                hwnd, rect_before_resize[0], rect_before_resize[1], new_w, new_h, True
            ),
        )
        rect_after = _guarded_target_action(
            "resize_read_after",
            hwnd,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: win32gui.GetWindowRect(hwnd),
        )
        actual_w = rect_after[2] - rect_after[0]
        actual_h = rect_after[3] - rect_after[1]
        checks["resize_verified"] = actual_w == new_w and actual_h == new_h
        checks["resize_expected"] = {"width": new_w, "height": new_h}
        checks["resize_actual"] = {"width": actual_w, "height": actual_h}

        # --- full window capture -------------------------------------------
        image, print_window_ok, cap_w, cap_h = _capture_window(
            hwnd,
            title,
            proc.pid,
            identity_revalidation_actions,
            capture_details,
        )
        expected_gdi_cleanup = {
            "previous_bitmap_restored",
            "bitmap_deleted",
            "compatible_dc_deleted",
            "window_dc_released",
        }
        checks["capture_gdi_cleanup_verified"] = bool(
            set(capture_details.get("gdi_cleanup_completed", []))
            == expected_gdi_cleanup
            and capture_details.get("bitmap_owner_detached") is True
            and not capture_details.get("gdi_cleanup_errors")
        )
        artifacts["capture_gdi_cleanup"] = capture_details
        forced_capture_failure_seen = False
        try:
            _capture_window(
                hwnd,
                title,
                proc.pid,
                identity_revalidation_actions,
                failed_capture_details,
                force_failure_after_selection=True,
            )
        except RuntimeError as exc:
            forced_capture_failure_seen = (
                str(exc) == "forced capture failure after bitmap selection"
            )
        checks["capture_gdi_cleanup_on_error_verified"] = bool(
            forced_capture_failure_seen
            and set(failed_capture_details.get("gdi_cleanup_completed", []))
            == expected_gdi_cleanup
            and failed_capture_details.get("bitmap_owner_detached") is True
            and not failed_capture_details.get("gdi_cleanup_errors")
        )
        artifacts["capture_gdi_cleanup_on_error"] = failed_capture_details
        full_path = output_dir / "full_window.png"
        image.save(full_path)
        full_non_uniform = _image_non_uniform(image)
        checks["full_capture_valid"] = bool(
            print_window_ok and cap_w > 0 and cap_h > 0 and full_non_uniform
        )
        artifacts["full_window_png"] = str(full_path)
        artifacts["full_window_size"] = [cap_w, cap_h]

        # --- explicit sub-region capture ------------------------------------
        region_box = (10, 10, min(210, cap_w), min(160, cap_h))
        region_image = image.crop(region_box)
        region_path = output_dir / "region.png"
        region_image.save(region_path)
        region_non_uniform = _image_non_uniform(region_image)
        checks["region_capture_valid"] = bool(
            region_image.width == (region_box[2] - region_box[0])
            and region_image.height == (region_box[3] - region_box[1])
            and region_non_uniform
        )
        artifacts["region_png"] = str(region_path)
        artifacts["region_size"] = [region_image.width, region_image.height]
        artifacts["region_rect"] = list(region_box)

        # --- direct text set + readback -------------------------------------
        # win32gui.GetWindowText()'s buffer marshaling was found unreliable
        # across the process boundary during validation (WM_GETTEXTLENGTH
        # confirmed the set succeeded while GetWindowText read back empty),
        # so text is both set and read back via explicit ctypes SendMessageW
        # calls with our own buffer, which round-trip correctly cross-process.
        edit_hwnd = _guarded_target_action(
            "edit_find",
            hwnd,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: _find_child_by_class(hwnd, "Edit"),
        )
        if edit_hwnd is None:
            raise RuntimeError("could not locate Edit child control on disposable target")
        user32 = ctypes.windll.user32
        _guarded_child_action(
            "edit_set_text",
            hwnd,
            edit_hwnd,
            "Edit",
            EDIT_CONTROL_ID,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: user32.SendMessageW(
                edit_hwnd, win32con.WM_SETTEXT, 0, TARGET_SAMPLE_TEXT
            ),
        )
        text_length = _guarded_child_action(
            "edit_get_text_length",
            hwnd,
            edit_hwnd,
            "Edit",
            EDIT_CONTROL_ID,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: user32.SendMessageW(
                edit_hwnd, win32con.WM_GETTEXTLENGTH, 0, 0
            ),
        )
        buf = ctypes.create_unicode_buffer(text_length + 1)
        _guarded_child_action(
            "edit_get_text",
            hwnd,
            edit_hwnd,
            "Edit",
            EDIT_CONTROL_ID,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: user32.SendMessageW(
                edit_hwnd, win32con.WM_GETTEXT, text_length + 1, buf
            ),
        )
        readback = buf.value
        checks["text_set_verified"] = readback == TARGET_SAMPLE_TEXT
        artifacts["edit_text_expected"] = TARGET_SAMPLE_TEXT
        artifacts["edit_text_readback"] = readback

        # --- direct button click + target-state verification ---------------
        button_hwnd = _guarded_target_action(
            "button_find",
            hwnd,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: _find_child_by_class(hwnd, "Button"),
        )
        if button_hwnd is None:
            raise RuntimeError("could not locate Button child control on disposable target")
        text_before_click = _guarded_child_action(
            "button_get_text_before",
            hwnd,
            button_hwnd,
            "Button",
            BUTTON_CONTROL_ID,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: win32gui.GetWindowText(button_hwnd),
        )
        _guarded_child_action(
            "button_click",
            hwnd,
            button_hwnd,
            "Button",
            BUTTON_CONTROL_ID,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: win32gui.SendMessage(button_hwnd, win32con.BM_CLICK, 0, 0),
        )
        checks["button_click_verified"] = _poll_until(
            lambda: _guarded_child_action(
                "button_poll_text",
                hwnd,
                button_hwnd,
                "Button",
                BUTTON_CONTROL_ID,
                title,
                proc.pid,
                identity_revalidation_actions,
                lambda: win32gui.GetWindowText(button_hwnd),
            )
            == "Click Me (1)"
        )
        artifacts["button_text_before"] = text_before_click
        artifacts["button_text_after"] = _guarded_child_action(
            "button_get_text_after",
            hwnd,
            button_hwnd,
            "Button",
            BUTTON_CONTROL_ID,
            title,
            proc.pid,
            identity_revalidation_actions,
            lambda: win32gui.GetWindowText(button_hwnd),
        )

        # --- cursor unchanged -------------------------------------------------
        cursor_after = win32api.GetCursorPos()
        checks["cursor_unchanged"] = cursor_before == cursor_after

    except Exception as exc:  # noqa: BLE001 - collected into JSON, not swallowed
        errors.append(str(exc))
    finally:
        cleanup_ok = True
        if hwnd is not None and proc is not None:
            try:
                _guarded_target_action(
                    "cleanup_close",
                    hwnd,
                    title,
                    proc.pid,
                    identity_revalidation_actions,
                    lambda: win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0),
                )
            except (RuntimeError, win32gui.error) as exc:
                errors.append(f"refused cleanup through HWND after identity check: {exc}")
                cleanup_ok = False
        if proc is not None:
            if hwnd is None and proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

        window_gone = True
        if hwnd is not None:
            window_gone = not win32gui.IsWindow(hwnd)
        process_gone = proc is None or proc.poll() is not None
        try:
            cursor_after_cleanup = win32api.GetCursorPos()
        except Exception as exc:  # noqa: BLE001 - telemetry must not skip cleanup
            errors.append(f"post-cleanup cursor sampling failed: {exc}")
        checks["cleanup_verified"] = bool(cleanup_ok and window_gone and process_gone)
        checks["cursor_unchanged_after_cleanup"] = bool(
            cursor_before is not None
            and cursor_after_cleanup is not None
            and cursor_before == cursor_after_cleanup
        )
        checks["cursor_post_cleanup_sample_verified"] = bool(
            window_gone and process_gone and cursor_after_cleanup is not None
        )
        checks["identity_revalidation_actions"] = identity_revalidation_actions
        checks["identity_revalidated_before_actions"] = (
            set(identity_revalidation_actions) == required_identity_actions
        )

    forced_focus_failure = {
        name: True for name in CORE_SELF_TEST_REQUIRED_CHECKS
    }
    forced_focus_failure["foreground_activation_succeeded"] = False
    checks["foreground_failure_rejected"] = not _core_self_test_succeeds(
        forced_focus_failure, []
    )
    success = _core_self_test_succeeds(checks, errors) and checks.get(
        "foreground_failure_rejected"
    ) is True

    result = {
        "command": "self-test",
        "success": success,
        "target": target_info,
        "checks": checks,
        "artifacts": artifacts,
        "cursor": {
            "before": list(cursor_before) if cursor_before is not None else None,
            "after": (
                list(cursor_after_cleanup)
                if cursor_after_cleanup is not None
                else None
            ),
        },
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if success else 1


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="windows_control_poc.py",
        description=(
            "Bounded Windows-control CLI POC: enumerate windows, inspect the "
            "foreground window, and self-test control primitives against a "
            "disposable target only. Standalone CLI, no MCP server, no "
            "Codex/Claude harness integration."
        ),
        epilog=(
            "Limitations: classic Win32 HWND/message control only (no UI "
            "Automation); SetForegroundWindow is subject to focus-stealing "
            "prevention and outcome is reported, not guaranteed; Windows "
            "Terminal/console and modern UWP/XAML surfaces are largely out of "
            "reach of this layer; no UAC/integrity elevation or secure-desktop "
            "interaction; SendInput is never used, so hardware-input-only "
            "controls cannot be exercised."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser(
        "list-windows", help="List visible titled top-level windows as a JSON array."
    )
    p_list.set_defaults(func=cmd_list_windows)

    p_fg = subparsers.add_parser(
        "foreground", help="Inspect the actual foreground window as a JSON object."
    )
    p_fg.set_defaults(func=cmd_foreground)

    p_self_test = subparsers.add_parser(
        "self-test",
        help=(
            "Spawn a disposable target window and exercise/verify focus, "
            "resize, capture, text-set, and click primitives against it only."
        ),
    )
    p_self_test.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write self-test evidence (JSON is also printed to stdout).",
    )
    p_self_test.set_defaults(func=cmd_self_test)

    p_target = subparsers.add_parser("_target-process", help=argparse.SUPPRESS)
    p_target.add_argument("--title", required=True)
    p_target.add_argument("--title-spinner-ms", type=int, default=0)
    p_target.set_defaults(
        func=lambda a: _run_target_process(a.title, a.title_spinner_ms)
    )

    return parser


_ALIASES = {
    "--list-windows": "list-windows",
    "--foreground": "foreground",
    "--self-test": "self-test",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _ALIASES:
        argv[0] = _ALIASES[argv[0]]

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - surfaced as an actionable CLI error
        _fail(f"{type(exc).__name__}: {exc}")
        return 1  # unreachable, _fail exits


if __name__ == "__main__":
    sys.exit(main())
