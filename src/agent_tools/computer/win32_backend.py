from __future__ import annotations

import ctypes
import importlib
import platform
import socket
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from ctypes import wintypes
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from agent_tools.computer.models import (
    MAX_CLASS_NAME_CHARS,
    MAX_PROCESS_NAME_CHARS,
    MAX_TITLE_CHARS,
    Bounds,
    ComputerError,
    NativeCaptureGeometry,
    WindowIdentity,
    available_section,
)

MAX_VISIBLE_WINDOWS = 100
MAX_BACKGROUND_PROCESSES = 250
TH32CS_SNAPPROCESS = 0x00000002
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_ELEVATION = 20
TOKEN_INTEGRITY_LEVEL = 25
UOI_NAME = 2
DESKTOP_READOBJECTS = 0x0001
GA_ROOT = 2
DWMWA_CLOAKED = 14
SW_RESTORE = 9
SW_SHOW = 5
SW_MINIMIZE = 6
SW_MAXIMIZE = 3
MONITOR_DEFAULTTONEAREST = 2
DPI_AWARENESS_CONTEXT_UNAWARE = -1
DPI_AWARENESS_CONTEXT_SYSTEM_AWARE = -2
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE = -3
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
DPI_AWARENESS_CONTEXT_UNAWARE_GDISCALED = -5
INTEGRITY_ORDER = {
    "untrusted": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "system": 4,
    "protected": 5,
}
PROTECTED_PROCESS_NAMES = {
    "consent.exe",
    "credentialuibroker.exe",
    "lockapp.exe",
    "logonui.exe",
    "winlogon.exe",
}
LOCK_UI_PROCESS_NAMES = {"lockapp.exe", "logonui.exe"}


@dataclass(frozen=True)
class Win32Modules:
    win32api: Any
    win32con: Any
    win32gui: Any
    win32process: Any
    win32ui: Any


class ProcessEntry32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class FileTime(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", wintypes.DWORD),
        ("dwHighDateTime", wintypes.DWORD),
    ]


class SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", wintypes.DWORD),
        ("BatteryFullLifeTime", wintypes.DWORD),
    ]


class Point(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class Rect(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", Rect),
        ("rcWork", Rect),
        ("dwFlags", wintypes.DWORD),
    ]


class SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]


class TokenMandatoryLabel(ctypes.Structure):
    _fields_ = [("Label", SidAndAttributes)]


def _require_windows() -> None:
    if sys.platform != "win32":
        raise ComputerError(
            "unsupported_platform",
            "Computer commands require Windows.",
            exit_code=2,
        )


@lru_cache(maxsize=1)
def load_win32_modules() -> Win32Modules:
    _require_windows()
    try:
        return Win32Modules(
            win32api=importlib.import_module("win32api"),
            win32con=importlib.import_module("win32con"),
            win32gui=importlib.import_module("win32gui"),
            win32process=importlib.import_module("win32process"),
            win32ui=importlib.import_module("win32ui"),
        )
    except ImportError as exc:
        raise ComputerError(
            "win32_backend_unavailable",
            "The Win32 backend is not installed; install the computer extra.",
            exit_code=2,
        ) from exc


def _windll(name: str) -> Any:
    _require_windows()
    return ctypes.WinDLL(name, use_last_error=True)


def _filetime_value(value: FileTime) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def _classify_lock_state(
    desktop_name: str,
    lock_ui_foreground: bool | None,
) -> str:
    if desktop_name.casefold() != "default" or lock_ui_foreground is True:
        return "locked"
    if lock_ui_foreground is None:
        return "unknown"
    return "unlocked"


def _is_pywin32_error(error: BaseException) -> bool:
    try:
        pywintypes = importlib.import_module("pywintypes")
    except ImportError:
        return False
    return isinstance(error, pywintypes.error)


def _post_delivery_error(error: ComputerError, method: str) -> ComputerError:
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details={**error.details, "outcome": "delivery_only", "method": method},
    )


def _rectangle_covered_by_monitors(
    requested: Bounds, monitors: list[Bounds]
) -> bool:
    if requested.width <= 0 or requested.height <= 0:
        return False
    left = requested.x
    right = requested.x + requested.width
    top = requested.y
    bottom = requested.y + requested.height
    x_edges = {left, right}
    for monitor in monitors:
        overlap_left = max(left, monitor.x)
        overlap_right = min(right, monitor.x + monitor.width)
        if overlap_left < overlap_right:
            x_edges.update((overlap_left, overlap_right))
    sorted_edges = sorted(x_edges)
    for slab_left, slab_right in zip(sorted_edges, sorted_edges[1:], strict=False):
        if slab_left >= slab_right:
            continue
        intervals = sorted(
            (
                max(top, monitor.y),
                min(bottom, monitor.y + monitor.height),
            )
            for monitor in monitors
            if monitor.x <= slab_left
            and monitor.x + monitor.width >= slab_right
            and monitor.y < bottom
            and monitor.y + monitor.height > top
        )
        covered_until = top
        for interval_top, interval_bottom in intervals:
            if interval_top > covered_until:
                break
            covered_until = max(covered_until, interval_bottom)
            if covered_until >= bottom:
                break
        if covered_until < bottom:
            return False
    return True


def _active_monitor_rectangles(modules: Win32Modules) -> list[Bounds]:
    try:
        monitors = [
            Bounds.from_ltrb(*map(int, monitor[2]))
            for monitor in modules.win32api.EnumDisplayMonitors()
        ]
    except Exception as exc:
        raise ComputerError(
            "display_layout_unavailable",
            "The active monitor layout could not be verified.",
        ) from exc
    if not monitors:
        raise ComputerError(
            "display_layout_unavailable",
            "The active monitor layout could not be verified.",
        )
    return monitors


def _post_restore_resize_error(
    backend: Win32Backend,
    error: ComputerError,
    *,
    expected: WindowIdentity,
    require_title_match: bool,
    requested_bounds: Bounds,
) -> ComputerError:
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details={
            **error.details,
            "outcome": "delivery_only",
            "method": "win32.ShowWindow",
            "restore_method": "win32.ShowWindow(SW_RESTORE)",
            "restore_performed": True,
            "restore_first": True,
            "restored_first": True,
            "state_after": _safe_window_state(
                backend,
                expected,
                require_title_match,
            ),
            "requested_bounds": requested_bounds.to_jsonable(),
            "actual_bounds": _safe_window_bounds(
                backend,
                expected,
                require_title_match,
            ),
            "postcondition_verified": False,
        },
    )


def _nested_restore_resize_error(
    backend: Win32Backend,
    error: ComputerError,
    *,
    expected: WindowIdentity,
    require_title_match: bool,
    requested_bounds: Bounds,
    restore_delivery_attempted: bool,
) -> ComputerError:
    state_after = _safe_window_state(
        backend,
        expected,
        require_title_match,
    )
    error_outcome = error.details.get("outcome")
    outcome = (
        str(error_outcome or "delivery_only")
        if restore_delivery_attempted
        else str(error_outcome)
        if error_outcome in {"rejected", "failed"}
        else "rejected"
    )
    restore_verified = restore_delivery_attempted and state_after == "normal"
    restore_result: bool | None = (
        True
        if restore_verified
        else None
        if restore_delivery_attempted
        else False
    )
    method = error.details.get("method")
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details={
            **error.details,
            "outcome": outcome,
            "method": (
                str(method)
                if method is not None
                else "win32.ShowWindow"
                if restore_delivery_attempted
                else None
            ),
            "requested_bounds": requested_bounds.to_jsonable(),
            "actual_bounds": _safe_window_bounds(
                backend,
                expected,
                require_title_match,
            ),
            "restore_first": True,
            "restored_first": restore_result,
            "restore_performed": restore_result,
            "restore_method": (
                "win32.ShowWindow(SW_RESTORE)"
                if restore_delivery_attempted
                else None
            ),
            "restore_status": (
                "verified"
                if restore_verified
                else "unknown"
                if restore_delivery_attempted
                else "not_attempted"
            ),
            "state_after": state_after,
            "postcondition_verified": False,
        },
    )


def _identity_bound_window_state(
    backend: Win32Backend,
    expected: WindowIdentity,
    require_title_match: bool,
) -> tuple[WindowIdentity, str]:
    backend._revalidate_action_target(expected, require_title_match)
    state = backend.window_state(expected.hwnd)
    actual = backend._revalidate_action_target(expected, require_title_match)
    return actual, state


def _safe_window_state(
    backend: Win32Backend,
    expected: WindowIdentity,
    require_title_match: bool,
) -> str:
    try:
        _actual, state = _identity_bound_window_state(
            backend,
            expected,
            require_title_match,
        )
        return state
    except Exception:
        return "unavailable"


def _safe_window_bounds(
    backend: Win32Backend,
    expected: WindowIdentity,
    require_title_match: bool,
) -> dict[str, int] | None:
    try:
        backend._revalidate_action_target(expected, require_title_match)
        actual = backend._revalidate_action_target(expected, require_title_match)
    except Exception:
        return None
    return actual.bounds.to_jsonable()


def _safe_foreground_hwnd(modules: Win32Modules) -> int | None:
    try:
        return int(modules.win32gui.GetForegroundWindow() or 0)
    except Exception:
        return None


def _pre_delivery_focus_outcome(error: ComputerError) -> str:
    outcome = error.details.get("outcome")
    if outcome in {"rejected", "failed", "delivery_only"}:
        return str(outcome)
    if error.code in {
        "computer_actions_disabled",
        "invalid_window",
        "window_not_visible",
        "window_untitled",
        "stale_window",
        "secure_desktop_unavailable",
        "protected_target",
        "target_integrity_unavailable",
        "target_integrity_denied",
        "target_identity_unavailable",
    }:
        return "rejected"
    return "failed"


def _focus_delivery_error(
    error: ComputerError,
    *,
    backend: Win32Backend,
    modules: Win32Modules,
    expected: WindowIdentity,
    require_title_match: bool,
    method: str,
    state_before: str,
    restore_performed: bool,
    attached_count: int,
    detached_count: int,
    failed_attachment_count: int,
    state_after: str | None = None,
    foreground_hwnd: int | None = None,
) -> ComputerError:
    observed_state = (
        state_after
        if state_after is not None
        else _safe_window_state(backend, expected, require_title_match)
    )
    observed_foreground = (
        foreground_hwnd
        if foreground_hwnd is not None
        else _safe_foreground_hwnd(modules)
    )
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details={
            **error.details,
            "method": error.details.get("method") or method,
            "state_before": state_before,
            "state_after": observed_state,
            "restore_performed": restore_performed,
            "restored_to_normal": (
                observed_state == "normal" if observed_state != "unavailable" else None
            ),
            "foreground_hwnd": observed_foreground,
            "input_attachments": {
                "attached_count": attached_count,
                "detached_count": detached_count,
                "failed_count": failed_attachment_count,
            },
            "verification_scope": "immediate",
            "postcondition_verified": False,
        },
    )


def _state_delivery_error(
    error: ComputerError,
    *,
    backend: Win32Backend,
    expected: WindowIdentity,
    require_title_match: bool,
    method: str,
    requested_state: str,
    state_before: str,
) -> ComputerError:
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details={
            **error.details,
            "outcome": "delivery_only",
            "method": method,
            "requested_state": requested_state,
            "state_before": state_before,
            "state_after": _safe_window_state(
                backend,
                expected,
                require_title_match,
            ),
            "postcondition_verified": False,
        },
    )


def _resize_delivery_error(
    error: ComputerError,
    *,
    backend: Win32Backend,
    expected: WindowIdentity,
    require_title_match: bool,
    method: str,
    requested_bounds: Bounds,
    restore_first: bool,
    restored_first: bool,
    actual_bounds: Bounds | None = None,
) -> ComputerError:
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details={
            **error.details,
            "outcome": "delivery_only",
            "method": method,
            "requested_bounds": requested_bounds.to_jsonable(),
            "actual_bounds": (
                actual_bounds.to_jsonable()
                if actual_bounds is not None
                else _safe_window_bounds(
                    backend,
                    expected,
                    require_title_match,
                )
            ),
            "restore_first": restore_first,
            "restored_first": restored_first,
            "restore_performed": restored_first,
            "restore_method": (
                "win32.ShowWindow(SW_RESTORE)" if restored_first else None
            ),
            "state_after": _safe_window_state(
                backend,
                expected,
                require_title_match,
            ),
            "postcondition_verified": False,
        },
    )


class Win32Backend:
    """Replaceable adapter for exact-window Win32 reads and verified actions."""

    def modules(self) -> Win32Modules:
        return load_win32_modules()

    def probe(self) -> dict[str, Any]:
        try:
            self.modules()
        except ComputerError as exc:
            return {
                "available": False,
                "error_code": exc.code,
                "capabilities": [],
            }
        return {
            "available": True,
            "backend": "win32",
            "capabilities": [
                "info",
                "windows",
                "focused",
                "screenshot",
                "focus",
                "restore",
                "minimize",
                "maximize",
                "resize",
            ],
        }

    def list_windows(self, *, max_items: int = MAX_VISIBLE_WINDOWS) -> dict[str, Any]:
        modules = self.modules()
        process_names = self._process_snapshot()
        items: list[dict[str, Any]] = []

        def callback(hwnd: int, _extra: object) -> bool:
            try:
                if not modules.win32gui.IsWindowVisible(hwnd) or self._is_cloaked(hwnd):
                    return True
                title = modules.win32gui.GetWindowText(hwnd)
                if not title:
                    return True
                items.append(self._window_info(hwnd, process_names))
            except Exception:  # noqa: BLE001 - windows can disappear during enumeration
                pass
            return True

        modules.win32gui.EnumWindows(callback, None)
        total_count = len(items)
        return available_section(
            count=total_count,
            truncated=total_count > max_items,
            items=items[:max_items],
        )

    def focused_window(self) -> dict[str, Any]:
        modules = self.modules()
        process_names = self._process_snapshot()
        for _attempt in range(2):
            hwnd = int(modules.win32gui.GetForegroundWindow() or 0)
            if not hwnd:
                return available_section(active=False, window=None)
            try:
                return available_section(
                    active=True,
                    window=self._window_info(hwnd, process_names),
                )
            except Exception:  # noqa: BLE001 - foreground can change between reads
                continue
        return {
            "available": False,
            "error_code": "focus_changed_during_read",
            "active": False,
            "window": None,
        }

    def background_processes(
        self, *, max_items: int = MAX_BACKGROUND_PROCESSES
    ) -> dict[str, Any]:
        process_names = self._process_snapshot()
        rows = [
            {
                "pid": pid,
                "name": name,
                "session_id": self._process_session_id(pid),
                "state": "running",
            }
            for pid, name in process_names.items()
            if pid > 0
        ]
        rows.sort(key=lambda row: (str(row["name"]).casefold(), str(row["pid"])))
        return available_section(
            count=len(rows),
            truncated=len(rows) > max_items,
            items=rows[:max_items],
        )

    def process_started(self, pid: int) -> int | None:
        """Return the exact process creation FILETIME used for owner identity."""
        return self._process_started(pid)

    def identity_session(self) -> dict[str, Any]:
        _require_windows()
        resolved_session_id = self._current_session_id()

        desktop_access, desktop_name, lock_state, lock_ui_foreground = self._desktop_state(
            resolved_session_id
        )
        elevated, integrity = self._security_state()
        return available_section(
            host=socket.gethostname(),
            user=self._windows_user(),
            session_id=resolved_session_id,
            session_name=self._session_name(resolved_session_id),
            interactive_desktop_access=desktop_access,
            lock_state=lock_state,
            lock_ui_foreground=lock_ui_foreground,
            desktop_usable=(
                desktop_access
                and lock_state == "unlocked"
                and bool(desktop_name)
                and desktop_name.casefold() == "default"
            ),
            desktop_name=desktop_name,
            elevated=elevated,
            integrity=integrity,
        )

    def system_info(self) -> dict[str, Any]:
        _require_windows()
        kernel32 = _windll("kernel32")
        version = sys.getwindowsversion()
        kernel32.GetTickCount64.restype = ctypes.c_ulonglong
        return available_section(
            windows={
                "major": int(version.major),
                "minor": int(version.minor),
                "build": int(version.build),
                "release": platform.release(),
            },
            architecture=platform.machine(),
            uptime_seconds=int(kernel32.GetTickCount64() // 1000),
            power=self._power_state(),
        )

    def display_info(self) -> dict[str, Any]:
        _require_windows()
        user32 = _windll("user32")
        metrics = user32.GetSystemMetrics
        virtual = Bounds(
            x=int(metrics(76)),
            y=int(metrics(77)),
            width=int(metrics(78)),
            height=int(metrics(79)),
        )
        primary = Bounds(x=0, y=0, width=int(metrics(0)), height=int(metrics(1)))
        dpi: int | None = None
        try:
            get_dpi = user32.GetDpiForSystem
            get_dpi.restype = wintypes.UINT
            dpi = int(get_dpi())
        except (AttributeError, OSError):
            pass
        return available_section(
            virtual_bounds=virtual.to_jsonable(),
            primary_bounds=primary.to_jsonable(),
            monitor_count=int(metrics(80)),
            dpi=dpi,
            scaling_percent=round(dpi * 100 / 96) if dpi else None,
        )

    def capture_identity(self, hwnd: int) -> WindowIdentity:
        modules = self.modules()
        if hwnd <= 0 or not modules.win32gui.IsWindow(hwnd):
            raise ComputerError("invalid_window", f"HWND {hwnd} is not a valid window.")
        user32 = _windll("user32")
        user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
        user32.GetAncestor.restype = wintypes.HWND
        root = int(user32.GetAncestor(hwnd, GA_ROOT) or 0)
        if root != hwnd:
            raise ComputerError(
                "ambiguous_window_target",
                f"HWND {hwnd} is not a top-level window.",
            )
        if not modules.win32gui.IsWindowVisible(hwnd) or self._is_cloaked(hwnd):
            raise ComputerError("window_not_visible", f"HWND {hwnd} is not visible.")
        title = str(modules.win32gui.GetWindowText(hwnd))
        left, top, right, bottom = modules.win32gui.GetWindowRect(hwnd)
        bounds = Bounds.from_ltrb(int(left), int(top), int(right), int(bottom))
        thread_id, pid = modules.win32process.GetWindowThreadProcessId(hwnd)
        process_names = self._process_snapshot()
        session_id = self._process_session_id(int(pid))
        desktop_access, desktop_name, _lock_state, _lock_ui = self._desktop_state(
            session_id
        )
        return WindowIdentity(
            hwnd=hwnd,
            pid=int(pid),
            thread_id=int(thread_id),
            process=process_names.get(int(pid)),
            title=title,
            class_name=str(modules.win32gui.GetClassName(hwnd)),
            bounds=bounds,
            process_started=self._process_started(int(pid)),
            session_id=session_id,
            desktop_name=desktop_name if desktop_access else None,
        )

    @contextmanager
    def native_pixel_context(self):
        """Temporarily make this thread PMv2-aware so geometry is physical pixels."""
        user32 = _windll("user32")
        try:
            setter = user32.SetThreadDpiAwarenessContext
            setter.argtypes = (wintypes.HANDLE,)
            setter.restype = wintypes.HANDLE
            comparer = user32.AreDpiAwarenessContextsEqual
            comparer.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            comparer.restype = wintypes.BOOL
            getter = user32.GetThreadDpiAwarenessContext
            getter.restype = wintypes.HANDLE
        except AttributeError as exc:
            raise ComputerError(
                "dpi_awareness_unavailable",
                "Native screenshot coordinates require per-monitor DPI awareness.",
            ) from exc
        requested = wintypes.HANDLE(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        try:
            previous = setter(requested)
        except Exception as exc:  # noqa: BLE001 - ctypes failures must fail closed
            raise ComputerError(
                "dpi_awareness_unavailable",
                "Native screenshot coordinates require per-monitor DPI awareness.",
            ) from exc
        if not previous:
            raise ComputerError(
                "dpi_awareness_unavailable",
                "Native screenshot coordinates require per-monitor DPI awareness.",
            )
        try:
            try:
                active = getter()
                active_matches = bool(active and comparer(active, requested))
            except Exception as exc:  # noqa: BLE001 - restoration still must run
                raise ComputerError(
                    "dpi_awareness_unavailable",
                    "The screenshot thread could not enter per-monitor DPI awareness.",
                ) from exc
            if not active_matches:
                raise ComputerError(
                    "dpi_awareness_unavailable",
                    "The screenshot thread could not enter per-monitor DPI awareness.",
                )
            yield
        finally:
            try:
                restored = setter(previous)
                current = getter()
                restore_matches = bool(current and comparer(current, previous))
            except Exception as exc:  # noqa: BLE001 - restoration failure is safety-critical
                raise ComputerError(
                    "dpi_awareness_restore_failed",
                    "The screenshot thread DPI context could not be restored safely.",
                ) from exc
            if not restored or not restore_matches:
                raise ComputerError(
                    "dpi_awareness_restore_failed",
                    "The screenshot thread DPI context could not be restored safely.",
                )

    def capture_native_geometry(self, hwnd: int) -> NativeCaptureGeometry:
        with self.native_pixel_context():
            identity = self._capture_screenshot_identity(hwnd)
            user32 = _windll("user32")
            user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(Rect))
            user32.GetClientRect.restype = wintypes.BOOL
            user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(Point))
            user32.ClientToScreen.restype = wintypes.BOOL
            user32.MonitorFromWindow.argtypes = (wintypes.HWND, wintypes.DWORD)
            user32.MonitorFromWindow.restype = wintypes.HANDLE
            user32.GetMonitorInfoW.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(MonitorInfo),
            )
            user32.GetMonitorInfoW.restype = wintypes.BOOL
            user32.GetDpiForSystem.restype = wintypes.UINT

            client_rect = Rect()
            client_origin = Point()
            client_rect_ok = user32.GetClientRect(hwnd, ctypes.byref(client_rect))
            client_origin_ok = user32.ClientToScreen(hwnd, ctypes.byref(client_origin))
            if not client_rect_ok or not client_origin_ok:
                raise ComputerError(
                    "capture_geometry_unavailable",
                    "The target client geometry could not be read safely.",
                )
            monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
            monitor_info = MonitorInfo(cbSize=ctypes.sizeof(MonitorInfo))
            if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(monitor_info)):
                raise ComputerError(
                    "capture_geometry_unavailable",
                    "The target monitor geometry could not be read safely.",
                )
            metrics = user32.GetSystemMetrics
            virtual_screen = Bounds(
                x=int(metrics(76)),
                y=int(metrics(77)),
                width=int(metrics(78)),
                height=int(metrics(79)),
            )
            window_dpi = self.window_dpi(hwnd)
            system_dpi = int(user32.GetDpiForSystem())
            if not window_dpi or system_dpi <= 0:
                raise ComputerError(
                    "capture_dpi_unavailable",
                    "The target DPI could not be read safely.",
                )
            awareness = self.window_dpi_awareness(hwnd)
            if awareness == "unknown":
                raise ComputerError(
                    "capture_dpi_unavailable",
                    "The target DPI awareness could not be read safely.",
                )
            return NativeCaptureGeometry(
                identity=identity,
                client_bounds=Bounds.from_ltrb(
                    int(client_origin.x),
                    int(client_origin.y),
                    int(client_origin.x + client_rect.right - client_rect.left),
                    int(client_origin.y + client_rect.bottom - client_rect.top),
                ),
                monitor_bounds=Bounds.from_ltrb(
                    int(monitor_info.rcMonitor.left),
                    int(monitor_info.rcMonitor.top),
                    int(monitor_info.rcMonitor.right),
                    int(monitor_info.rcMonitor.bottom),
                ),
                virtual_screen_bounds=virtual_screen,
                window_dpi=window_dpi,
                system_dpi=system_dpi,
                window_awareness=awareness,
                capture_thread_awareness="per_monitor_aware_v2",
            )

    def revalidate(self, expected: WindowIdentity) -> WindowIdentity:
        """Revalidate strong screenshot identity while allowing title/geometry drift."""
        self._require_screenshot_identity(expected)
        try:
            actual = self._capture_screenshot_identity(expected.hwnd)
        except ComputerError as exc:
            raise ComputerError(
                "stale_window",
                f"HWND {expected.hwnd} changed or closed before capture.",
                details={"identity_mismatch_fields": ["window"]},
            ) from exc
        mismatches = self.screenshot_identity_mismatches(expected, actual)
        if mismatches:
            raise ComputerError(
                "stale_window",
                f"HWND {expected.hwnd} changed or closed before capture.",
                details={
                    "identity_mismatch_fields": mismatches,
                    "title_changed": actual.title != expected.title,
                    "title_change_was_authoritative": False,
                },
            )
        return actual

    def screenshot_identity_mismatches(
        self,
        expected: WindowIdentity,
        actual: WindowIdentity,
    ) -> list[str]:
        """Return bounded strong-identity mismatch names for a screenshot target."""
        fields: list[str] = []
        comparisons = (
            ("hwnd", expected.hwnd, actual.hwnd),
            ("pid", expected.pid, actual.pid),
            (
                "process_start_identity",
                expected.process_started,
                actual.process_started,
            ),
            ("window_class", expected.class_name, actual.class_name),
            ("session_id", expected.session_id, actual.session_id),
            (
                "input_desktop",
                (expected.desktop_name or "").casefold(),
                (actual.desktop_name or "").casefold(),
            ),
        )
        for name, before, after in comparisons:
            if before != after:
                fields.append(name)
        return fields

    def _capture_screenshot_identity(self, hwnd: int) -> WindowIdentity:
        identity = self.capture_identity(hwnd)
        if identity.session_id is None or not identity.desktop_name:
            raise ComputerError(
                "target_identity_unavailable",
                "The screenshot target session or input desktop could not be verified.",
            )
        if identity.desktop_name.casefold() != "default":
            raise ComputerError(
                "secure_desktop_unavailable",
                "Screenshots require the unlocked default interactive desktop.",
            )
        return identity

    @staticmethod
    def _require_screenshot_identity(identity: WindowIdentity) -> None:
        if (
            identity.process is None
            or identity.process_started is None
            or identity.session_id is None
            or not identity.desktop_name
        ):
            raise ComputerError(
                "target_identity_unavailable",
                "The strong screenshot target identity could not be verified.",
            )

    def strong_identity_mismatches(
        self,
        expected: WindowIdentity,
        actual: WindowIdentity,
        *,
        require_title_match: bool = False,
    ) -> list[str]:
        """Return bounded action-identity mismatch names; titles are metadata by default."""
        comparisons: tuple[tuple[str, object, object], ...] = (
            ("hwnd", expected.hwnd, actual.hwnd),
            ("pid", expected.pid, actual.pid),
            ("thread_id", expected.thread_id, actual.thread_id),
            (
                "process_name",
                (expected.process or "").casefold(),
                (actual.process or "").casefold(),
            ),
            (
                "process_start_identity",
                expected.process_started,
                actual.process_started,
            ),
            ("window_class", expected.class_name, actual.class_name),
            ("session_id", expected.session_id, actual.session_id),
            (
                "input_desktop",
                (expected.desktop_name or "").casefold(),
                (actual.desktop_name or "").casefold(),
            ),
        )
        mismatches = [
            name for name, before, after in comparisons if before != after
        ]
        if require_title_match and expected.title != actual.title:
            mismatches.append("title")
        return mismatches

    def revalidate_identity(
        self,
        expected: WindowIdentity,
        *,
        require_title_match: bool = False,
    ) -> WindowIdentity:
        """Revalidate strong target identity while allowing presentation changes."""
        if (
            expected.process is None
            or expected.process_started is None
            or expected.session_id is None
            or not expected.desktop_name
        ):
            raise ComputerError(
                "target_identity_unavailable",
                "The strong target identity could not be verified for mutation.",
            )
        try:
            actual = self.capture_identity(expected.hwnd)
        except ComputerError as exc:
            raise ComputerError(
                "stale_window",
                f"HWND {expected.hwnd} changed or closed before the action.",
                details={"identity_mismatch_fields": ["window"]},
            ) from exc
        mismatches = self.strong_identity_mismatches(
            expected,
            actual,
            require_title_match=require_title_match,
        )
        if mismatches:
            raise ComputerError(
                "stale_window",
                f"HWND {expected.hwnd} changed or closed before the action.",
                details={
                    "identity_mismatch_fields": mismatches,
                    "title_changed": actual.title != expected.title,
                    "title_change_was_authoritative": require_title_match,
                },
            )
        return actual

    def assert_action_allowed(
        self,
        expected: WindowIdentity,
        *,
        require_title_match: bool = False,
    ) -> WindowIdentity:
        """Fail closed on non-interactive, secure, protected, or higher-integrity targets."""
        actual = (
            self.revalidate_identity(expected, require_title_match=True)
            if require_title_match
            else self.revalidate_identity(expected)
        )
        self._assert_desktop_action_allowed(actual.session_id)
        process_name = (actual.process or "").casefold()
        if process_name in PROTECTED_PROCESS_NAMES:
            raise ComputerError(
                "protected_target",
                "The selected Windows security surface cannot be automated.",
            )
        _current_elevated, current_integrity = self._security_state()
        _target_elevated, target_integrity = self._process_security_state(actual.pid)
        if current_integrity is None or target_integrity is None:
            raise ComputerError(
                "target_integrity_unavailable",
                "The target integrity boundary could not be verified.",
            )
        if INTEGRITY_ORDER[target_integrity] > INTEGRITY_ORDER[current_integrity]:
            raise ComputerError(
                "target_integrity_denied",
                "The target runs at a higher integrity level than AgentTools.",
            )
        return actual

    def assert_desktop_action_allowed(self) -> None:
        """Require the current owner session to have usable desktop authority."""
        self._assert_desktop_action_allowed(self._current_session_id())

    def _assert_desktop_action_allowed(self, session_id: int | None) -> None:
        desktop_access, desktop_name, lock_state, lock_ui_foreground = self._desktop_state(
            session_id
        )
        if (
            not desktop_access
            or lock_state != "unlocked"
            or not desktop_name
            or desktop_name.casefold() != "default"
        ):
            raise ComputerError(
                "secure_desktop_unavailable",
                "Computer actions require the unlocked default interactive desktop.",
                details={
                    "lock_state": lock_state,
                    "lock_ui_foreground": lock_ui_foreground,
                    "interactive_desktop_access": desktop_access,
                    "desktop_name": desktop_name,
                },
            )

    def _revalidate_action_target(
        self,
        expected: WindowIdentity,
        require_title_match: bool,
    ) -> WindowIdentity:
        if require_title_match:
            return self.revalidate_identity(expected, require_title_match=True)
        return self.revalidate_identity(expected)

    def _assert_action_target_allowed(
        self,
        expected: WindowIdentity,
        require_title_match: bool,
    ) -> WindowIdentity:
        if require_title_match:
            return self.assert_action_allowed(expected, require_title_match=True)
        return self.assert_action_allowed(expected)

    def focus_window(
        self,
        expected: WindowIdentity,
        *,
        mutation_guard: Callable[[], None] | None = None,
        require_title_match: bool = False,
    ) -> dict[str, Any]:
        """Restore and focus an exact target, then verify the foreground HWND."""
        modules = self.modules()
        try:
            _initial, state_before = _identity_bound_window_state(
                self,
                expected,
                require_title_match,
            )
        except Exception as raw_exc:
            if isinstance(raw_exc, ComputerError):
                raise
            raise ComputerError(
                "win32_action_failed",
                "Windows could not read the initial focus state.",
                details={
                    "outcome": "failed",
                    "method": "win32.window_state",
                    "state_before": "unavailable",
                    "state_after": "unavailable",
                    "postcondition_verified": False,
                },
            ) from raw_exc
        self._assert_action_target_allowed(expected, require_title_match)
        guard = mutation_guard or (lambda: None)
        try:
            current_tid = int(modules.win32api.GetCurrentThreadId())
            foreground = int(modules.win32gui.GetForegroundWindow() or 0)
            foreground_tid = (
                int(modules.win32process.GetWindowThreadProcessId(foreground)[0])
                if foreground
                else 0
            )
            target_tid = self._revalidate_action_target(
                expected, require_title_match
            ).thread_id
        except Exception as raw_exc:
            if isinstance(raw_exc, ComputerError):
                raise
            raise ComputerError(
                "win32_action_failed",
                "Windows could not read the initial focus routing state.",
                details={
                    "outcome": "failed",
                    "method": "win32.GetForegroundWindow",
                    "state_before": state_before,
                    "postcondition_verified": False,
                },
            ) from raw_exc
        thread_ids: list[int] = []
        for thread_id in (foreground_tid, target_tid):
            if thread_id and thread_id != current_tid and thread_id not in thread_ids:
                thread_ids.append(thread_id)

        attached: list[int] = []
        detached: list[int] = []
        attach_failures = 0
        delivered_method: str | None = None
        restore_performed = False
        primary_error: ComputerError | None = None
        try:
            for thread_id in thread_ids:
                self._assert_action_target_allowed(expected, require_title_match)
                guard()
                self._assert_action_target_allowed(expected, require_title_match)
                try:
                    modules.win32process.AttachThreadInput(current_tid, thread_id, True)
                except modules.win32process.error:
                    attach_failures += 1
                else:
                    attached.append(thread_id)
            if modules.win32gui.IsIconic(expected.hwnd):
                self._assert_action_target_allowed(expected, require_title_match)
                guard()
                self._assert_action_target_allowed(expected, require_title_match)
                if modules.win32gui.IsIconic(expected.hwnd):
                    delivered_method = "win32.ShowWindow"
                    restore_performed = True
                    modules.win32gui.ShowWindow(expected.hwnd, SW_RESTORE)
                deadline = time.monotonic() + 2.0
                while modules.win32gui.IsIconic(expected.hwnd) and time.monotonic() < deadline:
                    time.sleep(0.01)
                if modules.win32gui.IsIconic(expected.hwnd):
                    raise ComputerError(
                        "focus_restore_failed",
                        "The target window could not be restored before focusing.",
                        details={"outcome": "failed", "method": "win32.ShowWindow"},
                    )
            self._assert_action_target_allowed(expected, require_title_match)
            guard()
            self._assert_action_target_allowed(expected, require_title_match)
            delivered_method = "win32.ShowWindow"
            modules.win32gui.ShowWindow(expected.hwnd, SW_SHOW)
            self._assert_action_target_allowed(expected, require_title_match)
            guard()
            self._assert_action_target_allowed(expected, require_title_match)
            try:
                delivered_method = "win32.SetForegroundWindow"
                modules.win32gui.SetForegroundWindow(expected.hwnd)
            except modules.win32gui.error:
                pass
        except Exception as raw_exc:
            if isinstance(raw_exc, ComputerError):
                exc = raw_exc
            else:
                exc = ComputerError(
                    "win32_action_failed",
                    "Windows could not complete the focus action.",
                )
            if delivered_method is not None and "outcome" not in exc.details:
                primary_error = _post_delivery_error(exc, delivered_method)
            else:
                primary_error = exc
        finally:
            detach_errors: list[str] = []
            for thread_id in reversed(attached):
                try:
                    modules.win32process.AttachThreadInput(current_tid, thread_id, False)
                    detached.append(thread_id)
                except modules.win32process.error:
                    detach_errors.append(str(thread_id))
            if detach_errors:
                unresolved_attachment = len(attached) > len(detached)
                if primary_error is None:
                    primary_error = ComputerError(
                        "input_attachment_cleanup_failed",
                        "Windows input queues could not be detached safely.",
                        details={
                            "outcome": "delivery_only",
                            "method": "win32.AttachThreadInput",
                        },
                    )
                elif delivered_method is None and unresolved_attachment:
                    original_error = primary_error
                    primary_error = ComputerError(
                        original_error.code,
                        original_error.message,
                        exit_code=original_error.exit_code,
                        details={
                            **original_error.details,
                            "outcome": "delivery_only",
                            "method": "win32.AttachThreadInput",
                            "partial_mutation": "input_attachment",
                            "focus_failure": {
                                "code": original_error.code,
                                "outcome": _pre_delivery_focus_outcome(original_error),
                                "method": original_error.details.get("method"),
                            },
                        },
                    )
                primary_error.details["warnings"] = sorted(
                    {
                        *primary_error.details.get("warnings", []),
                        "input_attachment_cleanup_failed",
                    }
                )
                primary_error.details["input_attachment_cleanup"] = {
                    "status": "failed",
                    "count": len(detach_errors),
                }

        if primary_error is not None:
            if delivered_method is not None or primary_error.details.get(
                "partial_mutation"
            ) == "input_attachment":
                raise _focus_delivery_error(
                    primary_error,
                    backend=self,
                    modules=modules,
                    expected=expected,
                    require_title_match=require_title_match,
                    method=delivered_method or "win32.AttachThreadInput",
                    state_before=state_before,
                    restore_performed=restore_performed,
                    attached_count=len(attached),
                    detached_count=len(detached),
                    failed_attachment_count=attach_failures,
                )
            raise primary_error

        try:
            actual, state_after = _identity_bound_window_state(
                self,
                expected,
                require_title_match,
            )
        except Exception as raw_exc:
            exc = (
                raw_exc
                if isinstance(raw_exc, ComputerError)
                else ComputerError(
                    "win32_action_failed",
                    "Windows could not verify the final focus state.",
                )
            )
            raise _focus_delivery_error(
                _post_delivery_error(exc, "win32.SetForegroundWindow"),
                backend=self,
                modules=modules,
                expected=expected,
                require_title_match=require_title_match,
                method="win32.SetForegroundWindow",
                state_before=state_before,
                state_after="unavailable",
                restore_performed=restore_performed,
                attached_count=len(attached),
                detached_count=len(detached),
                failed_attachment_count=attach_failures,
            ) from raw_exc
        try:
            foreground_after = int(modules.win32gui.GetForegroundWindow() or 0)
        except Exception as raw_exc:
            raise _focus_delivery_error(
                ComputerError(
                    "win32_action_failed",
                    "Windows could not read the final foreground window.",
                    details={
                        "outcome": "delivery_only",
                        "method": "win32.SetForegroundWindow",
                    },
                ),
                backend=self,
                modules=modules,
                expected=expected,
                require_title_match=require_title_match,
                method="win32.SetForegroundWindow",
                state_before=state_before,
                state_after=state_after,
                restore_performed=restore_performed,
                attached_count=len(attached),
                detached_count=len(detached),
                failed_attachment_count=attach_failures,
            ) from raw_exc
        if foreground_after != expected.hwnd:
            raise _focus_delivery_error(
                ComputerError(
                    "focus_postcondition_failed",
                    "Windows did not make the exact target the foreground window.",
                    details={
                        "outcome": "failed",
                        "method": "win32.SetForegroundWindow",
                    },
                ),
                backend=self,
                modules=modules,
                expected=expected,
                require_title_match=require_title_match,
                method="win32.SetForegroundWindow",
                state_before=state_before,
                state_after=state_after,
                restore_performed=restore_performed,
                foreground_hwnd=foreground_after,
                attached_count=len(attached),
                detached_count=len(detached),
                failed_attachment_count=attach_failures,
            )
        return {
            "window": actual.public(),
            "foreground_before": foreground or None,
            "foreground_hwnd": foreground_after,
            "state_before": state_before,
            "state_after": state_after,
            "restore_performed": restore_performed,
            "restored_to_normal": state_after == "normal",
            "verification_scope": "immediate",
            "input_attachments": {
                "attached_count": len(attached),
                "detached_count": len(detached),
                "failed_count": attach_failures,
            },
            "method": "win32.SetForegroundWindow",
            "outcome": "postcondition_verified",
            "postcondition_verified": True,
            "changed": bool(
                restore_performed
                or foreground != expected.hwnd
                or state_before != state_after
            ),
        }

    def resize_window(
        self,
        expected: WindowIdentity,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        mutation_guard: Callable[[], None] | None = None,
        restore_first: bool = False,
        require_title_match: bool = False,
    ) -> dict[str, Any]:
        """Move/resize an exact normal window inside the virtual desktop and verify it."""
        modules = self.modules()
        guard = mutation_guard or (lambda: None)
        self._assert_action_target_allowed(expected, require_title_match)
        if width <= 0 or height <= 0:
            raise ComputerError("invalid_geometry", "Width and height must be positive.")
        requested = Bounds(x, y, width, height)
        initial_monitors = _active_monitor_rectangles(modules)
        if not _rectangle_covered_by_monitors(requested, initial_monitors):
            raise ComputerError(
                "geometry_offscreen",
                "The requested rectangle must be completely covered by active monitors.",
                details={"requested_bounds": requested.to_jsonable()},
            )
        try:
            _initial, state_before = _identity_bound_window_state(
                self,
                expected,
                require_title_match,
            )
        except Exception as raw_exc:
            if isinstance(raw_exc, ComputerError):
                raise
            raise ComputerError(
                "win32_action_failed",
                "Windows could not read the initial resize state.",
                details={
                    "outcome": "failed",
                    "method": "win32.window_state",
                    "requested_bounds": requested.to_jsonable(),
                    "actual_bounds": None,
                    "restore_first": restore_first,
                    "restored_first": False,
                    "restore_performed": False,
                    "restore_method": None,
                    "state_after": "unavailable",
                    "postcondition_verified": False,
                },
            ) from raw_exc
        restored_first = False
        if restore_first and state_before != "normal":
            restore_delivery_attempted = False

            def restore_guard() -> None:
                guard()

            def mark_restore_delivery_attempted() -> None:
                nonlocal restore_delivery_attempted
                restore_delivery_attempted = True

            try:
                self.set_window_state(
                    expected,
                    state="normal",
                    mutation_guard=restore_guard,
                    require_title_match=require_title_match,
                    _delivery_boundary=mark_restore_delivery_attempted,
                )
            except Exception as raw_exc:
                exc = (
                    raw_exc
                    if isinstance(raw_exc, ComputerError)
                    else ComputerError(
                        "win32_action_failed",
                        "Windows could not complete the restore-first action.",
                    )
                )
                raise _nested_restore_resize_error(
                    self,
                    exc,
                    expected=expected,
                    require_title_match=require_title_match,
                    requested_bounds=requested,
                    restore_delivery_attempted=restore_delivery_attempted,
                ) from raw_exc
            restored_first = True
        try:
            if modules.win32gui.IsIconic(expected.hwnd):
                raise ComputerError(
                    "window_minimized",
                    "Restore the target with computer focus before resizing it.",
                )
            if self._is_zoomed(expected.hwnd):
                raise ComputerError(
                    "window_maximized",
                    "Restore the maximized target before resizing it.",
                )
            self._assert_action_target_allowed(expected, require_title_match)
            if modules.win32gui.IsIconic(expected.hwnd):
                raise ComputerError(
                    "window_minimized",
                    "The target became minimized before resizing.",
                )
            if self._is_zoomed(expected.hwnd):
                raise ComputerError(
                    "window_maximized",
                    "The target became maximized before resizing.",
                )
            current_monitors = _active_monitor_rectangles(modules)
            if not _rectangle_covered_by_monitors(requested, current_monitors):
                raise ComputerError(
                    "geometry_offscreen",
                    "The requested rectangle is no longer covered by active monitors.",
                    details={"requested_bounds": requested.to_jsonable()},
                )
            guard()
            self._assert_action_target_allowed(expected, require_title_match)
            if modules.win32gui.IsIconic(expected.hwnd):
                raise ComputerError(
                    "window_minimized",
                    "The target became minimized at the resize delivery boundary.",
                )
            if self._is_zoomed(expected.hwnd):
                raise ComputerError(
                    "window_maximized",
                    "The target became maximized at the resize delivery boundary.",
                )
            delivery_monitors = _active_monitor_rectangles(modules)
            if not _rectangle_covered_by_monitors(requested, delivery_monitors):
                raise ComputerError(
                    "geometry_offscreen",
                    "The requested rectangle is not covered at the resize delivery boundary.",
                    details={"requested_bounds": requested.to_jsonable()},
                )
        except Exception as raw_exc:
            exc = (
                raw_exc
                if isinstance(raw_exc, ComputerError)
                else ComputerError(
                    "win32_action_failed",
                    "Windows could not verify the resize target before delivery.",
                )
            )
            if restored_first:
                raise _post_restore_resize_error(
                    self,
                    exc,
                    expected=expected,
                    require_title_match=require_title_match,
                    requested_bounds=requested,
                ) from exc
            if isinstance(raw_exc, ComputerError):
                raise
            raise exc from raw_exc
        try:
            modules.win32gui.MoveWindow(expected.hwnd, x, y, width, height, True)
        except Exception as raw_exc:
            exc = ComputerError(
                "win32_action_failed",
                "Windows could not complete the resize action.",
            )
            raise _resize_delivery_error(
                exc,
                backend=self,
                expected=expected,
                require_title_match=require_title_match,
                method="win32.MoveWindow",
                requested_bounds=requested,
                restore_first=restore_first,
                restored_first=restored_first,
            ) from raw_exc
        deadline = time.monotonic() + 2.0
        actual: WindowIdentity | None = None
        try:
            actual = self._revalidate_action_target(expected, require_title_match)
            while actual.bounds != Bounds(x, y, width, height) and time.monotonic() < deadline:
                time.sleep(0.01)
                actual = self._revalidate_action_target(expected, require_title_match)
        except Exception as raw_exc:
            exc = (
                raw_exc
                if isinstance(raw_exc, ComputerError)
                else ComputerError(
                    "win32_action_failed",
                    "Windows could not verify the resized window.",
                )
            )
            raise _resize_delivery_error(
                exc,
                backend=self,
                expected=expected,
                require_title_match=require_title_match,
                method="win32.MoveWindow",
                requested_bounds=requested,
                restore_first=restore_first,
                restored_first=restored_first,
                actual_bounds=actual.bounds if actual is not None else None,
            ) from raw_exc
        assert actual is not None
        if actual.bounds != requested:
            raise ComputerError(
                "resize_postcondition_failed",
                "The exact DPI-aware target rectangle was not observed after resizing.",
                details={
                    "outcome": "failed",
                    "method": "win32.MoveWindow",
                    "requested_bounds": requested.to_jsonable(),
                    "actual_bounds": actual.bounds.to_jsonable(),
                    "restore_first": restore_first,
                    "restored_first": restored_first,
                    "restore_performed": restored_first,
                    "restore_method": (
                        "win32.ShowWindow(SW_RESTORE)" if restored_first else None
                    ),
                    "state_after": _safe_window_state(
                        self,
                        expected,
                        require_title_match,
                    ),
                    "postcondition_verified": False,
                },
            )
        try:
            self._revalidate_action_target(expected, require_title_match)
            minimized = bool(modules.win32gui.IsIconic(expected.hwnd))
            maximized = bool(self._is_zoomed(expected.hwnd))
            actual = self._revalidate_action_target(expected, require_title_match)
        except Exception as raw_exc:
            exc = (
                raw_exc
                if isinstance(raw_exc, ComputerError)
                else ComputerError(
                    "win32_action_failed",
                    "Windows could not verify the final resize state.",
                )
            )
            raise _resize_delivery_error(
                exc,
                backend=self,
                expected=expected,
                require_title_match=require_title_match,
                method="win32.MoveWindow",
                requested_bounds=requested,
                restore_first=restore_first,
                restored_first=restored_first,
                actual_bounds=actual.bounds,
            ) from raw_exc
        if minimized or maximized:
            observed_state = "minimized" if minimized else "maximized"
            raise ComputerError(
                "resize_state_postcondition_failed",
                "The target did not remain in the normal window state after resizing.",
                details={
                    "outcome": "failed",
                    "method": "win32.MoveWindow",
                    "requested_bounds": requested.to_jsonable(),
                    "actual_bounds": actual.bounds.to_jsonable(),
                    "restore_first": restore_first,
                    "restored_first": restored_first,
                    "restore_performed": restored_first,
                    "restore_method": (
                        "win32.ShowWindow(SW_RESTORE)" if restored_first else None
                    ),
                    "state_after": observed_state,
                    "postcondition_verified": False,
                },
            )
        try:
            dpi = self.window_dpi(expected.hwnd)
        except Exception:
            dpi = None
        return {
            "window": actual.public(),
            "requested_bounds": requested.to_jsonable(),
            "actual_bounds": actual.bounds.to_jsonable(),
            "dpi": dpi,
            "state": "normal",
            "restore_first": restore_first,
            "restored_first": restored_first,
            "method": "win32.MoveWindow",
            "outcome": "postcondition_verified",
            "postcondition_verified": True,
        }

    def window_dpi(self, hwnd: int) -> int | None:
        user32 = _windll("user32")
        try:
            user32.GetDpiForWindow.argtypes = (wintypes.HWND,)
            user32.GetDpiForWindow.restype = wintypes.UINT
            value = int(user32.GetDpiForWindow(hwnd))
        except (AttributeError, OSError):
            return None
        return value or None

    def window_dpi_awareness(self, hwnd: int) -> str:
        user32 = _windll("user32")
        try:
            get_context = user32.GetWindowDpiAwarenessContext
            get_context.argtypes = (wintypes.HWND,)
            get_context.restype = wintypes.HANDLE
            comparer = user32.AreDpiAwarenessContextsEqual
            comparer.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            comparer.restype = wintypes.BOOL
        except AttributeError:
            return "unknown"
        context = get_context(hwnd)
        if not context:
            return "unknown"
        contexts = (
            (DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, "per_monitor_aware_v2"),
            (DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE, "per_monitor_aware"),
            (DPI_AWARENESS_CONTEXT_SYSTEM_AWARE, "system_aware"),
            (DPI_AWARENESS_CONTEXT_UNAWARE_GDISCALED, "unaware_gdi_scaled"),
            (DPI_AWARENESS_CONTEXT_UNAWARE, "unaware"),
        )
        for expected, name in contexts:
            if comparer(context, wintypes.HANDLE(expected)):
                return name
        return "unknown"

    def is_minimized(self, hwnd: int) -> bool:
        return bool(self.modules().win32gui.IsIconic(hwnd))

    def window_state(self, hwnd: int) -> str:
        modules = self.modules()
        if modules.win32gui.IsIconic(hwnd):
            return "minimized"
        if self._is_zoomed(hwnd):
            return "maximized"
        return "normal"

    def set_window_state(
        self,
        expected: WindowIdentity,
        *,
        state: str,
        mutation_guard: Callable[[], None] | None = None,
        require_title_match: bool = False,
        _delivery_boundary: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if state not in {"normal", "minimized", "maximized"}:
            raise ComputerError("invalid_window_state", "The requested window state is invalid.")
        modules = self.modules()
        guard = mutation_guard or (lambda: None)
        self._assert_action_target_allowed(expected, require_title_match)
        try:
            _initial, before = _identity_bound_window_state(
                self,
                expected,
                require_title_match,
            )
        except Exception as raw_exc:
            if isinstance(raw_exc, ComputerError):
                raise
            raise ComputerError(
                "win32_action_failed",
                "Windows could not read the initial window state.",
                details={
                    "outcome": "failed",
                    "method": "win32.window_state",
                    "requested_state": state,
                    "state_before": "unavailable",
                    "state_after": "unavailable",
                    "postcondition_verified": False,
                },
            ) from raw_exc
        command = {
            "normal": SW_RESTORE,
            "minimized": SW_MINIMIZE,
            "maximized": SW_MAXIMIZE,
        }[state]
        method = "win32.ShowWindow"
        deadline = time.monotonic() + 2.0
        guard()
        self._assert_action_target_allowed(expected, require_title_match)
        if _delivery_boundary is not None:
            _delivery_boundary()
        try:
            modules.win32gui.ShowWindow(expected.hwnd, command)
            actual, observed = _identity_bound_window_state(
                self,
                expected,
                require_title_match,
            )
            while observed != state and time.monotonic() < deadline:
                if state == "normal" and observed != "normal":
                    self._assert_action_target_allowed(expected, require_title_match)
                    guard()
                    self._assert_action_target_allowed(expected, require_title_match)
                    if _delivery_boundary is not None:
                        _delivery_boundary()
                    modules.win32gui.ShowWindow(expected.hwnd, SW_RESTORE)
                time.sleep(0.01)
                actual, observed = _identity_bound_window_state(
                    self,
                    expected,
                    require_title_match,
                )
        except Exception as raw_exc:
            exc = (
                raw_exc
                if isinstance(raw_exc, ComputerError)
                else ComputerError(
                    "win32_action_failed",
                    "Windows could not verify the requested window state.",
                )
            )
            raise _state_delivery_error(
                exc,
                backend=self,
                expected=expected,
                require_title_match=require_title_match,
                method=method,
                requested_state=state,
                state_before=before,
            ) from raw_exc
        if observed != state:
            raise ComputerError(
                "window_state_postcondition_failed",
                "Windows did not reach the requested window state.",
                details={
                    "outcome": "failed",
                    "method": method,
                    "requested_state": state,
                    "state_before": before,
                    "state_after": observed,
                    "postcondition_verified": False,
                },
            )
        return {
            "window": actual.public(),
            "state_before": before,
            "state_after": observed,
            "requested_state": state,
            "method": method,
            "outcome": "postcondition_verified",
            "postcondition_verified": True,
        }

    def _window_info(
        self,
        hwnd: int,
        process_names: dict[int, str],
    ) -> dict[str, Any]:
        modules = self.modules()
        left, top, right, bottom = modules.win32gui.GetWindowRect(hwnd)
        thread_id, pid = modules.win32process.GetWindowThreadProcessId(hwnd)
        if modules.win32gui.IsIconic(hwnd):
            state = "minimized"
        elif self._is_zoomed(hwnd):
            state = "maximized"
        else:
            state = "normal"
        return {
            "hwnd": int(hwnd),
            "pid": int(pid),
            "process": process_names.get(int(pid)),
            "title": str(modules.win32gui.GetWindowText(hwnd))[:MAX_TITLE_CHARS],
            "class": str(modules.win32gui.GetClassName(hwnd))[:MAX_CLASS_NAME_CHARS],
            "state": state,
            "bounds": Bounds.from_ltrb(
                int(left), int(top), int(right), int(bottom)
            ).to_jsonable(),
            "thread_id": int(thread_id),
        }

    def _is_cloaked(self, hwnd: int) -> bool:
        try:
            dwmapi = _windll("dwmapi")
            dwmapi.DwmGetWindowAttribute.argtypes = (
                wintypes.HWND,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
            )
            dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
            value = wintypes.DWORD()
            result = dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(DWMWA_CLOAKED),
                ctypes.byref(value),
                ctypes.sizeof(value),
            )
            return result == 0 and bool(value.value)
        except (AttributeError, OSError):
            return False

    def _is_zoomed(self, hwnd: int) -> bool:
        user32 = _windll("user32")
        user32.IsZoomed.argtypes = (wintypes.HWND,)
        user32.IsZoomed.restype = wintypes.BOOL
        return bool(user32.IsZoomed(hwnd))

    def _process_snapshot(self) -> dict[int, str]:
        _require_windows()
        kernel32 = _windll("kernel32")
        kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessEntry32W),
        )
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessEntry32W),
        )
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        invalid_handle = ctypes.c_void_p(-1).value
        if int(handle or 0) == int(invalid_handle or -1):
            return {}
        processes: dict[int, str] = {}
        try:
            entry = ProcessEntry32W()
            entry.dwSize = ctypes.sizeof(ProcessEntry32W)
            available = bool(kernel32.Process32FirstW(handle, ctypes.byref(entry)))
            while available:
                processes[int(entry.th32ProcessID)] = str(entry.szExeFile)[
                    :MAX_PROCESS_NAME_CHARS
                ]
                available = bool(kernel32.Process32NextW(handle, ctypes.byref(entry)))
        finally:
            kernel32.CloseHandle(handle)
        return processes

    def _process_session_id(self, pid: int) -> int | None:
        kernel32 = _windll("kernel32")
        session_id = wintypes.DWORD()
        kernel32.ProcessIdToSessionId.argtypes = (wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
        kernel32.ProcessIdToSessionId.restype = wintypes.BOOL
        if not kernel32.ProcessIdToSessionId(pid, ctypes.byref(session_id)):
            return None
        return int(session_id.value)

    def _process_started(self, pid: int) -> int | None:
        kernel32 = _windll("kernel32")
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
        )
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            created = FileTime()
            exited = FileTime()
            kernel = FileTime()
            user = FileTime()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return None
            return _filetime_value(created)
        finally:
            kernel32.CloseHandle(handle)

    def _windows_user(self) -> str | None:
        advapi32 = _windll("advapi32")
        size = wintypes.DWORD(256)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not advapi32.GetUserNameW(buffer, ctypes.byref(size)):
            return None
        return str(buffer.value)[:128]

    def _session_name(self, session_id: int | None) -> str | None:
        if session_id is None:
            return None
        wtsapi32 = _windll("wtsapi32")
        buffer = wintypes.LPWSTR()
        byte_count = wintypes.DWORD()
        wtsapi32.WTSQuerySessionInformationW.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.POINTER(wintypes.LPWSTR),
            ctypes.POINTER(wintypes.DWORD),
        )
        wtsapi32.WTSQuerySessionInformationW.restype = wintypes.BOOL
        wtsapi32.WTSFreeMemory.argtypes = (wintypes.LPVOID,)
        if not wtsapi32.WTSQuerySessionInformationW(
            None,
            session_id,
            6,
            ctypes.byref(buffer),
            ctypes.byref(byte_count),
        ):
            return None
        try:
            return str(buffer.value or "")[:128] or None
        finally:
            wtsapi32.WTSFreeMemory(buffer)

    def _desktop_state(
        self,
        session_id: int | None = None,
    ) -> tuple[bool, str | None, str, bool | None]:
        user32 = _windll("user32")
        user32.OpenInputDesktop.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        user32.OpenInputDesktop.restype = wintypes.HANDLE
        user32.CloseDesktop.argtypes = (wintypes.HANDLE,)
        user32.CloseDesktop.restype = wintypes.BOOL
        user32.GetUserObjectInformationW.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        user32.GetUserObjectInformationW.restype = wintypes.BOOL
        desktop = user32.OpenInputDesktop(0, False, DESKTOP_READOBJECTS)
        if not desktop:
            lock_ui_foreground = self._foreground_lock_ui(session_id)
            return (
                False,
                None,
                "locked" if lock_ui_foreground is True else "unknown",
                lock_ui_foreground,
            )
        try:
            needed = wintypes.DWORD()
            user32.GetUserObjectInformationW(desktop, UOI_NAME, None, 0, ctypes.byref(needed))
            if not needed.value:
                name = None
            else:
                character_count = max(
                    1, needed.value // ctypes.sizeof(wintypes.WCHAR)
                )
                buffer = ctypes.create_unicode_buffer(character_count)
                if not user32.GetUserObjectInformationW(
                    desktop,
                    UOI_NAME,
                    buffer,
                    needed.value,
                    ctypes.byref(needed),
                ):
                    name = None
                else:
                    name = str(buffer.value)[:128]
        finally:
            user32.CloseDesktop(desktop)
        lock_ui_foreground = self._foreground_lock_ui(session_id)
        if name is None:
            return True, None, "unknown", lock_ui_foreground
        return True, name, _classify_lock_state(name, lock_ui_foreground), lock_ui_foreground

    def _foreground_lock_ui(self, session_id: int | None) -> bool | None:
        if session_id is None:
            return None
        try:
            modules = self.modules()
            foreground = int(modules.win32gui.GetForegroundWindow() or 0)
            if foreground <= 0:
                return None
            _thread_id, pid = modules.win32process.GetWindowThreadProcessId(foreground)
            pid = int(pid)
            if pid <= 0:
                return None
            foreground_session = self._process_session_id(pid)
            if session_id is not None:
                if foreground_session is None:
                    return None
                if foreground_session != session_id:
                    return False
            process_name = self._process_snapshot().get(pid)
            if not process_name:
                return None
            verified_foreground = int(modules.win32gui.GetForegroundWindow() or 0)
            if verified_foreground != foreground:
                return None
            _verified_thread_id, verified_pid = (
                modules.win32process.GetWindowThreadProcessId(verified_foreground)
            )
            if int(verified_pid) != pid:
                return None
            return process_name.casefold() in LOCK_UI_PROCESS_NAMES
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        except Exception as error:
            if _is_pywin32_error(error):
                return None
            raise

    def _current_session_id(self) -> int | None:
        kernel32 = _windll("kernel32")
        return self._process_session_id(int(kernel32.GetCurrentProcessId()))

    def _security_state(self) -> tuple[bool | None, str | None]:
        return self._process_security_state(None)

    def _process_security_state(self, pid: int | None) -> tuple[bool | None, str | None]:
        advapi32 = _windll("advapi32")
        kernel32 = _windll("kernel32")
        token = wintypes.HANDLE()
        advapi32.OpenProcessToken.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        )
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        process_handle = None
        process = kernel32.GetCurrentProcess()
        if pid is not None:
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            process_handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not process_handle:
                return None, None
            process = process_handle
        if not advapi32.OpenProcessToken(process, TOKEN_QUERY, ctypes.byref(token)):
            if process_handle:
                kernel32.CloseHandle(process_handle)
            return None, None
        try:
            elevation = wintypes.DWORD()
            returned = wintypes.DWORD()
            elevated: bool | None = None
            if advapi32.GetTokenInformation(
                token,
                TOKEN_ELEVATION,
                ctypes.byref(elevation),
                ctypes.sizeof(elevation),
                ctypes.byref(returned),
            ):
                elevated = bool(elevation.value)
            integrity = self._token_integrity(advapi32, token)
            return elevated, integrity
        finally:
            kernel32.CloseHandle(token)
            if process_handle:
                kernel32.CloseHandle(process_handle)

    def _token_integrity(self, advapi32: Any, token: wintypes.HANDLE) -> str | None:
        size = wintypes.DWORD()
        advapi32.GetTokenInformation(
            token, TOKEN_INTEGRITY_LEVEL, None, 0, ctypes.byref(size)
        )
        if not size.value:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(
            token,
            TOKEN_INTEGRITY_LEVEL,
            buffer,
            size.value,
            ctypes.byref(size),
        ):
            return None
        label = ctypes.cast(buffer, ctypes.POINTER(TokenMandatoryLabel)).contents
        advapi32.GetSidSubAuthorityCount.argtypes = (wintypes.LPVOID,)
        advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(wintypes.BYTE)
        advapi32.GetSidSubAuthority.argtypes = (wintypes.LPVOID, wintypes.DWORD)
        advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
        count_ptr = advapi32.GetSidSubAuthorityCount(label.Label.Sid)
        if not count_ptr:
            return None
        count = int(count_ptr.contents.value)
        if count <= 0:
            return None
        rid_ptr = advapi32.GetSidSubAuthority(label.Label.Sid, count - 1)
        if not rid_ptr:
            return None
        rid = int(rid_ptr.contents.value)
        if rid < 0x1000:
            return "untrusted"
        if rid < 0x2000:
            return "low"
        if rid < 0x3000:
            return "medium"
        if rid < 0x4000:
            return "high"
        if rid < 0x5000:
            return "system"
        return "protected"

    def _power_state(self) -> dict[str, Any] | None:
        kernel32 = _windll("kernel32")
        status = SystemPowerStatus()
        if not kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
        ac_map = {0: "offline", 1: "online", 255: "unknown"}
        return {
            "ac": ac_map.get(int(status.ACLineStatus), "unknown"),
            "battery_percent": (
                None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent)
            ),
            "battery_saver": bool(status.SystemStatusFlag),
        }
