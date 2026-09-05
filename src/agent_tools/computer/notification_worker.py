from __future__ import annotations

import ctypes
import json
import os
import sys
import time
import unicodedata
from ctypes import wintypes
from pathlib import Path

NIM_ADD = 0x00000000
NIM_MODIFY = 0x00000001
NIM_DELETE = 0x00000002
NIF_MESSAGE = 0x00000001
NIF_ICON = 0x00000002
NIF_TIP = 0x00000004
NIF_INFO = 0x00000010
NIIF_INFO = 0x00000001
WM_USER = 0x0400
IDI_INFORMATION = 32516
MAX_INPUT_BYTES = 8_192
DISPLAY_SECONDS = 5.0
NOTIFICATION_MUTEX_NAME = r"Local\ai-nd-co.agent-tools.computer-notification.v1"
WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
WAIT_TIMEOUT = 0x00000102
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000


class NotifyIconDataW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT),
        ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HICON),
        ("szTip", wintypes.WCHAR * 128),
        ("dwState", wintypes.DWORD),
        ("dwStateMask", wintypes.DWORD),
        ("szInfo", wintypes.WCHAR * 256),
        ("uTimeoutOrVersion", wintypes.UINT),
        ("szInfoTitle", wintypes.WCHAR * 64),
        ("dwInfoFlags", wintypes.DWORD),
        ("guidItem", ctypes.c_byte * 16),
        ("hBalloonIcon", wintypes.HICON),
    ]


class Message(ctypes.Structure):
    _fields_ = [
        ("hWnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt_x", wintypes.LONG),
        ("pt_y", wintypes.LONG),
        ("lPrivate", wintypes.DWORD),
    ]


def main() -> int:
    worker_handle = 0
    try:
        worker_handle = _try_acquire_worker_mutex()
        if not worker_handle:
            _write_status("BUSY")
            return 7
        payload = _read_payload()
        if not _owner_alive(payload):
            _write_status("OWNER_GONE")
            return 8
        title = _bounded(payload.get("title"), 48)
        message = _bounded(payload.get("message"), 200)
        if not title or not message:
            return 2
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        user32.CreateWindowExW.restype = wintypes.HWND
        hwnd = int(
            user32.CreateWindowExW(
                0,
                "STATIC",
                "AgentToolsNotificationWorker",
                0,
                0,
                0,
                0,
                0,
                None,
                None,
                None,
                None,
            )
            or 0
        )
        if not hwnd:
            return 3
        icon = user32.LoadIconW(None, ctypes.c_void_p(IDI_INFORMATION))
        data = NotifyIconDataW()
        data.cbSize = ctypes.sizeof(NotifyIconDataW)
        data.hWnd = hwnd
        data.uID = 1
        data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        data.uCallbackMessage = WM_USER + 1
        data.hIcon = icon
        data.szTip = "AgentTools"
        added = False
        try:
            if _actions_disabled(payload):
                _write_status("DISABLED")
                return 6
            if not shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(data)):
                return 4
            added = True
            data.uFlags = NIF_INFO
            data.szInfo = message
            data.szInfoTitle = title
            data.dwInfoFlags = NIIF_INFO
            data.uTimeoutOrVersion = int(DISPLAY_SECONDS * 1_000)
            if _actions_disabled(payload):
                _write_status("DISABLED")
                return 6
            if not shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(data)):
                return 5
            _write_status("READY")
            sys.stdout.close()
            time.sleep(DISPLAY_SECONDS)
            return 0
        finally:
            if added:
                shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(data))
            user32.DestroyWindow(hwnd)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return 1
    finally:
        if worker_handle:
            _release_worker_mutex(worker_handle)


def _write_status(status: str) -> None:
    sys.stdout.write(f"{status}\n")
    sys.stdout.flush()


def _read_payload() -> dict[str, object]:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("notification payload too large")
    payload = json.loads(raw.decode("ascii"))
    if not isinstance(payload, dict):
        raise ValueError("notification payload must be an object")
    return payload


def _bounded(value: object, limit: int) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        units = len(value.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        return None
    if units > limit:
        return None
    if any(unicodedata.category(character) in {"Cc", "Cf"} for character in value):
        return None
    return value


def _owner_alive(payload: dict[str, object]) -> bool:
    owner = payload.get("owner")
    if not isinstance(owner, dict):
        return False
    pid = owner.get("pid")
    process_started = owner.get("processStarted")
    if type(pid) is not int or type(process_started) is not int or pid <= 0:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = int(
        kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE,
            False,
            pid,
        )
        or 0
    )
    if not handle:
        return False
    try:
        if int(kernel32.WaitForSingleObject(handle, 0)) != WAIT_TIMEOUT:
            return False
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return False
        actual_started = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
        return actual_started == process_started
    finally:
        kernel32.CloseHandle(handle)


def _actions_disabled(payload: dict[str, object]) -> bool:
    if os.environ.get("AGENT_TOOLS_COMPUTER_ACTIONS_DISABLED", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    marker = payload.get("disableMarker")
    if not isinstance(marker, str) or not marker or len(marker) > 1_024:
        return marker is not None
    path = Path(marker)
    if not path.is_absolute() or not path.drive or marker.startswith(("//", "\\\\")):
        return True
    try:
        return path.is_file()
    except OSError:
        return True


def _try_acquire_worker_mutex() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    )
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = int(kernel32.CreateMutexW(None, False, NOTIFICATION_MUTEX_NAME) or 0)
    if not handle:
        return 0
    result = int(kernel32.WaitForSingleObject(handle, 0))
    if result in {WAIT_OBJECT_0, WAIT_ABANDONED}:
        return handle
    kernel32.CloseHandle(handle)
    return 0


def _release_worker_mutex(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    try:
        kernel32.ReleaseMutex(handle)
    finally:
        kernel32.CloseHandle(handle)


if __name__ == "__main__":
    raise SystemExit(main())
