from __future__ import annotations

import asyncio
import ctypes
import importlib
import importlib.util
import ipaddress
import json
import locale
import os
import queue
import select
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_tools import __version__
from agent_tools.computer.actions import action_capabilities
from agent_tools.computer.models import (
    MAX_PROCESS_NAME_CHARS,
    MAX_TITLE_CHARS,
    ComputerError,
    available_section,
    bounded_text,
    unavailable_section,
)
from agent_tools.computer.uia_winapp import WINAPP_PROBE_MAX_SECONDS, WinAppAdapter
from agent_tools.computer.win32_backend import Win32Backend

MAX_EXTERNAL_OUTPUT_BYTES = 64 * 1024
NETWORK_COMMAND_BUDGET_SECONDS = 5.0
EXTERNAL_CLEANUP_BUDGET_SECONDS = 3.0
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
TH32CS_SNAPTHREAD = 0x00000004
TH32CS_SNAPPROCESS = 0x00000002
THREAD_SUSPEND_RESUME = 0x0002
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JobObjectBasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _ThreadEntry32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


class _ProcessEntry32W(ctypes.Structure):
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


class _FileTime(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", wintypes.DWORD),
        ("dwHighDateTime", wintypes.DWORD),
    ]


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    timeout_seconds: float
    function: Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class ExternalResult:
    returncode: int | None
    stdout: str
    error_code: str | None = None


@dataclass
class _WindowsProcessJob:
    handle: int
    kernel32: Any
    closed: bool = False
    root_process: subprocess.Popen[bytes] | None = None

    def assign(self, process: subprocess.Popen[bytes]) -> bool:
        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            return False
        assigned = bool(
            self.kernel32.AssignProcessToJobObject(
                wintypes.HANDLE(self.handle),
                wintypes.HANDLE(int(process_handle)),
            )
        )
        if assigned:
            self.root_process = process
        return assigned

    def active_process_count(self) -> int | None:
        information = _JobObjectBasicAccountingInformation()
        if not self.kernel32.QueryInformationJobObject(
            wintypes.HANDLE(self.handle),
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
            None,
        ):
            return None
        return int(information.ActiveProcesses)

    def terminate_and_wait(self, deadline: float) -> bool:
        active = self.active_process_count()
        if active is None:
            return False
        if active == 0:
            return True
        if not self.kernel32.TerminateJobObject(wintypes.HANDLE(self.handle), 1):
            return self.active_process_count() == 0
        while time.monotonic() < deadline:
            active = self.active_process_count()
            if active is None:
                return False
            if active == 0:
                return True
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return self.active_process_count() == 0

    def close(self) -> bool:
        if self.closed:
            return True
        if not self.kernel32.CloseHandle(wintypes.HANDLE(self.handle)):
            return False
        self.closed = True
        self.handle = 0
        return True


@dataclass(frozen=True)
class _ExactDescendantHandle:
    pid: int
    parent_pid: int
    created: int
    depth: int
    handle: int
    can_terminate: bool = True


@dataclass
class _WindowsDescendantTree:
    kernel32: Any
    root_pid: int
    root_created: int
    handles: dict[int, _ExactDescendantHandle]

    @classmethod
    def capture(cls, process: subprocess.Popen[bytes]) -> _WindowsDescendantTree | None:
        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            return None
        kernel32 = _windows_process_tree_kernel32()
        root_created = _process_created_filetime(kernel32, int(process_handle))
        if root_created is None:
            return None
        return cls(
            kernel32=kernel32,
            root_pid=int(process.pid),
            root_created=root_created,
            handles={},
        )

    def discover(self) -> tuple[bool, int]:
        snapshot_time = _system_filetime(self.kernel32)
        entries = _process_parent_snapshot(self.kernel32)
        if snapshot_time is None or entries is None:
            return False, 0
        lineage_depth = {
            self.root_pid: 0,
            **{
                descendant.pid: descendant.depth
                for descendant in self.handles.values()
            },
        }
        changed = True
        while changed:
            changed = False
            for pid, parent_pid in entries.items():
                if pid in lineage_depth or parent_pid not in lineage_depth:
                    continue
                lineage_depth[pid] = lineage_depth[parent_pid] + 1
                changed = True

        complete = True
        added = 0
        for pid, depth in sorted(lineage_depth.items(), key=lambda item: item[1]):
            if pid == self.root_pid or pid in self.handles:
                continue
            handle = int(
                self.kernel32.OpenProcess(
                    PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE,
                    False,
                    pid,
                )
                or 0
            )
            can_terminate = handle > 0
            if handle <= 0:
                handle = int(
                    self.kernel32.OpenProcess(
                        PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE,
                        False,
                        pid,
                    )
                    or 0
                )
            if handle <= 0:
                refreshed_entries = _process_parent_snapshot(self.kernel32)
                if (
                    refreshed_entries is None
                    or refreshed_entries.get(pid) == entries[pid]
                ):
                    complete = False
                continue
            created = _process_created_filetime(self.kernel32, handle)
            if created is None or not self.root_created <= created <= snapshot_time:
                complete = False
                self.kernel32.CloseHandle(wintypes.HANDLE(handle))
                continue
            self.handles[pid] = _ExactDescendantHandle(
                pid=pid,
                parent_pid=entries[pid],
                created=created,
                depth=depth,
                handle=handle,
                can_terminate=can_terminate,
            )
            if (
                not can_terminate
                and int(
                    self.kernel32.WaitForSingleObject(
                        wintypes.HANDLE(handle),
                        0,
                    )
                )
                != 0
            ):
                complete = False
            added += 1
        return complete, added

    def terminate_and_wait(self, deadline: float) -> bool:
        stable_rounds = 0
        while time.monotonic() < deadline:
            discovered, added = self.discover()
            for descendant in sorted(
                self.handles.values(),
                key=lambda item: item.depth,
            ):
                wait_result = int(
                    self.kernel32.WaitForSingleObject(
                        wintypes.HANDLE(descendant.handle),
                        0,
                    )
                )
                if wait_result == 0:
                    continue
                if not descendant.can_terminate:
                    continue
                if not self.kernel32.TerminateProcess(
                    wintypes.HANDLE(descendant.handle),
                    1,
                ):
                    wait_result = int(
                        self.kernel32.WaitForSingleObject(
                            wintypes.HANDLE(descendant.handle),
                            0,
                        )
                    )
                    if wait_result != 0:
                        continue
            all_exited = all(
                int(
                    self.kernel32.WaitForSingleObject(
                        wintypes.HANDLE(descendant.handle),
                        0,
                    )
                )
                == 0
                for descendant in self.handles.values()
            )
            if discovered and added == 0 and all_exited:
                stable_rounds += 1
                if stable_rounds >= 2:
                    return True
            else:
                stable_rounds = 0
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return False

    def close(self) -> bool:
        closed = True
        for pid, descendant in list(self.handles.items()):
            if self.kernel32.CloseHandle(wintypes.HANDLE(descendant.handle)):
                del self.handles[pid]
            else:
                closed = False
        return closed


def collect_info(
    *,
    background: bool,
    backend: Win32Backend | None = None,
) -> dict[str, Any]:
    resolved_backend = backend or Win32Backend()
    specs = [
        ProviderSpec("identity", 2.0, resolved_backend.identity_session),
        ProviderSpec("clock", 1.0, _clock_provider),
        ProviderSpec("system", 2.0, resolved_backend.system_info),
        ProviderSpec("display", 2.0, resolved_backend.display_info),
        ProviderSpec("focus", 2.0, resolved_backend.focused_window),
        ProviderSpec("visible_apps", 3.0, resolved_backend.list_windows),
        ProviderSpec("media", 2.5, _media_provider),
        ProviderSpec(
            "network",
            NETWORK_COMMAND_BUDGET_SECONDS + EXTERNAL_CLEANUP_BUDGET_SECONDS,
            _network_provider,
        ),
        ProviderSpec(
            "readiness",
            WINAPP_PROBE_MAX_SECONDS + 0.5,
            lambda: _readiness_provider(resolved_backend),
        ),
    ]
    if background:
        specs.append(
            ProviderSpec("background", 3.0, resolved_backend.background_processes)
        )
    return _run_providers(specs)


def _run_providers(specs: list[ProviderSpec]) -> dict[str, Any]:
    started_at = time.monotonic()
    result_queues: dict[str, queue.Queue[dict[str, Any]]] = {}

    for spec in specs:
        output: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        result_queues[spec.name] = output

        def runner(
            provider: ProviderSpec = spec,
            destination: queue.Queue[dict[str, Any]] = output,
        ) -> None:
            try:
                result = provider.function()
            except ComputerError as exc:
                result = unavailable_section(exc.code)
            except Exception:  # noqa: BLE001 - provider isolation is the contract
                result = unavailable_section(f"{provider.name}_unavailable")
            try:
                destination.put_nowait(result)
            except queue.Full:
                pass

        threading.Thread(
            target=runner,
            name=f"agent-tools-computer-{spec.name}",
            daemon=True,
        ).start()

    sections: dict[str, Any] = {}
    for spec in specs:
        deadline = started_at + spec.timeout_seconds
        remaining = max(0.0, deadline - time.monotonic())
        try:
            sections[spec.name] = result_queues[spec.name].get(timeout=remaining)
        except queue.Empty:
            sections[spec.name] = unavailable_section(f"{spec.name}_timeout")
    return sections


def _clock_provider() -> dict[str, Any]:
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    offset_seconds = int(offset.total_seconds()) if offset is not None else None
    return available_section(
        local_iso=now.isoformat(timespec="seconds"),
        timezone=now.tzname(),
        utc_offset_seconds=offset_seconds,
    )


def _media_provider() -> dict[str, Any]:
    if sys.platform != "win32":
        return unavailable_section("unsupported_platform", active=False)
    try:
        media_spec = importlib.util.find_spec("winsdk.windows.media.control")
    except ModuleNotFoundError:
        media_spec = None
    if media_spec is None:
        return unavailable_section("media_backend_unavailable", active=False)

    async def read_media() -> dict[str, Any]:
        module = importlib.import_module("winsdk.windows.media.control")
        manager_type = module.GlobalSystemMediaTransportControlsSessionManager
        manager = await manager_type.request_async()
        session = manager.get_current_session()
        if session is None:
            return available_section(active=False, session=None)
        properties = await session.try_get_media_properties_async()
        playback = session.get_playback_info()
        raw_status = playback.playback_status
        status = getattr(raw_status, "name", None) or str(raw_status).rsplit(".", 1)[-1]
        return available_section(
            active=True,
            session={
                "app": bounded_text(session.source_app_user_model_id, MAX_PROCESS_NAME_CHARS),
                "playback_state": str(status).casefold(),
                "title": bounded_text(properties.title or "", MAX_TITLE_CHARS) or None,
                "artist": bounded_text(properties.artist or "", MAX_TITLE_CHARS) or None,
            },
        )

    return asyncio.run(read_media())


def _network_provider() -> dict[str, Any]:
    if sys.platform != "win32":
        return unavailable_section("unsupported_platform")

    deadline = time.monotonic() + NETWORK_COMMAND_BUDGET_SECONDS
    details_timeout = _remaining_timeout(deadline, 3.25)
    details = (
        _windows_network_details(timeout_seconds=details_timeout)
        if details_timeout is not None
        else {"error_code": "network_deadline_exceeded"}
    )
    wifi_timeout = _remaining_timeout(deadline, 0.75)
    wifi = (
        _wifi_details(timeout_seconds=wifi_timeout)
        if wifi_timeout is not None
        else unavailable_section("network_deadline_exceeded", connected=False)
    )
    tailscale_timeout = _remaining_timeout(deadline, 0.75)
    tailscale = (
        _tailscale_details(timeout_seconds=tailscale_timeout)
        if tailscale_timeout is not None
        else unavailable_section(
            "network_deadline_exceeded",
            present=_find_tailscale() is not None,
            status="unknown",
            addresses=[],
        )
    )
    primary_address = details.get("primary_address") or _primary_address()
    connectivity = str(details.get("connectivity") or "unknown").casefold()
    connected = connectivity in {"internet", "localnetwork", "connected"} or bool(
        primary_address
    )
    section = available_section(
        connected=connected,
        connectivity=connectivity,
        interface=details.get("interface"),
        interface_type=details.get("interface_type"),
        primary_address=primary_address,
        wifi=wifi,
        tailscale=tailscale,
    )
    if details.get("error_code"):
        section["details_error_code"] = details["error_code"]
    return section


def _windows_network_details(*, timeout_seconds: float = 3.25) -> dict[str, Any]:
    powershell = _system_executable(
        "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
    )
    if powershell is None:
        return {"error_code": "network_backend_unavailable"}
    script = (
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);"
        "$c=@(Get-NetIPConfiguration -ErrorAction Stop | "
        "Where-Object {$_.NetAdapter.Status -eq 'Up'} | "
        "Sort-Object @{Expression={if ($_.NetProfile.IPv4Connectivity "
        "-eq 'Internet') {0} else {1}}},"
        "InterfaceMetric) | Select-Object -First 1;"
        "if ($null -eq $c) { '{}' } else { [pscustomobject]@{"
        "connectivity=[string]$c.NetProfile.IPv4Connectivity;"
        "interface=[string]$c.InterfaceAlias;"
        "interface_type=[string]$c.NetAdapter.MediaType;"
        "primary_address=[string](($c.IPv4Address | Select-Object -First 1).IPAddress)"
        "} | ConvertTo-Json -Compress }"
    )
    result = _run_external(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
        timeout_seconds,
    )
    if result.error_code or result.returncode != 0:
        return {"error_code": result.error_code or "network_query_failed"}
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {"error_code": "network_response_invalid"}
    if not isinstance(payload, dict):
        return {"error_code": "network_response_invalid"}
    return {
        "connectivity": bounded_text(payload.get("connectivity") or "unknown", 64),
        "interface": bounded_text(payload.get("interface") or "", 128) or None,
        "interface_type": bounded_text(payload.get("interface_type") or "", 64) or None,
        "primary_address": _validated_ip(payload.get("primary_address")),
    }


def _primary_address() -> str | None:
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.connect(("192.0.2.1", 9))
        return _validated_ip(connection.getsockname()[0])
    except OSError:
        return None
    finally:
        connection.close()


def _wifi_details(*, timeout_seconds: float = 0.75) -> dict[str, Any]:
    netsh = _system_executable("System32", "netsh.exe")
    if netsh is None:
        return unavailable_section("wifi_backend_unavailable", connected=False)
    result = _run_external(
        [str(netsh), "wlan", "show", "interfaces"], timeout_seconds
    )
    if result.error_code:
        return unavailable_section(result.error_code, connected=False)
    text = result.stdout
    if result.returncode != 0:
        return unavailable_section("wifi_query_failed", connected=False)
    interfaces = _parse_wifi_interfaces(text)
    if not interfaces:
        if "there is no wireless interface" in text.casefold():
            return available_section(connected=False, ssid=None, signal_percent=None)
        return unavailable_section("wifi_response_unrecognized", connected=False)
    fields = next(
        (item for item in interfaces if item.get("state", "").casefold() == "connected"),
        interfaces[0],
    )
    state = fields.get("state", "").casefold()
    ssid = fields.get("ssid")
    signal_text = fields.get("signal", "").rstrip("%")
    try:
        signal = int(signal_text) if signal_text else None
    except ValueError:
        signal = None
    return available_section(
        connected=state == "connected",
        ssid=bounded_text(ssid, 128) if ssid else None,
        signal_percent=signal if signal is None or 0 <= signal <= 100 else None,
    )


def _parse_wifi_interfaces(text: str) -> list[dict[str, str]]:
    interfaces: list[dict[str, str]] = []
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if "state" in fields:
                interfaces.append(fields)
            fields = {}
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().casefold()
        if normalized_key == "name" and "state" in fields:
            interfaces.append(fields)
            fields = {}
        fields[normalized_key] = value.strip()
    if "state" in fields:
        interfaces.append(fields)
    return interfaces


def _tailscale_details(*, timeout_seconds: float = 0.75) -> dict[str, Any]:
    executable = _find_tailscale()
    if executable is None:
        return unavailable_section(
            "tailscale_not_installed",
            present=False,
            status="not_installed",
            addresses=[],
        )
    result = _run_external([str(executable), "status", "--json"], timeout_seconds)
    if result.error_code:
        return unavailable_section(
            result.error_code,
            present=True,
            status="unknown",
            addresses=[],
        )
    if result.returncode != 0:
        return available_section(present=True, status="stopped", addresses=[])
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return unavailable_section(
            "tailscale_response_invalid",
            present=True,
            status="unknown",
            addresses=[],
        )
    if not isinstance(payload, dict):
        return unavailable_section(
            "tailscale_response_invalid",
            present=True,
            status="unknown",
            addresses=[],
        )
    self_data = payload.get("Self") if isinstance(payload.get("Self"), dict) else {}
    raw_addresses = self_data.get("TailscaleIPs", []) if isinstance(self_data, dict) else []
    addresses = [address for address in map(_validated_ip, raw_addresses[:4]) if address]
    return available_section(
        present=True,
        status=bounded_text(payload.get("BackendState") or "unknown", 64).casefold(),
        addresses=addresses,
    )


def _find_tailscale() -> Path | None:
    if sys.platform != "win32":
        return None
    try:
        winreg = importlib.import_module("winreg")
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Services\Tailscale",
        ) as key:
            image_path, _value_type = winreg.QueryValueEx(key, "ImagePath")
    except (OSError, ImportError):
        return None
    service_path = _service_executable_path(str(image_path))
    if service_path is None:
        return None
    client_path = service_path.with_name("tailscale.exe")
    return client_path if client_path.is_file() else None


def _readiness_provider(backend: Win32Backend) -> dict[str, Any]:
    uia_probe = WinAppAdapter(backend).probe()
    mutations = action_capabilities()
    identity = backend.identity_session()
    desktop_usable = bool(
        identity.get("available")
        and identity.get("session_id") is not None
        and identity.get("interactive_desktop_access") is True
        and identity.get("lock_state") == "unlocked"
        and str(identity.get("desktop_name") or "").casefold() == "default"
    )
    actions_available = bool(mutations["available"] and desktop_usable)
    if not desktop_usable:
        action_blocked_reason = "secure_desktop_unavailable"
    elif not mutations["available"]:
        action_blocked_reason = "computer_actions_disabled"
    else:
        action_blocked_reason = None
    capabilities = ["info", "windows", "focused", "screenshot", "capabilities"]
    if uia_probe.get("available"):
        capabilities.extend(
            ["inspect", "read", "scroll-areas", "invoke", "set-value", "scroll"]
        )
    capabilities.extend(["focus", "resize", "notify"])
    return available_section(
        agent_tools_version=__version__,
        read_only=False,
        action_lock=mutations["mutex"],
        actions_available=actions_available,
        actions_disabled=mutations["disabled"],
        action_blocked_reason=action_blocked_reason,
        desktop_usable=desktop_usable,
        lock_state=identity.get("lock_state"),
        lock_ui_foreground=identity.get("lock_ui_foreground"),
        action_busy_behavior=mutations["busy_behavior"],
        physical_input=False,
        capabilities=capabilities,
        backends={
            "win32": backend.probe(),
            "uia_winapp": uia_probe,
        },
    )


def _validated_ip(value: object) -> str | None:
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(str(value)))
    except ValueError:
        return None


def _remaining_timeout(deadline: float, maximum: float) -> float | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    return min(maximum, remaining)


def _windows_directory() -> Path | None:
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetWindowsDirectoryW.argtypes = (wintypes.LPWSTR, wintypes.UINT)
    kernel32.GetWindowsDirectoryW.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32_768)
    length = int(kernel32.GetWindowsDirectoryW(buffer, len(buffer)))
    if length <= 0 or length >= len(buffer):
        return None
    return Path(buffer.value)


def _system_executable(*relative_parts: str) -> Path | None:
    windows_directory = _windows_directory()
    if windows_directory is None:
        return None
    candidate = windows_directory.joinpath(*relative_parts)
    return candidate if candidate.is_file() else None


def _service_executable_path(value: str) -> Path | None:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith('"'):
        end_quote = stripped.find('"', 1)
        if end_quote <= 1:
            return None
        executable = stripped[1:end_quote]
    else:
        executable = stripped.split(maxsplit=1)[0]
    path = Path(executable)
    if not path.is_absolute():
        return None
    return path


def _create_external_process_job() -> _WindowsProcessJob | None:
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPDWORD,
    )
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = int(kernel32.CreateJobObjectW(None, None) or 0)
    if handle <= 0:
        return None
    job = _WindowsProcessJob(handle=handle, kernel32=kernel32)
    information = _JobObjectExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        wintypes.HANDLE(handle),
        JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        job.close()
        return None
    return job


def _windows_process_tree_kernel32() -> Any:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessEntry32W),
    )
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessEntry32W),
    )
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.GetSystemTimeAsFileTime.argtypes = (ctypes.POINTER(_FileTime),)
    kernel32.GetSystemTimeAsFileTime.restype = None
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _filetime_value(value: _FileTime) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def _process_created_filetime(kernel32: Any, handle: int) -> int | None:
    created = _FileTime()
    exited = _FileTime()
    kernel = _FileTime()
    user = _FileTime()
    if not kernel32.GetProcessTimes(
        wintypes.HANDLE(handle),
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        return None
    return _filetime_value(created)


def _system_filetime(kernel32: Any) -> int | None:
    value = _FileTime()
    try:
        kernel32.GetSystemTimeAsFileTime(ctypes.byref(value))
    except OSError:
        return None
    result = _filetime_value(value)
    return result if result > 0 else None


def _process_parent_snapshot(kernel32: Any) -> dict[int, int] | None:
    snapshot = int(kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) or 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot <= 0 or snapshot == int(invalid_handle or -1):
        return None
    entries: dict[int, int] = {}
    closed = True
    try:
        entry = _ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(_ProcessEntry32W)
        available = bool(
            kernel32.Process32FirstW(wintypes.HANDLE(snapshot), ctypes.byref(entry))
        )
        while available:
            entries[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            available = bool(
                kernel32.Process32NextW(
                    wintypes.HANDLE(snapshot),
                    ctypes.byref(entry),
                )
            )
    finally:
        closed = bool(kernel32.CloseHandle(wintypes.HANDLE(snapshot)))
    return entries if closed else None


def _run_external(args: list[str], timeout_seconds: float) -> ExternalResult:
    process_job = _create_external_process_job()
    if sys.platform == "win32" and process_job is None:
        return ExternalResult(None, "", "external_command_unavailable")
    try:
        result = _run_external_in_job(args, timeout_seconds, process_job)
    except BaseException:
        if process_job is not None:
            _finalize_external_process_job(process_job)
        raise
    if process_job is not None and not _finalize_external_process_job(process_job):
        return ExternalResult(None, "", "external_termination_failed")
    return result


def _run_external_in_job(
    args: list[str],
    timeout_seconds: float,
    process_job: _WindowsProcessJob | None,
) -> ExternalResult:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    if process_job is not None:
        creationflags |= getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=creationflags,
        )
    except OSError:
        return ExternalResult(None, "", "external_command_unavailable")
    if process_job is not None and not process_job.assign(process):
        _terminate_external_process(
            process,
            time.monotonic() + EXTERNAL_CLEANUP_BUDGET_SECONDS,
        )
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        return ExternalResult(None, "", "external_termination_failed")
    if process_job is not None and not _resume_external_process(process):
        _terminate_external_process(
            process,
            time.monotonic() + EXTERNAL_CLEANUP_BUDGET_SECONDS,
            process_job,
        )
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        return ExternalResult(None, "", "external_termination_failed")
    if process.stdout is None or process.stderr is None:
        if not _terminate_external_process(
            process,
            time.monotonic() + EXTERNAL_CLEANUP_BUDGET_SECONDS,
            process_job,
        ):
            return ExternalResult(None, "", "external_termination_failed")
        return ExternalResult(None, "", "external_pipe_unavailable")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    closed_streams: set[str] = set()
    total_bytes = 0

    def drain_available() -> str | None:
        nonlocal total_bytes
        for name, stream in streams.items():
            while name not in closed_streams:
                try:
                    chunk, closed = _read_pipe_available(stream)
                except OSError:
                    return "external_pipe_read_failed"
                if closed:
                    closed_streams.add(name)
                if not chunk:
                    break
                if total_bytes + len(chunk) > MAX_EXTERNAL_OUTPUT_BYTES:
                    return "external_output_too_large"
                total_bytes += len(chunk)
                buffers[name].extend(chunk)
        return None

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    failure_code: str | None = None
    try:
        while process.poll() is None:
            failure_code = drain_available()
            if failure_code is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            time.sleep(min(0.01, remaining))

        cleanup_deadline = time.monotonic() + EXTERNAL_CLEANUP_BUDGET_SECONDS
        if process.poll() is None and not _terminate_external_process(
            process,
            cleanup_deadline,
            process_job,
        ):
            return ExternalResult(None, "", "external_termination_failed")
        try:
            process.wait(timeout=_cleanup_remaining(cleanup_deadline))
        except (OSError, subprocess.TimeoutExpired):
            return ExternalResult(None, "", "external_termination_failed")
        if process_job is not None and not process_job.terminate_and_wait(cleanup_deadline):
            return ExternalResult(None, "", "external_termination_failed")
        drain_error = drain_available()
        if failure_code is None:
            failure_code = drain_error
    finally:
        for stream in streams.values():
            try:
                stream.close()
            except OSError:
                pass
    if timed_out:
        return ExternalResult(None, "", "external_command_timeout")
    if failure_code is not None:
        return ExternalResult(process.returncode, "", failure_code)
    return ExternalResult(process.returncode, _decode_output(bytes(buffers["stdout"])))


def _terminate_external_process(
    process: subprocess.Popen[bytes],
    deadline: float,
    process_job: _WindowsProcessJob | None = None,
) -> bool:
    descendants: _WindowsDescendantTree | None = None
    descendants_ready = True
    if sys.platform == "win32":
        descendants = _WindowsDescendantTree.capture(process)
        if descendants is None:
            descendants_ready = False
        else:
            descendants.discover()
    tree_reaped = True
    if process_job is not None:
        tree_reaped = process_job.terminate_and_wait(deadline)
    else:
        try:
            process.kill()
        except OSError:
            pass
    descendants_reaped = True
    descendants_closed = True
    if descendants is not None:
        descendants_reaped = descendants.terminate_and_wait(deadline)
        descendants_closed = descendants.close()
    try:
        process.wait(timeout=_cleanup_remaining(deadline))
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(
        descendants_ready
        and descendants_reaped
        and descendants_closed
        and tree_reaped
        and process.poll() is not None
    )


def _finalize_external_process_job(process_job: _WindowsProcessJob) -> bool:
    deadline = time.monotonic() + EXTERNAL_CLEANUP_BUDGET_SECONDS
    root_process = getattr(process_job, "root_process", None)
    if root_process is None:
        reaped = process_job.terminate_and_wait(deadline)
    else:
        reaped = _terminate_external_process(
            root_process,
            deadline,
            process_job,
        )
    closed = process_job.close()
    if not closed and time.monotonic() < deadline:
        closed = process_job.close()
    return reaped and closed


def _resume_external_process(process: subprocess.Popen[bytes]) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Thread32First.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ThreadEntry32),
    )
    kernel32.Thread32First.restype = wintypes.BOOL
    kernel32.Thread32Next.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ThreadEntry32),
    )
    kernel32.Thread32Next.restype = wintypes.BOOL
    kernel32.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.ResumeThread.argtypes = (wintypes.HANDLE,)
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = int(kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) or 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot <= 0 or snapshot == int(invalid_handle or -1):
        return False
    thread_handle = 0
    resumed = False
    handles_closed = True
    try:
        entry = _ThreadEntry32()
        entry.dwSize = ctypes.sizeof(_ThreadEntry32)
        available = bool(
            kernel32.Thread32First(wintypes.HANDLE(snapshot), ctypes.byref(entry))
        )
        while available:
            if int(entry.th32OwnerProcessID) == int(process.pid):
                thread_handle = int(
                    kernel32.OpenThread(
                        THREAD_SUSPEND_RESUME,
                        False,
                        int(entry.th32ThreadID),
                    )
                    or 0
                )
                break
            available = bool(
                kernel32.Thread32Next(wintypes.HANDLE(snapshot), ctypes.byref(entry))
            )
        if thread_handle > 0:
            previous_suspend_count = int(
                kernel32.ResumeThread(wintypes.HANDLE(thread_handle))
            )
            resumed = previous_suspend_count == 1
    finally:
        if thread_handle > 0:
            handles_closed = bool(
                kernel32.CloseHandle(wintypes.HANDLE(thread_handle))
            )
        handles_closed = bool(
            kernel32.CloseHandle(wintypes.HANDLE(snapshot))
        ) and handles_closed
    return resumed and handles_closed


def _read_pipe_available(stream: Any) -> tuple[bytes, bool]:
    if sys.platform == "win32":
        return _read_windows_pipe_available(stream)
    readable, _, _ = select.select([stream], [], [], 0)
    if not readable:
        return b"", False
    chunk = os.read(stream.fileno(), 4096)
    return chunk, not chunk


def _read_windows_pipe_available(stream: Any) -> tuple[bytes, bool]:
    msvcrt = importlib.import_module("msvcrt")
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


def _decode_output(value: bytes) -> str:
    if not value:
        return ""
    encodings = ["utf-8-sig"]
    if b"\x00" in value[:100]:
        encodings.append("utf-16")
    encodings.append(locale.getpreferredencoding(False))
    for encoding in encodings:
        try:
            return value.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return value.decode("utf-8", errors="replace")


def _cleanup_remaining(deadline: float) -> float:
    return max(0.001, deadline - time.monotonic())
