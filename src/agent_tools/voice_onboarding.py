from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import os
import platform
import secrets
import shutil
import signal
import socket
import ssl
import stat
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, TextIO

from agent_tools.tts_server import load_owner_only_bearer_token

VOICE_CONFIG_SCHEMA = "ai-nd-co.agent-tools.voice/v1"
VOICE_STATUS_SCHEMA = "ai-nd-co.agent-tools.voice-status/v1"
PAIRING_VERSION = 1
PAIRING_MIN_SECONDS = 15
PAIRING_MAX_SECONDS = 900
PAIRING_DEFAULT_SECONDS = 300
PAIRING_MIN_PORT = 49_152
PAIRING_MAX_PORT = 65_535
MAX_OWNER_OUTPUT_BYTES = 64 * 1024
MAX_VOICE_CONFIG_BYTES = 64 * 1024
MAX_PAIR_REQUEST_BYTES = 8 * 1024
MAX_PAIR_RESPONSE_BYTES = 32 * 1024
MAX_LABEL_CHARS = 64
MAX_ALIAS_CHARS = 32
PAIR_PATH = "/pair/v1/claim"
ACK_PATH = "/pair/v1/ack"
_VIRTUAL_ADAPTER_TERMS = (
    "bluetooth",
    "docker",
    "hyper-v",
    "loopback",
    "openvpn",
    "ppp",
    "tap",
    "tunnel",
    "virtual",
    "vmware",
    "vpn",
    "wireguard",
    "wsl",
)


class VoiceError(RuntimeError):
    def __init__(
        self, code: str, message: str, next_action: str, *, impact: str = "Voice setup stopped."
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.impact = impact
        self.next_action = next_action

    def to_public_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "cause": self.message,
            "impact": self.impact,
            "nextAction": self.next_action,
        }


@dataclass(frozen=True)
class AdapterAddress:
    address: str
    preferred: bool = True


@dataclass(frozen=True)
class AdapterSnapshot:
    interface_index: int
    name: str
    description: str
    kind: str
    operational: bool
    physical: bool
    gateway: str | None
    route_metric: int | None
    addresses: tuple[AdapterAddress, ...]


@dataclass(frozen=True)
class TailscaleSnapshot:
    state: str
    addresses: tuple[str, ...] = ()
    executable_sha256: str | None = None


@dataclass(frozen=True)
class NetworkSelection:
    address: str
    network_class: str
    interface_name: str
    route_metric: int | None
    confirmation_required: bool = False

    def to_public_dict(self) -> dict[str, object]:
        return {
            "address": self.address,
            "class": self.network_class,
            "interface": self.interface_name,
            "routeMetric": self.route_metric,
            "confirmationRequired": self.confirmation_required,
        }


class NetworkBackend(Protocol):
    def adapters(self) -> tuple[AdapterSnapshot, ...]: ...

    def tailscale(self) -> TailscaleSnapshot: ...


def _rfc1918(value: str) -> bool:
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError:
        return False
    return (
        address in ipaddress.IPv4Network("10.0.0.0/8")
        or address in ipaddress.IPv4Network("172.16.0.0/12")
        or address in ipaddress.IPv4Network("192.168.0.0/16")
    )


def _tailscale_ipv4(value: str) -> bool:
    try:
        return ipaddress.IPv4Address(value) in ipaddress.IPv4Network("100.64.0.0/10")
    except ipaddress.AddressValueError:
        return False


def _exact_private_endpoint(value: str, *, allow_loopback: bool = False) -> bool:
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError:
        return False
    if address.is_unspecified or address.is_multicast or address.is_link_local:
        return False
    if address.is_loopback:
        return allow_loopback and value == "127.0.0.1"
    return _rfc1918(value) or _tailscale_ipv4(value)


def _adapter_is_virtual(adapter: AdapterSnapshot) -> bool:
    text = f"{adapter.name} {adapter.description}".casefold()
    return any(term in text for term in _VIRTUAL_ADAPTER_TERMS)


def select_network(
    backend: NetworkBackend,
    *,
    mode: str,
    bind_ip: str | None = None,
) -> NetworkSelection:
    if mode not in {"auto", "tailscale", "lan"}:
        raise VoiceError(
            "network_mode_invalid", "Network mode is invalid.", "Use auto, tailscale, or lan."
        )
    adapters = backend.adapters()
    if bind_ip is not None:
        return _select_explicit_bind(adapters, bind_ip, mode)

    if mode in {"auto", "tailscale"}:
        snapshot = backend.tailscale()
        if snapshot.state == "healthy":
            if len(snapshot.addresses) != 1 or not _tailscale_ipv4(snapshot.addresses[0]):
                raise VoiceError(
                    "tailscale_address_ambiguous",
                    "Tailscale did not report one usable IPv4 address.",
                    "Run Tailscale repair, then retry voice setup.",
                )
            address = snapshot.addresses[0]
            matches = [
                adapter
                for adapter in adapters
                if adapter.operational
                and "tailscale" in f"{adapter.name} {adapter.description}".casefold()
                and any(item.preferred and item.address == address for item in adapter.addresses)
            ]
            if len(matches) != 1:
                raise VoiceError(
                    "tailscale_adapter_mismatch",
                    "The Tailscale address does not match one active adapter.",
                    "Restart Tailscale and retry voice setup.",
                )
            return NetworkSelection(address, "tailscale", matches[0].name, matches[0].route_metric)
        if mode == "tailscale":
            raise VoiceError(
                f"tailscale_{snapshot.state}",
                "Tailscale is not ready with one private IPv4 address.",
                "Start Tailscale, then retry voice setup.",
            )
        if snapshot.state not in {"absent", "stopped", "no_address"}:
            raise VoiceError(
                "tailscale_state_ambiguous",
                "Tailscale state could not be proved safe.",
                "Repair Tailscale or choose --network lan explicitly.",
            )

    return _select_lan(adapters)


def _select_explicit_bind(
    adapters: Sequence[AdapterSnapshot],
    bind_ip: str,
    mode: str,
) -> NetworkSelection:
    if bind_ip in {"0.0.0.0", "127.0.0.1"} or not _exact_private_endpoint(bind_ip):
        raise VoiceError(
            "bind_ip_not_private",
            "The requested bind address is not an exact private IPv4 address.",
            "Choose the current Tailscale address or a physical RFC1918 LAN address.",
        )
    matches = [
        adapter
        for adapter in adapters
        if adapter.operational
        and any(item.preferred and item.address == bind_ip for item in adapter.addresses)
    ]
    if len(matches) != 1:
        raise VoiceError(
            "bind_ip_not_current",
            "The requested bind address is not current on exactly one active adapter.",
            "Choose an address shown by the current active adapter.",
        )
    adapter = matches[0]
    is_tailscale = _tailscale_ipv4(bind_ip) and "tailscale" in (
        f"{adapter.name} {adapter.description}".casefold()
    )
    if mode == "tailscale" and not is_tailscale:
        raise VoiceError(
            "bind_ip_not_tailscale",
            "The requested address is not on the active Tailscale adapter.",
            "Choose the current Tailscale IPv4 address.",
        )
    if is_tailscale and mode != "lan":
        return NetworkSelection(bind_ip, "tailscale", adapter.name, adapter.route_metric)
    if (
        mode == "tailscale"
        or not _rfc1918(bind_ip)
        or not adapter.physical
        or adapter.kind not in {"ethernet", "wifi"}
        or adapter.gateway is None
        or adapter.gateway == "0.0.0.0"
        or adapter.route_metric is None
        or _adapter_is_virtual(adapter)
    ):
        raise VoiceError(
            "bind_ip_not_trusted_lan",
            "The requested LAN address is not current on a physical default-route adapter.",
            "Choose an active Ethernet or Wi-Fi RFC1918 address with a default gateway.",
        )
    return NetworkSelection(
        bind_ip,
        "trusted_lan",
        adapter.name,
        adapter.route_metric,
        confirmation_required=True,
    )


def _select_lan(adapters: Sequence[AdapterSnapshot]) -> NetworkSelection:
    candidates: list[tuple[AdapterSnapshot, str]] = []
    for adapter in adapters:
        if (
            not adapter.operational
            or not adapter.physical
            or adapter.kind not in {"ethernet", "wifi"}
            or adapter.gateway in {None, "0.0.0.0"}
            or adapter.route_metric is None
            or _adapter_is_virtual(adapter)
        ):
            continue
        addresses = sorted(
            item.address for item in adapter.addresses if item.preferred and _rfc1918(item.address)
        )
        if len(addresses) > 1:
            raise VoiceError(
                "lan_address_ambiguous",
                "An active LAN adapter has multiple usable private addresses.",
                "Retry with --bind-ip using the intended current address.",
            )
        if addresses:
            candidates.append((adapter, addresses[0]))
    if not candidates:
        raise VoiceError(
            "private_network_unavailable",
            "No active physical private LAN route is available.",
            "Connect Tailscale, Ethernet, or Wi-Fi and retry.",
        )
    lowest = min(
        adapter.route_metric for adapter, _ in candidates if adapter.route_metric is not None
    )
    winners = [
        (adapter, address) for adapter, address in candidates if adapter.route_metric == lowest
    ]
    if len(winners) != 1:
        raise VoiceError(
            "lan_route_ambiguous",
            "Multiple private LAN routes have the same preferred metric.",
            "Retry with --bind-ip using the intended current address.",
        )
    adapter, address = winners[0]
    return NetworkSelection(
        address,
        "trusted_lan",
        adapter.name,
        adapter.route_metric,
        confirmation_required=True,
    )


class WindowsNetworkBackend:
    def adapters(self) -> tuple[AdapterSnapshot, ...]:
        if sys.platform != "win32":
            raise VoiceError(
                "windows_required",
                "Voice network discovery requires Windows.",
                "Run this command on the target Windows PC.",
            )
        return _windows_adapters()

    def tailscale(self) -> TailscaleSnapshot:
        if sys.platform != "win32":
            return TailscaleSnapshot("absent")
        executable = _resolve_tailscale_executable()
        if executable is None:
            return TailscaleSnapshot("absent")
        before = _regular_file_sha256(executable, "tailscale_identity_invalid")
        status = _bounded_subprocess([str(executable), "status", "--json"], timeout=10.0)
        addresses = _bounded_subprocess([str(executable), "ip", "-4"], timeout=10.0)
        after = _regular_file_sha256(executable, "tailscale_identity_invalid")
        if before != after:
            raise VoiceError(
                "tailscale_identity_changed",
                "The Tailscale executable changed during discovery.",
                "Finish the Tailscale update, then retry.",
            )
        if status.returncode != 0:
            return TailscaleSnapshot("stopped", executable_sha256=before)
        try:
            payload = json.loads(status.stdout.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise VoiceError(
                "tailscale_output_invalid",
                "Tailscale returned invalid status data.",
                "Update or repair Tailscale, then retry.",
            ) from exc
        if not isinstance(payload, dict) or payload.get("BackendState") != "Running":
            return TailscaleSnapshot("stopped", executable_sha256=before)
        if addresses.returncode != 0:
            return TailscaleSnapshot("no_address", executable_sha256=before)
        try:
            values = tuple(
                line.strip()
                for line in addresses.stdout.decode("ascii").splitlines()
                if line.strip()
            )
        except UnicodeError as exc:
            raise VoiceError(
                "tailscale_output_invalid",
                "Tailscale returned invalid address data.",
                "Update or repair Tailscale, then retry.",
            ) from exc
        return TailscaleSnapshot("healthy", values, before)


@dataclass(frozen=True)
class _ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


_WINDOWS_OWNER_WRAPPER = "\n".join(
    (
        "import base64,json,subprocess,sys,win32event",
        "status=win32event.WaitForSingleObject(int(sys.argv[1]),win32event.INFINITE)",
        "if status != win32event.WAIT_OBJECT_0: raise SystemExit(125)",
        "encoded=sys.argv[2]+'='*(-len(sys.argv[2])%4)",
        "command=json.loads(base64.urlsafe_b64decode(encoded.encode('ascii')).decode('utf-8'))",
        "completed=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=sys.stdout.buffer,"
        "stderr=sys.stderr.buffer,shell=False,check=False)",
        "raise SystemExit(completed.returncode)",
    )
)


def _windows_kill_job(process: subprocess.Popen[bytes]) -> object:
    try:
        import win32api  # type: ignore[import-untyped]
        import win32job  # type: ignore[import-untyped]

        job = win32job.CreateJobObject(None, "")
        limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(
            job,
            win32job.JobObjectExtendedLimitInformation,
            limits,
        )
        process_handle = process._handle  # type: ignore[attr-defined]
        win32job.AssignProcessToJobObject(job, int(process_handle))
        return job
    except Exception as exc:
        if "job" in locals():
            win32api.CloseHandle(job)
        raise VoiceError(
            "child_process_failed",
            "A required owner command could not be contained.",
            "Verify Windows process containment, then retry.",
        ) from exc


def _spawn_contained_process(
    command: Sequence[str],
) -> tuple[subprocess.Popen[bytes], object | None]:
    if sys.platform != "win32":
        return (
            subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                env=_minimal_child_environment(),
                start_new_session=True,
            ),
            None,
        )

    import pywintypes  # type: ignore[import-untyped]
    import win32api
    import win32event  # type: ignore[import-untyped]

    security = pywintypes.SECURITY_ATTRIBUTES()
    security.bInheritHandle = True
    release_event = win32event.CreateEvent(security, False, False, "")
    process: subprocess.Popen[bytes] | None = None
    containment: object | None = None
    try:
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {"handle_list": [int(release_event)]}
        encoded_command = _b64url(
            json.dumps(list(command), ensure_ascii=True, separators=(",", ":")).encode("ascii")
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WINDOWS_OWNER_WRAPPER,
                str(int(release_event)),
                encoded_command,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=_minimal_child_environment(),
            startupinfo=startup,
            close_fds=True,
        )
        containment = _windows_kill_job(process)
        win32event.SetEvent(release_event)
        return process, containment
    except Exception as exc:
        if process is not None:
            _terminate_child_tree(process, containment)
        if isinstance(exc, VoiceError):
            raise
        raise VoiceError(
            "child_process_failed",
            "A required owner command could not be contained.",
            "Verify Windows process containment, then retry.",
        ) from exc
    finally:
        win32api.CloseHandle(release_event)


def _terminate_child_tree(process: subprocess.Popen[bytes], containment: object | None) -> None:
    if sys.platform == "win32":
        if containment is not None:
            import win32api

            win32api.CloseHandle(containment)
        elif process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.poll() is None:
                process.kill()
    if process.poll() is None:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


def _bounded_subprocess(command: Sequence[str], *, timeout: float) -> _ProcessResult:
    try:
        process, containment = _spawn_contained_process(command)
    except OSError as exc:
        raise VoiceError(
            "child_process_failed",
            "A required owner command did not complete.",
            "Verify the installed command and retry.",
        ) from exc
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("owner command pipes unavailable")

    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()
    read_failed = threading.Event()

    def drain(stream: Any, destination: bytearray) -> None:
        try:
            while chunk := stream.read(8192):
                remaining = MAX_OWNER_OUTPUT_BYTES + 1 - len(destination)
                if remaining > 0:
                    destination.extend(chunk[:remaining])
                if len(destination) > MAX_OWNER_OUTPUT_BYTES:
                    overflow.set()
        except OSError:
            read_failed.set()

    readers = [
        threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout
    timed_out = False
    while process.poll() is None:
        if overflow.is_set() or read_failed.is_set():
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        overflow.wait(min(0.05, remaining))

    inherited_pipes = False
    if process.poll() is not None and not overflow.is_set() and not read_failed.is_set():
        for reader in readers:
            reader.join(0.5)
        inherited_pipes = any(reader.is_alive() for reader in readers)
    _terminate_child_tree(process, containment)
    for reader in readers:
        reader.join(2.0)
    for reader, stream in zip(readers, (process.stdout, process.stderr), strict=True):
        if not reader.is_alive():
            stream.close()

    if overflow.is_set():
        raise VoiceError(
            "child_output_too_large",
            "A required owner command returned too much output.",
            "Repair that lifecycle owner before retrying.",
        )
    if (
        timed_out
        or inherited_pipes
        or read_failed.is_set()
        or any(reader.is_alive() for reader in readers)
    ):
        raise VoiceError(
            "child_process_failed",
            "A required owner command did not complete.",
            "Verify the installed command and retry.",
        )
    return _ProcessResult(process.returncode, bytes(stdout), bytes(stderr))


def _minimal_child_environment() -> dict[str, str]:
    allowed = ("SystemRoot", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT", "COMSPEC")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment["PYTHONUTF8"] = "1"
    return environment


def _resolve_tailscale_executable() -> Path | None:
    candidates: list[Path] = []
    for name in ("ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value) / "Tailscale" / "tailscale.exe")
    try:
        import winreg

        for root, key_name in (
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Tailscale IPN"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Tailscale IPN"),
        ):
            try:
                with winreg.OpenKey(root, key_name) as key:
                    install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                candidates.append(Path(str(install_dir)) / "tailscale.exe")
            except OSError:
                continue
    except ImportError:
        pass
    matches: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.lstat()
            if stat.S_ISREG(metadata.st_mode) and not _is_reparse(resolved, metadata):
                matches.append(resolved)
        except OSError:
            continue
    unique = {os.path.normcase(str(path)): path for path in matches}
    if len(unique) > 1:
        raise VoiceError(
            "tailscale_binary_ambiguous",
            "Multiple trusted Tailscale installations were found.",
            "Remove the stale installation, then retry.",
        )
    return next(iter(unique.values()), None)


def _windows_adapters() -> tuple[AdapterSnapshot, ...]:
    import ctypes
    from ctypes import wintypes

    af_inet = 2
    gaa_flag_include_prefix = 0x0010
    gaa_flag_include_gateways = 0x0080
    error_buffer_overflow = 111

    class SocketAddress(ctypes.Structure):
        _fields_ = [("address", ctypes.c_void_p), ("length", ctypes.c_int)]

    class UnicastAddress(ctypes.Structure):
        _fields_ = [
            ("alignment", ctypes.c_ulonglong),
            ("next", ctypes.c_void_p),
            ("socket_address", SocketAddress),
            ("prefix_origin", ctypes.c_int),
            ("suffix_origin", ctypes.c_int),
            ("dad_state", ctypes.c_int),
            ("valid_lifetime", wintypes.ULONG),
            ("preferred_lifetime", wintypes.ULONG),
            ("lease_lifetime", wintypes.ULONG),
            ("on_link_prefix_length", ctypes.c_ubyte),
        ]

    class GatewayAddress(ctypes.Structure):
        _fields_ = [
            ("alignment", ctypes.c_ulonglong),
            ("next", ctypes.c_void_p),
            ("socket_address", SocketAddress),
        ]

    class AdapterAddresses(ctypes.Structure):
        _fields_ = [
            ("alignment", ctypes.c_ulonglong),
            ("next", ctypes.c_void_p),
            ("adapter_name", ctypes.c_char_p),
            ("first_unicast", ctypes.c_void_p),
            ("first_anycast", ctypes.c_void_p),
            ("first_multicast", ctypes.c_void_p),
            ("first_dns", ctypes.c_void_p),
            ("dns_suffix", wintypes.LPWSTR),
            ("description", wintypes.LPWSTR),
            ("friendly_name", wintypes.LPWSTR),
            ("physical_address", ctypes.c_ubyte * 8),
            ("physical_address_length", wintypes.ULONG),
            ("flags", wintypes.ULONG),
            ("mtu", wintypes.ULONG),
            ("if_type", wintypes.ULONG),
            ("oper_status", wintypes.ULONG),
            ("ipv6_if_index", wintypes.ULONG),
            ("zone_indices", wintypes.ULONG * 16),
            ("first_prefix", ctypes.c_void_p),
            ("transmit_link_speed", ctypes.c_ulonglong),
            ("receive_link_speed", ctypes.c_ulonglong),
            ("first_wins", ctypes.c_void_p),
            ("first_gateway", ctypes.c_void_p),
            ("ipv4_metric", wintypes.ULONG),
            ("ipv6_metric", wintypes.ULONG),
            ("luid", ctypes.c_ulonglong),
        ]

    iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)
    size = wintypes.ULONG(15_000)
    for _ in range(3):
        buffer = ctypes.create_string_buffer(size.value)
        result = iphlpapi.GetAdaptersAddresses(
            af_inet,
            gaa_flag_include_prefix | gaa_flag_include_gateways,
            None,
            buffer,
            ctypes.byref(size),
        )
        if result == 0:
            break
        if result != error_buffer_overflow:
            raise VoiceError(
                "adapter_query_failed",
                "Windows network adapters could not be inspected.",
                "Retry after Windows networking is ready.",
            )
    else:
        raise VoiceError(
            "adapter_query_failed",
            "Windows network adapters changed during inspection.",
            "Retry after Windows networking settles.",
        )

    route_metrics = _windows_default_route_metrics()
    snapshots: list[AdapterSnapshot] = []
    pointer = ctypes.addressof(buffer)
    while pointer:
        adapter = ctypes.cast(pointer, ctypes.POINTER(AdapterAddresses)).contents
        addresses: list[AdapterAddress] = []
        unicast_pointer = adapter.first_unicast
        while unicast_pointer:
            item = ctypes.cast(unicast_pointer, ctypes.POINTER(UnicastAddress)).contents
            address = _sockaddr_ipv4(item.socket_address.address, item.socket_address.length)
            if address is not None:
                addresses.append(
                    AdapterAddress(address, item.dad_state == 4 and item.valid_lifetime != 0)
                )
            unicast_pointer = item.next
        gateways: list[str] = []
        gateway_pointer = adapter.first_gateway
        while gateway_pointer:
            gateway_item = ctypes.cast(gateway_pointer, ctypes.POINTER(GatewayAddress)).contents
            address = _sockaddr_ipv4(
                gateway_item.socket_address.address,
                gateway_item.socket_address.length,
            )
            if address not in {None, "0.0.0.0"}:
                gateways.append(address)
            gateway_pointer = gateway_item.next
        kind = {6: "ethernet", 71: "wifi", 24: "loopback", 131: "tunnel", 23: "ppp"}.get(
            int(adapter.if_type), "other"
        )
        # Alignment stores Length in the low 32 bits and IfIndex in the high 32 bits.
        if_index = int((adapter.alignment >> 32) & 0xFFFFFFFF)
        has_mac = int(adapter.physical_address_length) > 0
        snapshots.append(
            AdapterSnapshot(
                interface_index=if_index,
                name=adapter.friendly_name or "",
                description=adapter.description or "",
                kind=kind,
                operational=int(adapter.oper_status) == 1,
                physical=_windows_adapter_is_physical(if_index, kind, has_mac),
                gateway=gateways[0] if len(gateways) == 1 else None,
                route_metric=route_metrics.get(if_index),
                addresses=tuple(addresses),
            )
        )
        pointer = adapter.next
    return tuple(snapshots)


def _physical_adapter_metadata_is_authoritative(
    *,
    kind: str,
    has_mac: bool,
    hardware_interface: bool,
    physical_medium: int,
    access_type: int,
) -> bool:
    expected_medium = {"ethernet": 14, "wifi": 9}.get(kind)
    return (
        has_mac
        and hardware_interface
        and expected_medium is not None
        and physical_medium == expected_medium
        and access_type == 2
    )


def _windows_adapter_is_physical(interface_index: int, kind: str, has_mac: bool) -> bool:
    import ctypes
    from ctypes import wintypes

    class Guid(ctypes.Structure):
        _fields_ = [
            ("data1", wintypes.DWORD),
            ("data2", wintypes.WORD),
            ("data3", wintypes.WORD),
            ("data4", ctypes.c_ubyte * 8),
        ]

    class IfRow2(ctypes.Structure):
        _fields_ = [
            ("interface_luid", ctypes.c_ulonglong),
            ("interface_index", wintypes.ULONG),
            ("interface_guid", Guid),
            ("alias", wintypes.WCHAR * 257),
            ("description", wintypes.WCHAR * 257),
            ("physical_address_length", wintypes.ULONG),
            ("physical_address", ctypes.c_ubyte * 32),
            ("permanent_physical_address", ctypes.c_ubyte * 32),
            ("mtu", wintypes.ULONG),
            ("if_type", wintypes.ULONG),
            ("tunnel_type", wintypes.ULONG),
            ("media_type", wintypes.ULONG),
            ("physical_medium_type", wintypes.ULONG),
            ("access_type", wintypes.ULONG),
            ("direction_type", wintypes.ULONG),
            ("interface_flags", ctypes.c_ubyte),
            ("oper_status", wintypes.ULONG),
            ("admin_status", wintypes.ULONG),
            ("media_connect_state", wintypes.ULONG),
            ("network_guid", Guid),
            ("connection_type", wintypes.ULONG),
            ("transmit_link_speed", ctypes.c_ulonglong),
            ("receive_link_speed", ctypes.c_ulonglong),
            *[
                (name, ctypes.c_ulonglong)
                for name in (
                    "in_octets",
                    "in_ucast_pkts",
                    "in_nucast_pkts",
                    "in_discards",
                    "in_errors",
                    "in_unknown_protos",
                    "in_ucast_octets",
                    "in_multicast_octets",
                    "in_broadcast_octets",
                    "out_octets",
                    "out_ucast_pkts",
                    "out_nucast_pkts",
                    "out_discards",
                    "out_errors",
                    "out_ucast_octets",
                    "out_multicast_octets",
                    "out_broadcast_octets",
                    "out_qlen",
                )
            ],
        ]

    row = IfRow2()
    row.interface_index = interface_index
    try:
        result = ctypes.WinDLL("iphlpapi", use_last_error=True).GetIfEntry2(ctypes.byref(row))
    except (AttributeError, OSError):
        return False
    if result != 0:
        return False
    return _physical_adapter_metadata_is_authoritative(
        kind=kind,
        has_mac=has_mac and int(row.physical_address_length) > 0,
        hardware_interface=bool(int(row.interface_flags) & 0x01),
        physical_medium=int(row.physical_medium_type),
        access_type=int(row.access_type),
    )


def _sockaddr_ipv4(pointer: int | None, length: int) -> str | None:
    if not pointer or length < 8:
        return None
    import ctypes

    raw = ctypes.string_at(pointer, length)
    if int.from_bytes(raw[:2], sys.byteorder) != socket.AF_INET:
        return None
    return socket.inet_ntop(socket.AF_INET, raw[4:8])


def _windows_default_route_metrics() -> dict[int, int]:
    import ctypes
    from ctypes import wintypes

    error_insufficient_buffer = 122

    class ForwardRow(ctypes.Structure):
        _fields_ = [
            (name, wintypes.DWORD)
            for name in (
                "destination",
                "mask",
                "policy",
                "next_hop",
                "if_index",
                "route_type",
                "protocol",
                "age",
                "next_hop_as",
                "metric1",
                "metric2",
                "metric3",
                "metric4",
                "metric5",
            )
        ]

    iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)
    size = wintypes.ULONG(0)
    result = iphlpapi.GetIpForwardTable(None, ctypes.byref(size), False)
    if result not in {0, error_insufficient_buffer}:
        raise VoiceError(
            "route_query_failed",
            "Windows default routes could not be inspected.",
            "Retry after Windows networking is ready.",
        )
    buffer = ctypes.create_string_buffer(size.value)
    result = iphlpapi.GetIpForwardTable(buffer, ctypes.byref(size), False)
    if result != 0:
        raise VoiceError(
            "route_query_failed",
            "Windows default routes could not be inspected.",
            "Retry after Windows networking is ready.",
        )
    count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD))[0]
    base = ctypes.addressof(buffer) + ctypes.sizeof(wintypes.DWORD)
    metrics: dict[int, int] = {}
    for index in range(int(count)):
        row = ForwardRow.from_address(base + index * ctypes.sizeof(ForwardRow))
        if row.destination != 0 or row.mask != 0 or row.next_hop == 0 or row.metric1 == 0xFFFFFFFF:
            continue
        current = metrics.get(int(row.if_index))
        if current is None or int(row.metric1) < current:
            metrics[int(row.if_index)] = int(row.metric1)
    return metrics


def _default_voice_root() -> Path:
    override = os.environ.get("AGENT_TOOLS_VOICE_STATE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise VoiceError(
                "protected_storage_unavailable",
                "The owner-local application directory is unavailable.",
                "Sign in to the Windows owner profile and retry.",
            )
        return Path(local) / "ai-nd-co" / "agent-tools" / "voice"
    return (
        Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        / "agent-tools"
        / "voice"
    )


def _is_reparse(path: Path, metadata: os.stat_result) -> bool:
    if path.is_symlink():
        return True
    return bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _protect_windows_path(path: Path) -> None:
    try:
        import win32api
        import win32con  # type: ignore[import-untyped]
        import win32security  # type: ignore[import-untyped]
    except ImportError as exc:
        raise VoiceError(
            "protected_storage_unavailable",
            "Windows protected storage support is unavailable.",
            "Install AgentTools with its Windows dependencies, then retry.",
        ) from exc
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    owner_sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    acl = win32security.ACL()
    acl.AddAccessAllowedAce(win32security.ACL_REVISION, win32con.GENERIC_ALL, owner_sid)
    win32security.SetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION
        | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        owner_sid,
        None,
        acl,
        None,
    )


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or _is_reparse(path, metadata):
        raise VoiceError(
            "protected_storage_invalid",
            "Voice protected storage is not a stable directory.",
            "Remove the unsafe link or replacement, then retry.",
        )
    if sys.platform == "win32":
        _protect_windows_path(path)
    else:
        os.chmod(path, 0o700)
    _verify_owner_only(path, metadata=path.lstat())


def _verify_owner_only(path: Path, *, metadata: os.stat_result | None = None) -> None:
    from agent_tools.tts_server import _verify_owner_only_permissions

    current = metadata or path.lstat()
    try:
        _verify_owner_only_permissions(path, current)
    except OSError as exc:
        raise VoiceError(
            "protected_storage_invalid",
            "Voice protected storage permissions are unsafe.",
            "Restore owner-only permissions, then retry.",
        ) from exc


def _write_private_bytes(path: Path, data: bytes, *, replace: bool) -> None:
    _ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if sys.platform == "win32":
            _protect_windows_path(temporary)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise
            finally:
                temporary.unlink(missing_ok=True)
        if sys.platform == "win32":
            _protect_windows_path(path)
        else:
            os.chmod(path, 0o600)
        _verify_owner_only(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_private_json(path: Path, payload: Mapping[str, object], *, replace: bool = True) -> None:
    _write_private_bytes(path, _canonical_json(payload), replace=replace)


def _read_private_bytes(path: Path, maximum: int) -> bytes:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or _is_reparse(path, before)
            or not 0 < before.st_size <= maximum
        ):
            raise OSError("unsafe file")
        _verify_owner_only(path, metadata=before)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(opened):
                raise OSError("file changed")
            raw = os.read(descriptor, maximum + 1)
        finally:
            os.close(descriptor)
        after = path.lstat()
        if (
            _file_identity(before) != _file_identity(after)
            or len(raw) != before.st_size
            or _is_reparse(path, after)
        ):
            raise OSError("file changed")
        _verify_owner_only(path, metadata=after)
        return raw
    except (OSError, VoiceError) as exc:
        if isinstance(exc, VoiceError):
            raise
        raise VoiceError(
            "protected_file_invalid",
            "A voice configuration file is missing, unsafe, or changed during use.",
            "Run agent-tools voice repair after restoring the protected file.",
        ) from exc


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


@contextmanager
def _interprocess_file_lock(
    path: Path,
    *,
    busy_code: str,
    busy_message: str,
    busy_next_action: str,
    timeout: float,
) -> Iterator[None]:
    _ensure_private_directory(path.parent)
    if not path.exists():
        try:
            _write_private_bytes(path, b"\0", replace=False)
        except FileExistsError:
            pass
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_reparse(path, current)
            or _file_identity(opened) != _file_identity(current)
            or opened.st_size != 1
        ):
            raise VoiceError(
                "protected_storage_invalid",
                "A voice transaction guard has an unsafe identity.",
                "Inspect the protected voice state before retrying.",
            )
        _verify_owner_only(path, metadata=current)
        deadline = time.monotonic() + timeout
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise VoiceError(
                        busy_code,
                        busy_message,
                        busy_next_action,
                    ) from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _regular_file_sha256(path: Path, code: str) -> str:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or _is_reparse(path, before):
            raise OSError("unsafe file")
        digest = hashlib.sha256()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(opened):
                raise OSError("file changed")
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        after = path.lstat()
        if _file_identity(before) != _file_identity(after) or _is_reparse(path, after):
            raise OSError("file changed")
        return digest.hexdigest()
    except OSError as exc:
        raise VoiceError(
            code,
            "A required executable or configuration identity is unsafe.",
            "Restore the exact lifecycle owner files, then retry.",
        ) from exc


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str, *, expected_bytes: int | None = None) -> bytes:
    if (
        not value
        or len(value) > 2048
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in value
        )
    ):
        raise ValueError("invalid base64url")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64url") from exc
    if _b64url(raw) != value:
        raise ValueError("noncanonical base64url")
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise ValueError("invalid decoded length")
    return raw


def _validated_label(value: str | None) -> str:
    label = (value or f"{platform.node() or 'Home'} desktop").strip()
    if (
        not label
        or len(label) > MAX_LABEL_CHARS
        or any(unicodedata.category(character).startswith("C") for character in label)
    ):
        raise VoiceError(
            "label_invalid", "The server label is invalid.", "Use 1-64 visible characters."
        )
    return label


def _validated_alias(value: str | None, fingerprint: str) -> str:
    alias = (value or f"desktop-{fingerprint[:6]}").strip().casefold()
    if (
        not alias
        or len(alias) > MAX_ALIAS_CHARS
        or not alias[0].isalpha()
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in alias)
        or alias.endswith("-")
        or "--" in alias
    ):
        raise VoiceError(
            "alias_invalid",
            "The spoken alias is invalid.",
            "Use a unique lowercase alias with letters, numbers, and single hyphens.",
        )
    return alias


@dataclass(frozen=True)
class ServerIdentity:
    private_key: Any
    public_key: bytes
    fingerprint: str
    server_id: str


def load_or_create_identity(path: Path, *, create: bool = True) -> ServerIdentity:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:
        raise VoiceError(
            "voice_dependencies_missing",
            "Voice pairing cryptography is unavailable.",
            "Reinstall AgentTools with its current dependencies.",
        ) from exc
    if path.exists():
        raw = _read_private_bytes(path, 8192)
        try:
            payload = json.loads(raw.decode("ascii"))
            if set(payload) != {"v", "key"} or payload["v"] != 1:
                raise ValueError("schema")
            private_key = Ed25519PrivateKey.from_private_bytes(
                _b64url_decode(payload["key"], expected_bytes=32)
            )
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise VoiceError(
                "identity_invalid",
                "The PC voice identity is invalid.",
                "Restore its protected backup, or remove it only to create a new PC identity.",
            ) from exc
    elif create:
        private_key = Ed25519PrivateKey.generate()
        seed = private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        try:
            _write_private_json(path, {"v": 1, "key": _b64url(seed)}, replace=False)
        except FileExistsError:
            return load_or_create_identity(path, create=create)
    else:
        raise VoiceError(
            "identity_missing",
            "The PC voice identity is missing.",
            "Restore the protected identity before checking or repairing voice setup.",
        )
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    fingerprint = hashlib.sha256(public_key).hexdigest()
    prefix = base64.b32encode(bytes.fromhex(fingerprint)).decode("ascii").rstrip("=").lower()[:16]
    return ServerIdentity(private_key, public_key, fingerprint, f"srv_{prefix}")


@dataclass(frozen=True)
class OwnerCommand:
    name: str
    executable: str
    executable_sha256: str
    prefix: tuple[str, ...]
    entry_file: str | None
    entry_sha256: str | None
    config_file: str
    config_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> OwnerCommand:
        if not isinstance(value, dict):
            raise ValueError("owner")
        expected = {
            "name",
            "executable",
            "executableSha256",
            "prefix",
            "entryFile",
            "entrySha256",
            "configFile",
            "configSha256",
        }
        if set(value) != expected or value["name"] not in {"codex", "tts"}:
            raise ValueError("owner")
        prefix = value["prefix"]
        if (
            not isinstance(prefix, list)
            or len(prefix) > 8
            or any(not isinstance(item, str) or len(item) > 1024 for item in prefix)
        ):
            raise ValueError("prefix")
        entry_file = value["entryFile"]
        entry_sha256 = value["entrySha256"]
        if (entry_file is None) != (entry_sha256 is None):
            raise ValueError("entry")
        result = cls(
            name=value["name"],
            executable=value["executable"],
            executable_sha256=value["executableSha256"],
            prefix=tuple(prefix),
            entry_file=entry_file,
            entry_sha256=entry_sha256,
            config_file=value["configFile"],
            config_sha256=value["configSha256"],
        )
        for path_value in (result.executable, result.config_file, result.entry_file):
            if path_value is not None and not Path(path_value).is_absolute():
                raise ValueError("path")
        for digest in (result.executable_sha256, result.config_sha256, result.entry_sha256):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("sha")
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "executable": self.executable,
            "executableSha256": self.executable_sha256,
            "prefix": list(self.prefix),
            "entryFile": self.entry_file,
            "entrySha256": self.entry_sha256,
            "configFile": self.config_file,
            "configSha256": self.config_sha256,
        }


@dataclass(frozen=True)
class VoiceConfig:
    root: Path
    server_id: str
    label: str
    alias: str
    identity_fingerprint: str
    network: NetworkSelection
    codex_endpoint: str
    codex_token_file: Path
    codex_owner: OwnerCommand
    tts_origin: str | None
    tts_token_file: Path | None
    tts_owner: OwnerCommand | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": VOICE_CONFIG_SCHEMA,
            "server": {
                "id": self.server_id,
                "label": self.label,
                "alias": self.alias,
                "identityFingerprint": self.identity_fingerprint,
            },
            "network": self.network.to_public_dict(),
            "codex": {
                "endpoint": self.codex_endpoint,
                "tokenFile": str(self.codex_token_file),
                "owner": self.codex_owner.to_dict(),
            },
            "tts": None
            if self.tts_owner is None
            else {
                "origin": self.tts_origin,
                "tokenFile": str(self.tts_token_file),
                "owner": self.tts_owner.to_dict(),
            },
        }


def _validate_service_url(
    value: object,
    *,
    scheme: str,
    address: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError("service URL")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != scheme
        or parsed.hostname != address
        or parsed.port in {None, 0}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("service URL")
    return value


def load_voice_config(root: Path) -> VoiceConfig:
    path = root / "voice.json"
    raw = _read_private_bytes(path, MAX_VOICE_CONFIG_BYTES)
    try:
        payload = json.loads(raw.decode("ascii"))
        if not isinstance(payload, dict) or set(payload) != {
            "schemaVersion",
            "server",
            "network",
            "codex",
            "tts",
        }:
            raise ValueError("schema")
        if payload["schemaVersion"] != VOICE_CONFIG_SCHEMA:
            raise ValueError("version")
        server = payload["server"]
        network = payload["network"]
        codex = payload["codex"]
        if not isinstance(server, dict) or set(server) != {
            "id",
            "label",
            "alias",
            "identityFingerprint",
        }:
            raise ValueError("server")
        if not isinstance(network, dict) or set(network) != {
            "address",
            "class",
            "interface",
            "routeMetric",
            "confirmationRequired",
        }:
            raise ValueError("network")
        if not isinstance(codex, dict) or set(codex) != {"endpoint", "tokenFile", "owner"}:
            raise ValueError("codex")
        network_selection = NetworkSelection(
            address=network["address"],
            network_class=network["class"],
            interface_name=network["interface"],
            route_metric=network["routeMetric"],
            confirmation_required=network["confirmationRequired"],
        )
        if (
            not isinstance(network_selection.address, str)
            or not isinstance(network_selection.interface_name, str)
            or not network_selection.interface_name
            or any(
                unicodedata.category(character).startswith("C")
                for character in network_selection.interface_name
            )
            or (
                network_selection.route_metric is not None
                and (
                    type(network_selection.route_metric) is not int
                    or network_selection.route_metric < 0
                )
            )
            or type(network_selection.confirmation_required) is not bool
            or network_selection.network_class not in {"tailscale", "trusted_lan"}
            or network_selection.confirmation_required
            != (network_selection.network_class == "trusted_lan")
            or not _exact_private_endpoint(network_selection.address)
        ):
            raise ValueError("network")
        fingerprint = server["identityFingerprint"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("fingerprint")
        codex_endpoint = _validate_service_url(
            codex["endpoint"],
            scheme="ws",
            address=network_selection.address,
        )
        tts = payload["tts"]
        tts_origin: str | None = None
        tts_token_file: Path | None = None
        tts_owner: OwnerCommand | None = None
        if tts is not None:
            if not isinstance(tts, dict) or set(tts) != {"origin", "tokenFile", "owner"}:
                raise ValueError("tts")
            tts_origin = _validate_service_url(
                tts["origin"],
                scheme="http",
                address=network_selection.address,
            )
            tts_token_file = Path(tts["tokenFile"])
            tts_owner = OwnerCommand.from_dict(tts["owner"])
        result = VoiceConfig(
            root=root,
            server_id=server["id"],
            label=_validated_label(server["label"]),
            alias=_validated_alias(server["alias"], fingerprint),
            identity_fingerprint=fingerprint,
            network=network_selection,
            codex_endpoint=codex_endpoint,
            codex_token_file=Path(codex["tokenFile"]),
            codex_owner=OwnerCommand.from_dict(codex["owner"]),
            tts_origin=tts_origin,
            tts_token_file=tts_token_file,
            tts_owner=tts_owner,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeError) as exc:
        raise VoiceError(
            "voice_config_invalid",
            "The voice setup configuration is invalid.",
            "Restore the protected configuration or run voice setup again after review.",
        ) from exc
    _verify_owner_command(result.codex_owner)
    if not result.codex_token_file.is_absolute():
        raise VoiceError(
            "voice_config_invalid",
            "The voice setup configuration is invalid.",
            "Restore the protected configuration or run voice setup again after review.",
        )
    _load_voice_token(result.codex_token_file)
    if result.tts_owner is not None:
        _verify_owner_command(result.tts_owner)
        if result.tts_token_file is None or not result.tts_token_file.is_absolute():
            raise VoiceError(
                "voice_config_invalid",
                "The voice setup configuration is invalid.",
                "Restore the protected configuration or run voice setup again after review.",
            )
        _load_voice_token(result.tts_token_file)
    identity = load_or_create_identity(root / "identity.json", create=False)
    if (
        identity.fingerprint != result.identity_fingerprint
        or identity.server_id != result.server_id
    ):
        raise VoiceError(
            "identity_mismatch",
            "The PC identity does not match the voice configuration.",
            "Restore the matching protected identity and configuration.",
        )
    return result


def _verify_owner_command(owner: OwnerCommand) -> None:
    if (
        _regular_file_sha256(Path(owner.executable), "owner_executable_invalid")
        != owner.executable_sha256
    ):
        raise VoiceError(
            "owner_executable_changed",
            f"The {owner.name} lifecycle executable changed.",
            "Repair or restage the exact lifecycle owner before retrying.",
        )
    if _regular_file_sha256(Path(owner.config_file), "owner_config_invalid") != owner.config_sha256:
        raise VoiceError(
            "owner_config_changed",
            f"The {owner.name} lifecycle configuration changed.",
            "Run voice setup again only after reviewing the lifecycle configuration.",
        )
    if (
        owner.entry_file is not None
        and _regular_file_sha256(Path(owner.entry_file), "owner_entry_invalid")
        != owner.entry_sha256
    ):
        raise VoiceError(
            "owner_entry_changed",
            f"The {owner.name} lifecycle entry changed.",
            "Repair or restage the exact lifecycle owner before retrying.",
        )


class LifecycleInvoker:
    def invoke(self, owner: OwnerCommand, action: str, *, plan: bool = False) -> dict[str, object]:
        _verify_owner_command(owner)
        if action not in {"install", "status", "start", "restart", "stop", "uninstall"}:
            raise VoiceError(
                "owner_action_invalid",
                "A lifecycle action is invalid.",
                "Update AgentTools before retrying.",
            )
        command = [owner.executable, *owner.prefix, action, "--config", owner.config_file]
        if plan:
            if owner.name != "tts" or action == "status":
                raise VoiceError(
                    "owner_plan_unavailable",
                    f"The {owner.name} lifecycle owner does not support this plan.",
                    "Use voice setup --plan for the composed plan.",
                )
            command.append("--plan")
        completed = _bounded_subprocess(command, timeout=300.0)
        if completed.returncode != 0:
            raise VoiceError(
                f"{owner.name}_owner_failed",
                f"The {owner.name} lifecycle owner refused or failed the action.",
                f"Run agent-tools voice status, repair the {owner.name} owner, then retry.",
            )
        try:
            payload = json.loads(completed.stdout.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise VoiceError(
                f"{owner.name}_owner_output_invalid",
                f"The {owner.name} lifecycle owner returned invalid status.",
                f"Repair the {owner.name} lifecycle owner, then retry.",
            ) from exc
        if not isinstance(payload, dict):
            raise VoiceError(
                f"{owner.name}_owner_output_invalid",
                f"The {owner.name} lifecycle owner returned invalid status.",
                f"Repair the {owner.name} lifecycle owner, then retry.",
            )
        return payload


@dataclass(frozen=True)
class OwnerHealth:
    code: str
    healthy: bool
    exact: bool
    registered: bool
    owned: bool
    listening: bool
    changed: bool = False

    def to_public_dict(self) -> dict[str, object]:
        return asdict(self)


def _codex_health(payload: Mapping[str, object], *, changed: bool = False) -> OwnerHealth:
    try:
        task = payload["task"]
        runtime = payload["runtime"]
        if not isinstance(task, dict) or not isinstance(runtime, dict):
            raise TypeError
        app_server = runtime["appServer"]
        if not isinstance(app_server, dict):
            raise TypeError
        registered = task["registered"] is True
        exact = task["compatible"] is True and task["definitionExact"] is True
        owned = task["owned"] is True and task["installationActive"] is True
        listening = app_server["listening"] is True
        healthy = (
            registered
            and exact
            and owned
            and listening
            and app_server["authenticatedReady"] is True
        )
        if healthy:
            code = "ready"
        elif registered and (not exact or not owned):
            code = "foreign_task"
        elif listening:
            code = "unknown_listener"
        elif registered:
            code = "not_ready"
        else:
            code = "not_installed"
        return OwnerHealth(code, healthy, exact, registered, owned, listening, changed)
    except (KeyError, TypeError) as exc:
        raise VoiceError(
            "codex_owner_output_invalid",
            "The Codex lifecycle owner returned an unknown status schema.",
            "Update the lifecycle owner and AgentTools together.",
        ) from exc


def _tts_health(payload: Mapping[str, object], *, changed: bool = False) -> OwnerHealth:
    try:
        status = payload["status"]
        if not isinstance(status, dict):
            raise TypeError
        registered = status["task_present"] is True
        exact = (
            status["task_exact"] is True
            and status["config_exact"] is True
            and status["credential_exact"] is True
        )
        owned = exact and status["runner_alive"] is True and status["child_alive"] is True
        listening = status["listener_owned"] is True
        healthy = owned and listening and status["authenticated_ready"] is True
        code = "ready" if healthy else str(status.get("code", payload.get("code", "not_ready")))
        return OwnerHealth(code, healthy, exact, registered, owned, listening, changed)
    except (KeyError, TypeError) as exc:
        raise VoiceError(
            "tts_owner_output_invalid",
            "The TTS lifecycle owner returned an unknown status schema.",
            "Update the lifecycle owner and AgentTools together.",
        ) from exc


def inspect_owners(
    config: VoiceConfig, invoker: LifecycleInvoker
) -> tuple[OwnerHealth, OwnerHealth | None]:
    codex = _codex_health(invoker.invoke(config.codex_owner, "status"))
    tts = (
        _tts_health(invoker.invoke(config.tts_owner, "status"))
        if config.tts_owner is not None
        else None
    )
    return codex, tts


def _owner_spec_path(root: Path) -> Path:
    override = os.environ.get("AGENT_TOOLS_VOICE_OWNER_SPEC")
    return Path(override).resolve() if override else root / "owner-spec.json"


def load_owner_spec(
    root: Path,
    selection: NetworkSelection,
    *,
    include_tts: bool,
) -> tuple[OwnerCommand, Path, str, OwnerCommand | None, Path | None, str | None]:
    path = _owner_spec_path(root)
    if not path.exists():
        raise VoiceError(
            "codex_owner_contract_unavailable",
            "The Codex startup owner has not staged an exact-address setup contract.",
            "Install the compatible codex-appserver-startup owner, then retry voice setup.",
        )
    raw = _read_private_bytes(path, MAX_VOICE_CONFIG_BYTES)
    try:
        payload = json.loads(raw.decode("ascii"))
        if (
            not isinstance(payload, dict)
            or set(payload) != {"v", "network", "codex", "tts"}
            or payload["v"] != 1
        ):
            raise ValueError("schema")
        if payload["network"] != {"address": selection.address, "class": selection.network_class}:
            raise ValueError("network")
        codex = payload["codex"]
        if not isinstance(codex, dict) or set(codex) != {"owner", "endpoint", "tokenFile"}:
            raise ValueError("codex")
        codex_owner = OwnerCommand.from_dict(codex["owner"])
        if codex_owner.name != "codex":
            raise ValueError("codex owner")
        codex_endpoint = _validate_service_url(
            codex["endpoint"],
            scheme="ws",
            address=selection.address,
        )
        codex_token = Path(codex["tokenFile"])
        if not codex_token.is_absolute():
            raise ValueError("codex token")
        tts_owner: OwnerCommand | None = None
        tts_token: Path | None = None
        tts_origin: str | None = None
        if include_tts and payload["tts"] is not None:
            tts = payload["tts"]
            if not isinstance(tts, dict) or set(tts) != {"owner", "origin", "tokenFile"}:
                raise ValueError("tts")
            tts_owner = OwnerCommand.from_dict(tts["owner"])
            if tts_owner.name != "tts":
                raise ValueError("tts owner")
            tts_origin = _validate_service_url(
                tts["origin"],
                scheme="http",
                address=selection.address,
            )
            tts_token = Path(tts["tokenFile"])
            if not tts_token.is_absolute():
                raise ValueError("tts token")
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise VoiceError(
            "owner_spec_invalid",
            "The staged lifecycle owner contract is invalid or targets another address.",
            "Restage the exact-address owner contract, then retry.",
        ) from exc
    _verify_owner_command(codex_owner)
    _load_voice_token(codex_token)
    if tts_owner is not None and tts_token is not None:
        _verify_owner_command(tts_owner)
        _load_voice_token(tts_token)
    return codex_owner, codex_token, codex_endpoint, tts_owner, tts_token, tts_origin


def _load_voice_token(path: Path) -> str:
    try:
        return load_owner_only_bearer_token(path)
    except RuntimeError as exc:
        raise VoiceError(
            "credential_unavailable",
            "A required protected service credential is unavailable.",
            "Repair the exact lifecycle owner and retry.",
        ) from exc


@dataclass(frozen=True)
class VoiceStatus:
    code: str
    configured: bool
    codex: OwnerHealth | None
    tts: OwnerHealth | None
    network: NetworkSelection | None
    pairing_active: bool

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": VOICE_STATUS_SCHEMA,
            "status": self.code,
            "configured": self.configured,
            "codex": None if self.codex is None else self.codex.to_public_dict(),
            "tts": None if self.tts is None else self.tts.to_public_dict(),
            "network": None if self.network is None else self.network.to_public_dict(),
            "pairing": {"active": self.pairing_active},
        }


def voice_status(root: Path, invoker: LifecycleInvoker) -> VoiceStatus:
    config_path = root / "voice.json"
    if not config_path.exists():
        return VoiceStatus("not_configured", False, None, None, None, False)
    config = load_voice_config(root)
    codex, tts = inspect_owners(config, invoker)
    if codex.healthy:
        code = "ready" if tts is None or tts.healthy else "ready_local_tts"
    elif codex.code in {"unknown_listener", "foreign_task"}:
        code = "unsafe"
    else:
        code = "needs_repair"
    return VoiceStatus(code, True, codex, tts, config.network, _pairing_record_active(root))


def _read_pairing_record(root: Path) -> dict[str, object] | None:
    record = root / "pairing" / "active.json"
    if not record.exists():
        return None
    try:
        raw = _read_private_bytes(record, 4096)
        payload = json.loads(raw.decode("ascii"))
        if (
            not isinstance(payload, dict)
            or set(payload) != {"v", "expiresAt", "pid", "processGeneration", "sessionHash"}
            or payload["v"] != 1
            or type(payload["expiresAt"]) is not int
            or type(payload["pid"]) is not int
            or payload["pid"] <= 0
            or not isinstance(payload["processGeneration"], str)
            or not payload["processGeneration"]
            or not isinstance(payload["sessionHash"], str)
            or len(payload["sessionHash"]) != 64
            or any(character not in "0123456789abcdef" for character in payload["sessionHash"])
        ):
            raise ValueError("record")
        return payload
    except (VoiceError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise VoiceError(
            "pairing_record_invalid",
            "The protected pairing activity record is invalid.",
            "Inspect the protected pairing state before creating another invitation.",
        ) from exc


def _pairing_record_active(root: Path) -> bool:
    payload = _read_pairing_record(root)
    if payload is None:
        return False
    pid = payload["pid"]
    expires_at = payload["expiresAt"]
    assert isinstance(pid, int)
    assert isinstance(expires_at, int)
    if expires_at <= int(time.time()):
        return False
    observed = _process_generation(pid)
    return observed is not None and hmac.compare_digest(
        observed,
        str(payload["processGeneration"]),
    )


def _process_generation(pid: int) -> str | None:
    if os.name == "nt":
        try:
            from agent_tools.tts_startup import WindowsRuntimeBackend

            process = WindowsRuntimeBackend().inspect_process(pid)
            return None if process is None else str(process.creation_time_ns)
        except RuntimeError:
            return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
        return fields[21]
    except (OSError, IndexError, UnicodeError):
        return None


def _setup_journal(
    root: Path,
    phase: str,
    config: VoiceConfig,
    *,
    created_identity: bool,
    codex_changed: bool = False,
    tts_changed: bool = False,
) -> None:
    _write_private_json(
        root / "setup-transaction.json",
        {
            "v": 1,
            "phase": phase,
            "createdIdentity": created_identity,
            "codexChanged": codex_changed,
            "ttsChanged": tts_changed,
            "codexOwner": config.codex_owner.to_dict(),
            "ttsOwner": None if config.tts_owner is None else config.tts_owner.to_dict(),
        },
    )


def _remove_owned_file(path: Path) -> None:
    if not path.exists():
        return
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or _is_reparse(path, metadata):
        raise VoiceError(
            "owned_cleanup_refused",
            "A transaction artifact changed identity before cleanup.",
            "Inspect the protected voice state before retrying.",
        )
    _verify_owner_only(path, metadata=metadata)
    path.unlink()


def _read_setup_lock(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(_read_private_bytes(path, 4096).decode("ascii"))
        if (
            not isinstance(payload, dict)
            or set(payload) != {"v", "pid", "processGeneration", "lockId"}
            or payload["v"] != 1
            or type(payload["pid"]) is not int
            or payload["pid"] <= 0
            or not isinstance(payload["processGeneration"], str)
            or not payload["processGeneration"]
            or not isinstance(payload["lockId"], str)
            or len(payload["lockId"]) != 43
        ):
            raise ValueError("lock")
        _b64url_decode(payload["lockId"], expected_bytes=32)
        return payload
    except (VoiceError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise VoiceError(
            "setup_lock_invalid",
            "The protected voice setup lock is invalid.",
            "Inspect the protected voice state before retrying setup.",
        ) from exc


@contextmanager
def _voice_setup_lock(root: Path) -> Iterator[None]:
    generation = _process_generation(os.getpid())
    if generation is None:
        raise VoiceError(
            "setup_process_identity_unavailable",
            "The voice setup process identity could not be verified.",
            "Retry from the installed AgentTools process on Windows.",
        )
    path = root / "setup.lock"
    lock_id = _b64url(secrets.token_bytes(32))
    record = {
        "v": 1,
        "pid": os.getpid(),
        "processGeneration": generation,
        "lockId": lock_id,
    }
    with _interprocess_file_lock(
        root / ".setup.guard",
        busy_code="setup_already_running",
        busy_message="Another voice setup transaction is active.",
        busy_next_action="Wait for that transaction to finish before retrying.",
        timeout=0.0,
    ):
        if path.exists():
            previous = _read_setup_lock(path)
            previous_pid = previous["pid"]
            assert isinstance(previous_pid, int)
            observed = _process_generation(previous_pid)
            if observed is not None and hmac.compare_digest(
                observed,
                str(previous["processGeneration"]),
            ):
                raise VoiceError(
                    "setup_already_running",
                    "Another voice setup transaction is active.",
                    "Wait for that transaction to finish before retrying.",
                )
            _remove_owned_file(path)
        try:
            _write_private_json(path, record, replace=False)
            yield
        finally:
            if not path.exists():
                raise VoiceError(
                    "setup_lock_changed",
                    "The voice setup transaction lock disappeared.",
                    "Inspect the protected voice state before retrying.",
                )
            current = _read_setup_lock(path)
            if not hmac.compare_digest(str(current["lockId"]), lock_id):
                raise VoiceError(
                    "setup_lock_changed",
                    "The voice setup transaction lock changed identity.",
                    "Inspect the protected voice state before retrying.",
                )
            _remove_owned_file(path)


def _read_setup_journal(root: Path) -> dict[str, object] | None:
    path = root / "setup-transaction.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(_read_private_bytes(path, MAX_VOICE_CONFIG_BYTES).decode("ascii"))
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {
                "v",
                "phase",
                "createdIdentity",
                "codexChanged",
                "ttsChanged",
                "codexOwner",
                "ttsOwner",
            }
            or payload["v"] != 1
            or not isinstance(payload["phase"], str)
            or type(payload["createdIdentity"]) is not bool
            or type(payload["codexChanged"]) is not bool
            or type(payload["ttsChanged"]) is not bool
        ):
            raise ValueError("journal")
        codex_owner = OwnerCommand.from_dict(payload["codexOwner"])
        tts_owner = (
            None if payload["ttsOwner"] is None else OwnerCommand.from_dict(payload["ttsOwner"])
        )
        if codex_owner.name != "codex" or (tts_owner is not None and tts_owner.name != "tts"):
            raise ValueError("journal owner")
        return {**payload, "codexOwner": codex_owner, "ttsOwner": tts_owner}
    except (VoiceError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise VoiceError(
            "setup_journal_invalid",
            "The durable voice setup journal is invalid.",
            "Inspect the protected voice state before retrying setup.",
        ) from exc


def _recover_setup_transaction(root: Path, invoker: LifecycleInvoker) -> None:
    journal = _read_setup_journal(root)
    if journal is None:
        return
    if (root / "voice.json").exists():
        load_voice_config(root)
        _remove_owned_file(root / "setup-transaction.json")
        return
    tts_owner = journal["ttsOwner"]
    codex_owner = journal["codexOwner"]
    assert tts_owner is None or isinstance(tts_owner, OwnerCommand)
    assert isinstance(codex_owner, OwnerCommand)
    if journal["ttsChanged"] is True and tts_owner is not None:
        invoker.invoke(tts_owner, "uninstall")
    if journal["codexChanged"] is True:
        invoker.invoke(codex_owner, "uninstall")
    if journal["createdIdentity"] is True:
        _remove_owned_file(root / "identity.json")
    _remove_owned_file(root / "setup-transaction.json")


def setup_voice(
    root: Path,
    invoker: LifecycleInvoker,
    network_backend: NetworkBackend,
    *,
    label: str | None,
    alias: str | None,
    network_mode: str,
    bind_ip: str | None,
    include_tts: bool,
    trust_lan: bool,
    plan_only: bool,
) -> tuple[VoiceConfig | None, dict[str, object]]:
    if plan_only:
        return _setup_voice_transaction(
            root,
            invoker,
            network_backend,
            label=label,
            alias=alias,
            network_mode=network_mode,
            bind_ip=bind_ip,
            include_tts=include_tts,
            trust_lan=trust_lan,
            plan_only=True,
        )
    _ensure_private_directory(root)
    with _voice_setup_lock(root):
        _recover_setup_transaction(root, invoker)
        return _setup_voice_transaction(
            root,
            invoker,
            network_backend,
            label=label,
            alias=alias,
            network_mode=network_mode,
            bind_ip=bind_ip,
            include_tts=include_tts,
            trust_lan=trust_lan,
            plan_only=False,
        )


def _setup_voice_transaction(
    root: Path,
    invoker: LifecycleInvoker,
    network_backend: NetworkBackend,
    *,
    label: str | None,
    alias: str | None,
    network_mode: str,
    bind_ip: str | None,
    include_tts: bool,
    trust_lan: bool,
    plan_only: bool,
) -> tuple[VoiceConfig | None, dict[str, object]]:
    selection = select_network(network_backend, mode=network_mode, bind_ip=bind_ip)
    if selection.confirmation_required and not trust_lan and not plan_only:
        raise VoiceError(
            "lan_confirmation_required",
            "Trusted-LAN mode requires explicit confirmation.",
            "Review the local network, then rerun with --trust-lan.",
        )
    if plan_only:
        steps = ["validate_owner_contract", "verify_or_install_codex"]
        if include_tts:
            steps.append("verify_or_install_optional_tts")
        steps.append("create_expiring_pairing_invitation")
        return None, {
            "schemaVersion": VOICE_STATUS_SCHEMA,
            "action": "setup",
            "status": "plan",
            "network": selection.to_public_dict(),
            "steps": steps,
        }
    _ensure_private_directory(root)
    existing = root / "voice.json"
    if existing.exists():
        config = load_voice_config(root)
        configured_network = (
            config.network.address,
            config.network.network_class,
            config.network.interface_name,
        )
        selected_network = (selection.address, selection.network_class, selection.interface_name)
        if configured_network != selected_network:
            raise VoiceError(
                "setup_network_changed",
                "The existing voice setup uses another exact network address.",
                "Run voice repair on the existing setup or review a new setup transaction.",
            )
        if label is not None and _validated_label(label) != config.label:
            raise VoiceError(
                "setup_identity_conflict",
                "The requested label differs from the stable server identity.",
                "Use the existing label or remove the setup only after exporting its identity.",
            )
        if (
            alias is not None
            and _validated_alias(alias, config.identity_fingerprint) != config.alias
        ):
            raise VoiceError(
                "setup_identity_conflict",
                "The requested alias differs from the stable server identity.",
                "Use the existing alias or choose a new PC identity after review.",
            )
        codex = _codex_health(invoker.invoke(config.codex_owner, "status"))
        tts = (
            _tts_health(invoker.invoke(config.tts_owner, "status"))
            if include_tts and config.tts_owner is not None
            else None
        )
        if not codex.healthy:
            raise VoiceError(
                "codex_not_ready",
                "Codex is not authenticated and ready.",
                "Run agent-tools voice repair, then retry pairing.",
            )
        invitation_config = (
            config
            if include_tts
            else replace(config, tts_owner=None, tts_token_file=None, tts_origin=None)
        )
        return invitation_config, {
            "schemaVersion": VOICE_STATUS_SCHEMA,
            "action": "setup",
            "status": "ready" if tts is None or tts.healthy else "ready_local_tts",
            "changed": False,
            "network": selection.to_public_dict(),
        }

    server_label = _validated_label(label)
    if alias is not None:
        _validated_alias(alias, "0" * 64)
    (
        codex_owner,
        codex_token,
        codex_endpoint,
        tts_owner,
        tts_token,
        tts_origin,
    ) = load_owner_spec(root, selection, include_tts=include_tts)
    identity_path = root / "identity.json"
    created_identity = not identity_path.exists()
    identity = load_or_create_identity(identity_path)
    server_alias = _validated_alias(alias, identity.fingerprint)
    config = VoiceConfig(
        root=root,
        server_id=identity.server_id,
        label=server_label,
        alias=server_alias,
        identity_fingerprint=identity.fingerprint,
        network=selection,
        codex_endpoint=codex_endpoint,
        codex_token_file=codex_token,
        codex_owner=codex_owner,
        tts_origin=tts_origin,
        tts_token_file=tts_token,
        tts_owner=tts_owner,
    )
    try:
        _setup_journal(root, "prepared", config, created_identity=created_identity)
    except Exception:
        if created_identity:
            _remove_owned_file(identity_path)
        raise
    codex_changed = False
    tts_changed = False
    config_written = False
    tts_health: OwnerHealth | None = None
    try:
        initial_codex = _codex_health(invoker.invoke(codex_owner, "status"))
        if initial_codex.code in {"unknown_listener", "foreign_task"}:
            raise VoiceError(
                "codex_owner_unsafe",
                "Codex has an unknown task, identity, or listener.",
                "Repair the existing Codex lifecycle owner before setup.",
            )
        codex_changed = not initial_codex.registered
        _setup_journal(
            root,
            "codex_installing",
            config,
            created_identity=created_identity,
            codex_changed=codex_changed,
        )
        invoker.invoke(codex_owner, "install")
        _setup_journal(
            root,
            "codex_installed",
            config,
            created_identity=created_identity,
            codex_changed=codex_changed,
        )
        codex_health = _codex_health(invoker.invoke(codex_owner, "status"), changed=codex_changed)
        if not codex_health.healthy:
            raise VoiceError(
                "codex_not_ready",
                "Codex setup did not reach authenticated readiness.",
                "Run the Codex lifecycle status and repair its exact owner before retrying.",
            )
        if tts_owner is not None:
            try:
                initial_tts = _tts_health(invoker.invoke(tts_owner, "status"))
                if initial_tts.code in {"foreign_listener", "foreign_task"} or (
                    initial_tts.registered and not initial_tts.exact
                ):
                    raise VoiceError(
                        "tts_owner_unsafe",
                        "TTS has an unknown task, identity, or listener.",
                        "Repair the optional TTS lifecycle owner before enabling it.",
                    )
                tts_changed = not initial_tts.registered
                _setup_journal(
                    root,
                    "tts_installing",
                    config,
                    created_identity=created_identity,
                    codex_changed=codex_changed,
                    tts_changed=tts_changed,
                )
                invoker.invoke(tts_owner, "install")
                _setup_journal(
                    root,
                    "tts_installed",
                    config,
                    created_identity=created_identity,
                    codex_changed=codex_changed,
                    tts_changed=tts_changed,
                )
                tts_health = _tts_health(invoker.invoke(tts_owner, "status"), changed=tts_changed)
                if not tts_health.healthy:
                    if tts_changed:
                        invoker.invoke(tts_owner, "uninstall")
                        tts_changed = False
                    tts_owner = None
                    tts_token = None
                    tts_origin = None
            except VoiceError as exc:
                if exc.code == "tts_owner_unsafe":
                    raise
                if tts_changed and config.tts_owner is not None:
                    try:
                        invoker.invoke(config.tts_owner, "uninstall")
                    except VoiceError as rollback_exc:
                        raise VoiceError(
                            "tts_rollback_failed",
                            "Optional TTS setup failed and its new owner could not be removed.",
                            "Run voice setup again to resume protected transaction recovery.",
                        ) from rollback_exc
                    tts_changed = False
                    _setup_journal(
                        root,
                        "codex_installed",
                        config,
                        created_identity=created_identity,
                        codex_changed=codex_changed,
                    )
                tts_owner = None
                tts_token = None
                tts_origin = None
                tts_health = None
        final_config = replace(
            config,
            tts_owner=tts_owner,
            tts_token_file=tts_token,
            tts_origin=tts_origin,
        )
        _write_private_json(root / "voice.json", final_config.to_dict(), replace=False)
        config_written = True
        _setup_journal(
            root,
            "committed",
            final_config,
            created_identity=created_identity,
            codex_changed=codex_changed,
            tts_changed=tts_changed,
        )
        _remove_owned_file(root / "setup-transaction.json")
        return final_config, {
            "schemaVersion": VOICE_STATUS_SCHEMA,
            "action": "setup",
            "status": "ready" if tts_owner is not None else "ready_local_tts",
            "changed": True,
            "network": selection.to_public_dict(),
            "codex": codex_health.to_public_dict(),
            "tts": None if tts_health is None else tts_health.to_public_dict(),
        }
    except Exception:
        if config_written:
            raise
        rollback_complete = True
        if tts_changed and config.tts_owner is not None:
            try:
                invoker.invoke(config.tts_owner, "uninstall")
            except VoiceError:
                rollback_complete = False
        if codex_changed:
            try:
                invoker.invoke(config.codex_owner, "uninstall")
            except VoiceError:
                rollback_complete = False
        if created_identity:
            try:
                _remove_owned_file(identity_path)
            except VoiceError:
                rollback_complete = False
        if rollback_complete:
            _remove_owned_file(root / "setup-transaction.json")
        raise


def repair_voice(root: Path, invoker: LifecycleInvoker) -> dict[str, object]:
    config = load_voice_config(root)
    codex_before, tts_before = inspect_owners(config, invoker)
    if codex_before.code in {"unknown_listener", "foreign_task"}:
        raise VoiceError(
            "codex_owner_unsafe",
            "Codex has an unknown task, identity, or listener.",
            "Repair the Codex lifecycle owner directly before retrying.",
        )
    if tts_before is not None and (
        tts_before.code in {"foreign_listener", "foreign_task"}
        or (tts_before.registered and not tts_before.exact)
        or (tts_before.listening and not tts_before.owned)
    ):
        raise VoiceError(
            "tts_owner_unsafe",
            "TTS has an unknown task, identity, or listener.",
            "Repair the TTS lifecycle owner directly before retrying.",
        )
    if not codex_before.healthy:
        action = "install" if not codex_before.registered else "restart"
        invoker.invoke(config.codex_owner, action)
    codex_after = _codex_health(
        invoker.invoke(config.codex_owner, "status"), changed=not codex_before.healthy
    )
    if not codex_after.healthy:
        raise VoiceError(
            "codex_repair_incomplete",
            "Codex repair did not reach authenticated readiness.",
            "Inspect the Codex lifecycle status before retrying.",
        )
    tts_after = tts_before
    if config.tts_owner is not None and tts_before is not None and not tts_before.healthy:
        try:
            action = "install" if not tts_before.registered else "start"
            invoker.invoke(config.tts_owner, action)
            tts_after = _tts_health(invoker.invoke(config.tts_owner, "status"), changed=True)
        except VoiceError:
            tts_after = None
    return {
        "schemaVersion": VOICE_STATUS_SCHEMA,
        "action": "repair",
        "status": "ready" if tts_after is None or tts_after.healthy else "ready_local_tts",
        "codex": codex_after.to_public_dict(),
        "tts": None if tts_after is None else tts_after.to_public_dict(),
    }


@dataclass(frozen=True)
class PairingInvitation:
    uri: str
    fallback_code: str
    expires_at: int
    port: int
    tls_spki_sha256: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": PAIRING_VERSION,
            "pairingUri": self.uri,
            "checkCode": self.fallback_code,
            "expiresAt": self.expires_at,
        }


@dataclass
class _PairState:
    session_id: str
    nonce: str
    expires_at: int
    manifest_builder: Callable[[str, int], dict[str, object]]
    identity: ServerIdentity
    now: Callable[[], float]
    claimed_client: bytes | None = None
    cached_response: bytes | None = None
    manifest_digest: str | None = None
    consumed: bool = False

    def claim(self, payload: object) -> tuple[int, dict[str, object]]:
        if self.consumed:
            return 410, {"error": {"code": "pairing_consumed"}}
        if int(self.now()) >= self.expires_at:
            return 410, {"error": {"code": "pairing_expired"}}
        if not isinstance(payload, dict) or set(payload) != {
            "v",
            "sessionId",
            "nonce",
            "clientPublicKey",
        }:
            return 400, {"error": {"code": "claim_invalid"}}
        if payload["v"] != PAIRING_VERSION:
            return 400, {"error": {"code": "version_unsupported"}}
        if not isinstance(payload["sessionId"], str) or not hmac.compare_digest(
            payload["sessionId"], self.session_id
        ):
            return 404, {"error": {"code": "session_unknown"}}
        if not isinstance(payload["nonce"], str) or not hmac.compare_digest(
            payload["nonce"], self.nonce
        ):
            return 403, {"error": {"code": "nonce_invalid"}}
        client = payload["clientPublicKey"]
        try:
            if not isinstance(client, str):
                raise ValueError
            client_bytes = _b64url_decode(client, expected_bytes=32)
        except ValueError:
            return 400, {"error": {"code": "client_key_invalid"}}
        if self.claimed_client is not None and not hmac.compare_digest(
            client_bytes, self.claimed_client
        ):
            return 409, {"error": {"code": "pairing_claimed"}}
        if self.cached_response is None:
            self.claimed_client = client_bytes
            client_hash = hashlib.sha256(client_bytes).hexdigest()
            manifest = self.manifest_builder(client_hash, self.expires_at)
            canonical = _canonical_json(manifest)
            signature = self.identity.private_key.sign(canonical)
            self.manifest_digest = hashlib.sha256(canonical).hexdigest()
            response = {
                "manifest": manifest,
                "manifestSha256": self.manifest_digest,
                "signature": _b64url(signature),
            }
            self.cached_response = _canonical_json(response)
        return 200, json.loads(self.cached_response.decode("ascii"))

    def acknowledge(self, payload: object) -> tuple[int, dict[str, object]]:
        if self.consumed:
            return 410, {"error": {"code": "pairing_consumed"}}
        if int(self.now()) >= self.expires_at:
            return 410, {"error": {"code": "pairing_expired"}}
        if not isinstance(payload, dict) or set(payload) != {
            "v",
            "sessionId",
            "nonce",
            "clientPublicKey",
            "manifestSha256",
        }:
            return 400, {"error": {"code": "ack_invalid"}}
        client = payload.get("clientPublicKey")
        try:
            if not isinstance(client, str):
                raise ValueError
            client_bytes = _b64url_decode(client, expected_bytes=32)
        except ValueError:
            return 400, {"error": {"code": "client_key_invalid"}}
        values = (payload.get("sessionId"), payload.get("nonce"), payload.get("manifestSha256"))
        expected = (self.session_id, self.nonce, self.manifest_digest)
        for value, wanted in zip(values, expected, strict=True):
            if (
                not isinstance(value, str)
                or wanted is None
                or not hmac.compare_digest(value, wanted)
            ):
                return 403, {"error": {"code": "ack_not_claimant"}}
        if self.claimed_client is None or not hmac.compare_digest(
            client_bytes, self.claimed_client
        ):
            return 403, {"error": {"code": "ack_not_claimant"}}
        self.consumed = True
        return 200, {"ok": True}


class _ExactHttpsServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request: object, client_address: object) -> None:
        del request, client_address


class _PairingHandler(BaseHTTPRequestHandler):
    server_version = "AgentToolsPair/1"
    sys_version = ""

    def do_POST(self) -> None:
        state: _PairState = self.server.pair_state  # type: ignore[attr-defined]
        if self.path not in {PAIR_PATH, ACK_PATH}:
            self._send(404, {"error": {"code": "path_unknown"}})
            return
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError:
            self._send(411, {"error": {"code": "content_length_required"}})
            return
        if length <= 0 or length > MAX_PAIR_REQUEST_BYTES:
            self._send(413, {"error": {"code": "request_too_large"}})
            return
        if (
            self.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
            != "application/json"
        ):
            self._send(415, {"error": {"code": "content_type_invalid"}})
            return
        raw = self.rfile.read(length)
        if len(raw) != length:
            self._send(400, {"error": {"code": "request_truncated"}})
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._send(400, {"error": {"code": "json_invalid"}})
            return
        with self.server.pair_lock:  # type: ignore[attr-defined]
            status, response = (
                state.claim(payload) if self.path == PAIR_PATH else state.acknowledge(payload)
            )
        self._send(status, response)
        if self.path == ACK_PATH and status == 200:
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    def _send(self, status: int, payload: Mapping[str, object]) -> None:
        data = _canonical_json(payload)
        if len(data) > MAX_PAIR_RESPONSE_BYTES:
            status = 500
            data = b'{"error":{"code":"response_too_large"}}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class PairingServer:
    def __init__(
        self,
        config: VoiceConfig,
        identity: ServerIdentity,
        *,
        expires_seconds: int,
        allow_fixture_loopback: bool = False,
        now: Callable[[], float] = time.time,
        random_port: Callable[[], int] | None = None,
    ) -> None:
        if not PAIRING_MIN_SECONDS <= expires_seconds <= PAIRING_MAX_SECONDS:
            raise VoiceError(
                "pairing_expiry_invalid",
                "Pairing expiry is outside the supported range.",
                f"Choose {PAIRING_MIN_SECONDS}-{PAIRING_MAX_SECONDS} seconds.",
            )
        if not _exact_private_endpoint(
            config.network.address, allow_loopback=allow_fixture_loopback
        ):
            raise VoiceError(
                "pairing_bind_invalid",
                "Pairing requires one exact private IPv4 address.",
                "Repair the selected voice network before pairing.",
            )
        self.config = config
        self.identity = identity
        self.now = now
        self.expires_at = int(now()) + expires_seconds
        self.session_id = _b64url(secrets.token_bytes(16))
        self.nonce = _b64url(secrets.token_bytes(32))
        self._pairing_root = config.root / "pairing"
        self._session_hash = hashlib.sha256(self.session_id.encode("ascii")).hexdigest()
        self._session_root = self._pairing_root / self._session_hash[:24]
        self._server: _ExactHttpsServer | None = None
        self._thread: threading.Thread | None = None
        self._timer: threading.Timer | None = None
        self._started = False
        self._active_record_identity: tuple[int, int, int, int] | None = None
        self._codex_token: str | None = _load_voice_token(config.codex_token_file)
        self._tts_token: str | None = (
            _load_voice_token(config.tts_token_file) if config.tts_token_file is not None else None
        )
        self._process_generation = _process_generation(os.getpid())
        if self._process_generation is None:
            raise VoiceError(
                "pairing_process_identity_unavailable",
                "The pairing process identity could not be verified.",
                "Retry from the installed AgentTools process on Windows.",
            )
        self._random_port = random_port or (
            lambda: PAIRING_MIN_PORT + secrets.randbelow(PAIRING_MAX_PORT - PAIRING_MIN_PORT + 1)
        )
        self.invitation = self._prepare(allow_fixture_loopback)

    def _prepare(self, allow_fixture_loopback: bool) -> PairingInvitation:
        with _interprocess_file_lock(
            self.config.root / ".pairing.guard",
            busy_code="pairing_state_busy",
            busy_message="Another pairing invitation is being updated.",
            busy_next_action="Wait briefly, then retry pairing.",
            timeout=3.0,
        ):
            return self._prepare_guarded(allow_fixture_loopback)

    def _prepare_guarded(self, allow_fixture_loopback: bool) -> PairingInvitation:
        if allow_fixture_loopback and self.config.network.address != "127.0.0.1":
            raise AssertionError("fixture loopback mismatch")
        _ensure_private_directory(self._pairing_root)
        self._cleanup_stale_record()
        _ensure_private_directory(self._session_root)
        try:
            certificate, private_key, pin = _ephemeral_certificate(self.config.network.address)
            cert_path = self._session_root / "tls-cert.pem"
            key_path = self._session_root / "tls-key.pem"
            _write_private_bytes(cert_path, certificate, replace=False)
            _write_private_bytes(key_path, private_key, replace=False)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(certfile=cert_path, keyfile=key_path)
        except Exception:
            self._cleanup_session()
            raise
        state = _PairState(
            self.session_id,
            self.nonce,
            self.expires_at,
            self._build_manifest,
            self.identity,
            self.now,
        )
        server: _ExactHttpsServer | None = None
        for _ in range(64):
            port = self._random_port()
            if not PAIRING_MIN_PORT <= port <= PAIRING_MAX_PORT or port in {4222, 4223}:
                continue
            try:
                server = _ExactHttpsServer((self.config.network.address, port), _PairingHandler)
                break
            except OSError:
                continue
        if server is None:
            self._cleanup_session()
            raise VoiceError(
                "pairing_port_unavailable",
                "No random high pairing port could be reserved.",
                "Close the conflicting local service and retry pairing.",
            )
        try:
            server.socket = context.wrap_socket(server.socket, server_side=True)
        except Exception:
            server.server_close()
            self._cleanup_session()
            raise
        server.pair_state = state  # type: ignore[attr-defined]
        server.pair_lock = threading.Lock()  # type: ignore[attr-defined]
        self._server = server
        pair_url = f"https://{self.config.network.address}:{server.server_port}{PAIR_PATH}"
        bootstrap = {
            "a": self.config.alias,
            "c": self.config.network.network_class,
            "e": self.expires_at,
            "f": self.identity.fingerprint,
            "i": self.config.server_id,
            "l": self.config.label,
            "n": self.nonce,
            "p": pin,
            "s": self.session_id,
            "u": pair_url,
            "v": PAIRING_VERSION,
        }
        encoded = _b64url(_canonical_json(bootstrap))
        uri = f"voxflow://pair?d={encoded}"
        fallback = _fallback_code(uri)
        record = {
            "v": 1,
            "expiresAt": self.expires_at,
            "pid": os.getpid(),
            "processGeneration": self._process_generation,
            "sessionHash": self._session_hash,
        }
        active_record = self._pairing_root / "active.json"
        try:
            _write_private_json(active_record, record, replace=False)
            self._active_record_identity = _file_identity(active_record.lstat())
        except Exception:
            server.server_close()
            self._server = None
            self._cleanup_session()
            raise
        return PairingInvitation(uri, fallback, self.expires_at, server.server_port, pin)

    def _build_manifest(self, client_hash: str, expires_at: int) -> dict[str, object]:
        if self._codex_token is None:
            raise RuntimeError("pairing credential state invalid")
        issued_at = int(self.now())
        manifest: dict[str, object] = {
            "v": PAIRING_VERSION,
            "sessionId": self.session_id,
            "clientPublicKeySha256": client_hash,
            "issuedAt": issued_at,
            "expiresAt": expires_at,
            "server": {
                "id": self.config.server_id,
                "label": self.config.label,
                "alias": self.config.alias,
                "identityPublicKey": _b64url(self.identity.public_key),
                "fingerprint": self.identity.fingerprint,
            },
            "networkClass": self.config.network.network_class,
            "codex": {
                "endpoint": self.config.codex_endpoint,
                "capabilityToken": self._codex_token,
                "protocol": "ai-nd-co-codex-app-server",
                "version": 1,
            },
            "capabilities": ["codex.app-server.v1", "pairing.claimant-bound.v1"],
        }
        if self.config.tts_origin is not None and self._tts_token is not None:
            manifest["tts"] = {
                "origin": self.config.tts_origin,
                "bearerToken": self._tts_token,
                "version": 1,
            }
            manifest["capabilities"] = [
                "codex.app-server.v1",
                "pairing.claimant-bound.v1",
                "tts.remote.v1",
            ]
        return manifest

    def start(self) -> None:
        if self._server is None or self._thread is not None:
            raise RuntimeError("pairing server state invalid")
        thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        try:
            thread.start()
            self._thread = thread
            self._started = True
            delay = max(0.0, self.expires_at - self.now())
            self._timer = threading.Timer(delay, self._server.shutdown)
            self._timer.daemon = True
            self._timer.start()
        except RuntimeError as exc:
            self.close()
            raise VoiceError(
                "pairing_start_failed",
                "The pairing listener could not start.",
                "Retry after local thread resources are available.",
            ) from exc

    def wait(self) -> str:
        if self._thread is None:
            raise RuntimeError("pairing server not started")
        self._thread.join(max(1.0, self.expires_at - self.now() + 5.0))
        consumed = bool(self._server and self._server.pair_state.consumed)  # type: ignore[attr-defined]
        self.close()
        return "paired" if consumed else "expired"

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        if self._server is not None:
            if self._started:
                self._server.shutdown()
            state: _PairState = self._server.pair_state  # type: ignore[attr-defined]
            state.cached_response = None
            state.claimed_client = None
            state.manifest_digest = None
            self._server.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(2.0)
        release_error: VoiceError | None = None
        try:
            self._release_active_record()
        except VoiceError as exc:
            release_error = exc
        try:
            self._cleanup_session()
        finally:
            self._codex_token = None
            self._tts_token = None
            self._server = None
            self._thread = None
            self._timer = None
            self._started = False
        if release_error is not None:
            raise release_error

    def _release_active_record(self) -> None:
        if self._active_record_identity is None:
            return
        with _interprocess_file_lock(
            self.config.root / ".pairing.guard",
            busy_code="pairing_state_busy",
            busy_message="Another pairing invitation is being updated.",
            busy_next_action="Wait briefly, then retry pairing.",
            timeout=3.0,
        ):
            record = self._pairing_root / "active.json"
            if not record.exists():
                self._active_record_identity = None
                return
            payload = _read_pairing_record(self.config.root)
            if payload is None or not hmac.compare_digest(
                str(payload["sessionHash"]), self._session_hash
            ):
                self._active_record_identity = None
                return
            current = record.lstat()
            if _file_identity(current) != self._active_record_identity:
                raise VoiceError(
                    "pairing_record_changed",
                    "The active pairing record changed identity.",
                    "Inspect protected voice pairing state before retrying.",
                )
            _remove_owned_file(record)
            self._active_record_identity = None

    def _cleanup_stale_record(self) -> None:
        record = self._pairing_root / "active.json"
        if not record.exists():
            return
        payload = _read_pairing_record(self.config.root)
        if payload is None:
            return
        pid = payload["pid"]
        expires_at = payload["expiresAt"]
        assert isinstance(pid, int)
        assert isinstance(expires_at, int)
        generation = _process_generation(pid)
        active = (
            expires_at > int(self.now())
            and generation is not None
            and hmac.compare_digest(
                generation,
                str(payload["processGeneration"]),
            )
        )
        if active:
            raise VoiceError(
                "pairing_already_active",
                "Another pairing invitation is active.",
                "Finish or wait for that invitation before creating another.",
            )
        session_hash = str(payload["sessionHash"])
        _cleanup_pairing_directory(self._pairing_root, self._pairing_root / session_hash[:24])
        _remove_owned_file(record)

    def _cleanup_session(self) -> None:
        if not self._session_root.exists():
            return
        _cleanup_pairing_directory(self._pairing_root, self._session_root)


def _cleanup_pairing_directory(pairing_root: Path, session_root: Path) -> None:
    root = pairing_root.resolve()
    target = session_root.resolve()
    if not target.exists():
        return
    if target == root or root not in target.parents:
        raise VoiceError(
            "pairing_cleanup_refused",
            "Pairing cleanup target is outside protected storage.",
            "Inspect the voice pairing state before retrying.",
        )
    shutil.rmtree(target)


def _ephemeral_certificate(address: str) -> tuple[bytes, bytes, str]:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError as exc:
        raise VoiceError(
            "voice_dependencies_missing",
            "Voice pairing cryptography is unavailable.",
            "Reinstall AgentTools with its current dependencies.",
        ) from exc
    import datetime

    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.UTC)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AgentTools voice pairing")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(minutes=20))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(address))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    spki = key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return cert_pem, key_pem, _b64url(hashlib.sha256(spki).digest())


def _fallback_code(uri: str) -> str:
    value = base64.b32encode(hashlib.sha256(uri.encode("ascii")).digest()).decode("ascii")[:8]
    return f"{value[:4]}-{value[4:]}"


def _render_qr(uri: str) -> str:
    try:
        import qrcode  # type: ignore[import-untyped]
    except ImportError as exc:
        raise VoiceError(
            "voice_dependencies_missing",
            "Terminal QR support is unavailable.",
            "Reinstall AgentTools with its current dependencies.",
        ) from exc
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L, border=2)
    qr.add_data(uri)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    lines: list[str] = []
    for index in range(0, len(matrix), 2):
        top = matrix[index]
        bottom = matrix[index + 1] if index + 1 < len(matrix) else [False] * len(top)
        lines.append(
            "".join(
                "█" if upper and lower else "▀" if upper else "▄" if lower else " "
                for upper, lower in zip(top, bottom, strict=True)
            )
        )
    return "\n".join(lines)


def create_pairing_server(
    root: Path,
    invoker: LifecycleInvoker,
    *,
    expires_seconds: int,
    allow_fixture_loopback: bool = False,
    include_tts: bool = True,
) -> PairingServer:
    config = load_voice_config(root)
    codex = _codex_health(invoker.invoke(config.codex_owner, "status"))
    tts = (
        _tts_health(invoker.invoke(config.tts_owner, "status"))
        if include_tts and config.tts_owner is not None
        else None
    )
    if not codex.healthy:
        raise VoiceError(
            "codex_not_ready",
            "Codex is not authenticated and ready.",
            "Run agent-tools voice repair before pairing.",
        )
    if not include_tts or (config.tts_owner is not None and (tts is None or not tts.healthy)):
        config = replace(config, tts_owner=None, tts_token_file=None, tts_origin=None)
    identity = load_or_create_identity(root / "identity.json", create=False)
    return PairingServer(
        config,
        identity,
        expires_seconds=expires_seconds,
        allow_fixture_loopback=allow_fixture_loopback,
    )


def _render_status(status: VoiceStatus) -> str:
    if not status.configured:
        return "Voice: not configured\nNext: agent-tools voice setup"
    codex = (
        "ready"
        if status.codex and status.codex.healthy
        else (status.codex.code if status.codex else "unknown")
    )
    tts = "ready" if status.tts and status.tts.healthy else "device fallback"
    network = (
        f"{status.network.network_class} {status.network.address}" if status.network else "unknown"
    )
    pairing = "active" if status.pairing_active else "idle"
    return (
        f"Voice: {status.code}\nCodex: {codex}\nTTS: {tts}\nNetwork: {network}\nPairing: {pairing}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-tools voice", description="Set up and pair Vox with this PC."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    setup = subparsers.add_parser("setup", help="Verify services and open one pairing invitation.")
    setup.add_argument("--label")
    setup.add_argument("--alias")
    setup.add_argument("--network", choices=("auto", "tailscale", "lan"), default="auto")
    setup.add_argument("--bind-ip")
    setup.add_argument("--no-tts", action="store_true")
    setup.add_argument("--trust-lan", action="store_true")
    setup.add_argument("--plan", action="store_true")
    setup.add_argument("--json", action="store_true")
    status = subparsers.add_parser("status", help="Show bounded service and pairing status.")
    status.add_argument("--json", action="store_true")
    repair = subparsers.add_parser("repair", help="Repair only exact owned service lifecycles.")
    repair.add_argument("--json", action="store_true")
    pair = subparsers.add_parser("pair", help="Open a new short-lived pairing invitation.")
    pair.add_argument("--expires-seconds", type=int, default=PAIRING_DEFAULT_SECONDS)
    pair.add_argument("--json", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path | None = None,
    invoker: LifecycleInvoker | None = None,
    network_backend: NetworkBackend | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    pair_factory: Callable[..., PairingServer] = create_pairing_server,
) -> int:
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(arguments)
        voice_root = root or _default_voice_root()
        owner_invoker = invoker or LifecycleInvoker()
        if args.action == "status":
            status = voice_status(voice_root, owner_invoker)
            output.write(
                (
                    json.dumps(status.to_public_dict(), sort_keys=True)
                    if args.json
                    else _render_status(status)
                )
                + "\n"
            )
            return 0 if status.code not in {"unsafe", "needs_repair"} else 2
        if args.action == "repair":
            result = repair_voice(voice_root, owner_invoker)
            output.write(
                (
                    json.dumps(result, sort_keys=True)
                    if args.json
                    else "Voice: ready\nCodex: ready\n"
                )
                + "\n"
            )
            return 0
        if args.action == "setup":
            config, result = setup_voice(
                voice_root,
                owner_invoker,
                network_backend or WindowsNetworkBackend(),
                label=args.label,
                alias=args.alias,
                network_mode=args.network,
                bind_ip=args.bind_ip,
                include_tts=not args.no_tts,
                trust_lan=args.trust_lan,
                plan_only=args.plan,
            )
            if args.plan:
                output.write(
                    (
                        json.dumps(result, sort_keys=True)
                        if args.json
                        else _render_setup_plan(result)
                    )
                    + "\n"
                )
                return 0
            if config is None:
                raise RuntimeError("setup returned no config")
            pairing = pair_factory(
                voice_root,
                owner_invoker,
                expires_seconds=PAIRING_DEFAULT_SECONDS,
                include_tts=not args.no_tts,
            )
            return _run_pairing(pairing, output, json_output=args.json)
        pairing = pair_factory(voice_root, owner_invoker, expires_seconds=args.expires_seconds)
        return _run_pairing(pairing, output, json_output=args.json)
    except VoiceError as exc:
        if "--json" in arguments:
            errors.write(json.dumps({"error": exc.to_public_dict()}, sort_keys=True) + "\n")
        else:
            errors.write(f"Voice: {exc.message}\nImpact: {exc.impact}\nNext: {exc.next_action}\n")
        return 2


def _render_setup_plan(result: Mapping[str, object]) -> str:
    network = result["network"]
    assert isinstance(network, dict)
    steps = result["steps"]
    assert isinstance(steps, list)
    return "\n".join(
        [
            "VOICE SETUP PLAN",
            f"Network: {network['class']} {network['address']}",
            *(f"- {str(step).replace('_', ' ')}" for step in steps),
        ]
    )


def _run_pairing(pairing: PairingServer, output: TextIO, *, json_output: bool) -> int:
    if json_output:
        output.write(json.dumps(pairing.invitation.to_public_dict(), sort_keys=True) + "\n")
    else:
        output.write("VOICE PAIR\n")
        output.write(_render_qr(pairing.invitation.uri) + "\n")
        remaining = max(0, pairing.invitation.expires_at - int(time.time()))
        output.write(
            f"CODE {pairing.invitation.fallback_code}   "
            f"{remaining // 60:02d}:{remaining % 60:02d}\n"
        )
    output.flush()
    try:
        pairing.start()
        result = pairing.wait()
    finally:
        pairing.close()
    if json_output:
        output.write(json.dumps({"pairing": result}, sort_keys=True) + "\n")
    else:
        output.write("Voice: paired\n" if result == "paired" else "Voice: pairing expired\n")
    return 0 if result == "paired" else 2
