from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import socket
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO

from agent_tools.tts_server import (
    load_owner_only_bearer_token,
    validate_tts_server_bind,
)

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer, WriteableBuffer

TASK_NAME = "ai-nd-co AgentTools TTS (Owner Logon)"
TASK_DESCRIPTION_PREFIX = "ai-nd-co-agent-tools-tts:"
CONFIG_SCHEMA_VERSION = 1
TASK_LOGON_DELAY = "PT15S"
TASK_RESTART_INTERVAL = "PT1M"
TASK_RESTART_COUNT = 3
DEFAULT_READY_TIMEOUT_SECONDS = 240.0
DEFAULT_RELEASE_TIMEOUT_SECONDS = 35.0
DEFAULT_RESTART_DELAY_SECONDS = 60.0
DEFAULT_RESTART_RESET_SECONDS = 300.0
DEFAULT_MAX_RESTARTS = 3
DEFAULT_POLL_SECONDS = 0.2
DEFAULT_HEALTH_REPROBE_SECONDS = 30.0
DEFAULT_HEALTH_FAILURE_THRESHOLD = 3
STATE_FILENAME = "runtime-state.json"
LOCK_FILENAME = "runner.lock"
STOP_FILENAME = "stop-request.json"
TASK_XML_FILENAME = "scheduled-task.xml"
EVENT_LOG_FILENAME = "tts-startup.log"
MAX_CONFIG_BYTES = 64 * 1024
MAX_STATE_BYTES = 64 * 1024
MAX_EVENT_LOG_BYTES = 256 * 1024
MAX_BOOTSTRAP_BYTES = 64 * 1024
MAX_WINDOWS_PID = (1 << 32) - 1
MAX_WINDOWS_TIMESTAMP = (1 << 64) - 1

_RUNTIME_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "config_sha256",
        "config_file_fingerprint",
        "task_fingerprint",
        "credential_fingerprint",
        "runner_pid",
        "runner_creation_time_ns",
        "child_pid",
        "child_creation_time_ns",
        "phase",
        "restart_count",
        "authenticated_ready",
    }
)
_RUNTIME_PHASES = frozenset(
    {
        "starting",
        "ready",
        "stopped",
        "foreign_listener",
        "config_drift",
        "credential_drift",
        "executable_drift",
        "bootstrap_drift",
        "listener_release_timeout",
        "restart_exhausted",
        "restart_wait",
        "health_failed",
    }
)
_DRIFT_EXIT_CODES = {
    "config_drift": 23,
    "credential_drift": 24,
    "executable_drift": 25,
    "bootstrap_drift": 26,
}
_LOCK_FIELDS = frozenset(
    {
        "schema_version",
        "config_sha256",
        "config_file_fingerprint",
        "task_fingerprint",
        "runner_pid",
        "runner_creation_time_ns",
    }
)
_STOP_REQUEST_FIELDS = _LOCK_FIELDS

_EXACT_IMAGE_NATIVE_SOURCE_LINES = (
    "using System;",
    "using System.ComponentModel;",
    "using System.IO;",
    "using System.Runtime.InteropServices;",
    "using Microsoft.Win32.SafeHandles;",
    "public static class AgentToolsExactImage {",
    "  const uint GENERIC_READ = 0x80000000;",
    "  const uint FILE_SHARE_READ = 0x00000001;",
    "  const uint OPEN_EXISTING = 3;",
    "  const uint FILE_ATTRIBUTE_NORMAL = 0x00000080;",
    "  const uint FILE_ATTRIBUTE_DIRECTORY = 0x00000010;",
    "  const uint FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400;",
    "  const uint FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000;",
    "  [StructLayout(LayoutKind.Sequential)]",
    "  struct BY_HANDLE_FILE_INFORMATION {",
    "    public uint FileAttributes;",
    "    public System.Runtime.InteropServices.ComTypes.FILETIME CreationTime;",
    "    public System.Runtime.InteropServices.ComTypes.FILETIME LastAccessTime;",
    "    public System.Runtime.InteropServices.ComTypes.FILETIME LastWriteTime;",
    "    public uint VolumeSerialNumber;",
    "    public uint FileSizeHigh; public uint FileSizeLow;",
    "    public uint NumberOfLinks;",
    "    public uint FileIndexHigh; public uint FileIndexLow;",
    "  }",
    '  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]',
    "  static extern SafeFileHandle CreateFile("
    "string path, uint access, uint share, IntPtr security, uint creation, "
    "uint flags, IntPtr templateFile);",
    '  [DllImport("kernel32.dll", SetLastError=true)]',
    "  static extern bool GetFileInformationByHandle("
    "SafeFileHandle handle, out BY_HANDLE_FILE_INFORMATION info);",
    "  public static SafeFileHandle Open(string path) {",
    "    SafeFileHandle handle = CreateFile("
    "path, GENERIC_READ, FILE_SHARE_READ, IntPtr.Zero, OPEN_EXISTING, "
    "FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, IntPtr.Zero);",
    "    if (handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error());",
    "    BY_HANDLE_FILE_INFORMATION info;",
    "    if (!GetFileInformationByHandle(handle, out info)) {",
    "      handle.Dispose();",
    "      throw new Win32Exception(Marshal.GetLastWin32Error());",
    "    }",
    "    uint unsafeAttributes = FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT;",
    "    if ((info.FileAttributes & unsafeAttributes) != 0) {",
    '      handle.Dispose(); throw new IOException("unsafe image path");',
    "    }",
    "    return handle;",
    "  }",
    "}",
)

_CONFIG_FIELDS = frozenset(
    {
        "schemaVersion",
        "taskName",
        "ownerSid",
        "pythonExecutable",
        "pythonExecutableSha256",
        "moduleRoot",
        "bootstrapSha256",
        "workingDirectory",
        "tokenFile",
        "stateDirectory",
        "logDirectory",
        "host",
        "port",
        "device",
    }
)


class TtsStartupError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int

    @property
    def fingerprint(self) -> str:
        value = f"{self.device}:{self.inode}:{self.size}:{self.mtime_ns}"
        return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    creation_time_ns: int
    executable_path: str
    executable_sha256: str


@dataclass(frozen=True)
class ListenerIdentity:
    pid: int
    creation_time_ns: int


@dataclass(frozen=True)
class TtsStartupConfig:
    path: Path
    config_sha256: str
    config_file_identity: FileIdentity
    owner_sid: str
    python_executable: Path
    python_executable_sha256: str
    module_root: Path
    bootstrap_path: Path
    bootstrap_sha256: str
    bootstrap_source: str
    working_directory: Path
    token_file: Path
    credential_identity: FileIdentity
    state_directory: Path
    log_directory: Path
    host: str
    port: int
    device: str


@dataclass(frozen=True)
class ScheduledTaskDefinition:
    task_name: str
    owner_sid: str
    command: str
    arguments: str
    working_directory: str
    config_sha256: str
    config_file_fingerprint: str
    credential_fingerprint: str
    fingerprint: str


@dataclass(frozen=True)
class ScheduledTaskSnapshot:
    present: bool
    exact: bool = False
    enabled: bool = False
    running: bool = False
    state: str = "Absent"
    owned: bool = False


@dataclass(frozen=True)
class RuntimeState:
    schema_version: int
    config_sha256: str
    config_file_fingerprint: str
    task_fingerprint: str
    credential_fingerprint: str
    runner_pid: int
    runner_creation_time_ns: int
    child_pid: int | None
    child_creation_time_ns: int | None
    phase: str
    restart_count: int
    authenticated_ready: bool


@dataclass(frozen=True)
class LifecycleStatus:
    code: str
    task_present: bool
    task_exact: bool
    task_enabled: bool
    task_running: bool
    config_exact: bool
    credential_exact: bool
    runner_alive: bool
    child_alive: bool
    listener_owned: bool
    authenticated_ready: bool

    def to_public_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LifecyclePlan:
    action: str
    code: str
    allowed: bool
    changed: bool
    steps: tuple[str, ...]
    status: LifecycleStatus

    def to_public_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "code": self.code,
            "allowed": self.allowed,
            "changed": self.changed,
            "steps": list(self.steps),
            "status": self.status.to_public_dict(),
        }


@dataclass(frozen=True)
class RunnerPolicy:
    ready_timeout_seconds: float = DEFAULT_READY_TIMEOUT_SECONDS
    release_timeout_seconds: float = DEFAULT_RELEASE_TIMEOUT_SECONDS
    restart_delay_seconds: float = DEFAULT_RESTART_DELAY_SECONDS
    restart_reset_seconds: float = DEFAULT_RESTART_RESET_SECONDS
    max_restarts: int = DEFAULT_MAX_RESTARTS
    poll_seconds: float = DEFAULT_POLL_SECONDS
    health_reprobe_seconds: float = DEFAULT_HEALTH_REPROBE_SECONDS
    health_failure_threshold: int = DEFAULT_HEALTH_FAILURE_THRESHOLD


class ScheduledTaskBackend(Protocol):
    def query(self, definition: ScheduledTaskDefinition) -> ScheduledTaskSnapshot: ...

    def install(self, definition: ScheduledTaskDefinition, xml: str, xml_path: Path) -> bool: ...

    def start(self, definition: ScheduledTaskDefinition) -> bool: ...

    def disable(self, definition: ScheduledTaskDefinition) -> bool: ...

    def stop(self, definition: ScheduledTaskDefinition) -> bool: ...

    def uninstall(self, definition: ScheduledTaskDefinition) -> bool: ...


class ManagedChild(Protocol):
    @property
    def identity(self) -> ProcessIdentity: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


class RuntimeBackend(Protocol):
    def current_owner_sid(self) -> str: ...

    def current_process(self) -> ProcessIdentity: ...

    def inspect_process(self, pid: int) -> ProcessIdentity | None: ...

    def listeners(self, host: str, port: int) -> tuple[ListenerIdentity, ...]: ...

    def spawn_child(
        self,
        command: Sequence[str],
        *,
        expected_executable: Path,
        expected_executable_sha256: str,
        working_directory: Path,
        log_directory: Path,
    ) -> ManagedChild: ...

    def authenticated_health(
        self,
        host: str,
        port: int,
        token: str,
        *,
        expected_process: ProcessIdentity,
        timeout_seconds: float,
    ) -> bool: ...

    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


def load_startup_config(
    path: Path,
    *,
    current_owner_sid: str | None = None,
) -> TtsStartupConfig:
    raw, metadata = _read_stable_owner_only_file(path, MAX_CONFIG_BYTES, "config_invalid")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TtsStartupError("config_invalid", "TTS startup config is invalid.") from exc
    if not isinstance(payload, dict) or set(payload) != _CONFIG_FIELDS:
        raise TtsStartupError("config_invalid", "TTS startup config is invalid.")
    if (
        type(payload.get("schemaVersion")) is not int
        or payload.get("schemaVersion") != CONFIG_SCHEMA_VERSION
    ):
        raise TtsStartupError("config_invalid", "TTS startup config is invalid.")
    if payload.get("taskName") != TASK_NAME:
        raise TtsStartupError("task_name_invalid", "TTS startup task name is invalid.")

    owner_sid = _required_string(payload, "ownerSid")
    if current_owner_sid is not None and owner_sid != current_owner_sid:
        raise TtsStartupError("owner_mismatch", "TTS startup owner identity does not match.")

    python_executable = _absolute_path(payload, "pythonExecutable")
    module_root = _absolute_path(payload, "moduleRoot")
    working_directory = _absolute_path(payload, "workingDirectory")
    token_file = _absolute_path(payload, "tokenFile")
    state_directory = _absolute_path(payload, "stateDirectory")
    log_directory = _absolute_path(payload, "logDirectory")
    bootstrap_path = module_root / "agent_tools" / "tts_startup_bootstrap.pyw"

    python_sha256 = _required_sha256(payload, "pythonExecutableSha256")
    bootstrap_sha256 = _required_sha256(payload, "bootstrapSha256")
    if sys.platform == "win32" and python_executable.name.casefold() != "pythonw.exe":
        raise TtsStartupError("executable_not_hidden", "TTS startup executable must be windowless.")
    if _sha256_regular_file(python_executable, "executable_invalid") != python_sha256:
        raise TtsStartupError("executable_drift", "TTS startup executable identity changed.")
    bootstrap_bytes = _read_stable_regular_file(
        bootstrap_path, MAX_BOOTSTRAP_BYTES, "bootstrap_invalid"
    )
    if hashlib.sha256(bootstrap_bytes).hexdigest() != bootstrap_sha256:
        raise TtsStartupError("bootstrap_drift", "TTS startup bootstrap identity changed.")
    try:
        bootstrap_source = bootstrap_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise TtsStartupError("bootstrap_invalid", "TTS startup bootstrap is invalid.") from exc
    if not working_directory.is_dir():
        raise TtsStartupError(
            "working_directory_invalid", "TTS startup working directory is invalid."
        )
    _verify_owner_only_directory(state_directory, "state_directory_invalid")
    _verify_owner_only_directory(log_directory, "log_directory_invalid")

    host = _required_string(payload, "host")
    port = payload.get("port")
    if isinstance(port, bool) or not isinstance(port, int):
        raise TtsStartupError("endpoint_invalid", "TTS startup endpoint is invalid.")
    try:
        validate_tts_server_bind(host, port)
    except ValueError as exc:
        raise TtsStartupError("endpoint_invalid", "TTS startup endpoint is invalid.") from exc
    device = _required_string(payload, "device")
    if device not in {"cpu", "cuda"}:
        raise TtsStartupError("device_invalid", "TTS startup device is invalid.")

    _, token_metadata = _read_stable_owner_only_file(token_file, 512, "credential_invalid")
    credential_identity = _file_identity(token_metadata)
    config_identity = _file_identity(metadata)
    if config_identity.fingerprint == credential_identity.fingerprint:
        raise TtsStartupError("credential_invalid", "TTS startup credential is invalid.")
    reserved_paths = (
        state_directory / STATE_FILENAME,
        state_directory / LOCK_FILENAME,
        state_directory / f"{LOCK_FILENAME}.takeover",
        state_directory / STOP_FILENAME,
        state_directory / TASK_XML_FILENAME,
        log_directory / EVENT_LOG_FILENAME,
        log_directory / f"{EVENT_LOG_FILENAME}.1",
        log_directory / f"{EVENT_LOG_FILENAME}.2",
    )
    if any(
        _paths_collide(protected, reserved)
        for protected in (path, token_file)
        for reserved in reserved_paths
    ):
        raise TtsStartupError("protected_path_collision", "TTS startup protected path is reserved.")
    return TtsStartupConfig(
        path=path.resolve(),
        config_sha256=hashlib.sha256(raw).hexdigest(),
        config_file_identity=config_identity,
        owner_sid=owner_sid,
        python_executable=python_executable.resolve(),
        python_executable_sha256=python_sha256,
        module_root=module_root.resolve(),
        bootstrap_path=bootstrap_path.resolve(),
        bootstrap_sha256=bootstrap_sha256,
        bootstrap_source=bootstrap_source,
        working_directory=working_directory.resolve(),
        token_file=token_file.resolve(),
        credential_identity=credential_identity,
        state_directory=state_directory.resolve(),
        log_directory=log_directory.resolve(),
        host=host,
        port=port,
        device=device,
    )


def build_task_definition(config: TtsStartupConfig) -> ScheduledTaskDefinition:
    python_arguments = subprocess.list2cmdline(
        [
            "-I",
            "-c",
            config.bootstrap_source,
            "--module-root",
            str(config.module_root),
            "run",
            "--config",
            str(config.path),
            "--config-sha256",
            config.config_sha256,
            "--config-file-fingerprint",
            config.config_file_identity.fingerprint,
            "--credential-fingerprint",
            config.credential_identity.fingerprint,
        ]
    )
    command, arguments = _trusted_task_launcher(config, python_arguments)
    values = {
        "taskName": TASK_NAME,
        "ownerSid": config.owner_sid,
        "command": command,
        "arguments": arguments,
        "pythonExecutable": str(config.python_executable),
        "pythonExecutableSha256": config.python_executable_sha256,
        "bootstrapPath": str(config.bootstrap_path),
        "bootstrapSha256": config.bootstrap_sha256,
        "workingDirectory": str(config.working_directory),
        "configSha256": config.config_sha256,
        "configFileFingerprint": config.config_file_identity.fingerprint,
        "credentialFingerprint": config.credential_identity.fingerprint,
        "logonType": "InteractiveToken",
        "runLevel": "LeastPrivilege",
        "hidden": True,
        "logonDelay": TASK_LOGON_DELAY,
        "multipleInstances": "IgnoreNew",
        "restartCount": TASK_RESTART_COUNT,
        "restartInterval": TASK_RESTART_INTERVAL,
        "executionTimeLimit": "PT0S",
        "startWhenAvailable": True,
        "allowHardTerminate": True,
        "disallowStartIfOnBatteries": False,
        "stopIfGoingOnBatteries": False,
        "runOnlyIfNetworkAvailable": False,
    }
    fingerprint = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ScheduledTaskDefinition(
        task_name=TASK_NAME,
        owner_sid=config.owner_sid,
        command=command,
        arguments=arguments,
        working_directory=str(config.working_directory),
        config_sha256=config.config_sha256,
        config_file_fingerprint=config.config_file_identity.fingerprint,
        credential_fingerprint=config.credential_identity.fingerprint,
        fingerprint=fingerprint,
    )


def _trusted_task_launcher(config: TtsStartupConfig, python_arguments: str) -> tuple[str, str]:
    command = _windows_system_directory() / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    script = "\n".join(
        (
            "$ErrorActionPreference = 'Stop'",
            "$ProgressPreference = 'SilentlyContinue'",
            "$VerbosePreference = 'SilentlyContinue'",
            "$InformationPreference = 'SilentlyContinue'",
            f"$pythonPath = {_powershell_literal(str(config.python_executable))}",
            f"$expectedHash = {_powershell_literal(config.python_executable_sha256)}",
            f"$arguments = {_powershell_literal(python_arguments)}",
            f"$workingDirectory = {_powershell_literal(str(config.working_directory))}",
            "$nativeSource = @'",
            *_EXACT_IMAGE_NATIVE_SOURCE_LINES,
            "'@",
            "Add-Type -TypeDefinition $nativeSource | Out-Null",
            "$imageHandle = [AgentToolsExactImage]::Open($pythonPath)",
            "$stream = [IO.FileStream]::new($imageHandle, [IO.FileAccess]::Read)",
            "try {",
            "  $sha = [Security.Cryptography.SHA256]::Create()",
            "  try {",
            "    $hashBytes = $sha.ComputeHash($stream)",
            "    $actualHash = [BitConverter]::ToString($hashBytes)"
            ".Replace('-', '').ToLowerInvariant()",
            "  } finally { $sha.Dispose() }",
            "  if ($actualHash -ne $expectedHash) { exit 25 }",
            "  $start = [Diagnostics.ProcessStartInfo]::new()",
            "  $start.FileName = $pythonPath",
            "  $start.Arguments = $arguments",
            "  $start.WorkingDirectory = $workingDirectory",
            "  $start.UseShellExecute = $false",
            "  $start.CreateNoWindow = $true",
            "  $start.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden",
            "  $process = [Diagnostics.Process]::Start($start)",
            "  if ($null -eq $process) { exit 27 }",
            "  $process.WaitForExit()",
            "  exit $process.ExitCode",
            "} catch { exit 27 } finally { $stream.Dispose() }",
        )
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    arguments = subprocess.list2cmdline(
        (
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-EncodedCommand",
            encoded,
        )
    )
    return str(command), arguments


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _windows_system_directory() -> Path:
    if sys.platform != "win32":
        raise TtsStartupError("windows_required", "TTS startup lifecycle requires Windows.")
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
    kernel32.GetSystemDirectoryW.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel32.GetSystemDirectoryW(buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        raise TtsStartupError(
            "system_directory_failed", "TTS startup system directory lookup failed."
        )
    return Path(buffer.value).resolve()


def render_scheduled_task_xml(definition: ScheduledTaskDefinition) -> str:
    namespace = "http://schemas.microsoft.com/windows/2004/02/mit/task"
    ET.register_namespace("", namespace)
    task = ET.Element(f"{{{namespace}}}Task", {"version": "1.4"})
    registration = ET.SubElement(task, f"{{{namespace}}}RegistrationInfo")
    description = ET.SubElement(registration, f"{{{namespace}}}Description")
    description.text = TASK_DESCRIPTION_PREFIX + definition.fingerprint
    triggers = ET.SubElement(task, f"{{{namespace}}}Triggers")
    logon = ET.SubElement(triggers, f"{{{namespace}}}LogonTrigger")
    ET.SubElement(logon, f"{{{namespace}}}Enabled").text = "true"
    ET.SubElement(logon, f"{{{namespace}}}UserId").text = definition.owner_sid
    ET.SubElement(logon, f"{{{namespace}}}Delay").text = TASK_LOGON_DELAY
    principals = ET.SubElement(task, f"{{{namespace}}}Principals")
    principal = ET.SubElement(principals, f"{{{namespace}}}Principal", {"id": "Owner"})
    ET.SubElement(principal, f"{{{namespace}}}UserId").text = definition.owner_sid
    ET.SubElement(principal, f"{{{namespace}}}LogonType").text = "InteractiveToken"
    ET.SubElement(principal, f"{{{namespace}}}RunLevel").text = "LeastPrivilege"
    settings = ET.SubElement(task, f"{{{namespace}}}Settings")
    ET.SubElement(settings, f"{{{namespace}}}MultipleInstancesPolicy").text = "IgnoreNew"
    ET.SubElement(settings, f"{{{namespace}}}DisallowStartIfOnBatteries").text = "false"
    ET.SubElement(settings, f"{{{namespace}}}StopIfGoingOnBatteries").text = "false"
    ET.SubElement(settings, f"{{{namespace}}}AllowHardTerminate").text = "true"
    ET.SubElement(settings, f"{{{namespace}}}StartWhenAvailable").text = "true"
    ET.SubElement(settings, f"{{{namespace}}}RunOnlyIfNetworkAvailable").text = "false"
    ET.SubElement(settings, f"{{{namespace}}}ExecutionTimeLimit").text = "PT0S"
    ET.SubElement(settings, f"{{{namespace}}}Enabled").text = "true"
    ET.SubElement(settings, f"{{{namespace}}}Hidden").text = "true"
    restart = ET.SubElement(settings, f"{{{namespace}}}RestartOnFailure")
    ET.SubElement(restart, f"{{{namespace}}}Interval").text = TASK_RESTART_INTERVAL
    ET.SubElement(restart, f"{{{namespace}}}Count").text = str(TASK_RESTART_COUNT)
    actions = ET.SubElement(task, f"{{{namespace}}}Actions", {"Context": "Owner"})
    execute = ET.SubElement(actions, f"{{{namespace}}}Exec")
    ET.SubElement(execute, f"{{{namespace}}}Command").text = definition.command
    ET.SubElement(execute, f"{{{namespace}}}Arguments").text = definition.arguments
    ET.SubElement(execute, f"{{{namespace}}}WorkingDirectory").text = definition.working_directory
    return ET.tostring(task, encoding="unicode", xml_declaration=True)


def inspect_lifecycle(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    task: ScheduledTaskSnapshot,
    runtime: RuntimeBackend,
) -> LifecycleStatus:
    state_path = config.state_directory / STATE_FILENAME
    state = _read_runtime_state(state_path)
    listeners = runtime.listeners(config.host, config.port)
    if not task.present:
        code = "foreign_listener" if listeners else "task_absent"
        return _status(code, task, False, False, False, False, False, False)
    if not task.exact and not task.owned:
        return _status("foreign_task", task, False, False, False, False, False, False)
    if state_path.exists() and state is None:
        code = "foreign_listener" if listeners else "state_invalid"
        return _status(code, task, False, False, False, False, False, False)
    if state is None:
        if listeners:
            code = "foreign_listener"
        elif task.exact and task.running:
            code = "starting"
        else:
            code = "stopped" if task.exact else "task_drift"
        return _status(code, task, True, True, False, False, False, False)
    config_exact = (
        state.config_sha256 == config.config_sha256
        and state.config_file_fingerprint == config.config_file_identity.fingerprint
    )
    credential_exact = state.credential_fingerprint == config.credential_identity.fingerprint
    runner = runtime.inspect_process(state.runner_pid)
    runner_alive = _process_matches(
        runner,
        state.runner_creation_time_ns,
        config.python_executable,
        config.python_executable_sha256,
    )
    child = runtime.inspect_process(state.child_pid) if state.child_pid is not None else None
    child_alive = _process_matches(
        child,
        state.child_creation_time_ns,
        config.python_executable,
        config.python_executable_sha256,
    )
    listener_owned = (
        child_alive
        and child is not None
        and len(listeners) == 1
        and listeners[0].pid == child.pid
        and listeners[0].creation_time_ns == child.creation_time_ns
    )
    ready = False
    if config_exact and credential_exact and runner_alive and listener_owned:
        try:
            token = load_owner_only_bearer_token(config.token_file)
            before = runtime.listeners(config.host, config.port)
            if child is not None and _listeners_match_child(before, child):
                ready = runtime.authenticated_health(
                    config.host,
                    config.port,
                    token,
                    expected_process=child,
                    timeout_seconds=3.0,
                )
                after = runtime.listeners(config.host, config.port)
                ready = ready and _listeners_match_child(after, child)
        except RuntimeError:
            ready = False
    if not config_exact:
        code = "config_drift"
    elif not credential_exact:
        code = "credential_drift"
    elif state.task_fingerprint != definition.fingerprint or not task.exact:
        code = "task_drift"
    elif listeners and not listener_owned:
        code = "foreign_listener"
    elif not runner_alive and not child_alive and not listeners:
        code = "stale_state"
    elif ready:
        code = "ready"
    elif runner_alive:
        code = "starting"
    else:
        code = "state_mismatch"
    return LifecycleStatus(
        code=code,
        task_present=task.present,
        task_exact=task.exact,
        task_enabled=task.enabled,
        task_running=task.running,
        config_exact=config_exact,
        credential_exact=credential_exact,
        runner_alive=runner_alive,
        child_alive=child_alive,
        listener_owned=listener_owned,
        authenticated_ready=ready,
    )


def plan_lifecycle(action: str, status: LifecycleStatus) -> LifecyclePlan:
    if action == "status":
        return LifecyclePlan(action, status.code, True, False, (), status)
    if action == "install":
        if status.task_present and not status.task_exact:
            return _refused_plan(action, status.code, status)
        if status.task_exact:
            return LifecyclePlan(action, "already_installed", True, False, (), status)
        return LifecyclePlan(action, "install", True, True, ("register_exact_task",), status)
    if not status.task_present and action == "stop":
        if status.code == "foreign_listener":
            return _refused_plan(action, status.code, status)
        return LifecyclePlan(action, "already_stopped", True, False, (), status)
    if not status.task_present and action == "uninstall":
        return LifecyclePlan(action, "already_uninstalled", True, False, (), status)
    if not status.task_present:
        return _refused_plan(action, "task_absent", status)
    if not status.task_exact:
        return _refused_plan(action, "foreign_task", status)
    if action == "start":
        if status.code in {"ready", "starting"}:
            return LifecyclePlan(action, "already_started", True, False, (), status)
        if status.code not in {"stopped", "stale_state"}:
            return _refused_plan(action, status.code, status)
        steps = ("remove_owned_stale_state", "clear_owned_stop_request", "enable_and_start")
        return LifecyclePlan(action, "start", True, True, steps, status)
    if action == "stop":
        if status.code == "foreign_listener":
            foreign_steps = ["disable"] if status.task_enabled else []
            if status.task_running:
                foreign_steps.extend(("stop_exact_task", "wait_task_release"))
            return LifecyclePlan(
                action,
                "foreign_listener_preserved",
                True,
                bool(foreign_steps),
                tuple(foreign_steps),
                status,
            )
        if status.task_running and not status.runner_alive:
            return LifecyclePlan(
                action,
                "stop_startup_window",
                True,
                True,
                ("disable", "stop_exact_task", "wait_task_and_listener_release"),
                status,
            )
        if status.code in {"config_drift", "credential_drift", "task_drift"}:
            return _refused_plan(action, status.code, status)
        if (
            not status.task_running
            and not status.runner_alive
            and not status.child_alive
            and not status.listener_owned
        ):
            changed = status.task_enabled
            stop_steps: tuple[str, ...] = ("disable",) if changed else ()
            return LifecyclePlan(action, "already_stopped", True, changed, stop_steps, status)
        return LifecyclePlan(
            action,
            "stop",
            True,
            True,
            ("disable", "request_owned_stop", "wait_process_release", "wait_listener_release"),
            status,
        )
    if action == "uninstall":
        if status.code not in {"stopped", "stale_state"}:
            return _refused_plan(action, "not_stopped", status)
        return LifecyclePlan(
            action,
            "uninstall",
            True,
            True,
            ("disable", "remove_exact_task", "remove_owned_runtime_artifacts"),
            status,
        )
    raise ValueError(f"Unsupported TTS startup action: {action}")


def execute_lifecycle(
    action: str,
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    scheduler: ScheduledTaskBackend,
    runtime: RuntimeBackend,
    *,
    plan_only: bool = False,
    policy: RunnerPolicy | None = None,
) -> LifecyclePlan:
    task = scheduler.query(definition)
    status = inspect_lifecycle(config, definition, task, runtime)
    plan = plan_lifecycle(action, status)
    if action == "uninstall" and not task.present and _owned_artifacts_present(config):
        _validate_owned_install_artifacts(config, definition, runtime)
        plan = LifecyclePlan(
            action,
            "remove_owned_artifacts",
            True,
            True,
            ("remove_owned_runtime_artifacts",),
            status,
        )
    if plan_only or not plan.allowed or not plan.changed:
        return plan
    policy = policy or RunnerPolicy()
    if action == "install":
        _validate_owned_install_artifacts(config, definition, runtime)
        xml = render_scheduled_task_xml(definition)
        _write_private_text(config.state_directory / TASK_XML_FILENAME, xml)
        scheduler.install(
            definition,
            xml,
            config.state_directory / TASK_XML_FILENAME,
        )
    elif action == "start":
        _remove_owned_state_artifacts(config, definition, allow_runtime_state=True)
        scheduler.start(definition)
    elif action == "stop":
        scheduler.disable(definition)
        if plan.code == "stop":
            _write_stop_request(config, definition)
            _wait_for_external_stop(config, definition, scheduler, runtime, policy)
        elif plan.code == "stop_startup_window":
            scheduler.stop(definition)
            _wait_for_external_stop(config, definition, scheduler, runtime, policy)
        elif plan.code == "foreign_listener_preserved" and status.task_running:
            scheduler.stop(definition)
            _wait_for_task_release(definition, scheduler, runtime, policy)
    elif action == "uninstall":
        _validate_owned_install_artifacts(config, definition, runtime)
        if task.present:
            scheduler.disable(definition)
            scheduler.uninstall(definition)
        _remove_owned_state_artifacts(config, definition, allow_runtime_state=True)
        _remove_owned_install_artifacts(config)
    return plan


def run_owned_runner(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runtime: RuntimeBackend,
    *,
    expected_config_sha256: str,
    expected_config_file_fingerprint: str,
    expected_credential_fingerprint: str,
    policy: RunnerPolicy | None = None,
) -> int:
    policy = policy or RunnerPolicy()
    if config.config_sha256 != expected_config_sha256:
        raise TtsStartupError("config_drift", "TTS startup config identity changed.")
    if config.config_file_identity.fingerprint != expected_config_file_fingerprint:
        raise TtsStartupError("config_drift", "TTS startup config identity changed.")
    if config.credential_identity.fingerprint != expected_credential_fingerprint:
        raise TtsStartupError("credential_drift", "TTS startup credential identity changed.")
    if runtime.current_owner_sid() != config.owner_sid:
        raise TtsStartupError("owner_mismatch", "TTS startup owner identity does not match.")
    runner = runtime.current_process()
    if (
        Path(runner.executable_path).resolve() != config.python_executable
        or runner.executable_sha256 != config.python_executable_sha256
    ):
        raise TtsStartupError("executable_drift", "TTS startup executable identity changed.")
    lock_path = config.state_directory / LOCK_FILENAME
    _acquire_runner_lock(lock_path, config, definition, runner, runtime)
    active_child: ManagedChild | None = None
    try:
        if runtime.listeners(config.host, config.port):
            _publish_state(config, definition, runner, None, "foreign_listener", 0, False)
            return 20
        _publish_state(config, definition, runner, None, "starting", 0, False)
        restarts = 0
        while True:
            if _stop_requested(config, definition, runner):
                _publish_state(config, definition, runner, None, "stopped", restarts, False)
                return 0
            drift = _runtime_input_drift(config)
            if drift is not None:
                _publish_state(config, definition, runner, None, drift, restarts, False)
                return _DRIFT_EXIT_CODES[drift]
            child = runtime.spawn_child(
                _server_command(config),
                expected_executable=config.python_executable,
                expected_executable_sha256=config.python_executable_sha256,
                working_directory=config.working_directory,
                log_directory=config.log_directory,
            )
            active_child = child
            child_drift = _child_identity_drift(config, child.identity)
            if child_drift is not None:
                _terminate_and_release(config, child, runtime, policy)
                _publish_state(
                    config, definition, runner, child.identity, child_drift, restarts, False
                )
                return _DRIFT_EXIT_CODES[child_drift]
            drift = _runtime_input_drift(config)
            if drift is not None:
                _terminate_and_release(config, child, runtime, policy)
                _publish_state(config, definition, runner, child.identity, drift, restarts, False)
                return _DRIFT_EXIT_CODES[drift]
            _publish_state(config, definition, runner, child.identity, "starting", restarts, False)
            outcome, ready_at = _wait_for_ready_or_exit(
                config,
                definition,
                runner,
                child,
                runtime,
                policy,
                restarts,
            )
            if outcome == "stopped":
                return _finish_intentional_stop(
                    config, definition, runner, child, runtime, policy, restarts
                )
            if outcome == "foreign_listener":
                _terminate_and_release(config, child, runtime, policy)
                _publish_state(config, definition, runner, child.identity, outcome, restarts, False)
                return 20
            if outcome in _DRIFT_EXIT_CODES:
                _terminate_and_release(config, child, runtime, policy)
                _publish_state(config, definition, runner, child.identity, outcome, restarts, False)
                return _DRIFT_EXIT_CODES[outcome]
            if outcome == "ready":
                outcome = _monitor_ready_child(
                    config,
                    definition,
                    runner,
                    child,
                    runtime,
                    policy,
                    restarts,
                )
                if outcome == "stopped":
                    return _finish_intentional_stop(
                        config, definition, runner, child, runtime, policy, restarts
                    )
                if outcome == "foreign_listener":
                    _terminate_and_release(config, child, runtime, policy)
                    _publish_state(
                        config, definition, runner, child.identity, outcome, restarts, False
                    )
                    return 20
                if outcome in _DRIFT_EXIT_CODES:
                    _terminate_and_release(config, child, runtime, policy)
                    _publish_state(
                        config, definition, runner, child.identity, outcome, restarts, False
                    )
                    return _DRIFT_EXIT_CODES[outcome]
                if outcome == "health_failed":
                    _terminate_and_release(config, child, runtime, policy)
                    _publish_state(
                        config, definition, runner, child.identity, outcome, restarts, False
                    )
            else:
                _terminate_and_release(config, child, runtime, policy)
            released = _wait_for_release(config, child.identity, runtime, policy)
            if not released:
                _publish_state(
                    config,
                    definition,
                    runner,
                    child.identity,
                    "listener_release_timeout",
                    restarts,
                    False,
                )
                return 21
            active_child = None
            ran_for = 0.0 if ready_at is None else max(0.0, runtime.now() - ready_at)
            restarts = 0 if ran_for >= policy.restart_reset_seconds else restarts + 1
            if restarts > policy.max_restarts:
                _publish_state(
                    config, definition, runner, None, "restart_exhausted", restarts, False
                )
                return 22
            _publish_state(config, definition, runner, None, "restart_wait", restarts, False)
            delay_outcome = _interruptible_restart_delay(
                config, definition, runner, runtime, policy
            )
            if delay_outcome == "stopped":
                _publish_state(config, definition, runner, None, "stopped", restarts, False)
                return 0
            if delay_outcome in _DRIFT_EXIT_CODES:
                _publish_state(config, definition, runner, None, delay_outcome, restarts, False)
                return _DRIFT_EXIT_CODES[delay_outcome]
    finally:
        try:
            if active_child is not None:
                try:
                    _terminate_and_release(config, active_child, runtime, policy)
                except Exception:
                    try:
                        if active_child.poll() is None:
                            active_child.terminate()
                    except Exception:
                        pass
        finally:
            _remove_exact_lock(lock_path, config, definition, runner)


def _probe_child_authenticated_health(
    config: TtsStartupConfig,
    child: ManagedChild,
    runtime: RuntimeBackend,
    policy: RunnerPolicy,
) -> str:
    """Probe the owned child's authenticated health using the startup client.

    Returns "healthy", "unhealthy", "foreign_listener", or "readiness_failed" (token
    unreadable). The listener generation is rechecked on both sides of the bearer use.
    """
    try:
        token = load_owner_only_bearer_token(config.token_file)
    except RuntimeError:
        return "readiness_failed"
    before = runtime.listeners(config.host, config.port)
    if not _listeners_match_child(before, child.identity):
        return "foreign_listener"
    if not runtime.authenticated_health(
        config.host,
        config.port,
        token,
        expected_process=child.identity,
        timeout_seconds=min(3.0, policy.ready_timeout_seconds),
    ):
        return "unhealthy"
    after = runtime.listeners(config.host, config.port)
    if not _listeners_match_child(after, child.identity):
        return "foreign_listener"
    return "healthy"


def _wait_for_ready_or_exit(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runner: ProcessIdentity,
    child: ManagedChild,
    runtime: RuntimeBackend,
    policy: RunnerPolicy,
    restarts: int,
) -> tuple[str, float | None]:
    deadline = runtime.now() + policy.ready_timeout_seconds
    while runtime.now() < deadline:
        drift = _runtime_input_drift(config)
        if drift is not None:
            return drift, None
        if _stop_requested(config, definition, runner):
            return "stopped", None
        if child.poll() is not None:
            return "crashed", None
        listeners = runtime.listeners(config.host, config.port)
        if listeners:
            if not _listeners_match_child(listeners, child.identity):
                return "foreign_listener", None
            probe = _probe_child_authenticated_health(config, child, runtime, policy)
            if probe == "readiness_failed":
                return "readiness_failed", None
            if probe == "foreign_listener":
                return "foreign_listener", None
            if probe == "healthy":
                ready_at = runtime.now()
                _publish_state(config, definition, runner, child.identity, "ready", restarts, True)
                return "ready", ready_at
        runtime.sleep(policy.poll_seconds)
    return "readiness_failed", None


def _monitor_ready_child(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runner: ProcessIdentity,
    child: ManagedChild,
    runtime: RuntimeBackend,
    policy: RunnerPolicy,
    restarts: int,
) -> str:
    next_health_check = runtime.now() + policy.health_reprobe_seconds
    consecutive_health_failures = 0
    while True:
        drift = _runtime_input_drift(config)
        if drift is not None:
            return drift
        if _stop_requested(config, definition, runner):
            return "stopped"
        if child.poll() is not None:
            return "crashed"
        listeners = runtime.listeners(config.host, config.port)
        if not _listeners_match_child(listeners, child.identity):
            return "foreign_listener"
        if runtime.now() >= next_health_check:
            probe = _probe_child_authenticated_health(config, child, runtime, policy)
            next_health_check = runtime.now() + policy.health_reprobe_seconds
            if probe == "foreign_listener":
                return "foreign_listener"
            if probe == "healthy":
                consecutive_health_failures = 0
            else:
                consecutive_health_failures += 1
                if consecutive_health_failures >= policy.health_failure_threshold:
                    return "health_failed"
        _publish_state(config, definition, runner, child.identity, "ready", restarts, True)
        runtime.sleep(policy.poll_seconds)


def _finish_intentional_stop(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runner: ProcessIdentity,
    child: ManagedChild,
    runtime: RuntimeBackend,
    policy: RunnerPolicy,
    restarts: int,
) -> int:
    released = _terminate_and_release(config, child, runtime, policy)
    phase = "stopped" if released else "listener_release_timeout"
    _publish_state(config, definition, runner, child.identity, phase, restarts, False)
    return 0 if released else 21


def _terminate_and_release(
    config: TtsStartupConfig,
    child: ManagedChild,
    runtime: RuntimeBackend,
    policy: RunnerPolicy,
) -> bool:
    if child.poll() is None:
        child.terminate()
    return _wait_for_release(config, child.identity, runtime, policy)


def _wait_for_release(
    config: TtsStartupConfig,
    child: ProcessIdentity,
    runtime: RuntimeBackend,
    policy: RunnerPolicy,
) -> bool:
    deadline = runtime.now() + policy.release_timeout_seconds
    while True:
        process = runtime.inspect_process(child.pid)
        process_released = process is None or process.creation_time_ns != child.creation_time_ns
        listeners = runtime.listeners(config.host, config.port)
        listener_released = not listeners
        if process_released and listener_released:
            return True
        if runtime.now() >= deadline:
            return False
        runtime.sleep(policy.poll_seconds)


def _interruptible_restart_delay(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runner: ProcessIdentity,
    runtime: RuntimeBackend,
    policy: RunnerPolicy,
) -> str | None:
    deadline = runtime.now() + policy.restart_delay_seconds
    while runtime.now() < deadline:
        drift = _runtime_input_drift(config)
        if drift is not None:
            return drift
        if _stop_requested(config, definition, runner):
            return "stopped"
        runtime.sleep(min(policy.poll_seconds, max(0.0, deadline - runtime.now())))
    if _stop_requested(config, definition, runner):
        return "stopped"
    return _runtime_input_drift(config)


def _server_command(config: TtsStartupConfig) -> tuple[str, ...]:
    return (
        str(config.python_executable),
        "-I",
        "-c",
        config.bootstrap_source,
        "--module-root",
        str(config.module_root),
        "serve",
        "--config",
        str(config.path),
        "--config-sha256",
        config.config_sha256,
        "--config-file-fingerprint",
        config.config_file_identity.fingerprint,
        "--credential-fingerprint",
        config.credential_identity.fingerprint,
    )


def _publish_state(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runner: ProcessIdentity,
    child: ProcessIdentity | None,
    phase: str,
    restart_count: int,
    authenticated_ready: bool,
) -> None:
    state = RuntimeState(
        schema_version=1,
        config_sha256=config.config_sha256,
        config_file_fingerprint=config.config_file_identity.fingerprint,
        task_fingerprint=definition.fingerprint,
        credential_fingerprint=config.credential_identity.fingerprint,
        runner_pid=runner.pid,
        runner_creation_time_ns=runner.creation_time_ns,
        child_pid=None if child is None else child.pid,
        child_creation_time_ns=None if child is None else child.creation_time_ns,
        phase=phase,
        restart_count=restart_count,
        authenticated_ready=authenticated_ready,
    )
    _write_private_json(config.state_directory / STATE_FILENAME, asdict(state))


def _read_runtime_state(path: Path) -> RuntimeState | None:
    if not path.exists():
        return None
    try:
        raw, _ = _read_stable_owner_only_file(path, MAX_STATE_BYTES, "state_invalid")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != _RUNTIME_STATE_FIELDS:
            return None
        if not _valid_runtime_state_payload(payload):
            return None
        return RuntimeState(**payload)
    except (OSError, TypeError, ValueError, TtsStartupError, UnicodeError, json.JSONDecodeError):
        return None


def _valid_runtime_state_payload(payload: Mapping[str, object]) -> bool:
    hashes = (
        payload.get("config_sha256"),
        payload.get("config_file_fingerprint"),
        payload.get("task_fingerprint"),
        payload.get("credential_fingerprint"),
    )
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in hashes
        )
    ):
        return False
    if not _valid_windows_pid(payload.get("runner_pid")) or not _valid_windows_timestamp(
        payload.get("runner_creation_time_ns")
    ):
        return False
    child_pid = payload.get("child_pid")
    child_creation = payload.get("child_creation_time_ns")
    child_values_valid = child_pid is None and child_creation is None
    child_values_valid = child_values_valid or (
        _valid_windows_pid(child_pid) and _valid_windows_timestamp(child_creation)
    )
    restart_count = payload.get("restart_count")
    ready = payload.get("authenticated_ready")
    phase = payload.get("phase")
    return (
        child_values_valid
        and isinstance(restart_count, int)
        and not isinstance(restart_count, bool)
        and restart_count >= 0
        and type(ready) is bool
        and isinstance(phase, str)
        and phase in _RUNTIME_PHASES
        and ready == (phase == "ready")
        and (phase != "ready" or child_pid is not None)
    )


def _valid_windows_pid(value: object) -> bool:
    return type(value) is int and 0 < value <= MAX_WINDOWS_PID


def _valid_windows_timestamp(value: object) -> bool:
    return type(value) is int and 0 < value <= MAX_WINDOWS_TIMESTAMP


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_stop_request(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
) -> None:
    state = _read_runtime_state(config.state_directory / STATE_FILENAME)
    if state is None:
        return
    payload: dict[str, object] = {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "config_file_fingerprint": config.config_file_identity.fingerprint,
        "task_fingerprint": definition.fingerprint,
        "runner_pid": state.runner_pid,
        "runner_creation_time_ns": state.runner_creation_time_ns,
    }
    path = config.state_directory / STOP_FILENAME
    if path.exists():
        existing = _read_control_file(path, "stop_request_invalid")
        if not _valid_stop_request_payload(existing) or existing != payload:
            raise TtsStartupError("stop_request_invalid", "TTS startup stop request is invalid.")
        return
    _write_new_private_json(path, payload)


def _stop_requested(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runner: ProcessIdentity,
) -> bool:
    path = config.state_directory / STOP_FILENAME
    if not path.exists():
        return False
    try:
        raw, _ = _read_stable_owner_only_file(path, MAX_STATE_BYTES, "stop_request_invalid")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, TtsStartupError, UnicodeError, json.JSONDecodeError):
        return False
    expected: dict[str, object] = {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "config_file_fingerprint": config.config_file_identity.fingerprint,
        "task_fingerprint": definition.fingerprint,
        "runner_pid": runner.pid,
        "runner_creation_time_ns": runner.creation_time_ns,
    }
    return (
        isinstance(payload, dict) and _valid_stop_request_payload(payload) and payload == expected
    )


def _valid_stop_request_payload(payload: Mapping[str, object]) -> bool:
    return (
        set(payload) == _STOP_REQUEST_FIELDS
        and type(payload.get("schema_version")) is int
        and payload.get("schema_version") == 1
        and _valid_windows_pid(payload.get("runner_pid"))
        and _valid_windows_timestamp(payload.get("runner_creation_time_ns"))
        and all(
            _is_sha256(payload.get(field))
            for field in (
                "config_sha256",
                "config_file_fingerprint",
                "task_fingerprint",
            )
        )
    )


def _acquire_runner_lock(
    path: Path,
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runner: ProcessIdentity,
    runtime: RuntimeBackend,
) -> None:
    payload = {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "config_file_fingerprint": config.config_file_identity.fingerprint,
        "task_fingerprint": definition.fingerprint,
        "runner_pid": runner.pid,
        "runner_creation_time_ns": runner.creation_time_ns,
    }
    try:
        _write_new_private_json(path, payload)
        return
    except FileExistsError:
        pass
    takeover = path.with_name(path.name + ".takeover")
    _acquire_takeover_guard(takeover, payload, config, definition, runtime)
    try:
        existing = _read_lock(path)
        if existing is None:
            raise TtsStartupError("runner_lock_invalid", "TTS startup runner lock is invalid.")
        existing_pid = existing.get("runner_pid")
        existing_generation = existing.get("runner_creation_time_ns")
        if not _valid_windows_pid(existing_pid) or not _valid_windows_timestamp(
            existing_generation
        ):
            raise TtsStartupError("runner_lock_invalid", "TTS startup runner lock is invalid.")
        assert isinstance(existing_pid, int)
        assert isinstance(existing_generation, int)
        process = runtime.inspect_process(existing_pid)
        same_generation = process is not None and process.creation_time_ns == existing_generation
        if same_generation:
            raise TtsStartupError("runner_already_active", "TTS startup runner is already active.")
        if not _lock_matches_definition(existing, config, definition):
            raise TtsStartupError("runner_lock_invalid", "TTS startup runner lock is invalid.")
        if _read_lock(path) != existing:
            raise TtsStartupError("runner_lock_contended", "TTS startup runner lock changed.")
        path.unlink()
        _write_new_private_json(path, payload)
    finally:
        if _read_lock(takeover) == payload:
            takeover.unlink(missing_ok=True)


def _acquire_takeover_guard(
    path: Path,
    payload: Mapping[str, object],
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runtime: RuntimeBackend,
) -> None:
    try:
        _write_new_private_json(path, payload)
        return
    except FileExistsError:
        pass
    existing = _read_lock(path)
    if existing is None or not _lock_matches_definition(existing, config, definition):
        raise TtsStartupError("runner_lock_invalid", "TTS startup runner takeover lock is invalid.")
    existing_pid = existing.get("runner_pid")
    existing_generation = existing.get("runner_creation_time_ns")
    if not _valid_windows_pid(existing_pid) or not _valid_windows_timestamp(existing_generation):
        raise TtsStartupError("runner_lock_invalid", "TTS startup runner takeover lock is invalid.")
    assert isinstance(existing_pid, int)
    assert isinstance(existing_generation, int)
    process = runtime.inspect_process(existing_pid)
    if process is not None and process.creation_time_ns == existing_generation:
        raise TtsStartupError("runner_lock_contended", "TTS startup runner lock is contended.")
    if _read_lock(path) != existing:
        raise TtsStartupError("runner_lock_contended", "TTS startup runner lock changed.")
    path.unlink()
    try:
        _write_new_private_json(path, payload)
    except FileExistsError as exc:
        raise TtsStartupError(
            "runner_lock_contended", "TTS startup runner lock is contended."
        ) from exc


def _lock_matches_definition(
    payload: Mapping[str, object],
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
) -> bool:
    return (
        set(payload) == _LOCK_FIELDS
        and type(payload.get("schema_version")) is int
        and payload.get("schema_version") == 1
        and payload.get("config_sha256") == config.config_sha256
        and payload.get("config_file_fingerprint") == config.config_file_identity.fingerprint
        and payload.get("task_fingerprint") == definition.fingerprint
    )


def _read_lock(path: Path) -> dict[str, object] | None:
    try:
        raw, _ = _read_stable_owner_only_file(path, MAX_STATE_BYTES, "runner_lock_invalid")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, TtsStartupError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _remove_exact_lock(
    path: Path,
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runner: ProcessIdentity,
) -> None:
    existing = _read_lock(path)
    if existing == {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "config_file_fingerprint": config.config_file_identity.fingerprint,
        "task_fingerprint": definition.fingerprint,
        "runner_pid": runner.pid,
        "runner_creation_time_ns": runner.creation_time_ns,
    }:
        path.unlink(missing_ok=True)


def _wait_for_external_stop(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    scheduler: ScheduledTaskBackend,
    runtime: RuntimeBackend,
    policy: RunnerPolicy,
) -> None:
    deadline = runtime.now() + policy.release_timeout_seconds
    while runtime.now() < deadline:
        state = _read_runtime_state(config.state_directory / STATE_FILENAME)
        listeners = runtime.listeners(config.host, config.port)
        task_running = scheduler.query(definition).running
        if state is None:
            if not task_running and not listeners:
                return
        else:
            runner = runtime.inspect_process(state.runner_pid)
            child = runtime.inspect_process(state.child_pid) if state.child_pid else None
            runner_released = (
                runner is None or runner.creation_time_ns != state.runner_creation_time_ns
            )
            child_released = child is None or child.creation_time_ns != state.child_creation_time_ns
            if not task_running and runner_released and child_released and not listeners:
                return
        runtime.sleep(policy.poll_seconds)
    raise TtsStartupError(
        "stop_timeout", "TTS startup stop did not prove process and listener release."
    )


def _wait_for_task_release(
    definition: ScheduledTaskDefinition,
    scheduler: ScheduledTaskBackend,
    runtime: RuntimeBackend,
    policy: RunnerPolicy,
) -> None:
    deadline = runtime.now() + policy.release_timeout_seconds
    while runtime.now() < deadline:
        if not scheduler.query(definition).running:
            return
        runtime.sleep(policy.poll_seconds)
    raise TtsStartupError("stop_timeout", "TTS startup task instance did not stop.")


def _remove_owned_state_artifacts(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    *,
    allow_runtime_state: bool,
) -> None:
    stop_path = config.state_directory / STOP_FILENAME
    state_path = config.state_directory / STATE_FILENAME
    remove_stop = False
    if stop_path.exists():
        stop_request = _read_control_file(stop_path, "stop_request_invalid")
        if not _valid_stop_request_payload(stop_request) or any(
            stop_request.get(key) != value
            for key, value in {
                "config_sha256": config.config_sha256,
                "config_file_fingerprint": config.config_file_identity.fingerprint,
                "task_fingerprint": definition.fingerprint,
            }.items()
        ):
            raise TtsStartupError("stop_request_invalid", "TTS startup stop request is invalid.")
        remove_stop = True
    remove_state = allow_runtime_state and state_path.exists()
    if remove_state:
        state = _read_runtime_state(state_path)
        if state is None:
            raise TtsStartupError("state_invalid", "TTS startup state is invalid.")
        if (
            state.config_sha256 != config.config_sha256
            or state.config_file_fingerprint != config.config_file_identity.fingerprint
            or state.task_fingerprint != definition.fingerprint
            or state.credential_fingerprint != config.credential_identity.fingerprint
        ):
            raise TtsStartupError("state_drift", "TTS startup state identity changed.")
    if remove_stop:
        stop_path.unlink()
    if remove_state:
        state_path.unlink()


def _validate_owned_install_artifacts(
    config: TtsStartupConfig,
    definition: ScheduledTaskDefinition,
    runtime: RuntimeBackend,
) -> None:
    xml_path = config.state_directory / TASK_XML_FILENAME
    if xml_path.exists():
        raw, _ = _read_stable_owner_only_file(xml_path, MAX_CONFIG_BYTES, "task_xml_invalid")
        if raw.decode("utf-8") != render_scheduled_task_xml(definition):
            raise TtsStartupError("task_xml_invalid", "TTS startup task XML is invalid.")
    for path in (
        config.state_directory / LOCK_FILENAME,
        config.state_directory / f"{LOCK_FILENAME}.takeover",
    ):
        if not path.exists():
            continue
        payload = _read_lock(path)
        if payload is None or not _lock_matches_definition(payload, config, definition):
            raise TtsStartupError("runner_lock_invalid", "TTS startup runner lock is invalid.")
        pid = payload.get("runner_pid")
        generation = payload.get("runner_creation_time_ns")
        if not _valid_windows_pid(pid) or not _valid_windows_timestamp(generation):
            raise TtsStartupError("runner_lock_invalid", "TTS startup runner lock is invalid.")
        assert isinstance(pid, int)
        assert isinstance(generation, int)
        process = runtime.inspect_process(pid)
        if process is not None and process.creation_time_ns == generation:
            raise TtsStartupError("runner_already_active", "TTS startup runner is still active.")


def _remove_owned_install_artifacts(config: TtsStartupConfig) -> None:
    for path in (
        config.state_directory / LOCK_FILENAME,
        config.state_directory / f"{LOCK_FILENAME}.takeover",
        config.state_directory / TASK_XML_FILENAME,
    ):
        path.unlink(missing_ok=True)


def _owned_artifacts_present(config: TtsStartupConfig) -> bool:
    return any(
        path.exists()
        for path in (
            config.state_directory / STATE_FILENAME,
            config.state_directory / STOP_FILENAME,
            config.state_directory / LOCK_FILENAME,
            config.state_directory / f"{LOCK_FILENAME}.takeover",
            config.state_directory / TASK_XML_FILENAME,
        )
    )


def _process_matches(
    process: ProcessIdentity | None,
    creation_time_ns: int | None,
    executable_path: Path,
    executable_sha256: str,
) -> bool:
    return (
        process is not None
        and creation_time_ns is not None
        and process.creation_time_ns == creation_time_ns
        and Path(process.executable_path).resolve() == executable_path
        and process.executable_sha256 == executable_sha256
    )


def _listeners_match_child(
    listeners: Sequence[ListenerIdentity],
    child: ProcessIdentity,
) -> bool:
    return (
        len(listeners) == 1
        and listeners[0].pid == child.pid
        and listeners[0].creation_time_ns == child.creation_time_ns
    )


def _status(
    code: str,
    task: ScheduledTaskSnapshot,
    config_exact: bool,
    credential_exact: bool,
    runner_alive: bool,
    child_alive: bool,
    listener_owned: bool,
    authenticated_ready: bool,
) -> LifecycleStatus:
    return LifecycleStatus(
        code=code,
        task_present=task.present,
        task_exact=task.exact,
        task_enabled=task.enabled,
        task_running=task.running,
        config_exact=config_exact,
        credential_exact=credential_exact,
        runner_alive=runner_alive,
        child_alive=child_alive,
        listener_owned=listener_owned,
        authenticated_ready=authenticated_ready,
    )


def _refused_plan(action: str, code: str, status: LifecycleStatus) -> LifecyclePlan:
    return LifecyclePlan(action, code, False, False, (), status)


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TtsStartupError("config_invalid", "TTS startup config is invalid.")
    return value


def _required_sha256(payload: Mapping[str, object], name: str) -> str:
    value = _required_string(payload, name).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise TtsStartupError("config_invalid", "TTS startup config is invalid.")
    return value


def _absolute_path(payload: Mapping[str, object], name: str) -> Path:
    value = Path(_required_string(payload, name))
    if not value.is_absolute():
        raise TtsStartupError("config_invalid", "TTS startup config is invalid.")
    return value


def _paths_collide(left: Path, right: Path) -> bool:
    left_value = os.path.normcase(str(left.resolve()))
    right_value = os.path.normcase(str(right.resolve()))
    if left_value == right_value:
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _read_stable_owner_only_file(
    path: Path,
    maximum_bytes: int,
    code: str,
) -> tuple[bytes, os.stat_result]:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or _is_reparse_point(path, before):
            raise OSError("unsafe file type")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise OSError("unsafe file size")
        _verify_owner_only_permissions(path, before)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(opened):
                raise OSError("file changed before read")
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining > 0:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
        after = path.lstat()
        if _file_identity(before) != _file_identity(after) or _is_reparse_point(path, after):
            raise OSError("file changed during read")
        _verify_owner_only_permissions(path, after)
        return raw, after
    except OSError as exc:
        raise TtsStartupError(code, "TTS startup protected file is invalid.") from exc


def _snapshot_owner_only_file(path: Path, maximum_bytes: int, code: str) -> FileIdentity:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or _is_reparse_point(path, before):
            raise OSError("unsafe file type")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise OSError("unsafe file size")
        _verify_owner_only_permissions(path, before)
        after = path.lstat()
        if _file_identity(before) != _file_identity(after) or _is_reparse_point(path, after):
            raise OSError("file changed during identity check")
        _verify_owner_only_permissions(path, after)
        return _file_identity(after)
    except OSError as exc:
        raise TtsStartupError(code, "TTS startup protected file is invalid.") from exc


def _runtime_input_drift(config: TtsStartupConfig) -> str | None:
    try:
        if (
            _sha256_regular_file(config.python_executable, "executable_drift")
            != config.python_executable_sha256
        ):
            return "executable_drift"
    except TtsStartupError:
        return "executable_drift"
    try:
        if (
            _sha256_regular_file(config.bootstrap_path, "bootstrap_drift")
            != config.bootstrap_sha256
        ):
            return "bootstrap_drift"
    except TtsStartupError:
        return "bootstrap_drift"
    try:
        raw, metadata = _read_stable_owner_only_file(config.path, MAX_CONFIG_BYTES, "config_drift")
    except TtsStartupError:
        return "config_drift"
    if (
        hashlib.sha256(raw).hexdigest() != config.config_sha256
        or _file_identity(metadata) != config.config_file_identity
    ):
        return "config_drift"
    try:
        credential = _snapshot_owner_only_file(config.token_file, 512, "credential_drift")
    except TtsStartupError:
        return "credential_drift"
    if credential != config.credential_identity:
        return "credential_drift"
    return None


def _child_identity_drift(config: TtsStartupConfig, child: ProcessIdentity) -> str | None:
    if (
        Path(child.executable_path).resolve() != config.python_executable
        or child.executable_sha256 != config.python_executable_sha256
    ):
        return "executable_drift"
    return None


def _sha256_regular_file(path: Path, code: str) -> str:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or _is_reparse_point(path, before):
            raise OSError("unsafe file type")
        digest = hashlib.sha256()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(opened):
                raise OSError("file changed before hash")
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        after = path.lstat()
        if _file_identity(before) != _file_identity(after) or _is_reparse_point(path, after):
            raise OSError("file changed during hash")
        return digest.hexdigest()
    except OSError as exc:
        raise TtsStartupError(code, "TTS startup file identity is invalid.") from exc


def _read_stable_regular_file(path: Path, maximum_bytes: int, code: str) -> bytes:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or _is_reparse_point(path, before)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise OSError("unsafe file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(opened):
                raise OSError("file changed before read")
            raw = os.read(descriptor, maximum_bytes + 1)
        finally:
            os.close(descriptor)
        after = path.lstat()
        if (
            len(raw) != before.st_size
            or _file_identity(before) != _file_identity(after)
            or _is_reparse_point(path, after)
        ):
            raise OSError("file changed during read")
        return raw
    except OSError as exc:
        raise TtsStartupError(code, "TTS startup file identity is invalid.") from exc


def _verify_owner_only_directory(path: Path, code: str) -> None:
    try:
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or _is_reparse_point(path, metadata):
            raise OSError("unsafe directory type")
        _verify_owner_only_permissions(path, metadata)
    except OSError as exc:
        raise TtsStartupError(code, "TTS startup protected directory is invalid.") from exc


def _verify_owner_only_permissions(path: Path, metadata: os.stat_result) -> None:
    from agent_tools.tts_server import _verify_owner_only_permissions as verify

    verify(path, metadata)


def _is_reparse_point(path: Path, metadata: os.stat_result) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_attribute)


def _stable_device(device: int) -> int:
    # On Windows os.stat().st_dev is the volume serial number, reported by
    # Python 3.11 and older as the classic 32-bit serial
    # (BY_HANDLE_FILE_INFORMATION.dwVolumeSerialNumber) and by Python 3.13+
    # as the full 64-bit serial (FILE_ID_INFO.VolumeSerialNumber) whose low
    # 32 bits are the classic serial. Pinned file identities must not depend
    # on the interpreter build, so normalize to the classic serial on Windows;
    # POSIX device ids keep their full value.
    if sys.platform == "win32":
        return device & 0xFFFFFFFF
    return device


def _file_identity(metadata: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=_stable_device(metadata.st_dev),
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )


def _read_control_file(path: Path, code: str) -> dict[str, object]:
    try:
        raw, _ = _read_stable_owner_only_file(path, MAX_STATE_BYTES, code)
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TtsStartupError(code, "TTS startup control file is invalid.") from exc
    if not isinstance(payload, dict):
        raise TtsStartupError(code, "TTS startup control file is invalid.")
    return payload


def _write_private_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_private_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _write_private_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_new_private_json(path: Path, payload: Mapping[str, object]) -> None:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class _SubprocessChild:
    def __init__(self, process: subprocess.Popen[bytes], identity: ProcessIdentity) -> None:
        self._process = process
        self._identity = identity

    @property
    def identity(self) -> ProcessIdentity:
        return self._identity

    def poll(self) -> int | None:
        return self._process.poll()

    def terminate(self) -> None:
        self._process.terminate()


@contextmanager
def _windows_execution_lock(path: Path) -> Iterator[None]:
    if sys.platform != "win32":
        raise TtsStartupError("windows_required", "TTS startup execution lock requires Windows.")
    import ctypes
    from ctypes import wintypes

    generic_read = 0x80000000
    file_share_read = 0x00000001
    open_existing = 3
    file_attribute_normal = 0x00000080
    file_attribute_directory = 0x00000010
    file_attribute_reparse_point = 0x00000400
    file_flag_open_reparse_point = 0x00200000
    invalid_handle_value = ctypes.c_void_p(-1).value

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ByHandleFileInformation),
    ]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateFileW(
        str(path),
        generic_read,
        file_share_read,
        None,
        open_existing,
        file_attribute_normal | file_flag_open_reparse_point,
        None,
    )
    if handle == invalid_handle_value:
        raise TtsStartupError("executable_lock_failed", "TTS startup executable lock failed.")
    information = ByHandleFileInformation()
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
        kernel32.CloseHandle(handle)
        raise TtsStartupError("executable_lock_failed", "TTS startup executable lock failed.")
    unsafe_attributes = file_attribute_directory | file_attribute_reparse_point
    if information.file_attributes & unsafe_attributes:
        kernel32.CloseHandle(handle)
        raise TtsStartupError("executable_lock_failed", "TTS startup executable lock failed.")
    try:
        yield
    finally:
        kernel32.CloseHandle(handle)


class _DeadlineHealthSocket(socket.socket):
    """Health-probe TCP socket bounded by one total monotonic deadline.

    ``settimeout`` only bounds each individual blocking operation, so a peer that
    drip-feeds one byte per read can stall a probe indefinitely while every read
    stays under the nominal timeout. This socket re-derives the per-operation
    timeout from the remaining budget before every send and receive, including
    the reads ``http.client`` performs while parsing the response, so deadline
    expiry surfaces as ``TimeoutError`` (an ``OSError``) mapped to one failed
    probe instead of an unbounded stall.
    """

    def __init__(self, deadline: float) -> None:
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        self._deadline = deadline
        self.settimeout(self._remaining_timeout())

    def _remaining_timeout(self) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("TTS health probe total deadline expired.")
        return remaining

    def sendall(self, data: ReadableBuffer, flags: int = 0) -> None:
        view = memoryview(data)
        while len(view) > 0:
            self.settimeout(self._remaining_timeout())
            view = view[super().send(view, flags) :]

    def recv(self, bufsize: int, flags: int = 0) -> bytes:
        self.settimeout(self._remaining_timeout())
        return super().recv(bufsize, flags)

    def recv_into(self, buffer: WriteableBuffer, nbytes: int = 0, flags: int = 0) -> int:
        self.settimeout(self._remaining_timeout())
        return super().recv_into(buffer, nbytes, flags)


class WindowsRuntimeBackend:
    def current_owner_sid(self) -> str:
        return _current_windows_user_sid()

    def current_process(self) -> ProcessIdentity:
        process = self.inspect_process(os.getpid())
        if process is None:
            raise TtsStartupError("process_identity_failed", "TTS startup process identity failed.")
        return process

    def inspect_process(self, pid: int) -> ProcessIdentity | None:
        return _inspect_windows_process(pid)

    def listeners(self, host: str, port: int) -> tuple[ListenerIdentity, ...]:
        return _windows_tcp_listeners(host, port, self.inspect_process)

    def spawn_child(
        self,
        command: Sequence[str],
        *,
        expected_executable: Path,
        expected_executable_sha256: str,
        working_directory: Path,
        log_directory: Path,
    ) -> ManagedChild:
        _append_redacted_event(log_directory, "child_spawn")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        if not command or Path(command[0]).resolve() != expected_executable:
            raise TtsStartupError("executable_drift", "TTS startup executable identity changed.")
        with _windows_execution_lock(expected_executable):
            if (
                _sha256_regular_file(expected_executable, "executable_drift")
                != expected_executable_sha256
            ):
                raise TtsStartupError(
                    "executable_drift", "TTS startup executable identity changed."
                )
            process = subprocess.Popen(
                list(command),
                cwd=working_directory,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                creationflags=flags,
            )
        try:
            identity = self.inspect_process(process.pid)
            if identity is None:
                raise TtsStartupError("child_identity_failed", "TTS startup child identity failed.")
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise
        return _SubprocessChild(process, identity)

    def authenticated_health(
        self,
        host: str,
        port: int,
        token: str,
        *,
        expected_process: ProcessIdentity,
        timeout_seconds: float,
    ) -> bool:
        connection = _DeadlineHealthSocket(time.monotonic() + timeout_seconds)
        try:
            connection.connect((host, port))
            client_host, client_port = connection.getsockname()[:2]
            owner = _windows_tcp_connection_owner(
                server_host=host,
                server_port=port,
                client_host=str(client_host),
                client_port=int(client_port),
                process_inspector=self.inspect_process,
            )
            if owner != expected_process:
                return False
            request = (
                f"GET /healthz HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Authorization: Bearer {token}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            connection.sendall(request)
            response = http.client.HTTPResponse(connection)
            response.begin()
            body = response.read(1024)
            if response.status != 200:
                return False
            payload = json.loads(body.decode("ascii"))
            return (
                isinstance(payload, dict)
                and payload.get("status") == "ready"
                and payload.get("engines") == ["kokoro", "silero"]
                and len(payload) == 2
            )
        except (OSError, UnicodeError, json.JSONDecodeError, http.client.HTTPException):
            return False
        finally:
            connection.close()

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def _append_redacted_event(log_directory: Path, event: str) -> None:
    if event not in {"child_spawn"}:
        raise ValueError("unsupported TTS startup event")
    path = log_directory / EVENT_LOG_FILENAME
    data = (json.dumps({"event": event}, separators=(",", ":")) + "\n").encode("ascii")
    _rotate_log(path, maximum_bytes=MAX_EVENT_LOG_BYTES - len(data))
    try:
        if path.exists():
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or _is_reparse_point(path, metadata):
                raise OSError("unsafe log file")
            _verify_owner_only_permissions(path, metadata)
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "ab", buffering=0) as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise OSError("unsafe log file")
            _verify_owner_only_permissions(path, opened)
            handle.write(data)
    except OSError as exc:
        raise TtsStartupError("event_log_failed", "TTS startup event log failed.") from exc


def _rotate_log(path: Path, *, maximum_bytes: int, backups: int = 2) -> None:
    try:
        if not path.exists():
            return
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or _is_reparse_point(path, metadata):
            raise OSError("unsafe log file")
        if metadata.st_size <= maximum_bytes:
            return
        for index in range(backups, 0, -1):
            source = path if index == 1 else path.with_suffix(path.suffix + f".{index - 1}")
            target = path.with_suffix(path.suffix + f".{index}")
            if source.exists():
                os.replace(source, target)
    except OSError as exc:
        raise TtsStartupError("log_rotation_failed", "TTS startup log rotation failed.") from exc


def _current_windows_user_sid() -> str:
    if sys.platform != "win32":
        return f"uid:{os.getuid()}"
    import ctypes
    from ctypes import wintypes

    token_query = 0x0008
    token_user = 1
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), token_query, ctypes.byref(token)
    ):
        raise TtsStartupError("owner_identity_failed", "TTS startup owner identity failed.")
    try:
        length = wintypes.DWORD()
        advapi32.GetTokenInformation(token, token_user, None, 0, ctypes.byref(length))
        buffer = ctypes.create_string_buffer(length.value)
        if not advapi32.GetTokenInformation(
            token, token_user, buffer, length, ctypes.byref(length)
        ):
            raise TtsStartupError("owner_identity_failed", "TTS startup owner identity failed.")
        sid_pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        sid_text = wintypes.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(sid_pointer, ctypes.byref(sid_text)):
            raise TtsStartupError("owner_identity_failed", "TTS startup owner identity failed.")
        try:
            value = sid_text.value
            if value is None:
                raise TtsStartupError("owner_identity_failed", "TTS startup owner identity failed.")
            return value
        finally:
            kernel32.LocalFree(sid_text)
    finally:
        kernel32.CloseHandle(token)


def _inspect_windows_process(pid: int) -> ProcessIdentity | None:
    if sys.platform != "win32":
        if pid != os.getpid():
            return None
        return ProcessIdentity(
            pid,
            1,
            str(Path(sys.executable).resolve()),
            _sha256_regular_file(Path(sys.executable), "executable_invalid"),
        )
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        length = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(length.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
            return None
        creation_value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        executable_hash = _sha256_regular_file(Path(buffer.value), "executable_invalid")
        return ProcessIdentity(
            pid,
            creation_value * 100,
            str(Path(buffer.value).resolve()),
            executable_hash,
        )
    finally:
        kernel32.CloseHandle(handle)


def _windows_tcp_listeners(
    host: str,
    port: int,
    process_inspector: Callable[[int], ProcessIdentity | None],
) -> tuple[ListenerIdentity, ...]:
    rows: list[ListenerIdentity] = []
    for row in _windows_tcp_owner_rows(3):
        if row.local_host not in {host, "0.0.0.0"} or row.local_port != port:
            continue
        process = process_inspector(row.pid)
        if process is None:
            rows.append(ListenerIdentity(row.pid, -1))
        else:
            rows.append(ListenerIdentity(process.pid, process.creation_time_ns))
    return tuple(rows)


def _windows_tcp_connection_owner(
    *,
    server_host: str,
    server_port: int,
    client_host: str,
    client_port: int,
    process_inspector: Callable[[int], ProcessIdentity | None],
) -> ProcessIdentity | None:
    matches = [
        row
        for row in _windows_tcp_owner_rows(5)
        if row.state == 5
        and row.local_host == server_host
        and row.local_port == server_port
        and row.remote_host == client_host
        and row.remote_port == client_port
    ]
    if len(matches) != 1:
        return None
    return process_inspector(matches[0].pid)


@dataclass(frozen=True)
class _TcpOwnerRow:
    state: int
    local_host: str
    local_port: int
    remote_host: str
    remote_port: int
    pid: int


def _windows_tcp_owner_rows(table_class: int) -> tuple[_TcpOwnerRow, ...]:
    if sys.platform != "win32":
        return ()
    import ctypes
    from ctypes import wintypes

    af_inet = 2
    insufficient_buffer = 122

    class TcpRowOwnerPid(ctypes.Structure):
        _fields_ = [
            ("state", wintypes.DWORD),
            ("local_address", wintypes.DWORD),
            ("local_port", wintypes.DWORD),
            ("remote_address", wintypes.DWORD),
            ("remote_port", wintypes.DWORD),
            ("owning_pid", wintypes.DWORD),
        ]

    iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)
    iphlpapi.GetExtendedTcpTable.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.ULONG),
        wintypes.BOOL,
        wintypes.ULONG,
        ctypes.c_int,
        wintypes.ULONG,
    ]
    iphlpapi.GetExtendedTcpTable.restype = wintypes.DWORD
    size = wintypes.ULONG(0)
    result = iphlpapi.GetExtendedTcpTable(
        None,
        ctypes.byref(size),
        False,
        af_inet,
        table_class,
        0,
    )
    if result not in {0, insufficient_buffer}:
        raise TtsStartupError("listener_query_failed", "TTS startup listener query failed.")
    buffer = ctypes.create_string_buffer(size.value)
    result = iphlpapi.GetExtendedTcpTable(
        buffer,
        ctypes.byref(size),
        False,
        af_inet,
        table_class,
        0,
    )
    if result != 0:
        raise TtsStartupError("listener_query_failed", "TTS startup listener query failed.")
    count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD))[0]
    base = ctypes.addressof(buffer) + ctypes.sizeof(wintypes.DWORD)
    rows: list[_TcpOwnerRow] = []
    for index in range(count):
        row = TcpRowOwnerPid.from_address(base + index * ctypes.sizeof(TcpRowOwnerPid))
        rows.append(
            _TcpOwnerRow(
                state=int(row.state),
                local_host=socket.inet_ntoa(int(row.local_address).to_bytes(4, "little")),
                local_port=socket.ntohs(int(row.local_port) & 0xFFFF),
                remote_host=socket.inet_ntoa(int(row.remote_address).to_bytes(4, "little")),
                remote_port=socket.ntohs(int(row.remote_port) & 0xFFFF),
                pid=int(row.owning_pid),
            )
        )
    return tuple(rows)


class PowerShellScheduledTaskBackend:
    def __init__(self, script_path: Path | None = None) -> None:
        self.script_path = script_path or Path(__file__).with_name("tts_startup_task.ps1")

    def query(self, definition: ScheduledTaskDefinition) -> ScheduledTaskSnapshot:
        payload = self._invoke("Query", definition)
        return ScheduledTaskSnapshot(
            present=bool(payload.get("present")),
            exact=bool(payload.get("exact")),
            enabled=bool(payload.get("enabled")),
            running=bool(payload.get("running")),
            state=str(payload.get("state", "Unknown")),
            owned=bool(payload.get("owned")),
        )

    def install(self, definition: ScheduledTaskDefinition, xml: str, xml_path: Path) -> bool:
        del xml
        return bool(self._invoke("Install", definition, xml_path=xml_path).get("changed"))

    def start(self, definition: ScheduledTaskDefinition) -> bool:
        return bool(self._invoke("Start", definition).get("changed"))

    def disable(self, definition: ScheduledTaskDefinition) -> bool:
        return bool(self._invoke("Disable", definition).get("changed"))

    def stop(self, definition: ScheduledTaskDefinition) -> bool:
        return bool(self._invoke("Stop", definition).get("changed"))

    def uninstall(self, definition: ScheduledTaskDefinition) -> bool:
        return bool(self._invoke("Uninstall", definition).get("changed"))

    def _invoke(
        self,
        operation: str,
        definition: ScheduledTaskDefinition,
        *,
        xml_path: Path | None = None,
    ) -> dict[str, object]:
        if sys.platform != "win32":
            raise TtsStartupError("windows_required", "TTS startup lifecycle requires Windows.")
        command = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.script_path),
            "-Operation",
            operation,
            "-TaskName",
            definition.task_name,
            "-Fingerprint",
            definition.fingerprint,
            "-OwnerSid",
            definition.owner_sid,
            "-CommandPath",
            definition.command,
            "-Arguments",
            definition.arguments,
            "-WorkingDirectory",
            definition.working_directory,
        ]
        if xml_path is not None:
            command.extend(["-XmlPath", str(xml_path)])
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
            timeout=30,
        )
        if completed.returncode != 0:
            raise TtsStartupError(
                "task_scheduler_failed", "TTS startup Scheduled Task operation failed."
            )
        try:
            payload = json.loads(completed.stdout.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise TtsStartupError(
                "task_scheduler_failed", "TTS startup Scheduled Task operation failed."
            ) from exc
        if not isinstance(payload, dict):
            raise TtsStartupError(
                "task_scheduler_failed", "TTS startup Scheduled Task operation failed."
            )
        return payload


def _serve(
    config: TtsStartupConfig,
    expected_config: str,
    expected_config_file: str,
    expected_credential: str,
) -> int:
    if config.config_sha256 != expected_config:
        raise TtsStartupError("config_drift", "TTS startup config identity changed.")
    if config.config_file_identity.fingerprint != expected_config_file:
        raise TtsStartupError("config_drift", "TTS startup config identity changed.")
    if config.credential_identity.fingerprint != expected_credential:
        raise TtsStartupError("credential_drift", "TTS startup credential identity changed.")
    from agent_tools.tts_server import run_tts_server

    return run_tts_server(
        host=config.host,
        port=config.port,
        token_file=config.token_file,
        device=config.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-tools tts-startup")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("install", "status", "start", "stop", "uninstall"):
        action_parser = subparsers.add_parser(action)
        action_parser.add_argument("--config", required=True, type=Path)
        if action != "status":
            action_parser.add_argument("--plan", action="store_true")
    for action in ("run", "serve"):
        action_parser = subparsers.add_parser(action, help=argparse.SUPPRESS)
        action_parser.add_argument("--config", required=True, type=Path)
        action_parser.add_argument("--config-sha256", required=True)
        action_parser.add_argument("--config-file-fingerprint", required=True)
        action_parser.add_argument("--credential-fingerprint", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    scheduler: ScheduledTaskBackend | None = None,
    runtime: RuntimeBackend | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    try:
        args = build_parser().parse_args(argv)
        runtime = runtime or WindowsRuntimeBackend()
        config = load_startup_config(
            args.config,
            current_owner_sid=runtime.current_owner_sid(),
        )
        definition = build_task_definition(config)
        if args.action == "run":
            return run_owned_runner(
                config,
                definition,
                runtime,
                expected_config_sha256=args.config_sha256,
                expected_config_file_fingerprint=args.config_file_fingerprint,
                expected_credential_fingerprint=args.credential_fingerprint,
            )
        if args.action == "serve":
            return _serve(
                config,
                args.config_sha256,
                args.config_file_fingerprint,
                args.credential_fingerprint,
            )
        scheduler = scheduler or PowerShellScheduledTaskBackend()
        plan = execute_lifecycle(
            args.action,
            config,
            definition,
            scheduler,
            runtime,
            plan_only=bool(getattr(args, "plan", False)),
        )
        output.write(json.dumps(plan.to_public_dict(), sort_keys=True) + "\n")
        return 0 if plan.allowed else 2
    except TtsStartupError as exc:
        errors.write(json.dumps({"error": {"code": exc.code}}) + "\n")
        return 1
    except Exception:
        errors.write(json.dumps({"error": {"code": "internal_error"}}) + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
