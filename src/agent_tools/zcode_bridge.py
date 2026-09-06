from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import socketserver
import stat
import struct
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from agent_tools.tts_startup import (
    ProcessIdentity,
    _inspect_windows_process,
    _windows_tcp_listeners,
)
from agent_tools.voice_onboarding import (
    NetworkBackend,
    VoiceError,
    WindowsNetworkBackend,
    _bounded_subprocess,
    _ensure_private_directory,
    _file_identity,
    _interprocess_file_lock,
    _read_private_bytes,
    _regular_file_sha256,
    _write_private_bytes,
    _write_private_json,
    load_or_create_identity,
    select_network,
)
from agent_tools.zcode_relay import (
    AcknowledgedRelayProtocol,
    ChannelRpcClient,
    ChannelSubscription,
    RelayConflictError,
    RelayFrameIdentity,
    RelayIdentity,
    RelayOpClient,
    RelayTerminal,
    VSBuffer,
    WebSocketConnection,
    ZCodeRelayError,
    load_persisted_relay_identity,
)

BRIDGE_SCHEMA = "ai-nd-co.agent-tools.zcode-bridge-config/v1"
STATUS_SCHEMA = "ai-nd-co.agent-tools.zcode-bridge-status/v1"
CLIENT_SCHEMA = "ai-nd-co.agent-tools.zcode-bridge/v1"
EXPECTED_ZCODE_VERSION = "0.16.5"
EXPECTED_CONVERSATION_PROTOCOL = 3
DEFAULT_RELAY_URL = "wss://zcode.z.ai/ws"
MIN_PRIVATE_PORT = 49152
MAX_PRIVATE_PORT = 65535
RESERVED_PORTS = frozenset({4222, 4223})
CONFIG_FILENAME = "config.json"
BEARER_FILENAME = "bridge-bearer.txt"
IDENTITY_FILENAME = "bridge-identity.json"
WORKSPACE_IDENTITY_FILENAME = "workspace-identity.json"
STATE_FILENAME = "runtime-state.json"
STOP_FILENAME = "stop-request.json"
LIFECYCLE_LOCK_FILENAME = "lifecycle.lock"
EARLY_FAILURE_PREFIX = "failure-"
MAX_CONFIG_BYTES = 64 * 1024
MAX_STATE_BYTES = 64 * 1024
MAX_LOCAL_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_LOCAL_JSON_DEPTH = 64
MAX_LOCAL_CACHE = 512
LOCAL_SEND_TIMEOUT_SECONDS = 30.0
MAX_RECONNECTS = 3
_NAME_RE = re.compile(r"^[A-Za-z0-9._/:-]{1,160}$")
_ALLOWED_CALLS = frozenset(
    {
        ("zcode-task", "listTasks"),
        ("zcode-task", "createTask"),
        ("zcode-task", "sendPrompt"),
        ("zcode-task", "getTaskSnapshot"),
        ("zcode-task", "respondPermission"),
        ("zcode-task", "respondElicitation"),
        ("zcode-task", "stopGeneration"),
        ("zcode-task", "closeTask"),
        ("zcode-agent", "listSessions"),
        ("zcode-agent", "readSession"),
        ("zcode-agent", "listSessionSubagents"),
        ("zcode-agent", "sendConversationCommandV4"),
    }
)
_ALLOWED_EVENTS = frozenset(
    {
        ("zcode-task", "onDynamicTaskEvent"),
        ("zcode-agent", "onDynamicSessionEvent"),
    }
)
_ALLOWED_V4_COMMANDS = frozenset({"cancelBackgroundWork", "sendText"})
_TASK_ID_OPERATIONS = frozenset(
    {
        ("zcode-task", "sendPrompt"),
        ("zcode-task", "getTaskSnapshot"),
        ("zcode-task", "respondPermission"),
        ("zcode-task", "respondElicitation"),
        ("zcode-task", "stopGeneration"),
        ("zcode-task", "closeTask"),
        ("zcode-task", "onDynamicTaskEvent"),
    }
)
_SESSION_ID_OPERATIONS = frozenset(
    {
        ("zcode-agent", "readSession"),
        ("zcode-agent", "listSessionSubagents"),
        ("zcode-agent", "onDynamicSessionEvent"),
    }
)
_WORKSPACE_FIELDS = frozenset({"workspacePath", "workspaceIdentity"})
_OPERATION_FIELDS = {
    ("zcode-task", "listTasks"): _WORKSPACE_FIELDS,
    ("zcode-task", "createTask"): _WORKSPACE_FIELDS
    | {"mode", "deferPersistenceUntilFirstPrompt"},
    ("zcode-task", "sendPrompt"): {
        "taskId",
        "traceId",
        "content",
        "clientId",
        "clientMode",
    },
    ("zcode-task", "getTaskSnapshot"): _WORKSPACE_FIELDS
    | {"taskId", "clientMode", "messageLimit"},
    ("zcode-task", "respondPermission"): _WORKSPACE_FIELDS
    | {"taskId", "requestId", "optionId"},
    ("zcode-task", "respondElicitation"): _WORKSPACE_FIELDS
    | {"taskId", "requestId", "action", "content", "clientMode"},
    ("zcode-task", "stopGeneration"): _WORKSPACE_FIELDS | {"taskId", "runId"},
    ("zcode-task", "closeTask"): frozenset({"taskId"}),
    ("zcode-agent", "listSessions"): _WORKSPACE_FIELDS | {"includeArchived", "limit"},
    ("zcode-agent", "readSession"): _WORKSPACE_FIELDS | {"sessionId"},
    ("zcode-agent", "listSessionSubagents"): _WORKSPACE_FIELDS
    | {"sessionId", "endedCursor", "endedLimit"},
    ("zcode-agent", "sendConversationCommandV4"): _WORKSPACE_FIELDS
    | {"clientMode", "envelope"},
    ("zcode-task", "onDynamicTaskEvent"): _WORKSPACE_FIELDS
    | {"taskId", "deliveryKind"},
    ("zcode-agent", "onDynamicSessionEvent"): _WORKSPACE_FIELDS
    | {"sessionId", "deliveryKind", "includeSnapshot"},
}
_WORKSPACE_SCOPED_OPERATIONS = frozenset(
    {
        ("zcode-task", "listTasks"),
        ("zcode-task", "createTask"),
        ("zcode-task", "getTaskSnapshot"),
        ("zcode-task", "respondPermission"),
        ("zcode-task", "respondElicitation"),
        ("zcode-task", "stopGeneration"),
        ("zcode-agent", "listSessions"),
        ("zcode-agent", "readSession"),
        ("zcode-agent", "listSessionSubagents"),
        ("zcode-agent", "sendConversationCommandV4"),
        ("zcode-agent", "queryConversationCommandsV4"),
        ("zcode-task", "onDynamicTaskEvent"),
        ("zcode-agent", "onDynamicSessionEvent"),
    }
)


class BridgeError(RuntimeError):
    def __init__(self, code: str, message: str, next_action: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.next_action = next_action

    def to_public_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "cause": self.message,
            "impact": "ZCode bridge state was not changed.",
            "nextAction": self.next_action,
        }


@dataclass(frozen=True)
class BridgeConfig:
    root: Path
    config_sha256: str
    config_fingerprint: str
    bind_address: str
    port: int
    network_class: str
    workspace: Path
    workspace_fingerprint: str
    zcode_cli: Path
    zcode_cli_sha256: str
    zcode_version: str
    python_executable: Path
    python_executable_sha256: str
    module_file: Path
    module_sha256: str
    package_init_file: Path
    package_init_sha256: str
    relay_module_file: Path
    relay_module_sha256: str
    voice_module_file: Path
    voice_module_sha256: str
    startup_module_file: Path
    startup_module_sha256: str
    bearer_file: Path
    bearer_fingerprint: str
    server_id: str
    identity_fingerprint: str
    relay_url: str


@dataclass(frozen=True)
class RuntimeState:
    schema_version: int
    config_sha256: str
    config_fingerprint: str
    bearer_fingerprint: str
    module_sha256: str
    process_pid: int
    process_creation_time_ns: int
    launch_id: str
    phase: str
    relay_generation: int
    reconnect_count: int
    client_connected: bool
    last_code: str


@dataclass(frozen=True)
class BridgeStatus:
    code: str
    configured: bool
    server_id: str | None = None
    bind_address: str | None = None
    port: int | None = None
    zcode_version: str | None = None
    process_alive: bool = False
    listener_owned: bool = False
    relay_generation: int = 0
    reconnect_count: int = 0
    client_connected: bool = False

    def to_public_dict(self) -> dict[str, object]:
        return {"schema": STATUS_SCHEMA, **asdict(self)}


@dataclass(frozen=True)
class _LaunchGate:
    schema_version: int
    launch_id: str
    config_sha256: str
    process_pid: int
    process_creation_time_ns: int


def _default_root() -> Path:
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise BridgeError(
                "protected_storage_unavailable",
                "The owner-local application directory is unavailable.",
                "Sign in to the Windows owner profile and retry.",
            )
        return Path(local) / "ai-nd-co" / "agent-tools" / "zcode-bridge"
    return (
        Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        / "agent-tools"
        / "zcode-bridge"
    )


def _bridge_error(exc: Exception, code: str = "protected_state_invalid") -> BridgeError:
    if isinstance(exc, BridgeError):
        return exc
    if isinstance(exc, (VoiceError, ZCodeRelayError)):
        code = exc.code
    return BridgeError(
        code,
        "Protected ZCode bridge state is missing, unsafe, or changed.",
        "Inspect the bridge status and restore its exact protected state.",
    )


def _sha256(path: Path, code: str) -> str:
    try:
        return _regular_file_sha256(path.resolve(), code)
    except VoiceError as exc:
        raise _bridge_error(exc, code) from exc


def _fingerprint(path: Path) -> str:
    try:
        metadata = path.lstat()
        if path.is_symlink():
            raise OSError("link")
    except OSError as exc:
        raise _bridge_error(exc) from exc
    return hashlib.sha256(
        ":".join(str(value) for value in _file_identity(metadata)).encode("ascii")
    ).hexdigest()


def _read_private(path: Path, maximum: int) -> bytes:
    try:
        return _read_private_bytes(path, maximum)
    except VoiceError as exc:
        raise _bridge_error(exc) from exc


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )


def _private_address(value: str) -> bool:
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError:
        return False
    return (
        not address.is_unspecified
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and (
            address in ipaddress.IPv4Network("100.64.0.0/10")
            or address in ipaddress.IPv4Network("10.0.0.0/8")
            or address in ipaddress.IPv4Network("172.16.0.0/12")
            or address in ipaddress.IPv4Network("192.168.0.0/16")
        )
    )


def validate_bridge_bind(address: str, port: int) -> None:
    if not _private_address(address):
        raise BridgeError(
            "bind_not_private",
            "The bridge bind is not one exact private IPv4 address.",
            "Supply the current Tailscale address or an explicitly trusted physical LAN address.",
        )
    if port in RESERVED_PORTS:
        raise BridgeError(
            "port_reserved",
            "The bridge port collides with a reserved live service.",
            "Choose a different explicit private dynamic port.",
        )
    if type(port) is not int or not MIN_PRIVATE_PORT <= port <= MAX_PRIVATE_PORT:
        raise BridgeError(
            "port_not_private_dynamic",
            "The bridge port is outside the private dynamic range.",
            f"Choose one explicit port from {MIN_PRIVATE_PORT} through {MAX_PRIVATE_PORT}.",
        )


def _workspace_fingerprint(path: Path) -> str:
    try:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or bool(
                getattr(metadata, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            )
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise OSError("unsafe workspace")
    except OSError as exc:
        raise BridgeError(
            "workspace_invalid",
            "The selected ZCode workspace is not one stable directory.",
            "Supply the exact existing workspace directory.",
        ) from exc
    return hashlib.sha256(
        f"{metadata.st_dev}:{metadata.st_ino}:{metadata.st_ctime_ns}".encode("ascii")
    ).hexdigest()


def _validate_workspace(path: Path) -> Path:
    expanded = path.expanduser()
    _workspace_fingerprint(expanded)
    return expanded.resolve()


def _default_zcode_cli() -> Path:
    program_files = os.environ.get("ProgramFiles") or "C:/Program Files"
    return Path(program_files) / "ZCode" / "resources" / "glm" / "zcode.cjs"


def _installed_zcode_version(zcode_cli: Path) -> str:
    node = shutil.which("node")
    if not node:
        raise BridgeError(
            "node_missing",
            "The Node runtime required to verify ZCode is unavailable.",
            "Install or restore Node, then retry.",
        )
    try:
        result = _bounded_subprocess(
            (str(Path(node).resolve()), str(zcode_cli), "--version"), timeout=10.0
        )
    except VoiceError as exc:
        raise BridgeError(
            "zcode_version_unavailable",
            "The installed ZCode version could not be verified.",
            "Restore the official ZCode 0.16.5 installation, then retry.",
        ) from exc
    try:
        version = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise BridgeError(
            "zcode_version_invalid",
            "The installed ZCode version response is invalid.",
            "Restore the official ZCode 0.16.5 installation, then retry.",
        ) from exc
    if result.returncode != 0 or result.stderr or version != EXPECTED_ZCODE_VERSION:
        raise BridgeError(
            "zcode_version_mismatch",
            "The installed ZCode version does not match the pinned bridge contract.",
            f"Install ZCode {EXPECTED_ZCODE_VERSION}, then prepare the bridge again.",
        )
    return version


def _check_port_available(address: str, port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((address, port))
    except OSError as exc:
        raise BridgeError(
            "port_occupied",
            "The requested bridge address and port are already occupied.",
            "Choose another explicit private dynamic port.",
        ) from exc
    finally:
        probe.close()


def _bearer_value(path: Path) -> str:
    raw = _read_private(path, 256)
    try:
        value = raw.decode("ascii")
        decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (UnicodeError, ValueError) as exc:
        raise BridgeError(
            "bearer_invalid",
            "The protected bridge bearer is invalid.",
            "Restore its protected backup or prepare a new bridge root.",
        ) from exc
    if (
        len(decoded) != 32
        or base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != value
    ):
        raise BridgeError(
            "bearer_invalid",
            "The protected bridge bearer is invalid.",
            "Restore its protected backup or prepare a new bridge root.",
        )
    return value


def _ensure_bearer(path: Path) -> str:
    if path.exists():
        return _bearer_value(path)
    value = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    try:
        _write_private_bytes(path, value.encode("ascii"), replace=False)
    except (VoiceError, FileExistsError) as exc:
        if isinstance(exc, FileExistsError):
            return _bearer_value(path)
        raise _bridge_error(exc) from exc
    return value


def prepare_bridge(
    root: Path,
    *,
    bind_address: str,
    port: int,
    workspace: Path,
    network_mode: str,
    trust_lan: bool,
    network_backend: NetworkBackend,
    zcode_cli: Path | None = None,
) -> dict[str, object]:
    validate_bridge_bind(bind_address, port)
    if network_mode != "tailscale" or trust_lan:
        raise BridgeError(
            "network_mode_unsafe",
            "The ZCode bearer bridge is restricted to the encrypted Tailscale network.",
            "Choose --network tailscale and the exact current Tailscale IPv4 address.",
        )
    try:
        network = select_network(network_backend, mode=network_mode, bind_ip=bind_address)
    except VoiceError as exc:
        raise BridgeError(
            exc.code,
            "The requested bridge address is not current on one approved private adapter.",
            (
                "Choose the exact current Tailscale address or one explicitly trusted "
                "physical LAN address."
            ),
        ) from exc
    if network.confirmation_required:
        raise BridgeError(
            "network_mode_unsafe",
            "The ZCode bearer bridge cannot use a plaintext LAN endpoint.",
            "Choose the exact current Tailscale IPv4 address.",
        )
    resolved_root = root.expanduser().resolve()
    resolved_workspace = _validate_workspace(workspace)
    resolved_zcode = (zcode_cli or _default_zcode_cli()).expanduser().resolve()
    if not resolved_zcode.is_file() or resolved_zcode.is_symlink():
        raise BridgeError(
            "zcode_missing",
            "The official ZCode CLI is missing or unsafe.",
            "Restore the official ZCode installation, then retry.",
        )
    version = _installed_zcode_version(resolved_zcode)
    module_file = Path(__file__).resolve()
    package_init_file = module_file.with_name("__init__.py")
    relay_module_file = module_file.with_name("zcode_relay.py")
    voice_module_file = module_file.with_name("voice_onboarding.py")
    startup_module_file = module_file.with_name("tts_startup.py")
    python_executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    config_path = resolved_root / CONFIG_FILENAME
    if config_path.exists():
        current = load_bridge_config(resolved_root)
        expected = (
            current.bind_address == bind_address
            and current.port == port
            and current.workspace == resolved_workspace
            and current.zcode_cli == resolved_zcode
            and current.zcode_version == version
            and current.network_class == network.network_class
        )
        if not expected:
            raise BridgeError(
                "config_already_exists",
                "A different protected bridge configuration already exists.",
                "Stop it and use a separate protected root for another bridge configuration.",
            )
        return {
            "action": "prepare",
            "code": "already_prepared",
            "changed": False,
            "serverId": current.server_id,
            "bindAddress": current.bind_address,
            "port": current.port,
            "zcodeVersion": current.zcode_version,
        }
    _check_port_available(bind_address, port)
    try:
        _ensure_private_directory(resolved_root)
        bearer_path = resolved_root / BEARER_FILENAME
        _ensure_bearer(bearer_path)
        identity = load_or_create_identity(resolved_root / IDENTITY_FILENAME)
        payload: dict[str, object] = {
            "schema": BRIDGE_SCHEMA,
            "bindAddress": bind_address,
            "port": port,
            "networkClass": network.network_class,
            "workspace": str(resolved_workspace),
            "workspaceFingerprint": _workspace_fingerprint(resolved_workspace),
            "zcodeCli": str(resolved_zcode),
            "zcodeCliSha256": _sha256(resolved_zcode, "zcode_identity_invalid"),
            "zcodeVersion": version,
            "pythonExecutable": str(python_executable),
            "pythonExecutableSha256": _sha256(python_executable, "python_identity_invalid"),
            "moduleFile": str(module_file),
            "moduleSha256": _sha256(module_file, "module_identity_invalid"),
            "packageInitFile": str(package_init_file),
            "packageInitSha256": _sha256(package_init_file, "module_identity_invalid"),
            "relayModuleFile": str(relay_module_file),
            "relayModuleSha256": _sha256(relay_module_file, "module_identity_invalid"),
            "voiceModuleFile": str(voice_module_file),
            "voiceModuleSha256": _sha256(voice_module_file, "module_identity_invalid"),
            "startupModuleFile": str(startup_module_file),
            "startupModuleSha256": _sha256(startup_module_file, "module_identity_invalid"),
            "bearerFile": str(bearer_path),
            "bearerFingerprint": _fingerprint(bearer_path),
            "serverId": identity.server_id,
            "identityFingerprint": identity.fingerprint,
            "relayUrl": DEFAULT_RELAY_URL,
        }
        _write_private_json(config_path, payload, replace=False)
    except (VoiceError, FileExistsError) as exc:
        raise _bridge_error(exc) from exc
    return {
        "action": "prepare",
        "code": "prepared",
        "changed": True,
        "serverId": identity.server_id,
        "bindAddress": bind_address,
        "port": port,
        "zcodeVersion": version,
    }


_CONFIG_FIELDS = {
    "schema",
    "bindAddress",
    "port",
    "networkClass",
    "workspace",
    "workspaceFingerprint",
    "zcodeCli",
    "zcodeCliSha256",
    "zcodeVersion",
    "pythonExecutable",
    "pythonExecutableSha256",
    "moduleFile",
    "moduleSha256",
    "packageInitFile",
    "packageInitSha256",
    "relayModuleFile",
    "relayModuleSha256",
    "voiceModuleFile",
    "voiceModuleSha256",
    "startupModuleFile",
    "startupModuleSha256",
    "bearerFile",
    "bearerFingerprint",
    "serverId",
    "identityFingerprint",
    "relayUrl",
}


def load_bridge_config(root: Path) -> BridgeConfig:
    resolved_root = root.expanduser().resolve()
    path = resolved_root / CONFIG_FILENAME
    try:
        raw = _read_private(path, MAX_CONFIG_BYTES)
        payload = json.loads(raw.decode("ascii"))
        if not isinstance(payload, dict) or set(payload) != _CONFIG_FIELDS:
            raise ValueError("schema")
        if payload["schema"] != BRIDGE_SCHEMA or payload["zcodeVersion"] != EXPECTED_ZCODE_VERSION:
            raise ValueError("schema")
        bind_address = str(payload["bindAddress"])
        port = payload["port"]
        if type(port) is not int:
            raise ValueError("port")
        validate_bridge_bind(bind_address, port)
        paths = {
            name: Path(str(payload[field])).resolve()
            for name, field in (
                ("workspace", "workspace"),
                ("zcode", "zcodeCli"),
                ("python", "pythonExecutable"),
                ("module", "moduleFile"),
                ("package_init", "packageInitFile"),
                ("relay_module", "relayModuleFile"),
                ("voice_module", "voiceModuleFile"),
                ("startup_module", "startupModuleFile"),
                ("bearer", "bearerFile"),
            )
        }
        if paths["bearer"].parent != resolved_root:
            raise ValueError("bearer")
        hashes = (
            payload["zcodeCliSha256"],
            payload["pythonExecutableSha256"],
            payload["moduleSha256"],
            payload["packageInitSha256"],
            payload["workspaceFingerprint"],
            payload["relayModuleSha256"],
            payload["voiceModuleSha256"],
            payload["startupModuleSha256"],
            payload["bearerFingerprint"],
            payload["identityFingerprint"],
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("hash")
        if (
            payload["networkClass"] != "tailscale"
            or not isinstance(payload["serverId"], str)
            or not re.fullmatch(r"srv_[a-z2-7]{16}", payload["serverId"])
            or payload["relayUrl"] != DEFAULT_RELAY_URL
        ):
            raise ValueError("value")
    except (BridgeError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise _bridge_error(exc, "config_invalid") from exc
    config = BridgeConfig(
        root=resolved_root,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        config_fingerprint=_fingerprint(path),
        bind_address=bind_address,
        port=port,
        network_class=str(payload["networkClass"]),
        workspace=paths["workspace"],
        workspace_fingerprint=str(payload["workspaceFingerprint"]),
        zcode_cli=paths["zcode"],
        zcode_cli_sha256=str(payload["zcodeCliSha256"]),
        zcode_version=str(payload["zcodeVersion"]),
        python_executable=paths["python"],
        python_executable_sha256=str(payload["pythonExecutableSha256"]),
        module_file=paths["module"],
        module_sha256=str(payload["moduleSha256"]),
        package_init_file=paths["package_init"],
        package_init_sha256=str(payload["packageInitSha256"]),
        relay_module_file=paths["relay_module"],
        relay_module_sha256=str(payload["relayModuleSha256"]),
        voice_module_file=paths["voice_module"],
        voice_module_sha256=str(payload["voiceModuleSha256"]),
        startup_module_file=paths["startup_module"],
        startup_module_sha256=str(payload["startupModuleSha256"]),
        bearer_file=paths["bearer"],
        bearer_fingerprint=str(payload["bearerFingerprint"]),
        server_id=str(payload["serverId"]),
        identity_fingerprint=str(payload["identityFingerprint"]),
        relay_url=str(payload["relayUrl"]),
    )
    _validate_config_inputs(config)
    return config


def _validate_config_inputs(config: BridgeConfig) -> None:
    if _workspace_fingerprint(config.workspace) != config.workspace_fingerprint:
        raise BridgeError(
            "workspace_identity_drift",
            "The prepared workspace directory identity changed.",
            "Restore the exact workspace or prepare a new bridge root.",
        )
    if _sha256(config.zcode_cli, "zcode_identity_drift") != config.zcode_cli_sha256:
        raise BridgeError(
            "zcode_identity_drift",
            "The pinned ZCode CLI identity changed.",
            "Restore ZCode 0.16.5 and prepare a new bridge root.",
        )
    if (
        _sha256(config.python_executable, "python_identity_drift")
        != config.python_executable_sha256
    ):
        raise BridgeError(
            "python_identity_drift",
            "The bridge Python executable identity changed.",
            "Restore the exact AgentTools environment and prepare a new bridge root.",
        )
    if _sha256(config.module_file, "module_identity_drift") != config.module_sha256:
        raise BridgeError(
            "module_identity_drift",
            "The packaged bridge module identity changed.",
            "Prepare a new bridge root from the current AgentTools package.",
        )
    for path, expected in (
        (config.package_init_file, config.package_init_sha256),
        (config.relay_module_file, config.relay_module_sha256),
        (config.voice_module_file, config.voice_module_sha256),
        (config.startup_module_file, config.startup_module_sha256),
    ):
        if path.parent != config.module_file.parent or _sha256(
            path, "module_identity_drift"
        ) != expected:
            raise BridgeError(
                "module_identity_drift",
                "A packaged bridge dependency identity changed.",
                "Prepare a new bridge root from the current AgentTools package.",
            )
    if _fingerprint(config.bearer_file) != config.bearer_fingerprint:
        raise BridgeError(
            "bearer_identity_drift",
            "The protected bridge bearer identity changed.",
            "Restore the protected bearer or prepare a new bridge root.",
        )
    _bearer_value(config.bearer_file)


def _read_state(config: BridgeConfig) -> RuntimeState | None:
    path = config.root / STATE_FILENAME
    if not path.exists():
        return None
    try:
        raw = _read_private(path, MAX_STATE_BYTES)
        payload = json.loads(raw.decode("ascii"))
        expected = {
            "schema_version",
            "config_sha256",
            "config_fingerprint",
            "bearer_fingerprint",
            "module_sha256",
            "process_pid",
            "process_creation_time_ns",
            "launch_id",
            "phase",
            "relay_generation",
            "reconnect_count",
            "client_connected",
            "last_code",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("schema")
        state = RuntimeState(**payload)
        if (
            state.schema_version != 1
            or state.config_sha256 != config.config_sha256
            or state.config_fingerprint != config.config_fingerprint
            or state.bearer_fingerprint != config.bearer_fingerprint
            or state.module_sha256 != config.module_sha256
            or not 0 < state.process_pid <= 2**32 - 1
            or state.process_creation_time_ns <= 0
            or not re.fullmatch(r"[0-9a-f]{32}", state.launch_id)
            or state.phase
            not in {"starting", "relay_connecting", "ready", "reconnecting", "stopping", "fatal"}
            or state.relay_generation < 0
            or state.reconnect_count < 0
            or type(state.client_connected) is not bool
            or not re.fullmatch(r"[a-z0-9_]{1,80}", state.last_code)
        ):
            raise ValueError("value")
        return state
    except (BridgeError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise _bridge_error(exc, "state_invalid") from exc


def _process_for_state(config: BridgeConfig, state: RuntimeState) -> ProcessIdentity | None:
    process = _inspect_windows_process(state.process_pid)
    if process is None:
        return None
    if (
        process.creation_time_ns != state.process_creation_time_ns
        or Path(process.executable_path).resolve() != config.python_executable
        or process.executable_sha256 != config.python_executable_sha256
    ):
        raise BridgeError(
            "process_identity_mismatch",
            "The recorded bridge process identity no longer matches.",
            "Do not stop it by PID; inspect the protected lifecycle state.",
        )
    return process


def _listeners(config: BridgeConfig) -> tuple[object, ...]:
    return _windows_tcp_listeners(config.bind_address, config.port, _inspect_windows_process)


def inspect_bridge(config: BridgeConfig) -> BridgeStatus:
    state = _read_state(config)
    listeners = _listeners(config)
    if state is None:
        return BridgeStatus(
            code="foreign_listener" if listeners else "stopped",
            configured=True,
            server_id=config.server_id,
            bind_address=config.bind_address,
            port=config.port,
            zcode_version=config.zcode_version,
        )
    process = _process_for_state(config, state)
    if process is None:
        return BridgeStatus(
            code="foreign_listener" if listeners else "stale_state",
            configured=True,
            server_id=config.server_id,
            bind_address=config.bind_address,
            port=config.port,
            zcode_version=config.zcode_version,
            relay_generation=state.relay_generation,
            reconnect_count=state.reconnect_count,
            client_connected=False,
        )
    owned = (
        len(listeners) == 1
        and getattr(listeners[0], "pid", None) == process.pid
        and getattr(listeners[0], "creation_time_ns", None) == process.creation_time_ns
    )
    if listeners and not owned:
        code = "foreign_listener"
    elif state.phase == "fatal":
        code = state.last_code
    elif state.phase == "ready" and owned:
        code = "ready"
    elif state.phase in {"starting", "relay_connecting", "reconnecting", "stopping"}:
        code = state.phase
    else:
        code = "state_mismatch"
    return BridgeStatus(
        code=code,
        configured=True,
        server_id=config.server_id,
        bind_address=config.bind_address,
        port=config.port,
        zcode_version=config.zcode_version,
        process_alive=True,
        listener_owned=owned,
        relay_generation=state.relay_generation,
        reconnect_count=state.reconnect_count,
        client_connected=state.client_connected and code == "ready",
    )


def bridge_status(root: Path) -> BridgeStatus:
    if not (root.expanduser().resolve() / CONFIG_FILENAME).exists():
        return BridgeStatus(code="not_configured", configured=False)
    return inspect_bridge(load_bridge_config(root))


def _revalidate_current_network(
    config: BridgeConfig, network_backend: NetworkBackend | None = None
) -> None:
    if config.network_class != "tailscale":
        return
    try:
        selection = select_network(
            network_backend or WindowsNetworkBackend(),
            mode="tailscale",
            bind_ip=config.bind_address,
        )
    except VoiceError as exc:
        raise BridgeError(
            exc.code,
            "The prepared Tailscale address is no longer uniquely current.",
            "Restore the exact Tailscale adapter or prepare a new bridge root.",
        ) from exc
    if selection.network_class != "tailscale":
        raise BridgeError(
            "network_identity_drift",
            "The prepared address no longer belongs to the Tailscale adapter.",
            "Restore the exact Tailscale adapter or prepare a new bridge root.",
        )


def _minimal_environment() -> dict[str, str]:
    allowed = (
        "SystemRoot",
        "WINDIR",
        "TEMP",
        "TMP",
        "PATH",
        "PATHEXT",
        "COMSPEC",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "APPDATA",
    )
    result = {name: os.environ[name] for name in allowed if name in os.environ}
    result["PYTHONUTF8"] = "1"
    return result


_BOOTSTRAP = "\n".join(
    (
        "import hashlib,pathlib,sys",
        "root=sys.argv.pop(1)",
        "pins=[(pathlib.Path(sys.argv.pop(1)),sys.argv.pop(1)) for _ in range(5)]",
        (
            "if not all(hashlib.sha256(path.read_bytes()).hexdigest()==digest "
            "for path,digest in pins): raise SystemExit(78)"
        ),
        "sys.path.insert(0,root)",
        "from agent_tools.zcode_bridge import main",
        "raise SystemExit(main(sys.argv[1:]))",
    )
)


def _spawn_service(config: BridgeConfig, launch_id: str) -> subprocess.Popen[bytes]:
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        (
            str(config.python_executable),
            "-I",
            "-B",
            "-c",
            _BOOTSTRAP,
            str(config.module_file.parent.parent),
            str(config.module_file),
            config.module_sha256,
            str(config.package_init_file),
            config.package_init_sha256,
            str(config.relay_module_file),
            config.relay_module_sha256,
            str(config.voice_module_file),
            config.voice_module_sha256,
            str(config.startup_module_file),
            config.startup_module_sha256,
            "bridge",
            "_run",
            "--root",
            str(config.root),
            "--launch-id",
            launch_id,
            "--config-sha256",
            config.config_sha256,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        env=_minimal_environment(),
        close_fds=True,
        creationflags=flags,
        start_new_session=sys.platform != "win32",
    )


def _child_failure(config: BridgeConfig, launch_id: str) -> str | None:
    path = config.root / f"{EARLY_FAILURE_PREFIX}{launch_id}.json"
    if not path.exists():
        return None
    try:
        payload: object = json.loads(_read_private(path, MAX_STATE_BYTES).decode("ascii"))
    except (BridgeError, UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeError(
            "child_failure_invalid",
            "The disposable bridge child failure record is invalid.",
            "Leave the protected root untouched and inspect its exact file ownership.",
        ) from exc
    code = payload.get("code") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "launch_id", "config_sha256", "code"}
        or payload.get("schema_version") != 1
        or payload.get("launch_id") != launch_id
        or payload.get("config_sha256") != config.config_sha256
        or not isinstance(code, str)
        or not re.fullmatch(r"[a-z0-9_]{1,80}", code)
    ):
        raise BridgeError(
            "child_failure_invalid",
            "The disposable bridge child failure record does not own this launch.",
            "Leave the protected root untouched and inspect its exact file ownership.",
        )
    return code


def _raise_child_exited(config: BridgeConfig, launch_id: str) -> None:
    code = _child_failure(config, launch_id) or "bridge_exited"
    raise BridgeError(
        code,
        "The disposable bridge process exited before authenticated readiness.",
        "Check the protected configuration and retry from one stopped state.",
    )


def start_bridge(
    root: Path,
    *,
    timeout_seconds: float = 60.0,
    network_backend: NetworkBackend | None = None,
) -> dict[str, object]:
    config = load_bridge_config(root)
    _revalidate_current_network(config, network_backend)
    lock = config.root / LIFECYCLE_LOCK_FILENAME
    try:
        guard = _interprocess_file_lock(
            lock,
            busy_code="lifecycle_busy",
            busy_message="Another ZCode bridge lifecycle action is active.",
            busy_next_action="Wait for that bounded action to finish, then retry.",
            timeout=0.25,
        )
        with guard:
            status = inspect_bridge(config)
            if status.code in {"ready", "starting", "relay_connecting", "reconnecting"}:
                return {
                    "action": "start",
                    "code": "already_started",
                    "changed": False,
                    "status": status.to_public_dict(),
                }
            if status.code != "stopped":
                raise BridgeError(
                    status.code,
                    "The bridge lifecycle is not in one exact stopped state.",
                    "Run status; retire only a proved stale generation before retrying.",
                )
            _check_port_available(config.bind_address, config.port)
            launch_id = secrets.token_hex(16)
            gate_path = config.root / f"launch-{launch_id}.json"
            failure_path = config.root / f"{EARLY_FAILURE_PREFIX}{launch_id}.json"
            process = _spawn_service(config, launch_id)
            try:
                deadline = time.monotonic() + min(5.0, timeout_seconds)
                identity = None
                while time.monotonic() < deadline and process.poll() is None:
                    identity = _inspect_windows_process(process.pid)
                    if identity is not None:
                        break
                    time.sleep(0.05)
                if process.poll() is not None:
                    _raise_child_exited(config, launch_id)
                if (
                    identity is None
                    or Path(identity.executable_path).resolve() != config.python_executable
                    or identity.executable_sha256 != config.python_executable_sha256
                ):
                    raise BridgeError(
                        "process_identity_mismatch",
                        "The disposable bridge process identity could not be proved.",
                        "Restore the exact AgentTools environment, then retry.",
                    )
                gate = _LaunchGate(
                    1, launch_id, config.config_sha256, identity.pid, identity.creation_time_ns
                )
                _write_private_json(gate_path, asdict(gate), replace=False)
                deadline = time.monotonic() + timeout_seconds
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    current = inspect_bridge(config)
                    if current.code == "ready":
                        return {
                            "action": "start",
                            "code": "started",
                            "changed": True,
                            "status": current.to_public_dict(),
                        }
                    if current.code not in {
                        "stopped",
                        "starting",
                        "relay_connecting",
                        "reconnecting",
                    }:
                        raise BridgeError(
                            current.code,
                            "The bridge failed before authenticated readiness.",
                            "Inspect bounded status and resolve the reported condition.",
                        )
                    time.sleep(0.1)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)
                final = inspect_bridge(config)
                if process.poll() is not None and final.code == "stopped":
                    _raise_child_exited(config, launch_id)
                raise BridgeError(
                    final.code if final.code != "stopped" else "start_timeout",
                    "The bridge did not reach authenticated readiness in time.",
                    "Check relay availability and retry after retiring any stale generation.",
                )
            except Exception:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                raise
            finally:
                gate_path.unlink(missing_ok=True)
                failure_path.unlink(missing_ok=True)
    except VoiceError as exc:
        raise BridgeError(
            exc.code,
            "Another ZCode bridge lifecycle action is active.",
            "Wait for that bounded action to finish, then retry.",
        ) from exc


def _write_stop(config: BridgeConfig, state: RuntimeState) -> None:
    payload = {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "process_pid": state.process_pid,
        "process_creation_time_ns": state.process_creation_time_ns,
        "launch_id": state.launch_id,
    }
    path = config.root / STOP_FILENAME
    if path.exists():
        current = json.loads(_read_private(path, MAX_STATE_BYTES).decode("ascii"))
        if current != payload:
            raise BridgeError(
                "stop_request_invalid",
                "The existing bridge stop request does not own this generation.",
                "Inspect protected lifecycle state before retrying.",
            )
        return
    _write_private_json(path, payload, replace=False)


def stop_bridge(root: Path, *, timeout_seconds: float = 20.0) -> dict[str, object]:
    config = load_bridge_config(root)
    try:
        with _interprocess_file_lock(
            config.root / LIFECYCLE_LOCK_FILENAME,
            busy_code="lifecycle_busy",
            busy_message="Another ZCode bridge lifecycle action is active.",
            busy_next_action="Wait for that bounded action to finish, then retry.",
            timeout=0.25,
        ):
            status = inspect_bridge(config)
            if status.code == "stopped":
                return {
                    "action": "stop",
                    "code": "already_stopped",
                    "changed": False,
                    "status": status.to_public_dict(),
                }
            if status.code in {
                "stale_state",
                "foreign_listener",
                "process_identity_mismatch",
                "state_invalid",
            }:
                raise BridgeError(
                    status.code,
                    "The bridge generation cannot be stopped with exact ownership.",
                    "Retire only a proved stale generation; do not terminate by port or PID.",
                )
            state = _read_state(config)
            if state is None or _process_for_state(config, state) is None:
                raise BridgeError(
                    "stale_state",
                    "The bridge generation is stale.",
                    "Use retire after proving the process and listener are absent.",
                )
            _write_stop(config, state)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                process = _inspect_windows_process(state.process_pid)
                listeners = _listeners(config)
                if (
                    (process is None or process.creation_time_ns != state.process_creation_time_ns)
                    and not listeners
                    and not (config.root / STATE_FILENAME).exists()
                ):
                    final = inspect_bridge(config)
                    return {
                        "action": "stop",
                        "code": "stopped",
                        "changed": True,
                        "status": final.to_public_dict(),
                    }
                time.sleep(0.1)
            raise BridgeError(
                "stop_timeout",
                "The bridge did not prove process and listener release in time.",
                "Leave it untouched and inspect exact lifecycle status.",
            )
    except VoiceError as exc:
        raise _bridge_error(exc, exc.code) from exc


def retire_stale_bridge(root: Path) -> dict[str, object]:
    config = load_bridge_config(root)
    try:
        with _interprocess_file_lock(
            config.root / LIFECYCLE_LOCK_FILENAME,
            busy_code="lifecycle_busy",
            busy_message="Another ZCode bridge lifecycle action is active.",
            busy_next_action="Wait for that bounded action to finish, then retry.",
            timeout=0.25,
        ):
            status = inspect_bridge(config)
            if status.code == "stopped":
                return {
                    "action": "retire",
                    "code": "already_retired",
                    "changed": False,
                    "status": status.to_public_dict(),
                }
            if status.code != "stale_state":
                raise BridgeError(
                    status.code,
                    "The bridge state is not a proved stale owned generation.",
                    "Stop an active exact generation or leave foreign state untouched.",
                )
            if _listeners(config):
                raise BridgeError(
                    "foreign_listener",
                    "A listener still occupies the configured bridge endpoint.",
                    "Identify its owner without terminating by port.",
                )
            state_path = config.root / STATE_FILENAME
            state = _read_state(config)
            if state is None or _inspect_windows_process(state.process_pid) is not None:
                raise BridgeError(
                    "stale_state_unproved",
                    "The stale bridge generation could not be re-proved.",
                    "Re-run status and leave the state untouched.",
                )
            state_path.unlink()
            (config.root / STOP_FILENAME).unlink(missing_ok=True)
            return {
                "action": "retire",
                "code": "retired",
                "changed": True,
                "status": inspect_bridge(config).to_public_dict(),
            }
    except VoiceError as exc:
        raise _bridge_error(exc, exc.code) from exc


def _validate_workspace_pin_value(value: object, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or not 0 < len(value.encode("utf-8")) <= 4096
        or any(character in value for character in "\x00\r\n")
    ):
        raise BridgeError(
            "workspace_identity_invalid",
            "The relay workspace identity is invalid.",
            "Stop and re-verify the pinned ZCode workspace contract.",
        )
    return value


def _pin_relay_workspace(
    config: BridgeConfig, workspace_key: str, workspace_identity: str | None
) -> None:
    key = _validate_workspace_pin_value(workspace_key)
    identity = _validate_workspace_pin_value(workspace_identity, optional=True)
    assert isinstance(key, str)
    expected = {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "workspace_key": key,
        "workspace_identity": identity,
    }
    path = config.root / WORKSPACE_IDENTITY_FILENAME
    if not path.exists():
        try:
            _write_private_json(path, expected, replace=False)
            return
        except FileExistsError:
            pass
        except VoiceError as exc:
            raise _bridge_error(exc) from exc
    try:
        current = json.loads(_read_private(path, MAX_STATE_BYTES).decode("ascii"))
    except (BridgeError, UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeError(
            "workspace_identity_invalid",
            "The protected relay workspace identity is invalid.",
            "Leave the bridge stopped and inspect its protected workspace pin.",
        ) from exc
    if current != expected:
        raise BridgeError(
            "workspace_identity_drift",
            "The relay workspace identity changed for the prepared path.",
            "Restore the exact workspace or prepare a separate bridge root.",
        )


class _BridgeRelayController:
    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.persisted_identity: RelayIdentity | None = None
        self.terminal: RelayTerminal | None = None
        self.protocol: AcknowledgedRelayProtocol | None = None
        self.rpc: ChannelRpcClient | None = None
        self.workspace: dict[str, object] | None = None
        self.workspace_key: str | None = None
        self.bridge_identity: RelayFrameIdentity | None = None
        self.relay_generation = 0
        self.reconnect_count = 0
        self.last_code = "starting"
        self._fatal: BridgeError | None = None
        self._generation_handlers: list[Callable[[int], None]] = []
        self._lock = threading.RLock()
        self._client_id = f"agenttools-{secrets.token_hex(16)}"
        self._known_task_ids: set[str] = set()
        self._known_session_ids: set[str] = set()
        self._reconnecting = False
        self._rebind_committed = False
        self._candidate_terminal: RelayTerminal | None = None

    @property
    def fatal(self) -> BridgeError | None:
        return self._fatal

    def add_generation_handler(self, handler: Callable[[int], None]) -> None:
        self._generation_handlers.append(handler)

    def remove_generation_handler(self, handler: Callable[[int], None]) -> None:
        try:
            self._generation_handlers.remove(handler)
        except ValueError:
            pass

    def fail_local_delivery(self) -> None:
        with self._lock:
            if self._fatal is None:
                self._fatal = BridgeError(
                    "local_delivery_failed",
                    "An authorized local event could not be delivered safely.",
                    "Reconnect only after this bridge generation stops.",
                )

    def start(self) -> None:
        self.persisted_identity = load_persisted_relay_identity()
        try:
            self._connect(reconnect=False)
        except Exception:
            if self._candidate_terminal:
                self._candidate_terminal.close()
                self._candidate_terminal = None
            raise

    def call(
        self, generation: int, channel: str, method: str, argument: object, timeout: float
    ) -> object:
        with self._lock:
            if self._fatal:
                raise self._fatal
            if self._reconnecting:
                raise BridgeError(
                    "relay_reconnecting",
                    "The bridge is reconnecting to the ZCode relay.",
                    "Wait for the next ready generation before retrying safely.",
                )
            if generation != self.relay_generation:
                raise BridgeError(
                    "stale_generation",
                    "The client request targets a retired relay generation.",
                    "Use the latest bridge generation and retry only if the operation is safe.",
                )
            rpc = self.rpc
            scoped_argument = self._scope_argument(channel, method, argument)
        if rpc is None:
            raise BridgeError(
                "relay_reconnecting",
                "The bridge is reconnecting to the ZCode relay.",
                "Wait for the next bridge status event, then retry safely.",
            )
        try:
            result = rpc.call(channel, method, scoped_argument, timeout)
            with self._lock:
                self._record_authorized_result(channel, method, result)
            return result
        except ZCodeRelayError as exc:
            raise BridgeError(
                exc.code,
                "The ZCode relay request did not complete safely.",
                "Check bridge status before retrying the operation.",
            ) from exc

    def listen(
        self,
        generation: int,
        channel: str,
        event: str,
        argument: object,
        handler: Callable[[object], None],
    ) -> ChannelSubscription:
        with self._lock:
            if self._fatal:
                raise self._fatal
            if self._reconnecting:
                raise BridgeError(
                    "relay_reconnecting",
                    "The bridge is reconnecting to the ZCode relay.",
                    "Wait for the next ready generation before subscribing again.",
                )
            if generation != self.relay_generation or self.rpc is None:
                raise BridgeError(
                    "stale_generation",
                    "The subscription targets a retired relay generation.",
                    "Use the latest bridge generation and subscribe again.",
                )
            scoped_argument = self._scope_argument(channel, event, argument)
            return self.rpc.listen(channel, event, scoped_argument, handler)

    def _scope_argument(
        self, channel: str, operation: str, argument: object
    ) -> object:
        pair = (channel, operation)
        if pair not in _ALLOWED_CALLS and pair not in _ALLOWED_EVENTS:
            raise BridgeError(
                "operation_not_allowed",
                "The requested ZCode operation is not exposed by this bridge.",
                "Use one documented workspace-scoped bridge operation.",
            )
        if not isinstance(argument, Mapping):
            raise BridgeError(
                "operation_argument_invalid",
                "The ZCode operation argument must be one object.",
                "Send the documented bounded argument object.",
            )
        allowed_fields = _OPERATION_FIELDS[pair]
        if any(not isinstance(key, str) for key in argument) or set(argument) - allowed_fields:
            raise BridgeError(
                "operation_argument_invalid",
                "The ZCode operation argument has unsupported fields.",
                "Send only the documented fields for this bridge operation.",
            )
        argument = dict(argument)
        if pair in {
            ("zcode-task", "sendPrompt"),
            ("zcode-task", "getTaskSnapshot"),
            ("zcode-task", "respondElicitation"),
            ("zcode-agent", "sendConversationCommandV4"),
        }:
            argument["clientMode"] = "web-remote-replayable"
        if pair == ("zcode-task", "onDynamicTaskEvent"):
            argument["deliveryKind"] = "replayable"
        if pair == ("zcode-agent", "onDynamicSessionEvent"):
            argument["deliveryKind"] = "web-remote-replayable"
        is_v4_command = channel == "zcode-agent" and operation == "sendConversationCommandV4"
        if is_v4_command:
            if not isinstance(argument, Mapping) or not isinstance(
                argument.get("envelope"), Mapping
            ):
                raise BridgeError(
                    "conversation_envelope_invalid",
                    "The V4 conversation command envelope is invalid.",
                    "Send one object envelope through the prepared bridge workspace.",
                )
            envelope = dict(argument["envelope"])
            if set(envelope) - {
                "commandId",
                "clientId",
                "sessionId",
                "type",
                "payload",
                "issuedAt",
            } or not isinstance(envelope.get("payload"), Mapping):
                raise BridgeError(
                    "conversation_envelope_invalid",
                    "The V4 conversation command envelope is invalid.",
                    "Send only the documented bounded V4 command fields.",
                )
            if envelope.get("type") not in _ALLOWED_V4_COMMANDS:
                raise BridgeError(
                    "conversation_command_not_allowed",
                    "The V4 conversation command is not exposed by this bridge.",
                    "Use sendText or cancelBackgroundWork for one authorized session.",
                )
            envelope["clientId"] = self._client_id
            argument["envelope"] = envelope
        self._authorize_target_ids(channel, operation, argument)
        if pair not in _WORKSPACE_SCOPED_OPERATIONS and not (
            "workspacePath" in argument or "workspaceIdentity" in argument
        ):
            return dict(argument)
        workspace = self.workspace
        if workspace is None:
            raise BridgeError(
                "workspace_unavailable",
                "The exact relay workspace is unavailable.",
                "Wait for bridge readiness before retrying.",
            )
        expected_path = os.path.normcase(os.path.normpath(str(self.config.workspace)))
        supplied_path = argument.get("workspacePath")
        if supplied_path is not None and (
            not isinstance(supplied_path, str)
            or os.path.normcase(os.path.normpath(supplied_path)) != expected_path
        ):
            raise BridgeError(
                "workspace_scope_mismatch",
                "The client request targets a different workspace.",
                "Use the bridge's prepared workspace only.",
            )
        selected_identity = workspace.get("workspaceIdentity")
        supplied_identity = argument.get("workspaceIdentity")
        if supplied_identity is not None and supplied_identity != selected_identity:
            raise BridgeError(
                "workspace_scope_mismatch",
                "The client request targets a different workspace identity.",
                "Use the bridge's prepared workspace only.",
            )
        scoped = dict(argument)
        scoped["workspacePath"] = str(self.config.workspace)
        if isinstance(selected_identity, str) and selected_identity:
            scoped["workspaceIdentity"] = selected_identity
        else:
            scoped.pop("workspaceIdentity", None)
        return scoped

    def _authorize_target_ids(
        self, channel: str, operation: str, argument: Mapping[str, object]
    ) -> None:
        pair = (channel, operation)
        task_id = argument.get("taskId")
        session_id = argument.get("sessionId")
        if pair in _TASK_ID_OPERATIONS and (
            not isinstance(task_id, str) or task_id not in self._known_task_ids
        ):
            raise BridgeError(
                "task_scope_mismatch",
                "The task was not discovered in this bridge workspace.",
                "List or create the task through this bridge before using its ID.",
            )
        if pair in _SESSION_ID_OPERATIONS and (
            not isinstance(session_id, str) or session_id not in self._known_session_ids
        ):
            raise BridgeError(
                "session_scope_mismatch",
                "The session was not discovered in this bridge workspace.",
                "List or create the session through this bridge before using its ID.",
            )
        if channel == "zcode-agent" and operation == "sendConversationCommandV4":
            envelope = argument.get("envelope")
            assert isinstance(envelope, Mapping)
            envelope_session = envelope.get("sessionId")
            if not isinstance(envelope_session, str) or (
                envelope_session not in self._known_session_ids
            ):
                raise BridgeError(
                    "session_scope_mismatch",
                    "The V4 command session was not discovered in this bridge workspace.",
                    "List or create the session through this bridge before using its ID.",
                )
    def _record_authorized_result(
        self, channel: str, operation: str, result: object
    ) -> None:
        values: list[object] = []
        scoped_create = False
        if channel == "zcode-task" and operation == "listTasks" and isinstance(result, list):
            values = result
        elif channel == "zcode-agent" and operation == "listSessions" and isinstance(result, list):
            values = result
        elif channel == "zcode-task" and operation == "createTask":
            values = [result]
            scoped_create = True
        for value in values:
            if not isinstance(value, Mapping):
                continue
            if not scoped_create and not self._result_matches_workspace(value):
                continue
            task_id = value.get("taskId")
            session_id = value.get("sessionId")
            if isinstance(task_id, str) and _NAME_RE.fullmatch(task_id):
                self._known_task_ids.add(task_id)
                self._known_session_ids.add(task_id)
            if isinstance(session_id, str) and _NAME_RE.fullmatch(session_id):
                self._known_session_ids.add(session_id)

    def _result_matches_workspace(self, value: Mapping[str, object]) -> bool:
        nested = value.get("workspace")
        workspace = nested if isinstance(nested, Mapping) else value
        path = workspace.get("workspacePath")
        if not isinstance(path, str):
            return False
        expected = os.path.normcase(os.path.normpath(str(self.config.workspace)))
        if os.path.normcase(os.path.normpath(path)) != expected:
            return False
        configured_identity = (
            self.workspace.get("workspaceIdentity") if self.workspace is not None else None
        )
        observed_identity = workspace.get("workspaceIdentity")
        return observed_identity is None or observed_identity == configured_identity

    def monitor(self, timeout_seconds: float) -> None:
        terminal = self.terminal
        if terminal is None or not terminal.wait_failed(timeout_seconds):
            return
        failure = terminal.failure or ZCodeRelayError("relay_closed")
        if isinstance(failure, RelayConflictError):
            self._fatal = BridgeError(
                "relay_session_conflict",
                "The official relay reported another terminal session.",
                "Stop the bridge and retry only in an owner-approved quiet window.",
            )
            return
        with self._lock:
            self._reconnecting = True
        for attempt in range(1, MAX_RECONNECTS + 1):
            self.reconnect_count += 1
            self.last_code = "reconnecting"
            time.sleep(min(2.0, 0.25 * (2 ** (attempt - 1))))
            try:
                self._connect(reconnect=True)
                with self._lock:
                    self._reconnecting = False
                return
            except RelayConflictError:
                self._fatal = BridgeError(
                    "relay_session_conflict",
                    "The official relay reported another terminal session.",
                    "Stop the bridge and retry only in an owner-approved quiet window.",
                )
                break
            except (BridgeError, ZCodeRelayError, OSError):
                if self._candidate_terminal:
                    self._candidate_terminal.close()
                    self._candidate_terminal = None
                if self._rebind_committed:
                    self._fatal = BridgeError(
                        "relay_reconnect_commit_failed",
                        "The relay failed after committing a replacement bridge generation.",
                        "Stop and retire this generation before starting a new bridge.",
                    )
                    break
                continue
        if self._fatal is None:
            self._fatal = BridgeError(
                "relay_reconnect_exhausted",
                "The bridge could not recover its official relay connection.",
                "Check relay availability, then retire the stale generation before restarting.",
            )

    def close(self) -> None:
        with self._lock:
            terminal = self.terminal
            rpc = self.rpc
            self.terminal = None
            self.rpc = None
        if rpc:
            rpc.abort(ZCodeRelayError("bridge_stopping"))
        if terminal:
            terminal.close()

    def _connect(self, *, reconnect: bool) -> None:
        if self.persisted_identity is None:
            raise ZCodeRelayError("relay_pair_incomplete")
        terminal = RelayTerminal(self.persisted_identity, url=self.config.relay_url)
        self._candidate_terminal = terminal
        self._rebind_committed = False
        ops = RelayOpClient(terminal)
        early_frames: list[object] = []
        early_frames_lock = threading.Lock()

        def accept(payload: object) -> None:
            with early_frames_lock:
                protocol = self.protocol
                if protocol is not None and protocol.accept(payload):
                    return
                if isinstance(payload, dict) and payload.get("zcode_type") in {
                    "rpc-frame",
                    "rpc-frame-ack",
                }:
                    if len(early_frames) >= 16:
                        self._fatal = BridgeError(
                            "relay_early_frame_overflow",
                            "The official relay sent too many frames before bridge readiness.",
                            "Stop and re-verify the pinned carrier before restarting.",
                        )
                    else:
                        early_frames.append(payload)
                    return
            if not ops.accept(payload) and isinstance(payload, dict):
                kind = payload.get("zcode_type")
                if kind in {"bridge-degraded", "app-error"}:
                    self._fatal = BridgeError(
                        "relay_schema_or_transport_error",
                        "The official bridge reported a transport or schema failure.",
                        "Stop and re-verify the pinned ZCode bridge contract.",
                    )

        terminal.set_payload_handler(accept)
        terminal.connect()
        bootstrap = ops.request("bootstrap-request")
        result = bootstrap.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("workspaces"), list):
            terminal.close()
            raise ZCodeRelayError("bootstrap_schema_drift")
        expected = os.path.normcase(os.path.normpath(str(self.config.workspace)))
        matches = [
            value
            for value in result["workspaces"]
            if isinstance(value, dict)
            and os.path.normcase(os.path.normpath(str(value.get("workspacePath", "")))) == expected
        ]
        if len(matches) != 1:
            terminal.close()
            raise ZCodeRelayError("workspace_selection_ambiguous")
        workspace = matches[0]
        identity_value = workspace.get("workspaceIdentity")
        workspace_key = (
            identity_value.strip()
            if isinstance(identity_value, str) and identity_value.strip()
            else str(workspace["workspacePath"])
        )
        normalized_identity = (
            identity_value.strip()
            if isinstance(identity_value, str) and identity_value.strip()
            else None
        )
        _pin_relay_workspace(self.config, workspace_key, normalized_identity)
        if reconnect:
            if workspace_key != self.workspace_key:
                terminal.close()
                raise ZCodeRelayError("bridge_workspace_identity_drift")
            ops.request("workspace-reconnect-request", {"workspaceKey": workspace_key})
            if self.bridge_identity is None:
                raise ZCodeRelayError("bridge_identity_missing")
            requested_identity = RelayFrameIdentity(
                self.bridge_identity.bridge_session_id,
                (self.bridge_identity.bridge_generation or 0) + 1,
                self.bridge_identity.recovery_id,
            )
        else:
            requested_identity = RelayFrameIdentity(
                f"agenttools-{secrets.token_hex(16)}",
                1,
                f"agenttools-{secrets.token_hex(16)}",
            )
        ready = ops.request(
            "workspace-bridge-open",
            {**requested_identity.to_wire(), "workspaceKey": workspace_key},
            timeout_seconds=45.0,
        )
        bridge = ready.get("bridge")
        ready_result = ready.get("result")
        if bridge is None and isinstance(ready_result, dict):
            bridge = ready_result.get("bridge")
        if not isinstance(bridge, dict) or bridge.get("kind") != "local":
            terminal.close()
            raise ZCodeRelayError("bridge_ready_schema_drift")
        selected_path = os.path.normcase(os.path.normpath(str(bridge.get("workspacePath", ""))))
        if (
            selected_path != expected
            or bridge.get("workspaceKey") != workspace_key
            or (
                bridge.get("workspaceIdentity") is not None
                and bridge.get("workspaceIdentity") != normalized_identity
            )
        ):
            terminal.close()
            raise ZCodeRelayError("bridge_workspace_drift")
        frame_identity = RelayFrameIdentity(
            str(bridge.get("bridgeSessionId") or requested_identity.bridge_session_id),
            bridge.get("bridgeGeneration")
            if type(bridge.get("bridgeGeneration")) is int
            else requested_identity.bridge_generation,
            str(bridge.get("recoveryId") or requested_identity.recovery_id),
        )
        try:
            frame_identity.to_wire()
        except ZCodeRelayError:
            terminal.close()
            raise
        if frame_identity != requested_identity:
            terminal.close()
            raise ZCodeRelayError("bridge_identity_drift")
        with early_frames_lock:
            if reconnect and self.protocol is not None:
                self._rebind_committed = True
                self.protocol.rebind(frame_identity, terminal.send_payload)
                self.protocol.replay_unacknowledged()
                rpc = self.rpc
                if rpc is None:
                    raise ZCodeRelayError("channel_missing")
            else:
                protocol = AcknowledgedRelayProtocol(frame_identity, terminal.send_payload)
                protocol.on_degraded(
                    lambda code: setattr(
                        self,
                        "_fatal",
                        BridgeError(
                            code,
                            "The official relay carrier degraded.",
                            "Stop and re-verify the pinned carrier before restarting.",
                        ),
                    )
                )
                self.protocol = protocol
                rpc = ChannelRpcClient(protocol)
                self.rpc = rpc
            pending_early_frames = tuple(early_frames)
            early_frames.clear()
            for payload in pending_early_frames:
                if not self.protocol.accept(payload):
                    terminal.close()
                    raise ZCodeRelayError("bridge_early_frame_identity_drift")
        if self._fatal:
            terminal.close()
            raise self._fatal
        rpc.wait_initialized(30.0)
        hello = rpc.call("zcode-agent", "helloConversationV4", None, 30.0)
        if (
            not isinstance(hello, dict)
            or hello.get("protocolVersion") != EXPECTED_CONVERSATION_PROTOCOL
            or hello.get("clientMode") != "web-remote-replayable"
            or hello.get("deliveryProfile") != "replayable"
        ):
            terminal.close()
            raise ZCodeRelayError("conversation_schema_drift")
        rpc.call(
            "zcode-agent",
            "initializeConversationV4",
            {
                "kind": "clientHello",
                "protocolVersion": EXPECTED_CONVERSATION_PROTOCOL,
                "clientId": self._client_id,
                "clientKind": "web",
                "appVersion": "agent-tools-zcode-bridge/1",
            },
            30.0,
        )
        old_terminal = self.terminal
        with self._lock:
            self.terminal = terminal
            self.workspace = workspace
            self.workspace_key = workspace_key
            self.bridge_identity = frame_identity
            self.relay_generation += 1
            self.last_code = "ready"
            generation = self.relay_generation
            self._candidate_terminal = None
            self._rebind_committed = False
        if old_terminal and old_terminal is not terminal:
            old_terminal.close()
        for handler in tuple(self._generation_handlers):
            handler(generation)


@dataclass
class _CachedResponse:
    digest: str
    response: bytes


class _BridgeService:
    def __init__(
        self, config: BridgeConfig, state: RuntimeState, controller: _BridgeRelayController
    ) -> None:
        self.config = config
        self.state = state
        self.controller = controller
        self.server: _ExactBridgeServer | None = None
        self.client_connected = False
        self.stopping = False
        self.request_cache: OrderedDict[str, _CachedResponse] = OrderedDict()
        self.request_cache_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._state_lock = threading.Lock()

    def claim_client(self) -> bool:
        with self._client_lock:
            if self.client_connected or self.stopping:
                return False
            self.client_connected = True
        self.publish("ready", "ready")
        return True

    def release_client(self) -> None:
        with self._client_lock:
            self.client_connected = False
            should_publish = not self.stopping
        if should_publish:
            self.publish("ready", "ready")

    def mark_stopping(self) -> None:
        with self._client_lock:
            self.stopping = True

    def publish(self, phase: str, code: str) -> None:
        with self._state_lock:
            self.state = RuntimeState(
                schema_version=1,
                config_sha256=self.config.config_sha256,
                config_fingerprint=self.config.config_fingerprint,
                bearer_fingerprint=self.config.bearer_fingerprint,
                module_sha256=self.config.module_sha256,
                process_pid=self.state.process_pid,
                process_creation_time_ns=self.state.process_creation_time_ns,
                launch_id=self.state.launch_id,
                phase=phase,
                relay_generation=self.controller.relay_generation,
                reconnect_count=self.controller.reconnect_count,
                client_connected=self.client_connected,
                last_code=code,
            )
            _write_private_json(self.config.root / STATE_FILENAME, asdict(self.state), replace=True)


class _ExactBridgeServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, service: _BridgeService) -> None:
        self.service = service
        if sys.platform == "win32":
            self.allow_reuse_address = False
        super().__init__(
            (service.config.bind_address, service.config.port),
            _BridgeHandler,
            bind_and_activate=False,
        )
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        self.server_bind()
        self.server_activate()


class _BridgeHandler(socketserver.BaseRequestHandler):
    server: _ExactBridgeServer

    def handle(self) -> None:
        request = self.request
        request.settimeout(10.0)
        try:
            head, buffered = _read_upgrade(request)
            headers = _validate_upgrade(head, self.server.service, request)
            if headers is None:
                return
            if not self.server.service.claim_client():
                _http_error(request, 409, "bridge_busy")
                return
            try:
                key = headers["sec-websocket-key"]
                accept = base64.b64encode(
                    hashlib.sha1(
                        (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
                    ).digest()
                ).decode("ascii")
                response = (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n"
                    "Sec-WebSocket-Protocol: agenttools.zcode.v1\r\n\r\n"
                ).encode("ascii")
                request.sendall(response)
                _set_socket_send_timeout(request, LOCAL_SEND_TIMEOUT_SECONDS)
                request.settimeout(None)
                connection = WebSocketConnection(
                    request,
                    expect_masked=True,
                    send_masked=False,
                    buffered=buffered,
                    maximum_frame_bytes=MAX_LOCAL_MESSAGE_BYTES,
                    maximum_message_bytes=MAX_LOCAL_MESSAGE_BYTES,
                )
                _serve_client(connection, self.server.service)
            finally:
                self.server.service.release_client()
        except (BridgeError, ZCodeRelayError, OSError):
            try:
                request.close()
            except OSError:
                pass


def _set_socket_send_timeout(transport: socket.socket, seconds: float) -> None:
    milliseconds = max(1, min(int(seconds * 1000), 2**31 - 1))
    if sys.platform == "win32":
        value = struct.pack("I", milliseconds)
    else:
        whole = int(seconds)
        value = struct.pack("ll", whole, int((seconds - whole) * 1_000_000))
    transport.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, value)


def _read_upgrade(transport: socket.socket) -> tuple[bytes, bytes]:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = transport.recv(4096)
        if not chunk:
            raise BridgeError(
                "upgrade_closed",
                "The client closed before authentication.",
                "Reconnect with the complete bridge request.",
            )
        data.extend(chunk)
        if len(data) > 16 * 1024:
            raise BridgeError(
                "upgrade_too_large",
                "The client upgrade request is too large.",
                "Use the compact bridge WebSocket handshake.",
            )
    head, buffered = bytes(data).split(b"\r\n\r\n", 1)
    return head, buffered


def _validate_upgrade(
    raw: bytes,
    service: _BridgeService,
    transport: socket.socket,
) -> dict[str, str] | None:
    try:
        lines = raw.decode("latin1").split("\r\n")
    except UnicodeError:
        return None
    if lines[0] != "GET /zcode/v1 HTTP/1.1":
        _http_error(transport, 400, "upgrade_invalid")
        return None
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or line[0] in " \t" or ":" not in line:
            _http_error(transport, 400, "upgrade_invalid")
            return None
        name, value = line.split(":", 1)
        key = name.casefold()
        if key in headers:
            _http_error(transport, 400, "upgrade_invalid")
            return None
        headers[key] = value.strip()
    expected_host = f"{service.config.bind_address}:{service.config.port}"
    key = headers.get("sec-websocket-key", "")
    try:
        decoded_key = base64.b64decode(key, validate=True)
    except (ValueError, binascii.Error):
        _http_error(transport, 400, "upgrade_invalid")
        return None
    expected_bearer = _bearer_value(service.config.bearer_file)
    authorization = headers.get("authorization", "")
    supplied = authorization[len("Bearer ") :] if authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(expected_bearer.encode("ascii"), supplied.encode("ascii")):
        _http_error(transport, 401, "unauthorized")
        return None
    if (
        headers.get("host") != expected_host
        or headers.get("upgrade", "").casefold() != "websocket"
        or "upgrade"
        not in {item.strip().casefold() for item in headers.get("connection", "").split(",")}
        or headers.get("sec-websocket-version") != "13"
        or len(decoded_key) != 16
        or headers.get("sec-websocket-protocol") != "agenttools.zcode.v1"
        or "sec-websocket-extensions" in headers
    ):
        _http_error(transport, 400, "upgrade_invalid")
        return None
    return headers


def _http_error(transport: socket.socket | None, status: int, code: str) -> None:
    if transport is None:
        return
    body = json.dumps({"error": {"code": code}}, separators=(",", ":")).encode("ascii")
    reason = {400: "Bad Request", 401: "Unauthorized", 409: "Conflict"}.get(status, "Error")
    try:
        transport.sendall(
            (
                f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            ).encode("ascii")
            + body
        )
    except OSError:
        pass


def _json_default(value: object) -> object:
    if isinstance(value, VSBuffer):
        return {"$type": "vsbuffer", "base64": base64.b64encode(value.data).decode("ascii")}
    if isinstance(value, bytes):
        return {"$type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    raise TypeError(type(value).__name__)


def _validate_local_json_depth(value: object, depth: int = 0) -> None:
    if depth > MAX_LOCAL_JSON_DEPTH:
        raise ValueError("json depth")
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_local_json_depth(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_local_json_depth(item, depth + 1)


def _json_bytes(value: object) -> bytes:
    try:
        _validate_local_json_depth(value)
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            default=_json_default,
        ).encode("ascii")
    except (RecursionError, TypeError, ValueError) as exc:
        raise BridgeError(
            "local_response_invalid",
            "The authorized bridge response cannot be represented safely.",
            "Use a bounded ZCode operation with JSON-compatible output.",
        ) from exc
    if len(raw) > MAX_LOCAL_MESSAGE_BYTES:
        raise BridgeError(
            "local_response_too_large",
            "The authorized bridge response exceeds the local message limit.",
            "Use a bounded ZCode query or paging method.",
        )
    return raw


def _parse_client_message(payload: bytes) -> dict[str, object]:
    def no_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate")
            value[key] = item
        return value

    try:
        result = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
        )
        _validate_local_json_depth(result)
    except (RecursionError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BridgeError(
            "client_message_invalid",
            "The authorized client message is malformed.",
            "Send one bounded v1 bridge request.",
        ) from exc
    if not isinstance(result, dict):
        raise BridgeError(
            "client_message_invalid",
            "The authorized client message is malformed.",
            "Send one bounded v1 bridge request.",
        )
    return result


def _validate_name(value: object, field: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise BridgeError(
            "client_message_invalid",
            f"The bridge {field} is invalid.",
            "Use one bounded ZCode channel or method name.",
        )
    return value


def _serve_client(connection: WebSocketConnection, service: _BridgeService) -> None:
    subscriptions: dict[str, ChannelSubscription] = {}

    def send(value: object) -> None:
        connection._send(1, _json_bytes(value))

    def generation_changed(generation: int) -> None:
        try:
            send(
                {
                    "type": "bridge.status",
                    "schema": CLIENT_SCHEMA,
                    "generation": generation,
                    "state": "ready",
                }
            )
        except (BridgeError, ZCodeRelayError, OSError):
            pass

    service.controller.add_generation_handler(generation_changed)
    try:
        send(
            {
                "type": "bridge.ready",
                "schema": CLIENT_SCHEMA,
                "serverId": service.config.server_id,
                "generation": service.controller.relay_generation,
                "zcodeVersion": service.config.zcode_version,
            }
        )
        while True:
            opcode, payload = connection.receive()
            if opcode != 1:
                raise BridgeError(
                    "client_message_invalid",
                    "The bridge accepts text JSON messages only.",
                    "Send one bounded v1 text request.",
                )
            request = _parse_client_message(payload)
            request_id = request.get("id")
            if not isinstance(request_id, str) or not re.fullmatch(
                r"[A-Za-z0-9._~-]{1,128}", request_id
            ):
                raise BridgeError(
                    "client_id_invalid",
                    "The bridge request ID is invalid.",
                    "Use one short opaque request ID.",
                )
            digest = hashlib.sha256(_canonical_json(request)).hexdigest()
            with service.request_cache_lock:
                cached = service.request_cache.get(request_id)
                exhausted = cached is None and len(service.request_cache) >= MAX_LOCAL_CACHE
            if cached:
                if cached.digest != digest:
                    raise BridgeError(
                        "duplicate_request_conflict",
                        "A request ID was reused with different content.",
                        "Use a fresh request ID.",
                    )
                connection._send(1, cached.response)
                continue
            if exhausted:
                connection._send(
                    1,
                    _json_bytes(
                        {
                            "id": request_id,
                            "ok": False,
                            "error": {"code": "request_cache_exhausted"},
                        }
                    ),
                )
                continue
            try:
                response = _execute_client_request(request, service, subscriptions, send)
            except BridgeError as exc:
                response = {"id": request_id, "ok": False, "error": {"code": exc.code}}
            try:
                encoded = _json_bytes(response)
            except BridgeError as exc:
                encoded = _json_bytes(
                    {"id": request_id, "ok": False, "error": {"code": exc.code}}
                )
            with service.request_cache_lock:
                service.request_cache[request_id] = _CachedResponse(digest, encoded)
            connection._send(1, encoded)
    finally:
        service.controller.remove_generation_handler(generation_changed)
        for subscription in subscriptions.values():
            try:
                subscription.dispose()
            except ZCodeRelayError:
                pass
        connection.close()


def _execute_client_request(
    request: Mapping[str, object],
    service: _BridgeService,
    subscriptions: dict[str, ChannelSubscription],
    send: Callable[[object], None],
) -> dict[str, object]:
    operation = request.get("op")
    generation = request.get("generation")
    request_id = request["id"]
    if type(generation) is not int or generation < 1:
        raise BridgeError(
            "client_generation_invalid",
            "The bridge generation is invalid.",
            "Use the generation from bridge.ready.",
        )
    if operation == "call":
        if set(request) - {"id", "op", "generation", "channel", "method", "arg", "timeoutMs"}:
            raise BridgeError(
                "client_message_invalid",
                "The call request has unknown fields.",
                "Send only the documented v1 call fields.",
            )
        channel = _validate_name(request.get("channel"), "channel")
        method = _validate_name(request.get("method"), "method")
        timeout_ms = request.get("timeoutMs", 120000)
        if type(timeout_ms) is not int or not 100 <= timeout_ms <= 120000:
            raise BridgeError(
                "client_timeout_invalid",
                "The bridge timeout is outside its bounded range.",
                "Choose 100 through 120000 milliseconds.",
            )
        result = service.controller.call(
            generation, channel, method, request.get("arg"), timeout_ms / 1000
        )
        return {"id": request_id, "ok": True, "result": result}
    if operation == "listen":
        if set(request) != {"id", "op", "generation", "subscriptionId", "channel", "event", "arg"}:
            raise BridgeError(
                "client_message_invalid",
                "The listen request fields are invalid.",
                "Send the exact documented v1 listen fields.",
            )
        subscription_id = _validate_name(request.get("subscriptionId"), "subscription ID")
        if subscription_id in subscriptions:
            raise BridgeError(
                "subscription_exists",
                "The subscription ID is already active.",
                "Use a fresh subscription ID.",
            )
        channel = _validate_name(request.get("channel"), "channel")
        event = _validate_name(request.get("event"), "event")

        def deliver_event(value: object) -> None:
            try:
                send(
                    {
                        "type": "event",
                        "subscriptionId": subscription_id,
                        "generation": service.controller.relay_generation,
                        "data": value,
                    }
                )
            except (BridgeError, ZCodeRelayError, OSError):
                service.controller.fail_local_delivery()

        subscriptions[subscription_id] = service.controller.listen(
            generation,
            channel,
            event,
            request.get("arg"),
            deliver_event,
        )
        return {"id": request_id, "ok": True, "result": {"subscriptionId": subscription_id}}
    if operation == "dispose":
        if set(request) != {"id", "op", "generation", "subscriptionId"}:
            raise BridgeError(
                "client_message_invalid",
                "The dispose request fields are invalid.",
                "Send the exact documented v1 dispose fields.",
            )
        if generation != service.controller.relay_generation:
            raise BridgeError(
                "stale_generation",
                "The dispose targets a retired generation.",
                "Reconnect and replace stale subscriptions.",
            )
        subscription_id = _validate_name(request.get("subscriptionId"), "subscription ID")
        subscription = subscriptions.pop(subscription_id, None)
        if subscription:
            subscription.dispose()
        return {"id": request_id, "ok": True, "result": {"disposed": subscription is not None}}
    raise BridgeError(
        "client_operation_invalid",
        "The bridge operation is unsupported.",
        "Use call, listen, or dispose.",
    )


def _read_launch_gate(
    config: BridgeConfig, launch_id: str, timeout_seconds: float = 15.0
) -> _LaunchGate:
    path = config.root / f"launch-{launch_id}.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and not path.exists():
        time.sleep(0.05)
    try:
        payload = json.loads(_read_private(path, MAX_STATE_BYTES).decode("ascii"))
        gate = _LaunchGate(**payload)
    except (BridgeError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeError(
            "launch_gate_unreadable",
            "The disposable bridge launch identity was not proved.",
            "Retry through agent-tools zcode bridge start.",
        ) from exc
    if gate.schema_version != 1:
        code = "launch_gate_schema"
    elif gate.launch_id != launch_id or gate.config_sha256 != config.config_sha256:
        code = "launch_gate_config"
    elif gate.process_pid != os.getpid():
        code = "launch_gate_pid"
    else:
        process = _inspect_windows_process(os.getpid())
        if process is None:
            code = "launch_gate_process_missing"
        elif gate.process_creation_time_ns != process.creation_time_ns:
            code = "launch_gate_generation"
        elif Path(process.executable_path).resolve() != config.python_executable:
            code = "launch_gate_executable"
        elif process.executable_sha256 != config.python_executable_sha256:
            code = "launch_gate_executable_hash"
        else:
            return gate
    raise BridgeError(
        code,
        "The disposable bridge launch identity was not proved.",
        "Retry through agent-tools zcode bridge start.",
    )


def _stop_requested(config: BridgeConfig, state: RuntimeState) -> bool:
    path = config.root / STOP_FILENAME
    if not path.exists():
        return False
    try:
        payload: object = json.loads(_read_private(path, MAX_STATE_BYTES).decode("ascii"))
    except (BridgeError, UnicodeError, json.JSONDecodeError):
        return False
    return payload == {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "process_pid": state.process_pid,
        "process_creation_time_ns": state.process_creation_time_ns,
        "launch_id": state.launch_id,
    }


def _run_service(root: Path, launch_id: str, expected_config_sha256: str) -> int:
    config = load_bridge_config(root)
    _revalidate_current_network(config)
    if config.config_sha256 != expected_config_sha256:
        raise BridgeError(
            "config_identity_drift",
            "The bridge config changed before launch.",
            "Prepare a new bridge root.",
        )
    gate = _read_launch_gate(config, launch_id)
    initial = RuntimeState(
        1,
        config.config_sha256,
        config.config_fingerprint,
        config.bearer_fingerprint,
        config.module_sha256,
        gate.process_pid,
        gate.process_creation_time_ns,
        launch_id,
        "starting",
        0,
        0,
        False,
        "starting",
    )
    controller = _BridgeRelayController(config)
    service = _BridgeService(config, initial, controller)
    service.publish("relay_connecting", "relay_connecting")
    normal_stop = False
    server_thread: threading.Thread | None = None
    try:
        controller.start()
        server = _ExactBridgeServer(service)
        service.server = server
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        service.publish("ready", "ready")
        last_heartbeat = time.monotonic()
        while True:
            if _stop_requested(config, service.state):
                normal_stop = True
                service.mark_stopping()
                service.publish("stopping", "stopping")
                break
            controller.monitor(0.25)
            if controller.fatal:
                raise controller.fatal
            if controller.last_code == "reconnecting":
                service.publish("reconnecting", "reconnecting")
            elif (
                service.state.phase != "ready"
                or service.state.relay_generation != controller.relay_generation
            ):
                service.publish("ready", "ready")
            if time.monotonic() - last_heartbeat >= 10.0:
                if controller.terminal:
                    controller.terminal.heartbeat()
                last_heartbeat = time.monotonic()
    except (BridgeError, ZCodeRelayError, OSError) as exc:
        service.mark_stopping()
        code = (
            exc.code if isinstance(exc, (BridgeError, ZCodeRelayError)) else "listener_start_failed"
        )
        service.publish(
            "fatal", code if re.fullmatch(r"[a-z0-9_]{1,80}", code) else "internal_error"
        )
        return 2
    finally:
        service.mark_stopping()
        if service.server:
            service.server.shutdown()
            service.server.server_close()
        if server_thread:
            server_thread.join(timeout=3.0)
        controller.close()
        if normal_stop:
            state_path = config.root / STATE_FILENAME
            current = _read_state(config)
            if current == service.state:
                state_path.unlink(missing_ok=True)
            (config.root / STOP_FILENAME).unlink(missing_ok=True)
    return 0


def _render_status(status: BridgeStatus) -> str:
    if not status.configured:
        return "ZCode bridge: not configured"
    client = "connected" if status.client_connected else "idle"
    return (
        f"ZCode bridge: {status.code}\n"
        f"Endpoint: {status.bind_address}:{status.port}\n"
        f"Relay generation: {status.relay_generation}\n"
        f"Client: {client}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-tools zcode", description="Control the running ZCode Desktop harness."
    )
    bridge = parser.add_subparsers(dest="surface", required=True).add_parser(
        "bridge", help="Manage the disposable authenticated bridge."
    )
    actions = bridge.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser(
        "prepare", help="Create one protected exact-private bridge configuration."
    )
    prepare.add_argument("--bind", required=True)
    prepare.add_argument("--port", required=True, type=int)
    prepare.add_argument("--workspace", required=True, type=Path)
    prepare.add_argument("--network", required=True, choices=("tailscale",))
    prepare.add_argument("--zcode-cli", type=Path, help=argparse.SUPPRESS)
    prepare.add_argument("--root", type=Path)
    prepare.add_argument("--json", action="store_true")
    for name in ("start", "status", "stop", "retire"):
        action = actions.add_parser(name)
        action.add_argument("--root", type=Path)
        action.add_argument("--json", action="store_true")
    run = actions.add_parser("_run", help=argparse.SUPPRESS)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--launch-id", required=True)
    run.add_argument("--config-sha256", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    network_backend: NetworkBackend | None = None,
) -> int:
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(arguments)
        root = (args.root or _default_root()).expanduser().resolve()
        if args.action == "_run":
            try:
                return _run_service(root, args.launch_id, args.config_sha256)
            except BridgeError as exc:
                code = exc.code if re.fullmatch(r"[a-z0-9_]{1,80}", exc.code) else "internal_error"
                try:
                    _ensure_private_directory(root)
                    _write_private_json(
                        root / f"{EARLY_FAILURE_PREFIX}{args.launch_id}.json",
                        {
                            "schema_version": 1,
                            "launch_id": args.launch_id,
                            "config_sha256": args.config_sha256,
                            "code": code,
                        },
                        replace=False,
                    )
                except (VoiceError, FileExistsError):
                    pass
                return 2
        if args.action == "prepare":
            result: object = prepare_bridge(
                root,
                bind_address=args.bind,
                port=args.port,
                workspace=args.workspace,
                network_mode=args.network,
                trust_lan=False,
                network_backend=network_backend or WindowsNetworkBackend(),
                zcode_cli=args.zcode_cli,
            )
        elif args.action == "start":
            result = start_bridge(root, network_backend=network_backend)
        elif args.action == "stop":
            result = stop_bridge(root)
        elif args.action == "retire":
            result = retire_stale_bridge(root)
        else:
            result = bridge_status(root).to_public_dict()
        if getattr(args, "json", False):
            output.write(json.dumps(result, ensure_ascii=True, sort_keys=True) + "\n")
        elif isinstance(result, dict) and result.get("schema") == STATUS_SCHEMA:
            output.write(
                _render_status(
                    BridgeStatus(**{key: value for key, value in result.items() if key != "schema"})
                )
                + "\n"
            )
        else:
            assert isinstance(result, dict)
            status_value = result.get("status")
            if isinstance(status_value, dict):
                output.write(f"ZCode bridge: {result['code']}\n")
            else:
                endpoint = f"{result.get('bindAddress')}:{result.get('port')}"
                output.write(f"ZCode bridge: {result['code']}\nEndpoint: {endpoint}\n")
        return 0
    except BridgeError as exc:
        if "--json" in arguments:
            errors.write(
                json.dumps({"error": exc.to_public_dict()}, ensure_ascii=True, sort_keys=True)
                + "\n"
            )
        else:
            errors.write(f"ZCode bridge: {exc.message}\nNext: {exc.next_action}\n")
        return 2
    except Exception:
        if "--json" in arguments:
            errors.write(json.dumps({"error": {"code": "internal_error"}}) + "\n")
        else:
            errors.write("ZCode bridge: internal error\nNext: inspect bounded status and retry.\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
