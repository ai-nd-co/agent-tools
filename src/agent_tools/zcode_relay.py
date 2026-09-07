from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import importlib
import json
import os
import queue
import secrets
import socket
import ssl
import stat
import struct
import sys
import threading
import time
import urllib.parse
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAX_PHYSICAL_FRAME_BYTES = 1024 * 1024
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
MAX_FRAGMENTS = 64
MAX_WEBSOCKET_FRAGMENTS = 1024
MAX_SERIALIZER_DEPTH = 64
MAX_HTTP_HEADER_BYTES = 16 * 1024
MAX_TRANSPORT_ID_BYTES = 256
_TRANSPORT_ID_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-"
)
_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_SINGLETON_WEBSOCKET_HEADERS = frozenset(
    {
        "upgrade",
        "sec-websocket-accept",
        "sec-websocket-extensions",
        "sec-websocket-protocol",
    }
)


class ZCodeRelayError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RelayConflictError(ZCodeRelayError):
    pass


class RelayClosedError(ZCodeRelayError):
    pass


class RelaySendOutcomeUnknown(RelayClosedError):
    def __init__(self, message_seq: int) -> None:
        super().__init__("relay_send_outcome_unknown")
        self.message_seq = message_seq


class WebSocketProtocolError(ZCodeRelayError):
    pass


@dataclass(frozen=True)
class RelayIdentity:
    device_sid: str
    device_mid: str
    pass_hash: str


@dataclass(frozen=True)
class VSBuffer:
    data: bytes


@dataclass(frozen=True)
class RelayFrameIdentity:
    bridge_session_id: str
    bridge_generation: int | None = None
    recovery_id: str | None = None

    def to_wire(self) -> dict[str, object]:
        _transport_id(self.bridge_session_id, "bridgeSessionId")
        if self.recovery_id is not None:
            _transport_id(self.recovery_id, "recoveryId")
        result: dict[str, object] = {"bridgeSessionId": self.bridge_session_id}
        if self.bridge_generation is not None:
            if not 0 < self.bridge_generation <= 2**31 - 1:
                raise ZCodeRelayError("invalid_bridge_generation")
            result["bridgeGeneration"] = self.bridge_generation
        if self.recovery_id is not None:
            result["recoveryId"] = self.recovery_id
        return result


def _b64url_decode(value: str) -> bytes:
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ZCodeRelayError("credential_ciphertext_invalid") from exc
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value:
        raise ZCodeRelayError("credential_ciphertext_invalid")
    return raw


def credential_secret(
    *,
    platform: str | None = None,
    home: Path | None = None,
    username: str | None = None,
) -> str:
    resolved_home = (home or Path.home()).resolve()
    resolved_platform = platform or sys.platform
    resolved_username = username
    if not resolved_username:
        try:
            resolved_username = os.getlogin()
        except OSError:
            resolved_username = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
    return f"zcode-credential-fallback:{resolved_platform}:{resolved_home}:{resolved_username}"


def decrypt_credential(value: str, secret: str) -> str:
    if not isinstance(value, str) or not value.startswith("enc:v1:"):
        return value
    parts = value[len("enc:v1:") :].split(".")
    if len(parts) != 3:
        raise ZCodeRelayError("credential_ciphertext_invalid")
    iv, tag, ciphertext = (_b64url_decode(part) for part in parts)
    if len(iv) != 12 or len(tag) != 16:
        raise ZCodeRelayError("credential_ciphertext_invalid")
    key = hashlib.sha256(secret.encode("utf-8")).digest()
    try:
        plain = AESGCM(key).decrypt(iv, ciphertext + tag, None)
        return plain.decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ZCodeRelayError("credential_decrypt_failed") from exc


def _stable_json_file(path: Path, maximum_bytes: int) -> object:
    try:
        before = path.lstat()
        if (
            path.is_symlink()
            or bool(
                getattr(before, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            )
            or not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise OSError("unsafe file")
        _verify_pairing_permissions(path, before)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            opened_identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            )
            if identity != opened_identity or not stat.S_ISREG(opened.st_mode):
                raise OSError("file changed")
            raw = os.read(descriptor, maximum_bytes + 1)
        finally:
            os.close(descriptor)
        after = path.lstat()
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if (
            identity != after_identity
            or len(raw) != before.st_size
            or path.is_symlink()
            or bool(
                getattr(after, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            )
        ):
            raise OSError("file changed")
        _verify_pairing_permissions(path, after)
        return json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ZCodeRelayError("relay_pair_store_invalid") from exc


def _verify_pairing_permissions(path: Path, metadata: os.stat_result) -> None:
    if sys.platform != "win32":
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise OSError("unsafe permissions")
        return
    try:
        win32api = importlib.import_module("win32api")
        win32con = importlib.import_module("win32con")
        win32security = importlib.import_module("win32security")
        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
        )
        current = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        descriptor = win32security.GetNamedSecurityInfo(
            str(path),
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION,
        )
        owner = descriptor.GetSecurityDescriptorOwner()
        acl = descriptor.GetSecurityDescriptorDacl()
        current_text = win32security.ConvertSidToStringSid(current)
        allowed = {current_text, "S-1-5-18", "S-1-5-32-544"}
        if (
            win32security.ConvertSidToStringSid(owner) not in allowed
            or acl is None
            or acl.GetAceCount() < 1
        ):
            raise OSError("unsafe owner")
        granting = {
            win32security.ACCESS_ALLOWED_ACE_TYPE,
            getattr(win32security, "ACCESS_ALLOWED_COMPOUND_ACE_TYPE", 4),
            getattr(win32security, "ACCESS_ALLOWED_OBJECT_ACE_TYPE", 5),
            getattr(win32security, "ACCESS_ALLOWED_CALLBACK_ACE_TYPE", 9),
            getattr(win32security, "ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE", 11),
        }
        for index in range(acl.GetAceCount()):
            ace = acl.GetAce(index)
            if ace[0][0] not in granting:
                continue
            if ace[0][0] != win32security.ACCESS_ALLOWED_ACE_TYPE:
                raise OSError("unsupported grant")
            if win32security.ConvertSidToStringSid(ace[2]) not in allowed:
                raise OSError("broad grant")
    except Exception as exc:
        raise OSError("pairing ACL could not be verified") from exc


def load_persisted_relay_identity(home: Path | None = None) -> RelayIdentity:
    resolved_home = (home or Path.home()).resolve()
    root = resolved_home / ".zcode" / "v2"
    settings = _stable_json_file(root / "setting.json", 2 * 1024 * 1024)
    credentials = _stable_json_file(root / "credentials.json", 2 * 1024 * 1024)
    telemetry = _stable_json_file(root / "telemetry-state.json", 512 * 1024)
    if not all(isinstance(item, dict) for item in (settings, credentials, telemetry)):
        raise ZCodeRelayError("relay_pair_store_invalid")
    assert isinstance(settings, dict)
    assert isinstance(credentials, dict)
    assert isinstance(telemetry, dict)
    nested = settings.get("webRemoteControlExternalRelayDevice")
    device_sid = nested.get("deviceSid") if isinstance(nested, dict) else None
    device_sid = device_sid or settings.get("webRemoteControlExternalRelayDevice.deviceSid")
    stored = credentials.get("web-remote-control:external-relay:pass_hash")
    device_mid = telemetry.get("deviceMid") or telemetry.get("device_mid") or telemetry.get("mid")
    if not all(
        isinstance(value, str) and 0 < len(value) <= 4096
        for value in (device_sid, stored, device_mid)
    ):
        raise ZCodeRelayError("relay_pair_incomplete")
    assert isinstance(device_sid, str)
    assert isinstance(stored, str)
    assert isinstance(device_mid, str)
    pass_hash = decrypt_credential(
        stored,
        credential_secret(home=resolved_home),
    )
    if not pass_hash or len(pass_hash) > 4096:
        raise ZCodeRelayError("relay_pair_incomplete")
    return RelayIdentity(device_sid, device_mid, pass_hash)


def calculate_proof(pass_hash: str, nonce: str, role: str, device_sid: str) -> str:
    proof = hmac.new(
        pass_hash.encode("utf-8"),
        f"{nonce}|{role}|{device_sid}".encode(),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(proof).rstrip(b"=").decode("ascii")


def crc32(data: bytes) -> str:
    return f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"


def _transport_id(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not 0 < len(value.encode("utf-8")) <= MAX_TRANSPORT_ID_BYTES
        or any(character not in _TRANSPORT_ID_CHARS for character in value)
    ):
        raise ZCodeRelayError(f"invalid_{field}")


def relay_envelope_bytes(payload: Mapping[str, object]) -> int:
    envelope = {
        "type": "data",
        "payload": payload,
        "client_ts": 2**53 - 1,
        "server_ts": 2**53 - 1,
    }
    return len(json.dumps(envelope, separators=(",", ":"), ensure_ascii=True).encode("ascii"))


def encode_logical_message(
    data: bytes,
    identity: RelayFrameIdentity,
    *,
    first_seq: int,
    message_seq: int,
    maximum_frame_bytes: int = MAX_PHYSICAL_FRAME_BYTES,
) -> tuple[list[dict[str, object]], int]:
    if not 0 < len(data) <= MAX_MESSAGE_BYTES:
        raise ZCodeRelayError("logical_message_size")
    if not 1024 <= maximum_frame_bytes <= MAX_PHYSICAL_FRAME_BYTES:
        raise ZCodeRelayError("physical_frame_limit_invalid")
    common = identity.to_wire()
    checksum = {"algorithm": "crc32", "value": crc32(data)}
    fragment_size = min(len(data), int(maximum_frame_bytes * 0.70))
    frames: list[dict[str, object]] = []
    while fragment_size > 0:
        count = (len(data) + fragment_size - 1) // fragment_size
        if count > MAX_FRAGMENTS:
            raise ZCodeRelayError("too_many_fragments")
        candidate: list[dict[str, object]] = []
        for index in range(count):
            part = data[index * fragment_size : (index + 1) * fragment_size]
            frame: dict[str, object] = {
                "zcode_type": "rpc-frame",
                **common,
                "seq": first_seq + index,
                "messageSeq": message_seq,
                "fragmentIndex": index,
                "fragmentCount": count,
                "messageBytes": len(data),
                "checksum": checksum,
                "dataBase64": base64.b64encode(part).decode("ascii"),
            }
            if relay_envelope_bytes(frame) > maximum_frame_bytes:
                candidate = []
                break
            candidate.append(frame)
        if candidate:
            frames = candidate
            break
        fragment_size = int(fragment_size * 0.90)
    if not frames:
        raise ZCodeRelayError("physical_frame_fit_failed")
    return frames, first_seq + len(frames)


@dataclass
class _Assembly:
    fragment_count: int
    message_bytes: int
    checksum: str
    parts: list[bytes]
    staged_bytes: int
    started_at: float
    timer: threading.Timer | None = None


class AcknowledgedRelayProtocol:
    def __init__(
        self,
        identity: RelayFrameIdentity,
        send_payload: Callable[[dict[str, object]], None],
        *,
        maximum_frame_bytes: int = MAX_PHYSICAL_FRAME_BYTES,
        assembly_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._identity = identity
        self._send_payload = send_payload
        self._maximum_frame_bytes = maximum_frame_bytes
        self._assembly_timeout = assembly_timeout_seconds
        self._clock = clock
        self._next_seq = 1
        self._next_message_seq = 1
        self._expected_seq = 1
        self._expected_message_seq = 1
        self._highest_fully_sent = 0
        self._unacked: dict[int, list[dict[str, object]]] = {}
        self._assemblies: dict[int, _Assembly] = {}
        self._fingerprints: dict[int, str] = {}
        self._message_handlers: list[Callable[[bytes], None]] = []
        self._degraded_handlers: list[Callable[[str], None]] = []
        self._degraded_code: str | None = None
        self._lock = threading.RLock()

    @property
    def degraded_code(self) -> str | None:
        return self._degraded_code

    @property
    def identity(self) -> RelayFrameIdentity:
        return self._identity

    def on_message(self, handler: Callable[[bytes], None]) -> None:
        self._message_handlers.append(handler)

    def on_degraded(self, handler: Callable[[str], None]) -> None:
        self._degraded_handlers.append(handler)

    def send(self, data: bytes) -> int:
        with self._lock:
            if self._degraded_code:
                raise ZCodeRelayError(self._degraded_code)
            frames, self._next_seq = encode_logical_message(
                data,
                self._identity,
                first_seq=self._next_seq,
                message_seq=self._next_message_seq,
                maximum_frame_bytes=self._maximum_frame_bytes,
            )
            message_seq = self._next_message_seq
            self._next_message_seq += 1
            self._unacked[message_seq] = frames
            self._highest_fully_sent = message_seq
            try:
                for frame in frames:
                    self._send_payload(frame)
            except Exception as exc:
                # Once staged, every send failure is ambiguous: the peer may have
                # received any prefix. Preserve the exact frames for generation
                # replay and never expose the underlying error as definitive.
                raise RelaySendOutcomeUnknown(message_seq) from exc
            return message_seq

    def retire_outbound(self, code: str) -> None:
        with self._lock:
            self._fault(code)

    def rebind(
        self,
        identity: RelayFrameIdentity,
        send_payload: Callable[[dict[str, object]], None],
    ) -> None:
        with self._lock:
            if self._degraded_code:
                raise ZCodeRelayError(self._degraded_code)
            old = self._identity
            if (
                old.bridge_session_id != identity.bridge_session_id
                or old.recovery_id != identity.recovery_id
            ):
                self._fault("bridge_replacement")
                raise ZCodeRelayError("bridge_replacement")
            old_generation = old.bridge_generation or 0
            new_generation = identity.bridge_generation or 0
            if new_generation <= old_generation:
                self._fault("stale_bridge_generation")
                raise ZCodeRelayError("stale_bridge_generation")
            common = identity.to_wire()
            for frames in self._unacked.values():
                for frame in frames:
                    for key in ("bridgeSessionId", "bridgeGeneration", "recoveryId"):
                        frame.pop(key, None)
                    frame.update(common)
            self._identity = identity
            self._send_payload = send_payload

    def replay_unacknowledged(self) -> None:
        with self._lock:
            if self._degraded_code:
                raise ZCodeRelayError(self._degraded_code)
            for frames in self._unacked.values():
                for frame in frames:
                    self._send_payload(dict(frame))

    def accept(self, payload: object) -> bool:
        if not isinstance(payload, dict) or not self._same_identity(payload):
            return False
        with self._lock:
            if self._degraded_code:
                return True
            self._expire_assemblies()
            kind = payload.get("zcode_type")
            if kind == "rpc-frame-ack":
                ack = payload.get("ackMessageSeq")
                if type(ack) is not int or ack < 1:
                    self._fault("invalid_ack")
                elif ack > self._highest_fully_sent:
                    self._fault("future_ack")
                else:
                    for value in tuple(self._unacked):
                        if value <= ack:
                            self._unacked.pop(value, None)
                return True
            if kind != "rpc-frame":
                return False
            code, part = self._validate_frame(payload)
            if code:
                self._fault(code)
                return True
            assert part is not None
            seq = int(payload["seq"])
            message_seq = int(payload["messageSeq"])
            fingerprint = "|".join(
                (
                    str(message_seq),
                    str(payload["fragmentIndex"]),
                    str(payload["fragmentCount"]),
                    str(payload["messageBytes"]),
                    str(payload["checksum"]["value"]),
                    str(len(part)),
                    crc32(part),
                )
            )
            if seq < self._expected_seq:
                if self._fingerprints.get(seq) != fingerprint:
                    self._fault("conflicting_duplicate")
                elif message_seq < self._expected_message_seq:
                    self._ack(self._expected_message_seq - 1)
                return True
            if seq != self._expected_seq or message_seq != self._expected_message_seq:
                self._fault("rpc_frame_gap")
                return True
            assembly = self._assemblies.get(message_seq)
            checksum = str(payload["checksum"]["value"])
            if assembly is None:
                assembly = _Assembly(
                    int(payload["fragmentCount"]),
                    int(payload["messageBytes"]),
                    checksum,
                    [],
                    0,
                    self._clock(),
                )
                assembly.timer = threading.Timer(
                    self._assembly_timeout,
                    self._assembly_expired,
                    args=(message_seq,),
                )
                assembly.timer.daemon = True
                assembly.timer.start()
                self._assemblies[message_seq] = assembly
            if (
                int(payload["fragmentIndex"]) != len(assembly.parts)
                or int(payload["fragmentCount"]) != assembly.fragment_count
                or int(payload["messageBytes"]) != assembly.message_bytes
                or checksum != assembly.checksum
            ):
                self._fault("fragment_metadata")
                return True
            if assembly.staged_bytes + len(part) > min(assembly.message_bytes, MAX_MESSAGE_BYTES):
                self._fault("buffer_overflow")
                return True
            assembly.parts.append(part)
            assembly.staged_bytes += len(part)
            self._fingerprints[seq] = fingerprint
            while len(self._fingerprints) > 4096:
                self._fingerprints.pop(next(iter(self._fingerprints)))
            self._expected_seq += 1
            if len(assembly.parts) == assembly.fragment_count:
                message = b"".join(assembly.parts)
                self._assemblies.pop(message_seq, None)
                if assembly.timer:
                    assembly.timer.cancel()
                if len(message) != assembly.message_bytes or crc32(message) != assembly.checksum:
                    self._fault("checksum_mismatch")
                    return True
                self._expected_message_seq += 1
                for handler in tuple(self._message_handlers):
                    handler(message)
                self._ack(message_seq)
            return True

    def _same_identity(self, payload: Mapping[str, object]) -> bool:
        return (
            payload.get("bridgeSessionId") == self._identity.bridge_session_id
            and payload.get("bridgeGeneration") == self._identity.bridge_generation
            and payload.get("recoveryId") == self._identity.recovery_id
        )

    def _validate_frame(self, payload: Mapping[str, object]) -> tuple[str | None, bytes | None]:
        if relay_envelope_bytes(payload) > self._maximum_frame_bytes:
            return "physical_frame_size", None
        for name in ("seq", "messageSeq"):
            value = payload.get(name)
            if type(value) is not int or value < 1 or value > 2**53 - 1:
                return f"invalid_{name}", None
        index = payload.get("fragmentIndex")
        count = payload.get("fragmentCount")
        size = payload.get("messageBytes")
        if type(index) is not int or not 0 <= index < MAX_FRAGMENTS:
            return "invalid_fragment_index", None
        if type(count) is not int or not 1 <= count <= MAX_FRAGMENTS or index >= count:
            return "invalid_fragment_count", None
        if type(size) is not int or not 1 <= size <= MAX_MESSAGE_BYTES:
            return "invalid_message_bytes", None
        checksum = payload.get("checksum")
        if (
            not isinstance(checksum, dict)
            or checksum.get("algorithm") != "crc32"
            or not isinstance(checksum.get("value"), str)
            or len(checksum["value"]) != 8
            or any(character not in "0123456789abcdef" for character in checksum["value"])
        ):
            return "invalid_checksum", None
        encoded = payload.get("dataBase64")
        if not isinstance(encoded, str) or len(encoded) > (MAX_PHYSICAL_FRAME_BYTES * 4 // 3 + 8):
            return "invalid_base64", None
        try:
            part = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return "invalid_base64", None
        if not part or base64.b64encode(part).decode("ascii") != encoded or len(part) > size:
            return "noncanonical_base64", None
        return None, part

    def _expire_assemblies(self) -> None:
        now = self._clock()
        if any(
            now - value.started_at > self._assembly_timeout for value in self._assemblies.values()
        ):
            self._fault("assembly_timeout")

    def _assembly_expired(self, message_seq: int) -> None:
        with self._lock:
            if message_seq in self._assemblies:
                self._fault("assembly_timeout")

    def _ack(self, message_seq: int) -> None:
        self._send_payload(
            {
                "zcode_type": "rpc-frame-ack",
                **self._identity.to_wire(),
                "ackMessageSeq": message_seq,
            }
        )

    def _fault(self, code: str) -> None:
        if self._degraded_code:
            return
        self._degraded_code = code
        self._unacked.clear()
        for assembly in self._assemblies.values():
            if assembly.timer:
                assembly.timer.cancel()
        self._assemblies.clear()
        for handler in tuple(self._degraded_handlers):
            handler(code)


def encode_vql(value: int) -> bytes:
    if type(value) is not int or value < 0 or value > 2**53 - 1:
        raise ZCodeRelayError("vql_value_invalid")
    output = bytearray()
    while True:
        byte = value % 128
        value //= 128
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def decode_vql(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    multiplier = 1
    for _ in range(8):
        if offset >= len(data):
            raise ZCodeRelayError("vql_truncated")
        byte = data[offset]
        offset += 1
        value += (byte & 0x7F) * multiplier
        if value > 2**53 - 1:
            raise ZCodeRelayError("vql_overflow")
        if not byte & 0x80:
            return value, offset
        multiplier *= 128
    raise ZCodeRelayError("vql_overflow")


def _nested_json_default(value: object) -> object:
    if isinstance(value, VSBuffer):
        return {
            "__zcode_rpc_nested_vsbuffer_v1": True,
            "base64": base64.b64encode(value.data).decode("ascii"),
        }
    if isinstance(value, bytes):
        return {
            "__zcode_rpc_nested_uint8array_v1": True,
            "base64": base64.b64encode(value).decode("ascii"),
        }
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _nested_json_hook(value: dict[str, object]) -> object:
    encoded = value.get("base64")
    if not isinstance(encoded, str):
        return value
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return value
    if value.get("__zcode_rpc_nested_vsbuffer_v1") is True:
        return VSBuffer(raw)
    if value.get("__zcode_rpc_nested_uint8array_v1") is True:
        return raw
    return value


def _validate_nested_depth(value: object, depth: int = 0) -> None:
    if depth > MAX_SERIALIZER_DEPTH:
        raise ZCodeRelayError("serializer_depth_exceeded")
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_nested_depth(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_nested_depth(item, depth + 1)


def encode_value(value: object, *, _depth: int = 0) -> bytes:
    if _depth > MAX_SERIALIZER_DEPTH:
        raise ZCodeRelayError("serializer_depth_exceeded")
    if value is None:
        return b"\x00"
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"\x01" + encode_vql(len(raw)) + raw
    if isinstance(value, VSBuffer):
        return b"\x03" + encode_vql(len(value.data)) + value.data
    if isinstance(value, bytes):
        return b"\x02" + encode_vql(len(value)) + value
    if isinstance(value, (list, tuple)):
        if len(value) > 1_000_000:
            raise ZCodeRelayError("serializer_array_too_large")
        return b"\x04" + encode_vql(len(value)) + b"".join(
            encode_value(item, _depth=_depth + 1) for item in value
        )
    if type(value) is int and -(2**31) <= value <= 2**31 - 1:
        return b"\x06" + encode_vql(value & 0xFFFFFFFF)
    try:
        _validate_nested_depth(value, _depth)
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            default=_nested_json_default,
        ).encode("ascii")
    except (RecursionError, ValueError) as exc:
        raise ZCodeRelayError("serializer_depth_exceeded") from exc
    return b"\x05" + encode_vql(len(raw)) + raw


def decode_value(data: bytes, offset: int = 0, *, _depth: int = 0) -> tuple[object, int]:
    if _depth > MAX_SERIALIZER_DEPTH:
        raise ZCodeRelayError("serializer_depth_exceeded")
    if offset >= len(data):
        raise ZCodeRelayError("serializer_value_missing")
    tag = data[offset]
    offset += 1
    if tag == 0:
        return None, offset
    if tag in {1, 2, 3, 5}:
        length, offset = decode_vql(data, offset)
        end = offset + length
        if end > len(data) or length > MAX_MESSAGE_BYTES:
            raise ZCodeRelayError("serializer_value_truncated")
        raw = data[offset:end]
        if tag == 1:
            try:
                return raw.decode("utf-8", errors="strict"), end
            except UnicodeError as exc:
                raise ZCodeRelayError("serializer_utf8_invalid") from exc
        if tag == 2:
            return raw, end
        if tag == 3:
            return VSBuffer(raw), end
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_hook=_nested_json_hook,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
            )
            _validate_nested_depth(value)
            return value, end
        except (RecursionError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ZCodeRelayError("serializer_json_invalid") from exc
    if tag == 4:
        length, offset = decode_vql(data, offset)
        if length > 1_000_000:
            raise ZCodeRelayError("serializer_array_too_large")
        values: list[object] = []
        for _ in range(length):
            value, offset = decode_value(data, offset, _depth=_depth + 1)
            values.append(value)
        return values, offset
    if tag == 6:
        value, offset = decode_vql(data, offset)
        value &= 0xFFFFFFFF
        return (value - 0x100000000 if value > 0x7FFFFFFF else value), offset
    raise ZCodeRelayError("serializer_tag_unknown")


def encode_message(*values: object) -> bytes:
    return b"".join(encode_value(value) for value in values)


def decode_message(data: bytes) -> list[object]:
    values: list[object] = []
    offset = 0
    while offset < len(data):
        value, offset = decode_value(data, offset)
        values.append(value)
    return values


@dataclass
class _PendingCall:
    event: threading.Event
    result: object = None
    error: ZCodeRelayError | None = None


class ChannelSubscription:
    def __init__(self, owner: ChannelRpcClient, listener_id: int) -> None:
        self._owner = owner
        self._listener_id = listener_id
        self._disposed = False

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._owner._dispose(self._listener_id)


class ChannelRpcClient:
    def __init__(self, protocol: AcknowledgedRelayProtocol) -> None:
        self._protocol = protocol
        self._next_id = 1
        self._pending: dict[int, _PendingCall] = {}
        self._listeners: dict[int, Callable[[object], None]] = {}
        self._initialized = threading.Event()
        self._abort_error: ZCodeRelayError | None = None
        self._lock = threading.RLock()
        protocol.on_message(self._accept)
        protocol.on_degraded(lambda code: self._abort_after_retire(ZCodeRelayError(code)))

    def wait_initialized(self, timeout_seconds: float = 30.0) -> None:
        if self._abort_error:
            raise self._abort_error
        if not self._initialized.wait(timeout_seconds):
            raise ZCodeRelayError("channel_initialize_timeout")
        if self._abort_error:
            raise self._abort_error

    def call(
        self,
        channel: str,
        command: str,
        argument: object,
        timeout_seconds: float = 120.0,
    ) -> object:
        with self._lock:
            if self._abort_error:
                raise self._abort_error
            request_id = self._next_id
            self._next_id += 1
            pending = _PendingCall(threading.Event())
            self._pending[request_id] = pending
            try:
                self._protocol.send(
                    encode_message([100, request_id, channel, command], [argument])
                )
            except RelayClosedError:
                # Delivery is ambiguous. The generation-preserving replay owns the
                # eventual authoritative response for this still-pending request.
                pass
        if not pending.event.wait(timeout_seconds):
            with self._lock:
                still_pending = self._pending.pop(request_id, None) is not None
            if not still_pending:
                if pending.error:
                    raise pending.error
                return pending.result
            # A cancellation would itself be ambiguous and leave a replayable
            # mutation. Retire the whole ordered generation before publishing an
            # unknown outcome, so neither this request nor a later sequence can
            # execute after the caller receives that result.
            self._protocol.retire_outbound("channel_request_outcome_unknown")
            raise ZCodeRelayError("channel_request_outcome_unknown")
        if pending.error:
            raise pending.error
        return pending.result

    def listen(
        self,
        channel: str,
        event: str,
        argument: object,
        listener: Callable[[object], None],
    ) -> ChannelSubscription:
        with self._lock:
            if self._abort_error:
                raise self._abort_error
            listener_id = self._next_id
            self._next_id += 1
            self._listeners[listener_id] = listener
            try:
                self._protocol.send(
                    encode_message([102, listener_id, channel, event], argument)
                )
            except RelayClosedError:
                pass
            return ChannelSubscription(self, listener_id)

    def abort(self, error: ZCodeRelayError | None = None) -> None:
        resolved = error or ZCodeRelayError("channel_aborted")
        self._protocol.retire_outbound(resolved.code)
        self._abort_after_retire(resolved)

    def _abort_after_retire(self, error: ZCodeRelayError) -> None:
        with self._lock:
            if self._abort_error:
                return
            self._abort_error = error
            for pending in self._pending.values():
                pending.error = ZCodeRelayError("channel_request_outcome_unknown")
                pending.event.set()
            self._pending.clear()
            self._listeners.clear()
            self._initialized.set()

    def _dispose(self, listener_id: int) -> None:
        with self._lock:
            if self._listeners.pop(listener_id, None) is None or self._abort_error:
                return
            try:
                self._protocol.send(encode_message([103, listener_id], None))
            except RelayClosedError:
                pass

    def _accept(self, data: bytes) -> None:
        try:
            values = decode_message(data)
            if not values or not isinstance(values[0], list):
                raise ZCodeRelayError("channel_envelope_invalid")
            header = values[0]
            if not header or type(header[0]) is not int:
                raise ZCodeRelayError("channel_header_invalid")
            kind = header[0]
            if kind == 200:
                if len(header) != 1 or not (
                    len(values) == 1 or (len(values) == 2 and values[1] is None)
                ):
                    raise ZCodeRelayError("channel_initialize_invalid")
                self._initialized.set()
                return
            if kind not in {201, 202, 203, 204} or len(header) != 2 or type(header[1]) is not int:
                raise ZCodeRelayError("channel_message_kind_invalid")
            if len(values) != 2:
                raise ZCodeRelayError("channel_payload_invalid")
            message_id = header[1]
            with self._lock:
                if kind == 204:
                    listener = self._listeners.get(message_id)
                    if listener:
                        listener(values[1])
                    return
                pending = self._pending.pop(message_id, None)
                if not pending:
                    return
                if kind == 201:
                    pending.result = values[1]
                else:
                    remote = values[1]
                    pending.error = ZCodeRelayError(_channel_remote_error_code(remote))
                pending.event.set()
        except ZCodeRelayError as exc:
            self.abort(exc)


def _channel_remote_error_code(remote: object) -> str:
    if isinstance(remote, dict):
        text = " ".join(
            str(remote.get(key, "")) for key in ("code", "message", "reason", "error")
        ).casefold()
    else:
        text = str(remote).casefold()
    categories = (
        ("untrusted", "untrusted"),
        ("unsupported", "unsupported"),
        ("not supported", "unsupported"),
        ("not active", "inactive"),
        ("not found", "not_found"),
        ("unknown", "unknown"),
        ("invalid", "invalid"),
        ("stale", "stale"),
        ("permission", "permission"),
        ("denied", "denied"),
    )
    for marker, code in categories:
        if marker in text:
            return f"channel_remote_{code}"
    return "channel_remote_error"


def _valid_close_code(code: int) -> bool:
    if code in {1000, 1001, 1002, 1003, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014}:
        return True
    return 3000 <= code <= 4999


class WebSocketFrameCodec:
    def __init__(
        self,
        *,
        expect_masked: bool,
        maximum_frame_bytes: int = MAX_PHYSICAL_FRAME_BYTES,
        maximum_message_bytes: int = MAX_MESSAGE_BYTES,
    ) -> None:
        self.expect_masked = expect_masked
        self.maximum_frame_bytes = maximum_frame_bytes
        self.maximum_message_bytes = maximum_message_bytes
        self.buffer = bytearray()
        self.fragment_opcode: int | None = None
        self.fragment_parts: list[bytes] = []
        self.fragment_bytes = 0
        self.fragment_count = 0

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self.buffer.extend(data)
        messages: list[tuple[int, bytes]] = []
        while True:
            parsed = self._frame()
            if parsed is None:
                return messages
            fin, opcode, payload = parsed
            if opcode >= 8:
                if not fin or len(payload) > 125:
                    raise WebSocketProtocolError("websocket_control_frame_invalid")
                if opcode == 8:
                    if len(payload) == 1:
                        raise WebSocketProtocolError("websocket_close_payload_invalid")
                    if payload:
                        code = struct.unpack("!H", payload[:2])[0]
                        if not _valid_close_code(code):
                            raise WebSocketProtocolError("websocket_close_code_invalid")
                        try:
                            payload[2:].decode("utf-8", errors="strict")
                        except UnicodeError as exc:
                            raise WebSocketProtocolError("websocket_close_reason_invalid") from exc
                messages.append((opcode, payload))
                continue
            if opcode == 0:
                if self.fragment_opcode is None:
                    raise WebSocketProtocolError("websocket_unexpected_continuation")
                self.fragment_parts.append(payload)
                self.fragment_count += 1
                if self.fragment_count > MAX_WEBSOCKET_FRAGMENTS:
                    raise WebSocketProtocolError("websocket_fragment_count_exceeded")
                self.fragment_bytes += len(payload)
                if self.fragment_bytes > self.maximum_message_bytes:
                    raise WebSocketProtocolError("websocket_message_too_large")
                if fin:
                    complete = b"".join(self.fragment_parts)
                    original_opcode = self.fragment_opcode
                    self.fragment_opcode = None
                    self.fragment_parts = []
                    self.fragment_bytes = 0
                    self.fragment_count = 0
                    self._validate_data(original_opcode, complete)
                    messages.append((original_opcode, complete))
                continue
            if opcode not in {1, 2}:
                raise WebSocketProtocolError("websocket_opcode_invalid")
            if self.fragment_opcode is not None:
                raise WebSocketProtocolError("websocket_fragment_sequence_invalid")
            if fin:
                self._validate_data(opcode, payload)
                messages.append((opcode, payload))
            else:
                self.fragment_opcode = opcode
                self.fragment_parts = [payload]
                self.fragment_bytes = len(payload)
                self.fragment_count = 1
                if self.fragment_bytes > self.maximum_message_bytes:
                    raise WebSocketProtocolError("websocket_message_too_large")

    def _frame(self) -> tuple[bool, int, bytes] | None:
        if len(self.buffer) < 2:
            return None
        first, second = self.buffer[0], self.buffer[1]
        if first & 0x70:
            raise WebSocketProtocolError("websocket_rsv_invalid")
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        if masked != self.expect_masked:
            raise WebSocketProtocolError("websocket_mask_direction_invalid")
        length_code = second & 0x7F
        offset = 2
        if length_code == 126:
            if len(self.buffer) < 4:
                return None
            length = struct.unpack("!H", self.buffer[2:4])[0]
            offset = 4
            if length < 126:
                raise WebSocketProtocolError("websocket_length_noncanonical")
        elif length_code == 127:
            if len(self.buffer) < 10:
                return None
            raw = bytes(self.buffer[2:10])
            if raw[0] & 0x80:
                raise WebSocketProtocolError("websocket_length_invalid")
            length = struct.unpack("!Q", raw)[0]
            offset = 10
            if length < 65536:
                raise WebSocketProtocolError("websocket_length_noncanonical")
        else:
            length = length_code
        if length > self.maximum_frame_bytes:
            raise WebSocketProtocolError("websocket_frame_too_large")
        mask_bytes = 4 if masked else 0
        required = offset + mask_bytes + length
        if len(self.buffer) < required:
            return None
        mask = bytes(self.buffer[offset : offset + 4]) if masked else b""
        payload = bytes(self.buffer[offset + mask_bytes : required])
        del self.buffer[:required]
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return fin, opcode, payload

    @staticmethod
    def _validate_data(opcode: int, payload: bytes) -> None:
        if opcode == 1:
            try:
                payload.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise WebSocketProtocolError("websocket_text_utf8_invalid") from exc


def encode_websocket_frame(
    opcode: int,
    payload: bytes = b"",
    *,
    masked: bool,
    fin: bool = True,
    mask_key: bytes | None = None,
) -> bytes:
    if opcode not in {0, 1, 2, 8, 9, 10}:
        raise ValueError("unsupported opcode")
    if opcode >= 8 and (not fin or len(payload) > 125):
        raise ValueError("invalid control frame")
    first = (0x80 if fin else 0) | opcode
    mask_flag = 0x80 if masked else 0
    if len(payload) < 126:
        header = bytes((first, mask_flag | len(payload)))
    elif len(payload) <= 0xFFFF:
        header = bytes((first, mask_flag | 126)) + struct.pack("!H", len(payload))
    else:
        header = bytes((first, mask_flag | 127)) + struct.pack("!Q", len(payload))
    if not masked:
        return header + payload
    key = mask_key or secrets.token_bytes(4)
    if len(key) != 4:
        raise ValueError("mask key must be four bytes")
    masked_payload = bytes(byte ^ key[index % 4] for index, byte in enumerate(payload))
    return header + key + masked_payload


class WebSocketConnection:
    def __init__(
        self,
        transport: socket.socket,
        *,
        expect_masked: bool,
        send_masked: bool,
        buffered: bytes = b"",
        maximum_frame_bytes: int = MAX_PHYSICAL_FRAME_BYTES,
        maximum_message_bytes: int = MAX_MESSAGE_BYTES,
    ) -> None:
        self._transport = transport
        self._codec = WebSocketFrameCodec(
            expect_masked=expect_masked,
            maximum_frame_bytes=maximum_frame_bytes,
            maximum_message_bytes=maximum_message_bytes,
        )
        self._send_masked = send_masked
        self._buffered_events: queue.SimpleQueue[tuple[int, bytes]] = queue.SimpleQueue()
        self._send_lock = threading.Lock()
        self._closed = False
        if buffered:
            for event in self._codec.feed(buffered):
                self._buffered_events.put(event)

    @classmethod
    def connect(
        cls,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> WebSocketConnection:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname or parsed.fragment:
            raise ZCodeRelayError("websocket_url_invalid")
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        try:
            raw = socket.create_connection((parsed.hostname, port), timeout_seconds)
        except OSError as exc:
            raise RelayClosedError("websocket_io_error") from exc
        transport: socket.socket
        try:
            if parsed.scheme == "wss":
                transport = ssl.create_default_context().wrap_socket(
                    raw, server_hostname=parsed.hostname
                )
            else:
                transport = raw
            transport.settimeout(timeout_seconds)
        except OSError as exc:
            raw.close()
            raise RelayClosedError("websocket_io_error") from exc
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        target = parsed.path or "/"
        if parsed.query:
            target += f"?{parsed.query}"
        request_headers = {
            "Host": parsed.netloc,
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Key": key,
            "Sec-WebSocket-Version": "13",
            **(dict(headers) if headers else {}),
        }
        requested_protocols = [
            value
            for name, value in request_headers.items()
            if name.casefold() == "sec-websocket-protocol"
        ]
        if len(requested_protocols) > 1:
            transport.close()
            raise ZCodeRelayError("websocket_request_headers_invalid")
        requested_protocol = requested_protocols[0] if requested_protocols else None
        request = (
            f"GET {target} HTTP/1.1\r\n"
            + "".join(f"{name}: {value}\r\n" for name, value in request_headers.items())
            + "\r\n"
        ).encode("latin1")
        try:
            transport.sendall(request)
        except OSError as exc:
            transport.close()
            raise RelayClosedError("websocket_io_error") from exc
        response = bytearray()
        while b"\r\n\r\n" not in response:
            try:
                chunk = transport.recv(4096)
            except OSError as exc:
                transport.close()
                raise RelayClosedError("websocket_io_error") from exc
            if not chunk:
                transport.close()
                raise ZCodeRelayError("websocket_upgrade_closed")
            response.extend(chunk)
            if len(response) > MAX_HTTP_HEADER_BYTES:
                transport.close()
                raise ZCodeRelayError("websocket_upgrade_too_large")
        head, buffered = bytes(response).split(b"\r\n\r\n", 1)
        try:
            status, response_headers = _parse_http_headers(head)
        except ZCodeRelayError:
            transport.close()
            raise
        expected = base64.b64encode(hashlib.sha1((key + _GUID).encode("ascii")).digest()).decode(
            "ascii"
        )
        if (
            status != "HTTP/1.1 101 Switching Protocols"
            or response_headers.get("upgrade", "").casefold() != "websocket"
            or "upgrade"
            not in {
                token.strip().casefold()
                for token in response_headers.get("connection", "").split(",")
            }
            or response_headers.get("sec-websocket-accept") != expected
            or "sec-websocket-extensions" in response_headers
            or (
                requested_protocol is not None
                and response_headers.get("sec-websocket-protocol") != requested_protocol
            )
            or (
                requested_protocol is None
                and "sec-websocket-protocol" in response_headers
            )
        ):
            transport.close()
            raise ZCodeRelayError("websocket_upgrade_rejected")
        try:
            transport.settimeout(timeout_seconds)
        except OSError as exc:
            transport.close()
            raise RelayClosedError("websocket_io_error") from exc
        return cls(
            transport,
            expect_masked=False,
            send_masked=True,
            buffered=buffered,
        )

    def send_text(self, value: str) -> None:
        self._send(1, value.encode("utf-8"))

    def send_json(self, value: object) -> None:
        self.send_text(json.dumps(value, ensure_ascii=True, separators=(",", ":")))

    def receive(self) -> tuple[int, bytes]:
        while not self._closed:
            try:
                opcode, payload = self._buffered_events.get_nowait()
            except queue.Empty:
                try:
                    data = self._transport.recv(65536)
                except OSError as exc:
                    self._closed = True
                    raise RelayClosedError("websocket_io_error") from exc
                if not data:
                    self._closed = True
                    raise RelayClosedError("websocket_closed") from None
                try:
                    events = self._codec.feed(data)
                except WebSocketProtocolError:
                    self._protocol_close(1002)
                    raise
                for event in events:
                    self._buffered_events.put(event)
                continue
            if opcode == 9:
                self._send(10, payload)
                continue
            if opcode == 10:
                continue
            if opcode == 8:
                self._closed = True
                try:
                    self._send_raw(8, payload)
                finally:
                    self._transport.close()
                raise RelayClosedError("websocket_peer_closed")
            return opcode, payload
        raise RelayClosedError("websocket_closed")

    def close(self, code: int = 1000) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._send_raw(8, struct.pack("!H", code))
            self._transport.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        finally:
            self._transport.close()

    def _protocol_close(self, code: int) -> None:
        try:
            self._send_raw(8, struct.pack("!H", code))
        except OSError:
            pass
        self._closed = True
        self._transport.close()

    def _send(self, opcode: int, payload: bytes) -> None:
        if self._closed:
            raise RelayClosedError("websocket_closed")
        self._send_raw(opcode, payload)

    def _send_raw(self, opcode: int, payload: bytes) -> None:
        frame = encode_websocket_frame(opcode, payload, masked=self._send_masked)
        with self._send_lock:
            try:
                self._transport.sendall(frame)
            except OSError as exc:
                self._closed = True
                raise RelayClosedError("websocket_io_error") from exc


def _parse_http_headers(raw: bytes) -> tuple[str, dict[str, str]]:
    try:
        lines = raw.decode("latin1").split("\r\n")
    except UnicodeError as exc:
        raise ZCodeRelayError("websocket_upgrade_invalid") from exc
    if not lines or len(lines) > 128:
        raise ZCodeRelayError("websocket_upgrade_invalid")
    status = lines[0]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or line[0] in " \t" or ":" not in line:
            raise ZCodeRelayError("websocket_upgrade_invalid")
        name, value = line.split(":", 1)
        normalized = name.casefold()
        if normalized in headers:
            if normalized in _SINGLETON_WEBSOCKET_HEADERS:
                raise ZCodeRelayError("websocket_upgrade_duplicate_header")
            headers[normalized] = f"{headers[normalized]},{value.strip()}"
        else:
            headers[normalized] = value.strip()
    return status, headers


class RelayTerminal:
    def __init__(
        self,
        identity: RelayIdentity,
        *,
        url: str = "wss://zcode.z.ai/ws",
        name: str = "AgentTools ZCode bridge",
    ) -> None:
        self.identity = identity
        self.url = url
        self.name = name
        self._role = "terminal"
        self._connection: WebSocketConnection | None = None
        self._payload_handler: Callable[[object], None] | None = None
        self._failure: ZCodeRelayError | None = None
        self._failure_event = threading.Event()
        self._receiver: threading.Thread | None = None
        self._closed_intentionally = False

    @property
    def failure(self) -> ZCodeRelayError | None:
        return self._failure

    @property
    def failed(self) -> bool:
        return self._failure_event.is_set()

    def connect(self, timeout_seconds: float = 30.0) -> None:
        parsed = urllib.parse.urlsplit(self.url)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query.append(("mid", self.identity.device_mid))
        url = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), "")
        )
        connection = WebSocketConnection.connect(
            url,
            headers={"X-Device-ID": self.identity.device_mid},
            timeout_seconds=timeout_seconds,
        )
        self._connection = connection
        connection.send_json(
            {
                "type": "auth_init",
                "role": self._role,
                "device_sid": self.identity.device_sid,
                "meta": {"platform": sys.platform, "version": "0.16.5", "name": self.name},
                "client_ts": int(time.time() * 1000),
            }
        )
        deadline = time.monotonic() + timeout_seconds
        challenged = False
        while time.monotonic() < deadline:
            opcode, payload = connection.receive()
            if opcode != 1:
                raise ZCodeRelayError("relay_auth_non_text")
            message = _strict_json_object(payload, "relay_auth_json_invalid")
            kind = message.get("type")
            if kind == "auth_challenge":
                nonce = message.get("nonce")
                if challenged or not isinstance(nonce, str) or not 0 < len(nonce) <= 1024:
                    raise ZCodeRelayError("relay_auth_challenge_invalid")
                challenged = True
                connection.send_json(
                    {
                        "type": "auth_response",
                        "device_sid": self.identity.device_sid,
                        "proof": calculate_proof(
                            self.identity.pass_hash,
                            nonce,
                            self._role,
                            self.identity.device_sid,
                        ),
                        "client_ts": int(time.time() * 1000),
                    }
                )
            elif kind == "auth_ack" and challenged:
                self._receiver = threading.Thread(target=self._receive_loop, daemon=True)
                self._receiver.start()
                return
            elif kind == "error":
                raise _relay_error(message)
        raise ZCodeRelayError("relay_auth_timeout")

    def set_payload_handler(self, handler: Callable[[object], None]) -> None:
        self._payload_handler = handler

    def send_payload(self, payload: Mapping[str, object]) -> None:
        self._send_json(
            {"type": "data", "payload": dict(payload), "client_ts": int(time.time() * 1000)}
        )

    def heartbeat(self) -> None:
        self._send_json(
            {
                "type": "pair_status_query",
                "device_sid": self.identity.device_sid,
                "client_ts": int(time.time() * 1000),
            }
        )

    def wait_failed(self, timeout_seconds: float) -> bool:
        return self._failure_event.wait(timeout_seconds)

    def close(self) -> None:
        self._closed_intentionally = True
        if self._connection:
            self._connection.close()
        if self._receiver and self._receiver is not threading.current_thread():
            self._receiver.join(timeout=2.0)

    def _send_json(self, value: object) -> None:
        if self._failure:
            raise self._failure
        if not self._connection:
            raise RelayClosedError("relay_not_connected")
        self._connection.send_json(value)

    def _receive_loop(self) -> None:
        try:
            assert self._connection is not None
            while True:
                opcode, payload = self._connection.receive()
                if opcode != 1:
                    raise ZCodeRelayError("relay_message_non_text")
                message = _strict_json_object(payload, "relay_json_invalid")
                kind = message.get("type")
                if kind == "data":
                    if self._payload_handler:
                        self._payload_handler(message.get("payload"))
                elif kind == "error":
                    raise _relay_error(message)
                elif kind not in {"pair_status_ack", "auth_ack"}:
                    raise ZCodeRelayError("relay_schema_drift")
        except (ZCodeRelayError, OSError) as exc:
            if not self._closed_intentionally:
                self._failure = (
                    exc if isinstance(exc, ZCodeRelayError) else RelayClosedError("relay_io_error")
                )
                self._failure_event.set()


def _strict_json_object(raw: bytes, code: str) -> dict[str, object]:
    def no_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ZCodeRelayError(code)
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
        )
        _validate_nested_depth(value)
    except (
        RecursionError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        ZCodeRelayError,
    ) as exc:
        raise ZCodeRelayError(code) from exc
    if not isinstance(value, dict):
        raise ZCodeRelayError(code)
    return value


def _relay_error(message: Mapping[str, object]) -> ZCodeRelayError:
    candidates = (message.get("code"), message.get("reason"), message.get("error"))
    text = " ".join(str(value) for value in candidates if value is not None).casefold()
    if "kick" in text or "conflict" in text:
        return RelayConflictError("relay_session_conflict")
    return ZCodeRelayError("relay_error")


class RelayOpClient:
    def __init__(self, terminal: RelayTerminal) -> None:
        self.terminal = terminal
        self._pending: dict[str, _PendingCall] = {}
        self._lock = threading.Lock()

    def accept(self, payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        request_id = payload.get("requestId")
        if not isinstance(request_id, str):
            return False
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if not pending:
            return False
        if payload.get("success") is False or str(payload.get("zcode_type", "")).endswith("-error"):
            reason = str(
                payload.get("reason") or payload.get("zcode_type") or "relay_operation_error"
            )
            safe = (
                "relay_session_conflict"
                if "conflict" in reason.casefold()
                else "relay_operation_error"
            )
            pending.error = (
                RelayConflictError(safe)
                if safe == "relay_session_conflict"
                else ZCodeRelayError(safe)
            )
        else:
            pending.result = payload
        pending.event.set()
        return True

    def request(
        self,
        zcode_type: str,
        fields: Mapping[str, object] | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, object]:
        request_id = secrets.token_hex(16)
        pending = _PendingCall(threading.Event())
        with self._lock:
            self._pending[request_id] = pending
        self.terminal.send_payload(
            {"zcode_type": zcode_type, "requestId": request_id, **(dict(fields) if fields else {})}
        )
        if not pending.event.wait(timeout_seconds):
            with self._lock:
                self._pending.pop(request_id, None)
            raise ZCodeRelayError("relay_operation_timeout")
        if pending.error:
            raise pending.error
        if not isinstance(pending.result, dict):
            raise ZCodeRelayError("relay_operation_schema_drift")
        return pending.result

    def abort(self, error: ZCodeRelayError) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            item.error = error
            item.event.set()
