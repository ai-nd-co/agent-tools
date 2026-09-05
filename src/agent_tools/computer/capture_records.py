from __future__ import annotations

import ctypes
import hashlib
import hmac
import importlib
import json
import os
import re
import secrets
import stat
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path
from typing import Any, Protocol

from agent_tools.computer.models import ComputerError, WindowIdentity

CAPTURE_RECORD_SCHEMA = "agent-tools.capture-record/v1"
CAPTURE_RECORD_TTL_SECONDS = 60
MAX_CAPTURE_RECORDS = 64
MAX_CAPTURE_RECORD_BYTES = 64 * 1024
MAX_CAPTURE_STORE_ENTRIES = 256
CAPTURE_STORE_LOCK_NAME = ".store-lock"
CAPTURE_STORE_LOCK_TIMEOUT_MS = 2_000
CAPTURE_ID_PATTERN = re.compile(r"cap_[0-9a-f]{32}\Z")
KEY_BYTES = 32
FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_ALWAYS = 4
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
ERROR_SHARING_VIOLATION = 32
ERROR_LOCK_VIOLATION = 33
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class CaptureClock(Protocol):
    def now_unix_ms(self) -> int: ...


class _SystemCaptureClock:
    def now_unix_ms(self) -> int:
        return int(time.time() * 1000)


class CaptureRecordStore:
    """Owner-private, signed, bounded capture provenance records."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        clock: CaptureClock | None = None,
    ) -> None:
        self.root = root if root is not None else _default_capture_store_root()
        self._clock = clock or _SystemCaptureClock()

    def new_capture_id(self) -> str:
        return f"cap_{secrets.token_hex(16)}"

    def identity_reference(self, identity: WindowIdentity) -> dict[str, Any]:
        if (
            identity.process is None
            or identity.process_started is None
            or identity.session_id is None
            or not identity.desktop_name
        ):
            raise ComputerError(
                "capture_identity_unavailable",
                "The exact target process identity is unavailable for coordinate provenance.",
            )
        with _capture_store_lock(self.root):
            key = self._ensure_ready_locked()
            return _identity_reference(identity, key)

    def validate_identity_reference(
        self,
        reference: dict[str, Any],
        identity: WindowIdentity,
    ) -> None:
        with _capture_store_lock(self.root):
            key = self._ensure_ready_locked()
            current_reference = _identity_reference(identity, key)
            required_fields = (
                "hwnd",
                "pid",
                "process_started",
                "session_id",
                "class_hmac_sha256",
                "desktop_hmac_sha256",
                "fingerprint",
            )
            if any(
                reference.get(field) != current_reference.get(field)
                for field in required_fields
            ):
                raise ComputerError(
                    "capture_target_changed",
                    "The captured window identity changed; take a fresh screenshot.",
                )

    def save(self, payload: dict[str, Any]) -> None:
        capture_id = _validated_capture_id(payload.get("capture_id"))
        if payload.get("schema_version") != CAPTURE_RECORD_SCHEMA:
            raise ComputerError(
                "capture_record_invalid",
                "The capture provenance record is invalid.",
            )
        record_path = self._record_path(capture_id)
        committed = False
        try:
            with _capture_store_lock(self.root):
                key = self._ensure_ready_locked()
                canonical = _canonical_json(payload)
                envelope = {
                    "payload": payload,
                    "hmac_sha256": hmac.new(key, canonical, hashlib.sha256).hexdigest(),
                }
                encoded = _canonical_json(envelope)
                if len(encoded) > MAX_CAPTURE_RECORD_BYTES:
                    raise ComputerError(
                        "capture_record_too_large",
                        "The capture provenance record exceeded its safety limit.",
                    )
                self._prune_locked(keep_capture_id=capture_id)
                _atomic_private_write(record_path, encoded)
                committed = True
        except Exception:
            if committed:
                _delete_record_path(record_path)
            raise

    def load(self, capture_id: str) -> dict[str, Any]:
        validated_id = _validated_capture_id(capture_id)
        with _capture_store_lock(self.root):
            key = self._ensure_ready_locked()
            path = self._record_path(validated_id)
            payload = _read_signed_record(path, key)
            if payload.get("capture_id") != validated_id:
                raise ComputerError(
                    "capture_record_invalid",
                    "The capture provenance record is invalid.",
                )
            created_ms = payload.get("created_unix_ms")
            expires_ms = payload.get("expires_unix_ms")
            if type(created_ms) is not int or type(expires_ms) is not int:
                raise ComputerError(
                    "capture_record_invalid",
                    "The capture provenance record is invalid.",
                )
            current_ms = self.now_unix_ms()
            if not created_ms <= current_ms < expires_ms:
                _delete_record_path(path)
                raise ComputerError(
                    "capture_expired",
                    "The capture coordinate record expired; take a fresh screenshot.",
                )
            return payload

    def now_unix_ms(self) -> int:
        current_ms = self._clock.now_unix_ms()
        if type(current_ms) is not int or current_ms < 0:
            raise ComputerError(
                "capture_clock_invalid",
                "The capture freshness clock returned an invalid value.",
            )
        return current_ms

    def assert_fresh(self, payload: dict[str, Any]) -> int:
        created_ms = payload.get("created_unix_ms")
        expires_ms = payload.get("expires_unix_ms")
        if type(created_ms) is not int or type(expires_ms) is not int:
            raise ComputerError(
                "capture_record_invalid",
                "The capture provenance record is invalid.",
            )
        current_ms = self.now_unix_ms()
        if not created_ms <= current_ms < expires_ms:
            raise ComputerError(
                "capture_expired",
                "The capture coordinate record expired; take a fresh screenshot.",
            )
        return current_ms

    def delete(self, capture_id: str) -> None:
        validated_id = _validated_capture_id(capture_id)
        with _capture_store_lock(self.root):
            self._ensure_ready_locked()
            _delete_record_path(self._record_path(validated_id))

    def _record_path(self, capture_id: str) -> Path:
        return self.root / f"{capture_id}.json"

    def _ensure_ready_locked(self) -> bytes:
        if sys.platform != "win32":
            raise ComputerError(
                "unsupported_platform",
                "Screenshot capture records require Windows.",
                exit_code=2,
            )
        _prepare_capture_store_root(self.root)
        key_path = self.root / ".integrity-key"
        _create_integrity_key_exclusive(key_path)
        _reject_reparse_point(key_path, "capture_store_unavailable")
        try:
            key_stat = key_path.stat()
            key = key_path.read_bytes()
        except OSError as exc:
            raise ComputerError(
                "capture_store_unavailable",
                "The capture integrity key could not be read.",
            ) from exc
        if not stat.S_ISREG(key_stat.st_mode) or len(key) != KEY_BYTES:
            raise ComputerError(
                "capture_store_unavailable",
                "The capture integrity key is invalid.",
            )
        _set_owner_only_acl(key_path, directory=False)
        return key

    def _prune_locked(self, *, keep_capture_id: str) -> None:
        now = time.time()
        current_ms = int(now * 1000)
        entries, temp_entries = _scan_capture_store(self.root)
        ranked: list[tuple[bool, float, Path]] = []
        keep_path: Path | None = None
        keep_valid = True
        for entry in entries:
            path = Path(entry.path)
            is_keep = entry.name == f"{keep_capture_id}.json"
            if is_keep:
                keep_path = path
            try:
                metadata = entry.stat(follow_symlinks=False)
                valid = stat.S_ISREG(metadata.st_mode) and not _is_reparse_point(path)
                stale = (
                    current_ms - int(metadata.st_mtime * 1000)
                    > CAPTURE_RECORD_TTL_SECONDS * 2_000
                )
                if is_keep:
                    keep_valid = valid
                ranked.append((valid and not stale, metadata.st_mtime, path))
            except OSError:
                if is_keep:
                    keep_valid = False
                ranked.append((False, 0.0, path))

        if not keep_valid:
            raise ComputerError(
                "capture_store_capacity_unavailable",
                "The capture store could not replace its target record safely.",
            )

        existing_capacity = (
            MAX_CAPTURE_RECORDS if keep_path is not None else MAX_CAPTURE_RECORDS - 1
        )
        ranked.sort(key=lambda item: (item[0], item[1]))
        remaining = {path for _valid, _mtime, path in ranked}
        for _valid, _mtime, path in ranked:
            if len(remaining) <= existing_capacity:
                break
            if path == keep_path:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            if not path.exists():
                remaining.discard(path)

        physical_records, _physical_temps = _scan_capture_store(self.root)
        if len(physical_records) > existing_capacity:
            raise ComputerError(
                "capture_store_capacity_unavailable",
                "The capture store could not enforce its record limit safely.",
            )

        for entry in temp_entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
                if now - metadata.st_mtime > CAPTURE_RECORD_TTL_SECONDS * 2:
                    Path(entry.path).unlink(missing_ok=True)
            except OSError:
                continue


def _scan_capture_store(
    root: Path,
) -> tuple[list[os.DirEntry[str]], list[os.DirEntry[str]]]:
    records: list[os.DirEntry[str]] = []
    temporary: list[os.DirEntry[str]] = []
    retained_entries = 0
    try:
        for entry in os.scandir(root):
            retained_entries += 1
            if retained_entries > MAX_CAPTURE_STORE_ENTRIES:
                raise ComputerError(
                    "capture_store_overflow",
                    "The capture store exceeded its safety limit.",
                )
            if entry.name.startswith("cap_") and entry.name.endswith(".json"):
                records.append(entry)
            if (
                entry.name.startswith(".cap_")
                or entry.name.startswith("..integrity-key.")
            ) and entry.name.endswith(".tmp"):
                temporary.append(entry)
    except ComputerError:
        raise
    except OSError as exc:
        raise ComputerError(
            "capture_store_unavailable",
            "The capture store could not be inspected safely.",
        ) from exc
    return records, temporary


def _default_capture_store_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise ComputerError(
            "capture_store_unavailable",
            "LOCALAPPDATA is unavailable for owner-private capture records.",
        )
    return Path(local_app_data) / "AgentTools" / "computer" / "captures-v1"


def _prepare_capture_store_root(root: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ComputerError(
            "capture_store_unavailable",
            "The owner-private capture store could not be created.",
        ) from exc
    _reject_reparse_point(root, "capture_store_unavailable")
    _set_owner_only_acl(root, directory=True)


def _create_integrity_key_exclusive(path: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="..integrity-key.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(secrets.token_bytes(KEY_BYTES))
            temporary.flush()
            os.fsync(temporary.fileno())
        _set_owner_only_acl(temporary_path, directory=False)
        os.link(temporary_path, path)
    except FileExistsError:
        return
    except OSError as exc:
        raise ComputerError(
            "capture_store_unavailable",
            "The capture integrity key could not be created.",
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    _set_owner_only_acl(path, directory=False)


def _validated_capture_id(value: object) -> str:
    if not isinstance(value, str) or CAPTURE_ID_PATTERN.fullmatch(value) is None:
        raise ComputerError(
            "invalid_capture_id",
            "The capture ID is invalid.",
            exit_code=2,
        )
    return value


def _identity_reference(identity: WindowIdentity, key: bytes) -> dict[str, Any]:
    exact = {
        "hwnd": identity.hwnd,
        "pid": identity.pid,
        "class": identity.class_name,
        "process_started": identity.process_started,
        "session_id": identity.session_id,
        "desktop": identity.desktop_name,
    }
    fingerprint = hmac.new(
        key,
        b"window-identity-v2\0" + _canonical_json(exact),
        hashlib.sha256,
    ).hexdigest()

    def field_fingerprint(label: str, value: str | None) -> str:
        encoded = "<missing>" if value is None else value
        return hmac.new(
            key,
            f"window-{label}-v1\0{encoded}".encode(),
            hashlib.sha256,
        ).hexdigest()

    return {
        "hwnd": identity.hwnd,
        "pid": identity.pid,
        "thread_id": identity.thread_id,
        "process_started": identity.process_started,
        "session_id": identity.session_id,
        "process_hmac_sha256": field_fingerprint("process", identity.process),
        "title_hmac_sha256": field_fingerprint("title", identity.title),
        "class_hmac_sha256": field_fingerprint("class", identity.class_name),
        "desktop_hmac_sha256": field_fingerprint("desktop", identity.desktop_name),
        "fingerprint": fingerprint,
    }


def _read_signed_record(path: Path, key: bytes) -> dict[str, Any]:
    if not path.exists():
        raise ComputerError(
            "capture_record_missing",
            "The capture coordinate record is missing; take a fresh screenshot.",
        )
    _reject_reparse_point(path, "capture_record_invalid")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record could not be read safely.",
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_CAPTURE_RECORD_BYTES:
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record is invalid.",
        )
    _set_owner_only_acl(path, directory=False)
    try:
        encoded = path.read_bytes()
        envelope = json.loads(encoded.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record is invalid.",
        ) from exc
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "hmac_sha256"}:
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record is invalid.",
        )
    payload = envelope.get("payload")
    signature = envelope.get("hmac_sha256")
    if not isinstance(payload, dict) or not isinstance(signature, str):
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record is invalid.",
        )
    expected = hmac.new(key, _canonical_json(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ComputerError(
            "capture_record_altered",
            "The capture provenance record was altered; take a fresh screenshot.",
        )
    if payload.get("schema_version") != CAPTURE_RECORD_SCHEMA:
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record is invalid.",
        )
    return payload


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _atomic_private_write(path: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        _set_owner_only_acl(temporary_path, directory=False)
        os.replace(temporary_path, path)
        temporary_path = None
    except ComputerError:
        raise
    except OSError as exc:
        raise ComputerError(
            "capture_store_unavailable",
            "The owner-private capture store could not be written.",
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _delete_record_path(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise ComputerError(
            "capture_record_cleanup_failed",
            "The capture provenance record could not be removed safely.",
        ) from exc
    if path.exists():
        raise ComputerError(
            "capture_record_cleanup_failed",
            "The capture provenance record could not be removed safely.",
        )


@contextmanager
def _capture_store_lock(root: Path) -> Iterator[None]:
    if sys.platform != "win32":
        raise ComputerError(
            "unsupported_platform",
            "Screenshot capture records require Windows.",
            exit_code=2,
        )
    _prepare_capture_store_root(root)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    lock_path = root / CAPTURE_STORE_LOCK_NAME
    deadline = time.monotonic() + CAPTURE_STORE_LOCK_TIMEOUT_MS / 1000
    handle: int | None = None
    last_error = 0
    while time.monotonic() < deadline:
        ctypes.set_last_error(0)
        candidate = kernel32.CreateFileW(
            str(lock_path),
            GENERIC_READ | GENERIC_WRITE,
            0,
            None,
            OPEN_ALWAYS,
            FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if candidate and candidate != INVALID_HANDLE_VALUE:
            handle = int(candidate)
            break
        last_error = ctypes.get_last_error()
        if last_error not in {ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION}:
            break
        time.sleep(0.01)
    if handle is None:
        code = "capture_store_busy" if last_error in {
            ERROR_SHARING_VIOLATION,
            ERROR_LOCK_VIOLATION,
        } else "capture_store_unavailable"
        raise ComputerError(
            code,
            "The capture store lock could not be acquired safely.",
        )
    try:
        _reject_reparse_point(lock_path, "capture_store_unavailable")
        _set_owner_only_acl(lock_path, directory=False)
        yield
    finally:
        if not kernel32.CloseHandle(handle):
            raise ComputerError(
                "capture_store_busy",
                "The capture store lock could not be released safely.",
            )


def _set_owner_only_acl(path: Path, *, directory: bool) -> None:
    try:
        win32api = importlib.import_module("win32api")
        win32con = importlib.import_module("win32con")
        win32security = importlib.import_module("win32security")
        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
        )
        owner_sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        acl = win32security.ACL()
        inherit_flags = 0
        if directory:
            inherit_flags = (
                win32security.OBJECT_INHERIT_ACE | win32security.CONTAINER_INHERIT_ACE
            )
        acl.AddAccessAllowedAceEx(
            win32security.ACL_REVISION_DS,
            inherit_flags,
            win32con.GENERIC_ALL,
            owner_sid,
        )
        security_information = (
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION
            | win32security.PROTECTED_DACL_SECURITY_INFORMATION
        )
        win32security.SetNamedSecurityInfo(
            str(path),
            win32security.SE_FILE_OBJECT,
            security_information,
            owner_sid,
            None,
            acl,
            None,
        )
        descriptor = win32security.GetNamedSecurityInfo(
            str(path),
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION,
        )
        owner = descriptor.GetSecurityDescriptorOwner()
        actual_acl = descriptor.GetSecurityDescriptorDacl()
        owner_text = win32security.ConvertSidToStringSid(owner)
        expected_owner_text = win32security.ConvertSidToStringSid(owner_sid)
        if (
            owner_text != expected_owner_text
            or actual_acl is None
            or actual_acl.GetAceCount() < 1
        ):
            raise OSError("capture ACL verification failed")
        for index in range(actual_acl.GetAceCount()):
            ace = actual_acl.GetAce(index)
            ace_owner_text = win32security.ConvertSidToStringSid(ace[2])
            if (
                ace[0][0] != win32security.ACCESS_ALLOWED_ACE_TYPE
                or ace_owner_text != expected_owner_text
            ):
                raise OSError("capture ACL verification failed")
    except (ImportError, OSError) as exc:
        raise ComputerError(
            "capture_store_permissions_failed",
            "The capture store could not be restricted to the current owner.",
        ) from exc


def _reject_reparse_point(path: Path, code: str) -> None:
    if _is_reparse_point(path):
        raise ComputerError(
            code,
            "Capture provenance paths cannot use reparse points.",
        )


def _is_reparse_point(path: Path) -> bool:
    if sys.platform != "win32":
        return path.is_symlink()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetFileAttributesW.argtypes = (wintypes.LPCWSTR,)
    kernel32.GetFileAttributesW.restype = wintypes.DWORD
    attributes = int(kernel32.GetFileAttributesW(str(path)))
    if attributes == INVALID_FILE_ATTRIBUTES:
        return False
    return bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT)
