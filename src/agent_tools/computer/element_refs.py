from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from agent_tools.computer.capture_records import (
    KEY_BYTES,
    _atomic_private_write,
    _canonical_json,
    _capture_store_lock,
    _create_integrity_key_exclusive,
    _prepare_capture_store_root,
    _reject_reparse_point,
    _set_owner_only_acl,
)
from agent_tools.computer.models import (
    MAX_CLASS_NAME_CHARS,
    Bounds,
    ComputerError,
    WindowIdentity,
)

ELEMENT_REFERENCE_SCHEMA = "agent-tools.element-reference/v1"
ELEMENT_REFERENCE_TTL_SECONDS = 15
MAX_ELEMENT_REFERENCES = 256
MAX_ELEMENT_REFERENCE_BYTES = 32 * 1024
MAX_ELEMENT_REFERENCE_STORE_ENTRIES = 512
MAX_WINAPP_SELECTOR_CHARS = 240
ELEMENT_REFERENCE_PATTERN = re.compile(r"eref_[0-9a-f]{32}\Z")
ANCESTOR_SIGNATURE_PATTERN = re.compile(r"sha256:[0-9a-f]{24}\Z")


class ElementReferenceClock(Protocol):
    def now_unix_ms(self) -> int: ...


class _SystemElementReferenceClock:
    def now_unix_ms(self) -> int:
        return int(time.time() * 1000)


class ElementReferenceStore:
    """Owner-private, signed, short-lived inspect-to-action references."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        clock: ElementReferenceClock | None = None,
    ) -> None:
        self.root = root if root is not None else _default_store_root()
        self._clock = clock or _SystemElementReferenceClock()

    def issue(
        self,
        *,
        identity: WindowIdentity,
        element: dict[str, Any],
    ) -> dict[str, Any]:
        return self._issue_records(
            identity=identity,
            records=[_element_record(element)],
        )[0]

    def _issue_records(
        self,
        *,
        identity: WindowIdentity,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not records:
            return []
        if len(records) > MAX_ELEMENT_REFERENCES:
            raise ComputerError(
                "element_ref_store_capacity_unavailable",
                "The inspection returned too many action references for the bounded store.",
            )
        window = _window_record(identity)
        now_ms = self.now_unix_ms()
        expires_ms = now_ms + ELEMENT_REFERENCE_TTL_SECONDS * 1000
        payloads: list[dict[str, Any]] = []
        for record in records:
            reference = f"eref_{secrets.token_hex(16)}"
            payloads.append(
                {
                    "schema_version": ELEMENT_REFERENCE_SCHEMA,
                    "element_ref": reference,
                    "created_unix_ms": now_ms,
                    "expires_unix_ms": expires_ms,
                    "window": window,
                    "element": record,
                }
            )
        written: list[Path] = []
        with self._locked():
            key = self._ensure_ready_locked()
            self._prune_locked(now_ms=now_ms, reserve_count=len(payloads))
            try:
                for payload in payloads:
                    envelope = {
                        "payload": payload,
                        "hmac_sha256": hmac.new(
                            key,
                            _canonical_json(payload),
                            hashlib.sha256,
                        ).hexdigest(),
                    }
                    encoded = _canonical_json(envelope)
                    if len(encoded) > MAX_ELEMENT_REFERENCE_BYTES:
                        raise ComputerError(
                            "element_ref_too_large",
                            "The element reference exceeded its safety limit.",
                        )
                    path = self._path(str(payload["element_ref"]))
                    _atomic_private_write(path, encoded)
                    written.append(path)
            except ComputerError as exc:
                for path in written:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                if exc.code == "element_ref_too_large":
                    raise
                raise _store_error() from exc
        return [
            {
                "element_ref": payload["element_ref"],
                "expires_at": _timestamp(expires_ms),
            }
            for payload in payloads
        ]

    def resolve(
        self,
        reference: str,
        *,
        identity: WindowIdentity,
        require_title_match: bool = False,
    ) -> dict[str, Any]:
        validated = _validated_reference(reference)
        with self._locked():
            key = self._ensure_ready_locked()
            path = self._path(validated)
            payload = self._read_locked(path, key)
            now_ms = self.now_unix_ms()
            created_ms = payload.get("created_unix_ms")
            expires_ms = payload.get("expires_unix_ms")
            if (
                type(created_ms) is not int
                or type(expires_ms) is not int
                or not created_ms <= now_ms < expires_ms
            ):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise ComputerError(
                    "element_ref_expired",
                    "The element reference expired; inspect the window again.",
                    details={"resolution_stage": "reference_freshness"},
                )
        recorded_window = payload.get("window")
        if not isinstance(recorded_window, dict):
            raise _invalid_reference()
        mismatches = _window_mismatches(
            recorded_window,
            identity,
            require_title_match=require_title_match,
        )
        if mismatches:
            raise ComputerError(
                "stale_window",
                "The element reference belongs to a different window identity.",
                details={
                    "identity_mismatch_fields": mismatches,
                    "title_changed": recorded_window.get("title") != identity.title,
                    "title_change_was_authoritative": require_title_match,
                    "resolution_stage": "element_ref_window_identity",
                },
            )
        element = payload.get("element")
        if not isinstance(element, dict):
            raise _invalid_reference()
        return {
            "element_ref": validated,
            "created_unix_ms": payload["created_unix_ms"],
            "expires_unix_ms": payload["expires_unix_ms"],
            "expires_at": _timestamp(int(payload["expires_unix_ms"])),
            "window": recorded_window,
            "element": element,
        }

    def now_unix_ms(self) -> int:
        value = self._clock.now_unix_ms()
        if type(value) is not int or value < 0:
            raise ComputerError(
                "element_ref_clock_invalid",
                "The element reference clock returned an invalid value.",
            )
        return value

    def _read_locked(self, path: Path, key: bytes) -> dict[str, Any]:
        if not path.exists():
            raise ComputerError(
                "element_ref_missing",
                "The element reference is missing; inspect the window again.",
                details={"resolution_stage": "reference_lookup"},
            )
        _reject_reparse_point(path, "element_ref_invalid")
        try:
            metadata = path.stat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > MAX_ELEMENT_REFERENCE_BYTES
            ):
                raise _invalid_reference()
            _set_owner_only_acl(path, directory=False)
            envelope = json.loads(path.read_text(encoding="ascii"))
        except ComputerError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _invalid_reference() from exc
        if not isinstance(envelope, dict) or set(envelope) != {"payload", "hmac_sha256"}:
            raise _invalid_reference()
        payload = envelope.get("payload")
        signature = envelope.get("hmac_sha256")
        if not isinstance(payload, dict) or not isinstance(signature, str):
            raise _invalid_reference()
        expected = hmac.new(key, _canonical_json(payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ComputerError(
                "element_ref_altered",
                "The element reference was altered; inspect the window again.",
                details={"resolution_stage": "reference_integrity"},
            )
        if (
            payload.get("schema_version") != ELEMENT_REFERENCE_SCHEMA
            or payload.get("element_ref") != path.stem
        ):
            raise _invalid_reference()
        return payload

    def _path(self, reference: str) -> Path:
        return self.root / f"{reference}.json"

    def _ensure_ready_locked(self) -> bytes:
        try:
            _prepare_capture_store_root(self.root)
            key_path = self.root / ".integrity-key"
            _create_integrity_key_exclusive(key_path)
            _reject_reparse_point(key_path, "element_ref_store_unavailable")
            metadata = key_path.stat()
            key = key_path.read_bytes()
            if not stat.S_ISREG(metadata.st_mode) or len(key) != KEY_BYTES:
                raise _store_error()
            _set_owner_only_acl(key_path, directory=False)
            return key
        except ComputerError as exc:
            if exc.code == "element_ref_store_unavailable":
                raise
            raise _store_error() from exc
        except OSError as exc:
            raise _store_error() from exc

    def _prune_locked(self, *, now_ms: int, reserve_count: int) -> None:
        records: list[tuple[float, Path]] = []
        try:
            entries = list(os.scandir(self.root))
        except OSError as exc:
            raise _store_error() from exc
        if len(entries) > MAX_ELEMENT_REFERENCE_STORE_ENTRIES:
            raise ComputerError(
                "element_ref_store_overflow",
                "The element reference store exceeded its safety limit.",
            )
        for entry in entries:
            if not entry.name.startswith("eref_") or not entry.name.endswith(".json"):
                continue
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            expired = int(metadata.st_mtime * 1000) < (
                now_ms - ELEMENT_REFERENCE_TTL_SECONDS * 2000
            )
            if expired:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            records.append((metadata.st_mtime, path))
        records.sort()
        capacity = MAX_ELEMENT_REFERENCES - reserve_count
        while len(records) > capacity:
            _mtime, path = records.pop(0)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                records.append((_mtime, path))
                records.sort()
                break
        if len(records) > capacity:
            raise ComputerError(
                "element_ref_store_capacity_unavailable",
                "The element reference store could not enforce its record limit safely.",
            )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if sys.platform != "win32":
            raise ComputerError(
                "unsupported_platform",
                "Element references require Windows.",
                exit_code=2,
            )
        try:
            with _capture_store_lock(self.root):
                yield
        except ComputerError as exc:
            if exc.code.startswith("element_ref_"):
                raise
            raise _store_error() from exc


def attach_element_references(
    result: dict[str, Any],
    *,
    identity: WindowIdentity,
    collection: str = "elements",
    store: ElementReferenceStore | None = None,
) -> dict[str, Any]:
    rows = result.get(collection)
    if not isinstance(rows, list):
        return result
    resolved_store = store or ElementReferenceStore()
    issued: list[dict[str, Any] | None] = []
    records: list[dict[str, Any]] = []
    record_indexes: list[int] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        try:
            record = _element_record(raw)
        except ComputerError as exc:
            if exc.code == "element_ref_unavailable":
                unavailable_reason = (
                    raw.get("element_ref_reason")
                    if raw.get("element_ref_reason")
                    == "provider_filtered_hierarchy_inexact"
                    else "safe_locator_unavailable"
                )
                issued.append(
                    {
                        **raw,
                        "element_ref": None,
                        "element_ref_status": "unavailable",
                        "element_ref_reason": unavailable_reason,
                    }
                )
                continue
            raise
        record_indexes.append(len(issued))
        records.append(record)
        issued.append(dict(raw))
    public_records = resolved_store._issue_records(identity=identity, records=records)
    for index, public in zip(record_indexes, public_records, strict=True):
        raw = issued[index]
        assert isinstance(raw, dict)
        issued[index] = {**raw, **public, "element_ref_status": "available"}
    return {**result, collection: [item for item in issued if item is not None]}


def _window_record(identity: WindowIdentity) -> dict[str, Any]:
    if (
        identity.process is None
        or identity.process_started is None
        or identity.session_id is None
        or not identity.desktop_name
    ):
        raise ComputerError(
            "target_identity_unavailable",
            "The strong window identity is unavailable for an element reference.",
        )
    return {
        "hwnd": identity.hwnd,
        "pid": identity.pid,
        "thread_id": identity.thread_id,
        "process": identity.process,
        "process_started": identity.process_started,
        "title": identity.title,
        "class": identity.class_name,
        "session_id": identity.session_id,
        "desktop": identity.desktop_name,
    }


def _element_record(element: dict[str, Any]) -> dict[str, Any]:
    locator = element.get("locator")
    if "locator" in element and not isinstance(locator, dict):
        raise ComputerError(
            "element_ref_unavailable",
            "The semantic element cannot be bound to a safe action reference.",
        )
    if "locator" not in element:
        selector = element.get("element")
        locator = {
            "provider": element.get("provider") or "winapp",
            "strategy": "selector_v1",
            "selector": selector,
            "ancestor_signature": (
                (element.get("hierarchy") or {}).get("ancestor_signature")
                if isinstance(element.get("hierarchy"), dict)
                else None
            ),
        }
    assert isinstance(locator, dict)
    provider = _bounded_optional(
        locator.get("provider") or element.get("provider") or "winapp", 64
    )
    if provider is None:
        raise ComputerError(
            "element_ref_unavailable",
            "The semantic element cannot be bound to a safe action reference.",
        )
    selector = locator.get("selector") or element.get("element")
    selector_limit = MAX_WINAPP_SELECTOR_CHARS if provider == "winapp" else 512
    if not isinstance(selector, str) or not selector or len(selector) > selector_limit:
        raise ComputerError(
            "element_ref_unavailable",
            "The semantic element cannot be bound to a safe action reference.",
        )
    bounds = element.get("bounds")
    if not isinstance(bounds, dict):
        raise ComputerError(
            "element_ref_unavailable",
            "The semantic element has no bounded geometry for an action reference.",
        )
    try:
        normalized_bounds = Bounds(
            int(bounds["x"]),
            int(bounds["y"]),
            int(bounds["width"]),
            int(bounds["height"]),
        ).to_jsonable()
    except (KeyError, TypeError, ValueError) as exc:
        raise ComputerError(
            "element_ref_unavailable",
            "The semantic element geometry is invalid for an action reference.",
        ) from exc
    hierarchy = element.get("hierarchy")
    ancestor_signature = locator.get("ancestor_signature")
    if ancestor_signature is None and isinstance(hierarchy, dict):
        ancestor_signature = hierarchy.get("ancestor_signature")
    record = {
        "provider": provider,
        "selector": selector,
        "runtime_id": _bounded_optional(locator.get("runtime_id"), 512),
        "automation_id": _bounded_optional(
            locator.get("automation_id") or element.get("automation_id"), 160
        ),
        "control_type": _bounded_optional(
            element.get("control_type") or element.get("role"), 64
        ),
        "class": _bounded_optional(element.get("class"), MAX_CLASS_NAME_CHARS),
        "ancestor_signature": _bounded_optional(ancestor_signature, 96),
        "bounds": normalized_bounds,
        "enabled": _strict_bool(element.get("enabled")),
        "read_only": _strict_bool(element.get("read_only")),
        "password": _strict_bool(element.get("password")),
    }
    record_ancestor = record.get("ancestor_signature")
    if record["provider"] == "winapp" and (
        any(
            not isinstance(record.get(field), str) or not record.get(field)
            for field in ("automation_id", "control_type", "class")
        )
        or not isinstance(record_ancestor, str)
        or ANCESTOR_SIGNATURE_PATTERN.fullmatch(record_ancestor) is None
    ):
        raise ComputerError(
            "element_ref_unavailable",
            "The semantic element has incomplete stable UI Automation identity.",
        )
    return record


def _window_mismatches(
    recorded: dict[str, Any],
    actual: WindowIdentity,
    *,
    require_title_match: bool,
) -> list[str]:
    comparisons = (
        ("hwnd", recorded.get("hwnd"), actual.hwnd),
        ("pid", recorded.get("pid"), actual.pid),
        ("thread_id", recorded.get("thread_id"), actual.thread_id),
        (
            "process_name",
            str(recorded.get("process") or "").casefold(),
            str(actual.process or "").casefold(),
        ),
        (
            "process_start_identity",
            recorded.get("process_started"),
            actual.process_started,
        ),
        ("window_class", recorded.get("class"), actual.class_name),
        ("session_id", recorded.get("session_id"), actual.session_id),
        (
            "input_desktop",
            str(recorded.get("desktop") or "").casefold(),
            str(actual.desktop_name or "").casefold(),
        ),
    )
    output = [name for name, before, after in comparisons if before != after]
    if require_title_match and recorded.get("title") != actual.title:
        output.append("title")
    return output


def _validated_reference(value: str) -> str:
    if not isinstance(value, str) or ELEMENT_REFERENCE_PATTERN.fullmatch(value) is None:
        raise ComputerError(
            "invalid_element_ref",
            "The element reference is invalid.",
            exit_code=2,
            details={"resolution_stage": "reference_parse"},
        )
    return value


def _default_store_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise _store_error()
    return Path(local) / "AgentTools" / "computer" / "element-refs-v1"


def _strict_bool(value: object) -> bool | None:
    return value if type(value) is bool else None


def _bounded_optional(value: object, limit: int) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    return text if len(text) <= limit else None


def _timestamp(unix_ms: int) -> str:
    return datetime.fromtimestamp(unix_ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _invalid_reference() -> ComputerError:
    return ComputerError(
        "element_ref_invalid",
        "The element reference is invalid; inspect the window again.",
        details={"resolution_stage": "reference_validation"},
    )


def _store_error() -> ComputerError:
    return ComputerError(
        "element_ref_store_unavailable",
        "The owner-private element reference store is unavailable.",
    )
