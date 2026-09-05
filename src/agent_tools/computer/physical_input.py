from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from ctypes import wintypes
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from agent_tools.computer.capture_records import (
    KEY_BYTES,
    CaptureRecordStore,
    _atomic_private_write,
    _canonical_json,
    _capture_store_lock,
    _create_integrity_key_exclusive,
    _prepare_capture_store_root,
    _reject_reparse_point,
    _set_owner_only_acl,
)
from agent_tools.computer.models import ComputerError, WindowIdentity
from agent_tools.computer.screenshot import resolve_capture_point
from agent_tools.computer.win32_backend import Win32Backend

PHYSICAL_CAPTURE_MAX_AGE_SECONDS = 15
MAX_PHYSICAL_TEXT_CHARS = 16_000
MAX_WHEEL_STEPS = 120
STAGED_INPUT_SCHEMA = "agent-tools.staged-input/v1"
STAGED_INPUT_TTL_SECONDS = 120
MAX_STAGED_INPUT_RECORDS = 64
MAX_STAGED_INPUT_BYTES = 48 * 1024
STAGED_INPUT_PATTERN = re.compile(r"stage_[0-9a-f]{32}\Z")

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
WHEEL_DELTA = 120
GA_ROOT = 2

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_RETURN = 0x0D
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_MBUTTON = 0x04
VK_XBUTTON1 = 0x05
VK_XBUTTON2 = 0x06
VK_LWIN = 0x5B
VK_RWIN = 0x5C

_HELD_INPUT_KEYS = (
    VK_LBUTTON,
    VK_RBUTTON,
    VK_MBUTTON,
    VK_XBUTTON1,
    VK_XBUTTON2,
    VK_SHIFT,
    VK_CONTROL,
    VK_MENU,
    VK_LWIN,
    VK_RWIN,
)
_SHORTCUT_MODIFIERS = {"CTRL": VK_CONTROL, "SHIFT": VK_SHIFT, "ALT": VK_MENU}
_NAMED_KEYS = {
    "ENTER": VK_RETURN,
    "TAB": 0x09,
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "BACKSPACE": 0x08,
    "DELETE": 0x2E,
    "HOME": 0x24,
    "END": 0x23,
    "LEFT": 0x25,
    "UP": 0x26,
    "RIGHT": 0x27,
    "DOWN": 0x28,
    "PAGEUP": 0x21,
    "PAGEDOWN": 0x22,
}
_FORBIDDEN_SHORTCUTS = {
    (frozenset({"ALT"}), "F4"),
    (frozenset({"CTRL", "ALT"}), "DELETE"),
    (frozenset({"CTRL", "SHIFT"}), "ESC"),
    (frozenset({"CTRL", "SHIFT"}), "ESCAPE"),
}


class StageClock(Protocol):
    def now_unix_ms(self) -> int: ...


class _SystemStageClock:
    def now_unix_ms(self) -> int:
        return int(time.time() * 1000)


@dataclass(frozen=True)
class PhysicalCaptureTarget:
    capture_id: str
    identity: WindowIdentity
    record: dict[str, Any]
    image_point: dict[str, int] | None
    screen_point: dict[str, int] | None
    provenance: dict[str, Any]


class StagedInputStore:
    """Owner-private, signed, one-shot authority for a later Enter action."""

    def __init__(self, root: Path | None = None, *, clock: StageClock | None = None) -> None:
        self.root = root or _default_stage_root()
        self._clock = clock or _SystemStageClock()

    def issue(
        self,
        *,
        identity: WindowIdentity,
        capture: PhysicalCaptureTarget,
        input_generation: int,
        text: str,
        terminal_targeting: dict[str, Any],
    ) -> dict[str, Any]:
        now_ms = self.now_unix_ms()
        expires_ms = now_ms + STAGED_INPUT_TTL_SECONDS * 1000
        stage_id = f"stage_{secrets.token_hex(16)}"
        payload = {
            "schema_version": STAGED_INPUT_SCHEMA,
            "staged_input": stage_id,
            "created_unix_ms": now_ms,
            "expires_unix_ms": expires_ms,
            "window": _window_record(identity),
            "capture": {
                "capture_id": capture.capture_id,
                "geometry": capture.record.get("geometry"),
                "window_identity": capture.record.get("window_identity"),
            },
            "input_generation": input_generation,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_char_count": len(text),
            "terminal_targeting": terminal_targeting,
        }
        self._write(payload)
        return {
            "staged_input": stage_id,
            "expires_at": _timestamp(expires_ms),
            "ttl_seconds": STAGED_INPUT_TTL_SECONDS,
            "text_sha256": payload["text_sha256"],
            "text_char_count": len(text),
            "input_generation": input_generation,
        }

    def consume_for_enter(
        self,
        stage_id: str,
        *,
        identity: WindowIdentity,
        current_input_generation: int,
        evidence_capture: PhysicalCaptureTarget,
        draft_evidence: str,
    ) -> dict[str, Any]:
        validated = _validated_stage_id(stage_id)
        with self._locked():
            key = self._ensure_ready_locked()
            path = self.root / f"{validated}.json"
            payload = self._read_locked(path, key)
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise ComputerError(
                    "staged_input_invalidation_failed",
                    "The staged input could not be invalidated safely.",
                ) from exc
        now_ms = self.now_unix_ms()
        created = payload.get("created_unix_ms")
        expires = payload.get("expires_unix_ms")
        if (
            type(created) is not int
            or type(expires) is not int
            or not created <= now_ms < expires
        ):
            raise ComputerError(
                "staged_input_expired",
                "The staged input expired; stage the complete draft again.",
            )
        _validate_stage_window(payload.get("window"), identity)
        stored_generation = payload.get("input_generation")
        if (
            type(stored_generation) is not int
            or current_input_generation != stored_generation + 1
        ):
            raise ComputerError(
                "staged_input_generation_changed",
                "Another computer mutation occurred after staging; stage the draft again.",
            )
        capture = payload.get("capture")
        if not isinstance(capture, dict):
            raise _invalid_stage()
        if capture.get("geometry") != evidence_capture.record.get("geometry"):
            raise ComputerError(
                "staged_input_geometry_changed",
                "The terminal geometry changed after staging; stage the draft again.",
            )
        if draft_evidence not in {"exact", "collapsed_visible_match"}:
            raise ComputerError(
                "invalid_draft_evidence",
                "Draft evidence must be exact or collapsed_visible_match.",
                exit_code=2,
            )
        return {
            "staged_input": validated,
            "text_sha256": payload.get("text_sha256"),
            "text_char_count": payload.get("text_char_count"),
            "stage_input_generation": stored_generation,
            "draft_evidence": draft_evidence,
            "evidence_capture_id": evidence_capture.capture_id,
            "terminal_targeting": payload.get("terminal_targeting"),
        }

    def now_unix_ms(self) -> int:
        value = self._clock.now_unix_ms()
        if type(value) is not int or value < 0:
            raise ComputerError(
                "staged_input_clock_invalid",
                "The staged input clock returned an invalid value.",
            )
        return value

    def _write(self, payload: dict[str, Any]) -> None:
        with self._locked():
            key = self._ensure_ready_locked()
            self._prune_locked()
            encoded = _signed_payload(payload, key)
            if len(encoded) > MAX_STAGED_INPUT_BYTES:
                raise ComputerError(
                    "staged_input_too_large",
                    "The staged input record exceeded its safety limit.",
                )
            _atomic_private_write(
                self.root / f"{payload['staged_input']}.json",
                encoded,
            )

    def _read_locked(self, path: Path, key: bytes) -> dict[str, Any]:
        if not path.exists():
            raise ComputerError(
                "staged_input_missing",
                "The staged input is missing or already consumed.",
            )
        _reject_reparse_point(path, "staged_input_invalid")
        try:
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_STAGED_INPUT_BYTES:
                raise _invalid_stage()
            envelope = json.loads(path.read_text(encoding="ascii"))
        except ComputerError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _invalid_stage() from exc
        if not isinstance(envelope, dict) or set(envelope) != {"payload", "hmac_sha256"}:
            raise _invalid_stage()
        payload = envelope.get("payload")
        signature = envelope.get("hmac_sha256")
        if not isinstance(payload, dict) or not isinstance(signature, str):
            raise _invalid_stage()
        expected = hmac.new(key, _canonical_json(payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ComputerError(
                "staged_input_altered",
                "The staged input record was altered.",
            )
        if (
            payload.get("schema_version") != STAGED_INPUT_SCHEMA
            or payload.get("staged_input") != path.stem
        ):
            raise _invalid_stage()
        return payload

    def _ensure_ready_locked(self) -> bytes:
        _prepare_capture_store_root(self.root)
        key_path = self.root / ".integrity-key"
        _create_integrity_key_exclusive(key_path)
        _reject_reparse_point(key_path, "staged_input_store_unavailable")
        try:
            metadata = key_path.stat()
            key = key_path.read_bytes()
        except OSError as exc:
            raise ComputerError(
                "staged_input_store_unavailable",
                "The staged input store is unavailable.",
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or len(key) != KEY_BYTES:
            raise ComputerError(
                "staged_input_store_unavailable",
                "The staged input integrity key is invalid.",
            )
        _set_owner_only_acl(key_path, directory=False)
        return key

    def _prune_locked(self) -> None:
        rows: list[tuple[float, Path]] = []
        try:
            for entry in os.scandir(self.root):
                if entry.name.startswith("stage_") and entry.name.endswith(".json"):
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISREG(metadata.st_mode):
                        rows.append((metadata.st_mtime, Path(entry.path)))
        except OSError as exc:
            raise ComputerError(
                "staged_input_store_unavailable",
                "The staged input store could not be inspected.",
            ) from exc
        rows.sort()
        while len(rows) >= MAX_STAGED_INPUT_RECORDS:
            _mtime, path = rows.pop(0)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                rows.append((_mtime, path))
                break
        if len(rows) >= MAX_STAGED_INPUT_RECORDS:
            raise ComputerError(
                "staged_input_store_capacity_unavailable",
                "The staged input store could not enforce its capacity.",
            )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if sys.platform != "win32":
            raise ComputerError(
                "unsupported_platform",
                "Physical input requires Windows.",
                exit_code=2,
            )
        try:
            with _capture_store_lock(self.root):
                yield
        except ComputerError as exc:
            if exc.code.startswith("staged_input_"):
                raise
            raise ComputerError(
                "staged_input_store_unavailable",
                "The staged input store is unavailable.",
            ) from exc


class Win32PhysicalInput:
    def __init__(
        self,
        *,
        event_sender: Callable[[list[dict[str, int]]], int] | None = None,
    ) -> None:
        self._event_sender = event_sender

    def assert_no_held_input(self) -> None:
        user32 = _user32()
        held = [key for key in _HELD_INPUT_KEYS if int(user32.GetAsyncKeyState(key)) & 0x8000]
        if held:
            raise ComputerError(
                "physical_input_user_state_held",
                "A user modifier or pointer button is held; physical input was rejected.",
                details={"held_input_count": len(held), "outcome": "rejected"},
            )

    def assert_foreground_identity(
        self,
        backend: Win32Backend,
        identity: WindowIdentity,
    ) -> None:
        modules = backend.modules()
        foreground = int(modules.win32gui.GetForegroundWindow() or 0)
        if foreground != identity.hwnd:
            raise ComputerError(
                "physical_input_focus_changed",
                "The foreground window changed during physical input.",
                details={"outcome": "rejected", "foreground_hwnd": foreground},
            )
        backend.revalidate_identity(identity)

    def current_cursor(self, backend: Win32Backend) -> tuple[int, int]:
        x, y = backend.modules().win32api.GetCursorPos()
        return int(x), int(y)

    def validate_text_mapping(
        self,
        identity: WindowIdentity,
        text: str,
    ) -> None:
        self._map_text(identity, text)

    def move_pointer(
        self,
        backend: Win32Backend,
        identity: WindowIdentity,
        x: int,
        y: int,
    ) -> dict[str, Any]:
        self.assert_foreground_identity(backend, identity)
        event = _absolute_move_event(backend, x, y)
        delivered = self._send([event])
        if delivered != 1:
            raise _partial_delivery_error("win32.SendInput.mouse_move", 1, delivered)
        observed = self.current_cursor(backend)
        if observed != (x, y):
            raise ComputerError(
                "physical_pointer_postcondition_failed",
                "The pointer move was delivered but the final position did not match.",
                details={
                    "outcome": "delivery_only",
                    "method": "win32.SendInput.mouse_move",
                    "requested_cursor": {"x": x, "y": y},
                    "actual_cursor": {"x": observed[0], "y": observed[1]},
                },
            )
        return {
            "method": "win32.SendInput.mouse_move",
            "outcome": "postcondition_verified",
            "postcondition_verified": True,
            "cursor": {"x": x, "y": y},
            "events_intended": 1,
            "events_delivered": 1,
        }

    def click(
        self,
        backend: Win32Backend,
        identity: WindowIdentity,
        x: int,
        y: int,
        *,
        double: bool,
    ) -> dict[str, Any]:
        original = self.current_cursor(backend)
        events = [_absolute_move_event(backend, x, y)]
        click_count = 2 if double else 1
        for _index in range(click_count):
            events.extend(
                [
                    {"type": INPUT_MOUSE, "flags": MOUSEEVENTF_LEFTDOWN, "data": 0},
                    {"type": INPUT_MOUSE, "flags": MOUSEEVENTF_LEFTUP, "data": 0},
                ]
            )

        def deliver() -> dict[str, Any]:
            self.assert_foreground_identity(backend, identity)
            delivered = self._send(events)
            if delivered != len(events):
                self._release_delivered_mouse_buttons(events[:delivered])
                raise _partial_delivery_error("win32.SendInput.mouse_click", len(events), delivered)
            self.assert_foreground_identity(backend, identity)
            return {
                "method": (
                    "win32.SendInput.mouse_double_click"
                    if double
                    else "win32.SendInput.mouse_click"
                ),
                "outcome": "delivery_only",
                "postcondition_verified": False,
                "verification_required": True,
                "events_intended": len(events),
                "events_delivered": delivered,
            }

        return self._with_restored_cursor(backend, original, deliver)

    def type_text(
        self,
        backend: Win32Backend,
        identity: WindowIdentity,
        x: int,
        y: int,
        text: str,
        *,
        mutation_guard: Callable[[], None],
    ) -> dict[str, Any]:
        mapped, layout = self._map_text(identity, text)
        original = self.current_cursor(backend)
        clicked = False

        def deliver() -> dict[str, Any]:
            nonlocal clicked
            self.assert_foreground_identity(backend, identity)
            pointer_events = [
                _absolute_move_event(backend, x, y),
                {"type": INPUT_MOUSE, "flags": MOUSEEVENTF_LEFTDOWN, "data": 0},
                {"type": INPUT_MOUSE, "flags": MOUSEEVENTF_LEFTUP, "data": 0},
            ]
            delivered_pointer = self._send(pointer_events)
            if delivered_pointer != len(pointer_events):
                self._release_delivered_mouse_buttons(pointer_events[:delivered_pointer])
                raise _partial_delivery_error(
                    "win32.SendInput.stage_focus_click",
                    len(pointer_events),
                    delivered_pointer,
                )
            clicked = True
            delivered_chars = 0
            for index, (virtual_key, shifted) in enumerate(mapped):
                try:
                    mutation_guard()
                    self.assert_foreground_identity(backend, identity)
                    self.assert_no_held_input()
                    if int(_user32().GetKeyboardLayout(identity.thread_id) or 0) != layout:
                        raise ComputerError(
                            "keyboard_layout_changed",
                            "The target keyboard layout changed during text staging.",
                        )
                except Exception as raw_exc:
                    reason = (
                        raw_exc.code
                        if isinstance(raw_exc, ComputerError)
                        else "physical_text_guard_failed"
                    )
                    raise _partial_text_error(
                        text,
                        delivered_chars,
                        index,
                        reason,
                        focus_click_delivered=clicked,
                    ) from raw_exc
                events = _character_events(virtual_key, shifted)
                delivered_events = self._send(events)
                if delivered_events != len(events):
                    self._release_delivered_keys(events[:delivered_events])
                    raise _partial_text_error(
                        text,
                        delivered_chars,
                        index,
                        "send_input_partial",
                        focus_click_delivered=clicked,
                    )
                delivered_chars += 1
                try:
                    self.assert_foreground_identity(backend, identity)
                except Exception as raw_exc:
                    reason = (
                        raw_exc.code
                        if isinstance(raw_exc, ComputerError)
                        else "physical_text_focus_verification_failed"
                    )
                    raise _partial_text_error(
                        text,
                        delivered_chars,
                        delivered_chars,
                        reason,
                        focus_click_delivered=clicked,
                    ) from raw_exc
            return {
                "method": "win32.SendInput.VkKeyScanW",
                "outcome": "delivery_only",
                "postcondition_verified": False,
                "verification_required": True,
                "intended_character_count": len(text),
                "delivered_character_count": delivered_chars,
                "first_undelivered_position": None,
                "partial_delivery": False,
                "focus_click_delivered": clicked,
            }

        return self._with_restored_cursor(backend, original, deliver)

    def press_enter(
        self,
        backend: Win32Backend,
        identity: WindowIdentity,
    ) -> dict[str, Any]:
        self.assert_foreground_identity(backend, identity)
        events = _key_events(VK_RETURN)
        delivered = self._send(events)
        if delivered != len(events):
            self._release_delivered_keys(events[:delivered])
            raise _partial_delivery_error("win32.SendInput.key_enter", len(events), delivered)
        self.assert_foreground_identity(backend, identity)
        return {
            "method": "win32.SendInput.key_enter",
            "outcome": "delivery_only",
            "postcondition_verified": False,
            "verification_required": True,
            "events_intended": len(events),
            "events_delivered": delivered,
        }

    def shortcut(
        self,
        backend: Win32Backend,
        identity: WindowIdentity,
        keys: str,
    ) -> dict[str, Any]:
        modifiers, virtual_key, normalized = parse_shortcut(keys)
        self.assert_foreground_identity(backend, identity)
        events: list[dict[str, int]] = []
        for modifier in modifiers:
            events.append(_keyboard_event(_SHORTCUT_MODIFIERS[modifier], key_up=False))
        events.extend(_key_events(virtual_key))
        for modifier in reversed(modifiers):
            events.append(_keyboard_event(_SHORTCUT_MODIFIERS[modifier], key_up=True))
        delivered = self._send(events)
        if delivered != len(events):
            self._release_delivered_keys(events[:delivered])
            raise _partial_delivery_error("win32.SendInput.shortcut", len(events), delivered)
        self.assert_foreground_identity(backend, identity)
        return {
            "method": "win32.SendInput.shortcut",
            "outcome": "delivery_only",
            "postcondition_verified": False,
            "verification_required": True,
            "shortcut": normalized,
            "events_intended": len(events),
            "events_delivered": delivered,
        }

    def wheel(
        self,
        backend: Win32Backend,
        identity: WindowIdentity,
        x: int,
        y: int,
        *,
        vertical: int | None,
        horizontal: int | None,
    ) -> dict[str, Any]:
        count = validate_wheel_request(vertical=vertical, horizontal=horizontal)
        original = self.current_cursor(backend)
        flags = MOUSEEVENTF_WHEEL if vertical is not None else MOUSEEVENTF_HWHEEL
        events = [
            _absolute_move_event(backend, x, y),
            {"type": INPUT_MOUSE, "flags": flags, "data": int(count * WHEEL_DELTA)},
        ]

        def deliver() -> dict[str, Any]:
            self.assert_foreground_identity(backend, identity)
            delivered = self._send(events)
            if delivered != len(events):
                raise _partial_delivery_error("win32.SendInput.wheel", len(events), delivered)
            self.assert_foreground_identity(backend, identity)
            return {
                "method": "win32.SendInput.wheel",
                "outcome": "delivery_only",
                "postcondition_verified": False,
                "verification_required": True,
                "axis": "vertical" if vertical is not None else "horizontal",
                "steps": count,
                "events_intended": len(events),
                "events_delivered": delivered,
            }

        return self._with_restored_cursor(backend, original, deliver)

    def _map_text(
        self,
        identity: WindowIdentity,
        text: str,
    ) -> tuple[list[tuple[int, bool]], int]:
        if not text or len(text) > MAX_PHYSICAL_TEXT_CHARS:
            raise ComputerError(
                "invalid_physical_text",
                f"Physical text must contain 1 through {MAX_PHYSICAL_TEXT_CHARS} characters.",
                exit_code=2,
            )
        if any(character in "\r\n" or ord(character) < 32 for character in text):
            raise ComputerError(
                "physical_text_control_rejected",
                "Staged physical text cannot contain Enter, newline, or control characters.",
                exit_code=2,
            )
        user32 = _user32()
        user32.GetKeyboardLayout.argtypes = (wintypes.DWORD,)
        user32.GetKeyboardLayout.restype = wintypes.HKL
        user32.VkKeyScanW.argtypes = (wintypes.WCHAR,)
        user32.VkKeyScanW.restype = ctypes.c_short
        current_layout = int(user32.GetKeyboardLayout(0) or 0)
        target_layout = int(user32.GetKeyboardLayout(identity.thread_id) or 0)
        if not current_layout or current_layout != target_layout:
            raise ComputerError(
                "keyboard_layout_mismatch",
                "The target active keyboard layout differs from the input thread layout.",
            )
        mapped: list[tuple[int, bool]] = []
        for index, character in enumerate(text):
            result = int(user32.VkKeyScanW(character))
            if result == -1 or result & 0xFFFF == 0xFFFF:
                raise ComputerError(
                    "physical_character_unmappable",
                    "A character cannot be mapped by the active keyboard layout.",
                    details={"first_undelivered_position": index, "outcome": "rejected"},
                )
            virtual_key = result & 0xFF
            modifiers = (result >> 8) & 0xFF
            if modifiers & ~0x01:
                raise ComputerError(
                    "physical_character_requires_unsafe_modifier",
                    "A character requires Ctrl or Alt and cannot be staged safely.",
                    details={"first_undelivered_position": index, "outcome": "rejected"},
                )
            mapped.append((virtual_key, bool(modifiers & 0x01)))
        return mapped, target_layout

    def _restore_cursor(self, backend: Win32Backend, original: tuple[int, int]) -> None:
        event = _absolute_move_event(backend, original[0], original[1])
        delivered = self._send([event])
        if delivered != 1 or self.current_cursor(backend) != original:
            raise ComputerError(
                "physical_cursor_restore_failed",
                "The original cursor position could not be restored safely.",
                details={
                    "outcome": "delivery_only",
                    "method": "win32.SendInput.mouse_move",
                    "cursor_restored": False,
                },
            )

    def _with_restored_cursor(
        self,
        backend: Win32Backend,
        original: tuple[int, int],
        deliver: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            result = deliver()
        except ComputerError as primary_error:
            try:
                self._restore_cursor(backend, original)
            except Exception as raw_restore_error:  # noqa: BLE001 - preserve primary truth
                restore_error = _cursor_restore_error(raw_restore_error)
                raise _combined_cursor_restore_error(
                    restore_error=restore_error,
                    primary_error=primary_error,
                ) from primary_error
            raise
        except Exception:
            self._restore_cursor(backend, original)
            raise
        try:
            self._restore_cursor(backend, original)
        except Exception as raw_restore_error:  # noqa: BLE001 - preserve delivery truth
            restore_error = _cursor_restore_error(raw_restore_error)
            raise _combined_cursor_restore_error(
                restore_error=restore_error,
                primary_result=result,
            ) from restore_error
        return result

    def _release_delivered_keys(self, delivered: list[dict[str, int]]) -> None:
        down: list[int] = []
        for event in delivered:
            if event.get("type") != INPUT_KEYBOARD:
                continue
            vk = int(event.get("vk", 0))
            if event.get("flags", 0) & KEYEVENTF_KEYUP:
                if vk in down:
                    down.remove(vk)
            else:
                down.append(vk)
        if down:
            cleanup = [_keyboard_event(vk, key_up=True) for vk in reversed(down)]
            if self._send(cleanup) != len(cleanup):
                raise ComputerError(
                    "physical_input_cleanup_failed",
                    "Injected keyboard state could not be released safely.",
                    details={
                        "outcome": "delivery_only",
                        "method": "win32.SendInput.keyboard_cleanup",
                        "partial_delivery": True,
                    },
                )

    def _release_delivered_mouse_buttons(
        self,
        delivered: list[dict[str, int]],
    ) -> None:
        left_down = False
        for event in delivered:
            if event.get("type") != INPUT_MOUSE:
                continue
            flags = int(event.get("flags", 0))
            if flags & MOUSEEVENTF_LEFTDOWN:
                left_down = True
            if flags & MOUSEEVENTF_LEFTUP:
                left_down = False
        if left_down:
            cleanup = [
                {"type": INPUT_MOUSE, "flags": MOUSEEVENTF_LEFTUP, "data": 0}
            ]
            if self._send(cleanup) != len(cleanup):
                raise ComputerError(
                    "physical_input_cleanup_failed",
                    "Injected pointer-button state could not be released safely.",
                    details={
                        "outcome": "delivery_only",
                        "method": "win32.SendInput.pointer_cleanup",
                        "partial_delivery": True,
                    },
                )

    def _send(self, events: list[dict[str, int]]) -> int:
        if not events:
            return 0
        if self._event_sender is not None:
            return int(self._event_sender(events))
        return _send_input(events)


def resolve_physical_capture(
    *,
    hwnd: int,
    capture_id: str,
    image_x: int | None,
    image_y: int | None,
    backend: Win32Backend,
    record_store: CaptureRecordStore,
    require_point: bool,
) -> PhysicalCaptureTarget:
    if require_point and (image_x is None or image_y is None):
        raise ComputerError(
            "physical_image_point_required",
            "This physical action requires a capture image point.",
            exit_code=2,
        )
    record = record_store.load(capture_id)
    created = record.get("created_unix_ms")
    now = record_store.now_unix_ms()
    if type(created) is not int or now - created > PHYSICAL_CAPTURE_MAX_AGE_SECONDS * 1000:
        raise ComputerError(
            "physical_capture_too_old",
            "Physical input requires a screenshot no older than 15 seconds.",
        )
    window_reference = record.get("window_identity")
    if not isinstance(window_reference, dict) or window_reference.get("hwnd") != hwnd:
        raise ComputerError(
            "physical_capture_target_mismatch",
            "The capture belongs to a different target window.",
        )
    identity = backend.capture_identity(hwnd)
    backend.assert_action_allowed(identity)
    record_store.validate_identity_reference(window_reference, identity)
    point_result: dict[str, Any] | None = None
    if image_x is not None and image_y is not None:
        point_result = resolve_capture_point(
            capture_id=capture_id,
            image_x=image_x,
            image_y=image_y,
            backend=backend,
            record_store=record_store,
        )
        screen = point_result["native_screen_point"]
        _assert_unoccluded_target(backend, identity.hwnd, int(screen["x"]), int(screen["y"]))
    presentation_value = record.get("presentation")
    presentation: dict[str, Any] = (
        presentation_value if isinstance(presentation_value, dict) else {}
    )
    geometry_value = record.get("geometry")
    geometry: dict[str, Any] = geometry_value if isinstance(geometry_value, dict) else {}
    provenance = {
        "capture_id": capture_id,
        "image_sha256": presentation.get("sha256"),
        "profile": presentation.get("profile"),
        "capture_age_ms": max(0, now - created),
        "capture_ttl_remaining_ms": max(
            0,
            int(record.get("expires_unix_ms") or now) - now,
        ),
        "geometry_fingerprint": hashlib.sha256(
            json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return PhysicalCaptureTarget(
        capture_id=capture_id,
        identity=identity,
        record=record,
        image_point=(
            {"x": int(image_x), "y": int(image_y)}
            if image_x is not None and image_y is not None
            else None
        ),
        screen_point=(
            {
                "x": int(point_result["native_screen_point"]["x"]),
                "y": int(point_result["native_screen_point"]["y"]),
            }
            if point_result is not None
            else None
        ),
        provenance=provenance,
    )


def terminal_targeting_contract(
    identity: WindowIdentity,
    *,
    visible_tab_verified: bool,
) -> dict[str, Any]:
    is_terminal = (
        (identity.process or "").casefold() in {"windowsterminal.exe", "wt.exe"}
        or identity.class_name.casefold() == "cascadia_hosting_window_class"
    )
    if is_terminal and not visible_tab_verified:
        raise ComputerError(
            "terminal_tab_verification_required",
            "Windows Terminal tab identity is not reliable from HWND alone; "
            "verify visible tab content.",
            details={
                "outcome": "rejected",
                "terminal_targeting": {
                    "status": "limited",
                    "reason": "multi_tab_same_hwnd_not_semantically_bound",
                    "visible_tab_verification_required": True,
                },
            },
        )
    return {
        "status": "limited" if is_terminal else "not_applicable",
        "reason": "multi_tab_same_hwnd_not_semantically_bound" if is_terminal else None,
        "visible_tab_verification_required": is_terminal,
        "caller_visible_tab_verified": bool(is_terminal and visible_tab_verified),
    }


def parse_shortcut(value: str) -> tuple[list[str], int, str]:
    parts = [part.strip().upper() for part in value.split("+") if part.strip()]
    modifiers = [part for part in parts if part in _SHORTCUT_MODIFIERS]
    non_modifiers = [part for part in parts if part not in _SHORTCUT_MODIFIERS]
    if len(non_modifiers) != 1 or len(modifiers) != len(set(modifiers)):
        raise ComputerError(
            "invalid_shortcut",
            "A shortcut must contain unique Ctrl, Shift, or Alt modifiers and one key.",
            exit_code=2,
        )
    key_name = non_modifiers[0]
    modifier_set = frozenset(modifiers)
    if key_name == "ENTER":
        raise ComputerError(
            "forbidden_shortcut",
            "Enter must use the guarded staged-input key transaction.",
            exit_code=2,
        )
    if "WIN" in parts or (modifier_set, key_name) in _FORBIDDEN_SHORTCUTS:
        raise ComputerError(
            "forbidden_shortcut",
            "The requested system-global shortcut is forbidden.",
            exit_code=2,
        )
    virtual_key = _shortcut_virtual_key(key_name)
    ordered: list[str] = [
        name for name in ("CTRL", "SHIFT", "ALT") if name in modifier_set
    ]
    return ordered, virtual_key, "+".join([*ordered, key_name])


def validate_wheel_request(*, vertical: int | None, horizontal: int | None) -> int:
    if (vertical is None) == (horizontal is None):
        raise ComputerError(
            "invalid_wheel_request",
            "Choose exactly one vertical or horizontal wheel count.",
            exit_code=2,
            details={"outcome": "rejected"},
        )
    count = vertical if vertical is not None else horizontal
    assert count is not None
    if count == 0 or abs(count) > MAX_WHEEL_STEPS:
        raise ComputerError(
            "invalid_wheel_count",
            f"Wheel count must be non-zero and at most {MAX_WHEEL_STEPS} steps.",
            exit_code=2,
            details={"outcome": "rejected"},
        )
    return count


def _shortcut_virtual_key(name: str) -> int:
    if name in _NAMED_KEYS:
        return _NAMED_KEYS[name]
    if len(name) == 1 and name.isascii() and name.isalnum():
        return ord(name.upper())
    if name.startswith("F") and name[1:].isdigit() and 1 <= int(name[1:]) <= 12:
        return 0x6F + int(name[1:])
    raise ComputerError(
        "invalid_shortcut_key",
        "The shortcut key is unsupported.",
        exit_code=2,
    )


def _assert_unoccluded_target(backend: Win32Backend, hwnd: int, x: int, y: int) -> None:
    modules = backend.modules()
    user32 = _user32()
    user32.WindowFromPoint.argtypes = (wintypes.POINT,)
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    user32.GetAncestor.restype = wintypes.HWND
    point_window = int(user32.WindowFromPoint(wintypes.POINT(x, y)) or 0)
    root = int(user32.GetAncestor(point_window, GA_ROOT) or 0) if point_window else 0
    if root != hwnd or not modules.win32gui.IsWindowVisible(hwnd):
        raise ComputerError(
            "physical_target_occluded",
            "The capture-derived target point is no longer visibly owned by the target window.",
        )


def _absolute_move_event(backend: Win32Backend, x: int, y: int) -> dict[str, int]:
    display = backend.display_info()
    virtual = display.get("virtual_bounds") if isinstance(display, dict) else None
    if not isinstance(virtual, dict):
        raise ComputerError(
            "physical_virtual_screen_unavailable",
            "The virtual screen geometry is unavailable.",
        )
    vx, vy = int(virtual["x"]), int(virtual["y"])
    width, height = int(virtual["width"]), int(virtual["height"])
    if width <= 1 or height <= 1 or not (vx <= x < vx + width and vy <= y < vy + height):
        raise ComputerError(
            "physical_point_offscreen",
            "The capture-derived point is outside the virtual screen.",
        )
    absolute_x = round((x - vx) * 65535 / (width - 1))
    absolute_y = round((y - vy) * 65535 / (height - 1))
    return {
        "type": INPUT_MOUSE,
        "flags": MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        "dx": absolute_x,
        "dy": absolute_y,
        "data": 0,
    }


def _character_events(virtual_key: int, shifted: bool) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    if shifted:
        events.append(_keyboard_event(VK_SHIFT, key_up=False))
    events.extend(_key_events(virtual_key))
    if shifted:
        events.append(_keyboard_event(VK_SHIFT, key_up=True))
    return events


def _key_events(virtual_key: int) -> list[dict[str, int]]:
    return [
        _keyboard_event(virtual_key, key_up=False),
        _keyboard_event(virtual_key, key_up=True),
    ]


def _keyboard_event(virtual_key: int, *, key_up: bool) -> dict[str, int]:
    return {
        "type": INPUT_KEYBOARD,
        "vk": virtual_key,
        "flags": KEYEVENTF_KEYUP if key_up else 0,
    }


def _send_input(events: list[dict[str, int]]) -> int:
    user32 = _user32()
    input_array = (_INPUT * len(events))()
    for index, event in enumerate(events):
        if event["type"] == INPUT_MOUSE:
            input_array[index].type = INPUT_MOUSE
            input_array[index].union.mi = _MOUSEINPUT(
                dx=int(event.get("dx", 0)),
                dy=int(event.get("dy", 0)),
                mouseData=int(event.get("data", 0)) & 0xFFFFFFFF,
                dwFlags=int(event.get("flags", 0)),
                time=0,
                dwExtraInfo=0,
            )
        else:
            input_array[index].type = INPUT_KEYBOARD
            input_array[index].union.ki = _KEYBDINPUT(
                wVk=int(event.get("vk", 0)),
                wScan=0,
                dwFlags=int(event.get("flags", 0)),
                time=0,
                dwExtraInfo=0,
            )
    user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
    user32.SendInput.restype = wintypes.UINT
    return int(user32.SendInput(len(events), input_array, ctypes.sizeof(_INPUT)))


def _partial_delivery_error(method: str, intended: int, delivered: int) -> ComputerError:
    return ComputerError(
        "physical_input_partial",
        "Windows accepted only part of the physical input sequence.",
        details={
            "outcome": "delivery_only" if delivered else "failed",
            "method": method,
            "partial_delivery": delivered > 0,
            "events_intended": intended,
            "events_delivered": delivered,
        },
    )


def _cursor_restore_error(error: Exception) -> ComputerError:
    if isinstance(error, ComputerError):
        return error
    return ComputerError(
        "physical_cursor_restore_failed",
        "The original cursor position could not be restored safely.",
        details={
            "outcome": "delivery_only",
            "method": "win32.SendInput.mouse_move",
            "cursor_restored": False,
        },
    )


def _combined_cursor_restore_error(
    *,
    restore_error: ComputerError,
    primary_error: ComputerError | None = None,
    primary_result: dict[str, Any] | None = None,
) -> ComputerError:
    cleanup = {
        "status": "failed",
        "error_code": restore_error.code,
        "method": restore_error.details.get("method"),
    }
    if primary_error is not None:
        return ComputerError(
            primary_error.code,
            primary_error.message,
            exit_code=primary_error.exit_code,
            details={
                **primary_error.details,
                "cursor_restored": False,
                "cursor_cleanup": cleanup,
            },
        )
    if primary_result is None:
        raise AssertionError("Primary delivery evidence is required")
    return ComputerError(
        "physical_cursor_restore_failed",
        "Physical input was delivered, but the original cursor position was not restored.",
        details={
            **primary_result,
            "outcome": "delivery_only",
            "cursor_restored": False,
            "cursor_cleanup": cleanup,
        },
    )


def _partial_text_error(
    text: str,
    delivered: int,
    first_undelivered: int,
    reason: str,
    *,
    focus_click_delivered: bool = False,
) -> ComputerError:
    return ComputerError(
        "physical_text_partial",
        "Physical text delivery stopped before the complete draft was staged.",
        details={
            "outcome": "delivery_only" if delivered or focus_click_delivered else "failed",
            "method": "win32.SendInput.VkKeyScanW",
            "partial_delivery": delivered > 0 or focus_click_delivered,
            "intended_character_count": len(text),
            "delivered_character_count": delivered,
            "first_undelivered_position": first_undelivered,
            "focus_click_delivered": focus_click_delivered,
            "stage_valid": False,
            "automatic_cleanup_performed": False,
            "reason": reason,
        },
    )


def _window_record(identity: WindowIdentity) -> dict[str, Any]:
    if (
        identity.process is None
        or identity.process_started is None
        or identity.session_id is None
        or not identity.desktop_name
    ):
        raise ComputerError(
            "target_identity_unavailable",
            "The strong terminal identity is unavailable.",
        )
    return {
        "hwnd": identity.hwnd,
        "pid": identity.pid,
        "thread_id": identity.thread_id,
        "process": identity.process,
        "process_started": identity.process_started,
        "class": identity.class_name,
        "session_id": identity.session_id,
        "desktop": identity.desktop_name,
    }


def _validate_stage_window(record: object, identity: WindowIdentity) -> None:
    if not isinstance(record, dict):
        raise _invalid_stage()
    expected = _window_record(identity)
    mismatches = [key for key in expected if record.get(key) != expected[key]]
    if mismatches:
        raise ComputerError(
            "staged_input_target_changed",
            "The staged input target changed; stage the draft again.",
            details={"identity_mismatch_fields": mismatches[:8]},
        )


def _signed_payload(payload: dict[str, Any], key: bytes) -> bytes:
    return _canonical_json(
        {
            "payload": payload,
            "hmac_sha256": hmac.new(key, _canonical_json(payload), hashlib.sha256).hexdigest(),
        }
    )


def _validated_stage_id(value: str) -> str:
    if not isinstance(value, str) or STAGED_INPUT_PATTERN.fullmatch(value) is None:
        raise ComputerError(
            "invalid_staged_input",
            "The staged input ID is invalid.",
            exit_code=2,
        )
    return value


def _invalid_stage() -> ComputerError:
    return ComputerError("staged_input_invalid", "The staged input record is invalid.")


def _default_stage_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise ComputerError(
            "staged_input_store_unavailable",
            "LOCALAPPDATA is unavailable for staged input records.",
        )
    return Path(local) / "AgentTools" / "computer" / "staged-input-v1"


def _timestamp(unix_ms: int) -> str:
    return (
        datetime.fromtimestamp(unix_ms / 1000, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _user32() -> Any:
    if sys.platform != "win32":
        raise ComputerError(
            "unsupported_platform",
            "Physical input requires Windows.",
            exit_code=2,
        )
    return ctypes.WinDLL("user32", use_last_error=True)


_ULONG_PTR = wintypes.WPARAM


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT))


class _INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = (("type", wintypes.DWORD), ("union", _INPUTUNION))
