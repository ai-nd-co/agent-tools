from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_tools.computer.models import ComputerError, WindowIdentity

CONSEQUENCE_SETTLE_SECONDS = 2.0
MAX_TOPOLOGY_WINDOWS = 64
MAX_CHURN_ITEMS = 8

_SPINNER_PREFIX_RE = re.compile(
    r"^(?:[|/\\-]|[\u2800-\u28ff]|[\u25d0-\u25d3]|[\u25f4-\u25f7])(?:\s+|\s*[-:|]\s*)"
)
_PROGRESS_PREFIX_RE = re.compile(
    r"^(?:\[?\d{1,3}%\]?|\d{1,2}:\d{2}(?::\d{2})?)(?:\s+|\s*[-:|]\s*)"
)


@dataclass(frozen=True)
class ConsequenceSnapshot:
    public: dict[str, Any]
    raw_title: str | None
    target_identity: WindowIdentity | None
    topology: dict[str, tuple[int, int | None, int | None, str]]
    topology_truncated: bool


class ConsequenceObserver:
    """Capture and compare one bounded, privacy-safe mutation consequence."""

    def __init__(
        self,
        *,
        backend: Any,
        hwnd: int | None,
        settle_seconds: float | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        input_generation: int | None = None,
        input_generation_reader: Callable[[], int] | None = None,
        capture_provenance: dict[str, Any] | None = None,
        cursor_relevant: bool = False,
    ) -> None:
        self.backend = backend
        self.hwnd = hwnd
        self.settle_seconds = (
            _default_settle_seconds(backend)
            if settle_seconds is None
            else settle_seconds
        )
        self.sleeper = sleeper
        self.input_generation = input_generation
        self.input_generation_reader = input_generation_reader
        self.capture_provenance = _bounded_capture_provenance(capture_provenance)
        self.cursor_relevant = cursor_relevant
        self._key = secrets.token_bytes(32)

    def capture(self) -> ConsequenceSnapshot:
        target, target_error = self._target_identity()
        foreground_hwnd = self._foreground_hwnd()
        foreground_identity, foreground_error = self._foreground_identity(
            foreground_hwnd,
            target,
        )
        topology, topology_truncated = self._topology()
        target_public: dict[str, Any] = {
            "exists": target is not None,
            "observation_error": target_error,
        }
        raw_title: str | None = None
        if target is not None:
            raw_title = target.title
            target_public.update(
                {
                    "strong_identity": _public_identity(target, self._key),
                    "state": self._window_state(target.hwnd),
                    "bounds": target.bounds.to_jsonable(),
                    "monitor": self._monitor(target.hwnd),
                    "dpi": self._window_dpi(target.hwnd),
                    "foreground": foreground_hwnd == target.hwnd,
                    "z_order": _z_order_for(target.hwnd, topology),
                    "title_hmac_sha256": _fingerprint(
                        self._key, "title", target.title
                    ),
                }
            )
        foreground_public: dict[str, Any] = {
            "hwnd": foreground_hwnd,
            "observation_error": foreground_error,
        }
        active_monitor: dict[str, Any] | None = None
        if foreground_identity is not None:
            active_monitor = self._monitor(foreground_identity.hwnd)
            foreground_public.update(
                {
                    "strong_identity": _public_identity(foreground_identity, self._key),
                    "monitor": active_monitor,
                }
            )
        public: dict[str, Any] = {
            "target": target_public,
            "foreground": foreground_public,
            "active_monitor": active_monitor,
            "topology": {
                "count": len(topology),
                "truncated": topology_truncated,
                "digest": _topology_digest(self._key, topology),
            },
            "input_generation": self._input_generation(),
            "capture_provenance": self.capture_provenance,
        }
        if self.cursor_relevant:
            public["cursor"] = self._cursor()
        return ConsequenceSnapshot(
            public=public,
            raw_title=raw_title,
            target_identity=target,
            topology=topology,
            topology_truncated=topology_truncated,
        )

    def settle_and_compare(
        self,
        before: ConsequenceSnapshot,
        *,
        operation: str,
        expectation: dict[str, Any] | None,
        semantic_postcondition: dict[str, Any] | None = None,
        terminal_effect: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        settle_started = time.perf_counter()
        if self.settle_seconds > 0:
            self.sleeper(self.settle_seconds)
        settle_ms = _elapsed_ms(settle_started)
        capture_started = time.perf_counter()
        try:
            after = self.capture()
        except Exception as raw_exc:  # noqa: BLE001 - preserve delivery truth
            error = raw_exc if isinstance(raw_exc, ComputerError) else None
            return {
                "settle_window_ms": int(CONSEQUENCE_SETTLE_SECONDS * 1000),
                "observed_effect": "ambiguous",
                "expectation": "ambiguous_effect",
                "observation_error": error.code if error is not None else "snapshot_failed",
                "before": before.public,
                "after": None,
                "target_delta": {},
                "unrelated_window_churn": _empty_churn(),
                "semantic_postcondition": _bounded_semantic_postcondition(
                    semantic_postcondition
                ),
                "terminal_effect": _bounded_terminal_effect(terminal_effect),
                "phase_timings_ms": {
                    "settle_wait": settle_ms,
                    "after_capture": _elapsed_ms(capture_started),
                    "compare": 0.0,
                    "total": _elapsed_ms(total_started),
                },
            }
        capture_ms = _elapsed_ms(capture_started)
        compare_started = time.perf_counter()
        result = compare_snapshots(
            before,
            after,
            operation=operation,
            expectation=expectation,
            semantic_postcondition=semantic_postcondition,
            terminal_effect=terminal_effect,
        )
        result["phase_timings_ms"] = {
            "settle_wait": settle_ms,
            "after_capture": capture_ms,
            "compare": _elapsed_ms(compare_started),
            "total": _elapsed_ms(total_started),
        }
        return result

    def unavailable(self, *, reason: str) -> dict[str, Any]:
        return unavailable_consequence(reason)

    def _target_identity(self) -> tuple[WindowIdentity | None, str | None]:
        if self.hwnd is None:
            return None, "target_not_applicable"
        try:
            return self.backend.capture_identity(self.hwnd), None
        except ComputerError as exc:
            return None, exc.code
        except Exception:  # noqa: BLE001 - bounded observer must not mask delivery
            return None, "target_observation_failed"

    def _foreground_hwnd(self) -> int | None:
        try:
            modules = self.backend.modules()
            return int(modules.win32gui.GetForegroundWindow() or 0) or None
        except Exception:  # noqa: BLE001
            try:
                focused = self.backend.focused_window()
                window = focused.get("window") if isinstance(focused, dict) else None
                hwnd = window.get("hwnd") if isinstance(window, dict) else None
                return int(hwnd) if type(hwnd) is int and hwnd > 0 else None
            except Exception:  # noqa: BLE001
                return None

    def _foreground_identity(
        self,
        hwnd: int | None,
        target: WindowIdentity | None,
    ) -> tuple[WindowIdentity | None, str | None]:
        if hwnd is None:
            return None, "foreground_unavailable"
        if target is not None and hwnd == target.hwnd:
            return target, None
        try:
            return self.backend.capture_identity(hwnd), None
        except ComputerError as exc:
            return None, exc.code
        except Exception:  # noqa: BLE001
            return None, "foreground_identity_observation_failed"

    def _topology(self) -> tuple[dict[str, tuple[int, int | None, int | None, str]], bool]:
        try:
            modules = self.backend.modules()
        except Exception:  # noqa: BLE001
            return {}, False
        rows: list[int] = []

        def visit(hwnd: int, _extra: object) -> bool:
            try:
                if modules.win32gui.IsWindowVisible(hwnd):
                    rows.append(int(hwnd))
            except Exception:  # noqa: BLE001
                pass
            return len(rows) < MAX_TOPOLOGY_WINDOWS + 1

        try:
            modules.win32gui.EnumWindows(visit, None)
        except Exception:  # noqa: BLE001
            return {}, False
        truncated = len(rows) > MAX_TOPOLOGY_WINDOWS
        result: dict[str, tuple[int, int | None, int | None, str]] = {}
        for index, hwnd in enumerate(rows[:MAX_TOPOLOGY_WINDOWS]):
            try:
                _thread_id, pid = modules.win32process.GetWindowThreadProcessId(hwnd)
                class_name = str(modules.win32gui.GetClassName(hwnd))
                started = self.backend.process_started(int(pid))
                result[str(hwnd)] = (index, int(pid), started, class_name)
            except Exception:  # noqa: BLE001
                continue
        return result, truncated

    def _window_state(self, hwnd: int) -> str | None:
        try:
            return str(self.backend.window_state(hwnd))
        except Exception:  # noqa: BLE001
            return None

    def _window_dpi(self, hwnd: int) -> int | None:
        try:
            value = self.backend.window_dpi(hwnd)
            return int(value) if value is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _monitor(self, hwnd: int) -> dict[str, Any] | None:
        try:
            geometry = self.backend.capture_native_geometry(hwnd)
            result: dict[str, Any] = geometry.monitor_bounds.to_jsonable()
            return result
        except Exception:  # noqa: BLE001
            return None

    def _cursor(self) -> dict[str, int] | None:
        try:
            modules = self.backend.modules()
            x, y = modules.win32api.GetCursorPos()
            return {"x": int(x), "y": int(y)}
        except Exception:  # noqa: BLE001
            return None

    def _input_generation(self) -> int | None:
        if self.input_generation_reader is None:
            return self.input_generation
        try:
            value = self.input_generation_reader()
            return value if type(value) is int and value >= 0 else None
        except Exception:  # noqa: BLE001
            return None


def compare_snapshots(
    before: ConsequenceSnapshot,
    after: ConsequenceSnapshot,
    *,
    operation: str,
    expectation: dict[str, Any] | None,
    semantic_postcondition: dict[str, Any] | None = None,
    terminal_effect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta = _target_delta(before, after)
    churn = _topology_churn(before, after)
    semantic = _bounded_semantic_postcondition(semantic_postcondition)
    terminal = _bounded_terminal_effect(terminal_effect)
    meaningful_changed = bool(delta) or _semantic_changed(semantic) or _terminal_changed(terminal)
    snapshot_ambiguous = (
        before.public["target"].get("observation_error") not in {None, "invalid_window"}
        or after.public["target"].get("observation_error")
        not in {None, "invalid_window"}
        or before.public["foreground"].get("observation_error")
        not in {None, "foreground_unavailable"}
        or after.public["foreground"].get("observation_error")
        not in {None, "foreground_unavailable"}
    )
    if snapshot_ambiguous:
        observed_effect = "ambiguous"
    elif meaningful_changed:
        observed_effect = "changed"
    else:
        observed_effect = "no_observable_change"
    expectation_result = _classify_expectation(
        expectation,
        after=after,
        delta=delta,
        semantic=semantic,
        terminal=terminal,
        observed_effect=observed_effect,
    )
    return {
        "settle_window_ms": int(CONSEQUENCE_SETTLE_SECONDS * 1000),
        "observed_effect": observed_effect,
        "expectation": expectation_result,
        "operation": operation,
        "before": before.public,
        "after": after.public,
        "target_delta": delta,
        "unrelated_window_churn": churn,
        "semantic_postcondition": semantic,
        "terminal_effect": terminal,
    }


def delivery_evidence(
    *,
    status: str,
    method: object,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"not_attempted", "rejected", "partial", "delivered"}:
        status = "not_attempted"
    return {
        "status": status,
        "method": str(method)[:96] if method not in {None, ""} else None,
        **(details or {}),
    }


def unavailable_consequence(reason: str) -> dict[str, Any]:
    return {
        "settle_window_ms": int(CONSEQUENCE_SETTLE_SECONDS * 1000),
        "observed_effect": "ambiguous",
        "expectation": "ambiguous_effect",
        "observation_error": reason,
        "before": None,
        "after": None,
        "target_delta": {},
        "unrelated_window_churn": _empty_churn(),
        "semantic_postcondition": None,
        "terminal_effect": None,
        "phase_timings_ms": {},
    }


def _elapsed_ms(started: float) -> float:
    return round(max(0.0, (time.perf_counter() - started) * 1000.0), 3)


def delivery_from_result(result: dict[str, Any]) -> dict[str, Any]:
    outcome = str(result.get("outcome") or "failed")
    if result.get("partial_mutation") or result.get("partial_delivery"):
        status = "partial"
    elif outcome in {"postcondition_verified", "delivery_only"}:
        status = "delivered"
    elif outcome == "rejected":
        status = "rejected"
    else:
        status = "not_attempted"
    return delivery_evidence(status=status, method=result.get("method"))


def delivery_from_error(error: ComputerError) -> dict[str, Any]:
    outcome = str(error.details.get("outcome") or "failed")
    if error.details.get("partial_mutation") or error.details.get("partial_delivery"):
        status = "partial"
    elif outcome == "delivery_only":
        status = "delivered"
    elif outcome == "rejected":
        status = "rejected"
    else:
        status = "not_attempted"
    return delivery_evidence(status=status, method=error.details.get("method"))


def _target_delta(
    before: ConsequenceSnapshot,
    after: ConsequenceSnapshot,
) -> dict[str, Any]:
    left = before.public["target"]
    right = after.public["target"]
    delta: dict[str, Any] = {}
    for field in (
        "exists",
        "strong_identity",
        "state",
        "bounds",
        "monitor",
        "dpi",
        "foreground",
        "z_order",
    ):
        if left.get(field) != right.get(field):
            delta[field] = {"before": left.get(field), "after": right.get(field)}
    raw_changed = before.raw_title != after.raw_title
    title_class = _title_change_classification(before.raw_title, after.raw_title)
    if raw_changed:
        delta["title"] = {
            "raw_changed": True,
            "classification": title_class,
            "before_hmac_sha256": left.get("title_hmac_sha256"),
            "after_hmac_sha256": right.get("title_hmac_sha256"),
        }
    before_cursor = before.public.get("cursor")
    after_cursor = after.public.get("cursor")
    if before_cursor != after_cursor:
        delta["cursor"] = {"before": before_cursor, "after": after_cursor}
    before_foreground = before.public.get("foreground")
    after_foreground = after.public.get("foreground")
    if before_foreground != after_foreground:
        delta["foreground_owner"] = {
            "before": before_foreground,
            "after": after_foreground,
        }
    if before.public.get("active_monitor") != after.public.get("active_monitor"):
        delta["active_monitor"] = {
            "before": before.public.get("active_monitor"),
            "after": after.public.get("active_monitor"),
        }
    if before.public.get("input_generation") != after.public.get("input_generation"):
        delta["input_generation"] = {
            "before": before.public.get("input_generation"),
            "after": after.public.get("input_generation"),
        }
    return delta


def _topology_churn(
    before: ConsequenceSnapshot,
    after: ConsequenceSnapshot,
) -> dict[str, Any]:
    before_keys = set(before.topology)
    after_keys = set(after.topology)
    created = sorted(after_keys - before_keys)
    destroyed = sorted(before_keys - after_keys)
    return {
        "created_count": len(created),
        "destroyed_count": len(destroyed),
        "created_identity_digests": [
            _topology_item_digest(after.topology[key]) for key in created[:MAX_CHURN_ITEMS]
        ],
        "destroyed_identity_digests": [
            _topology_item_digest(before.topology[key]) for key in destroyed[:MAX_CHURN_ITEMS]
        ],
        "truncated": (
            before.topology_truncated
            or after.topology_truncated
            or len(created) > MAX_CHURN_ITEMS
            or len(destroyed) > MAX_CHURN_ITEMS
        ),
    }


def _classify_expectation(
    expectation: dict[str, Any] | None,
    *,
    after: ConsequenceSnapshot,
    delta: dict[str, Any],
    semantic: dict[str, Any] | None,
    terminal: dict[str, Any] | None,
    observed_effect: str,
) -> str:
    if not expectation:
        return "no_declared_expected_effect"
    if observed_effect == "ambiguous":
        return "ambiguous_effect"
    kind = expectation.get("kind")
    effect_verified = False
    if kind == "foreground":
        target_foreground = delta.get("foreground")
        effect_verified = bool(
            after.public["foreground"].get("hwnd") == expectation.get("hwnd")
            and isinstance(target_foreground, dict)
            and target_foreground.get("after") is True
        )
    elif kind == "state":
        state_delta = delta.get("state")
        effect_verified = bool(
            after.public["target"].get("state") == expectation.get("state")
            and isinstance(state_delta, dict)
            and state_delta.get("after") == expectation.get("state")
        )
    elif kind == "bounds":
        bounds_delta = delta.get("bounds")
        effect_verified = bool(
            after.public["target"].get("bounds") == expectation.get("bounds")
            and isinstance(bounds_delta, dict)
            and bounds_delta.get("after") == expectation.get("bounds")
        )
    elif kind == "exists":
        exists_delta = delta.get("exists")
        effect_verified = bool(
            after.public["target"].get("exists") == expectation.get("exists")
            and isinstance(exists_delta, dict)
            and exists_delta.get("after") == expectation.get("exists")
        )
    elif kind == "cursor":
        cursor_delta = delta.get("cursor")
        effect_verified = bool(
            after.public.get("cursor") == expectation.get("cursor")
            and isinstance(cursor_delta, dict)
            and cursor_delta.get("after") == expectation.get("cursor")
        )
    elif kind == "semantic":
        effect_verified = bool(
            semantic
            and semantic.get("verified") is True
            and (
                semantic.get("changed") is True
                or semantic.get("change_required") is False
            )
        )
    elif kind == "terminal_submission":
        effect_verified = bool(terminal and terminal.get("submission") == "observed")
    if effect_verified:
        return "expected_effect_verified"
    if observed_effect == "no_observable_change":
        return "unexpected_effect" if expectation.get("required") else "ambiguous_effect"
    return "unexpected_effect"


def _bounded_semantic_postcondition(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    allowed = {
        "verified",
        "changed",
        "change_required",
        "selected_before",
        "selected_after",
        "expanded_before",
        "expanded_after",
        "checked_before",
        "checked_after",
        "enabled_before",
        "enabled_after",
        "scroll_before",
        "scroll_after",
        "value_char_count",
        "readback_char_count",
    }
    return {key: value[key] for key in sorted(allowed & set(value))}


def _bounded_terminal_effect(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    allowed = {
        "character_delivery",
        "enter_delivery",
        "submission",
        "inserted_newline",
        "composer_change",
        "focus_change",
        "tab_change",
    }
    return {key: value[key] for key in sorted(allowed & set(value))}


def _semantic_changed(value: dict[str, Any] | None) -> bool:
    return bool(value and value.get("changed") is True)


def _terminal_changed(value: dict[str, Any] | None) -> bool:
    if not value:
        return False
    return bool(
        value.get("submission") == "observed"
        or value.get("inserted_newline") == "inserted"
        or value.get("composer_change") == "changed"
        or value.get("focus_change") == "changed"
        or value.get("tab_change") == "changed"
    )


def _public_identity(identity: WindowIdentity, key: bytes) -> dict[str, Any]:
    exact = {
        "hwnd": identity.hwnd,
        "pid": identity.pid,
        "thread_id": identity.thread_id,
        "process_started": identity.process_started,
        "session_id": identity.session_id,
        "process_hmac_sha256": _fingerprint(key, "process", identity.process or ""),
        "class_hmac_sha256": _fingerprint(key, "class", identity.class_name),
        "desktop_hmac_sha256": _fingerprint(key, "desktop", identity.desktop_name or ""),
    }
    exact["fingerprint"] = _fingerprint(
        key,
        "identity",
        json.dumps(exact, sort_keys=True, separators=(",", ":")),
    )
    return exact


def _fingerprint(key: bytes, label: str, value: str) -> str:
    return hmac.new(key, f"{label}\0{value}".encode(), hashlib.sha256).hexdigest()


def _topology_digest(
    key: bytes,
    topology: dict[str, tuple[int, int | None, int | None, str]],
) -> str:
    canonical = json.dumps(sorted(topology.items()), ensure_ascii=True, separators=(",", ":"))
    return _fingerprint(key, "topology", canonical)


def _topology_item_digest(value: tuple[int, int | None, int | None, str]) -> str:
    return "sha256:" + hashlib.sha256(repr(value[1:]).encode("utf-8")).hexdigest()[:24]


def _z_order_for(
    hwnd: int,
    topology: dict[str, tuple[int, int | None, int | None, str]],
) -> dict[str, int] | None:
    row = topology.get(str(hwnd))
    if row is None:
        return None
    return {"visible_index": row[0], "visible_count": len(topology)}


def _title_change_classification(before: str | None, after: str | None) -> str:
    if before == after:
        return "unchanged"
    if before is None or after is None:
        return "meaningful"
    if _strip_volatile_prefix(before) == _strip_volatile_prefix(after):
        return "volatile_only"
    return "meaningful"


def _strip_volatile_prefix(value: str) -> str:
    stripped = value
    for pattern in (_SPINNER_PREFIX_RE, _PROGRESS_PREFIX_RE):
        stripped = pattern.sub("", stripped, count=1)
    return stripped


def _bounded_capture_provenance(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    allowed = {"capture_id", "image_sha256", "profile", "geometry_fingerprint"}
    return {key: value[key] for key in sorted(allowed & set(value))}


def _default_settle_seconds(backend: Any) -> float:
    return (
        CONSEQUENCE_SETTLE_SECONDS
        if backend.__class__.__module__ == "agent_tools.computer.win32_backend"
        and backend.__class__.__name__ == "Win32Backend"
        else 0.0
    )


def _empty_churn() -> dict[str, Any]:
    return {
        "created_count": 0,
        "destroyed_count": 0,
        "created_identity_digests": [],
        "destroyed_identity_digests": [],
        "truncated": False,
    }
