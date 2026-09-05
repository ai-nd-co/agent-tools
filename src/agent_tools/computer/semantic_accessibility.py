from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sys
import time
from dataclasses import dataclass
from typing import Any, cast

from agent_tools.computer.element_refs import (
    ANCESTOR_SIGNATURE_PATTERN,
    ElementReferenceStore,
    attach_element_references,
)
from agent_tools.computer.models import ComputerError, WindowIdentity, bounded_text
from agent_tools.computer.uia_winapp import (
    MAX_BACKEND_OUTPUT_BYTES,
    MAX_ELEMENT_ID_CHARS,
    MAX_INSPECT_ELEMENTS,
    MAX_READ_CHARS,
    MAX_SELECTOR_CHARS,
    NATIVE_UIA_TIMEOUT_SECONDS,
    WinAppAdapter,
    WinAppBinding,
    _canonical_invariant_double,
    _compressed_powershell_command,
    _deadline_timeout,
    _range_value_from_bits,
    _require_deadline,
    _run_process,
    _trusted_powershell,
    _valid_range_value_bits,
    redact_credential_like_text,
)
from agent_tools.computer.win32_backend import Win32Backend

MAX_SEMANTIC_DEPTH = 12
DEEP_DIAGNOSTIC_MIN_DEPTH = 8
SEMANTIC_FALLBACK_TIMEOUT_SECONDS = 8.0
NATIVE_READ_TIMEOUT_SECONDS = 8.0
NATIVE_UIA_PROVIDER_VERSION = "windows-uia-managed-v1"
NATIVE_LOCATOR_PREFIX = "nuia-"
_NATIVE_VIEWS = ("raw", "control", "content")
_ACTION_PATTERNS = {
    "expand_collapse",
    "invoke",
    "range_value",
    "scroll",
    "selection_item",
    "toggle",
    "value",
}
_WINAPP_FALLBACK_ERRORS = {
    "backend_version_mismatch",
    "capability_missing",
    "uia_backend_error",
    "uia_backend_unreadable",
    "uia_backend_start_failed",
    "uia_invalid_response",
    "uia_output_too_large",
    "uia_pipe_unavailable",
    "uia_process_failed",
    "uia_provider_unavailable",
    "uia_timeout",
    "uia_version_probe_failed",
}
_NATIVE_DEGRADABLE_ERRORS = {
    "capability_missing",
    "native_uia_invalid_response",
    "native_uia_query_failed",
    "native_uia_unavailable",
    "uia_backend_start_failed",
    "uia_output_too_large",
    "uia_pipe_unavailable",
    "uia_process_failed",
    "uia_timeout",
}
_CONTAINER_CONTROL_TYPES = {"custom", "group", "pane", "window"}
_TRUNCATED_AUTH_TOKEN_RE = re.compile(
    r"(?i)(?P<prefix>\b(?:Basic|Bearer)\s+)[^\s,;}}\]]+"
)
_MAX_INVARIANT_DOUBLE_CHARS = 24
_NATIVE_INSPECTION_FIELDS = {
    "runtime_id",
    "parent_runtime_id",
    "automation_id",
    "name",
    "control_type",
    "class",
    "framework",
    "value",
    "text",
    "bounds",
    "depth",
    "enabled",
    "offscreen",
    "password",
    "read_only",
    "range_read_only",
    "patterns",
    "scroll",
}


@dataclass(frozen=True)
class NativeUiaBinding:
    view: str

    def public(self) -> dict[str, Any]:
        return {
            "name": f"native_uia_{self.view}",
            "version": NATIVE_UIA_PROVIDER_VERSION,
            "source": "windows_system",
            "actions_exposed": self.view == "raw",
        }


class NativeUiaAdapter:
    """Bounded fallback over Windows' managed UI Automation bridge."""

    def __init__(self, backend: Win32Backend | None = None) -> None:
        self.backend = backend or Win32Backend()

    def probe(self) -> dict[str, Any]:
        available = _trusted_powershell() is not None
        return {
            "available": available,
            "status": "ready" if available else "missing",
            "version": NATIVE_UIA_PROVIDER_VERSION,
            "views": list(_NATIVE_VIEWS),
            "actions_exposed": True,
        }

    def inspect(
        self,
        identity: WindowIdentity,
        *,
        view: str,
        depth: int,
        max_elements: int,
        deadline: float | None = None,
    ) -> tuple[NativeUiaBinding, dict[str, Any]]:
        normalized_view = view.casefold()
        if normalized_view not in _NATIVE_VIEWS:
            raise ComputerError("invalid_uia_view", "The native UIA view is invalid.")
        powershell = _trusted_powershell()
        if powershell is None:
            raise ComputerError(
                "native_uia_unavailable",
                "The trusted Windows UI Automation bridge is unavailable.",
            )
        request = {
            "depth": max(1, min(MAX_SEMANTIC_DEPTH, int(depth))),
            "maxElements": max(1, min(MAX_INSPECT_ELEMENTS, int(max_elements))),
        }
        self.backend.revalidate_identity(identity)
        _require_deadline(deadline)
        result = _run_process(
            powershell,
            [
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-STA",
                "-Command",
                _compressed_powershell_command(
                    _native_inspection_script(identity.hwnd, normalized_view)
                ),
            ],
            timeout_seconds=_deadline_timeout(NATIVE_UIA_TIMEOUT_SECONDS, deadline),
            max_output_bytes=MAX_BACKEND_OUTPUT_BYTES,
            input_bytes=json.dumps(request, ensure_ascii=True).encode("ascii"),
            deadline=deadline,
        )
        self.backend.revalidate_identity(identity)
        _require_deadline(deadline)
        if result.returncode != 0:
            raise ComputerError(
                "native_uia_query_failed",
                "The trusted native UI Automation provider rejected the request.",
            )
        try:
            payload = json.loads(result.stdout.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise ComputerError(
                "native_uia_invalid_response",
                "The native UI Automation provider returned an invalid bounded response.",
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
            raise ComputerError(
                "native_uia_invalid_response",
                "The native UI Automation provider returned an invalid bounded response.",
            )
        count = payload.get("count")
        truncated = payload.get("truncated")
        if (
            len(payload["elements"]) > request["maxElements"]
            or type(count) is not int
            or count != len(payload["elements"])
            or count > request["maxElements"]
            or type(truncated) is not bool
            or not all(
                _valid_native_inspection_row(
                    item,
                    requested_depth=request["depth"],
                    response_truncated=truncated,
                )
                for item in payload["elements"]
            )
        ):
            raise ComputerError(
                "native_uia_invalid_response",
                "The native UI Automation provider returned an invalid bounded response.",
            )
        elements = _normalize_native_elements(
            payload["elements"],
            provider=f"native_uia_{normalized_view}",
        )
        duplicate_runtime_ids = _duplicate_native_runtime_ids(payload["elements"])
        visible_elements = [item for item in elements if item.get("offscreen") is False]
        _require_deadline(deadline)
        return NativeUiaBinding(normalized_view), {
            "depth": request["depth"],
            "count": len(visible_elements),
            "provider_count": count,
            "truncated": truncated,
            "identity_ambiguous": bool(duplicate_runtime_ids),
            "ambiguous_runtime_id_count": len(duplicate_runtime_ids),
            "values_omitted": True,
            "elements": visible_elements,
        }

    def read(
        self,
        identity: WindowIdentity,
        *,
        element: str,
        max_chars: int,
    ) -> tuple[NativeUiaBinding, dict[str, Any]]:
        if not _is_native_locator(element):
            raise ComputerError("invalid_element", "The UI Automation element is invalid.")
        deadline = time.monotonic() + NATIVE_READ_TIMEOUT_SECONDS
        binding, before_inspection = self.inspect(
            identity,
            view="raw",
            depth=MAX_SEMANTIC_DEPTH,
            max_elements=MAX_INSPECT_ELEMENTS,
            deadline=deadline,
        )
        limit = max(1, min(MAX_READ_CHARS, int(max_chars)))
        before_match = _unique_native_match(before_inspection, element)
        if before_match.get("password") is not False:
            return binding, _redacted_native_read(
                element,
                password=before_match.get("password"),
            )
        locator = before_match.get("locator") or {}
        runtime_id = locator.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ComputerError(
                "stale_element",
                "The UI Automation element changed; inspect again.",
            )
        value = self._read_runtime_value(
            identity,
            runtime_id=runtime_id,
            max_chars=limit,
            deadline=deadline,
        )
        _after_binding, after_inspection = self.inspect(
            identity,
            view="raw",
            depth=MAX_SEMANTIC_DEPTH,
            max_elements=MAX_INSPECT_ELEMENTS,
            deadline=deadline,
        )
        after_match = _unique_native_match(after_inspection, element)
        after_locator = after_match.get("locator") or {}
        if after_locator.get("runtime_id") != runtime_id:
            raise ComputerError(
                "stale_element",
                "The UI Automation element changed; inspect again.",
            )
        if after_match.get("password") is not False:
            return binding, _redacted_native_read(
                element,
                password=after_match.get("password"),
            )
        match_count = value.get("match_count")
        if value.get("tree_truncated") is True:
            raise ComputerError(
                "ambiguous_element",
                "The bounded UI Automation tree cannot establish a unique locator.",
            )
        if match_count == 0:
            raise ComputerError(
                "stale_element",
                "The UI Automation element changed; inspect again.",
            )
        if match_count != 1:
            raise ComputerError("ambiguous_element", "The UI Automation locator is ambiguous.")
        if (
            value.get("runtime_id_before") != runtime_id
            or value.get("runtime_id_after") != runtime_id
        ):
            raise ComputerError(
                "stale_element",
                "The UI Automation element changed; inspect again.",
            )
        if value.get("password_before") is not False or value.get("password_after") is not False:
            password = (
                True
                if value.get("password_before") is True
                or value.get("password_after") is True
                else None
            )
            return binding, _redacted_native_read(element, password=password)
        value_status = value.get("value_status")
        if value_status in {"read", "range_read", "name_fallback"}:
            if value_status == "range_read":
                number = _range_value_from_bits(str(value.get("range_bits") or ""))
                raw_text = _canonical_invariant_double(number)
                char_count = _utf16_code_units(raw_text)
                truncated = False
            else:
                raw_text = str(value.get("text") or "")
                char_count = int(value.get("char_count") or 0)
                truncated = bool(value.get("truncated"))
        else:
            raise ComputerError(
                "native_uia_value_read_failed",
                "The native UI Automation provider could not read the selected value.",
            )
        raw_text, auth_token_redacted = _redact_auth_token(raw_text)
        text, redacted = redact_credential_like_text(raw_text)
        redacted = redacted or auth_token_redacted
        if redacted:
            char_count = _utf16_code_units(text)
        text, final_truncated = _truncate_utf16(text, limit)
        truncated = truncated or final_truncated
        _require_deadline(deadline)
        return binding, {
            "element": element,
            "resolved_element": element,
            "text": text,
            "char_count": char_count,
            "truncated": truncated,
            "redacted": redacted,
            "redaction_reason": "credential_like_text" if redacted else None,
            "_element_reference_source": after_match,
        }

    def _read_runtime_value(
        self,
        identity: WindowIdentity,
        *,
        runtime_id: str,
        max_chars: int,
        deadline: float,
    ) -> dict[str, Any]:
        powershell = _trusted_powershell()
        if powershell is None:
            raise ComputerError(
                "native_uia_unavailable",
                "The trusted Windows UI Automation bridge is unavailable.",
            )
        request = {
            "runtimeId": runtime_id,
            "maxChars": max_chars,
            "depth": MAX_SEMANTIC_DEPTH,
            "maxElements": MAX_INSPECT_ELEMENTS,
        }
        self.backend.revalidate_identity(identity)
        _require_deadline(deadline)
        result = _run_process(
            powershell,
            [
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-STA",
                "-Command",
                _compressed_powershell_command(_native_value_script(identity.hwnd)),
            ],
            timeout_seconds=_deadline_timeout(NATIVE_UIA_TIMEOUT_SECONDS, deadline),
            max_output_bytes=MAX_BACKEND_OUTPUT_BYTES,
            input_bytes=json.dumps(request, ensure_ascii=True).encode("ascii"),
            deadline=deadline,
        )
        self.backend.revalidate_identity(identity)
        _require_deadline(deadline)
        if result.returncode != 0:
            raise ComputerError(
                "native_uia_query_failed",
                "The trusted native UI Automation provider rejected the read request.",
            )
        try:
            payload = json.loads(result.stdout.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise ComputerError(
                "native_uia_invalid_response",
                "The native UI Automation provider returned an invalid bounded response.",
            ) from exc
        if not _valid_native_value_payload(
            payload,
            runtime_id=runtime_id,
            limit=max_chars,
        ):
            raise ComputerError(
                "native_uia_invalid_response",
                "The native UI Automation provider returned an invalid bounded response.",
            )
        _require_deadline(deadline)
        return cast(dict[str, Any], payload)


def _unique_native_match(inspection: dict[str, Any], element: str) -> dict[str, Any]:
    if inspection.get("truncated") is True or inspection.get("identity_ambiguous") is True:
        raise ComputerError(
            "ambiguous_element",
            "The bounded UI Automation tree cannot establish a unique locator.",
        )
    raw_elements = inspection.get("elements")
    elements = raw_elements if isinstance(raw_elements, list) else []
    matches: list[dict[str, Any]] = [
        item
        for item in elements
        if isinstance(item, dict)
        if item.get("element") == element
    ]
    if not matches:
        raise ComputerError("stale_element", "The UI Automation element changed; inspect again.")
    if len(matches) != 1:
        raise ComputerError("ambiguous_element", "The UI Automation locator is ambiguous.")
    return matches[0]


def _redacted_native_read(element: str, *, password: object) -> dict[str, Any]:
    return {
        "element": element,
        "resolved_element": element,
        "text": None,
        "char_count": 0,
        "truncated": False,
        "redacted": True,
        "redaction_reason": (
            "password_control" if password is True else "uia_sensitive_state_unavailable"
        ),
    }


def _valid_native_value_payload(
    payload: object,
    *,
    runtime_id: str,
    limit: int,
) -> bool:
    if not isinstance(payload, dict):
        return False
    match_count = payload.get("match_count")
    text = payload.get("text")
    char_count = payload.get("char_count")
    truncated = payload.get("truncated")
    value_supported = payload.get("value_supported")
    value_status = payload.get("value_status")
    range_bits = payload.get("range_bits")
    text_units = _utf16_code_units(text) if isinstance(text, str) else 0
    if (
        type(match_count) is not int
        or match_count < 0
        or match_count > MAX_INSPECT_ELEMENTS
        or (text is not None and not isinstance(text, str))
        or text_units > limit
        or type(char_count) is not int
        or char_count < 0
        or char_count > 2**31 - 1
        or type(truncated) is not bool
        or type(payload.get("tree_truncated")) is not bool
        or type(value_supported) is not bool
        or not isinstance(value_status, str)
        or value_status
        not in {
            "not_attempted",
            "name_fallback",
            "read",
            "range_read",
            "read_failed",
            "discarded_unsafe",
        }
        or (isinstance(text, str) and _has_unpaired_surrogate(text))
    ):
        return False
    for name in ("password_before", "password_after"):
        if payload.get(name) is not None and type(payload.get(name)) is not bool:
            return False
    for name in ("runtime_id_before", "runtime_id_after"):
        value = payload.get(name)
        if value is not None and (
            not isinstance(value, str)
            or _utf16_code_units(value) > 512
            or _has_unpaired_surrogate(value)
        ):
            return False
    value_safe = (
        match_count == 1
        and payload.get("tree_truncated") is False
        and payload.get("password_before") is False
        and payload.get("password_after") is False
        and payload.get("runtime_id_before") == runtime_id
        and payload.get("runtime_id_after") == runtime_id
    )
    cleared = text is None and char_count == 0 and truncated is False
    if not value_safe:
        return (
            value_supported is False
            and value_status in {"not_attempted", "discarded_unsafe"}
            and range_bits is None
            and cleared
        )
    if value_status == "read_failed":
        return value_supported is False and range_bits is None and cleared
    if value_status == "range_read":
        return value_supported is True and cleared and _valid_range_value_bits(range_bits)
    if range_bits is not None:
        return False
    if value_status not in {"read", "range_read", "name_fallback"}:
        return False
    if value_supported is not (value_status in {"read", "range_read"}):
        return False
    if not isinstance(text, str):
        return False
    if truncated:
        return text_units in {limit, max(0, limit - 1)} and char_count > limit
    return char_count == text_units


def inspect_semantic_window(
    *,
    hwnd: int,
    depth: int,
    interactive: bool,
    max_elements: int,
    backend: Win32Backend | None = None,
    winapp_adapter: WinAppAdapter | None = None,
    native_adapter: NativeUiaAdapter | None = None,
    element_ref_store: ElementReferenceStore | None = None,
    require_title_match: bool = False,
) -> dict[str, Any]:
    resolved_backend = backend or Win32Backend()
    identity = resolved_backend.capture_identity(hwnd)
    resolved_winapp = winapp_adapter or WinAppAdapter(resolved_backend)
    try:
        binding, inspection = resolved_winapp.inspect(
            identity,
            depth=depth,
            interactive=interactive,
            max_elements=max_elements,
        )
    except ComputerError as exc:
        if exc.code not in _WINAPP_FALLBACK_ERRORS:
            raise
        result = _inspect_native_only(
            identity=identity,
            depth=depth,
            interactive=interactive,
            max_elements=max_elements,
            backend=resolved_backend,
            native_adapter=native_adapter,
            winapp_error=exc,
        )
    else:
        result = _inspect_semantic_with_winapp(
            identity=identity,
            depth=depth,
            interactive=interactive,
            max_elements=max_elements,
            backend=resolved_backend,
            native_adapter=native_adapter,
            winapp_binding=binding,
            winapp_inspection=inspection,
        )
    actual = _revalidate_inspection_identity(
        resolved_backend,
        identity,
        require_title_match=require_title_match,
    )
    result = attach_element_references(
        result,
        identity=actual,
        store=element_ref_store,
    )
    return {
        **result,
        **_identity_transaction_metadata(identity, actual),
        "observation_tier": "uia_accessibility_provider",
    }


def _inspect_semantic_with_winapp(
    *,
    identity: WindowIdentity,
    depth: int,
    interactive: bool,
    max_elements: int,
    backend: Win32Backend,
    native_adapter: NativeUiaAdapter | None = None,
    winapp_binding: WinAppBinding,
    winapp_inspection: dict[str, Any],
) -> dict[str, Any]:
    binding = winapp_binding
    inspection = winapp_inspection
    winapp_elements = _normalize_winapp_elements(inspection.get("elements") or [])
    inspection = {**inspection, "elements": winapp_elements}
    shallow = _shallow_diagnostic(
        provider=binding.public()["name"],
        inspection=inspection,
        requested_depth=depth,
    )
    inconclusive = _surface_inconclusive_diagnostic(
        provider=binding.public()["name"],
        inspection=inspection,
        requested_depth=depth,
    )
    attempts = [
        _provider_attempt(
            binding.public(),
            inspection,
            status=_classification_status(
                shallow=shallow,
                inconclusive=inconclusive,
                selected=True,
            ),
            reason=_classification_reason(shallow, inconclusive),
        )
    ]
    selected_binding: dict[str, Any] = binding.public()
    selected_inspection = _filter_winapp_inspection(
        inspection,
        interactive=interactive,
    )
    selected_inconclusive = inconclusive
    if shallow is not None and depth < DEEP_DIAGNOSTIC_MIN_DEPTH:
        selected_inconclusive = _provider_diagnostics_inconclusive(
            provider=selected_binding["name"],
            inspection=inspection,
            requested_depth=depth,
            reason="deeper_provider_diagnostics_required",
            candidate_reason=shallow["reason"],
        )
        shallow = None
        attempts[0]["status"] = "inconclusive_selected"
        attempts[0]["reason"] = selected_inconclusive["reason"]

    native_deadline: float | None = None
    if depth >= DEEP_DIAGNOSTIC_MIN_DEPTH:
        deadline = time.monotonic() + SEMANTIC_FALLBACK_TIMEOUT_SECONDS
        native_deadline = deadline
        resolved_native = native_adapter or NativeUiaAdapter(backend)
        native_candidates: list[tuple[NativeUiaBinding, dict[str, Any]]] = []
        raw_shallow: dict[str, Any] | None = None
        raw_inconclusive: dict[str, Any] | None = None
        complete_native_shell_views: set[str] = set()
        for view in _NATIVE_VIEWS:
            try:
                native_binding, native_inspection = resolved_native.inspect(
                    identity,
                    view=view,
                    depth=depth,
                    max_elements=max_elements,
                    deadline=deadline,
                )
            except ComputerError as exc:
                attempts.append(
                    {
                        "provider": f"native_uia_{view}",
                        "status": "unavailable",
                        "error_code": exc.code,
                    }
                )
                if exc.code not in _NATIVE_DEGRADABLE_ERRORS:
                    raise
                continue
            native_shallow = _shallow_diagnostic(
                provider=native_binding.public()["name"],
                inspection=native_inspection,
                requested_depth=depth,
            )
            native_inconclusive = _surface_inconclusive_diagnostic(
                provider=native_binding.public()["name"],
                inspection=native_inspection,
                requested_depth=depth,
            )
            returned_inspection = _filter_native_inspection(
                native_inspection,
                interactive=interactive,
            )
            attempts.append(
                _provider_attempt(
                    native_binding.public(),
                    native_inspection,
                    status=_classification_status(
                        shallow=native_shallow,
                        inconclusive=native_inconclusive,
                        selected=False,
                    ),
                    reason=_classification_reason(native_shallow, native_inconclusive),
                )
            )
            if view == "raw":
                raw_shallow = native_shallow
                raw_inconclusive = native_inconclusive
            elif (
                native_shallow is None
                and native_inconclusive is None
                and not _has_document_element(native_inspection.get("elements") or [])
            ):
                complete_native_shell_views.add(view)
            if (
                view == "raw"
                and native_shallow is None
                and native_inconclusive is None
                and (not interactive or returned_inspection.get("elements"))
            ):
                native_candidates.append((native_binding, returned_inspection))
        selected_candidate = next(
            (
                candidate
                for candidate in native_candidates
                if _materially_better(candidate[1], selected_inspection)
            ),
            None,
        )
        truncated_raw_empty_document_consensus = bool(
            raw_inconclusive is not None
            and raw_inconclusive.get("reason")
            == "truncated_before_surface_classification"
            and raw_inconclusive.get("candidate_reason") == "empty_document_fragment"
            and complete_native_shell_views == {"control", "content"}
        )
        provider_consensus_empty_document = bool(
            raw_shallow is not None
            and raw_shallow.get("reason") == "empty_document_fragment"
            or truncated_raw_empty_document_consensus
        )
        if (
            (shallow is not None or selected_inconclusive is not None)
            and selected_candidate is not None
        ):
            selected_inspection = selected_candidate[1]
            selected_binding = _selected_native_binding(
                selected_candidate[0],
                selected_inspection,
            )
            shallow = None
            selected_inconclusive = None
            _mark_selected_attempt(
                attempts,
                selected_binding["name"],
                shallow=False,
            )
        elif (
            shallow is None
            and selected_inconclusive is None
            and provider_consensus_empty_document
            and not _has_document_element(winapp_elements)
        ):
            shallow = {
                "code": "semantic_surface_shallow",
                "provider": selected_binding["name"],
                **_surface_stats(inspection.get("elements") or []),
                "requested_depth": depth,
                "reason": "provider_consensus_empty_document_fragment",
            }
            attempts[0]["status"] = "shallow_selected"
            attempts[0]["reason"] = shallow["reason"]
        if (
            shallow is not None
            and raw_inconclusive is not None
            and not truncated_raw_empty_document_consensus
        ):
            selected_inconclusive = _provider_diagnostics_inconclusive(
                provider=selected_binding["name"],
                inspection=inspection,
                requested_depth=depth,
                reason="native_raw_surface_inconclusive",
                candidate_reason=shallow["reason"],
                diagnostic_reason=raw_inconclusive["reason"],
            )
            shallow = None
            attempts[0]["status"] = "inconclusive_selected"
            attempts[0]["reason"] = selected_inconclusive["reason"]

    result = {
        "window": identity.public(),
        "backend": selected_binding,
        **selected_inspection,
        "provider_attempts": attempts,
        "semantic_surface_shallow": shallow,
        "semantic_surface_inconclusive": selected_inconclusive,
    }
    fallbacks: dict[str, Any] = {}
    if shallow is not None:
        fallbacks["observation"] = {
                "kind": "targeted_screenshot",
                "status": "conditional",
                "when": "only_if_missing_state_is_genuinely_visual",
        }
    if _provider_actions_unavailable(selected_binding):
        fallbacks["action"] = {
            "kind": "guarded_physical_input",
            "status": "available",
            "reason": "requires_fresh_capture_and_explicit_allow_physical",
        }
    if _provider_actions_unavailable(selected_binding):
        result["semantic_action_unavailable"] = {
            "code": "semantic_provider_action_unsupported",
            "provider": selected_binding["name"],
            "reason": "provider_read_only",
        }
    if fallbacks:
        result["fallbacks"] = fallbacks
    if native_deadline is not None:
        _require_deadline(native_deadline)
    return result


def _inspect_native_only(
    *,
    identity: WindowIdentity,
    depth: int,
    interactive: bool,
    max_elements: int,
    backend: Win32Backend,
    native_adapter: NativeUiaAdapter | None,
    winapp_error: ComputerError,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = [
        {
            "provider": "winapp",
            "status": "unavailable",
            "error_code": winapp_error.code,
        }
    ]
    deadline = time.monotonic() + SEMANTIC_FALLBACK_TIMEOUT_SECONDS
    resolved_native = native_adapter or NativeUiaAdapter(backend)
    candidates: list[
        tuple[
            NativeUiaBinding,
            dict[str, Any],
            dict[str, Any] | None,
            dict[str, Any] | None,
        ]
    ] = []
    for view in _NATIVE_VIEWS:
        try:
            binding, inspection = resolved_native.inspect(
                identity,
                view=view,
                depth=depth,
                max_elements=max_elements,
                deadline=deadline,
            )
        except ComputerError as exc:
            attempts.append(
                {
                    "provider": f"native_uia_{view}",
                    "status": "unavailable",
                    "error_code": exc.code,
                }
            )
            if exc.code not in _NATIVE_DEGRADABLE_ERRORS:
                raise
            continue
        shallow = _shallow_diagnostic(
            provider=binding.public()["name"],
            inspection=inspection,
            requested_depth=depth,
        )
        inconclusive = _surface_inconclusive_diagnostic(
            provider=binding.public()["name"],
            inspection=inspection,
            requested_depth=depth,
        )
        returned_inspection = _filter_native_inspection(
            inspection,
            interactive=interactive,
        )
        attempts.append(
            _provider_attempt(
                binding.public(),
                inspection,
                status=_classification_status(
                    shallow=shallow,
                    inconclusive=inconclusive,
                    selected=False,
                ),
                reason=_classification_reason(shallow, inconclusive),
            )
        )
        if view == "raw":
            candidates.append((binding, returned_inspection, shallow, inconclusive))
    selected = candidates[0] if candidates else None
    if selected is None:
        raise ComputerError(
            "semantic_providers_unavailable",
            "No bounded semantic accessibility provider is available for the window.",
            details={"provider_attempts": attempts},
        )
    binding, inspection, shallow, inconclusive = selected
    selected_binding = _selected_native_binding(binding, inspection)
    _mark_selected_attempt(
        attempts,
        selected_binding["name"],
        shallow=shallow is not None,
        inconclusive=inconclusive is not None,
    )
    result = {
        "window": identity.public(),
        "backend": selected_binding,
        **inspection,
        "provider_attempts": attempts,
        "semantic_surface_shallow": shallow,
        "semantic_surface_inconclusive": inconclusive,
    }
    fallbacks: dict[str, Any] = {}
    if _provider_actions_unavailable(selected_binding):
        result["semantic_action_unavailable"] = {
            "code": "semantic_provider_action_unsupported",
            "provider": selected_binding["name"],
            "reason": "provider_read_only",
        }
        fallbacks["action"] = {
            "kind": "guarded_physical_input",
            "status": "available",
            "reason": "requires_fresh_capture_and_explicit_allow_physical",
        }
    if shallow is not None:
        fallbacks["observation"] = {
            "kind": "targeted_screenshot",
            "status": "conditional",
            "when": "only_if_missing_state_is_genuinely_visual",
        }
    if fallbacks:
        result["fallbacks"] = fallbacks
    _require_deadline(deadline)
    return result


def _filter_native_inspection(
    inspection: dict[str, Any],
    *,
    interactive: bool,
) -> dict[str, Any]:
    if not interactive:
        return inspection
    elements = [
        item
        for item in inspection.get("elements") or []
        if (
            isinstance(item, dict)
            and item.get("actionable")
            and item.get("enabled") is True
            and item.get("offscreen") is False
            and _has_addressable_native_locator(item)
        )
    ]
    return {
        **inspection,
        "interactive": True,
        "count": len(elements),
        "elements": elements,
    }


def _filter_winapp_inspection(
    inspection: dict[str, Any],
    *,
    interactive: bool,
) -> dict[str, Any]:
    if not interactive:
        return inspection
    provider_filter_applied = inspection.get("provider_filter_applied") is True
    elements = [
        item
        for item in inspection.get("elements") or []
        if (
            isinstance(item, dict)
            and item.get("actionable")
            and item.get("enabled") is True
            and item.get("offscreen") is False
            and isinstance(item.get("element"), str)
            and bool(item.get("element"))
            and (
                (
                    item.get("action_supported") is True
                    and isinstance(item.get("locator"), dict)
                )
                or (
                    provider_filter_applied
                    and item.get("element_ref_reason")
                    == "provider_filtered_hierarchy_inexact"
                )
            )
        )
    ]
    return {
        **inspection,
        "interactive": True,
        "count": len(elements),
        "elements": elements,
    }


def _mark_selected_attempt(
    attempts: list[dict[str, Any]],
    provider: str,
    *,
    shallow: bool,
    inconclusive: bool = False,
) -> None:
    for attempt in attempts:
        if attempt.get("provider") == provider:
            attempt["status"] = (
                "shallow_selected"
                if shallow
                else "inconclusive_selected" if inconclusive else "selected"
            )
        elif attempt.get("status") != "unavailable":
            attempt["status"] = "not_selected"


def _provider_actions_unavailable(binding: dict[str, Any]) -> bool:
    return binding.get("actions_exposed") is False


def _selected_native_binding(
    binding: NativeUiaBinding,
    inspection: dict[str, Any],
) -> dict[str, Any]:
    public = binding.public()
    public["actions_exposed"] = bool(
        binding.view == "raw"
        and any(
            isinstance(item, dict) and item.get("action_supported") is True
            for item in inspection.get("elements") or []
        )
    )
    return public


def read_semantic_element(
    *,
    hwnd: int,
    element: str,
    max_chars: int,
    backend: Win32Backend | None = None,
    winapp_adapter: WinAppAdapter | None = None,
    native_adapter: NativeUiaAdapter | None = None,
    element_ref_store: ElementReferenceStore | None = None,
    require_title_match: bool = False,
) -> dict[str, Any]:
    resolved_backend = backend or Win32Backend()
    identity = resolved_backend.capture_identity(hwnd)
    reference_source: dict[str, Any] | None = None
    reference_reason: str | None = None
    if element.startswith(NATIVE_LOCATOR_PREFIX):
        if not _is_native_locator(element):
            raise ComputerError("invalid_element", "The UI Automation element is invalid.")
        native_binding, reading = (native_adapter or NativeUiaAdapter(resolved_backend)).read(
            identity,
            element=element,
            max_chars=max_chars,
        )
        source = reading.pop("_element_reference_source", None)
        if isinstance(source, dict):
            reference_source = source
        else:
            reference_reason = "safe_locator_unavailable"
        public_binding = native_binding.public()
    else:
        resolved_winapp = winapp_adapter or WinAppAdapter(resolved_backend)
        winapp_binding, reading = resolved_winapp.read(
            identity,
            selector=element,
            max_chars=max_chars,
        )
        read_reference_identity = reading.pop("_element_reference_identity", None)
        if reading.get("redaction_reason") == "password_control":
            reference_reason = "sensitive_element"
        else:
            reference_source, reference_reason = _read_winapp_reference_source(
                resolved_winapp,
                identity,
                element,
                expected_identity=read_reference_identity,
            )
        public_binding = winapp_binding.public()
    actual = _revalidate_inspection_identity(
        resolved_backend,
        identity,
        require_title_match=require_title_match,
    )
    reference = _issue_read_reference(
        identity=actual,
        source=reference_source,
        reason=reference_reason,
        store=element_ref_store,
    )
    return {
        "window": actual.public(),
        "backend": public_binding,
        **reading,
        **reference,
        **_identity_transaction_metadata(identity, actual),
        "observation_tier": "uia_accessibility_provider",
    }


def _read_winapp_reference_source(
    adapter: WinAppAdapter,
    identity: WindowIdentity,
    element: str,
    *,
    expected_identity: object,
) -> tuple[dict[str, Any] | None, str | None]:
    stable_fields = ("automation_id", "control_type", "class", "ancestor_signature")
    if not isinstance(expected_identity, dict) or any(
        not isinstance(expected_identity.get(field), str)
        or not expected_identity.get(field)
        for field in stable_fields
    ):
        return None, "post_read_stable_identity_unavailable"
    try:
        _binding, inspection = adapter.inspect(
            identity,
            depth=MAX_SEMANTIC_DEPTH,
            interactive=False,
            max_elements=MAX_INSPECT_ELEMENTS,
        )
    except ComputerError as exc:
        return None, f"post_read_inspection_{exc.code}"[:96]
    matches = [
        item
        for item in _normalize_winapp_elements(inspection.get("elements") or [])
        if item.get("element") == element
    ]
    if len(matches) != 1:
        return None, "safe_locator_not_unique"
    try:
        _binding, _resolved_element, current = adapter._element_properties(
            identity,
            element,
        )
    except ComputerError as exc:
        return None, f"post_read_identity_{exc.code}"[:96]
    comparisons = (
        (
            "automation_id",
            str(expected_identity["automation_id"]),
            str(current.get("automation_id") or ""),
        ),
        (
            "control_type",
            str(expected_identity["control_type"]).casefold(),
            str(current.get("control_type") or "").casefold(),
        ),
        (
            "class",
            str(expected_identity["class"]).casefold(),
            str(current.get("class") or "").casefold(),
        ),
        (
            "ancestor_signature",
            str(expected_identity["ancestor_signature"]),
            str(current.get("ancestor_signature") or ""),
        ),
    )
    if any(expected != actual for _field, expected, actual in comparisons):
        return None, "post_read_stable_identity_changed"
    runtime_id = current.get("runtime_id")
    if not isinstance(runtime_id, str) or not runtime_id:
        return None, "post_read_stable_identity_unavailable"
    current_read_only = next(
        (
            state
            for state in (
                _strict_bool(current.get("value_read_only")),
                _strict_bool(current.get("range_value_read_only")),
            )
            if state is not None
        ),
        None,
    )
    matched_hierarchy = matches[0].get("hierarchy")
    if not isinstance(matched_hierarchy, dict):
        matched_hierarchy = {}
    source = {
        **matches[0],
        "automation_id": current.get("automation_id"),
        "control_type": current.get("control_type"),
        "class": current.get("class"),
        "enabled": current.get("enabled"),
        "read_only": current_read_only,
        "password": current.get("password"),
        "locator": {
            "provider": "winapp",
            "strategy": "selector_v1",
            "selector": element,
            "runtime_id": runtime_id,
            "automation_id": current.get("automation_id"),
            "ancestor_signature": current.get("ancestor_signature"),
        },
        "hierarchy": {
            **matched_hierarchy,
            "ancestor_signature": current.get("ancestor_signature"),
        },
    }
    return source, None


def _issue_read_reference(
    *,
    identity: WindowIdentity,
    source: dict[str, Any] | None,
    reason: str | None,
    store: ElementReferenceStore | None,
) -> dict[str, Any]:
    if source is None:
        return {
            "element_ref": None,
            "element_ref_status": "unavailable",
            "element_ref_reason": reason or "safe_locator_unavailable",
            "expires_at": None,
        }
    try:
        issued = (store or ElementReferenceStore()).issue(
            identity=identity,
            element=source,
        )
    except ComputerError as exc:
        if exc.code != "element_ref_unavailable":
            raise
        return {
            "element_ref": None,
            "element_ref_status": "unavailable",
            "element_ref_reason": "safe_locator_unavailable",
            "expires_at": None,
        }
    return {**issued, "element_ref_status": "available", "element_ref_reason": None}


def semantic_capabilities(
    *,
    native_adapter: NativeUiaAdapter | None = None,
) -> dict[str, Any]:
    legacy_available = sys.platform == "win32"
    return {
        "uia_native": (native_adapter or NativeUiaAdapter()).probe(),
        "legacy_accessibility": {
            "msaa": {
                "available": legacy_available,
                "runtime_provider": False,
                "status": (
                    "bakeoff_rejected_no_fixture_improvement"
                    if legacy_available
                    else "platform_unsupported"
                ),
            },
            "ia2": {
                "available": legacy_available,
                "runtime_provider": False,
                "status": (
                    "bakeoff_rejected_no_fixture_improvement"
                    if legacy_available
                    else "platform_unsupported"
                ),
            },
        },
    }


def _normalize_winapp_elements(elements: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    element_counts: dict[str, int] = {}
    for item in elements:
        if isinstance(item, dict) and isinstance(item.get("element"), str):
            element_key = str(item["element"])
            if element_key:
                element_counts[element_key] = element_counts.get(element_key, 0) + 1
    raw_by_element = {
        str(item.get("element")): item
        for item in elements
        if (
            isinstance(item, dict)
            and item.get("element")
            and element_counts.get(str(item.get("element"))) == 1
        )
    }
    parent_by_element = {
        str(item.get("element")): item.get("parent")
        for item in elements
        if isinstance(item, dict) and item.get("element")
    }
    for raw in elements:
        if not isinstance(raw, dict):
            continue
        element = raw.get("element")
        ancestor_signature = raw.get("_ancestor_signature")
        identity_exact = raw.get("_identity_exact") is not False
        ancestor_identity_exact = raw.get("_ancestor_identity_exact") is not False
        provider_filtered_hierarchy = raw.get("_provider_filtered_hierarchy") is True
        provider_ancestor_path = raw.get("_provider_ancestor_path")
        if not ancestor_identity_exact:
            ancestor_signature = None
        elif (
            not isinstance(ancestor_signature, str)
            or ANCESTOR_SIGNATURE_PATTERN.fullmatch(ancestor_signature) is None
        ):
            ancestors = _complete_winapp_ancestor_chain(
                str(element or ""),
                parent_by_element,
                raw_by_element,
            )
            ancestor_signature = (
                _ancestor_signature(ancestors) if ancestors is not None else None
            )
        public_raw = {key: value for key, value in raw.items() if not key.startswith("_")}
        locator = (
            {
                "provider": "winapp",
                "strategy": "selector_v1",
                "selector": element,
                "ancestor_signature": ancestor_signature,
            }
            if (
                isinstance(element, str)
                and element
                and element_counts.get(element) == 1
                and isinstance(ancestor_signature, str)
                and identity_exact
                and ancestor_identity_exact
            )
            else None
        )
        action_supported = bool(
            raw.get("actionable")
            and raw.get("enabled") is True
            and raw.get("offscreen") is False
            and locator
            and isinstance(raw.get("automation_id"), str)
            and bool(raw.get("automation_id"))
            and isinstance(raw.get("control_type"), str)
            and bool(raw.get("control_type"))
            and isinstance(raw.get("class"), str)
            and bool(raw.get("class"))
        )
        hierarchy = {
            "depth": raw.get("depth"),
            "parent": raw.get("parent"),
            "ancestor_signature": ancestor_signature,
        }
        if provider_filtered_hierarchy:
            hierarchy.update(
                {
                    "provider_ancestor_path": provider_ancestor_path,
                    "identity_complete": False,
                }
            )
        normalized.append(
            {
                **public_raw,
                "provider": "winapp",
                "role": raw.get("control_type"),
                "value": None,
                "text": None,
                "read_only": None,
                "password": None,
                "action_supported": action_supported,
                "hierarchy": hierarchy,
                "scroll": (
                    {"direction": raw.get("scroll_direction")}
                    if raw.get("scroll_direction")
                    else None
                ),
                "locator": locator,
                **(
                    {
                        "element_ref_reason": "provider_filtered_hierarchy_inexact",
                    }
                    if provider_filtered_hierarchy
                    else {}
                ),
            }
        )
    return normalized


def _normalize_native_elements(
    elements: list[Any],
    *,
    provider: str,
) -> list[dict[str, Any]]:
    safe_raw: list[dict[str, Any]] = []
    parent_by_runtime: dict[str, str | None] = {}
    raw_by_runtime: dict[str, dict[str, Any]] = {}
    duplicate_runtime_ids = _duplicate_native_runtime_ids(elements)
    for raw in elements:
        if not isinstance(raw, dict):
            continue
        runtime_id = _bounded_optional(raw.get("runtime_id"), 512)
        parent_runtime_id = _bounded_optional(raw.get("parent_runtime_id"), 512)
        if runtime_id in duplicate_runtime_ids:
            continue
        if runtime_id:
            parent_by_runtime[runtime_id] = (
                None if parent_runtime_id in duplicate_runtime_ids else parent_runtime_id
            )
            raw_by_runtime[runtime_id] = raw
        safe_raw.append(raw)

    normalized: list[dict[str, Any]] = []
    for raw in safe_raw:
        runtime_id = _bounded_optional(raw.get("runtime_id"), 512)
        automation_id = _bounded_optional(raw.get("automation_id"), 160)
        ancestors = _stable_ancestor_chain(
            runtime_id or "",
            parent_by_runtime,
            raw_by_runtime,
        )
        ancestor_signature = _ancestor_signature(ancestors)
        locator_key = "|".join(
            [provider, runtime_id or "", automation_id or "", ancestor_signature]
        )
        element = (
            NATIVE_LOCATOR_PREFIX
            + hashlib.sha256(locator_key.encode("utf-8")).hexdigest()[:24]
            if runtime_id
            else None
        )
        locator = (
            {
                "provider": provider,
                "strategy": "runtime_ancestor_v1",
                "runtime_id": runtime_id,
                "automation_id": automation_id,
                "ancestor_signature": ancestor_signature,
            }
            if element
            else None
        )
        patterns = [
            str(item)
            for item in raw.get("patterns") or []
            if isinstance(item, str) and item in _ACTION_PATTERNS
        ]
        read_only = _strict_bool(raw.get("read_only"))
        range_read_only = _strict_bool(raw.get("range_read_only"))
        mutation_read_only = [
            state
            for pattern, state in (
                ("value", read_only),
                ("range_value", range_read_only),
            )
            if pattern in patterns
        ]
        combined_read_only = (
            False
            if False in mutation_read_only
            else True
            if mutation_read_only and all(state is True for state in mutation_read_only)
            else None
        )
        scroll = _normalize_scroll(raw.get("scroll"))
        always_actionable = bool(
            set(patterns) & {"expand_collapse", "invoke", "selection_item", "toggle"}
        )
        value_actionable = "value" in patterns and read_only is False
        range_actionable = "range_value" in patterns and range_read_only is False
        scroll_actionable = bool(
            "scroll" in patterns
            and scroll
            and (
                scroll.get("horizontal_scrollable") is True
                or scroll.get("vertical_scrollable") is True
            )
        )
        actionable = bool(
            element
            and locator
            and (
                always_actionable
                or value_actionable
                or range_actionable
                or scroll_actionable
            )
        )
        control_type = _bounded_optional(raw.get("control_type"), 64)
        class_name = _bounded_optional(raw.get("class"), 160)
        action_supported = bool(
            actionable
            and provider == "native_uia_raw"
            and raw.get("enabled") is True
            and raw.get("offscreen") is False
            and raw.get("password") is False
            and control_type in {"TreeItem", "ListItem"}
            and "selection_item" in patterns
            and automation_id
            and class_name
            and ANCESTOR_SIGNATURE_PATTERN.fullmatch(ancestor_signature)
        )
        normalized.append(
            {
                "element": element,
                "provider": provider,
                "automation_id": automation_id,
                "name": _bounded_optional(raw.get("name"), 512),
                "role": control_type,
                "control_type": control_type,
                "class": class_name,
                "framework": _bounded_optional(raw.get("framework"), 64),
                "value": None,
                "text": None,
                "bounds": _normalize_bounds(raw.get("bounds")),
                "depth": _bounded_int(raw.get("depth"), minimum=0, maximum=MAX_SEMANTIC_DEPTH),
                "parent": parent_by_runtime.get(runtime_id or ""),
                "enabled": _strict_bool(raw.get("enabled")),
                "offscreen": _strict_bool(raw.get("offscreen")),
                "read_only": combined_read_only,
                "value_read_only": read_only,
                "range_read_only": range_read_only,
                "password": _strict_bool(raw.get("password")),
                "actionable": actionable,
                "action_supported": action_supported,
                "patterns": patterns,
                "hierarchy": {
                    "depth": _bounded_int(
                        raw.get("depth"), minimum=0, maximum=MAX_SEMANTIC_DEPTH
                    ),
                    "parent_runtime_id": parent_by_runtime.get(runtime_id or ""),
                    "ancestor_signature": ancestor_signature,
                },
                "scroll": scroll,
                "locator": locator,
            }
        )
    return normalized


def _provider_attempt(
    binding: dict[str, Any],
    inspection: dict[str, Any],
    *,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    stats = _surface_stats(inspection.get("elements") or [])
    return {
        "provider": binding.get("name"),
        "status": status,
        **stats,
        "truncated": bool(inspection.get("truncated")),
        "reason": reason,
    }


def _classification_status(
    *,
    shallow: dict[str, Any] | None,
    inconclusive: dict[str, Any] | None,
    selected: bool,
) -> str:
    if shallow is not None:
        return "shallow_selected" if selected else "shallow"
    if inconclusive is not None:
        return "inconclusive_selected" if selected else "inconclusive"
    return "selected" if selected else "diagnostic_only"


def _classification_reason(
    shallow: dict[str, Any] | None,
    inconclusive: dict[str, Any] | None,
) -> str | None:
    diagnostic = shallow or inconclusive
    reason = diagnostic.get("reason") if diagnostic else None
    return str(reason) if reason else None


def _surface_stats(elements: list[Any]) -> dict[str, int]:
    safe = [item for item in elements if isinstance(item, dict)]
    return {
        "element_count": len(safe),
        "maximum_depth": max((int(item.get("depth") or 0) for item in safe), default=0),
        "actionable_count": sum(_is_effectively_actionable(item) for item in safe),
        "text_bearing_count": sum(
            any(
                _has_meaningful_text(item.get(field))
                for field in ("name", "value", "text")
            )
            for item in safe
        ),
    }


def _has_meaningful_text(value: object) -> bool:
    return isinstance(value, str) and any(
        character.isprintable() and not character.isspace() for character in value
    )


def _is_effectively_actionable(item: dict[str, Any]) -> bool:
    return bool(
        item.get("actionable")
        and item.get("enabled") is True
        and item.get("offscreen") is False
        and isinstance(item.get("element"), str)
        and item.get("element")
        and isinstance(item.get("locator"), dict)
    )


def _shallow_diagnostic(
    *,
    provider: str,
    inspection: dict[str, Any],
    requested_depth: int,
) -> dict[str, Any] | None:
    if (
        inspection.get("truncated") is True
        or inspection.get("identity_ambiguous") is True
        or _reaches_depth_boundary(
        inspection,
        requested_depth=requested_depth,
        )
    ):
        return None
    return _shallow_candidate(
        provider=provider,
        inspection=inspection,
        requested_depth=requested_depth,
    )


def _shallow_candidate(
    *,
    provider: str,
    inspection: dict[str, Any],
    requested_depth: int,
) -> dict[str, Any] | None:
    elements = [item for item in inspection.get("elements") or [] if isinstance(item, dict)]
    stats = _surface_stats(elements)
    reason: str | None = None
    if not elements:
        reason = "no_semantic_elements"
    elif stats["actionable_count"] == 0 and stats["text_bearing_count"] == 0:
        reason = "no_actionable_or_text_bearing_elements"
    elif _has_empty_document_fragment(elements):
        reason = "empty_document_fragment"
    elif _is_container_only_surface(elements, stats):
        reason = "container_only_surface"
    if reason is None:
        return None
    return {
        "code": "semantic_surface_shallow",
        "provider": provider,
        **stats,
        "requested_depth": requested_depth,
        "reason": reason,
    }


def _surface_inconclusive_diagnostic(
    *,
    provider: str,
    inspection: dict[str, Any],
    requested_depth: int,
) -> dict[str, Any] | None:
    candidate = _shallow_candidate(
        provider=provider,
        inspection=inspection,
        requested_depth=requested_depth,
    )
    if inspection.get("identity_ambiguous") is True:
        reason = "ambiguous_runtime_ids"
        candidate_reason = (
            candidate["reason"] if candidate is not None else "partial_semantic_surface"
        )
    elif inspection.get("truncated") is True:
        reason = "truncated_before_surface_classification"
        candidate_reason = (
            candidate["reason"] if candidate is not None else "partial_semantic_surface"
        )
    elif _reaches_depth_boundary(inspection, requested_depth=requested_depth):
        if candidate is None:
            return None
        reason = "depth_limit_reached_before_surface_classification"
        candidate_reason = candidate["reason"]
    else:
        return None
    return {
        "code": "semantic_surface_inconclusive",
        "provider": provider,
        **_surface_stats(inspection.get("elements") or []),
        "requested_depth": requested_depth,
        "reason": reason,
        "candidate_reason": candidate_reason,
    }


def _provider_diagnostics_inconclusive(
    *,
    provider: str,
    inspection: dict[str, Any],
    requested_depth: int,
    reason: str,
    candidate_reason: str,
    diagnostic_reason: str | None = None,
) -> dict[str, Any]:
    diagnostic = {
        "code": "semantic_surface_inconclusive",
        "provider": provider,
        **_surface_stats(inspection.get("elements") or []),
        "requested_depth": requested_depth,
        "reason": reason,
        "candidate_reason": candidate_reason,
    }
    if diagnostic_reason is not None:
        diagnostic["diagnostic_reason"] = diagnostic_reason
    return diagnostic


def _reaches_depth_boundary(
    inspection: dict[str, Any],
    *,
    requested_depth: int,
) -> bool:
    if requested_depth < 1:
        return False
    return any(
        isinstance(item, dict)
        and type(item.get("depth")) is int
        and item["depth"] >= requested_depth
        for item in inspection.get("elements") or []
    )


def _has_empty_document_fragment(elements: list[dict[str, Any]]) -> bool:
    child_map: dict[str, list[dict[str, Any]]] = {}
    for item in elements:
        parent = item.get("parent")
        if isinstance(parent, str) and parent:
            child_map.setdefault(parent, []).append(item)
        hierarchy = item.get("hierarchy") or {}
        parent_runtime = hierarchy.get("parent_runtime_id")
        if isinstance(parent_runtime, str) and parent_runtime:
            child_map.setdefault(parent_runtime, []).append(item)
    documents = [
        item
        for item in elements
        if str(item.get("control_type") or "").casefold() == "document"
    ]
    if not documents:
        return False
    for item in documents:
        descendants = _descendants(_node_identity(item), child_map)
        meaningful = [
            child
            for child in descendants
            if any(
                _has_meaningful_text(child.get(field))
                for field in ("name", "value", "text")
            )
            or _is_effectively_actionable(child)
        ]
        if meaningful:
            return False
    return True


def _is_container_only_surface(
    elements: list[dict[str, Any]],
    stats: dict[str, int],
) -> bool:
    return (
        len(elements) <= 4
        and stats["maximum_depth"] <= 1
        and stats["actionable_count"] == 0
        and all(
            str(item.get("control_type") or "").casefold() in _CONTAINER_CONTROL_TYPES
            for item in elements
        )
        and not any(
            _has_meaningful_text(item.get(field))
            for item in elements
            for field in ("value", "text")
        )
    )


def _has_document_element(elements: list[dict[str, Any]]) -> bool:
    return any(
        str(item.get("control_type") or "").casefold() == "document"
        for item in elements
    )


def _descendants(
    parent: str,
    child_map: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    queue = list(child_map.get(parent, []))
    seen: set[int] = set()
    while queue and len(output) <= MAX_INSPECT_ELEMENTS:
        child = queue.pop(0)
        marker = id(child)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(child)
        locator = _node_identity(child)
        if locator:
            queue.extend(child_map.get(locator, []))
    return output


def _node_identity(item: dict[str, Any]) -> str:
    locator = item.get("locator") or {}
    runtime_id = locator.get("runtime_id")
    if isinstance(runtime_id, str) and runtime_id:
        return runtime_id
    element = item.get("element")
    return element if isinstance(element, str) else ""


def _materially_better(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    candidate_stats = _surface_stats(candidate.get("elements") or [])
    baseline_stats = _surface_stats(baseline.get("elements") or [])
    candidate_score = (
        candidate_stats["text_bearing_count"] * 4
        + candidate_stats["actionable_count"] * 3
        + min(candidate_stats["element_count"], 50)
    )
    baseline_score = (
        baseline_stats["text_bearing_count"] * 4
        + baseline_stats["actionable_count"] * 3
        + min(baseline_stats["element_count"], 50)
    )
    if candidate_stats["actionable_count"] > 0 and baseline_stats["actionable_count"] == 0:
        return True
    return candidate_score >= baseline_score + 8


def _ancestor_chain(value: str, parents: dict[str, Any]) -> list[str]:
    output: list[str] = []
    current: Any = parents.get(value)
    seen: set[str] = set()
    while isinstance(current, str) and current and current not in seen and len(output) < 12:
        seen.add(current)
        output.append(current)
        current = parents.get(current)
    return output


def _stable_ancestor_chain(
    value: str,
    parents: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
) -> list[str]:
    output: list[str] = []
    current: Any = parents.get(value)
    seen: set[str] = set()
    while isinstance(current, str) and current and current not in seen and len(output) < 12:
        seen.add(current)
        node = nodes.get(current) or {}
        output.append(
            "|".join(
                (
                    str(node.get("automation_id") or ""),
                    str(node.get("control_type") or node.get("role") or "").casefold(),
                    str(node.get("class") or "").casefold(),
                )
            )
        )
        current = parents.get(current)
    return output


def _complete_winapp_ancestor_chain(
    value: str,
    parents: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
) -> list[str] | None:
    output: list[str] = []
    current: Any = parents.get(value)
    seen: set[str] = set()
    while isinstance(current, str) and current and len(output) < 12:
        if current in seen or current not in nodes:
            return None
        seen.add(current)
        node = nodes[current]
        output.append(
            "|".join(
                (
                    str(node.get("automation_id") or ""),
                    str(node.get("control_type") or node.get("role") or "").casefold(),
                    str(node.get("class") or "").casefold(),
                )
            )
        )
        current = parents.get(current)
    return output


def _ancestor_signature(ancestors: list[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(ancestors).encode("utf-8")).hexdigest()[:24]


def _identity_transaction_metadata(
    before: WindowIdentity,
    after: WindowIdentity,
) -> dict[str, Any]:
    key = secrets.token_bytes(32)

    def fingerprint(value: str) -> str:
        return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()

    return {
        "strong_identity_verified": True,
        "identity_mismatch_fields": [],
        "title_changed_during_operation": before.title != after.title,
        "title_before_hmac_sha256": fingerprint(before.title),
        "title_after_hmac_sha256": fingerprint(after.title),
    }


def _revalidate_inspection_identity(
    backend: Win32Backend,
    identity: WindowIdentity,
    *,
    require_title_match: bool,
) -> WindowIdentity:
    if require_title_match:
        return backend.revalidate_identity(identity, require_title_match=True)
    return backend.revalidate_identity(identity)


def _bounded_optional(value: object, limit: int) -> str | None:
    if value in (None, ""):
        return None
    return bounded_text(value, limit)


def _bounded_int(value: object, *, minimum: int, maximum: int) -> int | None:
    if type(value) is not int or value < minimum or value > maximum:
        return None
    return value


def _strict_bool(value: object) -> bool | None:
    return value if type(value) is bool else None


def _utf16_code_units(value: str) -> int:
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _has_unpaired_surrogate(value: str) -> bool:
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 >= len(value) or not 0xDC00 <= ord(value[index + 1]) <= 0xDFFF:
                return True
            index += 2
            continue
        if 0xDC00 <= codepoint <= 0xDFFF:
            return True
        index += 1
    return False


def _redact_auth_token(value: str) -> tuple[str, bool]:
    redacted = False

    def replace(match: re.Match[str]) -> str:
        nonlocal redacted
        redacted = True
        return f"{match.group('prefix')}[REDACTED]"

    return _TRUNCATED_AUTH_TOKEN_RE.sub(replace, value), redacted


def _truncate_utf16(value: str, limit: int) -> tuple[str, bool]:
    budget = max(0, int(limit))
    index = 0
    units = 0
    while index < len(value):
        codepoint = ord(value[index])
        span = 1
        width = 2 if codepoint > 0xFFFF else 1
        if (
            0xD800 <= codepoint <= 0xDBFF
            and index + 1 < len(value)
            and 0xDC00 <= ord(value[index + 1]) <= 0xDFFF
        ):
            span = 2
            width = 2
        if units + width > budget:
            break
        units += width
        index += span
    return value[:index], index < len(value)


def _valid_native_inspection_row(
    item: object,
    *,
    requested_depth: int,
    response_truncated: bool,
) -> bool:
    if not isinstance(item, dict) or set(item) != _NATIVE_INSPECTION_FIELDS:
        return False
    for name, limit in (
        ("runtime_id", 512),
        ("parent_runtime_id", 512),
        ("automation_id", 160),
        ("name", 512),
        ("control_type", 64),
        ("class", 160),
        ("framework", 64),
    ):
        value = item.get(name)
        if value is not None and (
            not isinstance(value, str)
            or _utf16_code_units(value) > limit
            or _has_unpaired_surrogate(value)
        ):
            return False
    if item.get("value") is not None or item.get("text") is not None:
        return False
    bounds = item.get("bounds")
    if bounds is not None and _normalize_bounds(bounds) is None:
        return False
    depth = item.get("depth")
    if type(depth) is not int or not 0 <= depth <= requested_depth:
        return False
    if any(
        item.get(name) is not None and type(item.get(name)) is not bool
        for name in (
            "enabled",
            "offscreen",
            "password",
            "read_only",
            "range_read_only",
        )
    ):
        return False
    patterns = item.get("patterns")
    if (
        not isinstance(patterns, list)
        or len(patterns) > len(_ACTION_PATTERNS)
        or any(
            not isinstance(pattern, str) or pattern not in _ACTION_PATTERNS
            for pattern in patterns
        )
        or len(patterns) != len(set(patterns))
    ):
        return False
    if "value" not in patterns and item.get("read_only") is not None:
        return False
    if not response_truncated and "value" in patterns and type(item.get("read_only")) is not bool:
        return False
    if "range_value" not in patterns and item.get("range_read_only") is not None:
        return False
    if (
        not response_truncated
        and "range_value" in patterns
        and type(item.get("range_read_only")) is not bool
    ):
        return False
    scroll = item.get("scroll")
    if scroll is not None and not _valid_native_scroll(scroll):
        return False
    if "scroll" not in patterns:
        return scroll is None
    return response_truncated or scroll is not None


def _valid_native_scroll(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "horizontal_scrollable",
        "vertical_scrollable",
        "horizontal_percent",
        "vertical_percent",
    }:
        return False
    if any(
        type(value.get(name)) is not bool
        for name in ("horizontal_scrollable", "vertical_scrollable")
    ):
        return False
    for axis in ("horizontal", "vertical"):
        scrollable = value.get(f"{axis}_scrollable")
        percent = value.get(f"{axis}_percent")
        if (
            isinstance(percent, bool)
            or not isinstance(percent, (int, float))
        ):
            return False
        if scrollable is True and not 0 <= percent <= 100:
            return False
        if scrollable is False and percent != -1:
            return False
    return True


def _duplicate_native_runtime_ids(elements: list[Any]) -> set[str]:
    counts: dict[str, int] = {}
    for item in elements:
        runtime_id = item.get("runtime_id") if isinstance(item, dict) else None
        if isinstance(runtime_id, str) and runtime_id:
            counts[runtime_id] = counts.get(runtime_id, 0) + 1
    return {runtime_id for runtime_id, count in counts.items() if count > 1}


def _normalize_bounds(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict) or set(value) != {"x", "y", "width", "height"}:
        return None
    fields: dict[str, int] = {}
    for name in ("x", "y", "width", "height"):
        raw = value.get(name)
        if type(raw) is not int or not -(2**31) <= raw < 2**31:
            return None
        fields[name] = raw
    if fields["width"] < 0 or fields["height"] < 0:
        return None
    return fields


def _normalize_scroll(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "horizontal_scrollable": _strict_bool(value.get("horizontal_scrollable")),
        "vertical_scrollable": _strict_bool(value.get("vertical_scrollable")),
        "horizontal_percent": value.get("horizontal_percent")
        if isinstance(value.get("horizontal_percent"), (int, float))
        else None,
        "vertical_percent": value.get("vertical_percent")
        if isinstance(value.get("vertical_percent"), (int, float))
        else None,
    }


def _is_native_locator(value: str) -> bool:
    suffix = value[len(NATIVE_LOCATOR_PREFIX) :]
    return (
        value.startswith(NATIVE_LOCATOR_PREFIX)
        and len(value) == len(NATIVE_LOCATOR_PREFIX) + 24
        and all(character in "0123456789abcdef" for character in suffix)
        and len(value) <= min(MAX_SELECTOR_CHARS, MAX_ELEMENT_ID_CHARS)
    )


def _has_addressable_native_locator(item: dict[str, Any]) -> bool:
    element = item.get("element")
    locator = item.get("locator")
    return (
        isinstance(element, str)
        and _is_native_locator(element)
        and isinstance(locator, dict)
        and locator.get("provider") == item.get("provider")
        and locator.get("strategy") == "runtime_ancestor_v1"
        and isinstance(locator.get("runtime_id"), str)
        and bool(locator.get("runtime_id"))
    )


def _native_inspection_script(hwnd: int, view: str) -> str:
    walker = {
        "raw": "RawViewWalker",
        "control": "ControlViewWalker",
        "content": "ContentViewWalker",
    }[view]
    lines = [
        "$ErrorActionPreference='Stop';",
        "Add-Type -AssemblyName UIAutomationClient;",
        "Add-Type -AssemblyName UIAutomationTypes;",
        (
            "function Emit($payload,[int]$code){$payload|ConvertTo-Json "
            "-Compress -Depth 8;exit $code};"
        ),
        "$requestText=[Console]::In.ReadToEnd();",
        "$request=$requestText|ConvertFrom-Json;",
        f"$root=[Windows.Automation.AutomationElement]::FromHandle([IntPtr]{hwnd});",
        "if($null -eq $root){Emit @{error='root_unavailable'} 3};",
        f"$walker=[Windows.Automation.TreeWalker]::{walker};",
        "$items=[Collections.Generic.List[object]]::new();",
        "$queue=[Collections.Generic.Queue[object]]::new();",
        "$queue.Enqueue([pscustomobject]@{element=$root;depth=0;parent=$null});",
        "$truncated=$false;",
        (
            "function SafeProperty($element,$property){try{$value="
            "$element.GetCurrentPropertyValue($property,$true);"
            "if([object]::ReferenceEquals($value,[Windows.Automation.AutomationElement]::"
            "NotSupported)){return $null};return $value}"
            "catch{$script:truncated=$true;return $null}};"
        ),
        (
            "function SafeText($value,[int]$limit){if($null -eq $value){return $null};"
            "$text=[string]$value;if($text.Length -gt $limit){$take=$limit;"
            "if($take -gt 0 -and [char]::IsHighSurrogate($text[$take-1]) -and "
            "[char]::IsLowSurrogate($text[$take])){$take--};return "
            "$text.Substring(0,$take)};return $text};"
        ),
        (
            "function SafeIdentityText($value,[int]$limit){"
            "if($null -eq $value){return $null};$text=[string]$value;"
            "if($text.Length -gt $limit){$script:truncated=$true;return $null};"
            "return $text};"
        ),
        "while($queue.Count -gt 0){",
        (
            " $entry=$queue.Dequeue();if($items.Count -ge "
            "[int]$request.maxElements){$truncated=$true;break};"
        ),
        (
            " $element=$entry.element;$runtime=$null;try{$runtime="
            "($element.GetRuntimeId() -join ',')}catch{$truncated=$true};"
        ),
        (
            " $control=SafeProperty $element "
            "([Windows.Automation.AutomationElement]::ControlTypeProperty);"
        ),
        (
            " $controlName=$null;if($control -is [Windows.Automation.ControlType]){"
            "$controlName=$control.ProgrammaticName -replace '^ControlType\\.',''};"
        ),
        (
            " $bounds=SafeProperty $element "
            "([Windows.Automation.AutomationElement]::BoundingRectangleProperty);"
        ),
        (
            " $boundsOutput=$null;if($bounds -is [Windows.Rect]){$boundsOutput="
            "[pscustomobject]@{x=[int]$bounds.X;y=[int]$bounds.Y;"
            "width=[int]$bounds.Width;height=[int]$bounds.Height}};"
        ),
        " $patterns=[Collections.Generic.List[string]]::new();",
        " try{foreach($pattern in $element.GetSupportedPatterns()){",
        (
            "  $name=[Windows.Automation.Automation]::PatternName($pattern);"
            "if($name -eq 'Invoke'){$patterns.Add('invoke')}"
        ),
        (
            "  elseif($name -eq 'Value'){$patterns.Add('value')}"
            "elseif($name -eq 'RangeValue'){$patterns.Add('range_value')}"
        ),
        (
            "  elseif($name -eq 'Scroll'){$patterns.Add('scroll')}"
            "elseif($name -eq 'Toggle'){$patterns.Add('toggle')}"
        ),
        (
            "  elseif($name -eq 'SelectionItem'){$patterns.Add('selection_item')}"
            "elseif($name -eq 'ExpandCollapse'){$patterns.Add('expand_collapse')}}}"
            "catch{$truncated=$true};"
        ),
        (
            " $password=SafeProperty $element "
            "([Windows.Automation.AutomationElement]::IsPasswordProperty);"
        ),
        " $value=$null;$readOnly=$null;if($patterns.Contains('value')){try{",
        (
            "  $vp=$element.GetCurrentPattern([Windows.Automation.ValuePattern]::Pattern);"
            "$readOnly=$vp.Current.IsReadOnly;"
        ),
        " }catch{$truncated=$true}};",
        " $rangeReadOnly=$null;if($patterns.Contains('range_value')){try{",
        (
            "  $rp=$element.GetCurrentPattern("
            "[Windows.Automation.RangeValuePattern]::Pattern);"
            "$rangeReadOnly=$rp.Current.IsReadOnly;"
        ),
        " }catch{$truncated=$true}};",
        " $scroll=$null;if($patterns.Contains('scroll')){try{",
        (
            "  $sp=$element.GetCurrentPattern([Windows.Automation.ScrollPattern]::Pattern);"
            "$scroll=[pscustomobject]@{"
        ),
        (
            "   horizontal_scrollable=$sp.Current.HorizontallyScrollable;"
            "vertical_scrollable=$sp.Current.VerticallyScrollable;"
        ),
        (
            "   horizontal_percent=$sp.Current.HorizontalScrollPercent;"
            "vertical_percent=$sp.Current.VerticalScrollPercent}}"
            "catch{$truncated=$true}};"
        ),
        " $items.Add([pscustomobject]@{",
        "  runtime_id=SafeIdentityText $runtime 512;",
        "  parent_runtime_id=SafeIdentityText $entry.parent 512;",
        (
            "  automation_id=SafeIdentityText (SafeProperty $element "
            "([Windows.Automation.AutomationElement]::AutomationIdProperty)) 160;"
        ),
        (
            "  name=SafeText (SafeProperty $element "
            "([Windows.Automation.AutomationElement]::NameProperty)) 512;"
        ),
        "  control_type=SafeText $controlName 64;",
        (
            "  class=SafeText (SafeProperty $element "
            "([Windows.Automation.AutomationElement]::ClassNameProperty)) 160;"
        ),
        (
            "  framework=SafeText (SafeProperty $element "
            "([Windows.Automation.AutomationElement]::FrameworkIdProperty)) 64;"
        ),
        "  value=$value;text=$null;bounds=$boundsOutput;depth=[int]$entry.depth;",
        (
            "  enabled=SafeProperty $element "
            "([Windows.Automation.AutomationElement]::IsEnabledProperty);"
        ),
        (
            "  offscreen=SafeProperty $element "
            "([Windows.Automation.AutomationElement]::IsOffscreenProperty);"
        ),
        (
            "  password=$password;read_only=$readOnly;"
            "range_read_only=$rangeReadOnly;patterns=$patterns;scroll=$scroll});"
        ),
        " if([int]$entry.depth -ge [int]$request.depth){continue};",
        " try{$child=$walker.GetFirstChild($element)}catch{$truncated=$true;$child=$null};",
        " while($null -ne $child){",
        (
            "  if(($items.Count+$queue.Count) -ge [int]$request.maxElements){"
            "$truncated=$true;break};"
        ),
        (
            "  $queue.Enqueue([pscustomobject]@{element=$child;"
            "depth=[int]$entry.depth+1;parent=$runtime});"
        ),
        "  try{$child=$walker.GetNextSibling($child)}catch{$truncated=$true;$child=$null}",
        " }",
        "};",
        "Emit @{count=$items.Count;truncated=$truncated;elements=$items} 0;",
    ]
    return "\n".join(lines)


def _native_value_script(hwnd: int) -> str:
    lines = [
        "$ErrorActionPreference='Stop';",
        "Add-Type -AssemblyName UIAutomationClient;",
        "Add-Type -AssemblyName UIAutomationTypes;",
        (
            "function Emit($payload,[int]$code){$payload|ConvertTo-Json "
            "-Compress -Depth 5;exit $code};"
        ),
        "$requestText=[Console]::In.ReadToEnd();",
        "$request=$requestText|ConvertFrom-Json;",
        f"$root=[Windows.Automation.AutomationElement]::FromHandle([IntPtr]{hwnd});",
        "if($null -eq $root){Emit @{error='root_unavailable'} 3};",
        "$walker=[Windows.Automation.TreeWalker]::RawViewWalker;",
        "$queue=[Collections.Generic.Queue[object]]::new();",
        "$queue.Enqueue([pscustomobject]@{element=$root;depth=0});",
        "$matches=[Collections.Generic.List[object]]::new();",
        "$visited=0;$treeTruncated=$false;",
        (
            "function SafeProperty($element,$property){try{$value="
            "$element.GetCurrentPropertyValue($property,$true);"
            "if([object]::ReferenceEquals($value,[Windows.Automation.AutomationElement]::"
            "NotSupported)){return $null};return $value}catch{return $null}};"
        ),
        (
            "function PatternAvailability($element,$property){try{$value="
            "$element.GetCurrentPropertyValue($property,$true);"
            "if([object]::ReferenceEquals($value,"
            "[Windows.Automation.AutomationElement]::NotSupported)){"
            "return [pscustomobject]@{known=$true;value=$false}};"
            "if($value -isnot [bool]){return [pscustomobject]@{"
            "known=$false;value=$null}};return [pscustomobject]@{"
            "known=$true;value=[bool]$value}}catch{return [pscustomobject]@{"
            "known=$false;value=$null}}};"
        ),
        (
            "function RuntimeId($element){try{return ($element.GetRuntimeId() -join ',')}"
            "catch{$script:treeTruncated=$true;return $null}};"
        ),
        (
            "function SetBoundedText([string]$raw){$script:charCount=$raw.Length;"
            "if($raw.Length -gt [int]$request.maxChars){$take=[int]$request.maxChars;"
            "if($take -gt 0 -and [char]::IsHighSurrogate($raw[$take-1]) -and "
            "[char]::IsLowSurrogate($raw[$take])){$take--};"
            "$script:text=$raw.Substring(0,$take);$script:truncated=$true}"
            "else{$script:text=$raw}};"
        ),
        "while($queue.Count -gt 0){",
        (
            " $entry=$queue.Dequeue();if($visited -ge [int]$request.maxElements){"
            "$treeTruncated=$true;break};$visited++;"
        ),
        " $element=$entry.element;$runtime=RuntimeId $element;",
        (
            " if($runtime -eq [string]$request.runtimeId){"
            "$matches.Add($element)};"
        ),
        (
            " if([int]$entry.depth -ge [int]$request.depth){"
            "try{$boundaryChild=$walker.GetFirstChild($element)}"
            "catch{$treeTruncated=$true;$boundaryChild=$null};"
            "if($null -ne $boundaryChild){$treeTruncated=$true};continue};"
        ),
        (
            " try{$child=$walker.GetFirstChild($element)}"
            "catch{$treeTruncated=$true;$child=$null};"
        ),
        " while($null -ne $child){",
        (
            "  if(($visited+$queue.Count) -ge [int]$request.maxElements){"
            "$treeTruncated=$true;break};"
        ),
        (
            "  $queue.Enqueue([pscustomobject]@{element=$child;"
            "depth=[int]$entry.depth+1});"
        ),
        (
            "  try{$child=$walker.GetNextSibling($child)}"
            "catch{$treeTruncated=$true;$child=$null}"
        ),
        " }",
        "};",
        "$runtimeBefore=$null;$runtimeAfter=$null;",
        "$passwordBefore=$null;$passwordAfter=$null;",
        (
            "$valueSupported=$false;$valueStatus='not_attempted';"
            "$text=$null;$charCount=0;$truncated=$false;$rangeBits=$null;"
        ),
        "if($matches.Count -eq 1 -and -not $treeTruncated){",
        " $target=$matches[0];$runtimeBefore=RuntimeId $target;",
        (
            " $passwordBefore=SafeProperty $target "
            "([Windows.Automation.AutomationElement]::IsPasswordProperty);"
        ),
        (
            " if($runtimeBefore -eq [string]$request.runtimeId -and "
            "$passwordBefore -is [bool] -and $passwordBefore -eq $false){"
        ),
        (
            "  $valueAvailability=PatternAvailability $target "
            "([Windows.Automation.AutomationElement]::IsValuePatternAvailableProperty);"
        ),
        (
            "  $rangeAvailability=PatternAvailability $target "
            "([Windows.Automation.AutomationElement]::"
            "IsRangeValuePatternAvailableProperty);"
        ),
        (
            "  if($valueAvailability.known -and "
            "$valueAvailability.value -eq $true){try{"
        ),
        (
            "   $pattern=$target.GetCurrentPattern("
            "[Windows.Automation.ValuePattern]::Pattern);"
            "if($null -eq $pattern){throw 'value_pattern_missing'};"
        ),
        (
            "   $raw=[string]$pattern.Current.Value;$valueSupported=$true;"
            "$valueStatus='read';SetBoundedText $raw"
        ),
        (
            "  }catch{$valueSupported=$false;$valueStatus='read_failed';"
            "$text=$null;$charCount=0;$truncated=$false}}"
        ),
        (
            "  elseif($rangeAvailability.known -and "
            "$rangeAvailability.value -eq $true){try{"
        ),
        (
            "    $rangeValue=$target.GetCurrentPropertyValue("
            "[Windows.Automation.RangeValuePattern]::ValueProperty,$false);"
        ),
        (
            "    $number=[double]$rangeValue;"
            "if([double]::IsNaN($number) -or [double]::IsInfinity($number)){"
            "throw 'range_value_not_finite'};"
        ),
        (
            "    $bytes=[BitConverter]::GetBytes($number);"
            "if([BitConverter]::IsLittleEndian){[Array]::Reverse($bytes)};"
            "$rangeBits=([BitConverter]::ToString($bytes)).Replace('-','');"
            "$valueSupported=$true;$valueStatus='range_read'"
        ),
        (
            "   }catch{$valueSupported=$false;$valueStatus='read_failed';"
            "$text=$null;$charCount=0;$truncated=$false;$rangeBits=$null}}"
        ),
        (
            "  elseif($valueAvailability.known -and "
            "$rangeAvailability.known -and "
            "$valueAvailability.value -eq $false -and "
            "$rangeAvailability.value -eq $false){try{"
        ),
        (
            "    $nameValue=$target.GetCurrentPropertyValue("
            "[Windows.Automation.AutomationElement]::NameProperty,$true);"
        ),
        (
            "    if([object]::ReferenceEquals($nameValue,"
            "[Windows.Automation.AutomationElement]::NotSupported)){$raw=''}"
            "else{$raw=[string]$nameValue};"
        ),
        "    $valueStatus='name_fallback';SetBoundedText $raw",
        (
            "  }catch{$valueStatus='read_failed';$text=$null;"
            "$charCount=0;$truncated=$false}}"
        ),
        "  else{$valueStatus='read_failed'}",
        " }",
        (
            " $passwordAfter=SafeProperty $target "
            "([Windows.Automation.AutomationElement]::IsPasswordProperty);"
        ),
        " $runtimeAfter=RuntimeId $target;",
        " if($passwordBefore -isnot [bool]){$passwordBefore=$null};",
        " if($passwordAfter -isnot [bool]){$passwordAfter=$null};",
        (
            " if($passwordBefore -ne $false -or $passwordAfter -ne $false -or "
            "$runtimeBefore -ne [string]$request.runtimeId -or "
            "$runtimeAfter -ne [string]$request.runtimeId){"
        ),
        (
            "  $valueSupported=$false;$text=$null;$charCount=0;"
            "$truncated=$false;$rangeBits=$null;$valueStatus='discarded_unsafe'"
        ),
        " }",
        "};",
        "Emit @{",
        " match_count=$matches.Count;runtime_id_before=$runtimeBefore;",
        " runtime_id_after=$runtimeAfter;password_before=$passwordBefore;",
        (
            " password_after=$passwordAfter;value_supported=$valueSupported;"
            "value_status=$valueStatus;"
        ),
        " text=$text;char_count=$charCount;truncated=$truncated;range_bits=$rangeBits;",
        " tree_truncated=$treeTruncated",
        "} 0;",
    ]
    return "\n".join(lines)
