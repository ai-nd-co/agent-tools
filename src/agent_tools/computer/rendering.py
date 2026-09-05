from __future__ import annotations

import json
import unicodedata
from typing import Any

from agent_tools.computer.models import ComputerError, error_envelope, success_envelope


def render_json(command: str, data: dict[str, Any]) -> str:
    return json.dumps(success_envelope(command, data), indent=2, ensure_ascii=True) + "\n"


def render_json_error(command: str, error: ComputerError) -> str:
    return json.dumps(error_envelope(command, error), indent=2, ensure_ascii=True) + "\n"


def render_human_error(error: ComputerError) -> str:
    lines = [f"Error [{error.code}]: {_terminal_text(error.message)}"]
    details = error.details
    if details.get("operation_id"):
        lines.append(
            "Action: "
            f"operation={_value(details.get('operation'))} "
            f"id={_value(details.get('operation_id'))} "
            f"outcome={_value(details.get('outcome'))} "
            f"method={_value(details.get('method'))}"
        )
        notification = details.get("notification") or {}
        lines.append(
            "Notification: "
            f"{_value(notification.get('status'))}; "
            f"warnings={','.join(details.get('warnings') or []) or '-'}"
        )
    mismatch_fields = details.get("identity_mismatch_fields") or details.get(
        "element_mismatch_fields"
    )
    if mismatch_fields:
        lines.append(
            "Mismatch fields: "
            + ",".join(_terminal_text(item) for item in mismatch_fields[:8])
        )
    if details.get("resolution_stage"):
        lines.append(f"Resolution stage: {_value(details.get('resolution_stage'))}")
    if any(
        key in details for key in ("requested_state", "state_before", "state_after")
    ):
        lines.append(
            "State: "
            f"requested={_value(details.get('requested_state'))} "
            f"before={_value(details.get('state_before'))} "
            f"after={_value(details.get('state_after'))} "
            f"verified={_yes_no(details.get('postcondition_verified'))}"
        )
    focus = details.get("focus")
    if isinstance(focus, dict):
        lines.extend(_focus_lines(focus, prefix="Focus side effect"))
        if details.get("semantic_outcome"):
            lines.append(
                "Semantic action: "
                f"outcome={_value(details.get('semantic_outcome'))} "
                f"method={_value(details.get('semantic_method'))}"
            )
    if "requested_bounds" in details or "actual_bounds" in details:
        lines.append(
            "Bounds: "
            f"requested={_bounds(details.get('requested_bounds'))} "
            f"actual={_bounds(details.get('actual_bounds'))} "
            f"verified={_yes_no(details.get('postcondition_verified'))}"
        )
    if "restored_to_normal" in details:
        lines.append(
            "Focus restore: "
            f"performed={_yes_no(details.get('restore_performed'))} "
            f"normal={_yes_no(details.get('restored_to_normal'))}"
        )
    elif "restore_performed" in details or "restored_first" in details:
        lines.append(
            "Restore: "
            f"performed={_yes_no(details.get('restore_performed'))} "
            f"restored-first={_yes_no(details.get('restored_first'))} "
            f"status={_value(details.get('restore_status'))} "
            f"method={_value(details.get('restore_method'))}"
        )
    attachment_cleanup = details.get("input_attachment_cleanup")
    if isinstance(attachment_cleanup, dict):
        lines.append(
            "Input attachment cleanup: "
            f"status={_value(attachment_cleanup.get('status'))} "
            f"count={_value(attachment_cleanup.get('count'))}"
        )
    fallback = details.get("fallback")
    if isinstance(fallback, dict):
        lines.append(
            "Fallback: "
            f"{_value(fallback.get('kind'))}; status={_value(fallback.get('status'))}; "
            f"reason={_value(fallback.get('reason'))}"
        )
    delivery = details.get("delivery")
    if isinstance(delivery, dict):
        lines.append(
            "Delivery: "
            f"status={_value(delivery.get('status'))} "
            f"method={_value(delivery.get('method'))}"
        )
    consequence = details.get("consequence")
    if isinstance(consequence, dict):
        lines.append(
            "Observed consequence: "
            f"effect={_value(consequence.get('observed_effect'))} "
            f"expectation={_value(consequence.get('expectation'))}"
        )
    return "\n".join(lines) + "\n"


def render_info(data: dict[str, Any], *, background: bool) -> str:
    lines = ["AgentTools computer info"]
    identity = data.get("identity", {})
    if identity.get("available"):
        session_label = _join_nonempty(
            identity.get("session_id"), identity.get("session_name"), sep="/"
        )
        lines.append(
            "Identity: "
            f"host={_value(identity.get('host'))} user={_value(identity.get('user'))} "
            f"session={_value(session_label)} "
            f"desktop={_yes_no(identity.get('interactive_desktop_access'))} "
            f"lock={_value(identity.get('lock_state'))} "
            f"integrity={_value(identity.get('integrity'))} "
            f"elevated={_yes_no(identity.get('elevated'))}"
        )
    else:
        lines.append(_unavailable_line("Identity", identity))

    clock = data.get("clock", {})
    if clock.get("available"):
        lines.append(
            f"Clock: {_value(clock.get('local_iso'))} "
            f"({_value(clock.get('timezone'))}, {_format_offset(clock.get('utc_offset_seconds'))})"
        )
    else:
        lines.append(_unavailable_line("Clock", clock))

    system = data.get("system", {})
    if system.get("available"):
        windows = system.get("windows") or {}
        power = system.get("power") or {}
        lines.append(
            "System: "
            f"Windows {windows.get('major', '-')}.{windows.get('minor', '-')} "
            f"build {windows.get('build', '-')} {_value(system.get('architecture'))}; "
            f"uptime={_duration(system.get('uptime_seconds'))}; "
            f"power={_value(power.get('ac'))} battery={_percent(power.get('battery_percent'))}"
        )
    else:
        lines.append(_unavailable_line("System", system))

    display = data.get("display", {})
    if display.get("available"):
        lines.append(
            "Display: "
            f"virtual={_bounds(display.get('virtual_bounds'))} "
            f"primary={_bounds(display.get('primary_bounds'))} "
            f"monitors={_value(display.get('monitor_count'))} "
            f"scale={_percent(display.get('scaling_percent'))}"
        )
    else:
        lines.append(_unavailable_line("Display", display))

    focus = data.get("focus", {})
    if focus.get("available") and focus.get("active"):
        lines.append("Focus: " + _window_line(focus.get("window") or {}))
    elif focus.get("available"):
        lines.append("Focus: none")
    else:
        lines.append(_unavailable_line("Focus", focus))

    visible_apps = data.get("visible_apps", {})
    if visible_apps.get("available"):
        count = int(visible_apps.get("count") or 0)
        suffix = " (truncated)" if visible_apps.get("truncated") else ""
        lines.append(f"Visible apps ({count}){suffix}:")
        for window in visible_apps.get("items") or []:
            lines.append("  " + _window_line(window))
    else:
        lines.append(_unavailable_line("Visible apps", visible_apps))

    media = data.get("media", {})
    if media.get("available") and media.get("active"):
        media_session = media.get("session") or {}
        description = " - ".join(
            part
            for part in (media_session.get("artist"), media_session.get("title"))
            if part
        )
        lines.append(
            f"Media: {_value(media_session.get('app'))} "
            f"{_value(media_session.get('playback_state'))}"
            + (f"; {_value(description)}" if description else "")
        )
    elif media.get("available"):
        lines.append("Media: none")
    else:
        lines.append(_unavailable_line("Media", media))

    network = data.get("network", {})
    if network.get("available"):
        wifi = network.get("wifi") or {}
        tailscale = network.get("tailscale") or {}
        wifi_text = "off"
        if wifi.get("available") and wifi.get("connected"):
            wifi_text = f"{_value(wifi.get('ssid'))}/{_percent(wifi.get('signal_percent'))}"
        elif not wifi.get("available"):
            wifi_text = f"unavailable:{_value(wifi.get('error_code'))}"
        tailnet_addresses = ",".join(tailscale.get("addresses") or []) or "-"
        lines.append(
            "Network: "
            f"{_value(network.get('connectivity'))} "
            f"via={_value(network.get('interface'))}/{_value(network.get('interface_type'))} "
            f"ip={_value(network.get('primary_address'))} wifi={wifi_text} "
            f"tailscale={_value(tailscale.get('status'))}/{tailnet_addresses}"
        )
    else:
        lines.append(_unavailable_line("Network", network))

    readiness = data.get("readiness", {})
    if readiness.get("available"):
        backend_status = ",".join(
            _backend_status(name, details)
            for name, details in (readiness.get("backends") or {}).items()
        )
        lines.append(
            "Readiness: "
            f"AgentTools {_value(readiness.get('agent_tools_version'))}; "
            f"capabilities={','.join(readiness.get('capabilities') or [])}; "
            f"backends={backend_status or '-'}; "
            f"actions={_action_readiness(readiness)}; "
            f"desktop={_yes_no(readiness.get('desktop_usable'))}; "
            f"blocked={_value(readiness.get('action_blocked_reason'))}; "
            f"lock={_value(readiness.get('action_lock'))}"
        )
    else:
        lines.append(_unavailable_line("Readiness", readiness))

    if background:
        processes = data.get("background", {})
        if processes.get("available"):
            count = int(processes.get("count") or 0)
            suffix = " (truncated)" if processes.get("truncated") else ""
            lines.append(f"Background processes ({count}){suffix}:")
            for process in processes.get("items") or []:
                lines.append(
                    "  "
                    f"pid={_value(process.get('pid'))} name={_value(process.get('name'))} "
                    f"session={_value(process.get('session_id'))} "
                    f"state={_value(process.get('state'))}"
                )
        else:
            lines.append(_unavailable_line("Background processes", processes))
    else:
        lines.append("Background processes: hidden (use --background; filter with rg or jq).")
    return "\n".join(lines) + "\n"


def render_windows(data: dict[str, Any]) -> str:
    section = data.get("windows", {})
    if not section.get("available"):
        return _unavailable_line("Visible windows", section) + "\n"
    count = int(section.get("count") or 0)
    suffix = " (truncated)" if section.get("truncated") else ""
    lines = [f"Visible windows ({count}){suffix}:"]
    lines.extend("  " + _window_line(item) for item in section.get("items") or [])
    return "\n".join(lines) + "\n"


def render_focused(data: dict[str, Any]) -> str:
    section = data.get("focus", {})
    if not section.get("available"):
        return _unavailable_line("Focus", section) + "\n"
    if not section.get("active"):
        return "Focus: none\n"
    return "Focus: " + _window_line(section.get("window") or {}) + "\n"


def render_screenshot(data: dict[str, Any]) -> str:
    screenshot = data.get("screenshot", {})
    window = screenshot.get("window") or {}
    presentation = screenshot.get("presentation") or {}
    native_size = screenshot.get("native_pixel_size") or {}
    return (
        f"Saved {screenshot.get('width')}x{screenshot.get('height')} "
        f"{_value(presentation.get('profile'))} PNG "
        f"(native {native_size.get('width')}x{native_size.get('height')}) to "
        f"{_value(screenshot.get('output'))} (HWND {_value(window.get('hwnd'))}, "
        f"{_value(screenshot.get('backend'))}, capture "
        f"{_value(screenshot.get('capture_id'))}, expires "
        f"{_value(screenshot.get('expires_at'))}).\n"
    )


def render_ocr(data: dict[str, Any]) -> str:
    ocr = data.get("ocr") or {}
    source = ocr.get("source_capture") or {}
    uncertainty = ocr.get("uncertainty") or {}
    text = "\n".join(
        _terminal_text(line) for line in str(ocr.get("text") or "").splitlines()
    )
    body = text + ("\n" if text else "")
    flags = ",".join(_terminal_text(item) for item in uncertainty.get("flags") or []) or "none"
    return (
        body
        + f"[OCR capture={_value(source.get('capture_id'))} "
        f"confidence=unavailable uncertainty={flags}]\n"
    )


def render_inspect(data: dict[str, Any]) -> str:
    inspection = data.get("inspection") or {}
    window = inspection.get("window") or {}
    backend = inspection.get("backend") or {}
    count = int(inspection.get("count") or 0)
    suffix = " (truncated)" if inspection.get("truncated") else ""
    lines = [
        f"Semantic elements ({count}){suffix}: HWND {_value(window.get('hwnd'))} "
        f"via {_value(backend.get('name'))} {_value(backend.get('version'))}",
        f"Observation tier: {_value(inspection.get('observation_tier'))}",
    ]
    for element in inspection.get("elements") or []:
        flags: list[str] = []
        if element.get("actionable"):
            if backend.get("actions_exposed") is False:
                flags.extend(("semantic-read-only", "action-unsupported"))
            else:
                if element.get("enabled") is not True:
                    flags.append(
                        "disabled" if element.get("enabled") is False else "enabled-unknown"
                    )
                if element.get("offscreen") is not False:
                    flags.append(
                        "offscreen"
                        if element.get("offscreen") is True
                        else "visibility-unknown"
                    )
                if not (
                    isinstance(element.get("element"), str)
                    and element.get("element")
                    and isinstance(element.get("locator"), dict)
                ):
                    flags.append("unaddressable")
                if not flags:
                    flags.append("actionable")
        if element.get("scroll_direction"):
            flags.append(f"scroll:{element.get('scroll_direction')}")
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        lines.append(
            "  "
            f"{_value(element.get('element'))} {_value(element.get('control_type'))} "
            f"{_value(element.get('name'))} {_bounds(element.get('bounds'))}{flag_text} "
            f"ref={_value(element.get('element_ref'))} "
            f"ref-status={_value(element.get('element_ref_status'))} "
            f"expires={_value(element.get('expires_at'))} "
            f"ref-reason={_value(element.get('element_ref_reason'))}"
        )
    shallow = inspection.get("semantic_surface_shallow")
    if isinstance(shallow, dict):
        lines.append(
            "Shallow semantic surface: "
            f"{_value(shallow.get('reason'))}; provider={_value(shallow.get('provider'))}; "
            f"elements={_value(shallow.get('element_count'))}; "
            f"depth={_value(shallow.get('maximum_depth'))}; "
            f"actionable={_value(shallow.get('actionable_count'))}; "
            f"text={_value(shallow.get('text_bearing_count'))}"
        )
    inconclusive = inspection.get("semantic_surface_inconclusive")
    if isinstance(inconclusive, dict):
        lines.append(
            "Inconclusive semantic surface: "
            f"{_value(inconclusive.get('reason'))}; "
            f"candidate={_value(inconclusive.get('candidate_reason'))}; "
            f"provider={_value(inconclusive.get('provider'))}; "
            f"elements={_value(inconclusive.get('element_count'))}"
        )
    action_unavailable = inspection.get("semantic_action_unavailable")
    if isinstance(action_unavailable, dict):
        lines.append(
            "Semantic actions unavailable: "
            f"{_value(action_unavailable.get('reason'))}; "
            f"provider={_value(action_unavailable.get('provider'))}"
        )
    fallbacks = inspection.get("fallbacks") or {}
    for kind in ("observation", "action"):
        fallback = fallbacks.get(kind)
        if isinstance(fallback, dict):
            lines.append(
                f"{kind.capitalize()} fallback: "
                f"{_value(fallback.get('kind'))}; "
                f"status={_value(fallback.get('status'))}; "
                f"reason={_value(fallback.get('reason') or fallback.get('when'))}"
            )
    return "\n".join(lines) + "\n"


def render_read(data: dict[str, Any]) -> str:
    reading = data.get("reading") or {}
    metadata = (
        f"Tier: {_value(reading.get('observation_tier'))}; "
        f"ref={_value(reading.get('element_ref'))}; "
        f"ref-status={_value(reading.get('element_ref_status'))}; "
        f"expires={_value(reading.get('expires_at'))}; "
        f"ref-reason={_value(reading.get('element_ref_reason'))}\n"
    )
    if reading.get("redacted") and reading.get("text") is None:
        return (
            f"Element {_value(reading.get('element'))}: redacted "
            f"[{_value(reading.get('redaction_reason'))}]\n{metadata}"
        )
    suffix = " (truncated)" if reading.get("truncated") else ""
    text = _terminal_text(reading.get("text") or "")
    return (
        f"Element {_value(reading.get('element'))}: "
        f"{_value(reading.get('char_count'))} chars{suffix}\n{metadata}{text}\n"
    )


def render_scroll_areas(data: dict[str, Any]) -> str:
    scroll = data.get("scroll_areas") or {}
    window = scroll.get("window") or {}
    count = int(scroll.get("count") or 0)
    suffix = " (truncated)" if scroll.get("truncated") else ""
    lines = [
        f"Scroll areas ({count}){suffix}: HWND {_value(window.get('hwnd'))}",
        f"Observation tier: {_value(scroll.get('observation_tier'))}",
    ]
    for area in scroll.get("areas") or []:
        horizontal = area.get("horizontal") or {}
        vertical = area.get("vertical") or {}
        lines.append(
            "  "
            f"{_value(area.get('element'))} {_value(area.get('control_type'))} "
            f"viewport={_value((area.get('viewport') or {}).get('width'))}x"
            f"{_value((area.get('viewport') or {}).get('height'))} "
            f"h={_axis_text(horizontal)} v={_axis_text(vertical)}"
            f" ref={_value(area.get('element_ref'))}"
            f" ref-status={_value(area.get('element_ref_status'))}"
            f" expires={_value(area.get('expires_at'))}"
            f" ref-reason={_value(area.get('element_ref_reason'))}"
            + (f" warning={_value(area.get('warning'))}" if area.get("warning") else "")
        )
    warnings = scroll.get("degradation_warnings") or []
    if warnings:
        lines.append("Warnings: " + ",".join(_terminal_text(item) for item in warnings))
    return "\n".join(lines) + "\n"


def render_capabilities(data: dict[str, Any]) -> str:
    capabilities = data.get("capabilities") or {}
    backend = capabilities.get("uia_winapp") or {}
    status = "ready" if backend.get("available") else _value(backend.get("error_code"))
    lines = [
        "Computer capabilities: " + ",".join(capabilities.get("commands") or []),
        "WinApp UIA: "
        f"{status}; required="
        f"{_value(backend.get('required_version') or backend.get('version'))}; "
        f"runtime auto-download=off; actions={'on' if backend.get('actions_exposed') else 'off'}",
    ]
    native = capabilities.get("uia_native") or {}
    native_status = "ready" if native.get("available") else _value(native.get("status"))
    lines.append(
        "Native UIA diagnostics: "
        f"{native_status}; views={','.join(native.get('views') or [])}; actions=off"
    )
    ocr = capabilities.get("ocr") or {}
    lines.append(
        "Local terminal OCR: "
        f"{'ready' if ocr.get('available') else 'unavailable'}; "
        "native-capture-only=yes; implicit-capture=no; confidence=unavailable"
    )
    legacy = capabilities.get("legacy_accessibility") or {}
    for name in ("msaa", "ia2"):
        item = legacy.get(name) or {}
        lines.append(
            f"{name.upper()}: {_value(item.get('status'))}; runtime provider=off"
        )
    mutations = capabilities.get("mutations") or {}
    if mutations.get("disabled"):
        mutation_status = "disabled (emergency stop)"
    elif mutations.get("available"):
        mutation_status = (
            "available (semantic first; explicit capture-bound physical fallback)"
            if mutations.get("physical_input")
            else "available (semantic only; physical input unavailable)"
        )
    else:
        mutation_status = "unavailable"
    lines.append("Mutations: " + mutation_status)
    lines.append(
        "Action lock: "
        f"{mutations.get('mutex') or 'unavailable'}; "
        f"busy={mutations.get('busy_behavior') or 'unknown'}"
    )
    if not backend.get("available") and backend.get("install_hint"):
        lines.append("Install: " + _terminal_text(backend.get("install_hint")))
    return "\n".join(lines) + "\n"


def render_action(data: dict[str, Any]) -> str:
    action = data.get("action") or {}
    lines = [
        (
            f"Computer action {action.get('operation') or 'unknown'}: "
            f"{action.get('outcome') or 'unknown'} via {action.get('method') or 'unknown'}"
        ),
        f"  operation id: {action.get('operation_id') or 'unknown'}",
    ]
    if action.get("short_explanation"):
        lines.append(f"  reason: {_terminal_text(action.get('short_explanation'))}")
    notification = action.get("notification") or {}
    if notification.get("requested"):
        lines.append(f"  notification: {notification.get('status') or 'unknown'}")
    warnings = action.get("warnings") or []
    if warnings:
        lines.append("  warnings: " + ",".join(_terminal_text(item) for item in warnings))
    lines.append(f"  control tier: {_value(action.get('control_tier'))}")
    focus = action.get("focus")
    if isinstance(focus, dict):
        lines.extend(_focus_lines(focus, prefix="  focus"))
    if any(
        key in action for key in ("requested_state", "state_before", "state_after")
    ):
        lines.append(
            "  state: "
            f"requested={_value(action.get('requested_state'))} "
            f"before={_value(action.get('state_before'))} "
            f"after={_value(action.get('state_after'))} "
            f"verified={_yes_no(action.get('postcondition_verified'))}"
        )
    if "restore_performed" in action or "restored_to_normal" in action:
        lines.append(
            "  restore: "
            f"performed={_yes_no(action.get('restore_performed'))} "
            f"normal={_yes_no(action.get('restored_to_normal'))}"
        )
    if action.get("requested_bounds") or action.get("actual_bounds"):
        lines.append(
            "  bounds: "
            f"requested={_bounds(action.get('requested_bounds'))} "
            f"actual={_bounds(action.get('actual_bounds'))} "
            f"restore-first={_yes_no(action.get('restore_first'))} "
            f"restored-first={_yes_no(action.get('restored_first'))} "
            f"verified={_yes_no(action.get('postcondition_verified'))}"
        )
    if action.get("resolution_stage"):
        lines.append(f"  resolution stage: {_value(action.get('resolution_stage'))}")
    if action.get("verification_resolution_stage"):
        lines.append(
            "  verification resolution stage: "
            f"{_value(action.get('verification_resolution_stage'))}"
        )
    if action.get("element_ref"):
        lines.append(
            "  element ref: "
            f"{_value(action.get('element_ref'))}; "
            f"expires={_value(action.get('element_ref_expires_at'))}"
        )
    fallback = action.get("fallback")
    if isinstance(fallback, dict):
        lines.append(
            "  fallback: "
            f"{_value(fallback.get('kind'))}; status={_value(fallback.get('status'))}; "
            f"reason={_value(fallback.get('reason'))}"
        )
    elif "fallback_used" in action:
        lines.append(f"  fallback used: {_yes_no(action.get('fallback_used'))}")
    delivery = action.get("delivery")
    if isinstance(delivery, dict):
        lines.append(
            "  delivery: "
            f"status={_value(delivery.get('status'))} "
            f"method={_value(delivery.get('method'))}"
        )
    consequence = action.get("consequence")
    if isinstance(consequence, dict):
        lines.append(
            "  observed consequence: "
            f"effect={_value(consequence.get('observed_effect'))} "
            f"expectation={_value(consequence.get('expectation'))} "
            f"settle={_value(consequence.get('settle_window_ms'))}ms"
        )
        delta = consequence.get("target_delta")
        if isinstance(delta, dict) and delta:
            lines.append("  changed fields: " + ",".join(sorted(delta)[:12]))
    if action.get("staged_input"):
        lines.append(
            "  staged input: "
            f"{_value(action.get('staged_input'))}; "
            f"chars={_value(action.get('text_char_count'))}; "
            f"ttl={_value(action.get('ttl_seconds'))}s; "
            f"expires={_value(action.get('expires_at'))}"
        )
    if "delivered_character_count" in action:
        lines.append(
            "  staged characters: "
            f"{_value(action.get('delivered_character_count'))}/"
            f"{_value(action.get('intended_character_count'))}; "
            f"first-undelivered={_value(action.get('first_undelivered_position'))}"
        )
    return "\n".join(lines) + "\n"


def _focus_lines(focus: dict[str, Any], *, prefix: str) -> list[str]:
    lines = [
        (
            f"{prefix}: outcome={_value(focus.get('outcome'))} "
            f"method={_value(focus.get('method'))} "
            f"foreground={_value(focus.get('foreground_hwnd'))} "
            f"state={_value(focus.get('state_before'))}->{_value(focus.get('state_after'))} "
            f"verified={_yes_no(focus.get('postcondition_verified'))}"
        )
    ]
    if "restore_performed" in focus or "restored_to_normal" in focus:
        lines.append(
            f"{prefix} restore: "
            f"performed={_yes_no(focus.get('restore_performed'))} "
            f"normal={_yes_no(focus.get('restored_to_normal'))}"
        )
    return lines


def _window_line(window: dict[str, Any]) -> str:
    title = _terminal_text(window.get("title") or "-")
    return (
        f"hwnd={_value(window.get('hwnd'))} pid={_value(window.get('pid'))} "
        f"process={_value(window.get('process'))} state={_value(window.get('state'))} "
        f"bounds={_bounds(window.get('bounds'))} title={title}"
    )


def _unavailable_line(label: str, section: dict[str, Any]) -> str:
    return f"{label}: unavailable [{_value(section.get('error_code'))}]"


def _bounds(value: object) -> str:
    if not isinstance(value, dict):
        return "-"
    return (
        f"{value.get('x', '-')},{value.get('y', '-')} "
        f"{value.get('width', '-')}x{value.get('height', '-')}"
    )


def _value(value: object) -> str:
    return "-" if value in (None, "") else _terminal_text(value)


def _yes_no(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _percent(value: object) -> str:
    return "-" if value is None else f"{value}%"


def _duration(value: object) -> str:
    try:
        seconds = max(0, int(str(value)))
    except (TypeError, ValueError):
        return "-"
    days, remainder = divmod(seconds, 86_400)
    hours, minutes = divmod(remainder, 3_600)
    return f"{days}d{hours}h{minutes // 60}m"


def _axis_text(axis: dict[str, Any]) -> str:
    if not axis.get("scrollable"):
        return "no-scroll"
    percent = axis.get("percent")
    if percent is None:
        return "?%"
    before = "before" if axis.get("moreBefore") else "top"
    after = "after" if axis.get("moreAfter") else "end"
    return f"{percent}%/{before}/{after}"


def _format_offset(value: object) -> str:
    try:
        seconds = int(str(value))
    except (TypeError, ValueError):
        return "UTC?"
    sign = "+" if seconds >= 0 else "-"
    hours, remainder = divmod(abs(seconds), 3_600)
    return f"UTC{sign}{hours:02d}:{remainder // 60:02d}"


def _join_nonempty(*values: object, sep: str) -> str:
    return sep.join(str(value) for value in values if value not in (None, ""))


def _backend_status(name: str, details: dict[str, Any]) -> str:
    if details.get("available"):
        status = "ready"
    else:
        status = details.get("error_code") or details.get("status") or "off"
    return f"{name}:{status}"


def _action_readiness(readiness: dict[str, Any]) -> str:
    if readiness.get("actions_disabled"):
        return "disabled"
    if readiness.get("actions_available"):
        return "available"
    return "unavailable"


def _terminal_text(value: object) -> str:
    output: list[str] = []
    for character in str(value):
        if unicodedata.category(character) in {"Cc", "Cf"}:
            codepoint = ord(character)
            if codepoint <= 0xFF:
                output.append(f"\\x{codepoint:02x}")
            elif codepoint <= 0xFFFF:
                output.append(f"\\u{codepoint:04x}")
            else:
                output.append(f"\\U{codepoint:08x}")
        else:
            output.append(character)
    return "".join(output)
