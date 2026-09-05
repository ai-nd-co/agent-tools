from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from agent_tools.computer.actions import (
    MAX_VALUE_CHARS,
    action_capabilities,
    click_pointer,
    focus_window,
    invoke_element,
    move_pointer,
    notify_user,
    physical_shortcut,
    physical_wheel,
    press_staged_enter,
    resize_window,
    scroll_element,
    set_element_value,
    set_window_state,
    stage_physical_text,
)
from agent_tools.computer.models import ComputerError
from agent_tools.computer.ocr import (
    DEFAULT_MAX_OCR_CHARS,
    DEFAULT_MAX_OCR_LINES,
    MAX_OCR_CHARS,
    MAX_OCR_LINES,
    ocr_capabilities,
    ocr_capture,
)
from agent_tools.computer.physical_input import MAX_PHYSICAL_TEXT_CHARS
from agent_tools.computer.providers import collect_info
from agent_tools.computer.rendering import (
    render_action,
    render_capabilities,
    render_focused,
    render_human_error,
    render_info,
    render_inspect,
    render_json,
    render_json_error,
    render_ocr,
    render_read,
    render_screenshot,
    render_scroll_areas,
    render_windows,
)
from agent_tools.computer.screenshot import capture_screenshot, parse_region
from agent_tools.computer.semantic_accessibility import (
    MAX_SEMANTIC_DEPTH,
    inspect_semantic_window,
    read_semantic_element,
    semantic_capabilities,
)
from agent_tools.computer.uia_winapp import (
    MAX_INSPECT_ELEMENTS,
    MAX_READ_CHARS,
    MAX_SCROLL_AREAS,
    capabilities,
    list_scroll_areas,
)
from agent_tools.computer.win32_backend import Win32Backend


def add_computer_parser(subparsers: Any) -> None:
    computer_parser = subparsers.add_parser(
        "computer",
        help="Inspect Windows and perform guarded actions on exact targets.",
        description=(
            "Windows inspection is read-only; focus, resize, UI Automation, and notification "
            "commands are guarded mutations. Capture-bound physical input is an explicit "
            "last resort. Human output is compact; --json emits the versioned machine "
            "contract. Background processes are hidden unless requested."
        ),
        epilog=(
            "There is no built-in grep. Filter human output with `| rg PATTERN` or JSON "
            "with `| jq ...`."
        ),
    )
    commands = computer_parser.add_subparsers(dest="computer_command", required=True)

    info_parser = commands.add_parser(
        "info",
        help="Show compact system, session, focus, app, media, network, and readiness state.",
        description=(
            "Show compact best-effort computer context. Background processes are hidden by "
            "default; use --background to include bounded PID/name/session/state rows."
        ),
        epilog="Filter with normal shell pipes: `| rg PATTERN` or `--json | jq ...`.",
    )
    info_parser.add_argument(
        "--background",
        action="store_true",
        help="Include bounded background process rows (never command lines).",
    )
    info_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    windows_parser = commands.add_parser(
        "windows",
        help="List visible titled top-level windows.",
        epilog="Filter with `| rg PATTERN` or `--json | jq ...`.",
    )
    windows_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    focused_parser = commands.add_parser(
        "focused",
        help="Inspect the current foreground window.",
    )
    focused_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    screenshot_parser = commands.add_parser(
        "screenshot",
        help="Capture one exact visible top-level HWND without changing focus or the cursor.",
        description=(
            "Capture a caller-supplied HWND to a caller-supplied PNG path. Standard 720px "
            "presentation is the default; compact is 480px and native is explicit. Regions "
            "use native window-relative pixels. The exact target and DPI geometry are bound "
            "to a short-lived owner-private capture record."
        ),
    )
    screenshot_parser.add_argument("--hwnd", required=True, type=int, help="Target HWND integer.")
    screenshot_parser.add_argument("--output", required=True, type=Path, help="Output .png path.")
    screenshot_parser.add_argument(
        "--region",
        help="Optional window-relative x,y,width,height region.",
    )
    screenshot_parser.add_argument(
        "--profile",
        choices=("compact", "standard", "native"),
        default="standard",
        help="Presentation profile: compact 480px, standard 720px (default), or explicit native.",
    )
    screenshot_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    ocr_parser = commands.add_parser(
        "ocr",
        help="Read bounded text from an existing exact-window native screenshot capture.",
        description=(
            "Run local Windows OCR only on the unchanged profile-native PNG bound to an "
            "existing capture ID. This command never captures a window or desktop."
        ),
        epilog="Filter text with `| rg PATTERN` or JSON with `--json | jq ...`.",
    )
    ocr_parser.add_argument(
        "--from-capture",
        required=True,
        help="Fresh profile-native screenshot capture ID.",
    )
    ocr_parser.add_argument(
        "--last-lines",
        type=_bounded_integer("last-lines", 1, MAX_OCR_LINES),
        help=f"Return only the final N recognized lines (maximum {MAX_OCR_LINES}).",
    )
    ocr_parser.add_argument(
        "--max-lines",
        type=_bounded_integer("max-lines", 1, MAX_OCR_LINES),
        default=DEFAULT_MAX_OCR_LINES,
        help=f"Bound returned lines (default {DEFAULT_MAX_OCR_LINES}).",
    )
    ocr_parser.add_argument(
        "--max-chars",
        type=_bounded_integer("max-chars", 1, MAX_OCR_CHARS),
        default=DEFAULT_MAX_OCR_CHARS,
        help=f"Bound returned characters (default {DEFAULT_MAX_OCR_CHARS}).",
    )
    ocr_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    inspect_parser = commands.add_parser(
        "inspect",
        help="Inspect a bounded semantic UI Automation tree for one exact HWND.",
        description=(
            "Return visible semantic elements only. Element values are omitted; use read with "
            "an inspect-provided slug for bounded text."
        ),
    )
    inspect_parser.add_argument("--hwnd", required=True, type=int, help="Target HWND integer.")
    inspect_parser.add_argument(
        "--depth",
        type=_bounded_integer("depth", 1, MAX_SEMANTIC_DEPTH),
        default=3,
        help=(
            f"UIA tree depth from 1 through {MAX_SEMANTIC_DEPTH} (default: 3); "
            "depth 8+ enables bounded provider diagnostics for shallow surfaces."
        ),
    )
    inspect_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Return only interactive/invokable elements.",
    )
    inspect_parser.add_argument(
        "--max-elements",
        type=_bounded_integer("max-elements", 1, MAX_INSPECT_ELEMENTS),
        default=50,
        help=f"Maximum returned elements from 1 through {MAX_INSPECT_ELEMENTS} (default: 50).",
    )
    _add_title_match(inspect_parser)
    inspect_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    read_parser = commands.add_parser(
        "read",
        help="Read bounded semantic text from one inspect-provided element slug.",
        description=(
            "Password controls are never read. Credential-like values are redacted and output "
            "is bounded before it reaches the caller."
        ),
    )
    read_parser.add_argument("--hwnd", required=True, type=int, help="Target HWND integer.")
    read_parser.add_argument("--element", required=True, help="Element slug from inspect output.")
    read_parser.add_argument(
        "--max-chars",
        type=_bounded_integer("max-chars", 1, MAX_READ_CHARS),
        default=4_000,
        help=f"Maximum returned characters from 1 through {MAX_READ_CHARS} (default: 4000).",
    )
    _add_title_match(read_parser)
    read_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    scroll_parser = commands.add_parser(
        "scroll-areas",
        help="List semantic scroll containers and conservative remaining directions.",
        description=(
            "Read ScrollPattern state only. This command does not scroll, focus, click, type, or "
            "inject input."
        ),
    )
    scroll_parser.add_argument("--hwnd", required=True, type=int, help="Target HWND integer.")
    scroll_parser.add_argument(
        "--max-elements",
        type=_bounded_integer("max-elements", 1, MAX_SCROLL_AREAS),
        default=20,
        help=f"Maximum returned scroll areas from 1 through {MAX_SCROLL_AREAS} (default: 20).",
    )
    _add_title_match(scroll_parser)
    scroll_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    capabilities_parser = commands.add_parser(
        "capabilities",
        help="Show computer backends, the pinned WinApp version, and missing-backend hints.",
    )
    capabilities_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    focus_parser = commands.add_parser(
        "focus",
        help="Restore and focus one exact HWND with a verified foreground postcondition.",
    )
    focus_parser.add_argument("--hwnd", required=True, type=int, help="Target HWND integer.")
    _add_title_match(focus_parser)
    _add_short_explanation(focus_parser)
    focus_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    for command, help_text in (
        ("restore", "Restore one exact HWND to the normal window state."),
        ("minimize", "Minimize one exact HWND and verify the final state."),
        ("maximize", "Maximize one exact HWND and verify the final state."),
    ):
        state_parser = commands.add_parser(command, help=help_text)
        state_parser.add_argument(
            "--hwnd", required=True, type=int, help="Target HWND integer."
        )
        _add_title_match(state_parser)
        _add_short_explanation(state_parser)
        state_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    resize_parser = commands.add_parser(
        "resize",
        help="Move and resize one exact normal HWND inside the virtual desktop.",
    )
    resize_parser.add_argument("--hwnd", required=True, type=int, help="Target HWND integer.")
    resize_parser.add_argument("--x", required=True, type=int)
    resize_parser.add_argument("--y", required=True, type=int)
    resize_parser.add_argument("--width", required=True, type=int)
    resize_parser.add_argument("--height", required=True, type=int)
    resize_parser.add_argument(
        "--restore-first",
        action="store_true",
        help="Restore a minimized or maximized target before applying geometry.",
    )
    _add_title_match(resize_parser)
    _add_short_explanation(resize_parser)
    resize_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    invoke_parser = commands.add_parser(
        "invoke",
        help="Invoke one inspect-provided UIA element without physical input.",
    )
    invoke_parser.add_argument("--hwnd", required=True, type=int, help="Target HWND integer.")
    invoke_target = invoke_parser.add_mutually_exclusive_group(required=True)
    invoke_target.add_argument("--element", help="Legacy exact element slug from inspect.")
    invoke_target.add_argument(
        "--element-ref", help="Opaque short-lived element reference from inspect."
    )
    _add_title_match(invoke_parser)
    _add_short_explanation(invoke_parser)
    invoke_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    set_value_parser = commands.add_parser(
        "set-value",
        help="Set and read back one non-password UIA control value.",
        description=(
            "Use --stdin for sensitive or shell-hostile values. Values are never returned, "
            "journaled, or included in notifications."
        ),
    )
    set_value_parser.add_argument("--hwnd", required=True, type=int, help="Target HWND integer.")
    set_target = set_value_parser.add_mutually_exclusive_group(required=True)
    set_target.add_argument("--element", help="Legacy exact element slug from inspect.")
    set_target.add_argument(
        "--element-ref", help="Opaque short-lived element reference from inspect."
    )
    value_source = set_value_parser.add_mutually_exclusive_group(required=True)
    value_source.add_argument("--value", help="Value to set (prefer --stdin for sensitive data).")
    value_source.add_argument(
        "--stdin",
        action="store_true",
        help=f"Read up to {MAX_VALUE_CHARS} value characters from standard input.",
    )
    set_value_parser.add_argument(
        "--require-keyboard-focus",
        action="store_true",
        help="Require the exact UIA element to own keyboard focus before and after setting.",
    )
    _add_title_match(set_value_parser)
    _add_short_explanation(set_value_parser)
    set_value_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    action_scroll_parser = commands.add_parser(
        "scroll",
        help="Scroll one exact UIA element through ScrollPattern only.",
        description=(
            "No wheel or pointer fallback is used. Percent targets use UI Automation's "
            "ScrollPattern and must verify within tolerance."
        ),
    )
    action_scroll_parser.add_argument(
        "--hwnd", required=True, type=int, help="Target HWND integer."
    )
    scroll_target = action_scroll_parser.add_mutually_exclusive_group(required=True)
    scroll_target.add_argument(
        "--element", help="Legacy exact scroll element slug from scroll-areas."
    )
    scroll_target.add_argument(
        "--element-ref", help="Opaque short-lived element reference from scroll-areas."
    )
    scroll_request = action_scroll_parser.add_mutually_exclusive_group(required=True)
    scroll_request.add_argument("--direction", choices=("up", "down", "left", "right"))
    scroll_request.add_argument("--to", choices=("top", "bottom"))
    scroll_request.add_argument(
        "--percent",
        type=_bounded_float("percent", 0.0, 100.0),
        help="Target the primary scroll axis from 0 through 100 percent.",
    )
    _add_title_match(action_scroll_parser)
    _add_short_explanation(action_scroll_parser)
    action_scroll_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    for command, help_text in (
        ("move-pointer", "Move the pointer to one fresh capture-derived point."),
        ("click", "Click one fresh capture-derived point."),
    ):
        pointer_parser = commands.add_parser(command, help=help_text)
        _add_physical_target(pointer_parser, point=True)
        if command == "click":
            pointer_parser.add_argument("--double", action="store_true")
        _add_short_explanation(pointer_parser)
        pointer_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    type_parser = commands.add_parser(
        "type",
        help="Stage bounded text without Enter using guarded physical virtual keys.",
    )
    _add_physical_target(type_parser, point=True)
    text_source = type_parser.add_mutually_exclusive_group(required=True)
    text_source.add_argument("--text", help="Text to stage without Enter.")
    text_source.add_argument("--stdin", action="store_true", help="Read staged text from stdin.")
    type_parser.add_argument(
        "--visible-tab-verified",
        action="store_true",
        help="Confirm the caller verified visible terminal tab content from this capture.",
    )
    _add_short_explanation(type_parser)
    type_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    key_parser = commands.add_parser(
        "key",
        help="Send Enter once for an exact staged-input transaction.",
    )
    _add_physical_target(key_parser, point=True)
    key_parser.add_argument("--key", required=True, choices=("ENTER",))
    key_parser.add_argument("--staged-input", required=True)
    key_parser.add_argument(
        "--draft-evidence",
        required=True,
        choices=("exact", "collapsed_visible_match"),
    )
    key_parser.add_argument("--visible-tab-verified", action="store_true")
    _add_short_explanation(key_parser)
    key_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    shortcut_parser = commands.add_parser(
        "shortcut",
        help="Send one bounded Ctrl, Shift, or Alt shortcut to a capture-bound window.",
    )
    _add_physical_target(shortcut_parser, point=False)
    shortcut_parser.add_argument("--keys", required=True, help="Example: CTRL+K")
    shortcut_parser.add_argument("--visible-tab-verified", action="store_true")
    _add_short_explanation(shortcut_parser)
    shortcut_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    wheel_parser = commands.add_parser(
        "wheel",
        help="Send bounded vertical or horizontal wheel input at a capture-derived point.",
    )
    _add_physical_target(wheel_parser, point=True)
    wheel_request = wheel_parser.add_mutually_exclusive_group(required=True)
    wheel_request.add_argument("--vertical", type=int)
    wheel_request.add_argument("--horizontal", type=int)
    _add_short_explanation(wheel_parser)
    wheel_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")

    notify_parser = commands.add_parser(
        "notify",
        help="Show a non-blocking Windows popup notification.",
    )
    notify_parser.add_argument(
        "--short-explanation",
        required=True,
        help="Required concise notification text (1-160 characters).",
    )
    notify_parser.add_argument("--title", help="Optional short notification title.")
    notify_parser.add_argument("--json", action="store_true", help="Emit versioned JSON.")


def run_computer(args: argparse.Namespace) -> int:
    command = str(args.computer_command)
    json_output = bool(getattr(args, "json", False))
    try:
        data = _run_command(command, args)
    except ComputerError as exc:
        if json_output:
            sys.stdout.write(render_json_error(command, exc))
        else:
            sys.stderr.write(render_human_error(exc))
        return exc.exit_code
    except Exception:  # noqa: BLE001 - CLI must not expose internal tracebacks
        error = ComputerError(
            "internal_error",
            "The computer command failed unexpectedly.",
        )
        if json_output:
            sys.stdout.write(render_json_error(command, error))
        else:
            sys.stderr.write(render_human_error(error))
        return error.exit_code

    if json_output:
        sys.stdout.write(render_json(command, data))
    elif command == "info":
        sys.stdout.write(render_info(data, background=bool(args.background)))
    elif command == "windows":
        sys.stdout.write(render_windows(data))
    elif command == "focused":
        sys.stdout.write(render_focused(data))
    elif command == "screenshot":
        sys.stdout.write(render_screenshot(data))
    elif command == "ocr":
        sys.stdout.write(render_ocr(data))
    elif command == "inspect":
        sys.stdout.write(render_inspect(data))
    elif command == "read":
        sys.stdout.write(render_read(data))
    elif command == "scroll-areas":
        sys.stdout.write(render_scroll_areas(data))
    elif command == "capabilities":
        sys.stdout.write(render_capabilities(data))
    elif command in {
        "focus",
        "restore",
        "minimize",
        "maximize",
        "resize",
        "invoke",
        "set-value",
        "scroll",
        "move-pointer",
        "click",
        "type",
        "key",
        "shortcut",
        "wheel",
        "notify",
    }:
        sys.stdout.write(render_action(data))
    return 0


def is_computer_json_request(argv: list[str]) -> bool:
    return bool(argv and argv[0] == "computer" and "--json" in argv)


def render_computer_argument_error(argv: list[str]) -> int:
    command = "unknown"
    if len(argv) > 1 and not argv[1].startswith("-"):
        command = argv[1]
    error = ComputerError(
        "invalid_arguments",
        "The computer command arguments are invalid.",
        exit_code=2,
    )
    sys.stdout.write(render_json_error(command, error))
    return error.exit_code


def _run_command(command: str, args: argparse.Namespace) -> dict[str, Any]:
    if command == "ocr":
        return {
            "ocr": ocr_capture(
                capture_id=str(args.from_capture),
                last_lines=args.last_lines,
                max_lines=int(args.max_lines),
                max_chars=int(args.max_chars),
            )
        }
    backend = Win32Backend()
    if command == "info":
        return collect_info(background=bool(args.background), backend=backend)
    if command == "windows":
        return {"windows": backend.list_windows()}
    if command == "focused":
        return {"focus": backend.focused_window()}
    if command == "screenshot":
        return {
            "screenshot": capture_screenshot(
                hwnd=int(args.hwnd),
                output=args.output,
                region=parse_region(args.region),
                profile=str(args.profile),
                backend=backend,
            )
        }
    if command == "inspect":
        return {
            "inspection": inspect_semantic_window(
                hwnd=int(args.hwnd),
                depth=int(args.depth),
                interactive=bool(args.interactive),
                max_elements=int(args.max_elements),
                backend=backend,
                require_title_match=bool(args.require_title_match),
            )
        }
    if command == "read":
        read_kwargs: dict[str, Any] = {
            "hwnd": int(args.hwnd),
            "element": str(args.element),
            "max_chars": int(args.max_chars),
            "backend": backend,
        }
        if args.require_title_match:
            read_kwargs["require_title_match"] = True
        return {"reading": read_semantic_element(**read_kwargs)}
    if command == "scroll-areas":
        scroll_area_kwargs: dict[str, Any] = {
            "hwnd": int(args.hwnd),
            "max_elements": int(args.max_elements),
            "backend": backend,
        }
        if args.require_title_match:
            scroll_area_kwargs["require_title_match"] = True
        return {"scroll_areas": list_scroll_areas(**scroll_area_kwargs)}
    if command == "capabilities":
        result = capabilities()
        result.update(semantic_capabilities())
        result["ocr"] = ocr_capabilities()
        result["mutations"] = action_capabilities()
        return {"capabilities": result}
    if command == "focus":
        return {
            "action": focus_window(
                hwnd=int(args.hwnd),
                short_explanation=args.short_explanation,
                backend=backend,
                require_title_match=bool(args.require_title_match),
            )
        }
    if command in {"restore", "minimize", "maximize"}:
        state = {
            "restore": "normal",
            "minimize": "minimized",
            "maximize": "maximized",
        }[command]
        return {
            "action": set_window_state(
                hwnd=int(args.hwnd),
                state=state,
                short_explanation=args.short_explanation,
                backend=backend,
                require_title_match=bool(args.require_title_match),
            )
        }
    if command == "resize":
        return {
            "action": resize_window(
                hwnd=int(args.hwnd),
                x=int(args.x),
                y=int(args.y),
                width=int(args.width),
                height=int(args.height),
                short_explanation=args.short_explanation,
                backend=backend,
                restore_first=bool(args.restore_first),
                require_title_match=bool(args.require_title_match),
            )
        }
    if command == "invoke":
        return {
            "action": invoke_element(
                hwnd=int(args.hwnd),
                element=str(args.element) if args.element is not None else None,
                element_ref=args.element_ref,
                short_explanation=args.short_explanation,
                backend=backend,
                require_title_match=bool(args.require_title_match),
            )
        }
    if command == "set-value":
        value = _stdin_value() if bool(args.stdin) else str(args.value)
        return {
            "action": set_element_value(
                hwnd=int(args.hwnd),
                element=str(args.element) if args.element is not None else None,
                element_ref=args.element_ref,
                value=value,
                short_explanation=args.short_explanation,
                backend=backend,
                require_title_match=bool(args.require_title_match),
                require_keyboard_focus=bool(args.require_keyboard_focus),
            )
        }
    if command == "scroll":
        return {
            "action": scroll_element(
                hwnd=int(args.hwnd),
                element=str(args.element) if args.element is not None else None,
                element_ref=args.element_ref,
                direction=args.direction,
                to=args.to,
                percent=args.percent,
                short_explanation=args.short_explanation,
                backend=backend,
                require_title_match=bool(args.require_title_match),
            )
        }
    if command == "move-pointer":
        image_x, image_y = _parse_image_point(args.image_at)
        return {
            "action": move_pointer(
                hwnd=int(args.hwnd),
                capture_id=str(args.from_capture),
                image_x=image_x,
                image_y=image_y,
                allow_physical=bool(args.allow_physical),
                short_explanation=args.short_explanation,
                backend=backend,
            )
        }
    if command == "click":
        image_x, image_y = _parse_image_point(args.image_at)
        return {
            "action": click_pointer(
                hwnd=int(args.hwnd),
                capture_id=str(args.from_capture),
                image_x=image_x,
                image_y=image_y,
                double=bool(args.double),
                allow_physical=bool(args.allow_physical),
                short_explanation=args.short_explanation,
                backend=backend,
            )
        }
    if command == "type":
        image_x, image_y = _parse_image_point(args.image_at)
        value = _stdin_physical_text() if bool(args.stdin) else str(args.text)
        return {
            "action": stage_physical_text(
                hwnd=int(args.hwnd),
                capture_id=str(args.from_capture),
                image_x=image_x,
                image_y=image_y,
                text=value,
                allow_physical=bool(args.allow_physical),
                visible_tab_verified=bool(args.visible_tab_verified),
                short_explanation=args.short_explanation,
                backend=backend,
            )
        }
    if command == "key":
        image_x, image_y = _parse_image_point(args.image_at)
        return {
            "action": press_staged_enter(
                hwnd=int(args.hwnd),
                capture_id=str(args.from_capture),
                image_x=image_x,
                image_y=image_y,
                staged_input=str(args.staged_input),
                draft_evidence=str(args.draft_evidence),
                allow_physical=bool(args.allow_physical),
                visible_tab_verified=bool(args.visible_tab_verified),
                short_explanation=args.short_explanation,
                backend=backend,
            )
        }
    if command == "shortcut":
        return {
            "action": physical_shortcut(
                hwnd=int(args.hwnd),
                capture_id=str(args.from_capture),
                keys=str(args.keys),
                allow_physical=bool(args.allow_physical),
                visible_tab_verified=bool(args.visible_tab_verified),
                short_explanation=args.short_explanation,
                backend=backend,
            )
        }
    if command == "wheel":
        image_x, image_y = _parse_image_point(args.image_at)
        return {
            "action": physical_wheel(
                hwnd=int(args.hwnd),
                capture_id=str(args.from_capture),
                image_x=image_x,
                image_y=image_y,
                vertical=args.vertical,
                horizontal=args.horizontal,
                allow_physical=bool(args.allow_physical),
                short_explanation=args.short_explanation,
                backend=backend,
            )
        }
    if command == "notify":
        return {
            "action": notify_user(
                short_explanation=str(args.short_explanation),
                title=args.title,
            )
        }
    raise ComputerError("unknown_command", f"Unknown computer command: {command}", exit_code=2)


def _bounded_integer(label: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc
        if parsed < minimum or parsed > maximum:
            raise argparse.ArgumentTypeError(
                f"{label} must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def _bounded_float(label: str, minimum: float, maximum: float):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be a number") from exc
        if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
            raise argparse.ArgumentTypeError(
                f"{label} must be between {minimum:g} and {maximum:g}"
            )
        return parsed

    return parse


def _add_short_explanation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--short-explanation",
        help="Optional concise reason shown before the action and returned in its result.",
    )


def _add_title_match(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-title-match",
        action="store_true",
        help="Reject if the title changes during this operation; default is title-tolerant.",
    )


def _add_physical_target(parser: argparse.ArgumentParser, *, point: bool) -> None:
    parser.add_argument("--hwnd", required=True, type=int, help="Exact target HWND integer.")
    parser.add_argument("--from-capture", required=True, help="Fresh screenshot capture ID.")
    if point:
        parser.add_argument("--image-at", required=True, help="Presentation point x,y.")
    parser.add_argument(
        "--allow-physical",
        action="store_true",
        help="Explicitly authorize last-resort physical input for this one action.",
    )


def _parse_image_point(value: str) -> tuple[int, int]:
    try:
        parts = [int(item.strip()) for item in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ComputerError(
            "invalid_image_point",
            "Image point must be x,y using integers.",
            exit_code=2,
        ) from exc
    if len(parts) != 2 or any(item < 0 for item in parts):
        raise ComputerError(
            "invalid_image_point",
            "Image point must be two non-negative integers.",
            exit_code=2,
        )
    return parts[0], parts[1]


def _stdin_value() -> str:
    if sys.stdin.isatty():
        raise ComputerError(
            "stdin_required",
            "--stdin requires piped standard input.",
            exit_code=2,
        )
    value = sys.stdin.read(MAX_VALUE_CHARS + 1)
    if len(value) > MAX_VALUE_CHARS:
        raise ComputerError(
            "invalid_value",
            f"The piped value exceeds {MAX_VALUE_CHARS} characters.",
            exit_code=2,
        )
    return value


def _stdin_physical_text() -> str:
    if sys.stdin.isatty():
        raise ComputerError(
            "stdin_required",
            "--stdin requires piped standard input.",
            exit_code=2,
        )
    value = sys.stdin.read(MAX_PHYSICAL_TEXT_CHARS + 1)
    if len(value) > MAX_PHYSICAL_TEXT_CHARS:
        raise ComputerError(
            "invalid_physical_text",
            f"The piped text exceeds {MAX_PHYSICAL_TEXT_CHARS} characters.",
            exit_code=2,
        )
    return value
