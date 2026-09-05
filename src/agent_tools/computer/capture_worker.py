from __future__ import annotations

import json
import sys
from typing import Any

from agent_tools.computer.models import Bounds, ComputerError, WindowIdentity
from agent_tools.computer.screenshot import (
    _capture_desktop_full,
    _capture_full,
    _encode_png,
)
from agent_tools.computer.win32_backend import Win32Backend

MAX_REQUEST_BYTES = 128 * 1024


def main() -> int:
    try:
        request = _read_request()
        identity = _parse_identity(request.get("identity"))
        mode = request.get("mode", "print_window")
        if mode == "print_window":
            captured = _capture_full(Win32Backend(), identity)
        elif mode == "desktop_crop":
            captured = _capture_desktop_full(Win32Backend(), identity)
        else:
            raise ComputerError(
                "capture_worker_request_invalid",
                "The capture worker mode is invalid.",
            )
        payload = _encode_png(captured.image)
        _write_success(
            {
                "ok": True,
                "width": captured.width,
                "height": captured.height,
                "captured_unix_ms": captured.captured_unix_ms,
                "cleanup_verified": captured.cleanup_verified,
                "png_bytes": len(payload),
            },
            payload,
        )
        return 0
    except ComputerError as exc:
        _write_response(
            {
                "ok": False,
                "error": {"code": exc.code, "message": exc.message},
            }
        )
        return exc.exit_code
    except Exception:  # noqa: BLE001 - hidden worker must not leak a traceback
        _write_response(
            {
                "ok": False,
                "error": {
                    "code": "capture_worker_failed",
                    "message": "The capture worker failed unexpectedly.",
                },
            }
        )
        return 1


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ComputerError(
            "capture_worker_request_too_large",
            "The capture worker request exceeded its safety limit.",
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComputerError(
            "capture_worker_request_invalid",
            "The capture worker request is invalid.",
        ) from exc
    if not isinstance(value, dict):
        raise ComputerError(
            "capture_worker_request_invalid",
            "The capture worker request is invalid.",
        )
    return value


def _parse_identity(value: object) -> WindowIdentity:
    if not isinstance(value, dict) or not isinstance(value.get("bounds"), dict):
        raise ComputerError(
            "capture_worker_request_invalid",
            "The capture identity is invalid.",
        )
    bounds = value["bounds"]
    try:
        return WindowIdentity(
            hwnd=int(value["hwnd"]),
            pid=int(value["pid"]),
            thread_id=int(value["thread_id"]),
            process=str(value["process"]) if value.get("process") is not None else None,
            title=str(value["title"]),
            class_name=str(value["class_name"]),
            bounds=Bounds(
                x=int(bounds["x"]),
                y=int(bounds["y"]),
                width=int(bounds["width"]),
                height=int(bounds["height"]),
            ),
            process_started=(
                int(value["process_started"])
                if value.get("process_started") is not None
                else None
            ),
            session_id=(
                int(value["session_id"])
                if value.get("session_id") is not None
                else None
            ),
            desktop_name=(
                str(value["desktop_name"])
                if value.get("desktop_name") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ComputerError(
            "capture_worker_request_invalid",
            "The capture identity is invalid.",
        ) from exc


def _write_response(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(json.dumps(value, separators=(",", ":")).encode("ascii"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


def _write_success(header: dict[str, Any], payload: bytes) -> None:
    sys.stdout.buffer.write(json.dumps(header, separators=(",", ":")).encode("ascii"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    raise SystemExit(main())
