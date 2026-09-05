from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "agent-tools.computer/v1"
MAX_TITLE_CHARS = 240
MAX_PROCESS_NAME_CHARS = 128
MAX_CLASS_NAME_CHARS = 128


class ComputerError(RuntimeError):
    """Expected command failure with a stable public error code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        exit_code: int = 1,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = details or {}


@dataclass(frozen=True)
class Bounds:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_ltrb(cls, left: int, top: int, right: int, bottom: int) -> Bounds:
        return cls(x=left, y=top, width=right - left, height=bottom - top)

    def to_jsonable(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class WindowIdentity:
    hwnd: int
    pid: int
    thread_id: int
    process: str | None
    title: str
    class_name: str
    bounds: Bounds
    process_started: int | None
    session_id: int | None = None
    desktop_name: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "hwnd": self.hwnd,
            "pid": self.pid,
            "process": (
                bounded_text(self.process, MAX_PROCESS_NAME_CHARS)
                if self.process is not None
                else None
            ),
            "title": bounded_text(self.title, MAX_TITLE_CHARS),
            "class": bounded_text(self.class_name, MAX_CLASS_NAME_CHARS),
            "bounds": self.bounds.to_jsonable(),
        }


@dataclass(frozen=True)
class NativeCaptureGeometry:
    identity: WindowIdentity
    client_bounds: Bounds
    monitor_bounds: Bounds
    virtual_screen_bounds: Bounds
    window_dpi: int
    system_dpi: int
    window_awareness: str
    capture_thread_awareness: str

    def public(self) -> dict[str, Any]:
        return {
            "coordinate_space": "physical_screen_pixels",
            "window_bounds": self.identity.bounds.to_jsonable(),
            "client_bounds": self.client_bounds.to_jsonable(),
            "monitor_bounds": self.monitor_bounds.to_jsonable(),
            "virtual_screen_bounds": self.virtual_screen_bounds.to_jsonable(),
            "dpi": {
                "window": self.window_dpi,
                "system": self.system_dpi,
                "window_scale": {
                    "numerator": self.window_dpi,
                    "denominator": 96,
                },
                "window_awareness": self.window_awareness,
                "capture_thread_awareness": self.capture_thread_awareness,
            },
        }


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    def to_jsonable(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def bounded_text(value: object, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[: max(0, limit)]
    return text[: limit - 3] + "..."


def available_section(**values: Any) -> dict[str, Any]:
    return {"available": True, **values}


def unavailable_section(error_code: str, **values: Any) -> dict[str, Any]:
    return {"available": False, "error_code": error_code, **values}


def success_envelope(command: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": f"computer.{command}",
        "ok": True,
        "data": data,
        "error": None,
    }


def error_envelope(command: str, error: ComputerError) -> dict[str, Any]:
    error_data: dict[str, Any] = {
        "code": error.code,
        "message": error.message,
    }
    if error.details:
        error_data["details"] = error.details
    return {
        "schema_version": SCHEMA_VERSION,
        "command": f"computer.{command}",
        "ok": False,
        "data": None,
        "error": error_data,
    }
