from __future__ import annotations

import ctypes
import io
import sys
import tempfile
from ctypes import wintypes
from typing import Any

MIN_PROCESS_LIFECYCLE_SECONDS = 0.25


class ProcessInputStagingError(Exception):
    def __init__(self, stage_error: str, cleanup_error: str | None) -> None:
        super().__init__("Process input staging failed.")
        self.stage_error = stage_error
        self.cleanup_error = cleanup_error


def preloaded_process_stdin(payload: bytes) -> Any:
    """Return an unbuffered, delete-on-close file positioned for child stdin."""
    stream = tempfile.TemporaryFile(
        mode="w+b",
        buffering=0,
        prefix="agent-tools-process-input-",
    )
    try:
        view = memoryview(payload)
        while view:
            written = stream.write(view)
            if written is None or written <= 0:
                raise OSError("process input could not be staged")
            view = view[written:]
        stream.seek(0)
    except (OSError, ValueError) as exc:
        cleanup_error = close_preloaded_process_stdin(stream)
        raise ProcessInputStagingError(
            type(exc).__name__,
            cleanup_error,
        ) from exc
    return stream


def close_preloaded_process_stdin(stream: Any) -> str | None:
    """Close staged child input and return a bounded error name on failure."""
    if stream is None or stream.closed:
        return None
    try:
        stream.close()
    except OSError as exc:
        primary_error = type(exc).__name__
        underlying = getattr(stream, "file", None)
        if underlying is not None and not underlying.closed:
            try:
                if isinstance(underlying, io.FileIO):
                    io.FileIO.close(underlying)
                else:
                    underlying.close()
            except OSError as fallback:
                if not underlying.closed:
                    return f"{primary_error}+{type(fallback).__name__}"
        return primary_error
    return None


def close_stream_os_enforced(stream: Any) -> str | None:
    """Cancel pending pipe I/O, then close exactly once through its Python owner."""
    if stream is None or stream.closed:
        return None
    if sys.platform != "win32":
        try:
            stream.close()
        except OSError as exc:
            return type(exc).__name__
        return None

    cancellation_error = _cancel_stream_io(stream)
    try:
        if isinstance(stream, io.BufferedReader):
            io.BufferedReader.close(stream)
        elif isinstance(stream, io.BufferedWriter):
            io.BufferedWriter.close(stream)
        elif isinstance(stream, io.BufferedRandom):
            io.BufferedRandom.close(stream)
        elif isinstance(stream, io.FileIO):
            io.FileIO.close(stream)
        else:
            stream.close()
    except OSError as exc:
        if not stream.closed:
            return type(exc).__name__
    return cancellation_error


def _cancel_stream_io(stream: Any) -> str | None:
    import msvcrt

    try:
        file_descriptor = stream.fileno()
        raw_handle = msvcrt.get_osfhandle(file_descriptor)
    except (OSError, ValueError, io.UnsupportedOperation) as exc:
        return type(exc).__name__

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CancelIoEx.argtypes = (wintypes.HANDLE, wintypes.LPVOID)
    kernel32.CancelIoEx.restype = wintypes.BOOL
    if kernel32.CancelIoEx(wintypes.HANDLE(raw_handle), None):
        return None
    error = ctypes.get_last_error()
    return None if error in {0, 6, 1168} else f"CancelIoEx:{error}"
