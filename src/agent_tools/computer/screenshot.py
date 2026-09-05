from __future__ import annotations

import ctypes
import hashlib
import hmac
import importlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_tools.computer.capture_records import (
    CAPTURE_RECORD_SCHEMA,
    CAPTURE_RECORD_TTL_SECONDS,
    CaptureRecordStore,
    _is_reparse_point,
)
from agent_tools.computer.models import (
    Bounds,
    ComputerError,
    NativeCaptureGeometry,
    Region,
    WindowIdentity,
)
from agent_tools.computer.process_io import (
    MIN_PROCESS_LIFECYCLE_SECONDS,
    ProcessInputStagingError,
    close_preloaded_process_stdin,
    close_stream_os_enforced,
    preloaded_process_stdin,
)
from agent_tools.computer.uia_winapp import WinAppAdapter
from agent_tools.computer.win32_backend import (
    Win32Backend,
    _rectangle_covered_by_monitors,
    _windll,
)

MAX_CAPTURE_DIMENSION = 32_768
MAX_CAPTURE_PIXELS = 100_000_000
MAX_CAPTURE_WORKER_ERROR_BYTES = 16 * 1024
MAX_CAPTURE_PROTOCOL_OVERHEAD_BYTES = 16 * 1024
MAX_CAPTURE_WORKER_REQUEST_BYTES = 128 * 1024
CAPTURE_TIMEOUT_SECONDS = 4.0
MIN_CAPTURE_BACKEND_START_SECONDS = MIN_PROCESS_LIFECYCLE_SECONDS
FATAL_CAPTURE_ATTEMPT_ERRORS = {
    "capture_cleanup_failed",
    "capture_worker_termination_failed",
    "dpi_awareness_restore_failed",
    "uia_cleanup_failed",
    "uia_termination_failed",
}
PRINT_WINDOW_FLAGS = 0x00000000
SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
PRESENTATION_PROFILES = {"compact": 480, "standard": 720, "native": None}
DEFAULT_PRESENTATION_PROFILE = "standard"
MAX_PRESENTATION_PNG_BYTES = 8 * 1024 * 1024
MAX_QUALITY_ANALYSIS_EDGE = 512
DWMWA_EXTENDED_FRAME_BOUNDS = 9
QUALITY_THRESHOLDS = {
    "black_luminance_max": 8,
    "white_luminance_min": 247,
    "transparent_alpha_max": 8,
    "extreme_coverage_reject": 0.995,
    "transparent_coverage_reject": 0.98,
    "tiny_mark_dominant_ratio_min": 0.995,
    "tiny_mark_active_ratio_max": 0.005,
    "tiny_mark_bbox_fraction_max": 0.10,
    "tiny_mark_edge_density_max": 0.002,
    "low_information_luminance_spread_max": 4,
    "low_information_entropy_max": 0.15,
    "low_information_edge_density_max": 0.0005,
    "edge_luminance_delta_min": 12,
}


@dataclass(frozen=True)
class CapturedImage:
    image: Any
    width: int
    height: int
    captured_unix_ms: int
    backend: str
    cleanup_verified: bool
    backend_details: dict[str, Any] | None = None
    attempts: tuple[dict[str, Any], ...] = ()
    quality: CaptureQuality | None = None
    title_changed_during_operation: bool = False


@dataclass(frozen=True)
class CaptureQuality:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]
    analysis_size: tuple[int, int]

    def public(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "analysis_pixel_size": {
                "width": self.analysis_size[0],
                "height": self.analysis_size[1],
            },
            "metrics": self.metrics,
            "thresholds": QUALITY_THRESHOLDS,
        }


@dataclass(frozen=True)
class CaptureWorkerMetadata:
    width: int
    height: int
    png_bytes: int
    captured_unix_ms: int
    cleanup_verified: bool


@dataclass(frozen=True)
class ScreenshotTransaction:
    captured: CapturedImage
    geometry: NativeCaptureGeometry
    identity_before: WindowIdentity
    identity_after: WindowIdentity
    title_changed_during_operation: bool
    geometry_recapture_count: int


@dataclass(frozen=True)
class NativeCaptureSource:
    capture_id: str
    record: dict[str, Any]
    png_bytes: bytes
    validated_unix_ms: int


def parse_region(value: str | None) -> Region | None:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        raise ComputerError(
            "invalid_region",
            "Region must be x,y,width,height using integers.",
            exit_code=2,
        )
    try:
        x, y, width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ComputerError(
            "invalid_region",
            "Region must be x,y,width,height using integers.",
            exit_code=2,
        ) from exc
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ComputerError(
            "invalid_region",
            "Region coordinates must be non-negative and dimensions must be positive.",
            exit_code=2,
        )
    return Region(x=x, y=y, width=width, height=height)


def capture_screenshot(
    *,
    hwnd: int,
    output: Path,
    region: Region | None,
    profile: str = DEFAULT_PRESENTATION_PROFILE,
    backend: Win32Backend | None = None,
    record_store: CaptureRecordStore | None = None,
) -> dict[str, Any]:
    resolved_backend = backend or Win32Backend()
    resolved_store = record_store or CaptureRecordStore()
    output_path = _validate_output_path(output)
    resolved_profile = _validate_profile(profile)
    transaction = _capture_target_transaction(
        resolved_backend,
        hwnd,
        region=region,
    )
    geometry = transaction.geometry
    identity = geometry.identity
    captured = transaction.captured
    if not captured.cleanup_verified:
        raise ComputerError(
            "capture_cleanup_failed",
            "The capture backend could not verify resource cleanup.",
        )
    native_image = captured.image
    if region is not None:
        native_image = native_image.crop(
            (
                region.x,
                region.y,
                region.x + region.width,
                region.y + region.height,
            )
        )
    expected_width = region.width if region is not None else identity.bounds.width
    expected_height = region.height if region is not None else identity.bounds.height
    if native_image.width != expected_width or native_image.height != expected_height:
        raise ComputerError(
            "capture_dimensions_invalid",
            "The captured image dimensions do not match the requested target.",
        )
    quality = captured.quality or analyze_capture_quality(native_image)
    if not quality.accepted:
        raise ComputerError(
            "capture_content_invalid",
            "The capture backend returned a visually unusable image.",
            details={"quality": quality.public()},
        )

    presentation_width, presentation_height = presentation_size(
        expected_width,
        expected_height,
        resolved_profile,
    )
    presentation_image = _presentation_image(
        native_image,
        presentation_width,
        presentation_height,
    )
    payload = _encode_png(presentation_image)
    if resolved_profile != "native" and len(payload) > MAX_PRESENTATION_PNG_BYTES:
        raise ComputerError(
            "presentation_too_large",
            "The presentation screenshot exceeded its safety limit.",
        )
    _verify_png(payload, presentation_width, presentation_height)

    capture_id = resolved_store.new_capture_id()
    created_ms = captured.captured_unix_ms
    expires_ms = created_ms + CAPTURE_RECORD_TTL_SECONDS * 1000
    requested_crop = region or Region(
        x=0,
        y=0,
        width=identity.bounds.width,
        height=identity.bounds.height,
    )
    transform = build_transform(
        requested_crop,
        presentation_width,
        presentation_height,
    )
    identity_before_reference = resolved_store.identity_reference(
        transaction.identity_before
    )
    identity_reference = resolved_store.identity_reference(
        transaction.identity_after
    )
    target_transaction = {
        "strong_identity_verified": True,
        "identity_mismatch_fields": [],
        "title_changed_during_operation": (
            transaction.title_changed_during_operation
        ),
        "title_before_hmac_sha256": identity_before_reference[
            "title_hmac_sha256"
        ],
        "title_after_hmac_sha256": identity_reference["title_hmac_sha256"],
        "geometry_recapture_count": transaction.geometry_recapture_count,
    }
    crop_screen_bounds = {
        "x": identity.bounds.x + requested_crop.x,
        "y": identity.bounds.y + requested_crop.y,
        "width": requested_crop.width,
        "height": requested_crop.height,
    }
    image_sha256 = hashlib.sha256(payload).hexdigest()
    resolved_output = output_path.resolve()
    record = {
        "schema_version": CAPTURE_RECORD_SCHEMA,
        "capture_id": capture_id,
        "created_unix_ms": created_ms,
        "expires_unix_ms": expires_ms,
        "window_identity": identity_reference,
        "geometry": geometry.public(),
        "requested_native_crop": requested_crop.to_jsonable(),
        "requested_native_crop_screen_bounds": crop_screen_bounds,
        "native_pixel_size": {"width": expected_width, "height": expected_height},
        "presentation": {
            "profile": resolved_profile,
            "pixel_size": {
                "width": presentation_width,
                "height": presentation_height,
            },
            "path": str(resolved_output),
            "bytes": len(payload),
            "sha256": image_sha256,
        },
        "transform": transform,
        "backend": captured.backend,
        "backend_details": captured.backend_details or {},
        "capture_attempts": list(captured.attempts),
        "quality": quality.public(),
        "target_transaction": target_transaction,
    }
    response = {
        "available": True,
        "capture_id": capture_id,
        "captured_at": _timestamp(created_ms),
        "expires_at": _timestamp(expires_ms),
        "window": identity_reference,
        "geometry": geometry.public(),
        "region": region.to_jsonable() if region is not None else None,
        "requested_native_crop": requested_crop.to_jsonable(),
        "requested_native_crop_screen_bounds": crop_screen_bounds,
        "native_pixel_size": {"width": expected_width, "height": expected_height},
        "presentation": {
            "profile": resolved_profile,
            "pixel_size": {
                "width": presentation_width,
                "height": presentation_height,
            },
        },
        "transform": transform,
        "output": str(resolved_output),
        "width": presentation_width,
        "height": presentation_height,
        "bytes": len(payload),
        "sha256": image_sha256,
        "content_non_uniform": _image_non_uniform(presentation_image),
        "backend": captured.backend,
        "backend_details": captured.backend_details or {},
        "capture_attempts": list(captured.attempts),
        "quality": quality.public(),
        "limitations": _capture_limitations(captured.backend, captured.backend_details),
        "cleanup_verified": captured.cleanup_verified,
        "coordinate_authority": "capture_record_required",
        **target_transaction,
    }
    resolved_store.assert_fresh(record)
    _atomic_write(output_path, payload)
    resolved_store.assert_fresh(record)
    resolved_store.save(record)
    try:
        resolved_store.assert_fresh(record)
    except ComputerError:
        resolved_store.delete(capture_id)
        raise
    return response


def _capture_target_transaction(
    backend: Win32Backend,
    hwnd: int,
    *,
    region: Region | None,
) -> ScreenshotTransaction:
    deadline = time.monotonic() + CAPTURE_TIMEOUT_SECONDS
    geometry = backend.capture_native_geometry(hwnd)
    anchor = geometry.identity
    _validate_coordinate_identity(anchor)
    title_changed = False
    for attempt_index in range(2):
        _capture_time_remaining(deadline)
        _assert_strong_screenshot_identity(backend, anchor, geometry.identity)
        _validate_window_size(geometry.identity)
        if backend.is_minimized(hwnd):
            raise ComputerError(
                "window_minimized",
                f"HWND {hwnd} is minimized and cannot be captured safely.",
            )
        _validate_region(region, geometry.identity)
        captured: CapturedImage | None = None
        capture_error: ComputerError | None = None
        try:
            captured = _capture_bounded(
                backend,
                geometry.identity,
                region=region,
                deadline=deadline,
            )
        except ComputerError as exc:
            capture_error = exc
            if exc.code in FATAL_CAPTURE_ATTEMPT_ERRORS:
                raise

        if captured is not None:
            title_changed = (
                title_changed or captured.title_changed_during_operation
            )

        try:
            post_geometry = backend.capture_native_geometry(hwnd)
        except ComputerError as exc:
            if capture_error is not None and capture_error.code in {
                "stale_window",
                "target_identity_unavailable",
            }:
                raise capture_error from exc
            if exc.code in {
                "invalid_window",
                "window_not_visible",
                "window_untitled",
            }:
                raise ComputerError(
                    "stale_window",
                    f"HWND {hwnd} changed or closed during capture.",
                    details={"identity_mismatch_fields": ["window"]},
                ) from exc
            raise
        _assert_strong_screenshot_identity(backend, anchor, post_geometry.identity)
        title_changed = title_changed or (
            geometry.identity.title != post_geometry.identity.title
        )
        geometry_mismatches = _screenshot_geometry_mismatches(
            geometry,
            post_geometry,
        )
        if geometry_mismatches:
            if attempt_index == 0:
                geometry = post_geometry
                continue
            raise ComputerError(
                "capture_geometry_changed",
                "The target geometry did not remain stable during screenshot capture.",
                details={
                    "geometry_mismatch_fields": geometry_mismatches,
                    "geometry_recapture_count": 1,
                },
            )
        _capture_time_remaining(deadline)
        if capture_error is not None:
            raise capture_error
        if captured is None:
            raise ComputerError(
                "capture_backend_failed",
                "The capture transaction returned no screenshot.",
            )
        return ScreenshotTransaction(
            captured=captured,
            geometry=post_geometry,
            identity_before=anchor,
            identity_after=post_geometry.identity,
            title_changed_during_operation=title_changed,
            geometry_recapture_count=attempt_index,
        )
    raise ComputerError(
        "capture_geometry_changed",
        "The target geometry did not remain stable during screenshot capture.",
    )


def _assert_strong_screenshot_identity(
    backend: Win32Backend,
    expected: WindowIdentity,
    actual: WindowIdentity,
) -> None:
    if isinstance(backend, Win32Backend):
        mismatches = backend.screenshot_identity_mismatches(expected, actual)
    else:
        comparisons = (
            ("hwnd", expected.hwnd, actual.hwnd),
            ("pid", expected.pid, actual.pid),
            (
                "process_start_identity",
                expected.process_started,
                actual.process_started,
            ),
            ("window_class", expected.class_name, actual.class_name),
            ("session_id", expected.session_id, actual.session_id),
            (
                "input_desktop",
                (expected.desktop_name or "").casefold(),
                (actual.desktop_name or "").casefold(),
            ),
        )
        mismatches = [
            name for name, before, after in comparisons if before != after
        ]
    if mismatches:
        raise ComputerError(
            "stale_window",
            f"HWND {expected.hwnd} changed or closed before capture.",
            details={
                "identity_mismatch_fields": mismatches,
                "title_changed": actual.title != expected.title,
                "title_change_was_authoritative": False,
            },
        )


def _screenshot_geometry_mismatches(
    before: NativeCaptureGeometry,
    after: NativeCaptureGeometry,
) -> list[str]:
    comparisons = (
        ("window_bounds", before.identity.bounds, after.identity.bounds),
        ("client_bounds", before.client_bounds, after.client_bounds),
        ("monitor_bounds", before.monitor_bounds, after.monitor_bounds),
        (
            "virtual_screen_bounds",
            before.virtual_screen_bounds,
            after.virtual_screen_bounds,
        ),
        ("window_dpi", before.window_dpi, after.window_dpi),
        ("system_dpi", before.system_dpi, after.system_dpi),
        ("window_awareness", before.window_awareness, after.window_awareness),
        (
            "capture_thread_awareness",
            before.capture_thread_awareness,
            after.capture_thread_awareness,
        ),
    )
    return [name for name, old, new in comparisons if old != new]


def presentation_size(native_width: int, native_height: int, profile: str) -> tuple[int, int]:
    resolved_profile = _validate_profile(profile)
    if native_width <= 0 or native_height <= 0:
        raise ComputerError(
            "capture_dimensions_invalid",
            "Screenshot dimensions must be positive.",
        )
    max_edge = PRESENTATION_PROFILES[resolved_profile]
    if max_edge is None or max(native_width, native_height) <= max_edge:
        return native_width, native_height
    if native_width >= native_height:
        return max_edge, max(1, _round_half_up(native_height * max_edge, native_width))
    return max(1, _round_half_up(native_width * max_edge, native_height)), max_edge


def build_transform(
    native_crop: Region,
    presentation_width: int,
    presentation_height: int,
) -> dict[str, Any]:
    return {
        "coordinate_spaces": {
            "source": "presentation_pixel_index",
            "target": "native_window_pixel_index",
        },
        "rounding": "edge_aligned_nearest_half_up",
        "x": _axis_transform(native_crop.x, native_crop.width, presentation_width),
        "y": _axis_transform(native_crop.y, native_crop.height, presentation_height),
    }


def map_presentation_point(
    transform: dict[str, Any],
    image_x: int,
    image_y: int,
) -> tuple[int, int]:
    return (
        _map_axis_to_native(transform.get("x"), image_x),
        _map_axis_to_native(transform.get("y"), image_y),
    )


def map_native_point(
    transform: dict[str, Any],
    native_x: int,
    native_y: int,
) -> tuple[int, int]:
    return (
        _map_axis_to_presentation(transform.get("x"), native_x),
        _map_axis_to_presentation(transform.get("y"), native_y),
    )


def resolve_capture_point(
    *,
    capture_id: str,
    image_x: int,
    image_y: int,
    backend: Win32Backend | None = None,
    record_store: CaptureRecordStore | None = None,
) -> dict[str, Any]:
    if type(image_x) is not int or type(image_y) is not int:
        raise ComputerError(
            "invalid_image_point",
            "Presentation image coordinates must be integers.",
            exit_code=2,
        )
    resolved_store = record_store or CaptureRecordStore()
    record = resolved_store.load(capture_id)
    presentation = _required_dict(record, "presentation")
    pixel_size = _required_size(presentation, "pixel_size")
    _validate_capture_record_consistency(record, pixel_size)
    if not (0 <= image_x < pixel_size[0] and 0 <= image_y < pixel_size[1]):
        raise ComputerError(
            "image_point_out_of_bounds",
            "The presentation image coordinate is outside the captured image.",
            exit_code=2,
        )
    _validate_recorded_image(presentation, pixel_size)

    window_reference = _required_dict(record, "window_identity")
    hwnd = window_reference.get("hwnd")
    if type(hwnd) is not int or hwnd <= 0:
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record is invalid.",
        )
    resolved_backend = backend or Win32Backend()
    current_geometry = resolved_backend.capture_native_geometry(hwnd)
    resolved_store.validate_identity_reference(window_reference, current_geometry.identity)
    _validate_geometry_record(record, current_geometry)

    transform = _required_dict(record, "transform")
    native_x, native_y = map_presentation_point(transform, image_x, image_y)
    screen_x = current_geometry.identity.bounds.x + native_x
    screen_y = current_geometry.identity.bounds.y + native_y
    validated_ms = resolved_store.assert_fresh(record)
    return {
        "capture_id": capture_id,
        "image_point": {"x": image_x, "y": image_y},
        "native_window_point": {"x": native_x, "y": native_y},
        "native_screen_point": {"x": screen_x, "y": screen_y},
        "validated_at": _timestamp(validated_ms),
        "expires_at": _timestamp(int(record["expires_unix_ms"])),
        "coordinate_authority": "fresh_capture_bound_native",
        "action_revalidation_required": True,
    }


def resolve_native_capture_source(
    *,
    capture_id: str,
    record_store: CaptureRecordStore | None = None,
) -> NativeCaptureSource:
    """Resolve immutable native screenshot bytes without reading the live desktop."""
    resolved_store = record_store or CaptureRecordStore()
    record = resolved_store.load(capture_id)
    presentation = _required_dict(record, "presentation")
    pixel_size = _required_size(presentation, "pixel_size")
    _validate_capture_record_consistency(record, pixel_size)
    if presentation.get("profile") != "native":
        raise ComputerError(
            "ocr_capture_not_native",
            "OCR requires the original profile-native screenshot capture.",
            details={"profile": presentation.get("profile")},
        )
    if _required_size(record, "native_pixel_size") != pixel_size:
        raise ComputerError(
            "capture_record_invalid",
            "The native screenshot dimensions are invalid.",
        )
    _validate_capture_time_window_identity(record)
    target_transaction = _required_dict(record, "target_transaction")
    if (
        target_transaction.get("strong_identity_verified") is not True
        or target_transaction.get("identity_mismatch_fields") != []
    ):
        raise ComputerError(
            "capture_record_invalid",
            "The native screenshot identity transaction is invalid.",
        )
    png_bytes = _read_recorded_image_bytes(presentation, pixel_size)
    validated_ms = resolved_store.assert_fresh(record)
    return NativeCaptureSource(
        capture_id=capture_id,
        record=record,
        png_bytes=png_bytes,
        validated_unix_ms=validated_ms,
    )


def _validate_profile(profile: str) -> str:
    if not isinstance(profile, str) or profile not in PRESENTATION_PROFILES:
        raise ComputerError(
            "invalid_screenshot_profile",
            "Screenshot profile must be compact, standard, or native.",
            exit_code=2,
        )
    return profile


def _validate_coordinate_identity(identity: WindowIdentity) -> None:
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


def _presentation_image(image: Any, width: int, height: int) -> Any:
    if image.size == (width, height):
        return image.copy()
    image_module = _load_pillow()
    resampling = getattr(image_module, "Resampling", image_module)
    return image.resize((width, height), resample=resampling.LANCZOS)


def _axis_transform(
    native_origin: int,
    native_extent: int,
    presentation_extent: int,
) -> dict[str, int]:
    if native_extent <= 0 or presentation_extent <= 0:
        raise ComputerError(
            "capture_dimensions_invalid",
            "Screenshot transform dimensions must be positive.",
        )
    return {
        "native_origin": native_origin,
        "native_extent": native_extent,
        "presentation_extent": presentation_extent,
        "native_span_numerator": max(0, native_extent - 1),
        "presentation_span_denominator": max(1, presentation_extent - 1),
        "singleton_native_index": native_origin + _round_half_up(native_extent - 1, 2),
    }


def _map_axis_to_native(axis: object, coordinate: int) -> int:
    values = _validated_axis(axis)
    if type(coordinate) is not int or not 0 <= coordinate < values[2]:
        raise ComputerError(
            "image_point_out_of_bounds",
            "The presentation image coordinate is outside the captured image.",
            exit_code=2,
        )
    origin, native_extent, presentation_extent = values
    if presentation_extent == 1:
        return origin + _round_half_up(native_extent - 1, 2)
    return origin + _round_half_up(
        coordinate * (native_extent - 1),
        presentation_extent - 1,
    )


def _map_axis_to_presentation(axis: object, coordinate: int) -> int:
    origin, native_extent, presentation_extent = _validated_axis(axis)
    if type(coordinate) is not int or not origin <= coordinate < origin + native_extent:
        raise ComputerError(
            "native_point_out_of_bounds",
            "The native coordinate is outside the captured crop.",
            exit_code=2,
        )
    if native_extent == 1 or presentation_extent == 1:
        return 0
    return _round_half_up(
        (coordinate - origin) * (presentation_extent - 1),
        native_extent - 1,
    )


def _validated_axis(axis: object) -> tuple[int, int, int]:
    if not isinstance(axis, dict):
        raise ComputerError(
            "capture_record_invalid",
            "The capture transform is invalid.",
        )
    fields = ("native_origin", "native_extent", "presentation_extent")
    if any(type(axis.get(field)) is not int for field in fields):
        raise ComputerError(
            "capture_record_invalid",
            "The capture transform is invalid.",
        )
    origin = int(axis["native_origin"])
    native_extent = int(axis["native_extent"])
    presentation_extent = int(axis["presentation_extent"])
    if native_extent <= 0 or presentation_extent <= 0:
        raise ComputerError(
            "capture_record_invalid",
            "The capture transform is invalid.",
        )
    expected = _axis_transform(origin, native_extent, presentation_extent)
    if axis != expected:
        raise ComputerError(
            "capture_record_invalid",
            "The capture transform is invalid.",
        )
    return origin, native_extent, presentation_extent


def _round_half_up(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ComputerError(
            "capture_dimensions_invalid",
            "Screenshot scaling values are invalid.",
        )
    return (numerator * 2 + denominator) // (denominator * 2)


def _required_dict(parent: dict[str, Any], field: str) -> dict[str, Any]:
    value = parent.get(field)
    if not isinstance(value, dict):
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record is invalid.",
        )
    return value


def _required_size(parent: dict[str, Any], field: str) -> tuple[int, int]:
    value = _required_dict(parent, field)
    width = value.get("width")
    height = value.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record is invalid.",
        )
    return width, height


def _validate_recorded_image(
    presentation: dict[str, Any],
    expected_size: tuple[int, int],
) -> None:
    _read_recorded_image_bytes(presentation, expected_size)


def _read_recorded_image_bytes(
    presentation: dict[str, Any],
    expected_size: tuple[int, int],
) -> bytes:
    path_value = presentation.get("path")
    expected_bytes = presentation.get("bytes")
    expected_hash = presentation.get("sha256")
    if (
        not isinstance(path_value, str)
        or not path_value
        or type(expected_bytes) is not int
        or expected_bytes < 0
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or not Path(path_value).is_absolute()
    ):
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record is invalid.",
        )
    path = Path(path_value)
    if _is_reparse_point(path):
        raise ComputerError(
            "capture_image_altered",
            "The captured presentation image changed; take a fresh screenshot.",
        )
    try:
        with path.open("rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
                raise ComputerError(
                    "capture_image_altered",
                    "The captured presentation image changed; take a fresh screenshot.",
                )
            payload = stream.read(expected_bytes + 1)
            if len(payload) != expected_bytes:
                raise ComputerError(
                    "capture_image_altered",
                    "The captured presentation image changed; take a fresh screenshot.",
                )
            if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_hash):
                raise ComputerError(
                    "capture_image_altered",
                    "The captured presentation image changed; take a fresh screenshot.",
                )
            image_module = _load_pillow()
            with image_module.open(io.BytesIO(payload)) as image:
                if image.size != expected_size:
                    raise ComputerError(
                        "capture_image_dimensions_changed",
                        "The captured image dimensions changed; take a fresh screenshot.",
                    )
                image.verify()
    except ComputerError:
        raise
    except (OSError, ValueError) as exc:
        raise ComputerError(
            "capture_image_missing",
            "The captured presentation image is missing or unreadable.",
        ) from exc
    return payload


def _validate_capture_time_window_identity(record: dict[str, Any]) -> None:
    identity = _required_dict(record, "window_identity")
    integer_fields = ("hwnd", "pid", "thread_id", "process_started", "session_id")
    if any(type(identity.get(field)) is not int for field in integer_fields):
        raise ComputerError(
            "capture_record_invalid",
            "The capture-time window identity is invalid.",
        )
    if (
        int(identity["hwnd"]) <= 0
        or int(identity["pid"]) <= 0
        or int(identity["thread_id"]) <= 0
        or int(identity["process_started"]) <= 0
        or int(identity["session_id"]) < 0
    ):
        raise ComputerError(
            "capture_record_invalid",
            "The capture-time window identity is invalid.",
        )
    hash_fields = (
        "process_hmac_sha256",
        "title_hmac_sha256",
        "class_hmac_sha256",
        "desktop_hmac_sha256",
        "fingerprint",
    )
    if any(
        not isinstance(identity.get(field), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(identity[field])) is None
        for field in hash_fields
    ):
        raise ComputerError(
            "capture_record_invalid",
            "The capture-time window identity is invalid.",
        )


def _validate_geometry_record(
    record: dict[str, Any],
    current: NativeCaptureGeometry,
) -> None:
    recorded = _required_dict(record, "geometry")
    current_public = current.public()
    if recorded.get("coordinate_space") != "physical_screen_pixels":
        raise ComputerError(
            "capture_record_invalid",
            "The capture geometry is invalid.",
        )
    if recorded.get("window_bounds") != current_public["window_bounds"]:
        raise ComputerError(
            "capture_geometry_changed",
            "The captured window moved or resized; take a fresh screenshot.",
        )
    if recorded.get("client_bounds") != current_public["client_bounds"]:
        raise ComputerError(
            "capture_geometry_changed",
            "The captured client geometry changed; take a fresh screenshot.",
        )
    if (
        recorded.get("monitor_bounds") != current_public["monitor_bounds"]
        or recorded.get("virtual_screen_bounds") != current_public["virtual_screen_bounds"]
    ):
        raise ComputerError(
            "capture_screen_geometry_changed",
            "The screen geometry changed; take a fresh screenshot.",
        )
    if recorded.get("dpi") != current_public["dpi"]:
        raise ComputerError(
            "capture_dpi_changed",
            "The captured window DPI changed; take a fresh screenshot.",
        )


def _validate_capture_record_consistency(
    record: dict[str, Any],
    presentation_size_value: tuple[int, int],
) -> None:
    created_ms = record.get("created_unix_ms")
    expires_ms = record.get("expires_unix_ms")
    if (
        type(created_ms) is not int
        or type(expires_ms) is not int
        or expires_ms - created_ms != CAPTURE_RECORD_TTL_SECONDS * 1000
    ):
        raise ComputerError(
            "capture_record_invalid",
            "The capture provenance record is invalid.",
        )
    crop_value = _required_dict(record, "requested_native_crop")
    crop_fields = ("x", "y", "width", "height")
    if any(type(crop_value.get(field)) is not int for field in crop_fields):
        raise ComputerError(
            "capture_record_invalid",
            "The capture crop is invalid.",
        )
    crop = Region(
        x=int(crop_value["x"]),
        y=int(crop_value["y"]),
        width=int(crop_value["width"]),
        height=int(crop_value["height"]),
    )
    if crop.x < 0 or crop.y < 0 or crop.width <= 0 or crop.height <= 0:
        raise ComputerError(
            "capture_record_invalid",
            "The capture crop is invalid.",
        )
    native_size = _required_size(record, "native_pixel_size")
    if native_size != (crop.width, crop.height):
        raise ComputerError(
            "capture_record_invalid",
            "The capture crop dimensions are invalid.",
        )
    presentation = _required_dict(record, "presentation")
    profile = presentation.get("profile")
    if not isinstance(profile, str):
        raise ComputerError(
            "capture_record_invalid",
            "The screenshot profile is invalid.",
        )
    try:
        expected_presentation_size = presentation_size(crop.width, crop.height, profile)
    except ComputerError as exc:
        raise ComputerError(
            "capture_record_invalid",
            "The screenshot profile is invalid.",
        ) from exc
    if presentation_size_value != expected_presentation_size:
        raise ComputerError(
            "capture_record_invalid",
            "The presentation dimensions are invalid.",
        )
    expected_transform = build_transform(crop, *presentation_size_value)
    if record.get("transform") != expected_transform:
        raise ComputerError(
            "capture_record_invalid",
            "The capture transform is invalid.",
        )
    geometry = _required_dict(record, "geometry")
    window_bounds = _required_bounds(geometry, "window_bounds")
    if crop.x + crop.width > window_bounds.width or crop.y + crop.height > window_bounds.height:
        raise ComputerError(
            "capture_record_invalid",
            "The capture crop is outside the recorded window.",
        )
    expected_screen_crop = {
        "x": window_bounds.x + crop.x,
        "y": window_bounds.y + crop.y,
        "width": crop.width,
        "height": crop.height,
    }
    if record.get("requested_native_crop_screen_bounds") != expected_screen_crop:
        raise ComputerError(
            "capture_record_invalid",
            "The capture screen crop is invalid.",
        )


def _required_bounds(parent: dict[str, Any], field: str) -> Bounds:
    value = _required_dict(parent, field)
    fields = ("x", "y", "width", "height")
    if any(type(value.get(item)) is not int for item in fields):
        raise ComputerError(
            "capture_record_invalid",
            "The capture bounds are invalid.",
        )
    bounds = Bounds(
        x=int(value["x"]),
        y=int(value["y"]),
        width=int(value["width"]),
        height=int(value["height"]),
    )
    if bounds.width <= 0 or bounds.height <= 0:
        raise ComputerError(
            "capture_record_invalid",
            "The capture bounds are invalid.",
        )
    return bounds


def _timestamp(unix_ms: int) -> str:
    return (
        datetime.fromtimestamp(unix_ms / 1000, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _capture_bounded(
    backend: Win32Backend,
    identity: WindowIdentity,
    *,
    region: Region | None = None,
    deadline: float | None = None,
) -> CapturedImage:
    if not isinstance(backend, Win32Backend):
        captured = _capture_full(backend, identity)
        quality = analyze_capture_quality(_requested_native_image(captured.image, region))
        return CapturedImage(
            image=captured.image,
            width=captured.width,
            height=captured.height,
            captured_unix_ms=captured.captured_unix_ms,
            backend=captured.backend,
            cleanup_verified=captured.cleanup_verified,
            backend_details=captured.backend_details,
            attempts=(
                _capture_attempt(
                    captured.backend,
                    "accepted" if quality.accepted else "rejected",
                    0,
                    quality=quality,
                    error_code=None if quality.accepted else "capture_content_invalid",
                ),
            ),
            quality=quality,
        )

    attempts: list[dict[str, Any]] = []
    observed_title = identity.title
    title_changed = False
    capture_deadline = (
        deadline if deadline is not None else time.monotonic() + CAPTURE_TIMEOUT_SECONDS
    )
    candidates = (
        ("winapp_window_capture", _capture_winapp_window),
        ("win32_print_window_isolated", _capture_print_window_isolated),
        ("desktop_crop_visible", _capture_desktop_visible),
    )
    deadline_budget_exhausted = False
    for backend_name, capture in candidates:
        if (
            capture_deadline - time.monotonic()
            <= MIN_CAPTURE_BACKEND_START_SECONDS
        ):
            deadline_budget_exhausted = True
            break
        started = time.monotonic()
        try:
            _capture_time_remaining(capture_deadline)
            with backend.native_pixel_context():
                actual_before = backend.revalidate(identity)
            title_changed = title_changed or actual_before.title != observed_title
            observed_title = actual_before.title
            if backend.is_minimized(identity.hwnd):
                raise ComputerError(
                    "window_minimized",
                    f"HWND {identity.hwnd} became minimized during capture.",
                )
            remaining = capture_deadline - time.monotonic()
            if remaining <= MIN_CAPTURE_BACKEND_START_SECONDS:
                deadline_budget_exhausted = True
                break
            candidate_capture: CapturedImage | None = None
            capture_error: Exception | None = None
            try:
                candidate_capture = capture(
                    backend,
                    identity,
                    timeout_seconds=remaining,
                    deadline=capture_deadline,
                )
            except Exception as exc:  # noqa: BLE001 - revalidation must still run
                capture_error = exc
            try:
                with backend.native_pixel_context():
                    actual_after = backend.revalidate(identity)
                title_changed = title_changed or actual_after.title != observed_title
                observed_title = actual_after.title
            except ComputerError as exc:
                if capture_error is not None:
                    if (
                        isinstance(capture_error, ComputerError)
                        and capture_error.code in FATAL_CAPTURE_ATTEMPT_ERRORS
                    ):
                        capture_error.details.setdefault(
                            "secondary_errors",
                            [],
                        ).append(exc.code)
                        raise capture_error from exc
                    raise exc from capture_error
                raise
            if backend.is_minimized(identity.hwnd):
                raise ComputerError(
                    "window_minimized",
                    f"HWND {identity.hwnd} became minimized during capture.",
                )
            if capture_error is not None:
                if isinstance(capture_error, ComputerError):
                    if capture_error.code in FATAL_CAPTURE_ATTEMPT_ERRORS:
                        raise capture_error
                    _capture_time_remaining(capture_deadline)
                    raise capture_error
                _capture_time_remaining(capture_deadline)
                raise ComputerError(
                    "capture_backend_failed",
                    "The capture backend failed unexpectedly.",
                ) from capture_error
            _capture_time_remaining(capture_deadline)
            if candidate_capture is None:
                raise ComputerError(
                    "capture_backend_failed",
                    "The capture backend returned no screenshot.",
                )
            captured = candidate_capture
            if not captured.cleanup_verified:
                raise ComputerError(
                    "capture_cleanup_failed",
                    "The capture backend could not verify resource cleanup.",
                )
            _validate_captured_target_dimensions(captured, identity)
            quality = analyze_capture_quality(
                _requested_native_image(captured.image, region)
            )
            _capture_time_remaining(capture_deadline)
            duration_ms = _duration_ms(started)
            if not quality.accepted:
                attempts.append(
                    _capture_attempt(
                        backend_name,
                        "rejected",
                        duration_ms,
                        quality=quality,
                        error_code="capture_content_invalid",
                    )
                )
                continue
            attempts.append(
                _capture_attempt(
                    backend_name,
                    "accepted",
                    duration_ms,
                    quality=quality,
                )
            )
            return CapturedImage(
                image=captured.image,
                width=captured.width,
                height=captured.height,
                captured_unix_ms=captured.captured_unix_ms,
                backend=captured.backend,
                cleanup_verified=True,
                backend_details=captured.backend_details,
                attempts=tuple(attempts),
                quality=quality,
                title_changed_during_operation=title_changed,
            )
        except ComputerError as exc:
            attempts.append(
                _capture_attempt(
                    backend_name,
                    "failed",
                    _duration_ms(started),
                    error_code=exc.code,
                )
            )
            if exc.code in {
                "stale_window",
                "window_minimized",
                "window_not_visible",
                "target_identity_unavailable",
            } | FATAL_CAPTURE_ATTEMPT_ERRORS:
                raise ComputerError(
                    exc.code,
                    exc.message,
                    exit_code=exc.exit_code,
                    details={**exc.details, "capture_attempts": attempts},
                ) from exc

    quality_rejected = any(
        attempt.get("error_code") == "capture_content_invalid"
        for attempt in attempts
    )
    every_attempt_quality_rejected = bool(attempts) and all(
        attempt.get("error_code") == "capture_content_invalid"
        for attempt in attempts
    )
    chain_timed_out = deadline_budget_exhausted or time.monotonic() >= capture_deadline
    final_code = (
        "capture_timeout"
        if chain_timed_out
        else "capture_content_invalid" if quality_rejected else "capture_failed"
    )
    raise ComputerError(
        final_code,
        (
            "The capture backend chain did not finish within its safety deadline."
            if chain_timed_out
            else (
                "Every capture backend returned a visually unusable image."
                if every_attempt_quality_rejected
                else (
                    "No backend produced a usable image; at least one capture was "
                    "rejected by quality checks."
                    if quality_rejected
                    else "Every available capture backend failed."
                )
            )
        ),
        details={
            "capture_attempts": attempts,
            "limitations": [
                "minimized_or_cloaked_windows_are_not_captured",
                "protected_content_may_be_unavailable",
                "desktop_crop_requires_foreground_and_full_visibility",
            ],
        },
    )


def _capture_winapp_window(
    backend: Win32Backend,
    identity: WindowIdentity,
    *,
    timeout_seconds: float = CAPTURE_TIMEOUT_SECONDS,
    deadline: float | None = None,
) -> CapturedImage:
    capture_started_unix_ms = int(time.time() * 1000)
    temporary = tempfile.TemporaryDirectory(prefix="agent-tools-window-capture-")
    temporary_root = Path(temporary.name)
    capture_path = temporary_root / "window.png"
    primary_error: BaseException | None = None
    captured: CapturedImage | None = None
    try:
        adapter = WinAppAdapter(backend)
        screenshot_kwargs: dict[str, Any] = {
            "output": capture_path,
            "timeout_seconds": timeout_seconds,
        }
        if deadline is not None:
            screenshot_kwargs["deadline"] = deadline
        binding, metadata = adapter.screenshot(identity, **screenshot_kwargs)
        if not capture_path.is_file() or _is_reparse_point(capture_path):
            raise ComputerError(
                "capture_backend_failed",
                "WinApp did not produce a regular screenshot file.",
            )
        max_file_bytes = (
            identity.bounds.width * identity.bounds.height * 4
            + MAX_CAPTURE_PROTOCOL_OVERHEAD_BYTES
        )
        try:
            file_bytes = capture_path.stat().st_size
        except OSError as exc:
            raise ComputerError(
                "capture_backend_failed",
                "The WinApp screenshot metadata could not be verified.",
            ) from exc
        if file_bytes <= 0 or file_bytes > max_file_bytes:
            raise ComputerError(
                "capture_backend_output_too_large",
                "The WinApp screenshot exceeded its output safety limit.",
            )
        image = _load_capture_file(
            capture_path,
            int(metadata["width"]),
            int(metadata["height"]),
        )
        normalized, normalization = _normalize_winapp_capture(
            backend,
            identity,
            image,
        )
        captured = CapturedImage(
            image=normalized,
            width=int(normalized.width),
            height=int(normalized.height),
            captured_unix_ms=capture_started_unix_ms,
            backend="winapp_window_capture",
            cleanup_verified=True,
            backend_details={
                "component": "microsoft_winapp_cli",
                "component_version": binding.version,
                "capture_api": metadata["capture_api"],
                "effective_capture_api": metadata["effective_capture_api"],
                "upstream_fallback_possible": metadata[
                    "upstream_fallback_possible"
                ],
                "normalization": normalization,
            },
        )
    except BaseException as exc:
        primary_error = exc
    try:
        temporary.cleanup()
    except OSError as exc:
        raise ComputerError(
            "capture_cleanup_failed",
            "The WinApp screenshot temporary directory could not be removed.",
        ) from exc
    if primary_error is not None:
        raise primary_error
    if captured is None:
        raise ComputerError(
            "capture_backend_failed",
            "WinApp returned no screenshot.",
        )
    return captured


def _load_capture_file(path: Path, width: int, height: int) -> Any:
    image_module = _load_pillow()
    try:
        with image_module.open(path) as encoded:
            if encoded.size != (width, height):
                raise ComputerError(
                    "capture_dimensions_invalid",
                    "The WinApp screenshot dimensions do not match its response.",
                )
            encoded.load()
            mode = "RGBA" if "A" in encoded.getbands() else "RGB"
            return encoded.convert(mode).copy()
    except ComputerError:
        raise
    except Exception as exc:
        raise ComputerError(
            "capture_backend_failed",
            "WinApp produced an invalid screenshot file.",
        ) from exc


def _normalize_winapp_capture(
    backend: Win32Backend,
    identity: WindowIdentity,
    image: Any,
) -> tuple[Any, dict[str, Any]]:
    full_size = (identity.bounds.width, identity.bounds.height)
    source_size = (image.width, image.height)
    if image.size == full_size:
        return image, {
            "method": "exact_window_bounds",
            "source_pixel_size": {"width": image.width, "height": image.height},
            "canvas_offset": {"x": 0, "y": 0},
        }
    frame = _extended_frame_bounds(identity.hwnd)
    header_height = 0
    normalized_source = image
    target_kind: str | None = None
    candidate_sizes = [("window", full_size)]
    if frame is not None:
        candidate_sizes.append(("dwm_frame", (frame.width, frame.height)))
    for kind, (candidate_width, candidate_height) in candidate_sizes:
        extra_height = image.height - candidate_height
        if (
            image.width == candidate_width
            and 1 <= extra_height <= 128
        ):
            header_height = extra_height
            normalized_source = image.crop(
                (0, header_height, image.width, image.height)
            )
            target_kind = kind
            break
    if target_kind == "window":
        return normalized_source, {
            "method": "winapp_header_cropped_to_exact_window_bounds",
            "source_pixel_size": {"width": source_size[0], "height": source_size[1]},
            "header_crop_pixels": header_height,
            "canvas_offset": {"x": 0, "y": 0},
        }
    if target_kind == "dwm_frame":
        image = normalized_source
    if frame is None or image.size != (frame.width, frame.height):
        raise ComputerError(
            "capture_dimensions_invalid",
            "The WinApp screenshot did not match exact window or DWM frame bounds.",
            details={
                "source_pixel_size": {"width": source_size[0], "height": source_size[1]},
                "window_pixel_size": {
                    "width": identity.bounds.width,
                    "height": identity.bounds.height,
                },
                "dwm_frame_pixel_size": (
                    {"width": frame.width, "height": frame.height}
                    if frame is not None
                    else None
                ),
            },
        )
    offset_x = frame.x - identity.bounds.x
    offset_y = frame.y - identity.bounds.y
    if (
        offset_x < 0
        or offset_y < 0
        or offset_x + frame.width > identity.bounds.width
        or offset_y + frame.height > identity.bounds.height
    ):
        raise ComputerError(
            "capture_dimensions_invalid",
            "The DWM capture frame is outside the target window bounds.",
        )
    image_module = _load_pillow()
    fill = (0, 0, 0, 0) if image.mode == "RGBA" else (0, 0, 0)
    canvas = image_module.new(image.mode, full_size, fill)
    canvas.paste(image, (offset_x, offset_y))
    with backend.native_pixel_context():
        backend.revalidate(identity)
    return canvas, {
        "method": (
            "winapp_header_cropped_dwm_frame_to_window_canvas"
            if header_height
            else "dwm_extended_frame_to_window_canvas"
        ),
        "source_pixel_size": {"width": source_size[0], "height": source_size[1]},
        **(
            {"header_crop_pixels": header_height}
            if header_height
            else {}
        ),
        "canvas_offset": {"x": offset_x, "y": offset_y},
    }


def _extended_frame_bounds(hwnd: int) -> Bounds | None:
    try:
        dwmapi = _windll("dwmapi")
        dwmapi.DwmGetWindowAttribute.argtypes = (
            wintypes.HWND,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
        rect = wintypes.RECT()
        result = dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
    except (AttributeError, OSError):
        return None
    if result != 0:
        return None
    bounds = Bounds.from_ltrb(
        int(rect.left),
        int(rect.top),
        int(rect.right),
        int(rect.bottom),
    )
    return bounds if bounds.width > 0 and bounds.height > 0 else None


def _capture_desktop_visible(
    backend: Win32Backend,
    identity: WindowIdentity,
    *,
    timeout_seconds: float = CAPTURE_TIMEOUT_SECONDS,
    deadline: float | None = None,
) -> CapturedImage:
    return _capture_worker_isolated(
        backend,
        identity,
        mode="desktop_crop",
        backend_name="desktop_crop_visible",
        backend_details={
            "capture_api": "desktop_screen_crop",
            "visibility_requirement": "foreground_fully_visible_unoccluded",
            "focus_mutation_requested": False,
            "isolated_worker": True,
            "normalization": {
                "method": "dwm_extended_frame_masked_to_window_canvas"
            },
        },
        timeout_seconds=timeout_seconds,
        deadline=deadline,
    )


def _capture_desktop_full(
    backend: Win32Backend,
    identity: WindowIdentity,
) -> CapturedImage:
    with backend.native_pixel_context():
        return _capture_desktop_full_native(backend, identity)


def _capture_desktop_full_native(
    backend: Win32Backend,
    identity: WindowIdentity,
) -> CapturedImage:
    modules = backend.modules()
    user32 = _windll("user32")
    user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    user32.GetAncestor.restype = wintypes.HWND
    backend.revalidate(identity)
    if backend.is_minimized(identity.hwnd):
        raise ComputerError(
            "window_minimized",
            f"HWND {identity.hwnd} became minimized before desktop capture.",
        )
    foreground = int(modules.win32gui.GetForegroundWindow() or 0)
    foreground_root = int(user32.GetAncestor(foreground, 2) or 0) if foreground else 0
    if foreground_root != identity.hwnd:
        raise ComputerError(
            "desktop_crop_target_not_foreground",
            "The visible desktop fallback requires the target to already be foreground.",
        )
    if _window_above_intersects(backend, identity):
        raise ComputerError(
            "desktop_crop_target_occluded",
            "The visible desktop fallback requires an unoccluded target window.",
        )
    frame_before = _required_desktop_frame(identity)
    _assert_desktop_monitor_coverage(modules, identity.bounds)
    try:
        frame_image, captured_unix_ms = _capture_exact_desktop_region(
            modules,
            left=frame_before.x,
            top=frame_before.y,
            width=frame_before.width,
            height=frame_before.height,
        )
        frame_after = _required_desktop_frame(identity)
        if frame_after != frame_before:
            raise ComputerError(
                "desktop_crop_frame_changed",
                "The visible DWM frame changed during desktop capture.",
            )
        image = _place_desktop_frame_on_window_canvas(
            frame_image,
            identity,
            frame_before,
        )
    except ComputerError:
        raise
    except Exception as exc:  # noqa: BLE001 - native capture errors are bounded
        raise ComputerError(
            "desktop_crop_failed",
            "The visible desktop fallback could not capture the target.",
        ) from exc
    foreground_after = int(modules.win32gui.GetForegroundWindow() or 0)
    foreground_root_after = (
        int(user32.GetAncestor(foreground_after, 2) or 0)
        if foreground_after
        else 0
    )
    if foreground_root_after != identity.hwnd or _window_above_intersects(
        backend, identity
    ):
        raise ComputerError(
            "desktop_crop_visibility_changed",
            "The target visibility changed during the desktop capture.",
        )
    backend.revalidate(identity)
    if backend.is_minimized(identity.hwnd):
        raise ComputerError(
            "window_minimized",
            f"HWND {identity.hwnd} became minimized during desktop capture.",
        )
    return CapturedImage(
        image=image,
        width=int(image.width),
        height=int(image.height),
        captured_unix_ms=captured_unix_ms,
        backend="desktop_crop_visible",
        cleanup_verified=True,
        backend_details={
            "capture_api": "desktop_screen_crop",
            "visibility_requirement": "foreground_fully_visible_unoccluded",
            "focus_mutation_requested": False,
        },
    )


def _assert_desktop_monitor_coverage(modules: Any, requested: Bounds) -> None:
    try:
        monitor_rectangles = [
            Bounds.from_ltrb(*map(int, monitor[2]))
            for monitor in modules.win32api.EnumDisplayMonitors()
        ]
    except Exception as exc:
        raise ComputerError(
            "display_layout_unavailable",
            "The active monitor layout could not be verified for desktop capture.",
        ) from exc
    if not monitor_rectangles:
        raise ComputerError(
            "display_layout_unavailable",
            "The active monitor layout could not be verified for desktop capture.",
        )
    if not _rectangle_covered_by_monitors(requested, monitor_rectangles):
        raise ComputerError(
            "desktop_crop_target_offscreen",
            "The visible desktop fallback requires the full target on-screen.",
        )


def _capture_exact_desktop_region(
    modules: Any,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[Any, int]:
    image_module = _load_pillow()
    user32 = _windll("user32")
    gdi32 = _windll("gdi32")
    user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
    user32.ReleaseDC.restype = ctypes.c_int
    gdi32.DeleteDC.argtypes = (wintypes.HDC,)
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.BitBlt.argtypes = (
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.DWORD,
    )
    gdi32.BitBlt.restype = wintypes.BOOL
    cleanup_steps: list[tuple[str, Any]] = []
    cleanup_completed: list[str] = []
    cleanup_errors: list[str] = []
    image: Any | None = None
    captured_unix_ms: int | None = None
    try:
        desktop_dc = modules.win32gui.GetDC(0)
        if not desktop_dc:
            raise ComputerError(
                "desktop_crop_failed",
                "The desktop capture DC could not be acquired.",
            )

        def release_desktop_dc() -> None:
            if not user32.ReleaseDC(0, desktop_dc):
                raise RuntimeError("ReleaseDC failed")

        cleanup_steps.append(("desktop_dc_released", release_desktop_dc))
        compatible_dc = modules.win32gui.CreateCompatibleDC(desktop_dc)
        if not compatible_dc:
            raise ComputerError(
                "desktop_crop_failed",
                "The desktop capture buffer could not be created.",
            )

        def delete_compatible_dc() -> None:
            if not gdi32.DeleteDC(compatible_dc):
                raise RuntimeError("DeleteDC failed")

        cleanup_steps.append(("compatible_dc_deleted", delete_compatible_dc))
        bitmap_owner = modules.win32gui.CreateCompatibleBitmap(
            desktop_dc,
            width,
            height,
        )
        bitmap_handle = bitmap_owner.Detach()

        def delete_bitmap() -> None:
            if not gdi32.DeleteObject(bitmap_handle):
                raise RuntimeError("DeleteObject failed")

        cleanup_steps.append(("bitmap_deleted", delete_bitmap))
        previous_bitmap = modules.win32gui.SelectObject(compatible_dc, bitmap_handle)
        cleanup_steps.append(
            (
                "previous_bitmap_restored",
                lambda: modules.win32gui.SelectObject(
                    compatible_dc,
                    previous_bitmap,
                ),
            )
        )
        captured_unix_ms = int(time.time() * 1000)
        copied = bool(
            gdi32.BitBlt(
                compatible_dc,
                0,
                0,
                width,
                height,
                desktop_dc,
                left,
                top,
                SRCCOPY | CAPTUREBLT,
            )
        )
        if not copied:
            raise ComputerError(
                "desktop_crop_failed",
                "The exact desktop region could not be copied.",
            )
        bitmap = modules.win32ui.CreateBitmapFromHandle(bitmap_handle)
        info = bitmap.GetInfo()
        bits = bitmap.GetBitmapBits(True)
        if (int(info["bmWidth"]), int(info["bmHeight"])) != (width, height):
            raise ComputerError(
                "capture_dimensions_invalid",
                "The exact desktop capture dimensions changed unexpectedly.",
            )
        image = image_module.frombuffer(
            "RGB",
            (width, height),
            bits,
            "raw",
            "BGRX",
            0,
            1,
        ).copy()
    except ComputerError:
        raise
    except Exception as exc:
        raise ComputerError(
            "desktop_crop_failed",
            "The exact desktop region capture failed.",
        ) from exc
    finally:
        for label, cleanup in reversed(cleanup_steps):
            try:
                cleanup()
                cleanup_completed.append(label)
            except Exception as exc:  # noqa: BLE001 - every GDI cleanup failure matters
                cleanup_errors.append(f"{label}:{type(exc).__name__}")
        if cleanup_errors:
            raise ComputerError(
                "capture_cleanup_failed",
                "The desktop capture completed with a GDI cleanup failure.",
                details={"steps": cleanup_errors},
            )
    expected_cleanup = {
        "previous_bitmap_restored",
        "bitmap_deleted",
        "compatible_dc_deleted",
        "desktop_dc_released",
    }
    if (
        image is None
        or captured_unix_ms is None
        or set(cleanup_completed) != expected_cleanup
    ):
        raise ComputerError(
            "capture_cleanup_failed",
            "The desktop capture could not verify exact GDI cleanup.",
        )
    return image, captured_unix_ms


def _required_desktop_frame(identity: WindowIdentity) -> Bounds:
    frame = _extended_frame_bounds(identity.hwnd)
    if frame is None:
        raise ComputerError(
            "capture_geometry_unavailable",
            "The visible DWM frame could not be verified for desktop capture.",
        )
    offset_x = frame.x - identity.bounds.x
    offset_y = frame.y - identity.bounds.y
    if (
        offset_x < 0
        or offset_y < 0
        or offset_x + frame.width > identity.bounds.width
        or offset_y + frame.height > identity.bounds.height
    ):
        raise ComputerError(
            "capture_dimensions_invalid",
            "The verified DWM frame is outside the target window bounds.",
        )
    return frame


def _place_desktop_frame_on_window_canvas(
    frame_image: Any,
    identity: WindowIdentity,
    frame: Bounds,
) -> Any:
    verified_frame = _required_desktop_frame(identity)
    if verified_frame != frame:
        raise ComputerError(
            "desktop_crop_frame_changed",
            "The visible DWM frame changed before desktop capture composition.",
        )
    if frame_image.size != (frame.width, frame.height):
        raise ComputerError(
            "capture_dimensions_invalid",
            "The desktop capture does not match the pinned DWM frame.",
        )
    offset_x = frame.x - identity.bounds.x
    offset_y = frame.y - identity.bounds.y
    image_module = _load_pillow()
    canvas = image_module.new(
        "RGB",
        (identity.bounds.width, identity.bounds.height),
        (0, 0, 0),
    )
    canvas.paste(frame_image, (offset_x, offset_y))
    return canvas


def _mask_desktop_capture_to_dwm_frame(image: Any, identity: WindowIdentity) -> Any:
    frame = _required_desktop_frame(identity)
    if image.size != (identity.bounds.width, identity.bounds.height):
        raise ComputerError(
            "capture_dimensions_invalid",
            "The desktop capture does not match the target window bounds.",
        )
    offset_x = frame.x - identity.bounds.x
    offset_y = frame.y - identity.bounds.y
    return _place_desktop_frame_on_window_canvas(
        image.crop(
            (
                offset_x,
                offset_y,
                offset_x + frame.width,
                offset_y + frame.height,
            )
        ),
        identity,
        frame,
    )


def _window_above_intersects(
    backend: Win32Backend,
    identity: WindowIdentity,
) -> bool:
    modules = backend.modules()
    target_left = identity.bounds.x
    target_top = identity.bounds.y
    target_right = target_left + identity.bounds.width
    target_bottom = target_top + identity.bounds.height
    blocked = False
    target_seen = False

    def callback(hwnd: int, _extra: object) -> bool:
        nonlocal blocked, target_seen
        if int(hwnd) == identity.hwnd:
            target_seen = True
            return False
        try:
            if (
                not modules.win32gui.IsWindowVisible(hwnd)
                or modules.win32gui.IsIconic(hwnd)
                or backend._is_cloaked(hwnd)
            ):
                return True
            left, top, right, bottom = modules.win32gui.GetWindowRect(hwnd)
            if (
                int(left) < target_right
                and int(right) > target_left
                and int(top) < target_bottom
                and int(bottom) > target_top
            ):
                blocked = True
                return False
        except Exception:
            blocked = True
            return False
        return True

    try:
        modules.win32gui.EnumWindows(callback, None)
    except modules.win32gui.error:
        return True
    return blocked or not target_seen


def _capture_print_window_isolated(
    backend: Win32Backend,
    identity: WindowIdentity,
    *,
    timeout_seconds: float = CAPTURE_TIMEOUT_SECONDS,
    deadline: float | None = None,
) -> CapturedImage:
    return _capture_worker_isolated(
        backend,
        identity,
        mode="print_window",
        backend_name="win32_print_window_isolated",
        backend_details={
            "capture_api": "PrintWindow",
            "isolated_worker": True,
            "hardware_accelerated_content_may_be_unavailable": True,
        },
        timeout_seconds=timeout_seconds,
        deadline=deadline,
    )


def _capture_worker_isolated(
    backend: Win32Backend,
    identity: WindowIdentity,
    *,
    mode: str,
    backend_name: str,
    backend_details: dict[str, Any],
    timeout_seconds: float,
    deadline: float | None = None,
) -> CapturedImage:
    capture_deadline = (
        deadline if deadline is not None else time.monotonic() + timeout_seconds
    )
    process: subprocess.Popen[bytes] | None = None
    captured: CapturedImage | None = None
    active_error: ComputerError | None = None
    try:
        _capture_time_remaining(capture_deadline)
        request = {
            "identity": {
                "hwnd": identity.hwnd,
                "pid": identity.pid,
                "thread_id": identity.thread_id,
                "process": identity.process,
                "title": identity.title,
                "class_name": identity.class_name,
                "bounds": identity.bounds.to_jsonable(),
                "process_started": identity.process_started,
                "session_id": identity.session_id,
                "desktop_name": identity.desktop_name,
            },
            "mode": mode,
        }
        source_root = str(Path(__file__).resolve().parents[2])
        environment = _worker_environment(source_root)
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        )
        request_bytes = json.dumps(request).encode("utf-8")
        if len(request_bytes) > MAX_CAPTURE_WORKER_REQUEST_BYTES:
            raise ComputerError(
                "capture_worker_request_too_large",
                "The capture worker request exceeded its safety limit.",
            )
        try:
            request_stream = preloaded_process_stdin(request_bytes)
        except ProcessInputStagingError as exc:
            staging_details: dict[str, Any] = {
                "secondary_errors": [f"staging:{exc.stage_error}"]
            }
            if exc.cleanup_error is not None:
                staging_details["steps"] = [
                    f"staged_stdin:{exc.cleanup_error}"
                ]
                raise ComputerError(
                    "capture_cleanup_failed",
                    "The failed capture-worker input stage could not be closed safely.",
                    details=staging_details,
                ) from exc
            raise ComputerError(
                "capture_backend_failed",
                "The capture-worker input could not be staged.",
                details=staging_details,
            ) from exc
        spawn_error: BaseException | None = None
        try:
            if (
                capture_deadline - time.monotonic()
                <= MIN_PROCESS_LIFECYCLE_SECONDS
            ):
                raise ComputerError(
                    "capture_timeout",
                    "The capture backend did not have enough lifecycle budget to start safely.",
                )
            process = subprocess.Popen(
                [sys.executable, "-P", "-m", "agent_tools.computer.capture_worker"],
                stdin=request_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                shell=False,
                env=environment,
                cwd=source_root,
                creationflags=creationflags,
            )
        except BaseException as exc:
            spawn_error = exc
        staged_close_error = close_preloaded_process_stdin(request_stream)
        if staged_close_error is not None:
            staged_close_details: dict[str, Any] = {
                "steps": [f"staged_stdin:{staged_close_error}"]
            }
            if isinstance(spawn_error, ComputerError):
                staged_close_details["primary_error"] = spawn_error.code
            raise ComputerError(
                "capture_cleanup_failed",
                "The staged capture-worker input could not be closed safely.",
                details=staged_close_details,
            ) from spawn_error
        if spawn_error is not None:
            raise spawn_error
        if process is None:
            raise ComputerError(
                "capture_backend_failed",
                "The capture worker did not start.",
            )
        max_png_bytes = (
            identity.bounds.width * identity.bounds.height * 4
            + MAX_CAPTURE_PROTOCOL_OVERHEAD_BYTES
        )
        stdout, _stderr = _read_worker_bounded(
            process,
            timeout_seconds=min(
                timeout_seconds,
                _capture_time_remaining(capture_deadline),
            ),
            max_stdout_bytes=max_png_bytes,
            max_stderr_bytes=MAX_CAPTURE_WORKER_ERROR_BYTES,
            deadline=capture_deadline,
        )
        header, separator, png_payload = stdout.partition(b"\n")
        if not separator:
            raise ComputerError(
                "capture_worker_failed",
                "The capture worker returned an invalid response.",
            )
        try:
            response = json.loads(header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ComputerError(
                "capture_worker_failed",
                "The capture worker returned an invalid response.",
            ) from exc
        if not isinstance(response, dict):
            raise ComputerError(
                "capture_worker_failed",
                "The capture worker returned an invalid response.",
            )
        if process.returncode != 0 or response.get("ok") is not True:
            error = response.get("error") if isinstance(response, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            raise ComputerError(
                str(code or "capture_worker_failed"),
                str(message or "The capture worker failed."),
            )
        metadata = _parse_worker_metadata(response)

        _capture_time_remaining(capture_deadline)
        with backend.native_pixel_context():
            backend.revalidate(identity)
        if backend.is_minimized(identity.hwnd):
            raise ComputerError(
                "window_minimized",
                f"HWND {identity.hwnd} became minimized during capture.",
            )
        _validate_worker_target_dimensions(metadata, identity)
        if len(png_payload) != metadata.png_bytes:
            raise ComputerError(
                "capture_worker_failed",
                "The capture worker returned incomplete image bytes.",
            )
        image = _decode_worker_png(png_payload, metadata)
        _capture_time_remaining(capture_deadline)
        captured = CapturedImage(
            image=image,
            width=metadata.width,
            height=metadata.height,
            captured_unix_ms=metadata.captured_unix_ms,
            backend=backend_name,
            cleanup_verified=metadata.cleanup_verified,
            backend_details=backend_details,
        )
    except ComputerError as exc:
        active_error = exc
        raise
    finally:
        termination_error: ComputerError | None = None
        pipe_cleanup_errors: list[str] = []
        if process is not None and process.poll() is None:
            try:
                _kill_worker(process, capture_deadline)
            except ComputerError as exc:
                termination_error = exc
        if process is not None:
            pipe_cleanup_errors = _close_capture_worker_pipes(
                process,
                deadline=capture_deadline,
            )
        if termination_error is not None:
            if (
                active_error is not None
                and active_error.code == "capture_worker_termination_failed"
            ):
                _merge_secondary_cleanup_errors(
                    active_error,
                    ["outer:termination_retry_failed", *pipe_cleanup_errors],
                )
            else:
                secondary_errors = list(pipe_cleanup_errors)
                if active_error is not None:
                    secondary_errors.extend(
                        f"primary_cleanup:{step}"
                        for step in active_error.details.get("steps", [])
                        if isinstance(step, str)
                    )
                    secondary_errors.append(f"primary:{active_error.code}")
                _merge_secondary_cleanup_errors(
                    termination_error,
                    secondary_errors,
                )
                raise termination_error
        if pipe_cleanup_errors:
            if (
                active_error is not None
                and active_error.code == "capture_worker_termination_failed"
            ):
                _merge_secondary_cleanup_errors(active_error, pipe_cleanup_errors)
            elif (
                active_error is not None
                and active_error.code == "capture_cleanup_failed"
            ):
                _merge_cleanup_steps(active_error, pipe_cleanup_errors)
            else:
                raise ComputerError(
                    "capture_cleanup_failed",
                    "The capture worker pipes could not be closed safely.",
                    details={"steps": pipe_cleanup_errors},
                )
    _capture_time_remaining(capture_deadline)
    if captured is None:
        raise ComputerError(
            "capture_worker_failed",
            "The capture worker returned no screenshot.",
        )
    return captured


def _merge_secondary_cleanup_errors(
    error: ComputerError,
    additions: list[str],
) -> None:
    existing = error.details.get("secondary_cleanup_errors", [])
    merged = [item for item in existing if isinstance(item, str)]
    for item in additions:
        if item not in merged:
            merged.append(item)
    if merged:
        error.details["secondary_cleanup_errors"] = merged[:8]


def _merge_cleanup_steps(error: ComputerError, additions: list[str]) -> None:
    existing = error.details.get("steps", [])
    merged = [item for item in existing if isinstance(item, str)]
    for item in additions:
        if item not in merged:
            merged.append(item)
    if merged:
        error.details["steps"] = merged[:8]


def _parse_worker_metadata(response: dict[str, Any]) -> CaptureWorkerMetadata:
    integer_fields = ("width", "height", "png_bytes", "captured_unix_ms")
    if response.get("ok") is not True or any(
        type(response.get(field)) is not int for field in integer_fields
    ):
        raise ComputerError(
            "capture_worker_failed",
            "The capture worker returned invalid image metadata.",
        )
    width = response["width"]
    height = response["height"]
    png_bytes = response["png_bytes"]
    captured_unix_ms = response["captured_unix_ms"]
    cleanup_verified = response.get("cleanup_verified")
    if (
        width <= 0
        or height <= 0
        or png_bytes < 0
        or captured_unix_ms < 0
        or type(cleanup_verified) is not bool
    ):
        raise ComputerError(
            "capture_worker_failed",
            "The capture worker returned invalid image metadata.",
        )
    return CaptureWorkerMetadata(
        width=width,
        height=height,
        png_bytes=png_bytes,
        captured_unix_ms=captured_unix_ms,
        cleanup_verified=cleanup_verified,
    )


def _decode_worker_png(payload: bytes, metadata: CaptureWorkerMetadata) -> Any:
    image_module = _load_pillow()
    try:
        with image_module.open(io.BytesIO(payload)) as encoded_image:
            if encoded_image.size != (metadata.width, metadata.height):
                raise ComputerError(
                    "capture_dimensions_invalid",
                    "The capture worker image dimensions do not match its response.",
                )
            encoded_image.load()
            return encoded_image.convert("RGB").copy()
    except ComputerError:
        raise
    except Exception as exc:
        raise ComputerError(
            "capture_worker_failed",
            "The capture worker produced an invalid image.",
        ) from exc


def _validate_worker_target_dimensions(
    metadata: CaptureWorkerMetadata, identity: WindowIdentity
) -> None:
    if (metadata.width, metadata.height) != (
        identity.bounds.width,
        identity.bounds.height,
    ):
        raise ComputerError(
            "capture_dimensions_invalid",
            "The capture worker dimensions do not match the requested target.",
        )


def _kill_worker(process: subprocess.Popen[bytes], deadline: float) -> None:
    try:
        process.kill()
    except OSError as exc:
        raise ComputerError(
            "capture_worker_termination_failed",
            "The capture worker could not be terminated.",
        ) from exc
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ComputerError(
            "capture_worker_termination_failed",
            "The capture worker termination could not be verified before the deadline.",
        )
    try:
        process.wait(timeout=remaining)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComputerError(
            "capture_worker_termination_failed",
            "The capture worker could not be terminated.",
        ) from exc


def _read_worker_bounded(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    deadline: float | None = None,
) -> tuple[bytes, bytes]:
    cleanup_deadline = min(
        time.monotonic() + timeout_seconds,
        deadline if deadline is not None else math.inf,
    )
    cleanup_reserve = min(0.25, max(0.01, timeout_seconds / 2))
    io_deadline = cleanup_deadline - cleanup_reserve
    if process.stdout is None or process.stderr is None:
        _kill_worker(process, cleanup_deadline)
        raise ComputerError(
            "capture_worker_failed",
            "The capture worker pipes are unavailable.",
        )
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    timed_out = False
    output_too_large = False
    termination_error: ComputerError | None = None
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    closed: set[str] = set()
    read_error: OSError | None = None
    while True:
        made_progress = False
        try:
            for name, stream in streams.items():
                if name in closed:
                    continue
                chunk, stream_closed = _read_capture_pipe_available(stream)
                if stream_closed:
                    closed.add(name)
                if not chunk:
                    continue
                made_progress = True
                limit = max_stdout_bytes if name == "stdout" else max_stderr_bytes
                if len(buffers[name]) + len(chunk) > limit:
                    output_too_large = True
                    break
                buffers[name].extend(chunk)
        except OSError as exc:
            read_error = exc
            break
        if output_too_large:
            break
        if process.poll() is not None:
            if len(closed) == len(streams) or not made_progress:
                break
            continue
        if time.monotonic() >= io_deadline:
            timed_out = True
            break
        time.sleep(min(0.01, max(0.0, io_deadline - time.monotonic())))

    if process.poll() is None:
        try:
            _kill_worker(process, cleanup_deadline)
        except ComputerError as exc:
            termination_error = exc
    pipe_cleanup_errors: list[str] = []
    pipe_cleanup_errors = _close_capture_worker_pipes(
        process,
        deadline=cleanup_deadline,
    )
    if termination_error is not None:
        secondary_errors = list(pipe_cleanup_errors)
        if secondary_errors:
            termination_error.details["secondary_cleanup_errors"] = secondary_errors
        raise termination_error
    if pipe_cleanup_errors:
        raise ComputerError(
            "capture_cleanup_failed",
            "The capture worker pipes could not be closed safely.",
            details={"steps": pipe_cleanup_errors},
        )
    if timed_out:
        raise ComputerError(
            "capture_timeout",
            "The target window did not respond before the capture deadline.",
        )
    if output_too_large:
        raise ComputerError(
            "capture_worker_output_too_large",
            "The capture worker exceeded its output safety limit.",
        )
    if read_error is not None:
        raise ComputerError(
            "capture_worker_failed",
            "The capture worker output could not be read safely.",
        ) from read_error
    return bytes(buffers["stdout"]), bytes(buffers["stderr"])


def _read_capture_pipe_available(stream: Any) -> tuple[bytes, bool]:
    if sys.platform == "win32":
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.PeekNamedPipe.argtypes = (
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPDWORD,
            wintypes.LPDWORD,
            wintypes.LPDWORD,
        )
        kernel32.PeekNamedPipe.restype = wintypes.BOOL
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
        available = wintypes.DWORD()
        if not kernel32.PeekNamedPipe(
            handle,
            None,
            0,
            None,
            ctypes.byref(available),
            None,
        ):
            error = ctypes.get_last_error()
            if error in {109, 232}:
                return b"", True
            raise OSError(error, "PeekNamedPipe failed")
        if available.value == 0:
            return b"", False
        return os.read(stream.fileno(), min(4096, int(available.value))), False

    import select

    ready, _, _ = select.select([stream], [], [], 0)
    if not ready:
        return b"", False
    chunk = os.read(stream.fileno(), 4096)
    return chunk, not chunk


def _close_capture_worker_pipes(
    process: subprocess.Popen[bytes],
    *,
    deadline: float | None = None,
) -> list[str]:
    errors: list[str] = []
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, name, None)
        if stream is None or stream.closed:
            continue
        if deadline is not None and deadline - time.monotonic() <= 0:
            errors.append(f"{name}:deadline")
        close_error = close_stream_os_enforced(stream)
        if close_error is not None:
            errors.append(f"{name}:{close_error}")
    return errors


def _worker_environment(source_root: str) -> dict[str, str]:
    allowed_names = (
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "PATHEXT",
        "PYTHONHOME",
    )
    environment = {
        name: os.environ[name]
        for name in allowed_names
        if name in os.environ
    }
    environment.update(
        {
            "PYTHONPATH": source_root,
            "PYTHONSAFEPATH": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    return environment


def _capture_full(
    backend: Win32Backend,
    identity: WindowIdentity,
    *,
    force_failure_after_selection: bool = False,
) -> CapturedImage:
    if isinstance(backend, Win32Backend):
        with backend.native_pixel_context():
            return _capture_full_native(
                backend,
                identity,
                force_failure_after_selection=force_failure_after_selection,
            )
    return _capture_full_native(
        backend,
        identity,
        force_failure_after_selection=force_failure_after_selection,
    )


def _capture_full_native(
    backend: Win32Backend,
    identity: WindowIdentity,
    *,
    force_failure_after_selection: bool = False,
) -> CapturedImage:
    modules = backend.modules()
    image_module = _load_pillow()
    user32 = _windll("user32")
    gdi32 = _windll("gdi32")
    user32.PrintWindow.argtypes = (wintypes.HWND, wintypes.HDC, wintypes.UINT)
    user32.PrintWindow.restype = wintypes.BOOL
    user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
    user32.ReleaseDC.restype = ctypes.c_int
    gdi32.DeleteDC.argtypes = (wintypes.HDC,)
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
    gdi32.DeleteObject.restype = wintypes.BOOL

    width = identity.bounds.width
    height = identity.bounds.height
    cleanup_steps: list[tuple[str, Any]] = []
    cleanup_completed: list[str] = []
    cleanup_errors: list[str] = []
    image: Any | None = None
    captured_unix_ms: int | None = None

    try:
        backend.revalidate(identity)
        if backend.is_minimized(identity.hwnd):
            raise ComputerError(
                "window_minimized",
                f"HWND {identity.hwnd} became minimized before capture.",
            )
        hwnd_dc = modules.win32gui.GetWindowDC(identity.hwnd)
        if not hwnd_dc:
            raise ComputerError("capture_backend_failed", "Could not acquire the window DC.")

        def release_window_dc() -> None:
            if not user32.ReleaseDC(identity.hwnd, hwnd_dc):
                raise RuntimeError("ReleaseDC failed")

        cleanup_steps.append(("window_dc_released", release_window_dc))
        compatible_dc = modules.win32gui.CreateCompatibleDC(hwnd_dc)
        if not compatible_dc:
            raise ComputerError("capture_backend_failed", "Could not create a capture DC.")

        def delete_compatible_dc() -> None:
            if not gdi32.DeleteDC(compatible_dc):
                raise RuntimeError("DeleteDC failed")

        cleanup_steps.append(("compatible_dc_deleted", delete_compatible_dc))
        bitmap_owner = modules.win32gui.CreateCompatibleBitmap(hwnd_dc, width, height)
        bitmap_handle = bitmap_owner.Detach()

        def delete_bitmap() -> None:
            if not gdi32.DeleteObject(bitmap_handle):
                raise RuntimeError("DeleteObject failed")

        cleanup_steps.append(("bitmap_deleted", delete_bitmap))
        previous_bitmap = modules.win32gui.SelectObject(compatible_dc, bitmap_handle)
        cleanup_steps.append(
            (
                "previous_bitmap_restored",
                lambda: modules.win32gui.SelectObject(compatible_dc, previous_bitmap),
            )
        )
        if force_failure_after_selection:
            raise ComputerError(
                "forced_capture_failure",
                "Forced capture failure after bitmap selection.",
            )

        backend.revalidate(identity)
        if backend.is_minimized(identity.hwnd):
            raise ComputerError(
                "window_minimized",
                f"HWND {identity.hwnd} became minimized before capture.",
            )
        captured_unix_ms = int(time.time() * 1000)
        printed = bool(
            user32.PrintWindow(identity.hwnd, compatible_dc, PRINT_WINDOW_FLAGS)
        )
        if not printed:
            raise ComputerError(
                "capture_backend_failed",
                "PrintWindow did not capture the requested window.",
            )
        bitmap = modules.win32ui.CreateBitmapFromHandle(bitmap_handle)
        info = bitmap.GetInfo()
        bits = bitmap.GetBitmapBits(True)
        image = image_module.frombuffer(
            "RGB",
            (int(info["bmWidth"]), int(info["bmHeight"])),
            bits,
            "raw",
            "BGRX",
            0,
            1,
        ).copy()
    except ComputerError:
        raise
    except Exception as exc:
        raise ComputerError(
            "capture_backend_failed",
            "The Win32 capture backend failed.",
        ) from exc
    finally:
        for label, cleanup in reversed(cleanup_steps):
            try:
                cleanup()
                cleanup_completed.append(label)
            except Exception as exc:  # noqa: BLE001 - every GDI cleanup failure matters
                cleanup_errors.append(f"{label}:{type(exc).__name__}")
        if cleanup_errors:
            raise ComputerError(
                "capture_cleanup_failed",
                "The capture completed with a GDI cleanup failure.",
                details={"steps": cleanup_errors},
            )

    if image is None or captured_unix_ms is None:
        raise ComputerError("capture_backend_failed", "The capture backend returned no image.")
    expected_cleanup = {
        "previous_bitmap_restored",
        "bitmap_deleted",
        "compatible_dc_deleted",
        "window_dc_released",
    }
    return CapturedImage(
        image=image,
        width=int(image.width),
        height=int(image.height),
        captured_unix_ms=captured_unix_ms,
        backend="win32_print_window",
        cleanup_verified=set(cleanup_completed) == expected_cleanup,
    )


def _validate_window_size(identity: WindowIdentity) -> None:
    width = identity.bounds.width
    height = identity.bounds.height
    if width <= 0 or height <= 0:
        raise ComputerError(
            "invalid_window_bounds",
            f"HWND {identity.hwnd} has zero-sized bounds.",
        )
    if (
        width > MAX_CAPTURE_DIMENSION
        or height > MAX_CAPTURE_DIMENSION
        or width * height > MAX_CAPTURE_PIXELS
    ):
        raise ComputerError(
            "capture_too_large",
            "The requested window is larger than the capture safety limit.",
        )


def _validate_region(region: Region | None, identity: WindowIdentity) -> None:
    if region is None:
        return
    if (
        region.x + region.width > identity.bounds.width
        or region.y + region.height > identity.bounds.height
    ):
        raise ComputerError(
            "region_out_of_bounds",
            "The requested region extends outside the target window.",
            details={
                "window_width": identity.bounds.width,
                "window_height": identity.bounds.height,
            },
        )


def _validate_output_path(output: Path) -> Path:
    if output.suffix.casefold() != ".png":
        raise ComputerError(
            "invalid_output_path",
            "Screenshot output must use a .png extension.",
            exit_code=2,
        )
    parent = output.parent if str(output.parent) else Path(".")
    if not parent.is_dir():
        raise ComputerError(
            "output_parent_missing",
            "The screenshot output directory does not exist.",
            exit_code=2,
        )
    if output.exists() and not output.is_file():
        raise ComputerError(
            "invalid_output_path",
            "Screenshot output points to a non-file target.",
            exit_code=2,
        )
    if output.exists() and _is_reparse_point(output):
        raise ComputerError(
            "invalid_output_path",
            "Screenshot output cannot use a reparse point.",
            exit_code=2,
        )
    return output


def _load_pillow() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise ComputerError(
            "screenshot_backend_unavailable",
            "Pillow is not installed; install the computer extra.",
            exit_code=2,
        ) from exc


def _image_non_uniform(image: Any) -> bool:
    low, high = image.convert("L").getextrema()
    return bool(low != high)


def analyze_capture_quality(image: Any) -> CaptureQuality:
    """Measure useful structure from native-stage pixels on a bounded sample grid."""
    if image.width <= 0 or image.height <= 0:
        raise ComputerError(
            "capture_dimensions_invalid",
            "Screenshot quality analysis requires positive dimensions.",
        )
    analysis_width, analysis_height = _analysis_size(image.width, image.height)
    image_module = _load_pillow()
    resampling = getattr(image_module, "Resampling", image_module)
    if image.size == (analysis_width, analysis_height):
        sampled = image.convert("RGBA")
    else:
        sampled = image.resize(
            (analysis_width, analysis_height),
            resample=resampling.NEAREST,
        ).convert("RGBA")
    flattened = getattr(sampled, "get_flattened_data", sampled.getdata)
    pixels = list(flattened())
    total = len(pixels)
    transparent = 0
    visible_luminances: list[int] = []
    visible_quantized: list[tuple[int, int, int] | None] = []
    quantized_counts: dict[tuple[int, int, int], int] = {}
    luminance_grid: list[int | None] = []
    for red, green, blue, alpha in pixels:
        if alpha <= QUALITY_THRESHOLDS["transparent_alpha_max"]:
            transparent += 1
            visible_quantized.append(None)
            luminance_grid.append(None)
            continue
        luminance = (77 * red + 150 * green + 29 * blue) >> 8
        quantized = (red >> 3, green >> 3, blue >> 3)
        visible_luminances.append(luminance)
        visible_quantized.append(quantized)
        luminance_grid.append(luminance)
        quantized_counts[quantized] = quantized_counts.get(quantized, 0) + 1

    visible = len(visible_luminances)
    transparent_ratio = transparent / total
    if visible:
        black_ratio = sum(
            value <= QUALITY_THRESHOLDS["black_luminance_max"]
            for value in visible_luminances
        ) / visible
        white_ratio = sum(
            value >= QUALITY_THRESHOLDS["white_luminance_min"]
            for value in visible_luminances
        ) / visible
        dominant_color, dominant_count = max(
            quantized_counts.items(), key=lambda item: item[1]
        )
        dominant_ratio = dominant_count / visible
        active_indexes = [
            index
            for index, value in enumerate(visible_quantized)
            if value is not None and value != dominant_color
        ]
        active_ratio = len(active_indexes) / visible
        active_bbox_fraction = _active_bbox_fraction(
            active_indexes,
            analysis_width,
            analysis_height,
        )
        luminance_histogram = [0] * 256
        for value in visible_luminances:
            luminance_histogram[value] += 1
        luminance_min = min(visible_luminances)
        luminance_max = max(visible_luminances)
        percentile_low = _histogram_percentile(luminance_histogram, visible, 0.01)
        percentile_high = _histogram_percentile(luminance_histogram, visible, 0.99)
        luminance_spread = percentile_high - percentile_low
        entropy = -sum(
            (count / visible) * math.log2(count / visible)
            for count in luminance_histogram
            if count
        )
        edge_density = _edge_density(
            luminance_grid,
            analysis_width,
            analysis_height,
        )
    else:
        black_ratio = 0.0
        white_ratio = 0.0
        dominant_ratio = 0.0
        active_ratio = 0.0
        active_bbox_fraction = 0.0
        luminance_min = 0
        luminance_max = 0
        percentile_low = 0
        percentile_high = 0
        luminance_spread = 0
        entropy = 0.0
        edge_density = 0.0

    reasons: list[str] = []
    if transparent_ratio >= QUALITY_THRESHOLDS["transparent_coverage_reject"]:
        reasons.append("extreme_transparent_coverage")
    if visible and black_ratio >= QUALITY_THRESHOLDS["extreme_coverage_reject"]:
        reasons.append("extreme_black_coverage")
    if visible and white_ratio >= QUALITY_THRESHOLDS["extreme_coverage_reject"]:
        reasons.append("extreme_white_coverage")
    if (
        visible
        and dominant_ratio >= QUALITY_THRESHOLDS["tiny_mark_dominant_ratio_min"]
        and active_ratio <= QUALITY_THRESHOLDS["tiny_mark_active_ratio_max"]
        and active_bbox_fraction
        <= QUALITY_THRESHOLDS["tiny_mark_bbox_fraction_max"]
        and edge_density <= QUALITY_THRESHOLDS["tiny_mark_edge_density_max"]
    ):
        reasons.append("tiny_active_content")
    if (
        visible
        and luminance_spread
        <= QUALITY_THRESHOLDS["low_information_luminance_spread_max"]
        and entropy <= QUALITY_THRESHOLDS["low_information_entropy_max"]
        and edge_density
        <= QUALITY_THRESHOLDS["low_information_edge_density_max"]
    ):
        reasons.append("low_information_frame")
    metrics = {
        "transparent_coverage": round(transparent_ratio, 6),
        "near_black_coverage": round(black_ratio, 6),
        "near_white_coverage": round(white_ratio, 6),
        "dominant_color_ratio": round(dominant_ratio, 6),
        "active_pixel_ratio": round(active_ratio, 6),
        "active_bbox_fraction": round(active_bbox_fraction, 6),
        "luminance_min": luminance_min,
        "luminance_max": luminance_max,
        "luminance_percentile_01": percentile_low,
        "luminance_percentile_99": percentile_high,
        "luminance_percentile_spread": luminance_spread,
        "luminance_entropy_bits": round(entropy, 6),
        "edge_density": round(edge_density, 6),
    }
    return CaptureQuality(
        accepted=not reasons,
        reasons=tuple(reasons),
        metrics=metrics,
        analysis_size=(analysis_width, analysis_height),
    )


def _analysis_size(width: int, height: int) -> tuple[int, int]:
    if max(width, height) <= MAX_QUALITY_ANALYSIS_EDGE:
        return width, height
    if width >= height:
        return (
            MAX_QUALITY_ANALYSIS_EDGE,
            max(1, _round_half_up(height * MAX_QUALITY_ANALYSIS_EDGE, width)),
        )
    return (
        max(1, _round_half_up(width * MAX_QUALITY_ANALYSIS_EDGE, height)),
        MAX_QUALITY_ANALYSIS_EDGE,
    )


def _histogram_percentile(
    histogram: list[int],
    total: int,
    percentile: float,
) -> int:
    target = max(1, math.ceil(total * percentile))
    observed = 0
    for value, count in enumerate(histogram):
        observed += count
        if observed >= target:
            return value
    return 255


def _active_bbox_fraction(
    indexes: list[int],
    width: int,
    height: int,
) -> float:
    if not indexes:
        return 0.0
    xs = [index % width for index in indexes]
    ys = [index // width for index in indexes]
    bbox_pixels = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
    return bbox_pixels / (width * height)


def _edge_density(
    luminances: list[int | None],
    width: int,
    height: int,
) -> float:
    edge_count = 0
    comparison_count = 0
    threshold = QUALITY_THRESHOLDS["edge_luminance_delta_min"]
    for y in range(height):
        row = y * width
        for x in range(width):
            current = luminances[row + x]
            if current is None:
                continue
            if x + 1 < width:
                right = luminances[row + x + 1]
                if right is not None:
                    comparison_count += 1
                    edge_count += abs(current - right) >= threshold
            if y + 1 < height:
                below = luminances[row + width + x]
                if below is not None:
                    comparison_count += 1
                    edge_count += abs(current - below) >= threshold
    return edge_count / comparison_count if comparison_count else 0.0


def _requested_native_image(image: Any, region: Region | None) -> Any:
    if region is None:
        return image
    return image.crop(
        (
            region.x,
            region.y,
            region.x + region.width,
            region.y + region.height,
        )
    )


def _validate_captured_target_dimensions(
    captured: CapturedImage,
    identity: WindowIdentity,
) -> None:
    expected = (identity.bounds.width, identity.bounds.height)
    if (
        captured.width,
        captured.height,
        int(captured.image.width),
        int(captured.image.height),
    ) != (expected[0], expected[1], expected[0], expected[1]):
        raise ComputerError(
            "capture_dimensions_invalid",
            "The captured image dimensions do not match the exact target window.",
        )


def _capture_attempt(
    backend: str,
    outcome: str,
    duration_ms: int,
    *,
    quality: CaptureQuality | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "backend": backend,
        "outcome": outcome,
        "duration_ms": duration_ms,
    }
    if error_code is not None:
        result["error_code"] = error_code
    if quality is not None:
        result["quality"] = {
            "reasons": list(quality.reasons),
            "analysis_pixel_size": {
                "width": quality.analysis_size[0],
                "height": quality.analysis_size[1],
            },
            "metrics": quality.metrics,
        }
    return result


def _capture_time_remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ComputerError(
            "capture_timeout",
            "The capture backend chain did not finish within its safety deadline.",
        )
    return remaining


def _duration_ms(started: float) -> int:
    return min(60_000, max(0, int((time.monotonic() - started) * 1000)))


def _capture_limitations(
    backend: str,
    backend_details: dict[str, Any] | None,
) -> list[str]:
    limitations = ["protected_content_may_be_unavailable"]
    if backend == "winapp_window_capture":
        limitations.append("effective_capture_api_unreported_by_winapp_0.6.0")
    elif backend == "win32_print_window_isolated":
        limitations.append("hardware_accelerated_content_may_be_unavailable")
    elif backend == "desktop_crop_visible":
        limitations.append("requires_foreground_and_full_visibility")
    normalization_method = (
        backend_details.get("normalization", {}).get("method")
        if backend_details
        else None
    )
    if normalization_method in {
        "dwm_extended_frame_to_window_canvas",
        "dwm_extended_frame_masked_to_window_canvas",
        "winapp_header_cropped_dwm_frame_to_window_canvas",
    }:
        limitations.append("non_rendered_window_border_is_canvas_padding")
    return limitations


def _image_all_black(image: Any) -> bool:
    low, high = image.convert("L").getextrema()
    return bool(low == 0 and high == 0)


def _encode_png(image: Any) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _verify_png(payload: bytes, width: int, height: int) -> None:
    image_module = _load_pillow()
    try:
        with image_module.open(io.BytesIO(payload)) as image:
            image.verify()
        with image_module.open(io.BytesIO(payload)) as image:
            if image.size != (width, height):
                raise ComputerError(
                    "capture_dimensions_invalid",
                    "The encoded PNG dimensions do not match the request.",
                )
    except ComputerError:
        raise
    except Exception as exc:
        raise ComputerError(
            "capture_encoding_failed",
            "The screenshot PNG failed validation.",
        ) from exc


def _atomic_write(output: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    write_error: ComputerError | None = None
    cleanup_error: OSError | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    except OSError:
        write_error = ComputerError(
            "output_write_failed",
            "The screenshot could not be written to the requested output path.",
        )
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_error = exc
    if cleanup_error is not None:
        raise ComputerError(
            "output_cleanup_failed",
            "The temporary screenshot output could not be removed.",
            details={
                "original_error_code": (
                    write_error.code if write_error is not None else None
                )
            },
        ) from cleanup_error
    if write_error is not None:
        raise write_error
