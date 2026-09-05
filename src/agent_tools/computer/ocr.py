from __future__ import annotations

import asyncio
import importlib.metadata
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from agent_tools.computer.capture_records import CaptureRecordStore
from agent_tools.computer.models import ComputerError
from agent_tools.computer.screenshot import NativeCaptureSource, resolve_native_capture_source

OCR_RESULT_SCHEMA = "agent-tools.ocr/v1"
DEFAULT_MAX_OCR_LINES = 200
DEFAULT_MAX_OCR_CHARS = 16_000
MAX_OCR_LINES = 500
MAX_OCR_CHARS = 50_000
MAX_OCR_ENGINE_LINES = MAX_OCR_LINES
MAX_OCR_ENGINE_WORDS_PER_WINDOW = 5_000
MAX_OCR_LINE_CHARS = 4_096
OCR_BINDING_PACKAGE = "winrt-Windows.Media.Ocr"
AGENT_TOOLS_DISTRIBUTION = "ai-nd-co-agent-tools"
_CYRILLIC_RE = re.compile(r"[\u0400-\u052f]")
_LATIN_RE = re.compile(r"[A-Za-z]")


class OcrBackend(Protocol):
    def recognize(self, png_bytes: bytes, *, width: int, height: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _BoundedLine:
    text: str
    words: list[dict[str, Any]]
    bounds: dict[str, float] | None
    boxes_omitted_due_to_text_truncation: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bounds": self.bounds,
            "words": self.words,
            "boxes_omitted_due_to_text_truncation": (self.boxes_omitted_due_to_text_truncation),
        }


class WindowsOcrBackend:
    name = "windows_media_ocr"

    def recognize(self, png_bytes: bytes, *, width: int, height: int) -> dict[str, Any]:
        if sys.platform != "win32":
            raise ComputerError(
                "unsupported_platform",
                "Local terminal OCR requires Windows.",
                exit_code=2,
            )
        try:
            return asyncio.run(self._recognize(png_bytes, width=width, height=height))
        except ComputerError:
            raise
        except (ImportError, ModuleNotFoundError) as exc:
            raise ComputerError(
                "ocr_backend_unavailable",
                "The local Windows OCR binding is unavailable.",
                exit_code=2,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - WinRT failures become bounded public errors
            raise ComputerError(
                "ocr_failed",
                "The local Windows OCR engine could not read the native capture.",
                details={"failure_type": type(exc).__name__},
            ) from exc

    async def _recognize(
        self,
        png_bytes: bytes,
        *,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        try:
            (
                OcrEngine,
                Language,
                BitmapDecoder,
                DataWriter,
                InMemoryRandomAccessStream,
            ) = _load_windows_ocr_runtime()
        except (ImportError, ModuleNotFoundError) as exc:
            raise ComputerError(
                "ocr_backend_unavailable",
                "The local Windows OCR binding is unavailable.",
                exit_code=2,
            ) from exc

        max_dimension = int(OcrEngine.max_image_dimension)
        if width > max_dimension or height > max_dimension:
            raise ComputerError(
                "ocr_image_too_large",
                "The native capture exceeds the local Windows OCR dimension limit.",
                details={"max_dimension": max_dimension, "width": width, "height": height},
            )
        available = [item.language_tag for item in OcrEngine.available_recognizer_languages]
        engine = _select_engine(OcrEngine, Language, available)
        if engine is None:
            raise ComputerError(
                "ocr_language_unavailable",
                "Windows has no installed OCR language for this command.",
            )

        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        bitmap = None
        try:
            writer.write_bytes(png_bytes)
            await writer.store_async()
            writer.detach_stream()
            writer.close()
            stream.seek(0)
            decoder = await BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()
            result = await engine.recognize_async(bitmap)
            return _normalize_engine_result(
                result,
                engine_name=self.name,
                language_used=engine.recognizer_language.language_tag,
                available_languages=available,
                max_image_dimension=max_dimension,
                image_width=width,
                image_height=height,
            )
        finally:
            if bitmap is not None:
                bitmap.close()
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - already detached/closed is harmless
                pass
            stream.close()


def ocr_capture(
    *,
    capture_id: str,
    last_lines: int | None = None,
    max_lines: int = DEFAULT_MAX_OCR_LINES,
    max_chars: int = DEFAULT_MAX_OCR_CHARS,
    record_store: CaptureRecordStore | None = None,
    backend: OcrBackend | None = None,
) -> dict[str, Any]:
    _validate_output_limits(last_lines=last_lines, max_lines=max_lines, max_chars=max_chars)
    command_started = time.perf_counter()
    source = resolve_native_capture_source(
        capture_id=capture_id,
        record_store=record_store,
    )
    native_size = _required_size(source.record, "native_pixel_size")
    ocr_started = time.perf_counter()
    engine_result = (backend or WindowsOcrBackend()).recognize(
        source.png_bytes,
        width=native_size[0],
        height=native_size[1],
    )
    ocr_duration_ms = _elapsed_ms(ocr_started)
    bounded = _bounded_result(
        engine_result,
        last_lines=last_lines,
        max_lines=max_lines,
        max_chars=max_chars,
    )
    uncertainty = _uncertainty(engine_result, bounded)
    return {
        "schema_version": OCR_RESULT_SCHEMA,
        "source_capture": _source_provenance(source),
        "engine": {
            "name": engine_result.get("engine"),
            "binding_version": engine_result.get("binding_version"),
            "language_used": engine_result.get("language_used"),
            "available_languages": engine_result.get("available_languages") or [],
            "max_image_dimension": engine_result.get("max_image_dimension"),
            "confidence_available": False,
        },
        "text": bounded["text"],
        "lines": [line.public() for line in bounded["lines"]],
        "counts": bounded["counts"],
        "limits": bounded["limits"],
        "truncation": bounded["truncation"],
        "uncertainty": uncertainty,
        "timing": {
            "ocr_duration_ms": ocr_duration_ms,
            "command_elapsed_ms": _elapsed_ms(command_started),
            "source_capture_attempt_duration_ms": _capture_attempt_duration(source.record),
        },
        "observation_only": True,
        "authorizes_action": False,
    }


def ocr_capabilities() -> dict[str, Any]:
    source_version, distribution_version = _runtime_versions()
    version_match = bool(
        source_version is not None
        and distribution_version is not None
        and source_version == distribution_version
    )
    available = False
    binding_version: str | None = None
    languages: list[str] = []
    reason_code: str | None = None
    next_action: str | None = None
    failure_type: str | None = None
    if sys.platform != "win32":
        reason_code = "platform_unsupported"
        next_action = "Run local OCR from AgentTools on Windows 10 or Windows 11."
    elif not version_match:
        reason_code = "source_distribution_version_mismatch"
        next_action = "Reinstall this AgentTools source with the computer extra."
    else:
        try:
            binding_version, languages = _probe_windows_ocr()
        except (ImportError, ModuleNotFoundError, importlib.metadata.PackageNotFoundError):
            reason_code = "binding_not_installed"
            next_action = "Install or refresh AgentTools with the computer extra."
        except Exception as exc:  # noqa: BLE001 - public capability probe is bounded
            reason_code = "probe_failed"
            next_action = "Repair Windows OCR, then run computer capabilities again."
            failure_type = type(exc).__name__[:96]
        else:
            if languages:
                available = True
            else:
                reason_code = "no_windows_ocr_languages"
                next_action = "Install a Windows OCR language pack, then retry."
    return {
        "available": available,
        "status": "ready" if available else "unavailable",
        "reason_code": reason_code,
        "next_action": next_action,
        "engine": "windows_media_ocr",
        "source_version": source_version,
        "distribution_version": distribution_version,
        "source_distribution_version_match": version_match,
        "binding_version": binding_version,
        "available_languages": languages,
        "failure_type": failure_type,
        "local_only": True,
        "network": False,
        "requires_native_capture_id": True,
        "implicit_capture": False,
        "confidence_available": False,
    }


def _runtime_versions() -> tuple[str | None, str | None]:
    try:
        from agent_tools import __version__ as imported_source_version
    except (ImportError, AttributeError):
        source_version = None
    else:
        source_version = imported_source_version
    try:
        distribution_version = importlib.metadata.version(AGENT_TOOLS_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    return source_version, distribution_version


def _probe_windows_ocr() -> tuple[str, list[str]]:
    binding_version = importlib.metadata.version(OCR_BINDING_PACKAGE)
    (
        ocr_engine,
        language_type,
        _bitmap_decoder,
        _data_writer,
        _random_access_stream,
    ) = _load_windows_ocr_runtime()

    languages = [item.language_tag for item in ocr_engine.available_recognizer_languages]
    if languages:
        engine = _select_engine(ocr_engine, language_type, languages)
        if engine is None:
            raise RuntimeError("Windows OCR engine selection failed")
        language_tag = engine.recognizer_language.language_tag
        if not isinstance(language_tag, str) or not language_tag:
            raise RuntimeError("Windows OCR engine language is unavailable")
    return binding_version, languages


def _load_windows_ocr_runtime() -> tuple[Any, Any, Any, Any, Any]:
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

    return OcrEngine, Language, BitmapDecoder, DataWriter, InMemoryRandomAccessStream


def _select_engine(ocr_engine: Any, language_type: Any, available: list[str]) -> Any:
    russian = next((tag for tag in available if tag.casefold().startswith("ru")), None)
    if russian is not None:
        engine = ocr_engine.try_create_from_language(language_type(russian))
        if engine is not None:
            return engine
    return ocr_engine.try_create_from_user_profile_languages()


def _normalize_engine_result(
    result: Any,
    *,
    engine_name: str,
    language_used: str,
    available_languages: list[str],
    max_image_dimension: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    source_lines = list(result.lines)
    total_words = sum(len(list(line.words)) for line in source_lines)
    total_characters = sum(len(str(line.text)) for line in source_lines) + max(
        0, len(source_lines) - 1
    )
    head_indexes = list(range(min(len(source_lines), MAX_OCR_ENGINE_LINES)))
    tail_start = max(0, len(source_lines) - MAX_OCR_ENGINE_LINES)
    tail_indexes = list(range(tail_start, len(source_lines)))
    lines, head_truncated = _normalize_line_window(
        source_lines,
        indexes=head_indexes,
        image_width=image_width,
        image_height=image_height,
    )
    if tail_indexes == head_indexes:
        tail_lines = lines
        tail_truncated = head_truncated
    else:
        tail_lines, tail_truncated = _normalize_line_window(
            source_lines,
            indexes=tail_indexes,
            image_width=image_width,
            image_height=image_height,
        )
    retained_indexes = set(head_indexes + tail_indexes)
    engine_truncated = len(retained_indexes) < len(source_lines) or head_truncated or tail_truncated
    try:
        binding_version = importlib.metadata.version(OCR_BINDING_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        binding_version = None
    return {
        "engine": engine_name,
        "binding_version": binding_version,
        "language_used": language_used,
        "available_languages": available_languages,
        "max_image_dimension": max_image_dimension,
        "source_pixel_size": {"width": image_width, "height": image_height},
        "lines": lines,
        "tail_lines": tail_lines,
        "retained_line_count": len(retained_indexes),
        "original_line_count": len(source_lines),
        "original_word_count": total_words,
        "original_character_count": total_characters,
        "engine_output_truncated": engine_truncated,
    }


def _normalize_line_window(
    source_lines: list[Any],
    *,
    indexes: list[int],
    image_width: int,
    image_height: int,
) -> tuple[list[dict[str, Any]], bool]:
    lines: list[dict[str, Any]] = []
    retained_words = 0
    truncated = False
    for line_index in indexes:
        line = source_lines[line_index]
        words: list[dict[str, Any]] = []
        for word in list(line.words):
            if retained_words >= MAX_OCR_ENGINE_WORDS_PER_WINDOW:
                truncated = True
                break
            bounds = _rect(word.bounding_rect, image_width=image_width, image_height=image_height)
            words.append({"text": str(word.text)[:MAX_OCR_LINE_CHARS], "bounds": bounds})
            retained_words += 1
        text = str(line.text)
        if len(text) > MAX_OCR_LINE_CHARS:
            text = text[:MAX_OCR_LINE_CHARS]
            truncated = True
        lines.append(
            {
                "index": line_index,
                "text": text,
                "bounds": _union_bounds([word["bounds"] for word in words]),
                "words": words,
            }
        )
    return lines, truncated


def _bounded_result(
    engine_result: dict[str, Any],
    *,
    last_lines: int | None,
    max_lines: int,
    max_chars: int,
) -> dict[str, Any]:
    raw_lines = list(
        (engine_result.get("tail_lines") if last_lines is not None else None)
        or engine_result.get("lines")
        or []
    )
    original_text = "\n".join(str(line.get("text") or "") for line in raw_lines)
    selected = raw_lines[-last_lines:] if last_lines is not None else raw_lines
    selected_before_max_lines = len(selected)
    selected = selected[:max_lines]
    output: list[_BoundedLine] = []
    remaining = max_chars
    for line in selected:
        separator = 1 if output else 0
        if remaining <= separator:
            break
        remaining -= separator
        text = str(line.get("text") or "")
        if len(text) <= remaining:
            output.append(
                _BoundedLine(
                    text=text,
                    words=list(line.get("words") or []),
                    bounds=line.get("bounds"),
                )
            )
            remaining -= len(text)
            continue
        output.append(
            _BoundedLine(
                text=text[:remaining],
                words=[],
                bounds=line.get("bounds"),
                boxes_omitted_due_to_text_truncation=True,
            )
        )
        remaining = 0
        break
    text = "\n".join(line.text for line in output)
    original_line_count = int(engine_result.get("original_line_count", len(raw_lines)))
    original_word_count = int(engine_result.get("original_word_count", 0))
    returned_word_count = sum(len(line.words) for line in output)
    last_lines_applied = last_lines is not None and original_line_count > last_lines
    return {
        "text": text,
        "lines": output,
        "counts": {
            "original_lines": original_line_count,
            "recognized_lines_retained_by_engine": int(
                engine_result.get("retained_line_count", len(raw_lines))
            ),
            "returned_lines": len(output),
            "original_characters": int(
                engine_result.get("original_character_count", len(original_text))
            ),
            "returned_characters": len(text),
            "original_words": original_word_count,
            "returned_words_with_boxes": returned_word_count,
        },
        "limits": {
            "last_lines": last_lines,
            "max_lines": max_lines,
            "max_chars": max_chars,
        },
        "truncation": {
            "applied": (
                bool(engine_result.get("engine_output_truncated"))
                or last_lines_applied
                or len(selected) < len(raw_lines)
                or len(output) < len(selected)
                or len(text) < len("\n".join(str(line.get("text") or "") for line in selected))
            ),
            "engine_output_truncated": bool(engine_result.get("engine_output_truncated")),
            "last_lines_applied": last_lines_applied,
            "max_lines_applied": selected_before_max_lines > max_lines,
            "max_chars_applied": len(text)
            < len("\n".join(str(line.get("text") or "") for line in selected)),
        },
    }


def _uncertainty(
    engine_result: dict[str, Any],
    bounded: dict[str, Any],
) -> dict[str, Any]:
    lines = bounded["lines"]
    mixed_script_lines = [
        index
        for index, line in enumerate(lines)
        if _CYRILLIC_RE.search(line.text) and _LATIN_RE.search(line.text)
    ][:32]
    edge_lines = [
        index
        for index, line in enumerate(lines)
        if line.bounds is not None
        and _touches_image_edge(line.bounds, engine_result.get("source_pixel_size") or {})
    ][:32]
    flags = ["confidence_unknown", "layout_may_be_flattened", "character_substitutions_possible"]
    if not bounded["text"]:
        flags.append("no_text_detected")
    if mixed_script_lines:
        flags.append("mixed_script_recognition")
    if edge_lines:
        flags.append("edge_clipping_possible")
    if bounded["truncation"]["applied"]:
        flags.append("output_truncated")
    return {
        "flags": flags,
        "confidence_available": False,
        "numeric_confidence": None,
        "layout_fidelity": "line_order_and_word_boxes_only",
        "character_substitutions_possible": True,
        "mixed_script_line_indexes": mixed_script_lines,
        "edge_clipped_line_indexes": edge_lines,
        "language_coverage_limited": len(engine_result.get("available_languages") or []) < 2,
    }


def _source_provenance(source: NativeCaptureSource) -> dict[str, Any]:
    record = source.record
    presentation = _required_dict(record, "presentation")
    return {
        "capture_id": source.capture_id,
        "path": presentation.get("path"),
        "sha256": presentation.get("sha256"),
        "bytes": presentation.get("bytes"),
        "profile": presentation.get("profile"),
        "native_source_verified": presentation.get("profile") == "native",
        "captured_at": _timestamp(int(record["created_unix_ms"])),
        "expires_at": _timestamp(int(record["expires_unix_ms"])),
        "verified_at": _timestamp(source.validated_unix_ms),
        "capture_time_window_identity": record.get("window_identity"),
        "geometry": record.get("geometry"),
        "region": record.get("requested_native_crop"),
        "screen_region": record.get("requested_native_crop_screen_bounds"),
        "native_pixel_size": record.get("native_pixel_size"),
        "transform": record.get("transform"),
        "backend": record.get("backend"),
        "backend_details": record.get("backend_details"),
        "capture_attempts": record.get("capture_attempts"),
        "target_transaction": record.get("target_transaction"),
        "quality": record.get("quality"),
    }


def _validate_output_limits(*, last_lines: int | None, max_lines: int, max_chars: int) -> None:
    if last_lines is not None and (
        type(last_lines) is not int or not 1 <= last_lines <= MAX_OCR_LINES
    ):
        raise ComputerError(
            "invalid_ocr_limit",
            f"last-lines must be between 1 and {MAX_OCR_LINES}.",
            exit_code=2,
        )
    if type(max_lines) is not int or not 1 <= max_lines <= MAX_OCR_LINES:
        raise ComputerError(
            "invalid_ocr_limit",
            f"max-lines must be between 1 and {MAX_OCR_LINES}.",
            exit_code=2,
        )
    if type(max_chars) is not int or not 1 <= max_chars <= MAX_OCR_CHARS:
        raise ComputerError(
            "invalid_ocr_limit",
            f"max-chars must be between 1 and {MAX_OCR_CHARS}.",
            exit_code=2,
        )


def _required_size(record: dict[str, Any], field: str) -> tuple[int, int]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ComputerError("capture_record_invalid", "The capture provenance record is invalid.")
    width = value.get("width")
    height = value.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ComputerError("capture_record_invalid", "The capture provenance record is invalid.")
    return width, height


def _required_dict(record: dict[str, Any], field: str) -> dict[str, Any]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ComputerError("capture_record_invalid", "The capture provenance record is invalid.")
    return value


def _rect(
    rect: Any,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    values = {
        "x": float(rect.x),
        "y": float(rect.y),
        "width": float(rect.width),
        "height": float(rect.height),
    }
    if (
        any(not math.isfinite(value) for value in values.values())
        or values["x"] < 0
        or values["y"] < 0
        or values["width"] < 0
        or values["height"] < 0
        or values["x"] + values["width"] > image_width + 1
        or values["y"] + values["height"] > image_height + 1
    ):
        raise ComputerError("ocr_invalid_response", "Windows OCR returned invalid word bounds.")
    return {key: round(value, 3) for key, value in values.items()}


def _union_bounds(rectangles: list[dict[str, float]]) -> dict[str, float] | None:
    if not rectangles:
        return None
    left = min(rect["x"] for rect in rectangles)
    top = min(rect["y"] for rect in rectangles)
    right = max(rect["x"] + rect["width"] for rect in rectangles)
    bottom = max(rect["y"] + rect["height"] for rect in rectangles)
    return {
        "x": round(left, 3),
        "y": round(top, 3),
        "width": round(right - left, 3),
        "height": round(bottom - top, 3),
    }


def _touches_image_edge(bounds: dict[str, float], size: dict[str, Any]) -> bool:
    width = size.get("width")
    height = size.get("height")
    if type(width) is not int or type(height) is not int:
        return False
    return (
        bounds["x"] <= 1
        or bounds["y"] <= 1
        or bounds["x"] + bounds["width"] >= width - 1
        or bounds["y"] + bounds["height"] >= height - 1
    )


def _capture_attempt_duration(record: dict[str, Any]) -> int:
    total = 0
    for attempt in record.get("capture_attempts") or []:
        if isinstance(attempt, dict) and type(attempt.get("duration_ms")) is int:
            total += max(0, int(attempt["duration_ms"]))
    return total


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.perf_counter() - started) * 1000)))


def _timestamp(unix_ms: int) -> str:
    return (
        datetime.fromtimestamp(unix_ms / 1000, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
