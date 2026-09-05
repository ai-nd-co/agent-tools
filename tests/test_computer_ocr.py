from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont

from agent_tools import cli as root_cli
from agent_tools.computer import cli as computer_cli
from agent_tools.computer import ocr, screenshot
from agent_tools.computer.capture_records import CaptureRecordStore
from agent_tools.computer.models import (
    Bounds,
    ComputerError,
    NativeCaptureGeometry,
    Region,
    WindowIdentity,
)
from agent_tools.computer.rendering import render_ocr


class ManualClock:
    def __init__(self, now_ms: int) -> None:
        self.now_ms = now_ms

    def now_unix_ms(self) -> int:
        return self.now_ms


class FakeCaptureBackend:
    def __init__(self, *, width: int = 80, height: int = 60, dpi: int = 144) -> None:
        self.identity = WindowIdentity(
            hwnd=101,
            pid=202,
            thread_id=303,
            process="fixture.exe",
            title="Synthetic terminal",
            class_name="FixtureTerminal",
            bounds=Bounds(x=10, y=20, width=width, height=height),
            process_started=404,
            session_id=1,
            desktop_name="Default",
        )
        self.geometry = NativeCaptureGeometry(
            identity=self.identity,
            client_bounds=Bounds(x=18, y=50, width=max(1, width - 16), height=max(1, height - 38)),
            monitor_bounds=Bounds(x=0, y=0, width=2560, height=1440),
            virtual_screen_bounds=Bounds(x=0, y=0, width=2560, height=1440),
            window_dpi=dpi,
            system_dpi=96,
            window_awareness="per_monitor_aware_v2",
            capture_thread_awareness="per_monitor_aware_v2",
        )

    def capture_identity(self, _hwnd: int) -> WindowIdentity:
        return self.identity

    def capture_native_geometry(self, _hwnd: int) -> NativeCaptureGeometry:
        return self.geometry

    def is_minimized(self, _hwnd: int) -> bool:
        return False


class FakeOcrBackend:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = lines if lines is not None else ["first line", "second line", "third line"]
        self.calls: list[tuple[int, int, str]] = []

    def recognize(self, png_bytes: bytes, *, width: int, height: int) -> dict[str, object]:
        self.calls.append((width, height, hashlib.sha256(png_bytes).hexdigest()))
        rows = []
        for index, text in enumerate(self.lines):
            bounds = {"x": 5.0, "y": float(10 + index * 20), "width": 100.0, "height": 16.0}
            rows.append(
                {
                    "index": index,
                    "text": text,
                    "bounds": bounds,
                    "words": [{"text": text, "bounds": bounds}],
                }
            )
        return {
            "engine": "fixture_ocr",
            "binding_version": "1.0",
            "language_used": "ru",
            "available_languages": ["en-US", "ru"],
            "max_image_dimension": 10_000,
            "lines": rows,
            "original_line_count": len(rows),
            "original_word_count": len(rows),
            "original_character_count": sum(len(text) for text in self.lines)
            + max(0, len(self.lines) - 1),
            "engine_output_truncated": False,
        }


def _captured_image(width: int, height: int, captured_ms: int) -> screenshot.CapturedImage:
    image = Image.new("RGB", (width, height), "#101010")
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, max(3, width // 2), max(3, height // 2)), fill="#f0f0f0")
    return screenshot.CapturedImage(
        image=image,
        width=width,
        height=height,
        captured_unix_ms=captured_ms,
        backend="fixture_capture",
        backend_details={"fixture": True},
        attempts=(
            {
                "backend": "fixture_capture",
                "outcome": "accepted",
                "duration_ms": 17,
            },
        ),
        cleanup_verified=True,
    )


def _capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    profile: str = "native",
    width: int = 80,
    height: int = 60,
    region: Region | None = None,
    store: CaptureRecordStore | None = None,
) -> tuple[dict[str, object], CaptureRecordStore]:
    now_ms = store.now_unix_ms() if store is not None else int(time.time() * 1000)
    backend = FakeCaptureBackend(width=width, height=height)
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _captured_image(width, height, now_ms),
    )
    resolved_store = store or CaptureRecordStore(tmp_path / f"records-{profile}")
    result = screenshot.capture_screenshot(
        hwnd=backend.identity.hwnd,
        output=tmp_path / f"capture-{profile}.png",
        region=region,
        profile=profile,
        backend=backend,
        record_store=resolved_store,
    )
    return result, resolved_store


def test_ocr_preserves_native_capture_provenance_and_region(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, store = _capture(
        monkeypatch,
        tmp_path,
        region=Region(x=10, y=12, width=50, height=30),
    )
    image_path = Path(str(result["output"]))
    before_bytes = image_path.read_bytes()
    before_stat = image_path.stat()
    backend = FakeOcrBackend(["C:\\work\\app> pytest -q", "Русский вывод: готово!"])

    recognized = ocr.ocr_capture(
        capture_id=str(result["capture_id"]),
        record_store=store,
        backend=backend,
    )

    source = recognized["source_capture"]
    assert source["capture_id"] == result["capture_id"]
    assert source["profile"] == "native"
    assert source["path"] == str(image_path.resolve())
    assert source["sha256"] == hashlib.sha256(before_bytes).hexdigest()
    assert source["region"] == {"x": 10, "y": 12, "width": 50, "height": 30}
    assert source["native_pixel_size"] == {"width": 50, "height": 30}
    assert source["geometry"]["dpi"]["window"] == 144
    assert source["capture_time_window_identity"]["hwnd"] == 101
    assert source["backend"] == "fixture_capture"
    assert source["capture_attempts"][0]["duration_ms"] == 0
    assert recognized["observation_only"] is True
    assert recognized["authorizes_action"] is False
    assert recognized["engine"]["confidence_available"] is False
    assert recognized["uncertainty"]["numeric_confidence"] is None
    assert "layout_may_be_flattened" in recognized["uncertainty"]["flags"]
    assert backend.calls[0][:2] == (50, 30)
    assert image_path.read_bytes() == before_bytes
    assert image_path.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_native_2576_capture_is_eligible_and_720_profile_is_rejected_before_ocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native, native_store = _capture(
        monkeypatch,
        tmp_path,
        profile="native",
        width=2576,
        height=1426,
    )
    standard, standard_store = _capture(
        monkeypatch,
        tmp_path,
        profile="standard",
        width=2576,
        height=1426,
    )
    assert native["width"] == 2576
    assert native["height"] == 1426
    assert standard["width"] == 720
    assert standard["height"] == 399
    backend = FakeOcrBackend(["native terminal text"])

    recognized = ocr.ocr_capture(
        capture_id=str(native["capture_id"]),
        record_store=native_store,
        backend=backend,
    )
    assert recognized["text"] == "native terminal text"
    assert len(backend.calls) == 1

    with pytest.raises(ComputerError) as raised:
        ocr.ocr_capture(
            capture_id=str(standard["capture_id"]),
            record_store=standard_store,
            backend=backend,
        )
    assert raised.value.code == "ocr_capture_not_native"
    assert len(backend.calls) == 1


@pytest.mark.parametrize("mutation", ["missing", "tampered", "resized"])
def test_ocr_rejects_missing_tampered_or_resized_native_source_before_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    result, store = _capture(monkeypatch, tmp_path)
    path = Path(str(result["output"]))
    if mutation == "missing":
        path.unlink()
    elif mutation == "tampered":
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        Image.new("RGB", (40, 30), "white").save(path, format="PNG")
    backend = FakeOcrBackend()

    with pytest.raises(ComputerError) as raised:
        ocr.ocr_capture(
            capture_id=str(result["capture_id"]),
            record_store=store,
            backend=backend,
        )

    assert raised.value.code in {
        "capture_image_missing",
        "capture_image_altered",
        "capture_image_dimensions_changed",
    }
    assert backend.calls == []


def test_ocr_rejects_expired_capture_before_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = ManualClock(int(time.time() * 1000))
    store = CaptureRecordStore(tmp_path / "records-expiry", clock=clock)
    result, store = _capture(monkeypatch, tmp_path, store=store)
    clock.now_ms += 60_001
    backend = FakeOcrBackend()

    with pytest.raises(ComputerError) as raised:
        ocr.ocr_capture(
            capture_id=str(result["capture_id"]),
            record_store=store,
            backend=backend,
        )

    assert raised.value.code == "capture_expired"
    assert backend.calls == []


def test_ocr_accepts_capture_time_identity_without_live_window_revalidation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, store = _capture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        screenshot,
        "Win32Backend",
        lambda: (_ for _ in ()).throw(AssertionError("live backend must not be created")),
    )

    recognized = ocr.ocr_capture(
        capture_id=str(result["capture_id"]),
        record_store=store,
        backend=FakeOcrBackend(["title may have changed later"]),
    )

    assert recognized["text"] == "title may have changed later"


def test_last_lines_max_lines_and_max_chars_are_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, store = _capture(monkeypatch, tmp_path)
    backend = FakeOcrBackend(["alpha", "bravo", "charlie", "delta"])

    recognized = ocr.ocr_capture(
        capture_id=str(result["capture_id"]),
        last_lines=3,
        max_lines=2,
        max_chars=9,
        record_store=store,
        backend=backend,
    )

    assert recognized["text"] == "bravo\ncha"
    assert recognized["counts"]["original_lines"] == 4
    assert recognized["counts"]["returned_lines"] == 2
    assert recognized["counts"]["returned_characters"] == 9
    assert recognized["truncation"] == {
        "applied": True,
        "engine_output_truncated": False,
        "last_lines_applied": True,
        "max_lines_applied": True,
        "max_chars_applied": True,
    }
    assert recognized["lines"][1]["boxes_omitted_due_to_text_truncation"] is True
    assert recognized["lines"][1]["words"] == []


def test_last_lines_uses_actual_engine_tail_beyond_retained_head_cap() -> None:
    source_lines = [SimpleNamespace(text=f"line-{index:04d}", words=[]) for index in range(1_200)]
    normalized = ocr._normalize_engine_result(
        SimpleNamespace(lines=source_lines),
        engine_name="fixture_ocr",
        language_used="en-US",
        available_languages=["en-US"],
        max_image_dimension=10_000,
        image_width=1600,
        image_height=1400,
    )

    bounded = ocr._bounded_result(
        normalized,
        last_lines=3,
        max_lines=ocr.DEFAULT_MAX_OCR_LINES,
        max_chars=ocr.DEFAULT_MAX_OCR_CHARS,
    )

    assert bounded["text"] == "line-1197\nline-1198\nline-1199"
    assert bounded["counts"]["original_lines"] == 1_200
    assert bounded["counts"]["recognized_lines_retained_by_engine"] == 1_000
    assert bounded["truncation"]["engine_output_truncated"] is True


def test_last_lines_reports_omitted_lines_when_head_and_tail_windows_overlap() -> None:
    source_lines = [SimpleNamespace(text=f"line-{index:04d}", words=[]) for index in range(600)]
    normalized = ocr._normalize_engine_result(
        SimpleNamespace(lines=source_lines),
        engine_name="fixture_ocr",
        language_used="en-US",
        available_languages=["en-US"],
        max_image_dimension=10_000,
        image_width=1600,
        image_height=1400,
    )

    bounded = ocr._bounded_result(
        normalized,
        last_lines=500,
        max_lines=ocr.MAX_OCR_LINES,
        max_chars=ocr.MAX_OCR_CHARS,
    )

    assert bounded["text"].splitlines()[0] == "line-0100"
    assert bounded["text"].splitlines()[-1] == "line-0599"
    assert bounded["counts"]["recognized_lines_retained_by_engine"] == 600
    assert bounded["truncation"] == {
        "applied": True,
        "engine_output_truncated": False,
        "last_lines_applied": True,
        "max_lines_applied": False,
        "max_chars_applied": False,
    }


def test_empty_ocr_is_truthful_success_with_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, store = _capture(monkeypatch, tmp_path)

    recognized = ocr.ocr_capture(
        capture_id=str(result["capture_id"]),
        record_store=store,
        backend=FakeOcrBackend([]),
    )

    assert recognized["text"] == ""
    assert recognized["counts"]["returned_lines"] == 0
    assert "no_text_detected" in recognized["uncertainty"]["flags"]


def test_ocr_capabilities_reports_unsupported_platform_with_one_action(monkeypatch) -> None:
    monkeypatch.setattr(ocr.sys, "platform", "linux")
    monkeypatch.setattr(ocr, "_runtime_versions", lambda: ("0.7.1", "0.7.1"))

    result = ocr.ocr_capabilities()

    assert result["available"] is False
    assert result["reason_code"] == "platform_unsupported"
    assert isinstance(result["next_action"], str)


def test_ocr_capabilities_reports_source_distribution_mismatch_first(monkeypatch) -> None:
    monkeypatch.setattr(ocr.sys, "platform", "win32")
    monkeypatch.setattr(ocr, "_runtime_versions", lambda: ("0.7.1", "0.4.0"))
    monkeypatch.setattr(
        ocr,
        "_probe_windows_ocr",
        lambda: (_ for _ in ()).throw(AssertionError("binding probe must not run")),
    )

    result = ocr.ocr_capabilities()

    assert result["reason_code"] == "source_distribution_version_mismatch"
    assert result["source_distribution_version_match"] is False
    assert result["source_version"] == "0.7.1"
    assert result["distribution_version"] == "0.4.0"
    assert result["next_action"] == "Reinstall this AgentTools source with the computer extra."


def test_ocr_capabilities_reports_missing_binding(monkeypatch) -> None:
    monkeypatch.setattr(ocr.sys, "platform", "win32")
    monkeypatch.setattr(ocr, "_runtime_versions", lambda: ("0.7.1", "0.7.1"))
    monkeypatch.setattr(
        ocr,
        "_probe_windows_ocr",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError("fixture")),
    )

    result = ocr.ocr_capabilities()

    assert result["reason_code"] == "binding_not_installed"
    assert result["binding_version"] is None
    assert result["available_languages"] == []
    assert isinstance(result["next_action"], str)


def test_ocr_capabilities_reports_missing_windows_ocr_languages(monkeypatch) -> None:
    monkeypatch.setattr(ocr.sys, "platform", "win32")
    monkeypatch.setattr(ocr, "_runtime_versions", lambda: ("0.7.1", "0.7.1"))
    monkeypatch.setattr(ocr, "_probe_windows_ocr", lambda: ("3.2.1", []))

    result = ocr.ocr_capabilities()

    assert result["reason_code"] == "no_windows_ocr_languages"
    assert result["binding_version"] == "3.2.1"
    assert isinstance(result["next_action"], str)


def test_ocr_capabilities_reports_bounded_probe_failure(monkeypatch) -> None:
    monkeypatch.setattr(ocr.sys, "platform", "win32")
    monkeypatch.setattr(ocr, "_runtime_versions", lambda: ("0.7.1", "0.7.1"))
    monkeypatch.setattr(
        ocr,
        "_probe_windows_ocr",
        lambda: (_ for _ in ()).throw(RuntimeError("private fixture detail")),
    )

    result = ocr.ocr_capabilities()

    assert result["reason_code"] == "probe_failed"
    assert result["failure_type"] == "RuntimeError"
    assert "private fixture detail" not in str(result)
    assert isinstance(result["next_action"], str)


def test_ocr_capabilities_reports_ready_runtime(monkeypatch) -> None:
    monkeypatch.setattr(ocr.sys, "platform", "win32")
    monkeypatch.setattr(ocr, "_runtime_versions", lambda: ("0.7.1", "0.7.1"))
    monkeypatch.setattr(
        ocr,
        "_probe_windows_ocr",
        lambda: ("3.2.1", ["en-US", "ru-RU"]),
    )

    result = ocr.ocr_capabilities()

    assert result["available"] is True
    assert result["status"] == "ready"
    assert result["reason_code"] is None
    assert result["next_action"] is None


def test_ocr_probe_loads_full_runtime_and_creates_production_engine(monkeypatch) -> None:
    selected: list[tuple[object, object, list[str]]] = []
    engine = SimpleNamespace(recognizer_language=SimpleNamespace(language_tag="ru-RU"))
    ocr_engine = SimpleNamespace(
        available_recognizer_languages=[SimpleNamespace(language_tag="ru-RU")]
    )
    runtime = (ocr_engine, object(), object(), object(), object())
    monkeypatch.setattr(ocr.importlib.metadata, "version", lambda _name: "3.2.1")
    monkeypatch.setattr(ocr, "_load_windows_ocr_runtime", lambda: runtime)
    monkeypatch.setattr(
        ocr,
        "_select_engine",
        lambda candidate, language, languages: (
            selected.append((candidate, language, languages)) or engine
        ),
    )

    binding, languages = ocr._probe_windows_ocr()

    assert binding == "3.2.1"
    assert languages == ["ru-RU"]
    assert selected == [(ocr_engine, runtime[1], ["ru-RU"])]


def test_cli_ocr_does_not_construct_win32_or_capture_backend(monkeypatch, capsys) -> None:
    sample = {
        "text": "bounded terminal tail",
        "source_capture": {"capture_id": "cap_" + "a" * 32},
        "uncertainty": {"flags": ["confidence_unknown"]},
    }
    monkeypatch.setattr(computer_cli, "ocr_capture", lambda **_kwargs: sample)
    monkeypatch.setattr(
        computer_cli,
        "Win32Backend",
        lambda: (_ for _ in ()).throw(AssertionError("Win32 backend must not be constructed")),
    )
    monkeypatch.setattr(
        computer_cli,
        "capture_screenshot",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("capture must not run")),
    )

    assert root_cli.main(["computer", "ocr", "--from-capture", "cap_" + "a" * 32, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "computer.ocr"
    assert payload["data"]["ocr"]["text"] == "bounded terminal tail"


def test_human_ocr_output_preserves_lines_and_escapes_embedded_controls() -> None:
    rendered = render_ocr(
        {
            "ocr": {
                "text": "first line\nsecond\tline",
                "source_capture": {"capture_id": "cap_" + "a" * 32},
                "uncertainty": {"flags": ["confidence_unknown"]},
            }
        }
    )

    assert rendered.startswith("first line\nsecond\\x09line\n")
    assert "\\x0a" not in rendered


def test_cli_ocr_has_no_image_path_or_internal_grep_escape(capsys) -> None:
    for option in ("--input", "--path", "--hwnd", "--grep"):
        assert (
            root_cli.main(
                ["computer", "ocr", "--from-capture", "cap_" + "a" * 32, option, "value", "--json"]
            )
            == 2
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["error"]["code"] == "invalid_arguments"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows OCR integration")
def test_real_windows_ocr_reads_synthetic_english_russian_paths_and_punctuation() -> None:
    pytest.importorskip("winrt.windows.media.ocr")
    image = Image.new("RGB", (1600, 900), "#0c0c0c")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 34)
    expected_lines = [
        r"C:\projects\fixture> uv run pytest -q",
        "128 passed in 4.21s",
        "Русский терминал: проверка пути и пунктуации!",
        "prompt> fix --target src/agent_tools/computer/ocr.py",
    ]
    for index, line in enumerate(expected_lines):
        draw.text((70, 90 + index * 105), line, fill="#f2f2f2", font=font)
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    backend = ocr.WindowsOcrBackend()

    started = time.perf_counter()
    recognized = backend.recognize(payload.getvalue(), width=1600, height=900)
    elapsed = time.perf_counter() - started
    text = "\n".join(line["text"] for line in recognized["lines"])

    assert "pytest" in text
    assert "128 passed" in text
    assert "Русский терминал" in text
    assert "src/agent_tools/computer/ocr.py" in text
    assert recognized["language_used"].casefold().startswith("ru")
    assert all(line["bounds"] is not None for line in recognized["lines"])
    assert elapsed < 3.0
