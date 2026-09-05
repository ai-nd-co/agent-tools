from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from agent_tools.computer import capture_records, process_io, screenshot
from agent_tools.computer.capture_records import (
    CAPTURE_RECORD_SCHEMA,
    MAX_CAPTURE_RECORDS,
    CaptureRecordStore,
)
from agent_tools.computer.models import (
    Bounds,
    ComputerError,
    NativeCaptureGeometry,
    Region,
    WindowIdentity,
)
from agent_tools.computer.rendering import render_json, render_windows


class FakeBackend:
    def __init__(
        self,
        *,
        minimized: bool = False,
        width: int = 80,
        height: int = 60,
    ) -> None:
        self.minimized = minimized
        self.identity = WindowIdentity(
            hwnd=101,
            pid=202,
            thread_id=303,
            process="fixture.exe",
            title="Disposable fixture",
            class_name="Fixture",
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
            window_dpi=144,
            system_dpi=96,
            window_awareness="per_monitor_aware_v2",
            capture_thread_awareness="per_monitor_aware_v2",
        )

    def capture_identity(self, _hwnd: int) -> WindowIdentity:
        return self.identity

    def capture_native_geometry(self, _hwnd: int) -> NativeCaptureGeometry:
        return self.geometry

    def is_minimized(self, _hwnd: int) -> bool:
        return self.minimized


class ManualClock:
    def __init__(self, now_ms: int) -> None:
        self.now_ms = now_ms

    def now_unix_ms(self) -> int:
        return self.now_ms


def _non_uniform_capture(
    width: int = 80,
    height: int = 60,
    *,
    captured_unix_ms: int | None = None,
) -> screenshot.CapturedImage:
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).rectangle(
        (0, 0, max(1, width // 4), max(1, height // 4)),
        fill="black",
    )
    return screenshot.CapturedImage(
        image=image,
        width=width,
        height=height,
        captured_unix_ms=(
            int(time.time() * 1000) if captured_unix_ms is None else captured_unix_ms
        ),
        backend="fixture",
        cleanup_verified=True,
    )


def test_public_identity_is_bounded_without_weakening_internal_identity() -> None:
    identity = FakeBackend().identity
    long_identity = WindowIdentity(
        hwnd=identity.hwnd,
        pid=identity.pid,
        thread_id=identity.thread_id,
        process="p" * 300,
        title="t" * 500,
        class_name="c" * 300,
        bounds=identity.bounds,
        process_started=identity.process_started,
    )

    public = long_identity.public()

    assert len(long_identity.title) == 500
    assert len(public["title"]) == 240
    assert len(public["class"]) == 128
    assert len(public["process"]) == 128


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("1,2,3", "invalid_region"),
        ("1,2,nope,4", "invalid_region"),
        ("-1,2,3,4", "invalid_region"),
        ("1,2,0,4", "invalid_region"),
    ],
)
def test_parse_region_rejects_invalid_values(value: str, code: str) -> None:
    with pytest.raises(ComputerError, match="Region") as raised:
        screenshot.parse_region(value)
    assert raised.value.code == code


def test_resolver_allows_title_change_but_keeps_geometry_fail_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(),
    )
    store = CaptureRecordStore(tmp_path / "records-title")
    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / "title-coordinate.png",
        region=None,
        backend=backend,
        record_store=store,
    )
    backend.geometry = replace(
        backend.geometry,
        identity=replace(backend.geometry.identity, title="New mutable title"),
    )

    mapped = screenshot.resolve_capture_point(
        capture_id=result["capture_id"],
        image_x=10,
        image_y=10,
        backend=backend,
        record_store=store,
    )

    assert mapped["coordinate_authority"] == "fresh_capture_bound_native"


def test_region_capture_is_validated_and_written_atomically(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(),
    )
    output = tmp_path / "region.png"

    result = screenshot.capture_screenshot(
        hwnd=101,
        output=output,
        region=Region(x=0, y=0, width=40, height=30),
        backend=FakeBackend(),
        record_store=CaptureRecordStore(tmp_path / "records"),
    )

    assert result["width"] == 40
    assert result["height"] == 30
    assert result["content_non_uniform"] is True
    assert result["cleanup_verified"] is True
    assert len(result["sha256"]) == 64
    with Image.open(output) as image:
        assert image.size == (40, 30)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("native", "profile", "expected"),
    [
        ((1920, 1080), "standard", (720, 405)),
        ((1920, 1080), "compact", (480, 270)),
        ((1920, 1080), "native", (1920, 1080)),
        ((400, 300), "standard", (400, 300)),
        ((1001, 777), "standard", (720, 559)),
        ((777, 1001), "compact", (373, 480)),
    ],
)
def test_presentation_profiles_are_bounded_aspect_preserving_and_never_upscale(
    native: tuple[int, int],
    profile: str,
    expected: tuple[int, int],
) -> None:
    assert screenshot.presentation_size(*native, profile) == expected


def test_transform_maps_edges_exactly_and_round_trips_presentation_pixels() -> None:
    crop = Region(x=11, y=7, width=1001, height=777)
    presentation = screenshot.presentation_size(crop.width, crop.height, "standard")
    transform = screenshot.build_transform(crop, *presentation)

    assert screenshot.map_presentation_point(transform, 0, 0) == (11, 7)
    assert screenshot.map_presentation_point(
        transform, presentation[0] - 1, presentation[1] - 1
    ) == (1011, 783)
    for image_x, image_y in ((1, 1), (123, 321), (719, 558), (359, 279)):
        native_x, native_y = screenshot.map_presentation_point(
            transform, image_x, image_y
        )
        ideal_x = Fraction(image_x * (crop.width - 1), presentation[0] - 1) + crop.x
        ideal_y = Fraction(image_y * (crop.height - 1), presentation[1] - 1) + crop.y
        assert abs(Fraction(native_x) - ideal_x) <= Fraction(1, 2)
        assert abs(Fraction(native_y) - ideal_y) <= Fraction(1, 2)
        round_trip = screenshot.map_native_point(transform, native_x, native_y)
        assert abs(round_trip[0] - image_x) <= 1
        assert abs(round_trip[1] - image_y) <= 1


@pytest.mark.parametrize(
    ("native_extent", "expected_index"),
    [(2, 1), (3, 1), (4, 2), (5, 2)],
)
def test_singleton_presentation_axis_uses_native_midpoint_half_up(
    native_extent: int,
    expected_index: int,
) -> None:
    transform = screenshot.build_transform(Region(0, 0, native_extent, 1), 1, 1)

    assert transform["x"]["singleton_native_index"] == expected_index
    assert screenshot.map_presentation_point(transform, 0, 0) == (expected_index, 0)


def test_standard_capture_emits_bounded_metadata_and_resolves_native_point(
    monkeypatch, tmp_path: Path
) -> None:
    backend = FakeBackend(width=1920, height=1080)
    clock = ManualClock(1_800_000_000_000)
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(
            1920,
            1080,
            captured_unix_ms=clock.now_ms,
        ),
    )
    store = CaptureRecordStore(tmp_path / "records", clock=clock)
    output = tmp_path / "standard.png"

    result = screenshot.capture_screenshot(
        hwnd=101,
        output=output,
        region=None,
        backend=backend,
        record_store=store,
    )

    assert (result["width"], result["height"]) == (720, 405)
    assert result["native_pixel_size"] == {"width": 1920, "height": 1080}
    assert result["presentation"] == {
        "profile": "standard",
        "pixel_size": {"width": 720, "height": 405},
    }
    assert result["coordinate_authority"] == "capture_record_required"
    assert "title" not in result["window"]
    assert len(result["window"]["fingerprint"]) == 64
    assert result["geometry"]["coordinate_space"] == "physical_screen_pixels"
    assert result["geometry"]["client_bounds"] == backend.geometry.client_bounds.to_jsonable()
    assert result["geometry"]["dpi"]["window_scale"] == {
        "numerator": 144,
        "denominator": 96,
    }
    assert len(json.dumps(result, ensure_ascii=True)) < 8_000
    with Image.open(output) as image:
        assert image.size == (720, 405)

    clock.now_ms = 1_800_000_030_000
    mapped = screenshot.resolve_capture_point(
        capture_id=result["capture_id"],
        image_x=719,
        image_y=404,
        backend=backend,
        record_store=store,
    )
    assert mapped["native_window_point"] == {"x": 1919, "y": 1079}
    assert mapped["native_screen_point"] == {"x": 1929, "y": 1099}
    assert mapped["coordinate_authority"] == "fresh_capture_bound_native"
    assert mapped["action_revalidation_required"] is True


def test_title_change_is_non_authoritative_and_reported_with_safe_hashes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    before = backend.geometry
    after = replace(
        before,
        identity=replace(before.identity, title="Disposable fixture - frame 2"),
    )
    geometries = iter((before, after))
    monkeypatch.setattr(
        backend,
        "capture_native_geometry",
        lambda _hwnd: next(geometries),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(),
    )
    store = CaptureRecordStore(tmp_path / "records")

    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / "title-change.png",
        region=None,
        backend=backend,
        record_store=store,
    )

    assert result["strong_identity_verified"] is True
    assert result["identity_mismatch_fields"] == []
    assert result["title_changed_during_operation"] is True
    assert result["title_before_hmac_sha256"] != result["title_after_hmac_sha256"]
    assert "Disposable fixture" not in json.dumps(result)


def test_geometry_change_discards_frame_and_recaptures_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    before = backend.geometry
    moved_identity = replace(
        before.identity,
        bounds=replace(before.identity.bounds, x=before.identity.bounds.x + 25),
    )
    moved = replace(
        before,
        identity=moved_identity,
        client_bounds=replace(
            before.client_bounds,
            x=before.client_bounds.x + 25,
        ),
    )
    geometries = iter((before, moved, moved))
    monkeypatch.setattr(
        backend,
        "capture_native_geometry",
        lambda _hwnd: next(geometries),
    )
    captures = 0

    def capture(_backend, _identity):
        nonlocal captures
        captures += 1
        return _non_uniform_capture()

    monkeypatch.setattr(screenshot, "_capture_full", capture)

    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / "geometry-recapture.png",
        region=None,
        backend=backend,
        record_store=CaptureRecordStore(tmp_path / "geometry-records"),
    )

    assert captures == 2
    assert result["geometry_recapture_count"] == 1
    assert result["geometry"]["window_bounds"]["x"] == moved_identity.bounds.x
    assert result["requested_native_crop_screen_bounds"]["x"] == moved_identity.bounds.x


def test_second_geometry_change_fails_truthfully_without_writing_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    first = backend.geometry
    second = replace(
        first,
        identity=replace(
            first.identity,
            bounds=replace(first.identity.bounds, x=first.identity.bounds.x + 10),
        ),
    )
    third = replace(
        second,
        identity=replace(
            second.identity,
            bounds=replace(second.identity.bounds, x=second.identity.bounds.x + 10),
        ),
    )
    geometries = iter((first, second, third))
    monkeypatch.setattr(
        backend,
        "capture_native_geometry",
        lambda _hwnd: next(geometries),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(),
    )
    output = tmp_path / "unstable-geometry.png"

    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=output,
            region=None,
            backend=backend,
            record_store=CaptureRecordStore(tmp_path / "unstable-records"),
        )

    assert raised.value.code == "capture_geometry_changed"
    assert raised.value.details["geometry_mismatch_fields"] == ["window_bounds"]
    assert raised.value.details["geometry_recapture_count"] == 1
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "mismatch"),
    [
        ("pid", "pid"),
        ("process_started", "process_start_identity"),
        ("class_name", "window_class"),
        ("session_id", "session_id"),
        ("desktop_name", "input_desktop"),
    ],
)
def test_strong_identity_drift_reports_only_bounded_fields(
    field: str,
    mismatch: str,
) -> None:
    backend = FakeBackend()
    before = backend.identity
    changed_values = {
        "pid": before.pid + 1,
        "process_started": before.process_started + 1,
        "class_name": "ReplacementClass",
        "session_id": before.session_id + 1,
        "desktop_name": "ReplacementDesktop",
    }
    after = replace(before, **{field: changed_values[field]}, title="Same title")

    with pytest.raises(ComputerError) as raised:
        screenshot._assert_strong_screenshot_identity(backend, before, after)

    assert raised.value.code == "stale_window"
    assert raised.value.details == {
        "identity_mismatch_fields": [mismatch],
        "title_changed": True,
        "title_change_was_authoritative": False,
    }


def test_capture_lifetime_starts_when_pixels_are_captured(monkeypatch, tmp_path: Path) -> None:
    captured_ms = 1_800_000_000_000
    clock = ManualClock(captured_ms)
    backend = FakeBackend(width=1920, height=1080)
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(
            1920,
            1080,
            captured_unix_ms=captured_ms,
        ),
    )
    original_presentation_image = screenshot._presentation_image

    def expire_during_processing(image, width, height):
        result = original_presentation_image(image, width, height)
        clock.now_ms = captured_ms + 60_000
        return result

    monkeypatch.setattr(screenshot, "_presentation_image", expire_during_processing)
    store = CaptureRecordStore(tmp_path / "records", clock=clock)

    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=tmp_path / "expired-processing.png",
            region=None,
            backend=backend,
            record_store=store,
        )

    assert raised.value.code == "capture_expired"
    assert not (tmp_path / "expired-processing.png").exists()
    assert not list(store.root.glob("cap_*.json"))


def test_expiry_during_response_construction_cannot_publish_authority(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_ms = 1_800_000_000_000
    clock = ManualClock(captured_ms)
    backend = FakeBackend()
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(captured_unix_ms=captured_ms),
    )

    def expire_while_checking_content(_image):
        clock.now_ms = captured_ms + 60_000
        return True

    monkeypatch.setattr(screenshot, "_image_non_uniform", expire_while_checking_content)
    store = CaptureRecordStore(tmp_path / "records", clock=clock)
    output = tmp_path / "response-expired.png"

    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=output,
            region=None,
            backend=backend,
            record_store=store,
        )

    assert raised.value.code == "capture_expired"
    assert not output.exists()
    assert not list(store.root.glob("cap_*.json"))


def test_slow_output_write_cannot_publish_an_expired_record(monkeypatch, tmp_path: Path) -> None:
    captured_ms = 1_800_000_000_000
    clock = ManualClock(captured_ms)
    backend = FakeBackend()
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(captured_unix_ms=captured_ms),
    )
    original_atomic_write = screenshot._atomic_write
    output = tmp_path / "slow-output.png"

    def slow_atomic_write(path, payload):
        original_atomic_write(path, payload)
        clock.now_ms = captured_ms + 60_000

    monkeypatch.setattr(screenshot, "_atomic_write", slow_atomic_write)
    store = CaptureRecordStore(tmp_path / "records", clock=clock)

    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=output,
            region=None,
            backend=backend,
            record_store=store,
        )

    assert raised.value.code == "capture_expired"
    assert output.is_file()
    assert not list(store.root.glob("cap_*.json"))


def test_final_expiry_check_deletes_already_published_record(monkeypatch, tmp_path: Path) -> None:
    captured_ms = 1_800_000_000_000
    clock = ManualClock(captured_ms)
    backend = FakeBackend()
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(captured_unix_ms=captured_ms),
    )
    store = CaptureRecordStore(tmp_path / "records", clock=clock)
    original_save = store.save

    def save_then_expire(record):
        original_save(record)
        clock.now_ms = captured_ms + 60_000

    monkeypatch.setattr(store, "save", save_then_expire)

    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=tmp_path / "final-expired.png",
            region=None,
            backend=backend,
            record_store=store,
        )

    assert raised.value.code == "capture_expired"
    assert not list(store.root.glob("cap_*.json"))


def test_resolver_rechecks_expiry_after_image_and_live_geometry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_ms = 1_800_000_000_000
    clock = ManualClock(captured_ms)
    backend = FakeBackend()
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(captured_unix_ms=captured_ms),
    )
    store = CaptureRecordStore(tmp_path / "records", clock=clock)
    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / "resolver-expiry.png",
        region=None,
        backend=backend,
        record_store=store,
    )
    clock.now_ms = captured_ms + 59_999

    def geometry_at_expiry(_hwnd):
        clock.now_ms = captured_ms + 60_000
        return backend.geometry

    monkeypatch.setattr(backend, "capture_native_geometry", geometry_at_expiry)

    with pytest.raises(ComputerError) as raised:
        screenshot.resolve_capture_point(
            capture_id=result["capture_id"],
            image_x=0,
            image_y=0,
            backend=backend,
            record_store=store,
        )

    assert raised.value.code == "capture_expired"


@pytest.mark.parametrize(
    ("profile", "region", "expected"),
    [
        ("compact", Region(100, 50, 600, 200), (480, 160)),
        ("standard", Region(100, 50, 300, 200), (300, 200)),
        ("native", Region(100, 50, 600, 200), (600, 200)),
    ],
)
def test_crop_profiles_preserve_native_origin_and_do_not_upscale(
    monkeypatch,
    tmp_path: Path,
    profile: str,
    region: Region,
    expected: tuple[int, int],
) -> None:
    backend = FakeBackend(width=1000, height=800)
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(1000, 800),
    )
    store = CaptureRecordStore(tmp_path / f"records-{profile}")
    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / f"{profile}.png",
        region=region,
        profile=profile,
        backend=backend,
        record_store=store,
    )

    assert (result["width"], result["height"]) == expected
    assert result["requested_native_crop"] == region.to_jsonable()
    assert result["transform"]["x"]["native_origin"] == 100
    assert result["transform"]["y"]["native_origin"] == 50
    assert screenshot.map_presentation_point(result["transform"], 0, 0) == (100, 50)


def test_same_pixels_produce_stable_presentation_hashes(monkeypatch, tmp_path: Path) -> None:
    backend = FakeBackend(width=1200, height=900)
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(1200, 900),
    )
    hashes = []
    for index in range(2):
        result = screenshot.capture_screenshot(
            hwnd=101,
            output=tmp_path / f"stable-{index}.png",
            region=None,
            profile="standard",
            backend=backend,
            record_store=CaptureRecordStore(tmp_path / "records"),
        )
        hashes.append(result["sha256"])
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize(
    ("dpi", "profile", "expected"),
    [
        (96, "compact", (480, 320)),
        (144, "standard", (720, 480)),
        (192, "native", (1200, 800)),
    ],
)
def test_profiles_preserve_exact_dpi_metadata_across_settings(
    monkeypatch,
    tmp_path: Path,
    dpi: int,
    profile: str,
    expected: tuple[int, int],
) -> None:
    backend = FakeBackend(width=1200, height=800)
    backend.geometry = replace(backend.geometry, window_dpi=dpi)
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(1200, 800),
    )
    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / f"dpi-{dpi}.png",
        region=None,
        profile=profile,
        backend=backend,
        record_store=CaptureRecordStore(tmp_path / f"records-{dpi}"),
    )

    assert (result["width"], result["height"]) == expected
    assert result["geometry"]["dpi"]["window"] == dpi
    assert result["geometry"]["dpi"]["window_scale"] == {
        "numerator": dpi,
        "denominator": 96,
    }


def test_resolver_rejects_expired_missing_and_altered_records(
    monkeypatch, tmp_path: Path
) -> None:
    backend = FakeBackend(width=1200, height=900)
    clock = ManualClock(2_000_000_000_000)
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(
            1200,
            900,
            captured_unix_ms=clock.now_ms,
        ),
    )
    store = CaptureRecordStore(tmp_path / "records", clock=clock)
    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / "record.png",
        region=None,
        backend=backend,
        record_store=store,
    )
    capture_id = result["capture_id"]

    clock.now_ms = 2_000_000_060_000
    with pytest.raises(ComputerError) as expired:
        screenshot.resolve_capture_point(
            capture_id=capture_id,
            image_x=0,
            image_y=0,
            backend=backend,
            record_store=store,
        )
    assert expired.value.code == "capture_expired"

    missing_id = store.new_capture_id()
    with pytest.raises(ComputerError) as missing:
        screenshot.resolve_capture_point(
            capture_id=missing_id,
            image_x=0,
            image_y=0,
            backend=backend,
            record_store=store,
        )
    assert missing.value.code == "capture_record_missing"

    clock.now_ms = 2_000_000_061_000
    second = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / "altered-record.png",
        region=None,
        backend=backend,
        record_store=store,
    )
    record_path = store.root / f"{second['capture_id']}.json"
    envelope = json.loads(record_path.read_text(encoding="ascii"))
    envelope["payload"]["geometry"]["window_bounds"]["x"] += 1
    record_path.write_text(json.dumps(envelope), encoding="ascii")
    with pytest.raises(ComputerError) as altered:
        screenshot.resolve_capture_point(
            capture_id=second["capture_id"],
            image_x=0,
            image_y=0,
            backend=backend,
            record_store=store,
        )
    assert altered.value.code == "capture_record_altered"


def test_resolver_rejects_altered_image_and_mismatched_dimensions(
    monkeypatch, tmp_path: Path
) -> None:
    backend = FakeBackend(width=1200, height=900)
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(1200, 900),
    )
    store = CaptureRecordStore(tmp_path / "records")
    output = tmp_path / "image.png"
    result = screenshot.capture_screenshot(
        hwnd=101,
        output=output,
        region=None,
        backend=backend,
        record_store=store,
    )
    output.write_bytes(output.read_bytes() + b"altered")
    with pytest.raises(ComputerError) as altered:
        screenshot.resolve_capture_point(
            capture_id=result["capture_id"],
            image_x=0,
            image_y=0,
            backend=backend,
            record_store=store,
        )
    assert altered.value.code == "capture_image_altered"

    mismatched = tmp_path / "mismatched.png"
    Image.new("RGB", (10, 10), "red").save(mismatched)
    record = store.load(result["capture_id"])
    encoded = mismatched.read_bytes()
    record["presentation"].update(
        {
            "path": str(mismatched.resolve()),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    )
    store.save(record)
    with pytest.raises(ComputerError) as dimensions:
        screenshot.resolve_capture_point(
            capture_id=result["capture_id"],
            image_x=0,
            image_y=0,
            backend=backend,
            record_store=store,
        )
    assert dimensions.value.code == "capture_image_dimensions_changed"


def test_resolver_rejects_signed_but_cross_field_inconsistent_record(
    monkeypatch, tmp_path: Path
) -> None:
    backend = FakeBackend(width=1200, height=900)
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(1200, 900),
    )
    store = CaptureRecordStore(tmp_path / "records")
    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / "inconsistent.png",
        region=None,
        backend=backend,
        record_store=store,
    )
    record = store.load(result["capture_id"])
    record["transform"]["x"]["native_origin"] += 1
    store.save(record)

    with pytest.raises(ComputerError) as raised:
        screenshot.resolve_capture_point(
            capture_id=result["capture_id"],
            image_x=0,
            image_y=0,
            backend=backend,
            record_store=store,
        )
    assert raised.value.code == "capture_record_invalid"


@pytest.mark.parametrize(
    ("drift", "code"),
    [
        ("identity", "capture_target_changed"),
        ("window", "capture_geometry_changed"),
        ("client", "capture_geometry_changed"),
        ("screen", "capture_screen_geometry_changed"),
        ("dpi", "capture_dpi_changed"),
    ],
)
def test_resolver_rejects_identity_geometry_and_dpi_drift(
    monkeypatch, tmp_path: Path, drift: str, code: str
) -> None:
    backend = FakeBackend(width=1200, height=900)
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(1200, 900),
    )
    store = CaptureRecordStore(tmp_path / f"records-{drift}")
    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / f"{drift}.png",
        region=None,
        backend=backend,
        record_store=store,
    )
    geometry = backend.geometry
    if drift == "identity":
        identity = replace(
            geometry.identity,
            process_started=geometry.identity.process_started + 1,
        )
        backend.geometry = replace(geometry, identity=identity)
    elif drift == "window":
        identity = replace(
            geometry.identity,
            bounds=replace(geometry.identity.bounds, x=geometry.identity.bounds.x + 1),
        )
        backend.geometry = replace(geometry, identity=identity)
    elif drift == "client":
        backend.geometry = replace(
            geometry,
            client_bounds=replace(geometry.client_bounds, width=geometry.client_bounds.width - 1),
        )
    elif drift == "screen":
        backend.geometry = replace(
            geometry,
            virtual_screen_bounds=replace(
                geometry.virtual_screen_bounds,
                width=geometry.virtual_screen_bounds.width - 1,
            ),
        )
    else:
        backend.geometry = replace(geometry, window_dpi=120)

    with pytest.raises(ComputerError) as raised:
        screenshot.resolve_capture_point(
            capture_id=result["capture_id"],
            image_x=10,
            image_y=10,
            backend=backend,
            record_store=store,
        )
    assert raised.value.code == code


def test_capture_store_is_owner_only_and_bounded(monkeypatch, tmp_path: Path) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(),
    )
    store = CaptureRecordStore(tmp_path / "records")
    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / "acl.png",
        region=None,
        backend=backend,
        record_store=store,
    )

    import win32security

    for path in (
        store.root,
        store.root / ".integrity-key",
        store.root / f"{result['capture_id']}.json",
    ):
        descriptor = win32security.GetNamedSecurityInfo(
            str(path),
            win32security.SE_FILE_OBJECT,
            win32security.DACL_SECURITY_INFORMATION,
        )
        acl = descriptor.GetSecurityDescriptorDacl()
        assert acl.GetAceCount() >= 1
        owner_sids = {
            win32security.ConvertSidToStringSid(acl.GetAce(index)[2])
            for index in range(acl.GetAceCount())
        }
        assert len(owner_sids) == 1

    now_ms = int(time.time() * 1000)
    for _index in range(MAX_CAPTURE_RECORDS + 8):
        capture_id = store.new_capture_id()
        store.save(
            {
                "schema_version": CAPTURE_RECORD_SCHEMA,
                "capture_id": capture_id,
                "expires_unix_ms": now_ms + 60_000,
            }
        )
    assert len(list(store.root.glob("cap_*.json"))) <= MAX_CAPTURE_RECORDS


def test_capture_store_fails_closed_when_required_evictions_are_denied(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store = CaptureRecordStore(tmp_path / "records")
    for _index in range(MAX_CAPTURE_RECORDS):
        capture_id = store.new_capture_id()
        store.save({"schema_version": CAPTURE_RECORD_SCHEMA, "capture_id": capture_id})
    existing_records = set(store.root.glob("cap_*.json"))
    assert len(existing_records) == MAX_CAPTURE_RECORDS
    original_unlink = Path.unlink

    def deny_record_eviction(path, *args, **kwargs):
        if path in existing_records:
            raise PermissionError("fixture record is open without delete sharing")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_record_eviction)
    new_capture_id = store.new_capture_id()

    with pytest.raises(ComputerError) as raised:
        store.save(
            {"schema_version": CAPTURE_RECORD_SCHEMA, "capture_id": new_capture_id}
        )

    assert raised.value.code == "capture_store_capacity_unavailable"
    assert len(list(store.root.glob("cap_*.json"))) == MAX_CAPTURE_RECORDS
    assert not (store.root / f"{new_capture_id}.json").exists()


def test_capture_store_lock_is_cross_process_and_store_local(tmp_path: Path) -> None:
    root = tmp_path / "records"
    script = (
        "import sys,time\n"
        "from pathlib import Path\n"
        "from agent_tools.computer.capture_records import _capture_store_lock\n"
        "with _capture_store_lock(Path(sys.argv[1])):\n"
        " print('locked', flush=True)\n"
        " time.sleep(5)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(ComputerError) as raised:
            CaptureRecordStore(root).identity_reference(FakeBackend().identity)
        assert raised.value.code == "capture_store_busy"
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_integrity_key_creation_is_exclusive_without_store_lock(tmp_path: Path) -> None:
    root = tmp_path / "records"
    capture_records._prepare_capture_store_root(root)
    key_path = root / ".integrity-key"
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def create_key() -> None:
        try:
            barrier.wait(timeout=2)
            capture_records._create_integrity_key_exclusive(key_path)
        except Exception as exc:  # noqa: BLE001 - collect thread failures for assertion
            errors.append(exc)

    threads = [threading.Thread(target=create_key) for _index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    first_key = key_path.read_bytes()
    assert len(first_key) == capture_records.KEY_BYTES
    capture_records._create_integrity_key_exclusive(key_path)
    assert key_path.read_bytes() == first_key
    assert not list(root.glob("..integrity-key.*.tmp"))


def test_png_is_complete_before_signed_record_publication(monkeypatch, tmp_path: Path) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(),
    )
    store = CaptureRecordStore(tmp_path / "records")
    output = tmp_path / "published.png"
    original_save = store.save

    def save_after_image(record):
        assert output.is_file()
        encoded = output.read_bytes()
        assert len(encoded) == record["presentation"]["bytes"]
        assert hashlib.sha256(encoded).hexdigest() == record["presentation"]["sha256"]
        original_save(record)

    monkeypatch.setattr(store, "save", save_after_image)

    result = screenshot.capture_screenshot(
        hwnd=101,
        output=output,
        region=None,
        backend=backend,
        record_store=store,
    )

    assert (store.root / f"{result['capture_id']}.json").is_file()


def test_record_write_has_no_fallible_post_commit_acl_step(monkeypatch, tmp_path: Path) -> None:
    store = CaptureRecordStore(tmp_path / "records")
    capture_id = store.new_capture_id()
    record_path = store.root / f"{capture_id}.json"
    original_acl = capture_records._set_owner_only_acl

    def reject_final_record_acl(path, *, directory):
        if path == record_path:
            raise AssertionError("final record ACL touched after publication")
        original_acl(path, directory=directory)

    monkeypatch.setattr(capture_records, "_set_owner_only_acl", reject_final_record_acl)
    store.save({"schema_version": CAPTURE_RECORD_SCHEMA, "capture_id": capture_id})

    assert record_path.is_file()


def test_post_commit_lock_exit_failure_rolls_back_record(monkeypatch, tmp_path: Path) -> None:
    store = CaptureRecordStore(tmp_path / "records")
    capture_id = store.new_capture_id()
    record_path = store.root / f"{capture_id}.json"

    @contextmanager
    def fail_after_commit(_root):
        yield
        raise ComputerError("capture_store_busy", "fixture CloseHandle failure")

    monkeypatch.setattr(capture_records, "_capture_store_lock", fail_after_commit)

    with pytest.raises(ComputerError) as raised:
        store.save({"schema_version": CAPTURE_RECORD_SCHEMA, "capture_id": capture_id})

    assert raised.value.code == "capture_store_busy"
    assert not record_path.exists()


def test_capture_record_deletion_failures_are_visible(monkeypatch, tmp_path: Path) -> None:
    store = CaptureRecordStore(tmp_path / "records")
    capture_id = store.new_capture_id()
    record_path = store.root / f"{capture_id}.json"
    store.save({"schema_version": CAPTURE_RECORD_SCHEMA, "capture_id": capture_id})
    original_unlink = Path.unlink

    def fail_record_unlink(path, *args, **kwargs):
        if path == record_path:
            raise OSError("fixture deletion failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_record_unlink)

    with pytest.raises(ComputerError) as raised:
        store.delete(capture_id)

    assert raised.value.code == "capture_record_cleanup_failed"


def test_failed_output_write_removes_coordinate_record(monkeypatch, tmp_path: Path) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(),
    )
    store = CaptureRecordStore(tmp_path / "records")
    capture_id = "cap_" + "a" * 32
    monkeypatch.setattr(store, "new_capture_id", lambda: capture_id)
    monkeypatch.setattr(
        screenshot,
        "_atomic_write",
        lambda *_args: (_ for _ in ()).throw(
            ComputerError("output_write_failed", "fixture output failure")
        ),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=tmp_path / "failed.png",
            region=None,
            backend=backend,
            record_store=store,
        )
    assert raised.value.code == "output_write_failed"
    assert not (store.root / f"{capture_id}.json").exists()


def test_atomic_output_cleanup_failure_is_truthful(monkeypatch, tmp_path: Path) -> None:
    original_unlink = Path.unlink
    monkeypatch.setattr(
        screenshot.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("fixture replace failure")),
    )

    def fail_temporary_unlink(path: Path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise OSError("fixture cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)

    with pytest.raises(ComputerError) as raised:
        screenshot._atomic_write(tmp_path / "output.png", b"captured pixels")

    assert raised.value.code == "output_cleanup_failed"
    assert raised.value.details == {"original_error_code": "output_write_failed"}
    monkeypatch.setattr(Path, "unlink", original_unlink)
    for leftover in tmp_path.glob("*.tmp"):
        leftover.unlink()


def test_capture_requires_exact_process_identity_before_backend_capture(
    monkeypatch, tmp_path: Path
) -> None:
    backend = FakeBackend()
    identity = replace(backend.identity, process_started=None)
    backend.identity = identity
    backend.geometry = replace(backend.geometry, identity=identity)
    capture_called = False

    def capture(_backend, _identity):
        nonlocal capture_called
        capture_called = True
        return _non_uniform_capture()

    monkeypatch.setattr(screenshot, "_capture_full", capture)
    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=tmp_path / "identity.png",
            region=None,
            backend=backend,
            record_store=CaptureRecordStore(tmp_path / "records"),
        )
    assert raised.value.code == "capture_identity_unavailable"
    assert capture_called is False


def test_capture_rejects_reparse_output_before_backend_capture(
    monkeypatch, tmp_path: Path
) -> None:
    backend = FakeBackend()
    output = tmp_path / "existing.png"
    output.write_bytes(b"existing")
    capture_called = False

    def capture(_backend, _identity):
        nonlocal capture_called
        capture_called = True
        return _non_uniform_capture()

    monkeypatch.setattr(screenshot, "_capture_full", capture)
    monkeypatch.setattr(screenshot, "_is_reparse_point", lambda path: path == output)
    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=output,
            region=None,
            backend=backend,
            record_store=CaptureRecordStore(tmp_path / "records"),
        )
    assert raised.value.code == "invalid_output_path"
    assert capture_called is False
    assert output.read_bytes() == b"existing"


def test_out_of_bounds_region_fails_before_capture(monkeypatch, tmp_path: Path) -> None:
    capture_called = False

    def capture(_backend, _identity):
        nonlocal capture_called
        capture_called = True
        return _non_uniform_capture()

    monkeypatch.setattr(screenshot, "_capture_full", capture)

    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=tmp_path / "bad.png",
            region=Region(x=70, y=0, width=20, height=10),
            backend=FakeBackend(),
        )

    assert raised.value.code == "region_out_of_bounds"
    assert capture_called is False
    assert not (tmp_path / "bad.png").exists()


def test_minimized_window_fails_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: _non_uniform_capture(),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=tmp_path / "minimized.png",
            region=None,
            backend=FakeBackend(minimized=True),
        )

    assert raised.value.code == "window_minimized"
    assert not (tmp_path / "minimized.png").exists()


def test_all_black_capture_is_rejected_without_writing(monkeypatch, tmp_path: Path) -> None:
    uniform = screenshot.CapturedImage(
        image=Image.new("RGB", (80, 60), "black"),
        width=80,
        height=60,
        captured_unix_ms=int(time.time() * 1000),
        backend="fixture",
        cleanup_verified=True,
    )
    monkeypatch.setattr(screenshot, "_capture_full", lambda _backend, _identity: uniform)

    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=tmp_path / "uniform.png",
            region=None,
            backend=FakeBackend(),
        )

    assert raised.value.code == "capture_content_invalid"
    assert not (tmp_path / "uniform.png").exists()


def test_quality_rejects_extreme_white_and_transparent_frames() -> None:
    white = screenshot.analyze_capture_quality(Image.new("RGB", (800, 600), "white"))
    transparent = screenshot.analyze_capture_quality(
        Image.new("RGBA", (800, 600), (30, 80, 120, 0))
    )

    assert white.accepted is False
    assert "extreme_white_coverage" in white.reasons
    assert transparent.accepted is False
    assert "extreme_transparent_coverage" in transparent.reasons


def test_quality_rejects_near_black_tiny_mark_false_positive() -> None:
    image = Image.new("RGB", (1000, 700), (1, 1, 1))
    ImageDraw.Draw(image).rectangle((4, 4, 10, 10), fill=(0, 150, 210))

    quality = screenshot.analyze_capture_quality(image)

    assert quality.accepted is False
    assert "extreme_black_coverage" in quality.reasons
    assert quality.metrics["near_black_coverage"] >= 0.995


def test_quality_accepts_low_contrast_legitimate_dark_ui() -> None:
    image = Image.new("RGB", (900, 620), (16, 20, 24))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 870, 140), fill=(25, 31, 37), outline=(76, 122, 116), width=3)
    draw.rectangle((30, 170, 560, 580), fill=(19, 25, 30), outline=(62, 104, 98), width=2)
    draw.rectangle((590, 170, 870, 580), fill=(28, 35, 41))
    for index in range(7):
        y = 205 + index * 48
        draw.rectangle((65, y, 440 + index * 12, y + 8), fill=(78, 105, 103))
    draw.line((620, 510, 675, 420, 735, 455, 810, 290), fill=(180, 137, 61), width=5)

    quality = screenshot.analyze_capture_quality(image)

    assert quality.accepted is True
    assert quality.reasons == ()
    assert quality.metrics["luminance_percentile_spread"] > 4


def test_quality_accepts_terminal_like_dark_text() -> None:
    image = Image.new("RGB", (900, 600), (4, 7, 5))
    draw = ImageDraw.Draw(image)
    for row in range(24):
        y = 12 + row * 23
        color = (105, 176, 122) if row % 3 else (170, 184, 174)
        draw.rectangle((18, y, 250 + (row % 7) * 72, y + 5), fill=color)

    quality = screenshot.analyze_capture_quality(image)

    assert quality.accepted is True
    assert quality.metrics["near_black_coverage"] < 0.995
    assert quality.metrics["edge_density"] > 0.002


def test_quality_accepts_normal_colorful_ui() -> None:
    image = Image.new("RGB", (960, 640), (235, 241, 244))
    draw = ImageDraw.Draw(image)
    colors = [(8, 111, 131), (10, 155, 139), (220, 79, 69), (241, 179, 72)]
    for index, color in enumerate(colors):
        draw.rectangle((40 + index * 220, 50, 220 + index * 220, 590), fill=color)

    quality = screenshot.analyze_capture_quality(image)

    assert quality.accepted is True
    assert quality.metrics["dominant_color_ratio"] < 0.995


def test_quality_analysis_is_bounded_for_extreme_native_dimensions() -> None:
    image = Image.new("RGB", (screenshot.MAX_CAPTURE_DIMENSION, 1), (20, 30, 40))
    ImageDraw.Draw(image).rectangle((0, 0, 20_000, 0), fill=(180, 90, 40))

    quality = screenshot.analyze_capture_quality(image)

    assert quality.analysis_size == (screenshot.MAX_QUALITY_ANALYSIS_EDGE, 1)
    assert quality.accepted is True


def test_desktop_fallback_masks_pixels_outside_verified_dwm_frame(monkeypatch) -> None:
    identity = replace(
        FakeBackend(width=10, height=8).identity,
        bounds=Bounds(100, 200, 10, 8),
    )
    monkeypatch.setattr(
        screenshot,
        "_extended_frame_bounds",
        lambda _hwnd: Bounds(102, 201, 6, 5),
    )

    masked = screenshot._mask_desktop_capture_to_dwm_frame(
        Image.new("RGB", (10, 8), "white"),
        identity,
    )

    assert masked.getpixel((0, 0)) == (0, 0, 0)
    assert masked.getpixel((2, 1)) == (255, 255, 255)
    assert masked.getpixel((7, 5)) == (255, 255, 255)
    assert masked.getpixel((8, 6)) == (0, 0, 0)


def test_desktop_fallback_rejects_unverified_dwm_frame(monkeypatch) -> None:
    identity = FakeBackend(width=10, height=8).identity
    monkeypatch.setattr(
        screenshot,
        "_extended_frame_bounds",
        lambda _hwnd: None,
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._mask_desktop_capture_to_dwm_frame(
            Image.new("RGB", (10, 8), "white"),
            identity,
        )

    assert raised.value.code == "capture_geometry_unavailable"


def test_desktop_occlusion_check_fails_closed_when_z_order_is_unavailable() -> None:
    class Win32Gui:
        error = RuntimeError

        @staticmethod
        def EnumWindows(_callback, _extra) -> None:
            raise RuntimeError("fixture z-order failure")

    class Modules:
        win32gui = Win32Gui()

    class Backend:
        @staticmethod
        def modules():
            return Modules()

    assert screenshot._window_above_intersects(
        Backend(),
        FakeBackend().identity,
    ) is True


def test_desktop_occlusion_check_fails_closed_when_target_is_missing() -> None:
    class Win32Gui:
        error = RuntimeError

        @staticmethod
        def EnumWindows(_callback, _extra) -> None:
            return None

    class Backend:
        @staticmethod
        def modules():
            return SimpleNamespace(win32gui=Win32Gui())

    assert screenshot._window_above_intersects(
        Backend(),
        FakeBackend().identity,
    ) is True


def test_desktop_monitor_coverage_rejects_virtual_desktop_gap() -> None:
    modules = SimpleNamespace(
        win32api=SimpleNamespace(
            EnumDisplayMonitors=lambda: [
                (None, None, (0, 0, 100, 100)),
                (None, None, (150, 0, 250, 100)),
            ]
        )
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._assert_desktop_monitor_coverage(
            modules,
            Bounds(50, 10, 150, 80),
        )

    assert raised.value.code == "desktop_crop_target_offscreen"


def test_desktop_capture_preserves_specific_dwm_geometry_error(monkeypatch) -> None:
    class NativeFunction:
        def __init__(self, function) -> None:
            self.function = function
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.function(*args)

    identity = FakeBackend().identity
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(GetForegroundWindow=lambda: identity.hwnd),
        win32api=SimpleNamespace(
            EnumDisplayMonitors=lambda: [(None, None, (0, 0, 2560, 1440))]
        ),
    )
    backend = SimpleNamespace(
        modules=lambda: modules,
        revalidate=lambda expected: expected,
        is_minimized=lambda _hwnd: False,
    )
    user32 = SimpleNamespace(GetAncestor=NativeFunction(lambda *_args: identity.hwnd))
    monkeypatch.setattr(screenshot, "_windll", lambda _name: user32)
    monkeypatch.setattr(screenshot, "_window_above_intersects", lambda *_args: False)
    monkeypatch.setattr(
        screenshot,
        "_extended_frame_bounds",
        lambda _hwnd: identity.bounds,
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_exact_desktop_region",
        lambda *_args, **_kwargs: (Image.new("RGB", (80, 60)), 100_000),
    )

    def reject_composition(*_args):
        raise ComputerError(
            "capture_dimensions_invalid",
            "fixture DWM geometry mismatch",
        )

    monkeypatch.setattr(
        screenshot,
        "_place_desktop_frame_on_window_canvas",
        reject_composition,
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_desktop_full_native(backend, identity)

    assert raised.value.code == "capture_dimensions_invalid"


def test_desktop_capture_pins_frame_before_copy_and_captures_only_frame(
    monkeypatch,
) -> None:
    class NativeFunction:
        def __init__(self, function) -> None:
            self.function = function
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.function(*args)

    identity = replace(
        FakeBackend(width=100, height=80).identity,
        bounds=Bounds(100, 200, 100, 80),
    )
    frame = Bounds(103, 202, 94, 75)
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(GetForegroundWindow=lambda: identity.hwnd),
        win32api=SimpleNamespace(
            EnumDisplayMonitors=lambda: [(None, None, (0, 0, 2560, 1440))]
        ),
    )
    backend = SimpleNamespace(
        modules=lambda: modules,
        revalidate=lambda expected: expected,
        is_minimized=lambda _hwnd: False,
    )
    user32 = SimpleNamespace(GetAncestor=NativeFunction(lambda *_args: identity.hwnd))
    observed: list[dict[str, int]] = []
    monkeypatch.setattr(screenshot, "_windll", lambda _name: user32)
    monkeypatch.setattr(screenshot, "_window_above_intersects", lambda *_args: False)
    monkeypatch.setattr(screenshot, "_extended_frame_bounds", lambda _hwnd: frame)

    def capture_region(_modules, **kwargs):
        observed.append(kwargs)
        return Image.new("RGB", (frame.width, frame.height), "white"), 100_000

    monkeypatch.setattr(screenshot, "_capture_exact_desktop_region", capture_region)

    captured = screenshot._capture_desktop_full_native(backend, identity)

    assert observed == [
        {
            "left": frame.x,
            "top": frame.y,
            "width": frame.width,
            "height": frame.height,
        }
    ]
    assert captured.image.size == (100, 80)
    assert captured.image.getpixel((0, 0)) == (0, 0, 0)
    assert captured.image.getpixel((3, 2)) == (255, 255, 255)


def test_desktop_capture_rejects_dwm_frame_drift_after_copy(monkeypatch) -> None:
    class NativeFunction:
        def __init__(self, function) -> None:
            self.function = function
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.function(*args)

    identity = replace(
        FakeBackend(width=100, height=80).identity,
        bounds=Bounds(100, 200, 100, 80),
    )
    frames = iter((Bounds(103, 202, 94, 75), Bounds(104, 202, 93, 75)))
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(GetForegroundWindow=lambda: identity.hwnd),
        win32api=SimpleNamespace(
            EnumDisplayMonitors=lambda: [(None, None, (0, 0, 2560, 1440))]
        ),
    )
    backend = SimpleNamespace(
        modules=lambda: modules,
        revalidate=lambda expected: expected,
        is_minimized=lambda _hwnd: False,
    )
    user32 = SimpleNamespace(GetAncestor=NativeFunction(lambda *_args: identity.hwnd))
    monkeypatch.setattr(screenshot, "_windll", lambda _name: user32)
    monkeypatch.setattr(screenshot, "_window_above_intersects", lambda *_args: False)
    monkeypatch.setattr(screenshot, "_extended_frame_bounds", lambda _hwnd: next(frames))
    monkeypatch.setattr(
        screenshot,
        "_capture_exact_desktop_region",
        lambda *_args, **_kwargs: (Image.new("RGB", (94, 75), "white"), 100_000),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_desktop_full_native(backend, identity)

    assert raised.value.code == "desktop_crop_frame_changed"


def test_exact_desktop_capture_allocates_only_requested_region_and_timestamps_before_copy(
    monkeypatch,
) -> None:
    class NativeFunction:
        def __init__(self, function) -> None:
            self.function = function
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.function(*args)

    class BitmapOwner:
        @staticmethod
        def Detach() -> int:
            return 30

    class Bitmap:
        @staticmethod
        def GetInfo() -> dict[str, int]:
            return {"bmWidth": 3, "bmHeight": 2}

        @staticmethod
        def GetBitmapBits(_raw: bool) -> bytes:
            return bytes((30, 20, 10, 0)) * 6

    selected: list[tuple[int, int]] = []
    bitmap_sizes: list[tuple[int, int]] = []
    bitblt_calls: list[tuple[object, ...]] = []
    clock = [100.0]

    def bitblt(*args):
        bitblt_calls.append(args)
        clock[0] = 105.0
        return True

    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetDC=lambda _hwnd: 10,
            CreateCompatibleDC=lambda _dc: 20,
            CreateCompatibleBitmap=lambda _dc, width, height: (
                bitmap_sizes.append((width, height)) or BitmapOwner()
            ),
            SelectObject=lambda dc, bitmap: selected.append((dc, bitmap)) or 40,
        ),
        win32ui=SimpleNamespace(CreateBitmapFromHandle=lambda _handle: Bitmap()),
    )
    user32 = SimpleNamespace(ReleaseDC=NativeFunction(lambda *_args: True))
    gdi32 = SimpleNamespace(
        DeleteDC=NativeFunction(lambda *_args: True),
        DeleteObject=NativeFunction(lambda *_args: True),
        BitBlt=NativeFunction(bitblt),
    )
    monkeypatch.setattr(
        screenshot,
        "_windll",
        lambda name: user32 if name == "user32" else gdi32,
    )
    monkeypatch.setattr(screenshot.time, "time", lambda: clock[0])

    image, captured_unix_ms = screenshot._capture_exact_desktop_region(
        modules,
        left=-25,
        top=40,
        width=3,
        height=2,
    )

    assert image.size == (3, 2)
    assert image.getpixel((0, 0)) == (10, 20, 30)
    assert bitmap_sizes == [(3, 2)]
    assert bitblt_calls[0][5:9] == (10, -25, 40, screenshot.SRCCOPY | screenshot.CAPTUREBLT)
    assert captured_unix_ms == 100_000
    assert selected == [(20, 30), (20, 40)]


def test_winapp_capture_timestamp_is_conservative_before_backend_work(
    monkeypatch,
) -> None:
    identity = FakeBackend().identity
    clock = [100.0]

    class Binding:
        version = "0.6.0"

    class Adapter:
        def __init__(self, _backend) -> None:
            pass

        def screenshot(self, _identity, *, output, timeout_seconds):
            assert timeout_seconds == screenshot.CAPTURE_TIMEOUT_SECONDS
            output.write_bytes(b"fixture")
            clock[0] = 105.0
            return Binding(), {
                "width": 80,
                "height": 60,
                "capture_api": "windows_graphics_capture_preferred",
                "effective_capture_api": "fixture",
                "upstream_fallback_possible": "PrintWindow",
            }

    image = Image.new("RGB", (80, 60), "white")
    monkeypatch.setattr(screenshot.time, "time", lambda: clock[0])
    monkeypatch.setattr(screenshot, "WinAppAdapter", Adapter)
    monkeypatch.setattr(screenshot, "_load_capture_file", lambda *_args: image)
    monkeypatch.setattr(
        screenshot,
        "_normalize_winapp_capture",
        lambda _backend, _identity, captured: (
            captured,
            {
                "method": "exact_window_bounds",
                "source_pixel_size": {"width": 80, "height": 60},
                "canvas_offset": {"x": 0, "y": 0},
            },
        ),
    )

    captured = screenshot._capture_winapp_window(
        FakeBackend(),
        identity,
    )

    assert captured.captured_unix_ms == 100_000


def test_winapp_known_header_is_cropped_before_dwm_frame_normalization(
    monkeypatch,
) -> None:
    identity = FakeBackend(width=80, height=60).identity
    frame = Bounds(x=12, y=22, width=76, height=56)
    source = Image.new("RGB", (76, 84), "red")
    ImageDraw.Draw(source).rectangle((0, 28, 75, 83), fill="blue")
    backend = screenshot.Win32Backend()
    monkeypatch.setattr(screenshot, "_extended_frame_bounds", lambda _hwnd: frame)
    monkeypatch.setattr(
        backend,
        "native_pixel_context",
        lambda: contextmanager(lambda: (yield))(),
    )
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)

    normalized, details = screenshot._normalize_winapp_capture(
        backend,
        identity,
        source,
    )

    assert normalized.size == (80, 60)
    assert normalized.getpixel((2, 2)) == (0, 0, 255)
    assert normalized.getpixel((0, 0)) == (0, 0, 0)
    assert details == {
        "method": "winapp_header_cropped_dwm_frame_to_window_canvas",
        "source_pixel_size": {"width": 76, "height": 84},
        "header_crop_pixels": 28,
        "canvas_offset": {"x": 2, "y": 2},
    }


def test_printwindow_timestamp_is_conservative_before_pixel_copy(monkeypatch) -> None:
    class NativeFunction:
        def __init__(self, function) -> None:
            self.function = function
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.function(*args)

    class BitmapOwner:
        @staticmethod
        def Detach() -> int:
            return 30

    class Bitmap:
        @staticmethod
        def GetInfo() -> dict[str, int]:
            return {"bmWidth": 80, "bmHeight": 60}

        @staticmethod
        def GetBitmapBits(_raw: bool) -> bytes:
            clock[0] = 105.0
            return bytes((30, 20, 10, 0)) * (80 * 60)

    clock = [100.0]
    modules = SimpleNamespace(
        win32gui=SimpleNamespace(
            GetWindowDC=lambda _hwnd: 10,
            CreateCompatibleDC=lambda _dc: 20,
            CreateCompatibleBitmap=lambda _dc, _width, _height: BitmapOwner(),
            SelectObject=lambda _dc, bitmap: 40 if bitmap == 30 else 30,
        ),
        win32ui=SimpleNamespace(CreateBitmapFromHandle=lambda _handle: Bitmap()),
    )
    user32 = SimpleNamespace(
        PrintWindow=NativeFunction(lambda *_args: True),
        ReleaseDC=NativeFunction(lambda *_args: True),
    )
    gdi32 = SimpleNamespace(
        DeleteDC=NativeFunction(lambda *_args: True),
        DeleteObject=NativeFunction(lambda *_args: True),
    )
    backend = SimpleNamespace(
        modules=lambda: modules,
        revalidate=lambda expected: expected,
        is_minimized=lambda _hwnd: False,
    )
    monkeypatch.setattr(
        screenshot,
        "_windll",
        lambda name: user32 if name == "user32" else gdi32,
    )
    monkeypatch.setattr(screenshot.time, "time", lambda: clock[0])

    captured = screenshot._capture_full_native(
        backend,
        FakeBackend().identity,
    )

    assert captured.captured_unix_ms == 100_000


def test_capture_chain_falls_through_rejected_primary_without_losing_diagnostics(
    monkeypatch,
) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    dark = Image.new("RGB", (80, 60), (5, 7, 6))
    draw = ImageDraw.Draw(dark)
    for y in range(5, 55, 8):
        draw.rectangle((4, y, 72, y + 2), fill=(110, 170, 125))
    calls: list[str] = []
    monkeypatch.setattr(backend, "native_pixel_context", lambda: contextmanager(lambda: (yield))())
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)
    monkeypatch.setattr(
        screenshot,
        "_capture_winapp_window",
        lambda *_args, **_kwargs: (
            calls.append("winapp")
            or screenshot.CapturedImage(
                Image.new("RGB", (80, 60), "black"),
                80,
                60,
                1,
                "winapp_window_capture",
                True,
            )
        ),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_print_window_isolated",
        lambda *_args, **_kwargs: (
            calls.append("printwindow")
            or screenshot.CapturedImage(
                dark,
                80,
                60,
                2,
                "win32_print_window_isolated",
                True,
            )
        ),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_desktop_visible",
        lambda *_args, **_kwargs: pytest.fail("desktop fallback should not run"),
    )

    captured = screenshot._capture_bounded(backend, identity)

    assert calls == ["winapp", "printwindow"]
    assert captured.backend == "win32_print_window_isolated"
    assert [attempt["outcome"] for attempt in captured.attempts] == [
        "rejected",
        "accepted",
    ]
    assert captured.attempts[0]["error_code"] == "capture_content_invalid"


def test_capture_chain_aborts_on_identity_drift_without_more_backends(monkeypatch) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    calls: list[str] = []
    monkeypatch.setattr(backend, "native_pixel_context", lambda: contextmanager(lambda: (yield))())
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)

    def drift(*_args, **_kwargs):
        calls.append("winapp")
        raise ComputerError("stale_window", "fixture drift")

    monkeypatch.setattr(screenshot, "_capture_winapp_window", drift)
    monkeypatch.setattr(
        screenshot,
        "_capture_print_window_isolated",
        lambda *_args, **_kwargs: calls.append("printwindow"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == "stale_window"
    assert calls == ["winapp"]
    assert len(raised.value.details["capture_attempts"]) == 1


@pytest.mark.parametrize(
    "fatal_code",
    [
        "capture_cleanup_failed",
        "capture_worker_termination_failed",
        "dpi_awareness_restore_failed",
        "uia_cleanup_failed",
        "uia_termination_failed",
    ],
)
def test_capture_chain_never_falls_through_fatal_cleanup_failure(
    monkeypatch,
    fatal_code: str,
) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    calls: list[str] = []
    monkeypatch.setattr(
        backend,
        "native_pixel_context",
        lambda: contextmanager(lambda: (yield))(),
    )
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)

    def fail(*_args, **_kwargs):
        calls.append("winapp")
        raise ComputerError(fatal_code, "fixture fatal cleanup failure")

    monkeypatch.setattr(screenshot, "_capture_winapp_window", fail)
    monkeypatch.setattr(
        screenshot,
        "_capture_print_window_isolated",
        lambda *_args, **_kwargs: calls.append("printwindow"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == fatal_code
    assert calls == ["winapp"]
    assert raised.value.details["capture_attempts"][0]["error_code"] == fatal_code


def test_capture_chain_does_not_start_backend_without_cleanup_budget(
    monkeypatch,
) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    calls: list[str] = []
    for name in (
        "_capture_winapp_window",
        "_capture_print_window_isolated",
        "_capture_desktop_visible",
    ):
        monkeypatch.setattr(
            screenshot,
            name,
            lambda *_args, _name=name, **_kwargs: calls.append(_name),
        )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(
            backend,
            identity,
            deadline=(
                time.monotonic()
                + screenshot.MIN_CAPTURE_BACKEND_START_SECONDS / 2
            ),
        )

    assert raised.value.code == "capture_timeout"
    assert calls == []
    assert raised.value.details["capture_attempts"] == []


def test_capture_chain_rechecks_cleanup_budget_after_revalidation(
    monkeypatch,
) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    clock = {"now": 100.0}
    calls: list[str] = []

    def revalidate(expected):
        clock["now"] = 100.1
        return expected

    monkeypatch.setattr(screenshot.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        backend,
        "native_pixel_context",
        lambda: contextmanager(lambda: (yield))(),
    )
    monkeypatch.setattr(backend, "revalidate", revalidate)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)
    for name in (
        "_capture_winapp_window",
        "_capture_print_window_isolated",
        "_capture_desktop_visible",
    ):
        monkeypatch.setattr(
            screenshot,
            name,
            lambda *_args, _name=name, **_kwargs: calls.append(_name),
        )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(
            backend,
            identity,
            deadline=100.3,
        )

    assert raised.value.code == "capture_timeout"
    assert calls == []
    assert raised.value.details["capture_attempts"] == []


def test_isolated_worker_rechecks_cleanup_budget_after_input_staging(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    clock = {"now": 100.0}
    staged = io.BytesIO()

    def stage(_payload: bytes):
        clock["now"] = 100.1
        return staged

    monkeypatch.setattr(screenshot.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(screenshot, "preloaded_process_stdin", stage)
    monkeypatch.setattr(
        screenshot.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not start"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_worker_isolated(
            backend,
            backend.identity,
            mode="print_window",
            backend_name="win32_print_window_isolated",
            backend_details={},
            timeout_seconds=1.0,
            deadline=100.3,
        )

    assert raised.value.code == "capture_timeout"
    assert staged.closed is True


def test_isolated_worker_pre_spawn_staged_close_failure_is_fatal(
    monkeypatch,
) -> None:
    backend = FakeBackend()

    class FaultyClose(io.BytesIO):
        def close(self) -> None:
            raise OSError("fixture staged close failure")

    staged = FaultyClose()
    monkeypatch.setattr(
        screenshot,
        "preloaded_process_stdin",
        lambda _payload: staged,
    )
    monkeypatch.setattr(
        screenshot.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not start"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_worker_isolated(
            backend,
            backend.identity,
            mode="print_window",
            backend_name="win32_print_window_isolated",
            backend_details={},
            timeout_seconds=0.1,
        )

    assert raised.value.code == "capture_cleanup_failed"
    assert raised.value.details == {
        "steps": ["staged_stdin:OSError"],
        "primary_error": "capture_timeout",
    }


@pytest.mark.parametrize("failing_operation", ["write", "seek"])
def test_full_chain_staging_and_internal_close_failure_is_fatal(
    monkeypatch,
    failing_operation: str,
) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity

    class StagedOwner:
        def __init__(self) -> None:
            self.file = io.BytesIO()
            self.close_calls = 0

        @property
        def closed(self) -> bool:
            return self.file.closed

        def write(self, payload) -> int:
            if failing_operation == "write":
                raise OSError("fixture stage write failure")
            return self.file.write(payload)

        def seek(self, offset: int) -> int:
            if failing_operation == "seek":
                raise OSError("fixture stage seek failure")
            return self.file.seek(offset)

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("fixture wrapper close failure")

    owner = StagedOwner()
    monkeypatch.setattr(
        backend,
        "native_pixel_context",
        lambda: contextmanager(lambda: (yield))(),
    )
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)
    monkeypatch.setattr(
        screenshot,
        "_capture_winapp_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ComputerError("uia_capture_failed", "fixture WinApp failure")
        ),
    )
    monkeypatch.setattr(
        process_io.tempfile,
        "TemporaryFile",
        lambda **_kwargs: owner,
    )
    monkeypatch.setattr(
        screenshot.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not start"),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_desktop_visible",
        lambda *_args, **_kwargs: pytest.fail("desktop fallback should not run"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == "capture_cleanup_failed"
    assert raised.value.details["steps"] == ["staged_stdin:OSError"]
    assert raised.value.details["secondary_errors"] == ["staging:OSError"]
    assert owner.close_calls == 1
    assert owner.file.closed is True
    assert [
        attempt["backend"] for attempt in raised.value.details["capture_attempts"]
    ] == ["winapp_window_capture", "win32_print_window_isolated"]


def test_full_chain_post_spawn_staged_close_failure_cleans_without_fallback(
    monkeypatch,
) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity

    class FaultyClose(io.BytesIO):
        def close(self) -> None:
            raise OSError("fixture staged close failure")

    class Process:
        stdin = None

        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    process = Process()

    def close_pipes(target, **_kwargs):
        target.stdout.close()
        target.stderr.close()
        return []

    monkeypatch.setattr(
        backend,
        "native_pixel_context",
        lambda: contextmanager(lambda: (yield))(),
    )
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)
    monkeypatch.setattr(
        screenshot,
        "_capture_winapp_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ComputerError("uia_capture_failed", "fixture WinApp failure")
        ),
    )
    monkeypatch.setattr(
        screenshot,
        "preloaded_process_stdin",
        lambda _payload: FaultyClose(),
    )
    monkeypatch.setattr(
        screenshot.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(screenshot, "_close_capture_worker_pipes", close_pipes)
    monkeypatch.setattr(
        screenshot,
        "_capture_desktop_visible",
        lambda *_args, **_kwargs: pytest.fail("desktop fallback should not run"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == "capture_cleanup_failed"
    assert raised.value.details["steps"] == ["staged_stdin:OSError"]
    assert process.killed is True
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert [
        attempt["backend"] for attempt in raised.value.details["capture_attempts"]
    ] == ["winapp_window_capture", "win32_print_window_isolated"]


def test_staged_close_with_termination_failure_keeps_termination_primary(
    monkeypatch,
) -> None:
    backend = FakeBackend()

    class FaultyClose(io.BytesIO):
        def close(self) -> None:
            raise OSError("fixture staged close failure")

    class Process:
        stdin = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode = None

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            raise OSError("fixture termination failure")

    monkeypatch.setattr(
        screenshot,
        "preloaded_process_stdin",
        lambda _payload: FaultyClose(),
    )
    monkeypatch.setattr(
        screenshot.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Process(),
    )
    monkeypatch.setattr(
        screenshot,
        "_close_capture_worker_pipes",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_worker_isolated(
            backend,
            backend.identity,
            mode="print_window",
            backend_name="win32_print_window_isolated",
            backend_details={},
            timeout_seconds=1.0,
        )

    assert raised.value.code == "capture_worker_termination_failed"
    assert raised.value.details["secondary_cleanup_errors"] == [
        "primary_cleanup:staged_stdin:OSError",
        "primary:capture_cleanup_failed",
    ]


def test_full_isolated_worker_preserves_enriched_termination_failure(
    monkeypatch,
) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    primary = ComputerError(
        "capture_worker_termination_failed",
        "fixture inner termination failure",
        details={"secondary_cleanup_errors": ["stdout:CancelIoEx:5"]},
    )

    class Process:
        stdin = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode = None

        @staticmethod
        def poll() -> None:
            return None

    monkeypatch.setattr(
        backend,
        "native_pixel_context",
        lambda: contextmanager(lambda: (yield))(),
    )
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)
    monkeypatch.setattr(
        screenshot,
        "_capture_winapp_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ComputerError("uia_capture_failed", "fixture primary unavailable")
        ),
    )
    monkeypatch.setattr(
        screenshot.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Process(),
    )
    monkeypatch.setattr(
        screenshot,
        "_read_worker_bounded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    monkeypatch.setattr(
        screenshot,
        "_kill_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ComputerError(
                "capture_worker_termination_failed",
                "fixture outer termination retry failed",
            )
        ),
    )
    monkeypatch.setattr(
        screenshot,
        "_close_capture_worker_pipes",
        lambda *_args, **_kwargs: ["stderr:CancelIoEx:5"],
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_desktop_visible",
        lambda *_args, **_kwargs: pytest.fail("desktop fallback should not run"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == "capture_worker_termination_failed"
    assert raised.value.details["secondary_cleanup_errors"] == [
        "stdout:CancelIoEx:5",
        "outer:termination_retry_failed",
        "stderr:CancelIoEx:5",
    ]
    assert [
        attempt["backend"] for attempt in raised.value.details["capture_attempts"]
    ] == ["winapp_window_capture", "win32_print_window_isolated"]


def test_capture_chain_uses_win32_subclasses_instead_of_bypassing_chain(
    monkeypatch,
) -> None:
    class BackendSubclass(screenshot.Win32Backend):
        pass

    backend = BackendSubclass()
    identity = FakeBackend().identity
    calls: list[str] = []
    monkeypatch.setattr(
        backend,
        "native_pixel_context",
        lambda: contextmanager(lambda: (yield))(),
    )
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)
    monkeypatch.setattr(
        screenshot,
        "_capture_winapp_window",
        lambda *_args, **_kwargs: (
            calls.append("winapp") or _non_uniform_capture()
        ),
    )

    captured = screenshot._capture_bounded(backend, identity)

    assert calls == ["winapp"]
    assert captured.attempts[0]["outcome"] == "accepted"


def test_capture_chain_never_accepts_success_after_absolute_deadline(
    monkeypatch,
) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    clock = [100.0]
    calls: list[str] = []
    monkeypatch.setattr(screenshot.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        backend,
        "native_pixel_context",
        lambda: contextmanager(lambda: (yield))(),
    )
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)

    def late_success(*_args, **_kwargs):
        calls.append("winapp")
        clock[0] += screenshot.CAPTURE_TIMEOUT_SECONDS + 0.1
        return _non_uniform_capture()

    monkeypatch.setattr(screenshot, "_capture_winapp_window", late_success)
    monkeypatch.setattr(
        screenshot,
        "_capture_print_window_isolated",
        lambda *_args, **_kwargs: calls.append("printwindow"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == "capture_timeout"
    assert calls == ["winapp"]


def test_late_fatal_cleanup_is_not_masked_by_capture_timeout(monkeypatch) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    clock = [100.0]
    monkeypatch.setattr(screenshot.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        backend,
        "native_pixel_context",
        lambda: contextmanager(lambda: (yield))(),
    )
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)

    def late_cleanup(*_args, **_kwargs):
        clock[0] += screenshot.CAPTURE_TIMEOUT_SECONDS + 0.1
        raise ComputerError("uia_cleanup_failed", "fixture cleanup failure")

    monkeypatch.setattr(screenshot, "_capture_winapp_window", late_cleanup)

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == "uia_cleanup_failed"


def test_fatal_termination_survives_post_capture_identity_failure(monkeypatch) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    calls = 0
    fallbacks: list[str] = []
    monkeypatch.setattr(
        backend,
        "native_pixel_context",
        lambda: contextmanager(lambda: (yield))(),
    )

    def revalidate(expected):
        nonlocal calls
        calls += 1
        if calls == 1:
            return expected
        raise ComputerError("stale_window", "fixture post-capture drift")

    monkeypatch.setattr(backend, "revalidate", revalidate)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)
    monkeypatch.setattr(
        screenshot,
        "_capture_winapp_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ComputerError("uia_termination_failed", "fixture leaked helper")
        ),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_print_window_isolated",
        lambda *_args, **_kwargs: fallbacks.append("printwindow"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == "uia_termination_failed"
    assert raised.value.details["secondary_errors"] == ["stale_window"]
    assert fallbacks == []


def test_capture_chain_revalidates_before_and_after_each_failed_backend(
    monkeypatch,
) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    revalidations = 0
    monkeypatch.setattr(backend, "native_pixel_context", lambda: contextmanager(lambda: (yield))())

    def revalidate(expected):
        nonlocal revalidations
        revalidations += 1
        return expected

    monkeypatch.setattr(backend, "revalidate", revalidate)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)

    def fail(code: str):
        def capture(*_args, **_kwargs):
            raise ComputerError(code, "fixture failure")

        return capture

    monkeypatch.setattr(
        screenshot,
        "_capture_winapp_window",
        fail("capability_missing"),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_print_window_isolated",
        fail("capture_backend_failed"),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_desktop_visible",
        fail("desktop_crop_target_not_foreground"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == "capture_failed"
    assert revalidations == 6
    attempts = raised.value.details["capture_attempts"]
    assert len(attempts) == 3
    allowed_fields = {"backend", "outcome", "duration_ms", "error_code"}
    assert all(set(attempt) <= allowed_fields for attempt in attempts)
    assert len(json.dumps(raised.value.details, ensure_ascii=True)) < 2_000


def test_capture_chain_reports_mixed_quality_rejection_truthfully(monkeypatch) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    monkeypatch.setattr(backend, "native_pixel_context", lambda: contextmanager(lambda: (yield))())
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)
    monkeypatch.setattr(
        screenshot,
        "_capture_winapp_window",
        lambda *_args, **_kwargs: screenshot.CapturedImage(
            Image.new("RGB", (80, 60), "black"),
            80,
            60,
            1,
            "winapp_window_capture",
            True,
        ),
    )

    def fail(code: str):
        def capture(*_args, **_kwargs):
            raise ComputerError(code, "fixture failure")

        return capture

    monkeypatch.setattr(
        screenshot,
        "_capture_print_window_isolated",
        fail("capture_backend_failed"),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_desktop_visible",
        fail("desktop_crop_target_not_foreground"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == "capture_content_invalid"
    assert "at least one capture" in raised.value.message
    assert "Every capture backend returned" not in raised.value.message


def test_individual_timeout_does_not_claim_whole_chain_timed_out(monkeypatch) -> None:
    backend = screenshot.Win32Backend()
    identity = FakeBackend().identity
    monkeypatch.setattr(backend, "native_pixel_context", lambda: contextmanager(lambda: (yield))())
    monkeypatch.setattr(backend, "revalidate", lambda expected: expected)
    monkeypatch.setattr(backend, "is_minimized", lambda _hwnd: False)

    def fail(code: str):
        def capture(*_args, **_kwargs):
            raise ComputerError(code, "fixture failure")

        return capture

    monkeypatch.setattr(
        screenshot,
        "_capture_winapp_window",
        fail("uia_timeout"),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_print_window_isolated",
        fail("capture_backend_failed"),
    )
    monkeypatch.setattr(
        screenshot,
        "_capture_desktop_visible",
        fail("desktop_crop_target_not_foreground"),
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._capture_bounded(backend, identity)

    assert raised.value.code == "capture_failed"
    assert "safety deadline" not in raised.value.message


def test_unverified_capture_cleanup_is_rejected(monkeypatch, tmp_path: Path) -> None:
    capture = _non_uniform_capture()
    unverified = screenshot.CapturedImage(
        image=capture.image,
        width=capture.width,
        height=capture.height,
        captured_unix_ms=capture.captured_unix_ms,
        backend=capture.backend,
        cleanup_verified=False,
    )
    monkeypatch.setattr(screenshot, "_capture_full", lambda _backend, _identity: unverified)

    with pytest.raises(ComputerError) as raised:
        screenshot.capture_screenshot(
            hwnd=101,
            output=tmp_path / "unclean.png",
            region=None,
            backend=FakeBackend(),
        )

    assert raised.value.code == "capture_cleanup_failed"
    assert not (tmp_path / "unclean.png").exists()


def test_human_window_output_escapes_terminal_controls() -> None:
    output = render_windows(
        {
            "windows": {
                "available": True,
                "count": 1,
                "truncated": False,
                "items": [
                    {
                        "hwnd": 1,
                        "pid": 2,
                        "process": "fixture\x07.exe",
                        "state": "normal",
                        "bounds": {"x": 0, "y": 0, "width": 10, "height": 10},
                        "title": "title\x1b]0;spoof\x07\r\nnext\u202e",
                    }
                ],
            }
        }
    )

    assert "\x1b" not in output
    assert "\x07" not in output
    assert "\r" not in output
    assert "\nnext" not in output
    assert "\u202e" not in output
    assert "\\x1b" in output
    assert "\\x07" in output
    assert "\\x0d\\x0anext" in output
    assert "\\u202e" in output


def test_json_output_escapes_unicode_and_c1_controls() -> None:
    output = render_json("computer.windows", {"title": "safe\u202e\u0085\x7f"})

    assert "\u202e" not in output
    assert "\u0085" not in output
    assert "\x7f" not in output
    assert "\\u202e" in output
    assert "\\u0085" in output
    assert "\\u007f" in output


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ok", "yes"),
        ("width", "2"),
        ("height", 2.9),
        ("png_bytes", "70"),
        ("captured_unix_ms", "1000"),
        ("cleanup_verified", "false"),
    ],
)
def test_capture_worker_metadata_requires_exact_types(field: str, value: object) -> None:
    response = {
        "ok": True,
        "width": 2,
        "height": 2,
        "png_bytes": 70,
        "captured_unix_ms": 1_800_000_000_000,
        "cleanup_verified": True,
    }
    response[field] = value

    with pytest.raises(ComputerError) as raised:
        screenshot._parse_worker_metadata(response)

    assert raised.value.code == "capture_worker_failed"


def test_capture_worker_png_dimensions_are_checked_before_pixel_load(monkeypatch) -> None:
    load_called = False

    class EncodedImage:
        size = (3, 3)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def load(self) -> None:
            nonlocal load_called
            load_called = True

    class ImageModule:
        @staticmethod
        def open(_payload):
            return EncodedImage()

    monkeypatch.setattr(screenshot, "_load_pillow", lambda: ImageModule())
    metadata = screenshot.CaptureWorkerMetadata(
        width=2,
        height=2,
        png_bytes=1,
        captured_unix_ms=1_800_000_000_000,
        cleanup_verified=True,
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._decode_worker_png(b"x", metadata)

    assert raised.value.code == "capture_dimensions_invalid"
    assert load_called is False


def test_capture_worker_declared_dimensions_must_match_target() -> None:
    metadata = screenshot.CaptureWorkerMetadata(
        width=3,
        height=3,
        png_bytes=70,
        captured_unix_ms=1_800_000_000_000,
        cleanup_verified=True,
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._validate_worker_target_dimensions(metadata, FakeBackend().identity)

    assert raised.value.code == "capture_dimensions_invalid"


def test_capture_worker_output_is_capped_while_streaming() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; sys.stdout.write('x' * 20000); sys.stdout.flush()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._read_worker_bounded(
            process,
            timeout_seconds=2.0,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )

    assert raised.value.code == "capture_worker_output_too_large"
    assert process.poll() is not None


def test_capture_worker_environment_excludes_caller_secrets(monkeypatch) -> None:
    monkeypatch.setenv("TASK_270_SECRET", "must-not-pass")
    monkeypatch.setenv("PYTHONPATH", "hostile-path")

    environment = screenshot._worker_environment("trusted-source")

    assert "TASK_270_SECRET" not in environment
    assert environment["PYTHONPATH"] == "trusted-source"
    assert environment["PYTHONSAFEPATH"] == "1"
    assert "PATH" not in environment


def test_capture_worker_timeout_has_no_stdin_writer_thread() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    started = time.monotonic()
    with pytest.raises(ComputerError) as raised:
        screenshot._read_worker_bounded(
            process,
            timeout_seconds=0.2,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )

    assert raised.value.code == "capture_timeout"
    assert time.monotonic() - started < 3.0
    assert process.poll() is not None
    assert not any(
        thread.name == "agent-tools-capture-stdin"
        for thread in threading.enumerate()
    )


def test_capture_worker_pipe_close_failure_is_fatal(monkeypatch) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    original_close = screenshot._close_capture_worker_pipes
    monkeypatch.setattr(
        screenshot,
        "_close_capture_worker_pipes",
        lambda _process, **_kwargs: ["stdout:OSError"],
    )
    try:
        with pytest.raises(ComputerError) as raised:
            screenshot._read_worker_bounded(
                process,
                timeout_seconds=1.0,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
            )
    finally:
        original_close(process)

    assert raised.value.code == "capture_cleanup_failed"
    assert raised.value.details["steps"] == ["stdout:OSError"]


def test_capture_worker_failed_termination_is_fatal_and_deadline_bounded(
    monkeypatch,
) -> None:
    class Process:
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode = None

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            raise OSError("fixture termination failure")

    monkeypatch.setattr(
        screenshot,
        "_read_capture_pipe_available",
        lambda _stream: (b"", False),
    )
    started = time.monotonic()

    with pytest.raises(ComputerError) as raised:
        screenshot._read_worker_bounded(
            Process(),
            timeout_seconds=0.1,
            max_stdout_bytes=1_024,
            max_stderr_bytes=1_024,
        )

    assert raised.value.code == "capture_worker_termination_failed"
    assert time.monotonic() - started < 0.5


def test_capture_worker_termination_failure_precedes_pipe_cleanup(
    monkeypatch,
) -> None:
    class Process:
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode = None

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            raise OSError("fixture kill failure")

    monkeypatch.setattr(
        screenshot,
        "_read_capture_pipe_available",
        lambda _stream: (b"", False),
    )
    monkeypatch.setattr(
        screenshot,
        "close_stream_os_enforced",
        lambda _stream: "fixture_cleanup",
    )

    with pytest.raises(ComputerError) as raised:
        screenshot._read_worker_bounded(
            Process(),
            timeout_seconds=0.05,
            max_stdout_bytes=1_024,
            max_stderr_bytes=1_024,
        )

    assert raised.value.code == "capture_worker_termination_failed"
    assert raised.value.details["secondary_cleanup_errors"] == [
        "stdin:fixture_cleanup",
        "stdout:fixture_cleanup",
        "stderr:fixture_cleanup",
    ]


def test_header_cropped_dwm_padding_is_reported_in_final_limitations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture = _non_uniform_capture()
    monkeypatch.setattr(
        screenshot,
        "_capture_full",
        lambda _backend, _identity: screenshot.CapturedImage(
            image=capture.image,
            width=capture.width,
            height=capture.height,
            captured_unix_ms=capture.captured_unix_ms,
            backend="winapp_window_capture",
            cleanup_verified=True,
            backend_details={
                "normalization": {
                    "method": "winapp_header_cropped_dwm_frame_to_window_canvas"
                }
            },
        ),
    )

    result = screenshot.capture_screenshot(
        hwnd=101,
        output=tmp_path / "header-normalized.png",
        region=None,
        backend=FakeBackend(),
        record_store=CaptureRecordStore(tmp_path / "header-records"),
    )

    assert "non_rendered_window_border_is_canvas_padding" in result["limitations"]


def test_capture_worker_blocked_pipe_close_is_fatal_and_deadline_bounded(
    monkeypatch,
) -> None:
    class SlowClose(io.BytesIO):
        def close(self) -> None:
            time.sleep(2)
            super().close()

    class Process:
        stdin = io.BytesIO()
        stdout = SlowClose()
        stderr = io.BytesIO()
        returncode = 0

        @staticmethod
        def poll() -> int:
            return 0

    monkeypatch.setattr(
        screenshot,
        "_read_capture_pipe_available",
        lambda _stream: (b"", True),
    )
    monkeypatch.setattr(
        screenshot,
        "close_stream_os_enforced",
        lambda stream: "blocked_close" if isinstance(stream, SlowClose) else None,
    )
    started = time.monotonic()

    with pytest.raises(ComputerError) as raised:
        screenshot._read_worker_bounded(
            Process(),
            timeout_seconds=0.2,
            max_stdout_bytes=1_024,
            max_stderr_bytes=1_024,
        )

    assert raised.value.code == "capture_cleanup_failed"
    assert raised.value.details["steps"] == ["stdout:blocked_close"]
    assert time.monotonic() - started < 0.6
    assert not any(
        thread.name.startswith("agent-tools-capture-close-")
        for thread in threading.enumerate()
    )
