from __future__ import annotations

import importlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_tools.audio import play_wav_blocking

DirectPlaybackBackend = Literal["winsound", "afplay", "paplay", "aplay", "ffplay"]
PlaybackStrategy = Literal["controller-queue", "direct"]


@dataclass(frozen=True)
class PlaybackCapability:
    available: bool
    backend: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class PlaybackCapabilities:
    platform: str
    direct: PlaybackCapability
    controller_queue: PlaybackCapability


@dataclass(frozen=True)
class PlaybackPlan:
    available: bool
    strategy: PlaybackStrategy | None = None
    backend: str | None = None
    error_message: str | None = None


def detect_playback_capabilities(
    *,
    platform_name: str | None = None,
) -> PlaybackCapabilities:
    current_platform = platform_name or sys.platform
    return PlaybackCapabilities(
        platform=current_platform,
        direct=_detect_direct_playback_capability(current_platform),
        controller_queue=_detect_controller_queue_capability(current_platform),
    )


def plan_playback(*, platform_name: str | None = None) -> PlaybackPlan:
    capabilities = detect_playback_capabilities(platform_name=platform_name)
    if capabilities.platform == "win32":
        if capabilities.controller_queue.available:
            return PlaybackPlan(
                available=True,
                strategy="controller-queue",
                backend=capabilities.controller_queue.backend,
            )
        return PlaybackPlan(
            available=False,
            error_message=capabilities.controller_queue.error_message,
        )
    if capabilities.platform in {"linux", "darwin"}:
        if capabilities.direct.available:
            return PlaybackPlan(
                available=True,
                strategy="direct",
                backend=capabilities.direct.backend,
            )
        return PlaybackPlan(
            available=False,
            error_message=capabilities.direct.error_message,
        )
    return PlaybackPlan(
        available=False,
        error_message=(
            "Playback is supported only on Windows, macOS, and Linux."
        ),
    )


def play_direct_wav(
    wav_data: bytes,
    *,
    backend: str,
    output_path: Path | None = None,
) -> None:
    play_wav_blocking(wav_data, backend=backend, output_path=output_path)


def _detect_direct_playback_capability(platform_name: str) -> PlaybackCapability:
    if platform_name == "win32":
        return PlaybackCapability(available=True, backend="winsound")
    if platform_name == "darwin":
        backend = _first_available_backend(("afplay", "ffplay"))
        if backend is not None:
            return PlaybackCapability(available=True, backend=backend)
        return PlaybackCapability(
            available=False,
            error_message=(
                "macOS playback requires `afplay` (normally bundled with macOS) "
                "or `ffplay` on PATH."
            ),
        )
    if platform_name == "linux":
        backend = _first_available_backend(("paplay", "aplay", "ffplay"))
        if backend is not None:
            return PlaybackCapability(available=True, backend=backend)
        return PlaybackCapability(
            available=False,
            error_message=(
                "Linux playback requires one of `paplay` (PulseAudio), "
                "`aplay` (ALSA), or `ffplay` (FFmpeg) on PATH."
            ),
        )
    return PlaybackCapability(
        available=False,
        error_message="Playback is supported only on Windows, macOS, and Linux.",
    )


def _detect_controller_queue_capability(platform_name: str) -> PlaybackCapability:
    if platform_name not in {"win32", "darwin", "linux"}:
        return PlaybackCapability(
            available=False,
            error_message="The desktop controller is supported only on Windows, macOS, and Linux.",
        )
    try:
        importlib.import_module("PySide6.QtMultimedia")
        importlib.import_module("PySide6.QtWidgets")
    except ImportError:
        return PlaybackCapability(
            available=False,
            error_message=(
                "Desktop controller playback requires PySide6. "
                "Install with: pip install \"ai-nd-co-agent-tools[ui]\""
            ),
        )
    except Exception as exc:
        return PlaybackCapability(
            available=False,
            error_message=(
                "Desktop controller playback is unavailable because PySide6 multimedia "
                f"could not be imported: {exc}"
            ),
        )
    return PlaybackCapability(available=True, backend="pyside6-qmediaplayer")


def _first_available_backend(
    candidates: tuple[DirectPlaybackBackend, ...],
) -> DirectPlaybackBackend | None:
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    return None
