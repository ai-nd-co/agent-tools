from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final
from uuid import uuid4

from agent_tools.audio import wav_duration_ms
from agent_tools.runtime import audio_cache_dir, ensure_runtime_dirs

VOXFLOW_PACKAGE: Final = "com.ai_nd.co.voxflow"
VOXFLOW_ADB_ACTION: Final = f"{VOXFLOW_PACKAGE}.action.ENQUEUE_TTS"
VOXFLOW_RECEIVER_COMPONENT: Final = (
    f"{VOXFLOW_PACKAGE}/{VOXFLOW_PACKAGE}.service.AdbTtsCommandReceiver"
)
VOXFLOW_INBOX_DIR: Final = f"/sdcard/Android/data/{VOXFLOW_PACKAGE}/files/adb_inbox"
PLAYBACK_TARGET_WINDOWS: Final = "windows"
PLAYBACK_TARGET_VOXFLOW_ADB: Final = "voxflow-adb"
PLAYBACK_TARGET_CHOICES: Final[tuple[str, str]] = (
    PLAYBACK_TARGET_WINDOWS,
    PLAYBACK_TARGET_VOXFLOW_ADB,
)


@dataclass(frozen=True)
class VoxFlowAdbDispatchRequest:
    raw_text: str
    tts_text: str
    wav_data: bytes
    source_label: str | None
    voice: str
    language: str | None
    speed: float
    model: str | None
    reasoning_effort: str | None
    adb_serial: str | None = None


@dataclass(frozen=True)
class VoxFlowAdbDispatchResult:
    remote_item_id: str
    adb_serial: str
    remote_audio_path: str
    temp_write_ms: float
    push_ms: float
    broadcast_ms: float
    cleanup_ms: float

    @property
    def total_ms(self) -> float:
        return self.temp_write_ms + self.push_ms + self.broadcast_ms + self.cleanup_ms


def dispatch_to_voxflow_adb(request: VoxFlowAdbDispatchRequest) -> VoxFlowAdbDispatchResult:
    ensure_runtime_dirs()
    adb_serial = request.adb_serial or resolve_single_connected_device()
    remote_item_id = str(uuid4())
    remote_audio_path = f"{VOXFLOW_INBOX_DIR}/{remote_item_id}.wav"

    temp_write_started = datetime.now(tz=UTC)
    temp_path = _write_temp_wav(request.wav_data, remote_item_id)
    temp_write_ms = _elapsed_ms(temp_write_started)

    push_ms = 0.0
    broadcast_ms = 0.0
    cleanup_ms = 0.0
    push_succeeded = False
    try:
        push_started = datetime.now(tz=UTC)
        _run_adb(
            adb_serial,
            ["shell", "mkdir", "-p", VOXFLOW_INBOX_DIR],
            error_prefix="ADB inbox directory creation failed",
        )
        _run_adb(
            adb_serial,
            ["push", str(temp_path), remote_audio_path],
            error_prefix="ADB push failed",
        )
        push_ms = _elapsed_ms(push_started)
        push_succeeded = True

        created_at_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        broadcast_started = datetime.now(tz=UTC)
        duration_ms = wav_duration_ms(request.wav_data)
        _run_adb(
            adb_serial,
            _broadcast_args(
                remote_item_id=remote_item_id,
                created_at_ms=created_at_ms,
                remote_audio_path=remote_audio_path,
                duration_ms=duration_ms,
                request=request,
            ),
            error_prefix="ADB broadcast failed",
        )
        broadcast_ms = _elapsed_ms(broadcast_started)
    except Exception:
        if push_succeeded:
            cleanup_started = datetime.now(tz=UTC)
            try:
                _run_adb(
                    adb_serial,
                    ["shell", "rm", "-f", remote_audio_path],
                    error_prefix="ADB cleanup failed",
                )
            except Exception:
                pass
            cleanup_ms = _elapsed_ms(cleanup_started)
        raise
    finally:
        temp_path.unlink(missing_ok=True)

    return VoxFlowAdbDispatchResult(
        remote_item_id=remote_item_id,
        adb_serial=adb_serial,
        remote_audio_path=remote_audio_path,
        temp_write_ms=temp_write_ms,
        push_ms=push_ms,
        broadcast_ms=broadcast_ms,
        cleanup_ms=cleanup_ms,
    )


def resolve_single_connected_device() -> str:
    completed = subprocess.run(
        ["adb", "devices"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(_render_adb_failure("ADB device probe failed", completed))

    serials: list[str] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("List of devices attached"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serials.append(parts[0])

    if not serials:
        raise RuntimeError("No connected ADB device is available for VoxFlow playback.")
    if len(serials) > 1:
        raise RuntimeError(
            "Multiple ADB devices are connected. Pass --adb-serial or set AGENT_TOOLS_ADB_SERIAL."
        )
    return serials[0]


def _write_temp_wav(wav_data: bytes, remote_item_id: str) -> Path:
    ensure_runtime_dirs()
    with NamedTemporaryFile(
        mode="wb",
        suffix=".wav",
        prefix=f"voxflow-{remote_item_id[:8]}-",
        dir=audio_cache_dir(),
        delete=False,
    ) as handle:
        handle.write(wav_data)
        return Path(handle.name)


def _broadcast_args(
    *,
    remote_item_id: str,
    created_at_ms: int,
    remote_audio_path: str,
    duration_ms: int,
    request: VoxFlowAdbDispatchRequest,
) -> list[str]:
    args = [
        "shell",
        "am",
        "broadcast",
        "-n",
        VOXFLOW_RECEIVER_COMPONENT,
        "-a",
        VOXFLOW_ADB_ACTION,
        "--es",
        "item_id",
        remote_item_id,
        "--el",
        "created_at_ms",
        str(created_at_ms),
        "--es",
        "audio_path",
        remote_audio_path,
        "--es",
        "raw_text",
        request.raw_text,
        "--es",
        "tts_text",
        request.tts_text,
        "--ei",
        "duration_ms",
        str(duration_ms),
        "--es",
        "voice",
        request.voice,
        "--ef",
        "speed",
        str(request.speed),
    ]
    if request.source_label:
        args.extend(["--es", "source_label", request.source_label])
    if request.language:
        args.extend(["--es", "language", request.language])
    if request.model:
        args.extend(["--es", "model", request.model])
    if request.reasoning_effort:
        args.extend(["--es", "reasoning_effort", request.reasoning_effort])
    return args


def _run_adb(adb_serial: str, args: list[str], *, error_prefix: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["adb", "-s", adb_serial, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(_render_adb_failure(error_prefix, completed))
    return completed


def _render_adb_failure(prefix: str, completed: subprocess.CompletedProcess[str]) -> str:
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    details = stderr or stdout or f"exit code {completed.returncode}"
    return f"{prefix}: {details}"


def _elapsed_ms(started: datetime) -> float:
    return (datetime.now(tz=UTC) - started).total_seconds() * 1000.0
