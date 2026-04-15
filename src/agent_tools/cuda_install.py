from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass

from agent_tools.cuda_runtime import (
    SUPPORTED_CUDA_TRACKS,
    build_cuda_install_command,
    detect_nvidia_cuda_version,
    ensure_supported_cuda_install_platform,
    select_cuda_track,
)


@dataclass(frozen=True)
class InstallCudaResult:
    selected_track: str
    detected_cuda_version: str | None
    python_executable: str
    python_version: str
    platform: str
    machine: str
    install_command: tuple[str, ...]
    validation_payload: dict[str, object] | None


def install_cuda(
    *,
    cuda_track: str,
    validate: bool,
) -> InstallCudaResult:
    ensure_supported_cuda_install_platform()
    detected_cuda_version: str | None = None
    if cuda_track == "auto":
        detected_cuda_version = detect_nvidia_cuda_version()
        selected_track = select_cuda_track(detected_cuda_version)
    else:
        if cuda_track not in SUPPORTED_CUDA_TRACKS:
            raise ValueError(
                f"Unsupported CUDA track {cuda_track!r}. Expected one of "
                f"{SUPPORTED_CUDA_TRACKS} or 'auto'."
            )
        selected_track = cuda_track

    install_command = tuple(
        build_cuda_install_command(
            python_executable=sys.executable,
            cuda_track=selected_track,
        )
    )
    install_result = subprocess.run(install_command, check=False)
    if install_result.returncode != 0:
        raise RuntimeError(
            "PyTorch stack installation failed for the active Python interpreter "
            f"{sys.executable} (Python {platform.python_version()}, "
            f"{sys.platform}/{platform.machine()}) with exit code "
            f"{install_result.returncode}."
        )

    validation_payload: dict[str, object] | None = None
    if validate:
        validation_payload = run_cuda_validation()
        if not validation_payload.get("ok"):
            reason = validation_payload.get("reason") or "unknown CUDA validation failure"
            raise RuntimeError(
                f"CUDA installation completed, but validation failed: {reason}"
            )

    return InstallCudaResult(
        selected_track=selected_track,
        detected_cuda_version=detected_cuda_version,
        python_executable=sys.executable,
        python_version=platform.python_version(),
        platform=sys.platform,
        machine=platform.machine(),
        install_command=install_command,
        validation_payload=validation_payload,
    )


def run_cuda_validation() -> dict[str, object]:
    command = [sys.executable, "-m", "agent_tools", "cuda-self-check", "--json"]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    payload_text = (result.stdout or "").strip()
    if not payload_text:
        raise RuntimeError("CUDA validation produced no output.")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"CUDA validation returned invalid JSON: {payload_text!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("CUDA validation returned a non-object JSON payload.")
    if result.returncode != 0:
        payload.setdefault("ok", False)
    return {str(key): value for key, value in payload.items()}
