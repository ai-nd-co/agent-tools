from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from time import perf_counter

SUPPORTED_CUDA_TRACKS = ("cu130", "cu129", "cu128", "cu126", "cu124", "cu121")
CUDA_TRACK_MIN_VERSIONS = {
    "cu130": (13, 0),
    "cu129": (12, 9),
    "cu128": (12, 8),
    "cu126": (12, 6),
    "cu124": (12, 4),
    "cu121": (12, 1),
}
CUDA_VERSION_PATTERN = re.compile(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
SUPPORTED_INSTALL_PLATFORMS = {"win32", "linux"}
SUPPORTED_INSTALL_MACHINES = {"amd64", "x86_64"}


RunCallable = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class CudaProbeResult:
    ok: bool
    reason: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    device_count: int = 0
    device_name: str | None = None
    warning_text: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


@dataclass(frozen=True)
class DeviceResolution:
    requested_device: str
    resolved_device: str
    probe_ms: float = 0.0
    fallback_reason: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    device_name: str | None = None


def ensure_supported_cuda_install_platform(
    *,
    platform_name: str | None = None,
    machine: str | None = None,
) -> None:
    current_platform = platform_name or sys.platform
    current_machine = (machine or platform.machine() or "").lower()
    if current_platform not in SUPPORTED_INSTALL_PLATFORMS:
        raise RuntimeError(
            f"CUDA installation is supported only on {sorted(SUPPORTED_INSTALL_PLATFORMS)}, "
            f"got {current_platform!r}."
        )
    if current_machine not in SUPPORTED_INSTALL_MACHINES:
        raise RuntimeError(
            f"CUDA installation is supported only on {sorted(SUPPORTED_INSTALL_MACHINES)}, "
            f"got {current_machine!r}."
        )


def detect_nvidia_cuda_version(
    *,
    runner: RunCallable = subprocess.run,
) -> str:
    result = runner(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(
            "Could not run nvidia-smi. Install NVIDIA drivers first or pass --cuda-track "
            "explicitly."
        )
    match = CUDA_VERSION_PATTERN.search(output)
    if match is None:
        raise RuntimeError(
            "Could not determine CUDA runtime version from nvidia-smi output. "
            "Pass --cuda-track explicitly."
        )
    return match.group(1)


def select_cuda_track(cuda_version: str) -> str:
    parsed = _parse_version_tuple(cuda_version)
    for track in SUPPORTED_CUDA_TRACKS:
        if parsed >= CUDA_TRACK_MIN_VERSIONS[track]:
            return track
    raise RuntimeError(
        f"Detected CUDA runtime {cuda_version!r}, which is older than the minimum supported "
        "track cu121."
    )


def build_cuda_install_command(
    *,
    python_executable: str,
    cuda_track: str,
) -> list[str]:
    if cuda_track not in SUPPORTED_CUDA_TRACKS:
        raise ValueError(
            f"Unsupported CUDA track {cuda_track!r}. Expected one of {SUPPORTED_CUDA_TRACKS}."
        )
    return [
        python_executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--index-url",
        f"https://download.pytorch.org/whl/{cuda_track}",
        "torch",
    ]


def resolve_torch_device(requested_device: str) -> DeviceResolution:
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device {requested_device!r}. Expected one of ('auto', 'cpu', 'cuda')."
        )
    if requested_device == "cpu":
        return DeviceResolution(requested_device="cpu", resolved_device="cpu")

    probe_started = perf_counter()
    probe = probe_cuda_runtime()
    probe_ms = (perf_counter() - probe_started) * 1000.0

    if requested_device == "cuda":
        if not probe.ok:
            raise RuntimeError(
                "CUDA was requested but is not usable in this Python environment: "
                f"{probe.reason or 'unknown CUDA failure'}"
            )
        return DeviceResolution(
            requested_device="cuda",
            resolved_device="cuda",
            probe_ms=probe_ms,
            torch_version=probe.torch_version,
            torch_cuda_version=probe.torch_cuda_version,
            device_name=probe.device_name,
        )

    if probe.ok:
        return DeviceResolution(
            requested_device="auto",
            resolved_device="cuda",
            probe_ms=probe_ms,
            torch_version=probe.torch_version,
            torch_cuda_version=probe.torch_cuda_version,
            device_name=probe.device_name,
        )
    return DeviceResolution(
        requested_device="auto",
        resolved_device="cpu",
        probe_ms=probe_ms,
        fallback_reason=probe.reason,
        torch_version=probe.torch_version,
        torch_cuda_version=probe.torch_cuda_version,
        device_name=probe.device_name,
    )


def probe_cuda_runtime() -> CudaProbeResult:
    return _probe_cuda_runtime_cached()


def clear_cuda_probe_cache() -> None:
    _probe_cuda_runtime_cached.cache_clear()


@lru_cache(maxsize=1)
def _probe_cuda_runtime_cached() -> CudaProbeResult:
    warning_text: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    device_count = 0
    device_name: str | None = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import torch

            warning_text = _collect_warning_text(caught)
            torch_version = getattr(torch, "__version__", None)
            torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
            if not torch_cuda_version:
                return CudaProbeResult(
                    ok=False,
                    reason="Installed torch build does not include CUDA support.",
                    torch_version=torch_version,
                    torch_cuda_version=torch_cuda_version,
                    warning_text=warning_text,
                )
            if not torch.cuda.is_available():
                return CudaProbeResult(
                    ok=False,
                    reason="torch.cuda.is_available() returned false.",
                    torch_version=torch_version,
                    torch_cuda_version=torch_cuda_version,
                    warning_text=_collect_warning_text(caught),
                )
            device_count = int(torch.cuda.device_count())
            if device_count < 1:
                return CudaProbeResult(
                    ok=False,
                    reason="torch.cuda.device_count() returned zero.",
                    torch_version=torch_version,
                    torch_cuda_version=torch_cuda_version,
                    device_count=device_count,
                    warning_text=_collect_warning_text(caught),
                )
            device_name = torch.cuda.get_device_name(0)
            sample = torch.tensor([1.0], device="cuda")
            result = float((sample * 2.0).sum().item())
            if result <= 0.0:
                raise RuntimeError("CUDA validation produced an invalid test result.")
            return CudaProbeResult(
                ok=True,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                device_count=device_count,
                device_name=device_name,
                warning_text=_collect_warning_text(caught),
            )
    except Exception as exc:
        detail = warning_text or ""
        suffix = f" Warning: {detail}" if detail else ""
        return CudaProbeResult(
            ok=False,
            reason=f"{exc}{suffix}",
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            device_count=device_count,
            device_name=device_name,
            warning_text=warning_text,
        )


def _collect_warning_text(caught: list[warnings.WarningMessage]) -> str | None:
    messages: list[str] = []
    for warning in caught:
        text = str(warning.message).strip()
        if text:
            messages.append(text)
    if not messages:
        return None
    return " | ".join(messages)


def _parse_version_tuple(value: str) -> tuple[int, int]:
    parts = value.strip().split(".")
    if not parts or not parts[0].isdigit():
        raise RuntimeError(f"Invalid CUDA version string: {value!r}")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor)
