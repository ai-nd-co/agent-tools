from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import threading
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import metadata as importlib_metadata
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

CUDA_PROBE_TIMEOUT_SECONDS = 30.0
CUDA_PROBE_OUTPUT_LIMIT = 16_384
CUDA_PROBE_ENV_KEYS = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "PYTORCH_NVML_BASED_CUDA_CHECK",
    "SystemRoot",
    "TEMP",
    "TMP",
    "WINDIR",
)
_CUDA_PROBE_SCRIPT = r"""
import json
import warnings

payload = {
    "ok": False,
    "reason": None,
    "torch_version": None,
    "torch_cuda_version": None,
    "device_count": 0,
    "device_name": None,
    "warning_text": None,
}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import torch

    payload["torch_version"] = getattr(torch, "__version__", None)
    payload["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    if not payload["torch_cuda_version"]:
        payload["reason"] = "Installed torch build does not include CUDA support."
    elif not torch.cuda.is_available():
        payload["reason"] = "torch.cuda.is_available() returned false."
    else:
        payload["device_count"] = int(torch.cuda.device_count())
        if payload["device_count"] < 1:
            payload["reason"] = "torch.cuda.device_count() returned zero."
        else:
            payload["device_name"] = torch.cuda.get_device_name(0)
            sample = torch.tensor([1.0], device="cuda")
            result = float((sample * 2.0).sum().item())
            if result <= 0.0:
                raise RuntimeError("CUDA validation produced an invalid test result.")
            payload["ok"] = True
    messages = [str(item.message).strip() for item in caught if str(item.message).strip()]
    payload["warning_text"] = " | ".join(messages) or None
print(json.dumps(payload, sort_keys=True))
"""


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
class TtsRuntimeProbeResult:
    ok: bool
    cuda_ok: bool
    tts_stack_ok: bool
    python_executable: str
    python_version: str
    platform: str
    machine: str
    reason: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    torchvision_version: str | None = None
    torchaudio_version: str | None = None
    transformers_version: str | None = None
    kokoro_version: str | None = None
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
        "torchvision",
        "torchaudio",
    ]


def resolve_torch_device(requested_device: str) -> DeviceResolution:
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device {requested_device!r}. Expected one of ('auto', 'cpu', 'cuda')."
        )
    if requested_device == "cpu":
        torch_version = _distribution_version("torch")
        if _is_cuda_torch_build(torch_version):
            raise RuntimeError(
                "CPU synthesis requires a CPU-only PyTorch runtime because CUDA-enabled "
                "PyTorch can initialize the GPU driver while a TTS model loads. The active build "
                f"is {torch_version!r}. Create the locked AgentTools CPU runtime described in "
                "the README and use its agent-tools executable."
            )
        return DeviceResolution(
            requested_device="cpu",
            resolved_device="cpu",
            torch_version=torch_version,
        )

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
    fallback_torch_version = probe.torch_version or _distribution_version("torch")
    if _is_cuda_torch_build(fallback_torch_version):
        raise RuntimeError(
            "Automatic device selection cannot safely fall back to CPU because the active "
            f"PyTorch build is CUDA-enabled ({fallback_torch_version!r}) and the CUDA probe "
            "failed. Use a working CUDA runtime or the locked AgentTools CPU runtime "
            "described in the README."
        )
    return DeviceResolution(
        requested_device="auto",
        resolved_device="cpu",
        probe_ms=probe_ms,
        fallback_reason=probe.reason,
        torch_version=fallback_torch_version,
        torch_cuda_version=probe.torch_cuda_version,
        device_name=probe.device_name,
    )


def probe_cuda_runtime() -> CudaProbeResult:
    return _probe_cuda_runtime_cached()


def probe_tts_runtime() -> TtsRuntimeProbeResult:
    cuda_probe = probe_cuda_runtime()
    tts_probe = _probe_tts_stack()
    warning_text = _merge_detail_text(cuda_probe.warning_text, tts_probe.warning_text)
    reason = None
    if not cuda_probe.ok:
        reason = cuda_probe.reason
    elif not tts_probe.ok:
        reason = tts_probe.reason
    return TtsRuntimeProbeResult(
        ok=cuda_probe.ok and tts_probe.ok,
        cuda_ok=cuda_probe.ok,
        tts_stack_ok=tts_probe.ok,
        python_executable=sys.executable,
        python_version=platform.python_version(),
        platform=sys.platform,
        machine=platform.machine(),
        reason=reason,
        torch_version=cuda_probe.torch_version or _distribution_version("torch"),
        torch_cuda_version=cuda_probe.torch_cuda_version,
        torchvision_version=_distribution_version("torchvision"),
        torchaudio_version=_distribution_version("torchaudio"),
        transformers_version=_distribution_version("transformers"),
        kokoro_version=_distribution_version("kokoro"),
        device_count=cuda_probe.device_count,
        device_name=cuda_probe.device_name,
        warning_text=warning_text,
    )


def clear_cuda_probe_cache() -> None:
    _probe_cuda_runtime_cached.cache_clear()


@lru_cache(maxsize=1)
def _probe_cuda_runtime_cached() -> CudaProbeResult:
    try:
        result = _run_cuda_probe_subprocess()
    except _CudaProbeOutputLimit:
        return CudaProbeResult(
            ok=False,
            reason=f"CUDA probe exceeded the {CUDA_PROBE_OUTPUT_LIMIT}-byte output limit.",
        )
    except subprocess.TimeoutExpired:
        return CudaProbeResult(
            ok=False,
            reason=f"CUDA probe timed out after {CUDA_PROBE_TIMEOUT_SECONDS:g} seconds.",
        )
    except Exception as exc:
        return CudaProbeResult(
            ok=False,
            reason=f"CUDA probe could not start: {exc}",
        )
    stdout = _bounded_detail(result.stdout)
    stderr = _bounded_detail(result.stderr)
    if result.returncode != 0:
        detail = f" Detail: {stderr}" if stderr else ""
        return CudaProbeResult(
            ok=False,
            reason=f"CUDA probe subprocess exited with code {result.returncode}.{detail}",
            warning_text=stderr,
        )
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        return CudaProbeResult(
            ok=False,
            reason=f"CUDA probe returned invalid JSON: {exc}",
            warning_text=stderr,
        )
    if not isinstance(payload, dict):
        return CudaProbeResult(ok=False, reason="CUDA probe returned a non-object result.")
    return CudaProbeResult(
        ok=payload.get("ok") is True,
        reason=_optional_string(payload.get("reason")),
        torch_version=_optional_string(payload.get("torch_version")),
        torch_cuda_version=_optional_string(payload.get("torch_cuda_version")),
        device_count=_nonnegative_int(payload.get("device_count")),
        device_name=_optional_string(payload.get("device_name")),
        warning_text=_merge_detail_text(
            _optional_string(payload.get("warning_text")),
            stderr,
        ),
    )


def _run_cuda_probe_subprocess() -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key in CUDA_PROBE_ENV_KEYS
        if (value := os.environ.get(key)) is not None
    }
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        [sys.executable, "-I", "-X", "utf8", "-c", _CUDA_PROBE_SCRIPT],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()

    def drain(name: str) -> None:
        stream = getattr(process, name)
        if stream is None:
            return
        try:
            while chunk := stream.read(4096):
                remaining = CUDA_PROBE_OUTPUT_LIMIT - len(streams[name])
                if remaining > 0:
                    streams[name].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    overflow.set()
                    try:
                        process.kill()
                    except OSError:
                        pass
                    return
        except OSError:
            return

    readers = [
        threading.Thread(target=drain, args=(name,), daemon=True)
        for name in streams
    ]
    for reader in readers:
        reader.start()
    try:
        returncode = process.wait(timeout=CUDA_PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=5.0)
    if overflow.is_set():
        raise _CudaProbeOutputLimit
    return subprocess.CompletedProcess(
        args=process.args,
        returncode=returncode,
        stdout=streams["stdout"].decode("utf-8", errors="replace"),
        stderr=streams["stderr"].decode("utf-8", errors="replace"),
    )


def _bounded_detail(value: str | None) -> str:
    text = (value or "").strip()
    if len(text) <= CUDA_PROBE_OUTPUT_LIMIT:
        return text
    return f"{text[:CUDA_PROBE_OUTPUT_LIMIT]}...[truncated]"


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _is_cuda_torch_build(version: str | None) -> bool:
    if not version:
        return False
    return bool(re.search(r"\+cu[0-9]+(?:\.|$)", version, re.IGNORECASE))


class _CudaProbeOutputLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class _ImportProbeResult:
    ok: bool
    reason: str | None = None
    warning_text: str | None = None


def _probe_tts_stack() -> _ImportProbeResult:
    warning_text: str | None = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import torchaudio  # noqa: F401
            import torchvision  # noqa: F401
            import transformers  # noqa: F401
            from kokoro import KPipeline  # noqa: F401

            warning_text = _collect_warning_text(caught)
            return _ImportProbeResult(ok=True, warning_text=warning_text)
    except Exception as exc:
        detail = warning_text or ""
        suffix = f" Warning: {detail}" if detail else ""
        return _ImportProbeResult(
            ok=False,
            reason=f"{exc}{suffix}",
            warning_text=warning_text,
        )


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _collect_warning_text(caught: list[warnings.WarningMessage]) -> str | None:
    messages: list[str] = []
    for warning in caught:
        text = str(warning.message).strip()
        if text:
            messages.append(text)
    if not messages:
        return None
    return " | ".join(messages)


def _merge_detail_text(*parts: str | None) -> str | None:
    merged = [part for part in parts if part]
    if not merged:
        return None
    return " | ".join(merged)


def _parse_version_tuple(value: str) -> tuple[int, int]:
    parts = value.strip().split(".")
    if not parts or not parts[0].isdigit():
        raise RuntimeError(f"Invalid CUDA version string: {value!r}")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor)
