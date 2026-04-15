from __future__ import annotations

import json
import platform
import subprocess
import sys

import pytest

from agent_tools.cli import build_parser
from agent_tools.cuda_install import install_cuda, run_cuda_validation
from agent_tools.cuda_runtime import (
    CudaProbeResult,
    TtsRuntimeProbeResult,
    build_cuda_install_command,
    detect_nvidia_cuda_version,
    probe_tts_runtime,
    resolve_torch_device,
    select_cuda_track,
)


def test_select_cuda_track_prefers_latest_supported_runtime() -> None:
    assert select_cuda_track("13.1") == "cu130"
    assert select_cuda_track("12.9") == "cu129"
    assert select_cuda_track("12.8") == "cu128"
    assert select_cuda_track("12.7") == "cu126"
    assert select_cuda_track("12.4") == "cu124"
    assert select_cuda_track("12.1") == "cu121"


def test_detect_nvidia_cuda_version_parses_nvidia_smi_output() -> None:
    def fake_runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="Driver Version: 591.86    CUDA Version: 13.1\n",
            stderr="",
        )

    assert detect_nvidia_cuda_version(runner=fake_runner) == "13.1"


def test_build_cuda_install_command_uses_official_index() -> None:
    command = build_cuda_install_command(
        python_executable="python",
        cuda_track="cu130",
    )
    assert command == [
        "python",
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--index-url",
        "https://download.pytorch.org/whl/cu130",
        "torch",
        "torchvision",
        "torchaudio",
    ]


def test_cli_parser_accepts_install_cuda_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["install-cuda", "--cuda-track", "cu130", "--no-validate"])
    assert args.command == "install-cuda"
    assert args.cuda_track == "cu130"
    assert args.no_validate is True


def test_resolve_torch_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_tools.cuda_runtime as cuda_runtime

    monkeypatch.setattr(
        cuda_runtime,
        "probe_cuda_runtime",
        lambda: cuda_runtime.CudaProbeResult(
            ok=False,
            reason="torch.cuda.is_available() returned false.",
            torch_version="2.11.0",
        ),
    )

    result = resolve_torch_device("auto")
    assert result.requested_device == "auto"
    assert result.resolved_device == "cpu"
    assert result.fallback_reason == "torch.cuda.is_available() returned false."
    assert result.torch_version == "2.11.0"
    assert result.probe_ms >= 0.0


def test_resolve_torch_device_cuda_raises_on_failed_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_tools.cuda_runtime as cuda_runtime

    monkeypatch.setattr(
        cuda_runtime,
        "probe_cuda_runtime",
        lambda: cuda_runtime.CudaProbeResult(
            ok=False,
            reason="no kernel image is available for execution on the device",
        ),
    )

    with pytest.raises(RuntimeError, match="no kernel image"):
        resolve_torch_device("cuda")


def test_run_cuda_validation_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "ok": True,
        "cuda_ok": True,
        "tts_stack_ok": True,
        "python_executable": "python",
        "python_version": "3.11.9",
        "platform": "win32",
        "machine": "AMD64",
        "torch_version": "2.11.0+cu130",
        "torch_cuda_version": "13.0",
        "torchvision_version": "0.22.0+cu130",
        "torchaudio_version": "2.11.0+cu130",
        "transformers_version": "4.57.3",
        "kokoro_version": "0.9.4",
        "device_count": 1,
        "device_name": "RTX",
    }

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run_cuda_validation() == payload


def test_install_cuda_auto_selects_track_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_tools.cuda_install as cuda_install

    commands: list[list[str]] = []

    def fake_run(
        command: list[str] | tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(cuda_install, "ensure_supported_cuda_install_platform", lambda: None)
    monkeypatch.setattr(cuda_install, "detect_nvidia_cuda_version", lambda: "13.1")
    monkeypatch.setattr(cuda_install.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        cuda_install,
        "run_cuda_validation",
        lambda: {
            "ok": True,
            "cuda_ok": True,
            "tts_stack_ok": True,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": sys.platform,
            "machine": "AMD64",
            "device_name": "RTX",
        },
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = install_cuda(cuda_track="auto", validate=True)
    assert result.selected_track == "cu130"
    assert result.detected_cuda_version == "13.1"
    assert result.python_executable == sys.executable
    assert result.python_version == platform.python_version()
    assert commands == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
            "torch",
            "torchvision",
            "torchaudio",
        ]
    ]
    assert result.validation_payload == {
        "ok": True,
        "cuda_ok": True,
        "tts_stack_ok": True,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "machine": "AMD64",
        "device_name": "RTX",
    }


def test_probe_tts_runtime_combines_cuda_and_tts_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_tools.cuda_runtime as cuda_runtime

    monkeypatch.setattr(
        cuda_runtime,
        "probe_cuda_runtime",
        lambda: CudaProbeResult(
            ok=True,
            torch_version="2.11.0+cu130",
            torch_cuda_version="13.0",
            device_count=1,
            device_name="RTX",
        ),
    )
    monkeypatch.setattr(
        cuda_runtime,
        "_probe_tts_stack",
        lambda: cuda_runtime._ImportProbeResult(ok=True),
    )
    monkeypatch.setattr(cuda_runtime, "_distribution_version", lambda name: f"{name}-v")

    result = probe_tts_runtime()
    assert isinstance(result, TtsRuntimeProbeResult)
    assert result.ok is True
    assert result.cuda_ok is True
    assert result.tts_stack_ok is True
    assert result.torchvision_version == "torchvision-v"
    assert result.kokoro_version == "kokoro-v"


def test_probe_tts_runtime_reports_tts_stack_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_tools.cuda_runtime as cuda_runtime

    monkeypatch.setattr(
        cuda_runtime,
        "probe_cuda_runtime",
        lambda: CudaProbeResult(
            ok=True,
            torch_version="2.11.0+cu130",
            torch_cuda_version="13.0",
        ),
    )
    monkeypatch.setattr(
        cuda_runtime,
        "_probe_tts_stack",
        lambda: cuda_runtime._ImportProbeResult(
            ok=False,
            reason="operator torchvision::nms does not exist",
        ),
    )

    result = probe_tts_runtime()
    assert result.ok is False
    assert result.cuda_ok is True
    assert result.tts_stack_ok is False
    assert result.reason == "operator torchvision::nms does not exist"
