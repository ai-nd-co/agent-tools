from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np

from agent_tools.audio import concat_audio, wav_bytes
from agent_tools.cuda_runtime import DeviceResolution
from agent_tools.tts import ResolvedTtsOptions, TtsMetrics, TtsResult

SILERO_MODEL_VERSION = "v5_5_ru"
SILERO_MODEL_FILENAME = f"{SILERO_MODEL_VERSION}.pt"
SILERO_MODEL_URL = f"https://models.silero.ai/models/tts/ru/{SILERO_MODEL_FILENAME}"
SILERO_MODEL_SIZE = 145_420_684
SILERO_MODEL_SHA256 = "50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437"
SILERO_SAMPLE_RATE = 24_000
ENV_SILERO_CACHE_DIR = "AGENT_TOOLS_SILERO_CACHE_DIR"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_SYNTHESIS_CHARS = 500


def synthesize_silero_wav(
    text: str,
    *,
    options: ResolvedTtsOptions,
    device_resolution: DeviceResolution,
) -> TtsResult:
    total_started = perf_counter()
    import_started = perf_counter()
    torch = _load_torch()
    import_ms = (perf_counter() - import_started) * 1000.0

    pipeline_started = perf_counter()
    model_path = ensure_silero_model()
    model = _load_silero_model_cached(str(model_path), device_resolution.resolved_device)
    pipeline_init_ms = (perf_counter() - pipeline_started) * 1000.0

    text_chunks = _split_silero_text(text)
    audio_chunks: list[np.ndarray] = []
    generation_started = perf_counter()
    with torch.inference_mode():
        for chunk_number, text_chunk in enumerate(text_chunks, start=1):
            try:
                audio = model.apply_tts(
                    text=text_chunk,
                    speaker=options.voice,
                    sample_rate=SILERO_SAMPLE_RATE,
                    put_accent=True,
                    put_yo=True,
                    put_stress_homo=True,
                    put_yo_homo=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Silero v5_5_ru could not synthesize "
                    f"chunk {chunk_number} of {len(text_chunks)} "
                    f"({len(text_chunk)} characters): {exc}"
                ) from exc
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            samples = np.asarray(audio, dtype=np.float32).reshape(-1)
            if samples.size == 0:
                raise RuntimeError(
                    "Silero v5_5_ru did not produce any audio samples for "
                    f"chunk {chunk_number} of {len(text_chunks)}."
                )
            if not np.isfinite(samples).all():
                raise RuntimeError(
                    "Silero v5_5_ru produced non-finite audio samples for "
                    f"chunk {chunk_number} of {len(text_chunks)}."
                )
            audio_chunks.append(samples)
    generation_ms = (perf_counter() - generation_started) * 1000.0

    postprocess_started = perf_counter()
    wav = wav_bytes(concat_audio(audio_chunks), sample_rate=SILERO_SAMPLE_RATE)
    postprocess_ms = (perf_counter() - postprocess_started) * 1000.0
    total_ms = (perf_counter() - total_started) * 1000.0
    return TtsResult(
        wav=wav,
        sample_rate=SILERO_SAMPLE_RATE,
        chunks=len(audio_chunks),
        requested_device=device_resolution.requested_device,
        resolved_device=device_resolution.resolved_device,
        device_fallback_reason=device_resolution.fallback_reason,
        requested_engine=options.requested_engine,
        engine=options.engine,
        model=options.model,
        voice=options.voice,
        language=options.language,
        metrics=TtsMetrics(
            device_probe_ms=device_resolution.probe_ms,
            import_ms=import_ms,
            pipeline_init_ms=pipeline_init_ms,
            generation_ms=generation_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
            text_chars=len(text),
        ),
    )


def silero_cache_dir() -> Path:
    configured = os.environ.get(ENV_SILERO_CACHE_DIR)
    if configured and configured.strip():
        return Path(configured).expanduser()
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "ai-nd-co" / "agent-tools" / "models" / "silero"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
    return base / "agent-tools" / "models" / "silero"


def ensure_silero_model(*, cache_dir: Path | None = None) -> Path:
    target_dir = cache_dir or silero_cache_dir()
    target_path = target_dir / SILERO_MODEL_FILENAME
    if target_path.exists():
        _verify_model_file(target_path)
        return target_path

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Could not create the Silero model cache directory {target_dir}: {exc}"
        ) from exc

    temporary_path = target_dir / f".{SILERO_MODEL_FILENAME}.{uuid4().hex}.tmp"
    digest = hashlib.sha256()
    downloaded_size = 0
    try:
        request = Request(SILERO_MODEL_URL, headers={"User-Agent": "agent-tools-silero/1"})
        with urlopen(request, timeout=120) as response, temporary_path.open("xb") as output:
            while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                downloaded_size += len(chunk)
        _validate_model_identity(
            path=temporary_path,
            size=downloaded_size,
            sha256=digest.hexdigest(),
        )
        if target_path.exists():
            _verify_model_file(target_path)
            return target_path
        try:
            os.replace(temporary_path, target_path)
        except OSError:
            if not target_path.exists():
                raise
            _verify_model_file(target_path)
            return target_path
    except (OSError, URLError) as exc:
        raise RuntimeError(
            "Could not download the official Silero v5_5_ru model. "
            f"URL={SILERO_MODEL_URL} cache={target_path} error={exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return target_path


def _split_silero_text(text: str, *, max_chars: int = _MAX_SYNTHESIS_CHARS) -> list[str]:
    if max_chars <= 0:
        raise ValueError("Silero chunk size must be positive.")
    remaining = " ".join(text.split())
    chunks: list[str] = []
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        sentence_break = max(
            (window.rfind(f"{punctuation} ") + 1 for punctuation in ".!?…"),
            default=0,
        )
        break_at = sentence_break if sentence_break >= max_chars // 2 else 0
        if break_at == 0:
            break_at = window.rfind(" ", 0, max_chars + 1)
        if break_at <= 0:
            break_at = max_chars
        chunks.append(remaining[:break_at].strip())
        remaining = remaining[break_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _verify_model_file(path: Path) -> None:
    try:
        size = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            while chunk := model_file.read(_DOWNLOAD_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"Could not read the cached Silero model {path}: {exc}") from exc
    _validate_model_identity(path=path, size=size, sha256=digest.hexdigest())


def _validate_model_identity(*, path: Path, size: int, sha256: str) -> None:
    if size == SILERO_MODEL_SIZE and sha256 == SILERO_MODEL_SHA256:
        return
    raise RuntimeError(
        "Silero v5_5_ru model integrity check failed. "
        f"Expected size={SILERO_MODEL_SIZE} sha256={SILERO_MODEL_SHA256}; "
        f"got size={size} sha256={sha256} at {path}. Remove that file and retry."
    )


def _load_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Silero TTS requires the same working PyTorch runtime as Kokoro. "
            f"python={sys.executable} original_error={exc}"
        ) from exc
    return torch


@lru_cache(maxsize=2)
def _load_silero_model_cached(model_path: str, device: str) -> Any:
    torch = _load_torch()
    try:
        model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
        model.to(torch.device(device))
        if hasattr(model, "eval"):
            model.eval()
    except Exception as exc:
        raise RuntimeError(
            "The cached official Silero v5_5_ru model could not be loaded. "
            f"cache={model_path} device={device} error={exc}"
        ) from exc
    return model
