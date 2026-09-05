from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from math import isfinite
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from agent_tools.audio import KOKORO_SAMPLE_RATE, concat_audio, wav_bytes
from agent_tools.cuda_runtime import DeviceResolution, resolve_torch_device

SUPPORTED_KOKORO_LANGUAGES = ("a", "b", "e", "f", "h", "i", "j", "p", "z")
SUPPORTED_LANGUAGES = SUPPORTED_KOKORO_LANGUAGES + ("ru", "ru-RU")
SUPPORTED_TTS_ENGINES = ("auto", "kokoro", "silero")
SILERO_VOICES = ("aidar", "baya", "kseniya", "xenia", "eugene")
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
SILERO_MODEL_ID = "snakers4/silero-models:v5_5_ru"
DEFAULT_KOKORO_VOICE = "af_heart"
DEFAULT_SILERO_VOICE = "xenia"
RUSSIAN_LANGUAGE = "ru"
_RUSSIAN_CYRILLIC = frozenset("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")


@dataclass(frozen=True)
class TtsResult:
    wav: bytes
    sample_rate: int
    chunks: int
    requested_device: str = "auto"
    resolved_device: str = "cpu"
    device_fallback_reason: str | None = None
    requested_engine: str = "auto"
    engine: str = "kokoro"
    model: str | None = KOKORO_REPO_ID
    voice: str | None = None
    language: str | None = None
    metrics: TtsMetrics = field(default_factory=lambda: TtsMetrics())


@dataclass(frozen=True)
class TtsMetrics:
    device_probe_ms: float = 0.0
    import_ms: float = 0.0
    pipeline_init_ms: float = 0.0
    generation_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    text_chars: int = 0


@dataclass(frozen=True)
class ResolvedTtsOptions:
    requested_engine: str
    engine: str
    model: str
    voice: str
    language: str
    speed: float


class SynthesisEngine(Protocol):
    def synthesize(
        self,
        text: str,
        *,
        options: ResolvedTtsOptions,
        device_resolution: DeviceResolution,
    ) -> TtsResult: ...


class KokoroSynthesisEngine:
    def synthesize(
        self,
        text: str,
        *,
        options: ResolvedTtsOptions,
        device_resolution: DeviceResolution,
    ) -> TtsResult:
        return _synthesize_kokoro_wav(
            text,
            options=options,
            device_resolution=device_resolution,
        )


class SileroSynthesisEngine:
    def synthesize(
        self,
        text: str,
        *,
        options: ResolvedTtsOptions,
        device_resolution: DeviceResolution,
    ) -> TtsResult:
        from agent_tools.silero_tts import synthesize_silero_wav

        return synthesize_silero_wav(
            text,
            options=options,
            device_resolution=device_resolution,
        )


_SYNTHESIS_ENGINES: dict[str, SynthesisEngine] = {
    "kokoro": KokoroSynthesisEngine(),
    "silero": SileroSynthesisEngine(),
}


def synthesize_wav(
    text: str,
    *,
    voice: str | None = None,
    language: str | None = None,
    speed: float = 1.0,
    device: str = "auto",
    engine: str = "auto",
) -> TtsResult:
    if not text.strip():
        raise ValueError("Input text is empty.")
    options = resolve_tts_options(
        text,
        engine=engine,
        voice=voice,
        language=language,
        speed=speed,
    )
    device_resolution = resolve_torch_device(device)
    return _SYNTHESIS_ENGINES[options.engine].synthesize(
        text,
        options=options,
        device_resolution=device_resolution,
    )


def resolve_tts_engine(text: str, *, engine: str = "auto", language: str | None = None) -> str:
    requested_engine = engine.strip().lower()
    if requested_engine not in SUPPORTED_TTS_ENGINES:
        raise ValueError(
            f"Unsupported TTS engine {engine!r}. Expected one of {SUPPORTED_TTS_ENGINES}."
        )
    normalized_language = normalize_tts_language(language)
    if requested_engine != "auto":
        return requested_engine
    if normalized_language == RUSSIAN_LANGUAGE:
        return "silero"
    if is_clearly_russian(text):
        return "silero"
    return "kokoro"


def resolve_tts_options(
    text: str,
    *,
    engine: str = "auto",
    voice: str | None = None,
    language: str | None = None,
    speed: float = 1.0,
) -> ResolvedTtsOptions:
    requested_engine = engine.strip().lower()
    resolved_engine = resolve_tts_engine(text, engine=requested_engine, language=language)
    normalized_language = normalize_tts_language(language)
    normalized_voice = voice.strip().lower() if voice is not None else None
    if normalized_voice == "":
        raise ValueError("Voice cannot be empty.")
    if not isfinite(speed) or speed <= 0:
        raise ValueError(f"TTS speed must be a positive finite number, got {speed!r}.")

    if resolved_engine == "silero":
        if normalized_language not in {None, RUSSIAN_LANGUAGE}:
            raise ValueError(
                "Silero v5_5_ru supports only Russian language 'ru'. "
                f"Received language {language!r}; use --tts-engine kokoro for Kokoro languages."
            )
        effective_voice = normalized_voice or DEFAULT_SILERO_VOICE
        if effective_voice not in SILERO_VOICES:
            raise ValueError(
                f"Silero v5_5_ru voice {effective_voice!r} is unsupported. "
                f"Choose one of {SILERO_VOICES}, or use --tts-engine kokoro for Kokoro voices."
            )
        if speed != 1.0:
            raise ValueError(
                "Silero v5_5_ru supports speed 1.0 only in AgentTools; changing speed would "
                "require resampling, which AgentTools does not do silently."
            )
        return ResolvedTtsOptions(
            requested_engine=requested_engine,
            engine="silero",
            model=SILERO_MODEL_ID,
            voice=effective_voice,
            language=RUSSIAN_LANGUAGE,
            speed=speed,
        )

    if normalized_language == RUSSIAN_LANGUAGE:
        raise ValueError(
            "Kokoro does not support Russian language 'ru'. Use --tts-engine silero or auto."
        )
    effective_voice = normalized_voice or DEFAULT_KOKORO_VOICE
    if effective_voice in SILERO_VOICES:
        raise ValueError(
            f"Silero voice {effective_voice!r} is incompatible with Kokoro. "
            "Use --tts-engine silero with language 'ru', or choose a Kokoro voice."
        )
    lang_code = normalized_language or infer_language_from_voice(effective_voice)
    if lang_code not in SUPPORTED_KOKORO_LANGUAGES:
        raise ValueError(f"Unsupported Kokoro language code: {lang_code}")
    return ResolvedTtsOptions(
        requested_engine=requested_engine,
        engine="kokoro",
        model=KOKORO_REPO_ID,
        voice=effective_voice,
        language=lang_code,
        speed=speed,
    )


def normalize_tts_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip()
    if not normalized:
        raise ValueError("Language cannot be empty.")
    lowered = normalized.lower().replace("_", "-")
    if lowered in {"ru", "ru-ru"}:
        return RUSSIAN_LANGUAGE
    if lowered in SUPPORTED_KOKORO_LANGUAGES:
        return lowered
    raise ValueError(
        f"Unsupported TTS language {language!r}. Expected a Kokoro language code or 'ru'."
    )


def is_clearly_russian(text: str) -> bool:
    letters = [character.lower() for character in text if character.isalpha()]
    if not letters:
        return False
    russian_letters = [character for character in letters if character in _RUSSIAN_CYRILLIC]
    cyrillic_letters = [
        character for character in letters if "\u0400" <= character <= "\u052f"
    ]
    if len(russian_letters) < 4 or len(russian_letters) / len(letters) < 0.5:
        return False
    return len(russian_letters) == len(cyrillic_letters)


def _synthesize_kokoro_wav(
    text: str,
    *,
    options: ResolvedTtsOptions,
    device_resolution: DeviceResolution,
) -> TtsResult:
    total_started = perf_counter()
    lang_code = options.language

    pipeline_device = device_resolution.resolved_device
    import_started = perf_counter()
    KPipeline = _load_kokoro_pipeline()
    import_ms = (perf_counter() - import_started) * 1000.0

    pipeline_started = perf_counter()
    pipeline = _create_kokoro_pipeline(
        KPipeline,
        lang_code=lang_code,
        device=pipeline_device,
    )
    pipeline_init_ms = (perf_counter() - pipeline_started) * 1000.0
    audio_chunks: list[np.ndarray] = []

    generation_started = perf_counter()
    for result in pipeline(
        text,
        voice=options.voice,
        speed=options.speed,
        split_pattern=r"\n+",
    ):
        chunk = _extract_audio_chunk(result)
        if chunk is not None:
            audio_chunks.append(chunk)
    generation_ms = (perf_counter() - generation_started) * 1000.0

    if not audio_chunks:
        raise RuntimeError("Kokoro did not produce any audio chunks.")

    postprocess_started = perf_counter()
    merged = concat_audio(audio_chunks)
    wav = wav_bytes(merged, sample_rate=KOKORO_SAMPLE_RATE)
    postprocess_ms = (perf_counter() - postprocess_started) * 1000.0
    total_ms = (perf_counter() - total_started) * 1000.0
    return TtsResult(
        wav=wav,
        sample_rate=KOKORO_SAMPLE_RATE,
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


def infer_language_from_voice(voice: str) -> str:
    normalized = voice.strip().lower()
    if not normalized:
        raise ValueError("Voice cannot be empty.")
    lang_code = normalized[0]
    if lang_code not in SUPPORTED_KOKORO_LANGUAGES:
        raise ValueError(f"Could not infer Kokoro language from voice: {voice}")
    return lang_code


def _extract_audio_chunk(result: Any) -> np.ndarray | None:
    audio = getattr(result, "audio", None)
    if audio is None and isinstance(result, tuple) and len(result) >= 3:
        audio = result[2]
    if audio is None:
        return None
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def _load_kokoro_pipeline() -> Any:
    try:
        from kokoro import KPipeline
    except Exception as exc:
        raise RuntimeError(
            "Kokoro TTS could not be imported in this Python environment. "
            "The active PyTorch stack is likely missing or mismatched for this interpreter. "
            "Re-run `agent-tools install-cuda` with the same Python executable and retry. "
            f"python={sys.executable} python_version={platform.python_version()} "
            f"original_error={exc}"
        ) from exc
    return KPipeline


def _create_kokoro_pipeline(KPipeline: Any, *, lang_code: str, device: str) -> Any:
    if lang_code not in {"a", "b"} or _has_spacy_english_model():
        return KPipeline(lang_code=lang_code, repo_id=KOKORO_REPO_ID, device=device)

    try:
        from misaki.espeak import EspeakG2P  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeError(
            "Kokoro English speech needs either the en_core_web_sm package or the bundled "
            f"Misaki eSpeak fallback. original_error={exc}"
        ) from exc

    pipeline = KPipeline(lang_code="e", repo_id=KOKORO_REPO_ID, device=device)
    pipeline.g2p = EspeakG2P(language="en-gb" if lang_code == "b" else "en-us")
    return pipeline


def _has_spacy_english_model() -> bool:
    try:
        importlib_metadata.distribution("en-core-web-sm")
    except importlib_metadata.PackageNotFoundError:
        return False
    return True
