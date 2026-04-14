from __future__ import annotations

import json
import time

import numpy as np
from kokoro import KPipeline

TEXTS = {
    "short": "Hello. This is a short CPU benchmark for Kokoro text to speech.",
    "medium": (
        "Kokoro is an open-weight text to speech model. This benchmark measures only CPU "
        "execution. The goal is to estimate warm generation speed for a practical command-line "
        "pipeline that rewrites text and then synthesizes it."
    ),
    "long": (
        "Kokoro is an open-weight text to speech model with a small parameter count. "
        "In this benchmark, we force CPU execution and generate several sentences of speech to "
        "estimate real time performance. This is the performance that matters for a command-line "
        "pipeline, where the user pipes text into a transformer and then into text to speech. "
        "Cold start includes model and voice loading, while warm runs focus on steady state "
        "generation cost."
    ),
}


def generate_audio(pipe: KPipeline, text: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for result in pipe(text, voice="af_heart", speed=1.0, split_pattern=r"\n+"):
        audio = getattr(result, "audio", None)
        if audio is None and isinstance(result, tuple) and len(result) >= 3:
            audio = result[2]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def main() -> None:
    results: dict[str, object] = {}

    start_init = time.perf_counter()
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
    results["init_seconds"] = round(time.perf_counter() - start_init, 3)

    cold_start = time.perf_counter()
    cold_audio = generate_audio(pipe, TEXTS["medium"])
    cold_s = time.perf_counter() - cold_start
    cold_audio_s = len(cold_audio) / 24_000.0
    results["cold_medium"] = {
        "wall_seconds": round(cold_s, 3),
        "audio_seconds": round(cold_audio_s, 3),
        "rtf": round(cold_s / max(cold_audio_s, 1e-9), 3),
    }

    for name, text in TEXTS.items():
        _ = generate_audio(pipe, text)
        runs = []
        for _run in range(3):
            start = time.perf_counter()
            audio = generate_audio(pipe, text)
            wall = time.perf_counter() - start
            audio_seconds = len(audio) / 24_000.0
            runs.append(
                {
                    "wall_seconds": round(wall, 3),
                    "audio_seconds": round(audio_seconds, 3),
                    "rtf": round(wall / max(audio_seconds, 1e-9), 3),
                }
            )
        results[name] = {
            "avg_wall_seconds": round(sum(run["wall_seconds"] for run in runs) / len(runs), 3),
            "avg_audio_seconds": round(sum(run["audio_seconds"] for run in runs) / len(runs), 3),
            "avg_rtf": round(sum(run["rtf"] for run in runs) / len(runs), 3),
            "runs": runs,
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
