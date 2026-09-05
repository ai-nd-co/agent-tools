from __future__ import annotations

import pytest

from agent_tools.tts import (
    KOKORO_REPO_ID,
    SILERO_MODEL_ID,
    is_clearly_russian,
    normalize_tts_language,
    resolve_tts_engine,
    resolve_tts_options,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Привет, всё готово?", True),
        ("Файл README обновлён.", True),
        ("hello from Kokoro", False),
        ("status x = т", False),
        ("Український привіт", False),
    ],
)
def test_russian_detection_is_conservative(text: str, expected: bool) -> None:
    assert is_clearly_russian(text) is expected


def test_auto_engine_uses_explicit_russian_language() -> None:
    assert resolve_tts_engine("ASCII only", language="ru") == "silero"
    assert resolve_tts_engine("ASCII only", language="ru-RU") == "silero"
    assert normalize_tts_language("RU_ru") == "ru"


def test_explicit_engine_wins_over_text_detection() -> None:
    assert resolve_tts_engine("Привет, мир!", engine="kokoro") == "kokoro"
    assert resolve_tts_engine("Hello world", engine="silero") == "silero"


def test_engine_defaults_remain_truthful() -> None:
    english = resolve_tts_options("Hello world")
    russian = resolve_tts_options("Привет, мир!")

    assert (english.engine, english.model, english.voice, english.language) == (
        "kokoro",
        KOKORO_REPO_ID,
        "af_heart",
        "a",
    )
    assert (russian.engine, russian.model, russian.voice, russian.language) == (
        "silero",
        SILERO_MODEL_ID,
        "xenia",
        "ru",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"engine": "kokoro", "language": "ru"}, "Kokoro does not support Russian"),
        ({"engine": "silero", "language": "a"}, "supports only Russian"),
        ({"engine": "silero", "voice": "af_heart"}, "voice 'af_heart' is unsupported"),
        ({"engine": "kokoro", "voice": "xenia"}, "incompatible with Kokoro"),
        ({"engine": "silero", "speed": 1.2}, "supports speed 1.0 only"),
    ],
)
def test_incompatible_engine_options_fail_actionably(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_tts_options("Hello world", **kwargs)  # type: ignore[arg-type]
