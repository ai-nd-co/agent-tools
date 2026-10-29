from __future__ import annotations

import re
import time
import unicodedata

import pytest

import agent_tools.tts as tts_module
from agent_tools.ru_speech import speakable_russian_text
from agent_tools.tts import TtsResult

# The shape of the message Silero read without its Latin words (Prosell group, 2026-10-29).
MIXED_RUSSIAN = (
    "Руслан, staging уже подключён к Airwallex: Sean проверил оплату Visa, CI зелёный, "
    "k3s обновлён, PR #41 смёржен. Подробности: https://github.com/ai-nd-co/agent-tools/pull/41."
)
_UNSAYABLE = re.compile(r"[A-Za-z0-9]")


def _unsayable(spoken: str) -> list[str]:
    """Every Latin letter or digit, in any script block, that Silero would skip."""
    return [
        character
        for character in spoken
        if _UNSAYABLE.match(character)
        or unicodedata.decimal(character, None) is not None
        or "LATIN" in unicodedata.name(character, "")
    ]


def test_mixed_russian_message_reads_every_latin_word() -> None:
    spoken = speakable_russian_text(MIXED_RUSSIAN)

    assert _UNSAYABLE.search(spoken) is None, spoken
    for reading in (
        "стейджинг", "эйрволлекс", "шон", "виза", "си-ай", "кей-три-эс",
        "пи-ар номер сорок один", "гитхаб точка ком",
    ):
        assert reading in spoken
    assert spoken.startswith("Руслан, ")
    assert spoken.endswith("сорок один.")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10% ошибок", "десять процентов ошибок"),
        ("2.5 секунды", "два точка пять секунды"),
        ("в 2026 году", "в две тысячи двадцать шесть году"),
        ("21000 строк", "двадцать одна тысяча строк"),
        ("1000000 записей", "один миллион записей"),
        ("код 007", "код ноль ноль семь"),
        ("файл tts.py", "файл ти-ти-эс точка пи"),
        ("пишите alex@example.com", "пишите алекс эт эксампл точка ком"),
        ("класс HTTPServer", "класс эйч-ти-ти-пи сервер"),
        ("поле user_id", "поле юзер ай-ди"),
        ("Make it", "мейк ит"),
        ("naïve café", "нейв кейф"),
        ("Привет, мир.", "Привет, мир."),
        ("10-20 минут", "десять-двадцать минут"),
        ("код C# и F#", "код си шарп и эф шарп"),
        ("https://x.io/view?id=41#top", "экс точка ио слэш вью ай-ди сорок один"),
        ("Œuvre ４１", "оеувр сорок один"),
    ],
)
def test_rewrite_examples(text: str, expected: str) -> None:
    assert speakable_russian_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "x" * 40,
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789",
        "https://example.com/a_b/c-d?q=1#frag www.Example.org/x",
        "camelCaseIdentifier snake_case_name kebab-case-name dotted.module.path v2.5.1-rc1",
        "ÀÉÎÕÜ ßtraße Œuvre",
        "13 карт, 9999999999999 и 123456789012",
        "Œ Ø ẞ Æ Ł Þ ŋ ＡＢＣ ４１ ٤٢ ꜰ ʀ",
    ],
)
def test_no_latin_letter_or_digit_survives(text: str) -> None:
    spoken = speakable_russian_text(text)

    assert not _unsayable(spoken), spoken
    assert spoken.strip()


def test_a_long_token_without_an_at_sign_stays_linear() -> None:
    started = time.perf_counter()
    speakable_russian_text("a" * 100_000 + " и " + "1" * 50_000)

    assert time.perf_counter() - started < 5.0


def test_silero_input_is_rewritten_and_kokoro_input_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}

    class RecordingEngine:
        def __init__(self, name: str) -> None:
            self.name = name

        def synthesize(self, text: str, **_kwargs: object) -> TtsResult:
            seen[self.name] = text
            return TtsResult(wav=b"", sample_rate=24_000, chunks=1)

    monkeypatch.setattr(
        tts_module,
        "_SYNTHESIS_ENGINES",
        {"kokoro": RecordingEngine("kokoro"), "silero": RecordingEngine("silero")},
    )
    english = "Deploy PR #41 to staging via CI on k3s at 15:30, 10% faster."

    tts_module.synthesize_wav(MIXED_RUSSIAN, backend="local", device="cpu")
    tts_module.synthesize_wav(english, backend="local", device="cpu")

    assert _UNSAYABLE.search(seen["silero"]) is None
    assert "эйрволлекс" in seen["silero"]
    assert seen["kokoro"] == english


@pytest.mark.parametrize(
    ("text", "expect_rewritten"),
    [(MIXED_RUSSIAN, True), ("Номер " + "1" * 1000, False)],
)
def test_remote_gets_the_reading_unless_it_exceeds_the_server_limit(
    monkeypatch: pytest.MonkeyPatch, text: str, expect_rewritten: bool,
) -> None:
    import agent_tools.remote_tts as remote_module

    sent: list[str] = []

    def fake_remote(remote_text: str, **_kwargs: object) -> TtsResult:
        sent.append(remote_text)
        return TtsResult(wav=b"", sample_rate=24_000, chunks=1, engine="silero")

    monkeypatch.setattr(remote_module, "load_remote_tts_settings", lambda: object())
    monkeypatch.setattr(remote_module, "synthesize_remote_wav", fake_remote)

    tts_module.synthesize_wav(text, backend="remote", device="cpu")

    assert sent == [speakable_russian_text(text) if expect_rewritten else text]
