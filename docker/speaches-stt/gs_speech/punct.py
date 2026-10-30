"""Optional punctuation pass by a local LLM, for transcripts Whisper returned without punctuation.

Whisper large-v3 often writes an English piece of a voice note with no punctuation at all, and
Speaches decodes each piece on its own, so a long note comes back partly unpunctuated. This pass
asks an OpenAI-compatible chat model to add punctuation and capitals, and keeps its answer only if
every word is unchanged.

When it acts: POST /v1/audio/transcriptions with the header `X-GS-Punctuate: llm`, a complete 200
JSON answer (response_format json or verbose_json) of at most GS_PUNCT_MAX_BYTES, and either a
stretch of at least GS_PUNCT_MIN_RUN (40) words with no punctuation mark or at least
GS_PUNCT_PARAGRAPH_MIN_WORDS (80; 0 = off) words in all, which the model also splits into
paragraphs. Anything else passes through untouched, byte for byte.

The guard: the answer and the transcript are split on whitespace; there must be the same number of
tokens, and each pair may differ only in letter case and in sentence marks (. , ? ! ; : …) at the
token's two ends. Anything inside a token must be identical, so `we'll`/`well`, `1.5`/`15` or
`re-sign`/`resign` are refused. Then `text`, each segment's `text` and, when they line up one to
one, the `words` entries take the punctuated tokens; the top-level `text` also takes the model's
paragraph breaks (a blank line between paragraphs). Timestamps and every other field are unchanged.
Otherwise, and on any error, timeout or busy model, the original answer is sent. The header
`X-GS-Punctuated` says what happened: applied, unchanged, rejected, skipped, busy, timeout or error.
The log line has counts and timings only, never text.

Correction mode: the extra header `X-GS-Correct: context` (with `X-GS-Punctuate: llm`) asks the
model also to fix misheard words, names above all, using the chat's recent messages from
`X-GS-Context` (standard base64 of UTF-8 text, at most 4,000 characters; when invalid it is ignored,
when longer only its end is kept, and it is never logged). The pass then runs on any transcript of
at least GS_PUNCT_CORRECT_MIN_WORDS (3) words. The word guard is replaced by a looser one against
rewriting: the answer is aligned with the transcript (difflib on lower-case tokens without edge
punctuation; tokens with a digit keep everything but sentence marks), and at most max(2, 10%) of
the transcript's tokens may be dropped and as many added, and the answer must keep at least half
as many tokens. A correction replaces only the top-level
`text` (`X-GS-Punctuated: corrected`); segments and words stay as Whisper wrote them, since their
timings belong to Whisper's words. An answer that changed no word is used as in the normal pass
(applied or unchanged). It also refuses any change to a number (digits compared exactly, so a sign,
currency or percent counts) or to a negation, an insertion of more than one token, a deletion of
anything but a repeat or a sound, and a replacement of more than 2 tokens on either side. It does
not police person names: a context can make the model swap one name for another. A refused answer
sends the original (rejected); its punctuation is not reused, as it can put a stop mid-sentence.
The log line has counts and the refusal's kind only.

Hesitation sounds: after the guard, correction mode removes hesitation sounds ("um", "ммм"; SOUNDS)
from the top-level `text` in code, not by the model, and only on positive evidence: those of the
answer's own "language" (verbose_json; none without it), not next to a number, with a capital
only as "Um," at a sentence's start, and not at all in a note whose transcript or text quotes
something, talks about sounds or words, or writes a sound as a name ("UM", "Ask Um"). Word fillers
("you know", "like", "ну") are kept: punctuation cannot tell them from their meaning. That applies
to corrected, applied and unchanged answers and to Whisper's text after a refusal; segments and
words keep the sounds, and the outcome header stays as it was. The guard lets a correction drop a
run of that language's sounds (except in such a note) or an exact repeat. The log line counts the
removed sounds.

It wraps Admission from outside, so the LLM call happens after the Whisper permit is released: the
inner app runs as its own task, and once its answer is complete it has GS_PUNCT_APP_GRACE_S (5) to
return before it is cancelled (also if the request itself is cancelled). One LLM call at a time
(the model has one slot), at most GS_PUNCT_MAX_WAITING (2) waiting behind it, each bounded by
GS_PUNCT_TIMEOUT_S (60) including the wait. A client that disconnects meanwhile cancels its pass.

Env: GS_PUNCT_URL (chat completions URL; unset = off), GS_PUNCT_MODEL, GS_PUNCT_TIMEOUT_S,
GS_PUNCT_MIN_RUN, GS_PUNCT_PARAGRAPH_MIN_WORDS, GS_PUNCT_MAX_WORDS (3000), GS_PUNCT_MAX_BYTES
(8 MiB), GS_PUNCT_MAX_WAITING, GS_PUNCT_APP_GRACE_S, GS_PUNCT_CORRECT_MIN_WORDS (0 = mode off).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import difflib
import json
import logging
import math
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable, Mapping

logger = logging.getLogger("gs_speech.punct")

PATH = "/v1/audio/transcriptions"
OPT_IN = (b"x-gs-punctuate", b"llm")
EDGE = ".,?!;:…"
MARK = re.compile(r"[.,?!;:…]")
# Tested with Qwen3.8-27B IQ3_XXS on 20 of Davron's notes of 80+ words: 19 kept every word, 1 was
# refused by the guard; punctuation as good as the wording without paragraphs, 1-8 paragraphs a
# note. Asking also not to "join or split" words made the model return the text unpunctuated, so
# test any change to this wording.
SYSTEM = (
    "You restore punctuation and capitalization in speech transcripts and split them into "
    "paragraphs. Return the same text with punctuation (full stops, commas, question marks) and "
    "capital letters added, and start a new paragraph, separated by a blank line, wherever the "
    "topic or thought changes. Do not add, remove, replace, reorder or correct any word, and keep "
    "filler words and repeats exactly. Output only the text."
)
PARAGRAPH = re.compile(r"\n\s*\n")

CORRECT_OPT_IN = (b"x-gs-correct", b"context")
CONTEXT_HEADER = b"x-gs-context"
MAX_CONTEXT_CHARS = 4000
# 4,000 characters of UTF-8 are at most 16,000 bytes, 21,336 in base64; a longer header is ignored
# without decoding it.
MAX_CONTEXT_HEADER = 24 * 1024
# Tested with Qwen3.8-27B IQ3_XXS on 40 of Davron's notes (31 English, 9 Russian), each with his 3
# previous notes as context: 17 notes changed, by at most 6 tokens (of 324): names (Codex, Claude,
# k3s, Anthropic), repeated words dropped, a missing article or pronoun, hyphenation. No rephrasing.
# Without the context two of the name fixes were missed.
SYSTEM_CORRECT = (
    "You correct transcripts of voice notes made by speech recognition, which often mishears "
    "names and technical terms. Fix misheard words, punctuation and capitalization, and split the "
    "text into paragraphs, separated by a blank line, wherever the topic or thought changes. The "
    "recent messages of the chat, when given, show the names and terms the speaker is likely to "
    "use; never copy anything else from them. Names that come up often: Codex, Claude, Claude "
    "Code, claude-tools, k3s, Whisper, vox-flow, agent-hub, Telegram, Asana, Terraform, GPU. Keep "
    "the speaker's own words, word order and language: do not rephrase, summarise, translate, "
    "explain, add or remove anything, and keep filler words and repeats. Output only the "
    "corrected transcript."
)
# The guard against rewriting. On the same notes the corrections changed at most 5.8% of the tokens
# of a note of 50+ words (2 of a 16-word note); prompts asking to clean up, paraphrase or summarise
# changed 11.8-100% (33 of 36 answers; the other 3 changed 1 token of a 14-word note). Short notes
# may change 2 tokens, as one name can be two words ("cloud code"). On top of the share: no number
# or negation may change, an insertion is one token at most, a deletion only drops a repeat or a
# sound, and one replacement spans at most MAX_REPLACE tokens on either side. Person names are not
# policed: the owner chose free correction, and a context can talk the model into another name.
CORRECT_MAX_SHARE = 0.10
CORRECT_MIN_ALLOWED = 2
CORRECT_MIN_COUNT_RATIO = 0.5
# Words that may flip a note's meaning: a correction that drops, adds or replaces one is refused
# (the LLM turned "fifteen" into "fifty" and "don't deploy" into "deploy" when the context said so).


def _russian_numbers() -> list[str]:
    """Russian number words in their cases (a form may also be another word: семью, family)."""
    words = """ноль нуль нуля ноля нулю нолю нулём нолём нуле ноле нули ноли нулей нолей
    один одного одному одним одном одна одной одною одну одно
    одни одних одним одними два двух двум двумя две три трёх трех трём трем тремя четыре четырёх
    четырех четырём четырем четырьмя сорок сорока девяносто девяноста сто ста сотни сотен двести
    двухсот двумстам двумястами двухстах триста трёхсот трехсот тремстам тремястами трёхстах
    трехстах четыреста четырёхсот четырехсот четырёмстам четыремстам четырьмястами четырёхстах
    четырехстах полтора полторы полутора""".split()
    for stem in "пят шест сем девят десят".split():  # пять, пяти, пятью
        words += [stem + "ь", stem + "и", stem + "ью"]
    words += ["восемь", "восьми", "восемью", "восьмью", "восьмьюдесятью", "восьмьюстами"]
    for stem in "один две три четыр пят шест сем восем девят".split():  # одиннадцать ...
        words += [stem + "надцать", stem + "надцати", stem + "надцатью"]
    for stem in "двадцат тридцат".split():
        words += [stem + "ь", stem + "и", stem + "ью"]
    for nom, gen, ins in (  # пятьдесят, пятидесяти, пятьюдесятью; and the hundreds
        ("пятьдесят", "пятидесяти", "пятьюдесятью"),
        ("шестьдесят", "шестидесяти", "шестьюдесятью"),
        ("семьдесят", "семидесяти", "семьюдесятью"),
        ("восемьдесят", "восьмидесяти", "восемьюдесятью"),
        ("пятьсот", "пятисот", "пятьюстами"),
        ("шестьсот", "шестисот", "шестьюстами"),
        ("семьсот", "семисот", "семьюстами"),
        ("восемьсот", "восьмисот", "восемьюстами"),
        ("девятьсот", "девятисот", "девятьюстами"),
    ):
        words += [nom, gen, ins]
    for stem in "пятист шестист семист восьмист девятист".split():
        words += [stem + "ам", stem + "ах"]
    for stem in ("тысяч", "миллион", "миллиард"):
        ends = "а и е у ей ею ью ам ами ах" if stem == "тысяч" else "а у ом е ы ов ам ами ах"
        words += [stem] + [stem + e for e in ends.split()]
    return words


NUMBER_WORDS = frozenset(
    """zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen
    fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty
    ninety hundred hundreds thousand thousands million millions billion billions""".split()
    + [w.replace("ё", "е") for w in _russian_numbers()]  # compared with ё folded to е
)
NEGATIONS = frozenset("not no never nor none nothing cannot не нет ни никогда ничего".split())
# Sounds, which correction mode removes in code (strip_sounds): hesitation sounds only, on
# positive evidence that one is no word, unit, letter or name, and only in the note's own language
# (SOUNDS_BY_LANGUAGE: "um" is a Portuguese word). One goes in lower case, or with a capital only
# at a sentence's start and with its own comma or ellipsis ("Um, we merge."), bare or in round or
# square brackets, with sentence marks after it; in capitals ("UM"), with a capital otherwise ("Ask
# Um", "Dr. Um", "Um we merge"), with another symbol ("@um") or next to a number ("five um", "half
# um", "½ um") it stays, and a note about sounds or with quotes is left alone (META_STEMS).
# Left out: "er" (ER), "mm" (5 mm), "мм" (millimetres), "ah", "eh", "umm" (Umm Al Quwain), "э"
# and "эм" (the letters Э and М: "Нажми эм"), and every word filler: "you know" ("if you know, you
# know"), "I mean" ("fifty, I mean, fifteen"), "like" ("like, a hundred"), типа, как бы, вот, ну,
# короче, so, well, just, actually.
SOUNDS_BY_LANGUAGE = {
    "en": frozenset("uh um uhm erm hmm".split()),
    "ru": frozenset({"ммм"}),  # "м" three times or more; "мм" is millimetres
}
SOUNDS = frozenset().union(*SOUNDS_BY_LANGUAGE.values())
LANGUAGES = {"en": "en", "english": "en", "ru": "ru", "russian": "ru"}  # Speaches' "language"
# Abbreviations whose stop ends no sentence ("Dr. Um" is a name).
ABBREVIATIONS = frozenset("dr. mr. mrs. ms. prof. st. vs. etc. e.g. i.e. г. т.е.".split())
SOUND_SHAPE = re.compile(r"([(\[]*)([^\W\d_]+)([)\]]*)([,.?!…;:]*)")
PAIRS = {"(": ")", "[": "]"}
# A note that talks about sounds or words, or quotes something, is left alone as a whole: there a
# sound may be the subject ("Replace um with uh", "звуке эм") or a literal ("'um'"). It does so on
# any quote or backtick character, an apostrophe that is not inside a word, or a word starting with
# one of META_STEMS. 7 of the 40 test notes match (word, replace, quote, transcribe, quote marks),
# which kept 1 of their sounds.
META_STEMS = tuple(
    "sound word letter filler replac transcri spell quot звук слов букв символ междомет замен "
    "транскри цитат".split()
)
META_MARKS = re.compile(r'["“”«»‘`]|(?<![^\W_])[\'’]|[\'’](?![^\W_])')
FRACTIONS = frozenset(
    "half halves quarter quarters third thirds dozen dozens половина половину половины половиной "
    "пол четверть четверти треть трети".split()
)
# Never in a note, so a text past these is passed on unstripped: the longest of the 40 test notes
# has 1,897 characters, and 3,000 words (GS_PUNCT_MAX_WORDS) of speech are about 20,000.
STRIP_MAX_CHARS = 50_000
STRIP_MAX_SPACE = 1_000
APOSTROPHES = str.maketrans(dict.fromkeys("\u2018\u2019\u02bc\u02b9\u00b4\u0060\u2032\uff07", "'"))
HYPHENS = re.compile("[-\u2010-\u2015\u2212\ufe63\uff0d]")
DIGIT_NAMES = frozenset({"k3s"})  # names in SYSTEM_CORRECT with a digit: a fix to one is no number
MAX_REPLACE = 2  # tokens on either side of one replacement ("cloud code" -> "Claude Code")
ECHOED_TAGS = re.compile(r"\A\s*<transcript>|</transcript>\s*\Z")  # if the model repeats them

Complete = Callable[[str, int, float], Awaitable[str]]


def core(token: str) -> str:
    """What must not change: the token without sentence marks at its ends, in lower case."""
    return token.strip(EDGE).lower()


def longest_unpunctuated(tokens: list[str]) -> int:
    best = run = 0
    for t in tokens:
        run += 1
        if MARK.search(t):
            best, run = max(best, run), 0
    return max(best, run)


def same_words(src: list[str], got: list[str]) -> bool:
    return len(src) == len(got) and all(
        core(a) == core(b) and a.strip(EDGE) != "" and b.strip(EDGE) != ""
        for a, b in zip(src, got, strict=True)
    )


def layout(answer: str) -> str:
    """The answer's paragraphs, separated by one blank line, with single spaces inside each."""
    paragraphs = [" ".join(p.split()) for p in PARAGRAPH.split(answer.strip())]
    return "\n\n".join(p for p in paragraphs if p)


def rewrite(payload: dict, punctuated: list[str], text: str | None = None) -> dict | None:
    """The payload with punctuated tokens in place, or None if they do not line up exactly.

    `text`, when given, is the new top-level text (with paragraph breaks); it must consist of
    exactly the punctuated tokens. Segments and words take the tokens only, so their timing stays
    exact.
    """
    if text is not None and text.split() != punctuated:
        return None
    src = payload["text"].split()
    if not same_words(src, punctuated):
        return None
    out = dict(payload)
    out["text"] = text if text is not None else _replace_tokens(payload["text"], punctuated)
    segments = payload.get("segments")
    if isinstance(segments, list):
        seg_tokens = [str(s.get("text", "")).split() for s in segments]
        if not same_words([t for ts in seg_tokens for t in ts], punctuated):
            return None  # text and segments disagree: change nothing
        i, new_segments = 0, []
        for seg, ts in zip(segments, seg_tokens, strict=True):
            new_segments.append(
                {
                    **seg,
                    "text": _replace_tokens(str(seg.get("text", "")), punctuated[i : i + len(ts)]),
                }
            )
            i += len(ts)
        out["segments"] = new_segments
    words = payload.get("words")
    if isinstance(words, list) and words:
        w_tokens = [str(w.get("word", "")) for w in words]
        if all(len(t.split()) == 1 for t in w_tokens) and same_words(
            [t.strip() for t in w_tokens], punctuated
        ):
            out["words"] = [
                {**w, "word": _replace_tokens(w["word"], [p])}
                for w, p in zip(words, punctuated, strict=True)
            ]
    return out


def _replace_tokens(text: str, tokens: list[str]) -> str:
    """Replace the whitespace-separated tokens of `text` in order, keeping its whitespace."""
    it = iter(tokens)
    return re.sub(r"\S+", lambda _m: next(it), text)


def parse_context(raw: bytes | None) -> str:
    """The chat context from `X-GS-Context`: "" when absent or invalid, else its last 4,000 chars.

    A cut keeps whole lines when it can, so the oldest message is dropped rather than half of it.
    """
    if not raw or len(raw) > MAX_CONTEXT_HEADER:
        return ""
    try:
        text = base64.b64decode(raw.strip(), validate=True).decode("utf-8")
    except (binascii.Error, ValueError):
        return ""
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[-MAX_CONTEXT_CHARS:]
        if "\n" in text:
            text = text.split("\n", 1)[1]
    # The context sits between tags in the prompt; with no angle brackets it cannot close them.
    return text.replace("<", "‹").replace(">", "›").strip()


def correction_message(text: str, context: str) -> str:
    """The user message of a correction: the context, if any, then the transcript."""
    parts = []
    if context:
        parts.append(
            "Recent messages in this chat, oldest first:\n<context>\n" + context + "\n</context>"
        )
    parts.append("Transcript to correct:\n<transcript>\n" + text.strip() + "\n</transcript>")
    return "\n\n".join(parts)


def _bare(token: str) -> str:
    """A token for comparing wordings: lower case, without punctuation or symbols at its ends."""
    return _trim(token).lower()


def _trim(token: str) -> str:
    """The token without punctuation or symbols at its ends."""
    i, j = 0, len(token)
    while i < j and unicodedata.category(token[i])[0] in "PS":
        i += 1
    while j > i and unicodedata.category(token[j - 1])[0] in "PS":
        j -= 1
    return token[i:j]


def _key(token: str) -> str:
    """What a token is compared by. With a digit, all but sentence marks at its end, so a sign,
    currency, % or a leading decimal point (".5") counts."""
    return token.rstrip(EDGE).lower() if any(c.isdigit() for c in token) else _bare(token)


def _sensitive(token: str) -> bool:
    """A number or a negation, whose change could flip what the note says."""
    # Apostrophes first: NFKC splits U+00B4 into a space and a combining accent.
    bare = _bare(token).translate(APOSTROPHES)
    bare = unicodedata.normalize("NFKC", bare).translate(APOSTROPHES).replace("ё", "е")
    if any(c.isdigit() for c in token):
        return bare not in DIGIT_NAMES
    return (
        bare in NEGATIONS
        or bare.endswith("n't")
        or any(part in NUMBER_WORDS for part in HYPHENS.split(bare))
    )


def same_numbers(src: list[str], got: list[str]) -> bool:
    """For answers with the same words: whether every token with a digit is also the same by
    `_key` (same_words strips a leading "." too, so ".5" and "5" would pass)."""
    return all(
        _key(a) == _key(b) for a, b in zip(src, got, strict=True) if any(c.isdigit() for c in a + b)
    )


def _is_sound(
    tokens: list[str], i: int, seps: list[str] | None = None, sounds: frozenset[str] = SOUNDS
) -> bool:
    """Whether tokens[i] is one of `sounds` that may go (see SOUNDS); the note is no meta note."""
    m = SOUND_SHAPE.fullmatch(tokens[i])
    core = m.group(2) if m else ""
    if _sound_key(core) not in sounds or core not in (core.lower(), core.lower().capitalize()):
        return False
    if core != core.lower():  # "Um, we merge." only: at a sentence's start, with its own pause
        start = i == 0 or (seps is not None and "\n" in seps[i])
        start = start or (_stops(tokens[i - 1]) and tokens[i - 1].lower() not in ABBREVIATIONS)
        if not start or not m.group(4).startswith((",", "..", "…")):
            return False
    return not any(_number(t) for t in tokens[max(0, i - 1) : i] + tokens[i + 1 : i + 2])


def _sound_key(word: str) -> str:
    """The word as SOUNDS lists it: lower case, "м" three times or more as "ммм"."""
    word = word.lower()
    return "ммм" if len(word) >= 3 and set(word) == {"м"} else word


def _number(token: str) -> bool:
    """A number in digits (not a name like k3s) or in words, "twenty-five", "half" and "½"
    included."""
    bare = _bare(token).replace("ё", "е")
    if any(c.isnumeric() for c in bare):
        return bare not in DIGIT_NAMES
    return any(part in NUMBER_WORDS or part in FRACTIONS for part in HYPHENS.split(bare))


def _stops(token: str) -> bool:
    """Whether the token ends a sentence (. ? !), brackets aside; an ellipsis does not."""
    end = token.rstrip(")]")
    return end.endswith(("?", "!")) or (end.endswith(".") and not end.endswith(".."))


def _keeps_sounds(text: str) -> bool:
    """Whether no sound may go from this note: it quotes something or talks about sounds or words
    (see META_STEMS), or writes a sound as a name ("UM", "@um", "Ask Um"): a sound there that
    _is_sound would keep."""
    if META_MARKS.search(text):
        return True
    found = list(re.finditer(r"\S+", text))
    tokens = [m.group() for m in found]
    seps = [text[(found[k - 1].end() if k else 0) : m.start()] for k, m in enumerate(found)]
    for i, token in enumerate(tokens):
        bare = _bare(token)
        if bare.startswith(META_STEMS):
            return True
        bare = _sound_key(bare)
        if bare in SOUNDS and _bare(token) != _trim(token) and not _is_sound(tokens, i, seps):
            return True  # not in lower case, and not a removable "Um,"
        if bare in SOUNDS and not SOUND_SHAPE.fullmatch(token):
            return True
    return False


def note_sounds(payload: dict) -> frozenset[str]:
    """The sounds that may go from this answer: by its "language" (verbose_json), none if it has
    none or another one."""
    return SOUNDS_BY_LANGUAGE.get(
        LANGUAGES.get(str(payload.get("language", "")).lower(), ""), frozenset()
    )


def _ends(token: str, marks: str) -> bool:
    end = token.rstrip(")]")
    return bool(end) and end[-1] in marks


def _dropped_ok(src: list[str], i1: int, i2: int, sounds: frozenset[str]) -> bool:
    """Whether src[i1:i2] may be dropped: all of it `sounds` (none in a meta note), or a repeat
    of the words just before or after."""
    if sounds and all(_is_sound(src, i, sounds=sounds) for i in range(i1, i2)):
        return True
    run = [_bare(t) for t in src[i1:i2]]
    k = len(run)
    return run in (
        [_bare(t) for t in src[max(0, i1 - k) : i1]],
        [_bare(t) for t in src[i2 : i2 + k]],
    )


def strip_sounds(text: str, sounds: frozenset[str] = SOUNDS) -> tuple[str, int]:
    """`text` without its hesitation sounds (SOUNDS), and how many were removed. Every other token
    stays. Text that is sounds only, a meta note (_keeps_sounds), longer than STRIP_MAX_CHARS or
    with a space of over STRIP_MAX_SPACE is left as it is. Linear in the text's length.

    Each removed token's own punctuation: a comma after it goes with it, but one before it stays,
    as dropping it could join a word to the next clause ("No, um, do it." is not "No do it."); a
    sentence mark after it (. ? !) moves to the word before unless that one ends a sentence
    already, and so does a ; or : (in place of a comma there), while an ellipsis goes with it (a
    ? or ! after it still moves);
    brackets opened and closed within the removed words go with them, as do the marks inside
    them, others move to the word they belong to; the larger line break before it moves to the
    next word; and when it started a sentence with a capital, the next word takes the capital (a
    plain lower-case word only, so `k3s` stays as written).
    """
    if len(text) > STRIP_MAX_CHARS:
        logger.info("punct: sounds not stripped, %d characters", len(text))
        return text, 0
    found = list(re.finditer(r"\S+", text))
    tokens = [m.group() for m in found]
    seps = [text[(found[k - 1].end() if k else 0) : m.start()] for k, m in enumerate(found)]
    if any(len(sep) > STRIP_MAX_SPACE for sep in seps):
        logger.info("punct: sounds not stripped, a space of over %d characters", STRIP_MAX_SPACE)
        return text, 0
    n = len(tokens)
    gone = [_is_sound(tokens, i, seps, sounds) for i in range(n)]
    if not any(gone) or all(gone):
        return text, 0
    if _keeps_sounds(text):
        logger.info("punct: sounds not stripped, the note quotes or talks about words")
        return text, 0
    caps, last, i = set(), None, 0  # last: the last token kept so far
    lines = [sep.count("\n") for sep in seps]
    while i < n:
        if not gone[i]:
            last, i = i, i + 1
            continue
        j = i
        while j < n and gone[j]:
            j += 1
        r = j if j < n else None  # the next token kept
        opened, closed, inside = [], "", []  # brackets the removed words leave open or close
        for k in range(i, j):
            m = SOUND_SHAPE.fullmatch(tokens[k])
            for c in m.group(1) if m else "":
                opened.append(c)
            for c in m.group(3) if m else "":
                if opened and PAIRS[opened[-1]] == c:
                    opened.pop()
                else:
                    closed += c
            inside.append(bool(opened))  # its marks are inside the brackets: they go too
        if (closed and last is None) or (opened and r is None):
            gone[i:j] = [False] * (j - i)  # a bracket with no word to go to: keep the words
            last, i = j - 1, j
            continue
        if closed:
            tokens[last] += closed
        if opened:
            tokens[r] = "".join(opened) + tokens[r]
        stops = last is not None and _stops(tokens[last])  # kept up to date, not rescanned
        for k in range(i, j):
            marks = SOUND_SHAPE.fullmatch(tokens[k]).group(4)  # every removed token has this shape
            # An ellipsis is the hesitation and goes; a ? or ! after it ends the sentence ("um…?").
            ellipsis = ".." in marks or "…" in marks
            stop = re.search(r"[?!]+\Z" if ellipsis else r"[.?!]+", marks)
            pause = re.search(r"[;:]", marks)  # a clause break: kept, with no capital after it
            stop, pause = (None, None) if inside[k - i] else (stop, pause)
            if stop and last is not None and not stops:
                tokens[last] = tokens[last].rstrip(",;:") + stop.group()
                stops = True
            elif pause and last is not None and not stops and not _ends(tokens[last], ";:"):
                tokens[last] = tokens[last].rstrip(",") + pause.group()
            if r is not None and lines[k] > lines[r]:
                seps[r], lines[r] = seps[k], lines[k]
            starts = last is None or "\n" in seps[k] or stops
            if r is not None and starts and (_trim(tokens[k])[:1].isupper() or stop):
                caps.add(r)
        i = j
    for r in caps:
        word = _trim(tokens[r])
        if word.isalpha() and word.islower():
            at = tokens[r].find(word)
            tokens[r] = tokens[r][:at] + word[0].upper() + tokens[r][at + 1 :]
    kept = [k for k in range(n) if not gone[k]]
    if len(kept) == n:
        return text, 0
    seps[kept[0]] = seps[0]
    return "".join(seps[k] + tokens[k] for k in kept) + text[found[-1].end() :], sum(gone)


def _moved(sm: difflib.SequenceMatcher, src: list[str], got: list[str]) -> bool:
    """Whether a token was dropped in one place and added in another across a number or a
    negation: content moved over it, which can turn "Not ready. Ready now." into "Ready. Not
    ready now." with every change allowed. Two separate fixes of one word stay allowed."""
    gone: dict[str, list[int]] = {}  # key -> source positions it was dropped from
    new: dict[str, list[int]] = {}  # key -> source positions it was added at
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            for i in range(i1, i2):
                gone.setdefault(_key(src[i]), []).append(i)
            for t in got[j1:j2]:
                new.setdefault(_key(t), []).append(i1)
    if not gone.keys() & new.keys():
        return False
    count = [0]  # count[i]: sensitive tokens in src[:i], so any interval is checked at once
    for t in src:
        count.append(count[-1] + _sensitive(t))
    for key in gone.keys() & new.keys():
        for i in gone[key]:
            for j in new[key]:
                lo, hi = sorted((i, j))
                if count[min(hi + 1, len(src))] > count[lo]:
                    return True
    return False


def is_correction(
    src: list[str], got: list[str], sounds: frozenset[str] = SOUNDS
) -> tuple[bool, int, int, str]:
    """Whether `got` corrects `src` rather than rewriting it; tokens dropped and added; and the
    reason for a refusal ("" when accepted). `sounds` may be dropped (see note_sounds).

    Lengths that cannot pass are refused before the alignment, whose cost grows with both; the
    counts are then the length difference only.
    """
    allowed = max(CORRECT_MIN_ALLOWED, int(len(src) * CORRECT_MAX_SHARE))
    if len(got) > len(src) + allowed or len(got) < len(src) * CORRECT_MIN_COUNT_RATIO:
        return False, max(0, len(src) - len(got)), max(0, len(got) - len(src)), "length"
    sm = difflib.SequenceMatcher(
        None, [_key(t) for t in src], [_key(t) for t in got], autojunk=False
    )
    kept, reason, droppable = 0, "", None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            kept += i2 - i1
        elif reason:
            continue
        elif any(_sensitive(t) for t in src[i1:i2] + got[j1:j2]):
            reason = "number or negation"
        elif tag == "insert" and j2 - j1 > 1:
            reason = "insertion"
        elif tag == "delete":
            if droppable is None:  # once, and only if needed
                droppable = frozenset() if _keeps_sounds(" ".join(src)) else sounds
            reason = "" if _dropped_ok(src, i1, i2, droppable) else "deletion"
        elif tag == "replace" and max(i2 - i1, j2 - j1) > MAX_REPLACE:
            reason = "replacement"
    dropped, added = len(src) - kept, len(got) - kept
    if not reason and (dropped > allowed or added > allowed):
        reason = "share"
    if not reason and _moved(sm, src, got):
        reason = "move"
    return not reason, dropped, added, reason


MAX_LLM_BYTES = 1024 * 1024  # an answer of max_tokens is far smaller; anything bigger is refused


def make_complete(url: str, model: str, transport=None, system: str = SYSTEM) -> Complete:  # noqa: ANN001
    """One HTTP client per call: a few calls an hour, and nothing left open at shutdown."""
    import httpx

    async def complete(text: str, max_tokens: int, timeout: float) -> str:
        body = {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        }
        data = bytearray()
        headers = {"accept-encoding": "identity"}  # raw bytes only: nothing to inflate in memory
        async with httpx.AsyncClient(transport=transport) as client:
            async with client.stream("POST", url, json=body, headers=headers, timeout=timeout) as r:
                r.raise_for_status()
                if r.headers.get("content-encoding", "identity").lower() != "identity":
                    raise ValueError("LLM answer is compressed")
                async for chunk in r.aiter_raw():
                    if len(data) + len(chunk) > MAX_LLM_BYTES:
                        raise ValueError("LLM answer too large")
                    data += chunk
        return str(json.loads(bytes(data))["choices"][0]["message"]["content"])

    return complete


async def _stop(*tasks: asyncio.Future) -> None:
    """Cancel the tasks still running, once each, and wait until all have ended.

    Their own results and errors are collected, never raised. A cancellation of the caller that
    arrives meanwhile is held until every task has ended, then raised: cleanup is never cut short.
    """
    for t in tasks:
        if not t.done():
            t.cancel()
    pending = {t for t in tasks if not t.done()}
    interrupted = False
    while pending:
        try:
            _, pending = await asyncio.wait(pending)
        except asyncio.CancelledError:
            interrupted = True
    for t in tasks:
        _error(t)
    if interrupted:
        raise asyncio.CancelledError


async def _in_thread(fn: Callable, *args: object) -> object:
    """fn(*args) in a worker thread. A thread cannot be cancelled: a cancellation of the caller
    waits until it has ended and is then raised, so the caller's accounting (inflight) and its
    deadline cover the whole of the work."""
    fut = asyncio.get_running_loop().run_in_executor(None, fn, *args)
    interrupted = False
    while not fut.done():
        try:
            await asyncio.wait({fut})
        except asyncio.CancelledError:
            interrupted = True
    if interrupted:
        _error(fut)  # retrieved, so asyncio never logs it (its text may hold the transcript)
        raise asyncio.CancelledError
    return fut.result()


def _unwrap(answer: str, transcript: str) -> str:
    """The answer without the prompt's transcript tags, if the model repeated them."""
    return answer if "transcript>" in transcript.lower() else ECHOED_TAGS.sub("", answer)


def _error(task: asyncio.Future) -> BaseException | None:
    return None if task.cancelled() else task.exception()


async def _until_disconnect(receive) -> None:  # noqa: ANN001
    while (await receive())["type"] != "http.disconnect":
        pass


class Punctuate:
    """ASGI wrapper: buffers an opted-in transcription answer and maybe punctuates it."""

    def __init__(  # noqa: PLR0913
        self,
        app,  # noqa: ANN001
        complete: Complete,
        timeout: float = 60.0,
        min_run: int = 40,
        paragraph_min_words: int = 80,
        max_words: int = 3000,
        max_bytes: int = 8 * 1024 * 1024,
        max_waiting: int = 2,
        app_grace: float = 5.0,
        send_timeout: float = 60.0,
        correct: Complete | None = None,
        correct_min_words: int = 3,
    ) -> None:
        self.app = app
        self.complete = complete
        self.correct = correct  # the completion with SYSTEM_CORRECT; None = correction mode off
        self.correct_min_words = correct_min_words
        self.timeout = timeout
        self.min_run = min_run
        self.paragraph_min_words = paragraph_min_words
        self.max_words = max_words
        self.max_bytes = max_bytes
        self.max_waiting = max_waiting
        self.app_grace = app_grace
        self.send_timeout = send_timeout
        self._lock: asyncio.Lock | None = None
        self.inflight = 0  # passes holding or waiting for the LLM slot

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001, C901
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != PATH
            or OPT_IN not in [(k, v.strip().lower()) for k, v in scope.get("headers", [])]
        ):
            await self.app(scope, receive, send)
            return
        context = self._correction_context(scope.get("headers", []))
        start: dict | None = None
        chunks: list[bytes] = []
        size = 0
        passing = False
        complete = asyncio.Event()  # the app has sent its last body message

        async def capture(message) -> None:  # noqa: ANN001
            nonlocal start, size, passing
            kind = message["type"]
            last = kind == "http.response.body" and not message.get("more_body", False)
            if passing:
                await send(message)
                if last:
                    complete.set()
                return
            if kind == "http.response.start":
                ctype = dict(message.get("headers", [])).get(b"content-type", b"")
                if message["status"] != 200 or not ctype.startswith(b"application/json"):
                    passing = True
                    await send(message)
                    return
                start = message
                return
            if kind == "http.response.body" and start is not None and not complete.is_set():
                body = message.get("body", b"")
                size += len(body)
                if size > self.max_bytes:  # too big to hold: send what we have, then stream on
                    passing = True
                    await send(start)
                    for c in chunks:
                        await send({"type": "http.response.body", "body": c, "more_body": True})
                    chunks.clear()
                    await send(message)
                    if last:
                        complete.set()
                    return
                if body:
                    chunks.append(body)
                if last:
                    complete.set()
                return
            await send(message)

        # The app task is stopped on every way out of here, an outer cancellation included, so an
        # app stuck after its answer cannot keep Admission's Whisper permit.
        task = asyncio.ensure_future(self.app(scope, receive, capture))
        try:
            done = asyncio.ensure_future(complete.wait())
            try:
                await asyncio.wait({task, done}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                await _stop(done)
            if not complete.is_set():
                # The app ended without a whole answer: let its own error through, or say so.
                if _error(task) is not None:
                    raise _error(task)
                if start is not None and not passing:
                    chunks.clear()
                    await asyncio.wait_for(
                        _send_error(send, 500, b"Incomplete transcription response."),
                        timeout=self.send_timeout,
                    )
                return
            # The answer is whole. The app gets a short grace to return (Admission releases the
            # permit there); after that it is cancelled. An error after the answer changes nothing.
            await asyncio.wait({task}, timeout=self.app_grace)
            if not task.done():
                logger.warning(
                    "punct: app still running %.1f s after its answer; cancelled", self.app_grace
                )
            elif _error(task) is not None:
                logger.warning(
                    "punct: app raised after its answer: %s", type(_error(task)).__name__
                )
            await _stop(task)
            if passing:
                return
            body = b"".join(chunks)
            chunks.clear()
            result = await self._punctuate_unless_gone(body, receive, context)
            if result is None:
                return  # the client left; nobody to answer
            outcome, body = result
            headers = [(k, v) for k, v in start.get("headers", []) if k != b"content-length"]
            headers += [
                (b"content-length", str(len(body)).encode()),
                (b"x-gs-punctuated", outcome.encode()),
            ]

            async def finish() -> None:
                await send({**start, "headers": headers})
                await send({"type": "http.response.body", "body": body})

            await asyncio.wait_for(finish(), timeout=self.send_timeout)
        finally:
            await _stop(task)

    def _correction_context(self, headers: list) -> str | None:
        """None for the normal pass; the chat context ("" if none) when correction is asked for."""
        if self.correct is None or self.correct_min_words <= 0:
            return None
        found = [(k.lower(), v) for k, v in headers]
        if CORRECT_OPT_IN not in [(k, v.strip().lower()) for k, v in found]:
            return None
        try:
            return parse_context(next((v for k, v in found if k == CONTEXT_HEADER), None))
        except Exception as e:  # noqa: BLE001 - a bad context is no context
            logger.warning("punct: context: %s", type(e).__name__)
            return ""

    async def _punctuate_unless_gone(  # noqa: ANN001
        self, body: bytes, receive, context: str | None = None
    ) -> tuple[str, bytes] | None:
        """The pass, raced against the client leaving; None if it left (a tie counts as left).

        Only called once the app task has ended, so nothing else reads `receive` any more.
        """
        work = asyncio.ensure_future(self._punctuate(body, context))
        gone = asyncio.ensure_future(_until_disconnect(receive))
        try:
            await asyncio.wait({work, gone}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            await _stop(work, gone)
        if not gone.cancelled() and gone.exception() is None:
            return None  # the client is gone
        if work.cancelled():  # the watcher failed, not the client: answer with Whisper's text
            logger.warning("punct: disconnect watcher failed: %s", type(gone.exception()).__name__)
            return "error", body
        return work.result()

    async def _punctuate(self, body: bytes, context: str | None = None) -> tuple[str, bytes]:
        try:
            return await self._punctuate_or_raise(body, context)
        except Exception as e:  # noqa: BLE001 - never worse than Whisper's own answer
            logger.warning("punct: %s", type(e).__name__)
            return "error", body

    async def _punctuate_or_raise(  # noqa: C901
        self, body: bytes, context: str | None = None
    ) -> tuple[str, bytes]:
        begin = time.perf_counter()
        try:
            payload = json.loads(body)
            text = payload.get("text") if isinstance(payload, dict) else None
        except (ValueError, UnicodeDecodeError):
            return "skipped", body
        if not isinstance(text, str):
            return "skipped", body
        tokens = text.split()
        wanted = longest_unpunctuated(tokens) >= self.min_run or (
            self.paragraph_min_words > 0 and len(tokens) >= self.paragraph_min_words
        )
        if context is not None:
            wanted = wanted or len(tokens) >= self.correct_min_words
        if not tokens or len(tokens) > self.max_words or not wanted:
            return "skipped", body
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self.inflight >= 1 + self.max_waiting:  # one in the slot, max_waiting behind it
            outcome, out = "busy", body
        else:
            self.inflight += 1
            try:
                outcome, out = await self._ask(payload, tokens, body, begin, context)
            finally:
                self.inflight -= 1
        removed = 0
        if context is not None and outcome in STRIPPED:
            out, removed = _without_sounds(out, text)
        logger.info(
            "punct: %s, %d words, longest unpunctuated %d, %d sounds removed, %.0f ms",
            outcome,
            len(tokens),
            longest_unpunctuated(tokens),
            removed,
            (time.perf_counter() - begin) * 1000,
        )
        return outcome, out

    async def _ask(  # noqa: PLR0913
        self,
        payload: dict,
        tokens: list[str],
        body: bytes,
        begin: float,
        context: str | None = None,
    ) -> tuple[str, bytes]:
        holding = False
        try:
            async with asyncio.timeout(self.timeout):
                async with self._lock:
                    holding = True
                    remaining = max(1.0, self.timeout - (time.perf_counter() - begin))
                    # Budget from characters, not words: a Cyrillic word costs several tokens.
                    # Half the characters is a heuristic with room for marks and breaks (it worked
                    # on the 12 test notes, Russian included); a cut-short answer is refused anyway.
                    # The context does not count: the answer is the transcript only.
                    max_tokens = math.ceil(len(payload["text"]) / 2) + 64
                    if context is None:
                        got = await self.complete(payload["text"], max_tokens, remaining)
                    else:
                        message = correction_message(payload["text"], context)
                        got = await self.correct(message, max_tokens, remaining)
                if context is not None:
                    got = _unwrap(got, payload["text"])
                    text = layout(got)
                    answer = text.split()
                    if not (same_words(tokens, answer) and same_numbers(tokens, answer)):
                        # Words changed. Aligning takes up to about 1.2 s on 3,000 repeats of one
                        # word (a Whisper loop), so it runs in a thread, within the deadline.
                        return await _in_thread(
                            self._corrected, payload, tokens, body, text, bool(context)
                        )
        except TimeoutError:
            return ("timeout" if holding else "busy"), body
        except Exception as e:  # noqa: BLE001 - any failure keeps Whisper's answer
            logger.warning("punct: %s", type(e).__name__)
            return "error", body
        text = layout(got)
        new = rewrite(payload, text.split(), text)
        if new is None:
            return "rejected", body
        if text == layout(payload["text"]):
            return "unchanged", body
        return "applied", json.dumps(new, ensure_ascii=False).encode()

    def _corrected(
        self, payload: dict, tokens: list[str], body: bytes, text: str, with_context: bool
    ) -> tuple[str, bytes]:
        """A correction that changed words: the new top-level text, or the fallback if refused."""
        ok, dropped, added, reason = is_correction(tokens, text.split(), note_sounds(payload))
        logger.info(
            "punct: correction %s, %d words, %d dropped, %d added, count ratio %.2f, context %s",
            "accepted" if ok else f"refused ({reason})",
            len(tokens),
            dropped,
            added,
            len(text.split()) / len(tokens),
            "yes" if with_context else "no",
        )
        if ok:
            return "corrected", json.dumps({**payload, "text": text}, ensure_ascii=False).encode()
        return "rejected", body


# Correction mode removes hesitation sounds from the top-level text after the guard, so the guard
# sees the model's answer as it came: from a corrected, applied or unchanged answer, and from
# Whisper's own text when the answer was refused (rejected). Nothing goes if Whisper's transcript
# or the text itself keeps its sounds (_keeps_sounds): an answer may have dropped the quotes or
# capitals that made a sound a literal or a name, and only the sounds of the answer's "language"
# go (verbose_json carries it; claude-tools always asks for it), none without one. Busy, timeout,
# error and skipped keep
# Whisper's answer byte for byte. strip_sounds is linear in the
# text and skips one past STRIP_MAX_CHARS, so it runs on the event loop: well under a millisecond
# for a note, about 50 ms at the bound. Any error in it sends the answer as it was.
STRIPPED = ("corrected", "applied", "unchanged", "rejected")


def _without_sounds(body: bytes, transcript: str) -> tuple[bytes, int]:
    """The answer with sounds removed from its top-level `text`; segments and words unchanged."""
    try:
        payload = json.loads(body)
        sounds = note_sounds(payload)
        if not sounds:
            logger.info("punct: sounds not stripped, language %s", payload.get("language"))
            return body, 0
        if _keeps_sounds(transcript):  # what Whisper heard decides too, not only the answer
            logger.info("punct: sounds not stripped, the transcript quotes or names them")
            return body, 0
        text, removed = strip_sounds(payload["text"], sounds)
        if not removed:
            return body, 0
        return json.dumps({**payload, "text": text}, ensure_ascii=False).encode(), removed
    except Exception as e:  # noqa: BLE001 - the answer as it was
        logger.warning("punct: sounds: %s", type(e).__name__)
        return body, 0


async def _send_error(send, status: int, detail: bytes) -> None:  # noqa: ANN001
    body = b'{"detail":"' + detail + b'"}'
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def wrap(app, env: Mapping[str, str]):  # noqa: ANN001, ANN201
    """Punctuate around `app` when GS_PUNCT_URL is set; otherwise `app` unchanged."""
    url = env.get("GS_PUNCT_URL")
    if not url:
        return app
    model = env.get("GS_PUNCT_MODEL", "default")
    return Punctuate(
        app,
        make_complete(url, model),
        timeout=float(env.get("GS_PUNCT_TIMEOUT_S", "60")),
        min_run=int(env.get("GS_PUNCT_MIN_RUN", "40")),
        paragraph_min_words=int(env.get("GS_PUNCT_PARAGRAPH_MIN_WORDS", "80")),
        max_words=int(env.get("GS_PUNCT_MAX_WORDS", "3000")),
        max_bytes=int(env.get("GS_PUNCT_MAX_BYTES", str(8 * 1024 * 1024))),
        max_waiting=int(env.get("GS_PUNCT_MAX_WAITING", "2")),
        app_grace=float(env.get("GS_PUNCT_APP_GRACE_S", "5")),
        send_timeout=float(env.get("GS_SEND_TIMEOUT_S", "60")),
        correct=make_complete(url, model, system=SYSTEM_CORRECT),
        correct_min_words=int(env.get("GS_PUNCT_CORRECT_MIN_WORDS", "3")),
    )
