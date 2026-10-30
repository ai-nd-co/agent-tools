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
anything but a repeat or a filler, and a replacement of more than 2 tokens on either side. It does
not police person names: a context can make the model swap one name for another. A refused answer
sends the original (rejected); its punctuation is not reused, as it can put a stop mid-sentence.
The log line has counts and the refusal's kind only.

Fillers: after the guard, correction mode removes filler words from the top-level `text` in code,
not by the model, and only on positive evidence: hesitation sounds ("um", "э"; SOUNDS), and words
like "like", "you know", "ну" or "как бы" (WORD_FILLERS) only where the text's own punctuation sets
them off (", you know,"), so "I like it" or "Как бы ты решил?" stay. Corrected, applied and
unchanged answers lose every filler, a rejected one only SOUNDS; segments and words keep them, and
the outcome header stays as it was. The log line counts the removed tokens.

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
# filler, and one replacement spans at most MAX_REPLACE tokens on either side. Person names are not
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
# Fillers, which correction mode removes in code (strip_fillers) and a correction may also drop.
# Default keep: a word is removed only on positive evidence that it is a filler.
# SOUNDS are hesitation sounds that are no word, unit or abbreviation, in lower case or with a
# capital ("Um"); all capitals, or a digit next to it, keeps one. Not "er" (ER), "mm" (5 mm), "ah"
# or "eh", which can be words.
SOUNDS = frozenset("uh um uhm umm erm hmm э эм ммм".split())
# WORD_FILLERS are words with a meaning elsewhere ("I like it", "Как бы ты решил?", "путь короче"),
# so one is removed only where the text's own punctuation sets it off: with its own comma or
# ellipsis after it, and at a sentence's start or after a comma; or with a comma before it and the
# sentence's end after it (", you know."); never "like" before a number ("like, three": about
# three). Not "so", "well", "just", "actually", "basically", "right", "okay", "просто": in Davron's
# notes they mostly carry meaning ("so" 38 times in 40 notes).
WORD_FILLERS = tuple(  # longest first
    sorted(
        (tuple(p.split()) for p in "like|you know|i mean|типа|как бы|короче|вот|ну".split("|")),
        key=len,
        reverse=True,
    )
)
SENTENCE_END = ".?!…"
OPENERS = "([{«“„\"'"
CLOSERS = ")]}»”\"'"
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


def _filler_spans(
    tokens: list[str], seps: list[str] | None = None, everything: bool = True
) -> list[list[int]]:
    """The fillers among `tokens` (`seps`: the whitespace before each), each as the positions of its
    tokens; only SOUNDS unless `everything`. See SOUNDS and WORD_FILLERS for what counts."""
    n = len(tokens)
    seps = seps if seps is not None else [""] + [" "] * (n - 1)
    sound = [_is_sound(tokens, i) for i in range(n)]
    spans = [[i] for i in range(n) if sound[i]]
    if not everything:
        return spans
    keys = [_bare(t) for t in tokens]
    i = 0
    while i < n:
        k = next((len(p) for p in WORD_FILLERS if tuple(keys[i : i + len(p)]) == p), 0)
        if k and _plain(tokens, seps, i, i + k) and _set_off(tokens, seps, sound, i, i + k):
            spans.append(list(range(i, i + k)))
            i += k
        else:
            i += 1
    return spans


def _is_sound(tokens: list[str], i: int) -> bool:
    core = _trim(tokens[i])
    if core.lower() not in SOUNDS or core not in (core.lower(), core.lower().capitalize()):
        return False
    return not any(c.isdigit() for t in tokens[max(0, i - 1) : i + 2] for c in t)


def _plain(tokens: list[str], seps: list[str], i: int, j: int) -> bool:
    """Whether tokens[i:j] are one phrase: no punctuation but after the last, on one line."""
    return all(
        _trim(tokens[k])[:1] == tokens[k][:1]
        and (k == j - 1 or _trim(tokens[k]) == tokens[k])
        and (k == i or "\n" not in seps[k])
        for k in range(i, j)
    )


def _set_off(tokens: list[str], seps: list[str], sound: list[bool], i: int, j: int) -> bool:
    """Whether the punctuation around tokens[i:j] sets them off as a filler (see WORD_FILLERS).
    Sounds before them are looked past; a mark after a sound counts as the sentence's end."""
    last, core = tokens[j - 1], _trim(tokens[j - 1])
    own = last[last.find(core) + len(core) :]
    if any(c in CLOSERS for c in own):
        return False  # a bracket or a quote ends here: keep it simple and keep the words
    start, comma, k = "\n" in seps[i], False, i - 1
    while k >= 0:
        start = start or _ends(tokens[k], SENTENCE_END)
        if not sound[k]:
            comma = _ends(tokens[k], ",;:")
            break
        start = start or "\n" in seps[k]
        k -= 1
    start = start or k < 0
    if _like_a_number(tokens, i, j):
        return False  # "like three" is "about three"
    if own.endswith((",", "..", "…")):
        return start or comma
    return _ends(own, ".?!") and comma and not start


def _like_a_number(tokens: list[str], i: int, j: int) -> bool:
    """Whether tokens[i:j] is "like" before a number."""
    after = _bare(tokens[j]) if j < len(tokens) else ""
    return (_bare(tokens[i]),) == ("like",) and (
        after.replace("ё", "е") in NUMBER_WORDS
        or (any(c.isdigit() for c in after) and after not in DIGIT_NAMES)
    )


def _ends(token: str, marks: str) -> bool:
    """Whether the token, brackets and quotes aside, ends with one of `marks`."""
    end = token.rstrip(CLOSERS)
    return bool(end) and end[-1] in marks


def _dropped_ok(src: list[str], i1: int, i2: int, spans: list[list[int]]) -> bool:
    """Whether src[i1:i2] may be dropped: whole fillers of `spans` (the strip would remove them
    from the transcript too), and what is left is nothing or a repeat of the words just before or
    after."""
    filler = set()
    for span in spans:
        if i1 <= span[0] and span[-1] < i2:
            filler.update(span)
    rest = [_bare(src[i]) for i in range(i1, i2) if i not in filler]
    k = len(rest)
    before = [_bare(t) for t in src[max(0, i1 - k) : i1]]
    after = [_bare(t) for t in src[i2 : i2 + k]]
    return not rest or rest in (before, after)


def strip_fillers(text: str, everything: bool = True) -> tuple[str, int]:
    """`text` without its fillers, and how many tokens were removed; only SOUNDS unless
    `everything`. Every other token and every paragraph break stay. Text that is fillers only is
    left as it is. Linear in the text's length.

    Each removed token's own punctuation: a comma after it goes with it, but one before it stays,
    as dropping it could join a word to the next clause ("Не, ну, давай." is not "Не давай."); a
    sentence mark after it (. ? !) moves to the word before unless that one ends a sentence
    already, and an ellipsis goes with it; brackets or quotes around it alone go with it, others
    move to the word they belong to; a blank line before it moves to the next word; and when it
    started a sentence with a capital, the next word takes the capital (a plain lower-case word
    only, so `k3s` or `vox-flow` stay as written).
    """
    found = list(re.finditer(r"\S+", text))
    tokens = [m.group() for m in found]
    seps = [text[(found[k - 1].end() if k else 0) : m.start()] for k, m in enumerate(found)]
    gone = [False] * len(tokens)
    for span in _filler_spans(tokens, seps, everything):
        for i in span:
            gone[i] = True
    if not any(gone) or all(gone):
        return text, 0
    nxt, upcoming = [0] * len(tokens), None  # nxt[i]: the next token kept after i
    for i in reversed(range(len(tokens))):
        nxt[i] = upcoming
        upcoming = upcoming if gone[i] else i
    last = None  # the last token kept so far
    for i, tok in enumerate(tokens):
        if not gone[i]:
            last = i
            continue
        core = _trim(tok)
        at = tok.find(core)
        head, rest, r = tok[:at], tok[at + len(core) :], nxt[i]
        opened = "".join(c for c in head if c in OPENERS)
        closed = "".join(c for c in rest if c in CLOSERS)
        if not (opened and closed):  # a pair around this token alone goes with it
            if opened and r is not None:
                tokens[r] = opened + tokens[r]
            if closed and last is not None:
                tokens[last] += closed
        stop = None if (".." in rest or "…" in rest) else re.search(r"[.?!]+", rest)
        if stop and last is not None and not _ends(tokens[last], SENTENCE_END):
            tokens[last] = tokens[last].rstrip(",;:") + stop.group()
        if r is not None and "\n" in seps[i] and "\n" not in seps[r]:
            seps[r] = seps[i]
        starts = last is None or "\n" in seps[i] or _ends(tokens[last], ".?!")
        if r is not None and starts and (core[:1].isupper() or stop):
            word = _trim(tokens[r])
            if word.isalpha() and word.islower():
                at = tokens[r].find(word)
                tokens[r] = tokens[r][:at] + word[0].upper() + tokens[r][at + 1 :]
    kept = [k for k in range(len(tokens)) if not gone[k]]
    seps[kept[0]] = seps[0]
    tail = text[found[-1].end() :]
    return "".join(seps[k] + tokens[k] for k in kept) + tail, sum(gone)


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


def is_correction(src: list[str], got: list[str]) -> tuple[bool, int, int, str]:
    """Whether `got` corrects `src` rather than rewriting it; tokens dropped and added; and the
    reason for a refusal ("" when accepted).

    Lengths that cannot pass are refused before the alignment, whose cost grows with both; the
    counts are then the length difference only.
    """
    allowed = max(CORRECT_MIN_ALLOWED, int(len(src) * CORRECT_MAX_SHARE))
    if len(got) > len(src) + allowed or len(got) < len(src) * CORRECT_MIN_COUNT_RATIO:
        return False, max(0, len(src) - len(got)), max(0, len(got) - len(src)), "length"
    sm = difflib.SequenceMatcher(
        None, [_key(t) for t in src], [_key(t) for t in got], autojunk=False
    )
    kept, reason, spans = 0, "", None
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
            spans = _filler_spans(src) if spans is None else spans  # once, and only if needed
            reason = "" if _dropped_ok(src, i1, i2, spans) else "deletion"
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
            out, removed = _without_fillers(out, everything=outcome != "rejected")
        logger.info(
            "punct: %s, %d words, longest unpunctuated %d, %d fillers removed, %.0f ms",
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
        ok, dropped, added, reason = is_correction(tokens, text.split())
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


# Correction mode removes fillers from the top-level text after the guard, so the guard sees the
# model's answer as it came. Accepted answers (and an unchanged one, which the model read too) lose
# every filler; a refused answer, which sends Whisper's own text, loses only SOUNDS:
# that path is the safety net, so it gets no change that could alter a word's meaning. Busy,
# timeout, error and skipped keep Whisper's answer byte for byte. strip_fillers is linear in the
# text (at most GS_PUNCT_MAX_BYTES, GS_PUNCT_MAX_WORDS words), so it runs on the event loop: well
# under a millisecond for a note, tens of milliseconds for 100,000 characters. Any error in it
# sends the answer as it was.
STRIPPED = ("corrected", "applied", "unchanged", "rejected")


def _without_fillers(body: bytes, everything: bool) -> tuple[bytes, int]:
    """The answer with fillers removed from its top-level `text`; segments and words unchanged."""
    try:
        payload = json.loads(body)
        text, removed = strip_fillers(payload["text"], everything)
        if not removed:
            return body, 0
        return json.dumps({**payload, "text": text}, ensure_ascii=False).encode(), removed
    except Exception as e:  # noqa: BLE001 - the answer as it was
        logger.warning("punct: fillers: %s", type(e).__name__)
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
