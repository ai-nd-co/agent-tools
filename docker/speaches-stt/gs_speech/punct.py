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
punctuation), and at most max(2, 10%) of the transcript's tokens may be dropped and as many added,
and the answer must keep at least half as many tokens. A correction replaces only the top-level
`text` (`X-GS-Punctuated: corrected`); segments and words stay as Whisper wrote them, since their
timings belong to Whisper's words. An answer that changed no word is used as in the normal pass
(applied or unchanged). A refused answer falls back, without a second call, to its punctuation on
the words it kept unchanged, if it kept at least half of them (applied); otherwise the original
answer is sent (rejected). The log line has counts only.

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
# may change 2 tokens, as one name can be two words ("cloud code").
CORRECT_MAX_SHARE = 0.10
CORRECT_MIN_ALLOWED = 2
CORRECT_MIN_COUNT_RATIO = 0.5
FALLBACK_MIN_KEPT = 0.5  # below this the answer is another text, and its marks land on stray words
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
    i, j = 0, len(token)
    while i < j and unicodedata.category(token[i])[0] in "PS":
        i += 1
    while j > i and unicodedata.category(token[j - 1])[0] in "PS":
        j -= 1
    return token[i:j].lower()


def kept_count(src: list[str], got: list[str], key: Callable[[str], str]) -> int:
    """How many tokens of `src` appear in `got` in the same order (difflib's matching blocks)."""
    sm = difflib.SequenceMatcher(None, [key(t) for t in src], [key(t) for t in got], autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks())


def is_correction(src: list[str], got: list[str]) -> tuple[bool, int, int]:
    """Whether `got` corrects `src` rather than rewriting it; also tokens dropped and added.

    Lengths that cannot pass are refused before the alignment, whose cost grows with both; the
    counts are then the length difference only.
    """
    allowed = max(CORRECT_MIN_ALLOWED, int(len(src) * CORRECT_MAX_SHARE))
    if len(got) > len(src) + allowed or len(got) < len(src) * CORRECT_MIN_COUNT_RATIO:
        return False, max(0, len(src) - len(got)), max(0, len(got) - len(src))
    kept = kept_count(src, got, _bare)
    dropped, added = len(src) - kept, len(got) - kept
    return dropped <= allowed and added <= allowed, dropped, added


def fallback(src: list[str], answer: str) -> str | None:
    """A refused correction's marks, case and breaks on the words it kept; Whisper's elsewhere.

    The result has exactly the transcript's words, so it goes through the normal guard. None when
    the answer kept fewer than FALLBACK_MIN_KEPT of them, or has over twice as many tokens (which
    also bounds the alignment's cost).
    """
    paragraphs = [p.split() for p in layout(answer).split("\n\n") if p]
    got = [t for p in paragraphs for t in p]
    if len(got) > 2 * len(src):
        return None
    starts, i = set(), 0
    for p in paragraphs[:-1]:  # where the 2nd and later paragraphs start; never before the first
        i += len(p)
        starts.add(i)
    sm = difflib.SequenceMatcher(
        None, [core(t) for t in src], [core(t) for t in got], autojunk=False
    )
    out, breaks, kept = list(src), set(), 0
    for a, b, size in sm.get_matching_blocks():
        kept += size
        for k in range(size):
            out[a + k] = got[b + k]
            if b + k in starts and a + k > 0:
                breaks.add(a + k)
    if kept < len(src) * FALLBACK_MIN_KEPT:
        return None
    lines, cur = [], []
    for k, t in enumerate(out):
        if k in breaks:
            lines.append(" ".join(cur))
            cur = []
        cur.append(t)
    lines.append(" ".join(cur))
    return "\n\n".join(lines)


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
        raise asyncio.CancelledError
    return fut.result()


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
        logger.info(
            "punct: %s, %d words, longest unpunctuated %d, %.0f ms",
            outcome,
            len(tokens),
            longest_unpunctuated(tokens),
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
                    text = layout(ECHOED_TAGS.sub("", got))
                    if not same_words(tokens, text.split()):
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
        text = layout(got if context is None else ECHOED_TAGS.sub("", got))
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
        ok, dropped, added = is_correction(tokens, text.split())
        logger.info(
            "punct: correction %s, %d words, %d dropped, %d added, count ratio %.2f, context %s",
            "accepted" if ok else "refused",
            len(tokens),
            dropped,
            added,
            len(text.split()) / len(tokens),
            "yes" if with_context else "no",
        )
        if ok:
            return "corrected", json.dumps({**payload, "text": text}, ensure_ascii=False).encode()
        punctuated = fallback(tokens, text)
        new = None if punctuated is None else rewrite(payload, punctuated.split(), punctuated)
        if new is None or punctuated == layout(payload["text"]):
            return "rejected", body
        return "applied", json.dumps(new, ensure_ascii=False).encode()


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
