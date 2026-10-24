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

It wraps Admission from outside, so the LLM call happens after the Whisper permit is released: the
inner app runs as its own task, and once its answer is complete it has GS_PUNCT_APP_GRACE_S (5) to
return before it is cancelled (also if the request itself is cancelled). One LLM call at a time
(the model has one slot), at most GS_PUNCT_MAX_WAITING (2) waiting behind it, each bounded by
GS_PUNCT_TIMEOUT_S (60) including the wait. A client that disconnects meanwhile cancels its pass.

Env: GS_PUNCT_URL (chat completions URL; unset = off), GS_PUNCT_MODEL, GS_PUNCT_TIMEOUT_S,
GS_PUNCT_MIN_RUN, GS_PUNCT_PARAGRAPH_MIN_WORDS, GS_PUNCT_MAX_WORDS (3000), GS_PUNCT_MAX_BYTES
(8 MiB), GS_PUNCT_MAX_WAITING, GS_PUNCT_APP_GRACE_S.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
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


MAX_LLM_BYTES = 1024 * 1024  # an answer of max_tokens is far smaller; anything bigger is refused


def make_complete(url: str, model: str, transport=None) -> Complete:  # noqa: ANN001
    """One HTTP client per call: a few calls an hour, and nothing left open at shutdown."""
    import httpx

    async def complete(text: str, max_tokens: int, timeout: float) -> str:
        body = {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM},
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
    ) -> None:
        self.app = app
        self.complete = complete
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
            result = await self._punctuate_unless_gone(body, receive)
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

    async def _punctuate_unless_gone(self, body: bytes, receive) -> tuple[str, bytes] | None:  # noqa: ANN001
        """The pass, raced against the client leaving; None if it left (a tie counts as left).

        Only called once the app task has ended, so nothing else reads `receive` any more.
        """
        work = asyncio.ensure_future(self._punctuate(body))
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

    async def _punctuate(self, body: bytes) -> tuple[str, bytes]:
        try:
            return await self._punctuate_or_raise(body)
        except Exception as e:  # noqa: BLE001 - never worse than Whisper's own answer
            logger.warning("punct: %s", type(e).__name__)
            return "error", body

    async def _punctuate_or_raise(self, body: bytes) -> tuple[str, bytes]:
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
        if not tokens or len(tokens) > self.max_words or not wanted:
            return "skipped", body
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self.inflight >= 1 + self.max_waiting:  # one in the slot, max_waiting behind it
            outcome, out = "busy", body
        else:
            self.inflight += 1
            try:
                outcome, out = await self._ask(payload, tokens, body, begin)
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

    async def _ask(
        self, payload: dict, tokens: list[str], body: bytes, begin: float
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
                    max_tokens = math.ceil(len(payload["text"]) / 2) + 64
                    got = await self.complete(payload["text"], max_tokens, remaining)
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
    return Punctuate(
        app,
        make_complete(url, env.get("GS_PUNCT_MODEL", "default")),
        timeout=float(env.get("GS_PUNCT_TIMEOUT_S", "60")),
        min_run=int(env.get("GS_PUNCT_MIN_RUN", "40")),
        paragraph_min_words=int(env.get("GS_PUNCT_PARAGRAPH_MIN_WORDS", "80")),
        max_words=int(env.get("GS_PUNCT_MAX_WORDS", "3000")),
        max_bytes=int(env.get("GS_PUNCT_MAX_BYTES", str(8 * 1024 * 1024))),
        max_waiting=int(env.get("GS_PUNCT_MAX_WAITING", "2")),
        app_grace=float(env.get("GS_PUNCT_APP_GRACE_S", "5")),
        send_timeout=float(env.get("GS_SEND_TIMEOUT_S", "60")),
    )
