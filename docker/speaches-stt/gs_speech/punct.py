"""Optional punctuation pass by a local LLM, for transcripts Whisper returned without punctuation.

Whisper large-v3 often writes an English piece of a voice note with no punctuation at all, and
Speaches decodes each piece on its own, so a long note comes back partly unpunctuated. This pass
asks an OpenAI-compatible chat model to add punctuation and capitals, and keeps its answer only if
every word is unchanged.

When it acts: POST /v1/audio/transcriptions with the header `X-GS-Punctuate: llm`, a 200 JSON answer
(response_format json or verbose_json), and a stretch of at least GS_PUNCT_MIN_RUN (40) words with
no punctuation mark. Anything else passes through untouched, byte for byte.

The guard: the answer and the transcript are split on whitespace; there must be the same number of
tokens, and each pair must be the same word once case and punctuation are removed. Then `text`, each
segment's `text` and, when they line up one to one, the `words` entries take the punctuated tokens;
timestamps and every other field are unchanged. Otherwise, and on any error, timeout or busy model,
the original answer is sent. The header `X-GS-Punctuated` says what happened: applied, unchanged,
rejected, skipped, busy, timeout or error. The log line has counts and timings only, never text.

It wraps Admission from outside, so the LLM call happens after the Whisper permit is released. One
call at a time (the model has one slot), each bounded by GS_PUNCT_TIMEOUT_S (60) including the wait.

Env: GS_PUNCT_URL (chat completions URL; unset = off), GS_PUNCT_MODEL, GS_PUNCT_TIMEOUT_S,
GS_PUNCT_MIN_RUN, GS_PUNCT_MAX_WORDS (3000; longer transcripts are skipped).
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
MARK = re.compile(r"[.,?!;:…]")
# Tested on 17 of Davron's notes with Qwen3.8-27B: words kept in all 17. Asking also not to "join or
# split" words made the model return the text unpunctuated, so the wording stays as tested.
SYSTEM = (
    "You restore punctuation and capitalization in speech transcripts. Return the same text with "
    "punctuation (full stops, commas, question marks) and capital letters added. Do not add, "
    "remove, replace, reorder or correct any word, and keep filler words and repeats exactly. "
    "Output only the text."
)

Complete = Callable[[str, int, float], Awaitable[str]]


def bare(token: str) -> str:
    """A token without case and punctuation: what must not change."""
    return re.sub(r"[^\w']", "", token.casefold())


def longest_unpunctuated(tokens: list[str]) -> int:
    best = run = 0
    for t in tokens:
        run += 1
        if MARK.search(t):
            best, run = max(best, run), 0
    return max(best, run)


def same_words(src: list[str], got: list[str]) -> bool:
    return len(src) == len(got) and all(bare(a) == bare(b) for a, b in zip(src, got, strict=True))


def rewrite(payload: dict, punctuated: list[str]) -> dict | None:
    """The payload with punctuated tokens in place, or None if they do not line up exactly."""
    src = payload["text"].split()
    if not same_words(src, punctuated):
        return None
    out = dict(payload)
    out["text"] = _replace_tokens(payload["text"], punctuated)
    segments = payload.get("segments")
    if isinstance(segments, list):
        seg_tokens = [str(s.get("text", "")).split() for s in segments]
        if sum(map(len, seg_tokens)) != len(punctuated) or not same_words(
            [t for ts in seg_tokens for t in ts], punctuated
        ):
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


def make_complete(url: str, model: str) -> Complete:
    import httpx

    client = httpx.AsyncClient()

    async def complete(text: str, max_tokens: int, timeout: float) -> str:
        r = await client.post(
            url,
            json={
                "model": model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                ],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"])

    return complete


class Punctuate:
    """ASGI wrapper: buffers an opted-in transcription answer and maybe punctuates it."""

    def __init__(  # noqa: PLR0913
        self,
        app,  # noqa: ANN001
        complete: Complete,
        timeout: float = 60.0,
        min_run: int = 40,
        max_words: int = 3000,
        send_timeout: float = 60.0,
    ) -> None:
        self.app = app
        self.complete = complete
        self.timeout = timeout
        self.min_run = min_run
        self.max_words = max_words
        self.send_timeout = send_timeout
        self._lock: asyncio.Lock | None = None

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != PATH
            or OPT_IN not in [(k, v.strip().lower()) for k, v in scope.get("headers", [])]
        ):
            await self.app(scope, receive, send)
            return
        held: dict = {}
        chunks: list[bytes] = []
        passing = False

        async def capture(message) -> None:  # noqa: ANN001
            nonlocal passing
            if passing:
                await send(message)
                return
            if message["type"] == "http.response.start":
                ctype = dict(message.get("headers", [])).get(b"content-type", b"")
                if message["status"] != 200 or not ctype.startswith(b"application/json"):
                    passing = True
                    await send(message)
                    return
                held["start"] = message
                return
            if message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    held["done"] = True
                return
            await send(message)

        await self.app(scope, receive, capture)
        if passing or "start" not in held:
            return
        body = b"".join(chunks)
        outcome, body = await self._punctuate(body)
        headers = [(k, v) for k, v in held["start"].get("headers", []) if k != b"content-length"]
        headers += [
            (b"content-length", str(len(body)).encode()),
            (b"x-gs-punctuated", outcome.encode()),
        ]

        async def finish() -> None:
            await send({**held["start"], "headers": headers})
            await send({"type": "http.response.body", "body": body})

        await asyncio.wait_for(finish(), timeout=self.send_timeout)

    async def _punctuate(self, body: bytes) -> tuple[str, bytes]:
        start = time.perf_counter()
        try:
            payload = json.loads(body)
            text = payload.get("text") if isinstance(payload, dict) else None
        except (ValueError, UnicodeDecodeError):
            return "skipped", body
        if not isinstance(text, str):
            return "skipped", body
        tokens = text.split()
        if (
            not tokens
            or len(tokens) > self.max_words
            or longest_unpunctuated(tokens) < self.min_run
        ):
            return "skipped", body
        if self._lock is None:
            self._lock = asyncio.Lock()
        outcome, out, holding = "error", body, False
        try:
            async with asyncio.timeout(self.timeout):
                async with self._lock:
                    holding = True
                    remaining = max(1.0, self.timeout - (time.perf_counter() - start))
                    got = await self.complete(
                        text, math.ceil(len(tokens) * 1.4 * 1.3) + 16, remaining
                    )
            new = rewrite(payload, got.split())
            if new is None:
                outcome = "rejected"
            elif got.split() == tokens:
                outcome = "unchanged"
            else:
                outcome, out = "applied", json.dumps(new, ensure_ascii=False).encode()
        except TimeoutError:
            outcome = "timeout" if holding else "busy"
        except Exception as e:  # noqa: BLE001 - any failure keeps Whisper's answer
            logger.warning("punct: %s", type(e).__name__)
        logger.info(
            "punct: %s, %d words, longest unpunctuated %d, %.0f ms",
            outcome,
            len(tokens),
            longest_unpunctuated(tokens),
            (time.perf_counter() - start) * 1000,
        )
        return outcome, out


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
        max_words=int(env.get("GS_PUNCT_MAX_WORDS", "3000")),
        send_timeout=float(env.get("GS_SEND_TIMEOUT_S", "60")),
    )
