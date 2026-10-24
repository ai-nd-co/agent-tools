"""Unit tests for gs_speech.punct. No LLM: the completion is a stand-in. Run inside the image:

docker run --rm --entrypoint python <image> -I -m unittest discover -s /opt/gs-speech/tests -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest

sys.path.insert(0, "/opt/gs-speech")

import httpx  # noqa: E402
from gs_speech import punct  # noqa: E402

PATH = "/v1/audio/transcriptions"

BARE = " ".join(f"word{i}" for i in range(50))  # 50 words, no mark
PUNCT_TOKENS = [f"Word{i}." if i % 10 == 9 else f"word{i}," for i in range(50)]


def verbose(text: str) -> dict:
    tokens = text.split()
    return {
        "task": "transcribe",
        "language": "en",
        "duration": 20.0,
        "text": " " + " ".join(tokens),
        "segments": [
            {"id": 0, "start": 0.0, "end": 10.0, "text": " " + " ".join(tokens[:20])},
            {"id": 1, "start": 10.0, "end": 20.0, "text": " " + " ".join(tokens[20:])},
        ],
        "words": [
            {"word": " " + t, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, t in enumerate(tokens)
        ],
    }


def whisper_app(payload: object, status: int = 200, ctype: bytes = b"application/json"):  # noqa: ANN201
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    async def app(scope, receive, send) -> None:  # noqa: ANN001
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", ctype), (b"content-length", str(len(body)).encode())],
            }
        )
        half = len(body) // 2
        await send({"type": "http.response.body", "body": body[:half], "more_body": True})
        await send({"type": "http.response.body", "body": body[half:]})

    return app


class Fake:
    def __init__(self, answer: str | Exception | None = None, delay: float = 0.0) -> None:
        self.answer, self.delay, self.calls = answer, delay, []

    async def __call__(self, text: str, max_tokens: int, timeout: float) -> str:
        self.calls.append((text, max_tokens, timeout))
        await asyncio.sleep(self.delay)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer if self.answer is not None else " ".join(PUNCT_TOKENS)


def call(app, opt_in: bool = True, path: str = "/v1/audio/transcriptions"):  # noqa: ANN001, ANN201
    async def go():  # noqa: ANN202
        headers = {"x-gs-punctuate": "LLM"} if opt_in else {}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://x") as c:
            return await c.post(path, content=b"audio", headers=headers)

    return asyncio.run(go())


class Rewrite(unittest.TestCase):
    def test_applies_to_text_segments_and_words(self) -> None:
        p = verbose(BARE)
        out = punct.rewrite(p, PUNCT_TOKENS)
        self.assertEqual(out["text"], " " + " ".join(PUNCT_TOKENS))
        self.assertEqual(out["segments"][0]["text"], " " + " ".join(PUNCT_TOKENS[:20]))
        self.assertEqual(out["segments"][1]["text"], " " + " ".join(PUNCT_TOKENS[20:]))
        self.assertEqual(out["words"][9]["word"], " Word9.")
        self.assertEqual([w["start"] for w in out["words"]], [w["start"] for w in p["words"]])
        self.assertEqual(out["segments"][1]["start"], 10.0)
        self.assertEqual(p["text"], " " + BARE)  # the input is not modified

    def test_any_changed_word_is_refused(self) -> None:
        p = verbose(BARE)
        for got in (
            PUNCT_TOKENS[:-1],  # a word dropped
            PUNCT_TOKENS + ["extra"],  # a word added
            ["other,"] + PUNCT_TOKENS[1:],  # a word replaced
            PUNCT_TOKENS[1:2] + PUNCT_TOKENS[:1] + PUNCT_TOKENS[2:],  # reordered
            ["word0word1,"] + PUNCT_TOKENS[2:],  # two words joined
        ):
            self.assertIsNone(punct.rewrite(p, got))

    def test_segments_that_disagree_with_text_change_nothing(self) -> None:
        p = verbose(BARE)
        p["segments"][1]["text"] = " something else"
        self.assertIsNone(punct.rewrite(p, PUNCT_TOKENS))

    def test_words_left_alone_when_not_one_to_one(self) -> None:
        p = verbose(BARE)
        p["words"] = p["words"][:-1]
        out = punct.rewrite(p, PUNCT_TOKENS)
        self.assertEqual(out["words"], p["words"])
        self.assertEqual(out["text"], " " + " ".join(PUNCT_TOKENS))

    def test_plain_json_and_cyrillic(self) -> None:
        out = punct.rewrite({"text": "привет как дела"}, ["Привет,", "как", "дела?"])
        self.assertEqual(out, {"text": "Привет, как дела?"})

    def test_longest_unpunctuated(self) -> None:
        self.assertEqual(punct.longest_unpunctuated("a b. c d e, f".split()), 3)
        self.assertEqual(punct.longest_unpunctuated(BARE.split()), 50)


class Wrapper(unittest.TestCase):
    def test_applied(self) -> None:
        fake = Fake()
        r = call(punct.Punctuate(whisper_app(verbose(BARE)), fake))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["x-gs-punctuated"], "applied")
        self.assertEqual(int(r.headers["content-length"]), len(r.content))
        self.assertEqual(r.json()["text"], " " + " ".join(PUNCT_TOKENS))
        self.assertEqual(fake.calls[0][0], " " + BARE)
        self.assertGreaterEqual(fake.calls[0][1], 50)

    def test_rejected_keeps_original_bytes(self) -> None:
        payload = verbose(BARE)
        r = call(punct.Punctuate(whisper_app(payload), Fake("totally different words")))
        self.assertEqual(r.headers["x-gs-punctuated"], "rejected")
        self.assertEqual(r.content, json.dumps(payload).encode())

    def test_unchanged_when_the_model_adds_nothing(self) -> None:
        payload = verbose(BARE)
        r = call(punct.Punctuate(whisper_app(payload), Fake(BARE)))
        self.assertEqual((r.headers["x-gs-punctuated"], r.json()), ("unchanged", payload))

    def test_error_and_timeout_keep_original(self) -> None:
        payload = verbose(BARE)
        r = call(punct.Punctuate(whisper_app(payload), Fake(RuntimeError("down"))))
        self.assertEqual((r.headers["x-gs-punctuated"], r.json()), ("error", payload))
        r = call(punct.Punctuate(whisper_app(payload), Fake(delay=1.0), timeout=0.2))
        self.assertEqual((r.headers["x-gs-punctuated"], r.json()), ("timeout", payload))

    def test_busy_when_the_slot_is_taken(self) -> None:
        payload = verbose(BARE)
        fake = Fake()
        app = punct.Punctuate(whisper_app(payload), fake, timeout=0.2)

        async def go():  # noqa: ANN202
            app._lock = asyncio.Lock()
            await app._lock.acquire()  # another note is being punctuated
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://x"
            ) as c:
                return await c.post(
                    "/v1/audio/transcriptions", content=b"a", headers={"x-gs-punctuate": "llm"}
                )

        r = asyncio.run(go())
        self.assertEqual((r.headers["x-gs-punctuated"], r.json()), ("busy", payload))
        self.assertEqual(fake.calls, [])

    def test_skipped_when_already_punctuated_or_short(self) -> None:
        fake = Fake()
        for text in ("Hello there. " * 30, "just a few words"):
            r = call(punct.Punctuate(whisper_app({"text": text}), fake))
            self.assertEqual(r.headers["x-gs-punctuated"], "skipped")
            self.assertEqual(r.json(), {"text": text})
        self.assertEqual(fake.calls, [])

    def test_untouched_without_opt_in_or_off_path(self) -> None:
        fake = Fake()
        for kw in ({"opt_in": False}, {"path": "/v1/audio/translations"}):
            r = call(punct.Punctuate(whisper_app(verbose(BARE)), fake), **kw)
            self.assertNotIn("x-gs-punctuated", r.headers)
            self.assertEqual(r.json()["text"], " " + BARE)
        self.assertEqual(fake.calls, [])

    def test_non_json_and_errors_pass_through(self) -> None:
        fake = Fake()
        r = call(punct.Punctuate(whisper_app(BARE.encode(), ctype=b"text/plain"), fake))
        self.assertEqual((r.text, r.headers.get("x-gs-punctuated")), (BARE, None))
        r = call(punct.Punctuate(whisper_app({"detail": "x"}, status=503), fake))
        self.assertEqual((r.status_code, r.headers.get("x-gs-punctuated")), (503, None))
        r = call(punct.Punctuate(whisper_app(b"not json"), fake))
        self.assertEqual((r.content, r.headers["x-gs-punctuated"]), (b"not json", "skipped"))
        self.assertEqual(fake.calls, [])

    def test_wrap_is_off_without_url(self) -> None:
        app = whisper_app({"text": "x"})
        self.assertIs(punct.wrap(app, {}), app)
        self.assertIsInstance(
            punct.wrap(app, {"GS_PUNCT_URL": "http://llm/v1/chat/completions"}), punct.Punctuate
        )


def raw_call(app, receive_script, timeout: float = 5.0):  # noqa: ANN001, ANN201
    """Drive `app` directly: receive_script yields ASGI messages (awaiting as it likes)."""
    sent: list[dict] = []

    async def go() -> None:
        it = receive_script()

        async def receive():  # noqa: ANN202
            return await it.__anext__()

        async def send(message) -> None:  # noqa: ANN001
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/transcriptions",
            "headers": [(b"x-gs-punctuate", b"llm")],
        }
        await asyncio.wait_for(app(scope, receive, send), timeout)

    asyncio.run(go())
    return sent


class Hardening(unittest.TestCase):
    """Regressions for the Codex audit, round 1."""

    def test_guard_refuses_changes_inside_a_token(self) -> None:
        for a, b in (
            ("we\u2019ll", "well"),
            ("1.5", "15"),
            ("re-sign", "resign"),
            ("-5", "5"),
            ("cafe\u0301", "cafe"),
            ("it's", "its"),
            ("...", ""),
        ):
            self.assertFalse(punct.same_words([a], [b] if b else []), (a, b))
            self.assertFalse(punct.same_words([a], [b + "."] if b else ["."]), (a, b))
        self.assertTrue(
            punct.same_words(["hello", "us", "1.5", "co-op"], ["Hello,", "US.", "1.5?", "co-op!"])
        )
        self.assertTrue(punct.same_words(["\u2026and"], ["And"]))

    def test_app_error_after_the_answer_keeps_the_answer(self) -> None:
        inner = whisper_app(verbose(BARE))

        async def app(scope, receive, send) -> None:  # noqa: ANN001
            await inner(scope, receive, send)
            raise RuntimeError("cleanup failed")

        r = call(punct.Punctuate(app, Fake()))
        self.assertEqual(r.headers["x-gs-punctuated"], "applied")

    def test_app_hanging_after_the_answer_is_cancelled(self) -> None:
        inner = whisper_app(verbose(BARE))
        cancelled = []

        async def app(scope, receive, send) -> None:  # noqa: ANN001
            await inner(scope, receive, send)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        t0 = time.perf_counter()
        r = call(punct.Punctuate(app, Fake(), app_grace=0.2))
        self.assertEqual(r.headers["x-gs-punctuated"], "applied")
        self.assertLess(time.perf_counter() - t0, 3)
        self.assertEqual(cancelled, [True])

    def test_incomplete_answers_become_500(self) -> None:
        async def truncated(scope, receive, send) -> None:  # noqa: ANN001
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b'{"text": "par', "more_body": True})

        async def start_only(scope, receive, send) -> None:  # noqa: ANN001
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )

        fake = Fake()
        for app in (truncated, start_only):
            r = call(punct.Punctuate(app, fake))
            self.assertEqual(r.status_code, 500)
            self.assertEqual(r.json(), {"detail": "Incomplete transcription response."})
        self.assertEqual(fake.calls, [])

    def test_app_error_before_the_answer_propagates(self) -> None:
        async def broken(scope, receive, send) -> None:  # noqa: ANN001
            raise RuntimeError("whisper failed")

        with self.assertRaises(RuntimeError):
            call(punct.Punctuate(broken, Fake()))

    def test_oversized_answer_streams_through_unchanged(self) -> None:
        payload = verbose(BARE)
        fake = Fake()
        r = call(punct.Punctuate(whisper_app(payload), fake, max_bytes=100))
        self.assertNotIn("x-gs-punctuated", r.headers)
        self.assertEqual(r.content, json.dumps(payload).encode())
        self.assertEqual(fake.calls, [])

    def test_queue_cap_answers_busy_at_once(self) -> None:
        payload = verbose(BARE)
        fake = Fake()
        app = punct.Punctuate(whisper_app(payload), fake, timeout=5, max_waiting=1)
        app.inflight = 2  # one note in the slot, one waiting: the cap

        t0 = time.perf_counter()
        r = call(app)
        self.assertEqual((r.headers["x-gs-punctuated"], r.json()), ("busy", payload))
        self.assertLess(time.perf_counter() - t0, 1)
        self.assertEqual((app.inflight, fake.calls), (2, []))

    def test_cap_holds_across_a_lock_handoff(self) -> None:
        payload = verbose(BARE)
        release = asyncio.Event()

        async def gated(text: str, max_tokens: int, timeout: float) -> str:
            await release.wait()
            return text

        app = punct.Punctuate(whisper_app(payload), gated, timeout=5, max_waiting=1)

        async def go():  # noqa: ANN202
            h = {"x-gs-punctuate": "llm"}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://x"
            ) as c:
                first = [
                    asyncio.ensure_future(c.post(PATH, content=b"a", headers=h)) for _ in range(2)
                ]
                while app.inflight < 2:
                    await asyncio.sleep(0.01)
                release.set()  # the slot changes hands; a newcomer must still see the cap
                late = await c.post(PATH, content=b"b", headers=h)
                return [await f for f in first], late

        first, late = asyncio.run(go())
        self.assertEqual(
            sorted(r.headers["x-gs-punctuated"] for r in first), ["unchanged", "unchanged"]
        )
        self.assertIn(late.headers["x-gs-punctuated"], ("busy", "unchanged"))
        self.assertEqual(app.inflight, 0)

    def test_inflight_returns_to_zero(self) -> None:
        app = punct.Punctuate(whisper_app(verbose(BARE)), Fake(delay=0.5), timeout=0.2)
        call(app)
        self.assertEqual(app.inflight, 0)

    def test_outer_cancel_during_grace_stops_the_app(self) -> None:
        inner = whisper_app(verbose(BARE))
        cancelled = []

        async def app(scope, receive, send) -> None:  # noqa: ANN001
            await inner(scope, receive, send)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def script():  # noqa: ANN202
            yield {"type": "http.request", "body": b"audio", "more_body": False}
            await asyncio.sleep(30)
            yield {"type": "http.disconnect"}

        with self.assertRaises(TimeoutError):
            raw_call(punct.Punctuate(app, Fake(), app_grace=10), script, timeout=0.3)
        self.assertEqual(cancelled, [True])

    def test_overflow_then_hang_is_cancelled_after_grace(self) -> None:
        payload = verbose(BARE)
        inner = whisper_app(payload)
        cancelled = []

        async def app(scope, receive, send) -> None:  # noqa: ANN001
            await inner(scope, receive, send)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        t0 = time.perf_counter()
        r = call(punct.Punctuate(app, Fake(), max_bytes=100, app_grace=0.2))
        self.assertEqual(r.content, json.dumps(payload).encode())
        self.assertLess(time.perf_counter() - t0, 3)
        self.assertEqual(cancelled, [True])

    def test_llm_answer_size_is_capped(self) -> None:
        big = b'{"choices":[{"message":{"content":"' + b"x" * (punct.MAX_LLM_BYTES + 10) + b'"}}]}'
        ok = b'{"choices":[{"message":{"content":"Hello."}}]}'

        def complete_with(answer: bytes):  # noqa: ANN202
            t = httpx.MockTransport(
                lambda req: httpx.Response(200, stream=httpx.ByteStream(answer))
            )
            return punct.make_complete("http://llm/v1/chat/completions", "m", transport=t)

        self.assertEqual(asyncio.run(complete_with(ok)("hello", 10, 5)), "Hello.")
        with self.assertRaises(ValueError):
            asyncio.run(complete_with(big)("hello", 10, 5))

    def test_disconnect_cancels_the_pass_and_sends_nothing(self) -> None:
        started, cancelled = asyncio.Event(), []

        async def slow(text: str, max_tokens: int, timeout: float) -> str:
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return text

        async def script():  # noqa: ANN202
            yield {"type": "http.request", "body": b"audio", "more_body": False}
            await started.wait()
            yield {"type": "http.disconnect"}

        sent = raw_call(punct.Punctuate(whisper_app(verbose(BARE)), slow), script)
        self.assertEqual(sent, [])
        self.assertEqual(cancelled, [True])

    def test_stop_finishes_cleanup_before_passing_on_a_cancel(self) -> None:
        cleaned = []

        async def child() -> None:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.sleep(0.2)  # slow cleanup
                cleaned.append(True)
                raise

        async def go() -> None:
            c = asyncio.ensure_future(child())
            await asyncio.sleep(0)
            stopper = asyncio.ensure_future(punct._stop(c))
            await asyncio.sleep(0.05)
            stopper.cancel()  # the caller is cancelled while the child cleans up
            with self.assertRaises(asyncio.CancelledError):
                await stopper
            self.assertTrue(c.done())

        asyncio.run(go())
        self.assertEqual(cleaned, [True])

    def test_watcher_failure_still_answers(self) -> None:
        payload = verbose(BARE)

        async def script():  # noqa: ANN202
            yield {"type": "http.request", "body": b"audio", "more_body": False}
            raise RuntimeError("receive broke")

        sent = raw_call(punct.Punctuate(whisper_app(payload), Fake(delay=1.0)), script)
        headers = dict(sent[0]["headers"])
        self.assertEqual(headers[b"x-gs-punctuated"], b"error")
        self.assertEqual(sent[1]["body"], json.dumps(payload).encode())

    def test_compressed_llm_answer_is_refused(self) -> None:
        t = httpx.MockTransport(
            lambda req: httpx.Response(200, content=b"x", headers={"content-encoding": "gzip"})
        )
        complete = punct.make_complete("http://llm/v1/chat/completions", "m", transport=t)
        with self.assertRaises(ValueError):
            asyncio.run(complete("hello", 10, 5))


if __name__ == "__main__":
    unittest.main()
