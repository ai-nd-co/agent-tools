"""Unit tests for gs_speech. No GPU: Whisper is a stand-in. Run inside the image:

docker run --rm --entrypoint python <image> -I -m unittest discover -s /opt/gs-speech/tests -v
"""

from __future__ import annotations

import asyncio
import io
import sys
import threading
import time
import unittest
import wave
from contextlib import contextmanager

sys.path.insert(0, "/opt/gs-speech")

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from gs_speech import ext, langid  # noqa: E402
from gs_speech.serve import PortRouter  # noqa: E402


def wav_bytes(seconds: float = 1.0) -> bytes:
    t = np.arange(int(16000 * seconds)) / 16000
    pcm = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


class FakeWhisper:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0

    def detect_language(self, audio):  # noqa: ANN001, ANN201
        self.calls += 1
        assert audio.dtype == np.float32 and audio.ndim == 1
        time.sleep(self.delay)
        return "ru", 0.9, [("ru", 0.9), ("en", 0.07), ("uz", 0.03)]


class FakeModel:
    model_id = "Systran/faster-whisper-large-v3"
    compute_type = "int8"
    revision = "edaa852"

    def __init__(self, whisper: FakeWhisper | None) -> None:
        self.whisper = whisper

    @contextmanager
    def use(self):  # noqa: ANN201
        yield self.whisper


class AllowlistTest(unittest.TestCase):
    def test_parse(self) -> None:
        self.assertIsNone(ext.parse_allowlist(None))
        self.assertIsNone(ext.parse_allowlist("  "))
        got = ext.parse_allowlist("a/b, c/d\ne")
        self.assertEqual(got, frozenset({"a/b", "c/d", "e", "silero_vad_v5"}))

    def test_refuse(self) -> None:
        allow = ext.parse_allowlist("Systran/faster-whisper-large-v3")
        ext.refuse_unless_allowed(allow, "Systran/faster-whisper-large-v3")
        ext.refuse_unless_allowed(allow, "silero_vad_v5")
        ext.refuse_unless_allowed(None, "anything/at-all")
        with self.assertRaises(HTTPException) as cm:
            ext.refuse_unless_allowed(allow, "Systran/faster-whisper-tiny")
        self.assertEqual(cm.exception.status_code, 403)


class LanguageSetTest(unittest.TestCase):
    PROBS = [("lv", 0.321), ("en", 0.264), ("lt", 0.208), ("ru", 0.022)]

    def test_parse(self) -> None:
        self.assertIsNone(ext.parse_language_set(""))
        self.assertEqual(ext.parse_language_set("EN, ru"), frozenset({"en", "ru"}))

    def test_pick(self) -> None:
        d = ("lv", 0.321, self.PROBS)
        self.assertEqual(ext.pick_language(d, frozenset({"en", "ru"}))[:2], ("en", 0.264))
        self.assertIs(ext.pick_language(d, None), d)
        self.assertIs(ext.pick_language(d, frozenset({"lv", "en"})), d)  # detected one is allowed
        self.assertIs(ext.pick_language(d, frozenset({"uz"})), d)  # nothing allowed among the probs

    def test_only_inside_a_transcription(self) -> None:
        seen = []

        def handler() -> None:
            seen.append(getattr(ext._restrict, "langs", None))  # noqa: SLF001

        handler()
        ext.with_languages(handler, frozenset({"en"}))()
        handler()
        self.assertEqual(seen, [None, frozenset({"en"}), None])

    def test_generator_per_item_and_close(self) -> None:
        seen, closed = [], []

        def gen():  # noqa: ANN202
            try:
                for _ in range(3):
                    seen.append(getattr(ext._restrict, "langs", None))  # noqa: SLF001
                    yield 1
            finally:
                closed.append(True)

        g = ext.with_languages_generator(gen, frozenset({"ru"}))()
        next(g)
        self.assertIsNone(getattr(ext._restrict, "langs", None))  # noqa: SLF001 - not leaked between items
        next(g)
        g.close()
        self.assertEqual(seen, [frozenset({"ru"})] * 2)
        self.assertEqual(closed, [True])


class BatchSizeTest(unittest.TestCase):
    def test_default_only(self) -> None:
        seen = []

        def transcribe(self, audio, language=None, batch_size=8):  # noqa: ANN001, ANN202 - faster-whisper's shape
            seen.append(batch_size)

        wrapped = ext.with_batch_size(transcribe, 2)
        wrapped(None, "a")
        wrapped(None, "a", batch_size=8)
        wrapped(None, "a", None, 4)  # positional
        self.assertEqual(seen, [2, 8, 4])

    def test_real_signature(self) -> None:
        from faster_whisper import BatchedInferencePipeline

        self.assertIn(
            "batch_size",
            __import__("inspect").signature(BatchedInferencePipeline.transcribe).parameters,
        )


class InstallTest(unittest.TestCase):
    """Against the real Speaches classes in this image."""

    def test_install_wraps_and_refuses(self) -> None:
        from faster_whisper import WhisperModel
        from speaches.executors.shared.base_model_manager import BaseModelManager
        from speaches.executors.whisper import WhisperModelManager

        orig_detect = WhisperModel.detect_language
        state = ext.install(
            {
                "GS_WHISPER_PERMITS": "1",
                "GS_MODEL_ALLOWLIST": "Systran/faster-whisper-large-v3",
                "GS_STT_LANGUAGES": "en,ru",
            }
        )
        self.assertIs(ext.install({}), state)  # second call is a no-op
        self.assertEqual(state.langs, frozenset({"en", "ru"}))
        self.assertTrue(
            hasattr(WhisperModelManager.handle_non_streaming_transcription_request, "__wrapped__")
        )
        self.assertIs(WhisperModel.detect_language.__wrapped__, orig_detect)

        class Dummy(BaseModelManager):
            def _load_fn(self, model_id: str) -> str:
                return model_id

        mgr = Dummy(ttl=-1)
        self.assertEqual(
            mgr.load_model("Systran/faster-whisper-large-v3").model_id,
            "Systran/faster-whisper-large-v3",
        )
        self.assertEqual(
            mgr.load_model("whisper-1").model_id, "whisper-1"
        )  # alias of the allowed id
        with self.assertRaises(HTTPException) as cm:
            mgr.load_model("Systran/faster-whisper-tiny")
        self.assertEqual(cm.exception.status_code, 403)

    def test_stream_refused_before_it_starts(self) -> None:
        from types import SimpleNamespace

        from speaches.executors.whisper import WhisperModelManager

        ext.install({})
        if ext.STATE.allow is None:
            self.skipTest("installed without an allowlist")
        with self.assertRaises(
            HTTPException
        ) as cm:  # raised by the call itself, not on first iteration
            WhisperModelManager.handle_transcription_request(
                object(), SimpleNamespace(model="Systran/faster-whisper-tiny", stream=True)
            )
        self.assertEqual(cm.exception.status_code, 403)

    def test_restricting_detect(self) -> None:
        detect = ext.restricting_detect(lambda self, **kw: ("lv", 0.321, LanguageSetTest.PROBS))
        self.assertEqual(detect(None, audio=None)[0], "lv")  # the language-ID listener's view
        with ext.restricted_languages(frozenset({"en", "ru"})):
            self.assertEqual(detect(None, audio=None)[:2], ("en", 0.264))  # inside a transcription
        self.assertEqual(detect(None, audio=None)[0], "lv")


class LangIdTest(unittest.TestCase):
    def client(self, whisper: FakeWhisper | None, **env: str) -> TestClient:
        return TestClient(langid.make_app(env=env, model=FakeModel(whisper)))

    def post(self, c: TestClient, data: bytes, ctype: str = "audio/wav", **kw):  # noqa: ANN003, ANN201
        return c.post("/v1/langid", files={"file": ("a", data, ctype)}, **kw)

    def test_ok_wav(self) -> None:
        w = FakeWhisper()
        r = self.post(self.client(w), wav_bytes())
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["language"], "ru")
        self.assertEqual(body["probabilities"], {"ru": 0.9, "en": 0.07, "uz": 0.03})
        self.assertEqual(
            (body["model"], body["revision"]), ("Systran/faster-whisper-large-v3", "edaa852")
        )
        self.assertAlmostEqual(body["audio_seconds"], 1.0, places=2)
        self.assertEqual(w.calls, 1)

    def test_wav_matches_speaches_decoder(self) -> None:
        from speaches.dependencies import audio_file_dependency
        from starlette.datastructures import Headers, UploadFile

        data = wav_bytes(2.0)
        ours = langid.decode(data, "audio/wav")
        theirs = audio_file_dependency(
            UploadFile(
                io.BytesIO(data), filename="a.wav", headers=Headers({"content-type": "audio/wav"})
            )
        ).data
        self.assertEqual(ours.shape, theirs.shape)
        self.assertLess(float(np.max(np.abs(ours - theirs))), 1e-3)

    def test_first_30_seconds_only(self) -> None:
        self.assertEqual(
            len(langid.decode(np.zeros(16000 * 40, "<i2").tobytes(), "audio/pcm")), 16000 * 30
        )

    def test_pcm(self) -> None:
        r = self.post(self.client(FakeWhisper()), np.zeros(8000, "<i2").tobytes(), "audio/pcm")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertAlmostEqual(r.json()["audio_seconds"], 0.5, places=2)

    def test_bad_inputs(self) -> None:
        c = self.client(FakeWhisper())
        self.assertEqual(self.post(c, b"", "audio/pcm").status_code, 400)  # empty
        self.assertEqual(self.post(c, b"\x00\x01\x02", "audio/pcm").status_code, 400)  # odd length
        self.assertEqual(self.post(c, b"not audio").status_code, 415)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 800)
        self.assertEqual(self.post(c, buf.getvalue()).status_code, 415)  # 8 kHz
        self.assertEqual(
            self.post(c, wav_bytes()[:-1]).status_code, 400
        )  # truncated to an odd length
        self.assertEqual(c.post("/v1/langid", data={"x": "1"}).status_code, 400)  # no file field
        self.assertEqual(
            self.post(c, wav_bytes()[:-2]).status_code, 400
        )  # truncated by an even count
        riff = (
            b"RIFF"
            + (100).to_bytes(4, "little")
            + b"WAVE"
            + b"junk"
            + (10**6).to_bytes(4, "little")
            + b"\0" * 8
        )
        self.assertEqual(self.post(c, riff).status_code, 415)  # chunk past the RIFF boundary
        self.assertEqual(
            self.post(c, b"\0" * (16000 * 2 * 31 + 1), "audio/pcm").status_code, 400
        )  # odd, past 30 s
        self.assertEqual(c.app.state.slots.used, 0)

    def test_not_loaded(self) -> None:
        self.assertEqual(self.post(self.client(None), wav_bytes()).status_code, 503)
        self.assertEqual(self.client(None).get("/health").json()["model_loaded"], False)

    def test_too_big_refused_before_reading(self) -> None:
        r = self.post(self.client(FakeWhisper(), GS_LID_MAX_BYTES="1000"), wav_bytes())
        self.assertEqual(r.status_code, 413)

    def test_key(self) -> None:
        c = self.client(FakeWhisper(), GS_LID_API_KEY="k" * 40)
        self.assertEqual(self.post(c, wav_bytes()).status_code, 401)
        self.assertEqual(
            self.post(c, wav_bytes(), headers={"Authorization": "Bearer " + "k" * 40}).status_code,
            200,
        )

    def test_request_id(self) -> None:
        c = self.client(FakeWhisper())
        self.assertEqual(
            self.post(c, wav_bytes(), headers={"X-Request-Id": "job-42:a"}).json()["request_id"],
            "job-42:a",
        )
        self.assertIsNone(
            self.post(c, wav_bytes(), headers={"X-Request-Id": "bad id"}).json()["request_id"]
        )
        self.assertIsNone(self.post(c, wav_bytes()).json()["request_id"])

    def test_busy_cap_and_release(self) -> None:
        app = langid.make_app(
            env={"GS_LID_MAX_OUTSTANDING": "1"}, model=FakeModel(FakeWhisper(delay=0.5))
        )
        results: list[int] = []

        def call() -> None:
            with TestClient(app) as c:
                results.append(self.post(c, wav_bytes()).status_code)

        first = threading.Thread(target=call)
        first.start()
        time.sleep(0.2)
        call()
        first.join()
        self.assertEqual(sorted(results), [200, 503])
        time.sleep(0.05)
        self.assertEqual(app.state.slots.used, 0)

    def test_slot_released_when_detection_fails(self) -> None:
        class Boom(FakeWhisper):
            def detect_language(self, audio):  # noqa: ANN001, ANN201
                raise RuntimeError("cuda")

        app = langid.make_app(env={}, model=FakeModel(Boom()))
        with TestClient(app, raise_server_exceptions=False) as c:
            self.assertEqual(self.post(c, wav_bytes()).status_code, 500)
        time.sleep(0.05)
        self.assertEqual(app.state.slots.used, 0)


class BorrowTest(unittest.TestCase):
    """SpeachesWhisper.use never calls the lazy loader and holds a reference while in use."""

    def model(self, sdm):  # noqa: ANN001, ANN201
        m = langid.SpeachesWhisper("m", "int8")
        m._manager = lambda: type("M", (), {"loaded_models": {"m": sdm} if sdm else {}})()  # noqa: SLF001
        return m

    def test_borrow(self) -> None:
        from speaches.executors.shared.base_model_manager import SelfDisposingModel

        loads = []
        sdm = SelfDisposingModel("m", load_fn=lambda: loads.append(1) or "weights", ttl=-1)
        with self.model(sdm).use() as w:  # registered but not loaded: no load
            self.assertIsNone(w)
        self.assertEqual(loads, [])
        sdm.model = "weights"
        with self.model(sdm).use() as w:
            self.assertEqual((w, sdm.ref_count), ("weights", 1))
        self.assertEqual(sdm.ref_count, 0)
        with self.model(None).use() as w:  # not registered
            self.assertIsNone(w)


class AdmissionTest(unittest.TestCase):
    KEY = "k" * 40

    def app(self, max_body: int = 1 << 20, **kw):  # noqa: ANN003, ANN201
        inner = FastAPI()
        state = {"now": 0, "peak": 0, "bodies": [], "calls": 0}

        @inner.post("/v1/audio/transcriptions")
        async def t(request: Request) -> str:
            state["calls"] += 1
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            state["bodies"].append(await request.body())
            await asyncio.sleep(0.1)
            state["now"] -= 1
            return "ok"

        inner.get("/api/ps")(lambda: "ps")
        inner.post("/v1/models/{m:path}")(lambda m: "dl")
        return ext.Admission(inner, 1, self.KEY, max_body, **kw), state

    def run_async(self, app, calls):  # noqa: ANN001, ANN201
        import httpx

        async def go():  # noqa: ANN202
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://x"
            ) as c:
                return await asyncio.gather(*(call(c) for call in calls))

        return asyncio.run(go())

    def auth(self) -> dict:
        return {"Authorization": "Bearer " + self.KEY}

    def test_one_at_a_time_and_body_intact(self) -> None:
        app, state = self.app()
        calls = [
            lambda c, i=i: c.post(
                "/v1/audio/transcriptions", content=b"x" * (1000 + i), headers=self.auth()
            )
            for i in range(4)
        ]
        self.assertEqual([r.status_code for r in self.run_async(app, calls)], [200] * 4)
        self.assertEqual(state["peak"], 1)
        self.assertEqual(sorted(len(b) for b in state["bodies"]), [1000, 1001, 1002, 1003])

    def test_slow_upload_holds_nothing(self) -> None:
        app, state = self.app()

        async def slow_body():  # noqa: ANN202
            yield b"a"
            await asyncio.sleep(0.6)  # a phone on a bad link
            yield b"b"

        t0 = []

        async def fast(c):  # noqa: ANN001, ANN202
            await asyncio.sleep(0.05)
            start = time.perf_counter()
            r = await c.post("/v1/audio/transcriptions", content=b"z", headers=self.auth())
            t0.append(time.perf_counter() - start)
            return r

        rs = self.run_async(
            app,
            [
                lambda c: c.post(
                    "/v1/audio/transcriptions", content=slow_body(), headers=self.auth()
                ),
                fast,
            ],
        )
        self.assertEqual([r.status_code for r in rs], [200, 200])
        self.assertLess(t0[0], 0.4)  # did not wait for the slow upload
        self.assertIn(b"ab", state["bodies"])

    def test_body_cap(self) -> None:
        app, _ = self.app(max_body=100)
        r = self.run_async(
            app,
            [lambda c: c.post("/v1/audio/transcriptions", content=b"x" * 101, headers=self.auth())],
        )[0]
        self.assertEqual(r.status_code, 413)

    def test_large_body_replayed_in_pieces_intact(self) -> None:
        app, state = self.app(max_body=1 << 22)
        data = bytes(range(256)) * 4096  # 1 MiB, 16 replay pieces
        r = self.run_async(
            app, [lambda c: c.post("/v1/audio/transcriptions", content=data, headers=self.auth())]
        )[0]
        self.assertEqual(r.status_code, 200)
        self.assertEqual(state["bodies"], [data])

    def test_auth_like_speaches(self) -> None:
        app, state = self.app()
        rs = self.run_async(
            app,
            [
                lambda c: c.post("/v1/audio/transcriptions", content=b"x"),
                lambda c: c.post(
                    "/v1/audio/transcriptions",
                    content=b"x",
                    headers={"Authorization": "Bearer wrong"},
                ),
                lambda c: c.post(
                    "/v1/audio/transcriptions",
                    content=b"x",
                    headers={"Authorization": "bearer " + self.KEY},
                ),
            ],
        )
        self.assertEqual(
            [r.status_code for r in rs], [403, 403, 200]
        )  # scheme in any case, as HTTPBearer
        self.assertEqual(state["calls"], 1)

    def test_too_many_pending(self) -> None:
        app, _ = self.app(max_pending=2)
        calls = [
            lambda c: c.post("/v1/audio/transcriptions", content=b"x", headers=self.auth())
            for _ in range(4)
        ]
        codes = sorted(r.status_code for r in self.run_async(app, calls))
        self.assertEqual(codes, [200, 200, 503, 503])
        self.assertEqual(app.pending, 0)

    def test_disconnect_while_queued_is_dropped(self) -> None:
        app, state = self.app()

        async def go() -> None:
            async def holder_receive():  # noqa: ANN202
                return {"type": "http.request", "body": b"a", "more_body": False}

            sent = []

            async def send(m) -> None:  # noqa: ANN001
                sent.append(m)

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/v1/audio/transcriptions",
                "headers": [(b"authorization", ("Bearer " + self.KEY).encode())],
                "query_string": b"",
                "root_path": "",
                "scheme": "http",
                "server": ("x", 80),
                "client": ("c", 1),
                "http_version": "1.1",
                "raw_path": b"/v1/audio/transcriptions",
            }
            first = asyncio.create_task(app(scope, holder_receive, send))
            await asyncio.sleep(0.02)  # the first one holds the permit for 0.1 s
            msgs = [{"type": "http.request", "body": b"b", "more_body": False}]

            async def leaving_receive():  # noqa: ANN202
                if msgs:
                    return msgs.pop(0)
                await asyncio.sleep(0.01)
                return {"type": "http.disconnect"}

            await app(scope, leaving_receive, send)
            await first

        asyncio.run(go())
        self.assertEqual(state["calls"], 1)  # the second never reached the app
        self.assertEqual((app.pending, app.running), (0, 0))
        self.assertFalse(app._sem.locked())  # noqa: SLF001 - the permit is free again

    def test_disconnect_wins_a_tie_with_the_permit(self) -> None:
        app, state = self.app()

        async def go() -> None:
            msgs = [
                {"type": "http.request", "body": b"a", "more_body": False},
                {"type": "http.disconnect"},
            ]

            async def receive():  # noqa: ANN202 - the disconnect is already there when the permit is free
                return msgs.pop(0) if msgs else {"type": "http.disconnect"}

            async def send(m) -> None:  # noqa: ANN001
                pass

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/v1/audio/transcriptions",
                "headers": [(b"authorization", ("Bearer " + self.KEY).encode())],
                "query_string": b"",
                "root_path": "",
                "scheme": "http",
                "server": ("x", 80),
                "client": ("c", 1),
                "http_version": "1.1",
                "raw_path": b"/v1/audio/transcriptions",
            }
            await app(scope, receive, send)

        asyncio.run(go())
        self.assertEqual(state["calls"], 0)
        self.assertFalse(app._sem.locked())  # noqa: SLF001
        self.assertEqual((app.pending, app.running), (0, 0))

    def test_cancelled_while_queued_strands_nothing(self) -> None:
        app, state = self.app()

        async def go() -> None:
            app._sem = asyncio.Semaphore(1)  # noqa: SLF001
            await app._sem.acquire()  # noqa: SLF001 - someone else holds the permit

            async def receive():  # noqa: ANN202 - the client stays
                await asyncio.sleep(10)
                return {"type": "http.disconnect"}

            task = asyncio.create_task(app._acquire_unless_gone(receive))  # noqa: SLF001
            await asyncio.sleep(0.05)
            app._sem.release()  # noqa: SLF001 - the permit is granted to the waiter...
            task.cancel()  # ...and the request is cancelled in the same instant
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.05)
            self.assertFalse(app._sem.locked())  # noqa: SLF001 - nobody holds it

        asyncio.run(go())

    def test_stalled_reader_releases_the_permit(self) -> None:
        app, state = self.app(send_timeout=0.2)

        async def go() -> None:
            async def receive():  # noqa: ANN202
                return {"type": "http.request", "body": b"a", "more_body": False}

            async def stuck_send(m) -> None:  # noqa: ANN001
                await asyncio.sleep(10)

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/v1/audio/transcriptions",
                "headers": [(b"authorization", ("Bearer " + self.KEY).encode())],
                "query_string": b"",
                "root_path": "",
                "scheme": "http",
                "server": ("x", 80),
                "client": ("c", 1),
                "http_version": "1.1",
                "raw_path": b"/v1/audio/transcriptions",
            }
            with self.assertRaises(TimeoutError):
                await app(scope, receive, stuck_send)

        asyncio.run(go())
        self.assertFalse(app._sem.locked())  # noqa: SLF001
        self.assertEqual((app.pending, app.running), (0, 0))

    def test_model_admin_blocked(self) -> None:
        app, _ = self.app()
        c = TestClient(app)
        self.assertEqual(c.post("/api/ps/Systran/faster-whisper-large-v3").status_code, 403)
        self.assertEqual(c.delete("/api/ps/Systran/faster-whisper-large-v3").status_code, 403)
        self.assertEqual(c.delete("/v1/models/Systran/faster-whisper-large-v3").status_code, 403)
        self.assertEqual(c.get("/api/ps").status_code, 200)  # the probe's read
        self.assertEqual(
            c.post("/v1/models/Systran/faster-whisper-large-v3").status_code, 200
        )  # the warm-up


class PortRouterTest(unittest.TestCase):
    def test_dispatch_by_local_port(self) -> None:
        main, lid = FastAPI(), FastAPI()
        main.get("/who")(lambda: "main")
        lid.get("/who")(lambda: "lid")
        router = PortRouter(main, lid, 8002)
        self.assertEqual(TestClient(router, base_url="http://x:8000").get("/who").json(), "main")
        self.assertEqual(TestClient(router, base_url="http://x:8002").get("/who").json(), "lid")
        off = PortRouter(main, None, 0)
        self.assertEqual(TestClient(off, base_url="http://x:8002").get("/who").json(), "main")


if __name__ == "__main__":
    unittest.main()
