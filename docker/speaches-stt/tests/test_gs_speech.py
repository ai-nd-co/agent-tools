"""Unit tests for gs_speech. No GPU: Whisper is a stand-in. Run inside the image:

    docker run --rm --entrypoint python <image> -I -m unittest discover -s /opt/gs-speech/tests -v
"""

from __future__ import annotations

from contextlib import contextmanager
import io
import sys
import threading
import time
import unittest
import wave

sys.path.insert(0, "/opt/gs-speech")

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
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


class GateTest(unittest.TestCase):
    def test_one_permit_serialises(self) -> None:
        gate = ext.Gate(1)
        peak = []

        def job() -> None:
            with gate.hold("t"):
                peak.append(gate.running)
                time.sleep(0.05)

        threads = [threading.Thread(target=job) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(max(peak), 1)
        self.assertEqual((gate.running, gate.waiting), (0, 0))

    def test_generator_holds_until_closed(self) -> None:
        gate = ext.Gate(1)

        def gen():  # noqa: ANN202
            yield 1
            yield 2

        wrapped = ext.gated_generator(gen, gate, "t")()
        self.assertEqual(gate.running, 0)  # nothing taken before the first item
        next(wrapped)
        self.assertEqual(gate.running, 1)
        wrapped.close()  # a client hanging up
        self.assertEqual(gate.running, 0)

    def test_released_on_error(self) -> None:
        gate = ext.Gate(1)

        def boom() -> None:
            raise RuntimeError

        with self.assertRaises(RuntimeError):
            ext.gated(boom, gate, "t")()
        self.assertTrue(gate._sem.acquire(blocking=False))  # noqa: SLF001


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


class InstallTest(unittest.TestCase):
    """Against the real Speaches classes in this image."""

    def test_install_wraps_and_refuses(self) -> None:
        from speaches.executors.shared.base_model_manager import BaseModelManager
        from speaches.executors.whisper import WhisperModelManager

        from faster_whisper import WhisperModel

        orig_detect = WhisperModel.detect_language
        state = ext.install(
            {"GS_WHISPER_PERMITS": "1", "GS_MODEL_ALLOWLIST": "Systran/faster-whisper-large-v3", "GS_STT_LANGUAGES": "en,ru"}
        )
        self.assertIs(ext.install({}), state)  # second call is a no-op
        self.assertEqual(state.langs, frozenset({"en", "ru"}))
        self.assertTrue(hasattr(WhisperModelManager.handle_non_streaming_transcription_request, "__wrapped__"))
        self.assertTrue(hasattr(WhisperModelManager.handle_translation_request, "__wrapped__"))
        self.assertIs(WhisperModel.detect_language.__wrapped__, orig_detect)

        class Dummy(BaseModelManager):
            def _load_fn(self, model_id: str) -> str:
                return model_id

        mgr = Dummy(ttl=-1)
        self.assertEqual(mgr.load_model("Systran/faster-whisper-large-v3").model_id, "Systran/faster-whisper-large-v3")
        self.assertEqual(mgr.load_model("whisper-1").model_id, "whisper-1")  # alias of the allowed id
        with self.assertRaises(HTTPException) as cm:
            mgr.load_model("Systran/faster-whisper-tiny")
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

    def test_ok_real_decoder(self) -> None:
        w = FakeWhisper()
        r = self.client(w).post("/v1/langid", files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["language"], "ru")
        self.assertEqual(body["probabilities"], {"ru": 0.9, "en": 0.07, "uz": 0.03})
        self.assertEqual(body["model"], "Systran/faster-whisper-large-v3")
        self.assertEqual(body["revision"], "edaa852")
        self.assertAlmostEqual(body["audio_seconds"], 1.0, places=2)
        self.assertEqual(w.calls, 1)

    def test_other_wav_falls_back(self) -> None:
        t = np.arange(8000) / 8000
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes((0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes())
        r = self.client(FakeWhisper()).post("/v1/langid", files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertAlmostEqual(r.json()["audio_seconds"], 1.0, places=2)  # resampled to 16 kHz by PyAV

    def test_fast_wav_matches_pyav(self) -> None:
        from speaches.dependencies import audio_file_dependency
        from starlette.datastructures import Headers, UploadFile

        data = wav_bytes(2.0)
        mk = lambda: UploadFile(io.BytesIO(data), filename="a.wav", headers=Headers({"content-type": "audio/wav"}))  # noqa: E731
        fast, slow = langid.decode(mk()).data, audio_file_dependency(mk()).data
        self.assertEqual(fast.shape, slow.shape)
        self.assertLess(float(np.max(np.abs(fast - slow))), 1e-3)

    def test_request_id(self) -> None:
        c, files = self.client(FakeWhisper()), {"file": ("a.wav", wav_bytes(), "audio/wav")}
        self.assertEqual(c.post("/v1/langid", files=files, headers={"X-Request-Id": "job-42:a"}).json()["request_id"], "job-42:a")
        self.assertIsNone(c.post("/v1/langid", files=files, headers={"X-Request-Id": "bad id\nx"}).json()["request_id"])
        self.assertIsNone(c.post("/v1/langid", files=files).json()["request_id"])

    def test_pcm(self) -> None:
        pcm = (np.zeros(8000)).astype("<i2").tobytes()
        r = self.client(FakeWhisper()).post("/v1/langid", files={"file": ("a.pcm", pcm, "audio/pcm")})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertAlmostEqual(r.json()["audio_seconds"], 0.5, places=2)

    def test_not_loaded(self) -> None:
        r = self.client(None).post("/v1/langid", files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(self.client(None).get("/health").json()["model_loaded"], False)

    def test_too_big(self) -> None:
        r = self.client(FakeWhisper(), GS_LID_MAX_BYTES="1000").post(
            "/v1/langid", files={"file": ("a.wav", wav_bytes(), "audio/wav")}
        )
        self.assertEqual(r.status_code, 413)

    def test_bad_audio(self) -> None:
        r = self.client(FakeWhisper()).post("/v1/langid", files={"file": ("a.wav", b"not audio", "audio/wav")})
        self.assertIn(r.status_code, (400, 415))

    def test_key(self) -> None:
        c = self.client(FakeWhisper(), GS_LID_API_KEY="k" * 40)
        files = {"file": ("a.wav", wav_bytes(), "audio/wav")}
        self.assertEqual(c.post("/v1/langid", files=files).status_code, 401)
        ok = c.post("/v1/langid", files=files, headers={"Authorization": "Bearer " + "k" * 40})
        self.assertEqual(ok.status_code, 200)

    def test_busy_cap(self) -> None:
        app = langid.make_app(env={"GS_LID_MAX_OUTSTANDING": "1"}, model=FakeModel(FakeWhisper(delay=0.5)))
        results: list[int] = []

        def call() -> None:
            with TestClient(app) as c:
                results.append(c.post("/v1/langid", files={"file": ("a.wav", wav_bytes(), "audio/wav")}).status_code)

        first = threading.Thread(target=call)
        first.start()
        time.sleep(0.2)
        call()
        first.join()
        self.assertEqual(sorted(results), [200, 503])
        self.assertEqual(app.state.slots.used, 0)


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
