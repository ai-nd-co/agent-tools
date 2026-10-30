"""Unit tests for gs_speech.tts. No model, no GPU: the session is a stand-in. Run inside the image:

docker run --rm --entrypoint python <image> -I -m unittest discover -s /opt/gs-speech/tests -v
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, "/opt/gs-speech")

from gs_speech import tts  # noqa: E402

IN_IMAGE = os.path.isdir("/home/ubuntu/speaches/src/speaches")


class FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0


class FakeSession:
    calls: list = []

    def __init__(self, path_or_bytes, sess_options=None, providers=None, **kwargs) -> None:  # noqa: ANN001, ANN003
        FakeSession.calls.append((path_or_bytes, sess_options, providers, kwargs))


def fake_ort() -> types.SimpleNamespace:
    FakeSession.calls = []
    return types.SimpleNamespace(InferenceSession=FakeSession, SessionOptions=FakeSessionOptions)


# The shape of speaches.executors.kokoro this image pins (Speaches' own lines, trimmed).
KOKORO_SOURCE = """
class KokoroModelManager:
    ort_providers = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]

    def _load_fn(self, model_id):
        inf_sess = InferenceSession("model.onnx", providers=self.ort_providers)
        return inf_sess
"""


def fake_kokoro(ort: types.SimpleNamespace, source: str = KOKORO_SOURCE) -> types.ModuleType:
    module = types.ModuleType("speaches.executors.kokoro")
    module.InferenceSession = ort.InferenceSession
    exec(source, module.__dict__)  # noqa: S102 - fixed test source
    return module


class ParseTest(unittest.TestCase):
    def test_threads(self) -> None:
        self.assertEqual(tts.parse_threads(None), 4)
        self.assertEqual(tts.parse_threads(" "), 4)
        self.assertEqual(tts.parse_threads("2"), 2)
        self.assertEqual(tts.parse_threads("0"), 0)
        with self.assertRaises(ValueError):
            tts.parse_threads("-1")


class InstallTest(unittest.TestCase):
    def test_cpu_only_with_threads(self) -> None:
        ort = fake_ort()
        module = fake_kokoro(ort)
        self.assertTrue(tts.install({}, module=module, ort=ort))
        module.KokoroModelManager()._load_fn("speaches-ai/Kokoro-82M-v1.0-ONNX")
        [(path, so, providers, kwargs)] = FakeSession.calls
        self.assertEqual(path, "model.onnx")
        self.assertEqual(providers, ["CPUExecutionProvider"])
        self.assertEqual((so.intra_op_num_threads, so.inter_op_num_threads), (4, 1))
        self.assertEqual(kwargs, {})

    def test_thread_count_from_env(self) -> None:
        ort = fake_ort()
        module = fake_kokoro(ort)
        tts.install({"GS_KOKORO_THREADS": "3"}, module=module, ort=ort)
        module.KokoroModelManager()._load_fn("m")
        self.assertEqual(FakeSession.calls[0][1].intra_op_num_threads, 3)

    def test_second_install_is_a_no_op(self) -> None:
        ort = fake_ort()
        module = fake_kokoro(ort)
        tts.install({}, module=module, ort=ort)
        wrapped = module.InferenceSession
        self.assertTrue(tts.install({"GS_KOKORO_THREADS": "8"}, module=module, ort=ort))
        self.assertIs(module.InferenceSession, wrapped)

    def test_off_switch_leaves_speaches_alone(self) -> None:
        ort = fake_ort()
        module = fake_kokoro(ort)
        self.assertFalse(tts.install({"GS_KOKORO_THREADS": "0"}, module=module, ort=ort))
        self.assertIs(module.InferenceSession, FakeSession)
        module.KokoroModelManager()._load_fn("m")
        [(_, so, providers, _)] = FakeSession.calls
        self.assertIsNone(so)
        self.assertEqual(providers[0][0], "CUDAExecutionProvider")

    def test_mismatch_logs_and_leaves_speaches_alone(self) -> None:
        cases = {
            "no _load_fn": "class KokoroModelManager:\n    pass\n",
            "no manager": "x = 1\n",
            "_load_fn without InferenceSession": (
                "class KokoroModelManager:\n    def _load_fn(self, model_id):\n        return 1\n"
            ),
        }
        for name, source in cases.items():
            with self.subTest(name):
                ort = fake_ort()
                module = fake_kokoro(ort, source)
                with self.assertLogs("gs_speech", "WARNING") as logs:
                    self.assertFalse(tts.install({}, module=module, ort=ort))
                self.assertIn("left on Speaches' providers", logs.output[0])
                self.assertIs(module.InferenceSession, FakeSession)

    def test_foreign_session_name_is_left_alone(self) -> None:
        ort = fake_ort()
        module = fake_kokoro(ort)
        module.InferenceSession = object  # someone else's object under that name
        with self.assertLogs("gs_speech", "WARNING"):
            self.assertFalse(tts.install({}, module=module, ort=ort))
        self.assertIs(module.InferenceSession, object)

    def test_module_without_session_is_left_alone(self) -> None:
        ort = fake_ort()
        broken = types.ModuleType("broken")  # no InferenceSession at all
        with self.assertLogs("gs_speech", "WARNING"):
            self.assertFalse(tts.install({}, module=broken, ort=ort))

    def test_import_failure_does_not_raise(self) -> None:
        # A None entry in sys.modules makes that import raise ImportError.
        blocked = {
            "speaches": {"speaches": None, "speaches.executors": None},
            "onnxruntime": {"onnxruntime": None},
        }
        for name, entries in blocked.items():
            with self.subTest(name), mock.patch.dict(sys.modules, entries):
                ort = fake_ort()
                kwargs = {"ort": ort} if name == "speaches" else {"module": fake_kokoro(ort)}
                with self.assertLogs("gs_speech", "WARNING") as logs:
                    self.assertFalse(tts.install({}, **kwargs))
                self.assertIn("import failed", logs.output[0])

    def test_bad_thread_value_warns_and_keeps_cpu(self) -> None:
        for raw in ("abc", "-1"):
            with self.subTest(raw):
                ort = fake_ort()
                module = fake_kokoro(ort)
                with self.assertLogs("gs_speech", "WARNING") as logs:
                    self.assertTrue(tts.install({"GS_KOKORO_THREADS": raw}, module=module, ort=ort))
                self.assertIn("GS_KOKORO_THREADS", logs.output[0])
                module.KokoroModelManager()._load_fn("m")
                [(_, so, providers, _)] = FakeSession.calls
                self.assertEqual(providers, ["CPUExecutionProvider"])
                self.assertEqual(so.intra_op_num_threads, 4)


class LauncherTest(unittest.TestCase):
    """serve.main installs the Kokoro hook before the server runs; nothing is served or bound."""

    def test_install_before_serving(self) -> None:
        try:
            from gs_speech import ext, serve
        except ImportError as e:  # fastapi/uvicorn: present in the image, so never skipped there
            if IN_IMAGE:
                raise
            self.skipTest(f"needs the image's packages: {e}")
        order = []
        fake_main = types.ModuleType("speaches.main")
        fake_main.create_app = lambda: order.append("create_app") or object()
        fake_deps = types.ModuleType("speaches.dependencies")
        fake_deps.get_config = lambda: types.SimpleNamespace(api_key=None)
        state = types.SimpleNamespace(permits=1, limits={})
        server = mock.Mock()
        server.run.side_effect = lambda **_: order.append("run")
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "speaches": types.ModuleType("speaches"),
                    "speaches.main": fake_main,
                    "speaches.dependencies": fake_deps,
                },
            ),
            mock.patch.dict(os.environ, {"GS_LID_PORT": "0"}),
            mock.patch.object(ext, "install", side_effect=lambda: order.append("ext") or state),
            mock.patch.object(ext, "Admission"),
            mock.patch.object(tts, "install", side_effect=lambda: order.append("tts")),
            mock.patch.object(serve, "bind"),
            mock.patch.object(serve.uvicorn, "Server", return_value=server),
        ):
            serve.main()
        self.assertEqual(order, ["create_app", "ext", "tts", "run"])


@unittest.skipUnless(IN_IMAGE, "needs the Speaches install in the image")
class RealSpeachesTest(unittest.TestCase):
    """Against the real Speaches module and onnxruntime in this image; installs nothing."""

    def test_pinned_speaches_matches(self) -> None:
        import onnxruntime
        from speaches.executors import kokoro, silero_vad_v5

        self.assertIsNone(tts.mismatch(kokoro, onnxruntime))
        self.assertIsNot(silero_vad_v5.__dict__, kokoro.__dict__)  # VAD resolves its own name

    def test_real_load_fn_hands_kokoro_a_cpu_session(self) -> None:
        """The real KokoroModelManager._load_fn, with only model files, the onnxruntime
        constructor and Kokoro.from_session stubbed: no model is read."""
        import onnxruntime
        from speaches.config import OrtOptions
        from speaches.executors import kokoro

        class Recorder:
            def __init__(self, path, sess_options=None, providers=None, **kwargs) -> None:  # noqa: ANN001, ANN003
                self.path, self.sess_options, self.providers = path, sess_options, providers

        got = []
        real_session = onnxruntime.InferenceSession
        files = types.SimpleNamespace(model="model.onnx", voices="voices.bin")
        ort = types.SimpleNamespace(
            InferenceSession=Recorder, SessionOptions=onnxruntime.SessionOptions
        )
        with (
            mock.patch.object(kokoro, "InferenceSession", Recorder),
            mock.patch.object(kokoro, tts._INSTALLED_MARK, False, create=True),  # noqa: SLF001
            mock.patch.object(kokoro.kokoro_model_registry, "get_model_files", return_value=files),
            mock.patch.object(
                kokoro, "Kokoro", types.SimpleNamespace(from_session=lambda s, v: got.append(s))
            ),
        ):
            self.assertTrue(tts.install({}, module=kokoro, ort=ort))
            kokoro.KokoroModelManager(ttl=-1, ort_opts=OrtOptions())._load_fn("kokoro")
        [sess] = got
        self.assertIsInstance(sess, Recorder)
        self.assertEqual(sess.providers, ["CPUExecutionProvider"])
        self.assertIsInstance(sess.sess_options, onnxruntime.SessionOptions)
        self.assertEqual(sess.sess_options.intra_op_num_threads, 4)
        self.assertEqual(sess.sess_options.inter_op_num_threads, 1)
        self.assertIs(onnxruntime.InferenceSession, real_session)  # no global onnxruntime patch
        self.assertIs(kokoro.InferenceSession, real_session)  # restored after the test

    def test_real_session_options_take_threads(self) -> None:
        import onnxruntime

        seen = []
        make = tts.cpu_session(
            lambda path, **kw: seen.append(kw), onnxruntime.SessionOptions, threads=4
        )
        make("model.onnx", providers=[("CUDAExecutionProvider", {})])
        so = seen[0]["sess_options"]
        self.assertEqual((so.intra_op_num_threads, so.inter_op_num_threads), (4, 1))
        self.assertEqual(seen[0]["providers"], ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
