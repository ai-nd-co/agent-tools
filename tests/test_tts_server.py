from __future__ import annotations

import hashlib
import http.client
import importlib
import json
import socket
import sys
import threading
import time
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_tools.tts_server import (
    MAX_AUDIO_BYTES,
    MAX_REQUEST_BODY_BYTES,
    TtsHttpServer,
    TtsService,
    TtsServiceUnavailable,
    _synthesize_prepared_text,
    load_owner_only_bearer_token,
    validate_tts_server_bind,
)

TOKEN = "task298-owner-only-token-0123456789abcdef"


def _server_address(server: TtsHttpServer) -> tuple[str, int]:
    host, port = server.server_address[:2]
    assert isinstance(host, str)
    assert isinstance(port, int)
    return host, port


def _wav_bytes(*, frames: int = 2_400, rate: int = 24_000, channels: int = 1) -> bytes:
    import io

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * frames * channels)
    return output.getvalue()


def _result(language: str, *, wav: bytes | None = None) -> Any:
    is_russian = language in {"ru", "ru-RU"}
    api_language = "ru" if is_russian else "b" if language == "en-GB" else "a"
    return SimpleNamespace(
        tts_result=SimpleNamespace(
            wav=_wav_bytes() if wav is None else wav,
            engine="silero" if is_russian else "kokoro",
            language=api_language,
        )
    )


def _ready_service(synthesizer: Any | None = None) -> TtsService:
    service = TtsService(
        bearer_token=TOKEN,
        synthesizer=synthesizer or (lambda _text, language, _device: _result(language)),
    )
    service.prewarm()
    return service


@contextmanager
def _running_server(service: TtsService) -> Iterator[tuple[str, int]]:
    server = TtsHttpServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _server_address(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def _request(
    address: tuple[str, int],
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    token: str | None = TOKEN,
    content_type: str | None = "application/json",
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    headers = dict(extra_headers or {})
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if content_type is not None:
        headers["Content-Type"] = content_type
    connection = http.client.HTTPConnection(*address, timeout=3)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _post_payload(
    address: tuple[str, int],
    payload: object,
    *,
    content_type: str | None = "application/json",
) -> tuple[int, dict[str, str], bytes]:
    return _request(
        address,
        "POST",
        "/v1/tts",
        body=json.dumps(payload).encode("utf-8"),
        content_type=content_type,
    )


def test_health_requires_bearer_and_reports_ready_without_secrets() -> None:
    with _running_server(_ready_service()) as address:
        missing = _request(address, "GET", "/healthz", token=None, content_type=None)
        wrong = _request(address, "GET", "/healthz", token="x", content_type=None)
        success = _request(address, "GET", "/healthz", content_type=None)

    assert missing[0] == wrong[0] == 401
    assert json.loads(missing[2])["error"]["code"] == "unauthorized"
    assert TOKEN.encode() not in missing[2] + wrong[2] + success[2]
    assert success[0] == 200
    assert json.loads(success[2]) == {"status": "ready", "engines": ["kokoro", "silero"]}
    assert success[1]["Cache-Control"] == "no-store"
    assert success[1]["X-Content-Type-Options"] == "nosniff"


def test_health_is_unavailable_until_both_engines_are_pre_warmed() -> None:
    service = TtsService(bearer_token=TOKEN, synthesizer=lambda *_args: _result("en-US"))
    with _running_server(service) as address:
        status, _, body = _request(address, "GET", "/healthz", content_type=None)

    assert status == 503
    assert json.loads(body)["error"]["code"] == "service_unavailable"


def test_prewarm_requires_truthful_kokoro_and_silero() -> None:
    service = TtsService(bearer_token=TOKEN, synthesizer=lambda *_args: _result("en-US"))

    with pytest.raises(TtsServiceUnavailable, match="silero prewarm failed"):
        service.prewarm()

    assert service.ready is False


def test_hung_prewarm_is_bounded_and_never_becomes_ready() -> None:
    entered = threading.Event()
    release = threading.Event()

    def hung_synthesizer(_text: str, language: str, _device: str) -> Any:
        entered.set()
        assert release.wait(2)
        return _result(language)

    service = TtsService(
        bearer_token=TOKEN,
        synthesizer=hung_synthesizer,
        prewarm_timeout_seconds=0.05,
    )
    started = time.perf_counter()
    try:
        with pytest.raises(TtsServiceUnavailable, match="kokoro prewarm failed"):
            service.prewarm()
        assert time.perf_counter() - started < 0.5
        assert entered.is_set()
        assert service.ready is False
    finally:
        release.set()


def test_default_synthesizer_uses_no_transform_and_maps_api_language(
    monkeypatch: Any,
) -> None:
    import agent_tools.tts_server as server_module

    captured: dict[str, object] = {}

    def fake_ttsify(text: str, options: object) -> Any:
        captured.update(text=text, options=options)
        return _result("ru")

    monkeypatch.setattr(server_module, "ttsify_text", fake_ttsify)

    result = _synthesize_prepared_text("Точный текст.", "ru-RU", "cpu")

    assert result.tts_result.engine == "silero"
    assert captured["text"] == "Точный текст."
    options: Any = captured["options"]
    assert options.no_transform is True
    assert options.language == "ru"
    assert options.device == "cpu"


@pytest.mark.parametrize("language", ["auto", "en-US", "en-GB", "ru", "ru-RU"])
def test_post_returns_direct_valid_wav_and_truthful_headers(language: str) -> None:
    with _running_server(_ready_service()) as address:
        status, headers, body = _post_payload(
            address,
            {"requestId": "phone-123", "text": "Prepared speech.", "language": language},
        )

    assert status == 200
    assert headers["Content-Type"] == "audio/wav"
    assert int(headers["Content-Length"]) == len(body)
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Audio-SHA256"] == hashlib.sha256(body).hexdigest()
    assert int(headers["X-Audio-Duration-Ms"]) == 100
    assert headers["X-Request-Id"] == "phone-123"
    expected_engine = "silero" if language in {"ru", "ru-RU"} else "kokoro"
    assert headers["X-TTS-Engine"] == expected_engine
    expected_language = (
        "ru-RU" if language in {"ru", "ru-RU"} else "en-GB" if language == "en-GB" else "en-US"
    )
    assert headers["X-TTS-Language"] == expected_language
    with wave.open(__import__("io").BytesIO(body), "rb") as wav_file:
        assert (wav_file.getnchannels(), wav_file.getsampwidth(), wav_file.getframerate()) == (
            1,
            2,
            24_000,
        )


def test_repeated_real_socket_delivery_matches_full_length_and_hash() -> None:
    wav_by_language = {
        "en-US": _wav_bytes(frames=34_200),
        "ru-RU": _wav_bytes(frames=35_700),
    }

    def synthesizer(_text: str, language: str, _device: str) -> Any:
        wav_language = "ru-RU" if language in {"ru", "ru-RU"} else "en-US"
        return _result(language, wav=wav_by_language[wav_language])

    with _running_server(_ready_service(synthesizer)) as address:
        for index in range(16):
            language = "en-US" if index % 2 == 0 else "ru-RU"
            status, headers, body = _post_payload(
                address,
                {
                    "requestId": f"delivery-{index}",
                    "text": "Synthetic delivery probe.",
                    "language": language,
                },
            )

            assert status == 200
            assert int(headers["Content-Length"]) == len(wav_by_language[language])
            assert len(body) == len(wav_by_language[language])
            assert headers["X-Audio-SHA256"] == hashlib.sha256(body).hexdigest()


def test_http11_wav_connection_serves_second_request_before_framework_close() -> None:
    class TrackingServer(TtsHttpServer):
        def __init__(self, server_address: tuple[str, int], service: TtsService) -> None:
            self.shutdown_request_started = threading.Event()
            super().__init__(server_address, service)

        def shutdown_request(self, request: Any) -> None:
            self.shutdown_request_started.set()
            super().shutdown_request(request)

    wav = _wav_bytes(frames=34_200)
    service = _ready_service(lambda _text, language, _device: _result(language, wav=wav))
    server = TrackingServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = _server_address(server)
    client = socket.create_connection(address, timeout=2)

    def request(request_id: str) -> bytes:
        payload = json.dumps(
            {
                "requestId": request_id,
                "text": "Synthetic delivery probe.",
                "language": "en-US",
            }
        ).encode()
        return (
            b"POST /v1/tts HTTP/1.1\r\n"
            + f"Host: {address[0]}\r\n".encode()
            + f"Authorization: Bearer {TOKEN}\r\n".encode()
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )

    def read_response() -> tuple[dict[str, str], bytes]:
        response = bytearray()
        while b"\r\n\r\n" not in response:
            chunk = client.recv(257)
            assert chunk, "response closed before headers completed"
            response.extend(chunk)
        raw_headers, body_prefix = bytes(response).split(b"\r\n\r\n", 1)
        headers = {
            name.decode("ascii").lower(): value.decode("ascii").strip()
            for line in raw_headers.split(b"\r\n")[1:]
            for name, value in [line.split(b":", 1)]
        }
        expected_length = int(headers["content-length"])
        body = bytearray(body_prefix)
        while len(body) < expected_length:
            chunk = client.recv(min(113, expected_length - len(body)))
            assert chunk, "response closed before the declared WAV body completed"
            body.extend(chunk)
        return headers, bytes(body)

    try:
        client.sendall(request("keepalive-1") + request("keepalive-2"))
        first_headers, first_body = read_response()
        assert "connection" not in first_headers
        assert len(first_body) == len(wav)
        assert first_headers["x-audio-sha256"] == hashlib.sha256(first_body).hexdigest()
        assert not server.shutdown_request_started.wait(0.1)

        second_headers, second_body = read_response()
        assert len(second_body) == len(wav)
        assert second_headers["x-request-id"] == "keepalive-2"
        assert second_headers["x-audio-sha256"] == hashlib.sha256(second_body).hexdigest()
        assert not server.shutdown_request_started.wait(0.1)

        client.close()
        assert server.shutdown_request_started.wait(1)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize(
    ("http_version", "connection_header"),
    [("HTTP/1.0", None), ("HTTP/1.1", "close")],
)
def test_successful_wav_honors_request_close(
    http_version: str,
    connection_header: str | None,
) -> None:
    wav = _wav_bytes(frames=34_200)
    service = _ready_service(lambda _text, language, _device: _result(language, wav=wav))
    with _running_server(service) as address:
        client = socket.create_connection(address, timeout=2)
        stream = client.makefile("rb")
        payload = json.dumps(
            {
                "requestId": "request-close",
                "text": "Synthetic delivery probe.",
                "language": "en-US",
            }
        ).encode()
        request_lines = [
            f"POST /v1/tts {http_version}",
            f"Host: {address[0]}",
            f"Authorization: Bearer {TOKEN}",
            "Content-Type: application/json",
            f"Content-Length: {len(payload)}",
        ]
        if connection_header is not None:
            request_lines.append(f"Connection: {connection_header}")
        request = ("\r\n".join(request_lines) + "\r\n\r\n").encode() + payload
        try:
            client.sendall(request)
            status_line = stream.readline()
            assert status_line.startswith(b"HTTP/1.1 200")
            headers: dict[str, str] = {}
            while True:
                line = stream.readline()
                assert line, "response closed before headers completed"
                if line == b"\r\n":
                    break
                name, value = line.split(b":", 1)
                headers[name.decode("ascii").lower()] = value.decode("ascii").strip()
            expected_length = int(headers["content-length"])
            body = stream.read(expected_length)
            assert len(body) == expected_length == len(wav)
            assert headers["x-audio-sha256"] == hashlib.sha256(body).hexdigest()
            assert stream.read(1) == b""
        finally:
            stream.close()
            client.close()


def test_http11_wav_idle_connection_teardown_is_bounded(monkeypatch: Any) -> None:
    import agent_tools.tts_server as server_module

    class TrackingServer(TtsHttpServer):
        def __init__(self, server_address: tuple[str, int], service: TtsService) -> None:
            self.shutdown_request_started = threading.Event()
            super().__init__(server_address, service)

        def shutdown_request(self, request: Any) -> None:
            self.shutdown_request_started.set()
            super().shutdown_request(request)

    monkeypatch.setattr(server_module, "SOCKET_IO_TIMEOUT_SECONDS", 0.05)
    server = TrackingServer(("127.0.0.1", 0), _ready_service())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*_server_address(server), timeout=2)
    payload = json.dumps(
        {
            "requestId": "idle-keepalive",
            "text": "Synthetic delivery probe.",
            "language": "en-US",
        }
    ).encode()
    try:
        connection.request(
            "POST",
            "/v1/tts",
            body=payload,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        assert response.status == 200
        assert len(response.read()) == int(response.headers["Content-Length"])
        assert server.shutdown_request_started.wait(1)
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_unauthenticated_json_response_does_not_wait_for_peer_close() -> None:
    class TrackingServer(TtsHttpServer):
        def __init__(self, server_address: tuple[str, int], service: TtsService) -> None:
            self.shutdown_request_started = threading.Event()
            super().__init__(server_address, service)

        def shutdown_request(self, request: Any) -> None:
            self.shutdown_request_started.set()
            super().shutdown_request(request)

    server = TrackingServer(("127.0.0.1", 0), _ready_service())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = _server_address(server)
    client = socket.create_connection(address, timeout=2)
    request = (
        b"GET /healthz HTTP/1.1\r\n"
        + f"Host: {address[0]}\r\n".encode()
        + b"Connection: keep-alive\r\n\r\n"
    )
    try:
        client.sendall(request)
        assert server.shutdown_request_started.wait(0.5)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"requestId": "id", "text": "ok"}, "invalid_fields"),
        (
            {"requestId": "id", "text": "ok", "language": "auto", "extra": True},
            "invalid_fields",
        ),
        ({"requestId": "bad id", "text": "ok", "language": "auto"}, "invalid_request_id"),
        ({"requestId": "id", "text": "  ", "language": "auto"}, "invalid_text"),
        ({"requestId": "id", "text": "x" * 4_001, "language": "auto"}, "text_too_long"),
        ({"requestId": "id", "text": "ok", "language": "de-DE"}, "invalid_language"),
    ],
)
def test_post_rejects_invalid_request_fields(payload: object, code: str) -> None:
    with _running_server(_ready_service()) as address:
        status, _, body = _post_payload(address, payload)

    assert status == 400
    assert json.loads(body)["error"]["code"] == code
    assert b"ok" not in body


def test_post_rejects_malformed_duplicate_and_oversized_json() -> None:
    duplicate = b'{"requestId":"a","requestId":"b","text":"x","language":"auto"}'
    oversized = b"{" + b" " * MAX_REQUEST_BODY_BYTES + b"}"
    with _running_server(_ready_service()) as address:
        malformed = _request(address, "POST", "/v1/tts", body=b"{not-json")
        duplicated = _request(address, "POST", "/v1/tts", body=duplicate)
        too_large = _request(address, "POST", "/v1/tts", body=oversized)

    assert malformed[0] == duplicated[0] == 400
    assert json.loads(malformed[2])["error"]["code"] == "malformed_json"
    assert json.loads(duplicated[2])["error"]["code"] == "malformed_json"
    assert too_large[0] == 413
    assert json.loads(too_large[2])["error"]["code"] == "request_too_large"


def test_post_rejects_media_query_absolute_target_and_methods() -> None:
    payload = {"requestId": "id", "text": "ok", "language": "auto"}
    with _running_server(_ready_service()) as address:
        media = _post_payload(address, payload, content_type="text/plain")
        query = _request(address, "GET", "/healthz?token=private", content_type=None)
        method = _request(address, "DELETE", "/v1/tts", body=b"")
        unknown_method = _request(address, "TRACE", "/v1/tts", body=b"")
        missing = _request(address, "GET", "/missing", content_type=None)

    assert media[0] == 415
    assert json.loads(media[2])["error"]["code"] == "unsupported_media_type"
    assert query[0] == 400
    assert b"private" not in query[2]
    assert method[0] == 405
    assert unknown_method[0] == 405
    assert json.loads(unknown_method[2])["error"]["code"] == "method_not_allowed"
    assert missing[0] == 404


def test_concurrent_synthesis_returns_busy_without_queueing() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def synthesizer(text: str, language: str, _device: str) -> Any:
        calls.append(text)
        if text == "first":
            entered.set()
            assert release.wait(3)
        return _result(language)

    service = _ready_service(synthesizer)
    with _running_server(service) as address:
        first_result: list[tuple[int, dict[str, str], bytes]] = []
        first = threading.Thread(
            target=lambda: first_result.append(
                _post_payload(
                    address,
                    {"requestId": "first", "text": "first", "language": "en-US"},
                )
            )
        )
        first.start()
        assert entered.wait(2)
        second = _post_payload(
            address,
            {"requestId": "second", "text": "second", "language": "en-US"},
        )
        release.set()
        first.join(timeout=3)

    assert second[0] == 429
    assert json.loads(second[2])["error"]["code"] == "busy"
    assert second[1]["Retry-After"] == "1"
    assert first_result[0][0] == 200
    assert "second" not in calls


def test_hung_request_fails_closed_and_does_not_start_a_second_job() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def hung_synthesizer(text: str, language: str, _device: str) -> Any:
        calls.append(text)
        entered.set()
        assert release.wait(2)
        return _result(language)

    service = _ready_service()
    service._synthesizer = hung_synthesizer
    service._synthesis_timeout_seconds = 0.05
    try:
        with _running_server(service) as address:
            started = time.perf_counter()
            timed_out = _post_payload(
                address,
                {"requestId": "first", "text": "first", "language": "en-US"},
            )
            assert time.perf_counter() - started < 0.5
            assert entered.is_set()
            health = _request(address, "GET", "/healthz", content_type=None)
            second = _post_payload(
                address,
                {"requestId": "second", "text": "second", "language": "en-US"},
            )
    finally:
        release.set()

    assert timed_out[0] == 503
    assert json.loads(timed_out[2])["error"]["code"] == "synthesis_timeout"
    assert health[0] == 503
    assert second[0] == 503
    assert json.loads(second[2])["error"]["code"] == "synthesis_unavailable"
    assert calls == ["first"]
    assert service.ready is False


def test_server_shutdown_is_bounded_while_timed_out_worker_is_still_alive() -> None:
    release = threading.Event()

    def hung_synthesizer(_text: str, language: str, _device: str) -> Any:
        assert release.wait(2)
        return _result(language)

    service = _ready_service()
    service._synthesizer = hung_synthesizer
    service._synthesis_timeout_seconds = 0.05
    server = TtsHttpServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = _post_payload(
            _server_address(server),
            {"requestId": "hung", "text": "hung", "language": "en-US"},
        )
        assert response[0] == 503
        started = time.perf_counter()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
        assert time.perf_counter() - started < 1
        assert not thread.is_alive()
    finally:
        release.set()


@pytest.mark.parametrize(
    ("synthesizer", "code"),
    [
        (
            lambda *_args: (_ for _ in ()).throw(RuntimeError("private text C:/secret")),
            "synthesis_unavailable",
        ),
        (lambda _text, language, _device: _result(language, wav=b"not wav"), "invalid_audio"),
        (
            lambda _text, language, _device: _result(language, wav=b"R" * (MAX_AUDIO_BYTES + 1)),
            "audio_too_large",
        ),
        (
            lambda _text, language, _device: _result(language, wav=_wav_bytes(channels=2)),
            "invalid_audio",
        ),
    ],
)
def test_synthesis_failures_are_bounded_and_private(synthesizer: Any, code: str) -> None:
    service = _ready_service()
    service._synthesizer = synthesizer
    with _running_server(service) as address:
        status, _, body = _post_payload(
            address,
            {"requestId": "id", "text": "private spoken text", "language": "en-US"},
        )

    assert status == 503
    assert json.loads(body)["error"]["code"] == code
    assert b"private spoken text" not in body
    assert b"C:/secret" not in body
    assert b"Traceback" not in body


def test_client_disconnect_releases_single_job_and_server_stays_healthy() -> None:
    completed = threading.Event()

    def synthesizer(_text: str, language: str, _device: str) -> Any:
        time.sleep(0.05)
        completed.set()
        return _result(language)

    service = _ready_service(synthesizer)
    body = json.dumps({"requestId": "gone", "text": "disconnect", "language": "en-US"}).encode()
    with _running_server(service) as address:
        client = socket.create_connection(address, timeout=2)
        request = (
            b"POST /v1/tts HTTP/1.1\r\n"
            + f"Host: {address[0]}\r\n".encode()
            + f"Authorization: Bearer {TOKEN}\r\n".encode()
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        client.sendall(request)
        client.close()
        assert completed.wait(2)
        time.sleep(0.05)
        health = _request(address, "GET", "/healthz", content_type=None)

    assert health[0] == 200


def test_shutdown_releases_exact_port() -> None:
    service = _ready_service()
    server = TtsHttpServer(("127.0.0.1", 0), service)
    address = _server_address(server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)

    replacement = TtsHttpServer(address, service)
    replacement.server_close()


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "localhost", "127.0.0.2", "169.254.2.2", "8.8.8.8"],
)
def test_bind_rejects_wildcard_public_link_local_and_alias_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="exact loopback"):
        validate_tts_server_bind(host, 4223)


@pytest.mark.parametrize("host", ["127.0.0.1", "100.64.0.4", "100.100.20.30", "192.168.1.10"])
def test_bind_accepts_exact_loopback_tailscale_and_rfc1918(host: str) -> None:
    validate_tts_server_bind(host, 4223)


def test_bind_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="between 1 and 65535"):
        validate_tts_server_bind("127.0.0.1", 0)


def test_cli_contract_defaults_to_loopback_port_4223_and_cpu(monkeypatch: Any) -> None:
    import agent_tools.cli as cli_module

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "run_tts_server",
        lambda **values: captured.update(values) or 0,
    )

    assert cli_module.main(["tts-server", "--token-file", "private-token-file"]) == 0

    assert captured == {
        "host": "127.0.0.1",
        "port": 4223,
        "token_file": Path("private-token-file"),
        "device": "cpu",
    }


def test_owner_only_token_file_is_read_without_exposure(tmp_path: Path) -> None:
    token_path = tmp_path / "tts-token"
    token_path.write_text(TOKEN + "\n", encoding="ascii")
    _restrict_to_current_owner(token_path)

    assert load_owner_only_bearer_token(token_path) == TOKEN

    token_path.write_text("short", encoding="ascii")
    _restrict_to_current_owner(token_path)
    with pytest.raises(RuntimeError, match="stable owner-only regular file") as error:
        load_owner_only_bearer_token(token_path)
    assert str(token_path) not in str(error.value)
    assert "short" not in str(error.value)


def test_token_file_with_another_principal_access_is_rejected(tmp_path: Path) -> None:
    token_path = tmp_path / "broad-token"
    token_path.write_text(TOKEN, encoding="ascii")
    _grant_another_principal_read_access(token_path)

    with pytest.raises(RuntimeError, match="stable owner-only regular file"):
        load_owner_only_bearer_token(token_path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACL exception contract")
def test_pywin32_acl_failure_is_normalized_before_cli_output(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_tools.cli as cli_module
    import agent_tools.tts_server as server_module

    token_path = tmp_path / "private-token-file"
    token_path.write_text(TOKEN, encoding="ascii")
    _restrict_to_current_owner(token_path)
    real_import = server_module.importlib.import_module

    def failing_import(name: str) -> Any:
        module = real_import(name)
        if name == "win32api":
            return SimpleNamespace(
                GetCurrentProcess=lambda: (_ for _ in ()).throw(
                    RuntimeError(f"private ACL failure at {token_path} for {TOKEN}")
                )
            )
        return module

    monkeypatch.setattr(server_module.importlib, "import_module", failing_import)

    result = cli_module.main(["tts-server", "--token-file", str(token_path)])
    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == ""
    assert "stable owner-only regular file" in captured.err
    assert str(token_path) not in captured.err
    assert TOKEN not in captured.err
    assert "Traceback" not in captured.err


def _restrict_to_current_owner(path: Path) -> None:
    if sys.platform != "win32":
        path.chmod(0o600)
        return
    win32api = importlib.import_module("win32api")
    win32con = importlib.import_module("win32con")
    win32security = importlib.import_module("win32security")
    process_token = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
    )
    owner_sid = win32security.GetTokenInformation(process_token, win32security.TokenUser)[0]
    acl = win32security.ACL()
    acl.AddAccessAllowedAceEx(
        win32security.ACL_REVISION_DS,
        0,
        win32con.GENERIC_ALL,
        owner_sid,
    )
    win32security.SetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION
        | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        owner_sid,
        None,
        acl,
        None,
    )


def _grant_another_principal_read_access(path: Path) -> None:
    if sys.platform != "win32":
        path.chmod(0o644)
        return
    win32api = importlib.import_module("win32api")
    win32con = importlib.import_module("win32con")
    win32security = importlib.import_module("win32security")
    process_token = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
    )
    owner_sid = win32security.GetTokenInformation(process_token, win32security.TokenUser)[0]
    everyone_sid = win32security.CreateWellKnownSid(win32security.WinWorldSid, None)
    acl = win32security.ACL()
    acl.AddAccessAllowedAceEx(
        win32security.ACL_REVISION_DS,
        0,
        win32con.GENERIC_ALL,
        owner_sid,
    )
    acl.AddAccessAllowedAceEx(
        win32security.ACL_REVISION_DS,
        0,
        win32con.GENERIC_READ,
        everyone_sid,
    )
    win32security.SetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION
        | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        owner_sid,
        None,
        acl,
        None,
    )
