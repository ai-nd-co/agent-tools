from __future__ import annotations

import ctypes
import hashlib
import hmac
import importlib
import io
import ipaddress
import json
import os
import re
import signal
import stat
import sys
import threading
import warnings
import wave
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agent_tools.ttsify import TtsifyOptions, TtsifyResult, ttsify_text

DEFAULT_TTS_SERVER_HOST = "127.0.0.1"
DEFAULT_TTS_SERVER_PORT = 4223
ALLOWED_API_LANGUAGES = ("auto", "en-US", "en-GB", "ru", "ru-RU")
MAX_REQUEST_BODY_BYTES = 64 * 1024
MAX_OVERSIZE_DRAIN_BYTES = MAX_REQUEST_BODY_BYTES + 4 * 1024
MAX_TEXT_CHARACTERS = 4_000
MAX_AUDIO_BYTES = 12 * 1024 * 1024
MAX_REQUEST_ID_CHARACTERS = 128
MAX_TOKEN_BYTES = 512
MIN_TOKEN_CHARACTERS = 32
SOCKET_IO_TIMEOUT_SECONDS = 10.0
PREWARM_TIMEOUT_SECONDS = 180.0
SYNTHESIS_TIMEOUT_SECONDS = 180.0

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,127}\Z")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9._~+/=-]+\Z")
_REQUEST_FIELDS = frozenset({"requestId", "text", "language"})
_API_TO_TTSIFY_LANGUAGE = {
    "auto": None,
    "en-US": "a",
    "en-GB": "b",
    "ru": "ru",
    "ru-RU": "ru",
}
_TTSIFY_TO_API_LANGUAGE = {
    "a": "en-US",
    "b": "en-GB",
    "ru": "ru-RU",
    "ru-RU": "ru-RU",
}


class TtsServerError(RuntimeError):
    pass


class TtsServiceUnavailable(TtsServerError):
    pass


class TtsServiceBusy(TtsServerError):
    pass


class TtsSynthesisTimedOut(TtsServerError):
    pass


class InvalidSynthesizedAudio(TtsServerError):
    pass


class SynthesizedAudioTooLarge(InvalidSynthesizedAudio):
    pass


@dataclass(frozen=True)
class TtsRequest:
    request_id: str
    text: str
    language: str


@dataclass(frozen=True)
class TtsAudioResponse:
    request_id: str
    wav: bytes
    sha256: str
    duration_ms: int
    engine: str
    language: str


Synthesizer = Callable[[str, str, str], TtsifyResult]


class TtsService:
    def __init__(
        self,
        *,
        bearer_token: str,
        device: str = "cpu",
        synthesizer: Synthesizer | None = None,
        prewarm_timeout_seconds: float = PREWARM_TIMEOUT_SECONDS,
        synthesis_timeout_seconds: float = SYNTHESIS_TIMEOUT_SECONDS,
    ) -> None:
        self.bearer_token = bearer_token
        self.device = device
        self._synthesizer = synthesizer or _synthesize_prepared_text
        self._job_lock = threading.Lock()
        self._ready = False
        self._prewarm_timeout_seconds = prewarm_timeout_seconds
        self._synthesis_timeout_seconds = synthesis_timeout_seconds

    @property
    def ready(self) -> bool:
        return self._ready

    def prewarm(self) -> None:
        self._ready = False
        probes = (("Ready.", "en-US", "kokoro"), ("Готово.", "ru", "silero"))
        for text, language, expected_engine in probes:
            try:
                response = self._run_synthesis_job(
                    request_id="prewarm",
                    text=text,
                    requested_language=language,
                    timeout_seconds=self._prewarm_timeout_seconds,
                )
                if response.engine != expected_engine:
                    raise InvalidSynthesizedAudio("The prewarm engine did not match.")
            except Exception as exc:
                raise TtsServiceUnavailable(
                    f"TTS server {expected_engine} prewarm failed."
                ) from exc
        self._ready = True

    def synthesize(self, request: TtsRequest) -> TtsAudioResponse:
        if not self._ready:
            raise TtsServiceUnavailable("The TTS service is not ready.")
        return self._run_synthesis_job(
            request_id=request.request_id,
            text=request.text,
            requested_language=request.language,
            timeout_seconds=self._synthesis_timeout_seconds,
        )

    def _run_synthesis_job(
        self,
        *,
        request_id: str,
        text: str,
        requested_language: str,
        timeout_seconds: float,
    ) -> TtsAudioResponse:
        if timeout_seconds <= 0:
            raise ValueError("TTS synthesis timeout must be positive.")
        if not self._job_lock.acquire(blocking=False):
            raise TtsServiceBusy("The TTS service is busy.")
        completed = threading.Event()
        outcome: list[TtsAudioResponse | Exception] = []

        def synthesize() -> None:
            try:
                result = self._synthesizer(text, requested_language, self.device)
                outcome.append(
                    _validate_synthesized_result(
                        request_id=request_id,
                        result=result,
                        requested_language=requested_language,
                    )
                )
            except Exception as exc:
                outcome.append(exc)
            finally:
                self._job_lock.release()
                completed.set()

        worker = threading.Thread(target=synthesize, daemon=True, name="agent-tools-tts-job")
        try:
            worker.start()
        except Exception:
            self._job_lock.release()
            raise
        if not completed.wait(timeout_seconds):
            self._ready = False
            raise TtsSynthesisTimedOut("TTS synthesis timed out.")
        if not outcome:
            self._ready = False
            raise TtsServiceUnavailable("TTS synthesis ended without a result.")
        result_or_error = outcome[0]
        if isinstance(result_or_error, Exception):
            raise result_or_error
        return result_or_error


class TtsHttpServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True
    block_on_close = False

    def __init__(self, server_address: tuple[str, int], service: TtsService) -> None:
        self.service = service
        super().__init__(server_address, TtsRequestHandler)

    def get_request(self) -> tuple[Any, Any]:
        request, client_address = super().get_request()
        request.settimeout(SOCKET_IO_TIMEOUT_SECONDS)
        return request, client_address

    def handle_error(self, request: Any, client_address: Any) -> None:
        # BaseServer prints tracebacks. Requests are intentionally reduced to a safe failure.
        return None


class TtsRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AgentToolsTTS"
    sys_version = ""

    @property
    def tts_server(self) -> TtsHttpServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:
        if not self._route_is("/healthz"):
            return
        if not self._authorize():
            return
        if not self.tts_server.service.ready:
            self._send_json_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "service_unavailable",
                "TTS service is unavailable.",
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {"status": "ready", "engines": ["kokoro", "silero"]},
        )

    def do_POST(self) -> None:
        if not self._route_is("/v1/tts"):
            return
        if not self._authorize():
            return
        request = self._read_request()
        if request is None:
            return
        try:
            response = self.tts_server.service.synthesize(request)
        except TtsServiceBusy:
            self._send_json_error(
                HTTPStatus.TOO_MANY_REQUESTS,
                "busy",
                "TTS service is busy.",
                extra_headers={"Retry-After": "1"},
            )
            return
        except TtsSynthesisTimedOut:
            self._send_json_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "synthesis_timeout",
                "TTS synthesis timed out.",
            )
            return
        except SynthesizedAudioTooLarge:
            self._send_json_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "audio_too_large",
                "TTS service produced oversized audio.",
            )
            return
        except InvalidSynthesizedAudio:
            self._send_json_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "invalid_audio",
                "TTS service produced invalid audio.",
            )
            return
        except Exception:
            self._send_json_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "synthesis_unavailable",
                "TTS synthesis is unavailable.",
            )
            return
        self._send_wav(response)

    def do_HEAD(self) -> None:
        self._reject_method()

    def do_PUT(self) -> None:
        self._reject_method()

    def do_PATCH(self) -> None:
        self._reject_method()

    def do_DELETE(self) -> None:
        self._reject_method()

    def do_OPTIONS(self) -> None:
        self._reject_method()

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        del message, explain
        if code == HTTPStatus.NOT_IMPLEMENTED:
            self._reject_method()
            return
        self._send_json_error(
            code,
            "invalid_http_request",
            "The HTTP request is invalid.",
        )

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        del code, size

    def log_error(self, format: str, *args: object) -> None:
        del format, args

    def _route_is(self, expected_path: str) -> bool:
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            self._send_json_error(
                HTTPStatus.BAD_REQUEST,
                "query_not_allowed",
                "Query strings and absolute request targets are not allowed.",
            )
            return False
        if parsed.path != expected_path:
            self._send_json_error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                "The requested endpoint does not exist.",
            )
            return False
        return True

    def _authorize(self) -> bool:
        values = self.headers.get_all("Authorization", failobj=[])
        if len(values) != 1:
            self._send_auth_error()
            return False
        scheme, separator, candidate = values[0].partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not candidate:
            self._send_auth_error()
            return False
        if not hmac.compare_digest(candidate, self.tts_server.service.bearer_token):
            self._send_auth_error()
            return False
        return True

    def _send_auth_error(self) -> None:
        self._send_json_error(
            HTTPStatus.UNAUTHORIZED,
            "unauthorized",
            "Bearer authentication is required.",
            extra_headers={"WWW-Authenticate": "Bearer"},
        )

    def _read_request(self) -> TtsRequest | None:
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding:
            self._send_json_error(
                HTTPStatus.BAD_REQUEST,
                "unsupported_transfer_encoding",
                "Transfer encoding is not supported.",
            )
            return None
        content_types = self.headers.get_all("Content-Type", failobj=[])
        if len(content_types) != 1 or not _is_json_content_type(content_types[0]):
            self._send_json_error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json with UTF-8 encoding.",
            )
            return None
        lengths = self.headers.get_all("Content-Length", failobj=[])
        if len(lengths) != 1 or not lengths[0].isdigit():
            self._send_json_error(
                HTTPStatus.LENGTH_REQUIRED,
                "content_length_required",
                "One valid Content-Length header is required.",
            )
            return None
        content_length = int(lengths[0])
        if content_length <= 0:
            self._send_json_error(
                HTTPStatus.BAD_REQUEST,
                "malformed_json",
                "The JSON request is malformed.",
            )
            return None
        if content_length > MAX_REQUEST_BODY_BYTES:
            if content_length <= MAX_OVERSIZE_DRAIN_BYTES:
                try:
                    self.rfile.read(content_length)
                except OSError:
                    pass
            self._send_json_error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                "The request body is too large.",
            )
            return None
        try:
            body = self.rfile.read(content_length)
        except OSError:
            body = b""
        if len(body) != content_length:
            self._send_json_error(
                HTTPStatus.BAD_REQUEST,
                "incomplete_request",
                "The request body is incomplete.",
            )
            return None
        try:
            decoded = body.decode("utf-8")
            payload = json.loads(decoded, object_pairs_hook=_unique_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._send_json_error(
                HTTPStatus.BAD_REQUEST,
                "malformed_json",
                "The JSON request is malformed.",
            )
            return None
        try:
            return _validate_request_payload(payload)
        except ValueError as exc:
            code = str(exc)
            messages = {
                "invalid_fields": "The request fields are invalid.",
                "invalid_request_id": "The request ID is invalid.",
                "invalid_text": "The speech text is invalid.",
                "text_too_long": "The speech text is too long.",
                "invalid_language": "The speech language is invalid.",
            }
            self._send_json_error(
                HTTPStatus.BAD_REQUEST,
                code if code in messages else "invalid_request",
                messages.get(code, "The speech request is invalid."),
            )
            return None

    def _reject_method(self) -> None:
        self._send_json_error(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "The HTTP method is not allowed.",
            extra_headers={"Allow": "GET, POST"},
        )

    def _send_wav(self, response: TtsAudioResponse) -> None:
        headers = {
            "Content-Type": "audio/wav",
            "Content-Length": str(len(response.wav)),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Audio-SHA256": response.sha256,
            "X-Audio-Duration-Ms": str(response.duration_ms),
            "X-TTS-Engine": response.engine,
            "X-TTS-Language": response.language,
            "X-Request-Id": response.request_id,
        }
        try:
            self.send_response(HTTPStatus.OK)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response.wav)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True

    def _send_json_error(
        self,
        status: int,
        code: str,
        message: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._send_json(
            status,
            {"error": {"code": code, "message": message}},
            extra_headers=extra_headers,
        )

    def _send_json(
        self,
        status: int,
        payload: dict[str, object],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.close_connection = True


def run_tts_server(
    *,
    host: str,
    port: int,
    token_file: Path,
    device: str,
) -> int:
    validate_tts_server_bind(host, port)
    token = load_owner_only_bearer_token(token_file)
    service = TtsService(bearer_token=token, device=device)
    service.prewarm()
    try:
        server = TtsHttpServer((host, port), service)
    except OSError as exc:
        raise TtsServerError(
            "TTS server could not bind to the configured private address."
        ) from exc
    previous_sigbreak: Any = None
    sigbreak_number = getattr(signal, "SIGBREAK", None)
    sigbreak = signal.Signals(sigbreak_number) if sigbreak_number is not None else None
    if sys.platform == "win32" and sigbreak is not None:
        previous_sigbreak = signal.getsignal(sigbreak)
        signal.signal(sigbreak, _raise_keyboard_interrupt)
    try:
        print(f"TTS server ready on {host}:{port} (kokoro, silero; device={device}).")
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
    finally:
        server.server_close()
        if previous_sigbreak is not None and sigbreak is not None:
            signal.signal(sigbreak, previous_sigbreak)
    return 0


def validate_tts_server_bind(host: str, port: int) -> None:
    try:
        address = ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError as exc:
        raise ValueError(
            "TTS server host must be exact loopback, Tailscale IPv4, or RFC1918 IPv4."
        ) from exc
    private_lan = any(
        address in network
        for network in (
            ipaddress.IPv4Network("10.0.0.0/8"),
            ipaddress.IPv4Network("172.16.0.0/12"),
            ipaddress.IPv4Network("192.168.0.0/16"),
        )
    )
    tailscale = address in ipaddress.IPv4Network("100.64.0.0/10")
    if host != "127.0.0.1" and not private_lan and not tailscale:
        raise ValueError("TTS server host must be exact loopback, Tailscale IPv4, or RFC1918 IPv4.")
    if isinstance(port, bool) or not 1 <= port <= 65_535:
        raise ValueError("TTS server port must be between 1 and 65535.")


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
    del signum, frame
    raise KeyboardInterrupt


def load_owner_only_bearer_token(path: Path) -> str:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or _is_reparse_point(path, before):
            raise OSError("unsafe token file type")
        if before.st_size <= 0 or before.st_size > MAX_TOKEN_BYTES:
            raise OSError("unsafe token file size")
        _verify_owner_only_permissions(path, before)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            if not _same_file(before, opened):
                raise OSError("token file changed before read")
            raw = os.read(descriptor, MAX_TOKEN_BYTES + 1)
        finally:
            os.close(descriptor)
        after = path.lstat()
        if not _same_file(before, after) or _is_reparse_point(path, after):
            raise OSError("token file changed during read")
        _verify_owner_only_permissions(path, after)
        token = raw.decode("ascii").rstrip("\r\n")
        if (
            len(token) < MIN_TOKEN_CHARACTERS
            or len(token.encode("ascii")) > MAX_TOKEN_BYTES
            or _TOKEN_PATTERN.fullmatch(token) is None
        ):
            raise OSError("unsafe token format")
        return token
    except (OSError, UnicodeError) as exc:
        raise TtsServerError(
            "TTS server token file must be a stable owner-only regular file containing one "
            "strong bearer token."
        ) from exc


def _synthesize_prepared_text(text: str, language: str, device: str) -> TtsifyResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ttsify_text(
            text,
            TtsifyOptions(
                no_transform=True,
                language=_API_TO_TTSIFY_LANGUAGE[language],
                device=device,
            ),
        )


def _validate_synthesized_result(
    *,
    request_id: str,
    result: TtsifyResult,
    requested_language: str,
) -> TtsAudioResponse:
    wav = result.tts_result.wav
    if not isinstance(wav, bytes) or not wav:
        raise InvalidSynthesizedAudio("The synthesized WAV size is invalid.")
    if len(wav) > MAX_AUDIO_BYTES:
        raise SynthesizedAudioTooLarge("The synthesized WAV is too large.")
    if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise InvalidSynthesizedAudio("The synthesized WAV container is invalid.")
    declared_size = int.from_bytes(wav[4:8], byteorder="little", signed=False) + 8
    if declared_size != len(wav):
        raise InvalidSynthesizedAudio("The synthesized WAV length is invalid.")
    try:
        with wave.open(io.BytesIO(wav), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            compression = wav_file.getcomptype()
            pcm = wav_file.readframes(frames + 1)
    except (EOFError, wave.Error) as exc:
        raise InvalidSynthesizedAudio("The synthesized WAV cannot be parsed.") from exc
    if (
        channels != 1
        or sample_width != 2
        or sample_rate <= 0
        or frames <= 0
        or compression != "NONE"
        or len(pcm) != frames * channels * sample_width
    ):
        raise InvalidSynthesizedAudio("The synthesized WAV format is invalid.")
    engine = result.tts_result.engine
    language = _TTSIFY_TO_API_LANGUAGE.get(result.tts_result.language or "")
    expected_engine = (
        "kokoro"
        if requested_language in {"en-US", "en-GB"}
        else "silero"
        if requested_language in {"ru", "ru-RU"}
        else None
    )
    expected_language = (
        requested_language
        if requested_language in {"en-US", "en-GB"}
        else "ru-RU"
        if requested_language in {"ru", "ru-RU"}
        else None
    )
    if (
        engine not in {"kokoro", "silero"}
        or language is None
        or (expected_engine is not None and engine != expected_engine)
        or (expected_language is not None and language != expected_language)
    ):
        raise InvalidSynthesizedAudio("The synthesized metadata is invalid.")
    duration_ms = max(1, (frames * 1_000) // sample_rate)
    return TtsAudioResponse(
        request_id=request_id,
        wav=wav,
        sha256=hashlib.sha256(wav).hexdigest(),
        duration_ms=duration_ms,
        engine=engine,
        language=language,
    )


def _validate_request_payload(payload: object) -> TtsRequest:
    if not isinstance(payload, dict) or set(payload) != _REQUEST_FIELDS:
        raise ValueError("invalid_fields")
    request_id = payload.get("requestId")
    text = payload.get("text")
    language = payload.get("language")
    if not isinstance(request_id, str) or _REQUEST_ID_PATTERN.fullmatch(request_id) is None:
        raise ValueError("invalid_request_id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("invalid_text")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError("text_too_long")
    if not isinstance(language, str) or language not in ALLOWED_API_LANGUAGES:
        raise ValueError("invalid_language")
    return TtsRequest(request_id=request_id, text=text, language=language)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _is_json_content_type(value: str | None) -> bool:
    if value is None:
        return False
    parts = [part.strip() for part in value.split(";")]
    if not parts or parts[0].lower() != "application/json":
        return False
    parameters = parts[1:]
    return len(parameters) <= 1 and all(
        parameter.lower() == "charset=utf-8" for parameter in parameters
    )


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _is_reparse_point(path: Path, metadata: os.stat_result) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_attribute)


def _verify_owner_only_permissions(path: Path, metadata: os.stat_result) -> None:
    if sys.platform == "win32":
        _verify_windows_owner_only_acl(path)
        return
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise OSError("token file is not owner-only")


def _verify_windows_owner_only_acl(path: Path) -> None:
    try:
        win32api = importlib.import_module("win32api")
        win32con = importlib.import_module("win32con")
        win32security = importlib.import_module("win32security")
        process_token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
        )
        current_sid = win32security.GetTokenInformation(process_token, win32security.TokenUser)[0]
        descriptor = win32security.GetNamedSecurityInfo(
            str(path),
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION,
        )
        owner_sid = descriptor.GetSecurityDescriptorOwner()
        acl = descriptor.GetSecurityDescriptorDacl()
        current_sid_text = win32security.ConvertSidToStringSid(current_sid)
        owner_sid_text = win32security.ConvertSidToStringSid(owner_sid)
        if owner_sid_text != current_sid_text or acl is None or acl.GetAceCount() < 1:
            raise OSError("token owner or ACL is invalid")
        owner_has_access = False
        access_granting_ace_types = {
            win32security.ACCESS_ALLOWED_ACE_TYPE,
            getattr(win32security, "ACCESS_ALLOWED_COMPOUND_ACE_TYPE", 4),
            getattr(win32security, "ACCESS_ALLOWED_OBJECT_ACE_TYPE", 5),
            getattr(win32security, "ACCESS_ALLOWED_CALLBACK_ACE_TYPE", 9),
            getattr(win32security, "ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE", 11),
        }
        for index in range(acl.GetAceCount()):
            ace = acl.GetAce(index)
            ace_type = ace[0][0]
            if ace_type not in access_granting_ace_types:
                continue
            if ace_type != win32security.ACCESS_ALLOWED_ACE_TYPE:
                raise OSError("token ACL uses an unsupported access grant")
            ace_sid_text = win32security.ConvertSidToStringSid(ace[2])
            if ace_sid_text != current_sid_text:
                raise OSError("token ACL grants another principal access")
            owner_has_access = True
        if not owner_has_access:
            raise OSError("token ACL does not grant owner access")
    except ImportError:
        _verify_windows_owner_only_acl_native(path)
    except OSError as exc:
        raise OSError("token ACL could not be verified") from exc
    except Exception as exc:
        raise OSError("token ACL could not be verified") from exc


def _verify_windows_owner_only_acl_native(path: Path) -> None:
    from ctypes import wintypes

    token_query = 0x0008
    token_user = 1
    se_file_object = 1
    owner_security_information = 0x00000001
    dacl_security_information = 0x00000004
    access_allowed_ace_type = 0
    access_granting_ace_types = {0, 4, 5, 9, 11}

    class Acl(ctypes.Structure):
        _fields_ = [
            ("revision", ctypes.c_ubyte),
            ("reserved1", ctypes.c_ubyte),
            ("size", wintypes.WORD),
            ("ace_count", wintypes.WORD),
            ("reserved2", wintypes.WORD),
        ]

    class SidAndAttributes(ctypes.Structure):
        _fields_ = [("sid", wintypes.LPVOID), ("attributes", wintypes.DWORD)]

    class TokenUser(ctypes.Structure):
        _fields_ = [("user", SidAndAttributes)]

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(ctypes.POINTER(Acl)),
        ctypes.POINTER(ctypes.POINTER(Acl)),
        ctypes.POINTER(wintypes.LPVOID),
    ]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.GetAce.argtypes = [
        ctypes.POINTER(Acl),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
    ]
    advapi32.GetAce.restype = wintypes.BOOL
    advapi32.EqualSid.argtypes = [wintypes.LPVOID, wintypes.LPVOID]
    advapi32.EqualSid.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL

    token_handle = wintypes.HANDLE()
    security_descriptor = wintypes.LPVOID()
    try:
        if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(), token_query, ctypes.byref(token_handle)
        ):
            raise OSError("current user token could not be opened")
        required = wintypes.DWORD()
        advapi32.GetTokenInformation(
            token_handle,
            token_user,
            None,
            0,
            ctypes.byref(required),
        )
        if required.value <= 0:
            raise OSError("current user identity could not be measured")
        token_buffer = ctypes.create_string_buffer(required.value)
        if not advapi32.GetTokenInformation(
            token_handle,
            token_user,
            token_buffer,
            required,
            ctypes.byref(required),
        ):
            raise OSError("current user identity could not be read")
        current_sid = ctypes.cast(token_buffer, ctypes.POINTER(TokenUser)).contents.user.sid

        owner_sid = wintypes.LPVOID()
        dacl = ctypes.POINTER(Acl)()
        status = advapi32.GetNamedSecurityInfoW(
            str(path),
            se_file_object,
            owner_security_information | dacl_security_information,
            ctypes.byref(owner_sid),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(security_descriptor),
        )
        if status != 0 or not owner_sid or not dacl:
            raise OSError("token security descriptor could not be read")
        if not advapi32.EqualSid(owner_sid, current_sid):
            raise OSError("token owner does not match the current user")
        if dacl.contents.ace_count < 1:
            raise OSError("token ACL is empty")
        owner_has_access = False
        for index in range(dacl.contents.ace_count):
            ace = wintypes.LPVOID()
            if not advapi32.GetAce(dacl, index, ctypes.byref(ace)):
                raise OSError("token ACL entry could not be read")
            ace_address = ace.value
            if ace_address is None:
                raise OSError("token ACL entry is missing")
            ace_type = ctypes.c_ubyte.from_address(ace_address).value
            if ace_type not in access_granting_ace_types:
                continue
            if ace_type != access_allowed_ace_type:
                raise OSError("token ACL uses an unsupported access grant")
            ace_sid = wintypes.LPVOID(ace_address + 8)
            if not advapi32.EqualSid(ace_sid, current_sid):
                raise OSError("token ACL grants another principal access")
            owner_has_access = True
        if not owner_has_access:
            raise OSError("token ACL does not grant owner access")
    except (OSError, ValueError) as exc:
        raise OSError("token ACL could not be verified") from exc
    finally:
        if security_descriptor:
            kernel32.LocalFree(security_descriptor)
        if token_handle:
            kernel32.CloseHandle(token_handle)
