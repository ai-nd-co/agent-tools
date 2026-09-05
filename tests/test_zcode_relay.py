from __future__ import annotations

import base64
import hashlib
import struct
import threading
from pathlib import Path

import pytest

import agent_tools.zcode_relay as relay
from agent_tools.zcode_relay import (
    AcknowledgedRelayProtocol,
    ChannelRpcClient,
    RelayClosedError,
    RelayConflictError,
    RelayFrameIdentity,
    RelayIdentity,
    RelayTerminal,
    VSBuffer,
    WebSocketFrameCodec,
    WebSocketProtocolError,
    ZCodeRelayError,
    _channel_remote_error_code,
    _parse_http_headers,
    _relay_error,
    _strict_json_object,
    decode_message,
    encode_logical_message,
    encode_message,
    encode_websocket_frame,
)


def test_vsbuffer_tag_three_and_nested_values_round_trip() -> None:
    direct = VSBuffer(b"\x00buffer\xff")
    nested = {"direct": direct, "bytes": b"nested"}

    decoded = decode_message(encode_message(direct, nested))

    assert decoded == [direct, nested]
    assert encode_message(direct).startswith(b"\x03")


def test_relay_rebind_replays_with_new_generation_and_deduplicates() -> None:
    original = RelayFrameIdentity("bridge", 1, "recovery")
    rebound = RelayFrameIdentity("bridge", 2, "recovery")
    first_transport: list[dict[str, object]] = []
    replay_transport: list[dict[str, object]] = []
    sender = AcknowledgedRelayProtocol(original, first_transport.append, maximum_frame_bytes=1024)

    sender.send(b"x" * 2048)
    sender.rebind(rebound, replay_transport.append)
    sender.replay_unacknowledged()

    assert len(first_transport) > 1
    assert replay_transport
    assert {frame["bridgeGeneration"] for frame in replay_transport} == {2}
    received: list[bytes] = []
    acknowledgements: list[dict[str, object]] = []
    receiver = AcknowledgedRelayProtocol(
        rebound,
        acknowledgements.append,
        maximum_frame_bytes=1024,
    )
    receiver.on_message(received.append)
    for frame in replay_transport:
        assert receiver.accept(frame)
    for frame in replay_transport:
        assert receiver.accept(frame)

    assert received == [b"x" * 2048]
    assert acknowledgements[-1]["ackMessageSeq"] == 1


def test_relay_conflicting_duplicate_and_stale_rebind_fail_closed() -> None:
    identity = RelayFrameIdentity("bridge", 5, "recovery")
    frames, _ = encode_logical_message(b"hello", identity, first_seq=1, message_seq=1)
    receiver = AcknowledgedRelayProtocol(identity, lambda _value: None)
    assert receiver.accept(frames[0])
    conflict = dict(frames[0])
    conflict["dataBase64"] = base64.b64encode(b"jello").decode("ascii")

    assert receiver.accept(conflict)
    assert receiver.degraded_code == "conflicting_duplicate"

    sender = AcknowledgedRelayProtocol(identity, lambda _value: None)
    with pytest.raises(ZCodeRelayError, match="stale_bridge_generation"):
        sender.rebind(RelayFrameIdentity("bridge", 5, "recovery"), lambda _value: None)


@pytest.mark.parametrize("reason", ["session-conflict", "paired terminal kicked"])
def test_relay_kick_and_conflict_are_distinct_fatal_errors(reason: str) -> None:
    assert isinstance(_relay_error({"reason": reason}), RelayConflictError)


def test_websocket_fragmentation_allows_interleaved_control_frame() -> None:
    codec = WebSocketFrameCodec(expect_masked=False)
    wire = b"".join(
        (
            encode_websocket_frame(1, b"hel", masked=False, fin=False),
            encode_websocket_frame(9, b"ping", masked=False),
            encode_websocket_frame(0, b"lo", masked=False),
        )
    )

    assert codec.feed(wire) == [(9, b"ping"), (1, b"hello")]


def test_websocket_masked_frame_decodes_for_server_direction() -> None:
    codec = WebSocketFrameCodec(expect_masked=True)

    assert codec.feed(encode_websocket_frame(1, b"hello", masked=True, mask_key=b"mask")) == [
        (1, b"hello")
    ]


@pytest.mark.parametrize(
    ("wire", "code"),
    [
        (b"\xc1\x00", "websocket_rsv_invalid"),
        (b"\x81\x00", "websocket_mask_direction_invalid"),
        (b"\x09\x80mask", "websocket_control_frame_invalid"),
        (b"\x82\xfe\x00\x01", "websocket_length_noncanonical"),
        (
            encode_websocket_frame(1, b"\xff", masked=True, mask_key=b"mask"),
            "websocket_text_utf8_invalid",
        ),
        (
            encode_websocket_frame(8, struct.pack("!H", 1005), masked=True, mask_key=b"mask"),
            "websocket_close_code_invalid",
        ),
    ],
)
def test_websocket_rejects_malformed_frames(wire: bytes, code: str) -> None:
    codec = WebSocketFrameCodec(expect_masked=True)

    with pytest.raises(WebSocketProtocolError, match=code):
        codec.feed(wire)


def test_websocket_rejects_oversized_physical_frame_before_buffering_payload() -> None:
    codec = WebSocketFrameCodec(expect_masked=False, maximum_frame_bytes=4)

    with pytest.raises(WebSocketProtocolError, match="websocket_frame_too_large"):
        codec.feed(encode_websocket_frame(2, b"12345", masked=False))


def test_websocket_upgrade_allows_repeatable_benign_response_headers() -> None:
    status, headers = _parse_http_headers(
        b"HTTP/1.1 101 Switching Protocols\r\nSet-Cookie: a=1\r\nSet-Cookie: b=2"
    )

    assert status == "HTTP/1.1 101 Switching Protocols"
    assert headers["set-cookie"] == "a=1,b=2"


def test_websocket_upgrade_rejects_duplicate_security_headers() -> None:
    with pytest.raises(ZCodeRelayError, match="websocket_upgrade_duplicate_header"):
        _parse_http_headers(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Sec-WebSocket-Accept: one\r\nSec-WebSocket-Accept: two"
        )


@pytest.mark.parametrize("payload", [encode_message([200]), encode_message([200], None)])
def test_channel_initialization_accepts_canonical_optional_undefined_payload(
    payload: bytes,
) -> None:
    identity = RelayFrameIdentity("bridge", 1, "recovery")
    protocol = AcknowledgedRelayProtocol(identity, lambda _value: None)
    client = ChannelRpcClient(protocol)

    client._accept(payload)

    client.wait_initialized(0.01)


def test_channel_initialization_rejects_noncanonical_payload() -> None:
    identity = RelayFrameIdentity("bridge", 1, "recovery")
    protocol = AcknowledgedRelayProtocol(identity, lambda _value: None)
    client = ChannelRpcClient(protocol)

    client._accept(encode_message([200], "unexpected"))

    with pytest.raises(ZCodeRelayError, match="channel_initialize_invalid"):
        client.wait_initialized(0.01)


@pytest.mark.parametrize(
    ("remote", "code"),
    [
        ({"code": "BAD_REQUEST"}, "channel_remote_error"),
        ({"code": "session_very_secret_token"}, "channel_remote_error"),
        ({"message": "connection is untrusted"}, "channel_remote_untrusted"),
        ({"message": "session is not active"}, "channel_remote_inactive"),
        ({"reason": "unsupported method"}, "channel_remote_unsupported"),
        ({"message": "sensitive detail"}, "channel_remote_error"),
    ],
)
def test_channel_remote_errors_are_bounded_categories(remote: object, code: str) -> None:
    assert _channel_remote_error_code(remote) == code


def test_websocket_rejects_excessive_zero_byte_fragments() -> None:
    codec = WebSocketFrameCodec(expect_masked=False)
    codec.feed(encode_websocket_frame(1, b"", masked=False, fin=False))
    continuation = encode_websocket_frame(0, b"", masked=False, fin=False)

    for _ in range(1023):
        codec.feed(continuation)
    with pytest.raises(WebSocketProtocolError, match="websocket_fragment_count_exceeded"):
        codec.feed(continuation)


def test_serializer_depth_is_bounded_on_encode_and_decode() -> None:
    encoded = b"\x00"
    for _ in range(66):
        encoded = b"\x04\x01" + encoded
    with pytest.raises(ZCodeRelayError, match="serializer_depth_exceeded"):
        decode_message(encoded)

    nested: object = None
    for _ in range(100):
        nested = [nested]
    with pytest.raises(ZCodeRelayError, match="serializer_depth_exceeded"):
        encode_message(nested)


def test_relay_assembly_expires_without_another_payload() -> None:
    identity = RelayFrameIdentity("bridge", 1, "recovery")
    frames, _ = encode_logical_message(
        b"x" * 2048,
        identity,
        first_seq=1,
        message_seq=1,
        maximum_frame_bytes=1024,
    )
    degraded = threading.Event()
    receiver = AcknowledgedRelayProtocol(
        identity,
        lambda _value: None,
        maximum_frame_bytes=1024,
        assembly_timeout_seconds=0.01,
    )
    receiver.on_degraded(lambda _code: degraded.set())

    assert len(frames) > 1
    assert receiver.accept(frames[0])
    assert degraded.wait(0.5)
    assert receiver.degraded_code == "assembly_timeout"


def test_pairing_store_rejects_unverifiable_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pair.json"
    path.write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        relay,
        "_verify_pairing_permissions",
        lambda _path, _metadata: (_ for _ in ()).throw(OSError("unsafe")),
    )

    with pytest.raises(ZCodeRelayError, match="relay_pair_store_invalid"):
        relay._stable_json_file(path, 100)


def test_websocket_client_rejects_subprotocol_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_nonce = b"0" * 16
    key = base64.b64encode(fixed_nonce).decode("ascii")
    accept = base64.b64encode(
        hashlib.sha1((key + relay._GUID).encode("ascii")).digest()
    ).decode("ascii")

    class SocketFixture:
        def __init__(self) -> None:
            self.closed = False

        def settimeout(self, _timeout: float) -> None:
            return None

        def sendall(self, _payload: bytes) -> None:
            return None

        def recv(self, _size: int) -> bytes:
            return (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "Sec-WebSocket-Protocol: wrong.protocol\r\n\r\n"
            ).encode("ascii")

        def close(self) -> None:
            self.closed = True

    transport = SocketFixture()
    monkeypatch.setattr(relay.secrets, "token_bytes", lambda _size: fixed_nonce)
    monkeypatch.setattr(relay.socket, "create_connection", lambda *_args: transport)

    with pytest.raises(ZCodeRelayError, match="websocket_upgrade_rejected"):
        relay.WebSocketConnection.connect(
            "ws://example.invalid/ws",
            headers={"Sec-WebSocket-Protocol": "agenttools.zcode.v1"},
        )
    assert transport.closed


def test_relay_receiver_normalizes_socket_failure() -> None:
    class BrokenConnection:
        @staticmethod
        def receive() -> tuple[int, bytes]:
            raise OSError("socket failed")

    terminal = RelayTerminal(RelayIdentity("sid", "mid", "hash"))
    terminal._connection = BrokenConnection()  # type: ignore[assignment]

    terminal._receive_loop()

    assert isinstance(terminal.failure, RelayClosedError)
    assert terminal.failure.code == "relay_io_error"


def test_channel_call_waits_for_replay_after_generic_staged_send_error() -> None:
    first_send = threading.Event()

    def broken_send(_payload: dict[str, object]) -> None:
        first_send.set()
        raise ZCodeRelayError("relay_schema_drift")

    protocol = AcknowledgedRelayProtocol(
        RelayFrameIdentity("bridge", 1, "recovery"), broken_send
    )
    client = ChannelRpcClient(protocol)
    results: list[object] = []
    failures: list[BaseException] = []

    def call() -> None:
        try:
            results.append(client.call("zcode-task", "createTask", {}, 1.0))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=call)
    worker.start()
    assert first_send.wait(0.5)
    replayed: list[dict[str, object]] = []
    protocol.rebind(RelayFrameIdentity("bridge", 2, "recovery"), replayed.append)
    protocol.replay_unacknowledged()
    client._accept(encode_message([201, 1], {"taskId": "one"}))
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert failures == []
    assert results == [{"taskId": "one"}]
    assert len(replayed) == 1
    replay_message = base64.b64decode(str(replayed[0]["dataBase64"]), validate=True)
    assert decode_message(replay_message) == [
        [100, 1, "zcode-task", "createTask"],
        [{}],
    ]


def test_channel_timeout_retires_ambiguous_mutation_before_returning_unknown() -> None:
    def broken_send(_payload: dict[str, object]) -> None:
        raise ZCodeRelayError("relay_schema_drift")

    protocol = AcknowledgedRelayProtocol(
        RelayFrameIdentity("bridge", 1, "recovery"), broken_send
    )
    client = ChannelRpcClient(protocol)

    with pytest.raises(ZCodeRelayError, match="channel_request_outcome_unknown"):
        client.call("zcode-task", "createTask", {}, 0.01)

    assert protocol.degraded_code == "channel_request_outcome_unknown"
    assert protocol._unacked == {}
    with pytest.raises(ZCodeRelayError, match="channel_request_outcome_unknown"):
        protocol.rebind(
            RelayFrameIdentity("bridge", 2, "recovery"),
            lambda _payload: None,
        )


def test_channel_abort_retires_inflight_mutation_and_reports_unknown() -> None:
    sent = threading.Event()

    def accept_send(_payload: dict[str, object]) -> None:
        sent.set()

    protocol = AcknowledgedRelayProtocol(
        RelayFrameIdentity("bridge", 1, "recovery"), accept_send
    )
    client = ChannelRpcClient(protocol)
    failures: list[str] = []

    def call() -> None:
        try:
            client.call("zcode-task", "createTask", {}, 1.0)
        except ZCodeRelayError as exc:
            failures.append(exc.code)

    worker = threading.Thread(target=call)
    worker.start()
    assert sent.wait(0.5)
    client.abort(ZCodeRelayError("bridge_stopping"))
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert failures == ["channel_request_outcome_unknown"]
    assert protocol._unacked == {}
    assert protocol.degraded_code == "bridge_stopping"


@pytest.mark.parametrize(
    "payload",
    [
        b'{"type":NaN}',
        (b'{"type":' + b"[" * 70 + b"null" + b"]" * 70 + b"}"),
    ],
)
def test_relay_json_rejects_nonfinite_and_deep_values(payload: bytes) -> None:
    with pytest.raises(ZCodeRelayError, match="relay_json_invalid"):
        _strict_json_object(payload, "relay_json_invalid")
