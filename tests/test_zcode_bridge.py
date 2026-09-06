from __future__ import annotations

import hashlib
import json
import socket
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_tools.zcode_bridge as bridge
import agent_tools.zcode_relay as relay
from agent_tools.voice_onboarding import AdapterAddress, AdapterSnapshot, TailscaleSnapshot
from agent_tools.zcode_bridge import (
    BridgeConfig,
    BridgeError,
    BridgeStatus,
    RuntimeState,
    _BridgeService,
    _ExactBridgeServer,
    _execute_client_request,
    _parse_client_message,
    _serve_client,
    prepare_bridge,
    start_bridge,
    stop_bridge,
    validate_bridge_bind,
)

TAILSCALE_IP = "100.64.10.20"
BEARER = "task320-local-bearer-0123456789abcdef"


def test_child_interpreter_disables_bytecode_in_protected_package(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda argv, **kwargs: captured.append(argv))
    bridge._spawn_service(_config(tmp_path, 50000), "fixture")
    assert captured[0][1:4] == ("-I", "-B", "-c")


@dataclass
class FakeNetwork:
    adapters_value: tuple[AdapterSnapshot, ...]
    tailscale_value: TailscaleSnapshot

    def adapters(self) -> tuple[AdapterSnapshot, ...]:
        return self.adapters_value

    def tailscale(self) -> TailscaleSnapshot:
        return self.tailscale_value


def _adapter(name: str, address: str, *, metric: int = 10) -> AdapterSnapshot:
    return AdapterSnapshot(
        interface_index=metric,
        name=name,
        description=name,
        kind="tunnel",
        operational=True,
        physical=False,
        gateway=None,
        route_metric=metric,
        addresses=(AdapterAddress(address, True),),
    )


def _network(*addresses: str) -> FakeNetwork:
    return FakeNetwork(
        tuple(_adapter(f"Tailscale {index}", address) for index, address in enumerate(addresses)),
        TailscaleSnapshot("healthy", tuple(addresses)),
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _config(tmp_path: Path, port: int) -> BridgeConfig:
    executable = Path(__file__).resolve()
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    bearer = tmp_path / "bearer"
    bearer.write_text(BEARER, encoding="ascii")
    return BridgeConfig(
        root=tmp_path,
        config_sha256="a" * 64,
        config_fingerprint="b" * 64,
        bind_address="127.0.0.1",
        port=port,
        network_class="test",
        workspace=tmp_path,
        workspace_fingerprint=bridge._workspace_fingerprint(tmp_path),
        zcode_cli=executable,
        zcode_cli_sha256=digest,
        zcode_version="0.16.5",
        python_executable=executable,
        python_executable_sha256=digest,
        module_file=executable,
        module_sha256=digest,
        package_init_file=executable,
        package_init_sha256=digest,
        relay_module_file=executable,
        relay_module_sha256=digest,
        voice_module_file=executable,
        voice_module_sha256=digest,
        startup_module_file=executable,
        startup_module_sha256=digest,
        bearer_file=bearer,
        bearer_fingerprint="c" * 64,
        server_id="server",
        identity_fingerprint="d" * 64,
        relay_url="wss://example.invalid/ws",
    )


class FakeController:
    relay_generation = 4
    reconnect_count = 0

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str, object, float]] = []
        self.event_handler: object | None = None
        self.local_delivery_failed = False

    def call(
        self, generation: int, channel: str, method: str, argument: object, timeout: float
    ) -> object:
        if generation != self.relay_generation:
            raise BridgeError("stale_generation", "stale", "reconnect")
        self.calls.append((generation, channel, method, argument, timeout))
        return {"accepted": True}

    def add_generation_handler(self, _handler: object) -> None:
        return None

    def remove_generation_handler(self, _handler: object) -> None:
        return None

    def listen(
        self,
        generation: int,
        _channel: str,
        _event: str,
        _argument: object,
        handler: object,
    ) -> object:
        assert generation == self.relay_generation
        self.event_handler = handler
        return SimpleNamespace(dispose=lambda: None)

    def fail_local_delivery(self) -> None:
        self.local_delivery_failed = True


def _service(tmp_path: Path, port: int) -> _BridgeService:
    config = _config(tmp_path, port)
    controller = FakeController()
    state = RuntimeState(
        1,
        config.config_sha256,
        config.config_fingerprint,
        config.bearer_fingerprint,
        config.module_sha256,
        10,
        20,
        "launch",
        "ready",
        controller.relay_generation,
        0,
        False,
        "ready",
    )
    return _BridgeService(config, state, controller)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "address",
    ["0.0.0.0", "127.0.0.1", "169.254.1.2", "8.8.8.8", "::1"],
)
def test_bridge_rejects_wildcard_loopback_link_local_public_and_ipv6(address: str) -> None:
    with pytest.raises(BridgeError) as failure:
        validate_bridge_bind(address, 55000)
    assert failure.value.code == "bind_not_private"


@pytest.mark.parametrize("port", [0, 49151, 65536])
def test_bridge_rejects_non_ephemeral_ports(port: int) -> None:
    with pytest.raises(BridgeError) as failure:
        validate_bridge_bind(TAILSCALE_IP, port)
    assert failure.value.code == "port_not_private_dynamic"


@pytest.mark.parametrize("port", [4222, 4223])
def test_bridge_rejects_reserved_live_ports(port: int) -> None:
    with pytest.raises(BridgeError) as failure:
        validate_bridge_bind(TAILSCALE_IP, port)
    assert failure.value.code == "port_reserved"


def test_prepare_rejects_ambiguous_explicit_tailscale_bind(tmp_path: Path) -> None:
    zcode = tmp_path / "zcode.cjs"
    zcode.write_text("fixture", encoding="ascii")

    with pytest.raises(BridgeError) as failure:
        prepare_bridge(
            tmp_path / "state",
            bind_address=TAILSCALE_IP,
            port=55000,
            workspace=tmp_path,
            network_mode="tailscale",
            trust_lan=False,
            network_backend=_network(TAILSCALE_IP, TAILSCALE_IP),
            zcode_cli=zcode,
        )
    assert failure.value.code == "bind_ip_not_current"


def test_prepare_rejects_wrong_zcode_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zcode = tmp_path / "zcode.cjs"
    zcode.write_text("fixture", encoding="ascii")
    monkeypatch.setattr(bridge.shutil, "which", lambda _name: str(zcode))
    monkeypatch.setattr(
        bridge,
        "_bounded_subprocess",
        lambda _command, timeout: SimpleNamespace(stdout=b"0.16.6\n", stderr=b"", returncode=0),
    )

    with pytest.raises(BridgeError) as failure:
        bridge._installed_zcode_version(zcode)
    assert failure.value.code == "zcode_version_mismatch"


def test_prepare_reports_occupied_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zcode = tmp_path / "zcode.cjs"
    zcode.write_text("fixture", encoding="ascii")
    monkeypatch.setattr(bridge, "_installed_zcode_version", lambda _path: "0.16.5")
    monkeypatch.setattr(
        bridge,
        "_check_port_available",
        lambda _address, _port: (_ for _ in ()).throw(
            BridgeError("port_occupied", "occupied", "choose another")
        ),
    )

    with pytest.raises(BridgeError) as failure:
        prepare_bridge(
            tmp_path / "state",
            bind_address=TAILSCALE_IP,
            port=55000,
            workspace=tmp_path,
            network_mode="tailscale",
            trust_lan=False,
            network_backend=_network(TAILSCALE_IP),
            zcode_cli=zcode,
        )
    assert failure.value.code == "port_occupied"


def test_bridge_refuses_bad_bearer_before_websocket_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _free_port()
    service = _service(tmp_path, port)
    monkeypatch.setattr(bridge, "_bearer_value", lambda _path: BEARER)
    monkeypatch.setattr(service, "publish", lambda _phase, _code: None)
    server = _ExactBridgeServer(service)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as client:
            client.sendall(
                (
                    "GET /zcode/v1 HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{port}\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    "Sec-WebSocket-Key: MDEyMzQ1Njc4OWFiY2RlZg==\r\n"
                    "Sec-WebSocket-Version: 13\r\n"
                    "Sec-WebSocket-Protocol: agenttools.zcode.v1\r\n"
                    "Authorization: Bearer wrong\r\n\r\n"
                ).encode("ascii")
            )
            assert client.recv(4096).startswith(b"HTTP/1.1 401 Unauthorized")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert service.client_connected is False


def test_client_call_rejects_stale_generation_and_bounds_timeout(tmp_path: Path) -> None:
    service = _service(tmp_path, _free_port())
    subscriptions: dict[str, object] = {}
    request = {
        "id": "one",
        "op": "call",
        "generation": 3,
        "channel": "zcode-task",
        "method": "getTaskSnapshot",
        "arg": {},
        "timeoutMs": 1000,
    }

    with pytest.raises(BridgeError) as failure:
        _execute_client_request(request, service, subscriptions, lambda _value: None)  # type: ignore[arg-type]
    assert failure.value.code == "stale_generation"

    request["generation"] = 4
    request["timeoutMs"] = 120001
    with pytest.raises(BridgeError) as failure:
        _execute_client_request(request, service, subscriptions, lambda _value: None)  # type: ignore[arg-type]
    assert failure.value.code == "client_timeout_invalid"


def test_stop_marker_is_generation_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, _free_port())
    monkeypatch.setattr(bridge, "_read_private", lambda path, _maximum: path.read_bytes())
    monkeypatch.setattr(
        bridge,
        "_write_private_json",
        lambda path, value, replace: path.write_text(json.dumps(value), encoding="ascii"),
    )

    bridge._write_stop(service.config, service.state)
    assert bridge._stop_requested(service.config, service.state)
    newer = RuntimeState(**{**service.state.__dict__, "process_creation_time_ns": 21})
    assert not bridge._stop_requested(service.config, newer)


class FakeConnection:
    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = payloads
        self.sent: list[bytes] = []
        self.closed = False

    def receive(self) -> tuple[int, bytes]:
        if not self.payloads:
            raise OSError("fixture complete")
        return 1, self.payloads.pop(0)

    def _send(self, opcode: int, payload: bytes) -> None:
        assert opcode == 1
        self.sent.append(payload)

    def close(self) -> None:
        self.closed = True


def test_client_duplicate_request_is_suppressed_and_cached(tmp_path: Path) -> None:
    service = _service(tmp_path, _free_port())
    request = json.dumps(
        {
            "id": "same",
            "op": "call",
            "generation": 4,
            "channel": "zcode-task",
            "method": "getTaskSnapshot",
            "arg": {},
            "timeoutMs": 1000,
        },
        separators=(",", ":"),
    ).encode("ascii")
    connection = FakeConnection([request, request])

    with pytest.raises(OSError, match="fixture complete"):
        _serve_client(connection, service)  # type: ignore[arg-type]

    assert len(service.controller.calls) == 1  # type: ignore[attr-defined]
    assert connection.sent[1] == connection.sent[2]
    assert connection.closed


def test_generic_staged_send_replays_once_then_caches_authoritative_result(
    tmp_path: Path,
) -> None:
    first_send = threading.Event()
    replayed: list[dict[str, object]] = []

    def generic_failure(_payload: dict[str, object]) -> None:
        first_send.set()
        raise relay.ZCodeRelayError("relay_schema_drift")

    protocol = relay.AcknowledgedRelayProtocol(
        relay.RelayFrameIdentity("bridge", 1, "recovery"), generic_failure
    )
    rpc = relay.ChannelRpcClient(protocol)

    class ReplayController(FakeController):
        def call(
            self,
            generation: int,
            channel: str,
            method: str,
            argument: object,
            timeout: float,
        ) -> object:
            self.calls.append((generation, channel, method, argument, timeout))
            try:
                return rpc.call(channel, method, argument, timeout)
            except relay.ZCodeRelayError as exc:
                raise BridgeError(exc.code, "relay failed", "check status") from exc

    service = _service(tmp_path, _free_port())
    controller = ReplayController()
    service.controller = controller  # type: ignore[assignment]
    request = json.dumps(
        {
            "id": "generic-send-replay",
            "op": "call",
            "generation": 4,
            "channel": "zcode-task",
            "method": "createTask",
            "arg": {},
            "timeoutMs": 1000,
        },
        separators=(",", ":"),
    ).encode("ascii")
    connection = FakeConnection([request, request])

    def reconnect() -> None:
        assert first_send.wait(0.5)
        protocol.rebind(
            relay.RelayFrameIdentity("bridge", 2, "recovery"), replayed.append
        )
        protocol.replay_unacknowledged()
        rpc._accept(relay.encode_message([201, 1], {"taskId": "one"}))

    reconnect_worker = threading.Thread(target=reconnect)
    reconnect_worker.start()
    with pytest.raises(OSError, match="fixture complete"):
        _serve_client(connection, service)  # type: ignore[arg-type]
    reconnect_worker.join(timeout=1.0)

    assert not reconnect_worker.is_alive()
    assert len(controller.calls) == 1
    assert len(replayed) == 1
    assert json.loads(connection.sent[1]) == {
        "id": "generic-send-replay",
        "ok": True,
        "result": {"taskId": "one"},
    }
    assert connection.sent[1] == connection.sent[2]


def test_generic_staged_send_timeout_caches_unknown_after_retiring_replay(
    tmp_path: Path,
) -> None:
    def generic_failure(_payload: dict[str, object]) -> None:
        raise relay.ZCodeRelayError("relay_schema_drift")

    protocol = relay.AcknowledgedRelayProtocol(
        relay.RelayFrameIdentity("bridge", 1, "recovery"), generic_failure
    )
    rpc = relay.ChannelRpcClient(protocol)

    class TimeoutController(FakeController):
        def call(
            self,
            generation: int,
            channel: str,
            method: str,
            argument: object,
            timeout: float,
        ) -> object:
            self.calls.append((generation, channel, method, argument, timeout))
            try:
                return rpc.call(channel, method, argument, timeout)
            except relay.ZCodeRelayError as exc:
                raise BridgeError(exc.code, "relay failed", "check status") from exc

    service = _service(tmp_path, _free_port())
    controller = TimeoutController()
    service.controller = controller  # type: ignore[assignment]
    request = json.dumps(
        {
            "id": "generic-send-timeout",
            "op": "call",
            "generation": 4,
            "channel": "zcode-task",
            "method": "createTask",
            "arg": {},
            "timeoutMs": 100,
        },
        separators=(",", ":"),
    ).encode("ascii")
    connection = FakeConnection([request, request])

    with pytest.raises(OSError, match="fixture complete"):
        _serve_client(connection, service)  # type: ignore[arg-type]

    assert len(controller.calls) == 1
    assert json.loads(connection.sent[1]) == {
        "id": "generic-send-timeout",
        "ok": False,
        "error": {"code": "channel_request_outcome_unknown"},
    }
    assert connection.sent[1] == connection.sent[2]
    assert protocol._unacked == {}
    with pytest.raises(relay.ZCodeRelayError, match="channel_request_outcome_unknown"):
        protocol.rebind(
            relay.RelayFrameIdentity("bridge", 2, "recovery"),
            lambda _payload: None,
        )


def test_client_duplicate_request_cache_survives_local_reconnect(tmp_path: Path) -> None:
    service = _service(tmp_path, _free_port())
    request = json.dumps(
        {
            "id": "same-across-connections",
            "op": "call",
            "generation": 4,
            "channel": "zcode-task",
            "method": "getTaskSnapshot",
            "arg": {},
            "timeoutMs": 1000,
        },
        separators=(",", ":"),
    ).encode("ascii")
    first = FakeConnection([request])
    second = FakeConnection([request])

    with pytest.raises(OSError, match="fixture complete"):
        _serve_client(first, service)  # type: ignore[arg-type]
    with pytest.raises(OSError, match="fixture complete"):
        _serve_client(second, service)  # type: ignore[arg-type]

    assert len(service.controller.calls) == 1  # type: ignore[attr-defined]
    assert first.sent[1] == second.sent[1]


def test_client_request_cache_fails_closed_instead_of_evicting(tmp_path: Path) -> None:
    service = _service(tmp_path, _free_port())
    for index in range(bridge.MAX_LOCAL_CACHE):
        service.request_cache[f"old-{index}"] = bridge._CachedResponse(
            "digest", b'{"ok":true}'
        )
    request = json.dumps(
        {
            "id": "new-request",
            "op": "call",
            "generation": 4,
            "channel": "zcode-task",
            "method": "listTasks",
            "arg": {},
        },
        separators=(",", ":"),
    ).encode("ascii")
    connection = FakeConnection([request])

    with pytest.raises(OSError, match="fixture complete"):
        _serve_client(connection, service)  # type: ignore[arg-type]

    response = json.loads(connection.sent[1])
    assert response["error"]["code"] == "request_cache_exhausted"
    assert service.controller.calls == []  # type: ignore[attr-defined]
    assert len(service.request_cache) == bridge.MAX_LOCAL_CACHE
    assert "old-0" in service.request_cache


def test_oversized_subscription_event_fails_bridge_without_escaping(tmp_path: Path) -> None:
    service = _service(tmp_path, _free_port())
    request = {
        "id": "listen",
        "op": "listen",
        "generation": 4,
        "subscriptionId": "events",
        "channel": "zcode-agent",
        "event": "onDynamicSessionEvent",
        "arg": {},
    }

    _execute_client_request(
        request,
        service,
        {},
        lambda _value: (_ for _ in ()).throw(
            BridgeError("local_response_too_large", "large", "reconnect")
        ),
    )
    handler = service.controller.event_handler  # type: ignore[attr-defined]
    assert callable(handler)
    handler({"large": True})
    assert service.controller.local_delivery_failed  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b'{"id":"one","id":"two"}',
        b"[]",
    ],
)
def test_client_schema_drift_and_malformed_json_fail_closed(payload: bytes) -> None:
    with pytest.raises(BridgeError) as failure:
        _parse_client_message(payload)
    assert failure.value.code == "client_message_invalid"


def test_client_json_depth_is_bounded() -> None:
    nested: object = None
    for _ in range(70):
        nested = [nested]

    with pytest.raises(BridgeError) as failure:
        _parse_client_message(json.dumps({"arg": nested}).encode("ascii"))
    assert failure.value.code == "client_message_invalid"

    with pytest.raises(BridgeError) as failure:
        _parse_client_message(b'{"arg":NaN}')
    assert failure.value.code == "client_message_invalid"


def test_start_reports_crashed_child_and_cleans_launch_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, _free_port())

    class CrashedProcess:
        pid = 32001

        @staticmethod
        def poll() -> int:
            return 7

    monkeypatch.setattr(bridge, "load_bridge_config", lambda _root: config)
    monkeypatch.setattr(
        bridge,
        "_interprocess_file_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        bridge,
        "inspect_bridge",
        lambda _config: BridgeStatus("stopped", True),
    )
    monkeypatch.setattr(bridge, "_check_port_available", lambda _address, _port: None)
    monkeypatch.setattr(bridge, "_spawn_service", lambda _config, _launch_id: CrashedProcess())

    with pytest.raises(BridgeError) as failure:
        start_bridge(tmp_path, timeout_seconds=0.1)
    assert failure.value.code == "bridge_exited"
    assert not tuple(tmp_path.glob("launch-*.json"))


def test_stop_waits_for_exact_generation_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, _free_port())
    statuses = iter(
        (
            BridgeStatus("ready", True, process_alive=True, listener_owned=True),
            BridgeStatus("stopped", True),
        )
    )
    stop_writes: list[RuntimeState] = []
    monkeypatch.setattr(bridge, "load_bridge_config", lambda _root: service.config)
    monkeypatch.setattr(
        bridge,
        "_interprocess_file_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(bridge, "inspect_bridge", lambda _config: next(statuses))
    monkeypatch.setattr(bridge, "_read_state", lambda _config: service.state)
    monkeypatch.setattr(
        bridge,
        "_process_for_state",
        lambda _config, _state: SimpleNamespace(pid=10, creation_time_ns=20),
    )
    monkeypatch.setattr(bridge, "_write_stop", lambda _config, state: stop_writes.append(state))
    monkeypatch.setattr(bridge, "_inspect_windows_process", lambda _pid: None)
    monkeypatch.setattr(bridge, "_listeners", lambda _config: ())

    result = stop_bridge(tmp_path, timeout_seconds=0.1)

    assert result["code"] == "stopped"
    assert stop_writes == [service.state]


def test_stop_refuses_stale_state_without_process_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, _free_port())
    monkeypatch.setattr(bridge, "load_bridge_config", lambda _root: config)
    monkeypatch.setattr(
        bridge,
        "_interprocess_file_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        bridge,
        "inspect_bridge",
        lambda _config: BridgeStatus("stale_state", True),
    )

    with pytest.raises(BridgeError) as failure:
        stop_bridge(tmp_path, timeout_seconds=0.1)
    assert failure.value.code == "stale_state"


def test_lifecycle_idempotence_does_not_spawn_or_stop_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, _free_port())
    monkeypatch.setattr(bridge, "load_bridge_config", lambda _root: config)
    monkeypatch.setattr(
        bridge,
        "_interprocess_file_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        bridge,
        "inspect_bridge",
        lambda _config: BridgeStatus("ready", True),
    )
    monkeypatch.setattr(
        bridge,
        "_spawn_service",
        lambda *_args: pytest.fail("idempotent start spawned a child"),
    )
    assert start_bridge(tmp_path)["code"] == "already_started"

    monkeypatch.setattr(
        bridge,
        "inspect_bridge",
        lambda _config: BridgeStatus("stopped", True),
    )
    assert stop_bridge(tmp_path)["code"] == "already_stopped"


def test_late_client_release_cannot_overwrite_stopping_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, _free_port())
    publications: list[tuple[str, str]] = []
    monkeypatch.setattr(service, "publish", lambda phase, code: publications.append((phase, code)))
    service.client_connected = True
    service.mark_stopping()

    service.release_client()

    assert service.client_connected is False
    assert publications == []


def test_relay_controller_pins_selected_workspace_identity(tmp_path: Path) -> None:
    config = _config(tmp_path, _free_port())
    config = BridgeConfig(**{**config.__dict__, "workspace": tmp_path.resolve()})
    controller = bridge._BridgeRelayController(config)
    controller.workspace = {
        "workspacePath": str(tmp_path.resolve()),
        "workspaceIdentity": "selected-workspace",
    }

    scoped = controller._scope_argument(
        "zcode-agent",
        "listSessions",
        {"workspacePath": str(tmp_path.resolve()), "limit": 50},
    )

    assert scoped == {
        "workspacePath": str(tmp_path.resolve()),
        "workspaceIdentity": "selected-workspace",
        "limit": 50,
    }
    with pytest.raises(BridgeError) as failure:
        controller._scope_argument(
            "zcode-agent",
            "listSessions",
            {"workspacePath": str(tmp_path / "other")},
        )
    assert failure.value.code == "workspace_scope_mismatch"


def test_relay_controller_owns_v4_command_client_identity(tmp_path: Path) -> None:
    config = _config(tmp_path, _free_port())
    config = BridgeConfig(**{**config.__dict__, "workspace": tmp_path.resolve()})
    controller = bridge._BridgeRelayController(config)
    controller.workspace = {
        "workspacePath": str(tmp_path.resolve()),
        "workspaceIdentity": "selected-workspace",
    }
    controller._client_id = "bridge-owned-client"
    controller._known_session_ids.add("owned-session")

    scoped = controller._scope_argument(
        "zcode-agent",
        "sendConversationCommandV4",
        {
            "workspacePath": str(tmp_path.resolve()),
            "envelope": {
                "clientId": "untrusted-client",
                "type": "sendText",
                "sessionId": "owned-session",
                "payload": {},
            },
        },
    )

    assert scoped["envelope"]["clientId"] == "bridge-owned-client"
    assert scoped["workspaceIdentity"] == "selected-workspace"
    with pytest.raises(BridgeError) as failure:
        controller._scope_argument(
            "zcode-agent",
            "sendConversationCommandV4",
            {"workspacePath": str(tmp_path.resolve())},
        )
    assert failure.value.code == "conversation_envelope_invalid"


def test_relay_controller_rejects_unknown_operations_and_foreign_ids(tmp_path: Path) -> None:
    controller = bridge._BridgeRelayController(_config(tmp_path, _free_port()))
    controller.workspace = {"workspacePath": str(tmp_path.resolve())}

    with pytest.raises(BridgeError) as failure:
        controller._scope_argument("zcode-agent", "deleteSession", {})
    assert failure.value.code == "operation_not_allowed"

    with pytest.raises(BridgeError) as failure:
        controller._scope_argument("zcode-task", "listTasks", {"unexpected": True})
    assert failure.value.code == "operation_argument_invalid"

    with pytest.raises(BridgeError) as failure:
        controller._scope_argument("zcode-task", "closeTask", {"taskId": "foreign"})
    assert failure.value.code == "task_scope_mismatch"

    controller._record_authorized_result(
        "zcode-task", "listTasks", [{"taskId": "owned", "workspacePath": str(tmp_path)}]
    )
    assert controller._scope_argument("zcode-task", "closeTask", {"taskId": "owned"}) == {
        "taskId": "owned"
    }

    controller._record_authorized_result(
        "zcode-task",
        "listTasks",
        [{"taskId": "foreign-listed", "workspacePath": str(tmp_path / "other")}],
    )
    with pytest.raises(BridgeError) as failure:
        controller._scope_argument(
            "zcode-task", "closeTask", {"taskId": "foreign-listed"}
        )
    assert failure.value.code == "task_scope_mismatch"


def test_relay_controller_requires_ready_workspace_and_forces_replay_delivery(
    tmp_path: Path,
) -> None:
    controller = bridge._BridgeRelayController(_config(tmp_path, _free_port()))

    with pytest.raises(BridgeError) as failure:
        controller._scope_argument("zcode-task", "listTasks", {})
    assert failure.value.code == "workspace_unavailable"

    controller.workspace = {"workspacePath": str(tmp_path.resolve())}
    controller._known_session_ids.add("owned-session")
    scoped = controller._scope_argument(
        "zcode-agent",
        "onDynamicSessionEvent",
        {"sessionId": "owned-session", "deliveryKind": "desktop-continuous"},
    )
    assert scoped["deliveryKind"] == "web-remote-replayable"


def test_relay_controller_restricts_v4_command_types(tmp_path: Path) -> None:
    controller = bridge._BridgeRelayController(_config(tmp_path, _free_port()))
    controller.workspace = {"workspacePath": str(tmp_path.resolve())}
    controller._known_session_ids.add("owned-session")

    with pytest.raises(BridgeError) as failure:
        controller._scope_argument(
            "zcode-agent",
            "sendConversationCommandV4",
            {
                "envelope": {
                    "type": "deleteSession",
                    "sessionId": "owned-session",
                    "payload": {},
                }
            },
        )
    assert failure.value.code == "conversation_command_not_allowed"


def test_relay_controller_rejects_calls_while_reconnecting(tmp_path: Path) -> None:
    controller = bridge._BridgeRelayController(_config(tmp_path, _free_port()))
    controller.relay_generation = 1
    controller._reconnecting = True

    with pytest.raises(BridgeError) as failure:
        controller.call(1, "zcode-task", "listTasks", {}, 1.0)
    assert failure.value.code == "relay_reconnecting"


def test_relay_controller_fails_closed_after_rebind_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = bridge._BridgeRelayController(_config(tmp_path, _free_port()))

    class FailedTerminal:
        failure = bridge.ZCodeRelayError("websocket_closed")

        @staticmethod
        def wait_failed(_timeout: float) -> bool:
            return True

    controller.terminal = FailedTerminal()  # type: ignore[assignment]
    monkeypatch.setattr(bridge.time, "sleep", lambda _delay: None)

    def failed_connect(*, reconnect: bool) -> None:
        assert reconnect
        controller._rebind_committed = True
        raise bridge.ZCodeRelayError("channel_initialize_timeout")

    monkeypatch.setattr(controller, "_connect", failed_connect)

    controller.monitor(0.01)

    assert controller.fatal is not None
    assert controller.fatal.code == "relay_reconnect_commit_failed"


def test_prepare_rejects_plaintext_lan_mode(tmp_path: Path) -> None:
    with pytest.raises(BridgeError) as failure:
        prepare_bridge(
            tmp_path / "state",
            bind_address="192.168.1.10",
            port=55000,
            workspace=tmp_path,
            network_mode="lan",
            trust_lan=True,
            network_backend=_network(TAILSCALE_IP),
            zcode_cli=tmp_path / "unused",
        )
    assert failure.value.code == "network_mode_unsafe"


def test_start_revalidates_prepared_tailscale_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = BridgeConfig(
        **{
            **_config(tmp_path, _free_port()).__dict__,
            "bind_address": TAILSCALE_IP,
            "network_class": "tailscale",
        }
    )
    monkeypatch.setattr(bridge, "load_bridge_config", lambda _root: config)

    with pytest.raises(BridgeError) as failure:
        start_bridge(tmp_path, timeout_seconds=0.1, network_backend=_network("100.64.10.21"))
    assert failure.value.code == "bind_ip_not_current"


def test_config_validation_pins_imported_bridge_dependencies(tmp_path: Path) -> None:
    config = BridgeConfig(
        **{**_config(tmp_path, _free_port()).__dict__, "relay_module_sha256": "0" * 64}
    )

    with pytest.raises(BridgeError) as failure:
        bridge._validate_config_inputs(config)
    assert failure.value.code == "module_identity_drift"

    config = BridgeConfig(
        **{**_config(tmp_path, _free_port()).__dict__, "package_init_sha256": "0" * 64}
    )
    with pytest.raises(BridgeError) as failure:
        bridge._validate_config_inputs(config)
    assert failure.value.code == "module_identity_drift"


def test_relay_workspace_identity_pin_rejects_same_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, _free_port())
    monkeypatch.setattr(
        bridge,
        "_write_private_json",
        lambda path, value, replace: path.write_text(json.dumps(value), encoding="ascii"),
    )
    monkeypatch.setattr(bridge, "_read_private", lambda path, _maximum: path.read_bytes())

    bridge._pin_relay_workspace(config, "workspace-one", "identity-one")
    bridge._pin_relay_workspace(config, "workspace-one", "identity-one")
    with pytest.raises(BridgeError) as failure:
        bridge._pin_relay_workspace(config, "workspace-two", "identity-two")
    assert failure.value.code == "workspace_identity_drift"


def test_local_socket_send_deadline_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[int, int, bytes]] = []
    transport = SimpleNamespace(
        setsockopt=lambda level, option, value: captured.append((level, option, value))
    )
    monkeypatch.setattr(bridge.sys, "platform", "win32")

    bridge._set_socket_send_timeout(transport, 30.0)  # type: ignore[arg-type]

    assert captured
    assert captured[0][0:2] == (socket.SOL_SOCKET, socket.SO_SNDTIMEO)
    assert captured[0][2] == bridge.struct.pack("I", 30_000)
