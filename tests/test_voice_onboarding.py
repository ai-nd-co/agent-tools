from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import os
import socket
import ssl
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

import agent_tools.voice_onboarding as voice
from agent_tools import cli
from agent_tools.voice_onboarding import (
    AdapterAddress,
    AdapterSnapshot,
    NetworkSelection,
    OwnerCommand,
    PairingServer,
    TailscaleSnapshot,
    VoiceConfig,
    VoiceError,
    load_or_create_identity,
    select_network,
    setup_voice,
)

CODEX_TOKEN = "fixture-codex-token-0123456789abcdef"
TTS_TOKEN = "fixture-tts-token-0123456789abcdef"
TAILSCALE_IP = "100.64.10.20"


def _adapter(
    name: str,
    address: str,
    *,
    metric: int = 10,
    kind: str = "wifi",
    physical: bool = True,
    gateway: str | None = "192.168.1.1",
    preferred: bool = True,
) -> AdapterSnapshot:
    return AdapterSnapshot(
        interface_index=metric + 1,
        name=name,
        description=name,
        kind=kind,
        operational=True,
        physical=physical,
        gateway=gateway,
        route_metric=metric,
        addresses=(AdapterAddress(address, preferred),),
    )


@dataclass
class FakeNetwork:
    adapter_values: tuple[AdapterSnapshot, ...]
    tailscale_value: TailscaleSnapshot

    def adapters(self) -> tuple[AdapterSnapshot, ...]:
        return self.adapter_values

    def tailscale(self) -> TailscaleSnapshot:
        return self.tailscale_value


def _tailscale_network(*, state: str = "healthy") -> FakeNetwork:
    return FakeNetwork(
        (_adapter("Tailscale", TAILSCALE_IP, kind="tunnel", physical=False),),
        TailscaleSnapshot(state, (TAILSCALE_IP,) if state == "healthy" else ()),
    )


def test_network_prefers_one_cross_checked_tailscale_address() -> None:
    selected = select_network(_tailscale_network(), mode="auto")

    assert selected == NetworkSelection(TAILSCALE_IP, "tailscale", "Tailscale", 10)


def test_network_falls_back_to_lowest_metric_physical_lan() -> None:
    backend = FakeNetwork(
        (
            _adapter("Wi-Fi", "192.168.1.20", metric=20),
            _adapter("Ethernet", "10.1.2.3", metric=5, kind="ethernet"),
            _adapter("VMware Virtual", "192.168.50.4", metric=1, physical=False),
        ),
        TailscaleSnapshot("absent"),
    )

    selected = select_network(backend, mode="auto")

    assert selected.address == "10.1.2.3"
    assert selected.interface_name == "Ethernet"
    assert selected.confirmation_required is True


@pytest.mark.parametrize(
    ("kind", "hardware", "medium", "access", "expected"),
    [
        ("ethernet", True, 14, 2, True),
        ("wifi", True, 9, 2, True),
        ("ethernet", False, 14, 2, False),
        ("ethernet", True, 0, 2, False),
        ("ethernet", True, 14, 0, False),
        ("other", True, 14, 2, False),
    ],
)
def test_physical_adapter_requires_authoritative_windows_metadata(
    kind: str,
    hardware: bool,
    medium: int,
    access: int,
    expected: bool,
) -> None:
    assert (
        voice._physical_adapter_metadata_is_authoritative(
            kind=kind,
            has_mac=True,
            hardware_interface=hardware,
            physical_medium=medium,
            access_type=access,
        )
        is expected
    )


def test_generic_virtual_ethernet_metadata_fails_lan_selection_closed() -> None:
    metadata_proves_physical = voice._physical_adapter_metadata_is_authoritative(
        kind="ethernet",
        has_mac=True,
        hardware_interface=False,
        physical_medium=14,
        access_type=2,
    )
    backend = FakeNetwork(
        (
            _adapter(
                "Local Area Connection",
                "192.168.8.10",
                kind="ethernet",
                physical=metadata_proves_physical,
            ),
        ),
        TailscaleSnapshot("absent"),
    )

    with pytest.raises(VoiceError) as caught:
        select_network(backend, mode="lan")

    assert caught.value.code == "private_network_unavailable"


@pytest.mark.parametrize(
    ("backend", "code"),
    [
        (
            FakeNetwork(
                (_adapter("Tailscale", TAILSCALE_IP, kind="tunnel", physical=False),),
                TailscaleSnapshot("healthy", (TAILSCALE_IP, "100.64.10.21")),
            ),
            "tailscale_address_ambiguous",
        ),
        (_tailscale_network(state="unknown"), "tailscale_state_ambiguous"),
        (
            FakeNetwork(
                (
                    _adapter("Wi-Fi", "192.168.1.20", metric=10),
                    _adapter("Ethernet", "10.1.2.3", metric=10, kind="ethernet"),
                ),
                TailscaleSnapshot("absent"),
            ),
            "lan_route_ambiguous",
        ),
    ],
)
def test_network_ambiguity_fails_closed(backend: FakeNetwork, code: str) -> None:
    with pytest.raises(VoiceError) as caught:
        select_network(backend, mode="auto")

    assert caught.value.code == code


@pytest.mark.parametrize("address", ["0.0.0.0", "127.0.0.1", "8.8.8.8", "169.254.2.2"])
def test_explicit_bind_rejects_non_private_endpoints(address: str) -> None:
    with pytest.raises(VoiceError) as caught:
        select_network(_tailscale_network(), mode="auto", bind_ip=address)

    assert caught.value.code == "bind_ip_not_private"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _owner(root: Path, name: str) -> OwnerCommand:
    executable = root / f"{name}-owner.exe"
    config = root / f"{name}-startup.json"
    executable.write_bytes(f"{name}-owner".encode("ascii"))
    config.write_bytes(b"{}")
    return OwnerCommand(
        name=name,
        executable=str(executable.resolve()),
        executable_sha256=_sha(executable),
        prefix=(),
        entry_file=None,
        entry_sha256=None,
        config_file=str(config.resolve()),
        config_sha256=_sha(config),
    )


def _codex_status(
    *,
    installed: bool,
    healthy: bool,
    unknown: bool = False,
    owned: bool = True,
) -> dict[str, object]:
    return {
        "task": {
            "registered": installed,
            "compatible": not unknown,
            "definitionExact": not unknown,
            "owned": installed and not unknown and owned,
            "installationActive": installed and not unknown and owned,
        },
        "runtime": {
            "appServer": {
                "listening": unknown or healthy,
                "authenticatedReady": healthy,
            }
        },
    }


def _tts_status(*, installed: bool, healthy: bool, exact: bool = True) -> dict[str, object]:
    return {
        "code": "ready" if healthy else "not_ready",
        "status": {
            "code": "ready" if healthy else "not_ready",
            "task_present": installed,
            "task_exact": exact,
            "config_exact": exact,
            "credential_exact": exact,
            "runner_alive": healthy,
            "child_alive": healthy,
            "listener_owned": healthy,
            "authenticated_ready": healthy,
        },
    }


class FakeInvoker:
    def __init__(
        self,
        *,
        codex_unknown: bool = False,
        codex_unowned: bool = False,
        codex_never_ready: bool = False,
        tts_never_ready: bool = False,
        tts_drifted: bool = False,
    ) -> None:
        self.codex_installed = False
        self.tts_installed = False
        self.codex_unknown = codex_unknown
        self.codex_unowned = codex_unowned
        self.codex_never_ready = codex_never_ready
        self.tts_never_ready = tts_never_ready
        self.tts_drifted = tts_drifted
        self.calls: list[tuple[str, str]] = []

    def invoke(self, owner: OwnerCommand, action: str, *, plan: bool = False) -> dict[str, object]:
        assert plan is False
        self.calls.append((owner.name, action))
        if action == "install":
            changed = not (self.codex_installed if owner.name == "codex" else self.tts_installed)
            if owner.name == "codex":
                self.codex_installed = True
            else:
                self.tts_installed = True
            return {"changed": changed}
        if action == "uninstall":
            if owner.name == "codex":
                self.codex_installed = False
            else:
                self.tts_installed = False
            return {"changed": True}
        if action in {"restart", "start"}:
            return {"changed": True}
        if owner.name == "codex":
            healthy = self.codex_installed and not self.codex_never_ready
            return _codex_status(
                installed=self.codex_installed,
                healthy=healthy,
                unknown=self.codex_unknown,
                owned=not self.codex_unowned,
            )
        return _tts_status(
            installed=self.tts_installed,
            healthy=self.tts_installed and not self.tts_never_ready,
            exact=not self.tts_drifted,
        )


def _write_owner_spec(root: Path, *, include_tts: bool = True) -> tuple[OwnerCommand, OwnerCommand]:
    voice._ensure_private_directory(root)
    codex_owner = _owner(root, "codex")
    tts_owner = _owner(root, "tts")
    codex_token = root / "codex.token"
    tts_token = root / "tts.token"
    voice._write_private_bytes(codex_token, CODEX_TOKEN.encode("ascii"), replace=False)
    voice._write_private_bytes(tts_token, TTS_TOKEN.encode("ascii"), replace=False)
    payload: dict[str, object] = {
        "v": 1,
        "network": {"address": TAILSCALE_IP, "class": "tailscale"},
        "codex": {
            "owner": codex_owner.to_dict(),
            "endpoint": f"ws://{TAILSCALE_IP}:4222",
            "tokenFile": str(codex_token.resolve()),
        },
        "tts": None,
    }
    if include_tts:
        payload["tts"] = {
            "owner": tts_owner.to_dict(),
            "origin": f"http://{TAILSCALE_IP}:4223",
            "tokenFile": str(tts_token.resolve()),
        }
    voice._write_private_json(root / "owner-spec.json", payload, replace=False)
    return codex_owner, tts_owner


def _setup(root: Path, invoker: FakeInvoker, *, include_tts: bool = True) -> VoiceConfig:
    config, _ = setup_voice(
        root,
        invoker,
        _tailscale_network(),
        label="Studio PC",
        alias="studio",
        network_mode="auto",
        bind_ip=None,
        include_tts=include_tts,
        trust_lan=False,
        plan_only=False,
    )
    assert config is not None
    return config


def test_setup_composes_owners_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "voice"
    _write_owner_spec(root)
    invoker = FakeInvoker()

    first = _setup(root, invoker)
    first_installs = [call for call in invoker.calls if call[1] == "install"]
    second = _setup(root, invoker)

    assert first.server_id == second.server_id
    assert first.identity_fingerprint == second.identity_fingerprint
    assert first.codex_endpoint.endswith(":4222")
    assert first.tts_origin is not None and first.tts_origin.endswith(":4223")
    assert first_installs == [("codex", "install"), ("tts", "install")]
    assert [call for call in invoker.calls if call[1] == "install"] == first_installs


def test_existing_setup_no_tts_skips_tts_and_strips_invitation_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "voice"
    _write_owner_spec(root)
    invoker = FakeInvoker()
    _setup(root, invoker)
    invoker.calls.clear()

    invitation_config = _setup(root, invoker, include_tts=False)

    assert invitation_config.tts_owner is None
    assert invitation_config.tts_token_file is None
    assert invitation_config.tts_origin is None
    assert invoker.calls == [("codex", "status")]
    captured: list[VoiceConfig] = []

    def capture_pairing(
        config: VoiceConfig,
        _identity: object,
        **_kwargs: object,
    ) -> object:
        captured.append(config)
        return object()

    monkeypatch.setattr(voice, "PairingServer", capture_pairing)
    invoker.calls.clear()
    voice.create_pairing_server(root, invoker, expires_seconds=60, include_tts=False)

    assert captured[0].tts_owner is None
    assert captured[0].tts_token_file is None
    assert captured[0].tts_origin is None
    assert invoker.calls == [("codex", "status")]
    identity = load_or_create_identity(root / "identity.json", create=False)
    manifest_owner = object.__new__(PairingServer)
    manifest_owner.config = captured[0]
    manifest_owner.identity = identity
    manifest_owner.session_id = "fixture-session"
    manifest_owner.now = lambda: 1_000.0
    manifest_owner._codex_token = CODEX_TOKEN
    manifest_owner._tts_token = None
    state = voice._PairState(
        "fixture-session",
        "fixture-nonce",
        1_060,
        manifest_owner._build_manifest,
        identity,
        lambda: 1_000.0,
    )
    status, response = state.claim(
        {
            "v": 1,
            "sessionId": "fixture-session",
            "nonce": "fixture-nonce",
            "clientPublicKey": _client_key(7),
        }
    )

    assert status == 200
    assert "tts" not in response["manifest"]
    assert "tts.remote.v1" not in response["manifest"]["capabilities"]


def test_setup_refuses_exact_but_unowned_codex_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "voice"
    _write_owner_spec(root, include_tts=False)
    invoker = FakeInvoker(codex_unowned=True, codex_never_ready=True)
    invoker.codex_installed = True

    with pytest.raises(VoiceError) as caught:
        _setup(root, invoker, include_tts=False)

    assert caught.value.code == "codex_owner_unsafe"
    assert invoker.calls == [("codex", "status")]
    assert not (root / "voice.json").exists()
    assert not (root / "identity.json").exists()


def test_repair_refuses_exact_but_unowned_codex_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "voice"
    _write_owner_spec(root, include_tts=False)
    invoker = FakeInvoker()
    _setup(root, invoker, include_tts=False)
    invoker.codex_unowned = True
    invoker.codex_never_ready = True
    invoker.calls.clear()

    with pytest.raises(VoiceError) as caught:
        voice.repair_voice(root, invoker)

    assert caught.value.code == "codex_owner_unsafe"
    assert invoker.calls == [("codex", "status")]


def test_repair_refuses_registered_drifted_tts_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "voice"
    _write_owner_spec(root)
    invoker = FakeInvoker()
    _setup(root, invoker)
    invoker.tts_drifted = True
    invoker.tts_never_ready = True
    invoker.calls.clear()

    with pytest.raises(VoiceError) as caught:
        voice.repair_voice(root, invoker)

    assert caught.value.code == "tts_owner_unsafe"
    assert invoker.calls == [("codex", "status"), ("tts", "status")]


def test_setup_requires_explicit_trusted_lan_confirmation(tmp_path: Path) -> None:
    backend = FakeNetwork(
        (_adapter("Wi-Fi", "192.168.1.20"),),
        TailscaleSnapshot("absent"),
    )

    with pytest.raises(VoiceError) as caught:
        setup_voice(
            tmp_path / "voice",
            FakeInvoker(),
            backend,
            label=None,
            alias=None,
            network_mode="auto",
            bind_ip=None,
            include_tts=False,
            trust_lan=False,
            plan_only=False,
        )

    assert caught.value.code == "lan_confirmation_required"


def test_setup_refuses_unknown_codex_listener_and_rolls_back_identity(tmp_path: Path) -> None:
    root = tmp_path / "voice"
    _write_owner_spec(root, include_tts=False)

    with pytest.raises(VoiceError) as caught:
        _setup(root, FakeInvoker(codex_unknown=True), include_tts=False)

    assert caught.value.code == "codex_owner_unsafe"
    assert not (root / "identity.json").exists()
    assert not (root / "voice.json").exists()


def test_setup_rolls_back_only_new_codex_owner_when_readiness_fails(tmp_path: Path) -> None:
    root = tmp_path / "voice"
    _write_owner_spec(root, include_tts=False)
    invoker = FakeInvoker(codex_never_ready=True)

    with pytest.raises(VoiceError) as caught:
        _setup(root, invoker, include_tts=False)

    assert caught.value.code == "codex_not_ready"
    assert ("codex", "uninstall") in invoker.calls
    assert invoker.codex_installed is False
    assert not (root / "identity.json").exists()
    assert not (root / "setup-transaction.json").exists()


def test_failed_optional_tts_is_removed_without_disabling_codex(tmp_path: Path) -> None:
    root = tmp_path / "voice"
    _write_owner_spec(root)
    invoker = FakeInvoker(tts_never_ready=True)

    config = _setup(root, invoker)

    assert config.tts_owner is None
    assert config.tts_origin is None
    assert invoker.codex_installed is True
    assert invoker.tts_installed is False
    assert ("tts", "uninstall") in invoker.calls


def test_missing_owner_contract_does_not_create_identity(tmp_path: Path) -> None:
    root = tmp_path / "voice"

    with pytest.raises(VoiceError) as caught:
        _setup(root, FakeInvoker(), include_tts=False)

    assert caught.value.code == "codex_owner_contract_unavailable"
    assert not (root / "identity.json").exists()


def _pairing_config(root: Path) -> tuple[VoiceConfig, object]:
    voice._ensure_private_directory(root)
    codex_owner = _owner(root, "codex")
    token = root / "codex.token"
    voice._write_private_bytes(token, CODEX_TOKEN.encode("ascii"), replace=False)
    identity = load_or_create_identity(root / "identity.json")
    config = VoiceConfig(
        root=root,
        server_id=identity.server_id,
        label="Fixture PC",
        alias="fixture",
        identity_fingerprint=identity.fingerprint,
        network=NetworkSelection("127.0.0.1", "trusted_lan", "Fixture", 1, True),
        codex_endpoint="ws://127.0.0.1:54222",
        codex_token_file=token.resolve(),
        codex_owner=codex_owner,
        tts_origin=None,
        tts_token_file=None,
        tts_owner=None,
    )
    return config, identity


def _free_high_port() -> int:
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.bind(("127.0.0.1", 0))
            port = int(candidate.getsockname()[1])
        if voice.PAIRING_MIN_PORT <= port <= voice.PAIRING_MAX_PORT:
            return port
    raise AssertionError("no high fixture port")


def _bootstrap(uri: str) -> dict[str, object]:
    encoded = urllib.parse.parse_qs(urllib.parse.urlsplit(uri).query)["d"][0]
    encoded += "=" * (-len(encoded) % 4)
    return json.loads(base64.urlsafe_b64decode(encoded).decode("ascii"))


def _request(
    pairing: PairingServer,
    path: str,
    payload: Mapping[str, object],
) -> tuple[int, dict[str, object], bytes]:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    connection = http.client.HTTPSConnection(
        "127.0.0.1",
        pairing.invitation.port,
        context=context,
        timeout=3,
    )
    connection.connect()
    assert connection.sock is not None
    certificate = connection.sock.getpeercert(binary_form=True)
    connection.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    body = json.loads(response.read().decode("ascii"))
    connection.close()
    return response.status, body, certificate


def _raw_request(
    pairing: PairingServer,
    path: str,
    body: bytes,
    *,
    content_type: str = "application/json",
) -> tuple[int, dict[str, object]]:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    connection = http.client.HTTPSConnection(
        "127.0.0.1",
        pairing.invitation.port,
        context=context,
        timeout=3,
    )
    connection.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": content_type},
    )
    response = connection.getresponse()
    payload = json.loads(response.read().decode("ascii"))
    connection.close()
    return response.status, payload


def _client_key(byte: int) -> str:
    return base64.urlsafe_b64encode(bytes([byte]) * 32).decode("ascii").rstrip("=")


def test_pairing_is_pinned_signed_claimant_bound_idempotent_and_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    config, identity = _pairing_config(tmp_path / "voice")
    pairing = PairingServer(
        config,
        identity,
        expires_seconds=60,
        allow_fixture_loopback=True,
        random_port=lambda: _free_high_port(),
    )
    bootstrap = _bootstrap(pairing.invitation.uri)
    first_key = _client_key(1)
    second_key = _client_key(2)
    claim = {
        "v": 1,
        "sessionId": bootstrap["s"],
        "nonce": bootstrap["n"],
        "clientPublicKey": first_key,
    }
    pairing.start()
    try:
        status, first, certificate_der = _request(pairing, voice.PAIR_PATH, claim)
        retry_status, retry, _ = _request(pairing, voice.PAIR_PATH, claim)
        stolen_status, stolen, _ = _request(
            pairing,
            voice.PAIR_PATH,
            {**claim, "clientPublicKey": second_key},
        )

        certificate = x509.load_der_x509_certificate(certificate_der)
        spki = certificate.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        observed_pin = (
            base64.urlsafe_b64encode(hashlib.sha256(spki).digest()).decode("ascii").rstrip("=")
        )
        assert status == retry_status == 200
        assert first == retry
        assert stolen_status == 409
        assert stolen == {"error": {"code": "pairing_claimed"}}
        assert observed_pin == bootstrap["p"] == pairing.invitation.tls_spki_sha256
        assert certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value.get_values_for_type(x509.IPAddress) == [
            __import__("ipaddress").ip_address("127.0.0.1")
        ]

        manifest = first["manifest"]
        canonical = voice._canonical_json(manifest)
        identity.private_key.public_key().verify(
            voice._b64url_decode(first["signature"]),
            canonical,
        )
        assert first["manifestSha256"] == hashlib.sha256(canonical).hexdigest()
        assert manifest["clientPublicKeySha256"] == hashlib.sha256(bytes([1]) * 32).hexdigest()
        assert manifest["codex"]["capabilityToken"] == CODEX_TOKEN

        wrong_ack, _, _ = _request(
            pairing,
            voice.ACK_PATH,
            {
                "v": 1,
                "sessionId": bootstrap["s"],
                "nonce": bootstrap["n"],
                "clientPublicKey": second_key,
                "manifestSha256": first["manifestSha256"],
            },
        )
        assert wrong_ack == 403
        good_ack, _, _ = _request(
            pairing,
            voice.ACK_PATH,
            {
                "v": 1,
                "sessionId": bootstrap["s"],
                "nonce": bootstrap["n"],
                "clientPublicKey": first_key,
                "manifestSha256": first["manifestSha256"],
            },
        )
        assert good_ack == 200
        replay_status, replay = pairing._server.pair_state.claim(claim)
        assert replay_status == 410
        assert replay == {"error": {"code": "pairing_consumed"}}
    finally:
        pairing.close()

    assert not (config.root / "pairing" / "active.json").exists()
    assert not any((config.root / "pairing").iterdir())


def test_pairing_qr_excludes_credentials_paths_identity_secret_and_username(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    config, identity = _pairing_config(tmp_path / "voice")
    pairing = PairingServer(
        config,
        identity,
        expires_seconds=60,
        allow_fixture_loopback=True,
        random_port=lambda: _free_high_port(),
    )
    try:
        qr = pairing.invitation.uri
        identity_payload = (config.root / "identity.json").read_text(encoding="ascii")
        assert CODEX_TOKEN not in qr
        assert str(config.root) not in qr
        assert os.environ.get("USERNAME", "fixture-username-not-present") not in qr
        assert identity_payload not in qr
        bootstrap = _bootstrap(qr)
        assert set(bootstrap) == {"v", "u", "s", "n", "e", "p", "i", "f", "l", "a", "c"}
        assert bootstrap["i"] == config.server_id
        assert bootstrap["f"] == config.identity_fingerprint
        assert bootstrap["l"] == config.label
        assert bootstrap["a"] == config.alias
        assert bootstrap["c"] == config.network.network_class
    finally:
        pairing.close()


def test_pairing_rejects_malformed_oversized_and_unknown_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    config, identity = _pairing_config(tmp_path / "voice")
    pairing = PairingServer(
        config,
        identity,
        expires_seconds=60,
        allow_fixture_loopback=True,
        random_port=lambda: _free_high_port(),
    )
    bootstrap = _bootstrap(pairing.invitation.uri)
    pairing.start()
    try:
        malformed = _raw_request(pairing, voice.PAIR_PATH, b"{")
        oversized = _raw_request(
            pairing,
            voice.PAIR_PATH,
            b"x" * (voice.MAX_PAIR_REQUEST_BYTES + 1),
        )
        unknown = _request(
            pairing,
            voice.PAIR_PATH,
            {
                "v": 2,
                "sessionId": bootstrap["s"],
                "nonce": bootstrap["n"],
                "clientPublicKey": _client_key(5),
            },
        )[:2]
    finally:
        pairing.close()

    assert malformed == (400, {"error": {"code": "json_invalid"}})
    assert oversized == (413, {"error": {"code": "request_too_large"}})
    assert unknown == (400, {"error": {"code": "version_unsupported"}})


def test_pairing_rejects_noncanonical_client_key_without_claiming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    config, identity = _pairing_config(tmp_path / "voice")
    pairing = PairingServer(
        config,
        identity,
        expires_seconds=60,
        allow_fixture_loopback=True,
        random_port=lambda: _free_high_port(),
    )
    bootstrap = _bootstrap(pairing.invitation.uri)
    canonical = _client_key(1)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    equivalent = next(
        canonical[:-1] + character
        for character in alphabet
        if character != canonical[-1]
        and base64.urlsafe_b64decode(canonical[:-1] + character + "=")
        == base64.urlsafe_b64decode(canonical + "=")
    )
    pairing.start()
    try:
        invalid_status, invalid, _ = _request(
            pairing,
            voice.PAIR_PATH,
            {
                "v": 1,
                "sessionId": bootstrap["s"],
                "nonce": bootstrap["n"],
                "clientPublicKey": equivalent,
            },
        )
        valid_status, _, _ = _request(
            pairing,
            voice.PAIR_PATH,
            {
                "v": 1,
                "sessionId": bootstrap["s"],
                "nonce": bootstrap["n"],
                "clientPublicKey": canonical,
            },
        )
    finally:
        pairing.close()

    assert invalid_status == 400
    assert invalid == {"error": {"code": "client_key_invalid"}}
    assert valid_status == 200


def test_second_live_pairing_session_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    config, identity = _pairing_config(tmp_path / "voice")
    first = PairingServer(
        config,
        identity,
        expires_seconds=60,
        allow_fixture_loopback=True,
        random_port=lambda: _free_high_port(),
    )
    try:
        with pytest.raises(VoiceError) as caught:
            PairingServer(
                config,
                identity,
                expires_seconds=60,
                allow_fixture_loopback=True,
                random_port=lambda: _free_high_port(),
            )
        assert caught.value.code == "pairing_already_active"
    finally:
        first.close()


def test_expired_server_close_cannot_remove_successor_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    now = [1_000.0]
    config, identity = _pairing_config(tmp_path / "voice")
    expired = PairingServer(
        config,
        identity,
        expires_seconds=15,
        allow_fixture_loopback=True,
        now=lambda: now[0],
        random_port=lambda: _free_high_port(),
    )
    now[0] = 1_016.0
    successor = PairingServer(
        config,
        identity,
        expires_seconds=60,
        allow_fixture_loopback=True,
        now=lambda: now[0],
        random_port=lambda: _free_high_port(),
    )
    try:
        expired.close()
        with pytest.raises(VoiceError) as caught:
            PairingServer(
                config,
                identity,
                expires_seconds=60,
                allow_fixture_loopback=True,
                now=lambda: now[0],
                random_port=lambda: _free_high_port(),
            )
        assert caught.value.code == "pairing_already_active"
    finally:
        successor.close()


def test_pairing_expiry_replay_and_stale_artifacts_clean_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    now = [1_000.0]
    config, identity = _pairing_config(tmp_path / "voice")
    pairing = PairingServer(
        config,
        identity,
        expires_seconds=15,
        allow_fixture_loopback=True,
        now=lambda: now[0],
        random_port=lambda: _free_high_port(),
    )
    bootstrap = _bootstrap(pairing.invitation.uri)
    claim = {
        "v": 1,
        "sessionId": bootstrap["s"],
        "nonce": bootstrap["n"],
        "clientPublicKey": _client_key(3),
    }
    now[0] = 1_016.0
    assert pairing._server.pair_state.claim(claim) == (
        410,
        {"error": {"code": "pairing_expired"}},
    )
    pairing.close()

    stale_root = config.root / "pairing" / ("a" * 24)
    voice._ensure_private_directory(stale_root)
    voice._write_private_bytes(stale_root / "secret", b"expired", replace=False)
    voice._write_private_json(
        config.root / "pairing" / "active.json",
        {
            "v": 1,
            "expiresAt": 1,
            "pid": os.getpid(),
            "processGeneration": "old-generation",
            "sessionHash": "a" * 64,
        },
        replace=False,
    )
    replacement = PairingServer(
        config,
        identity,
        expires_seconds=15,
        allow_fixture_loopback=True,
        random_port=lambda: _free_high_port(),
    )
    replacement.close()
    assert not stale_root.exists()


def test_pairing_bind_failure_removes_ephemeral_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    config, identity = _pairing_config(tmp_path / "voice")

    class RefusingServer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("fixture bind disappeared")

    monkeypatch.setattr(voice, "_ExactHttpsServer", RefusingServer)
    with pytest.raises(VoiceError) as caught:
        PairingServer(
            config,
            identity,
            expires_seconds=15,
            allow_fixture_loopback=True,
            random_port=lambda: 55_555,
        )

    assert caught.value.code == "pairing_port_unavailable"
    assert not any((config.root / "pairing").iterdir())


def test_status_load_does_not_recreate_missing_identity(tmp_path: Path) -> None:
    root = tmp_path / "voice"
    _write_owner_spec(root, include_tts=False)
    invoker = FakeInvoker()
    _setup(root, invoker, include_tts=False)
    (root / "identity.json").unlink()

    with pytest.raises(VoiceError) as caught:
        voice.voice_status(root, invoker)

    assert caught.value.code == "identity_missing"
    assert not (root / "identity.json").exists()


def test_top_level_cli_delegates_all_voice_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[list[str]] = []
    monkeypatch.setattr(cli, "run_voice", lambda values: observed.append(values) or 7)

    result = cli._main(["voice", "pair", "--expires-seconds", "45", "--json"])

    assert result == 7
    assert observed == [["pair", "--expires-seconds", "45", "--json"]]


def test_cli_errors_are_actionable_and_json_is_machine_readable(tmp_path: Path) -> None:
    human_error = io.StringIO()
    json_error = io.StringIO()

    human_result = voice.main(
        ["pair"],
        root=tmp_path / "missing",
        stderr=human_error,
    )
    json_result = voice.main(
        ["pair", "--json"],
        root=tmp_path / "missing",
        stderr=json_error,
    )

    assert human_result == json_result == 2
    assert human_error.getvalue().startswith("Voice: ")
    assert "\nImpact: " in human_error.getvalue()
    assert "\nNext: " in human_error.getvalue()
    assert set(json.loads(json_error.getvalue())["error"]) == {
        "code",
        "cause",
        "impact",
        "nextAction",
    }


def test_owner_command_output_is_stopped_at_the_memory_limit() -> None:
    script = (
        "import sys,time;"
        f"sys.stdout.buffer.write(b'x'*{voice.MAX_OWNER_OUTPUT_BYTES + 8192});"
        "sys.stdout.buffer.flush();time.sleep(30)"
    )
    started = time.monotonic()

    with pytest.raises(VoiceError) as caught:
        voice._bounded_subprocess([sys.executable, "-c", script], timeout=20.0)

    assert caught.value.code == "child_output_too_large"
    assert time.monotonic() - started < 5.0


def test_owner_command_descendant_cannot_hold_capture_pipes_open(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "descendant.pid"
    descendant = "import time;time.sleep(30)"
    owner = "\n".join(
        [
            "import subprocess,sys",
            "from pathlib import Path",
            "child = subprocess.Popen(",
            f"    [sys.executable, '-c', {descendant!r}],",
            "    stdin=subprocess.DEVNULL,",
            "    stdout=sys.stdout,",
            "    stderr=sys.stderr,",
            "    close_fds=False,",
            ")",
            "Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')",
        ]
    )
    started = time.monotonic()

    with pytest.raises(VoiceError) as caught:
        voice._bounded_subprocess([sys.executable, "-c", owner, str(child_pid_file)], timeout=20.0)

    assert caught.value.code == "child_process_failed"
    assert time.monotonic() - started < 5.0
    child_pid = int(child_pid_file.read_text(encoding="ascii"))
    deadline = time.monotonic() + 2.0
    while voice._process_generation(child_pid) is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert voice._process_generation(child_pid) is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job assignment ordering")
def test_windows_owner_cannot_execute_before_job_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "owner-executed"
    original_assign = voice._windows_kill_job
    assignment_observed = False

    def delayed_assignment(process: subprocess.Popen[bytes]) -> object:
        nonlocal assignment_observed
        time.sleep(0.2)
        assert not marker.exists()
        assignment_observed = True
        return original_assign(process)

    monkeypatch.setattr(voice, "_windows_kill_job", delayed_assignment)
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path;import sys;Path(sys.argv[1]).write_bytes(b'ok')",
        str(marker),
    ]

    completed = voice._bounded_subprocess(command, timeout=5.0)

    assert completed.returncode == 0
    assert assignment_observed is True
    assert marker.read_bytes() == b"ok"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job assignment ordering")
def test_windows_assignment_failure_never_executes_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "owner-must-not-execute"

    def refuse_assignment(_process: subprocess.Popen[bytes]) -> object:
        raise VoiceError(
            "child_process_failed",
            "fixture assignment failure",
            "fixture",
        )

    monkeypatch.setattr(voice, "_windows_kill_job", refuse_assignment)
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path;import sys;Path(sys.argv[1]).write_bytes(b'unsafe')",
        str(marker),
    ]

    with pytest.raises(VoiceError) as caught:
        voice._bounded_subprocess(command, timeout=5.0)

    assert caught.value.code == "child_process_failed"
    assert not marker.exists()


def test_active_setup_lock_refuses_a_second_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    root = tmp_path / "voice"
    _write_owner_spec(root, include_tts=False)
    voice._write_private_json(
        root / "setup.lock",
        {
            "v": 1,
            "pid": os.getpid(),
            "processGeneration": "fixture-generation",
            "lockId": voice._b64url(bytes(range(32))),
        },
        replace=False,
    )

    with pytest.raises(VoiceError) as caught:
        _setup(root, FakeInvoker(), include_tts=False)

    assert caught.value.code == "setup_already_running"
    voice._remove_owned_file(root / "setup.lock")


def test_setup_lock_is_exclusive_across_real_processes(tmp_path: Path) -> None:
    root = tmp_path / "voice"
    script = "\n".join(
        [
            "import sys",
            "from pathlib import Path",
            "from agent_tools.voice_onboarding import VoiceError, _voice_setup_lock",
            "try:",
            "    with _voice_setup_lock(Path(sys.argv[1])):",
            "        print('acquired')",
            "except VoiceError as exc:",
            "    print(exc.code)",
        ]
    )

    with voice._voice_setup_lock(root):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(root)],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "setup_already_running"
    assert completed.stderr == ""


@pytest.mark.parametrize("guard_name", [".setup.guard", ".pairing.guard"])
def test_guard_publication_recovers_when_interrupted_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guard_name: str,
) -> None:
    guard = tmp_path / "voice" / guard_name
    original_write = voice._write_private_bytes
    interrupted = False

    def interrupt_first_publish(path: Path, data: bytes, *, replace: bool) -> None:
        nonlocal interrupted
        if path == guard and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        original_write(path, data, replace=replace)

    monkeypatch.setattr(voice, "_write_private_bytes", interrupt_first_publish)
    with pytest.raises(KeyboardInterrupt):
        with voice._interprocess_file_lock(
            guard,
            busy_code="fixture_busy",
            busy_message="fixture",
            busy_next_action="fixture",
            timeout=0.0,
        ):
            pass
    assert not guard.exists()

    monkeypatch.setattr(voice, "_write_private_bytes", original_write)
    with voice._interprocess_file_lock(
        guard,
        busy_code="fixture_busy",
        busy_message="fixture",
        busy_next_action="fixture",
        timeout=0.0,
    ):
        assert guard.exists()
    assert guard.read_bytes() == b"\0"


def test_interrupted_owner_install_recovers_from_durable_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    root = tmp_path / "voice"
    _write_owner_spec(root, include_tts=False)

    class InterruptingInvoker(FakeInvoker):
        interrupt = True

        def invoke(
            self, owner: OwnerCommand, action: str, *, plan: bool = False
        ) -> dict[str, object]:
            result = super().invoke(owner, action, plan=plan)
            if owner.name == "codex" and action == "install" and self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt
            return result

    invoker = InterruptingInvoker()
    with pytest.raises(KeyboardInterrupt):
        _setup(root, invoker, include_tts=False)

    assert invoker.codex_installed is True
    assert (root / "setup-transaction.json").exists()
    assert (root / "identity.json").exists()

    recovered = _setup(root, invoker, include_tts=False)

    assert recovered.codex_endpoint.endswith(":4222")
    assert invoker.calls.count(("codex", "uninstall")) == 1
    assert invoker.calls.count(("codex", "install")) == 2
    assert not (root / "setup-transaction.json").exists()


def test_completed_config_rolls_forward_after_interrupted_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice, "_process_generation", lambda _pid: "fixture-generation")
    root = tmp_path / "voice"
    _write_owner_spec(root, include_tts=False)
    invoker = FakeInvoker()
    original_journal = voice._setup_journal

    def interrupt_commit(*args: object, **kwargs: object) -> None:
        if args[1] == "committed":
            raise KeyboardInterrupt
        original_journal(*args, **kwargs)

    monkeypatch.setattr(voice, "_setup_journal", interrupt_commit)
    with pytest.raises(KeyboardInterrupt):
        _setup(root, invoker, include_tts=False)

    assert (root / "voice.json").exists()
    assert (root / "setup-transaction.json").exists()
    install_count = invoker.calls.count(("codex", "install"))

    recovered = _setup(root, invoker, include_tts=False)

    assert recovered.server_id.startswith("srv_")
    assert invoker.calls.count(("codex", "install")) == install_count
    assert not (root / "setup-transaction.json").exists()
