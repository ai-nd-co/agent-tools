"""Owner-login gate and bounded supervision for the existing ZCode bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

RETRY_DELAYS = (10.0, 30.0, 60.0)
STABLE_RESET = 300.0
STATE_FILE = "startup-wrapper.json"
STOP_FILE = "startup-wrapper-stop.json"


class StartupError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Observation:
    code: str
    generation: tuple[int, int, str] | None = None
    high_owner_session: bool = False
    last_code: str = ""


class BridgeHost(Protocol):
    def producer(self, *, start: bool) -> None: ...
    def observe(self) -> Observation: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def retire(self) -> None: ...


def supervise(
    host: BridgeHost,
    stopped: Callable[[], bool],
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Only crashes consume this budget; the bridge owns relay reconnect attempts."""
    if stopped():
        return "stopped"
    host.producer(start=True)
    failures = 0
    ready_since: float | None = None
    generation: tuple[int, int, str] | None = None
    next_producer_check = clock() + 10
    attach_deadline = clock() + 60
    while True:
        if stopped():
            host.stop()
            return "stopped"
        observation = host.observe()
        if observation.code == "stopped":
            if generation is not None:
                # A clean stop is intentional. Only a proved stale crash is restarted.
                return "stopped"
            try:
                host.start()
            except StartupError as exc:
                if exc.code != "bridge_exited":
                    raise
                observation = host.observe()
                if observation.code != "stale_state":
                    raise
            else:
                observation = host.observe()
        if observation.code == "stale_state":
            if observation.last_code not in {
                "ready", "starting", "relay_connecting", "reconnecting"
            }:
                raise StartupError("bridge_fatal_state")
            if generation is not None and observation.generation != generation:
                raise StartupError("bridge_generation_changed")
            if failures == len(RETRY_DELAYS):
                raise StartupError("bridge_retry_exhausted")
            deadline = clock() + RETRY_DELAYS[failures]
            failures += 1
            while clock() < deadline:
                if stopped():
                    return "stopped"
                sleep(min(0.25, deadline - clock()))
            if stopped():
                return "stopped"
            host.producer(start=False)
            host.retire()
            generation = None
            ready_since = None
            attach_deadline = clock() + 60
            continue
        if observation.code not in {"ready", "starting", "relay_connecting", "reconnecting"}:
            raise StartupError("bridge_" + observation.code)
        if not observation.high_owner_session:
            raise StartupError("bridge_not_high_owner_session")
        if generation is not None and observation.generation != generation:
            raise StartupError("bridge_generation_changed")
        generation = observation.generation
        if observation.code == "ready":
            attach_deadline = clock() + 60
            if ready_since is None:
                ready_since = clock()
            elif clock() - ready_since >= STABLE_RESET:
                failures = 0
        else:
            ready_since = None
            if clock() >= attach_deadline:
                raise StartupError("bridge_readiness_timeout")
        if clock() >= next_producer_check:
            host.producer(start=False)
            next_producer_check = clock() + 10
        sleep(0.25)


def render_task_xml(owner_sid: str, command: Path, arguments: list[str], working: Path) -> str:
    """Preparation only: activation requires a complete protected runtime package."""
    if not re.fullmatch(r"S-1-5-21-\d+-\d+-\d+-\d+", owner_sid):
        raise StartupError("owner_invalid")
    if not command.is_absolute() or not working.is_absolute():
        raise StartupError("task_path_invalid")
    if any("\x00" in value or "\n" in value or "\r" in value for value in arguments):
        raise StartupError("task_arguments_invalid")
    root = ET.Element("Task", version="1.4", xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task")
    action_text = subprocess.list2cmdline(arguments)
    fingerprint = hashlib.sha256(
        json.dumps([owner_sid, str(command), action_text, str(working)]).encode()
    ).hexdigest()

    def value(parent: ET.Element, tag: str, text: str) -> None:
        ET.SubElement(parent, tag).text = text

    registration = ET.SubElement(root, "RegistrationInfo")
    value(registration, "Description", "ai-nd-co-zcode-startup:" + fingerprint)
    trigger = ET.SubElement(ET.SubElement(root, "Triggers"), "LogonTrigger")
    value(trigger, "Enabled", "true")
    value(trigger, "UserId", owner_sid)
    principal = ET.SubElement(ET.SubElement(root, "Principals"), "Principal", id="Owner")
    value(principal, "UserId", owner_sid)
    value(principal, "LogonType", "InteractiveToken")
    value(principal, "RunLevel", "HighestAvailable")
    settings = ET.SubElement(root, "Settings")
    for tag, text in (
        ("MultipleInstancesPolicy", "IgnoreNew"),
        ("DisallowStartIfOnBatteries", "false"),
        ("StopIfGoingOnBatteries", "false"),
        ("AllowHardTerminate", "false"),
        ("StartWhenAvailable", "true"),
        ("Enabled", "false"),
        ("Hidden", "true"),
        ("ExecutionTimeLimit", "PT0S"),
    ):
        value(settings, tag, text)
    action = ET.SubElement(ET.SubElement(root, "Actions", Context="Owner"), "Exec")
    value(action, "Command", str(command))
    value(action, "Arguments", action_text)
    value(action, "WorkingDirectory", str(working))
    return ET.tostring(root, encoding="unicode")


def _high_owner_session(pid: int, owner_sid: str) -> bool:
    import win32api
    import win32con
    import win32security
    import win32ts

    try:
        with closing(win32api.OpenProcess(0x1000, False, pid)) as process:
            with closing(win32security.OpenProcessToken(process, win32con.TOKEN_QUERY)) as token:
                owner = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
                integrity = win32security.GetTokenInformation(
                    token, win32security.TokenIntegrityLevel
                )[0]
                rid = int(win32security.ConvertSidToStringSid(integrity).rsplit("-", 1)[1])
                session = win32ts.ProcessIdToSessionId(pid)
                return bool(
                    win32security.ConvertSidToStringSid(owner) == owner_sid
                    and win32security.GetTokenInformation(token, win32security.TokenElevation)
                    and rid == 12288
                    and session == win32ts.ProcessIdToSessionId(os.getpid())
                    and session != 0
                )
    except Exception:
        return False


class WindowsBridgeHost:
    def __init__(self, root: Path, executable: Path, sha256: str, owner_sid: str) -> None:
        from agent_tools import zcode_bridge

        self.bridge = zcode_bridge
        self.root = root
        self.executable = executable
        self.sha256 = sha256
        self.owner_sid = owner_sid
        if not _high_owner_session(os.getpid(), owner_sid):
            raise StartupError("startup_not_high_owner_interactive")

    def producer(self, *, start: bool) -> None:
        from agent_tools.tts_startup import _windows_system_directory

        helper = Path(__file__).with_name("zcode_producer_startup.ps1")
        result = subprocess.run(
            [str(_windows_system_directory() / "WindowsPowerShell/v1.0/powershell.exe"),
             "-NoProfile", "-NonInteractive", "-File", str(helper),
             "-Operation", "Start" if start else "Probe", "-Executable", str(self.executable),
             "-Sha256", self.sha256, "-OwnerSid", self.owner_sid],
            capture_output=True, timeout=75, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            code = json.loads(result.stdout).get("code")
        except (ValueError, AttributeError):
            raise StartupError("producer_observation_failed") from None
        if result.returncode or code != "producer_ready":
            safe = "producer_observation_failed"
            if isinstance(code, str) and re.fullmatch(r"[a-z_]{1,80}", code):
                safe = code
            raise StartupError(safe)

    def observe(self) -> Observation:
        config = self.bridge.load_bridge_config(self.root)
        status = self.bridge.inspect_bridge(config)
        state = self.bridge._read_state(config)
        generation = None if state is None else (
            state.process_pid, state.process_creation_time_ns, state.launch_id
        )
        return Observation(
            status.code, generation,
            state is not None and _high_owner_session(state.process_pid, self.owner_sid),
            "" if state is None else state.last_code,
        )

    def start(self) -> None:
        try:
            self.bridge.start_bridge(self.root, timeout_seconds=60)
        except self.bridge.BridgeError as exc:
            raise StartupError(exc.code) from None

    def stop(self) -> None:
        self.bridge.stop_bridge(self.root)

    def retire(self) -> None:
        self.bridge.retire_stale_bridge(self.root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "stop"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--producer", type=Path)
    parser.add_argument("--producer-sha256")
    parser.add_argument("--owner-sid")
    args = parser.parse_args(argv)
    try:
        from agent_tools.tts_startup import _inspect_windows_process
        from agent_tools.voice_onboarding import (
            _interprocess_file_lock,
            _read_private_bytes,
            _write_private_json,
        )
        from agent_tools.zcode_bridge import inspect_bridge, load_bridge_config

        config = load_bridge_config(args.root)
        state_path, stop_path = config.root / STATE_FILE, config.root / STOP_FILE
        if args.action in {"status", "stop"}:
            state = json.loads(_read_private_bytes(state_path, 4096))
            if (
                not isinstance(state, dict) or state.get("schema") != 1
                or state.get("config") != config.config_sha256
                or type(state.get("runner_pid")) is not int
                or type(state.get("runner_created")) is not int
                or not re.fullmatch(r"[a-f0-9]{32}", str(state.get("episode", "")))
            ):
                raise StartupError("wrapper_state_invalid")

            def runner_alive() -> bool:
                identity = _inspect_windows_process(state["runner_pid"])
                return identity is not None and identity.creation_time_ns == state["runner_created"]

            if args.action == "status":
                status = inspect_bridge(config)
                print(json.dumps({
                    "code": "running" if runner_alive() else "wrapper_stopped",
                    "bridge": status.code,
                }))
                return 0
            if not runner_alive():
                raise StartupError("wrapper_stale")
            _write_private_json(stop_path, state, replace=True)
            deadline = time.monotonic() + 90
            while runner_alive() and time.monotonic() < deadline:
                time.sleep(0.1)
            if runner_alive():
                raise StartupError("wrapper_stop_timeout")
            status = inspect_bridge(config)
            if status.code not in {"stopped", "stale_state"} or status.process_alive:
                raise StartupError("bridge_release_unproved")
            print(json.dumps({"code": "stopped", "bridge": status.code}))
            return 0
        if not args.producer or not args.producer_sha256 or not args.owner_sid:
            raise StartupError("startup_arguments_invalid")
        host = WindowsBridgeHost(config.root, args.producer, args.producer_sha256, args.owner_sid)
        with _interprocess_file_lock(
            config.root / "startup-wrapper.lock", busy_code="startup_busy",
            busy_message="ZCode startup is already running.", busy_next_action="Use status.",
            timeout=0.25,
        ):
            import secrets

            identity = _inspect_windows_process(os.getpid())
            if identity is None:
                raise StartupError("wrapper_identity_unproved")
            state = {
                "schema": 1, "episode": secrets.token_hex(16), "config": config.config_sha256,
                "runner_pid": identity.pid, "runner_created": identity.creation_time_ns,
            }
            _write_private_json(state_path, state, replace=True)

            def stopped() -> bool:
                return stop_path.exists() and json.loads(
                    _read_private_bytes(stop_path, 4096)
                ) == state

            code = supervise(host, stopped)
            print(json.dumps({"code": code}))
            return 0
    except Exception as exc:
        code = getattr(exc, "code", "startup_failed")
        if not isinstance(code, str) or not re.fullmatch(r"[a-z_]{1,80}", code):
            code = "startup_failed"
        print(json.dumps({"code": code}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
