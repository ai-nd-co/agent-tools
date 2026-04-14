from __future__ import annotations

import json
import socket
import subprocess
import sys
import time

from agent_tools.runtime import CONTROLLER_HOST, CONTROLLER_PORT

WINDOWS_DETACHED_FLAGS = 0x00000008 | 0x00000200


def send_controller_command(
    action: str,
    *,
    timeout_seconds: float = 0.75,
) -> bool:
    payload = json.dumps({"action": action}).encode("utf-8") + b"\n"
    try:
        with socket.create_connection(
            (CONTROLLER_HOST, CONTROLLER_PORT),
            timeout=timeout_seconds,
        ) as sock:
            sock.sendall(payload)
            sock.shutdown(socket.SHUT_WR)
            return True
    except OSError:
        return False


def ensure_controller_running(*, show_window: bool, detached: bool) -> bool:
    action = "show" if show_window else "refresh"
    if send_controller_command(action):
        return False
    launch_controller(hidden=not show_window, detached=detached)
    for _ in range(20):
        if send_controller_command(action):
            return True
        time.sleep(0.1)
    return True


def launch_controller(*, hidden: bool, detached: bool) -> None:
    args = [sys.executable, "-m", "agent_tools", "ui"]
    if hidden:
        args.append("--hidden")
    creationflags = WINDOWS_DETACHED_FLAGS if sys.platform == "win32" and detached else 0
    subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creationflags,
    )
