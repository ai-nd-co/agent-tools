from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from uuid import uuid4

from agent_tools.runtime import CONTROLLER_HOST, CONTROLLER_PORT

WINDOWS_DETACHED_FLAGS = 0x00000008 | 0x00000200


@dataclass
class ProcessingNotice:
    progress_id: str
    available: bool

    def update(
        self,
        *,
        stage: str,
        preview_text: str | None = None,
        detail_text: str | None = None,
    ) -> bool:
        if not self.available:
            return False
        payload: dict[str, object] = {
            "action": "processing-update",
            "progress_id": self.progress_id,
            "stage": stage,
        }
        if preview_text is not None:
            payload["preview_text"] = preview_text
        if detail_text is not None:
            payload["detail_text"] = detail_text
        self.available = send_controller_payload(payload)
        return self.available

    def finish(self) -> bool:
        if not self.available:
            return False
        self.available = send_controller_payload(
            {
                "action": "processing-finish",
                "progress_id": self.progress_id,
            }
        )
        return self.available


def send_controller_payload(
    payload: dict[str, object],
    *,
    timeout_seconds: float = 0.75,
) -> bool:
    data = json.dumps(payload).encode("utf-8") + b"\n"
    try:
        with socket.create_connection(
            (CONTROLLER_HOST, CONTROLLER_PORT),
            timeout=timeout_seconds,
        ) as sock:
            sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
            return True
    except OSError:
        return False

def send_controller_command(
    action: str,
    *,
    timeout_seconds: float = 0.75,
) -> bool:
    return send_controller_payload({"action": action}, timeout_seconds=timeout_seconds)



def start_processing_notice(
    *,
    source_label: str | None,
    preview_text: str,
    detail_text: str | None = None,
    stage: str = "Processing audio",
) -> ProcessingNotice:
    progress_id = str(uuid4())
    available = send_controller_payload(
        {
            "action": "processing-start",
            "progress_id": progress_id,
            "source_label": source_label,
            "preview_text": preview_text,
            "detail_text": detail_text or preview_text,
            "stage": stage,
        }
    )
    return ProcessingNotice(progress_id=progress_id, available=available)

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
