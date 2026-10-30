"""Launcher: `python -m gs_speech.serve`, replacing `uvicorn --factory speaches.main:create_app`.

One uvicorn server listens on two sockets. Connections to UVICORN_PORT (default 8000) reach
Speaches exactly as before. Connections to GS_LID_PORT (default 8002; 0 turns it off) reach only
the language-ID app, so the two ports can be given different NetworkPolicy rules and neither app
is reachable through the other's port. One server also means one set of signal handlers: SIGTERM
drains both ports together.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import TYPE_CHECKING

import uvicorn

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("gs_speech")


class PortRouter:
    """Sends each connection to the app for the local port it arrived on."""

    def __init__(self, main: ASGIApp, lid: ASGIApp | None, lid_port: int) -> None:
        self.main = main
        self.lid = lid
        self.lid_port = lid_port

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.main(
                scope, receive, send
            )  # only Speaches has startup work (Admission passes it on)
            return
        server = scope.get("server")
        if self.lid is not None and server is not None and server[1] == self.lid_port:
            await self.lid(scope, receive, send)
        else:
            await self.main(scope, receive, send)


def bind(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.set_inheritable(True)
    return sock


def main() -> None:
    from speaches.main import create_app

    from gs_speech import ext, langid, punct, tts

    app = create_app()  # sets up logging; the wrappers below act on the classes, used per request
    state = ext.install()
    tts.install()  # before serving: Kokoro's session is built on its first load

    host = os.environ.get("UVICORN_HOST", "127.0.0.1")
    port = int(os.environ.get("UVICORN_PORT", "8000"))
    lid_port = int(os.environ.get("GS_LID_PORT", "8002"))
    lid_app = langid.make_app() if lid_port else None

    from speaches.dependencies import get_config

    key = get_config().api_key
    main_app = punct.wrap(
        ext.Admission(app, state.permits, key.get_secret_value() if key else None, **state.limits),
        os.environ,
    )
    # h11 refuses a request whose headers exceed 16 KiB, and X-GS-Context (punct.py) may be up to
    # 21,336 bytes of base64 on its own; 64 KiB leaves room for it and the usual headers.
    config = uvicorn.Config(
        PortRouter(main_app, lid_app, lid_port),
        host=host,
        port=port,
        lifespan="on",
        h11_max_incomplete_event_size=64 * 1024,
    )
    server = uvicorn.Server(config)
    sockets = [bind(host, port)]
    if lid_app is not None:
        sockets.append(bind(host, lid_port))
        logger.info("gs_speech: language ID on %s:%d", host, lid_port)
    logger.info("gs_speech: Speaches on %s:%d", host, port)
    server.run(sockets=sockets)


if __name__ == "__main__":
    main()
