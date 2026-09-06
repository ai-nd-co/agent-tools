"""Loopback-only Android fixture using the production authenticated bridge/controller.

The hermetic boundary is the desktop RPC peer; no ZCode process or login is opened.
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from test_zcode_bridge import _service

from agent_tools import zcode_bridge as bridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    service = _service(args.state.resolve(), args.port)
    service.publish = lambda *_: None
    bridge._bearer_value = lambda _: "task341-zcode-capability-token-00000000"
    original_http_error = bridge._http_error

    def http_error(transport, status, code):
        print(f"HTTP refusal {status} {code}", flush=True)
        original_http_error(transport, status, code)

    bridge._http_error = http_error
    controller = bridge._BridgeRelayController(service.config)
    controller.relay_generation = 4
    controller.workspace = {"workspacePath": str(service.config.workspace)}
    service.controller = controller
    counts: Counter[str] = Counter()
    listeners: dict[str, object] = {}
    executions: dict[str, tuple[str, str]] = {}
    sequence = 0
    tasks = {
        "ztask-fixture": {
            "taskId": "ztask-fixture", "title": "ZCode fixture conversation",
            "workspacePath": str(service.config.workspace), "status": "idle",
            "updatedAt": 1788480000,
        }
    }

    def event(
        kind: str, payload: dict, command_id: str | None = None,
        session_id: str = "ztask-fixture", turn_id: str = "zturn-fixture",
    ) -> None:
        nonlocal sequence
        sequence += 1
        value = {
            "type": "session.event",
            "event": {
                "eventId": f"event-{sequence}", "seq": sequence,
                "sessionId": session_id, "turnId": turn_id,
                "type": kind, "payload": payload,
                **({"sourceCommandId": command_id} if command_id else {}),
            },
        }
        handler = listeners.get("onDynamicSessionEvent")
        if handler:
            handler(value)

    def lifecycle(trace_id: str, session_id: str) -> None:
        if session_id != "ztask-fixture":
            turn_id, execution_id = "question-turn", "question-execution"
            executions[session_id] = turn_id, execution_id
            event("turn.started", {
                "inputId": "question-input", "foregroundExecutionId": execution_id,
            }, trace_id, session_id, turn_id)
            threading.Timer(0.2, lambda: event("userInput.requested", {
                "requestId": "question-fixture", "prompt": "Which file should I use?",
            }, None, session_id, turn_id)).start()
            return
        event("turn.started", {
            "inputId": "native-input", "messageId": "user-message",
            "foregroundExecutionId": "foreground-execution",
        }, trace_id)
        for message_id, visibility, text in (
            ("visible-message", "visible", "The ZCode fixture is working."),
            ("hidden-message", "hidden", "hidden reasoning must never be spoken"),
        ):
            event("message.upserted", {"message": {
                "info": {"messageId": message_id, "role": "assistant", "semantics": {
                    "kind": "assistant_response", "uiVisibility": visibility,
                    "transcriptVisibility": visibility, "providerVisibility": visibility,
                }}, "parts": [{"type": "text", "text": ""}],
            }})
            event("part.delta", {"messageId": message_id, "field": "text", "delta": text})
        threading.Timer(0.5, lambda: event("turn.completed", {
            "response": "The ZCode fixture journey completed safely.", "resultType": "success",
        })).start()

    class RpcPeer:
        def call(self, channel, method, argument, timeout):
            counts[f"{channel}/{method}"] += 1
            (args.state / "ledger.json").write_text(json.dumps(dict(counts)), encoding="utf-8")
            if method == "listTasks":
                return list(tasks.values())
            if method == "getTaskSnapshot":
                return {"meta": tasks[argument["taskId"]], "messages": [], "runtime": {}}
            if method == "createTask":
                assert argument["mode"] == "yolo"
                task_id = f"created-{len(tasks)}"
                tasks[task_id] = {**tasks["ztask-fixture"], "taskId": task_id, "title": "New task"}
                return tasks[task_id]
            if method == "renameTask":
                tasks[argument["taskId"]]["title"] = argument["title"]
                return tasks[argument["taskId"]]
            if method == "sendPrompt":
                threading.Timer(
                    0.3, lifecycle, args=(argument["traceId"], argument["taskId"])
                ).start()
                return None
            if method == "respondElicitation":
                assert argument["requestId"] == "question-fixture"
                assert argument["action"] == "decline"
                return True
            if method == "sendConversationCommandV4":
                envelope = argument["envelope"]
                assert envelope["type"] == "stop"
                session_id = envelope["sessionId"]
                turn_id, execution_id = executions[session_id]
                assert envelope["payload"] == {"expectedForegroundExecutionId": execution_id}
                threading.Timer(2.0, lambda: event("turn.completed", {
                    "response": "Question declined.", "resultType": "cancelled",
                }, None, session_id, turn_id)).start()
                return {"commandId": envelope["commandId"], "status": "accepted"}
            raise AssertionError(f"Unexpected hermetic RPC: {channel}/{method}")

        def listen(self, channel, event_name, argument, handler):
            listeners[event_name] = handler

            def dispose():
                if listeners.get(event_name) is handler:
                    listeners.pop(event_name, None)

            return SimpleNamespace(dispose=dispose)

    controller.rpc = RpcPeer()

    class Control(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.reply({
                "zcodeMethodCounts": dict(counts), "workspace": str(service.config.workspace)
            })

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            service.config = replace(service.config, server_id=body["serverId"])
            self.reply({"ok": True})

        def reply(self, value):
            content = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    control = ThreadingHTTPServer(("127.0.0.1", args.control_port), Control)
    threading.Thread(target=control.serve_forever, daemon=True).start()
    server = bridge._ExactBridgeServer(service)
    try:
        server.serve_forever()
    finally:
        control.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
