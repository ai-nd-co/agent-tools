Resume the Task 298 response-delivery audit after the two MEDIUM findings from your first pass.

Both findings were addressed narrowly in the same worktree:

1. The peer-close delivery handshake is now WAV-only. `_send_json` no longer calls it, so unauthenticated/error requests do not incur the one-second wait. `test_unauthenticated_json_response_does_not_wait_for_peer_close` protects that boundary using a real threaded server and a client intentionally kept open.
2. The response tests now include deterministic transport fault injection. `test_wav_delivery_acknowledges_queued_tail_before_framework_close` models the observed nine-byte queued tail: framework closure discards it unless the WAV-specific shutdown/read acknowledgement runs first. The test asserts exact bytes and SHA after simulated framework close. The raw-socket receive loops also fail explicitly on EOF.

Current verification is clean:

- `uv run pytest -q tests/test_tts_server.py` -> 42 passed
- `uv run ruff check src/agent_tools/tts_server.py tests/test_tts_server.py` -> clean
- `uv run mypy src/agent_tools/tts_server.py tests/test_tts_server.py` -> clean

Please inspect the current files directly, verify that both original findings are resolved, and audit the narrow delivery repair for new correctness, security, lifecycle, or regression problems. Report findings first with severity and file/line references. If no actionable findings remain, say CLEAN explicitly. Do not edit files.
