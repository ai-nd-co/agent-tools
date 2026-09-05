Resume the Task 298 response-delivery audit again. The prior half-close repair was falsified by the exact Android acceptance path after your CLEAN verdict, so this is a materially revised lifecycle fix.

Observed evidence against the half-close generation on `100.64.0.4:4223`:

- Exact shipped Android OkHttp transport on the API 35 emulator over Tailscale failed 10/10 English requests.
- Every declared 68,444-byte body ended 6-14 bytes short. OkHttp raised `ProtocolException` during response-body read.
- A Russian example declared 71,444 bytes and received 71,438.
- Client validation correctly rejected all truncated bodies; no retry or byte reconstruction exists.

Root lifecycle flaw in the previous repair:

- `_finish_wav_delivery` called `shutdown(SHUT_WR)` immediately after the write, so FIN was still initiated before Android received the tail.
- Its subsequent `recv` wait could consume a pipelined next HTTP request rather than acknowledge body delivery.

Current narrow fix:

- Successful WAV responses retain exact `Content-Length` and SHA but no longer emit `Connection: close`, no longer call `shutdown`, and no longer force `close_connection = True` after a successful write.
- `wfile.flush()` remains explicit. Normal HTTP/1.1 persistence lets the client finish the Content-Length-framed body without an immediate FIN boundary.
- A request that already requires closure (HTTP/1.0 or client `Connection: close`) is still honored by the base handler. Write/flush errors still force close.
- JSON/auth/error responses retain `Connection: close` and immediate teardown.
- Idle successful connections remain bounded by the existing per-socket 10-second server timeout.

Regression changes:

- A raw real-socket test reads one exact WAV, proves framework teardown has not started, sends a second authenticated request on the same socket, reads its exact WAV/hash/request ID, then closes and proves teardown.
- A separate test lowers the server socket timeout and proves an idle successful persistent connection is torn down boundedly.
- The unauthenticated JSON immediate-teardown regression remains.
- The prior synthetic tail-ack and half-close tests were removed because Android falsified their model.

Verification:

- Pre-fix control: the new HTTP/1.1 reuse regression fails against the half-close implementation because the first response declares `Connection: close`.
- `uv run pytest -q tests/test_tts_server.py` -> 41 passed.
- Ruff and strict mypy for the server/test files -> clean.
- Exact shipped Android OkHttp A/B through `adb reverse` to isolated loopback port 4224: 10/10 English bodies at 68,444 bytes and 5/5 Russian bodies at 71,444 bytes, all metadata/hash-valid, zero truncation/NETWORK events. The harness force-stopped between runs and created fresh transports, so this proves 15 fresh connections rather than relying on socket reuse.

Please inspect the current files directly. Audit protocol correctness, connection lifecycle/resource bounds, request smuggling/pipelining implications, auth/error behavior, shutdown behavior, and whether the regressions meaningfully protect the Android fix. Findings first with severity and file/line references. If no actionable findings remain, say CLEAN explicitly. Do not edit files or perform live device/server actions.
