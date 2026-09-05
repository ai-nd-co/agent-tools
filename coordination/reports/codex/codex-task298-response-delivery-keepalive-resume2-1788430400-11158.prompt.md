The LOW regression-coverage finding is fixed.

- The real raw-socket HTTP/1.1 test now sends both authenticated POST requests in one `sendall`, then reads and validates both exact Content-Length/SHA responses, proving pipelined input is processed rather than consumed by response-finish logic.
- A parameterized raw-socket regression now covers both HTTP/1.0 default close and HTTP/1.1 `Connection: close`, validates the exact WAV/hash, and requires EOF immediately after the declared response body.
- Full focused suite is now 43 passed; Ruff and strict mypy remain clean.

Please inspect the current tests and return the final findings-first verdict. If the finding is resolved and no actionable issue remains, say CLEAN explicitly. Do not edit files or perform live actions.
