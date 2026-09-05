# agent-tools

Python CLI tools for:

- transforming raw text into TTS-ready narration and synthesizing it in one command
- transforming piped text through either the private Codex backend used by local Codex or an experimental Claude Code CLI wrapper
- synthesizing English with Kokoro-82M and Russian with Silero `v5_5_ru`
- auto-TTS of completed Codex and Claude Code replies through installed desktop hooks

This repo is intentionally wired to local Codex and Claude Code installs.

Release policy:

- semantic-release owns version bumps, changelog updates, and `py-v*` tags
- do not manually edit `project.version` in `pyproject.toml` during normal work
- do not create release tags by hand unless the release workflow explicitly calls for it

## Status

This is an **experimental public package** with a **private Codex dependency** and an
**experimental Claude Code CLI transform path**.

The `transform` command mirrors the current request shape used by the local Codex source tree and depends on ChatGPT-backed auth in `~/.codex/auth.json`.

The Claude Code path uses the official `claude` CLI in headless `-p` mode with a constrained
single-turn wrapper. It does not rely on any private Claude Code API.

It does **not** use:

- `codex exec`
- `codex app-server`
- the public OpenAI API key flow

That means:

- you must already be logged into local Codex
- backend compatibility can break if Codex internals or backend contracts change
- this package is best suited for users who already use local Codex

## Requirements

- Python 3.11+
- local Codex already logged in via ChatGPT
- local Claude Code CLI installed if you want Claude-backed transforms or Claude auto-TTS integration
- `espeak-ng` installed for best Kokoro English fallback behavior

## Install

```bash
cd repos/agent-tools
uv venv
uv pip install -e ".[dev]"
```

Public package install:

```bash
pip install ai-nd-co-agent-tools
```

UI-enabled install:

```bash
pip install "ai-nd-co-agent-tools[ui]"
```

Install a CUDA-enabled PyTorch stack for this CLI environment:

```bash
agent-tools install-cuda
```

Pass an explicit track if you do not want auto-detection:

```bash
agent-tools install-cuda --cuda-track cu130
```

For a service that explicitly runs Kokoro on CPU, use a separate locked CPU runtime rather than
reusing an environment previously modified by `install-cuda`. On Windows, create it outside any
gateway-approved workspace so the gateway can pin its executable as a stable regular file:

```bash
UV_PROJECT_ENVIRONMENT="$LOCALAPPDATA/ai-nd-co/agent-tools-cpu" \
  uv sync --locked --no-editable --python 3.13 --extra computer
```

Run that command from this repository. The gateway executable is then
`%LOCALAPPDATA%\ai-nd-co\agent-tools-cpu\Scripts\agent-tools.exe`. Explicit `--device cpu`
rejects CUDA-enabled Torch wheels before Kokoro imports them; this prevents a broken GPU runtime
from crashing an otherwise CPU-only synthesis process.

## Usage

### Single-command path: `ttsify`

```bash
echo "Turn this note into natural spoken narration." | agent-tools ttsify --output-file out.wav
```

To synthesize already-prepared text without any LLM or network transformation, add
`--no-transform`:

```bash
echo "This text is ready to speak." \
  | agent-tools ttsify --no-transform --device cpu --output-file out.wav
```

`ttsify` uses a built-in, language-preserving rewrite prompt and then sends the final text to the
selected local TTS engine. English remains on Kokoro. Explicit Russian or clearly Russian final text
uses Silero `v5_5_ru` automatically.

Default `ttsify` settings:

- model: `gpt-5.4-mini`
- reasoning effort: `medium`
- TTS engine: `auto`
- English voice: `af_heart`
- Russian voice: `xenia`

Configurable via env vars:

```bash
AGENT_TOOLS_CODEX_MODEL=gpt-5.4-mini
AGENT_TOOLS_CODEX_REASONING_EFFORT=medium
AGENT_TOOLS_KOKORO_VOICE=af_heart
AGENT_TOOLS_KOKORO_LANGUAGE=a
AGENT_TOOLS_KOKORO_SPEED=1.0
AGENT_TOOLS_KOKORO_DEVICE=auto
AGENT_TOOLS_TTS_ENGINE=auto
AGENT_TOOLS_SILERO_VOICE=xenia
AGENT_TOOLS_SILERO_CACHE_DIR=/path/to/model-cache
AGENT_TOOLS_TRANSFORM_PROVIDER=codex
AGENT_TOOLS_CLAUDE_CODE_MODEL=haiku
AGENT_TOOLS_CLAUDE_CODE_EFFORT=low
AGENT_TOOLS_CLAUDE_CODE_BARE=false
```

Claude Code transform models are intentionally limited to:

- `haiku`
- `sonnet`

`opus` is rejected by the CLI and runtime wrapper.

CLI flags override env vars.

Queue for playback on Windows:

```bash
echo "Turn this note into natural spoken narration." | agent-tools ttsify --output-mode play --source agent-a
```

### Optional private TTS service

Serve prepared text as direct WAV responses for a private client such as Vox:

```bash
agent-tools tts-server \
  --host 100.64.0.4 \
  --port 4223 \
  --token-file /path/to/owner-only-tts-token \
  --device cpu
```

The service prewarms both Kokoro English and Silero Russian before it becomes ready. It accepts one
synthesis job at a time and does not queue, persist, replay, proxy, or observe Codex traffic.

- `GET /healthz` requires `Authorization: Bearer ...` and returns `200` only when both engines are
  ready.
- `POST /v1/tts` requires authenticated `application/json` with exactly `requestId`, `text`, and
  `language`. Languages are `auto`, `en-US`, `en-GB`, `ru`, and `ru-RU`.
- Successful synthesis returns `audio/wav` directly with byte length, SHA-256, duration, effective
  engine/language, `no-store`, `nosniff`, and the echoed request ID in headers.
- Text is limited to 4,000 characters and WAV output to 12 MiB. A concurrent request receives `429`
  so the client can immediately use local TTS.
- Exact `127.0.0.1`, Tailscale IPv4, and RFC1918 IPv4 binds are accepted. Wildcard, hostname,
  public, link-local, query-credential, redirect, and unauthenticated use is rejected. Exposing an
  RFC1918 bind is appropriate only on an explicitly trusted LAN.

The bearer file is separate from Codex and Vox credentials. It must be a stable regular file owned
and readable only by the current OS user. Request text, bearer values, request bodies, filesystem
paths, and tracebacks are never written to normal service logs or error responses.

#### Owner-logon lifecycle

`agent-tools tts-startup` manages one exact task named
`ai-nd-co AgentTools TTS (Owner Logon)`. Its protected JSON config pins the current owner SID, Python
executable and SHA-256, AgentTools module root/bootstrap SHA-256, working directory, existing bearer
file, state/log directories, exact private endpoint, and CPU/CUDA device. The lifecycle actions are:

```bash
agent-tools tts-startup install --config /absolute/path/tts-startup.json --plan
agent-tools tts-startup status --config /absolute/path/tts-startup.json
agent-tools tts-startup start --config /absolute/path/tts-startup.json --plan
agent-tools tts-startup stop --config /absolute/path/tts-startup.json --plan
agent-tools tts-startup uninstall --config /absolute/path/tts-startup.json --plan
```

Remove `--plan` only during an explicitly approved local activation window. Installation uses the
current user's `InteractiveToken`, least privilege, an exact user-logon trigger, a hidden long-running
task, `IgnoreNew`, and three one-minute failure restarts. Its Windows-protected inline launcher verifies
and read-locks the pinned Python image through process creation. The runner separately owns a bounded child
restart budget, treats an exact stop request as success, and will not adopt or terminate a listener
whose PID and process generation do not match its child. Status output reports only typed identity and
readiness booleans; it never emits the bearer, its path, process command lines, or private file paths.

### Vox voice onboarding

Use the composed voice CLI after compatible Codex and optional TTS lifecycle owners have staged their
protected exact-address contract:

```bash
agent-tools voice setup --label "Studio PC" --alias studio
agent-tools voice status
agent-tools voice repair
agent-tools voice pair --expires-seconds 300
```

`voice setup --plan` performs network discovery only and reports the bounded actions without creating
identity, configuration, credentials, tasks, or listeners. Auto discovery prefers one cross-checked
Tailscale IPv4 address. It falls back only to one unambiguous lowest-metric RFC1918 address on a physical
Ethernet or Wi-Fi default route; that path requires `--trust-lan`. Wildcard, public, link-local, stale,
virtual, and ambiguous candidates are rejected.

Setup composes the existing lifecycle owners and writes an owner-protected stable Ed25519 PC identity,
voice configuration, setup lock, and recoverable transaction journal. Unknown tasks, executable drift,
configuration drift, identities, and listeners fail closed. Optional TTS failure is rolled back and
omitted while Codex remains usable.

Setup and pair print a five-minute terminal QR and non-secret check code. The QR carries only the exact
ephemeral HTTPS claim URL, expiry, nonce, TLS SPKI pin, stable public identity, label, alias, and network
class. The pinned response is signed by the stable PC identity and is the only pairing message containing
service credentials. The first app public key owns the invitation, same-key retries are idempotent, and a
matching acknowledgement consumes it. Pairing files and the exact high-port listener are removed after
acknowledgement or expiry.

### Disposable ZCode Desktop bridge

The ZCode bridge controls the already-running, already-paired ZCode Desktop harness through its official
Web Remote Control relay. It pins the installed ZCode CLI to `0.16.5`, loads the Desktop relay pair only in
memory, and creates a separate protected bearer and stable AgentTools bridge identity.

Prepare one exact Tailscale endpoint and existing workspace, then supervise the disposable process:

```bash
agent-tools zcode bridge prepare \
  --bind 100.64.0.4 \
  --port 55320 \
  --network tailscale \
  --workspace C:/projects/ai-nd-co
agent-tools zcode bridge start
agent-tools zcode bridge status --json
agent-tools zcode bridge stop
```

The bearer bridge is restricted to the encrypted Tailscale network; plaintext RFC1918 LAN endpoints are
not supported. Wildcard, loopback, public, link-local, IPv6, ambiguous, and stale adapter addresses are rejected. Ports must be explicitly chosen
from `49152` through `65535`; live ports `4222` and `4223` are always refused. `prepare` and lifecycle
commands are idempotent, and `retire` removes only proved stale runtime state after both the exact process
generation and listener are absent.

The local boundary is WebSocket `/zcode/v1` with subprotocol `agenttools.zcode.v1` and the protected
bridge bearer. It admits one authenticated client, accepts bounded `call`, `listen`, and `dispose`
messages, rejects stale relay generations, and suppresses exact duplicate request IDs. Status and errors
contain typed state only; relay pairing values, bearer values, request bodies, command lines, and owner
conversation content are never emitted. `start` creates no scheduled task or persistent registration.

### Worklog analysis

Analyze local Codex rollout logs and Claude Code project logs:

```bash
agent-tools worklog
```

Filter by local Asia/Tashkent dates and export JSON if needed:

```bash
agent-tools worklog \
  --since 2026-04-01 \
  --until 2026-04-30 \
  --format json \
  --output worklog-april.json
```

Default accounting rules:

- each real human prompt contributes a full **10 minute** active block
- prompts you typed still count even if they start with JSON, XML, HTML, or code
- overlapping blocks are merged by interval union
- when a block crosses midnight or a week boundary, it is split at that boundary
- weeks are grouped as **Monday-Sunday** in **Asia/Tashkent**
- repo attribution chooses one **primary repo** per merged interval
- ambiguous intervals are preserved under `multi/unknown`

Current log sources:

- Codex: `~/.codex/sessions/**/*.jsonl`
- Claude Code: `~/.claude/projects/**/*.jsonl`

Repo attribution uses the strongest available evidence in this order:

- changed file paths
- explicit file or directory paths in tool calls, prompts, or commands
- tool workdir/cwd
- session cwd

Claude subagent logs are not counted as additional human time.
Automatic Claude tool-result wrapper rows are not counted as human prompts.

### Desktop integrations

Install both supported desktop integrations:

```bash
agent-tools install-integrations
```

Install only one provider if needed:

```bash
agent-tools install-codex-integration
agent-tools install-claude-integration
```

- On native Windows Codex, this installs a `notify` command in `~/.codex/config.toml`.
- The Windows installer now points `notify` at a stable wrapper script:
  `~/.codex/scripts/agent-tools-notify.cmd`
- That wrapper is machine-local and is allowed to discover the correct local Python
  installation at runtime (`py`, `python`, then `pyenv` as a last fallback), so the
  synced Codex config does not need to hard-code a Python version path.
- On non-Windows, this keeps the Stop-hook integration path.
- Claude Code integration installs an AgentTools `Stop` hook into `~/.claude/settings.json`
  and writes the hook script to `~/.claude/agent-tools/stop_tts.sh`.
- The compatibility alias `agent-tools install-codex-stop-hook` remains available.

Windows debug logs:

- `~/.codex/notify_tts.log`
- `~/.codex/notify_tts_agent_tools.log`
- `~/.codex/scripts/agent-tools-notify.cmd`

On Windows, Codex passes the notify payload as the final JSON argv argument to the installed
notify command. AgentTools installs a wrapper `.cmd` file so the notify target stays stable even
when Python lives in different places on different machines.

This enqueues the generated audio, starts the background controller if needed, and returns
immediately.

Claude Code hook logs:

- `~/.claude/agent-tools/stop_tts.log`
- `~/.claude/agent-tools/stop_tts_agent_tools.log`

Shared performance log:

- Windows: `%LOCALAPPDATA%/AgentTools/state/performance.jsonl`
- Linux/macOS: `~/.local/share/agent-tools/state/performance.jsonl`

### Windows semantic accessibility

`computer inspect` uses the pinned WinApp provider by default and returns one normalized, bounded
element schema. Each element identifies its provider, role/control type, name/value/text fields,
bounds, enabled/read-only/password/actionable state, hierarchy signature, scroll state, and locator
metadata. Values remain omitted until an explicit bounded `computer read` call.

```bash
agent-tools computer inspect --hwnd 123 --json
agent-tools computer read --hwnd 123 --element edit-a1b2 --max-chars 4000 --json
agent-tools computer inspect --hwnd 123 --depth 12 --max-elements 200 --json
agent-tools computer invoke --hwnd 123 --element-ref eref_... --json
agent-tools computer capabilities --json
```

- Depths 1-7 stay on the compact WinApp path. Explicit depth 8-12 runs one bounded diagnostic
  budget across native UIA Raw, Control, and Content views; only Raw can become the fallback.
- Provider selection is deterministic. Native Raw is selected only when it exposes a non-shallow,
  materially better tree; native diagnostics are read-only and their `nuia-*` locators reject action
  commands with `semantic_provider_action_unsupported`.
- A shallow result includes `semantic_surface_shallow` with provider, element count, maximum depth,
  actionable count, text-bearing count, and reason. JSON also records every provider attempt.
- A truncated or depth-limited shallow-looking prefix is reported as
  `semantic_surface_inconclusive`; it does not trigger an observation fallback because controls may
  exist beyond the bounded prefix.
- A compact shallow-looking WinApp result requires an explicit depth 8-12 provider diagnostic before
  screenshot observation can be suggested. An inconclusive Native Raw diagnostic likewise suppresses
  screenshot fallback until a complete bounded result is available.
- MSAA/IAccessible and IAccessible2 were exercised as generic candidates but are not runtime providers:
  local Qt/Chrome bakeoffs found no stable fixture-control improvement over WinApp/native UIA.
- Observation and action fallbacks are independent. A shallow result may conditionally suggest a
  targeted screenshot for genuinely visual missing state. A guarded physical fallback is reported
  as available only when the caller supplies a fresh capture and explicitly authorizes one action.
- `inspect`, `read`, and `scroll-areas` issue owner-private `element_ref` values that expire after
  15 seconds. A read reports a bounded unavailable reason when its post-read target cannot be safely
  rebound inside the bounded semantic tree.
  Semantic `invoke`, `set-value`, and `scroll` prefer these references. Resolution uses exact
  RuntimeId first, then exactly-one AutomationId/control-type/class/ancestor metadata match so a
  provider rebuild does not create a false stale-element failure. Zero or multiple matches reject.
- Window titles are descriptive by default. `--require-title-match` enables strict compatibility
  behavior; PID, thread, process start, class, session, and input-desktop changes always reject.

Window state and semantic actions remain API/accessibility operations; they do not inject pointer or
keyboard input:

```bash
agent-tools computer restore --hwnd 123 --json
agent-tools computer minimize --hwnd 123 --json
agent-tools computer maximize --hwnd 123 --json
agent-tools computer resize --hwnd 123 --x 20 --y 20 --width 1200 --height 800 \
  --restore-first --json
agent-tools computer invoke --hwnd 123 --element-ref eref_... --json
agent-tools computer set-value --hwnd 123 --element-ref eref_... --stdin --json
agent-tools computer scroll --hwnd 123 --element-ref eref_... --direction down --json
```

Semantic mutations reclaim and verify the exact target foreground window inside the same serialized
operation. The native worker checks reference freshness, strong window identity, foreground ownership,
element identity, safety state, and the global emergency stop immediately before delivery.

### Windows computer screenshots

Prefer an application/API integration first, then `computer inspect`, `read`, and `scroll-areas`
when UI Automation exposes useful structure. Use a targeted screenshot only for visual, canvas, GPU,
or otherwise inaccessible state:

```bash
agent-tools computer screenshot --hwnd 123 --output window.png --json
agent-tools computer screenshot --hwnd 123 --output crop.png \
  --region 200,120,640,480 --profile compact --json
agent-tools computer screenshot --hwnd 123 --output native.png --profile native --json
agent-tools computer ocr --from-capture <native-capture-id> --last-lines 40
agent-tools computer ocr --from-capture <native-capture-id> --max-lines 100 \
  --max-chars 12000 --json
```

- `standard` is the default and bounds the longest edge to 720px.
- `compact` bounds the longest edge to 480px.
- `native` is explicit. No profile ever upscales.
- Local terminal OCR accepts only the fresh capture ID for the unchanged original `native` PNG.
  It rejects `compact`, `standard`, resized, copied, missing, expired, or hash-mismatched images before
  recognition and never invokes a screenshot backend or silently retries with a 480/720 derivative.
  Use ordinary `rg` or `jq` pipes; there is no OCR-specific grep option.
- OCR returns bounded text plus line/word boxes in native capture coordinates, the signed capture-time
  identity/geometry/region/hash/backend provenance, output truncation, and explicit unknown-confidence,
  substitution, clipping, and flattened-layout uncertainty. It is observation only and never proves a
  terminal draft or authorizes Enter. Keep application APIs and UI Automation ahead of OCR; use model
  vision only when the OCR uncertainty or spatial layout requires a separate inspection.
- `--region` coordinates are native window-relative pixels. Prefer a targeted 480/720 crop when a
  full-window presentation lacks detail.
- Capture tries the pinned Microsoft WinApp window path first, isolated `PrintWindow` second, and an
  isolated exact desktop crop only when the target is already foreground, fully on-screen, and
  unoccluded. The desktop fallback never requests focus and masks pixels outside the verified DWM
  frame before returning the window-coordinate canvas. WinApp 0.6.0 prefers Windows Graphics Capture
  but does not report whether its internal `PrintWindow` fallback ran, so JSON reports that limitation
  rather than claiming an unverified effective API.
- Every attempt revalidates strong target identity before and after capture: HWND, PID, process start,
  window class, Windows session, and input desktop. Window titles are mutable metadata, so title-only
  changes do not make a valid target stale. JSON reports `strong_identity_verified`, bounded
  `identity_mismatch_fields`, `title_changed_during_operation`, and keyed before/after title hashes;
  it does not expose raw titles through the capture record.
- A bounds, client-bounds, monitor, virtual-screen, DPI, or awareness change discards that frame and
  allows one recapture inside the original deadline. Returned pixels, geometry, transform, and capture
  record therefore describe one stable final geometry. WinApp 0.6.0's bounded screenshot header is
  removed before its verified window/DWM-frame image is normalized.
- Quality is checked on native-stage pixels before crop presentation resizing. JSON emits bounded
  black/white/transparent coverage, quantized dominant-color and active-content ratios, active
  bounding-box fraction, 1st-to-99th percentile luminance spread, luminance entropy, edge density,
  and the exact thresholds. Frames are rejected at 99.5% near-black/near-white coverage, 98%
  transparency, tiny-mark combinations (99.5% dominant color with at most 0.5% active pixels, 10%
  active bounds, and 0.2% edges), or jointly flat luminance/entropy/edge signals. These thresholds are
  covered by black, white, transparent, tiny-mark, legitimate dark UI, terminal, and colorful tests.
- `content_non_uniform` remains for response compatibility but is not a usefulness decision. Use the
  `quality` and `capture_attempts` fields. Failed backends and rejected frames are reported with
  bounded error codes; a visually useless frame is never returned as success.
- A presentation PNG is observation only, never coordinate authority. The JSON includes a short-lived
  opaque capture ID, native geometry/DPI, exact rational transform, image hash, and expiry. A future
  physical-input command must accept image coordinates only with that capture ID, validate the
  owner-private record and image, and re-read target identity/geometry/DPI immediately before input.
- Edge pixels map exactly when a presentation axis has at least two pixels. If scaling collapses an
  axis to one pixel, that pixel maps to the native midpoint using nearest-half-up rounding.
- Bare action coordinates are native coordinates; resized-image coordinates must never be passed as
  native coordinates.
- Screenshots do not authorize focus changes, clicks, or physical input. Guarded physical input is a
  separate last-resort operation and must revalidate a fresh capture record immediately before use.

### Windows guarded physical fallback

Use physical input only after application APIs and semantic UI Automation are unavailable. Every
command requires `--allow-physical`, binds to one exact HWND and fresh screenshot capture, rechecks
strong window identity, geometry, DPI, foreground ownership, held input, and the global input
generation immediately before delivery, and then observes one privacy-safe consequence for two
seconds. A delivered physical action consumes its capture authority exactly once and is never
automatically retried or resent.

```bash
agent-tools computer screenshot --hwnd 123 --output fixture.png --json
agent-tools computer click --hwnd 123 --from-capture <capture-id> \
  --image-at 240,180 --allow-physical --json

agent-tools computer screenshot --hwnd 123 --output fixture-draft.png --json
agent-tools computer type --hwnd 123 --from-capture <capture-id> \
  --image-at 240,180 --text "review before submit" --allow-physical --json
```

`type` stages text without Enter and returns a signed, one-shot staged-input ID. The caller must let
the application settle, verify the visible draft in a fresh capture, and send Enter as a separate
transaction:

```bash
agent-tools computer screenshot --hwnd 123 --output fixture-verified.png --json
agent-tools computer key --hwnd 123 --from-capture <fresh-capture-id> \
  --image-at 240,180 --key ENTER --staged-input <staged-input-id> \
  --draft-evidence exact --allow-physical --json
```

- Ordinary text uses the active layout's `VkKeyScanW` virtual-key mapping. Unmappable characters
  reject before focus or delivery; AgentTools does not use Unicode packets, the clipboard,
  `PostMessage`, or shell paste.
- Partial text delivery reports the exact first undelivered character and leaves the visible partial
  draft untouched. There is no automatic Enter or cleanup. Recovery is a new, explicit transaction:
  take a fresh capture, use a caller-chosen bounded shortcut such as Ctrl+K when appropriate, then
  re-stage and visibly verify the draft.
- Click, type, key, shortcut, and wheel restore the prior cursor position when possible. Only
  `move-pointer` intentionally leaves the cursor at the capture-derived point.
- Windows Terminal can host multiple tabs inside one HWND without exposing a stable semantic tab
  identity. Terminal physical input therefore requires `--visible-tab-verified`; a same-HWND tab
  change remains an explicit targeting limitation and is reported rather than silently assumed safe.

Each JSONL row records one completed command or dispatch with stage timings such as
`transform_ms`, `tts_generation_ms`, `tts_postprocess_ms`, `enqueue_ms`, and `total_ms`.

You can also manage AgentTools auto-TTS integration from the desktop controller UI and tray menu.
AgentTools only needs **one** backend to be available:

- Codex, with local ChatGPT login working
- or Claude Code installed on PATH

If neither backend is available, the UI shows an info-only message telling you to install or sign
in to any one of them first.

### Transform text

```bash
echo "Rewrite this into short spoken narration." | agent-tools transform \
  --system-prompt-file prompt_examples/rewrite_for_tts.md
```

Optional controls:

```bash
echo "Input text" | agent-tools transform \
  --system-prompt-file prompt_examples/rewrite_for_tts.md \
  --provider codex \
  --model gpt-5 \
  --reasoning-effort medium \
  --fast
```

Experimental Claude Code-backed transform:

```bash
echo "Input text" | agent-tools transform \
  --system-prompt-file prompt_examples/rewrite_for_tts.md \
  --provider claude-code \
  --claude-model haiku \
  --claude-effort low
```

What the Claude wrapper does:

- runs `claude -p` in a temporary minimal working directory
- forces `--max-turns 1`, `--tools ""`, `--no-session-persistence`, and `--output-format json`
- uses `--system-prompt` with the same rewrite prompt file you pass to `transform`
- does **not** require or use any unofficial Claude Code API

Important limitation:

- `--claude-bare` is supported, but it is **off by default** because local Claude help shows bare
  mode only reads `ANTHROPIC_API_KEY` or `apiKeyHelper` auth. If you rely on normal Claude login
  state, keep `--claude-bare` off.

### Text to speech

```bash
echo "Hello world." | agent-tools tts --output-file hello.wav
```

Russian text selects the official Silero model automatically:

```bash
echo "Всё готово. На двери висит замок, а старинный замок далеко?" \
  | agent-tools ttsify --no-transform --language ru --device cpu \
    --output-file russian.wav
```

Select the engine and one of the five Silero voices explicitly when needed:

```bash
echo "Привет, мир!" \
  | agent-tools tts --tts-engine silero --language ru --voice baya \
    --device cpu --output-file russian-baya.wav
```

Silero voices are `aidar`, `baya`, `kseniya`, `xenia`, and `eugene`. AgentTools uses 24 kHz mono
PCM for both engines and does not silently resample. Silero speed is therefore fixed at `1.0`.

Queue already-prepared speech on Windows:

```bash
echo "Hello world." | agent-tools tts --output-mode play --source agent-a
```

### Desktop controller UI

```bash
agent-tools ui
```

If the controller is already running, this focuses the existing window instead of starting a
second process.

The controller behavior is:

- when neither Codex nor Claude Code is available, the normal playback UI is hidden and the window
  shows an info-only message telling you to install or sign in to any one backend first
- when either backend is available, the normal playback UI stays usable
- the single "Enable Codex + Claude Code integration" checkbox is the install state for the
  integration:
  - checked = install/fix the relevant Codex + Claude integration artifacts
  - unchecked = remove/uninstall those artifacts
- there is no separate user-facing "soft disable" state for the checkbox flow; if users want the
  integration off, unchecking the box removes the installed integration artifacts
- a dropdown chooses the default transform engine used for `ttsify` and desktop auto-TTS:
  Codex or Claude Code
- if the saved/default provider is unavailable but another backend is available, AgentTools falls
  back automatically unless you explicitly force a provider on the CLI

Current known-good Windows state after checking the box:

- `~/.codex/config.toml` contains a `notify = ["cmd.exe", "/d", "/c", "...agent-tools-notify.cmd"]`
  entry
- `~/.codex/scripts/agent-tools-notify.cmd` exists
- Codex TUI no longer needs a machine-specific Python path in synced config
- unchecking the box removes the wrapper and removes the `notify` config entry

### End-to-end pipeline

```bash
cat input.txt | agent-tools transform \
  --system-prompt-file prompt_examples/rewrite_for_tts.md \
  | agent-tools tts --voice af_heart --output-file out.wav
```

## Notes

- `ttsify` is the recommended end-user path; `transform` and `tts` remain available as building blocks.
- `transform` reads stdin by default and writes plain text to stdout.
- `tts` reads stdin by default and writes WAV bytes to stdout unless `--output-file` is set.
- `tts` and `ttsify` support `--output-mode play` on Windows.
- in play mode, audio is queued into a single background controller process.
- `agent-tools ui` launches or focuses the popup/tray controller window.
- controller shortcuts: `Space` pause/resume, `Esc` stop, `Ctrl+R` replay, `Ctrl+N` next.
- `tts` and `ttsify` default to `--device auto`.
- `tts` and `ttsify` accept `--tts-engine auto|kokoro|silero`; an explicit engine wins.
- auto mode selects Silero for explicit `ru`/`ru-RU`, or when the final synthesized text contains at
  least four Russian Cyrillic letters, those letters are at least half of all letters, and no
  non-Russian Cyrillic letters are present. Otherwise it keeps Kokoro and `af_heart`.
- auto device selection uses a real CUDA probe, not just `torch.cuda.is_available()`.
- CUDA probing runs in a bounded child process so a native driver crash becomes a diagnostic
  fallback/error instead of terminating the caller.
- English speech falls back to the installed eSpeak backend when `en_core_web_sm` is absent;
  runtime model installation is never attempted by this fallback.
- `agent-tools install-cuda` reinstalls the full PyTorch stack (`torch`, `torchvision`, `torchaudio`) into the current Python environment and validates the full Kokoro import chain in a fresh subprocess by default.
- `transform` refreshes ChatGPT tokens when the Codex backend returns `401`.
- the experimental Claude transform path uses the official `claude` CLI only; it is intentionally
  limited to one turn with tools disabled
- Native Windows Codex uses `notify`; `hooks.json` lifecycle hooks are not used there.
- semantic-release now owns future Python package version bumps and `py-v*` tags.

## Silero model cache and license

AgentTools downloads only the official versioned artifact:

- model: `v5_5_ru`
- URL: `https://models.silero.ai/models/tts/ru/v5_5_ru.pt`
- size: `145420684` bytes
- SHA-256: `50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437`

The download is written atomically, verified before use, and reused offline from the standard user
cache. Override the directory with `AGENT_TOOLS_SILERO_CACHE_DIR`. AgentTools loads the packaged
model directly with PyTorch; it does not clone Silero or install runtime code.

Silero's Russian model is published under
[CC BY-NC-SA 4.0](https://github.com/snakers4/silero-models/blob/master/LICENSE), which restricts it
to non-commercial use and requires attribution/share-alike when applicable. AgentTools code remains
under its own Apache-2.0 license; the model is downloaded separately from Silero.

## CPU performance

Measured on this machine on **April 15, 2026** with **forced CPU**:

| Scenario | Wall time | Audio time | Real-time factor |
|---|---:|---:|---:|
| first-ever cold init after dependency/model setup | ~43.1s | n/a | n/a |
| cached init | ~2.9s | n/a | n/a |
| warm short | 0.309s | 4.80s | 0.064 |
| warm medium | 1.199s | 15.53s | 0.077 |
| warm long | 2.514s | 26.70s | 0.094 |

Interpretation:

- warm CPU generation on this machine is about **10x-15x faster than realtime**
- the main cost is **cold startup/model load**, not steady-state synthesis

To reproduce locally:

```bash
python scripts/benchmark_tts_cpu.py
```

## Troubleshooting

- Missing `~/.codex/auth.json`: run `codex login`
- Expired auth: rerun `codex login` if refresh fails permanently
- Missing `espeak-ng`: install it for better English fallback behavior
- Silero cache integrity failure: remove only the reported `v5_5_ru.pt` cache file and retry
- Silero first run offline: connect once so the verified official model can populate the user cache
- Slow first run: expected; Kokoro downloads voices/models and initializes the pipeline
- After changing Python versions for the interpreter that runs `agent-tools`, rerun `agent-tools install-cuda` in that same interpreter to repair the PyTorch stack for Kokoro
- If Windows Codex logs `after_agent hook failed` with `program not found`, rerun the checkbox
  install flow or run `agent-tools install-codex-integration` again. The expected repaired config
  should point to `cmd.exe /d /c ~/.codex/scripts/agent-tools-notify.cmd`, not directly to a
  version-specific Python path.
- If a synced `~/.codex/config.toml` still contains an older `notify = ["agent-tools", ...]` or a
  hard-coded Python path from a previous machine, reinstall the Codex integration on the current
  machine so the wrapper path is rewritten locally.
