# agent-tools

Python CLI tools for:

- transforming raw text into TTS-ready narration and synthesizing it in one command
- transforming piped text through the private Codex ChatGPT-backed backend used by local Codex
- synthesizing the result to WAV with Kokoro-82M

This repo is intentionally wired to the local Codex login state in `~/.codex/`.

Release policy:

- semantic-release owns version bumps, changelog updates, and `py-v*` tags
- do not manually edit `project.version` in `pyproject.toml` during normal work
- do not create release tags by hand unless the release workflow explicitly calls for it

## Status

This is an **experimental public package** with a **private Codex dependency**.

The `transform` command mirrors the current request shape used by the local Codex source tree and depends on ChatGPT-backed auth in `~/.codex/auth.json`.

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

Install a CUDA-enabled PyTorch runtime for this CLI environment:

```bash
agent-tools install-cuda
```

Pass an explicit track if you do not want auto-detection:

```bash
agent-tools install-cuda --cuda-track cu130
```

## Usage

### Single-command path: `ttsify`

```bash
echo "Turn this note into natural spoken narration." | agent-tools ttsify --output-file out.wav
```

`ttsify` uses a built-in rewrite prompt stored in the package and then pipes the transformed text
into Kokoro TTS.

Default `ttsify` settings:

- model: `gpt-5.4-mini`
- voice: `af_heart`

Configurable via env vars:

```bash
AGENT_TOOLS_CODEX_MODEL=gpt-5.4-mini
AGENT_TOOLS_CODEX_REASONING_EFFORT=medium
AGENT_TOOLS_KOKORO_VOICE=af_heart
AGENT_TOOLS_KOKORO_LANGUAGE=a
AGENT_TOOLS_KOKORO_SPEED=1.0
AGENT_TOOLS_KOKORO_DEVICE=auto
```

CLI flags override env vars.

Queue for playback on Windows:

```bash
echo "Turn this note into natural spoken narration." | agent-tools ttsify --output-mode play --source agent-a
```

### Codex integration

Install the supported Codex integration for the current platform:

```bash
agent-tools install-codex-integration
```

- On native Windows Codex, this installs a `notify` command in `~/.codex/config.toml`.
- On non-Windows, this keeps the Stop-hook integration path.
- The compatibility alias `agent-tools install-codex-stop-hook` remains available.

Windows debug logs:

- `~/.codex/notify_tts.log`
- `~/.codex/notify_tts_agent_tools.log`

On Windows, Codex passes the notify payload as the final JSON argv argument to the installed
Python command. No PowerShell or bash wrapper is used.

This enqueues the generated audio, starts the background controller if needed, and returns
immediately.

### Transform text

```bash
echo "Rewrite this into short spoken narration." | agent-tools transform \
  --system-prompt-file prompt_examples/rewrite_for_tts.md
```

Optional controls:

```bash
echo "Input text" | agent-tools transform \
  --system-prompt-file prompt_examples/rewrite_for_tts.md \
  --model gpt-5 \
  --reasoning-effort medium \
  --fast
```

### Text to speech

```bash
echo "Hello world." | agent-tools tts --output-file hello.wav
```

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
- auto device selection uses a real CUDA probe, not just `torch.cuda.is_available()`.
- `agent-tools install-cuda` installs a CUDA-enabled PyTorch build into the current Python environment and validates it in a fresh subprocess by default.
- `transform` refreshes ChatGPT tokens when the Codex backend returns `401`.
- Native Windows Codex uses `notify`; `hooks.json` lifecycle hooks are not used there.
- semantic-release now owns future Python package version bumps and `py-v*` tags.

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
- Slow first run: expected; Kokoro downloads voices/models and initializes the pipeline
