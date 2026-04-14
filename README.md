# agent-tools

Python CLI tools for:

- transforming piped text through the private Codex ChatGPT-backed backend used by local Codex
- synthesizing the result to WAV with Kokoro-82M

This repo is intentionally wired to the local Codex login state in `~/.codex/`.

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

- Python 3.12+
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
pip install ai-agent-tools
```

## Usage

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

### End-to-end pipeline

```bash
cat input.txt | agent-tools transform \
  --system-prompt-file prompt_examples/rewrite_for_tts.md \
  | agent-tools tts --voice af_heart --output-file out.wav
```

## Notes

- `transform` reads stdin by default and writes plain text to stdout.
- `tts` reads stdin by default and writes WAV bytes to stdout unless `--output-file` is set.
- `tts --device auto` prefers CUDA and falls back to CPU.
- `transform` refreshes ChatGPT tokens when the Codex backend returns `401`.

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
