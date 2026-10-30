"""ai-nd-co additions to Speaches for the office cluster's speech-stt.

- ext: admission control for Whisper and a model allowlist, installed by wrapping Speaches'
  classes at runtime. No Speaches file is edited, so patch 433's hash guard on whisper.py holds.
- tts: Kokoro text-to-speech on the CPU with a fixed thread count, so it takes no GPU memory.
- langid: a language-ID-only listener that reuses the Whisper model already loaded.
- serve: the launcher. One uvicorn server, two sockets: Speaches on :8000, language ID on :8002.
"""
