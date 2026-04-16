#!/usr/bin/env bash
set -euo pipefail

hook_root="${HOME}/.codex/hooks"
state_dir="${hook_root}/state"
audio_tmp_dir="${hook_root}/tmp-audio"
mkdir -p "${state_dir}" "${audio_tmp_dir}"

# Best-effort cleanup of old state/audio artifacts.
find "${state_dir}" -type f -mtime +7 -delete 2>/dev/null || true
find "${audio_tmp_dir}" -type f -mtime +2 -delete 2>/dev/null || true

payload="$(cat)"
if [[ -z "${payload}" ]]; then
  exit 0
fi

parsed_json="$({ PAYLOAD_JSON="${payload}" python - <<'PY'
import json
import os

payload = os.environ.get('PAYLOAD_JSON', '')
if not payload:
    print('{}')
    raise SystemExit(0)

try:
    data = json.loads(payload)
except Exception:
    print('{}')
    raise SystemExit(0)

out = {
    'turn_id': data.get('turn_id') or '',
    'session_id': data.get('session_id') or '',
    'stop_hook_active': bool(data.get('stop_hook_active')),
    'last_assistant_message': data.get('last_assistant_message') or '',
}
print(json.dumps(out))
PY
} 2>/dev/null || true)"

if [[ -z "${parsed_json}" ]]; then
  exit 0
fi

read_json_field() {
  local field="$1"
  PAYLOAD_JSON="${parsed_json}" python - "$field" <<'PY'
import json
import os
import sys
field = sys.argv[1]
payload = os.environ.get('PAYLOAD_JSON', '{}')
try:
    data = json.loads(payload)
except Exception:
    print("")
    raise SystemExit(0)
value = data.get(field)
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(str(value))
PY
}

turn_id="$(read_json_field turn_id)"
session_id="$(read_json_field session_id)"
stop_hook_active="$(read_json_field stop_hook_active)"
message="$(read_json_field last_assistant_message)"

if [[ "${stop_hook_active}" == "true" ]]; then
  exit 0
fi

if [[ -z "${message//[[:space:]]/}" ]]; then
  exit 0
fi

if [[ -n "${turn_id}" ]]; then
  turn_marker="${state_dir}/${turn_id}.done"
  if [[ -f "${turn_marker}" ]]; then
    exit 0
  fi
  : > "${turn_marker}"
fi

source_label="codex-stop"
if [[ -n "${session_id}" ]]; then
  source_label="codex-stop:${session_id}"
fi

if ! command -v agent-tools >/dev/null 2>&1; then
  echo "stop_tts.sh: agent-tools not found on PATH" >&2
  exit 0
fi

run_windows_queue() {
  printf '%s' "${message}" | env AGENT_TOOLS_CODEX_INTEGRATION_TRIGGERED=1 AGENT_TOOLS_SOURCE="${source_label}" agent-tools ttsify --output-mode play >/dev/null 2>&1 &
}

run_file_then_player() {
  local audio_file="$1"
  printf '%s' "${message}" | env AGENT_TOOLS_CODEX_INTEGRATION_TRIGGERED=1 AGENT_TOOLS_SOURCE="${source_label}" agent-tools ttsify --output-file "${audio_file}" >/dev/null 2>&1 || return 1

  if command -v afplay >/dev/null 2>&1; then
    nohup afplay "${audio_file}" >/dev/null 2>&1 &
    return 0
  fi
  if command -v paplay >/dev/null 2>&1; then
    nohup paplay "${audio_file}" >/dev/null 2>&1 &
    return 0
  fi
  if command -v aplay >/dev/null 2>&1; then
    nohup aplay "${audio_file}" >/dev/null 2>&1 &
    return 0
  fi
  if command -v ffplay >/dev/null 2>&1; then
    nohup ffplay -nodisp -autoexit "${audio_file}" >/dev/null 2>&1 &
    return 0
  fi
  return 1
}

uname_s="$(uname -s 2>/dev/null || echo unknown)"
case "${uname_s}" in
  MINGW*|MSYS*|CYGWIN*)
    run_windows_queue
    ;;
  Darwin)
    audio_file="${audio_tmp_dir}/${turn_id:-$(date +%s)}.wav"
    run_file_then_player "${audio_file}" || echo "stop_tts.sh: no supported audio player found on macOS" >&2
    ;;
  Linux)
    audio_file="${audio_tmp_dir}/${turn_id:-$(date +%s)}.wav"
    run_file_then_player "${audio_file}" || echo "stop_tts.sh: no supported Linux audio player found (tried paplay/aplay/ffplay)" >&2
    ;;
  *)
    echo "stop_tts.sh: unsupported platform ${uname_s}" >&2
    ;;
esac

# Stop hooks must not emit plain text on stdout.
exit 0
