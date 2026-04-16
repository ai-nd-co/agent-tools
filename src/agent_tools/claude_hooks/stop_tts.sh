#!/usr/bin/env bash
set -euo pipefail

hook_root="${HOME}/.claude/agent-tools"
log_file="${hook_root}/stop_tts.log"
child_log="${hook_root}/stop_tts_agent_tools.log"
touch "${log_file}" "${child_log}" 2>/dev/null || true

log() {
  local timestamp
  timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z' 2>/dev/null || date)"
  printf '%s %s\n' "${timestamp}" "$*" >>"${log_file}" 2>/dev/null || true
}

log "hook_start pid=$$ pwd=$(pwd) uname=$(uname -s 2>/dev/null || echo unknown)"

payload="$(cat)"
log "payload_bytes=${#payload}"
if [[ -z "${payload}" ]]; then
  log "payload_empty_exit"
  exit 0
fi

parsed_json="$({ PAYLOAD_JSON="${payload}" python - <<'PY'
import json
import os

payload = os.environ.get("PAYLOAD_JSON", "")
if not payload:
    print("{}")
    raise SystemExit(0)

try:
    data = json.loads(payload)
except Exception:
    print("{}")
    raise SystemExit(0)

out = {
    "session_id": data.get("session_id") or "",
    "stop_hook_active": bool(data.get("stop_hook_active")),
    "last_assistant_message": data.get("last_assistant_message") or "",
}
print(json.dumps(out))
PY
} 2>>"${child_log}" || true)"

if [[ -z "${parsed_json}" ]]; then
  log "parsed_json_empty_exit"
  exit 0
fi

read_json_field() {
  local field="$1"
  PAYLOAD_JSON="${parsed_json}" python - "$field" <<'PY'
import json
import os
import sys

field = sys.argv[1]
payload = os.environ.get("PAYLOAD_JSON", "{}")
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

session_id="$(read_json_field session_id)"
stop_hook_active="$(read_json_field stop_hook_active)"
message="$(read_json_field last_assistant_message)"

log "parsed session_id=${session_id:-<empty>} stop_hook_active=${stop_hook_active} message_bytes=${#message}"

if [[ "${stop_hook_active}" == "true" ]]; then
  log "stop_hook_active_exit"
  exit 0
fi

if [[ -z "${message//[[:space:]]/}" ]]; then
  log "message_blank_exit"
  exit 0
fi

source_label="claude-stop"
if [[ -n "${session_id}" ]]; then
  source_label="claude-stop:${session_id}"
fi

agent_tools_path="$(command -v agent-tools 2>/dev/null || true)"
if [[ -z "${agent_tools_path}" ]]; then
  log "agent_tools_missing PATH=${PATH}"
  exit 0
fi
log "agent_tools_path=${agent_tools_path} source_label=${source_label}"

(
  printf '%s' "${message}" | env AGENT_TOOLS_CLAUDE_INTEGRATION_TRIGGERED=1 AGENT_TOOLS_SOURCE="${source_label}" agent-tools ttsify --output-mode play
) >>"${child_log}" 2>&1 &
child_pid=$!
log "spawned_agent_tools_playback pid=${child_pid}"

log "hook_exit_ok"
exit 0
