# ZCode owner-login wrapper preparation

`agent_tools.zcode_startup` uses the existing bridge start/status/stop/retire contracts. The producer helper proves the installed Desktop root and its Electron Node host/app-server children are High under the same interactive owner. Electron also uses ZCode.exe for app-server children; executable name alone is not a Desktop identity. An existing Medium, unknown or ambiguous producer is refused without replacement. A missing producer may be launched once using its pinned installed executable; backend readiness must appear within 60 seconds. No GUI sign-in automation or engine CLI invention is involved.

The wrapper owns crash retries at 10/30/60 seconds, resetting after five stable minutes. The bridge retains its own relay reconnection logic; the wrapper does not restart reconnecting or authentication/config/identity-failed generations. A clean bridge stop ends the episode. An episode-bound stop request interrupts backoff; the CLI waits for exact wrapper release and bridge release before returning stopped. No port/PID kill is added. Producer Desktop is never stopped by bridge lifecycle operations.

Prepared entry points:

- `python -m agent_tools.zcode_startup run --root <enrolled-bridge-root> --producer <canonical-ZCode.exe> --producer-sha256 <sha256> --owner-sid <owner-SID>`
- `python -m agent_tools.zcode_startup status --root <enrolled-bridge-root>`
- `python -m agent_tools.zcode_startup stop --root <enrolled-bridge-root>`

These are source preparation commands, not an activation instruction. `render_task_xml` produces a disabled owner-logon InteractiveToken/HighestAvailable definition with no Scheduler restart loop and no hard termination. It does not register a task. The dedicated stop command must release the wrapper/bridge before a task is disabled or removed. Status reports the wrapper and bridge separately; a running wrapper does not claim authenticated readiness.

Remaining activation preparation:

1. Integrate final Task352 bridge/relay bytes and the enrolled bridge configuration into a fresh protected package. The package builder and complete Python/dependency fixture now exist; no mutable repository, user-writable venv or unprotected Python startup/import path may become a Highest task action. Bootstrap refuses a bridge interpreter/module path outside its package.
2. The disabled registration/receipt/removal helper and real task-owned High/Medium/write-refusal/foreign-task fixtures are implemented. Final supervised bridge stop/no-respawn and authenticated backend readiness still require the enrolled final backend package. Producer tests do not launch real ZCode.
3. Reuse enrolled relay/workspace identity and its private endpoint. Do not generate new production credentials, alter history or point at Codex/TTS ports. Stop/cutover of the existing producer and live ZCode activation require the separately coordinated activation gate. Actual logoff/logon is a separate proof from manual Scheduler start.

The wrapper does not replace the existing bridge service with a new generic daemon. A wrapper crash leaves the existing bridge's ownership/lifecycle records intact; recovery must inspect them. Do not treat raw Scheduler termination as successful contained bridge stop.

Package preparation after the 4300a25 source milestone:

- `zcode_startup_package.prepare_package` copies a base Windows Python runtime, a supplied complete dependency environment, and AgentTools source into a new directory. It excludes ambient base site-packages and bytecode, refuses editable/unapproved .pth hooks, hashes every file and uses the CWR Task354 admin/SYSTEM-write, owner-read ACL helper including ancestor replacement checks. Supply a lean dependency environment with the declared bridge import closure; do not pass a development venv containing editable imports.
- Immutable settings contain exactly ownerSid, bridgeRoot, bridgeConfigSha256, producer and producerSha256. No credentials enter the package manifest. The protected launcher verifies all listed bytes and rejects unlisted files before Python starts. Both wrapper and bridge child use `-I -B`; inherited Python/Node/Electron injection variables are cleared at the relevant launch boundary.
- `prepare_task_receipt(package, manifest_sha256, owner_sid, destination)` verifies the package and cold imports, then writes a new protected receipt for a disabled owner-logon definition. Use its SHA256 with `zcode_startup_task.ps1 -Operation InstallDisabled|Query|RemoveDisabled -ReceiptRoot <root> -ReceiptSha256 <digest> -OwnerSid <sid>`. The helper saves exact normalized registered XML and refuses foreign, active or changed task removal. An interrupted registration without its saved XML is an explicit refusal requiring inspection, not permission to overwrite the task.
- No activation operation is exposed. Final package construction must use the final bridge bytes and prepared/enrolled config; the retained native fixtures intentionally have no enrolled bridge config. Manual fixture starts prove tokens/ACL boundaries, not actual owner logoff/logon or production bridge readiness.
