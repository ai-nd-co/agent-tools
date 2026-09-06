# ZCode owner-login wrapper preparation

`agent_tools.zcode_startup` uses the existing bridge start/status/stop/retire contracts. The producer helper proves the installed Desktop root and its Electron Node host/app-server children are High under the same interactive owner. Electron also uses ZCode.exe for app-server children; executable name alone is not a Desktop identity. An existing Medium, unknown or ambiguous producer is refused without replacement. A missing producer may be launched once using its pinned installed executable; backend readiness must appear within 60 seconds. No GUI sign-in automation or engine CLI invention is involved.

The wrapper owns crash retries at 10/30/60 seconds, resetting after five stable minutes. The bridge retains its own relay reconnection logic; the wrapper does not restart reconnecting or authentication/config/identity-failed generations. A clean bridge stop ends the episode. An episode-bound stop request interrupts backoff; the CLI waits for exact wrapper release and bridge release before returning stopped. No port/PID kill is added. Producer Desktop is never stopped by bridge lifecycle operations.

Prepared entry points:

- `python -m agent_tools.zcode_startup run --root <enrolled-bridge-root> --producer <canonical-ZCode.exe> --producer-sha256 <sha256> --owner-sid <owner-SID>`
- `python -m agent_tools.zcode_startup status --root <enrolled-bridge-root>`
- `python -m agent_tools.zcode_startup stop --root <enrolled-bridge-root>`

These are source preparation commands, not an activation instruction. `render_task_xml` produces a disabled owner-logon InteractiveToken/HighestAvailable definition with no Scheduler restart loop and no hard termination. It does not register a task. The dedicated stop command must release the wrapper/bridge before a task is disabled or removed. Status reports the wrapper and bridge separately; a running wrapper does not claim authenticated readiness.

Remaining activation preparation:

1. Integrate the final Task352 bridge/relay bytes before producing one protected, complete Python/AgentTools/dependency bootstrap. Bind installed producer and prepared bridge configuration to that manifest. No mutable repository, user-writable venv or unprotected Python startup/import path may become a Highest task action. This wrapper checkpoint does not provide that complete package yet.
2. Prepare exact disabled task registration/CAS/removal against that package, with protected definition receipt and foreign-task refusal. Prove the final wrapper's real Highest/Medium tokens, intentional stop/no-respawn and backend authenticated readiness using task-owned fixtures. Current tests exercise real producer helper code with disposable process observations, not a real ZCode launch.
3. Reuse enrolled relay/workspace identity and its private endpoint. Do not generate new production credentials, alter history or point at Codex/TTS ports. Stop/cutover of the existing producer and live ZCode activation require the separately coordinated activation gate. Actual logoff/logon is a separate proof from manual Scheduler start.

The wrapper does not replace the existing bridge service with a new generic daemon. A wrapper crash leaves the existing bridge's ownership/lifecycle records intact; recovery must inspect them. Do not treat raw Scheduler termination as successful contained bridge stop.
