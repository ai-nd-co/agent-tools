from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    settings = json.loads((root / "startup.json").read_text("utf-8"))
    sys.path.insert(0, str(root / "app"))
    from agent_tools.zcode_bridge import load_bridge_config
    from agent_tools.zcode_startup import _high_owner_session
    from agent_tools.zcode_startup import main as startup_main

    operation = sys.argv[1]
    if operation == "probe":
        print(json.dumps({
            "code": "package_ready", "pid": os.getpid(),
            "highOwnerSession": _high_owner_session(os.getpid(), settings["ownerSid"]),
        }))
        return 0
    bridge_root = Path(settings["bridgeRoot"])
    digest = hashlib.sha256((bridge_root / "config.json").read_bytes()).hexdigest()
    if digest != settings["bridgeConfigSha256"]:
        raise ValueError("bridge configuration changed")
    config = load_bridge_config(bridge_root)
    if config.python_executable != (root / "runtime/python.exe").resolve():
        raise ValueError("bridge interpreter outside package")
    for field in (
        "module_file", "package_init_file", "relay_module_file", "voice_module_file",
        "startup_module_file",
    ):
        if not getattr(config, field).is_relative_to(root / "app/agent_tools"):
            raise ValueError("bridge import outside package")
    return startup_main([
        operation, "--root", str(bridge_root), "--producer", settings["producer"],
        "--producer-sha256", settings["producerSha256"], "--owner-sid", settings["ownerSid"],
    ])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print('{"code":"startup_package_invalid"}')
        raise SystemExit(2) from None
