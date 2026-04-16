from __future__ import annotations

import json
from pathlib import Path

from agent_tools.agent_integration import load_agent_integration_status


def test_load_agent_integration_status_requires_both_providers(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")

    codex_home = tmp_path / ".codex"
    hook_dir = codex_home / "hooks"
    hook_dir.mkdir(parents=True)
    (codex_home / "config.toml").write_text(
        "[features]\n"
        "codex_hooks = true\n",
        encoding="utf-8",
    )
    (codex_home / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "bash -lc '$HOME/.codex/hooks/stop_tts.sh'",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (hook_dir / "stop_tts.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    status = load_agent_integration_status(
        codex_home=codex_home,
        claude_home=tmp_path / ".claude",
        platform_name="linux",
    )

    assert status.install_state == "missing"
    assert status.effective_enabled is False
