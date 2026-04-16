from __future__ import annotations

import json
from pathlib import Path

from agent_tools.agent_integration import (
    load_agent_integration_status,
    resolve_transform_provider_or_fallback,
)


def test_load_agent_integration_status_allows_one_available_provider(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    import agent_tools.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "app_root", lambda: tmp_path / "app")
    monkeypatch.setattr("shutil.which", lambda name: None if name == "claude" else None)

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
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "account_id": "account-1",
                },
            }
        ),
        encoding="utf-8",
    )

    status = load_agent_integration_status(
        codex_home=codex_home,
        claude_home=tmp_path / ".claude",
        platform_name="linux",
    )

    assert status.install_state == "installed"
    assert status.integration_state == "installed"
    assert status.available_providers == ("codex",)
    assert status.any_provider_available is True


def test_resolve_transform_provider_or_fallback_uses_available_provider() -> None:
    provider = resolve_transform_provider_or_fallback(
        requested_provider="claude-code",
        available_providers=("codex",),
        explicit=False,
    )

    assert provider == "codex"


def test_resolve_transform_provider_or_fallback_rejects_explicit_missing_provider() -> None:
    try:
        resolve_transform_provider_or_fallback(
            requested_provider="claude-code",
            available_providers=("codex",),
            explicit=True,
        )
    except RuntimeError as exc:
        assert "claude-code is not available" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
