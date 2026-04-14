from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_tools.codex_auth import (
    AuthState,
    MissingTokenError,
    UnsupportedAuthModeError,
    load_auth_state,
    persist_tokens,
)


def write_auth(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_auth_state_reads_chatgpt_tokens(tmp_path: Path) -> None:
    auth_path = tmp_path / ".codex" / "auth.json"
    write_auth(
        auth_path,
        {
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": "access-1",
                "refresh_token": "refresh-1",
                "account_id": "account-1",
                "id_token": "header.payload.sig",
            },
        },
    )

    state = load_auth_state(tmp_path / ".codex")
    assert state.tokens.access_token == "access-1"
    assert state.tokens.account_id == "account-1"


def test_load_auth_state_rejects_non_chatgpt_mode(tmp_path: Path) -> None:
    auth_path = tmp_path / ".codex" / "auth.json"
    write_auth(
        auth_path,
        {
            "auth_mode": "apikey",
            "tokens": {
                "access_token": "access-1",
                "refresh_token": "refresh-1",
                "account_id": "account-1",
            },
        },
    )

    with pytest.raises(UnsupportedAuthModeError):
        load_auth_state(tmp_path / ".codex")


def test_load_auth_state_requires_tokens(tmp_path: Path) -> None:
    auth_path = tmp_path / ".codex" / "auth.json"
    write_auth(auth_path, {"auth_mode": "chatgpt", "tokens": {}})

    with pytest.raises(MissingTokenError):
        load_auth_state(tmp_path / ".codex")


def test_persist_tokens_updates_auth_file(tmp_path: Path) -> None:
    auth_path = tmp_path / ".codex" / "auth.json"
    write_auth(
        auth_path,
        {
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "account_id": "account-1",
            },
        },
    )
    state: AuthState = load_auth_state(tmp_path / ".codex")

    refreshed = persist_tokens(
        state,
        access_token="new-access",
        refresh_token="new-refresh",
        id_token=None,
    )

    assert refreshed.tokens.access_token == "new-access"
    stored = json.loads(auth_path.read_text(encoding="utf-8"))
    assert stored["tokens"]["refresh_token"] == "new-refresh"
    assert isinstance(stored["last_refresh"], str)
