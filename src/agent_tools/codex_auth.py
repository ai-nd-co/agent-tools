from __future__ import annotations

import base64
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from agent_tools.codex_config import build_user_agent, resolve_codex_home

REFRESH_TOKEN_URL = "https://auth.openai.com/oauth/token"
REFRESH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"


class AuthError(RuntimeError):
    pass


class UnsupportedAuthModeError(AuthError):
    pass


class MissingTokenError(AuthError):
    pass


class PermanentRefreshError(AuthError):
    pass


@dataclass(frozen=True)
class ChatgptTokens:
    access_token: str
    refresh_token: str
    account_id: str
    id_token: str | None


@dataclass(frozen=True)
class AuthState:
    path: Path
    auth_mode: str | None
    tokens: ChatgptTokens
    last_refresh: str | None
    raw: dict[str, Any]


def load_auth_state(codex_home: Path | None = None) -> AuthState:
    home = resolve_codex_home(codex_home)
    path = home / "auth.json"
    if not path.exists():
        raise AuthError(f"Codex auth file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    auth_mode = _coerce_str(raw.get("auth_mode"))
    if auth_mode != "chatgpt":
        raise UnsupportedAuthModeError(
            f"Expected ChatGPT Codex login in {path}, got auth_mode={auth_mode!r}"
        )
    tokens_raw = raw.get("tokens")
    if not isinstance(tokens_raw, dict):
        raise MissingTokenError(f"ChatGPT tokens missing in {path}")
    access_token = _required_str(tokens_raw.get("access_token"), "access_token", path)
    refresh_token = _required_str(tokens_raw.get("refresh_token"), "refresh_token", path)
    account_id = _coerce_str(tokens_raw.get("account_id")) or _extract_account_id(tokens_raw)
    if account_id is None:
        raise MissingTokenError(f"ChatGPT account_id missing in {path}")
    id_token = _coerce_str(tokens_raw.get("id_token"))
    return AuthState(
        path=path,
        auth_mode=auth_mode,
        tokens=ChatgptTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            account_id=account_id,
            id_token=id_token,
        ),
        last_refresh=_coerce_str(raw.get("last_refresh")),
        raw=raw,
    )


def reload_auth_state(previous: AuthState) -> tuple[AuthState, bool]:
    current = load_auth_state(previous.path.parent)
    changed = (
        current.tokens.access_token != previous.tokens.access_token
        or current.tokens.refresh_token != previous.tokens.refresh_token
        or current.tokens.account_id != previous.tokens.account_id
    )
    return current, changed


def refresh_chatgpt_tokens(
    auth_state: AuthState,
    *,
    originator: str,
    version: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None = None,
) -> AuthState:
    headers = {
        "content-type": "application/json",
        "originator": originator,
        "user-agent": build_user_agent(originator, version),
    }
    payload = {
        "client_id": REFRESH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": auth_state.tokens.refresh_token,
    }
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout, transport=transport, follow_redirects=True) as client:
        response = client.post(REFRESH_TOKEN_URL, headers=headers, json=payload)
    if response.status_code == 401:
        raise PermanentRefreshError(_classify_refresh_failure(response.text))
    response.raise_for_status()
    data = response.json()
    access_token = _coerce_str(data.get("access_token")) or auth_state.tokens.access_token
    refresh_token = _coerce_str(data.get("refresh_token")) or auth_state.tokens.refresh_token
    id_token = _coerce_str(data.get("id_token")) or auth_state.tokens.id_token
    return persist_tokens(
        auth_state,
        access_token=access_token,
        refresh_token=refresh_token,
        id_token=id_token,
    )


def persist_tokens(
    auth_state: AuthState,
    *,
    access_token: str,
    refresh_token: str,
    id_token: str | None,
) -> AuthState:
    payload = dict(auth_state.raw)
    tokens = dict(payload.get("tokens", {}))
    tokens["access_token"] = access_token
    tokens["refresh_token"] = refresh_token
    tokens["account_id"] = auth_state.tokens.account_id
    if id_token is not None:
        tokens["id_token"] = id_token
    payload["tokens"] = tokens
    payload["last_refresh"] = datetime.now(tz=UTC).isoformat()
    _atomic_write_json(auth_state.path, payload)
    return load_auth_state(auth_state.path.parent)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _classify_refresh_failure(body: str) -> str:
    code = _extract_error_code(body)
    if code == "refresh_token_expired":
        return "Codex ChatGPT refresh token expired."
    if code == "refresh_token_reused":
        return "Codex ChatGPT refresh token exhausted."
    if code == "refresh_token_invalidated":
        return "Codex ChatGPT refresh token revoked."
    return "Codex ChatGPT token refresh failed permanently."


def _extract_error_code(body: str) -> str | None:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            if isinstance(code, str):
                return code
        if isinstance(error, str):
            return error
        code = parsed.get("code")
        if isinstance(code, str):
            return code
    return None


def _extract_account_id(tokens_raw: dict[str, Any]) -> str | None:
    for field in ("id_token", "access_token"):
        token = _coerce_str(tokens_raw.get(field))
        if not token:
            continue
        account_id = _extract_account_id_from_jwt(token)
        if account_id:
            return account_id
    return None


def _extract_account_id_from_jwt(jwt: str) -> str | None:
    parts = jwt.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload + padding)
        data = json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    auth_claims = data.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id.strip():
            return account_id.strip()
    return None


def _required_str(value: object, field_name: str, path: Path) -> str:
    text = _coerce_str(value)
    if text is None:
        raise MissingTokenError(f"Missing {field_name} in {path}")
    return text


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None
