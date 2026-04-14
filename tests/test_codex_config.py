from __future__ import annotations

import json
from pathlib import Path

from agent_tools.codex_config import DEFAULT_CHATGPT_CODEX_BASE_URL, load_codex_defaults


def test_load_codex_defaults_reads_config_and_version(tmp_path: Path) -> None:
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        'model = "gpt-5"\nmodel_reasoning_effort = "medium"\n',
        encoding="utf-8",
    )
    (codex_home / "version.json").write_text(
        json.dumps({"latest_version": "0.120.0"}),
        encoding="utf-8",
    )

    defaults = load_codex_defaults(codex_home)
    assert defaults.model == "gpt-5"
    assert defaults.reasoning_effort == "medium"
    assert defaults.version == "0.120.0"
    assert defaults.base_url == DEFAULT_CHATGPT_CODEX_BASE_URL


def test_load_codex_defaults_falls_back_when_files_missing(tmp_path: Path) -> None:
    defaults = load_codex_defaults(tmp_path / ".codex")
    assert defaults.model is None
    assert defaults.reasoning_effort is None
    assert defaults.version == "0.0.0"
