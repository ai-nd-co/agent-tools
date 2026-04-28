from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

from agent_tools.worklog import (
    RepoEvidence,
    RepoIdentity,
    RepoResolver,
    WorkTurn,
    generate_worklog_report,
    iter_work_turns,
    merge_turns_into_intervals,
    summarize_weeks,
)


def test_merge_turns_into_intervals_merges_overlapping_blocks() -> None:
    repo = RepoIdentity(key="org/pdd-test", display_name="pdd-test")
    first = WorkTurn(
        turn_id="t1",
        source="codex",
        session_id="s1",
        prompt_timestamp=datetime(2026, 4, 21, 23, 58, tzinfo=UTC),
        nominal_end=datetime(2026, 4, 22, 0, 8, tzinfo=UTC),
        observed_end=datetime(2026, 4, 22, 0, 1, tzinfo=UTC),
        observed_turn_seconds=180.0,
        credited_seconds=600.0,
        prompt_preview="first",
        repo_evidence=(RepoEvidence(repo, 50, "session"),),
    )
    second = WorkTurn(
        turn_id="t2",
        source="claude",
        session_id="s2",
        prompt_timestamp=datetime(2026, 4, 22, 0, 5, tzinfo=UTC),
        nominal_end=datetime(2026, 4, 22, 0, 15, tzinfo=UTC),
        observed_end=datetime(2026, 4, 22, 0, 7, tzinfo=UTC),
        observed_turn_seconds=120.0,
        credited_seconds=600.0,
        prompt_preview="second",
        repo_evidence=(RepoEvidence(repo, 60, "tool"),),
    )

    intervals = merge_turns_into_intervals([first, second])

    assert len(intervals) == 1
    assert intervals[0].credited_seconds == 17 * 60
    assert intervals[0].repo.display_name == "pdd-test"

    weeks = summarize_weeks(intervals, timezone_name="Asia/Tashkent")
    assert len(weeks) == 1
    assert weeks[0].week_start == date(2026, 4, 20)


def test_iter_work_turns_parses_codex_and_attributes_subrepo(tmp_path: Path) -> None:
    parent = _init_git_repo(tmp_path / "workspace", "git@github.com:ai-nd-co/ai-nd-co.git")
    _init_git_repo(parent / "repos" / "pdd-test", "git@github.com:ai-nd-co/pdd-test.git")
    codex_home = tmp_path / ".codex"
    session_dir = codex_home / "sessions" / "2026" / "04" / "23"
    session_dir.mkdir(parents=True)
    session_path = session_dir / "rollout-test.jsonl"
    rows = [
        {
            "timestamp": "2026-04-23T00:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": "sess-1",
                "cwd": str(parent),
                "git": {"repository_url": "git@github.com:ai-nd-co/ai-nd-co.git"},
            },
        },
        {
            "timestamp": "2026-04-23T00:00:00Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "inspect repo"},
        },
        {
            "timestamp": "2026-04-23T00:01:00Z",
            "type": "event_msg",
            "payload": {
                "type": "exec_command_end",
                "cwd": str(parent),
                "command": "cd repos/pdd-test && git status",
            },
        },
        {
            "timestamp": "2026-04-23T00:04:00Z",
            "type": "event_msg",
            "payload": {"type": "task_complete"},
        },
    ]
    session_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    turns = iter_work_turns(
        resolver=RepoResolver(),
        warnings=[],
        codex_home=codex_home,
        claude_home=tmp_path / ".claude",
    )

    assert len(turns) == 1
    assert turns[0].observed_turn_seconds == 240.0
    assert any(evidence.repo.display_name == "pdd-test" for evidence in turns[0].repo_evidence)


def test_iter_work_turns_parses_claude_and_ignores_tool_result_user_rows(tmp_path: Path) -> None:
    parent = _init_git_repo(tmp_path / "workspace", "git@github.com:ai-nd-co/ai-nd-co.git")
    child = _init_git_repo(parent / "repos" / "pdd-test", "git@github.com:ai-nd-co/pdd-test.git")
    claude_home = tmp_path / ".claude"
    project_dir = claude_home / "projects" / "C--projects-ai-nd-co"
    project_dir.mkdir(parents=True)
    session_path = project_dir / "session-1.jsonl"
    rows = [
        {
            "timestamp": "2026-04-23T00:00:00Z",
            "type": "user",
            "userType": "external",
            "cwd": str(parent),
            "sessionId": "session-1",
            "message": {"role": "user", "content": "please inspect pdd-test"},
        },
        {
            "timestamp": "2026-04-23T00:00:05Z",
            "type": "assistant",
            "cwd": str(parent),
            "sessionId": "session-1",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "input": {"prompt": "look in repos/pdd-test"}}],
            },
        },
        {
            "timestamp": "2026-04-23T00:00:07Z",
            "type": "user",
            "userType": "external",
            "cwd": str(child),
            "sessionId": "session-1",
            "message": {
                "role": "user",
                "content": json.dumps(
                    {"tool_use_id": "toolu_1", "type": "tool_result", "content": "ok"}
                ),
            },
        },
        {
            "timestamp": "2026-04-23T00:02:00Z",
            "type": "assistant",
            "cwd": str(child),
            "sessionId": "session-1",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        },
    ]
    session_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    turns = iter_work_turns(
        resolver=RepoResolver(),
        warnings=[],
        codex_home=tmp_path / ".codex",
        claude_home=claude_home,
    )

    assert len(turns) == 1
    assert turns[0].observed_turn_seconds == 120.0
    assert any(evidence.repo.display_name == "pdd-test" for evidence in turns[0].repo_evidence)


def test_claude_counts_human_prompt_that_starts_with_json(tmp_path: Path) -> None:
    parent = _init_git_repo(tmp_path / "workspace", "git@github.com:ai-nd-co/ai-nd-co.git")
    claude_home = tmp_path / ".claude"
    project_dir = claude_home / "projects" / "C--projects-ai-nd-co"
    project_dir.mkdir(parents=True)
    session_path = project_dir / "session-2.jsonl"
    rows = [
        {
            "timestamp": "2026-04-23T00:00:00Z",
            "type": "user",
            "userType": "external",
            "cwd": str(parent),
            "sessionId": "session-2",
            "message": {"role": "user", "content": '{"foo":1, "bar":2}'},
        },
        {
            "timestamp": "2026-04-23T00:00:02Z",
            "type": "assistant",
            "cwd": str(parent),
            "sessionId": "session-2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        },
    ]
    session_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    turns = iter_work_turns(
        resolver=RepoResolver(),
        warnings=[],
        codex_home=tmp_path / ".codex",
        claude_home=claude_home,
    )

    assert len(turns) == 1
    assert turns[0].prompt_preview.startswith('{"foo":1')


def test_generate_worklog_report_clips_to_tashkent_local_date(tmp_path: Path) -> None:
    codex_home = tmp_path / ".codex"
    session_dir = codex_home / "sessions" / "2026" / "04" / "23"
    session_dir.mkdir(parents=True)
    session_path = session_dir / "rollout-test.jsonl"
    rows = [
        {
            "timestamp": "2026-04-20T18:59:00Z",
            "type": "session_meta",
            "payload": {"id": "sess-1", "cwd": str(tmp_path)},
        },
        {
            "timestamp": "2026-04-20T18:59:00Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "one"},
        },
        {
            "timestamp": "2026-04-20T19:11:00Z",
            "type": "event_msg",
            "payload": {"type": "task_complete"},
        },
    ]
    session_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = generate_worklog_report(
        since=date(2026, 4, 21),
        until=date(2026, 4, 21),
        codex_home=codex_home,
        claude_home=tmp_path / ".claude",
    )

    assert report.turn_count == 1
    assert report.total_hours == 9 / 60
    assert report.turns[0].credited_seconds == 9 * 60


def test_iter_work_turns_salvages_good_rows_from_broken_jsonl(tmp_path: Path) -> None:
    codex_home = tmp_path / ".codex"
    session_dir = codex_home / "sessions" / "2026" / "04" / "23"
    session_dir.mkdir(parents=True)
    session_path = session_dir / "rollout-bad.jsonl"
    lines = [
        json.dumps(
            {
                "timestamp": "2026-04-23T00:00:00Z",
                "type": "session_meta",
                "payload": {"id": "sess-1", "cwd": str(tmp_path)},
            }
        ),
        '{"timestamp": "2026-04-23T00:00:01Z", "type": ',
        json.dumps(
            {
                "timestamp": "2026-04-23T00:00:02Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "hello"},
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-04-23T00:00:04Z",
                "type": "event_msg",
                "payload": {"type": "task_complete"},
            }
        ),
    ]
    session_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    warnings: list[str] = []

    turns = iter_work_turns(
        resolver=RepoResolver(),
        warnings=warnings,
        codex_home=codex_home,
        claude_home=tmp_path / ".claude",
    )

    assert len(turns) == 1
    assert warnings


def _init_git_repo(path: Path, remote_url: str) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "remote", "add", "origin", remote_url], check=True)
    (path / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True)
    return path
