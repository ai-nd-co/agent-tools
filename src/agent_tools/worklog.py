from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

WORKLOG_TIMEZONE = "Asia/Tashkent"
DEFAULT_ACTIVE_BLOCK = timedelta(minutes=10)
MULTI_UNKNOWN_REPO_KEY = "multi/unknown"
PATH_HINT_RE = re.compile(r"[A-Za-z0-9._-]+(?:[\\/][A-Za-z0-9._-]+)+")
WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")
COMMAND_PATH_RE = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:cd|git\s+-C)\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s;&|]+))"
)


@dataclass(frozen=True)
class RepoIdentity:
    key: str
    display_name: str
    git_root: str | None = None
    repository_url: str | None = None


@dataclass(frozen=True)
class RepoEvidence:
    repo: RepoIdentity
    weight: int
    reason: str


@dataclass(frozen=True)
class WorkTurn:
    turn_id: str
    source: str
    session_id: str
    prompt_timestamp: datetime
    nominal_end: datetime
    observed_end: datetime | None
    observed_turn_seconds: float | None
    credited_seconds: float
    prompt_preview: str
    repo_evidence: tuple[RepoEvidence, ...]


@dataclass(frozen=True)
class WorkInterval:
    interval_id: str
    start: datetime
    end: datetime
    credited_seconds: float
    turn_ids: tuple[str, ...]
    turn_count: int
    sources: tuple[str, ...]
    repo: RepoIdentity
    ambiguous_repo: bool


@dataclass(frozen=True)
class WeeklyHours:
    week_start: date
    week_end: date
    seconds: float

    @property
    def hours(self) -> float:
        return self.seconds / 3600.0


@dataclass(frozen=True)
class RepoHours:
    repo_key: str
    repo_name: str
    seconds: float
    interval_count: int
    turn_count: int

    @property
    def hours(self) -> float:
        return self.seconds / 3600.0


@dataclass(frozen=True)
class WorklogReport:
    time_zone: str
    generated_at: datetime
    since: date | None
    until: date | None
    total_hours: float
    turn_count: int
    merged_interval_count: int
    weeks: tuple[WeeklyHours, ...]
    repos: tuple[RepoHours, ...]
    intervals: tuple[WorkInterval, ...]
    turns: tuple[WorkTurn, ...]
    warnings: tuple[str, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "summary": {
                "time_zone": self.time_zone,
                "generated_at": self.generated_at.isoformat(),
                "since": self.since.isoformat() if self.since is not None else None,
                "until": self.until.isoformat() if self.until is not None else None,
                "total_hours": round(self.total_hours, 4),
                "turn_count": self.turn_count,
                "merged_interval_count": self.merged_interval_count,
                "warning_count": len(self.warnings),
            },
            "weeks": [
                {
                    "week_start": item.week_start.isoformat(),
                    "week_end": item.week_end.isoformat(),
                    "seconds": round(item.seconds, 3),
                    "hours": round(item.hours, 4),
                }
                for item in self.weeks
            ],
            "repos": [
                {
                    "repo_key": item.repo_key,
                    "repo_name": item.repo_name,
                    "seconds": round(item.seconds, 3),
                    "hours": round(item.hours, 4),
                    "interval_count": item.interval_count,
                    "turn_count": item.turn_count,
                }
                for item in self.repos
            ],
            "intervals": [
                {
                    "interval_id": item.interval_id,
                    "start": item.start.isoformat(),
                    "end": item.end.isoformat(),
                    "credited_seconds": round(item.credited_seconds, 3),
                    "credited_hours": round(item.credited_seconds / 3600.0, 4),
                    "turn_ids": list(item.turn_ids),
                    "turn_count": item.turn_count,
                    "sources": list(item.sources),
                    "repo_key": item.repo.key,
                    "repo_name": item.repo.display_name,
                    "ambiguous_repo": item.ambiguous_repo,
                }
                for item in self.intervals
            ],
            "turns": [
                {
                    "turn_id": item.turn_id,
                    "source": item.source,
                    "session_id": item.session_id,
                    "prompt_timestamp": item.prompt_timestamp.isoformat(),
                    "nominal_end": item.nominal_end.isoformat(),
                    "credited_seconds": round(item.credited_seconds, 3),
                    "prompt_preview": item.prompt_preview,
                    "repo_evidence": [
                        {
                            "repo_key": evidence.repo.key,
                            "repo_name": evidence.repo.display_name,
                            "weight": evidence.weight,
                            "reason": evidence.reason,
                        }
                        for evidence in item.repo_evidence
                    ],
                }
                for item in self.turns
            ],
            "warnings": list(self.warnings),
        }


@dataclass
class _TurnBuilder:
    turn_id: str
    source: str
    session_id: str
    prompt_timestamp: datetime
    prompt_preview: str
    observed_end: datetime | None = None
    repo_evidence: list[RepoEvidence] = field(default_factory=list)

    def update_observed_end(self, timestamp: datetime) -> None:
        if self.observed_end is None or timestamp > self.observed_end:
            self.observed_end = timestamp

    def add_evidence(self, evidence: RepoEvidence | None) -> None:
        if evidence is not None:
            self.repo_evidence.append(evidence)

    def add_evidences(self, evidences: list[RepoEvidence]) -> None:
        self.repo_evidence.extend(evidences)

    def build(self) -> WorkTurn:
        observed_turn_seconds: float | None = None
        if self.observed_end is not None and self.observed_end >= self.prompt_timestamp:
            observed_turn_seconds = (self.observed_end - self.prompt_timestamp).total_seconds()
        return WorkTurn(
            turn_id=self.turn_id,
            source=self.source,
            session_id=self.session_id,
            prompt_timestamp=self.prompt_timestamp,
            nominal_end=self.prompt_timestamp + DEFAULT_ACTIVE_BLOCK,
            observed_end=self.observed_end,
            observed_turn_seconds=observed_turn_seconds,
            credited_seconds=DEFAULT_ACTIVE_BLOCK.total_seconds(),
            prompt_preview=self.prompt_preview,
            repo_evidence=tuple(self.repo_evidence),
        )


class RepoResolver:
    def __init__(self) -> None:
        self._path_cache: dict[str, RepoIdentity | None] = {}
        self._repo_cache: dict[str, RepoIdentity] = {}

    def repo_from_path(
        self, raw_path: str | None, *, base_dir: str | None = None
    ) -> RepoIdentity | None:
        path = _coerce_existing_path(raw_path, base_dir=base_dir)
        if path is None:
            return None
        cache_key = str(path)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        git_root = self._git_output(path, "rev-parse", "--show-toplevel")
        if git_root is None:
            self._path_cache[cache_key] = None
            return None
        repository_url = self._git_output(
            Path(git_root), "config", "--get", "remote.origin.url"
        )
        canonical_key = _canonical_repo_key(repository_url) or _normalize_display_path(git_root)
        cached = self._repo_cache.get(canonical_key)
        if cached is not None:
            self._path_cache[cache_key] = cached
            return cached
        repo = RepoIdentity(
            key=canonical_key,
            display_name=_display_name_from_repo(repository_url, git_root),
            git_root=_normalize_display_path(git_root),
            repository_url=repository_url,
        )
        self._repo_cache[canonical_key] = repo
        self._path_cache[cache_key] = repo
        return repo

    def repo_from_session_git(
        self, repository_url: str | None, cwd: str | None
    ) -> RepoIdentity | None:
        path_repo = self.repo_from_path(cwd)
        if path_repo is not None:
            return path_repo
        if repository_url is None:
            return None
        canonical_key = _canonical_repo_key(repository_url) or repository_url
        cached = self._repo_cache.get(canonical_key)
        if cached is not None:
            return cached
        repo = RepoIdentity(
            key=canonical_key,
            display_name=_display_name_from_repo(repository_url, cwd),
            git_root=_normalize_display_path(cwd) if cwd is not None else None,
            repository_url=repository_url,
        )
        self._repo_cache[canonical_key] = repo
        return repo

    def _git_output(self, path: Path, *args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), *args],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return value or None


def generate_worklog_report(
    *,
    since: date | None = None,
    until: date | None = None,
    codex_home: Path | None = None,
    claude_home: Path | None = None,
    timezone_name: str = WORKLOG_TIMEZONE,
) -> WorklogReport:
    resolver = RepoResolver()
    warnings: list[str] = []
    turns = iter_work_turns(
        resolver=resolver,
        warnings=warnings,
        codex_home=codex_home,
        claude_home=claude_home,
    )
    turns = _clip_turns_to_window(
        turns, since=since, until=until, timezone_name=timezone_name
    )
    intervals = merge_turns_into_intervals(turns)
    weeks = summarize_weeks(intervals, timezone_name=timezone_name)
    repos = summarize_repos(intervals, turns)
    total_seconds = sum(interval.credited_seconds for interval in intervals)
    return WorklogReport(
        time_zone=timezone_name,
        generated_at=datetime.now(tz=UTC),
        since=since,
        until=until,
        total_hours=total_seconds / 3600.0,
        turn_count=len(turns),
        merged_interval_count=len(intervals),
        weeks=tuple(weeks),
        repos=tuple(repos),
        intervals=tuple(intervals),
        turns=tuple(turns),
        warnings=tuple(warnings),
    )


def iter_work_turns(
    *,
    resolver: RepoResolver,
    warnings: list[str],
    codex_home: Path | None = None,
    claude_home: Path | None = None,
) -> list[WorkTurn]:
    codex_root = codex_home or (Path.home() / ".codex")
    claude_root = claude_home or (Path.home() / ".claude")
    turns: list[WorkTurn] = []
    for path in sorted((codex_root / "sessions").rglob("*.jsonl")):
        turns.extend(_parse_codex_session(path, resolver=resolver, warnings=warnings))
    for path in sorted((claude_root / "projects").rglob("*.jsonl")):
        path_text = path.as_posix()
        if "/subagents/" in path_text or ".backup-" in path.name or "sync-conflict" in path.name:
            continue
        turns.extend(_parse_claude_session(path, resolver=resolver, warnings=warnings))
    turns.sort(key=lambda item: item.prompt_timestamp)
    return turns


def merge_turns_into_intervals(turns: list[WorkTurn]) -> list[WorkInterval]:
    if not turns:
        return []
    intervals: list[WorkInterval] = []
    current_turns = [turns[0]]
    current_start = turns[0].prompt_timestamp
    current_end = turns[0].nominal_end
    for turn in turns[1:]:
        if turn.prompt_timestamp <= current_end:
            current_turns.append(turn)
            if turn.nominal_end > current_end:
                current_end = turn.nominal_end
            continue
        intervals.append(
            _build_interval(len(intervals) + 1, current_start, current_end, current_turns)
        )
        current_turns = [turn]
        current_start = turn.prompt_timestamp
        current_end = turn.nominal_end
    intervals.append(_build_interval(len(intervals) + 1, current_start, current_end, current_turns))
    return intervals


def summarize_weeks(intervals: list[WorkInterval], *, timezone_name: str) -> list[WeeklyHours]:
    timezone = ZoneInfo(timezone_name)
    totals: dict[date, float] = defaultdict(float)
    for interval in intervals:
        for week_start, seconds in _split_interval_by_week(interval.start, interval.end, timezone):
            totals[week_start] += seconds
    return [
        WeeklyHours(
            week_start=week_start,
            week_end=week_start + timedelta(days=6),
            seconds=totals[week_start],
        )
        for week_start in sorted(totals)
    ]


def summarize_repos(intervals: list[WorkInterval], turns: list[WorkTurn]) -> list[RepoHours]:
    turn_by_id = {turn.turn_id: turn for turn in turns}
    seconds_by_repo: dict[str, float] = defaultdict(float)
    interval_count_by_repo: dict[str, int] = defaultdict(int)
    turn_count_by_repo: dict[str, int] = defaultdict(int)
    name_by_repo: dict[str, str] = {}
    for interval in intervals:
        repo_key = interval.repo.key
        seconds_by_repo[repo_key] += interval.credited_seconds
        interval_count_by_repo[repo_key] += 1
        turn_count_by_repo[repo_key] += sum(
            1 for turn_id in interval.turn_ids if turn_id in turn_by_id
        )
        name_by_repo[repo_key] = interval.repo.display_name
    items = [
        RepoHours(
            repo_key=repo_key,
            repo_name=name_by_repo[repo_key],
            seconds=seconds_by_repo[repo_key],
            interval_count=interval_count_by_repo[repo_key],
            turn_count=turn_count_by_repo[repo_key],
        )
        for repo_key in seconds_by_repo
    ]
    items.sort(key=lambda item: (-item.seconds, item.repo_name.lower()))
    return items


def render_worklog_table(report: WorklogReport) -> str:
    lines = ["Worklog"]
    if len(report.weeks) == 1:
        week = report.weeks[0]
        lines.append(
            "  Week: "
            f"{week.week_start.isoformat()} to {week.week_end.isoformat()} | "
            f"Hours: {report.total_hours:.2f}"
        )
    else:
        lines.append(f"  Hours: {report.total_hours:.2f}")
    lines.append(
        "  "
        f"Intervals: {report.merged_interval_count} | "
        f"Turns: {report.turn_count} | "
        f"Time zone: {report.time_zone}"
    )
    if report.since is not None or report.until is not None:
        lines.append(
            "  Date filter: "
            f"{report.since.isoformat() if report.since is not None else '...'}"
            f" to {report.until.isoformat() if report.until is not None else '...'}"
        )
    if report.weeks:
        lines.extend(
            [
                "",
                "Weekly hours",
                _render_table(
                    ("Week start", "Week end", "Hours"),
                    [
                        (
                            item.week_start.isoformat(),
                            item.week_end.isoformat(),
                            f"{item.hours:.2f}",
                        )
                        for item in report.weeks
                    ],
                ),
            ]
        )
    if report.repos:
        total_seconds = sum(item.seconds for item in report.repos)
        lines.extend(
            [
                "",
                "Repo hours",
                _render_table(
                    ("Repo", "Hours", "Share", "Intervals"),
                    [
                        (
                            item.repo_name,
                            f"{item.hours:.2f}",
                            _format_share(item.seconds, total_seconds),
                            str(item.interval_count),
                        )
                        for item in report.repos
                    ],
                ),
            ]
        )
    if report.warnings:
        lines.extend(["", "Warnings"])
        lines.extend(f"  - {warning}" for warning in report.warnings)
    return "\n".join(lines) + "\n"


def _parse_codex_session(
    path: Path, *, resolver: RepoResolver, warnings: list[str]
) -> list[WorkTurn]:
    rows = _load_jsonl_rows(path, warnings, "Codex")
    if not rows:
        return []
    meta_payload = _find_codex_session_meta(rows)
    session_id = str(meta_payload.get("id") or path.stem)
    session_cwd = _string_or_none(meta_payload.get("cwd"))
    session_repo = resolver.repo_from_session_git(
        _string_or_none((meta_payload.get("git") or {}).get("repository_url")),
        session_cwd,
    )
    turns: list[WorkTurn] = []
    current: _TurnBuilder | None = None
    turn_index = 0
    for row in rows:
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is None:
            continue
        row_type = row.get("type")
        payload = row.get("payload") or {}
        if row_type == "event_msg" and payload.get("type") == "user_message":
            if current is not None:
                turns.append(current.build())
            turn_index += 1
            current = _TurnBuilder(
                turn_id=f"codex:{session_id}:{turn_index}",
                source="codex",
                session_id=session_id,
                prompt_timestamp=timestamp,
                prompt_preview=_limit_text(_string_or_none(payload.get("message")) or "", 140),
            )
            if session_repo is not None:
                current.add_evidence(RepoEvidence(session_repo, 10, "codex_session_cwd"))
            continue
        if current is None:
            continue
        if row_type == "event_msg":
            payload_type = _string_or_none(payload.get("type"))
            if payload_type in {"agent_message", "task_complete"}:
                current.update_observed_end(timestamp)
            current.add_evidences(_repo_evidence_from_codex_event(payload, resolver, session_cwd))
            continue
        if row_type == "response_item":
            role = _string_or_none(payload.get("role"))
            payload_type = _string_or_none(payload.get("type"))
            if role == "assistant":
                current.update_observed_end(timestamp)
            if payload_type in {"function_call", "custom_tool_call"}:
                current.add_evidences(
                    _repo_evidence_from_codex_tool(payload, resolver, session_cwd)
                )
    if current is not None:
        turns.append(current.build())
    return turns


def _parse_claude_session(
    path: Path, *, resolver: RepoResolver, warnings: list[str]
) -> list[WorkTurn]:
    rows = _load_jsonl_rows(path, warnings, "Claude")
    if not rows:
        return []
    turns: list[WorkTurn] = []
    current: _TurnBuilder | None = None
    session_id = path.stem
    turn_index = 0
    for row in rows:
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is None:
            continue
        row_type = _string_or_none(row.get("type"))
        row_cwd = _string_or_none(row.get("cwd"))
        row_repo = resolver.repo_from_path(row_cwd)
        row_text = _extract_claude_message_text(row)
        if row_type == "user" and _is_human_claude_prompt(row, row_text):
            if current is not None:
                turns.append(current.build())
            turn_index += 1
            current = _TurnBuilder(
                turn_id=f"claude:{session_id}:{turn_index}",
                source="claude",
                session_id=_string_or_none(row.get("sessionId")) or session_id,
                prompt_timestamp=timestamp,
                prompt_preview=_limit_text(row_text, 140),
            )
            if row_repo is not None:
                current.add_evidence(RepoEvidence(row_repo, 10, "claude_session_cwd"))
            continue
        if current is None:
            continue
        if row_type == "assistant":
            current.update_observed_end(timestamp)
            current.add_evidences(_repo_evidence_from_claude_assistant(row, resolver, row_cwd))
            continue
        if row_type == "user" and _is_claude_tool_result_row(row_text):
            current.update_observed_end(timestamp)
        if row_repo is not None:
            current.add_evidence(RepoEvidence(row_repo, 50, "claude_record_cwd"))
        hint = _repo_hint_evidence(row_text, resolver, row_cwd, 45, "claude_message_hint")
        if hint is not None:
            current.add_evidence(hint)
    if current is not None:
        turns.append(current.build())
    return turns


def _repo_evidence_from_codex_event(
    payload: dict[str, Any],
    resolver: RepoResolver,
    session_cwd: str | None,
) -> list[RepoEvidence]:
    evidences: list[RepoEvidence] = []
    event_cwd = _string_or_none(payload.get("cwd"))
    base_dir = event_cwd or session_cwd
    if payload.get("type") == "patch_apply_end":
        changes = payload.get("changes")
        if isinstance(changes, dict):
            for raw_path in changes.keys():
                repo = resolver.repo_from_path(str(raw_path), base_dir=base_dir)
                if repo is not None:
                    evidences.append(RepoEvidence(repo, 100, "codex_patch_changes"))
    if payload.get("type") == "exec_command_end":
        for raw_path in _paths_from_command(payload.get("command")):
            repo = resolver.repo_from_path(raw_path, base_dir=base_dir)
            if repo is not None:
                evidences.append(RepoEvidence(repo, 60, "codex_exec_command"))
    event_repo = resolver.repo_from_path(event_cwd, base_dir=session_cwd)
    if event_repo is not None:
        evidences.append(RepoEvidence(event_repo, 35, "codex_event_cwd"))
    return evidences


def _repo_evidence_from_codex_tool(
    payload: dict[str, Any],
    resolver: RepoResolver,
    session_cwd: str | None,
) -> list[RepoEvidence]:
    evidences: list[RepoEvidence] = []
    raw_text = _string_or_none(payload.get("arguments")) or _string_or_none(payload.get("input"))
    if raw_text is None:
        return evidences
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        base_dir = _string_or_none(parsed.get("workdir")) or session_cwd
        command = _string_or_none(parsed.get("command"))
        if command is not None:
            for raw_path in _paths_from_command(command):
                repo = resolver.repo_from_path(raw_path, base_dir=base_dir)
                if repo is not None:
                    evidences.append(RepoEvidence(repo, 60, "codex_shell_command"))
        for key in ("workdir", "cwd", "path"):
            repo = resolver.repo_from_path(_string_or_none(parsed.get(key)), base_dir=session_cwd)
            if repo is not None:
                evidences.append(RepoEvidence(repo, 55, f"codex_tool_{key}"))
        for value in parsed.values():
            if isinstance(value, str):
                evidence = _repo_hint_evidence(value, resolver, session_cwd, 45, "codex_tool_hint")
                if evidence is not None:
                    evidences.append(evidence)
        return evidences
    hint = _repo_hint_evidence(raw_text, resolver, session_cwd, 40, "codex_tool_text")
    if hint is not None:
        evidences.append(hint)
    return evidences


def _repo_evidence_from_claude_assistant(
    row: dict[str, Any],
    resolver: RepoResolver,
    row_cwd: str | None,
) -> list[RepoEvidence]:
    evidences: list[RepoEvidence] = []
    message = row.get("message")
    if not isinstance(message, dict):
        return evidences
    content = message.get("content")
    if not isinstance(content, list):
        return evidences
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool_use":
            continue
        input_payload = item.get("input")
        if not isinstance(input_payload, dict):
            continue
        for key in ("cwd", "workdir", "path"):
            repo = resolver.repo_from_path(
                _string_or_none(input_payload.get(key)), base_dir=row_cwd
            )
            if repo is not None:
                evidences.append(RepoEvidence(repo, 55, f"claude_tool_{key}"))
        for value in input_payload.values():
            if isinstance(value, str):
                evidence = _repo_hint_evidence(value, resolver, row_cwd, 45, "claude_tool_hint")
                if evidence is not None:
                    evidences.append(evidence)
    return evidences


def _repo_hint_evidence(
    text: str | None,
    resolver: RepoResolver,
    base_dir: str | None,
    weight: int,
    reason: str,
) -> RepoEvidence | None:
    if text is None:
        return None
    for raw_path in _extract_path_hints(text):
        repo = resolver.repo_from_path(raw_path, base_dir=base_dir)
        if repo is not None:
            return RepoEvidence(repo, weight, reason)
    return None


def _paths_from_command(command: Any) -> list[str]:
    if isinstance(command, list):
        text = " ".join(str(part) for part in command)
    elif isinstance(command, str):
        text = command
    else:
        return []
    results: list[str] = []
    for match in COMMAND_PATH_RE.finditer(text):
        raw = match.group(1) or match.group(2) or match.group(3)
        if raw:
            results.append(raw.strip())
    return results


def _extract_path_hints(text: str) -> list[str]:
    normalized = text.replace("\\\\", "/")
    results: list[str] = []
    for match in PATH_HINT_RE.finditer(normalized):
        candidate = match.group(0)
        if "://" in candidate:
            continue
        results.append(candidate)
    return results


def _is_human_claude_prompt(row: dict[str, Any], text: str) -> bool:
    if _string_or_none(row.get("userType")) != "external":
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("<command-name>"):
        return True
    if stripped.startswith("<local-command-stdout>") or stripped.startswith("<task-notification>"):
        return False
    if _is_claude_tool_result_row(stripped):
        return False
    return True


def _is_claude_tool_result_row(text: str) -> bool:
    if not text.startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return bool(payload.get("type") == "tool_result" or "tool_use_id" in payload)


def _extract_claude_message_text(row: dict[str, Any]) -> str:
    message = row.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return ""


def _build_interval(
    index: int, start: datetime, end: datetime, turns: list[WorkTurn]
) -> WorkInterval:
    repo, ambiguous = _choose_primary_repo(turns)
    return WorkInterval(
        interval_id=f"interval-{index}",
        start=start,
        end=end,
        credited_seconds=(end - start).total_seconds(),
        turn_ids=tuple(turn.turn_id for turn in turns),
        turn_count=len(turns),
        sources=tuple(sorted({turn.source for turn in turns})),
        repo=repo,
        ambiguous_repo=ambiguous,
    )


def _choose_primary_repo(turns: list[WorkTurn]) -> tuple[RepoIdentity, bool]:
    scores: dict[str, int] = defaultdict(int)
    repo_by_key: dict[str, RepoIdentity] = {}
    for turn in turns:
        for evidence in turn.repo_evidence:
            scores[evidence.repo.key] += evidence.weight
            repo_by_key[evidence.repo.key] = evidence.repo
    if not scores:
        return RepoIdentity(MULTI_UNKNOWN_REPO_KEY, MULTI_UNKNOWN_REPO_KEY), True
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_key, top_score = ordered[0]
    if len(ordered) > 1 and ordered[1][1] == top_score:
        return RepoIdentity(MULTI_UNKNOWN_REPO_KEY, MULTI_UNKNOWN_REPO_KEY), True
    return repo_by_key[top_key], False


def _split_interval_by_week(
    start: datetime, end: datetime, timezone: ZoneInfo
) -> list[tuple[date, float]]:
    local_start = start.astimezone(timezone)
    local_end = end.astimezone(timezone)
    results: list[tuple[date, float]] = []
    cursor = local_start
    while cursor < local_end:
        week_start = cursor.date() - timedelta(days=cursor.weekday())
        next_week = datetime.combine(
            week_start + timedelta(days=7), time.min, tzinfo=timezone
        )
        segment_end = min(local_end, next_week)
        results.append((week_start, (segment_end - cursor).total_seconds()))
        cursor = segment_end
    return results


def _clip_turns_to_window(
    turns: list[WorkTurn], *, since: date | None, until: date | None, timezone_name: str
) -> list[WorkTurn]:
    if since is None and until is None:
        return turns
    timezone = ZoneInfo(timezone_name)
    window_start = (
        datetime.combine(since, time.min, tzinfo=timezone).astimezone(UTC)
        if since is not None
        else None
    )
    window_end = (
        datetime.combine(until + timedelta(days=1), time.min, tzinfo=timezone).astimezone(UTC)
        if until is not None
        else None
    )
    clipped: list[WorkTurn] = []
    for turn in turns:
        start = turn.prompt_timestamp
        end = turn.nominal_end
        if window_start is not None and end <= window_start:
            continue
        if window_end is not None and start >= window_end:
            continue
        clipped_start = max(start, window_start) if window_start is not None else start
        clipped_end = min(end, window_end) if window_end is not None else end
        if clipped_end <= clipped_start:
            continue
        clipped.append(
            replace(
                turn,
                prompt_timestamp=clipped_start,
                nominal_end=clipped_end,
                credited_seconds=(clipped_end - clipped_start).total_seconds(),
            )
        )
    return clipped


def _load_jsonl_rows(path: Path, warnings: list[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for lineno, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.append(f"Failed to parse {label} session {path}:{lineno}: {exc}")
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except (OSError, UnicodeDecodeError) as exc:
        warnings.append(f"Failed to read {label} session {path}: {exc}")
    return rows


def _find_codex_session_meta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("type") == "session_meta" and isinstance(row.get("payload"), dict):
            return row["payload"]
    return {}


def _render_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, divider, *body])


def _format_share(seconds: float, total_seconds: float) -> str:
    if total_seconds <= 0:
        return "0.0%"
    return f"{(seconds / total_seconds) * 100.0:.1f}%"


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _coerce_existing_path(raw_path: str | None, *, base_dir: str | None = None) -> Path | None:
    if raw_path is None:
        return None
    value = raw_path.strip().strip("\"'")
    if not value:
        return None
    candidates: list[Path] = []
    if WINDOWS_ABS_RE.match(value):
        candidates.append(Path(value))
    else:
        candidate = Path(value)
        if candidate.is_absolute():
            candidates.append(candidate)
        if base_dir is not None:
            candidates.append(Path(base_dir) / value)
        candidates.append(candidate)
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved.exists():
            return resolved
    return None


def _canonical_repo_key(repository_url: str | None) -> str | None:
    if repository_url is None:
        return None
    value = repository_url.strip()
    if not value:
        return None
    if value.startswith("git@") and ":" in value:
        value = value.split(":", 1)[1]
    elif "://" in value:
        value = value.split("://", 1)[1]
        if "/" in value:
            value = value.split("/", 1)[1]
    value = value.removesuffix(".git").strip("/")
    return value or None


def _display_name_from_repo(repository_url: str | None, git_root: str | None) -> str:
    canonical = _canonical_repo_key(repository_url)
    if canonical is not None and "/" in canonical:
        return canonical.rsplit("/", 1)[1]
    if canonical is not None:
        return canonical
    if git_root is not None:
        return Path(git_root).name or _normalize_display_path(git_root)
    return MULTI_UNKNOWN_REPO_KEY


def _normalize_display_path(value: str | None) -> str:
    if value is None:
        return MULTI_UNKNOWN_REPO_KEY
    return value.replace("\\", "/")


def _limit_text(text: str, length: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= length:
        return compact
    return compact[: length - 1].rstrip() + "…"
