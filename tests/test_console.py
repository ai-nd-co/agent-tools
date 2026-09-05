from __future__ import annotations

import pytest

from agent_tools import cli
from agent_tools.console import configure_windows_utf8


class ReconfigurableStream:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)


class FrozenStream:
    pass


class FailingStream:
    def reconfigure(self, **_kwargs: str) -> None:
        raise OSError("fixture stream is frozen")


def test_windows_cli_streams_are_reconfigured_to_strict_utf8() -> None:
    stdout = ReconfigurableStream()
    stderr = ReconfigurableStream()

    configure_windows_utf8(
        stdout=stdout,
        stderr=stderr,
        platform_name="win32",
    )

    expected = [{"encoding": "utf-8", "errors": "strict"}]
    assert stdout.calls == expected
    assert stderr.calls == expected


def test_non_windows_cli_streams_are_unchanged() -> None:
    stdout = ReconfigurableStream()
    stderr = ReconfigurableStream()

    configure_windows_utf8(
        stdout=stdout,
        stderr=stderr,
        platform_name="linux",
    )

    assert stdout.calls == []
    assert stderr.calls == []


def test_missing_or_frozen_stream_reconfigure_is_safe() -> None:
    configure_windows_utf8(
        stdout=FrozenStream(),
        stderr=FailingStream(),
        platform_name="win32",
    )


def test_cli_configures_utf8_before_argument_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli, "configure_windows_utf8", lambda: events.append("utf8"))

    class Parser:
        def parse_args(self, _arguments):
            events.append("parse")
            raise RuntimeError("fixture")

    monkeypatch.setattr(cli, "build_parser", Parser)

    assert cli.main([]) == 1
    assert events == ["utf8", "parse"]
