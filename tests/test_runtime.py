from __future__ import annotations

from agent_tools.runtime import DEFAULT_CONTROLLER_PORT, _controller_port_from_env


def test_controller_port_uses_safe_default_when_unset() -> None:
    assert _controller_port_from_env({}) == DEFAULT_CONTROLLER_PORT


def test_controller_port_respects_valid_env_override() -> None:
    assert _controller_port_from_env({"AGENT_TOOLS_CONTROLLER_PORT": "52000"}) == 52000


def test_controller_port_ignores_invalid_env_override() -> None:
    assert _controller_port_from_env({"AGENT_TOOLS_CONTROLLER_PORT": "invalid"}) == (
        DEFAULT_CONTROLLER_PORT
    )


def test_controller_port_ignores_out_of_range_env_override() -> None:
    assert _controller_port_from_env({"AGENT_TOOLS_CONTROLLER_PORT": "70000"}) == (
        DEFAULT_CONTROLLER_PORT
    )
