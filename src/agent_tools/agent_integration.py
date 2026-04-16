from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_tools.claude_integration import ClaudeIntegrationStatus, load_claude_integration_status
from agent_tools.codex_integration import (
    CodexIntegrationStatus,
    load_codex_integration_enabled,
    load_codex_integration_status,
    set_codex_integration_enabled,
)
from agent_tools.hook_install import (
    InstallAgentIntegrationsResult,
    install_agent_integrations,
)

AgentIntegrationInstallState = Literal["installed", "missing", "broken"]


@dataclass(frozen=True)
class AgentIntegrationStatus:
    enabled: bool
    install_state: AgentIntegrationInstallState
    codex: CodexIntegrationStatus
    claude: ClaudeIntegrationStatus
    issues: tuple[str, ...] = ()

    @property
    def effective_enabled(self) -> bool:
        return self.install_state == "installed" and self.enabled


def load_agent_integration_status(
    codex_home: Path | None = None,
    claude_home: Path | None = None,
    *,
    platform_name: str | None = None,
) -> AgentIntegrationStatus:
    codex = load_codex_integration_status(codex_home, platform_name=platform_name)
    claude = load_claude_integration_status(claude_home)
    enabled = load_codex_integration_enabled()
    install_state = _combine_install_states(codex.install_state, claude.install_state)
    issues = tuple(f"codex:{issue}" for issue in codex.issues) + tuple(
        f"claude:{issue}" for issue in claude.issues
    )
    return AgentIntegrationStatus(
        enabled=enabled,
        install_state=install_state,
        codex=codex,
        claude=claude,
        issues=issues,
    )


def install_all_integrations(
    codex_home: Path | None = None,
    claude_home: Path | None = None,
    *,
    platform_name: str | None = None,
) -> InstallAgentIntegrationsResult:
    return install_agent_integrations(
        codex_home=codex_home,
        claude_home=claude_home,
        platform_name=platform_name,
    )


def set_agent_integration_enabled(enabled: bool) -> None:
    set_codex_integration_enabled(enabled)


def agent_integration_status_text(status: AgentIntegrationStatus) -> str:
    if status.install_state == "installed":
        if not status.enabled:
            return "Off - soft disabled"
        codex_mode = "notify" if status.codex.mode == "notify" else "stop hook"
        return f"On - Codex {codex_mode} + Claude stop hook"
    if status.install_state == "broken":
        return "Off - repair needed"
    return "Off - not installed"


def agent_integration_toggle_checked(status: AgentIntegrationStatus) -> bool:
    return status.effective_enabled


def should_show_agent_install_panel(status: AgentIntegrationStatus) -> bool:
    return status.install_state != "installed"


def agent_integration_install_title(status: AgentIntegrationStatus) -> str:
    if status.install_state == "broken":
        return "We need to repair AgentTools"
    return "We need to install AgentTools"


def agent_integration_install_body(status: AgentIntegrationStatus) -> str:
    if status.install_state == "broken":
        return (
            "The AgentTools integration looks incomplete. Repair it so completed "
            "Codex and Claude Code replies can be queued as spoken audio in this controller."
        )
    return (
        "Install it once to connect Codex and Claude Code to this controller. "
        "Completed Codex and Claude Code replies are queued as spoken audio automatically."
    )


def agent_integration_install_action_text(status: AgentIntegrationStatus) -> str:
    if status.install_state == "broken":
        return "Repair"
    return "Install"


def agent_integration_tooltip(status: AgentIntegrationStatus) -> str:
    lines = [
        f"Codex home: {status.codex.codex_home}",
        f"Claude home: {status.claude.claude_home}",
    ]
    if status.issues:
        lines.append(f"Issues: {', '.join(status.issues)}")
    return "\n".join(lines)


def _combine_install_states(
    codex_state: AgentIntegrationInstallState,
    claude_state: AgentIntegrationInstallState,
) -> AgentIntegrationInstallState:
    if codex_state == "installed" and claude_state == "installed":
        return "installed"
    if "broken" in (codex_state, claude_state):
        return "broken"
    return "missing"
