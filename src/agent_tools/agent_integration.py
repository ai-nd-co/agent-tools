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
TransformProvider = Literal["codex", "claude-code"]


@dataclass(frozen=True)
class AgentIntegrationStatus:
    enabled: bool
    install_state: AgentIntegrationInstallState
    integration_state: AgentIntegrationInstallState
    available_providers: tuple[TransformProvider, ...]
    codex: CodexIntegrationStatus
    claude: ClaudeIntegrationStatus
    availability_issues: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def effective_enabled(self) -> bool:
        return self.integration_state == "installed" and self.enabled

    @property
    def any_provider_available(self) -> bool:
        return bool(self.available_providers)


def load_agent_integration_status(
    codex_home: Path | None = None,
    claude_home: Path | None = None,
    *,
    platform_name: str | None = None,
) -> AgentIntegrationStatus:
    codex = load_codex_integration_status(codex_home, platform_name=platform_name)
    claude = load_claude_integration_status(claude_home)
    enabled = load_codex_integration_enabled()
    available_providers = available_transform_providers(codex=codex, claude=claude)
    install_state = "installed" if available_providers else "missing"
    integration_state = _combine_integration_states(codex=codex, claude=claude)
    availability_issues = tuple(f"codex:{issue}" for issue in codex.availability_issues) + tuple(
        f"claude:{issue}" for issue in claude.availability_issues
    )
    issues = tuple(f"codex:{issue}" for issue in codex.issues) + tuple(
        f"claude:{issue}" for issue in claude.issues
    )
    return AgentIntegrationStatus(
        enabled=enabled,
        install_state=install_state,
        integration_state=integration_state,
        available_providers=available_providers,
        codex=codex,
        claude=claude,
        availability_issues=availability_issues,
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
    if not status.available_providers:
        return "Install Codex or Claude Code first"
    provider_label = _available_provider_label(status.available_providers)
    if status.integration_state == "installed":
        if not status.enabled:
            return f"Off - soft disabled ({provider_label} available)"
        return f"On - {provider_label} available"
    if status.integration_state == "broken":
        return f"{provider_label} available - auto-TTS integration needs repair"
    return f"{provider_label} available - auto-TTS integration not installed"


def agent_integration_toggle_checked(status: AgentIntegrationStatus) -> bool:
    return status.effective_enabled


def should_show_agent_install_panel(status: AgentIntegrationStatus) -> bool:
    return not status.any_provider_available


def agent_integration_install_title(status: AgentIntegrationStatus) -> str:
    return "Install Codex or Claude Code first"


def agent_integration_install_body(status: AgentIntegrationStatus) -> str:
    return (
        "AgentTools can work with either Codex or Claude Code, but it does not install them "
        "for you. Install or sign in to any one backend first, then reopen this controller."
    )


def agent_integration_install_action_text(status: AgentIntegrationStatus) -> str:
    if not status.any_provider_available:
        return ""
    if status.integration_state == "broken":
        return "Repair"
    if status.integration_state == "missing":
        return "Install"
    return ""


def agent_integration_tooltip(status: AgentIntegrationStatus) -> str:
    lines = [
        f"Codex home: {status.codex.codex_home}",
        f"Claude home: {status.claude.claude_home}",
    ]
    if status.availability_issues:
        lines.append(f"Availability: {', '.join(status.availability_issues)}")
    if status.issues:
        lines.append(f"Issues: {', '.join(status.issues)}")
    return "\n".join(lines)


def available_transform_providers(
    *,
    codex: CodexIntegrationStatus,
    claude: ClaudeIntegrationStatus,
) -> tuple[TransformProvider, ...]:
    providers: list[TransformProvider] = []
    if codex.available:
        providers.append("codex")
    if claude.available:
        providers.append("claude-code")
    return tuple(providers)


def resolve_transform_provider_or_fallback(
    *,
    requested_provider: TransformProvider,
    available_providers: tuple[TransformProvider, ...],
    explicit: bool,
) -> TransformProvider:
    if requested_provider in available_providers:
        return requested_provider
    if explicit:
        available_text = ", ".join(available_providers) or "none"
        raise RuntimeError(
            f"{requested_provider} is not available. Available providers: {available_text}."
        )
    if available_providers:
        return available_providers[0]
    raise RuntimeError(
        "Install or sign in to Codex, or install Claude Code. AgentTools can use either one."
    )


def selected_provider_fallback_note(
    *,
    selected_provider: TransformProvider,
    available_providers: tuple[TransformProvider, ...],
) -> str | None:
    if selected_provider in available_providers:
        return None
    if not available_providers:
        return None
    return (
        f"{_provider_label(selected_provider)} selected, using "
        f"{_provider_label(available_providers[0])} because it is not available."
    )


def _combine_integration_states(
    *,
    codex: CodexIntegrationStatus,
    claude: ClaudeIntegrationStatus,
) -> AgentIntegrationInstallState:
    states: list[AgentIntegrationInstallState] = []
    if codex.available:
        states.append(codex.install_state)
    if claude.available:
        states.append(claude.install_state)
    if not states:
        return "missing"
    if any(state == "broken" for state in states):
        return "broken"
    if all(state == "installed" for state in states):
        return "installed"
    return "missing"


def _available_provider_label(providers: tuple[TransformProvider, ...]) -> str:
    if len(providers) == 2:
        return "Codex and Claude Code"
    if providers:
        return _provider_label(providers[0])
    return "No providers"


def _provider_label(provider: TransformProvider) -> str:
    return "Claude Code" if provider == "claude-code" else "Codex"
