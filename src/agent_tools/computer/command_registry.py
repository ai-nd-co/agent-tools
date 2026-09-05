"""Single source of truth for computer command inventories.

`info` readiness, `capabilities`, and the mutation capability block all derive
their command lists from this registry so the three inventories can never drift
apart. Kinds separate observation from mutation families; requirement gates
name the backend that must be available before a command is offered.
"""

from __future__ import annotations

import importlib.util
import sys

KIND_OBSERVATION = "observation"
KIND_WINDOW_MUTATION = "window_mutation"
KIND_SEMANTIC_MUTATION = "semantic_mutation"
KIND_PHYSICAL_INPUT = "physical_input"
KIND_NOTIFICATION = "notification"

REQUIREMENT_UIA_WINAPP = "uia_winapp"
REQUIREMENT_MUTATIONS = "mutations"
REQUIREMENT_OCR = "ocr"

_COMMANDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("info", KIND_OBSERVATION, ()),
    ("windows", KIND_OBSERVATION, ()),
    ("focused", KIND_OBSERVATION, ()),
    ("screenshot", KIND_OBSERVATION, ()),
    ("ocr", KIND_OBSERVATION, (REQUIREMENT_OCR,)),
    ("inspect", KIND_OBSERVATION, (REQUIREMENT_UIA_WINAPP,)),
    ("read", KIND_OBSERVATION, (REQUIREMENT_UIA_WINAPP,)),
    ("scroll-areas", KIND_OBSERVATION, (REQUIREMENT_UIA_WINAPP,)),
    ("capabilities", KIND_OBSERVATION, ()),
    ("focus", KIND_WINDOW_MUTATION, (REQUIREMENT_MUTATIONS,)),
    ("restore", KIND_WINDOW_MUTATION, (REQUIREMENT_MUTATIONS,)),
    ("minimize", KIND_WINDOW_MUTATION, (REQUIREMENT_MUTATIONS,)),
    ("maximize", KIND_WINDOW_MUTATION, (REQUIREMENT_MUTATIONS,)),
    ("resize", KIND_WINDOW_MUTATION, (REQUIREMENT_MUTATIONS,)),
    ("invoke", KIND_SEMANTIC_MUTATION, (REQUIREMENT_UIA_WINAPP, REQUIREMENT_MUTATIONS)),
    ("set-value", KIND_SEMANTIC_MUTATION, (REQUIREMENT_UIA_WINAPP, REQUIREMENT_MUTATIONS)),
    ("scroll", KIND_SEMANTIC_MUTATION, (REQUIREMENT_UIA_WINAPP, REQUIREMENT_MUTATIONS)),
    ("move-pointer", KIND_PHYSICAL_INPUT, (REQUIREMENT_MUTATIONS,)),
    ("click", KIND_PHYSICAL_INPUT, (REQUIREMENT_MUTATIONS,)),
    ("type", KIND_PHYSICAL_INPUT, (REQUIREMENT_MUTATIONS,)),
    ("key", KIND_PHYSICAL_INPUT, (REQUIREMENT_MUTATIONS,)),
    ("shortcut", KIND_PHYSICAL_INPUT, (REQUIREMENT_MUTATIONS,)),
    ("wheel", KIND_PHYSICAL_INPUT, (REQUIREMENT_MUTATIONS,)),
    ("notify", KIND_NOTIFICATION, (REQUIREMENT_MUTATIONS,)),
)

COMMAND_REGISTRY: dict[str, dict[str, str | list[str]]] = {
    name: {"kind": kind, "requires": list(requires)}
    for name, kind, requires in _COMMANDS
}

MUTATION_KINDS = frozenset(
    {
        KIND_WINDOW_MUTATION,
        KIND_SEMANTIC_MUTATION,
        KIND_PHYSICAL_INPUT,
        KIND_NOTIFICATION,
    }
)


def all_command_names() -> list[str]:
    """Every computer command in stable help order."""
    return [name for name, _kind, _requires in _COMMANDS]


def command_names_by_kind() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name, kind, _requires in _COMMANDS:
        grouped.setdefault(kind, []).append(name)
    return grouped


def mutation_command_names() -> list[str]:
    """Every guarded mutation command (includes physical input commands)."""
    return [name for name, kind, _requires in _COMMANDS if kind in MUTATION_KINDS]


def ocr_supported() -> bool:
    """Cheap platform/binding check; full OCR readiness lives in capabilities."""
    if sys.platform != "win32":
        return False
    try:
        return importlib.util.find_spec("winrt.windows.media.ocr") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def available_command_names(
    *,
    uia_winapp_available: bool,
    mutations_available: bool,
    ocr_available: bool,
) -> list[str]:
    """Registry commands that pass their requirement gates right now."""
    satisfied = {
        REQUIREMENT_UIA_WINAPP: uia_winapp_available,
        REQUIREMENT_MUTATIONS: mutations_available,
        REQUIREMENT_OCR: ocr_available,
    }
    return [
        name
        for name, _kind, requires in _COMMANDS
        if all(satisfied[requirement] for requirement in requires)
    ]
