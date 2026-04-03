"""Stable OmicVerse lookup helpers exposed via ``omicverse.utils``.

These wrappers make the ovagent registry/skill lookup flow available from the
same Python runtime that executes OmicVerse analysis code. This lets external
agent systems call OmicVerse-native lookup helpers without relying on a
separate Pantheon toolset process.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .ovagent.bootstrap import initialize_skill_registry
from .ovagent.registry_scanner import RegistryScanner

logger = logging.getLogger(__name__)


def _registry_unavailable_message() -> str:
    return (
        "OmicVerse RegistryScanner not available -- "
        "use `import omicverse as ov; help(ov)` for API reference."
    )


class _RegistryLookupContext:
    """Minimal ovagent context for ``handle_search_functions``."""

    def __init__(self, scanner: RegistryScanner):
        self._scanner = scanner

    def _collect_static_registry_entries(
        self,
        request: str,
        max_entries: int = 8,
    ) -> list[dict[str, Any]]:
        return self._scanner.collect_static_entries(
            request,
            max_entries=max_entries,
        )


class _SkillLookupContext:
    """Minimal ovagent context for ``handle_skill``."""

    def __init__(self, skill_registry: Any, max_body_chars: int):
        self.skill_registry = skill_registry
        self._llm = None
        self._max_body_chars = max_body_chars

    def _load_skill_guidance(self, skill_name: str) -> str:
        if not self.skill_registry or not self.skill_registry.skill_metadata:
            return json.dumps({"error": "No project skills are available."})
        if not skill_name or not skill_name.strip():
            return json.dumps(
                {"error": "Provide a skill name to load guidance."}
            )

        slug = skill_name.strip().lower()
        definition = self.skill_registry.load_full_skill(slug)
        if not definition:
            return json.dumps(
                {"error": f"Skill '{skill_name}' not found."}
            )

        return json.dumps(
            {
                "name": definition.name,
                "description": definition.description,
                "instructions": definition.prompt_instructions(
                    max_chars=self._max_body_chars,
                    provider=None,
                ),
                "path": str(definition.path),
                "metadata": definition.metadata,
            },
            indent=2,
        )


def _create_registry_scanner() -> RegistryScanner:
    scanner = RegistryScanner()
    scanner.ensure_runtime_registry()
    scanner.load_static_entries()
    return scanner


def _create_skill_registry() -> Any:
    registry, _ = initialize_skill_registry()
    return registry


def _delegate_registry_lookup(ctx: Any, query: str) -> str:
    from .ovagent.tool_runtime_exec import handle_search_functions

    return handle_search_functions(ctx, query)


def _delegate_skill_lookup(ctx: Any, query: str) -> str:
    from .ovagent.tool_runtime_workspace import handle_skill

    return handle_skill(ctx, query)


def registry_lookup(
    query: str,
    max_results: int = 15,
) -> str:
    """Search the OmicVerse function registry from the current runtime.

    Parameters
    ----------
    query
        Natural-language description of the function or operation needed.
    max_results
        Soft cap for caller compatibility. The delegated ovagent formatter
        currently applies its own presentation limit.
    """

    _ = max_results
    try:
        scanner = _create_registry_scanner()
    except Exception as exc:
        logger.warning("RegistryScanner init failed: %s", exc)
        return _registry_unavailable_message()

    try:
        return _delegate_registry_lookup(
            _RegistryLookupContext(scanner),
            query,
        )
    except Exception as exc:
        logger.warning("registry_lookup delegation failed: %s", exc)
        return _registry_unavailable_message()


def skill_lookup(
    query: str,
    max_body_chars: int = 3000,
) -> str:
    """Search and load OmicVerse workflow guidance from the current runtime."""

    try:
        registry = _create_skill_registry()
    except Exception as exc:
        logger.warning("Skill registry init failed: %s", exc)
        registry = None

    try:
        return _delegate_skill_lookup(
            _SkillLookupContext(registry, max_body_chars=max_body_chars),
            query,
        )
    except Exception as exc:
        logger.warning("skill_lookup delegation failed: %s", exc)
        return "Skills matched but content could not be loaded."


__all__ = [
    "RegistryScanner",
    "initialize_skill_registry",
    "registry_lookup",
    "skill_lookup",
]
