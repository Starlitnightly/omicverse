"""Codegen / tool-dispatch facade mixin — residual compatibility surface.

After task-049 rewired ovagent consumers (TurnController, ToolRuntime,
AnalysisExecutor, SubagentController) to hold direct subsystem references,
the majority of the original ~45-method delegation surface became dead code.

This mixin now retains only:

* **Lazy-init properties** (``_codegen``, ``_scanner``) — required so that
  ``OmicVerseAgent`` instances constructed via ``__new__`` (common in tests)
  still get working subsystem instances on first access.
* **``_collect_static_registry_entries``** — still accessed via duck-typed
  ``getattr(ctx, ...)`` in ``tool_runtime_exec.handle_search_functions``.

All other wrappers were removed in task-050. A closure test in
``tests/utils/test_smart_agent.py`` prevents silent regrowth.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .codegen_pipeline import CodegenPipeline
    from .registry_scanner import RegistryScanner

logger = logging.getLogger(__name__)


class CodegenToolDispatchFacadeMixin:
    """Residual mixin providing lazy-init properties and minimal compatibility wrappers.

    Expects the host class to set ``_codegen_pipeline`` and
    ``_registry_scanner`` before any delegate method is called (or allows
    them to be lazily constructed via the properties below).
    """

    # ================================================================
    # Lazy service constructors (for __new__-based instances in tests)
    # ================================================================

    @property
    def _codegen(self) -> "CodegenPipeline":
        """Lazily create CodegenPipeline for agents constructed via __new__."""
        from .codegen_pipeline import CodegenPipeline as _CodegenPipeline

        pipeline = getattr(self, "_codegen_pipeline", None)
        if pipeline is None:
            pipeline = _CodegenPipeline(self)
            self._codegen_pipeline = pipeline  # type: ignore[attr-defined]
        return pipeline

    @property
    def _scanner(self) -> "RegistryScanner":
        """Lazily create RegistryScanner for agents constructed via __new__."""
        from .registry_scanner import RegistryScanner as _RegistryScanner

        scanner = getattr(self, "_registry_scanner", None)
        if scanner is None:
            scanner = _RegistryScanner()
            self._registry_scanner = scanner  # type: ignore[attr-defined]
        return scanner

    # ================================================================
    # Compatibility wrapper — accessed via getattr in tool_runtime_exec
    # ================================================================

    def _collect_static_registry_entries(
        self, request: str, max_entries: int = 8
    ) -> List[Dict[str, Any]]:
        """Search the static AST-derived registry snapshot.

        Kept because ``tool_runtime_exec.handle_search_functions`` accesses
        this via ``getattr(ctx, "_collect_static_registry_entries", None)``.
        """
        return self._scanner.collect_static_entries(request, max_entries)

    def _collect_relevant_registry_entries(
        self, request: str, max_entries: int = 8
    ) -> List[Dict[str, Any]]:
        """Search the merged runtime/static registry with unified re-ranking."""

        return self._scanner.collect_relevant_entries(request, max_entries)
