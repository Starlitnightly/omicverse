"""Session/context facade mixin — extracted from OmicVerseAgent.

Owns lazy-service construction, session-ID delegation, tracing/runtime
bootstrap, and the public session/context API surface.  The concrete
agent class inherits this mixin so that the methods appear on
``OmicVerseAgent`` without bloating its class body.

The mixin relies on attributes set by the host ``__init__`` (e.g.
``_config``, ``_llm``, ``model``, ``enable_filesystem_context``) and
never stores new top-level attributes beyond the service delegates it
manages.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from ..filesystem_context import FilesystemContextManager
    from .session_context import ContextService, SessionService

logger = logging.getLogger(__name__)


class SessionContextFacadeMixin:
    """Mixin providing session, context, tracing, and service-wiring delegation.

    Expects the host class to set the following attributes before calling
    the bootstrap helpers:

    * ``_config``, ``_llm``, ``model``
    * ``_filesystem_context``, ``enable_filesystem_context``
    * ``use_notebook_execution``, ``max_prompts_per_session``
    * ``_notebook_executor``
    """

    # ----------------------------------------------------------------
    # Bootstrap: session, context, tracing, and runtime initialization
    # ----------------------------------------------------------------

    def _initialize_session_context_tracing(
        self,
        *,
        ctx_storage_dir: Optional[Path] = None,
    ) -> None:
        """Bootstrap filesystem context, session history, tracing, and OV runtime.

        Called once from ``__init__`` after the LLM backend is created.
        """
        from .bootstrap import (
            initialize_filesystem_context,
            initialize_ov_runtime,
            initialize_session_history,
            initialize_tracing,
        )

        self.enable_filesystem_context, self._filesystem_context = (
            initialize_filesystem_context(
                enabled=self.enable_filesystem_context,
                storage_dir=ctx_storage_dir,
            )
        )
        self._session_history = initialize_session_history(self._config)
        self._trace_store, self._context_compactor = initialize_tracing(
            self._config, self._llm, self.model,
        )
        self._ov_runtime = initialize_ov_runtime(self._detect_repo_root())

    def _wire_session_context_services(self) -> None:
        """Construct SessionService and ContextService delegates."""
        from .session_context import ContextService, SessionService

        self._session_service = SessionService(self)
        self._context_service = ContextService(self)

    # ----------------------------------------------------------------
    # Lazy service constructors (for __new__-based instances)
    # ----------------------------------------------------------------

    # Class-level lock for thread-safe lazy service initialization.
    # Double-checked locking ensures at most one service instance is
    # created even when multiple threads race on first access.
    _service_init_lock = threading.Lock()

    def _get_session_service(self) -> "SessionService":
        """Lazily construct SessionService for legacy __new__-based instances.

        Thread-safe: uses double-checked locking so concurrent callers
        never create duplicate service instances.
        """
        service = getattr(self, "_session_service", None)
        if service is not None:
            return service
        with self._service_init_lock:
            service = getattr(self, "_session_service", None)
            if service is None:
                from .session_context import SessionService

                service = SessionService(self)
                self._session_service = service
        return service

    def _get_context_service(self) -> "ContextService":
        """Lazily construct ContextService for legacy __new__-based instances.

        Thread-safe: uses double-checked locking so concurrent callers
        never create duplicate service instances.
        """
        service = getattr(self, "_context_service", None)
        if service is not None:
            return service
        with self._service_init_lock:
            service = getattr(self, "_context_service", None)
            if service is None:
                from .session_context import ContextService

                service = ContextService(self)
                self._context_service = service
        return service

    # ----------------------------------------------------------------
    # Session ID delegation (used by ovagent modules via AgentContext)
    # ----------------------------------------------------------------

    def _get_harness_session_id(self) -> str:
        """Best-effort session identifier for harness traces/history."""
        return self._get_session_service().get_harness_session_id()

    def _get_runtime_session_id(self) -> str:
        """Return the session key used by the harness runtime registry."""
        return self._get_session_service().get_runtime_session_id()

    def _refresh_runtime_working_directory(self) -> str:
        """Keep runtime cwd aligned with the active worktree / filesystem context."""
        return self._get_session_service().refresh_runtime_working_directory()

    # ----------------------------------------------------------------
    # Runtime helpers
    # ----------------------------------------------------------------

    def _detect_repo_root(self, cwd: Optional[Path] = None) -> Optional[Path]:
        """Walk up from *cwd* (or runtime working dir) to find a .git root."""
        current = (cwd or Path(self._refresh_runtime_working_directory())).resolve()
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists():
                return candidate
        return None

    # ----------------------------------------------------------------
    # Public Session Management API
    # ----------------------------------------------------------------

    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current notebook session."""
        return self._get_session_service().get_current_session_info()

    def restart_session(self):
        """Manually restart notebook session (clear memory, start fresh)."""
        self._get_session_service().restart_session()

    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get history of all archived notebook sessions."""
        return self._get_session_service().get_session_history()

    # ----------------------------------------------------------------
    # Public Filesystem Context Management API
    # ----------------------------------------------------------------

    @property
    def filesystem_context(self) -> Optional["FilesystemContextManager"]:
        """Get the filesystem context manager."""
        return self._get_context_service().filesystem_context

    def write_note(
        self,
        key: str,
        content: Union[str, Dict[str, Any]],
        category: str = "notes",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Write a note to the filesystem context workspace."""
        return self._get_context_service().write_note(key, content, category, metadata)

    def search_context(
        self,
        pattern: str,
        match_type: str = "glob",
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search the filesystem context for relevant notes."""
        return self._get_context_service().search_context(pattern, match_type, max_results)

    def get_relevant_context(self, query: str, max_tokens: int = 1000) -> str:
        """Get context relevant to a query."""
        return self._get_context_service().get_relevant_context(query, max_tokens)

    def save_plan(self, steps: List[Dict[str, Any]]) -> Optional[str]:
        """Save an execution plan."""
        return self._get_context_service().save_plan(steps)

    def update_plan_step(
        self,
        step_index: int,
        status: str,
        result: Optional[str] = None,
    ) -> None:
        """Update the status of a plan step."""
        self._get_context_service().update_plan_step(step_index, status, result)

    def get_workspace_summary(self) -> str:
        """Get a summary of the filesystem context workspace."""
        return self._get_context_service().get_workspace_summary()

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the filesystem context workspace."""
        return self._get_context_service().get_context_stats()
