"""SessionService & ContextService — session lifecycle and filesystem context.

Extracted from ``smart_agent.py`` so that session reset/history access,
session-ID resolution, runtime working-directory tracking, and
filesystem/notebook context delegation live in focused service code
instead of being scattered across the monolithic agent class.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..harness.runtime_state import runtime_state

if TYPE_CHECKING:
    from ..filesystem_context import FilesystemContextManager
    from .prompt_builder import build_filesystem_context_instructions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SessionService
# ---------------------------------------------------------------------------

class SessionService:
    """Owns session lifecycle, ID resolution, and runtime directory tracking.

    Parameters
    ----------
    ctx
        The owning agent instance (accessed via protocol surface).
    """

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    # -- session ID helpers --------------------------------------------------

    def get_harness_session_id(self) -> str:
        """Best-effort session identifier for harness traces/history."""
        web_session_id = getattr(self._ctx, "_web_session_id", "")
        if web_session_id:
            return web_session_id
        fs_ctx = getattr(self._ctx, "_filesystem_context", None)
        if fs_ctx is not None:
            return fs_ctx.session_id
        nb = getattr(self._ctx, "_notebook_executor", None)
        if nb is not None and nb.current_session:
            return nb.current_session.get("session_id", "")
        return ""

    def get_runtime_session_id(self) -> str:
        """Return the session key used by the harness runtime registry."""
        return self.get_harness_session_id() or "default"

    # -- runtime working directory ------------------------------------------

    def refresh_runtime_working_directory(self) -> str:
        """Keep runtime cwd aligned with the active worktree / filesystem context."""
        session_id = self.get_runtime_session_id()
        cwd = runtime_state.get_working_directory(session_id)
        if cwd:
            return cwd
        current = os.getcwd()
        fs_ctx = getattr(self._ctx, "_filesystem_context", None)
        if fs_ctx is not None:
            current = str(fs_ctx._workspace_dir)
        runtime_state.set_working_directory(session_id, current)
        return current

    # -- notebook session lifecycle -----------------------------------------

    def restart_session(self) -> None:
        """Manually restart notebook session (clear memory, start fresh).

        This forces a new session to be created on the next execution,
        useful for freeing memory or starting with a clean state.
        """
        ctx = self._ctx
        if ctx.use_notebook_execution and ctx._notebook_executor:
            if ctx._notebook_executor.current_session:
                print("⚙ = Manually restarting session...")
                ctx._notebook_executor._archive_current_session()
                ctx._notebook_executor.current_session = None
                ctx._notebook_executor.session_prompt_count = 0
                print("✓ Session cleared. Next prompt will start new session.")
            else:
                print("💡 No active session to restart")
        else:
            print("⚠️  Notebook execution is not enabled")

    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get history of all archived notebook sessions.

        Returns
        -------
        list of dict
            List of session history dictionaries, each containing
            session_id, notebook_path, prompt_count, start_time,
            end_time, and executions.
        """
        ctx = self._ctx
        if ctx.use_notebook_execution and ctx._notebook_executor:
            return ctx._notebook_executor.session_history
        return []

    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current notebook session.

        Returns
        -------
        dict or None
            Session information dictionary, or None if notebook execution
            is disabled or no session exists.
        """
        ctx = self._ctx
        if not ctx.use_notebook_execution or not ctx._notebook_executor:
            return None
        if not ctx._notebook_executor.current_session:
            return None

        session = ctx._notebook_executor.current_session
        return {
            "session_id": session["session_id"],
            "notebook_path": str(session["notebook_path"]),
            "prompt_count": ctx._notebook_executor.session_prompt_count,
            "max_prompts": ctx._notebook_executor.max_prompts_per_session,
            "remaining_prompts": ctx.max_prompts_per_session
            - ctx._notebook_executor.session_prompt_count,
            "start_time": session["start_time"].isoformat(),
        }


# ---------------------------------------------------------------------------
# ContextService
# ---------------------------------------------------------------------------

class ContextService:
    """Owns filesystem context delegation (notes, plans, search, stats).

    Parameters
    ----------
    ctx
        The owning agent instance (accessed via protocol surface).
    """

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    # -- core accessor ------------------------------------------------------

    @property
    def filesystem_context(self) -> Optional["FilesystemContextManager"]:
        """Return the filesystem context manager if enabled, else None."""
        ctx = self._ctx
        return ctx._filesystem_context if ctx.enable_filesystem_context else None

    # -- prompt fragment -----------------------------------------------------

    def build_filesystem_context_instructions(self) -> str:
        """Build instructions for using the filesystem context workspace."""
        from .prompt_builder import build_filesystem_context_instructions

        fs_ctx = getattr(self._ctx, "_filesystem_context", None)
        session_id = fs_ctx.session_id if fs_ctx else "N/A"
        return build_filesystem_context_instructions(session_id)

    # -- note operations -----------------------------------------------------

    def write_note(
        self,
        key: str,
        content: Union[str, Dict[str, Any]],
        category: str = "notes",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Write a note to the filesystem context workspace.

        Returns the path to the stored note, or None if context is disabled.
        """
        fs = self._ctx._filesystem_context
        if not fs:
            return None
        try:
            return fs.write_note(key, content, category, metadata)
        except Exception as e:
            logger.warning("Failed to write note: %s", e)
            return None

    def search_context(
        self,
        pattern: str,
        match_type: str = "glob",
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search the filesystem context for relevant notes."""
        fs = self._ctx._filesystem_context
        if not fs:
            return []
        try:
            results = fs.search_context(pattern, match_type, max_results=max_results)
            return [
                {
                    "key": r.key,
                    "category": r.category,
                    "preview": r.content_preview,
                    "relevance": r.relevance_score,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("Failed to search context: %s", e)
            return []

    def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 1000,
    ) -> str:
        """Get context relevant to a query, formatted for LLM injection."""
        fs = self._ctx._filesystem_context
        if not fs:
            return ""
        try:
            return fs.get_relevant_context(query, max_tokens)
        except Exception as e:
            logger.warning("Failed to get relevant context: %s", e)
            return ""

    # -- plan operations -----------------------------------------------------

    def save_plan(self, steps: List[Dict[str, Any]]) -> Optional[str]:
        """Save an execution plan to the filesystem context."""
        fs = self._ctx._filesystem_context
        if not fs:
            return None
        try:
            return fs.write_plan(steps)
        except Exception as e:
            logger.warning("Failed to save plan: %s", e)
            return None

    def update_plan_step(
        self,
        step_index: int,
        status: str,
        result: Optional[str] = None,
    ) -> None:
        """Update the status of a plan step."""
        fs = self._ctx._filesystem_context
        if not fs:
            return
        try:
            fs.update_plan_step(step_index, status, result)
        except Exception as e:
            logger.warning("Failed to update plan step: %s", e)

    # -- workspace summary / stats -------------------------------------------

    def get_workspace_summary(self) -> str:
        """Get a summary of the filesystem context workspace."""
        fs = self._ctx._filesystem_context
        if not fs:
            return "Filesystem context is disabled."
        try:
            return fs.get_session_summary()
        except Exception as e:
            logger.warning("Failed to get workspace summary: %s", e)
            return f"Error getting workspace summary: {e}"

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the filesystem context workspace."""
        fs = self._ctx._filesystem_context
        if not fs:
            return {"enabled": False}
        try:
            stats = fs.get_workspace_stats()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.warning("Failed to get context stats: %s", e)
            return {"enabled": True, "error": str(e)}
