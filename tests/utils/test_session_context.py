"""Tests for SessionService and ContextService extracted from smart_agent.py.

Verifies that the extracted services preserve the original behaviour of
restart_session, get_session_history, get_current_session_info,
session-ID resolution, runtime working-directory tracking, and
filesystem context delegation.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Lightweight package stubs — avoids pulling in heavy omicverse.__init__
# which requires pandas/numpy at import time.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SAVED = {
    name: sys.modules.get(name)
    for name in ["omicverse", "omicverse.utils"]
}
for name in ["omicverse", "omicverse.utils"]:
    sys.modules.pop(name, None)

_ov_pkg = types.ModuleType("omicverse")
_ov_pkg.__path__ = [str(PACKAGE_ROOT)]
_ov_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse", loader=None, is_package=True)
sys.modules["omicverse"] = _ov_pkg

_utils_pkg = types.ModuleType("omicverse.utils")
_utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
_utils_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse.utils", loader=None, is_package=True)
sys.modules["omicverse.utils"] = _utils_pkg
_ov_pkg.utils = _utils_pkg

from omicverse.utils.ovagent.session_context import ContextService, SessionService

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Lightweight test doubles
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    web_session_id: str = "",
    filesystem_context: Any = None,
    notebook_executor: Any = None,
    use_notebook_execution: bool = True,
    enable_filesystem_context: bool = True,
    max_prompts_per_session: int = 5,
) -> SimpleNamespace:
    """Build a minimal agent-like context for service tests."""
    return SimpleNamespace(
        _web_session_id=web_session_id,
        _filesystem_context=filesystem_context,
        _notebook_executor=notebook_executor,
        use_notebook_execution=use_notebook_execution,
        enable_filesystem_context=enable_filesystem_context,
        max_prompts_per_session=max_prompts_per_session,
    )


class FakeNotebookExecutor:
    """Mimics the notebook executor API used by SessionService."""

    def __init__(
        self,
        *,
        session_id: str = "nb-session-1",
        prompt_count: int = 2,
        max_prompts: int = 5,
        has_session: bool = True,
    ) -> None:
        self.session_prompt_count = prompt_count
        self.max_prompts_per_session = max_prompts
        self.session_history: List[Dict[str, Any]] = [
            {"session_id": "archived-1", "prompt_count": 3}
        ]
        self.current_session: Optional[Dict[str, Any]] = (
            {
                "session_id": session_id,
                "notebook_path": "/tmp/test.ipynb",
                "start_time": datetime(2026, 1, 1, 12, 0),
            }
            if has_session
            else None
        )
        self._archive_called = False

    def _archive_current_session(self) -> None:
        self._archive_called = True


class FakeFilesystemContext:
    """Mimics the FilesystemContextManager API used by ContextService."""

    def __init__(self, session_id: str = "fs-session-1") -> None:
        self.session_id = session_id
        self._workspace_dir = "/tmp/workspace"
        self._notes: Dict[str, Any] = {}
        self._plan: Optional[List[Dict[str, Any]]] = None

    def write_note(
        self, key: str, content: Any, category: str = "notes", metadata: Any = None
    ) -> str:
        self._notes[key] = {"content": content, "category": category}
        return f"/tmp/workspace/{category}/{key}.json"

    def search_context(
        self, pattern: str, match_type: str = "glob", *, max_results: int = 10
    ) -> list:
        return [
            SimpleNamespace(
                key="match-1",
                category="notes",
                content_preview="preview",
                relevance_score=0.9,
            )
        ]

    def get_relevant_context(self, query: str, max_tokens: int = 1000) -> str:
        return f"context for: {query}"

    def write_plan(self, steps: list) -> str:
        self._plan = steps
        return "/tmp/workspace/plans/plan.json"

    def update_plan_step(
        self, step_index: int, status: str, result: Optional[str] = None
    ) -> None:
        if self._plan and 0 <= step_index < len(self._plan):
            self._plan[step_index]["status"] = status

    def get_session_summary(self) -> str:
        return "Session summary text"

    def get_workspace_stats(self) -> dict:
        return {"total_notes": 5, "total_size_bytes": 1234}


# ===================================================================
# SessionService tests
# ===================================================================


class TestSessionServiceSessionId:
    """Test session-ID resolution in SessionService."""

    def test_web_session_id_takes_priority(self):
        ctx = _make_ctx(
            web_session_id="web-1",
            filesystem_context=FakeFilesystemContext("fs-1"),
        )
        svc = SessionService(ctx)
        assert svc.get_harness_session_id() == "web-1"

    def test_filesystem_context_session_id_used_if_no_web(self):
        ctx = _make_ctx(filesystem_context=FakeFilesystemContext("fs-1"))
        svc = SessionService(ctx)
        assert svc.get_harness_session_id() == "fs-1"

    def test_notebook_session_id_used_as_fallback(self):
        nb = FakeNotebookExecutor(session_id="nb-1")
        ctx = _make_ctx(notebook_executor=nb)
        svc = SessionService(ctx)
        assert svc.get_harness_session_id() == "nb-1"

    def test_empty_when_nothing_available(self):
        ctx = _make_ctx()
        svc = SessionService(ctx)
        assert svc.get_harness_session_id() == ""

    def test_runtime_session_id_defaults_to_default(self):
        ctx = _make_ctx()
        svc = SessionService(ctx)
        assert svc.get_runtime_session_id() == "default"

    def test_runtime_session_id_returns_harness_id(self):
        ctx = _make_ctx(web_session_id="web-2")
        svc = SessionService(ctx)
        assert svc.get_runtime_session_id() == "web-2"


class TestSessionServiceLifecycle:
    """Test restart_session and get_session_history."""

    def test_restart_session_archives_and_clears(self, capsys):
        nb = FakeNotebookExecutor()
        ctx = _make_ctx(notebook_executor=nb)
        svc = SessionService(ctx)

        svc.restart_session()

        assert nb._archive_called
        assert nb.current_session is None
        assert nb.session_prompt_count == 0
        out = capsys.readouterr().out
        assert "Manually restarting" in out
        assert "Session cleared" in out

    def test_restart_no_active_session(self, capsys):
        nb = FakeNotebookExecutor(has_session=False)
        ctx = _make_ctx(notebook_executor=nb)
        svc = SessionService(ctx)

        svc.restart_session()

        assert not nb._archive_called
        out = capsys.readouterr().out
        assert "No active session" in out

    def test_restart_not_enabled(self, capsys):
        ctx = _make_ctx(use_notebook_execution=False)
        svc = SessionService(ctx)

        svc.restart_session()

        out = capsys.readouterr().out
        assert "not enabled" in out

    def test_get_session_history_returns_list(self):
        nb = FakeNotebookExecutor()
        ctx = _make_ctx(notebook_executor=nb)
        svc = SessionService(ctx)

        history = svc.get_session_history()
        assert len(history) == 1
        assert history[0]["session_id"] == "archived-1"

    def test_get_session_history_empty_when_disabled(self):
        ctx = _make_ctx(use_notebook_execution=False)
        svc = SessionService(ctx)
        assert svc.get_session_history() == []

    def test_get_current_session_info(self):
        nb = FakeNotebookExecutor(
            session_id="nb-info", prompt_count=2, max_prompts=5
        )
        ctx = _make_ctx(notebook_executor=nb, max_prompts_per_session=5)
        svc = SessionService(ctx)

        info = svc.get_current_session_info()
        assert info is not None
        assert info["session_id"] == "nb-info"
        assert info["prompt_count"] == 2
        assert info["max_prompts"] == 5
        assert info["remaining_prompts"] == 3
        assert info["start_time"] == "2026-01-01T12:00:00"

    def test_get_current_session_info_none_when_disabled(self):
        ctx = _make_ctx(use_notebook_execution=False)
        svc = SessionService(ctx)
        assert svc.get_current_session_info() is None

    def test_get_current_session_info_none_when_no_session(self):
        nb = FakeNotebookExecutor(has_session=False)
        ctx = _make_ctx(notebook_executor=nb)
        svc = SessionService(ctx)
        assert svc.get_current_session_info() is None


class TestSessionServiceRuntimeDir:
    """Test refresh_runtime_working_directory."""

    def test_uses_cached_cwd_from_runtime_state(self):
        from omicverse.utils.harness.runtime_state import runtime_state

        ctx = _make_ctx(web_session_id="sess-dir-test")
        svc = SessionService(ctx)

        runtime_state.set_working_directory("sess-dir-test", "/cached/dir")
        try:
            result = svc.refresh_runtime_working_directory()
            assert result == "/cached/dir"
        finally:
            runtime_state.delete_session("sess-dir-test")

    def test_sets_filesystem_context_workspace_when_cwd_empty(self):
        from omicverse.utils.harness.runtime_state import runtime_state

        fs = FakeFilesystemContext()
        ctx = _make_ctx(web_session_id="sess-ws-test", filesystem_context=fs)
        svc = SessionService(ctx)

        # Force empty working directory on the session to trigger fallback
        runtime_state.delete_session("sess-ws-test")
        state = runtime_state.get_or_create_session("sess-ws-test")
        state.working_directory = ""

        result = svc.refresh_runtime_working_directory()
        assert result == "/tmp/workspace"
        # Clean up
        runtime_state.delete_session("sess-ws-test")


# ===================================================================
# ContextService tests
# ===================================================================


class TestContextServiceProperty:
    """Test filesystem_context property on ContextService."""

    def test_returns_context_when_enabled(self):
        fs = FakeFilesystemContext()
        ctx = _make_ctx(filesystem_context=fs, enable_filesystem_context=True)
        svc = ContextService(ctx)
        assert svc.filesystem_context is fs

    def test_returns_none_when_disabled(self):
        fs = FakeFilesystemContext()
        ctx = _make_ctx(filesystem_context=fs, enable_filesystem_context=False)
        svc = ContextService(ctx)
        assert svc.filesystem_context is None


class TestContextServiceNotes:
    """Test note operations on ContextService."""

    def test_write_note(self):
        fs = FakeFilesystemContext()
        ctx = _make_ctx(filesystem_context=fs)
        svc = ContextService(ctx)

        path = svc.write_note("qc_stats", {"n_cells": 100}, "results")
        assert path is not None
        assert "qc_stats" in fs._notes

    def test_write_note_returns_none_when_no_context(self):
        ctx = _make_ctx(filesystem_context=None)
        svc = ContextService(ctx)
        assert svc.write_note("key", "value") is None

    def test_search_context(self):
        fs = FakeFilesystemContext()
        ctx = _make_ctx(filesystem_context=fs)
        svc = ContextService(ctx)

        results = svc.search_context("match*")
        assert len(results) == 1
        assert results[0]["key"] == "match-1"
        assert results[0]["relevance"] == 0.9

    def test_search_context_empty_when_no_context(self):
        ctx = _make_ctx(filesystem_context=None)
        svc = ContextService(ctx)
        assert svc.search_context("*") == []

    def test_get_relevant_context(self):
        fs = FakeFilesystemContext()
        ctx = _make_ctx(filesystem_context=fs)
        svc = ContextService(ctx)

        result = svc.get_relevant_context("clustering")
        assert "clustering" in result

    def test_get_relevant_context_empty_when_no_context(self):
        ctx = _make_ctx(filesystem_context=None)
        svc = ContextService(ctx)
        assert svc.get_relevant_context("x") == ""


class TestContextServicePlans:
    """Test plan operations on ContextService."""

    def test_save_plan(self):
        fs = FakeFilesystemContext()
        ctx = _make_ctx(filesystem_context=fs)
        svc = ContextService(ctx)

        steps = [{"description": "QC", "status": "pending"}]
        path = svc.save_plan(steps)
        assert path is not None
        assert fs._plan == steps

    def test_save_plan_none_when_no_context(self):
        ctx = _make_ctx(filesystem_context=None)
        svc = ContextService(ctx)
        assert svc.save_plan([]) is None

    def test_update_plan_step(self):
        fs = FakeFilesystemContext()
        ctx = _make_ctx(filesystem_context=fs)
        svc = ContextService(ctx)

        svc.save_plan([{"description": "QC", "status": "pending"}])
        svc.update_plan_step(0, "completed")
        assert fs._plan[0]["status"] == "completed"


class TestContextServiceSummaryStats:
    """Test workspace summary and stats on ContextService."""

    def test_get_workspace_summary(self):
        fs = FakeFilesystemContext()
        ctx = _make_ctx(filesystem_context=fs)
        svc = ContextService(ctx)
        assert svc.get_workspace_summary() == "Session summary text"

    def test_get_workspace_summary_disabled(self):
        ctx = _make_ctx(filesystem_context=None)
        svc = ContextService(ctx)
        assert "disabled" in svc.get_workspace_summary()

    def test_get_context_stats(self):
        fs = FakeFilesystemContext()
        ctx = _make_ctx(filesystem_context=fs)
        svc = ContextService(ctx)

        stats = svc.get_context_stats()
        assert stats["enabled"] is True
        assert stats["total_notes"] == 5

    def test_get_context_stats_disabled(self):
        ctx = _make_ctx(filesystem_context=None)
        svc = ContextService(ctx)
        stats = svc.get_context_stats()
        assert stats == {"enabled": False}


class TestContextServiceErrorHandling:
    """Verify graceful degradation when filesystem context raises."""

    def test_write_note_catches_exception(self):
        class BrokenFS:
            def write_note(self, *a, **kw):
                raise RuntimeError("disk full")

        ctx = _make_ctx(filesystem_context=BrokenFS())
        svc = ContextService(ctx)
        assert svc.write_note("key", "val") is None

    def test_search_context_catches_exception(self):
        class BrokenFS:
            def search_context(self, *a, **kw):
                raise RuntimeError("oops")

        ctx = _make_ctx(filesystem_context=BrokenFS())
        svc = ContextService(ctx)
        assert svc.search_context("*") == []

    def test_get_relevant_context_catches_exception(self):
        class BrokenFS:
            def get_relevant_context(self, *a, **kw):
                raise RuntimeError("oops")

        ctx = _make_ctx(filesystem_context=BrokenFS())
        svc = ContextService(ctx)
        assert svc.get_relevant_context("q") == ""

    def test_get_workspace_summary_catches_exception(self):
        class BrokenFS:
            def get_session_summary(self):
                raise RuntimeError("oops")

        ctx = _make_ctx(filesystem_context=BrokenFS())
        svc = ContextService(ctx)
        assert "Error" in svc.get_workspace_summary()

    def test_get_context_stats_catches_exception(self):
        class BrokenFS:
            def get_workspace_stats(self):
                raise RuntimeError("oops")

        ctx = _make_ctx(filesystem_context=BrokenFS())
        svc = ContextService(ctx)
        stats = svc.get_context_stats()
        assert stats["enabled"] is True
        assert "error" in stats
