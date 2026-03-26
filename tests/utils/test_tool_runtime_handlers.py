"""Tests for extracted tool runtime handler modules.

Verifies that tool_runtime_io, tool_runtime_web, and tool_runtime_workspace
export the expected handler functions and that those functions are reachable
through the ToolRuntime registry-driven dispatch.

These tests run under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""

import asyncio
import importlib
import importlib.machinery
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Handler tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs to avoid heavy omicverse.__init__
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
_ov_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = _ov_pkg

_utils_pkg = types.ModuleType("omicverse.utils")
_utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
_utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = _utils_pkg
_ov_pkg.utils = _utils_pkg

from omicverse.utils.ovagent import (  # noqa: E402
    ToolRuntime,
)
from omicverse.utils.ovagent import tool_runtime_io  # noqa: E402
from omicverse.utils.ovagent import tool_runtime_web  # noqa: E402
from omicverse.utils.ovagent import tool_runtime_workspace  # noqa: E402
from omicverse.utils.ovagent.tool_runtime import LEGACY_AGENT_TOOLS  # noqa: E402
from omicverse.utils.harness.runtime_state import runtime_state  # noqa: E402

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------

_SESSION_COUNTER = 0


def _unique_session_id() -> str:
    global _SESSION_COUNTER
    _SESSION_COUNTER += 1
    return f"test-handlers-{_SESSION_COUNTER}"


class _DummyExecutor:
    pass


class _DummyCtx:
    """Minimal AgentContext stub."""

    LEGACY_AGENT_TOOLS = LEGACY_AGENT_TOOLS

    def __init__(self, session_id: str = ""):
        self._session_id = session_id or _unique_session_id()
        self.model = "gpt-5.4"
        self.provider = "openai"
        self.endpoint = "https://api.openai.com/v1"
        self.api_key = None
        self._llm = None
        self._config = SimpleNamespace()
        self._security_config = SimpleNamespace()
        self._security_scanner = SimpleNamespace()
        self._filesystem_context = None
        self.skill_registry = None
        self._notebook_executor = None
        self._ov_runtime = None
        self._trace_store = None
        self._session_history = None
        self._context_compactor = None
        self._approval_handler = None
        self._reporter = MagicMock()
        self.last_usage = None
        self.last_usage_breakdown = {}
        self._last_run_trace = None
        self._active_run_id = ""
        self._web_session_id = ""
        self._managed_api_env = {}
        self._code_only_mode = False
        self._code_only_captured_code = ""
        self._code_only_captured_history = []
        self.use_notebook_execution = False
        self.enable_filesystem_context = False

    def _get_runtime_session_id(self) -> str:
        return self._session_id

    def _emit(self, level, message, category=""):
        return None

    def _get_harness_session_id(self) -> str:
        return self._session_id

    def _get_visible_agent_tools(self, *, allowed_names=None):
        return []

    def _get_loaded_tool_names(self):
        return []

    def _refresh_runtime_working_directory(self) -> str:
        return "."

    def _tool_blocked_in_plan_mode(self, tool_name):
        return False

    def _detect_repo_root(self, cwd=None):
        return None

    def _resolve_local_path(self, file_path, *, allow_relative=False):
        return Path(file_path)

    def _ensure_server_tool_mode(self, tool_name):
        return None

    def _request_interaction(self, payload):
        raise NotImplementedError

    def _request_tool_approval(self, tool_name, *, reason, payload):
        return None

    def _load_skill_guidance(self, slug):
        return f"guidance:{slug}"

    def _extract_python_code(self, text):
        return text

    def _extract_python_code_strict(self, text):
        return text

    def _gather_code_candidates(self, text):
        return [text]

    def _normalize_code_candidate(self, code):
        return code

    def _collect_static_registry_entries(self, query, max_entries=20):
        return []

    def _collect_runtime_registry_entries(self, query, max_entries=20):
        return []

    def _review_generated_code_lightweight(self, request, code, entries):
        return code

    def _contains_forbidden_scanpy_usage(self, code):
        return False

    def _rewrite_scanpy_calls_with_registry(self, code, entries):
        return code

    def _run_agentic_loop(self, *args, **kwargs):
        raise NotImplementedError

    def _build_agentic_system_prompt(self):
        return ""

    def _normalize_registry_entry_for_codegen(self, entry):
        return entry

    from contextlib import contextmanager

    @contextmanager
    def _temporary_api_keys(self):
        yield


# ---------------------------------------------------------------------------
# Module-level export tests
# ---------------------------------------------------------------------------


class TestIOModuleExports:
    """tool_runtime_io exports all expected handler functions."""

    EXPECTED = {
        "handle_read", "handle_edit", "handle_write",
        "handle_glob", "handle_grep", "handle_notebook_edit",
    }

    def test_all_handlers_exported(self):
        for name in self.EXPECTED:
            assert hasattr(tool_runtime_io, name)
            assert callable(getattr(tool_runtime_io, name))


class TestWebModuleExports:
    """tool_runtime_web exports all expected handler functions."""

    EXPECTED = {
        "handle_web_fetch", "handle_web_search", "handle_web_download",
    }

    def test_all_handlers_exported(self):
        for name in self.EXPECTED:
            assert hasattr(tool_runtime_web, name)
            assert callable(getattr(tool_runtime_web, name))


class TestWorkspaceModuleExports:
    """tool_runtime_workspace exports all expected handler functions."""

    EXPECTED = {
        "handle_create_task", "handle_get_task", "handle_list_tasks",
        "handle_task_output", "handle_task_stop", "handle_task_update",
        "handle_enter_plan_mode", "handle_exit_plan_mode",
        "handle_enter_worktree",
        "handle_skill", "handle_search_skills",
        "handle_list_mcp_resources", "handle_read_mcp_resource",
        "handle_ask_user_question",
    }

    def test_all_handlers_exported(self):
        for name in self.EXPECTED:
            assert hasattr(tool_runtime_workspace, name)
            assert callable(getattr(tool_runtime_workspace, name))


# ---------------------------------------------------------------------------
# Dispatch-through tests: extracted handlers reachable via ToolRuntime
# ---------------------------------------------------------------------------


class TestDispatchThroughIO:
    """IO tool dispatch still works through ToolRuntime after extraction."""

    def test_dispatch_inspect_data_no_adata(self):
        rt = ToolRuntime(_DummyCtx(), _DummyExecutor())
        tc = SimpleNamespace(name="inspect_data", arguments={"aspect": "shape"})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert "No dataset" in result

    def test_registry_resolves_io_handler_keys(self):
        rt = ToolRuntime(_DummyCtx(), _DummyExecutor())
        for canonical_name in ["Read", "Edit", "Write", "Glob", "Grep", "NotebookEdit"]:
            assert rt.registry.get_handler(canonical_name) is not None, (
                f"IO handler for '{canonical_name}' not bound"
            )


class TestDispatchThroughWeb:
    """Web tool dispatch still works through ToolRuntime after extraction."""

    def test_registry_resolves_web_handler_keys(self):
        rt = ToolRuntime(_DummyCtx(), _DummyExecutor())
        for canonical_name in ["WebFetch", "WebSearch", "web_download"]:
            assert rt.registry.get_handler(canonical_name) is not None, (
                f"Web handler for '{canonical_name}' not bound"
            )


class TestDispatchThroughWorkspace:
    """Workspace tool dispatch still works through ToolRuntime after extraction."""

    def test_registry_resolves_workspace_handler_keys(self):
        rt = ToolRuntime(_DummyCtx(), _DummyExecutor())
        # Use canonical names for registry lookup
        for canonical_name in [
            "TaskCreate", "TaskGet", "TaskList", "TaskOutput",
            "TaskStop", "TaskUpdate",
            "EnterPlanMode", "ExitPlanMode", "EnterWorktree",
            "Skill", "ListMcpResourcesTool", "ReadMcpResourceTool",
        ]:
            assert rt.registry.get_handler(canonical_name) is not None, (
                f"Workspace handler for '{canonical_name}' not bound"
            )

    def test_dispatch_finish_still_works(self):
        rt = ToolRuntime(_DummyCtx(), _DummyExecutor())
        tc = SimpleNamespace(name="finish", arguments={"summary": "done"})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert result == {"finished": True, "summary": "done"}

    def test_dispatch_plan_mode_via_workspace(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        sid = ctx._session_id
        # Enter plan mode through dispatch
        tc = SimpleNamespace(name="EnterPlanMode", arguments={"reason": "test"})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        parsed = json.loads(result)
        assert parsed.get("enabled") is True
        # Exit plan mode
        tc = SimpleNamespace(name="ExitPlanMode", arguments={"summary": "done"})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        parsed = json.loads(result)
        assert parsed.get("enabled") is False

    def test_dispatch_task_lifecycle_via_workspace(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        # Create
        tc = SimpleNamespace(
            name="TaskCreate",
            arguments={"title": "test task", "description": "desc"},
        )
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        parsed = json.loads(result)
        task_id = parsed.get("task_id", "")
        assert task_id
        # Get
        tc = SimpleNamespace(name="TaskGet", arguments={"task_id": task_id})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        parsed = json.loads(result)
        assert parsed.get("title") == "test task"
        # List
        tc = SimpleNamespace(name="TaskList", arguments={})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        parsed = json.loads(result)
        assert isinstance(parsed.get("tasks"), list)

    def test_dispatch_mcp_not_configured(self):
        rt = ToolRuntime(_DummyCtx(), _DummyExecutor())
        tc = SimpleNamespace(
            name="ListMcpResourcesTool",
            arguments={"server": ""},
        )
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        parsed = json.loads(result)
        assert parsed.get("available") is False


# ---------------------------------------------------------------------------
# Direct handler function tests
# ---------------------------------------------------------------------------


class TestWebHandlersDirect:
    """Web handlers are pure functions — test directly without ToolRuntime."""

    def test_web_search_bad_query_returns_string(self):
        # A real web call may fail in CI; verify it returns a string
        result = tool_runtime_web.handle_web_search("", num_results=1)
        assert isinstance(result, str)

    def test_web_download_bad_url_returns_error(self):
        result = tool_runtime_web.handle_web_download(
            "http://localhost:99999/noexist.zip"
        )
        assert "error" in result.lower() or "Error" in result


class TestWorkspaceHandlersDirect:
    """Workspace handlers tested with minimal context double."""

    def test_handle_create_and_get_task(self):
        ctx = _DummyCtx()
        result = tool_runtime_workspace.handle_create_task(
            ctx, "my task", description="desc"
        )
        parsed = json.loads(result)
        task_id = parsed["task_id"]
        result2 = tool_runtime_workspace.handle_get_task(ctx, task_id)
        parsed2 = json.loads(result2)
        assert parsed2["title"] == "my task"

    def test_handle_enter_exit_plan_mode(self):
        ctx = _DummyCtx()
        result = tool_runtime_workspace.handle_enter_plan_mode(ctx, reason="test")
        parsed = json.loads(result)
        assert parsed["enabled"] is True
        result2 = tool_runtime_workspace.handle_exit_plan_mode(ctx, summary="done")
        parsed2 = json.loads(result2)
        assert parsed2["enabled"] is False

    def test_handle_list_mcp_not_configured(self):
        result = tool_runtime_workspace.handle_list_mcp_resources()
        parsed = json.loads(result)
        assert parsed["available"] is False

    def test_handle_skill_load_mode(self):
        ctx = _DummyCtx()
        result = tool_runtime_workspace.handle_skill(ctx, "test-skill", mode="load")
        assert result == "guidance:test-skill"

    def test_handle_search_skills_no_registry(self):
        ctx = _DummyCtx()
        ctx.skill_registry = None
        result = tool_runtime_workspace.handle_search_skills(ctx, "anything")
        assert "No domain skills" in result


# ---------------------------------------------------------------------------
# Facade slimming verification
# ---------------------------------------------------------------------------


class TestFacadeSlimming:
    """ToolRuntime no longer contains IO/web/workspace implementations."""

    IO_REMOVED = {
        "_tool_read", "_tool_edit", "_tool_write",
        "_tool_glob", "_tool_grep", "_tool_notebook_edit",
    }
    WEB_REMOVED = {
        "_tool_web_fetch", "_tool_web_search", "_tool_web_download",
    }
    WORKSPACE_REMOVED = {
        "_tool_create_task", "_tool_get_task", "_tool_list_tasks",
        "_tool_task_output", "_tool_task_stop", "_tool_task_update",
        "_tool_enter_plan_mode", "_tool_exit_plan_mode",
        "_tool_enter_worktree", "_tool_skill", "_tool_search_skills",
        "_tool_list_mcp_resources", "_tool_read_mcp_resource",
        "_tool_ask_user_question",
    }

    def test_io_methods_removed(self):
        for name in self.IO_REMOVED:
            assert not hasattr(ToolRuntime, name), (
                f"ToolRuntime still has '{name}'"
            )

    def test_web_methods_removed(self):
        for name in self.WEB_REMOVED:
            assert not hasattr(ToolRuntime, name), (
                f"ToolRuntime still has '{name}'"
            )

    def test_workspace_methods_removed(self):
        for name in self.WORKSPACE_REMOVED:
            assert not hasattr(ToolRuntime, name), (
                f"ToolRuntime still has '{name}'"
            )

    def test_facade_still_has_execution_methods(self):
        kept = {
            "_tool_execute_code", "_tool_run_snippet", "_tool_bash",
            "_tool_finish", "_tool_tool_search", "_tool_inspect_data",
            "_tool_search_functions",
        }
        for name in kept:
            assert hasattr(ToolRuntime, name), (
                f"ToolRuntime should still have '{name}'"
            )

    def test_all_handler_keys_still_bound(self):
        rt = ToolRuntime(_DummyCtx(), _DummyExecutor())
        unresolved = rt.registry.validate_handlers()
        assert unresolved == [], f"Unresolved handler keys: {unresolved}"

    def test_dispatch_tool_still_works(self):
        rt = ToolRuntime(_DummyCtx(), _DummyExecutor())
        tc = SimpleNamespace(name="finish", arguments={"summary": "ok"})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert result["finished"] is True

    def test_tool_runtime_line_count_reduced(self):
        """Verify tool_runtime.py shrank significantly."""
        src = Path(tool_runtime_io.__file__).parent / "tool_runtime.py"
        line_count = len(src.read_text().splitlines())
        # Before extraction: ~1791 lines; after: should be well under 1200
        assert line_count < 1200, (
            f"tool_runtime.py is still {line_count} lines — "
            f"expected significant reduction from extraction"
        )
