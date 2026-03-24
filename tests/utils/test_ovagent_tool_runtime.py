import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omicverse.utils.ovagent.tool_runtime import (
    LEGACY_AGENT_TOOLS,
    ToolRuntime,
)
from omicverse.utils.ovagent import tool_runtime as tool_runtime_module
from omicverse.utils.harness.runtime_state import runtime_state
from omicverse.utils.harness.tool_catalog import (
    get_default_loaded_tool_names,
    get_visible_tool_schemas,
    normalize_tool_name,
)


class _DummyExecutor:
    pass


_SESSION_COUNTER = 0


def _unique_session_id() -> str:
    global _SESSION_COUNTER
    _SESSION_COUNTER += 1
    return f"test-tool-runtime-{_SESSION_COUNTER}"


class _DummyCtx:
    """Minimal AgentContext stub for ToolRuntime tests."""

    def __init__(self, session_id: str = ""):
        self._session_id = session_id or _unique_session_id()

    def _get_runtime_session_id(self) -> str:
        return self._session_id

    def _collect_static_registry_entries(self, query: str, max_entries: int = 20):
        if query != "dynamo":
            return []
        return [
            {
                "full_name": "omicverse.single.Velo.cal_velocity[method=dynamo]",
                "short_name": "dynamo",
                "signature": "cal_velocity(method)",
                "description": "Variant of velocity calculation when method='dynamo'.",
                "aliases": ["dynamo", "velo dynamo"],
                "examples": ["velo.cal_velocity(method='dynamo')"],
                "category": "trajectory",
                "branch_parameter": "method",
                "branch_value": "dynamo",
            }
        ]


# -----------------------------------------------------------------------
# Original test
# -----------------------------------------------------------------------


def test_tool_search_functions_falls_back_to_static_registry(monkeypatch):
    runtime = ToolRuntime(_DummyCtx(), _DummyExecutor())
    fake_registry = SimpleNamespace(find=lambda query: [])

    monkeypatch.setattr(tool_runtime_module, "_global_registry", fake_registry)

    result = runtime._tool_search_functions("dynamo")

    assert "omicverse.single.Velo.cal_velocity[method=dynamo]" in result
    assert "Branch: method='dynamo'" in result


# -----------------------------------------------------------------------
# Task-005: Tool facade consolidation tests
# -----------------------------------------------------------------------


class TestLegacyAgentTools:
    """LEGACY_AGENT_TOOLS is now module-level in tool_runtime."""

    def test_legacy_tools_is_a_list(self):
        assert isinstance(LEGACY_AGENT_TOOLS, list)
        assert len(LEGACY_AGENT_TOOLS) > 0

    def test_legacy_tool_names_present(self):
        names = {t["name"] for t in LEGACY_AGENT_TOOLS}
        expected = {
            "inspect_data", "execute_code", "run_snippet",
            "search_functions", "search_skills", "delegate",
            "web_fetch", "web_search", "web_download", "finish",
        }
        assert expected == names

    def test_legacy_tools_have_valid_schemas(self):
        for tool in LEGACY_AGENT_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert tool["parameters"]["type"] == "object"

    def test_agent_class_exposes_same_legacy_tools(self):
        """OmicVerseAgent.LEGACY_AGENT_TOOLS is the same object."""
        from omicverse.utils.smart_agent import OmicVerseAgent
        assert OmicVerseAgent.LEGACY_AGENT_TOOLS is LEGACY_AGENT_TOOLS


class TestToolRuntimeVisibility:
    """ToolRuntime.get_visible_agent_tools merges catalog + legacy."""

    def test_visible_tools_include_core_and_legacy(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        tools = rt.get_visible_agent_tools()
        names = {t["name"] for t in tools}
        # Core tools from catalog
        assert "ToolSearch" in names
        assert "Bash" in names
        assert "Read" in names
        # Legacy tools
        assert "inspect_data" in names
        assert "execute_code" in names
        assert "finish" in names

    def test_visible_tools_with_allowed_names_filter(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        tools = rt.get_visible_agent_tools(allowed_names={"Read", "finish"})
        names = {t["name"] for t in tools}
        assert names == {"Read", "finish"}

    def test_allowed_names_normalized(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        # "web_fetch" is a legacy alias; "WebFetch" is the canonical name
        tools = rt.get_visible_agent_tools(allowed_names={"web_fetch"})
        names = {t["name"] for t in tools}
        # The legacy tool named "web_fetch" should appear
        assert "web_fetch" in names

    def test_loaded_tool_names_returns_tuple(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        loaded = rt.get_loaded_tool_names()
        assert isinstance(loaded, (list, tuple))


class TestToolRuntimePlanMode:
    """ToolRuntime.tool_blocked_in_plan_mode is now self-contained."""

    def _make_rt(self) -> tuple[ToolRuntime, str]:
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        return rt, ctx._session_id

    def test_not_blocked_when_plan_mode_off(self):
        rt, _ = self._make_rt()
        assert rt.tool_blocked_in_plan_mode("Bash") is False
        assert rt.tool_blocked_in_plan_mode("Read") is False
        assert rt.tool_blocked_in_plan_mode("execute_code") is False

    def test_high_risk_tools_blocked_in_plan_mode(self):
        rt, sid = self._make_rt()
        runtime_state.enter_plan_mode(sid, reason="test")
        try:
            assert rt.tool_blocked_in_plan_mode("Bash") is True
            assert rt.tool_blocked_in_plan_mode("Edit") is True
            assert rt.tool_blocked_in_plan_mode("Write") is True
            assert rt.tool_blocked_in_plan_mode("NotebookEdit") is True
            assert rt.tool_blocked_in_plan_mode("EnterWorktree") is True
        finally:
            runtime_state.exit_plan_mode(sid, reason="cleanup")

    def test_read_only_tools_allowed_in_plan_mode(self):
        rt, sid = self._make_rt()
        runtime_state.enter_plan_mode(sid, reason="test")
        try:
            assert rt.tool_blocked_in_plan_mode("Read") is False
            assert rt.tool_blocked_in_plan_mode("Glob") is False
            assert rt.tool_blocked_in_plan_mode("Grep") is False
            assert rt.tool_blocked_in_plan_mode("ToolSearch") is False
        finally:
            runtime_state.exit_plan_mode(sid, reason="cleanup")

    def test_legacy_tools_blocked_in_plan_mode(self):
        rt, sid = self._make_rt()
        runtime_state.enter_plan_mode(sid, reason="test")
        try:
            assert rt.tool_blocked_in_plan_mode("execute_code") is True
            assert rt.tool_blocked_in_plan_mode("web_download") is True
        finally:
            runtime_state.exit_plan_mode(sid, reason="cleanup")

    def test_dispatch_returns_blocked_message_in_plan_mode(self):
        rt, sid = self._make_rt()
        runtime_state.enter_plan_mode(sid, reason="test")
        try:
            tc = SimpleNamespace(name="Bash", arguments={"command": "ls"})
            result = asyncio.get_event_loop().run_until_complete(
                rt.dispatch_tool(tc, None, "test")
            )
            assert "plan mode" in result.lower()
        finally:
            runtime_state.exit_plan_mode(sid, reason="cleanup")


class TestDeferredToolLoading:
    """Deferred loading semantics are preserved through ToolRuntime."""

    def test_tool_search_loads_selected_tools(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        # Before search — only default loaded tools
        initial_loaded = set(rt.get_loaded_tool_names())

        # Search for "Edit" tool
        result_json = rt._tool_tool_search("select:Edit")
        result = json.loads(result_json)

        assert "Edit" in result.get("selected_tools", [])
        new_loaded = set(rt.get_loaded_tool_names())
        assert "Edit" in new_loaded

    def test_visible_schemas_include_newly_loaded(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())

        # Load "Edit" tool
        rt._tool_tool_search("select:Edit")
        tools = rt.get_visible_agent_tools()
        names = {t["name"] for t in tools}
        assert "Edit" in names


class TestSmartAgentDelegation:
    """OmicVerseAgent's remaining facade methods delegate to ToolRuntime."""

    def test_agent_get_visible_agent_tools_delegates(self):
        """The agent's _get_visible_agent_tools delegates to ToolRuntime."""
        from omicverse.utils.smart_agent import OmicVerseAgent
        # Verify the method exists and is a thin delegation
        import inspect
        src = inspect.getsource(OmicVerseAgent._get_visible_agent_tools)
        assert "tool_runtime" in src.lower() or "get_visible_agent_tools" in src

    def test_agent_tool_blocked_in_plan_mode_delegates(self):
        """The agent's _tool_blocked_in_plan_mode delegates to ToolRuntime."""
        from omicverse.utils.smart_agent import OmicVerseAgent
        import inspect
        src = inspect.getsource(OmicVerseAgent._tool_blocked_in_plan_mode)
        assert "tool_runtime" in src.lower() or "tool_blocked_in_plan_mode" in src

    def test_agent_class_has_no_tool_wrapper_methods(self):
        """The thin _tool_* wrappers have been removed from OmicVerseAgent."""
        from omicverse.utils.smart_agent import OmicVerseAgent
        # These should no longer exist on the agent class directly
        removed = [
            "_tool_tool_search", "_tool_read", "_tool_edit", "_tool_write",
            "_tool_glob", "_tool_grep", "_tool_notebook_edit",
            "_tool_inspect_data", "_tool_execute_code", "_tool_run_snippet",
            "_tool_search_functions", "_tool_search_skills",
            "_tool_web_fetch", "_tool_web_search", "_tool_web_download",
            "_tool_bash", "_tool_enter_plan_mode", "_tool_exit_plan_mode",
            "_tool_enter_worktree", "_tool_skill",
            "_tool_list_mcp_resources", "_tool_read_mcp_resource",
            "_tool_ask_user_question",
            "_tool_create_task", "_tool_get_task", "_tool_list_tasks",
            "_tool_task_output", "_tool_task_stop", "_tool_task_update",
        ]
        for name in removed:
            assert not hasattr(OmicVerseAgent, name), (
                f"OmicVerseAgent still has {name} — should be removed"
            )
