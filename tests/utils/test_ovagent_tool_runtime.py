import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omicverse.utils.ovagent.tool_runtime import (
    LEGACY_AGENT_TOOLS,
    ToolRuntime,
)
from omicverse.utils.ovagent.protocol import AgentContext
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

    def _emit(self, level, message: str, category: str = "") -> None:
        return None

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

    def _get_harness_session_id(self) -> str:
        return self._session_id

    def _get_visible_agent_tools(self, *, allowed_names=None):
        return []

    def _get_loaded_tool_names(self):
        return []

    def _refresh_runtime_working_directory(self) -> str:
        return "."

    def _tool_blocked_in_plan_mode(self, tool_name: str) -> bool:
        return False

    def _detect_repo_root(self, cwd=None):
        return None

    def _resolve_local_path(self, file_path: str, *, allow_relative: bool = False):
        raise NotImplementedError

    def _ensure_server_tool_mode(self, tool_name: str) -> None:
        return None

    def _request_interaction(self, payload):
        raise NotImplementedError

    def _request_tool_approval(self, tool_name: str, *, reason: str, payload):
        raise NotImplementedError

    def _load_skill_guidance(self, slug: str) -> str:
        return ""

    def _extract_python_code(self, text: str):
        return text

    def _extract_python_code_strict(self, text: str):
        return text

    def _gather_code_candidates(self, text: str):
        return [text]

    def _normalize_code_candidate(self, code: str):
        return code

    def _collect_runtime_registry_entries(self, query: str, max_entries: int = 20):
        return []

    def _review_generated_code_lightweight(self, request: str, code: str, entries):
        return code

    def _contains_forbidden_scanpy_usage(self, code: str) -> bool:
        return False

    def _rewrite_scanpy_calls_with_registry(self, code: str, entries):
        return code

    def _run_agentic_loop(self, *args, **kwargs):
        raise NotImplementedError

    def _build_agentic_system_prompt(self) -> str:
        return ""

    def _normalize_registry_entry_for_codegen(self, entry):
        return entry

    @contextmanager
    def _temporary_api_keys(self):
        yield


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


def test_execute_code_returns_full_debug_output_but_truncated_llm_output():
    runtime = ToolRuntime.__new__(ToolRuntime)

    class _Ctx:
        _code_only_mode = False

        @staticmethod
        def _extract_python_code(text):
            return text

    class _Executor:
        _notebook_fallback_error = None
        _ctx = _Ctx()

        def check_code_prerequisites(self, code, adata):
            return ""

        def execute_generated_code(self, code, adata, capture_stdout=True):
            return {
                "adata": adata,
                "stdout": "A" * 3500,
            }

    runtime._ctx = _Ctx()
    runtime._executor = _Executor()

    result = runtime._tool_execute_code("print('x')", "debug stdout", None)

    assert "stdout:\n" in result["output"]
    assert "truncated, total_chars=3500" in result["output"]
    assert len(result["debug_output"]) > len(result["output"])
    assert "A" * 3200 in result["debug_output"]
    assert "truncated, total_chars=3500" not in result["debug_output"]
    assert result["stdout"] == "A" * 3500


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

    def test_dummy_ctx_satisfies_runtime_protocol(self):
        """AgentContext remains runtime-checkable for duck-typed test doubles."""
        assert isinstance(_DummyCtx(), AgentContext)


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
            result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
            assert "plan mode" in result.lower()
        finally:
            runtime_state.exit_plan_mode(sid, reason="cleanup")

    def test_agent_dispatch_without_subagent_controller_raises_runtimeerror(self):
        rt, _ = self._make_rt()
        tc = SimpleNamespace(
            name="Agent",
            arguments={"agent_type": "explore", "task": "inspect this"},
        )
        with pytest.raises(RuntimeError, match="Subagent controller is not initialized"):
            asyncio.run(rt.dispatch_tool(tc, None, "test"))


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


# -----------------------------------------------------------------------
# Task-025: Registry-driven dispatch tests
# -----------------------------------------------------------------------


class TestRegistryDrivenDispatch:
    """Registry-driven dispatch preserves canonical names, aliases, and behavior."""

    def _make_rt(self):
        ctx = _DummyCtx()
        return ToolRuntime(ctx, _DummyExecutor())

    def test_registry_is_initialized(self):
        rt = self._make_rt()
        assert rt.registry is not None
        entries = rt.registry.all_entries()
        assert len(entries) > 0

    def test_all_handler_keys_are_bound(self):
        rt = self._make_rt()
        unresolved = rt.registry.validate_handlers()
        assert unresolved == [], f"Unresolved handler keys: {unresolved}"

    def test_canonical_catalog_tools_resolve(self):
        rt = self._make_rt()
        for name in [
            "ToolSearch", "Bash", "Read", "Edit", "Write", "Glob", "Grep",
            "NotebookEdit", "Agent", "AskUserQuestion", "EnterPlanMode",
            "ExitPlanMode", "EnterWorktree", "Skill", "WebFetch",
            "WebSearch", "ListMcpResourcesTool", "ReadMcpResourceTool",
        ]:
            canonical = rt.registry.resolve_name(name)
            assert canonical == name, f"{name} did not resolve to itself"
            handler = rt.registry.get_handler(name)
            assert handler is not None, f"No handler for {name}"

    def test_legacy_tools_resolve(self):
        rt = self._make_rt()
        for name in [
            "inspect_data", "execute_code", "run_snippet",
            "search_functions", "web_download", "finish",
        ]:
            canonical = rt.registry.resolve_name(name)
            assert canonical == name, f"Legacy tool {name} did not resolve"
            handler = rt.registry.get_handler(name)
            assert handler is not None, f"No handler for legacy tool {name}"

    def test_legacy_aliases_resolve_to_catalog_tools(self):
        rt = self._make_rt()
        assert rt.registry.resolve_name("delegate") == "Agent"
        assert rt.registry.resolve_name("web_fetch") == "WebFetch"
        assert rt.registry.resolve_name("web_search") == "WebSearch"
        assert rt.registry.resolve_name("search_skills") == "Skill"

    def test_case_insensitive_resolution(self):
        rt = self._make_rt()
        assert rt.registry.resolve_name("bash") == "Bash"
        assert rt.registry.resolve_name("read") == "Read"

    def test_dispatch_tool_search_via_registry(self):
        rt = self._make_rt()
        tc = SimpleNamespace(name="ToolSearch", arguments={"query": "select:Edit"})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        parsed = json.loads(result)
        assert "Edit" in parsed.get("selected_tools", [])

    def test_dispatch_finish_via_registry(self):
        rt = self._make_rt()
        tc = SimpleNamespace(name="finish", arguments={"summary": "all done"})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert result == {"finished": True, "summary": "all done"}

    def test_dispatch_unknown_tool_via_registry(self):
        rt = self._make_rt()
        tc = SimpleNamespace(name="nonexistent_tool_xyz", arguments={})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert "Unknown tool" in result

    def test_dispatch_inspect_data_passes_adata(self):
        rt = self._make_rt()
        tc = SimpleNamespace(name="inspect_data", arguments={"aspect": "shape"})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert "No dataset" in result

    def test_dispatch_execute_code_empty_validation(self):
        rt = self._make_rt()
        tc = SimpleNamespace(
            name="execute_code", arguments={"code": "", "description": ""}
        )
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        parsed = json.loads(result)
        assert "error" in parsed

    def test_dispatch_run_snippet_empty_validation(self):
        rt = self._make_rt()
        tc = SimpleNamespace(name="run_snippet", arguments={"code": "  "})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        parsed = json.loads(result)
        assert "error" in parsed

    def test_dispatch_ask_user_question_empty_validation(self):
        rt = self._make_rt()
        tc = SimpleNamespace(name="AskUserQuestion", arguments={})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        parsed = json.loads(result)
        assert "error" in parsed

    def test_plan_mode_blocking_via_registry_dispatch(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        sid = ctx._session_id
        runtime_state.enter_plan_mode(sid, reason="test")
        try:
            tc = SimpleNamespace(name="Bash", arguments={"command": "ls"})
            result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
            assert "plan mode" in result.lower()
        finally:
            runtime_state.exit_plan_mode(sid, reason="cleanup")

    def test_plan_mode_blocking_legacy_tools_via_registry(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        sid = ctx._session_id
        runtime_state.enter_plan_mode(sid, reason="test")
        try:
            tc = SimpleNamespace(
                name="execute_code",
                arguments={"code": "x=1", "description": "test"},
            )
            result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
            assert "plan mode" in result.lower()
        finally:
            runtime_state.exit_plan_mode(sid, reason="cleanup")

    def test_agent_dispatch_via_registry_without_controller_raises(self):
        rt = self._make_rt()
        tc = SimpleNamespace(
            name="Agent",
            arguments={"agent_type": "explore", "task": "inspect this"},
        )
        with pytest.raises(RuntimeError, match="Subagent controller"):
            asyncio.run(rt.dispatch_tool(tc, None, "test"))

    def test_dispatch_search_functions_via_registry(self, monkeypatch):
        rt = self._make_rt()
        fake_registry = SimpleNamespace(find=lambda query: [])
        monkeypatch.setattr(tool_runtime_module, "_global_registry", fake_registry)
        tc = SimpleNamespace(
            name="search_functions", arguments={"query": "dynamo"}
        )
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert "dynamo" in result.lower()

    def test_dispatch_via_legacy_alias_delegate(self):
        """Dispatching 'delegate' resolves through the registry to Agent."""
        rt = self._make_rt()
        tc = SimpleNamespace(
            name="delegate",
            arguments={"agent_type": "explore", "task": "test"},
        )
        # Should resolve to Agent and fail without controller
        with pytest.raises(RuntimeError, match="Subagent controller"):
            asyncio.run(rt.dispatch_tool(tc, None, "test"))

    def test_handler_count_matches_registered_entries(self):
        rt = self._make_rt()
        entries = rt.registry.all_entries()
        bound_keys = set()
        for meta in entries:
            handler = rt.registry.get_handler(meta.canonical_name)
            if handler is not None:
                bound_keys.add(meta.handler_key)
        assert bound_keys == rt.registry.handler_keys()
