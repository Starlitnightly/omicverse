"""Integration tests: subagent isolation boundaries.

Validates that SubagentController's isolation guarantees hold at the
integration level.  Each test class targets a specific isolation boundary
documented in the SubagentRuntime / SubagentController docstrings:

- Usage tracking isolation (parent vs. subagent ``last_usage``)
- Tool schema snapshotting (frozen at creation, not live)
- Permission policy enforcement (denied tools never reach dispatch)
- adata mutation gating (``can_mutate_adata`` flag)
- Budget manager isolation (subagent-local, not shared)
- Max turns enforcement (subagent exits after ``max_turns``)

Uses the shared harness fixtures from ``tests.integration`` and
requires the ``OV_AGENT_RUN_HARNESS_TESTS=1`` environment gate.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, FrozenSet, List, Optional, Set

import pytest

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Subagent isolation integration tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# -- harness imports ----------------------------------------------------------
from tests.integration.fakes import (
    ChatResponse,
    FakeLLM,
    FakeToolRegistry,
    ToolCall,
    Usage,
)
from tests.integration.helpers import (
    build_fake_llm,
    make_chat_response,
    make_tool_call,
    make_usage,
)

# -- production imports -------------------------------------------------------
from omicverse.utils.agent_config import AgentConfig, SubagentConfig
from omicverse.utils.ovagent.context_budget import ContextBudgetManager
from omicverse.utils.ovagent.permission_policy import (
    PermissionDecision,
    PermissionVerdict,
)
from omicverse.utils.ovagent.subagent_controller import (
    SubagentController,
    SubagentRuntime,
)
from omicverse.utils.ovagent.tool_registry import (
    ApprovalClass,
    IsolationMode,
    OutputTier,
)


# ===================================================================
#  Test doubles — minimal fakes scoped to SubagentController needs
# ===================================================================


class FakeAgentContext:
    """Minimal AgentContext protocol implementation for SubagentController.

    Exposes the exact surface that ``SubagentController.__init__`` and
    ``run_subagent`` touch: ``_llm``, ``_config``, ``model``,
    ``last_usage``, and ``_get_visible_agent_tools``.
    """

    def __init__(
        self,
        llm: FakeLLM,
        config: AgentConfig,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._llm = llm
        self._config = config
        self.model = "fake-model"
        self.last_usage: Any = None
        self._tool_schemas = tool_schemas or []

    def _get_visible_agent_tools(
        self, *, allowed_names: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        if allowed_names:
            return [s for s in self._tool_schemas if s.get("name") in allowed_names]
        return list(self._tool_schemas)


class FakePromptBuilder:
    """Minimal PromptBuilder that returns static strings."""

    def build_subagent_system_prompt(self, agent_type: str, context: str) -> str:
        return f"You are a {agent_type} subagent."

    def build_subagent_user_message(self, task: str, adata: Any) -> str:
        return f"Task: {task}"


class FakeSubagentToolRuntime:
    """Fake tool runtime with ``dispatch_tool`` matching the real interface.

    The SubagentController calls ``self._tool_runtime.dispatch_tool(tc,
    working_adata, task, permission_policy=...)``, and also accesses
    ``self._tool_runtime.registry``.  This fake satisfies both.

    Parameters
    ----------
    handlers : dict
        Mapping of tool name to a callable ``(tool_call, adata, task) -> Any``.
    registry_tools : dict
        Tool name -> dict of attributes for ``FakeToolRegistry.get()`` results.
        Must include ``output_tier`` for the budget-manager truncation path.
    """

    # Default attributes that PermissionPolicy.check() accesses on
    # registry.get() results (steps 4-5 in the resolution chain).
    _REGISTRY_DEFAULTS: Dict[str, Any] = {
        "approval_class": ApprovalClass.allow,
        "isolation_mode": IsolationMode.none,
        "output_tier": OutputTier.standard,
    }

    def __init__(
        self,
        handlers: Optional[Dict[str, Any]] = None,
        registry_tools: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._handlers = dict(handlers or {})
        # Ensure every registry entry has the attributes PermissionPolicy needs.
        enriched: Dict[str, Dict[str, Any]] = {}
        for name, attrs in (registry_tools or {}).items():
            merged = dict(self._REGISTRY_DEFAULTS)
            merged.update(attrs)
            enriched[name] = merged
        self.registry = FakeToolRegistry(enriched)
        self.dispatch_calls: List[Dict[str, Any]] = []

    async def dispatch_tool(
        self,
        tool_call: Any,
        current_adata: Any,
        request: str,
        *,
        permission_policy: Any = None,
    ) -> Any:
        self.dispatch_calls.append({
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "adata": current_adata,
            "request": request,
        })
        handler = self._handlers.get(tool_call.name)
        if handler is None:
            return f"[FakeSubagentToolRuntime] unknown tool: {tool_call.name}"
        return handler(tool_call, current_adata, request)


# -- helper builders ----------------------------------------------------------


def _make_tool_schema(name: str, description: str = "") -> Dict[str, Any]:
    """Build a minimal tool-schema dict."""
    return {
        "name": name,
        "description": description or f"{name} tool",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def _build_controller(
    llm: FakeLLM,
    tool_runtime: FakeSubagentToolRuntime,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    config: Optional[AgentConfig] = None,
) -> SubagentController:
    """Wire up a SubagentController with all test doubles."""
    cfg = config or AgentConfig()
    ctx = FakeAgentContext(llm, cfg, tool_schemas=tool_schemas or [])
    prompt_builder = FakePromptBuilder()
    return SubagentController(ctx, prompt_builder, tool_runtime)


# ===================================================================
#  Test: usage tracking isolation
# ===================================================================


class TestSubagentUsageIsolation:
    """Parent ``last_usage`` must NOT be mutated by a subagent run."""

    @pytest.mark.asyncio
    async def test_parent_usage_untouched_by_subagent(self):
        """Run a subagent that records usage; verify parent is unmodified."""
        parent_usage = make_usage(100, 200, model="parent-model")

        # LLM returns a plain text response (no tool calls -> single turn)
        subagent_usage = make_usage(5, 10, model="fake-model")
        llm = build_fake_llm([
            make_chat_response(content="Subagent done.", usage=subagent_usage),
        ])

        tool_runtime = FakeSubagentToolRuntime()
        cfg = AgentConfig()
        ctx = FakeAgentContext(llm, cfg)
        ctx.last_usage = parent_usage  # set parent's usage

        controller = SubagentController(ctx, FakePromptBuilder(), tool_runtime)

        result = await controller.run_subagent(
            agent_type="explore",
            task="Look at the data",
            adata=None,
        )

        # Parent usage must remain the original object, untouched
        assert ctx.last_usage is parent_usage
        assert ctx.last_usage.input_tokens == 100
        assert ctx.last_usage.output_tokens == 200

        # Returned usage comes from the subagent runtime
        assert result["last_usage"] is subagent_usage
        assert result["last_usage"].input_tokens == 5

    @pytest.mark.asyncio
    async def test_subagent_usage_returned_in_result(self):
        """Verify the result dict carries the subagent's last_usage."""
        usage_turn1 = make_usage(1, 2)
        usage_turn2 = make_usage(3, 4)

        tc = make_tool_call("inspect_data", {"aspect": "shape"})
        llm = build_fake_llm([
            make_chat_response(tool_calls=[tc], usage=usage_turn1),
            make_chat_response(content="Done.", usage=usage_turn2),
        ])

        tool_runtime = FakeSubagentToolRuntime(
            handlers={"inspect_data": lambda tc, ad, req: "shape: (100, 5)"},
            registry_tools={"inspect_data": {"output_tier": OutputTier.standard}},
        )
        schemas = [_make_tool_schema("inspect_data"), _make_tool_schema("finish")]
        controller = _build_controller(llm, tool_runtime, tool_schemas=schemas)

        result = await controller.run_subagent(
            agent_type="explore", task="inspect", adata=None,
        )

        # last_usage should be from the final LLM turn
        assert result["last_usage"] is usage_turn2


# ===================================================================
#  Test: tool schema snapshotting
# ===================================================================


class TestSubagentToolSchemaSnapshot:
    """Subagent tool schemas are frozen at creation time."""

    @pytest.mark.asyncio
    async def test_schema_snapshot_isolated_from_parent_mutations(self):
        """Mutating parent schemas after subagent creation has no effect."""
        schema_a = _make_tool_schema("inspect_data")
        schema_b = _make_tool_schema("finish")
        parent_schemas = [schema_a, schema_b]

        llm = build_fake_llm([
            make_chat_response(content="Done."),
        ])

        tool_runtime = FakeSubagentToolRuntime()
        cfg = AgentConfig()
        ctx = FakeAgentContext(llm, cfg, tool_schemas=parent_schemas)
        prompt_builder = FakePromptBuilder()
        controller = SubagentController(ctx, prompt_builder, tool_runtime)

        # Create a runtime directly to inspect the snapshot
        explore_config = cfg.get_subagent_config("explore")
        runtime = controller._create_subagent_runtime(
            agent_type="explore",
            allowed_tools=frozenset(explore_config.allowed_tools),
            max_turns=explore_config.max_turns,
        )

        # Snapshot should contain both schemas (they are in explore's allowed list)
        snapshot_names = {s["name"] for s in runtime.tool_schemas}
        assert "inspect_data" in snapshot_names
        assert "finish" in snapshot_names
        original_count = len(runtime.tool_schemas)

        # Now mutate the parent's schemas
        ctx._tool_schemas.append(_make_tool_schema("execute_code"))
        ctx._tool_schemas.append(_make_tool_schema("new_tool"))

        # Subagent snapshot is unaffected
        assert len(runtime.tool_schemas) == original_count
        snapshot_names_after = {s["name"] for s in runtime.tool_schemas}
        assert "execute_code" not in snapshot_names_after
        assert "new_tool" not in snapshot_names_after


# ===================================================================
#  Test: permission policy enforcement
# ===================================================================


class TestSubagentPermissionEnforcement:
    """Subagent tool calls outside the allowed set must be denied."""

    @pytest.mark.asyncio
    async def test_denied_tool_never_reaches_dispatch(self):
        """A tool NOT in the allowed set produces a denial message."""
        # explore subagent allows: inspect_data, run_snippet, search_functions,
        # web_fetch, web_search, finish — but NOT execute_code.
        tc_denied = make_tool_call("execute_code", {"code": "import os"})
        llm = build_fake_llm([
            # Turn 1: LLM requests a denied tool
            make_chat_response(tool_calls=[tc_denied]),
            # Turn 2: LLM gives up after denial
            make_chat_response(content="Cannot execute code."),
        ])

        tool_runtime = FakeSubagentToolRuntime(
            handlers={"execute_code": lambda tc, ad, req: "should not run"},
            registry_tools={"execute_code": {"output_tier": OutputTier.verbose}},
        )
        schemas = [
            _make_tool_schema("inspect_data"),
            _make_tool_schema("finish"),
            _make_tool_schema("execute_code"),
        ]
        controller = _build_controller(llm, tool_runtime, tool_schemas=schemas)

        result = await controller.run_subagent(
            agent_type="explore", task="explore the data", adata=None,
        )

        # dispatch_tool should NEVER have been called
        assert len(tool_runtime.dispatch_calls) == 0

        # The denial message should appear in the conversation
        # (check LLM's recorded chat calls for the tool result message)
        all_messages = []
        for call in llm.chat_calls:
            all_messages.extend(call["messages"])

        denial_messages = [
            m for m in all_messages
            if m.get("role") == "tool" and "Permission denied" in m.get("content", "")
        ]
        assert len(denial_messages) >= 1
        assert "execute_code" in denial_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_allowed_tool_passes_through(self):
        """A tool IN the allowed set reaches dispatch normally."""
        tc_allowed = make_tool_call("inspect_data", {"aspect": "shape"})
        llm = build_fake_llm([
            make_chat_response(tool_calls=[tc_allowed]),
            make_chat_response(content="Data has 100 rows."),
        ])

        tool_runtime = FakeSubagentToolRuntime(
            handlers={"inspect_data": lambda tc, ad, req: "shape: (100, 5)"},
            registry_tools={"inspect_data": {"output_tier": OutputTier.standard}},
        )
        schemas = [_make_tool_schema("inspect_data"), _make_tool_schema("finish")]
        controller = _build_controller(llm, tool_runtime, tool_schemas=schemas)

        result = await controller.run_subagent(
            agent_type="explore", task="inspect", adata=None,
        )

        # dispatch_tool SHOULD have been called for the allowed tool
        assert len(tool_runtime.dispatch_calls) == 1
        assert tool_runtime.dispatch_calls[0]["name"] == "inspect_data"


# ===================================================================
#  Test: adata mutation gating
# ===================================================================


class TestSubagentAdataMutationGating:
    """adata updates are gated by ``can_mutate_adata``."""

    @pytest.mark.asyncio
    async def test_mutation_blocked_when_flag_is_false(self):
        """With can_mutate_adata=False, original adata is preserved."""
        original_adata = SimpleNamespace(shape=(100, 5), tag="original")
        new_adata = SimpleNamespace(shape=(200, 10), tag="new")

        tc = make_tool_call("execute_code", {"code": "adata = transform(adata)"})
        llm = build_fake_llm([
            make_chat_response(tool_calls=[tc]),
            make_chat_response(content="Done."),
        ])

        def _execute_code_handler(tool_call, adata, request):
            return {"adata": new_adata, "output": "Transformed."}

        tool_runtime = FakeSubagentToolRuntime(
            handlers={"execute_code": _execute_code_handler},
            registry_tools={"execute_code": {"output_tier": OutputTier.verbose}},
        )

        # Use "explore" which has can_mutate_adata=False
        schemas = [_make_tool_schema("execute_code"), _make_tool_schema("finish")]
        # Override explore to include execute_code for this test
        cfg = AgentConfig(subagent_overrides={
            "explore": {"allowed_tools": ["execute_code", "inspect_data", "finish"]},
        })
        ctx = FakeAgentContext(llm, cfg, tool_schemas=schemas)
        controller = SubagentController(ctx, FakePromptBuilder(), tool_runtime)

        result = await controller.run_subagent(
            agent_type="explore", task="explore", adata=original_adata,
        )

        # explore has can_mutate_adata=False -> original adata returned
        assert result["adata"] is original_adata
        assert result["adata"].tag == "original"

    @pytest.mark.asyncio
    async def test_mutation_allowed_when_flag_is_true(self):
        """With can_mutate_adata=True, adata is updated from execute_code."""
        original_adata = SimpleNamespace(shape=(100, 5), tag="original")
        new_adata = SimpleNamespace(shape=(200, 10), tag="new")

        tc = make_tool_call("execute_code", {"code": "adata = transform(adata)"})
        llm = build_fake_llm([
            make_chat_response(tool_calls=[tc]),
            make_chat_response(content="Done."),
        ])

        def _execute_code_handler(tool_call, adata, request):
            return {"adata": new_adata, "output": "Transformed."}

        tool_runtime = FakeSubagentToolRuntime(
            handlers={"execute_code": _execute_code_handler},
            registry_tools={"execute_code": {"output_tier": OutputTier.verbose}},
        )

        # "execute" subagent has can_mutate_adata=True
        schemas = [_make_tool_schema("execute_code"), _make_tool_schema("finish")]
        cfg = AgentConfig()
        ctx = FakeAgentContext(llm, cfg, tool_schemas=schemas)
        controller = SubagentController(ctx, FakePromptBuilder(), tool_runtime)

        result = await controller.run_subagent(
            agent_type="execute", task="run code", adata=original_adata,
        )

        # execute has can_mutate_adata=True -> new adata returned
        assert result["adata"] is new_adata
        assert result["adata"].tag == "new"


# ===================================================================
#  Test: budget manager isolation
# ===================================================================


class TestSubagentBudgetManagerIsolation:
    """Each subagent runtime gets its own budget manager."""

    @pytest.mark.asyncio
    async def test_each_runtime_gets_own_budget_manager(self):
        """Two runtimes created from the same controller have distinct managers."""
        llm = build_fake_llm([])
        tool_runtime = FakeSubagentToolRuntime()
        cfg = AgentConfig()
        ctx = FakeAgentContext(llm, cfg)
        controller = SubagentController(ctx, FakePromptBuilder(), tool_runtime)

        explore_config = cfg.get_subagent_config("explore")
        runtime_a = controller._create_subagent_runtime(
            agent_type="explore",
            allowed_tools=frozenset(explore_config.allowed_tools),
            max_turns=explore_config.max_turns,
        )
        runtime_b = controller._create_subagent_runtime(
            agent_type="explore",
            allowed_tools=frozenset(explore_config.allowed_tools),
            max_turns=explore_config.max_turns,
        )

        # Distinct instances
        assert runtime_a.budget_manager is not runtime_b.budget_manager
        assert isinstance(runtime_a.budget_manager, ContextBudgetManager)
        assert isinstance(runtime_b.budget_manager, ContextBudgetManager)

    @pytest.mark.asyncio
    async def test_budget_manager_uses_subagent_tier_policies(self):
        """Subagent budget managers use tighter truncation policies."""
        llm = build_fake_llm([])
        tool_runtime = FakeSubagentToolRuntime()
        cfg = AgentConfig()
        ctx = FakeAgentContext(llm, cfg)
        controller = SubagentController(ctx, FakePromptBuilder(), tool_runtime)

        explore_config = cfg.get_subagent_config("explore")
        runtime = controller._create_subagent_runtime(
            agent_type="explore",
            allowed_tools=frozenset(explore_config.allowed_tools),
            max_turns=explore_config.max_turns,
        )

        # Subagent policies have max_tokens=300/1200/1800 vs default 500/2000/3000
        policies = runtime.budget_manager.tier_policies
        assert policies[OutputTier.minimal].max_tokens < 500
        assert policies[OutputTier.standard].max_tokens < 2000
        assert policies[OutputTier.verbose].max_tokens < 3000


# ===================================================================
#  Test: max turns enforcement
# ===================================================================


class TestSubagentMaxTurns:
    """Subagent must stop after ``max_turns`` even if LLM keeps calling tools."""

    @pytest.mark.asyncio
    async def test_stops_after_max_turns(self):
        """With max_turns=2, subagent exits after 2 LLM turns."""
        # Every response requests a tool call -> subagent never finishes naturally
        tc = make_tool_call("inspect_data", {"aspect": "shape"})
        llm = build_fake_llm([
            make_chat_response(tool_calls=[tc]),  # turn 1
            make_chat_response(tool_calls=[tc]),  # turn 2
            # turn 3 would happen, but max_turns=2 stops it
            make_chat_response(content="Should never reach here."),
        ])

        tool_runtime = FakeSubagentToolRuntime(
            handlers={"inspect_data": lambda tc, ad, req: "shape: (100, 5)"},
            registry_tools={"inspect_data": {"output_tier": OutputTier.standard}},
        )
        schemas = [_make_tool_schema("inspect_data"), _make_tool_schema("finish")]

        # Override explore to have max_turns=2
        cfg = AgentConfig(subagent_overrides={
            "explore": {"max_turns": 2},
        })
        ctx = FakeAgentContext(llm, cfg, tool_schemas=schemas)
        controller = SubagentController(ctx, FakePromptBuilder(), tool_runtime)

        result = await controller.run_subagent(
            agent_type="explore", task="keep inspecting", adata=None,
        )

        # Result should contain the timeout message
        assert "max turns" in result["result"].lower() or "reached max turns" in result["result"].lower()

        # LLM should have been called exactly 2 times (max_turns=2)
        assert len(llm.chat_calls) == 2

    @pytest.mark.asyncio
    async def test_finishes_early_when_no_tool_calls(self):
        """Subagent returns before max_turns if LLM produces no tool calls."""
        llm = build_fake_llm([
            make_chat_response(content="All done, no tools needed."),
        ])

        tool_runtime = FakeSubagentToolRuntime()
        schemas = [_make_tool_schema("inspect_data"), _make_tool_schema("finish")]
        cfg = AgentConfig(subagent_overrides={"explore": {"max_turns": 10}})
        ctx = FakeAgentContext(llm, cfg, tool_schemas=schemas)
        controller = SubagentController(ctx, FakePromptBuilder(), tool_runtime)

        result = await controller.run_subagent(
            agent_type="explore", task="quick check", adata=None,
        )

        assert result["result"] == "All done, no tools needed."
        assert len(llm.chat_calls) == 1

    @pytest.mark.asyncio
    async def test_finish_tool_exits_early(self):
        """The ``finish`` tool causes immediate return before max_turns."""
        tc_finish = make_tool_call("finish", {"summary": "Exploration complete."})
        llm = build_fake_llm([
            make_chat_response(tool_calls=[tc_finish]),
            # This should never be consumed
            make_chat_response(content="Unreachable."),
        ])

        def _finish_handler(tool_call, adata, request):
            return {"summary": tool_call.arguments.get("summary", "")}

        tool_runtime = FakeSubagentToolRuntime(
            handlers={"finish": _finish_handler},
            registry_tools={"finish": {"output_tier": OutputTier.minimal}},
        )
        schemas = [_make_tool_schema("inspect_data"), _make_tool_schema("finish")]
        controller = _build_controller(llm, tool_runtime, tool_schemas=schemas)

        result = await controller.run_subagent(
            agent_type="explore", task="explore", adata=None,
        )

        assert result["result"] == "Exploration complete."
        # Only 1 LLM call consumed (finish exits the loop)
        assert len(llm.chat_calls) == 1
        # Second response is still unconsumed
        assert llm.remaining_responses == 1
