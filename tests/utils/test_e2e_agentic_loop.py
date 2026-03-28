"""End-to-end tests for the main agentic loop.

Exercises the full OmicVerseAgent facade (``_run_agentic_loop``,
``run_async``, ``stream_async``) using the shared integration harness
fakes.  No real provider credentials are required.

Covers:
- Mock tool-call execution and result propagation
- Text-only retry/recovery via the FollowUpGate
- Final response validation (adata, summary, trace)
- ``stream_async`` event emission
- Multi-turn tool dispatch
- ``finish`` tool early exit

Gate: ``OV_AGENT_RUN_HARNESS_TESTS=1``
"""
from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import os
import sys
import types
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Module-level bootstrap: load smart_agent.py in isolation (same pattern as
# test_smart_agent.py) so we don't need the full omicverse install tree.
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, "omicverse")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_ORIGINAL_MODULES = {
    name: sys.modules.get(name)
    for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]
}
for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]:
    sys.modules.pop(name, None)

omicverse_pkg = types.ModuleType("omicverse")
omicverse_pkg.__path__ = [PACKAGE_ROOT]
omicverse_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = omicverse_pkg

utils_pkg = types.ModuleType("omicverse.utils")
utils_pkg.__path__ = [os.path.join(PACKAGE_ROOT, "utils")]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = utils_pkg
omicverse_pkg.utils = utils_pkg

smart_agent_spec = importlib.util.spec_from_file_location(
    "omicverse.utils.smart_agent",
    os.path.join(PACKAGE_ROOT, "utils", "smart_agent.py"),
)
smart_agent_module = importlib.util.module_from_spec(smart_agent_spec)
sys.modules["omicverse.utils.smart_agent"] = smart_agent_module
assert smart_agent_spec.loader is not None
smart_agent_spec.loader.exec_module(smart_agent_module)

OmicVerseAgent = smart_agent_module.OmicVerseAgent

for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Agentic loop e2e tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Production imports (only under the gate)
# ---------------------------------------------------------------------------

from omicverse.utils.ovagent.turn_controller import TurnController
from omicverse.utils.ovagent.tool_registry import OutputTier, ParallelClass

# Shared integration harness fakes
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
from omicverse.utils.harness import build_stream_event


# ===================================================================
#  Test doubles compatible with TurnController
# ===================================================================


class _E2EToolRuntime:
    """Fake ToolRuntime matching the interface TurnController.dispatch_tool expects.

    Uses the integration harness's FakeToolRegistry for name resolution and
    metadata, while providing an async ``dispatch_tool(tool_call, adata, request)``
    signature.
    """

    def __init__(
        self,
        handlers: Optional[Dict[str, Any]] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        registry_tools: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._handlers: Dict[str, Any] = dict(handlers or {})
        self._tool_schemas = list(tool_schemas or [])
        enriched: Dict[str, Dict[str, Any]] = {}
        for name, attrs in (registry_tools or {}).items():
            merged: Dict[str, Any] = {
                "output_tier": OutputTier.standard,
                "parallel_class": ParallelClass.stateful,
            }
            merged.update(attrs)
            enriched[name] = merged
        self.registry = FakeToolRegistry(enriched)
        self.dispatch_calls: List[Dict[str, Any]] = []

    async def dispatch_tool(
        self, tool_call: Any, current_adata: Any, request: str, **kw: Any
    ) -> Any:
        self.dispatch_calls.append({
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "adata": current_adata,
        })
        handler = self._handlers.get(tool_call.name)
        if handler is None:
            return f"[E2EToolRuntime] unknown tool: {tool_call.name}"
        return handler(tool_call, current_adata, request)

    def get_visible_agent_tools(
        self, *, allowed_names: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        if allowed_names is not None:
            return [s for s in self._tool_schemas if s.get("name") in allowed_names]
        return list(self._tool_schemas)


class _E2EPromptBuilder:
    """Minimal PromptBuilder satisfying the TurnController protocol."""

    def build_agentic_system_prompt(self) -> str:
        return "You are a test agent."

    def build_initial_user_message(
        self, request: str, adata: Any, extra_content: Any = None
    ) -> str:
        return request


def _make_tool_schema(name: str, description: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "description": description or f"{name} tool",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


# ===================================================================
#  Agent wiring helper
# ===================================================================


def _build_e2e_agent(
    llm: FakeLLM,
    tool_runtime: _E2EToolRuntime,
    *,
    max_agent_turns: int = 10,
) -> OmicVerseAgent:
    """Wire up an OmicVerseAgent with fakes for end-to-end loop testing.

    The agent uses a real TurnController (production code path) but with
    fake LLM/tool backends so no credentials are needed.
    """
    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    agent.model = llm.config.model
    agent.provider = llm.config.provider
    agent.last_usage = None
    agent._approval_handler = None
    agent._trace_store = None
    agent._context_compactor = None
    agent._session_history = None
    agent._last_run_trace = None
    agent._ov_runtime = None
    agent._active_run_id = None
    agent.skill_registry = None
    agent._llm = llm
    agent._tool_runtime = tool_runtime

    agent._config = SimpleNamespace(
        execution=SimpleNamespace(max_agent_turns=max_agent_turns),
        harness=SimpleNamespace(
            include_recent_failures_in_prompt=False,
            max_recent_failures=3,
            record_artifacts=False,
        ),
    )
    agent._get_harness_session_id = MethodType(lambda self: "e2e-session", agent)
    agent._get_runtime_session_id = MethodType(lambda self: "e2e-session", agent)
    agent._get_visible_agent_tools = MethodType(
        lambda self, allowed_names=None: tool_runtime.get_visible_agent_tools(
            allowed_names=allowed_names
        ),
        agent,
    )

    agent._turn_controller = TurnController(
        agent, _E2EPromptBuilder(), tool_runtime,
    )

    return agent


def _build_raw_message_for_tool_calls(tool_calls: List[ToolCall]) -> Dict[str, Any]:
    """Build a minimal raw_message dict for a response containing tool calls."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            }
            for tc in tool_calls
        ],
    }


def _resp_tool(
    tool_calls: List[ToolCall],
    usage: Optional[Usage] = None,
) -> ChatResponse:
    """Build a ChatResponse with tool calls and correct raw_message."""
    return ChatResponse(
        content=None,
        tool_calls=tool_calls,
        stop_reason="tool_use",
        usage=usage or make_usage(),
        raw_message=_build_raw_message_for_tool_calls(tool_calls),
    )


def _resp_text(text: str, usage: Optional[Usage] = None) -> ChatResponse:
    """Build a text-only ChatResponse with correct raw_message."""
    return ChatResponse(
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=usage or make_usage(),
        raw_message={"role": "assistant", "content": text},
    )


# ===================================================================
#  1. Happy path: tool call → finish
# ===================================================================


class TestAgenticLoopHappyPath:
    """Agent dispatches tool calls and exits when LLM calls ``finish``."""

    def test_single_tool_then_finish(self):
        """LLM calls one tool, then finish → loop returns adata."""
        tc_inspect = make_tool_call("inspect_data", {"aspect": "shape"})
        tc_finish = make_tool_call("finish", {"summary": "Data inspected."})

        llm = build_fake_llm([
            _resp_tool([tc_inspect]),
            _resp_tool([tc_finish]),
        ])

        tool_runtime = _E2EToolRuntime(
            handlers={
                "inspect_data": lambda tc, ad, req: "shape: (100, 5)",
                "finish": lambda tc, ad, req: f"Task finished: {tc.arguments.get('summary', '')}",
            },
            tool_schemas=[_make_tool_schema("inspect_data"), _make_tool_schema("finish")],
            registry_tools={
                "inspect_data": {"output_tier": OutputTier.standard},
                "finish": {"output_tier": OutputTier.minimal},
            },
        )

        sentinel_adata = SimpleNamespace(shape=(100, 5), tag="original")
        agent = _build_e2e_agent(llm, tool_runtime)

        result = asyncio.run(agent._run_agentic_loop("inspect the data", sentinel_adata))

        # adata returned unchanged (no execute_code that mutates it)
        assert result is sentinel_adata

        # Tools were dispatched
        assert len(tool_runtime.dispatch_calls) == 2
        assert tool_runtime.dispatch_calls[0]["name"] == "inspect_data"
        assert tool_runtime.dispatch_calls[1]["name"] == "finish"

        # LLM was called exactly twice
        assert len(llm.chat_calls) == 2

    def test_text_only_final_response(self):
        """LLM returns text without tool calls → loop exits with the text."""
        llm = build_fake_llm([
            make_chat_response(content="The data looks good. No action needed."),
        ])
        tool_runtime = _E2EToolRuntime(
            tool_schemas=[_make_tool_schema("finish")],
        )

        agent = _build_e2e_agent(llm, tool_runtime)
        result = asyncio.run(agent._run_agentic_loop("what does this data look like?", None))

        assert result is None  # adata was None
        assert len(llm.chat_calls) == 1


# ===================================================================
#  2. Text-only retry recovery
# ===================================================================


class TestTextOnlyRetryRecovery:
    """FollowUpGate retries when LLM promises action but returns no tools."""

    def test_promissory_text_triggers_retry_then_tool_call(self):
        """Promissory text → retry → tool call → finish."""
        tc_fetch = make_tool_call("WebFetch", {"url": "https://example.com"})
        tc_finish = make_tool_call("finish", {"summary": "Fetched and analyzed."})

        llm = build_fake_llm([
            # Turn 1: promissory text-only
            _resp_text("Let me first fetch the page to understand the data."),
            # Turn 2: actual tool call after retry nudge
            _resp_tool([tc_fetch]),
            # Turn 3: finish
            _resp_tool([tc_finish]),
        ])

        tool_runtime = _E2EToolRuntime(
            handlers={
                "WebFetch": lambda tc, ad, req: "Page content from example.com",
                "finish": lambda tc, ad, req: "done",
            },
            tool_schemas=[
                _make_tool_schema("WebFetch"),
                _make_tool_schema("finish"),
            ],
            registry_tools={
                "WebFetch": {"output_tier": OutputTier.standard},
                "finish": {"output_tier": OutputTier.minimal},
            },
        )

        agent = _build_e2e_agent(llm, tool_runtime)
        result = asyncio.run(agent._run_agentic_loop(
            "https://example.com\nanalyze this page",
            None,
        ))

        # LLM called 3 times: promissory text, retry with tool, then finish
        assert len(llm.chat_calls) == 3

        # The retry should have used tool_choice="required"
        assert llm.chat_calls[0]["tool_choice"] == "required"
        assert llm.chat_calls[1]["tool_choice"] == "required"

        # Tool was dispatched after retry
        assert any(c["name"] == "WebFetch" for c in tool_runtime.dispatch_calls)

    def test_retry_exhaustion_exits_gracefully(self):
        """When retries exhaust, loop exits without crash."""
        # All responses are promissory text-only → exhaustion
        responses = [
            _resp_text("Let me analyze this for you."),
            _resp_text("I'll start by fetching the data now."),
            _resp_text("Going to retrieve the dataset next."),
            _resp_text("I will now fetch the page."),
        ]

        llm = build_fake_llm(responses)
        tool_runtime = _E2EToolRuntime(
            tool_schemas=[_make_tool_schema("WebFetch"), _make_tool_schema("finish")],
        )

        agent = _build_e2e_agent(llm, tool_runtime, max_agent_turns=10)
        result = asyncio.run(agent._run_agentic_loop(
            "https://example.com\nfetch this",
            None,
        ))

        # Should exit without raising
        assert result is None
        # No tools were dispatched
        assert len(tool_runtime.dispatch_calls) == 0


# ===================================================================
#  3. adata mutation through execute_code
# ===================================================================


class TestAdataMutationThroughTool:
    """execute_code tool updates adata when result contains 'adata' key."""

    def test_execute_code_updates_adata(self):
        """execute_code returning {adata: X} causes loop to track the new adata."""
        original = SimpleNamespace(shape=(100, 5), tag="original")
        updated = SimpleNamespace(shape=(80, 5), tag="filtered")

        tc_exec = make_tool_call("execute_code", {
            "code": "adata = adata[adata.obs['n_genes'] > 200]",
            "description": "Filter low-quality cells",
        })
        tc_finish = make_tool_call("finish", {"summary": "Filtered."})

        llm = build_fake_llm([
            _resp_tool([tc_exec]),
            _resp_tool([tc_finish]),
        ])

        tool_runtime = _E2EToolRuntime(
            handlers={
                "execute_code": lambda tc, ad, req: {
                    "adata": updated,
                    "output": "Filtered to 80 cells.",
                    "stdout": "Filtered to 80 cells.",
                },
                "finish": lambda tc, ad, req: "done",
            },
            tool_schemas=[
                _make_tool_schema("execute_code"),
                _make_tool_schema("finish"),
            ],
            registry_tools={
                "execute_code": {"output_tier": OutputTier.verbose},
                "finish": {"output_tier": OutputTier.minimal},
            },
        )

        agent = _build_e2e_agent(llm, tool_runtime)
        result = asyncio.run(agent._run_agentic_loop("filter cells", original))

        # adata was updated by execute_code
        assert result is updated
        assert result.tag == "filtered"
        assert result.shape == (80, 5)


# ===================================================================
#  4. Multi-tool dispatch in a single turn
# ===================================================================


class TestMultiToolDispatch:
    """Multiple tool calls in a single LLM response are all dispatched."""

    def test_two_tools_in_one_turn(self):
        """LLM returns two tool calls; both are dispatched before next turn."""
        tc_inspect = make_tool_call("inspect_data", {"aspect": "shape"})
        tc_search = make_tool_call("search_functions", {"query": "qc"})
        tc_finish = make_tool_call("finish", {"summary": "Done."})

        llm = build_fake_llm([
            _resp_tool([tc_inspect, tc_search]),
            _resp_tool([tc_finish]),
        ])

        tool_runtime = _E2EToolRuntime(
            handlers={
                "inspect_data": lambda tc, ad, req: "shape: (200, 10)",
                "search_functions": lambda tc, ad, req: "ov.pp.qc found",
                "finish": lambda tc, ad, req: "done",
            },
            tool_schemas=[
                _make_tool_schema("inspect_data"),
                _make_tool_schema("search_functions"),
                _make_tool_schema("finish"),
            ],
            registry_tools={
                "inspect_data": {"output_tier": OutputTier.standard},
                "search_functions": {"output_tier": OutputTier.standard},
                "finish": {"output_tier": OutputTier.minimal},
            },
        )

        agent = _build_e2e_agent(llm, tool_runtime)
        result = asyncio.run(agent._run_agentic_loop("run qc", None))

        # Both tools were dispatched
        dispatched_names = [c["name"] for c in tool_runtime.dispatch_calls]
        assert "inspect_data" in dispatched_names
        assert "search_functions" in dispatched_names
        assert "finish" in dispatched_names
        assert len(tool_runtime.dispatch_calls) == 3


# ===================================================================
#  5. stream_async event emission
# ===================================================================


class TestStreamAsyncEventEmission:
    """stream_async yields events for tool calls and lifecycle."""

    def test_stream_emits_tool_and_done_events(self):
        """stream_async yields tool_call, item_started, item_completed, and done."""
        tc_finish = make_tool_call("finish", {"summary": "All done."})

        llm = build_fake_llm([_resp_tool([tc_finish])])

        tool_runtime = _E2EToolRuntime(
            handlers={"finish": lambda tc, ad, req: "done"},
            tool_schemas=[_make_tool_schema("finish")],
            registry_tools={"finish": {"output_tier": OutputTier.minimal}},
        )

        agent = _build_e2e_agent(llm, tool_runtime)

        async def _collect():
            events = []
            async for event in agent.stream_async("do it", None):
                events.append(event)
            return events

        events = asyncio.run(_collect())

        event_types = [e.get("type") for e in events]

        # Must contain at minimum: status (turn_started), tool_call,
        # item_started, item_completed, done
        assert "tool_call" in event_types
        assert "done" in event_types

        # The done event carries the finish summary
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) >= 1
        assert "All done." in str(done_events[-1].get("content", ""))

    def test_stream_emits_retry_status_on_text_only(self):
        """stream_async emits follow_up_required status during text-only retries."""
        tc_finish = make_tool_call("finish", {"summary": "Finally done."})

        llm = build_fake_llm([
            # Turn 1: promissory text
            _resp_text("Let me fetch that for you."),
            # Turn 2: tool call after retry
            _resp_tool([tc_finish]),
        ])

        tool_runtime = _E2EToolRuntime(
            handlers={"finish": lambda tc, ad, req: "done"},
            tool_schemas=[_make_tool_schema("WebFetch"), _make_tool_schema("finish")],
            registry_tools={"finish": {"output_tier": OutputTier.minimal}},
        )

        agent = _build_e2e_agent(llm, tool_runtime)

        async def _collect():
            events = []
            async for event in agent.stream_async(
                "https://example.com\nfetch this", None
            ):
                events.append(event)
            return events

        events = asyncio.run(_collect())

        # Should contain a status event with follow_up_required
        status_events = [
            e for e in events
            if e.get("type") == "status"
            and isinstance(e.get("content"), dict)
            and e["content"].get("follow_up_required")
        ]
        assert len(status_events) >= 1


# ===================================================================
#  6. run_async public facade exercises the full loop
# ===================================================================


class TestRunAsyncFacade:
    """run_async() delegates to _run_agentic_mode which runs the full loop."""

    def test_run_async_returns_adata_from_loop(self):
        """run_async returns the adata produced by the agentic loop."""
        sentinel = SimpleNamespace(shape=(50, 10), tag="sentinel")
        tc_finish = make_tool_call("finish", {"summary": "Done."})

        llm = build_fake_llm([_resp_tool([tc_finish])])

        tool_runtime = _E2EToolRuntime(
            handlers={"finish": lambda tc, ad, req: "done"},
            tool_schemas=[_make_tool_schema("finish")],
            registry_tools={"finish": {"output_tier": OutputTier.minimal}},
        )

        agent = _build_e2e_agent(llm, tool_runtime)
        # Stub _detect_direct_python_request to return None (not direct python)
        agent._detect_direct_python_request = lambda request: None
        # Stub _emit to avoid reporter dependency
        agent._emit = lambda level, message, category="": None

        result = asyncio.run(agent.run_async("inspect the data", sentinel))

        assert result is sentinel

    def test_run_async_direct_python_bypass(self):
        """run_async with direct python code bypasses the agentic loop."""
        agent = OmicVerseAgent.__new__(OmicVerseAgent)
        agent.provider = "openai"
        agent.last_usage = None
        agent.last_usage_breakdown = {
            "generation": None, "reflection": [], "review": [], "total": None
        }
        agent._emit = lambda level, message, category="": None

        executed = {}
        agent._detect_direct_python_request = lambda request: "x = 42"
        agent._execute_generated_code = lambda code, adata: (
            executed.update(code=code) or adata
        )

        result = asyncio.run(agent.run_async("x = 42", None))

        assert result is None  # adata was None, returned as-is
        assert executed["code"] == "x = 42"
        assert agent.last_usage is None


# ===================================================================
#  7. Trace metadata validation
# ===================================================================


class TestTraceMetadata:
    """The agentic loop populates _last_run_trace with correct metadata."""

    def test_trace_has_session_and_status(self):
        """After a successful loop, _last_run_trace records status=success."""
        tc_finish = make_tool_call("finish", {"summary": "Complete."})

        llm = build_fake_llm([_resp_tool([tc_finish])])

        tool_runtime = _E2EToolRuntime(
            handlers={"finish": lambda tc, ad, req: "done"},
            tool_schemas=[_make_tool_schema("finish")],
            registry_tools={"finish": {"output_tier": OutputTier.minimal}},
        )

        agent = _build_e2e_agent(llm, tool_runtime)
        asyncio.run(agent._run_agentic_loop("check data", None))

        trace = agent._last_run_trace
        assert trace is not None
        assert trace.session_id == "e2e-session"
        assert trace.status == "success"

    def test_trace_records_steps(self):
        """Trace steps include tool_call entries for dispatched tools."""
        tc_inspect = make_tool_call("inspect_data", {"aspect": "shape"})
        tc_finish = make_tool_call("finish", {"summary": "Done."})

        llm = build_fake_llm([
            _resp_tool([tc_inspect]),
            _resp_tool([tc_finish]),
        ])

        tool_runtime = _E2EToolRuntime(
            handlers={
                "inspect_data": lambda tc, ad, req: "shape: (100, 5)",
                "finish": lambda tc, ad, req: "done",
            },
            tool_schemas=[_make_tool_schema("inspect_data"), _make_tool_schema("finish")],
            registry_tools={
                "inspect_data": {"output_tier": OutputTier.standard},
                "finish": {"output_tier": OutputTier.minimal},
            },
        )

        agent = _build_e2e_agent(llm, tool_runtime)
        asyncio.run(agent._run_agentic_loop("inspect", None))

        trace = agent._last_run_trace
        step_types = [s.step_type for s in trace.steps]
        assert "tool_call" in step_types
        assert "done" in step_types


# ===================================================================
#  8. Max turns enforcement
# ===================================================================


class TestMaxTurnsEnforcement:
    """Loop exits when max_turns is reached."""

    def test_exits_after_max_turns(self):
        """With max_turns=2, loop stops even if LLM keeps calling tools."""
        tc = make_tool_call("inspect_data", {"aspect": "shape"})
        responses = [_resp_tool([tc]) for _ in range(5)]

        llm = build_fake_llm(responses)
        tool_runtime = _E2EToolRuntime(
            handlers={"inspect_data": lambda tc, ad, req: "shape: (100, 5)"},
            tool_schemas=[_make_tool_schema("inspect_data"), _make_tool_schema("finish")],
            registry_tools={"inspect_data": {"output_tier": OutputTier.standard}},
        )

        agent = _build_e2e_agent(llm, tool_runtime, max_agent_turns=2)
        result = asyncio.run(agent._run_agentic_loop("keep inspecting", None))

        # Should only have made 2 LLM calls
        assert len(llm.chat_calls) == 2
        # Trace records max_turns status
        assert agent._last_run_trace.status == "max_turns"


# ===================================================================
#  9. Cancellation support
# ===================================================================


class TestCancellationSupport:
    """cancel_event stops the loop at the next safe checkpoint."""

    def test_cancel_before_first_llm_call(self):
        """Pre-set cancel_event → loop returns immediately."""
        import threading

        llm = build_fake_llm([make_chat_response(content="should not reach")])
        tool_runtime = _E2EToolRuntime(
            tool_schemas=[_make_tool_schema("finish")],
        )

        agent = _build_e2e_agent(llm, tool_runtime)
        cancel = threading.Event()
        cancel.set()  # pre-set

        result = asyncio.run(agent._run_agentic_loop(
            "test", None, cancel_event=cancel,
        ))

        assert result is None
        # LLM was never called
        assert len(llm.chat_calls) == 0
        assert agent._last_run_trace.status == "cancelled"
