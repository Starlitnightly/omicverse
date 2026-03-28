"""Tests verifying the integration-test harness itself.

These tests confirm that every fake, fixture, and helper works as specified
so that downstream integration suites can rely on the harness without
second-guessing its behavior.

Acceptance criteria covered:
  AC-001-1: Shared fake LLM and tool-runtime fixtures exist.
  AC-001-2: Harness utilities exercise agent/Jarvis layers without network.
  AC-001-3: Fixture code is reusable (single import, no duplication).
  AC-001-4: Integrates with existing pytest structure.
  AC-001-5: No production code depends on test-only harness helpers.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

# -- harness imports (AC-001-3: single import location) -------------------
from tests.integration.fakes import (
    AgentRunResult,
    ChatResponse,
    FakeExecutionAdapter,
    FakeLLM,
    FakePresenter,
    FakeRouter,
    FakeToolRuntime,
    FakeToolRegistry,
    ToolCall,
    Usage,
)
from tests.integration.helpers import (
    build_fake_llm,
    build_fake_tool_runtime,
    build_jarvis_runtime_deps,
    make_agent_run_result,
    make_chat_response,
    make_envelope,
    make_route,
    make_tool_call,
    make_usage,
)


_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}


# ===================================================================
#  FakeLLM — chat / run / stream
# ===================================================================


class TestFakeLLMChat:
    """FakeLLM.chat() returns staged responses in FIFO order."""

    @pytest.mark.asyncio
    async def test_returns_string_as_chat_response(self):
        llm = FakeLLM(["first", "second"])
        r1 = await llm.chat([{"role": "user", "content": "hi"}])

        assert isinstance(r1, ChatResponse)
        assert r1.content == "first"
        assert r1.stop_reason == "end_turn"
        assert r1.tool_calls == []

    @pytest.mark.asyncio
    async def test_returns_chat_response_as_is(self):
        staged = make_chat_response(content="staged", stop_reason="tool_use")
        llm = FakeLLM([staged])
        r = await llm.chat([])

        assert r is staged
        assert r.content == "staged"
        assert r.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_returns_chat_response_with_tool_calls(self):
        tc = make_tool_call("inspect_data", {"aspect": "shape"})
        staged = make_chat_response(tool_calls=[tc], stop_reason="tool_use")
        llm = FakeLLM([staged])
        r = await llm.chat([])

        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "inspect_data"
        assert r.tool_calls[0].arguments == {"aspect": "shape"}

    @pytest.mark.asyncio
    async def test_exhausted_returns_default_response(self):
        llm = FakeLLM([])
        r = await llm.chat([])

        assert r.content == "<exhausted>"
        assert r.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_records_calls(self):
        llm = FakeLLM(["ok"])
        msgs = [{"role": "user", "content": "test"}]
        tools = [{"name": "t"}]
        await llm.chat(msgs, tools=tools, tool_choice="auto")

        assert len(llm.chat_calls) == 1
        assert llm.chat_calls[0]["messages"] is msgs
        assert llm.chat_calls[0]["tools"] is tools
        assert llm.chat_calls[0]["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_updates_last_usage(self):
        llm = FakeLLM(["hi"])
        assert llm.last_usage is None
        await llm.chat([])
        assert llm.last_usage is not None


class TestFakeLLMRun:
    """FakeLLM.run() returns text and records prompts."""

    @pytest.mark.asyncio
    async def test_returns_text(self):
        llm = FakeLLM(["hello world"])
        result = await llm.run("prompt")

        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_records_prompts(self):
        llm = FakeLLM(["a", "b"])
        await llm.run("first")
        await llm.run("second")

        assert llm.run_calls == ["first", "second"]


class TestFakeLLMStream:
    """FakeLLM.stream() yields chunks."""

    @pytest.mark.asyncio
    async def test_yields_chunks(self):
        llm = FakeLLM(["abcdef"])
        chunks = []
        async for chunk in llm.stream("prompt"):
            chunks.append(chunk)

        reassembled = "".join(chunks)
        assert reassembled == "abcdef"
        assert len(chunks) > 1  # actually streams, not one blob

    @pytest.mark.asyncio
    async def test_records_prompts(self):
        llm = FakeLLM(["x"])
        async for _ in llm.stream("my prompt"):
            pass

        assert llm.stream_calls == ["my prompt"]


class TestFakeLLMConfig:
    """FakeLLM exposes a config namespace matching OmicVerseLLMBackend."""

    def test_config_defaults(self):
        llm = FakeLLM()
        assert llm.config.model == "fake-model"
        assert llm.config.provider == "fake"

    def test_config_overrides(self):
        llm = FakeLLM(model="gpt-5.2", provider="openai")
        assert llm.config.model == "gpt-5.2"
        assert llm.config.provider == "openai"

    def test_remaining_responses(self):
        llm = FakeLLM(["a", "b", "c"])
        assert llm.remaining_responses == 3

    def test_format_tool_result_message(self):
        llm = FakeLLM()
        msg = llm.format_tool_result_message("call-1", "echo", "result-text")
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call-1"
        assert msg["name"] == "echo"
        assert msg["content"] == "result-text"


# ===================================================================
#  FakeToolRuntime & FakeToolRegistry
# ===================================================================


class TestFakeToolRegistry:
    def test_resolve_name_passthrough(self):
        reg = FakeToolRegistry()
        assert reg.resolve_name("foo") == "foo"

    def test_get_returns_none_for_unknown(self):
        reg = FakeToolRegistry()
        assert reg.get("unknown") is None

    def test_get_returns_namespace_for_known(self):
        reg = FakeToolRegistry({"echo": {"description": "test"}})
        meta = reg.get("echo")
        assert meta is not None
        assert meta.canonical_name == "echo"
        assert meta.description == "test"

    def test_validate_handlers_empty(self):
        reg = FakeToolRegistry()
        assert reg.validate_handlers() == []


class TestFakeToolRuntime:
    @pytest.mark.asyncio
    async def test_dispatch_calls_handler(self):
        def _echo(args, adata, request):
            return f"echo:{args['text']}"

        rt = FakeToolRuntime({"echo": _echo})
        result = await rt.dispatch("echo", {"text": "hi"}, None, "req")

        assert result == "echo:hi"
        assert rt.dispatch_calls == [("echo", {"text": "hi"})]

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self):
        rt = FakeToolRuntime()
        result = await rt.dispatch("missing", {}, None, "req")

        assert "unknown tool" in result

    @pytest.mark.asyncio
    async def test_dispatch_async_handler(self):
        async def _async_handler(args, adata, request):
            return "async-result"

        rt = FakeToolRuntime({"atool": _async_handler})
        result = await rt.dispatch("atool", {}, None, "req")

        assert result == "async-result"

    def test_get_visible_agent_tools(self):
        schemas = [{"name": "a"}, {"name": "b"}]
        rt = FakeToolRuntime(tool_schemas=schemas)

        assert rt.get_visible_agent_tools() == schemas
        assert rt.get_visible_agent_tools(allowed_names={"a"}) == [{"name": "a"}]

    def test_set_subagent_controller_noop(self):
        rt = FakeToolRuntime()
        rt.set_subagent_controller(object())  # should not raise


# ===================================================================
#  Jarvis fakes
# ===================================================================


class TestFakePresenter:
    def test_records_ack(self):
        p = FakePresenter()
        p.ack("env", "sess")

        assert len(p.events) == 1
        assert p.events[0][0] == "ack"

    def test_records_multiple_methods(self):
        p = FakePresenter()
        p.ack("e", "s")
        p.draft_open("r")
        p.analysis_error("r", "oops")

        assert [name for name, _ in p.events] == [
            "ack", "draft_open", "analysis_error"
        ]

    def test_method_calls_filter(self):
        p = FakePresenter()
        p.ack("e", "s")
        p.draft_open("r1")
        p.draft_open("r2")

        opens = p.method_calls("draft_open")
        assert len(opens) == 2
        assert opens[0]["route"] == "r1"
        assert opens[1]["route"] == "r2"

    def test_typing_returns_none(self):
        p = FakePresenter()
        assert p.typing("route") is None

    def test_final_events_returns_empty_list(self):
        p = FakePresenter()
        result = p.final_events(
            "route", session="s", user_text="u", llm_text="l", result="r"
        )
        assert result == []


class TestFakeExecutionAdapter:
    @pytest.mark.asyncio
    async def test_returns_default_result(self):
        adapter = FakeExecutionAdapter()
        r = await adapter.run("session", "do something")

        assert r.summary == "ok"
        assert r.error is None

    @pytest.mark.asyncio
    async def test_returns_custom_result(self):
        custom = make_agent_run_result(summary="custom", error="boom")
        adapter = FakeExecutionAdapter(result=custom)
        r = await adapter.run("session", "do something")

        assert r.summary == "custom"
        assert r.error == "boom"

    @pytest.mark.asyncio
    async def test_records_calls(self):
        adapter = FakeExecutionAdapter()
        await adapter.run("s1", "r1", adata="adata1")

        assert len(adapter.run_calls) == 1
        assert adapter.run_calls[0]["session"] == "s1"
        assert adapter.run_calls[0]["request"] == "r1"
        assert adapter.run_calls[0]["adata"] == "adata1"


class TestFakeRouter:
    def test_returns_default_session(self):
        router = FakeRouter()
        s = router.get_session("anything")
        assert s is not None

    def test_returns_per_route_override(self):
        sentinel = object()
        router = FakeRouter(sessions={"key1": sentinel})

        class _Route:
            def route_key(self):
                return "key1"

        assert router.get_session(_Route()) is sentinel


# ===================================================================
#  Helper factories
# ===================================================================


class TestHelperFactories:
    def test_make_tool_call(self):
        tc = make_tool_call("execute_code", {"code": "print(1)"})
        assert tc.name == "execute_code"
        assert tc.arguments == {"code": "print(1)"}
        assert tc.id.startswith("call_")

    def test_make_usage(self):
        u = make_usage(5, 15)
        assert u.input_tokens == 5
        assert u.output_tokens == 15
        assert u.total_tokens == 20

    def test_make_chat_response_defaults(self):
        cr = make_chat_response()
        assert cr.content is None
        assert cr.tool_calls == []
        assert cr.stop_reason == "end_turn"
        assert cr.usage is not None

    def test_make_chat_response_with_content(self):
        cr = make_chat_response(content="hello", stop_reason="tool_use")
        assert cr.content == "hello"
        assert cr.stop_reason == "tool_use"

    def test_make_agent_run_result(self):
        r = make_agent_run_result(summary="done", error=None)
        assert r.summary == "done"
        assert r.error is None
        assert r.figures == []

    def test_build_fake_llm_shortcut(self):
        llm = build_fake_llm(["a"], model="m", provider="p")
        assert isinstance(llm, FakeLLM)
        assert llm.config.model == "m"

    def test_build_fake_tool_runtime_shortcut(self):
        rt = build_fake_tool_runtime({"t": lambda a, d, r: "ok"})
        assert isinstance(rt, FakeToolRuntime)


# ===================================================================
#  Jarvis model factories (conditional on import availability)
# ===================================================================


try:
    from omicverse.jarvis.runtime.models import (
        ConversationRoute,
        MessageEnvelope,
    )
    _HAS_JARVIS = True
except Exception:
    _HAS_JARVIS = False


@pytest.mark.skipif(not _HAS_JARVIS, reason="Jarvis runtime models not importable")
class TestJarvisFactories:
    def test_make_route(self):
        route = make_route(channel="telegram", scope_id="chat-42")
        assert isinstance(route, ConversationRoute)
        assert route.channel == "telegram"
        assert route.scope_id == "chat-42"
        assert route.scope_type == "dm"

    def test_make_envelope(self):
        env = make_envelope(text="analyze this", sender_id="user-7")
        assert isinstance(env, MessageEnvelope)
        assert env.text == "analyze this"
        assert env.sender_id == "user-7"

    def test_make_envelope_with_custom_route(self):
        route = make_route(channel="discord", scope_type="group", scope_id="g1")
        env = make_envelope(text="hi", route=route)
        assert env.route.channel == "discord"
        assert env.route.scope_type == "group"


# ===================================================================
#  Jarvis runtime deps bundle
# ===================================================================


@pytest.mark.skipif(not _HAS_JARVIS, reason="Jarvis runtime models not importable")
class TestJarvisRuntimeDepsBundle:
    def test_build_returns_all_keys(self):
        deps = build_jarvis_runtime_deps()
        assert "presenter" in deps
        assert "execution_adapter" in deps
        assert "router" in deps
        assert "deliver" in deps
        assert "delivered" in deps
        assert isinstance(deps["presenter"], FakePresenter)
        assert isinstance(deps["execution_adapter"], FakeExecutionAdapter)
        assert isinstance(deps["router"], FakeRouter)
        assert callable(deps["deliver"])
        assert isinstance(deps["delivered"], list)

    @pytest.mark.asyncio
    async def test_deliver_accumulates_events(self):
        deps = build_jarvis_runtime_deps()
        await deps["deliver"]("event-1")
        await deps["deliver"]("event-2")
        assert deps["delivered"] == ["event-1", "event-2"]


# ===================================================================
#  Fixture integration (verifies conftest wiring)
# ===================================================================


class TestFixtureWiring:
    """Verify that the conftest fixtures are usable from this test file."""

    def test_fake_llm_fixture_is_factory(self, fake_llm):
        llm = fake_llm(["response"])
        assert isinstance(llm, FakeLLM)
        assert llm.remaining_responses == 1

    def test_simple_fake_llm_fixture(self, simple_fake_llm):
        assert isinstance(simple_fake_llm, FakeLLM)
        assert simple_fake_llm.remaining_responses == 1

    @pytest.mark.asyncio
    async def test_echo_tool_runtime_fixture(self, echo_tool_runtime):
        assert isinstance(echo_tool_runtime, FakeToolRuntime)
        result = await echo_tool_runtime.dispatch("echo", {"text": "hi"}, None, "req")
        assert "hi" in result
        schemas = echo_tool_runtime.get_visible_agent_tools()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "echo"

    def test_fake_presenter_fixture(self, fake_presenter):
        assert isinstance(fake_presenter, FakePresenter)
        assert fake_presenter.events == []

    def test_fake_execution_adapter_fixture(self, fake_execution_adapter):
        adapter = fake_execution_adapter()
        assert isinstance(adapter, FakeExecutionAdapter)

    @pytest.mark.skipif(not _HAS_JARVIS, reason="Jarvis models not importable")
    def test_fake_route_fixture(self, fake_route):
        assert fake_route is not None

    @pytest.mark.skipif(not _HAS_JARVIS, reason="Jarvis models not importable")
    def test_fake_envelope_fixture(self, fake_envelope):
        assert fake_envelope is not None

    @pytest.mark.skipif(not _HAS_JARVIS, reason="Jarvis models not importable")
    def test_jarvis_runtime_deps_fixture(self, jarvis_runtime_deps):
        assert "presenter" in jarvis_runtime_deps
        assert "delivered" in jarvis_runtime_deps


# ===================================================================
#  Multi-turn conversation scenario (integration smoke test)
# ===================================================================


class TestMultiTurnScenario:
    """Verify that fakes compose for a realistic multi-turn agent flow."""

    @pytest.mark.asyncio
    async def test_tool_call_then_text_response(self):
        tc = make_tool_call("inspect_data", {"aspect": "shape"})
        responses = [
            make_chat_response(tool_calls=[tc], stop_reason="tool_use"),
            make_chat_response(content="The data has 100 rows.", stop_reason="end_turn"),
        ]
        llm = FakeLLM(responses)
        rt = FakeToolRuntime(
            {"inspect_data": lambda a, d, r: "shape: (100, 5)"}
        )

        # Turn 1: LLM requests a tool call
        r1 = await llm.chat([{"role": "user", "content": "describe the data"}])
        assert r1.stop_reason == "tool_use"
        assert len(r1.tool_calls) == 1

        # Dispatch the tool
        tool_result = await rt.dispatch(
            r1.tool_calls[0].name, r1.tool_calls[0].arguments, None, "describe"
        )
        assert "100" in tool_result

        # Turn 2: Feed tool result back, get final answer
        r2 = await llm.chat([
            {"role": "user", "content": "describe the data"},
            {"role": "assistant", "content": None, "tool_calls": [r1.tool_calls[0]]},
            llm.format_tool_result_message(r1.tool_calls[0].id, "inspect_data", tool_result),
        ])
        assert r2.stop_reason == "end_turn"
        assert "100 rows" in r2.content

        # Verify call history
        assert len(llm.chat_calls) == 2
        assert len(rt.dispatch_calls) == 1

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_sequence(self):
        tc1 = make_tool_call("search_functions", {"query": "pca"})
        tc2 = make_tool_call("execute_code", {"code": "print(1)"})
        responses = [
            make_chat_response(tool_calls=[tc1], stop_reason="tool_use"),
            make_chat_response(tool_calls=[tc2], stop_reason="tool_use"),
            make_chat_response(content="Done.", stop_reason="end_turn"),
        ]
        llm = FakeLLM(responses)
        rt = FakeToolRuntime({
            "search_functions": lambda a, d, r: "found: ov.pp.pca",
            "execute_code": lambda a, d, r: "output: 1",
        })

        turns = []
        for _ in range(3):
            resp = await llm.chat([])
            turns.append(resp)
            if resp.tool_calls:
                for tc in resp.tool_calls:
                    await rt.dispatch(tc.name, tc.arguments, None, "req")

        assert turns[0].stop_reason == "tool_use"
        assert turns[1].stop_reason == "tool_use"
        assert turns[2].stop_reason == "end_turn"
        assert len(rt.dispatch_calls) == 2


# ===================================================================
#  AC-001-5: No production code depends on test-only harness helpers
# ===================================================================


class TestNoProductionDependency:
    """Verify that production source files do not import from the test harness."""

    def test_no_production_imports_from_harness(self):
        """Scan production Python files for imports of the integration harness."""
        from pathlib import Path

        prod_root = Path(__file__).resolve().parents[2] / "omicverse"
        violations = []

        for py_file in prod_root.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(source.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "tests.integration" in stripped or "tests/integration" in stripped:
                    violations.append(f"{py_file}:{i}: {stripped}")

        assert violations == [], (
            f"Production code must not import from the test harness:\n"
            + "\n".join(violations)
        )
