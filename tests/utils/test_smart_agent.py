import ast
import asyncio
import importlib.machinery
import importlib.util
import os
import sys
import threading
import types
from pathlib import Path
from types import MethodType
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ORIGINAL_MODULES = {
    name: sys.modules.get(name)
    for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]
}
for name in ["omicverse", "omicverse.utils", "omicverse.utils.smart_agent"]:
    sys.modules.pop(name, None)

omicverse_pkg = types.ModuleType("omicverse")
omicverse_pkg.__path__ = [str(PACKAGE_ROOT)]
omicverse_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = omicverse_pkg

utils_pkg = types.ModuleType("omicverse.utils")
utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = utils_pkg
omicverse_pkg.utils = utils_pkg

smart_agent_spec = importlib.util.spec_from_file_location(
    "omicverse.utils.smart_agent", PACKAGE_ROOT / "utils" / "smart_agent.py"
)
smart_agent_module = importlib.util.module_from_spec(smart_agent_spec)
sys.modules["omicverse.utils.smart_agent"] = smart_agent_module
assert smart_agent_spec.loader is not None
smart_agent_spec.loader.exec_module(smart_agent_module)

OmicVerseAgent = smart_agent_module.OmicVerseAgent
_run_coroutine_sync = smart_agent_module._run_coroutine_sync
from omicverse.utils.harness import build_stream_event
from omicverse.utils.ovagent.tool_runtime import ToolRuntime

for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module

_RUN_HARNESS_TESTS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _build_agent(return_value):
    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    async def _fake_run_async(self, request, adata):
        return {
            "request": request,
            "adata": adata,
            "value": return_value,
        }

    agent.run_async = MethodType(_fake_run_async, agent)
    return agent


def test_extract_python_code_includes_function_defs():
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    response_text = """
LLM response:
import numpy as np
def summarize_counts(adata):
    counts = []
    for value in np.sum(adata.X, axis=1):
        counts.append(float(value))
    return counts
for _ in range(1):
    totals = summarize_counts(adata)
    print(min(totals))
"""

    code = agent._codegen.extract_python_code(response_text)

    assert "def summarize_counts" in code
    assert "counts.append" in code
    ast.parse(code)


def test_run_outside_event_loop(monkeypatch):
    agent = _build_agent(return_value="plain")
    original_asyncio_run = asyncio.run
    run_threads = []

    def _tracking_run(coro):
        run_threads.append(threading.current_thread().name)
        return original_asyncio_run(coro)

    monkeypatch.setattr(asyncio, "run", _tracking_run)

    sentinel = object()
    result = agent.run("plain request", sentinel)

    assert result["request"] == "plain request"
    assert result["adata"] is sentinel
    assert result["value"] == "plain"
    assert run_threads == [threading.current_thread().name]


def test_run_inside_running_loop(monkeypatch):
    agent = _build_agent(return_value="nested")
    original_asyncio_run = asyncio.run
    run_threads = []

    def _tracking_run(coro):
        run_threads.append(threading.current_thread().name)
        return original_asyncio_run(coro)

    async def _caller():
        monkeypatch.setattr(asyncio, "run", _tracking_run)
        sentinel = object()
        result = agent.run("loop request", sentinel)

        assert result["request"] == "loop request"
        assert result["adata"] is sentinel
        assert result["value"] == "nested"
        assert run_threads and run_threads[0] != threading.current_thread().name

    asyncio.run(_caller())


@pytest.mark.skipif(
    not _RUN_HARNESS_TESTS,
    reason="Harness streaming test is server-only and requires OV_AGENT_RUN_HARNESS_TESTS=1.",
)
def test_stream_async_emits_harness_metadata():
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    agent._last_run_trace = None
    agent._get_harness_session_id = MethodType(lambda self: "session-1", agent)

    async def _fake_run_agentic_loop(self, request, adata, event_callback=None, cancel_event=None, history=None, approval_handler=None, request_content=None):
        await event_callback(build_stream_event(
            "tool_call",
            {"name": "inspect_data", "arguments": {"aspect": "shape"}},
            turn_id="turn-1",
            trace_id="trace-1",
            step_id="step-1",
            session_id="session-1",
            category="tool",
        ))

    agent._run_agentic_loop = MethodType(_fake_run_agentic_loop, agent)

    async def _collect():
        events = []
        async for event in agent.stream_async("inspect", object()):
            events.append(event)
        assert len(events) == 1
        assert events[0]["type"] == "tool_call"
        assert events[0]["turn_id"] == "turn-1"
        assert events[0]["trace_id"] == "trace-1"
        assert events[0]["step_id"] == "step-1"
        assert events[0]["session_id"] == "session-1"

    asyncio.run(_collect())


def test_url_request_requires_tool_action_and_promissory_text_retries():
    # After task-050, FollowUpGate methods are no longer on the facade mixin;
    # test them directly on the FollowUpGate class.
    from omicverse.utils.ovagent.turn_controller import FollowUpGate

    assert FollowUpGate.request_requires_tool_action(
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135893\n分析这个数据",
        None,
    ) is True
    assert FollowUpGate.response_is_promissory("Let me first fetch the GEO page to understand this dataset.") is True
    assert FollowUpGate.select_tool_choice(
        request="https://example.com\nanalyze this",
        adata=None,
        turn_index=0,
        had_meaningful_tool_call=False,
        forced_retry=False,
    ) == "required"


def test_generate_code_async_reuses_agentic_loop_and_captures_execute_code():
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    agent.provider = "openai"
    agent._code_only_mode = False
    agent._code_only_captured_code = ""
    agent._code_only_captured_history = []

    seen = {}

    async def _fake_run_agentic_loop(self, request, adata, event_callback=None, cancel_event=None, history=None, approval_handler=None, request_content=None):
        seen["request"] = request
        assert self._code_only_mode is True
        self._codegen.capture_code_only_snippet("import omicverse as ov\nov.pp.qc(adata)")
        await event_callback(build_stream_event(
            "tool_call",
            {"name": "execute_code", "arguments": {"description": "qc"}},
            turn_id="turn-1",
            trace_id="trace-1",
            session_id="session-1",
            category="tool",
        ))
        await event_callback(build_stream_event(
            "code",
            "import omicverse as ov\nov.pp.qc(adata)",
            turn_id="turn-1",
            trace_id="trace-1",
            session_id="session-1",
            category="execution",
        ))
        await event_callback(build_stream_event(
            "done",
            "captured",
            turn_id="turn-1",
            trace_id="trace-1",
            session_id="session-1",
            category="lifecycle",
        ))

    agent._run_agentic_loop = MethodType(_fake_run_agentic_loop, agent)

    result = asyncio.run(agent.generate_code_async("basic qc and clustering", None))

    assert result == "import omicverse as ov\nov.pp.qc(adata)"
    assert "CLAW REQUEST MODE" in seen["request"]
    assert "basic qc and clustering" in seen["request"]
    assert agent._code_only_mode is False


def test_tool_execute_code_in_code_only_mode_captures_without_execution():
    from omicverse.utils.ovagent.tool_runtime_exec import handle_execute_code
    captured = {}

    class _FakePipeline:
        def capture_code_only_snippet(self, code, description=""):
            captured["code"] = code
            captured["description"] = description

    class _Ctx:
        _code_only_mode = True

    class _Executor:
        _codegen_pipeline = _FakePipeline()

        def check_code_prerequisites(self, code, adata):
            raise AssertionError("should not check prerequisites in code-only mode")

        def execute_generated_code(self, code, adata, capture_stdout=True):
            raise AssertionError("should not execute code in code-only mode")

    result = handle_execute_code(
        _Ctx(), _Executor(),
        "import omicverse as ov\nov.pp.pca(adata)", "pca", None,
    )

    assert "captured generated Python code" in result["output"]
    assert captured["description"] == "pca"
    assert "ov.pp.pca" in captured["code"]


def test_static_registry_scanner_indexes_celltypist_method_branch():
    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    entries = agent._scanner.load_static_entries()

    assert any(
        entry.get("source") == "static_ast_branch"
        and "celltypist" == str(entry.get("branch_value", "")).lower()
        and "Annotation.annotate" in str(entry.get("full_name", ""))
        for entry in entries
    )


def test_static_registry_scanner_indexes_dynamo_method_branch():
    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    entries = agent._scanner.load_static_entries()

    assert any(
        entry.get("source") == "static_ast_branch"
        and "dynamo" == str(entry.get("branch_value", "")).lower()
        and "Velo" in str(entry.get("full_name", ""))
        for entry in entries
    )


@pytest.mark.skipif(
    not _RUN_HARNESS_TESTS,
    reason="Loop retry regression is validated only in the Taiwan harness environment.",
)
def test_agentic_loop_retries_after_text_only_promise_until_tool_call():
    from omicverse.utils.ovagent.turn_controller import TurnController

    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    agent.model = "claude-opus-4-6-thinking"
    agent.provider = "anthropic"
    agent.last_usage = None
    agent._approval_handler = None
    agent._trace_store = None
    agent._context_compactor = None
    agent._session_history = None
    agent._last_run_trace = None
    agent._ov_runtime = None
    agent._active_run_id = None
    agent.skill_registry = None
    agent._config = SimpleNamespace(
        execution=SimpleNamespace(max_agent_turns=4),
        harness=SimpleNamespace(
            include_recent_failures_in_prompt=False,
            max_recent_failures=3,
            record_artifacts=False,
        ),
    )
    agent._get_harness_session_id = MethodType(lambda self: "session-test", agent)
    agent._get_runtime_session_id = MethodType(lambda self: "session-test", agent)
    agent._get_visible_agent_tools = MethodType(
        lambda self, allowed_names=None: [
            {"name": "WebFetch", "description": "fetch", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
            {"name": "finish", "description": "finish", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
        ],
        agent,
    )

    chat_calls = []
    dispatch_log = []

    responses = [
        SimpleNamespace(
            content="Let me first fetch the GEO page to understand this dataset and find the download links.",
            tool_calls=[],
            usage=None,
            raw_message={"role": "assistant", "content": "Let me first fetch the GEO page to understand this dataset and find the download links."},
        ),
        SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="tool_1",
                    name="WebFetch",
                    arguments={"url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135893"},
                )
            ],
            usage=None,
            raw_message={"role": "assistant", "content": [{"type": "tool_use", "id": "tool_1", "name": "WebFetch", "input": {"url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135893"}}]},
        ),
        SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="tool_2",
                    name="finish",
                    arguments={"summary": "Fetched the GEO page and continued the workflow."},
                )
            ],
            usage=None,
            raw_message={"role": "assistant", "content": [{"type": "tool_use", "id": "tool_2", "name": "finish", "input": {"summary": "Fetched the GEO page and continued the workflow."}}]},
        ),
    ]

    class _FakeLLM:
        def __init__(self, staged_responses):
            self._responses = list(staged_responses)
            self.config = SimpleNamespace(model="claude-opus-4-6-thinking", provider="anthropic")

        async def chat(self, messages, tools=None, tool_choice=None):
            chat_calls.append({"tool_choice": tool_choice, "message_count": len(messages)})
            if not self._responses:
                raise AssertionError("No more fake responses available")
            return self._responses.pop(0)

    agent._llm = _FakeLLM(responses)

    # Minimal prompt builder for TurnController
    class _FakePromptBuilder:
        def build_agentic_system_prompt(self):
            return "You are a test agent."
        def build_initial_user_message(self, request, adata, extra_content=None):
            return request

    # Minimal tool registry and runtime for TurnController
    class _FakeToolRegistry:
        def resolve_name(self, name):
            return name
        def get(self, name):
            return None

    class _FakeToolRuntime:
        def __init__(self):
            self.registry = _FakeToolRegistry()
        def get_visible_agent_tools(self, *, allowed_names=None):
            return [
                {"name": "WebFetch", "description": "fetch", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
                {"name": "finish", "description": "finish", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
            ]
        async def dispatch_tool(self, tool_call, current_adata, request, **kw):
            dispatch_log.append(tool_call.name)
            if tool_call.name == "WebFetch":
                return "Content from GEO page"
            if tool_call.name == "finish":
                return {"finished": True, "summary": tool_call.arguments.get("summary", "")}
            return "ok"

    agent._turn_controller = TurnController(
        agent, _FakePromptBuilder(), _FakeToolRuntime(),
    )

    async def _run():
        result = await agent._run_agentic_loop(
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135893\n分析这个数据",
            None,
        )
        assert result is None

    asyncio.run(_run())

    assert [call["tool_choice"] for call in chat_calls[:2]] == ["required", "required"]
    assert "WebFetch" in dispatch_log


# -----------------------------------------------------------------------
# Task-008: Runtime observability tests
# -----------------------------------------------------------------------

from omicverse.utils.agent_reporter import (
    AgentEvent,
    EventLevel,
    SilentReporter,
)


def test_run_async_silent_reporter_no_stdout():
    """AC-001.2: SilentReporter mode produces no stdout leakage for runtime paths."""
    import io
    from contextlib import redirect_stdout

    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    # Wire up SilentReporter exactly as __init__ would
    agent._reporter = SilentReporter()

    def _emit(level, message, category=""):
        agent._reporter.emit(AgentEvent(level=level, message=message, category=category))

    agent._emit = _emit
    agent.provider = "python"
    agent.last_usage = None
    agent.last_usage_breakdown = {
        "generation": None,
        "reflection": [],
        "review": [],
        "total": None,
    }

    # Stub the direct-python path so run_async never touches the LLM.
    # After task-050, smart_agent calls subsystems directly instead of
    # facade wrappers, so we mock the subsystem objects.
    from types import SimpleNamespace as _NS
    agent._codegen_pipeline = _NS(detect_direct_python_request=lambda request: "x = 1")
    agent._analysis_executor = _NS(execute_generated_code=lambda code, adata: adata)

    captured = io.StringIO()
    with redirect_stdout(captured):
        result = asyncio.run(agent.run_async("x = 1", None))

    stdout_text = captured.getvalue()
    assert stdout_text == "", (
        f"SilentReporter leaked to stdout: {stdout_text!r}"
    )


def test_del_fallback_logs_debug_on_exception(caplog):
    """AC-001.3/4: __del__ fallback sites emit debug diagnostics."""
    import logging as _logging

    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    class _FailingExecutor:
        def shutdown(self):
            raise RuntimeError("test shutdown boom")

    agent._notebook_executor = _FailingExecutor()
    agent._filesystem_context = None  # only notebook path triggers

    with caplog.at_level(_logging.DEBUG, logger="omicverse.utils.smart_agent"):
        agent.__del__()

    debug_msgs = [
        rec
        for rec in caplog.records
        if rec.levelno == _logging.DEBUG and "notebook executor shutdown failed" in rec.message
    ]
    assert len(debug_msgs) == 1, (
        f"Expected exactly 1 debug log about notebook executor, got {len(debug_msgs)}"
    )


# -----------------------------------------------------------------------
# Task-014: Sync/async entrypoint stabilization
# -----------------------------------------------------------------------

import traceback


def _build_agent_that_raises(exc):
    """Build a minimal agent whose run_async raises *exc*."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    async def _fake_run_async(self, request, adata):
        raise exc

    agent.run_async = MethodType(_fake_run_async, agent)
    return agent


# --- AC-001.1: Traceback preservation ---------------------------------

def test_run_preserves_traceback_outside_event_loop():
    """run() outside a loop preserves the async traceback chain."""
    agent = _build_agent_that_raises(ValueError("outside-loop-error"))

    with pytest.raises(ValueError, match="outside-loop-error") as exc_info:
        agent.run("fail", None)

    tb_text = "".join(traceback.format_exception(
        type(exc_info.value), exc_info.value, exc_info.value.__traceback__,
    ))
    assert "_fake_run_async" in tb_text, (
        f"Traceback should contain the original async function name:\n{tb_text}"
    )


def test_run_preserves_traceback_inside_running_loop():
    """run() inside a running loop preserves the async traceback chain."""
    agent = _build_agent_that_raises(ValueError("inside-loop-error"))

    async def _caller():
        with pytest.raises(ValueError, match="inside-loop-error") as exc_info:
            agent.run("fail", None)

        tb_text = "".join(traceback.format_exception(
            type(exc_info.value), exc_info.value, exc_info.value.__traceback__,
        ))
        assert "_fake_run_async" in tb_text, (
            f"Traceback should contain the original async function:\n{tb_text}"
        )

    asyncio.run(_caller())


def test_run_preserves_exception_type_and_message():
    """Exception type and message survive the sync bridge unchanged."""

    class CustomAgentError(RuntimeError):
        pass

    agent = _build_agent_that_raises(CustomAgentError("custom-payload"))

    with pytest.raises(CustomAgentError, match="custom-payload"):
        agent.run("fail", None)


def test_run_preserves_traceback_inside_loop_chained():
    """The traceback chain inside a running loop includes intermediate frames."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)

    async def _inner_helper():
        raise RuntimeError("deep-origin")

    async def _fake_run_async(self, request, adata):
        await _inner_helper()

    agent.run_async = MethodType(_fake_run_async, agent)

    async def _caller():
        with pytest.raises(RuntimeError, match="deep-origin") as exc_info:
            agent.run("fail", None)

        tb_text = "".join(traceback.format_exception(
            type(exc_info.value), exc_info.value, exc_info.value.__traceback__,
        ))
        assert "_inner_helper" in tb_text, (
            f"Traceback should include the inner helper frame:\n{tb_text}"
        )

    asyncio.run(_caller())


# --- AC-001.2: Event-loop-present regression tests --------------------

def test_event_loop_present_uses_worker_thread():
    """When a loop is already running, _run_coroutine_sync delegates to a thread."""
    worker_thread_names = []
    main_thread_name = threading.current_thread().name

    async def _track_thread():
        worker_thread_names.append(threading.current_thread().name)
        return "threaded-result"

    async def _caller():
        result = _run_coroutine_sync(_track_thread())
        assert result == "threaded-result"
        assert len(worker_thread_names) == 1
        assert worker_thread_names[0] != main_thread_name

    asyncio.run(_caller())


def test_no_event_loop_uses_main_thread():
    """Without a running loop, _run_coroutine_sync uses asyncio.run on the current thread."""
    worker_thread_names = []

    async def _track_thread():
        worker_thread_names.append(threading.current_thread().name)
        return "direct-result"

    result = _run_coroutine_sync(_track_thread())
    assert result == "direct-result"
    assert len(worker_thread_names) == 1
    assert worker_thread_names[0] == threading.current_thread().name


# --- AC-001.3: No silent nest_asyncio patching ------------------------

def test_no_nest_asyncio_import_after_run():
    """run() must not import or patch nest_asyncio as a side effect."""
    agent = _build_agent(return_value="clean")
    agent.run("test", None)
    assert "nest_asyncio" not in sys.modules, (
        "nest_asyncio was imported as a side effect of run()"
    )


def test_no_nest_asyncio_import_inside_running_loop():
    """run() inside a running loop must not import nest_asyncio."""
    agent = _build_agent(return_value="clean-nested")

    async def _caller():
        agent.run("test", None)
        assert "nest_asyncio" not in sys.modules, (
            "nest_asyncio was imported inside a running loop"
        )

    asyncio.run(_caller())


# --- AC-001.4: Public signature unchanged -----------------------------

def test_run_signature_unchanged():
    """run() accepts (request: str, adata: Any) and returns Any."""
    import inspect
    sig = inspect.signature(OmicVerseAgent.run)
    param_names = list(sig.parameters.keys())
    assert param_names == ["self", "request", "adata"]


def test_generate_code_signature_unchanged():
    """generate_code() signature is preserved."""
    import inspect
    sig = inspect.signature(OmicVerseAgent.generate_code)
    param_names = list(sig.parameters.keys())
    assert param_names == ["self", "request", "adata", "max_functions", "progress_callback"]


def test_run_async_signature_unchanged():
    """run_async() signature is preserved."""
    import inspect
    sig = inspect.signature(OmicVerseAgent.run_async)
    param_names = list(sig.parameters.keys())
    assert param_names == ["self", "request", "adata"]


def test_generate_code_async_signature_unchanged():
    """generate_code_async() signature is preserved."""
    import inspect
    sig = inspect.signature(OmicVerseAgent.generate_code_async)
    param_names = list(sig.parameters.keys())
    assert param_names == ["self", "request", "adata", "max_functions", "progress_callback"]


# --- AC-001.5: Backward compatibility ---------------------------------

def test_run_returns_value_outside_loop():
    """run() returns the coroutine result when no loop is running."""
    sentinel = object()
    agent = _build_agent(return_value=sentinel)
    result = agent.run("request", None)
    assert result["value"] is sentinel


def test_run_returns_value_inside_loop():
    """run() returns the coroutine result when called inside a running loop."""
    sentinel = object()
    agent = _build_agent(return_value=sentinel)

    async def _caller():
        result = agent.run("request", None)
        assert result["value"] is sentinel

    asyncio.run(_caller())


def test_generate_code_sync_delegates_to_async():
    """generate_code() calls generate_code_async through the sync bridge."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    calls = []

    async def _fake_generate_code_async(self, request, adata=None, *, max_functions=8, progress_callback=None):
        calls.append({"request": request, "adata": adata, "max_functions": max_functions})
        return "generated-code"

    agent.generate_code_async = MethodType(_fake_generate_code_async, agent)

    result = agent.generate_code("build a plot", None, max_functions=5)
    assert result == "generated-code"
    assert len(calls) == 1
    assert calls[0]["request"] == "build a plot"
    assert calls[0]["max_functions"] == 5


# -----------------------------------------------------------------------
# Task-027 (reconciled task-013): Codegen / tool-dispatch facade extraction
# Task-050: Facade wrapper collapse and closure audit
# -----------------------------------------------------------------------

from omicverse.utils.ovagent.codegen_tool_facade import CodegenToolDispatchFacadeMixin


def test_codegen_tool_facade_mixin_is_base_of_agent():
    """AC-001.1: OmicVerseAgent inherits from CodegenToolDispatchFacadeMixin."""
    assert issubclass(OmicVerseAgent, CodegenToolDispatchFacadeMixin)


# -- Surviving mixin surface (task-050) ----------------------------------

# After task-050, CodegenToolDispatchFacadeMixin retains only:
#   properties: _codegen, _scanner
#   methods:    _collect_static_registry_entries
# All other ~42 wrapper methods were removed because consumers were rewired
# to hold direct subsystem references in task-049.

_ALLOWED_MIXIN_SURFACE = {
    # Lazy-init properties (needed for __new__-based test agents)
    "_codegen",
    "_scanner",
    # Compatibility wrapper (accessed via getattr in tool_runtime_exec)
    "_collect_static_registry_entries",
}


def test_mixin_surface_is_minimal():
    """AC-050.1: Mixin defines only the allowed residual surface."""
    dunder_ignore = {"__module__", "__doc__", "__dict__", "__weakref__",
                     "__firstlineno__", "__static_attributes__", "__qualname__"}
    actual = set(CodegenToolDispatchFacadeMixin.__dict__.keys()) - dunder_ignore
    assert actual == _ALLOWED_MIXIN_SURFACE, (
        f"Mixin surface drift — "
        f"unexpected: {actual - _ALLOWED_MIXIN_SURFACE}, "
        f"missing: {_ALLOWED_MIXIN_SURFACE - actual}"
    )


def test_collect_static_registry_entries_available_on_agent():
    """AC-050.2: The compatibility wrapper resolves on the concrete agent."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    assert callable(getattr(agent, "_collect_static_registry_entries", None))


def test_codegen_lazy_property_via_mixin():
    """AC-050.2: _codegen lazy property works through mixin for __new__ instances."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    pipeline = agent._codegen
    from omicverse.utils.ovagent.codegen_pipeline import CodegenPipeline
    assert isinstance(pipeline, CodegenPipeline)
    assert agent._codegen is pipeline


def test_scanner_lazy_property_via_mixin():
    """AC-050.2: _scanner lazy property works through mixin for __new__ instances."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    scanner = agent._scanner
    from omicverse.utils.ovagent.registry_scanner import RegistryScanner
    assert isinstance(scanner, RegistryScanner)
    assert agent._scanner is scanner


def test_extract_python_code_via_codegen_property():
    """AC-050.2: Code extraction works through _codegen property (no facade wrapper)."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    response = """Here is the code:
```python
import omicverse as ov
ov.pp.qc(adata)
```
"""
    code = agent._codegen.extract_python_code(response)
    assert "ov.pp.qc" in code
    ast.parse(code)


# -- Closure tests: removed wrappers must NOT reappear -------------------

# Exhaustive list of wrappers removed in task-050.  If any name reappears
# on CodegenToolDispatchFacadeMixin, the closure test fails — forcing an
# explicit justification before the surface can grow again.
_RETIRED_FACADE_WRAPPERS = frozenset({
    # Tool visibility (ToolRuntime)
    "_get_visible_agent_tools",
    "_get_loaded_tool_names",
    "_tool_blocked_in_plan_mode",
    # Tool dispatch (ToolRuntime)
    "_dispatch_tool",
    # FollowUp gate (TurnController / FollowUpGate)
    "_request_requires_tool_action",
    "_response_is_promissory",
    "_select_agent_tool_choice",
    # Registry scanner (all but _collect_static_registry_entries)
    "_load_static_registry_entries",
    "_collect_relevant_registry_entries",
    "_score_registry_entry_for_codegen",
    # Codegen pipeline
    "_extract_python_code",
    "_normalize_registry_entry_for_codegen",
    "_capture_code_only_snippet",
    "_select_codegen_skill_matches",
    "_format_registry_context_for_codegen",
    "_format_prerequisites_for_codegen_entry",
    "_build_code_generation_system_prompt",
    "_build_code_generation_user_prompt",
    "_contains_forbidden_scanpy_usage",
    "_rewrite_scanpy_calls_with_registry",
    "_rewrite_code_without_scanpy",
    "_review_generated_code_lightweight",
    "_build_code_only_agentic_request",
    "_generate_code_via_agentic_loop",
    "_gather_code_candidates",
    "_looks_like_python",
    "_extract_inline_python",
    "_normalize_code_candidate",
    "_extract_python_code_strict",
    "_review_result",
    "_reflect_on_code",
    "_detect_direct_python_request",
    "_merge_usage_stats",
    # Analysis executor
    "_check_code_prerequisites",
    "_apply_execution_error_fix",
    "_extract_package_name",
    "_auto_install_package",
    "_diagnose_error_with_llm",
    "_validate_outputs",
    "_generate_completion_code",
    "_request_approval",
    "_execute_generated_code",
    "_normalize_doublet_obs",
    "_process_context_directives",
    "_build_sandbox_globals",
})


def test_retired_wrappers_not_on_mixin():
    """AC-050.3: Removed wrappers must not silently reappear on the mixin."""
    mixin_names = set(CodegenToolDispatchFacadeMixin.__dict__.keys())
    regrown = mixin_names & _RETIRED_FACADE_WRAPPERS
    assert not regrown, (
        f"Retired facade wrappers reappeared on CodegenToolDispatchFacadeMixin — "
        f"if intentional, remove them from _RETIRED_FACADE_WRAPPERS and justify: "
        f"{sorted(regrown)}"
    )


def test_mixin_method_count_ceiling():
    """AC-050.3: Hard ceiling prevents gradual wrapper accumulation."""
    dunder_ignore = {"__module__", "__doc__", "__dict__", "__weakref__",
                     "__firstlineno__", "__static_attributes__", "__qualname__"}
    actual_count = len(set(CodegenToolDispatchFacadeMixin.__dict__.keys()) - dunder_ignore)
    # Current surface is 3 items (2 properties + 1 method).
    # Allow a small buffer for justified additions, but fail if it grows
    # back toward the old ~45-method shape.
    assert actual_count <= 6, (
        f"CodegenToolDispatchFacadeMixin has {actual_count} members — "
        f"expected ≤ 6.  If growth is justified, raise the ceiling and "
        f"document the reason."
    )


def test_no_provider_logic_in_facade_mixin():
    """AC-001.5: Mixin does not import or reference provider-specific modules."""
    import inspect
    source = inspect.getsource(CodegenToolDispatchFacadeMixin)
    for provider_term in ["openai", "anthropic", "google", "bedrock", "groq"]:
        assert provider_term not in source.lower(), (
            f"Provider logic '{provider_term}' found in CodegenToolDispatchFacadeMixin"
        )


def test_mixin_methods_not_duplicated_on_agent_class():
    """AC-001.1: Mixin members are NOT redefined on OmicVerseAgent itself."""
    dunder_ignore = {"__module__", "__doc__", "__dict__", "__weakref__",
                     "__firstlineno__", "__static_attributes__", "__qualname__"}
    agent_own = set(OmicVerseAgent.__dict__.keys()) - dunder_ignore
    mixin_own = set(CodegenToolDispatchFacadeMixin.__dict__.keys()) - dunder_ignore
    overlap = agent_own & mixin_own
    assert not overlap, (
        f"These members are duplicated on both OmicVerseAgent and the mixin: {overlap}"
    )


# -----------------------------------------------------------------------
# Task-028: Function-registry prompt footprint optimization
# -----------------------------------------------------------------------


def test_compact_registry_summary_returns_category_lines():
    """AC-028.1: _get_compact_registry_summary produces category-level lines, not full JSON."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    summary = agent._get_compact_registry_summary()
    # Must be a non-empty string
    assert isinstance(summary, str)
    assert len(summary) > 0
    # Must NOT contain the full JSON dump pattern
    assert '"signature"' not in summary, "Compact summary should not contain full function signatures"
    assert '"examples"' not in summary, "Compact summary should not contain example lists"
    assert '"aliases"' not in summary, "Compact summary should not contain alias lists"
    # Must contain category markers
    assert "**" in summary or "functions)" in summary, (
        "Summary should contain category headings with function counts"
    )


def test_compact_registry_summary_is_much_smaller_than_full_dump():
    """AC-028.1: The compact summary is significantly smaller than the old full JSON dump."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    compact = agent._get_compact_registry_summary()
    # The compact summary should be well under 5 KB
    assert len(compact) < 5000, (
        f"Compact summary is {len(compact)} chars — expected under 5000"
    )


def test_setup_agent_instructions_do_not_contain_full_registry():
    """AC-028.1: _setup_agent system prompt uses compact summary, not full registry."""
    import inspect
    source = inspect.getsource(OmicVerseAgent._setup_agent)
    # The old code called _get_available_functions_info which returned full JSON
    assert "_get_available_functions_info" not in source, (
        "_setup_agent should no longer call _get_available_functions_info"
    )
    # The new code should reference the compact summary
    assert "_get_compact_registry_summary" in source, (
        "_setup_agent should use _get_compact_registry_summary"
    )
    # The old "Here are all the currently registered functions" dump header is gone
    assert "Here are all the currently registered functions" not in source, (
        "System prompt should not dump all functions"
    )


def test_search_functions_tool_still_works():
    """AC-028.2: search_functions tool remains functional for on-demand lookup."""
    from omicverse.utils.ovagent.tool_runtime_exec import handle_search_functions
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    # search_functions must return results for a known domain keyword
    result = handle_search_functions(agent, "qc")
    assert isinstance(result, str)
    # Should either find matches or return a "no functions" message
    assert "qc" in result.lower() or "No functions found" in result


def test_search_functions_tool_description_mentions_signatures():
    """AC-028.2: search_functions tool description is informative."""
    from omicverse.utils.ovagent.tool_runtime import LEGACY_AGENT_TOOLS
    sf_tool = next(
        (t for t in LEGACY_AGENT_TOOLS if t["name"] == "search_functions"),
        None,
    )
    assert sf_tool is not None, "search_functions tool must exist"
    desc = sf_tool["description"]
    assert "signature" in desc.lower() or "parameter" in desc.lower(), (
        "search_functions description should mention it returns signatures/parameters"
    )


def test_codegen_flows_still_discover_tools_via_scanner():
    """AC-028.3: Codegen flows still discover tools through the registry scanner."""
    agent = OmicVerseAgent.__new__(OmicVerseAgent)
    # After task-050, use the scanner directly instead of the removed facade wrapper
    entries = agent._scanner.collect_relevant_entries("quality control", max_entries=5)
    assert isinstance(entries, list)
    # The scanner should still find QC-related entries
    if entries:
        names = [e.get("full_name", "") for e in entries]
        assert any("qc" in n.lower() for n in names), (
            f"Expected QC-related entries, got: {names}"
        )


def test_no_new_runtime_dependencies():
    """AC-028.4: No new runtime dependencies introduced."""
    import importlib
    # The compact summary only uses stdlib (json removed, uses string formatting)
    # and existing internal modules. Verify no new third-party imports.
    source_path = Path(__file__).resolve().parents[2] / "omicverse" / "utils" / "smart_agent.py"
    source = source_path.read_text(encoding="utf-8")
    # Check that no new third-party imports were added beyond what existed
    # (the file already imports json, os, re, sys, etc.)
    new_suspects = ["yaml", "toml", "rich", "click", "pydantic"]
    for suspect in new_suspects:
        assert f"import {suspect}" not in source, (
            f"New runtime dependency '{suspect}' found in smart_agent.py"
        )


# -----------------------------------------------------------------------
# Task-049: Direct consumer rewiring tests
# -----------------------------------------------------------------------

import inspect as _inspect


def test_subagent_controller_uses_tool_runtime_directly():
    """AC-049.1: SubagentController calls tool_runtime.get_visible_agent_tools, not ctx."""
    from omicverse.utils.ovagent.subagent_controller import SubagentController

    src = _inspect.getsource(SubagentController._create_subagent_runtime)
    assert "self._tool_runtime.get_visible_agent_tools" in src, (
        "SubagentController should call self._tool_runtime.get_visible_agent_tools"
    )
    assert "self._ctx._get_visible_agent_tools" not in src, (
        "SubagentController should NOT route through ctx._get_visible_agent_tools"
    )


def test_turn_controller_uses_tool_runtime_directly():
    """AC-049.1: TurnController calls tool_runtime.get_visible_agent_tools, not ctx."""
    from omicverse.utils.ovagent.turn_controller import TurnController

    src = _inspect.getsource(TurnController.run_agentic_loop)
    assert "self._tool_runtime.get_visible_agent_tools" in src, (
        "TurnController should call self._tool_runtime.get_visible_agent_tools"
    )
    assert "ctx._get_visible_agent_tools" not in src, (
        "TurnController should NOT route through ctx._get_visible_agent_tools"
    )


def test_analysis_executor_holds_direct_codegen_pipeline():
    """AC-049.1: AnalysisExecutor resolves extract_code_fn from its own _codegen_pipeline."""
    from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor

    src = _inspect.getsource(AnalysisExecutor._get_extract_code_fn)
    assert "self._codegen_pipeline" in src, (
        "AnalysisExecutor should use self._codegen_pipeline"
    )
    assert 'self._ctx' not in src, (
        "AnalysisExecutor._get_extract_code_fn should NOT reach through ctx"
    )


def test_tool_runtime_exec_uses_executor_codegen_pipeline():
    """AC-049.1: handle_execute_code gets codegen_pipeline from executor, not ctx."""
    from omicverse.utils.ovagent import tool_runtime_exec

    src = _inspect.getsource(tool_runtime_exec.handle_execute_code)
    assert 'getattr(executor, "_codegen_pipeline"' in src, (
        "handle_execute_code should get pipeline from executor"
    )
    assert 'getattr(ctx, "_codegen_pipeline"' not in src, (
        "handle_execute_code should NOT reach through ctx for codegen_pipeline"
    )


def test_repair_loop_uses_executor_codegen_pipeline():
    """AC-049.1: ExecutionRepairLoop resolves extract_code_fn from executor directly."""
    from omicverse.utils.ovagent.repair_loop import ExecutionRepairLoop

    src = _inspect.getsource(ExecutionRepairLoop.run)
    assert 'getattr(self._executor, "_codegen_pipeline"' in src, (
        "RepairLoop should get pipeline from self._executor._codegen_pipeline"
    )
    assert "_ctx._codegen_pipeline" not in src, (
        "RepairLoop should NOT reach through executor._ctx._codegen_pipeline"
    )


def test_protocol_does_not_declare_retired_facade_methods():
    """AC-049.1: AgentContext protocol no longer declares tool facade methods."""
    from omicverse.utils.ovagent.protocol import AgentContext

    protocol_methods = set(dir(AgentContext))
    retired = {"_get_visible_agent_tools", "_get_loaded_tool_names", "_tool_blocked_in_plan_mode"}
    for name in retired:
        assert name not in protocol_methods, (
            f"AgentContext should no longer declare {name} after consumer rewiring"
        )


def test_analysis_executor_set_codegen_pipeline_wiring():
    """AC-049.1: AnalysisExecutor.set_codegen_pipeline correctly late-binds the pipeline."""
    from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor

    class _StubCtx:
        pass

    executor = AnalysisExecutor(_StubCtx())
    assert executor._codegen_pipeline is None

    sentinel = object()
    executor.set_codegen_pipeline(sentinel)
    assert executor._codegen_pipeline is sentinel


def test_smart_agent_init_wires_codegen_pipeline_to_executor():
    """AC-049.1: OmicVerseAgent.__init__ wires codegen_pipeline to analysis_executor."""
    src = _inspect.getsource(OmicVerseAgent.__init__)
    assert "set_codegen_pipeline" in src, (
        "OmicVerseAgent.__init__ should call analysis_executor.set_codegen_pipeline"
    )


def test_no_new_wrapper_layer_introduced():
    """AC-049.3: No new compatibility wrapper layer replaces the removed indirection."""
    from omicverse.utils.ovagent import codegen_tool_facade

    src = _inspect.getsource(codegen_tool_facade.CodegenToolDispatchFacadeMixin)
    # The facade should NOT have grown new delegation wrappers for the rewired paths
    assert "get_visible_agent_tools_direct" not in src
    assert "tool_runtime_proxy" not in src
