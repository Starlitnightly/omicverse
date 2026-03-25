"""
Tests for ovagent.event_stream — structured event emission.

Validates that:
1. EventBus correctly routes events to the underlying Reporter.
2. All runtime event helpers produce events with correct categories.
3. SilentReporter captures events without stdout output.
4. PrintReporter preserves trace-output compatibility.
5. Bootstrap/auth modules emit structured events when event_bus is provided.
6. Turn/tool/recovery status is emitted via structured events.
"""

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: same isolation pattern as other test files
# ---------------------------------------------------------------------------

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

from omicverse.utils.agent_reporter import (
    AgentEvent,
    EventLevel,
    PrintReporter,
    SilentReporter,
)
from omicverse.utils.ovagent.event_stream import EventBus, make_event_bus

for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CollectingReporter:
    """Reporter that collects events in a list for assertion."""

    def __init__(self):
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)


# ===================================================================
# 1. EventBus core
# ===================================================================

class TestEventBusCore:

    def test_emit_delegates_to_reporter(self):
        """EventBus.emit creates an AgentEvent and delegates to reporter."""
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.emit(EventLevel.INFO, "hello", category="test")
        assert len(reporter.events) == 1
        event = reporter.events[0]
        assert event.level == EventLevel.INFO
        assert event.message == "hello"
        assert event.category == "test"

    def test_emit_with_data(self):
        """EventBus.emit passes data to AgentEvent."""
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.emit(EventLevel.DEBUG, "msg", data={"key": "val"})
        assert reporter.events[0].data == {"key": "val"}

    def test_make_event_bus_default_uses_print_reporter(self, capsys):
        """make_event_bus() with no args creates a PrintReporter-backed bus."""
        bus = make_event_bus()
        bus.emit(EventLevel.DEBUG, "visible output")
        captured = capsys.readouterr()
        assert "visible output" in captured.out


# ===================================================================
# 2. Init events
# ===================================================================

class TestInitEvents:

    def test_init_emits_debug_with_init_category(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.init("subsystem ready", component="security")
        assert len(reporter.events) == 1
        ev = reporter.events[0]
        assert ev.level == EventLevel.DEBUG
        assert ev.category == "init"
        assert "subsystem ready" in ev.message

    def test_init_warning_emits_warning_level(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.init_warning("skill load failed")
        assert reporter.events[0].level == EventLevel.WARNING
        assert reporter.events[0].category == "init"

    def test_init_error_emits_error_level(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.init_error("fatal init error")
        assert reporter.events[0].level == EventLevel.ERROR


# ===================================================================
# 3. Turn lifecycle events
# ===================================================================

class TestTurnLifecycleEvents:

    def test_turn_start(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.turn_start(3, 15)
        ev = reporter.events[0]
        assert ev.category == "turn"
        assert "Turn 3/15" in ev.message
        assert ev.data["turn"] == 3
        assert ev.data["max_turns"] == 15

    def test_turn_cancelled(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.turn_cancelled("Cancelled before LLM call")
        ev = reporter.events[0]
        assert ev.category == "turn"
        assert "Cancelled before LLM call" in ev.message
        assert ev.data["cancelled"] is True

    def test_turn_max_reached(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.turn_max_reached(15)
        ev = reporter.events[0]
        assert ev.category == "turn"
        assert "Max turns (15)" in ev.message


# ===================================================================
# 4. Tool events
# ===================================================================

class TestToolEvents:

    def test_tool_dispatch(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.tool_dispatch("execute_code", ["code", "description"])
        ev = reporter.events[0]
        assert ev.category == "tool"
        assert "execute_code" in ev.message
        assert ev.data["tool_name"] == "execute_code"

    def test_tool_result(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.tool_result("execute_code", "Normalized data")
        ev = reporter.events[0]
        assert ev.category == "tool"
        assert "Normalized data" in ev.message

    def test_tool_finished(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.tool_finished("Analysis complete")
        ev = reporter.events[0]
        assert ev.category == "tool"
        assert "Finished: Analysis complete" in ev.message

    def test_tool_delegated(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.tool_delegated("explore", "Find clustering methods")
        ev = reporter.events[0]
        assert ev.category == "tool"
        assert "explore" in ev.message


# ===================================================================
# 5. Subagent events
# ===================================================================

class TestSubagentEvents:

    def test_subagent_turn(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.subagent_turn("explore", 2, 5)
        ev = reporter.events[0]
        assert ev.category == "subagent"
        assert "[explore]" in ev.message
        assert "Turn 2/5" in ev.message

    def test_subagent_tool(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.subagent_tool("plan", "search_functions", ["query"])
        ev = reporter.events[0]
        assert ev.category == "subagent"
        assert "[plan]" in ev.message
        assert "search_functions" in ev.message

    def test_subagent_finished(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.subagent_finished("execute", "Code ran successfully")
        ev = reporter.events[0]
        assert ev.category == "subagent"
        assert "[execute]" in ev.message


# ===================================================================
# 6. LLM / response events
# ===================================================================

class TestLlmEvents:

    def test_llm_response(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.llm_response("Here is the analysis plan")
        ev = reporter.events[0]
        assert ev.category == "llm"
        assert "Agent response" in ev.message

    def test_llm_response_truncates(self):
        """LLM response message is truncated to 200 chars."""
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        long_text = "x" * 500
        bus.llm_response(long_text)
        # The message should contain at most 200 chars of the text
        ev = reporter.events[0]
        assert len(ev.message) < 300  # 200 content + prefix


# ===================================================================
# 7. Recovery events
# ===================================================================

class TestRecoveryEvents:

    def test_recovery_attempt(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.recovery_attempt("diagnosis", "   🔬 LLM diagnosing execution error...")
        ev = reporter.events[0]
        assert ev.category == "recovery"
        assert ev.data["phase"] == "diagnosis"

    def test_execution_status(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.execution_status("   ✅ Successfully installed numpy")
        ev = reporter.events[0]
        assert ev.category == "execution"


# ===================================================================
# 8. Request lifecycle events
# ===================================================================

class TestRequestLifecycleEvents:

    def test_request_start_with_dataset(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.request_start("cluster analysis", dataset_shape=(1000, 2000))
        ev = reporter.events[0]
        assert ev.category == "lifecycle"
        assert "cluster analysis" in ev.message
        assert "1000" in ev.message
        assert ev.data["dataset_shape"] == (1000, 2000)

    def test_request_start_without_dataset(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.request_start("what is UMAP?")
        ev = reporter.events[0]
        assert "None (knowledge query)" in ev.message

    def test_request_success(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.request_success()
        ev = reporter.events[0]
        assert "SUCCESS" in ev.message
        assert ev.category == "lifecycle"

    def test_request_error(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.request_error("timeout expired")
        ev = reporter.events[0]
        assert "ERROR" in ev.message
        assert "timeout expired" in ev.message


# ===================================================================
# 9. Trace output compatibility
# ===================================================================

class TestTraceCompatibility:

    def test_print_reporter_preserves_output(self, capsys):
        """PrintReporter backed EventBus produces the same stdout as raw print."""
        bus = EventBus(PrintReporter())
        bus.turn_start(3, 15)
        captured = capsys.readouterr()
        assert "Turn 3/15" in captured.out

    def test_silent_reporter_no_stdout(self, capsys):
        """SilentReporter backed EventBus produces no stdout."""
        bus = EventBus(SilentReporter())
        bus.turn_start(1, 10)
        bus.tool_dispatch("execute_code", ["code"])
        bus.request_success()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_conversation_log_saved(self):
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        bus.conversation_log_saved("/tmp/test_log.json")
        ev = reporter.events[0]
        assert ev.category == "trace"
        assert "/tmp/test_log.json" in ev.message


# ===================================================================
# 10. Bootstrap event_bus integration
# ===================================================================

class TestBootstrapEventBusIntegration:

    def test_bootstrap_notebook_disabled_uses_event_bus(self):
        """initialize_notebook_executor emits through event_bus instead of print."""
        from omicverse.utils.ovagent.bootstrap import initialize_notebook_executor
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        use_nb, executor = initialize_notebook_executor(
            use_notebook=False,
            storage_dir=None,
            max_prompts_per_session=5,
            keep_notebooks=True,
            timeout=600,
            strict_kernel_validation=True,
            event_bus=bus,
        )
        assert use_nb is False
        assert executor is None
        assert len(reporter.events) == 1
        assert "in-process execution" in reporter.events[0].message

    def test_bootstrap_filesystem_disabled_uses_event_bus(self):
        """initialize_filesystem_context emits through event_bus instead of print."""
        from omicverse.utils.ovagent.bootstrap import initialize_filesystem_context
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        enabled, ctx = initialize_filesystem_context(
            enabled=False,
            storage_dir=None,
            event_bus=bus,
        )
        assert enabled is False
        assert ctx is None
        assert len(reporter.events) == 1
        assert "Filesystem context disabled" in reporter.events[0].message

    def test_bootstrap_security_uses_event_bus(self):
        """initialize_security emits through event_bus instead of print."""
        from omicverse.utils.ovagent.bootstrap import initialize_security
        from omicverse.utils.agent_config import AgentConfig
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        sec_config, scanner = initialize_security(AgentConfig(), event_bus=bus)
        assert sec_config is not None
        assert scanner is not None
        assert len(reporter.events) == 1
        assert "Security scanner enabled" in reporter.events[0].message

    def test_display_reflection_config_uses_event_bus(self):
        """display_reflection_config emits through event_bus."""
        from omicverse.utils.ovagent.bootstrap import display_reflection_config
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        display_reflection_config(True, 2, True, event_bus=bus)
        messages = [ev.message for ev in reporter.events]
        assert any("Reflection enabled" in m for m in messages)
        assert any("Result review enabled" in m for m in messages)


# ===================================================================
# 11. Auth event_bus integration
# ===================================================================

class TestAuthEventBusIntegration:

    def test_resolve_model_proxy_uses_event_bus(self):
        """resolve_model_and_provider emits through event_bus."""
        from omicverse.utils.ovagent.auth import resolve_model_and_provider
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        result = resolve_model_and_provider(
            model="my-model",
            api_key="test-key",
            endpoint="https://proxy.example.com/v1",
            event_bus=bus,
        )
        assert result.model == "my-model"
        assert len(reporter.events) >= 1
        assert any("Proxy mode" in ev.message for ev in reporter.events)

    def test_display_backend_info_uses_event_bus(self):
        """display_backend_info emits through event_bus."""
        from omicverse.utils.ovagent.auth import display_backend_info
        reporter = CollectingReporter()
        bus = EventBus(reporter)
        display_backend_info(
            model="gpt-5.2",
            endpoint="https://api.openai.com/v1",
            provider="openai",
            api_key=None,
            managed_env={},
            event_bus=bus,
        )
        messages = [ev.message for ev in reporter.events]
        assert any("Model:" in m for m in messages)
        assert any("Provider:" in m for m in messages)


# ===================================================================
# 12. No ad-hoc print in core modules
# ===================================================================

class TestNoPrintInCoreModules:
    """Verify that core runtime modules no longer contain bare print() calls."""

    @pytest.mark.parametrize("module_relpath", [
        "omicverse/utils/ovagent/turn_controller.py",
    ])
    def test_no_bare_print_in_module(self, module_relpath):
        """Core runtime module should not contain bare print() calls."""
        module_path = PROJECT_ROOT / module_relpath
        source = module_path.read_text(encoding="utf-8")
        # Find all lines that start with whitespace + "print(" which are
        # actual code-level print calls (not in strings/comments)
        import ast
        tree = ast.parse(source)
        print_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    print_calls.append(node.lineno)
        assert print_calls == [], (
            f"Bare print() calls found at lines {print_calls} in {module_relpath}. "
            "Use EventBus instead."
        )
