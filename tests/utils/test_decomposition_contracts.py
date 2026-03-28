"""Decomposition contract tests for oversized OVAgent runtime modules.

Locks the current public behavior, import surfaces, and method inventories
of TurnController, ToolRuntime, and AnalysisExecutor so that later
extraction tasks can shrink the files without behavior drift.

Each test section documents the target decomposition seam — the future
module that will own the tested methods — so extraction tasks know exactly
what to move and what must remain on the facade.

These tests are intentionally lightweight: no real LLM calls, no heavy
optional dependencies.  They run under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""

import ast
import importlib
import importlib.machinery
import inspect
import os
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Guard: these tests require the harness env-var
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Decomposition contract tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs to avoid heavy omicverse.__init__
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"
OVAGENT_DIR = PACKAGE_ROOT / "utils" / "ovagent"

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

import omicverse.utils.ovagent as ovagent_pkg
from omicverse.utils.ovagent import (
    AnalysisExecutor,
    FollowUpGate,
    ProactiveCodeTransformer,
    PromptBuilder,
    SubagentController,
    ToolRuntime,
    TurnController,
)
from omicverse.utils.ovagent.turn_controller import ConvergenceMonitor
from omicverse.utils.ovagent.tool_runtime import LEGACY_AGENT_TOOLS
from omicverse.utils.ovagent.codegen_pipeline import CodegenPipeline
from omicverse.utils.ovagent.registry_scanner import RegistryScanner
from omicverse.utils.ovagent.protocol import AgentContext

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Minimal context double (matches AgentContext protocol)
# ---------------------------------------------------------------------------

class _MinimalCtx:
    """Lightweight AgentContext double for decomposition tests."""

    LEGACY_AGENT_TOOLS = LEGACY_AGENT_TOOLS

    def __init__(self):
        self.model = "test-model"
        self.provider = "openai"
        self.endpoint = "https://api.example.com"
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
        self._web_session_id = "test-decomp-session"
        self._managed_api_env = {}
        self._code_only_mode = False
        self._code_only_captured_code = ""
        self._code_only_captured_history = []
        self.use_notebook_execution = False
        self.enable_filesystem_context = False

    def _emit(self, level, message, category=""):
        pass

    def _get_harness_session_id(self):
        return self._web_session_id

    def _get_runtime_session_id(self):
        return self._web_session_id or "default"

    def _refresh_runtime_working_directory(self):
        return "."

    def _detect_repo_root(self, cwd=None):
        return None

    def _resolve_local_path(self, file_path, *, allow_relative=False):
        raise NotImplementedError

    def _ensure_server_tool_mode(self, tool_name):
        pass

    def _request_interaction(self, payload):
        raise NotImplementedError

    def _request_tool_approval(self, tool_name, *, reason, payload):
        raise NotImplementedError

    def _load_skill_guidance(self, slug):
        return ""

    def _build_agentic_system_prompt(self):
        return "You are a test agent."

    async def _run_agentic_loop(self, request, adata, event_callback=None,
                                cancel_event=None, history=None,
                                approval_handler=None, request_content=None):
        return adata


class _DummyExecutor:
    pass


# ===================================================================
# SECTION 1: TurnController decomposition contracts
# ===================================================================


class TestTurnControllerFacadeContract:
    """TurnController stays as the orchestration facade after decomposition.

    Target: turn_controller.py (stays)
    """

    def test_construction_signature(self):
        """TurnController(ctx, prompt_builder, tool_runtime) is the stable API."""
        ctx = _MinimalCtx()
        pb = PromptBuilder(ctx)
        rt = ToolRuntime(ctx, _DummyExecutor())
        tc = TurnController(ctx, pb, rt)
        assert tc is not None

    def test_run_agentic_loop_is_async(self):
        """run_agentic_loop is the sole public entry point and must be async."""
        assert inspect.iscoroutinefunction(TurnController.run_agentic_loop)

    def test_run_agentic_loop_signature(self):
        sig = inspect.signature(TurnController.run_agentic_loop)
        params = list(sig.parameters.keys())
        assert params[0] == "self"
        assert "request" in params
        assert "adata" in params
        assert "event_callback" in params
        assert "cancel_event" in params
        assert "history" in params


class TestFollowUpGateSeam:
    """FollowUpGate is extracted to turn_followup.py during decomposition.

    Target: turn_followup.py
    """

    EXPECTED_CLASS_ATTRS = {
        "NON_COMPLETING_TOOLS",
        "URL_PATTERN",
        "ACTION_REQUEST_PATTERN",
        "PROMISSORY_PATTERN",
        "BLOCKER_PATTERN",
        "RESULT_PATTERN",
    }

    EXPECTED_METHODS = {
        "request_requires_tool_action",
        "response_is_promissory",
        "select_tool_choice",
        "should_continue_after_text",
        "build_no_tool_follow_up",
        "tool_counts_as_meaningful_progress",
    }

    def test_class_attributes_present(self):
        for attr in self.EXPECTED_CLASS_ATTRS:
            assert hasattr(FollowUpGate, attr), (
                f"FollowUpGate missing class attr '{attr}'"
            )

    def test_all_methods_are_classmethods(self):
        for name in self.EXPECTED_METHODS:
            method = getattr(FollowUpGate, name, None)
            assert method is not None, f"FollowUpGate missing method '{name}'"
            assert callable(method)

    def test_method_inventory_complete(self):
        """No undocumented public classmethods exist on FollowUpGate."""
        actual_public = {
            name for name, obj in inspect.getmembers(FollowUpGate)
            if not name.startswith("_") and callable(obj)
        }
        assert actual_public == self.EXPECTED_METHODS, (
            f"FollowUpGate method drift: "
            f"added={actual_public - self.EXPECTED_METHODS}, "
            f"removed={self.EXPECTED_METHODS - actual_public}"
        )

    def test_non_completing_tools_is_frozenset(self):
        assert isinstance(FollowUpGate.NON_COMPLETING_TOOLS, frozenset)
        assert len(FollowUpGate.NON_COMPLETING_TOOLS) > 0

    def test_patterns_are_compiled_regex(self):
        for attr in ["URL_PATTERN", "ACTION_REQUEST_PATTERN",
                      "PROMISSORY_PATTERN", "BLOCKER_PATTERN", "RESULT_PATTERN"]:
            pattern = getattr(FollowUpGate, attr)
            assert isinstance(pattern, re.Pattern), (
                f"FollowUpGate.{attr} should be a compiled regex"
            )

    def test_request_requires_tool_action_contract(self):
        assert FollowUpGate.request_requires_tool_action("", None) is False
        assert FollowUpGate.request_requires_tool_action(
            "analyze data", None
        ) is True
        assert FollowUpGate.request_requires_tool_action(
            "hello", object()
        ) is True

    def test_response_is_promissory_contract(self):
        assert FollowUpGate.response_is_promissory("Let me analyze that") is True
        assert FollowUpGate.response_is_promissory("Here are the results") is False

    def test_select_tool_choice_first_turn(self):
        result = FollowUpGate.select_tool_choice(
            request="analyze data", adata=object(), turn_index=0,
            had_meaningful_tool_call=False, forced_retry=False,
        )
        assert result == "required"

    def test_tool_counts_as_meaningful_progress(self):
        assert FollowUpGate.tool_counts_as_meaningful_progress("execute_code") is True
        assert FollowUpGate.tool_counts_as_meaningful_progress("read") is False


class TestConvergenceMonitorSeam:
    """ConvergenceMonitor is extracted to turn_followup.py during decomposition.

    Target: turn_followup.py
    """

    EXPECTED_CLASS_ATTRS = {
        "READ_ONLY_TOOLS",
        "ARTIFACT_TOOLS",
        "THRESHOLD",
        "ESCALATION_LEVELS",
    }

    EXPECTED_METHODS = {
        "record_turn",
        "should_inject",
        "should_force_tool_choice",
        "build_steering_message",
    }

    def test_class_attributes_present(self):
        for attr in self.EXPECTED_CLASS_ATTRS:
            assert hasattr(ConvergenceMonitor, attr), (
                f"ConvergenceMonitor missing class attr '{attr}'"
            )

    def test_method_inventory_complete(self):
        actual_public = {
            name for name, obj in inspect.getmembers(ConvergenceMonitor)
            if not name.startswith("_") and callable(obj)
        }
        assert actual_public == self.EXPECTED_METHODS, (
            f"ConvergenceMonitor method drift: "
            f"added={actual_public - self.EXPECTED_METHODS}, "
            f"removed={self.EXPECTED_METHODS - actual_public}"
        )

    def test_read_only_tools_is_frozenset(self):
        assert isinstance(ConvergenceMonitor.READ_ONLY_TOOLS, frozenset)
        assert "run_snippet" in ConvergenceMonitor.READ_ONLY_TOOLS

    def test_artifact_tools_is_frozenset(self):
        assert isinstance(ConvergenceMonitor.ARTIFACT_TOOLS, frozenset)
        assert "execute_code" in ConvergenceMonitor.ARTIFACT_TOOLS

    def test_construction_with_prompt(self):
        cm = ConvergenceMonitor("Test prompt")
        assert cm is not None

    def test_steering_escalation_levels(self):
        cm = ConvergenceMonitor(
            "OUTPUT CONTRACT\n* figure.png: analysis figure\n* result.csv: data"
        )
        # Simulate read-only plateau
        for _ in range(ConvergenceMonitor.THRESHOLD):
            cm.record_turn(["run_snippet"])
        assert cm.should_inject() is True
        msg1 = cm.build_steering_message()
        assert "figure.png" in msg1
        assert cm.should_force_tool_choice() is False

    def test_execute_code_resets_plateau(self):
        cm = ConvergenceMonitor(
            "OUTPUT CONTRACT\n* result.csv: data"
        )
        cm.record_turn(["run_snippet"])
        cm.record_turn(["run_snippet"])
        cm.record_turn(["execute_code"])
        assert cm.should_inject() is False


class TestTurnControllerArtifactSeam:
    """Artifact/log persistence methods on TurnController.

    Target: turn_artifacts.py
    These methods will be extracted to a dedicated artifact persistence module.
    The facade will delegate to the extracted module.
    """

    ARTIFACT_METHODS = {
        "_persist_harness_history",
        "_save_conversation_log",
        "_persist_tool_debug_output",
        "_persist_execute_code_source",
        "_persist_execute_code_stdout",
        "_slugify_for_filename",
        "_log_tool_debug_output",
        "_log_execute_code_source",
        "_log_execute_code_stdout",
    }

    def test_artifact_methods_exist_on_turn_controller(self):
        """All artifact/log methods exist on TurnController before extraction."""
        for name in self.ARTIFACT_METHODS:
            assert hasattr(TurnController, name), (
                f"TurnController missing artifact method '{name}'"
            )

    def test_static_methods_identified(self):
        """Static methods can be moved without instance binding."""
        static_methods = {
            "_save_conversation_log",
            "_slugify_for_filename",
            "_log_tool_debug_output",
            "_log_execute_code_source",
            "_log_execute_code_stdout",
        }
        for name in static_methods:
            raw = inspect.getattr_static(TurnController, name)
            assert isinstance(raw, staticmethod), (
                f"TurnController.{name} should be staticmethod"
            )

    def test_instance_methods_identified(self):
        """Instance methods need ctx/filesystem_context wiring."""
        instance_methods = {
            "_persist_harness_history",
            "_persist_tool_debug_output",
            "_persist_execute_code_source",
            "_persist_execute_code_stdout",
        }
        for name in instance_methods:
            raw = inspect.getattr_static(TurnController, name)
            assert not isinstance(raw, (staticmethod, classmethod)), (
                f"TurnController.{name} should be a regular instance method"
            )

    def test_slugify_behavior_contract(self):
        assert TurnController._slugify_for_filename("hello world!") == "hello_world"
        assert TurnController._slugify_for_filename("") == "tool"
        result = TurnController._slugify_for_filename("a" * 100)
        assert len(result) <= 48


# ===================================================================
# SECTION 2: ToolRuntime decomposition contracts
# ===================================================================


class TestToolRuntimeFacadeContract:
    """ToolRuntime stays as the dispatch facade after decomposition.

    Target: tool_runtime.py (stays)
    """

    def test_construction_signature(self):
        """ToolRuntime(ctx, executor) is the stable construction API."""
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        assert rt is not None

    def test_public_method_inventory(self):
        """Only these methods are public on the facade."""
        expected_public = {
            "dispatch_tool",
            "get_visible_agent_tools",
            "get_loaded_tool_names",
            "tool_blocked_in_plan_mode",
            "set_subagent_controller",
            "registry",  # property
        }
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        actual_public = {
            name for name in dir(rt)
            if not name.startswith("_")
            and name != "registry"
            and callable(getattr(rt, name))
        }
        # Add property
        if hasattr(type(rt), "registry") and isinstance(
            inspect.getattr_static(type(rt), "registry"), property
        ):
            actual_public.add("registry")
        assert actual_public == expected_public, (
            f"ToolRuntime public API drift: "
            f"added={actual_public - expected_public}, "
            f"removed={expected_public - actual_public}"
        )

    def test_dispatch_tool_is_async(self):
        assert inspect.iscoroutinefunction(ToolRuntime.dispatch_tool)

    def test_registry_property(self):
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        assert rt.registry is not None

    def test_set_subagent_controller(self):
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        mock_ctrl = MagicMock()
        rt.set_subagent_controller(mock_ctrl)
        assert rt._subagent_controller is mock_ctrl


class TestToolRuntimeHandlerRegistry:
    """All tool handlers are registered and bound correctly.

    This test locks the registry-handler binding so extraction tasks
    can verify that moved handlers still get registered.
    """

    # Handler keys as used in _bind_handlers (register_handler first arg)
    EXPECTED_HANDLER_KEYS = {
        "tool_search", "bash", "read", "edit", "write", "glob", "grep",
        "notebook_edit", "inspect_data", "execute_code", "run_snippet",
        "search_functions", "agent", "ask_user_question",
        "task_create", "task_get", "task_list", "task_output",
        "task_stop", "task_update",
        "enter_plan_mode", "exit_plan_mode", "enter_worktree",
        "skill", "web_fetch", "web_search", "web_download",
        "list_mcp_resources", "read_mcp_resource", "finish",
    }

    def test_all_handler_keys_bound(self):
        """Every expected handler key has a callable bound in the registry."""
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        # Access the internal _handlers dict to check handler keys directly
        bound_keys = set(rt.registry._handlers.keys())
        missing = self.EXPECTED_HANDLER_KEYS - bound_keys
        assert not missing, f"Missing handler key bindings: {missing}"

    def test_no_missing_handler_bindings(self):
        """Every tool that declares a handler_key has a handler bound."""
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        missing = rt.registry.validate_handlers()
        assert not missing, f"Tools with missing handlers: {missing}"


class TestToolRuntimeExecSeam:
    """Execution/agent/shell tools — extracted to tool_runtime_exec.py."""

    EXEC_HANDLER_FUNCTIONS = {
        "handle_execute_code",
        "handle_run_snippet",
        "handle_bash",
        "handle_finish",
        "handle_inspect_data",
        "handle_search_functions",
        "handle_tool_search",
        "handle_agent",
        "background_bash_worker",
    }

    # Thin dispatch wrappers that remain on the facade for input
    # validation and argument unpacking.
    DISPATCH_WRAPPERS = {
        "_dispatch_execute_code",
        "_dispatch_run_snippet",
        "_dispatch_agent",
    }

    # No full tool implementations remain on the facade.
    EXEC_METHODS: set = set()

    def test_exec_handlers_exist_in_extracted_module(self):
        from omicverse.utils.ovagent import tool_runtime_exec
        for name in self.EXEC_HANDLER_FUNCTIONS:
            assert hasattr(tool_runtime_exec, name), (
                f"tool_runtime_exec missing handler '{name}'"
            )
            assert callable(getattr(tool_runtime_exec, name))

    def test_dispatch_wrappers_on_facade(self):
        for name in self.DISPATCH_WRAPPERS:
            assert hasattr(ToolRuntime, name), (
                f"ToolRuntime missing dispatch wrapper '{name}'"
            )

    def test_dispatch_agent_is_async(self):
        assert inspect.iscoroutinefunction(ToolRuntime._dispatch_agent)

    def test_handle_agent_is_async(self):
        from omicverse.utils.ovagent import tool_runtime_exec
        assert inspect.iscoroutinefunction(tool_runtime_exec.handle_agent)

    def test_exec_methods_removed_from_facade(self):
        removed = {
            "_tool_execute_code", "_tool_run_snippet", "_tool_bash",
            "_tool_finish", "_background_bash_worker",
            "_tool_inspect_data", "_tool_search_functions",
            "_tool_tool_search",
        }
        for name in removed:
            assert not hasattr(ToolRuntime, name), (
                f"ToolRuntime still has '{name}' — should be extracted"
            )


class TestToolRuntimeIOSeam:
    """Filesystem/notebook tools — extracted to tool_runtime_io.py."""

    IO_HANDLER_FUNCTIONS = {
        "handle_read",
        "handle_edit",
        "handle_write",
        "handle_glob",
        "handle_grep",
        "handle_notebook_edit",
    }

    # These methods no longer live on ToolRuntime; they are module-level
    # functions in tool_runtime_io.
    IO_METHODS: set = set()

    def test_io_handlers_exist_in_extracted_module(self):
        from omicverse.utils.ovagent import tool_runtime_io
        for name in self.IO_HANDLER_FUNCTIONS:
            assert hasattr(tool_runtime_io, name), (
                f"tool_runtime_io missing handler '{name}'"
            )
            assert callable(getattr(tool_runtime_io, name))

    def test_io_methods_removed_from_facade(self):
        removed = {
            "_tool_read", "_tool_edit", "_tool_write",
            "_tool_glob", "_tool_grep", "_tool_notebook_edit",
        }
        for name in removed:
            assert not hasattr(ToolRuntime, name), (
                f"ToolRuntime still has '{name}' — should be extracted"
            )


class TestToolRuntimeWebSeam:
    """Web/network tools — extracted to tool_runtime_web.py."""

    WEB_HANDLER_FUNCTIONS = {
        "handle_web_fetch",
        "handle_web_search",
        "handle_web_download",
    }

    # These methods no longer live on ToolRuntime.
    WEB_METHODS: set = set()

    def test_web_handlers_exist_in_extracted_module(self):
        from omicverse.utils.ovagent import tool_runtime_web
        for name in self.WEB_HANDLER_FUNCTIONS:
            assert hasattr(tool_runtime_web, name), (
                f"tool_runtime_web missing handler '{name}'"
            )
            assert callable(getattr(tool_runtime_web, name))

    def test_web_methods_removed_from_facade(self):
        removed = {"_tool_web_fetch", "_tool_web_search", "_tool_web_download"}
        for name in removed:
            assert not hasattr(ToolRuntime, name), (
                f"ToolRuntime still has '{name}' — should be extracted"
            )


class TestToolRuntimeWorkspaceSeam:
    """Task/plan/worktree/skill/MCP tools — extracted to tool_runtime_workspace.py.

    Methods that remain on the facade (core/execution family) are tracked
    separately in the facade-methods set below.
    """

    WORKSPACE_HANDLER_FUNCTIONS = {
        "handle_create_task",
        "handle_get_task",
        "handle_list_tasks",
        "handle_task_output",
        "handle_task_stop",
        "handle_task_update",
        "handle_enter_plan_mode",
        "handle_exit_plan_mode",
        "handle_enter_worktree",
        "handle_skill",
        "handle_search_skills",
        "handle_list_mcp_resources",
        "handle_read_mcp_resource",
        "handle_ask_user_question",
    }

    # The ask-user dispatch wrapper remains on the facade; execution-family
    # methods that were formerly listed here have been extracted to
    # tool_runtime_exec.py.
    FACADE_METHODS = {
        "_dispatch_ask_user_question",
    }

    # Empty — everything formerly here is now in the extracted module or
    # reclassified into FACADE_METHODS.
    WORKSPACE_METHODS: set = set()

    def test_workspace_handlers_exist_in_extracted_module(self):
        from omicverse.utils.ovagent import tool_runtime_workspace
        for name in self.WORKSPACE_HANDLER_FUNCTIONS:
            assert hasattr(tool_runtime_workspace, name), (
                f"tool_runtime_workspace missing handler '{name}'"
            )
            assert callable(getattr(tool_runtime_workspace, name))

    def test_facade_methods_still_on_runtime(self):
        for name in self.FACADE_METHODS:
            assert hasattr(ToolRuntime, name), (
                f"ToolRuntime should still have facade method '{name}'"
            )

    def test_workspace_methods_removed_from_facade(self):
        removed = {
            "_tool_create_task", "_tool_get_task", "_tool_list_tasks",
            "_tool_task_output", "_tool_task_stop", "_tool_task_update",
            "_tool_enter_plan_mode", "_tool_exit_plan_mode",
            "_tool_enter_worktree", "_tool_skill", "_tool_search_skills",
            "_tool_list_mcp_resources", "_tool_read_mcp_resource",
            "_tool_ask_user_question",
        }
        for name in removed:
            assert not hasattr(ToolRuntime, name), (
                f"ToolRuntime still has '{name}' — should be extracted"
            )


class TestToolRuntimeMethodPartition:
    """All _tool_* and _dispatch_* methods on ToolRuntime are accounted for.

    After full extraction (IO/web/workspace/exec), only thin dispatch
    wrappers remain on the facade.  This test ensures no method is forgotten.
    """

    # Methods that remain on the ToolRuntime facade after all extraction.
    ALL_FACADE_METHODS = (
        TestToolRuntimeExecSeam.DISPATCH_WRAPPERS
        | TestToolRuntimeWorkspaceSeam.FACADE_METHODS
    )

    def test_no_undocumented_tool_methods(self):
        """Every _tool_* and _dispatch_* method on the facade is accounted for."""
        actual_tool_methods = {
            name for name in dir(ToolRuntime)
            if (name.startswith("_tool_") or name.startswith("_dispatch_"))
            and callable(getattr(ToolRuntime, name))
        }
        undocumented = actual_tool_methods - self.ALL_FACADE_METHODS
        assert not undocumented, (
            f"Undocumented ToolRuntime methods (assign to a seam): {undocumented}"
        )

    def test_no_overlap_between_seams(self):
        """No method appears in more than one extraction target."""
        seams = [
            TestToolRuntimeExecSeam.EXEC_METHODS,
            TestToolRuntimeIOSeam.IO_METHODS,
            TestToolRuntimeWebSeam.WEB_METHODS,
            TestToolRuntimeWorkspaceSeam.WORKSPACE_METHODS,
        ]
        for i, a in enumerate(seams):
            for j, b in enumerate(seams):
                if i >= j:
                    continue
                overlap = a & b
                assert not overlap, (
                    f"Seam overlap between seam {i} and seam {j}: {overlap}"
                )


class TestLegacyAgentToolsContract:
    """LEGACY_AGENT_TOOLS constant on tool_runtime module.

    This stays in tool_runtime.py as a module-level constant.
    """

    EXPECTED_TOOL_NAMES = {
        "inspect_data", "execute_code", "run_snippet",
        "search_functions", "search_skills", "delegate",
        "web_fetch", "web_search", "web_download", "finish",
    }

    def test_legacy_tools_is_list(self):
        assert isinstance(LEGACY_AGENT_TOOLS, list)

    def test_expected_names(self):
        actual = {t["name"] for t in LEGACY_AGENT_TOOLS}
        assert actual == self.EXPECTED_TOOL_NAMES

    def test_each_tool_has_schema(self):
        for tool in LEGACY_AGENT_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert tool["parameters"]["type"] == "object"


# ===================================================================
# SECTION 3: AnalysisExecutor decomposition contracts
# ===================================================================


class TestAnalysisExecutorFacadeContract:
    """AnalysisExecutor stays as the execution facade after decomposition.

    Target: analysis_executor.py (stays)
    """

    def test_construction_signature(self):
        """AnalysisExecutor(ctx) is the stable construction API."""
        ctx = _MinimalCtx()
        ae = AnalysisExecutor(ctx)
        assert ae is not None
        assert ae._ctx is ctx

    def test_public_method_inventory(self):
        """Lock all public methods on AnalysisExecutor."""
        expected_public = {
            "check_code_prerequisites",
            "apply_execution_error_fix",
            "extract_package_name",
            "auto_install_package",
            "diagnose_error_with_llm",
            "validate_outputs",
            "generate_completion_code",
            "request_approval",
            "execute_snippet_readonly",
            "execute_generated_code",
            "normalize_doublet_obs",
            "process_context_directives",
            "build_sandbox_globals",
            "set_codegen_pipeline",
        }
        actual_public = {
            name for name in dir(AnalysisExecutor)
            if not name.startswith("_")
            and callable(getattr(AnalysisExecutor, name))
        }
        assert actual_public == expected_public, (
            f"AnalysisExecutor public API drift: "
            f"added={actual_public - expected_public}, "
            f"removed={expected_public - actual_public}"
        )


class TestProactiveCodeTransformerSeam:
    """ProactiveCodeTransformer is extracted to analysis_transformer.py.

    Target: analysis_transformer.py
    """

    EXPECTED_CLASS_ATTRS = {
        "INPLACE_FUNCTIONS",
        "KWARG_RENAMES",
    }

    EXPECTED_METHODS = {
        "transform",
    }

    def test_class_attributes_present(self):
        for attr in self.EXPECTED_CLASS_ATTRS:
            assert hasattr(ProactiveCodeTransformer, attr), (
                f"ProactiveCodeTransformer missing '{attr}'"
            )

    def test_inplace_functions_is_set(self):
        assert isinstance(ProactiveCodeTransformer.INPLACE_FUNCTIONS, set)
        assert "pca" in ProactiveCodeTransformer.INPLACE_FUNCTIONS
        assert "scale" in ProactiveCodeTransformer.INPLACE_FUNCTIONS
        assert "neighbors" in ProactiveCodeTransformer.INPLACE_FUNCTIONS

    def test_transform_public_method(self):
        t = ProactiveCodeTransformer()
        assert callable(t.transform)

    def test_transform_preserves_valid_code(self):
        t = ProactiveCodeTransformer()
        code = "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2]})"
        result = t.transform(code)
        assert "import pandas as pd" in result

    def test_transform_fixes_show_true(self):
        t = ProactiveCodeTransformer()
        result = t.transform("plt.show(show=True)")
        assert "show=False" in result

    def test_transform_fixes_inplace_assignment(self):
        t = ProactiveCodeTransformer()
        code = "adata = ov.pp.pca(adata)"
        result = t.transform(code)
        assert "adata = ov.pp.pca" not in result
        assert "ov.pp.pca(adata)" in result

    def test_transform_adds_matplotlib_preamble(self):
        t = ProactiveCodeTransformer()
        result = t.transform("print('hello')")
        assert "_mpl.use('Agg')" in result

    def test_standalone_no_agent_state(self):
        """ProactiveCodeTransformer can be constructed with zero args — stateless."""
        t = ProactiveCodeTransformer()
        assert t is not None
        # No instance attributes set by __init__ beyond inherited defaults
        assert not hasattr(t, "_ctx")

    def test_internal_fix_methods_inventory(self):
        """Lock the private fix methods that must move with the transformer."""
        expected_private = {
            "_prepend_matplotlib_noninteractive",
            "_fix_show_true",
            "_fix_inplace_assignments_regex",
            "_fix_fstring_print_regex",
            "_convert_fstring_line",
            "_fix_cat_accessor_regex",
            "_fix_kwarg_renames",
            "_MATPLOTLIB_PREAMBLE",
        }
        actual_private = {
            name for name in dir(ProactiveCodeTransformer)
            if name.startswith("_") and not name.startswith("__")
        }
        missing = expected_private - actual_private
        assert not missing, (
            f"ProactiveCodeTransformer missing internal methods: {missing}"
        )


class TestAnalysisDiagnosticsSeam:
    """Prerequisite checks, error recovery, LLM diagnosis methods.

    Target: analysis_diagnostics.py
    """

    DIAGNOSTIC_METHODS = {
        "check_code_prerequisites",
        "apply_execution_error_fix",
        "extract_package_name",
        "auto_install_package",
        "diagnose_error_with_llm",
        "generate_completion_code",
        "validate_outputs",
    }

    def test_diagnostic_methods_exist(self):
        for name in self.DIAGNOSTIC_METHODS:
            assert hasattr(AnalysisExecutor, name), (
                f"AnalysisExecutor missing diagnostic method '{name}'"
            )

    def test_extract_package_name_is_static(self):
        raw = inspect.getattr_static(AnalysisExecutor, "extract_package_name")
        assert isinstance(raw, staticmethod)

    def test_diagnose_error_with_llm_is_async(self):
        assert inspect.iscoroutinefunction(AnalysisExecutor.diagnose_error_with_llm)

    def test_generate_completion_code_is_async(self):
        assert inspect.iscoroutinefunction(AnalysisExecutor.generate_completion_code)


class TestAnalysisSandboxSeam:
    """Sandbox setup and execution methods.

    Target: analysis_sandbox.py
    """

    SANDBOX_METHODS = {
        "build_sandbox_globals",
        "execute_generated_code",
        "execute_snippet_readonly",
        "request_approval",
        "normalize_doublet_obs",
    }

    def test_sandbox_methods_exist(self):
        for name in self.SANDBOX_METHODS:
            assert hasattr(AnalysisExecutor, name), (
                f"AnalysisExecutor missing sandbox method '{name}'"
            )

    def test_normalize_doublet_obs_is_static(self):
        raw = inspect.getattr_static(AnalysisExecutor, "normalize_doublet_obs")
        assert isinstance(raw, staticmethod)


class TestAnalysisContextDirectiveSeam:
    """Context-directive handling methods on AnalysisExecutor.

    Target: analysis_sandbox.py (moves with sandbox execution)
    """

    DIRECTIVE_METHODS = {
        "process_context_directives",
        "_handle_context_write",
        "_handle_context_update",
        "_parse_plan_step",
    }

    def test_directive_methods_exist(self):
        for name in self.DIRECTIVE_METHODS:
            assert hasattr(AnalysisExecutor, name), (
                f"AnalysisExecutor missing directive method '{name}'"
            )

    def test_parse_plan_step_is_static(self):
        raw = inspect.getattr_static(AnalysisExecutor, "_parse_plan_step")
        assert isinstance(raw, staticmethod)


class TestAnalysisExecutorMethodPartition:
    """All public methods on AnalysisExecutor are accounted for in a seam."""

    ALL_DOCUMENTED = (
        TestAnalysisDiagnosticsSeam.DIAGNOSTIC_METHODS
        | TestAnalysisSandboxSeam.SANDBOX_METHODS
    )

    def test_no_undocumented_public_methods(self):
        actual_public = {
            name for name in dir(AnalysisExecutor)
            if not name.startswith("_")
            and callable(getattr(AnalysisExecutor, name))
        }
        # process_context_directives is covered by the directive seam
        # set_codegen_pipeline is a late-bind wiring method (task-049)
        documented = self.ALL_DOCUMENTED | {"process_context_directives", "set_codegen_pipeline"}
        undocumented = actual_public - documented
        assert not undocumented, (
            f"Undocumented AnalysisExecutor public methods: {undocumented}"
        )


class TestPackageAliasesContract:
    """_PACKAGE_ALIASES module-level constant in analysis_executor.py.

    Stays with the diagnostics seam (analysis_diagnostics.py).
    """

    def test_package_aliases_accessible(self):
        from omicverse.utils.ovagent.analysis_executor import _PACKAGE_ALIASES
        assert isinstance(_PACKAGE_ALIASES, dict)
        assert len(_PACKAGE_ALIASES) > 0

    def test_known_aliases_present(self):
        from omicverse.utils.ovagent.analysis_executor import _PACKAGE_ALIASES
        assert _PACKAGE_ALIASES.get("cv2") == "opencv-python"
        assert _PACKAGE_ALIASES.get("sklearn") == "scikit-learn"
        assert _PACKAGE_ALIASES.get("PIL") == "Pillow"


# ===================================================================
# SECTION 4: Cross-module import surface audit
# ===================================================================


class TestOvagentInitExportsAllFacades:
    """ovagent/__init__.py exports all three facade classes plus their helpers."""

    def test_turn_controller_exports(self):
        assert "TurnController" in ovagent_pkg.__all__
        assert "FollowUpGate" in ovagent_pkg.__all__

    def test_tool_runtime_exports(self):
        assert "ToolRuntime" in ovagent_pkg.__all__

    def test_analysis_executor_exports(self):
        assert "AnalysisExecutor" in ovagent_pkg.__all__
        assert "ProactiveCodeTransformer" in ovagent_pkg.__all__


class TestModuleLevelImportSurface:
    """Lock the import statements each module uses from other ovagent modules.

    After decomposition, these imports must still resolve — either from the
    facade or from the new extracted modules.
    """

    def _extract_ovagent_imports(self, module_path: Path) -> set:
        """Return set of imported names from ovagent sibling or project modules."""
        source = module_path.read_text()
        tree = ast.parse(source, filename=str(module_path))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Relative imports (from .foo import X) have node.level > 0
                if node.level and node.level > 0:
                    for alias in node.names:
                        imported.add(alias.name)
                elif node.module and "ovagent" in node.module:
                    for alias in node.names:
                        imported.add(alias.name)
        return imported

    def test_turn_controller_internal_imports(self):
        imports = self._extract_ovagent_imports(
            OVAGENT_DIR / "turn_controller.py"
        )
        expected_subset = {
            "BudgetSliceType", "ContextBudgetManager",
            "RuntimeEventEmitter", "OutputTier",
            "ToolScheduler", "execute_batch", "ScheduledCall",
        }
        missing = expected_subset - imports
        assert not missing, (
            f"turn_controller.py missing expected imports: {missing}"
        )

    def test_tool_runtime_internal_imports(self):
        imports = self._extract_ovagent_imports(
            OVAGENT_DIR / "tool_runtime.py"
        )
        # tool_runtime lazily imports from analysis_executor and repair_loop
        # but statically uses tool_catalog
        assert len(imports) > 0

    def test_analysis_executor_has_no_ovagent_runtime_imports(self):
        """AnalysisExecutor should not import from turn_controller or tool_runtime."""
        imports = self._extract_ovagent_imports(
            OVAGENT_DIR / "analysis_executor.py"
        )
        forbidden = {"TurnController", "ToolRuntime", "ToolScheduler"}
        violation = imports & forbidden
        assert not violation, (
            f"analysis_executor.py should not import {violation} — "
            "it sits below the runtime facade layer"
        )


# ===================================================================
# SECTION 5: No-behavior-change guards
# ===================================================================


class TestNoBehaviorChange:
    """Verify this task introduces no runtime behavior changes."""

    def test_no_new_source_modules(self):
        """No new .py files added to ovagent/ by this task."""
        known_modules = {
            "__init__.py", "analysis_executor.py", "auth.py",
            "analysis_transformer.py", "analysis_diagnostics.py",
            "analysis_sandbox.py",
            "bootstrap.py", "codegen_pipeline.py", "codegen_tool_facade.py",
            "context_budget.py",
            "event_stream.py", "permission_policy.py", "prompt_builder.py",
            "prompt_templates.py", "protocol.py", "registry_scanner.py",
            "repair_loop.py", "run_store.py", "runtime.py",
            "session_context.py", "session_facade.py", "subagent_controller.py",
            "tool_registry.py", "tool_runtime.py", "tool_runtime_exec.py",
            "tool_runtime_io.py", "tool_runtime_web.py",
            "tool_runtime_workspace.py", "tool_scheduler.py",
            "turn_controller.py", "turn_followup.py", "turn_artifacts.py",
            "workflow.py",
        }
        actual = {
            f.name for f in OVAGENT_DIR.iterdir()
            if f.suffix == ".py" and not f.name.startswith("__pycache__")
        }
        new_modules = actual - known_modules
        assert not new_modules, (
            f"Unexpected new modules in ovagent/: {new_modules}"
        )

    def test_no_new_dependencies(self):
        """pyproject.toml core dependencies unchanged."""
        text = (PROJECT_ROOT / "pyproject.toml").read_text()
        in_deps, deps = False, []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("dependencies"):
                in_deps = True
                continue
            if in_deps and s == "]":
                break
            if in_deps:
                c = s.strip("',\" ")
                if c and not c.startswith("#"):
                    deps.append(c)
        dep_names = {
            re.match(r"^([A-Za-z0-9_-]+)", d).group(1).lower()
            for d in deps if re.match(r"^([A-Za-z0-9_-]+)", d)
        }
        known = {
            "numpy", "scanpy", "pandas", "matplotlib", "scikit-learn", "scipy",
            "networkx", "multiprocess", "seaborn", "datetime", "statsmodels",
            "ipywidgets", "pygam", "igraph", "tqdm", "adjusttext", "scikit-misc",
            "scikit-image", "plotly", "numba", "requests", "transformers",
            "marsilea", "openai", "omicverse-skills", "omicverse-notebook",
            "zarr", "anndata", "setuptools", "wheel", "cython",
        }
        new = dep_names - known
        assert not new, f"New core dependencies detected: {new}"


# ===================================================================
# SECTION 6: AgentContext protocol contraction contracts (task-048)
# ===================================================================


class TestProtocolContraction:
    """Lock the contracted AgentContext protocol surface.

    After task-048, ovagent consumers no longer depend on facade-only
    codegen helper methods on AgentContext.  These tests verify:

    1. The protocol does not expose the retired facade helpers.
    2. Consumers reach subsystems directly (via _codegen_pipeline or
       RegistryScanner) rather than through wrapper methods on ctx.
    3. A minimal context double that lacks the retired methods still
       satisfies the protocol for ovagent consumer construction.
    """

    def test_protocol_does_not_require_extract_python_code(self):
        """AgentContext protocol must not declare _extract_python_code."""
        assert "_extract_python_code" not in AgentContext.__protocol_attrs__, (
            "_extract_python_code should not be in the AgentContext protocol"
        )

    def test_protocol_does_not_require_normalize_registry_entry(self):
        """AgentContext protocol must not declare _normalize_registry_entry_for_codegen."""
        assert "_normalize_registry_entry_for_codegen" not in AgentContext.__protocol_attrs__, (
            "_normalize_registry_entry_for_codegen should not be in the AgentContext protocol"
        )

    def test_protocol_does_not_require_extract_python_code_strict(self):
        """AgentContext protocol must not declare _extract_python_code_strict."""
        assert "_extract_python_code_strict" not in AgentContext.__protocol_attrs__, (
            "_extract_python_code_strict should not be in the AgentContext protocol"
        )

    def test_codegen_pipeline_uses_registry_scanner_directly(self):
        """CodegenPipeline.rewrite_scanpy_calls_with_registry uses RegistryScanner
        directly, not ctx._normalize_registry_entry_for_codegen."""
        source = (OVAGENT_DIR / "codegen_pipeline.py").read_text()
        assert "ctx._normalize_registry_entry_for_codegen" not in source, (
            "codegen_pipeline.py should use RegistryScanner.normalize_entry, "
            "not ctx._normalize_registry_entry_for_codegen"
        )
        assert "RegistryScanner.normalize_entry" in source

    def test_analysis_diagnostics_does_not_call_ctx_extract(self):
        """analysis_diagnostics.py must not call ctx._extract_python_code."""
        source = (OVAGENT_DIR / "analysis_diagnostics.py").read_text()
        assert "ctx._extract_python_code" not in source, (
            "analysis_diagnostics.py should use extract_code_fn parameter, "
            "not ctx._extract_python_code"
        )

    def test_tool_runtime_exec_does_not_call_ctx_extract(self):
        """tool_runtime_exec.py must not call ctx._extract_python_code."""
        source = (OVAGENT_DIR / "tool_runtime_exec.py").read_text()
        assert 'ctx, "_extract_python_code"' not in source, (
            "tool_runtime_exec.py should resolve extract fn from "
            "_codegen_pipeline, not ctx._extract_python_code"
        )

    def test_repair_loop_does_not_call_ctx_extract(self):
        """repair_loop.py must not call ctx._extract_python_code."""
        source = (OVAGENT_DIR / "repair_loop.py").read_text()
        assert 'ctx, "_extract_python_code"' not in source, (
            "repair_loop.py should resolve extract fn from "
            "_codegen_pipeline, not ctx._extract_python_code"
        )

    def test_retired_methods_not_needed_on_minimal_context(self):
        """A minimal context without retired facade methods can construct
        CodegenPipeline and RegistryScanner subsystems."""
        ctx = _MinimalCtx()
        # Verify the retired methods are absent
        assert not hasattr(ctx, "_extract_python_code")
        assert not hasattr(ctx, "_normalize_registry_entry_for_codegen")
        assert not hasattr(ctx, "_extract_python_code_strict")

        # CodegenPipeline should still construct
        pipeline = CodegenPipeline(ctx)
        assert pipeline is not None

        # extract_python_code should work via subsystem directly
        code = pipeline.extract_python_code(
            "```python\nimport omicverse as ov\nov.pp.qc(adata)\n```"
        )
        assert "ov.pp.qc" in code

    def test_registry_scanner_normalize_entry_is_static(self):
        """RegistryScanner.normalize_entry is a static method callable
        without any context object."""
        entry = {"full_name": "omicverse.pp.qc", "name": "qc"}
        normalized = RegistryScanner.normalize_entry(entry)
        assert "full_name" in normalized
        assert normalized["full_name"].startswith("ov.")
