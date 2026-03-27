"""Regression tests for the turn_controller.py decomposition.

Verifies that the extraction of FollowUpGate and ConvergenceMonitor into
``turn_followup.py``, and artifact helpers into ``turn_artifacts.py``,
preserves backward-compatible imports, class identity, and behavioral
contracts of every extracted component.

These tests are intentionally lightweight: no real LLM calls, no heavy
optional dependencies.  They run under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import inspect
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
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
    reason="Decomposition regression tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
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

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

# Canonical import paths (new extracted modules)
from omicverse.utils.ovagent.turn_followup import FollowUpGate as FollowUpGate_canonical
from omicverse.utils.ovagent.turn_followup import ConvergenceMonitor as ConvergenceMonitor_canonical

# Backward-compatible import paths (via facade)
from omicverse.utils.ovagent.turn_controller import FollowUpGate as FollowUpGate_compat
from omicverse.utils.ovagent.turn_controller import ConvergenceMonitor as ConvergenceMonitor_compat
from omicverse.utils.ovagent.turn_controller import TurnController

# Artifact module-level functions
from omicverse.utils.ovagent.turn_artifacts import (
    persist_harness_history,
    save_conversation_log,
    slugify_for_filename,
    persist_tool_debug_output,
    persist_execute_code_source,
    persist_execute_code_stdout,
    log_tool_debug_output,
    log_execute_code_source,
    log_execute_code_stdout,
)

# Package-level exports
import omicverse.utils.ovagent as ovagent_pkg

# ---------------------------------------------------------------------------
# Restore saved module state
# ---------------------------------------------------------------------------

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ===================================================================
# 1. Import compatibility
# ===================================================================


class TestImportCompatibility:
    """Verify backward-compatible imports after extraction."""

    def test_followup_gate_identity(self):
        """FollowUpGate from turn_controller is the same class as from turn_followup."""
        assert FollowUpGate_canonical is FollowUpGate_compat

    def test_convergence_monitor_identity(self):
        """ConvergenceMonitor from turn_controller is the same class as from turn_followup."""
        assert ConvergenceMonitor_canonical is ConvergenceMonitor_compat

    def test_package_exports_followup_gate(self):
        """FollowUpGate is exported from the ovagent package."""
        assert hasattr(ovagent_pkg, "FollowUpGate")
        assert ovagent_pkg.FollowUpGate is FollowUpGate_canonical

    def test_package_exports_convergence_monitor(self):
        """ConvergenceMonitor is exported from the ovagent package."""
        assert hasattr(ovagent_pkg, "ConvergenceMonitor")
        assert ovagent_pkg.ConvergenceMonitor is ConvergenceMonitor_canonical

    def test_package_exports_turn_controller(self):
        """TurnController is still exported from the ovagent package."""
        assert hasattr(ovagent_pkg, "TurnController")
        assert "TurnController" in ovagent_pkg.__all__
        assert "FollowUpGate" in ovagent_pkg.__all__
        assert "ConvergenceMonitor" in ovagent_pkg.__all__

    def test_turn_followup_module_exists(self):
        """turn_followup.py exists as a standalone module."""
        assert (OVAGENT_DIR / "turn_followup.py").exists()

    def test_turn_artifacts_module_exists(self):
        """turn_artifacts.py exists as a standalone module."""
        assert (OVAGENT_DIR / "turn_artifacts.py").exists()


# ===================================================================
# 2. FollowUpGate behavioral regression
# ===================================================================


class TestFollowUpGateRegression:
    """Behavioral regression: FollowUpGate must behave identically after extraction."""

    def test_request_requires_tool_action_empty(self):
        assert FollowUpGate_canonical.request_requires_tool_action("", None) is False

    def test_request_requires_tool_action_url(self):
        assert FollowUpGate_canonical.request_requires_tool_action("Check https://example.com", None) is True

    def test_request_requires_tool_action_adata(self):
        assert FollowUpGate_canonical.request_requires_tool_action("analyze", MagicMock()) is True

    def test_request_requires_tool_action_chinese(self):
        assert FollowUpGate_canonical.request_requires_tool_action("把分析后的图发送出来", None) is True

    def test_request_requires_tool_action_action_word(self):
        assert FollowUpGate_canonical.request_requires_tool_action("Please plot the UMAP", None) is True

    def test_response_is_promissory_true(self):
        assert FollowUpGate_canonical.response_is_promissory("Let me analyze that") is True

    def test_response_is_promissory_false(self):
        assert FollowUpGate_canonical.response_is_promissory("Here are the results") is False

    def test_response_is_promissory_empty(self):
        assert FollowUpGate_canonical.response_is_promissory("") is False

    def test_select_tool_choice_forced_retry(self):
        result = FollowUpGate_canonical.select_tool_choice(
            request="analyze data",
            adata=None,
            turn_index=0,
            had_meaningful_tool_call=False,
            forced_retry=True,
        )
        assert result == "required"

    def test_select_tool_choice_first_turn_action(self):
        result = FollowUpGate_canonical.select_tool_choice(
            request="plot UMAP",
            adata=None,
            turn_index=0,
            had_meaningful_tool_call=False,
            forced_retry=False,
        )
        assert result == "required"

    def test_select_tool_choice_auto_after_meaningful(self):
        result = FollowUpGate_canonical.select_tool_choice(
            request="plot UMAP",
            adata=None,
            turn_index=0,
            had_meaningful_tool_call=True,
            forced_retry=False,
        )
        assert result == "auto"

    def test_should_continue_after_text_promissory(self):
        assert FollowUpGate_canonical.should_continue_after_text(
            request="plot the UMAP",
            response_content="Let me analyze that for you.",
            adata=None,
            had_meaningful_tool_call=False,
        ) is True

    def test_should_continue_after_text_blocker(self):
        assert FollowUpGate_canonical.should_continue_after_text(
            request="plot the UMAP",
            response_content="I cannot do that, permission denied.",
            adata=None,
            had_meaningful_tool_call=False,
        ) is False

    def test_should_continue_after_text_already_has_tool(self):
        assert FollowUpGate_canonical.should_continue_after_text(
            request="plot the UMAP",
            response_content="Let me do that.",
            adata=None,
            had_meaningful_tool_call=True,
        ) is False

    def test_tool_counts_as_meaningful_progress_execute_code(self):
        assert FollowUpGate_canonical.tool_counts_as_meaningful_progress("execute_code") is True

    def test_tool_counts_as_meaningful_progress_read(self):
        assert FollowUpGate_canonical.tool_counts_as_meaningful_progress("read") is False

    def test_tool_counts_as_meaningful_progress_inspect_data(self):
        assert FollowUpGate_canonical.tool_counts_as_meaningful_progress("InspectData") is False

    def test_build_no_tool_follow_up_basic(self):
        msg = FollowUpGate_canonical.build_no_tool_follow_up("analyze data")
        assert "tool" in msg.lower() or "call" in msg.lower()

    def test_build_no_tool_follow_up_url(self):
        msg = FollowUpGate_canonical.build_no_tool_follow_up("fetch https://example.com")
        assert "WebFetch" in msg or "web_fetch" in msg

    def test_build_no_tool_follow_up_last_retry(self):
        msg = FollowUpGate_canonical.build_no_tool_follow_up("analyze", retry_count=1, max_retries=2)
        assert "MUST" in msg

    def test_non_completing_tools_frozenset(self):
        assert isinstance(FollowUpGate_canonical.NON_COMPLETING_TOOLS, frozenset)
        assert len(FollowUpGate_canonical.NON_COMPLETING_TOOLS) > 0


# ===================================================================
# 3. ConvergenceMonitor behavioral regression
# ===================================================================


class TestConvergenceMonitorRegression:
    """Behavioral regression: ConvergenceMonitor must behave identically after extraction."""

    def test_no_injection_without_contract(self):
        cm = ConvergenceMonitor_canonical("Test prompt without output contract")
        cm.record_turn(["run_snippet"])
        cm.record_turn(["run_snippet"])
        cm.record_turn(["run_snippet"])
        assert cm.should_inject() is False

    def test_injection_with_contract(self):
        prompt = "OUTPUT CONTRACT\n* figure_umap: UMAP plot\n* table: gene table"
        cm = ConvergenceMonitor_canonical(prompt)
        for _ in range(ConvergenceMonitor_canonical.THRESHOLD):
            cm.record_turn(["run_snippet"])
        assert cm.should_inject() is True

    def test_no_injection_after_execute_code(self):
        prompt = "OUTPUT CONTRACT\n* figure_umap: UMAP plot"
        cm = ConvergenceMonitor_canonical(prompt)
        cm.record_turn(["execute_code"])
        for _ in range(5):
            cm.record_turn(["run_snippet"])
        assert cm.should_inject() is False

    def test_steering_escalation_levels(self):
        prompt = "OUTPUT CONTRACT\n* figure_umap: UMAP plot"
        cm = ConvergenceMonitor_canonical(prompt)
        for _ in range(ConvergenceMonitor_canonical.THRESHOLD):
            cm.record_turn(["run_snippet"])
        msg1 = cm.build_steering_message()
        assert "execute_code" in msg1
        assert cm._escalation == 1

        cm._consecutive_readonly = ConvergenceMonitor_canonical.THRESHOLD
        msg2 = cm.build_steering_message()
        assert "IMPORTANT" in msg2
        assert cm._escalation == 2

        cm._consecutive_readonly = ConvergenceMonitor_canonical.THRESHOLD
        msg3 = cm.build_steering_message()
        assert "URGENT" in msg3
        assert cm._force_execute_next is True

    def test_force_tool_choice_after_level3(self):
        prompt = "OUTPUT CONTRACT\n* figure: plot"
        cm = ConvergenceMonitor_canonical(prompt)
        for _ in range(ConvergenceMonitor_canonical.THRESHOLD):
            cm.record_turn(["run_snippet"])
        cm.build_steering_message()  # level 1
        cm._consecutive_readonly = ConvergenceMonitor_canonical.THRESHOLD
        cm.build_steering_message()  # level 2
        cm._consecutive_readonly = ConvergenceMonitor_canonical.THRESHOLD
        cm.build_steering_message()  # level 3
        assert cm.should_force_tool_choice() is True

    def test_class_constants(self):
        assert isinstance(ConvergenceMonitor_canonical.READ_ONLY_TOOLS, frozenset)
        assert "run_snippet" in ConvergenceMonitor_canonical.READ_ONLY_TOOLS
        assert isinstance(ConvergenceMonitor_canonical.ARTIFACT_TOOLS, frozenset)
        assert "execute_code" in ConvergenceMonitor_canonical.ARTIFACT_TOOLS
        assert ConvergenceMonitor_canonical.THRESHOLD == 2
        assert ConvergenceMonitor_canonical.ESCALATION_LEVELS == 3


# ===================================================================
# 4. Artifact helper behavioral regression
# ===================================================================


class TestArtifactHelperRegression:
    """Behavioral regression: artifact helpers work via both module functions and TurnController delegation."""

    def test_slugify_for_filename_basic(self):
        assert slugify_for_filename("hello world!") == "hello_world"

    def test_slugify_for_filename_empty(self):
        assert slugify_for_filename("") == "tool"

    def test_slugify_for_filename_max_len(self):
        result = slugify_for_filename("a" * 100)
        assert len(result) == 48

    def test_slugify_via_turn_controller(self):
        """TurnController._slugify_for_filename delegates to the same function."""
        assert TurnController._slugify_for_filename("test case!") == slugify_for_filename("test case!")

    def test_persist_tool_debug_output_writes_file(self, tmp_path):
        ctx = SimpleNamespace(
            _filesystem_context=SimpleNamespace(workspace_dir=tmp_path),
        )
        path = persist_tool_debug_output(
            ctx,
            "execute_code",
            "stdout:\nfull debug output",
            turn_index=3,
            tool_index=1,
            description="Plot T-cell marker UMAP",
        )
        assert path is not None
        assert path.exists()
        assert path.parent == tmp_path / "results"
        text = path.read_text(encoding="utf-8")
        assert "tool=execute_code" in text
        assert "turn=3" in text
        assert text.endswith("stdout:\nfull debug output")

    def test_persist_tool_debug_output_via_turn_controller(self, tmp_path):
        """TurnController._persist_tool_debug_output delegates correctly."""
        ctx = SimpleNamespace(
            _filesystem_context=SimpleNamespace(workspace_dir=tmp_path),
        )
        controller = TurnController(
            ctx=ctx,
            prompt_builder=MagicMock(),
            tool_runtime=MagicMock(),
        )
        path = controller._persist_tool_debug_output(
            "execute_code",
            "debug output",
            turn_index=1,
            tool_index=1,
            description="test",
        )
        assert path is not None
        assert path.exists()

    def test_persist_execute_code_source_writes_py(self, tmp_path):
        ctx = SimpleNamespace(
            _filesystem_context=SimpleNamespace(workspace_dir=tmp_path),
        )
        path = persist_execute_code_source(
            ctx,
            "import scanpy as sc\nsc.pl.umap(adata)\n",
            turn_index=2,
            tool_index=1,
            description="Plot UMAP",
        )
        assert path is not None
        assert path.suffix == ".py"
        assert "sc.pl.umap" in path.read_text(encoding="utf-8")

    def test_persist_execute_code_stdout_writes_log(self, tmp_path):
        ctx = SimpleNamespace(
            _filesystem_context=SimpleNamespace(workspace_dir=tmp_path),
        )
        path = persist_execute_code_stdout(
            ctx,
            "UMAP completed\ncells=700\n",
            turn_index=2,
            tool_index=1,
            description="Plot UMAP",
        )
        assert path is not None
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "tool=execute_code_stdout" in text
        assert text.endswith("UMAP completed\ncells=700\n")

    def test_persist_tool_debug_output_no_workspace(self):
        ctx = SimpleNamespace(_filesystem_context=None)
        path = persist_tool_debug_output(
            ctx, "tool", "output", turn_index=1, tool_index=1
        )
        assert path is None

    def test_persist_tool_debug_output_empty_output(self, tmp_path):
        ctx = SimpleNamespace(
            _filesystem_context=SimpleNamespace(workspace_dir=tmp_path),
        )
        path = persist_tool_debug_output(
            ctx, "tool", "", turn_index=1, tool_index=1
        )
        assert path is None

    def test_log_tool_debug_output_chunks(self, caplog):
        import logging

        long_output = "B" * 4500
        with caplog.at_level(logging.INFO):
            log_tool_debug_output(
                tool_name="execute_code",
                output=long_output,
                turn_index=2,
                tool_index=1,
                path=Path("/tmp/fake.log"),
                chunk_size=4000,
            )
        messages = [record.message for record in caplog.records]
        assert any("execute_code_result_saved" in m for m in messages)
        assert any("chunk=1/2" in m for m in messages)
        assert any("chunk=2/2" in m for m in messages)

    def test_log_execute_code_source_chunks(self, caplog):
        import logging

        long_code = "print('x')\n" * 500
        with caplog.at_level(logging.INFO):
            log_execute_code_source(
                code=long_code,
                turn_index=2,
                tool_index=1,
                path=Path("/tmp/fake.py"),
                chunk_size=4000,
            )
        messages = [record.message for record in caplog.records]
        assert any("execute_code_source_saved" in m for m in messages)

    def test_log_execute_code_stdout_chunks(self, caplog):
        import logging

        long_stdout = "stdout line\n" * 500
        with caplog.at_level(logging.INFO):
            log_execute_code_stdout(
                stdout=long_stdout,
                turn_index=2,
                tool_index=1,
                path=Path("/tmp/stdout.log"),
                chunk_size=4000,
            )
        messages = [record.message for record in caplog.records]
        assert any("execute_code_stdout_saved" in m for m in messages)

    def test_save_conversation_log_no_env(self, monkeypatch):
        """No-op when OV_AGENT_LOG_DIR is not set."""
        monkeypatch.delenv("OV_AGENT_LOG_DIR", raising=False)
        save_conversation_log([{"role": "user", "content": "hello"}])
        # Should not raise

    def test_save_conversation_log_writes_file(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        monkeypatch.setenv("OV_AGENT_LOG_DIR", str(log_dir))
        save_conversation_log([{"role": "user", "content": "hello"}])
        files = list(log_dir.glob("agent_conversation_*.json"))
        assert len(files) == 1
        import json

        data = json.loads(files[0].read_text())
        assert isinstance(data, list)


# ===================================================================
# 5. TurnController facade preservation
# ===================================================================


class TestTurnControllerFacadePreserved:
    """TurnController still has all expected methods after decomposition."""

    def test_constructor_signature(self):
        sig = inspect.signature(TurnController.__init__)
        params = list(sig.parameters.keys())
        assert params == ["self", "ctx", "prompt_builder", "tool_runtime"]

    def test_run_agentic_loop_is_async(self):
        assert inspect.iscoroutinefunction(TurnController.run_agentic_loop)

    def test_run_agentic_loop_signature(self):
        sig = inspect.signature(TurnController.run_agentic_loop)
        params = list(sig.parameters.keys())
        assert "request" in params
        assert "adata" in params
        assert "event_callback" in params
        assert "cancel_event" in params

    def test_delegation_methods_exist(self):
        """All artifact delegation methods exist on TurnController."""
        for name in [
            "_persist_harness_history",
            "_save_conversation_log",
            "_slugify_for_filename",
            "_persist_tool_debug_output",
            "_persist_execute_code_source",
            "_persist_execute_code_stdout",
            "_log_tool_debug_output",
            "_log_execute_code_source",
            "_log_execute_code_stdout",
        ]:
            assert hasattr(TurnController, name), f"Missing: {name}"

    def test_static_methods_preserved(self):
        """Static methods remain static after delegation."""
        for name in [
            "_save_conversation_log",
            "_slugify_for_filename",
            "_log_tool_debug_output",
            "_log_execute_code_source",
            "_log_execute_code_stdout",
        ]:
            raw = inspect.getattr_static(TurnController, name)
            assert isinstance(raw, staticmethod), f"{name} should be staticmethod"

    def test_append_tool_results_exists(self):
        assert hasattr(TurnController, "_append_tool_results")
