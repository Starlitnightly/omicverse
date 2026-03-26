"""Tests for ovagent.event_stream — runtime observability contract.

Verifies:
  - RuntimeEventEmitter routes events through logger + harness sinks
  - No print() calls remain in core runtime modules (turn_controller,
    tool_runtime, subagent_controller)
  - Bootstrap CLI banners are preserved and explicitly scoped
  - Structured events cover dispatch, results, retries, and completion
"""

import ast
import asyncio
import importlib
import importlib.machinery
import importlib.util
import inspect
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any
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
    reason="Observability contract tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
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

# Now import the modules under test
from omicverse.utils.ovagent.event_stream import (
    RuntimeEventEmitter,
    _summarize,
)

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Paths to source modules for AST-level print detection
# ---------------------------------------------------------------------------

OVAGENT_DIR = PACKAGE_ROOT / "utils" / "ovagent"

# Core runtime modules that MUST NOT contain print() calls
RUNTIME_MODULES = [
    OVAGENT_DIR / "turn_controller.py",
    OVAGENT_DIR / "tool_runtime.py",
    OVAGENT_DIR / "subagent_controller.py",
]

# CLI banner modules where print() is explicitly allowed
CLI_BANNER_MODULES = [
    OVAGENT_DIR / "bootstrap.py",
    OVAGENT_DIR / "auth.py",
]


# ===========================================================================
# 1. No print() in core runtime modules
# ===========================================================================


def _find_print_calls(filepath: Path) -> list[int]:
    """Return line numbers of print() calls in *filepath* using AST."""
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Direct call: print(...)
            if isinstance(func, ast.Name) and func.id == "print":
                lines.append(node.lineno)
            # builtins.print(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "print"
                and isinstance(func.value, ast.Name)
                and func.value.id == "builtins"
            ):
                lines.append(node.lineno)
    return lines


class TestNoPrintInRuntimeModules:
    """Core runtime modules must not use print() for state reporting."""

    @pytest.mark.parametrize(
        "module_path",
        RUNTIME_MODULES,
        ids=[p.stem for p in RUNTIME_MODULES],
    )
    def test_no_print_calls(self, module_path: Path):
        lines = _find_print_calls(module_path)
        assert lines == [], (
            f"{module_path.name} contains print() calls on lines {lines}. "
            f"Runtime state must go through logger/event_stream, not stdout."
        )


class TestCLIBannersPreserved:
    """Bootstrap and auth modules retain their CLI banner print() calls."""

    @pytest.mark.parametrize(
        "module_path",
        CLI_BANNER_MODULES,
        ids=[p.stem for p in CLI_BANNER_MODULES],
    )
    def test_cli_banners_exist(self, module_path: Path):
        lines = _find_print_calls(module_path)
        assert len(lines) > 0, (
            f"{module_path.name} should retain CLI banner print() calls "
            f"as non-authoritative human feedback."
        )

    def test_bootstrap_docstring_declares_cli_contract(self):
        source = (OVAGENT_DIR / "bootstrap.py").read_text()
        assert "CLI banner contract" in source, (
            "bootstrap.py must explicitly declare its CLI banner contract "
            "in the module docstring."
        )


# ===========================================================================
# 2. RuntimeEventEmitter unit tests
# ===========================================================================


class TestRuntimeEventEmitter:
    """Test the unified observability surface."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_emit_logs_to_logger(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.emit("status", {"key": "value"}, category="test_cat"))
        assert any("runtime_event" in rec.message for rec in caplog.records)
        assert any("source=test" in rec.message for rec in caplog.records)

    def test_emit_records_to_trace(self):
        recorder = MagicMock()
        recorder.add_event = MagicMock()
        emitter = RuntimeEventEmitter(recorder=recorder, source="test")
        self._run(emitter.emit("status", "hello"))
        recorder.add_event.assert_called_once()
        event = recorder.add_event.call_args[0][0]
        assert event["type"] == "status"

    def test_emit_calls_event_callback(self):
        captured = []

        async def callback(event):
            captured.append(event)

        emitter = RuntimeEventEmitter(event_callback=callback, source="test")
        self._run(emitter.emit("tool_call", {"name": "bash"}))
        assert len(captured) == 1
        assert captured[0]["type"] == "tool_call"

    def test_emit_without_callback_or_recorder(self, caplog):
        """Emitter works with no recorder or callback — logger only."""
        emitter = RuntimeEventEmitter(source="bare")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.emit("status", "test"))
        assert any("runtime_event" in rec.message for rec in caplog.records)

    def test_emit_extra_merged(self):
        captured = []

        async def callback(event):
            captured.append(event)

        emitter = RuntimeEventEmitter(event_callback=callback, source="test")
        self._run(
            emitter.emit(
                "done", "ok", extra={"cancelled": True}
            )
        )
        assert captured[0]["cancelled"] is True

    def test_turn_started(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.turn_started(0, 15, "auto"))
        assert any("turn_started" in rec.message for rec in caplog.records)
        assert any("turn=1/15" in rec.message for rec in caplog.records)

    def test_turn_cancelled(self):
        captured = []

        async def callback(event):
            captured.append(event)

        emitter = RuntimeEventEmitter(event_callback=callback, source="test")
        self._run(emitter.turn_cancelled("before_llm_call"))
        assert len(captured) == 1
        assert captured[0]["type"] == "done"
        assert captured[0].get("cancelled") is True

    def test_max_turns_reached(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.WARNING, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.max_turns_reached(15))
        assert any("max_turns_reached" in rec.message for rec in caplog.records)

    def test_tool_dispatched(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.tool_dispatched("bash", {"command": "ls"}))
        assert any("tool_dispatched" in rec.message for rec in caplog.records)
        assert any("name=bash" in rec.message for rec in caplog.records)

    def test_tool_completed(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.tool_completed("bash", status="success"))
        assert any("tool_completed" in rec.message for rec in caplog.records)

    def test_execution_completed(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.execution_completed("Run analysis"))
        assert any("execution_completed" in rec.message for rec in caplog.records)

    def test_delegation_completed(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.delegation_completed("explore", task="find data"))
        assert any("delegation" in rec.message for rec in caplog.records)

    def test_task_finished(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.task_finished("Analysis complete"))
        assert any("task_finished" in rec.message for rec in caplog.records)

    def test_subagent_turn(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.subagent_turn("explore", 0, 5))
        assert any("subagent_turn" in rec.message for rec in caplog.records)

    def test_subagent_tool(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.subagent_tool("explore", "bash", {"command": "ls"}))
        assert any("subagent_tool" in rec.message for rec in caplog.records)

    def test_subagent_finished(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.subagent_finished("explore", "Found 3 files"))
        assert any("subagent_finished" in rec.message for rec in caplog.records)

    def test_conversation_log_saved(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            emitter.conversation_log_saved("/tmp/test.json")
        assert any("conversation_log_saved" in rec.message for rec in caplog.records)

    def test_agent_response(self, caplog):
        emitter = RuntimeEventEmitter(source="test")
        with caplog.at_level(logging.INFO, logger="omicverse.utils.ovagent.event_stream"):
            self._run(emitter.agent_response("Here is the result"))
        assert any("agent_response" in rec.message for rec in caplog.records)


# ===========================================================================
# 3. _summarize helper
# ===========================================================================


class TestSummarize:
    def test_none(self):
        assert _summarize(None) == "-"

    def test_string(self):
        assert _summarize("hello") == "hello"

    def test_long_string_truncated(self):
        long = "x" * 200
        assert len(_summarize(long)) <= 120

    def test_dict_shows_keys(self):
        result = _summarize({"name": "bash", "args": {}})
        assert "name" in result
        assert "args" in result

    def test_other_type(self):
        assert _summarize(42) == "42"


# ===========================================================================
# 4. Export contract — RuntimeEventEmitter in __init__.py
# ===========================================================================


class TestExportContract:
    def test_event_emitter_exported(self):
        import omicverse.utils.ovagent as pkg
        assert hasattr(pkg, "RuntimeEventEmitter")
        assert pkg.RuntimeEventEmitter is RuntimeEventEmitter

    def test_event_emitter_in_all(self):
        import omicverse.utils.ovagent as pkg
        assert "RuntimeEventEmitter" in pkg.__all__


# ===========================================================================
# 5. Observability coherence — structured events cover all lifecycle phases
# ===========================================================================


class TestObservabilityCoherence:
    """Verify that the emitter covers all required lifecycle phases."""

    REQUIRED_METHODS = [
        "turn_started",
        "turn_cancelled",
        "max_turns_reached",
        "tool_dispatched",
        "tool_completed",
        "execution_completed",
        "delegation_completed",
        "task_finished",
        "subagent_turn",
        "subagent_tool",
        "subagent_finished",
        "agent_response",
        "conversation_log_saved",
    ]

    def test_emitter_has_all_lifecycle_methods(self):
        for method_name in self.REQUIRED_METHODS:
            assert hasattr(RuntimeEventEmitter, method_name), (
                f"RuntimeEventEmitter missing required method: {method_name}"
            )
            method = getattr(RuntimeEventEmitter, method_name)
            assert callable(method), (
                f"RuntimeEventEmitter.{method_name} is not callable"
            )

    def test_emitter_methods_are_documented(self):
        for method_name in self.REQUIRED_METHODS:
            method = getattr(RuntimeEventEmitter, method_name)
            assert method.__doc__, (
                f"RuntimeEventEmitter.{method_name} lacks a docstring"
            )
