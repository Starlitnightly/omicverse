"""Tests for the ExecutionRepairLoop and FailureEnvelope subsystem.

Acceptance criteria verified:
  - Execution failures produce structured diagnostic payloads (FailureEnvelope).
  - Recovery supports multiple bounded attempts.
  - LLM-guided repair receives code + error + traceback + dataset context in a
    consistent format.
  - Regex transforms remain optional guardrails instead of owning the full
    repair path.
"""

import asyncio
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Guard: these tests require the harness env-var
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Repair-loop tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs
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

from omicverse.utils.ovagent.repair_loop import (
    DEFAULT_MAX_RETRIES,
    ExecutionRepairLoop,
    FailureEnvelope,
    RepairAttempt,
    RepairResult,
    build_dataset_context,
    build_llm_repair_prompt,
)

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAdata:
    """Minimal AnnData stand-in for dataset context tests."""

    def __init__(self, n_obs=100, n_vars=200):
        self.shape = (n_obs, n_vars)
        self.obs = SimpleNamespace(columns=["batch", "cell_type", "n_genes"])
        self.var = SimpleNamespace(columns=["gene_name", "highly_variable"])
        self.obsm = SimpleNamespace(keys=lambda: ["X_pca", "X_umap"])


class _FakeExecutor:
    """Minimal AnalysisExecutor double for repair loop tests."""

    def __init__(
        self,
        exec_side_effects=None,
        guardrail_return=None,
        llm_return=None,
    ):
        self._exec_call_count = 0
        self._exec_side_effects = exec_side_effects or []
        self._guardrail_return = guardrail_return
        self._llm_return = llm_return
        self._ctx = SimpleNamespace(
            _llm=None,
            _extract_python_code=lambda text: text,
        )

    def execute_generated_code(self, code, adata, capture_stdout=True):
        idx = self._exec_call_count
        self._exec_call_count += 1
        if idx < len(self._exec_side_effects):
            effect = self._exec_side_effects[idx]
            if isinstance(effect, Exception):
                raise effect
            return effect
        return {"stdout": "", "adata": adata}

    def apply_execution_error_fix(self, code, error_msg):
        return self._guardrail_return

    async def diagnose_error_with_llm(self, code, error_msg, traceback_str, adata):
        return self._llm_return


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ===================================================================
# 1. FailureEnvelope — structured diagnostic payload
# ===================================================================


class TestFailureEnvelope:
    """Execution failures produce structured diagnostic payloads."""

    def test_from_exception_basic(self):
        try:
            raise ValueError("bad column 'batch'")
        except ValueError as exc:
            env = FailureEnvelope.from_exception(
                exc, phase="execution", retry_count=0, code="print(1)"
            )

        assert env.phase == "execution"
        assert env.exception == "ValueError"
        assert "bad column" in env.summary
        assert env.retry_count == 0
        assert env.code == "print(1)"
        assert env.retry_safe is True
        assert isinstance(env.repair_hints, list)

    def test_from_exception_preserves_traceback(self):
        try:
            raise RuntimeError("segfault simulation")
        except RuntimeError as exc:
            env = FailureEnvelope.from_exception(exc, phase="transform")

        assert "RuntimeError" in env.traceback_excerpt
        assert "segfault simulation" in env.traceback_excerpt

    def test_to_dict_roundtrip(self):
        env = FailureEnvelope(
            phase="validation",
            exception="AssertionError",
            summary="expected 3 got 5",
            traceback_excerpt="Traceback...",
            retry_count=2,
            repair_hints=["check shape"],
            retry_safe=False,
            code="assert x == 3",
            dataset_context="100 obs x 200 vars",
        )
        d = env.to_dict()
        assert d["phase"] == "validation"
        assert d["exception"] == "AssertionError"
        assert d["summary"] == "expected 3 got 5"
        assert d["retry_count"] == 2
        assert d["repair_hints"] == ["check shape"]
        assert d["retry_safe"] is False
        assert d["code"] == "assert x == 3"
        assert d["dataset_context"] == "100 obs x 200 vars"

    def test_envelope_contract_fields(self):
        """All fields from the interface contract are present."""
        env = FailureEnvelope(
            phase="execution",
            exception="TypeError",
            summary="test",
            traceback_excerpt="tb",
            retry_count=0,
        )
        required_fields = {
            "phase", "exception", "summary", "traceback_excerpt",
            "retry_count", "repair_hints", "retry_safe",
        }
        actual = set(env.to_dict().keys())
        assert required_fields.issubset(actual), (
            f"Missing contract fields: {required_fields - actual}"
        )

    def test_traceback_bounded_length(self):
        """Traceback excerpt is bounded to prevent unbounded payloads."""
        try:
            # Generate a long traceback chain
            exec("1/0")
        except Exception as exc:
            env = FailureEnvelope.from_exception(exc)

        assert len(env.traceback_excerpt) <= 2000

    def test_summary_bounded_length(self):
        """Summary is bounded even for very long exception messages."""
        try:
            raise ValueError("x" * 1000)
        except ValueError as exc:
            env = FailureEnvelope.from_exception(exc)

        assert len(env.summary) <= 500


# ===================================================================
# 2. build_dataset_context
# ===================================================================


class TestBuildDatasetContext:
    def test_with_adata(self):
        adata = _FakeAdata()
        ctx = build_dataset_context(adata)
        assert "100 obs x 200 vars" in ctx
        assert "batch" in ctx

    def test_with_none(self):
        assert build_dataset_context(None) == ""


# ===================================================================
# 3. build_llm_repair_prompt — consistent format
# ===================================================================


class TestBuildLLMRepairPrompt:
    """LLM-guided repair receives code + error + traceback + dataset context."""

    def test_prompt_contains_all_sections(self):
        env = FailureEnvelope(
            phase="execution",
            exception="TypeError",
            summary="unsupported operand",
            traceback_excerpt="File test.py line 5\nTypeError: ...",
            retry_count=1,
            code="x = 1 + 'a'",
            dataset_context="100 obs x 200 vars",
            repair_hints=["check types"],
        )
        prompt = build_llm_repair_prompt(env)

        assert "--- PHASE ---" in prompt
        assert "execution" in prompt
        assert "--- EXCEPTION ---" in prompt
        assert "TypeError" in prompt
        assert "--- TRACEBACK ---" in prompt
        assert "--- CODE ---" in prompt
        assert "x = 1 + 'a'" in prompt
        assert "--- DATASET ---" in prompt
        assert "100 obs x 200 vars" in prompt
        assert "--- REPAIR HINTS ---" in prompt
        assert "check types" in prompt
        assert "--- RETRY COUNT ---" in prompt
        assert "1" in prompt

    def test_prompt_without_dataset(self):
        env = FailureEnvelope(
            phase="execution",
            exception="NameError",
            summary="name 'x' is not defined",
            traceback_excerpt="tb...",
            retry_count=0,
            code="print(x)",
        )
        prompt = build_llm_repair_prompt(env)
        assert "--- DATASET ---" not in prompt
        assert "--- CODE ---" in prompt

    def test_prompt_without_hints(self):
        env = FailureEnvelope(
            phase="execution",
            exception="NameError",
            summary="name 'x' is not defined",
            traceback_excerpt="tb...",
            retry_count=0,
            code="print(x)",
            repair_hints=[],
        )
        prompt = build_llm_repair_prompt(env)
        assert "--- REPAIR HINTS ---" not in prompt


# ===================================================================
# 4. ExecutionRepairLoop — bounded retries
# ===================================================================


class TestExecutionRepairLoopBounded:
    """Recovery supports multiple bounded attempts."""

    def test_success_on_first_try(self):
        executor = _FakeExecutor(
            exec_side_effects=[{"stdout": "ok", "adata": None}]
        )
        loop = ExecutionRepairLoop(executor, max_retries=3)
        result = _run(loop.run("print(1)", None))

        assert result.success is True
        assert len(result.attempts) == 1
        assert result.final_envelope is None

    def test_guardrail_recovery(self):
        """Guardrail produces a fix and the second attempt succeeds."""
        executor = _FakeExecutor(
            exec_side_effects=[
                ValueError("dtype error"),
                {"stdout": "fixed", "adata": None},
            ],
            guardrail_return="fixed_code",
        )
        loop = ExecutionRepairLoop(executor, max_retries=3)
        result = _run(loop.run("broken_code", None))

        assert result.success is True
        assert len(result.attempts) == 2
        assert result.attempts[0].strategy == "guardrail"
        assert result.attempts[0].success is False
        assert result.attempts[1].success is True

    def test_llm_recovery(self):
        """LLM diagnosis produces a fix after guardrail returns None."""
        executor = _FakeExecutor(
            exec_side_effects=[
                ValueError("unknown error"),
                {"stdout": "llm fixed", "adata": None},
            ],
            guardrail_return=None,
            llm_return="llm_fixed_code",
        )
        loop = ExecutionRepairLoop(executor, max_retries=3)
        result = _run(loop.run("broken_code", None))

        assert result.success is True
        assert len(result.attempts) == 2
        assert result.attempts[0].strategy == "llm"

    def test_max_retries_respected(self):
        """Loop stops after max_retries even if failures continue."""
        executor = _FakeExecutor(
            exec_side_effects=[
                ValueError("err1"),
                ValueError("err2"),
                ValueError("err3"),
                ValueError("err4"),
            ],
            guardrail_return="always_different_code",
        )
        # Need the guardrail to actually produce different code each time
        call_count = [0]
        original_fix = executor.apply_execution_error_fix

        def varying_fix(code, error_msg):
            call_count[0] += 1
            return f"fixed_v{call_count[0]}"

        executor.apply_execution_error_fix = varying_fix
        loop = ExecutionRepairLoop(executor, max_retries=2)
        result = _run(loop.run("code", None))

        assert result.success is False
        assert result.final_envelope is not None
        # 1 initial + max_retries = 3 total exec attempts
        assert len(result.attempts) <= 3

    def test_no_repair_strategy_exits_early(self):
        """When neither guardrail nor LLM produce new code, loop exits."""
        executor = _FakeExecutor(
            exec_side_effects=[ValueError("unfixable")],
            guardrail_return=None,
            llm_return=None,
        )
        loop = ExecutionRepairLoop(executor, max_retries=3)
        result = _run(loop.run("code", None))

        assert result.success is False
        assert result.final_envelope is not None
        assert result.final_envelope.retry_safe is False
        assert "no repair strategy" in result.final_envelope.repair_hints[0]

    def test_default_max_retries(self):
        assert DEFAULT_MAX_RETRIES == 3


# ===================================================================
# 5. Regex transforms are optional guardrails
# ===================================================================


class TestGuardrailsOptional:
    """Regex transforms remain optional guardrails, not the primary path."""

    def test_guardrail_skipped_when_no_match(self):
        """When guardrail returns None, loop falls through to LLM."""
        executor = _FakeExecutor(
            exec_side_effects=[
                ValueError("obscure error"),
                {"stdout": "ok", "adata": None},
            ],
            guardrail_return=None,
            llm_return="llm_fixed_code",
        )
        loop = ExecutionRepairLoop(executor, max_retries=3)
        result = _run(loop.run("code", None))

        assert result.success is True
        assert result.attempts[0].strategy == "llm"

    def test_guardrail_returns_same_code_falls_through(self):
        """When guardrail returns same code, it's not treated as a fix."""
        executor = _FakeExecutor(
            exec_side_effects=[
                ValueError("error"),
                {"stdout": "ok", "adata": None},
            ],
            guardrail_return=None,  # will be overridden
            llm_return="llm_fixed_code",
        )
        # Make guardrail return the same code
        executor.apply_execution_error_fix = lambda code, err: code
        loop = ExecutionRepairLoop(executor, max_retries=3)
        result = _run(loop.run("code", None))

        assert result.success is True
        # Should have used LLM, not guardrail
        assert result.attempts[0].strategy == "llm"

    def test_guardrail_used_as_first_attempt_when_available(self):
        """When guardrail matches, it's tried before LLM (as a guardrail)."""
        executor = _FakeExecutor(
            exec_side_effects=[
                ValueError("dtype error"),
                {"stdout": "ok", "adata": None},
            ],
            guardrail_return="guardrail_fixed",
        )
        loop = ExecutionRepairLoop(executor, max_retries=3)
        result = _run(loop.run("original_code", None))

        assert result.success is True
        assert result.attempts[0].strategy == "guardrail"


# ===================================================================
# 6. Structured diagnostic in repair result
# ===================================================================


class TestRepairResultDiagnostics:
    """RepairResult carries structured diagnostics on failure."""

    def test_failure_result_has_envelope(self):
        executor = _FakeExecutor(
            exec_side_effects=[TypeError("bad type")],
            guardrail_return=None,
            llm_return=None,
        )
        loop = ExecutionRepairLoop(executor, max_retries=3)
        result = _run(loop.run("code", _FakeAdata()))

        assert result.success is False
        env = result.final_envelope
        assert env is not None
        assert env.phase == "execution"
        assert env.exception == "TypeError"
        assert "bad type" in env.summary
        assert env.code == "code"
        assert "100 obs x 200 vars" in env.dataset_context

    def test_success_result_has_no_envelope(self):
        executor = _FakeExecutor(
            exec_side_effects=[{"stdout": "", "adata": None}]
        )
        loop = ExecutionRepairLoop(executor, max_retries=3)
        result = _run(loop.run("code", None))

        assert result.success is True
        assert result.final_envelope is None

    def test_attempts_list_tracks_history(self):
        call_count = [0]

        def varying_fix(code, error_msg):
            call_count[0] += 1
            return f"v{call_count[0]}"

        executor = _FakeExecutor(
            exec_side_effects=[
                ValueError("err1"),
                ValueError("err2"),
                {"stdout": "ok", "adata": None},
            ],
        )
        executor.apply_execution_error_fix = varying_fix
        loop = ExecutionRepairLoop(executor, max_retries=3)
        result = _run(loop.run("code", None))

        assert result.success is True
        assert len(result.attempts) == 3
        assert result.attempts[0].success is False
        assert result.attempts[1].success is False
        assert result.attempts[2].success is True


# ===================================================================
# 7. Integration: __init__.py exports include repair_loop types
# ===================================================================


class TestRepairLoopExports:
    """New repair_loop types are exported from ovagent package."""

    def test_exports_present(self):
        import omicverse.utils.ovagent as ovagent_pkg

        expected = {
            "FailureEnvelope",
            "ExecutionRepairLoop",
            "RepairAttempt",
            "RepairResult",
            "DEFAULT_MAX_RETRIES",
            "build_dataset_context",
            "build_llm_repair_prompt",
        }
        actual = set(ovagent_pkg.__all__)
        missing = expected - actual
        assert not missing, f"Missing repair_loop exports: {missing}"

    def test_exports_resolve(self):
        import omicverse.utils.ovagent as ovagent_pkg

        for name in [
            "FailureEnvelope",
            "ExecutionRepairLoop",
            "RepairAttempt",
            "RepairResult",
        ]:
            obj = getattr(ovagent_pkg, name, None)
            assert obj is not None, f"Export '{name}' is None"


# ===================================================================
# 8. Custom exec_fn support
# ===================================================================


class TestCustomExecFn:
    """Repair loop supports custom execution functions."""

    def test_custom_exec_fn(self):
        executor = _FakeExecutor()
        loop = ExecutionRepairLoop(executor, max_retries=1)

        call_log = []

        def custom_exec(code, adata):
            call_log.append(code)
            return {"stdout": "custom", "adata": adata}

        result = _run(loop.run(
            "code", None, exec_fn=custom_exec, phase="custom"
        ))

        assert result.success is True
        assert call_log == ["code"]
