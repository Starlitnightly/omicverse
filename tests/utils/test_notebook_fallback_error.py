"""Tests for notebook fallback error reporting in ToolRuntime._tool_execute_code.

Verifies that when notebook session execution fails and falls back to
in-process execution, the error is captured in _notebook_fallback_error
and surfaced in the tool output so the LLM can read and fix it.
"""

from __future__ import annotations

import json
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Minimal stubs so we can import ToolRuntime without the full omicverse stack
# ---------------------------------------------------------------------------

def _make_mock_ctx():
    """Build a minimal AgentContext-like mock."""
    ctx = MagicMock()
    ctx._tool_blocked_in_plan_mode.return_value = False
    ctx._code_only_mode = False
    ctx.use_notebook_execution = False   # default: no notebook
    ctx._notebook_executor = None
    ctx._security_scanner = MagicMock()
    ctx._security_scanner.scan.return_value = []
    ctx._security_scanner.has_critical.return_value = False
    ctx._security_config = MagicMock()
    ctx._security_config.approval_mode = None
    ctx._temporary_api_keys = MagicMock()
    ctx._temporary_api_keys.return_value.__enter__ = MagicMock(return_value=None)
    ctx._temporary_api_keys.return_value.__exit__ = MagicMock(return_value=False)
    ctx.enable_filesystem_context = False
    ctx._filesystem_context = None
    ctx._config = MagicMock()
    ctx._config.execution.sandbox_fallback_policy = None
    return ctx


def _make_mock_executor(ctx):
    """Build a minimal AnalysisExecutor-like mock."""
    from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
    executor = MagicMock(spec=AnalysisExecutor)
    executor._notebook_fallback_error = None
    executor.check_code_prerequisites.return_value = ""
    return executor


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestNotebookFallbackErrorReporting(unittest.TestCase):

    def _make_tool_runtime(self):
        from omicverse.utils.ovagent.tool_runtime import ToolRuntime
        ctx = _make_mock_ctx()
        executor = _make_mock_executor(ctx)
        rt = ToolRuntime(ctx=ctx, executor=executor)
        return rt, ctx, executor

    # ------------------------------------------------------------------
    # 1. Happy path: notebook succeeds — no error in output
    # ------------------------------------------------------------------
    def test_no_error_when_execution_succeeds(self):
        rt, ctx, executor = self._make_tool_runtime()
        import anndata
        import numpy as np
        adata = anndata.AnnData(np.zeros((5, 3)))

        # Simulate: execute_generated_code returns dict (capture_stdout=True path)
        executor.execute_generated_code.return_value = {
            "adata": adata,
            "stdout": "hello from notebook\n",
        }
        executor._notebook_fallback_error = None  # no error

        result = rt._tool_execute_code("x = 1", "simple assignment", adata)

        self.assertIsInstance(result, dict)
        self.assertNotIn("WARNING", result["output"])
        self.assertIn("hello from notebook", result["output"])
        print("[PASS] test_no_error_when_execution_succeeds")

    # ------------------------------------------------------------------
    # 2. Notebook fails → fallback → error IS reported in output
    # ------------------------------------------------------------------
    def test_error_reported_when_notebook_fallback_occurs(self):
        rt, ctx, executor = self._make_tool_runtime()
        import anndata
        import numpy as np
        adata = anndata.AnnData(np.zeros((5, 3)))

        # Simulate: notebook failed, fell back to in-process; executor stores the error
        simulated_error = "AttributeError: PathCollection.set() got an unexpected keyword argument 'figsize'"

        def fake_execute(code, adata, capture_stdout=False):
            # The executor sets the fallback error (mimicking WARN_AND_FALLBACK branch)
            executor._notebook_fallback_error = simulated_error
            # Returns dict (in-process fallback with capture_stdout)
            return {"adata": adata, "stdout": ""}

        executor.execute_generated_code.side_effect = fake_execute

        result = rt._tool_execute_code(
            "ov.pl.embedding(adata, figsize=(6,6))",  # bad kwarg
            "plot embedding",
            adata,
        )

        self.assertIsInstance(result, dict)
        output = result["output"]
        print(f"  output snippet: {output[:300]}")
        self.assertIn("WARNING", output)
        self.assertIn("notebook session execution failed", output)
        self.assertIn(simulated_error, output)
        self.assertIn("Fell back to in-process execution", output)
        print("[PASS] test_error_reported_when_notebook_fallback_occurs")

    # ------------------------------------------------------------------
    # 3. Empty code → early rejection before executor is called
    # ------------------------------------------------------------------
    def test_empty_code_returns_error_json(self):
        from omicverse.utils.ovagent.tool_runtime import ToolRuntime
        import anndata, numpy as np

        ctx = _make_mock_ctx()
        executor = _make_mock_executor(ctx)
        rt = ToolRuntime(ctx=ctx, executor=executor)

        adata = anndata.AnnData(np.zeros((5, 3)))

        # dispatch_tool path for empty code
        # We test the guard directly via the dispatch args
        import asyncio

        class FakeToolCall:
            name = "execute_code"
            arguments = {"code": "   ", "description": "empty"}

        async def run():
            return await rt.dispatch_tool(FakeToolCall(), adata, "test")

        raw = asyncio.run(run())
        parsed = json.loads(raw)
        self.assertIn("error", parsed)
        self.assertIn("non-empty", parsed["error"])
        executor.execute_generated_code.assert_not_called()
        print("[PASS] test_empty_code_returns_error_json")

    # ------------------------------------------------------------------
    # 4. Prereq warnings appear in output alongside normal stdout
    # ------------------------------------------------------------------
    def test_prereq_warnings_included_in_output(self):
        rt, ctx, executor = self._make_tool_runtime()
        import anndata, numpy as np
        adata = anndata.AnnData(np.zeros((5, 3)))

        executor.check_code_prerequisites.return_value = "adata.X is not raw counts"
        executor.execute_generated_code.return_value = {
            "adata": adata,
            "stdout": "done\n",
        }
        executor._notebook_fallback_error = None

        result = rt._tool_execute_code("ov.pp.normalize(adata)", "normalize", adata)

        output = result["output"]
        print(f"  output snippet: {output[:300]}")
        self.assertIn("PREREQUISITE WARNINGS", output)
        self.assertIn("adata.X is not raw counts", output)
        self.assertIn("done", output)
        print("[PASS] test_prereq_warnings_included_in_output")

    # ------------------------------------------------------------------
    # 5. Both notebook error AND prereq warning appear together
    # ------------------------------------------------------------------
    def test_notebook_error_and_prereq_warning_combined(self):
        rt, ctx, executor = self._make_tool_runtime()
        import anndata, numpy as np
        adata = anndata.AnnData(np.zeros((5, 3)))

        simulated_error = "KernelError: something went wrong in the kernel"
        executor.check_code_prerequisites.return_value = "Missing normalization step"

        def fake_execute(code, adata, capture_stdout=False):
            executor._notebook_fallback_error = simulated_error
            return {"adata": adata, "stdout": "partial output\n"}

        executor.execute_generated_code.side_effect = fake_execute

        result = rt._tool_execute_code("bad_code()", "bad call", adata)
        output = result["output"]
        print(f"  output snippet: {output[:400]}")
        self.assertIn("WARNING", output)
        self.assertIn(simulated_error, output)
        self.assertIn("PREREQUISITE WARNINGS", output)
        self.assertIn("Missing normalization step", output)
        self.assertIn("partial output", output)
        print("[PASS] test_notebook_error_and_prereq_warning_combined")


if __name__ == "__main__":
    unittest.main(verbosity=2)
