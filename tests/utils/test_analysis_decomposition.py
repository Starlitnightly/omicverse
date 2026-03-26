"""Tests for analysis_executor Phase 4 decomposition.

Verifies that:
- analysis_transformer.py exposes ProactiveCodeTransformer independently
- analysis_diagnostics.py exposes diagnostic/repair helpers independently
- analysis_sandbox.py exposes sandbox/execution helpers independently
- analysis_executor.py remains a thin facade re-exporting all public API
- Existing import paths (from analysis_executor) still work
- No inline implementations remain in the facade
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

OVAGENT_DIR = Path(__file__).resolve().parent.parent.parent / "omicverse" / "utils" / "ovagent"


# -------------------------------------------------------------------
# Helper: minimal ctx mock for facade construction
# -------------------------------------------------------------------

class _MinimalSecurityConfig:
    approval_mode = MagicMock()
    approval_mode.value = "never"
    restrict_introspection = False
    extra_blocked_modules = set()


class _MinimalCtx:
    _config = None
    _llm = None
    _security_scanner = MagicMock()
    _security_config = _MinimalSecurityConfig()
    _approval_handler = None
    _last_run_trace = None
    _filesystem_context = None
    _notebook_executor = None
    use_notebook_execution = False
    enable_filesystem_context = False

    def _get_harness_session_id(self):
        return "test-session"

    def _extract_python_code(self, text):
        return text

    def _temporary_api_keys(self):
        from contextlib import nullcontext
        return nullcontext()

    def _emit(self, level, msg, category):
        pass


# ===================================================================
# SECTION 1: Extracted module independence
# ===================================================================


class TestAnalysisTransformerModule:
    """analysis_transformer.py is an independent module with ProactiveCodeTransformer."""

    def test_direct_import(self):
        from omicverse.utils.ovagent.analysis_transformer import ProactiveCodeTransformer
        t = ProactiveCodeTransformer()
        assert t is not None
        assert not hasattr(t, "_ctx")

    def test_transform_works_standalone(self):
        from omicverse.utils.ovagent.analysis_transformer import ProactiveCodeTransformer
        t = ProactiveCodeTransformer()
        result = t.transform("print('hello')")
        assert "_mpl.use('Agg')" in result

    def test_class_attrs_present(self):
        from omicverse.utils.ovagent.analysis_transformer import ProactiveCodeTransformer
        assert isinstance(ProactiveCodeTransformer.INPLACE_FUNCTIONS, set)
        assert isinstance(ProactiveCodeTransformer.KWARG_RENAMES, dict)

    def test_module_has_no_agentcontext_import(self):
        """analysis_transformer.py should not depend on AgentContext at runtime."""
        source = (OVAGENT_DIR / "analysis_transformer.py").read_text()
        tree = ast.parse(source)
        runtime_imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                for alias in node.names:
                    runtime_imports.add(alias.name)
        assert "AgentContext" not in runtime_imports


class TestAnalysisDiagnosticsModule:
    """analysis_diagnostics.py exposes diagnostic helpers independently."""

    def test_direct_imports(self):
        from omicverse.utils.ovagent.analysis_diagnostics import (
            _PACKAGE_ALIASES,
            check_code_prerequisites,
            apply_execution_error_fix,
            extract_package_name,
            auto_install_package,
            diagnose_error_with_llm,
            validate_outputs,
            generate_completion_code,
        )
        assert isinstance(_PACKAGE_ALIASES, dict)
        assert callable(check_code_prerequisites)
        assert callable(apply_execution_error_fix)
        assert callable(extract_package_name)
        assert callable(auto_install_package)
        assert inspect.iscoroutinefunction(diagnose_error_with_llm)
        assert inspect.iscoroutinefunction(generate_completion_code)

    def test_extract_package_name_standalone(self):
        from omicverse.utils.ovagent.analysis_diagnostics import extract_package_name
        assert extract_package_name("No module named 'torch'") == "torch"
        assert extract_package_name("some other error") is None

    def test_validate_outputs_standalone(self):
        from omicverse.utils.ovagent.analysis_diagnostics import validate_outputs
        missing = validate_outputs("print('hello')")
        assert isinstance(missing, list)
        assert len(missing) == 0

    def test_package_aliases_known_entries(self):
        from omicverse.utils.ovagent.analysis_diagnostics import _PACKAGE_ALIASES
        assert _PACKAGE_ALIASES.get("cv2") == "opencv-python"
        assert _PACKAGE_ALIASES.get("sklearn") == "scikit-learn"
        assert _PACKAGE_ALIASES.get("PIL") == "Pillow"


class TestAnalysisSandboxModule:
    """analysis_sandbox.py exposes sandbox helpers independently."""

    def test_direct_imports(self):
        from omicverse.utils.ovagent.analysis_sandbox import (
            build_sandbox_globals,
            execute_generated_code,
            execute_snippet_readonly,
            request_approval,
            normalize_doublet_obs,
            process_context_directives,
            figure_autosave_dir,
            inject_figure_autosave,
        )
        assert callable(build_sandbox_globals)
        assert callable(execute_generated_code)
        assert callable(execute_snippet_readonly)
        assert callable(request_approval)
        assert callable(normalize_doublet_obs)
        assert callable(process_context_directives)
        assert callable(figure_autosave_dir)
        assert callable(inject_figure_autosave)

    def test_normalize_doublet_obs_standalone(self):
        from omicverse.utils.ovagent.analysis_sandbox import normalize_doublet_obs
        mock_adata = MagicMock()
        mock_adata.obs.columns = ["predicted_doublet"]
        mock_adata.obs.__contains__ = lambda s, x: x in ["predicted_doublet"]
        normalize_doublet_obs(mock_adata)

    def test_build_sandbox_globals_with_ctx(self):
        from omicverse.utils.ovagent.analysis_sandbox import build_sandbox_globals
        ctx = _MinimalCtx()
        g = build_sandbox_globals(ctx)
        assert "__builtins__" in g
        assert "print" in g["__builtins__"]

    def test_parse_plan_step_standalone(self):
        from omicverse.utils.ovagent.analysis_sandbox import _parse_plan_step
        result = _parse_plan_step("Load data [pending]")
        assert result is not None
        assert result["status"] == "pending"
        assert "Load data" in result["description"]


# ===================================================================
# SECTION 2: Facade backward compatibility
# ===================================================================


class TestFacadeReExports:
    """analysis_executor.py re-exports all symbols that callers expect."""

    def test_proactive_code_transformer_reexport(self):
        from omicverse.utils.ovagent.analysis_executor import ProactiveCodeTransformer
        t = ProactiveCodeTransformer()
        assert t.transform is not None

    def test_package_aliases_reexport(self):
        from omicverse.utils.ovagent.analysis_executor import _PACKAGE_ALIASES
        assert isinstance(_PACKAGE_ALIASES, dict)
        assert _PACKAGE_ALIASES.get("cv2") == "opencv-python"

    def test_analysis_executor_class_reexport(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        ctx = _MinimalCtx()
        ae = AnalysisExecutor(ctx)
        assert ae._ctx is ctx

    def test_ovagent_init_imports(self):
        from omicverse.utils.ovagent import AnalysisExecutor, ProactiveCodeTransformer
        assert AnalysisExecutor is not None
        assert ProactiveCodeTransformer is not None


class TestFacadeMethodDelegation:
    """AnalysisExecutor methods delegate to extracted helper modules."""

    def test_check_code_prerequisites_delegates(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        ctx = _MinimalCtx()
        ae = AnalysisExecutor(ctx)
        result = ae.check_code_prerequisites("print('hi')", None)
        assert isinstance(result, str)

    def test_extract_package_name_static_delegates(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        assert AnalysisExecutor.extract_package_name("No module named 'foo'") == "foo"

    def test_validate_outputs_delegates(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        ctx = _MinimalCtx()
        ae = AnalysisExecutor(ctx)
        result = ae.validate_outputs("print('hello')")
        assert isinstance(result, list)

    def test_normalize_doublet_obs_static_delegates(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        mock_adata = MagicMock()
        mock_adata.obs.columns = []
        mock_adata.obs.__contains__ = lambda s, x: False
        AnalysisExecutor.normalize_doublet_obs(mock_adata)

    def test_build_sandbox_globals_delegates(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        ctx = _MinimalCtx()
        ae = AnalysisExecutor(ctx)
        g = ae.build_sandbox_globals()
        assert "__builtins__" in g


# ===================================================================
# SECTION 3: Facade is thin — no inline implementation
# ===================================================================


class TestFacadeIsThin:
    """analysis_executor.py should only contain delegation, no business logic."""

    def test_facade_line_count(self):
        """Facade should be significantly smaller than the original (~973 lines)."""
        source = (OVAGENT_DIR / "analysis_executor.py").read_text()
        lines = [l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")]
        # The facade should be under 150 non-blank non-comment lines
        assert len(lines) < 150, (
            f"analysis_executor.py has {len(lines)} non-blank lines — "
            "expected < 150 for a thin facade"
        )

    def test_no_regex_imports_in_facade(self):
        """Facade should not import re — regex logic lives in helpers."""
        source = (OVAGENT_DIR / "analysis_executor.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "re", "Facade imports 're' — logic not fully extracted"
            if isinstance(node, ast.ImportFrom) and node.module == "re":
                pytest.fail("Facade imports from 're' — logic not fully extracted")

    def test_no_builtins_import_in_facade(self):
        """Facade should not import builtins — sandbox logic lives in analysis_sandbox."""
        source = (OVAGENT_DIR / "analysis_executor.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "builtins", "Facade imports 'builtins'"

    def test_extracted_modules_exist(self):
        assert (OVAGENT_DIR / "analysis_transformer.py").exists()
        assert (OVAGENT_DIR / "analysis_diagnostics.py").exists()
        assert (OVAGENT_DIR / "analysis_sandbox.py").exists()

    def test_facade_has_no_class_body_logic(self):
        """Every method body should be a single return/delegation statement."""
        source = (OVAGENT_DIR / "analysis_executor.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "AnalysisExecutor":
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == "__init__":
                            continue
                        # Each method body should have at most 2 statements
                        # (a return or an expression)
                        body_stmts = [
                            s for s in item.body
                            if not isinstance(s, ast.Expr)
                            or not isinstance(s.value, (ast.Constant, ast.Str))
                        ]
                        assert len(body_stmts) <= 2, (
                            f"AnalysisExecutor.{item.name} has {len(body_stmts)} "
                            "statements — expected thin delegation"
                        )


# ===================================================================
# SECTION 4: Public API inventory (must match decomposition contracts)
# ===================================================================


class TestPublicAPIPreserved:
    """AnalysisExecutor still exposes all 13 public methods."""

    EXPECTED_PUBLIC = {
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
    }

    def test_all_public_methods_present(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        actual_public = {
            name for name in dir(AnalysisExecutor)
            if not name.startswith("_")
            and callable(getattr(AnalysisExecutor, name))
        }
        assert actual_public == self.EXPECTED_PUBLIC, (
            f"API drift: added={actual_public - self.EXPECTED_PUBLIC}, "
            f"removed={self.EXPECTED_PUBLIC - actual_public}"
        )

    def test_extract_package_name_is_static(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        raw = inspect.getattr_static(AnalysisExecutor, "extract_package_name")
        assert isinstance(raw, staticmethod)

    def test_normalize_doublet_obs_is_static(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        raw = inspect.getattr_static(AnalysisExecutor, "normalize_doublet_obs")
        assert isinstance(raw, staticmethod)

    def test_diagnose_error_with_llm_is_async(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        assert inspect.iscoroutinefunction(AnalysisExecutor.diagnose_error_with_llm)

    def test_generate_completion_code_is_async(self):
        from omicverse.utils.ovagent.analysis_executor import AnalysisExecutor
        assert inspect.iscoroutinefunction(AnalysisExecutor.generate_completion_code)
