"""Regression tests for review comment 4134941752 confirmed findings.

Covers three implementation fixes:
1. agent_backend_common._get_shared_executor — no private _shutdown probe,
   no stale atexit accumulation.
2. analysis_diagnostics.auto_install_package — built-in blocked packages
   are preserved even when config supplies a custom blocklist.
3. analysis_sandbox._handle_context_write — CONTEXT_WRITE no longer uses
   unconstrained eval over local_vars.
"""

from __future__ import annotations

import ast
import concurrent.futures
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


UTILS_DIR = Path(__file__).resolve().parents[2] / "omicverse" / "utils"
OVAGENT_DIR = UTILS_DIR / "ovagent"


# ===================================================================
# 1. Shared executor lifecycle (agent_backend_common)
# ===================================================================


class TestSharedExecutorLifecycle:
    """_get_shared_executor must not probe private attrs or accumulate atexit."""

    def test_no_private_shutdown_access_in_source(self):
        """Source code must not reference _shutdown on the executor."""
        source = (UTILS_DIR / "agent_backend_common.py").read_text()
        assert "._shutdown" not in source, (
            "agent_backend_common.py still probes the private _shutdown attribute"
        )

    def test_get_shared_executor_returns_executor(self):
        from omicverse.utils.agent_backend_common import _get_shared_executor
        exc = _get_shared_executor()
        assert isinstance(exc, concurrent.futures.ThreadPoolExecutor)

    def test_get_shared_executor_returns_same_instance(self):
        from omicverse.utils.agent_backend_common import _get_shared_executor
        a = _get_shared_executor()
        b = _get_shared_executor()
        assert a is b

    def test_atexit_registered_only_once(self):
        """atexit.register should be called at most once, not per-executor."""
        import omicverse.utils.agent_backend_common as mod

        # Reset global state for a clean test
        old_exc = mod._SHARED_EXECUTOR
        old_flag = mod._EXECUTOR_ATEXIT_REGISTERED
        mod._SHARED_EXECUTOR = None
        mod._EXECUTOR_ATEXIT_REGISTERED = False

        try:
            with patch("atexit.register") as mock_register:
                mod._get_shared_executor()
                mod._get_shared_executor()
                mod._get_shared_executor()
                # Should only register once regardless of how many times called
                assert mock_register.call_count == 1
        finally:
            # Restore
            mod._SHARED_EXECUTOR = old_exc
            mod._EXECUTOR_ATEXIT_REGISTERED = old_flag

    def test_shutdown_helper_shuts_down_current_executor(self):
        """_shutdown_shared_executor cleanly shuts down the current executor."""
        import omicverse.utils.agent_backend_common as mod

        mock_exc = MagicMock(spec=concurrent.futures.ThreadPoolExecutor)
        old = mod._SHARED_EXECUTOR
        mod._SHARED_EXECUTOR = mock_exc
        try:
            mod._shutdown_shared_executor()
            mock_exc.shutdown.assert_called_once_with(wait=False)
        finally:
            mod._SHARED_EXECUTOR = old

    def test_shutdown_helper_noop_when_none(self):
        """_shutdown_shared_executor is safe when no executor exists."""
        import omicverse.utils.agent_backend_common as mod

        old = mod._SHARED_EXECUTOR
        mod._SHARED_EXECUTOR = None
        try:
            mod._shutdown_shared_executor()  # should not raise
        finally:
            mod._SHARED_EXECUTOR = old


# ===================================================================
# 2. Package blocklist safety (analysis_diagnostics)
# ===================================================================


class TestPackageBlocklistPreservation:
    """auto_install_package must always block built-in dangerous packages."""

    _BUILTIN_BLOCKED = {"os", "sys", "subprocess", "shutil", "signal", "ctypes"}

    def _make_ctx(self, config_blocklist=None):
        ctx = MagicMock()
        if config_blocklist is not None:
            ctx._config.execution.package_blocklist = config_blocklist
        else:
            ctx._config = None
        return ctx

    def test_builtin_blocked_without_config(self):
        """All built-in blocked packages are rejected when no config is set."""
        from omicverse.utils.ovagent.analysis_diagnostics import auto_install_package
        ctx = self._make_ctx(config_blocklist=None)
        for pkg in self._BUILTIN_BLOCKED:
            result = auto_install_package(ctx, pkg)
            assert result is False, f"{pkg} should be blocked without config"

    def test_builtin_blocked_with_empty_config_blocklist(self):
        """Built-in blocked packages survive even if config provides an empty list."""
        from omicverse.utils.ovagent.analysis_diagnostics import auto_install_package
        ctx = self._make_ctx(config_blocklist=[])
        for pkg in self._BUILTIN_BLOCKED:
            result = auto_install_package(ctx, pkg)
            assert result is False, f"{pkg} should still be blocked with empty config list"

    def test_builtin_blocked_with_custom_config_blocklist(self):
        """Built-in blocked packages are retained when config adds custom entries."""
        from omicverse.utils.ovagent.analysis_diagnostics import auto_install_package
        ctx = self._make_ctx(config_blocklist=["custom_bad_pkg"])
        for pkg in self._BUILTIN_BLOCKED:
            result = auto_install_package(ctx, pkg)
            assert result is False, f"{pkg} should still be blocked alongside config entries"

    def test_config_blocklist_entries_also_blocked(self):
        """Config-supplied blocklist entries are also enforced."""
        from omicverse.utils.ovagent.analysis_diagnostics import auto_install_package
        ctx = self._make_ctx(config_blocklist=["custom_bad_pkg"])
        result = auto_install_package(ctx, "custom_bad_pkg")
        assert result is False, "Config-supplied blocklist entry should be blocked"

    def test_source_uses_union_not_replacement(self):
        """Source code must union built-in and config blocklists, not replace."""
        source = (OVAGENT_DIR / "analysis_diagnostics.py").read_text()
        # The old pattern was: blocklist = cfg.execution.package_blocklist
        # (simple assignment replaces the defaults)
        # Check that the source does NOT have a bare replacement pattern
        assert "blocklist = cfg.execution.package_blocklist" not in source, (
            "Blocklist should be unioned with defaults, not replaced"
        )


# ===================================================================
# 3. CONTEXT_WRITE safe expression handling (analysis_sandbox)
# ===================================================================


class TestContextWriteSafeEval:
    """_handle_context_write must not use unconstrained eval."""

    def test_no_eval_call_in_source(self):
        """analysis_sandbox.py must not call eval() anywhere."""
        source = (OVAGENT_DIR / "analysis_sandbox.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "eval":
                    pytest.fail(
                        f"analysis_sandbox.py still uses eval() at line {node.lineno}"
                    )

    def test_safe_resolve_direct_name(self):
        """Direct variable name lookup works."""
        from omicverse.utils.ovagent.analysis_sandbox import _safe_resolve_expr
        result = _safe_resolve_expr("my_var", {"my_var": 42})
        assert result == 42

    def test_safe_resolve_dotted_attr(self):
        """Dotted attribute access resolves through getattr chain."""
        from omicverse.utils.ovagent.analysis_sandbox import _safe_resolve_expr
        obj = MagicMock()
        obj.shape = (100, 200)
        result = _safe_resolve_expr("adata.shape", {"adata": obj})
        assert result == (100, 200)

    def test_safe_resolve_literal_string(self):
        """Quoted string literals are resolved via ast.literal_eval."""
        from omicverse.utils.ovagent.analysis_sandbox import _safe_resolve_expr
        result = _safe_resolve_expr("'hello world'", {})
        assert result == "hello world"

    def test_safe_resolve_literal_number(self):
        from omicverse.utils.ovagent.analysis_sandbox import _safe_resolve_expr
        assert _safe_resolve_expr("42", {}) == 42
        assert _safe_resolve_expr("3.14", {}) == 3.14

    def test_safe_resolve_literal_list(self):
        from omicverse.utils.ovagent.analysis_sandbox import _safe_resolve_expr
        assert _safe_resolve_expr("[1, 2, 3]", {}) == [1, 2, 3]

    def test_safe_resolve_literal_dict(self):
        from omicverse.utils.ovagent.analysis_sandbox import _safe_resolve_expr
        assert _safe_resolve_expr("{'a': 1}", {}) == {"a": 1}

    def test_safe_resolve_fallback_to_string(self):
        """Unknown expressions fall back to the raw string."""
        from omicverse.utils.ovagent.analysis_sandbox import _safe_resolve_expr
        result = _safe_resolve_expr("some complex thing", {})
        assert result == "some complex thing"

    def test_safe_resolve_rejects_function_calls(self):
        """Function calls in expressions are not evaluated — returned as string."""
        from omicverse.utils.ovagent.analysis_sandbox import _safe_resolve_expr
        result = _safe_resolve_expr("len(data)", {"data": [1, 2, 3]})
        assert result == "len(data)"

    def test_safe_resolve_rejects_dunder_access(self):
        """__class__.__subclasses__ style attacks return raw string."""
        from omicverse.utils.ovagent.analysis_sandbox import _safe_resolve_expr
        # __class__ is not a valid identifier-only dotted chain that exists in local_vars
        result = _safe_resolve_expr("().__class__.__subclasses__()", {})
        assert isinstance(result, str)

    def test_context_write_uses_safe_resolve(self):
        """Full CONTEXT_WRITE directive uses _safe_resolve_expr, not eval."""
        from omicverse.utils.ovagent.analysis_sandbox import _handle_context_write

        ctx = MagicMock()
        fc = MagicMock()
        ctx._filesystem_context = fc

        local_vars = {"result_count": 42}
        _handle_context_write(
            ctx, "# CONTEXT_WRITE: stats_result -> result_count", local_vars,
        )
        fc.write_note.assert_called_once()
        args = fc.write_note.call_args
        assert args[0][0] == "stats_result"
        assert args[0][1] == 42
        assert args[0][2] == "results"

    def test_context_write_literal_value(self):
        """CONTEXT_WRITE with a literal value resolves without eval."""
        from omicverse.utils.ovagent.analysis_sandbox import _handle_context_write

        ctx = MagicMock()
        fc = MagicMock()
        ctx._filesystem_context = fc

        _handle_context_write(
            ctx, "# CONTEXT_WRITE: note_key -> 'hello'", {},
        )
        fc.write_note.assert_called_once()
        assert fc.write_note.call_args[0][1] == "hello"

    def test_context_write_dotted_attr(self):
        """CONTEXT_WRITE with dotted attribute access works."""
        from omicverse.utils.ovagent.analysis_sandbox import _handle_context_write

        ctx = MagicMock()
        fc = MagicMock()
        ctx._filesystem_context = fc

        obj = MagicMock()
        obj.n_obs = 5000
        _handle_context_write(
            ctx, "# CONTEXT_WRITE: result_cells -> adata.n_obs", {"adata": obj},
        )
        fc.write_note.assert_called_once()
        assert fc.write_note.call_args[0][1] == 5000


# ===================================================================
# 4. No dependency or interface drift
# ===================================================================


class TestNoDependencyDrift:
    """Fixes must not introduce new imports or change public surfaces."""

    @staticmethod
    def _top_level_imports(path: Path) -> set:
        tree = ast.parse(path.read_text())
        result = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module:
                    result.add(node.module.split(".")[0])
        return result

    @pytest.mark.parametrize("mod_path", [
        UTILS_DIR / "agent_backend_common.py",
        OVAGENT_DIR / "analysis_diagnostics.py",
        OVAGENT_DIR / "analysis_sandbox.py",
    ])
    def test_no_new_heavy_imports(self, mod_path):
        """Modified modules must not add top-level heavy package imports."""
        imports = self._top_level_imports(mod_path)
        for banned in ["numpy", "pandas", "torch"]:
            assert banned not in imports, (
                f"{mod_path.name} has new top-level import of {banned}"
            )

    def test_get_shared_executor_still_exported(self):
        from omicverse.utils.agent_backend_common import _get_shared_executor
        assert callable(_get_shared_executor)

    def test_auto_install_package_still_exported(self):
        from omicverse.utils.ovagent.analysis_diagnostics import auto_install_package
        assert callable(auto_install_package)

    def test_handle_context_write_still_exported(self):
        from omicverse.utils.ovagent.analysis_sandbox import _handle_context_write
        assert callable(_handle_context_write)

    def test_safe_resolve_expr_exported(self):
        from omicverse.utils.ovagent.analysis_sandbox import _safe_resolve_expr
        assert callable(_safe_resolve_expr)
