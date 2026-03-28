"""
Regression tests for local Python executor security hardening (task-007).

Covers:
1. Restricted builtins — dangerous eval/import primitives removed from sandbox
2. SafeOsProxy injection — import os returns safe proxy, blocks dangerous ops
3. SyntaxError propagation — scanner SyntaxError surfaces as RuntimeError
4. Benign execution — normal code still works under restricted sandbox
5. Public signature preservation
"""

import pytest

from omicverse.utils.agent_backend import OmicVerseLLMBackend
from omicverse.utils.model_config import ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_local_backend(monkeypatch):
    """Create a backend configured for local Python execution."""
    monkeypatch.setattr(
        ModelConfig, "get_provider_from_model", lambda *a, **kw: "python"
    )
    return OmicVerseLLMBackend(
        system_prompt="test", model="python", api_key=""
    )


# ---------------------------------------------------------------------------
# 1. Restricted builtins
# ---------------------------------------------------------------------------


class TestRestrictedBuiltins:
    """Dangerous eval/import builtins must not be available inside the sandbox.

    These tests reference the names without calling them to avoid triggering
    the AST security scanner (which independently blocks eval()/exec() calls).
    This verifies that the restricted builtins dict itself is effective.
    """

    def test_eval_not_accessible(self, monkeypatch):
        backend = _make_local_backend(monkeypatch)
        with pytest.raises(RuntimeError, match="NameError"):
            backend._run_python_local("x = eval")

    def test_exec_not_accessible(self, monkeypatch):
        backend = _make_local_backend(monkeypatch)
        with pytest.raises(RuntimeError, match="NameError"):
            backend._run_python_local("x = exec")

    def test_compile_not_accessible(self, monkeypatch):
        backend = _make_local_backend(monkeypatch)
        with pytest.raises(RuntimeError, match="NameError"):
            backend._run_python_local("x = compile")

    def test_dunder_import_is_safe_replacement(self, monkeypatch):
        """__import__ in sandbox is the safe replacement, not the real one."""
        backend = _make_local_backend(monkeypatch)
        # Access the name (not a call) to verify it's our safe wrapper
        result = backend._run_python_local(
            "result = __import__.__name__"
        )
        assert result == "'limited_import'"

    def test_breakpoint_not_accessible(self, monkeypatch):
        """breakpoint() is not caught by the AST scanner, only by builtins."""
        backend = _make_local_backend(monkeypatch)
        with pytest.raises(RuntimeError, match="NameError"):
            backend._run_python_local("breakpoint()")

    def test_benign_builtins_still_work(self, monkeypatch):
        """print, len, range, int, str, list, dict, etc. remain available."""
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local("print(len(list(range(5))))")
        assert result == "5"

    def test_open_still_available(self, monkeypatch):
        """open() is intentionally kept for bioinformatics file I/O."""
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local("result = callable(open)")
        assert result == "True"

    def test_type_and_isinstance_work(self, monkeypatch):
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local("print(isinstance(42, int))")
        assert result == "True"

    def test_denied_builtins_list(self):
        """Verify that the sandbox excludes dangerous builtins."""
        from omicverse.utils.agent_sandbox import _EXCLUDED_BUILTINS
        expected_minimum = {"eval", "exec", "compile", "__import__", "breakpoint"}
        assert expected_minimum <= _EXCLUDED_BUILTINS


# ---------------------------------------------------------------------------
# 2. SafeOsProxy injection
# ---------------------------------------------------------------------------


class TestSafeOsProxyInjection:
    """import os inside sandbox must return SafeOsProxy, blocking dangerous ops."""

    def test_os_global_is_safe_proxy(self, monkeypatch):
        """The pre-injected os global is a SafeOsProxy."""
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local("result = type(os).__name__")
        assert result == "'SafeOsProxy'"

    def test_import_os_returns_safe_proxy(self, monkeypatch):
        """import os returns SafeOsProxy, not the real os module."""
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local(
            "import os\nresult = type(os).__name__"
        )
        assert result == "'SafeOsProxy'"

    def test_os_putenv_blocked_at_runtime(self, monkeypatch):
        """os.putenv() is blocked by SafeOsProxy (not caught by AST scanner)."""
        backend = _make_local_backend(monkeypatch)
        with pytest.raises(RuntimeError, match="blocked"):
            backend._run_python_local("import os\nos.putenv('X', 'Y')")

    def test_os_unsetenv_blocked_at_runtime(self, monkeypatch):
        """os.unsetenv() is blocked by SafeOsProxy."""
        backend = _make_local_backend(monkeypatch)
        with pytest.raises(RuntimeError, match="blocked"):
            backend._run_python_local("import os\nos.unsetenv('X')")

    def test_os_path_join_works(self, monkeypatch):
        """os.path operations remain functional through SafeOsProxy."""
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local(
            "import os\nresult = os.path.join('a', 'b', 'c')"
        )
        assert "a" in result and "b" in result and "c" in result

    def test_os_getcwd_works(self, monkeypatch):
        """os.getcwd() is a safe operation and should work."""
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local("import os\nresult = os.getcwd()")
        assert result  # Should return a non-empty path string

    def test_from_os_path_import_works(self, monkeypatch):
        """from os.path import join should work through SafeOsProxy."""
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local(
            "from os.path import join\nresult = join('x', 'y')"
        )
        assert "x" in result and "y" in result

    def test_from_os_import_path(self, monkeypatch):
        """from os import path should return real os.path via SafeOsProxy."""
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local(
            "from os import path\nresult = path.join('a', 'b')"
        )
        assert "a" in result and "b" in result

    def test_os_environ_readable(self, monkeypatch):
        """os.environ should be readable (not in blocked list)."""
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local(
            "import os\nresult = type(os.environ).__name__"
        )
        assert result  # Should return the type name without error


# ---------------------------------------------------------------------------
# 3. SyntaxError propagation
# ---------------------------------------------------------------------------


class TestSyntaxErrorPropagation:
    """SyntaxError must surface as RuntimeError, not be silently swallowed."""

    def test_syntax_error_raises_runtime_error(self, monkeypatch):
        backend = _make_local_backend(monkeypatch)
        with pytest.raises(RuntimeError, match="Python execution failed"):
            backend._run_python_local("def foo(:\n  pass")

    def test_syntax_error_is_diagnosable(self, monkeypatch):
        """RuntimeError from syntax error contains useful diagnostic info."""
        backend = _make_local_backend(monkeypatch)
        with pytest.raises(RuntimeError) as exc_info:
            backend._run_python_local("if True\n  print('oops')")
        msg = str(exc_info.value)
        assert "Python execution failed" in msg

    def test_syntax_error_preserves_chain(self, monkeypatch):
        """The original SyntaxError is chained as __cause__."""
        backend = _make_local_backend(monkeypatch)
        with pytest.raises(RuntimeError) as exc_info:
            backend._run_python_local("x = =")
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, SyntaxError)

    def test_valid_code_no_syntax_error(self, monkeypatch):
        """Valid code executes without SyntaxError issues."""
        backend = _make_local_backend(monkeypatch)
        result = backend._run_python_local("print(2 + 2)")
        assert result == "4"


# ---------------------------------------------------------------------------
# 4. Public signature preservation
# ---------------------------------------------------------------------------


class TestPublicSignaturePreserved:
    """OmicVerseLLMBackend public API must remain unchanged."""

    def test_run_signature_unchanged(self):
        import inspect
        sig = inspect.signature(OmicVerseLLMBackend.run)
        params = list(sig.parameters.keys())
        assert params == ["self", "user_prompt"]

    def test_chat_signature_unchanged(self):
        import inspect
        sig = inspect.signature(OmicVerseLLMBackend.chat)
        params = list(sig.parameters.keys())
        assert params == ["self", "messages", "tools", "tool_choice"]

    def test_stream_signature_unchanged(self):
        import inspect
        sig = inspect.signature(OmicVerseLLMBackend.stream)
        params = list(sig.parameters.keys())
        assert params == ["self", "user_prompt"]
