"""Tests for PR #601 P0 security fixes (task-052).

Covers three confirmed findings from Claude review comment 4134544053:
1. OpenAI backend log/error paths no longer expose raw endpoint URLs
2. Analysis diagnostics only auto-installs validated package names
3. Sandbox open() is workspace-bounded, not unrestricted builtins.open

These tests run under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="P0 security tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight stubs to avoid heavy omicverse.__init__
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for name in ["omicverse", "omicverse.utils"]:
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(PACKAGE_ROOT if name == "omicverse" else PACKAGE_ROOT / "utils")]
        pkg.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = pkg
        if name == "omicverse.utils":
            sys.modules["omicverse"].utils = pkg  # type: ignore[attr-defined]


# ===================================================================
# SECTION 1: OpenAI backend endpoint redaction
# ===================================================================


class TestEndpointRedaction:
    """_redact_url strips path/port from endpoint URLs."""

    def test_redacts_full_url(self):
        from omicverse.utils.agent_backend_openai import _redact_url
        result = _redact_url("https://api.openai.com/v1/chat/completions")
        # Exact equality avoids CodeQL incomplete-URL-substring-sanitization pattern
        assert result == "https://api.openai.com/..."

    def test_redacts_custom_port(self):
        from omicverse.utils.agent_backend_openai import _redact_url
        result = _redact_url("http://my-server.example.com:8443/secret-path/v1")
        assert result == "http://my-server.example.com/..."

    def test_redacts_localhost(self):
        from omicverse.utils.agent_backend_openai import _redact_url
        result = _redact_url("http://localhost:11434/v1")
        assert result == "http://localhost/..."

    def test_redacts_empty_or_none(self):
        from omicverse.utils.agent_backend_openai import _redact_url
        assert _redact_url("") == "https://unknown/..."
        assert _redact_url(None) == "https://unknown/..."  # type: ignore[arg-type]

    def test_connection_error_redacted(self):
        """_wrap_openai_connection_error must not expose raw endpoint."""
        from omicverse.utils.agent_backend_openai import _wrap_openai_connection_error

        backend = MagicMock()
        exc = ConnectionRefusedError("Connection refused")
        err = _wrap_openai_connection_error(
            backend, exc, "https://secret-host.internal:9999/private/v1",
        )
        msg = str(err)
        # Full message equality — verifies port/path redaction without substring hostname check
        assert msg == (
            "OpenAI-compatible connection failed for"
            " https://secret-host.internal/...: Connection refused"
        )

    def test_ollama_connection_error_redacted(self):
        """Ollama connection error must not expose raw endpoint path."""
        from omicverse.utils.agent_backend_openai import _wrap_openai_connection_error

        backend = MagicMock()
        exc = ConnectionRefusedError("Connection refused")
        err = _wrap_openai_connection_error(backend, exc, "http://localhost:11434/v1")
        msg = str(err)
        # Full message equality — verifies port redaction and Ollama detection
        assert msg == (
            "Could not connect to Ollama at http://localhost/...."
            " Start the Ollama server and verify the model is installed."
        )


# ===================================================================
# SECTION 2: Package name validation
# ===================================================================


class TestPackageNameValidation:
    """extract_package_name and auto_install_package reject malicious names."""

    def test_valid_names_pass(self):
        from omicverse.utils.ovagent.analysis_diagnostics import _is_valid_package_name
        assert _is_valid_package_name("numpy")
        assert _is_valid_package_name("scikit-learn")
        assert _is_valid_package_name("torch")
        assert _is_valid_package_name("umap-learn")
        assert _is_valid_package_name("Pillow")
        assert _is_valid_package_name("opencv-python")
        assert _is_valid_package_name("scvi-tools")
        assert _is_valid_package_name("A1")

    def test_rejects_shell_injection(self):
        from omicverse.utils.ovagent.analysis_diagnostics import _is_valid_package_name
        assert not _is_valid_package_name("numpy; rm -rf /")
        assert not _is_valid_package_name("numpy && curl evil.com")
        assert not _is_valid_package_name("$(whoami)")
        assert not _is_valid_package_name("`id`")

    def test_rejects_pip_flags(self):
        from omicverse.utils.ovagent.analysis_diagnostics import _is_valid_package_name
        assert not _is_valid_package_name("--index-url")
        assert not _is_valid_package_name("-e")
        assert not _is_valid_package_name("--extra-index-url=evil")

    def test_rejects_path_traversal(self):
        from omicverse.utils.ovagent.analysis_diagnostics import _is_valid_package_name
        assert not _is_valid_package_name("../../../etc/passwd")
        assert not _is_valid_package_name("/tmp/evil.whl")

    def test_rejects_empty_and_long(self):
        from omicverse.utils.ovagent.analysis_diagnostics import _is_valid_package_name
        assert not _is_valid_package_name("")
        assert not _is_valid_package_name("a" * 200)

    def test_rejects_whitespace(self):
        from omicverse.utils.ovagent.analysis_diagnostics import _is_valid_package_name
        assert not _is_valid_package_name("numpy pandas")
        assert not _is_valid_package_name("numpy\tpandas")

    def test_extract_package_name_validates(self):
        from omicverse.utils.ovagent.analysis_diagnostics import extract_package_name
        # Normal case
        assert extract_package_name("No module named 'numpy'") == "numpy"
        # Malicious case: injected command should fail validation
        assert extract_package_name("No module named '--index-url'") is None
        assert extract_package_name("No module named 'foo; rm -rf /'") is None

    def test_auto_install_rejects_invalid_name(self):
        from omicverse.utils.ovagent.analysis_diagnostics import auto_install_package
        ctx = MagicMock()
        ctx._config = None
        # Should return False for invalid package names without calling pip
        assert auto_install_package(ctx, "--extra-index-url=evil") is False
        assert auto_install_package(ctx, "$(whoami)") is False


# ===================================================================
# SECTION 3: Sandbox open() workspace boundary
# ===================================================================


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


class TestSandboxOpenBoundary:
    """Sandbox open() must deny access outside the workspace boundary."""

    def test_open_allows_workspace_files(self):
        """open() inside the workspace directory should work normally."""
        from omicverse.utils.ovagent.analysis_sandbox import build_sandbox_globals

        ctx = _MinimalCtx()
        # Set up a real temp workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            fs_ctx = MagicMock()
            fs_ctx.workspace_dir = tmpdir
            ctx._filesystem_context = fs_ctx

            sandbox = build_sandbox_globals(ctx)
            sandbox_open = sandbox["__builtins__"]["open"]

            # Write and read inside workspace
            test_file = os.path.join(tmpdir, "test.txt")
            with sandbox_open(test_file, "w") as f:
                f.write("hello")
            with sandbox_open(test_file, "r") as f:
                assert f.read() == "hello"

    def test_open_allows_cwd_files(self):
        """open() for files in the current working directory should work."""
        from omicverse.utils.ovagent.analysis_sandbox import build_sandbox_globals

        ctx = _MinimalCtx()
        sandbox = build_sandbox_globals(ctx)
        sandbox_open = sandbox["__builtins__"]["open"]

        # Create a temp file in CWD and verify access
        cwd = os.getcwd()
        test_path = os.path.join(cwd, "_test_sandbox_cwd.tmp")
        try:
            with sandbox_open(test_path, "w") as f:
                f.write("cwd test")
            with sandbox_open(test_path, "r") as f:
                assert f.read() == "cwd test"
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)

    def test_open_allows_tmp_files(self):
        """open() for files in system temp directory should work."""
        from omicverse.utils.ovagent.analysis_sandbox import build_sandbox_globals

        ctx = _MinimalCtx()
        sandbox = build_sandbox_globals(ctx)
        sandbox_open = sandbox["__builtins__"]["open"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            tmp_path = f.name
        try:
            with sandbox_open(tmp_path, "w") as f:
                f.write("tmp test")
            with sandbox_open(tmp_path, "r") as f:
                assert f.read() == "tmp test"
        finally:
            os.unlink(tmp_path)

    def test_open_denies_outside_workspace(self):
        """open() must raise PermissionError for paths outside allowed roots."""
        from omicverse.utils.ovagent.analysis_sandbox import build_sandbox_globals

        ctx = _MinimalCtx()
        with tempfile.TemporaryDirectory() as tmpdir:
            fs_ctx = MagicMock()
            fs_ctx.workspace_dir = tmpdir
            ctx._filesystem_context = fs_ctx

            sandbox = build_sandbox_globals(ctx)
            sandbox_open = sandbox["__builtins__"]["open"]

            # Try to read a system file outside the workspace
            with pytest.raises(PermissionError, match="outside the workspace boundary"):
                sandbox_open("/etc/hostname", "r")

    def test_open_denies_path_traversal(self):
        """open() must block path traversal attempts."""
        from omicverse.utils.ovagent.analysis_sandbox import build_sandbox_globals

        ctx = _MinimalCtx()
        with tempfile.TemporaryDirectory() as tmpdir:
            fs_ctx = MagicMock()
            fs_ctx.workspace_dir = tmpdir
            ctx._filesystem_context = fs_ctx

            sandbox = build_sandbox_globals(ctx)
            sandbox_open = sandbox["__builtins__"]["open"]

            traversal_path = os.path.join(tmpdir, "..", "..", "etc", "passwd")
            with pytest.raises(PermissionError, match="outside the workspace boundary"):
                sandbox_open(traversal_path, "r")

    def test_open_is_not_builtin_open(self):
        """The sandbox must NOT expose builtins.open directly."""
        import builtins
        from omicverse.utils.ovagent.analysis_sandbox import build_sandbox_globals

        ctx = _MinimalCtx()
        sandbox = build_sandbox_globals(ctx)
        sandbox_open = sandbox["__builtins__"]["open"]
        assert sandbox_open is not builtins.open

    def test_sandbox_globals_no_raw_open_in_allowed_builtins(self):
        """Verify 'open' is no longer in the allowed_builtins list as raw builtin."""
        import ast
        source_path = (
            Path(__file__).resolve().parents[2]
            / "omicverse" / "utils" / "ovagent" / "analysis_sandbox.py"
        )
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_sandbox_globals":
                # Find the allowed_builtins list
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name) and target.id == "allowed_builtins":
                                if isinstance(stmt.value, ast.List):
                                    names = [
                                        elt.value
                                        for elt in stmt.value.elts
                                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                                    ]
                                    assert "open" not in names, (
                                        "builtins.open must not appear in the allowed_builtins list; "
                                        "use the workspace-bounded wrapper instead"
                                    )
                                    return
        pytest.fail("Could not find allowed_builtins list in build_sandbox_globals")
