"""Integration tests: security sandbox enforcement.

Validates that CodeSecurityScanner, SafeOsProxy, and SecurityConfig compose
correctly across security levels.  These tests exercise the layers as the
production ``agent_backend._run_python_local`` path uses them:

    1. Scan code with ``CodeSecurityScanner``
    2. Check ``has_critical(violations)``
    3. If critical, block execution

No production behaviour is modified -- all assertions validate existing
behaviour of the sandbox stack.
"""
from __future__ import annotations

import os

import pytest

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Security sandbox integration tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

from omicverse.utils.agent_sandbox import (
    CodeSecurityScanner,
    SafeOsProxy,
    SecurityConfig,
    SecurityLevel,
    SecurityViolation,
    ApprovalMode,
)
from omicverse.utils.agent_errors import SecurityViolationError


# ===================================================================
#  Helpers
# ===================================================================

def _scanner_for(level: SecurityLevel) -> CodeSecurityScanner:
    """Return a scanner configured from a preset security level."""
    return CodeSecurityScanner(SecurityConfig.from_level(level))


def _categories(violations: list[SecurityViolation]) -> list[str]:
    """Extract sorted-by-line category names."""
    return [v.category for v in violations]


def _severities(violations: list[SecurityViolation]) -> list[str]:
    """Extract sorted-by-line severity strings."""
    return [v.severity for v in violations]


# ===================================================================
#  1. Scanner + Config integration across security levels
# ===================================================================


class TestSecurityLevelPresets:
    """Verify that from_level presets configure the scanner correctly."""

    # -- STRICT --------------------------------------------------------

    def test_strict_shell_access_is_critical(self):
        scanner = _scanner_for(SecurityLevel.STRICT)
        violations = scanner.scan('os.system("ls")')
        assert scanner.has_critical(violations)
        assert violations[0].category == "shell_access"
        assert violations[0].severity == "critical"

    def test_strict_file_danger_is_critical(self):
        scanner = _scanner_for(SecurityLevel.STRICT)
        violations = scanner.scan('shutil.rmtree("/tmp/data")')
        assert scanner.has_critical(violations)
        assert violations[0].category == "file_danger"
        assert violations[0].severity == "critical"

    def test_strict_dangerous_import_is_critical(self):
        scanner = _scanner_for(SecurityLevel.STRICT)
        violations = scanner.scan("import subprocess")
        assert scanner.has_critical(violations)
        assert violations[0].category == "dangerous_import"
        assert violations[0].severity == "critical"

    # -- STANDARD ------------------------------------------------------

    def test_standard_file_danger_is_warning(self):
        scanner = _scanner_for(SecurityLevel.STANDARD)
        violations = scanner.scan('shutil.rmtree("/tmp/data")')
        assert len(violations) == 1
        assert violations[0].category == "file_danger"
        assert violations[0].severity == "warning"
        assert not scanner.has_critical(violations)

    def test_standard_shell_access_still_critical(self):
        scanner = _scanner_for(SecurityLevel.STANDARD)
        violations = scanner.scan('os.system("cmd")')
        assert scanner.has_critical(violations)
        assert violations[0].severity == "critical"

    def test_standard_dangerous_import_still_critical(self):
        scanner = _scanner_for(SecurityLevel.STANDARD)
        violations = scanner.scan("import subprocess")
        assert scanner.has_critical(violations)

    # -- PERMISSIVE ----------------------------------------------------

    def test_permissive_file_danger_is_warning(self):
        scanner = _scanner_for(SecurityLevel.PERMISSIVE)
        violations = scanner.scan('shutil.move("a", "b")')
        assert len(violations) == 1
        assert violations[0].severity == "warning"

    def test_permissive_dangerous_import_is_warning(self):
        scanner = _scanner_for(SecurityLevel.PERMISSIVE)
        violations = scanner.scan("import socket")
        assert len(violations) == 1
        assert violations[0].severity == "warning"
        assert not scanner.has_critical(violations)

    def test_permissive_dangerous_builtin_is_warning(self):
        scanner = _scanner_for(SecurityLevel.PERMISSIVE)
        violations = scanner.scan('eval("1+1")')
        assert len(violations) == 1
        assert violations[0].category == "dangerous_builtin"
        assert violations[0].severity == "warning"

    def test_permissive_allows_requests(self):
        scanner = _scanner_for(SecurityLevel.PERMISSIVE)
        violations = scanner.scan("import requests")
        assert violations == []

    def test_permissive_allows_urllib(self):
        scanner = _scanner_for(SecurityLevel.PERMISSIVE)
        violations = scanner.scan("import urllib")
        assert violations == []

    def test_permissive_allows_http(self):
        scanner = _scanner_for(SecurityLevel.PERMISSIVE)
        violations = scanner.scan("import http")
        assert violations == []


# ===================================================================
#  2. End-to-end scan-then-block flow
# ===================================================================


class TestScanThenBlockFlow:
    """Simulate the agent_backend._run_python_local scan-then-block pattern."""

    def _should_block(self, code: str, level: SecurityLevel = SecurityLevel.STANDARD) -> bool:
        """Return True if code would be blocked at the given level."""
        scanner = _scanner_for(level)
        violations = scanner.scan(code)
        return scanner.has_critical(violations)

    # -- import subprocess -> blocked at STANDARD -----------------------

    def test_import_subprocess_blocked_at_standard(self):
        assert self._should_block("import subprocess", SecurityLevel.STANDARD)

    def test_import_subprocess_blocked_at_strict(self):
        assert self._should_block("import subprocess", SecurityLevel.STRICT)

    # -- os.system -> blocked at all levels -----------------------------

    def test_os_system_blocked_at_strict(self):
        assert self._should_block('os.system("ls")', SecurityLevel.STRICT)

    def test_os_system_blocked_at_standard(self):
        assert self._should_block('os.system("ls")', SecurityLevel.STANDARD)

    def test_os_system_blocked_at_permissive(self):
        assert self._should_block('os.system("ls")', SecurityLevel.PERMISSIVE)

    # -- os.remove -> always shell_access (contains "os.") --------------

    def test_os_remove_blocked_at_strict(self):
        scanner = _scanner_for(SecurityLevel.STRICT)
        violations = scanner.scan('os.remove("file.txt")')
        assert scanner.has_critical(violations)
        assert violations[0].category == "shell_access"

    def test_os_remove_blocked_at_standard(self):
        # os.remove is categorized as shell_access (not file_danger),
        # so STANDARD's file_danger->warning override does not apply.
        scanner = _scanner_for(SecurityLevel.STANDARD)
        violations = scanner.scan('os.remove("file.txt")')
        assert scanner.has_critical(violations)
        assert violations[0].category == "shell_access"
        assert violations[0].severity == "critical"

    # -- eval -> blocked ------------------------------------------------

    def test_eval_blocked_at_standard(self):
        assert self._should_block('eval("1+1")', SecurityLevel.STANDARD)

    def test_eval_blocked_at_strict(self):
        assert self._should_block('eval("1+1")', SecurityLevel.STRICT)

    # -- __builtins__ access -> blocked ---------------------------------

    def test_builtins_access_blocked(self):
        assert self._should_block("x = __builtins__")

    # -- __subclasses__() -> blocked ------------------------------------

    def test_subclasses_call_blocked(self):
        assert self._should_block("obj.__subclasses__()")

    # -- SecurityViolationError integration -----------------------------

    def test_violation_error_carries_violations_list(self):
        """Validate the pattern used in _run_python_local."""
        scanner = _scanner_for(SecurityLevel.STANDARD)
        code = 'os.system("rm -rf /")'
        violations = scanner.scan(code)
        assert scanner.has_critical(violations)

        report = scanner.format_report(violations)
        err = SecurityViolationError(
            f"Code blocked by security scanner:\n{report}",
            violations=violations,
        )
        assert isinstance(err, SecurityViolationError)
        assert len(err.violations) > 0
        assert "CRITICAL" in str(err)


# ===================================================================
#  3. SafeOsProxy enforcement
# ===================================================================


class TestSafeOsProxyEnforcement:
    """SafeOsProxy blocks dangerous ops and passes through safe ones."""

    @pytest.fixture
    def proxy(self) -> SafeOsProxy:
        return SafeOsProxy()

    # -- safe operations pass through -----------------------------------

    def test_path_join(self, proxy: SafeOsProxy):
        assert proxy.path.join("a", "b") == os.path.join("a", "b")

    def test_getcwd(self, proxy: SafeOsProxy):
        assert proxy.getcwd() == os.getcwd()

    def test_listdir(self, proxy: SafeOsProxy):
        result = proxy.listdir(".")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_path_is_directly_accessible(self, proxy: SafeOsProxy):
        assert proxy.path is os.path

    def test_sep_available(self, proxy: SafeOsProxy):
        assert proxy.sep == os.sep

    # -- blocked operations raise SecurityViolationError -----------------

    def test_system_blocked(self, proxy: SafeOsProxy):
        with pytest.raises(SecurityViolationError, match="os.system"):
            proxy.system("ls")

    def test_remove_blocked(self, proxy: SafeOsProxy):
        with pytest.raises(SecurityViolationError, match="os.remove"):
            proxy.remove("file.txt")

    def test_kill_blocked(self, proxy: SafeOsProxy):
        with pytest.raises(SecurityViolationError, match="os.kill"):
            proxy.kill(1, 9)

    def test_popen_blocked(self, proxy: SafeOsProxy):
        with pytest.raises(SecurityViolationError, match="os.popen"):
            proxy.popen("ls")

    def test_unlink_blocked(self, proxy: SafeOsProxy):
        with pytest.raises(SecurityViolationError, match="os.unlink"):
            proxy.unlink("file.txt")

    def test_chmod_blocked(self, proxy: SafeOsProxy):
        with pytest.raises(SecurityViolationError, match="os.chmod"):
            proxy.chmod("/tmp", 0o777)

    def test_fork_blocked(self, proxy: SafeOsProxy):
        with pytest.raises(SecurityViolationError, match="os.fork"):
            proxy.fork()

    def test_putenv_blocked(self, proxy: SafeOsProxy):
        with pytest.raises(SecurityViolationError, match="os.putenv"):
            proxy.putenv("KEY", "VALUE")

    def test_error_is_subclass_of_sandbox_denied(self, proxy: SafeOsProxy):
        with pytest.raises(SecurityViolationError) as exc_info:
            proxy.system("ls")
        # SecurityViolationError -> SandboxDeniedError -> ExecutionError -> OVAgentError
        from omicverse.utils.agent_errors import SandboxDeniedError
        assert isinstance(exc_info.value, SandboxDeniedError)

    def test_repr(self, proxy: SafeOsProxy):
        assert "SafeOsProxy" in repr(proxy)


# ===================================================================
#  4. Custom config composition
# ===================================================================


class TestCustomConfigComposition:
    """SecurityConfig extra_* fields compose with defaults correctly."""

    def test_extra_blocked_modules_catches_pickle(self):
        cfg = SecurityConfig(extra_blocked_modules=frozenset({"pickle"}))
        scanner = CodeSecurityScanner(cfg)
        violations = scanner.scan("import pickle")
        assert len(violations) == 1
        assert violations[0].category == "dangerous_import"
        assert violations[0].severity == "critical"

    def test_extra_blocked_calls_catches_custom_call(self):
        cfg = SecurityConfig(extra_blocked_calls=frozenset({"numpy.save"}))
        scanner = CodeSecurityScanner(cfg)
        violations = scanner.scan('numpy.save("out.npy", arr)')
        assert len(violations) == 1
        assert violations[0].category == "file_danger"
        assert "numpy.save" in violations[0].description

    def test_allowed_import_roots_whitelists_requests(self):
        cfg = SecurityConfig(allowed_import_roots=frozenset({"requests"}))
        scanner = CodeSecurityScanner(cfg)
        violations = scanner.scan("import requests")
        assert violations == []

    def test_allowed_import_roots_does_not_affect_other_blocks(self):
        cfg = SecurityConfig(allowed_import_roots=frozenset({"requests"}))
        scanner = CodeSecurityScanner(cfg)
        # subprocess is still blocked
        violations = scanner.scan("import subprocess")
        assert len(violations) == 1
        assert scanner.has_critical(violations)

    def test_extra_blocked_modules_stack_with_defaults(self):
        cfg = SecurityConfig(extra_blocked_modules=frozenset({"pickle"}))
        scanner = CodeSecurityScanner(cfg)
        # Both pickle (extra) and subprocess (default) should be blocked
        violations = scanner.scan("import pickle\nimport subprocess")
        assert len(violations) == 2
        categories = {v.category for v in violations}
        assert categories == {"dangerous_import"}


# ===================================================================
#  5. Severity override propagation
# ===================================================================


class TestSeverityOverrides:
    """severity_overrides flow from SecurityConfig through to violations."""

    def test_shell_access_downgraded_to_warning(self):
        cfg = SecurityConfig(severity_overrides={"shell_access": "warning"})
        scanner = CodeSecurityScanner(cfg)
        violations = scanner.scan('os.system("cmd")')
        assert len(violations) == 1
        assert violations[0].category == "shell_access"
        assert violations[0].severity == "warning"
        assert not scanner.has_critical(violations)

    def test_dangerous_builtin_downgraded_to_warning(self):
        cfg = SecurityConfig(severity_overrides={"dangerous_builtin": "warning"})
        scanner = CodeSecurityScanner(cfg)
        violations = scanner.scan('eval("x")')
        assert len(violations) == 1
        assert violations[0].severity == "warning"
        assert not scanner.has_critical(violations)

    def test_dangerous_import_downgraded_to_warning(self):
        cfg = SecurityConfig(severity_overrides={"dangerous_import": "warning"})
        scanner = CodeSecurityScanner(cfg)
        violations = scanner.scan("import subprocess")
        assert len(violations) == 1
        assert violations[0].severity == "warning"

    def test_override_does_not_affect_other_categories(self):
        cfg = SecurityConfig(severity_overrides={"shell_access": "warning"})
        scanner = CodeSecurityScanner(cfg)
        # dangerous_import is not overridden -- should stay critical
        violations = scanner.scan("import subprocess")
        assert violations[0].severity == "critical"

    def test_sandbox_escape_is_not_overridable_via_severity_overrides(self):
        # sandbox_escape violations are hardcoded to "critical" in _check_attribute
        # and _check_name; they do not consult severity_overrides.
        cfg = SecurityConfig(severity_overrides={"sandbox_escape": "warning"})
        scanner = CodeSecurityScanner(cfg)
        violations = scanner.scan("obj.__subclasses__()")
        escape_violations = [v for v in violations if v.category == "sandbox_escape"]
        assert len(escape_violations) >= 1
        assert all(v.severity == "critical" for v in escape_violations)


# ===================================================================
#  6. Safe imports pass through cleanly
# ===================================================================


class TestSafeCodePassesCleanly:
    """Safe bioinformatics code produces zero violations at any level."""

    @pytest.fixture(params=[SecurityLevel.STRICT, SecurityLevel.STANDARD, SecurityLevel.PERMISSIVE])
    def scanner(self, request) -> CodeSecurityScanner:
        return _scanner_for(request.param)

    def test_numpy_import(self, scanner: CodeSecurityScanner):
        assert scanner.scan("import numpy as np") == []

    def test_pandas_import(self, scanner: CodeSecurityScanner):
        assert scanner.scan("import pandas as pd") == []

    def test_scanpy_import(self, scanner: CodeSecurityScanner):
        assert scanner.scan("import scanpy as sc") == []

    def test_importlib_metadata(self, scanner: CodeSecurityScanner):
        assert scanner.scan("from importlib.metadata import version") == []

    def test_simple_math(self, scanner: CodeSecurityScanner):
        assert scanner.scan("x = 1 + 2") == []

    def test_function_definition(self, scanner: CodeSecurityScanner):
        code = "def greet(name):\n    return f'Hello, {name}!'"
        assert scanner.scan(code) == []

    def test_list_comprehension(self, scanner: CodeSecurityScanner):
        code = "squares = [x**2 for x in range(10)]"
        assert scanner.scan(code) == []

    def test_multiline_data_pipeline(self, scanner: CodeSecurityScanner):
        code = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "import scanpy as sc\n"
            "x = np.array([1, 2, 3])\n"
            "df = pd.DataFrame({'a': x})\n"
        )
        assert scanner.scan(code) == []


# ===================================================================
#  7. Dunder assignment detection
# ===================================================================


class TestDunderAssignmentDetection:
    """Dunder assignments produce warning-level violations."""

    def test_builtins_assign_produces_warning(self):
        scanner = CodeSecurityScanner()
        violations = scanner.scan("__builtins__ = {}")
        dunder_violations = [v for v in violations if v.category == "dunder_assign"]
        assert len(dunder_violations) == 1
        assert dunder_violations[0].severity == "warning"
        assert "__builtins__" in dunder_violations[0].description

    def test_builtins_assign_also_triggers_sandbox_escape(self):
        # __builtins__ is both in BLOCKED_NAMES (sandbox_escape, critical)
        # and detected as a dunder assignment (dunder_assign, warning).
        scanner = CodeSecurityScanner()
        violations = scanner.scan("__builtins__ = {}")
        categories = {v.category for v in violations}
        assert "sandbox_escape" in categories
        assert "dunder_assign" in categories

    def test_class_dunder_assign(self):
        scanner = CodeSecurityScanner()
        violations = scanner.scan("obj.__class__ = X")
        dunder_violations = [v for v in violations if v.category == "dunder_assign"]
        assert len(dunder_violations) == 1
        assert dunder_violations[0].severity == "warning"
        assert "__class__" in dunder_violations[0].description

    def test_regular_assignment_no_dunder_violation(self):
        scanner = CodeSecurityScanner()
        violations = scanner.scan("x = 42")
        dunder_violations = [v for v in violations if v.category == "dunder_assign"]
        assert dunder_violations == []


# ===================================================================
#  8. Multi-violation report formatting
# ===================================================================


class TestMultiViolationReport:
    """format_report produces correct human-readable output."""

    def test_no_violations_message(self):
        scanner = CodeSecurityScanner()
        report = scanner.format_report([])
        assert report == "No security issues detected."

    def test_single_violation_report(self):
        scanner = CodeSecurityScanner()
        violations = scanner.scan('os.system("cmd")')
        report = scanner.format_report(violations)
        assert "1 issue(s)" in report
        assert "[CRITICAL]" in report
        assert "shell_access" in report

    def test_multi_violation_report_structure(self):
        scanner = CodeSecurityScanner()
        code = "import subprocess\nos.system('ls')\neval('1+1')"
        violations = scanner.scan(code)
        report = scanner.format_report(violations)

        assert "3 issue(s)" in report

        lines = report.splitlines()
        # Header + 3 violation lines
        assert len(lines) == 4
        assert lines[0].startswith("Security scan found")

        # Each violation line should contain marker, line number, and category
        for line in lines[1:]:
            assert "[CRITICAL]" in line or "[WARNING]" in line
            assert "Line " in line

    def test_mixed_severity_report(self):
        # STANDARD: file_danger is warning, rest are critical
        scanner = _scanner_for(SecurityLevel.STANDARD)
        code = "os.system('ls')\nshutil.rmtree('/tmp')"
        violations = scanner.scan(code)
        report = scanner.format_report(violations)

        assert "[CRITICAL]" in report
        assert "[WARNING]" in report

    def test_report_lines_ordered_by_source_line(self):
        scanner = CodeSecurityScanner()
        code = "eval('a')\nos.system('b')\nimport subprocess"
        violations = scanner.scan(code)
        report = scanner.format_report(violations)
        lines = report.splitlines()[1:]  # skip header

        # Extract line numbers from report lines
        line_numbers = []
        for line in lines:
            # Format: "  [CRITICAL] Line N: ..."
            after_line = line.split("Line ")[1]
            num = int(after_line.split(":")[0])
            line_numbers.append(num)

        assert line_numbers == sorted(line_numbers)


# ===================================================================
#  9. Runtime import blocking (defense-in-depth beyond scanner)
# ===================================================================


class TestRuntimeImportBlocking:
    """Verify that _run_python_local blocks importlib/ctypes/cffi at runtime
    via the _safe_import hook, not relying solely on the AST scanner.

    The scanner also catches these imports, so to isolate the runtime layer
    we monkeypatch the scanner to report no critical violations, letting
    code reach execution where _safe_import enforces the block.
    """

    @pytest.fixture
    def local_backend(self, monkeypatch):
        from omicverse.utils.agent_backend import OmicVerseLLMBackend
        from omicverse.utils.model_config import ModelConfig
        monkeypatch.setattr(
            ModelConfig, "get_provider_from_model", lambda *a, **kw: "python"
        )
        return OmicVerseLLMBackend(
            system_prompt="test", model="python", api_key=""
        )

    @pytest.fixture
    def _bypass_scanner(self, monkeypatch):
        """Disable the AST scanner so code reaches the runtime import hook."""
        monkeypatch.setattr(CodeSecurityScanner, "scan", lambda self, code: [])
        monkeypatch.setattr(CodeSecurityScanner, "has_critical", lambda self, v: False)

    def test_import_importlib_blocked_at_runtime(
        self, local_backend, _bypass_scanner
    ):
        """import importlib must be blocked by limited_import, not just scanner."""
        with pytest.raises(RuntimeError, match="(?i)importlib.*blocked"):
            local_backend._run_python_local("import importlib")

    def test_importlib_import_module_blocked(
        self, local_backend, _bypass_scanner
    ):
        """importlib.import_module escape path is blocked at runtime."""
        with pytest.raises(RuntimeError, match="(?i)importlib.*blocked"):
            local_backend._run_python_local(
                "import importlib\nimportlib.import_module('subprocess')"
            )

    def test_from_importlib_import_blocked(
        self, local_backend, _bypass_scanner
    ):
        """from importlib import import_module is blocked at runtime."""
        with pytest.raises(RuntimeError, match="(?i)importlib.*blocked"):
            local_backend._run_python_local(
                "from importlib import import_module"
            )

    def test_import_ctypes_blocked_at_runtime(
        self, local_backend, _bypass_scanner
    ):
        """import ctypes must be blocked at runtime."""
        with pytest.raises(RuntimeError, match="(?i)ctypes.*blocked"):
            local_backend._run_python_local("import ctypes")

    def test_import_cffi_blocked_at_runtime(
        self, local_backend, _bypass_scanner
    ):
        """import cffi must be blocked at runtime."""
        with pytest.raises(RuntimeError, match="(?i)cffi.*blocked"):
            local_backend._run_python_local("import cffi")

    def test_import_subprocess_blocked_at_runtime(
        self, local_backend, _bypass_scanner
    ):
        """import subprocess must be blocked at runtime, not just scanner."""
        with pytest.raises(RuntimeError, match="(?i)subprocess.*blocked"):
            local_backend._run_python_local("import subprocess")

    def test_import_sys_blocked_at_runtime(
        self, local_backend, _bypass_scanner
    ):
        """import sys must be blocked to prevent sys.path/sys.modules access."""
        with pytest.raises(RuntimeError, match="(?i)sys.*blocked"):
            local_backend._run_python_local("import sys")

    def test_safe_imports_still_work(
        self, local_backend, _bypass_scanner
    ):
        """Non-blocked imports (json, etc.) still function with scanner bypassed."""
        result = local_backend._run_python_local(
            "import json\nprint(json.dumps({'ok': True}))"
        )
        assert '"ok": true' in result


# ===================================================================
# 10. Runtime bypass vector enforcement (build_sandbox_globals)
# ===================================================================

from omicverse.utils.agent_sandbox import build_sandbox_globals


class TestRuntimeGetAttrBypass:
    """getattr(os, 'system') and similar dynamic attribute access must be
    blocked at runtime by SafeOsProxy, not only by the AST scanner."""

    def test_getattr_os_system_blocked(self):
        globs = build_sandbox_globals()
        with pytest.raises(SecurityViolationError, match="os.system"):
            exec("result = getattr(os, 'system')", globs)

    def test_getattr_os_popen_blocked(self):
        globs = build_sandbox_globals()
        with pytest.raises(SecurityViolationError, match="os.popen"):
            exec("result = getattr(os, 'popen')", globs)

    def test_getattr_os_execv_blocked(self):
        globs = build_sandbox_globals()
        with pytest.raises(SecurityViolationError, match="os.execv"):
            exec("getattr(os, 'execv')", globs)

    def test_getattr_os_kill_blocked(self):
        globs = build_sandbox_globals()
        with pytest.raises(SecurityViolationError, match="os.kill"):
            exec("getattr(os, 'kill')", globs)

    def test_getattr_os_path_allowed(self):
        """os.path is safe and must remain accessible."""
        globs = build_sandbox_globals()
        loc = {}
        exec("result = getattr(os, 'path')", globs, loc)
        assert loc["result"] is os.path

    def test_os_getcwd_allowed(self):
        globs = build_sandbox_globals()
        loc = {}
        exec("result = os.getcwd()", globs, loc)
        assert loc["result"] == os.getcwd()


class TestRuntimeDunderImportBypass:
    """Direct __import__ usage must go through restricted limited_import."""

    def test_dunder_import_subprocess_blocked(self):
        globs = build_sandbox_globals()
        with pytest.raises(ImportError, match="subprocess.*blocked"):
            exec("mod = __import__('subprocess')", globs)

    def test_dunder_import_socket_blocked(self):
        globs = build_sandbox_globals()
        with pytest.raises(ImportError, match="socket.*blocked"):
            exec("mod = __import__('socket')", globs)

    def test_dunder_import_sys_blocked(self):
        globs = build_sandbox_globals()
        with pytest.raises(ImportError, match="sys.*blocked"):
            exec("mod = __import__('sys')", globs)

    def test_dunder_import_os_returns_proxy(self):
        """__import__('os') must return SafeOsProxy, not real os."""
        from omicverse.utils.agent_sandbox import SafeOsProxy
        globs = build_sandbox_globals()
        loc = {}
        exec("mod = __import__('os')", globs, loc)
        assert isinstance(loc["mod"], SafeOsProxy)

    def test_dunder_import_json_allowed(self):
        globs = build_sandbox_globals()
        loc = {}
        exec("mod = __import__('json')", globs, loc)
        import json
        assert loc["mod"] is json

    def test_import_importlib_metadata_allowed(self):
        """Safe importlib sub-modules must remain importable."""
        globs = build_sandbox_globals()
        loc = {}
        exec("from importlib import metadata\nresult = hasattr(metadata, 'version')", globs, loc)
        assert loc["result"] is True


class TestRuntimeSysPathManipulation:
    """sys.path and sys.modules manipulation must be blocked."""

    def test_import_sys_blocked(self):
        globs = build_sandbox_globals()
        with pytest.raises(ImportError, match="sys.*blocked"):
            exec("import sys", globs)

    def test_from_sys_import_blocked(self):
        globs = build_sandbox_globals()
        with pytest.raises(ImportError, match="sys.*blocked"):
            exec("from sys import path", globs)

    def test_sys_modules_inaccessible(self):
        """Without sys, sys.modules cannot be used to retrieve blocked modules."""
        globs = build_sandbox_globals()
        with pytest.raises(ImportError, match="sys.*blocked"):
            exec("import sys; mod = sys.modules['os']", globs)


class TestRuntimeIntrospectionEscapes:
    """Dangerous builtins (eval, exec, compile, globals, locals) must not
    be available in the sandbox namespace."""

    def test_eval_not_available(self):
        globs = build_sandbox_globals()
        with pytest.raises(NameError):
            exec("eval('1+1')", globs)

    def test_exec_not_available(self):
        globs = build_sandbox_globals()
        with pytest.raises(NameError):
            exec("exec('x = 1')", globs)

    def test_compile_not_available(self):
        globs = build_sandbox_globals()
        with pytest.raises(NameError):
            exec("compile('1+1', '<test>', 'exec')", globs)

    def test_globals_not_available(self):
        globs = build_sandbox_globals()
        with pytest.raises(NameError):
            exec("g = globals()", globs)

    def test_locals_not_available(self):
        globs = build_sandbox_globals()
        with pytest.raises(NameError):
            exec("l = locals()", globs)

    def test_breakpoint_not_available(self):
        globs = build_sandbox_globals()
        with pytest.raises(NameError):
            exec("breakpoint()", globs)

    def test_standard_builtins_still_work(self):
        """Common builtins like len, range, print must still be available."""
        globs = build_sandbox_globals()
        loc = {}
        exec("result = len(list(range(5)))", globs, loc)
        assert loc["result"] == 5

    def test_import_os_then_getattr_system_blocked(self):
        """Full chain: import os; getattr(os, 'system') must be caught."""
        globs = build_sandbox_globals()
        with pytest.raises(SecurityViolationError, match="os.system"):
            exec("import os\ngetattr(os, 'system')", globs)

    def test_restricted_import_still_blocks_after_getattr(self):
        """The restricted __import__ obtained via builtins still enforces blocks."""
        globs = build_sandbox_globals()
        with pytest.raises(ImportError, match="blocked"):
            exec(
                "imp = __builtins__['__import__']\n"
                "imp('subprocess')",
                globs,
            )


class TestRuntimeLegitimateCodeWorks:
    """Verify that legitimate data-science code runs successfully in the
    sandboxed environment."""

    def test_math_operations(self):
        globs = build_sandbox_globals()
        loc = {}
        exec("import math\nresult = math.sqrt(144)", globs, loc)
        assert loc["result"] == 12.0

    def test_json_roundtrip(self):
        globs = build_sandbox_globals()
        loc = {}
        exec(
            "import json\n"
            "data = {'key': [1, 2, 3]}\n"
            "result = json.loads(json.dumps(data))",
            globs, loc,
        )
        assert loc["result"] == {"key": [1, 2, 3]}

    def test_os_path_join(self):
        globs = build_sandbox_globals()
        loc = {}
        exec("result = os.path.join('a', 'b', 'c')", globs, loc)
        assert loc["result"] == os.path.join("a", "b", "c")

    def test_from_os_path_import_join(self):
        globs = build_sandbox_globals()
        loc = {}
        exec("from os.path import join\nresult = join('x', 'y')", globs, loc)
        assert loc["result"] == os.path.join("x", "y")

    def test_list_comprehension_and_builtins(self):
        globs = build_sandbox_globals()
        loc = {}
        exec(
            "result = sorted([x**2 for x in range(5)], reverse=True)",
            globs, loc,
        )
        assert loc["result"] == [16, 9, 4, 1, 0]

    def test_exception_handling(self):
        globs = build_sandbox_globals()
        loc = {}
        exec(
            "try:\n"
            "    1 / 0\n"
            "except ZeroDivisionError:\n"
            "    result = 'caught'\n",
            globs, loc,
        )
        assert loc["result"] == "caught"


# ===================================================================
# 11. pandas.eval scanner classification correction
# ===================================================================


class TestPandasEvalClassification:
    """pandas.eval / pd.eval must be categorized as code_execution,
    not file_danger, and must have critical severity."""

    def test_pd_eval_detected_as_code_execution(self):
        scanner = CodeSecurityScanner()
        violations = scanner.scan('pd.eval("x + 1")')
        eval_v = [v for v in violations if "pd.eval" in v.description]
        assert len(eval_v) == 1
        assert eval_v[0].category == "code_execution"
        assert eval_v[0].severity == "critical"

    def test_pandas_eval_detected_as_code_execution(self):
        scanner = CodeSecurityScanner()
        violations = scanner.scan('pandas.eval("x + 1")')
        eval_v = [v for v in violations if "pandas.eval" in v.description]
        assert len(eval_v) == 1
        assert eval_v[0].category == "code_execution"
        assert eval_v[0].severity == "critical"

    def test_pd_eval_not_file_danger(self):
        scanner = CodeSecurityScanner()
        violations = scanner.scan('pd.eval("something")')
        for v in violations:
            if "eval" in v.description.lower():
                assert v.category != "file_danger", (
                    f"pandas.eval must not be categorized as file_danger, "
                    f"got: {v.category}"
                )

    def test_pd_eval_blocked_at_standard_level(self):
        scanner = _scanner_for(SecurityLevel.STANDARD)
        violations = scanner.scan('pd.eval("1+1")')
        assert scanner.has_critical(violations)

    def test_code_execution_severity_overridable(self):
        cfg = SecurityConfig(severity_overrides={"code_execution": "warning"})
        scanner = CodeSecurityScanner(cfg)
        violations = scanner.scan('pd.eval("x")')
        eval_v = [v for v in violations if v.category == "code_execution"]
        assert len(eval_v) == 1
        assert eval_v[0].severity == "warning"

    def test_cffi_in_scanner_blocked_imports(self):
        """cffi must be in the scanner's BLOCKED_IMPORT_ROOTS."""
        scanner = CodeSecurityScanner()
        violations = scanner.scan("import cffi")
        assert len(violations) == 1
        assert violations[0].category == "dangerous_import"
        assert violations[0].severity == "critical"
