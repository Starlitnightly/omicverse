"""
Pre-execution security analysis and runtime sandbox hardening for OmicVerse Agent.

Inspired by OpenCode's two-layer defense pattern (AST analysis + runtime restrictions),
adapted for bioinformatics use cases where file I/O and data-science libraries are
essential but shell access and system manipulation are not.

This module provides:
- ``CodeSecurityScanner``: AST-based pre-execution analysis that detects dangerous
  patterns *before* any code runs.
- ``SafeOsProxy``: A drop-in replacement for the ``os`` module that blocks dangerous
  operations while allowing path manipulation and directory reading.
- ``SecurityConfig`` / ``ApprovalMode``: Configurable security policy.

.. note::
    These are defense-in-depth / UX measures, not a true security boundary.
    For untrusted input, use OS-level isolation (containers, VMs).
"""

from __future__ import annotations

import ast
import builtins as _builtins
import logging
import os
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ApprovalMode(Enum):
    """Controls when the user is prompted to approve code before execution."""
    NEVER = "never"                # Execute immediately (current default behavior)
    ALWAYS = "always"              # Always show code and ask before execution
    ON_VIOLATION = "on_violation"  # Ask only when scanner finds issues


class SecurityLevel(Enum):
    """Pre-defined security level presets.

    STRICT   — Full sandbox; all blocks active. Best for untrusted code.
    STANDARD — Relaxed: file ops are warnings, dynamic imports blocked.
    PERMISSIVE — Trust mode: minimal restrictions, most operations allowed.
    """
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"


@dataclass
class SecurityConfig:
    """Configurable security policy for the agent sandbox.

    Parameters
    ----------
    level : SecurityLevel
        Pre-defined security level preset. Overrides individual settings
        when set via ``from_level()``.
    approval_mode : ApprovalMode
        When to prompt the user for approval before executing generated code.
    extra_blocked_modules : frozenset[str]
        Additional module root names to block beyond the built-in deny list.
    extra_blocked_calls : frozenset[str]
        Additional dotted callable names to flag in the AST scanner.
    allow_dynamic_imports : bool
        Whether unknown modules can be imported at runtime. When *False*
        (default), only pre-approved modules are loadable.
    dynamic_import_allowlist : frozenset[str]
        Module root names that may be dynamically imported even when
        ``allow_dynamic_imports`` is False.
    restrict_introspection : bool
        Remove ``globals()`` and ``locals()`` from the sandbox builtins.
    severity_overrides : dict
        Per-category severity overrides. Maps category name (e.g. "file_danger",
        "dangerous_import") to severity ("critical" or "warning").
    allowed_import_roots : frozenset[str]
        Module roots to REMOVE from the blocked list. Used by permissive mode
        to allow e.g. ``requests``.
    """
    level: SecurityLevel = SecurityLevel.STANDARD
    approval_mode: ApprovalMode = ApprovalMode.NEVER
    extra_blocked_modules: FrozenSet[str] = field(default_factory=frozenset)
    extra_blocked_calls: FrozenSet[str] = field(default_factory=frozenset)
    allow_dynamic_imports: bool = False
    dynamic_import_allowlist: FrozenSet[str] = field(default_factory=frozenset)
    restrict_introspection: bool = True
    severity_overrides: dict = field(default_factory=dict)
    allowed_import_roots: FrozenSet[str] = field(default_factory=frozenset)

    @classmethod
    def from_level(cls, level: SecurityLevel) -> "SecurityConfig":
        """Create a SecurityConfig from a preset security level."""
        if isinstance(level, str):
            level = SecurityLevel(level)

        if level == SecurityLevel.STRICT:
            return cls(
                level=level,
                approval_mode=ApprovalMode.ON_VIOLATION,
                allow_dynamic_imports=False,
                restrict_introspection=True,
                severity_overrides={},
                allowed_import_roots=frozenset(),
            )
        elif level == SecurityLevel.STANDARD:
            return cls(
                level=level,
                approval_mode=ApprovalMode.NEVER,
                allow_dynamic_imports=False,
                restrict_introspection=True,
                severity_overrides={
                    "file_danger": "warning",
                },
                allowed_import_roots=frozenset(),
            )
        elif level == SecurityLevel.PERMISSIVE:
            return cls(
                level=level,
                approval_mode=ApprovalMode.NEVER,
                allow_dynamic_imports=True,
                restrict_introspection=False,
                severity_overrides={
                    "file_danger": "warning",
                    "dangerous_import": "warning",
                    "dangerous_builtin": "warning",
                },
                allowed_import_roots=frozenset({"requests", "urllib", "http"}),
            )
        else:
            return cls(level=level)


# ---------------------------------------------------------------------------
# Security violation dataclass
# ---------------------------------------------------------------------------

@dataclass
class SecurityViolation:
    """A single security issue found during AST scanning."""
    category: str       # e.g. "shell_access", "sandbox_escape", "dangerous_builtin"
    description: str    # Human-readable explanation
    line: int           # Source line number (1-based)
    node_type: str      # AST node class name
    severity: str       # "critical" | "warning"


# ---------------------------------------------------------------------------
# AST-based pre-execution security scanner
# ---------------------------------------------------------------------------

class CodeSecurityScanner:
    """Walk the parsed AST and detect dangerous patterns before execution.

    This is a UX / defense-in-depth measure, not a true security boundary.
    Critical violations block execution; warnings are logged but allowed.

    Examples
    --------
    >>> scanner = CodeSecurityScanner()
    >>> violations = scanner.scan("import subprocess; subprocess.run(['ls'])")
    >>> scanner.has_critical(violations)
    True
    """

    # Dangerous callable patterns (dotted attribute chains)
    BLOCKED_CALLS: FrozenSet[str] = frozenset({
        # Shell access via os
        "os.system", "os.popen",
        "os.execl", "os.execle", "os.execlp", "os.execlpe",
        "os.execv", "os.execve", "os.execvp", "os.execvpe",
        "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
        "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
        "os.fork", "os.forkpty",
        "os.kill", "os.killpg",
        "os.popen2", "os.popen3", "os.popen4",
        # Dangerous file operations
        "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
        "os.rename", "os.renames", "os.replace",
        "os.link", "os.symlink",
        "os.chown", "os.chmod", "os.chroot", "os.lchmod", "os.lchown",
        # subprocess (backup in case it gets through module block)
        "subprocess.call", "subprocess.run", "subprocess.Popen",
        "subprocess.check_call", "subprocess.check_output",
        "subprocess.getoutput", "subprocess.getstatusoutput",
        # shutil dangerous ops
        "shutil.rmtree", "shutil.move",
        # Code execution via data-science libraries
        "pandas.eval", "pd.eval",
    })

    # Dangerous attribute access (sandbox escape vectors)
    BLOCKED_ATTRS: FrozenSet[str] = frozenset({
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__code__",
        "__closure__",
        "__func__",
    })

    # Dangerous bare name access
    BLOCKED_NAMES: FrozenSet[str] = frozenset({
        "__builtins__",
        "__loader__",
        "__spec__",
    })

    # Calls that represent code-execution risk (not file I/O or shell access)
    CODE_EXECUTION_CALLS: FrozenSet[str] = frozenset({
        "pandas.eval", "pd.eval",
    })

    # Modules that should never appear in import statements
    BLOCKED_IMPORT_ROOTS: FrozenSet[str] = frozenset({
        "subprocess", "socket", "ssl", "urllib", "http",
        "ftplib", "smtplib", "telnetlib", "paramiko", "requests",
        "importlib", "ctypes", "cffi", "multiprocessing",
    })

    SAFE_IMPORT_SUBMODULES: FrozenSet[str] = frozenset({
        "importlib.metadata",
        "importlib.resources",
    })

    def __init__(self, config: Optional[SecurityConfig] = None) -> None:
        self.config = config or SecurityConfig()
        self._blocked_calls = self.BLOCKED_CALLS | self.config.extra_blocked_calls
        self._blocked_import_roots = (
            (self.BLOCKED_IMPORT_ROOTS | self.config.extra_blocked_modules)
            - self.config.allowed_import_roots
        )
        self._severity_overrides = dict(self.config.severity_overrides)

    # ---- public API --------------------------------------------------------

    def scan(self, code: str) -> List[SecurityViolation]:
        """Parse *code* and return all security violations found.

        Returns an empty list when no issues are detected.
        Raises ``SyntaxError`` if *code* cannot be parsed.
        """
        tree = ast.parse(code)
        violations: List[SecurityViolation] = []
        for node in ast.walk(tree):
            violations.extend(self._check_call(node))
            violations.extend(self._check_attribute(node))
            violations.extend(self._check_name(node))
            violations.extend(self._check_import(node))
            violations.extend(self._check_dunder_assign(node))
        violations.sort(key=lambda v: v.line)
        return violations

    def has_critical(self, violations: List[SecurityViolation]) -> bool:
        """Return *True* if any violation has severity ``'critical'``."""
        return any(v.severity == "critical" for v in violations)

    def format_report(self, violations: List[SecurityViolation]) -> str:
        """Format violations into a human-readable report string."""
        if not violations:
            return "No security issues detected."
        lines = [f"Security scan found {len(violations)} issue(s):"]
        for v in violations:
            marker = "CRITICAL" if v.severity == "critical" else "WARNING"
            lines.append(
                f"  [{marker}] Line {v.line}: {v.description} ({v.category})"
            )
        return "\n".join(lines)

    # ---- private checks ----------------------------------------------------

    @staticmethod
    def _resolve_call_name(node: ast.expr) -> Optional[str]:
        """Resolve a call target to a dotted string like ``os.system``."""
        parts: list[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None

    def _check_call(self, node: ast.AST) -> List[SecurityViolation]:
        if not isinstance(node, ast.Call):
            return []
        violations: List[SecurityViolation] = []
        line = getattr(node, "lineno", 0)

        # Resolve dotted call name (e.g. os.system)
        name = self._resolve_call_name(node.func)
        if name:
            # Check exact match or prefix match (os.exec covers os.execl etc.)
            for blocked in self._blocked_calls:
                if name == blocked or name.startswith(blocked + "."):
                    if blocked in self.CODE_EXECUTION_CALLS:
                        category = "code_execution"
                    elif "os." in blocked or "subprocess" in blocked:
                        category = "shell_access"
                    else:
                        category = "file_danger"
                    severity = self._severity_overrides.get(category, "critical")
                    violations.append(SecurityViolation(
                        category=category,
                        description=f"Blocked call: {name}()",
                        line=line,
                        node_type="Call",
                        severity=severity,
                    ))
                    break

            # Check dangerous builtins used as bare calls
            if name in ("eval", "exec", "compile", "__import__"):
                severity = self._severity_overrides.get("dangerous_builtin", "critical")
                violations.append(SecurityViolation(
                    category="dangerous_builtin",
                    description=f"Dangerous builtin call: {name}()",
                    line=line,
                    node_type="Call",
                    severity=severity,
                ))

        return violations

    def _check_attribute(self, node: ast.AST) -> List[SecurityViolation]:
        if not isinstance(node, ast.Attribute):
            return []
        violations: List[SecurityViolation] = []
        line = getattr(node, "lineno", 0)

        if node.attr in self.BLOCKED_ATTRS:
            violations.append(SecurityViolation(
                category="sandbox_escape",
                description=f"Dangerous attribute access: .{node.attr}",
                line=line,
                node_type="Attribute",
                severity="critical",
            ))

        return violations

    def _check_name(self, node: ast.AST) -> List[SecurityViolation]:
        if not isinstance(node, ast.Name):
            return []
        violations: List[SecurityViolation] = []
        line = getattr(node, "lineno", 0)

        if node.id in self.BLOCKED_NAMES:
            violations.append(SecurityViolation(
                category="sandbox_escape",
                description=f"Dangerous name access: {node.id}",
                line=line,
                node_type="Name",
                severity="critical",
            ))

        return violations

    def _check_import(self, node: ast.AST) -> List[SecurityViolation]:
        violations: List[SecurityViolation] = []
        line = getattr(node, "lineno", 0)

        severity = self._severity_overrides.get("dangerous_import", "critical")

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in self.SAFE_IMPORT_SUBMODULES:
                    continue
                root = alias.name.split(".")[0]
                if root in self._blocked_import_roots:
                    violations.append(SecurityViolation(
                        category="dangerous_import",
                        description=f"Blocked module import: {alias.name}",
                        line=line,
                        node_type="Import",
                        severity=severity,
                    ))

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.module in self.SAFE_IMPORT_SUBMODULES:
                    return violations
                if node.module == "importlib":
                    imported_names = {alias.name for alias in node.names}
                    if imported_names and imported_names <= {"metadata", "resources"}:
                        return violations
                root = node.module.split(".")[0]
                if root in self._blocked_import_roots:
                    violations.append(SecurityViolation(
                        category="dangerous_import",
                        description=f"Blocked module import: from {node.module}",
                        line=line,
                        node_type="ImportFrom",
                        severity=severity,
                    ))

        return violations

    def _check_dunder_assign(self, node: ast.AST) -> List[SecurityViolation]:
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            return []
        violations: List[SecurityViolation] = []
        line = getattr(node, "lineno", 0)

        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            name = None
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            if name and name.startswith("__") and name.endswith("__"):
                violations.append(SecurityViolation(
                    category="dunder_assign",
                    description=f"Assignment to dunder name: {name}",
                    line=line,
                    node_type=type(node).__name__,
                    severity="warning",
                ))

        return violations


# ---------------------------------------------------------------------------
# Safe os module proxy
# ---------------------------------------------------------------------------

# Module-level reference to the real os module.  Intentionally NOT stored on
# SafeOsProxy instances so that proxy.__dict__ never exposes a path back to
# unrestricted os behaviour.
_REAL_OS_MODULE = os


class SafeOsProxy(types.ModuleType):
    """Drop-in proxy for the ``os`` module that blocks dangerous operations.

    Allowed operations include path manipulation, directory reading/creation,
    platform constants, and read-only system info — everything needed for
    bioinformatics data I/O.

    Blocked operations include shell execution, process management, file
    deletion, permission changes, and environment mutation.

    The proxy uses an **explicit allowlist** model: only attributes listed in
    ``_SAFE_ATTRS`` are forwarded to the real ``os`` module.  Everything else
    — including unknown future ``os`` additions — is denied by default.
    """

    _BLOCKED_ATTRS: FrozenSet[str] = frozenset({
        # Shell execution
        "system", "popen",
        "execl", "execle", "execlp", "execlpe",
        "execv", "execve", "execvp", "execvpe",
        "spawnl", "spawnle", "spawnlp", "spawnlpe",
        "spawnv", "spawnve", "spawnvp", "spawnvpe",
        "popen2", "popen3", "popen4",
        # Process management
        "fork", "forkpty",
        "kill", "killpg",
        # File deletion / mutation
        "remove", "unlink", "rmdir", "removedirs",
        "rename", "renames", "replace",
        "link", "symlink",
        # Permission changes
        "chmod", "chown", "chroot", "lchmod", "lchown",
        # Environment mutation
        "putenv", "unsetenv",
    })

    _SAFE_ATTRS: FrozenSet[str] = frozenset({
        # Directory / file-status operations (read-only or create-only)
        "getcwd", "listdir", "scandir", "walk",
        "makedirs", "mkdir",
        "stat", "lstat", "access",
        "readlink",
        # Platform constants
        "sep", "altsep", "extsep", "linesep", "pathsep",
        "curdir", "pardir", "devnull", "name",
        # Environment reading (mutation stays blocked above)
        "environ", "getenv", "environb", "getenvb",
        # System info (read-only)
        "cpu_count", "getpid", "getppid", "uname", "getlogin",
        # Path encoding helpers
        "fspath", "fsencode", "fsdecode",
        # Misc safe
        "urandom", "strerror", "get_terminal_size",
    })

    def __init__(self) -> None:
        super().__init__("os")
        # Expose os.path directly — it's entirely safe.
        # NOTE: the real os module is NOT stored on the instance; see
        # _REAL_OS_MODULE at module scope.
        self.__dict__["path"] = os.path

    def __getattr__(self, name: str) -> Any:
        # Blocked attrs get an explicit, descriptive error.
        if name in self._BLOCKED_ATTRS:
            from .agent_errors import SecurityViolationError
            raise SecurityViolationError(
                f"os.{name}() is blocked in the OmicVerse agent sandbox. "
                f"This operation is not needed for bioinformatics analysis."
            )
        # Only explicitly safe attrs are forwarded to the real os module.
        if name in self._SAFE_ATTRS:
            return getattr(_REAL_OS_MODULE, name)
        raise AttributeError(
            f"module 'os' has no attribute '{name}' in the OmicVerse agent sandbox"
        )

    def __dir__(self) -> List[str]:
        return sorted(set(self._SAFE_ATTRS) | {"path"})

    def __repr__(self) -> str:
        return "<SafeOsProxy: restricted os module for OmicVerse agent>"


# ---------------------------------------------------------------------------
# Runtime sandbox globals builder
# ---------------------------------------------------------------------------

# Modules blocked at runtime import time — superset of the scanner's
# BLOCKED_IMPORT_ROOTS, extended with sys (path/module manipulation)
# and cffi (foreign-function interface escape).
BLOCKED_RUNTIME_IMPORT_ROOTS: FrozenSet[str] = frozenset({
    "subprocess", "socket", "ssl", "urllib", "http",
    "ftplib", "smtplib", "telnetlib", "paramiko", "requests",
    "importlib", "ctypes", "cffi", "multiprocessing", "sys",
})

_SAFE_IMPORT_SUBMODULES: FrozenSet[str] = frozenset({
    "importlib.metadata",
    "importlib.resources",
})

# Builtins explicitly excluded from the runtime sandbox.
_EXCLUDED_BUILTINS: FrozenSet[str] = frozenset({
    "eval", "exec", "compile",   # arbitrary code execution
    "globals", "locals",          # namespace introspection
    "breakpoint",                 # debugger entry
    "exit", "quit",               # interpreter termination
    "__import__",                 # replaced with restricted version
})


def build_sandbox_globals(
    *,
    security_config: Optional[SecurityConfig] = None,
) -> Dict[str, Any]:
    """Build restricted globals dict for local Python ``exec()``.

    Returns a namespace with:

    * Allowlisted builtins (no ``eval``/``exec``/``compile``/``globals``/``locals``)
    * A restricted ``__import__`` that blocks dangerous modules at runtime
    * ``os`` replaced with :class:`SafeOsProxy`

    This is the runtime enforcement counterpart of :class:`CodeSecurityScanner`.
    The scanner catches patterns at AST level; these globals catch bypass
    vectors that only manifest at execution time (e.g. ``getattr(os, 'system')``
    or ``__import__('subprocess')``).
    """
    config = security_config or SecurityConfig()

    # -- restricted builtins ------------------------------------------------
    excluded = set(_EXCLUDED_BUILTINS)
    if not config.restrict_introspection:
        excluded -= {"globals", "locals"}

    safe_builtins: Dict[str, Any] = {}
    for name in dir(_builtins):
        if name.startswith("_"):
            continue
        if name in excluded:
            continue
        obj = getattr(_builtins, name, None)
        if obj is not None:
            safe_builtins[name] = obj

    # -- restricted import --------------------------------------------------
    deny_roots = BLOCKED_RUNTIME_IMPORT_ROOTS | config.extra_blocked_modules
    os_proxy = SafeOsProxy()

    def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        fromlist_names = {str(f) for f in (fromlist or ())}
        allow_safe = (
            name in _SAFE_IMPORT_SUBMODULES
            or any(name.startswith(f"{sm}.") for sm in _SAFE_IMPORT_SUBMODULES)
            or (name == "importlib" and fromlist_names
                and fromlist_names <= {"metadata", "resources"})
        )
        if root in deny_roots and not allow_safe:
            raise ImportError(
                f"Module '{name}' is blocked inside the OmicVerse agent sandbox."
            )
        # Redirect os imports to SafeOsProxy so getattr(os, 'system') is
        # caught at runtime rather than returning the real os attribute.
        if root == "os":
            if fromlist and name != "os":
                # e.g. from os.path import join — return real submodule
                return _real_import(name, globals, locals, fromlist, level)
            # import os / import os.path / from os import X — proxy
            _real_import(name, globals, locals, fromlist, level)
            return os_proxy
        return _real_import(name, globals, locals, fromlist, level)

    _real_import = __import__
    safe_builtins["__import__"] = limited_import

    # -- assemble globals ---------------------------------------------------
    sandbox_globals: Dict[str, Any] = {
        "__name__": "__main__",
        "__builtins__": safe_builtins,
        "os": os_proxy,
    }
    return sandbox_globals
