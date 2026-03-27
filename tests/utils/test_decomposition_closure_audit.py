"""Decomposition closure audit for OVAgent oversized-file wave.

Verifies that:
1. turn_controller.py, tool_runtime.py, analysis_executor.py, and
   agent_backend.py are materially smaller than their pre-decomposition
   baselines.
2. Public facade imports remain intact.
3. Extracted helper modules exist and are properly wired.
4. No unexpected dependency additions.
5. Maintainability guardrails prevent regression back to monolith sizes.

These tests run under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""

import ast
import importlib
import importlib.machinery
import os
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
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
    reason="Closure audit tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs to avoid heavy omicverse.__init__
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"
OVAGENT_DIR = PACKAGE_ROOT / "utils" / "ovagent"
BACKEND_DIR = PACKAGE_ROOT / "utils"

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

import omicverse.utils.ovagent as ovagent_pkg  # noqa: E402
from omicverse.utils.ovagent import (  # noqa: E402
    AnalysisExecutor,
    ConvergenceMonitor,
    FollowUpGate,
    ProactiveCodeTransformer,
    PromptBuilder,
    SubagentController,
    ToolRuntime,
    TurnController,
)
from omicverse.utils.ovagent.tool_runtime import LEGACY_AGENT_TOOLS  # noqa: E402

# Import backend facade
from omicverse.utils.agent_backend import OmicVerseLLMBackend  # noqa: E402
from omicverse.utils.agent_backend_common import (  # noqa: E402
    BackendConfig,
    ChatResponse,
    ToolCall,
    Usage,
)

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod

# ---------------------------------------------------------------------------
# Minimal context double
# ---------------------------------------------------------------------------

_STUB_METHODS = [
    "_emit", "_get_harness_session_id", "_get_runtime_session_id",
    "_get_visible_agent_tools", "_get_loaded_tool_names",
    "_refresh_runtime_working_directory", "_tool_blocked_in_plan_mode",
    "_detect_repo_root", "_resolve_local_path", "_ensure_server_tool_mode",
    "_request_interaction", "_request_tool_approval", "_load_skill_guidance",
    "_extract_python_code", "_extract_python_code_strict",
    "_gather_code_candidates", "_normalize_code_candidate",
    "_collect_static_registry_entries", "_collect_runtime_registry_entries",
    "_review_generated_code_lightweight", "_contains_forbidden_scanpy_usage",
    "_rewrite_scanpy_calls_with_registry",
    "_normalize_registry_entry_for_codegen", "_build_agentic_system_prompt",
]


class _MinimalCtx:
    LEGACY_AGENT_TOOLS = LEGACY_AGENT_TOOLS

    def __init__(self):
        self.model = "test-model"
        self.provider = "openai"
        self.endpoint = "https://api.example.com"
        self.api_key = None
        self._llm = None
        self._config = SimpleNamespace()
        self._security_config = SimpleNamespace()
        self._security_scanner = SimpleNamespace()
        self._filesystem_context = None
        self.skill_registry = None
        self._notebook_executor = None
        self._ov_runtime = None
        self._trace_store = None
        self._session_history = None
        self._context_compactor = None
        self._approval_handler = None
        self._reporter = MagicMock()
        self.last_usage = None
        self.last_usage_breakdown = {}
        self._last_run_trace = None
        self._active_run_id = ""
        self._web_session_id = "test-closure-session"
        self._managed_api_env = {}
        self._code_only_mode = False
        self._code_only_captured_code = ""
        self._code_only_captured_history = []
        self.use_notebook_execution = False
        self.enable_filesystem_context = False
        for name in _STUB_METHODS:
            if not hasattr(self, name):
                setattr(self, name, lambda *a, _n=name, **kw: (
                    [] if "collect" in _n or "gather" in _n or "visible" in _n or "loaded" in _n
                    else "" if "build" in _n or "extract" in _n or "load" in _n or "rewrite" in _n
                    else None
                ))

    def _get_harness_session_id(self):
        return self._web_session_id

    def _get_runtime_session_id(self):
        return self._web_session_id or "default"

    def _get_visible_agent_tools(self, *, allowed_names=None):
        return []

    def _get_loaded_tool_names(self):
        return []

    async def _run_agentic_loop(self, request, adata, event_callback=None,
                                cancel_event=None, history=None,
                                approval_handler=None, request_content=None):
        return adata


class _DummyExecutor:
    pass


# ===================================================================
# 1. Facade size reduction audit
# ===================================================================

# Pre-decomposition baselines (from the context manifest)
_BASELINES = {
    "turn_controller.py": {"lines": 1862, "bytes": 74754},
    "tool_runtime.py": {"lines": 1791, "bytes": 66789},
    "analysis_executor.py": {"lines": 973, "bytes": 42584},
    "agent_backend.py": {"lines": 3128, "bytes": 132380},
}

# Minimum required reduction (fraction) to count as "materially smaller"
_MIN_LINE_REDUCTION = 0.15  # at least 15% line reduction


class TestFacadeSizeReduction:
    """Each facade file must be materially smaller than its baseline."""

    @pytest.mark.parametrize("filename,baseline", list(_BASELINES.items()))
    def test_facade_line_count_reduced(self, filename, baseline):
        if filename == "agent_backend.py":
            path = BACKEND_DIR / filename
        else:
            path = OVAGENT_DIR / filename
        assert path.exists(), f"{filename} does not exist"
        current_lines = len(path.read_text().splitlines())
        max_allowed = int(baseline["lines"] * (1 - _MIN_LINE_REDUCTION))
        assert current_lines <= max_allowed, (
            f"{filename}: {current_lines} lines, expected <= {max_allowed} "
            f"(baseline {baseline['lines']}, min reduction {_MIN_LINE_REDUCTION:.0%})"
        )

    @pytest.mark.parametrize("filename,baseline", list(_BASELINES.items()))
    def test_facade_byte_size_reduced(self, filename, baseline):
        if filename == "agent_backend.py":
            path = BACKEND_DIR / filename
        else:
            path = OVAGENT_DIR / filename
        assert path.exists(), f"{filename} does not exist"
        current_bytes = path.stat().st_size
        max_allowed = int(baseline["bytes"] * (1 - _MIN_LINE_REDUCTION))
        assert current_bytes <= max_allowed, (
            f"{filename}: {current_bytes} bytes, expected <= {max_allowed} "
            f"(baseline {baseline['bytes']})"
        )


# Guardrail: prevent regression back toward monolith sizes.
# These ceilings are set above current sizes but well below baselines.
_SIZE_CEILINGS = {
    "turn_controller.py": 1600,      # current ~1410, baseline 1862
    "tool_runtime.py": 600,           # current ~474, baseline 1791
    "analysis_executor.py": 250,      # current ~150, baseline 973
    "agent_backend.py": 800,          # current ~664, baseline 3128
}


class TestMaintainabilityGuardrails:
    """Line-count ceilings prevent facades from re-accumulating bulk."""

    @pytest.mark.parametrize("filename,ceiling", list(_SIZE_CEILINGS.items()))
    def test_facade_stays_below_ceiling(self, filename, ceiling):
        if filename == "agent_backend.py":
            path = BACKEND_DIR / filename
        else:
            path = OVAGENT_DIR / filename
        current_lines = len(path.read_text().splitlines())
        assert current_lines <= ceiling, (
            f"{filename}: {current_lines} lines exceeds ceiling {ceiling}. "
            "If this is intentional new functionality, update the ceiling."
        )


# ===================================================================
# 2. Extracted module existence and non-emptiness
# ===================================================================

_EXPECTED_EXTRACTED_OVAGENT = [
    "turn_followup.py",
    "turn_artifacts.py",
    "tool_runtime_exec.py",
    "tool_runtime_io.py",
    "tool_runtime_web.py",
    "tool_runtime_workspace.py",
    "analysis_transformer.py",
    "analysis_diagnostics.py",
    "analysis_sandbox.py",
]

_EXPECTED_EXTRACTED_BACKEND = [
    "agent_backend_common.py",
    "agent_backend_openai.py",
    "agent_backend_anthropic.py",
    "agent_backend_gemini.py",
    "agent_backend_dashscope.py",
    "agent_backend_streaming.py",
]


class TestExtractedModulesExist:
    """All extracted helper modules exist and are non-trivial."""

    @pytest.mark.parametrize("filename", _EXPECTED_EXTRACTED_OVAGENT)
    def test_ovagent_extracted_module_exists(self, filename):
        path = OVAGENT_DIR / filename
        assert path.exists(), f"Missing extracted module: {filename}"
        lines = len(path.read_text().splitlines())
        assert lines >= 20, f"{filename} is suspiciously small ({lines} lines)"

    @pytest.mark.parametrize("filename", _EXPECTED_EXTRACTED_BACKEND)
    def test_backend_extracted_module_exists(self, filename):
        path = BACKEND_DIR / filename
        assert path.exists(), f"Missing extracted module: {filename}"
        lines = len(path.read_text().splitlines())
        assert lines >= 20, f"{filename} is suspiciously small ({lines} lines)"


# ===================================================================
# 3. Public facade imports
# ===================================================================


class TestOvagentPublicImports:
    """All symbols in ovagent __init__.py __all__ resolve correctly."""

    def test_all_symbols_resolve(self):
        for name in ovagent_pkg.__all__:
            obj = getattr(ovagent_pkg, name, None)
            assert obj is not None, f"ovagent.__all__ lists '{name}' but it resolves to None"

    def test_key_facade_classes_importable(self):
        assert TurnController is not None
        assert ToolRuntime is not None
        assert AnalysisExecutor is not None
        assert FollowUpGate is not None
        assert ConvergenceMonitor is not None
        assert ProactiveCodeTransformer is not None

    def test_turn_controller_backward_compat_reexports(self):
        """FollowUpGate and ConvergenceMonitor importable from turn_controller."""
        from omicverse.utils.ovagent.turn_controller import FollowUpGate as FG
        from omicverse.utils.ovagent.turn_controller import ConvergenceMonitor as CM
        assert FG is FollowUpGate
        assert CM is ConvergenceMonitor

    def test_analysis_executor_backward_compat_reexports(self):
        """ProactiveCodeTransformer importable from analysis_executor."""
        from omicverse.utils.ovagent.analysis_executor import ProactiveCodeTransformer as PCT
        assert PCT is ProactiveCodeTransformer


class TestBackendPublicImports:
    """Backend facade re-exports shared types correctly."""

    def test_backend_class_importable(self):
        assert OmicVerseLLMBackend is not None

    def test_shared_types_reexported(self):
        from omicverse.utils.agent_backend import (
            BackendConfig as BC,
            ChatResponse as CR,
            ToolCall as TC,
            Usage as U,
        )
        assert BC is BackendConfig
        assert CR is ChatResponse
        assert TC is ToolCall
        assert U is Usage

    def test_provider_adapters_importable(self):
        """Internal provider modules are importable (not necessarily public)."""
        from omicverse.utils import agent_backend_openai
        from omicverse.utils import agent_backend_anthropic
        from omicverse.utils import agent_backend_gemini
        from omicverse.utils import agent_backend_dashscope
        from omicverse.utils import agent_backend_streaming
        assert agent_backend_openai is not None
        assert agent_backend_anthropic is not None
        assert agent_backend_gemini is not None
        assert agent_backend_dashscope is not None
        assert agent_backend_streaming is not None


# ===================================================================
# 4. Facade construction contracts still hold
# ===================================================================


class TestFacadeConstructionContracts:
    """Facades instantiate with the documented signatures."""

    def test_turn_controller_3arg(self):
        ctx = _MinimalCtx()
        pb = PromptBuilder(ctx)
        rt = ToolRuntime(ctx, _DummyExecutor())
        tc = TurnController(ctx, pb, rt)
        assert tc is not None

    def test_tool_runtime_2arg(self):
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        assert rt is not None
        assert rt.registry is not None

    def test_analysis_executor_1arg(self):
        ctx = _MinimalCtx()
        ex = AnalysisExecutor(ctx)
        assert ex is not None

    def test_subagent_controller_3arg(self):
        ctx = _MinimalCtx()
        pb = PromptBuilder(ctx)
        rt = ToolRuntime(ctx, _DummyExecutor())
        sc = SubagentController(ctx, pb, rt)
        assert sc is not None

    def test_tool_runtime_late_binds_subagent_controller(self):
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        rt.set_subagent_controller(object())
        assert rt._subagent_controller is not None


# ===================================================================
# 5. Delegation wiring: facades delegate to extracted modules
# ===================================================================


class TestAnalysisExecutorDelegation:
    """AnalysisExecutor methods delegate to extracted helpers."""

    def test_facade_delegates_to_diagnostics(self):
        """Key diagnostic methods exist and are callable."""
        ctx = _MinimalCtx()
        ex = AnalysisExecutor(ctx)
        for method_name in [
            "check_code_prerequisites",
            "apply_execution_error_fix",
            "extract_package_name",
            "auto_install_package",
            "validate_outputs",
        ]:
            assert hasattr(ex, method_name), f"Missing: {method_name}"
            assert callable(getattr(ex, method_name))

    def test_facade_delegates_to_sandbox(self):
        """Key sandbox methods exist and are callable."""
        ctx = _MinimalCtx()
        ex = AnalysisExecutor(ctx)
        for method_name in [
            "request_approval",
            "execute_snippet_readonly",
            "execute_generated_code",
            "build_sandbox_globals",
            "process_context_directives",
            "normalize_doublet_obs",
        ]:
            assert hasattr(ex, method_name), f"Missing: {method_name}"
            assert callable(getattr(ex, method_name))


class TestToolRuntimeDelegation:
    """ToolRuntime dispatches to handler modules via registry."""

    def test_all_legacy_tools_have_handlers(self):
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        for tool_schema in LEGACY_AGENT_TOOLS:
            name = tool_schema["name"]
            handler = rt.registry.get_handler(name)
            assert handler is not None, f"No handler for legacy tool '{name}'"

    def test_core_catalog_tools_have_handlers(self):
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        catalog_tools = [
            "bash", "read", "edit", "write", "glob", "grep",
            "notebook_edit", "web_fetch", "web_search", "web_download",
            "tool_search", "enter_plan_mode", "exit_plan_mode",
        ]
        for name in catalog_tools:
            handler = rt.registry.get_handler(name)
            assert handler is not None, f"No handler for catalog tool '{name}'"


# ===================================================================
# 6. No unexpected dependency additions
# ===================================================================


class TestNoDependencyDrift:
    """The decomposition wave did not add new project dependencies."""

    KNOWN_DEPS = {
        "numpy", "scanpy", "pandas", "matplotlib", "scikit-learn", "scipy",
        "networkx", "multiprocess", "seaborn", "datetime", "statsmodels",
        "ipywidgets", "pygam", "igraph", "tqdm", "adjusttext", "scikit-misc",
        "scikit-image", "plotly", "numba", "requests", "transformers",
        "marsilea", "openai", "omicverse-skills", "omicverse-notebook",
        "zarr", "anndata", "setuptools", "wheel", "cython",
    }

    def test_no_new_core_dependencies(self):
        text = (PROJECT_ROOT / "pyproject.toml").read_text()
        in_deps, deps = False, []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("dependencies"):
                in_deps = True
                continue
            if in_deps and s == "]":
                break
            if in_deps:
                c = s.strip("',\" ")
                if c and not c.startswith("#"):
                    deps.append(c)
        actual = {
            re.match(r"^([A-Za-z0-9_-]+)", d).group(1).lower()
            for d in deps if re.match(r"^([A-Za-z0-9_-]+)", d)
        }
        new = actual - self.KNOWN_DEPS
        assert not new, f"Unexpected new dependencies: {new}"

    def test_extracted_modules_use_only_stdlib_and_project(self):
        """Extracted modules must not introduce third-party imports."""
        stdlib = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
        all_extracted = (
            [OVAGENT_DIR / f for f in _EXPECTED_EXTRACTED_OVAGENT]
            + [BACKEND_DIR / f for f in _EXPECTED_EXTRACTED_BACKEND]
        )
        for mod_path in all_extracted:
            if not mod_path.exists():
                continue
            tree = ast.parse(mod_path.read_text(), filename=str(mod_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in stdlib or alias.name.startswith(("omicverse", ".")):
                            continue
                        # Allow known project-level dependencies used in handlers
                        # Known project-level or declared-optional deps
                        if top in {"requests", "openai", "anthropic", "google",
                                   "dashscope", "httpx", "aiohttp",
                                   "nbformat", "pypdf", "pandas", "numpy"}:
                            continue
                        pytest.fail(
                            f"{mod_path.name} imports unexpected external: {alias.name}"
                        )


# ===================================================================
# 7. Module graph coherence
# ===================================================================


class TestModuleGraphCoherence:
    """The facade-to-helper module graph is properly structured."""

    def test_turn_controller_imports_extracted_helpers(self):
        """turn_controller.py imports from turn_followup and turn_artifacts."""
        source = (OVAGENT_DIR / "turn_controller.py").read_text()
        assert "from .turn_followup import" in source
        assert "from .turn_artifacts import" in source

    def test_tool_runtime_imports_handler_modules(self):
        """tool_runtime.py imports from all four handler modules."""
        source = (OVAGENT_DIR / "tool_runtime.py").read_text()
        for mod in ["tool_runtime_exec", "tool_runtime_io",
                     "tool_runtime_web", "tool_runtime_workspace"]:
            assert mod in source, f"tool_runtime.py missing import of {mod}"

    def test_analysis_executor_imports_extracted_helpers(self):
        """analysis_executor.py imports from transformer, diagnostics, sandbox."""
        source = (OVAGENT_DIR / "analysis_executor.py").read_text()
        assert "analysis_transformer" in source
        assert "analysis_diagnostics" in source
        assert "analysis_sandbox" in source

    def test_agent_backend_imports_provider_modules(self):
        """agent_backend.py imports from all provider adapter modules."""
        source = (BACKEND_DIR / "agent_backend.py").read_text()
        for mod in ["agent_backend_common", "agent_backend_openai",
                     "agent_backend_anthropic", "agent_backend_gemini",
                     "agent_backend_dashscope", "agent_backend_streaming"]:
            assert mod in source, f"agent_backend.py missing import of {mod}"

    def test_no_circular_imports_in_extracted_modules(self):
        """Extracted modules must not import their parent facade."""
        checks = [
            (OVAGENT_DIR / "turn_followup.py", "turn_controller"),
            (OVAGENT_DIR / "turn_artifacts.py", "turn_controller"),
            (OVAGENT_DIR / "tool_runtime_exec.py", "tool_runtime"),
            (OVAGENT_DIR / "tool_runtime_io.py", "tool_runtime"),
            (OVAGENT_DIR / "tool_runtime_web.py", "tool_runtime"),
            (OVAGENT_DIR / "tool_runtime_workspace.py", "tool_runtime"),
            (OVAGENT_DIR / "analysis_transformer.py", "analysis_executor"),
            (OVAGENT_DIR / "analysis_diagnostics.py", "analysis_executor"),
            (OVAGENT_DIR / "analysis_sandbox.py", "analysis_executor"),
            (BACKEND_DIR / "agent_backend_common.py", "agent_backend"),
            (BACKEND_DIR / "agent_backend_openai.py", "agent_backend"),
            (BACKEND_DIR / "agent_backend_anthropic.py", "agent_backend"),
            (BACKEND_DIR / "agent_backend_gemini.py", "agent_backend"),
            (BACKEND_DIR / "agent_backend_dashscope.py", "agent_backend"),
            (BACKEND_DIR / "agent_backend_streaming.py", "agent_backend"),
        ]
        for mod_path, forbidden_import in checks:
            if not mod_path.exists():
                continue
            source = mod_path.read_text()
            # Check for direct "from .forbidden_import import" or
            # "import ...forbidden_import"
            pattern = rf"(?:from\s+\.{forbidden_import}\s+import|import\s+\S*{forbidden_import}(?:\s|$))"
            matches = re.findall(pattern, source)
            # Filter out cases where the module name is a substring of a
            # longer import (e.g., "agent_backend_common" contains
            # "agent_backend")
            real_matches = [
                m for m in matches
                if not re.search(rf"{forbidden_import}_\w+", m)
            ]
            assert not real_matches, (
                f"{mod_path.name} has circular import of {forbidden_import}: {real_matches}"
            )


# ===================================================================
# 8. Responsibility narrowing verification
# ===================================================================


class TestResponsibilityNarrowing:
    """Each facade has a narrower responsibility than its baseline.

    Verified by checking that extracted domain logic is NOT duplicated
    in the facade source — it should exist only in the helper module.
    """

    def test_turn_controller_no_followup_gate_class_def(self):
        """FollowUpGate class body was extracted to turn_followup.py."""
        source = (OVAGENT_DIR / "turn_controller.py").read_text()
        assert "class FollowUpGate" not in source

    def test_turn_controller_no_convergence_monitor_class_def(self):
        """ConvergenceMonitor class body was extracted to turn_followup.py."""
        source = (OVAGENT_DIR / "turn_controller.py").read_text()
        assert "class ConvergenceMonitor" not in source

    def test_tool_runtime_no_concrete_handler_bodies(self):
        """Concrete handler implementations live in handler modules, not the facade."""
        source = (OVAGENT_DIR / "tool_runtime.py").read_text()
        # These function bodies should not appear in the facade
        for pattern in [
            "def handle_bash(",
            "def handle_read(",
            "def handle_edit(",
            "def handle_web_fetch(",
            "def handle_web_search(",
            "def handle_create_task(",
        ]:
            assert pattern not in source, (
                f"tool_runtime.py still contains handler: {pattern}"
            )

    def test_analysis_executor_no_transformer_class_def(self):
        """ProactiveCodeTransformer body was extracted."""
        source = (OVAGENT_DIR / "analysis_executor.py").read_text()
        assert "class ProactiveCodeTransformer" not in source

    def test_analysis_executor_no_sandbox_globals_body(self):
        """build_sandbox_globals body was extracted to analysis_sandbox.py."""
        source = (OVAGENT_DIR / "analysis_executor.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "build_sandbox_globals"):
                # The facade method should be a thin delegation (few lines)
                assert len(node.body) <= 3, (
                    "build_sandbox_globals in facade has too many statements "
                    f"({len(node.body)}); should delegate to analysis_sandbox"
                )

    def test_agent_backend_no_provider_request_bodies(self):
        """Provider-specific request building was extracted."""
        source = (BACKEND_DIR / "agent_backend.py").read_text()
        # Heavy provider-specific logic should not be in the facade
        for pattern in [
            "def _build_anthropic_request(",
            "def _build_gemini_request(",
            "def _build_dashscope_request(",
        ]:
            assert pattern not in source, (
                f"agent_backend.py still contains provider-specific: {pattern}"
            )


# ===================================================================
# 9. Overall decomposition summary metrics
# ===================================================================


class TestDecompositionSummaryMetrics:
    """Aggregate metrics across all four facades."""

    def test_total_facade_lines_below_threshold(self):
        """Combined facade line count is well below baseline total (7,754)."""
        total = 0
        for filename in _BASELINES:
            if filename == "agent_backend.py":
                path = BACKEND_DIR / filename
            else:
                path = OVAGENT_DIR / filename
            total += len(path.read_text().splitlines())
        baseline_total = sum(b["lines"] for b in _BASELINES.values())
        # Must be at least 40% smaller overall
        assert total <= baseline_total * 0.60, (
            f"Total facade lines {total} is not < 60% of baseline {baseline_total}"
        )

    def test_total_facade_bytes_below_threshold(self):
        """Combined facade byte size is well below baseline total."""
        total = 0
        for filename in _BASELINES:
            if filename == "agent_backend.py":
                path = BACKEND_DIR / filename
            else:
                path = OVAGENT_DIR / filename
            total += path.stat().st_size
        baseline_total = sum(b["bytes"] for b in _BASELINES.values())
        assert total <= baseline_total * 0.60, (
            f"Total facade bytes {total} is not < 60% of baseline {baseline_total}"
        )

    def test_extracted_modules_account_for_bulk(self):
        """The extracted modules collectively contain substantial code."""
        total_extracted = 0
        for f in _EXPECTED_EXTRACTED_OVAGENT:
            p = OVAGENT_DIR / f
            if p.exists():
                total_extracted += len(p.read_text().splitlines())
        for f in _EXPECTED_EXTRACTED_BACKEND:
            p = BACKEND_DIR / f
            if p.exists():
                total_extracted += len(p.read_text().splitlines())
        # Extracted modules should collectively have at least 3000 lines
        assert total_extracted >= 3000, (
            f"Extracted modules total only {total_extracted} lines; "
            "expected >= 3000 to confirm bulk was actually moved"
        )
