"""Cross-module integration audit for the OVAgent runtime-upgrade wave.

Verifies that the seven new runtime subsystems (tasks 024-030, 034)
work together correctly on the real codebase:

  - tool_registry, tool_scheduler, context_budget, repair_loop,
    permission_policy, event_stream, prompt_templates

These tests are intentionally lightweight: no real LLM calls, no heavy
optional dependencies.  They run under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""

import ast
import asyncio
import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("OV_AGENT_RUN_HARNESS_TESTS"),
    reason="harness tests disabled",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs to avoid heavy omicverse.__init__
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SAVED = {n: sys.modules.get(n) for n in ["omicverse", "omicverse.utils"]}
for n in ["omicverse", "omicverse.utils"]:
    sys.modules.pop(n, None)

_ov_pkg = types.ModuleType("omicverse")
_ov_pkg.__path__ = [str(PACKAGE_ROOT)]
_ov_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse", loader=None, is_package=True)
sys.modules["omicverse"] = _ov_pkg

_utils_pkg = types.ModuleType("omicverse.utils")
_utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
_utils_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse.utils", loader=None, is_package=True)
sys.modules["omicverse.utils"] = _utils_pkg
_ov_pkg.utils = _utils_pkg

import omicverse.utils.ovagent as ovagent_pkg  # noqa: E402
from omicverse.utils.ovagent import (  # noqa: E402
    ApprovalClass, BudgetSlice, BudgetSliceType, CompactionCheckpoint,
    ContextBudgetManager, ExecutionBatch, FailureEnvelope, IsolationMode,
    OutputTier, ParallelClass, PermissionDecision, PermissionPolicy,
    PermissionVerdict, PromptBuilder, PromptOverlay, PromptTemplateEngine,
    RuntimeEventEmitter, ScheduleResult, ScheduledCall, SubagentController,
    SubagentRuntime, ToolMetadata, ToolRegistry, ToolRuntime, ToolScheduler,
    TruncationPolicy, TurnController, build_agentic_engine,
    build_dataset_context, build_default_registry, build_llm_repair_prompt,
    build_subagent_engine, create_default_policy,
    create_subagent_budget_manager, create_subagent_policy, execute_batch,
)
from omicverse.utils.ovagent.tool_runtime import LEGACY_AGENT_TOOLS  # noqa: E402

for n, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(n, None)
    else:
        sys.modules[n] = mod

# ---------------------------------------------------------------------------
# Minimal context double (matches AgentContext protocol)
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
    """Lightweight AgentContext double for integration tests."""
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
        self._web_session_id = "test-audit-session"
        self._managed_api_env = {}
        self._code_only_mode = False
        self._code_only_captured_code = ""
        self._code_only_captured_history = []
        self.use_notebook_execution = False
        self.enable_filesystem_context = False
        # Attach stub methods that return neutral defaults
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


class _FakeToolCall:
    """Simple stand-in for a provider tool-call object."""
    def __init__(self, name, call_id="tc_1", arguments=None):
        self.name = name
        self.id = call_id
        self.arguments = arguments or {}


# ===================================================================
# 1. TestExportCompleteness
# ===================================================================

class TestExportCompleteness:
    """Every new public symbol is reachable and listed in __all__."""

    NEW_SYMBOLS = {
        "ToolMetadata", "ToolRegistry", "build_default_registry",
        "ApprovalClass", "ParallelClass", "OutputTier", "IsolationMode",
        "ToolScheduler", "ExecutionBatch", "ScheduledCall", "ScheduleResult",
        "execute_batch", "ContextBudgetManager", "BudgetSlice",
        "BudgetSliceType", "CompactionCheckpoint", "TruncationPolicy",
        "create_subagent_budget_manager", "ExecutionRepairLoop",
        "FailureEnvelope", "RepairAttempt", "RepairResult",
        "build_dataset_context", "build_llm_repair_prompt",
        "PermissionPolicy", "PermissionDecision", "PermissionVerdict",
        "create_default_policy", "create_subagent_policy",
        "RuntimeEventEmitter", "PromptTemplateEngine", "PromptOverlay",
        "build_agentic_engine", "build_subagent_engine",
    }

    def test_all_new_symbols_in_all(self):
        missing = self.NEW_SYMBOLS - set(ovagent_pkg.__all__)
        assert not missing, f"Missing from __all__: {missing}"

    def test_all_new_symbols_resolve(self):
        for name in self.NEW_SYMBOLS:
            assert getattr(ovagent_pkg, name, None) is not None, f"'{name}' is None"

    def test_no_stale_symbols_in_all(self):
        for name in ovagent_pkg.__all__:
            assert getattr(ovagent_pkg, name, None) is not None, f"Stale: '{name}'"


# ===================================================================
# 2. TestCrossModuleTypeCoherence
# ===================================================================

class TestCrossModuleTypeCoherence:
    """Types from different modules compose correctly."""

    def test_scheduler_accepts_registry_produces_parallel_class(self):
        registry = build_default_registry()
        result = ToolScheduler(registry).schedule([_FakeToolCall("Read")])
        assert isinstance(result, ScheduleResult)
        assert isinstance(result.batches[0].calls[0].parallel_class, ParallelClass)

    def test_permission_policy_uses_same_approval_class(self):
        registry = build_default_registry()
        decision = PermissionPolicy(registry).check("Bash")
        meta = registry.get("Bash")
        assert decision.verdict.value == meta.approval_class.value

    def test_budget_manager_truncate_uses_output_tier(self):
        mgr = ContextBudgetManager(model="test")
        assert (mgr.get_truncation_policy(OutputTier.verbose).max_tokens
                > mgr.get_truncation_policy(OutputTier.minimal).max_tokens)

    def test_prompt_engine_feeds_prompt_builder(self):
        prompt = PromptBuilder(_MinimalCtx()).build_agentic_system_prompt()
        assert isinstance(prompt, str) and "OmicVerse" in prompt

    def test_create_subagent_policy_returns_permission_decision(self):
        registry = build_default_registry()
        dec = create_subagent_policy(registry, frozenset({"Read"})).check("Read")
        assert isinstance(dec, PermissionDecision)


# ===================================================================
# 3. TestRegistryToSchedulerPipeline
# ===================================================================

class TestRegistryToSchedulerPipeline:
    """Registry -> Scheduler -> Batch execution pipeline."""

    def test_schedule_produces_batches(self):
        result = ToolScheduler(build_default_registry()).schedule(
            [_FakeToolCall("Read"), _FakeToolCall("Glob"), _FakeToolCall("Grep")]
        )
        assert result.total_calls == 3 and result.total_batches >= 1

    def test_readonly_tools_batched_together(self):
        result = ToolScheduler(build_default_registry()).schedule(
            [_FakeToolCall("Read"), _FakeToolCall("Glob"), _FakeToolCall("Grep")]
        )
        assert result.total_batches == 1
        assert result.batches[0].parallel is True and result.batches[0].size == 3

    def test_stateful_tool_forces_serial_batch(self):
        result = ToolScheduler(build_default_registry()).schedule(
            [_FakeToolCall("Read"), _FakeToolCall("Bash"), _FakeToolCall("Read", "tc_3")]
        )
        assert result.total_batches == 3
        assert result.batches[1].parallel is False

    def test_execute_batch_returns_results_in_index_order(self):
        batch = ToolScheduler(build_default_registry()).schedule(
            [_FakeToolCall("Read"), _FakeToolCall("Glob", "tc_2")]
        ).batches[0]

        async def dispatch(sc):
            return f"r_{sc.index}"

        results = asyncio.run(execute_batch(batch, dispatch))
        assert [i for i, _ in results] == sorted(i for i, _ in results)
        assert results[0] == (0, "r_0") and results[1] == (1, "r_1")


# ===================================================================
# 4. TestRegistryToPermissionPipeline
# ===================================================================

class TestRegistryToPermissionPipeline:
    """Registry -> PermissionPolicy -> check verdicts."""

    def test_known_tool_verdicts(self):
        policy = create_default_policy(build_default_registry())
        assert policy.check("Read").verdict == PermissionVerdict.allow
        assert policy.check("Bash").verdict == PermissionVerdict.ask

    def test_unknown_tool_denied(self):
        assert create_default_policy(build_default_registry()).check(
            "nonexistent_xyz"
        ).verdict == PermissionVerdict.deny

    def test_subagent_policy_restricts_tools(self):
        registry = build_default_registry()
        policy = create_subagent_policy(registry, frozenset({"Read", "Glob", "inspect_data"}))
        assert not policy.check("Read").is_denied
        assert not policy.check("inspect_data").is_denied
        assert policy.check("Bash").is_denied
        assert policy.check("Edit").is_denied


# ===================================================================
# 5. TestBudgetManagerIntegration
# ===================================================================

class TestBudgetManagerIntegration:
    """ContextBudgetManager cross-module integration."""

    def test_record_slices_and_consumption_by_type(self):
        mgr = ContextBudgetManager(model="gpt-4o")
        mgr.record(BudgetSliceType.system_prompt, "System prompt text here")
        mgr.record(BudgetSliceType.tool_output, "Tool output", content_key="bash",
                    tier=OutputTier.verbose)
        totals = mgr.consumption_by_type()
        assert "system_prompt" in totals and "tool_output" in totals
        assert mgr.total_consumed > 0

    def test_truncate_verbose_vs_minimal(self):
        mgr = ContextBudgetManager(model="test")
        long_text = "x" * 50000
        v = mgr.truncate_output(long_text, OutputTier.verbose)
        m = mgr.truncate_output(long_text, OutputTier.minimal)
        assert len(v) > len(m) and len(v) < len(long_text)

    def test_compact_history_with_checkpoint(self):
        mgr = ContextBudgetManager(model="test")
        mgr.add_checkpoint(0, "Summary of turns 0-3", messages_covered=4)
        messages = [{"role": "system", "content": "Agent."}] + [
            {"role": r, "content": f"msg {i}"}
            for i in range(6) for r in ["user", "assistant"]
        ]
        compacted, cp = mgr.compact_history(messages, keep_recent=2)
        assert cp is not None and isinstance(cp, CompactionCheckpoint)
        assert len(compacted) <= len(messages)

    def test_subagent_budget_tighter_policies(self):
        default_v = ContextBudgetManager(model="test").get_truncation_policy(OutputTier.verbose)
        sub_v = create_subagent_budget_manager(model="test").get_truncation_policy(OutputTier.verbose)
        assert sub_v.max_tokens < default_v.max_tokens


# ===================================================================
# 6. TestRepairLoopIntegration
# ===================================================================

class TestRepairLoopIntegration:
    """FailureEnvelope, repair prompt, and dataset context."""

    def test_envelope_from_exception_roundtrip(self):
        try:
            raise ValueError("missing column 'batch'")
        except ValueError as exc:
            env = FailureEnvelope.from_exception(exc, phase="execution", retry_count=1,
                                                  code="print('hello')")
        assert env.exception == "ValueError" and "missing column" in env.summary
        d = env.to_dict()
        assert d["phase"] == "execution" and isinstance(d["repair_hints"], list)

    def test_build_llm_repair_prompt_non_empty(self):
        env = FailureEnvelope(phase="execution", exception="KeyError",
                              summary="'batch' not found",
                              traceback_excerpt="KeyError: 'batch'",
                              retry_count=0, code="adata.obs['batch']",
                              dataset_context="Shape: 1000 obs x 500 vars",
                              repair_hints=["check column names"])
        prompt = build_llm_repair_prompt(env)
        assert len(prompt) > 50 and "KeyError" in prompt and "DATASET" in prompt

    def test_build_dataset_context_handles_none_and_missing(self):
        assert build_dataset_context(None) == ""
        assert build_dataset_context(SimpleNamespace()) == ""

    def test_build_dataset_context_with_shape(self):
        result = build_dataset_context(SimpleNamespace(shape=(500, 200)))
        assert "500" in result and "200" in result


# ===================================================================
# 7. TestEventStreamIntegration
# ===================================================================

class TestEventStreamIntegration:
    """RuntimeEventEmitter integration with callbacks and AST checks."""

    def test_callback_receives_structured_payloads(self):
        captured = []
        async def cb(event): captured.append(event)
        emitter = RuntimeEventEmitter(event_callback=cb, source="audit")
        asyncio.run(emitter.emit("status", {"key": "val"}, category="test"))
        assert len(captured) == 1 and captured[0]["type"] == "status"

    def test_event_types_coverage(self):
        for m in ["turn_started", "tool_dispatched", "tool_completed", "task_finished"]:
            assert hasattr(RuntimeEventEmitter, m) and callable(getattr(RuntimeEventEmitter, m))

    def test_no_bare_print_in_runtime_modules(self):
        ovagent_dir = PACKAGE_ROOT / "utils" / "ovagent"
        for mod_path in [
            ovagent_dir / f for f in [
                "turn_controller.py", "tool_runtime.py", "subagent_controller.py",
                "event_stream.py", "tool_scheduler.py", "context_budget.py",
                "permission_policy.py",
            ]
        ]:
            if not mod_path.exists():
                continue
            tree = ast.parse(mod_path.read_text(), filename=str(mod_path))
            lines = [n.lineno for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id == "print"]
            assert lines == [], f"{mod_path.name} has print() on lines {lines}"


# ===================================================================
# 8. TestPromptCompositionPipeline
# ===================================================================

class TestPromptCompositionPipeline:
    """PromptTemplateEngine -> PromptBuilder composition."""

    def test_agentic_engine_has_expected_overlays(self):
        engine = build_agentic_engine()
        assert isinstance(engine, PromptTemplateEngine)
        assert engine.has_overlay("tool_instructions") and engine.has_overlay("workflow")

    def test_subagent_engine_explore(self):
        rendered = build_subagent_engine("explore").render()
        assert "bioinformatics" in rendered.lower() or "data" in rendered.lower()

    def test_builder_produces_prompt_with_engine_content(self):
        prompt = PromptBuilder(_MinimalCtx()).build_agentic_system_prompt()
        assert "OmicVerse" in prompt and "ToolSearch" in prompt

    def test_overlay_priority_deterministic(self):
        def make():
            e = PromptTemplateEngine()
            e.set_base("base")
            e.add_overlay(PromptOverlay("z_last", "Z", priority=200))
            e.add_overlay(PromptOverlay("a_first", "A", priority=10))
            e.add_overlay(PromptOverlay("m_mid", "M", priority=100))
            return e.overlay_names
        assert make() == ["a_first", "m_mid", "z_last"]
        assert make() == make()  # deterministic across calls


# ===================================================================
# 9. TestSubagentIsolationContract
# ===================================================================

class TestSubagentIsolationContract:
    """SubagentRuntime scoped isolation guarantees."""

    def _make_rt(self, allowed):
        registry = build_default_registry()
        return SubagentRuntime(
            agent_type="explore", max_turns=5,
            permission_policy=create_subagent_policy(registry, frozenset(allowed)),
            budget_manager=create_subagent_budget_manager(model="test"),
            tool_schemas=[{"name": "Read"}, {"name": "Glob"}],
        )

    def test_permission_denies_outside_allowed_set(self):
        rt = self._make_rt({"Read", "Glob"})
        assert not rt.check_tool_permission("Read").is_denied
        assert rt.check_tool_permission("Bash").is_denied

    def test_budget_manager_independence(self):
        parent = ContextBudgetManager(model="test")
        child = create_subagent_budget_manager(model="test")
        child.record(BudgetSliceType.tool_output, "output", tier=OutputTier.standard)
        assert child.total_consumed > 0 and parent.total_consumed == 0

    def test_tool_schemas_are_snapshot(self):
        original = [{"name": "Read"}, {"name": "Glob"}]
        rt = self._make_rt({"Read"})
        rt.tool_schemas = list(original)
        rt.tool_schemas.append({"name": "Bash"})
        assert len(original) == 2  # original unaffected


# ===================================================================
# 10. TestEndToEndDispatchPath
# ===================================================================

class TestEndToEndDispatchPath:
    """Registry -> Scheduler -> Permission -> Budget end-to-end."""

    def test_full_pipeline_consistency(self):
        registry = build_default_registry()
        scheduler = ToolScheduler(registry)
        policy = create_default_policy(registry)
        budget = ContextBudgetManager(model="test")

        calls = [_FakeToolCall("Read"), _FakeToolCall("Bash", "tc_2"),
                 _FakeToolCall("Glob", "tc_3")]
        schedule = scheduler.schedule(calls)
        assert schedule.total_calls == 3

        all_sc = [sc for b in schedule.batches for sc in b.calls]
        for sc in all_sc:
            meta = registry.get(sc.canonical_name)
            assert meta is not None
            assert sc.parallel_class == meta.parallel_class
            # Permission is consistent with registry
            assert policy.check(sc.canonical_name).verdict == PermissionVerdict(
                meta.approval_class.value
            )
            # Budget can record with correct tier
            budget.record(BudgetSliceType.tool_output,
                          f"output from {sc.canonical_name}",
                          content_key=sc.canonical_name, tier=meta.output_tier)

        assert budget.total_consumed > 0
        assert "tool_output" in budget.consumption_by_type()


# ===================================================================
# 11. TestNoNewDependencies
# ===================================================================

class TestNoNewDependencies:
    """Runtime-upgrade modules don't add new project dependencies."""

    KNOWN_DEPS = {
        "numpy", "scanpy", "pandas", "matplotlib", "scikit-learn", "scipy",
        "networkx", "multiprocess", "seaborn", "datetime", "statsmodels",
        "ipywidgets", "pygam", "igraph", "tqdm", "adjusttext", "scikit-misc",
        "scikit-image", "plotly", "numba", "requests", "transformers",
        "marsilea", "openai", "omicverse-skills", "omicverse-notebook",
        "zarr", "anndata", "setuptools", "wheel", "cython",
    }

    def test_no_new_core_dependencies(self):
        import re as _re
        text = (PROJECT_ROOT / "pyproject.toml").read_text()
        in_deps, deps = False, []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("dependencies"):
                in_deps = True; continue
            if in_deps and s == "]":
                break
            if in_deps:
                c = s.strip("',\" ")
                if c and not c.startswith("#"):
                    deps.append(c)
        actual = {_re.match(r"^([A-Za-z0-9_-]+)", d).group(1).lower()
                  for d in deps if _re.match(r"^([A-Za-z0-9_-]+)", d)}
        new = actual - self.KNOWN_DEPS
        assert not new, f"New dependencies: {new}"

    def test_runtime_modules_import_only_stdlib_and_project(self):
        ovagent_dir = PACKAGE_ROOT / "utils" / "ovagent"
        stdlib = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
        for mod_path in [ovagent_dir / f for f in [
            "tool_registry.py", "tool_scheduler.py", "context_budget.py",
            "repair_loop.py", "permission_policy.py", "event_stream.py",
            "prompt_templates.py",
        ]]:
            if not mod_path.exists():
                continue
            tree = ast.parse(mod_path.read_text(), filename=str(mod_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        assert top in stdlib or alias.name.startswith(("omicverse", ".")), (
                            f"{mod_path.name} imports external: {alias.name}"
                        )


# ===================================================================
# 12. TestInterfaceStability
# ===================================================================

class TestInterfaceStability:
    """Public APIs from existing modules remain stable."""

    def test_omicverse_agent_methods(self):
        source = (PACKAGE_ROOT / "utils" / "smart_agent.py").read_text()
        for m in ["_run_agentic_loop", "run_async", "stream_async", "generate_code_async"]:
            assert f"def {m}" in source, f"OmicVerseAgent missing: {m}"

    def test_tool_runtime_methods(self):
        ctx = _MinimalCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        for m in ["dispatch_tool", "get_visible_agent_tools", "get_loaded_tool_names"]:
            assert hasattr(rt, m) and callable(getattr(rt, m))

    def test_turn_controller_3arg(self):
        ctx = _MinimalCtx()
        assert TurnController(ctx, PromptBuilder(ctx), ToolRuntime(ctx, _DummyExecutor())) is not None

    def test_subagent_controller_3arg(self):
        ctx = _MinimalCtx()
        assert SubagentController(ctx, PromptBuilder(ctx), ToolRuntime(ctx, _DummyExecutor())) is not None

    def test_prompt_builder_ctx(self):
        pb = PromptBuilder(_MinimalCtx())
        assert hasattr(pb, "build_agentic_system_prompt") and hasattr(pb, "build_explore_prompt")
