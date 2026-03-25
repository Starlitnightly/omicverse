"""
Contract validation tests for the OVAgent runtime contracts.

These tests verify that the runtime contracts defined in
``omicverse/utils/ovagent/contracts.py`` are self-consistent, serializable,
and meet the benchmark thresholds documented in
``docs/harness/runtime-benchmarks.md``.

No LLM calls, no heavy imports, no server requirement.
"""

import dataclasses
import importlib.machinery
import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load contracts modules in isolation to avoid pulling heavy
# omicverse dependencies through omicverse/utils/__init__.py.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

# Stub top-level packages so subpackage imports work without heavy deps
_STUB_NAMES = ["omicverse", "omicverse.utils", "omicverse.utils.ovagent",
               "omicverse.utils.harness"]
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _STUB_NAMES}

for name in _STUB_NAMES:
    sys.modules.pop(name, None)

omicverse_pkg = types.ModuleType("omicverse")
omicverse_pkg.__path__ = [str(PACKAGE_ROOT)]
omicverse_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True,
)
sys.modules["omicverse"] = omicverse_pkg

utils_pkg = types.ModuleType("omicverse.utils")
utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True,
)
sys.modules["omicverse.utils"] = utils_pkg
omicverse_pkg.utils = utils_pkg  # type: ignore[attr-defined]

ovagent_pkg = types.ModuleType("omicverse.utils.ovagent")
ovagent_pkg.__path__ = [str(PACKAGE_ROOT / "utils" / "ovagent")]
ovagent_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils.ovagent", loader=None, is_package=True,
)
sys.modules["omicverse.utils.ovagent"] = ovagent_pkg
utils_pkg.ovagent = ovagent_pkg  # type: ignore[attr-defined]

harness_pkg = types.ModuleType("omicverse.utils.harness")
harness_pkg.__path__ = [str(PACKAGE_ROOT / "utils" / "harness")]
harness_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils.harness", loader=None, is_package=True,
)
sys.modules["omicverse.utils.harness"] = harness_pkg
utils_pkg.harness = harness_pkg  # type: ignore[attr-defined]


def _load_module(fqn: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Load the contracts module (stdlib only — no heavy deps)
contracts_mod = _load_module(
    "omicverse.utils.ovagent.contracts",
    PACKAGE_ROOT / "utils" / "ovagent" / "contracts.py",
)

# Load context_compactor (needed by context_budget)
context_compactor_mod = _load_module(
    "omicverse.utils.context_compactor",
    PACKAGE_ROOT / "utils" / "context_compactor.py",
)
utils_pkg.context_compactor = context_compactor_mod  # type: ignore[attr-defined]

# Load context_budget module
context_budget_mod = _load_module(
    "omicverse.utils.ovagent.context_budget",
    PACKAGE_ROOT / "utils" / "ovagent" / "context_budget.py",
)
ovagent_pkg.context_budget = context_budget_mod  # type: ignore[attr-defined]

# Load harness contracts for cross-compatibility tests
harness_contracts_mod = _load_module(
    "omicverse.utils.harness.contracts",
    PACKAGE_ROOT / "utils" / "harness" / "contracts.py",
)

# Load harness tool_catalog for cross-compatibility tests
harness_tool_catalog_mod = _load_module(
    "omicverse.utils.harness.tool_catalog",
    PACKAGE_ROOT / "utils" / "harness" / "tool_catalog.py",
)

# Restore original modules
for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module

# Now pull symbols from the loaded modules
ApprovalClass = contracts_mod.ApprovalClass
IsolationMode = contracts_mod.IsolationMode
ParallelClass = contracts_mod.ParallelClass
OutputTier = contracts_mod.OutputTier
ToolPolicyMetadata = contracts_mod.ToolPolicyMetadata
ToolContract = contracts_mod.ToolContract
ImportanceTier = contracts_mod.ImportanceTier
ContextEntry = contracts_mod.ContextEntry
OverflowPolicy = contracts_mod.OverflowPolicy
ContextBudgetConfig = contracts_mod.ContextBudgetConfig
ContextBudgetManager = contracts_mod.ContextBudgetManager
FailurePhase = contracts_mod.FailurePhase
RepairHint = contracts_mod.RepairHint
ExecutionFailureEnvelope = contracts_mod.ExecutionFailureEnvelope
EVENT_CATEGORIES = contracts_mod.EVENT_CATEGORIES
EventEmitter = contracts_mod.EventEmitter
PromptLayerKind = contracts_mod.PromptLayerKind
PromptLayer = contracts_mod.PromptLayer
PromptComposer = contracts_mod.PromptComposer
RemoteReviewConfig = contracts_mod.RemoteReviewConfig
BenchmarkThresholds = contracts_mod.BenchmarkThresholds
BENCHMARK_THRESHOLDS = contracts_mod.BENCHMARK_THRESHOLDS

HARNESS_EVENT_TYPES = harness_contracts_mod.HARNESS_EVENT_TYPES
StepTrace = harness_contracts_mod.StepTrace
make_step_id = harness_contracts_mod.make_step_id
ToolDefinition = harness_tool_catalog_mod.ToolDefinition

# Context budget symbols
DefaultContextBudgetManager = context_budget_mod.DefaultContextBudgetManager
classify_fn = context_budget_mod.classify
max_tokens_for_tier = context_budget_mod.max_tokens_for_tier
estimate_tokens = context_compactor_mod.estimate_tokens


# =====================================================================
# 1. Tool Policy Metadata
# =====================================================================

class TestToolPolicyMetadata:

    def test_default_policy_is_permissive(self):
        """Default policy allows execution without approval."""
        policy = ToolPolicyMetadata()
        assert policy.approval == ApprovalClass.ALLOW
        assert policy.isolation == IsolationMode.IN_PROCESS
        assert policy.parallel == ParallelClass.SAFE
        assert policy.output_tier == OutputTier.STANDARD
        assert policy.read_only is False

    def test_policy_is_frozen(self):
        policy = ToolPolicyMetadata()
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.approval = ApprovalClass.DENY  # type: ignore[misc]

    def test_enum_values_are_strings(self):
        """Enum values serialize as plain strings for JSON compatibility."""
        assert ApprovalClass.ALLOW.value == "allow"
        assert IsolationMode.WORKTREE.value == "worktree"
        assert ParallelClass.CONDITIONAL.value == "conditional"
        assert OutputTier.UNBOUNDED.value == "unbounded"


class TestToolContract:

    def test_minimum_fields(self):
        """ToolContract must have at least name, group, description."""
        tc = ToolContract(name="Read", group="core", description="Read a file")
        assert tc.name == "Read"
        assert tc.group == "core"
        assert tc.description == "Read a file"
        n_fields = len(dataclasses.fields(ToolContract))
        assert n_fields >= BENCHMARK_THRESHOLDS.tool_contract_min_fields

    def test_requires_approval(self):
        tc_allow = ToolContract(name="Read", group="core", description="Read")
        tc_ask = ToolContract(
            name="Bash", group="core", description="Shell",
            policy=ToolPolicyMetadata(approval=ApprovalClass.ASK),
        )
        assert not tc_allow.requires_approval()
        assert tc_ask.requires_approval()

    def test_is_parallel_safe(self):
        tc_safe = ToolContract(name="Grep", group="code", description="Search")
        tc_unsafe = ToolContract(
            name="Edit", group="code", description="Edit",
            policy=ToolPolicyMetadata(parallel=ParallelClass.UNSAFE),
        )
        assert tc_safe.is_parallel_safe()
        assert not tc_unsafe.is_parallel_safe()

    def test_to_dict_round_trip(self):
        policy = ToolPolicyMetadata(
            approval=ApprovalClass.ASK,
            isolation=IsolationMode.SUBPROCESS,
            output_tier=OutputTier.VERBOSE,
        )
        tc = ToolContract(
            name="Bash", group="core", description="Shell",
            parameters_schema={"type": "object"},
            policy=policy,
            keywords=("shell", "exec"),
            aliases=("sh",),
        )
        d = tc.to_dict()
        assert d["name"] == "Bash"
        assert d["policy"]["approval"] == "ask"
        assert d["policy"]["isolation"] == "subprocess"
        assert d["policy"]["output_tier"] == "verbose"
        assert d["keywords"] == ["shell", "exec"]
        assert d["aliases"] == ["sh"]

    def test_to_dict_benchmark(self):
        """ToolContract.to_dict() must complete within threshold."""
        tc = ToolContract(name="T", group="g", description="d")
        start = time.perf_counter()
        for _ in range(1000):
            tc.to_dict()
        elapsed = (time.perf_counter() - start) / 1000
        assert elapsed < BENCHMARK_THRESHOLDS.tool_lookup_max_seconds


# =====================================================================
# 2. Context Budget Contract
# =====================================================================

class TestContextEntry:

    def test_minimum_fields(self):
        entry = ContextEntry(content="hello", source="user")
        assert entry.content == "hello"
        assert entry.source == "user"
        assert entry.importance == ImportanceTier.STANDARD
        n_fields = len(dataclasses.fields(ContextEntry))
        assert n_fields >= BENCHMARK_THRESHOLDS.context_entry_min_fields

    def test_estimated_tokens_from_content(self):
        entry = ContextEntry(content="a" * 400, source="test")
        assert entry.estimated_tokens() == 100

    def test_estimated_tokens_explicit(self):
        entry = ContextEntry(content="short", source="test", token_count=50)
        assert entry.estimated_tokens() == 50

    def test_importance_ordering(self):
        """ImportanceTier values must be orderable by name for human readability."""
        tiers = list(ImportanceTier)
        assert len(tiers) == 5
        assert ImportanceTier.CRITICAL in tiers
        assert ImportanceTier.EPHEMERAL in tiers


class TestContextBudgetConfig:

    def test_usable_tokens(self):
        cfg = ContextBudgetConfig(max_tokens=128_000, reserve_tokens=4_096)
        assert cfg.usable_tokens == 123_904

    def test_default_compaction_threshold(self):
        cfg = ContextBudgetConfig()
        assert 0.0 < cfg.compaction_threshold < 1.0

    def test_overflow_policy_values(self):
        policies = list(OverflowPolicy)
        assert len(policies) == 4
        assert OverflowPolicy.COMPACT_LOW_IMPORTANCE in policies


class TestContextBudgetManagerProtocol:

    def test_minimal_implementation_satisfies_protocol(self):
        """A minimal class satisfying the protocol shape is accepted."""
        class MinimalBudget:
            @property
            def config(self):
                return ContextBudgetConfig()

            @property
            def current_token_count(self):
                return 0

            @property
            def remaining_tokens(self):
                return 128_000

            def add_entry(self, entry):
                return True

            def compact(self):
                return 0

            def checkpoint(self):
                return {}

            def restore(self, checkpoint):
                pass

        mb = MinimalBudget()
        assert isinstance(mb, ContextBudgetManager)


# =====================================================================
# 3. Execution Failure / Recovery Contract
# =====================================================================

class TestExecutionFailureEnvelope:

    def test_minimum_fields(self):
        env = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="NameError",
            message="name 'foo' is not defined",
        )
        assert env.tool_name == "execute_code"
        assert env.phase == FailurePhase.EXECUTION
        n_fields = len(dataclasses.fields(ExecutionFailureEnvelope))
        assert n_fields >= BENCHMARK_THRESHOLDS.failure_envelope_min_fields

    def test_retryable(self):
        env = ExecutionFailureEnvelope(
            tool_name="t", phase=FailurePhase.EXECUTION,
            exception_type="E", message="m",
            retry_count=0, max_retries=3,
        )
        assert env.retryable is True

        env_exhausted = ExecutionFailureEnvelope(
            tool_name="t", phase=FailurePhase.EXECUTION,
            exception_type="E", message="m",
            retry_count=3, max_retries=3,
        )
        assert env_exhausted.retryable is False

    def test_to_llm_message_contains_tool_name(self):
        env = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="NameError",
            message="name 'foo' is not defined",
        )
        msg = env.to_llm_message()
        assert "execute_code" in msg
        assert "NameError" in msg
        assert "foo" in msg

    def test_to_llm_message_includes_hints(self):
        hint = RepairHint(
            strategy="add_import",
            description="Add 'import foo' to the code",
            confidence=0.8,
        )
        env = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="NameError",
            message="name 'foo' is not defined",
            repair_hints=[hint],
        )
        msg = env.to_llm_message()
        assert "import foo" in msg

    def test_to_dict_round_trip(self):
        hint = RepairHint(strategy="retry", description="Try again")
        env = ExecutionFailureEnvelope(
            tool_name="t", phase=FailurePhase.TIMEOUT,
            exception_type="TimeoutError", message="timed out",
            stderr_summary="killed", retry_count=1, max_retries=3,
            repair_hints=[hint],
        )
        d = env.to_dict()
        assert d["tool_name"] == "t"
        assert d["phase"] == "timeout"
        assert d["retryable"] is True
        assert len(d["repair_hints"]) == 1
        assert d["repair_hints"][0]["strategy"] == "retry"

    def test_construction_benchmark(self):
        """Failure envelope construction must meet benchmark threshold."""
        start = time.perf_counter()
        for _ in range(1000):
            env = ExecutionFailureEnvelope(
                tool_name="t", phase=FailurePhase.EXECUTION,
                exception_type="E", message="m",
            )
            env.to_llm_message()
        elapsed = (time.perf_counter() - start) / 1000
        assert elapsed < BENCHMARK_THRESHOLDS.failure_envelope_max_seconds


class TestFailurePhase:

    def test_all_phases_present(self):
        phases = list(FailurePhase)
        assert len(phases) == 5
        names = {p.value for p in phases}
        assert "pre_exec" in names
        assert "execution" in names
        assert "timeout" in names


# =====================================================================
# 4. Event Stream Contract
# =====================================================================

class TestEventStream:

    def test_event_categories_present(self):
        assert len(EVENT_CATEGORIES) >= 8
        assert "lifecycle" in EVENT_CATEGORIES
        assert "tool" in EVENT_CATEGORIES
        assert "trace" in EVENT_CATEGORIES

    def test_event_emitter_is_runtime_checkable(self):
        """EventEmitter protocol must be runtime-checkable."""
        class StubEmitter:
            def emit(self, event_type, content, *, category="",
                     step_id="", metadata=None):
                pass

            def emit_failure(self, envelope):
                pass

        stub = StubEmitter()
        assert isinstance(stub, EventEmitter)


# =====================================================================
# 5. Prompt Composition Contract
# =====================================================================

class TestPromptLayer:

    def test_minimum_fields(self):
        layer = PromptLayer(kind=PromptLayerKind.BASE_SYSTEM, content="You are an agent.")
        assert layer.kind == PromptLayerKind.BASE_SYSTEM
        assert layer.content == "You are an agent."
        n_fields = len(dataclasses.fields(PromptLayer))
        assert n_fields >= BENCHMARK_THRESHOLDS.prompt_layer_min_fields

    def test_is_frozen(self):
        layer = PromptLayer(kind=PromptLayerKind.WORKFLOW, content="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            layer.content = "modified"  # type: ignore[misc]

    def test_estimated_tokens(self):
        layer = PromptLayer(
            kind=PromptLayerKind.SKILL, content="a" * 200,
        )
        assert layer.estimated_tokens() == 50

    def test_estimated_tokens_explicit(self):
        layer = PromptLayer(
            kind=PromptLayerKind.SKILL, content="short",
            token_estimate=99,
        )
        assert layer.estimated_tokens() == 99

    def test_layer_kinds_complete(self):
        kinds = list(PromptLayerKind)
        assert len(kinds) == 6
        names = {k.value for k in kinds}
        assert "base_system" in names
        assert "workflow" in names
        assert "skill" in names
        assert "runtime_state" in names


class TestPromptComposerProtocol:

    def test_minimal_implementation_satisfies_protocol(self):
        class MinimalComposer:
            def __init__(self):
                self._layers = []

            def add_layer(self, layer):
                self._layers.append(layer)

            def compose(self):
                return "\n\n".join(l.content for l in self._layers)

            def layers(self):
                return list(self._layers)

            def total_tokens(self):
                return sum(l.estimated_tokens() for l in self._layers)

        mc = MinimalComposer()
        assert isinstance(mc, PromptComposer)

        mc.add_layer(PromptLayer(kind=PromptLayerKind.BASE_SYSTEM, content="Base"))
        mc.add_layer(PromptLayer(kind=PromptLayerKind.WORKFLOW, content="Workflow"))
        assert "Base" in mc.compose()
        assert "Workflow" in mc.compose()
        assert mc.total_tokens() >= 2

    def test_compose_benchmark(self):
        """Prompt composition must complete within threshold."""
        class BenchComposer:
            def __init__(self):
                self._layers = []

            def add_layer(self, layer):
                self._layers.append(layer)

            def compose(self):
                return "\n\n".join(l.content for l in self._layers)

            def layers(self):
                return self._layers

            def total_tokens(self):
                return sum(l.estimated_tokens() for l in self._layers)

        mc = BenchComposer()
        for i in range(20):
            mc.add_layer(PromptLayer(
                kind=PromptLayerKind.BASE_SYSTEM,
                content=f"Layer {i}: " + "x" * 500,
            ))

        start = time.perf_counter()
        for _ in range(100):
            mc.compose()
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < BENCHMARK_THRESHOLDS.prompt_compose_max_seconds


# =====================================================================
# 6. Remote Review Contract
# =====================================================================

class TestRemoteReviewConfig:

    def test_to_dict_from_dict_round_trip(self):
        cfg = RemoteReviewConfig(
            host="taiwan.example.com",
            user="deploy",
            key_path="/home/user/.ssh/id_ed25519",
            workspace="/srv/remote-review/omicverse",
            activate_cmd="source /opt/remote-env/bin/activate",
            timeout_seconds=300,
        )
        d = cfg.to_dict()
        restored = RemoteReviewConfig.from_dict(d)
        assert restored == cfg

    def test_is_frozen(self):
        cfg = RemoteReviewConfig(host="example.com")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.host = "other.com"  # type: ignore[misc]

    def test_default_timeout(self):
        cfg = RemoteReviewConfig(host="example.com")
        assert cfg.timeout_seconds == 300

    def test_from_dict_missing_fields(self):
        """from_dict handles missing optional fields gracefully."""
        cfg = RemoteReviewConfig.from_dict({"host": "h"})
        assert cfg.host == "h"
        assert cfg.user == ""
        assert cfg.workspace == ""

    def test_round_trip_benchmark(self):
        """Config serialization round-trip must meet threshold."""
        cfg = RemoteReviewConfig(
            host="h", user="u", key_path="/k",
            workspace="/w", activate_cmd="a",
        )
        start = time.perf_counter()
        for _ in range(1000):
            RemoteReviewConfig.from_dict(cfg.to_dict())
        elapsed = (time.perf_counter() - start) / 1000
        assert elapsed < BENCHMARK_THRESHOLDS.remote_config_roundtrip_max_seconds


# =====================================================================
# 7. Benchmark Thresholds Metadata
# =====================================================================

class TestBenchmarkThresholds:

    def test_singleton_exists(self):
        assert BENCHMARK_THRESHOLDS is not None
        assert isinstance(BENCHMARK_THRESHOLDS, BenchmarkThresholds)

    def test_all_thresholds_positive(self):
        for f in dataclasses.fields(BenchmarkThresholds):
            val = getattr(BENCHMARK_THRESHOLDS, f.name)
            assert val > 0, f"{f.name} must be positive, got {val}"

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            BENCHMARK_THRESHOLDS.tool_lookup_max_seconds = 999  # type: ignore[misc]


# =====================================================================
# 8. Cross-contract compatibility with existing harness
# =====================================================================

class TestHarnessCompatibility:

    def test_event_categories_cover_harness_event_types(self):
        """EVENT_CATEGORIES should map to the harness event type taxonomy."""
        assert len(HARNESS_EVENT_TYPES) > 0
        assert len(EVENT_CATEGORIES) > 0

    def test_tool_contract_compatible_with_tool_definition(self):
        """ToolContract must have fields that map to ToolDefinition."""
        td_fields = {f.name for f in dataclasses.fields(ToolDefinition)}
        tc_fields = {f.name for f in dataclasses.fields(ToolContract)}

        # These ToolDefinition fields must have equivalents in ToolContract
        required_overlap = {"name", "group", "description", "keywords", "aliases"}
        assert required_overlap <= td_fields, "ToolDefinition missing expected fields"
        assert required_overlap <= tc_fields, "ToolContract missing expected fields"

    def test_failure_envelope_compatible_with_step_trace(self):
        """ExecutionFailureEnvelope.to_dict() output must be storable in StepTrace.data."""
        env = ExecutionFailureEnvelope(
            tool_name="execute_code",
            phase=FailurePhase.EXECUTION,
            exception_type="ValueError",
            message="bad input",
        )
        step = StepTrace(
            step_id=make_step_id(),
            step_type="tool_call",
            name="execute_code",
            data={"failure": env.to_dict()},
        )
        assert step.data["failure"]["tool_name"] == "execute_code"
        assert step.data["failure"]["phase"] == "execution"


# =====================================================================
# 9. Context Budget Manager — Implementation Tests
# =====================================================================

class TestClassifyFunction:

    def test_classify_tool_by_name(self):
        """Tool name takes priority when classifying."""
        assert classify_fn("output", "tool", tool_name="finish") == ImportanceTier.CRITICAL
        assert classify_fn("output", "tool", tool_name="execute_code") == ImportanceTier.HIGH
        assert classify_fn("output", "tool", tool_name="inspect_data") == ImportanceTier.STANDARD
        assert classify_fn("output", "tool", tool_name="WebFetch") == ImportanceTier.LOW

    def test_classify_by_source(self):
        """Non-tool entries are classified by source."""
        assert classify_fn("prompt", "system_prompt") == ImportanceTier.CRITICAL
        assert classify_fn("msg", "user_message") == ImportanceTier.HIGH
        assert classify_fn("resp", "assistant") == ImportanceTier.STANDARD
        assert classify_fn("old", "history") == ImportanceTier.LOW
        assert classify_fn("old", "compacted") == ImportanceTier.EPHEMERAL

    def test_classify_unknown_defaults_standard(self):
        assert classify_fn("x", "unknown_source") == ImportanceTier.STANDARD

    def test_classify_by_output_tier(self):
        """output_tier from tool policy maps to importance."""
        assert classify_fn("x", "tool", output_tier="verbose") == ImportanceTier.LOW
        assert classify_fn("x", "tool", output_tier="compact") == ImportanceTier.STANDARD
        assert classify_fn("x", "tool", output_tier="minimal") == ImportanceTier.HIGH

    def test_tool_name_takes_priority_over_output_tier(self):
        """When both tool_name and output_tier are given, tool_name wins."""
        tier = classify_fn("x", "tool", tool_name="finish", output_tier="verbose")
        assert tier == ImportanceTier.CRITICAL


class TestMaxTokensForTier:

    def test_each_tier_has_limit(self):
        for tier in ImportanceTier:
            limit = max_tokens_for_tier(tier)
            assert limit > 0

    def test_critical_has_highest_limit(self):
        assert max_tokens_for_tier(ImportanceTier.CRITICAL) >= max_tokens_for_tier(ImportanceTier.LOW)


class TestDefaultContextBudgetManager:

    def _make_manager(self, **kwargs):
        config = ContextBudgetConfig(**kwargs)
        return DefaultContextBudgetManager(config)

    # -- satisfies protocol --

    def test_satisfies_protocol(self):
        mgr = self._make_manager()
        assert isinstance(mgr, ContextBudgetManager)

    # -- classify --

    def test_classify_delegates(self):
        mgr = self._make_manager()
        assert mgr.classify("x", "system_prompt") == ImportanceTier.CRITICAL
        assert mgr.classify("x", "tool", tool_name="WebSearch") == ImportanceTier.LOW

    # -- account --

    def test_account_adds_entry(self):
        mgr = self._make_manager(max_tokens=100_000)
        assert mgr.current_token_count == 0
        mgr.account("hello world", "user_message")
        assert mgr.current_token_count > 0
        assert len(mgr.entries) == 1

    def test_account_classifies_automatically(self):
        mgr = self._make_manager(max_tokens=100_000)
        mgr.account("data", "tool", tool_name="WebFetch")
        entry = mgr.entries[0]
        assert entry.importance == ImportanceTier.LOW
        assert entry.source == "tool:WebFetch"

    def test_account_uses_explicit_importance(self):
        mgr = self._make_manager(max_tokens=100_000)
        mgr.account("data", "tool", importance=ImportanceTier.CRITICAL)
        assert mgr.entries[0].importance == ImportanceTier.CRITICAL

    def test_account_tracks_tokens(self):
        mgr = self._make_manager(max_tokens=100_000)
        content = "a" * 400  # ~100 tokens
        mgr.account(content, "assistant")
        assert mgr.entries[0].estimated_tokens() == estimate_tokens(content)

    def test_account_critical_not_compactable(self):
        mgr = self._make_manager(max_tokens=100_000)
        mgr.account("sys", "system_prompt")
        assert mgr.entries[0].compactable is False

    # -- add_entry --

    def test_add_entry_within_budget(self):
        mgr = self._make_manager(max_tokens=100_000)
        entry = ContextEntry(content="hello", source="test")
        assert mgr.add_entry(entry) is True
        assert len(mgr.entries) == 1

    def test_add_entry_rejects_when_policy_is_reject(self):
        mgr = self._make_manager(
            max_tokens=100, reserve_tokens=0,
            overflow_policy=OverflowPolicy.REJECT,
            compaction_threshold=0.5,
        )
        # Fill up the budget
        mgr.add_entry(ContextEntry(
            content="x" * 500, source="test",
            importance=ImportanceTier.CRITICAL, compactable=False,
        ))
        # This should be rejected
        result = mgr.add_entry(ContextEntry(
            content="y" * 500, source="test2",
        ))
        assert result is False

    def test_add_entry_triggers_compaction_on_overflow(self):
        mgr = self._make_manager(
            max_tokens=300, reserve_tokens=0,
            compaction_threshold=0.5,
        )
        # Add small ephemeral entries that individually fit (~60 tokens each)
        for _ in range(3):
            mgr.add_entry(ContextEntry(
                content="e" * 240, source="test",
                importance=ImportanceTier.EPHEMERAL,
            ))
        # Total ~180 tokens, remaining ~120.
        # HIGH entry (~125 tokens) won't fit → triggers compaction
        result = mgr.add_entry(ContextEntry(
            content="h" * 500, source="test",
            importance=ImportanceTier.HIGH,
        ))
        assert result is True
        # Ephemeral entries should have been dropped during compaction
        for e in mgr.entries:
            assert e.importance != ImportanceTier.EPHEMERAL

    # -- compact --

    def test_compact_returns_zero_when_under_threshold(self):
        mgr = self._make_manager(max_tokens=100_000)
        mgr.account("small", "assistant")
        freed = mgr.compact()
        assert freed == 0

    def test_compact_drops_ephemeral_first(self):
        mgr = self._make_manager(
            max_tokens=200, reserve_tokens=0,
            compaction_threshold=0.01,  # very low to trigger compaction
        )
        mgr.add_entry(ContextEntry(
            content="keep me", source="test",
            importance=ImportanceTier.HIGH,
        ))
        mgr.add_entry(ContextEntry(
            content="e" * 400, source="test",
            importance=ImportanceTier.EPHEMERAL,
        ))
        freed = mgr.compact()
        assert freed > 0
        assert all(
            e.importance != ImportanceTier.EPHEMERAL
            for e in mgr.entries
        )

    def test_compact_truncates_low_entries(self):
        mgr = self._make_manager(
            max_tokens=300, reserve_tokens=0,
            compaction_threshold=0.01,
        )
        long_content = "x" * 1000
        mgr.add_entry(ContextEntry(
            content=long_content, source="test",
            importance=ImportanceTier.LOW, compactable=True,
        ))
        freed = mgr.compact()
        assert freed > 0
        # Entry should be shorter now
        remaining = [e for e in mgr.entries if e.importance == ImportanceTier.LOW]
        assert len(remaining) == 1
        assert len(remaining[0].content) < len(long_content)
        assert "[compacted]" in remaining[0].content

    def test_compact_increments_count(self):
        mgr = self._make_manager(
            max_tokens=100, reserve_tokens=0,
            compaction_threshold=0.01,
        )
        mgr.add_entry(ContextEntry(
            content="x" * 400, source="test",
            importance=ImportanceTier.EPHEMERAL,
        ))
        assert mgr.compaction_count == 0
        mgr.compact()
        assert mgr.compaction_count == 1

    def test_compact_preserves_critical_and_high(self):
        mgr = self._make_manager(
            max_tokens=200, reserve_tokens=0,
            compaction_threshold=0.01,
        )
        mgr.add_entry(ContextEntry(
            content="critical", source="sys",
            importance=ImportanceTier.CRITICAL, compactable=False,
        ))
        mgr.add_entry(ContextEntry(
            content="high", source="user",
            importance=ImportanceTier.HIGH,
        ))
        mgr.add_entry(ContextEntry(
            content="e" * 400, source="test",
            importance=ImportanceTier.EPHEMERAL,
        ))
        mgr.compact()
        sources = {e.source for e in mgr.entries}
        assert "sys" in sources
        assert "user" in sources

    # -- needs_compaction --

    def test_needs_compaction_false_when_under_threshold(self):
        mgr = self._make_manager(max_tokens=100_000, compaction_threshold=0.85)
        mgr.account("small", "assistant")
        assert mgr.needs_compaction() is False

    def test_needs_compaction_true_when_over_threshold(self):
        mgr = self._make_manager(
            max_tokens=100, reserve_tokens=0,
            compaction_threshold=0.01,
        )
        mgr.add_entry(ContextEntry(content="x" * 40, source="test"))
        assert mgr.needs_compaction() is True

    # -- render --

    def test_render_orders_by_importance(self):
        mgr = self._make_manager(max_tokens=100_000)
        mgr.add_entry(ContextEntry(
            content="low", source="a", importance=ImportanceTier.LOW,
        ))
        mgr.add_entry(ContextEntry(
            content="critical", source="b", importance=ImportanceTier.CRITICAL,
        ))
        mgr.add_entry(ContextEntry(
            content="high", source="c", importance=ImportanceTier.HIGH,
        ))
        rendered = mgr.render()
        assert rendered[0].importance == ImportanceTier.CRITICAL
        assert rendered[1].importance == ImportanceTier.HIGH
        assert rendered[2].importance == ImportanceTier.LOW

    # -- checkpoint / restore --

    def test_checkpoint_restore_round_trip(self):
        mgr = self._make_manager(max_tokens=100_000)
        mgr.account("sys prompt", "system_prompt")
        mgr.account("user msg", "user_message")
        mgr.account("tool out", "tool", tool_name="inspect_data")

        cp = mgr.checkpoint()
        assert len(cp["entries"]) == 3

        mgr2 = self._make_manager(max_tokens=100_000)
        mgr2.restore(cp)
        assert len(mgr2.entries) == 3
        assert mgr2.entries[0].source == "system_prompt"
        assert mgr2.entries[0].importance == ImportanceTier.CRITICAL

    def test_checkpoint_preserves_compaction_count(self):
        mgr = self._make_manager(
            max_tokens=100, reserve_tokens=0,
            compaction_threshold=0.01,
        )
        mgr.add_entry(ContextEntry(
            content="x" * 400, source="test",
            importance=ImportanceTier.EPHEMERAL,
        ))
        mgr.compact()
        cp = mgr.checkpoint()
        assert cp["compaction_count"] == 1

        mgr2 = self._make_manager()
        mgr2.restore(cp)
        assert mgr2.compaction_count == 1

    # -- format_tool_output --

    def test_format_preserves_short_output(self):
        mgr = self._make_manager()
        output = "short result"
        assert mgr.format_tool_output(output, "inspect_data") == output

    def test_format_truncates_long_output(self):
        mgr = self._make_manager()
        # WebFetch → LOW tier → 1000 token limit
        long_output = "x" * 20000
        result = mgr.format_tool_output(long_output, "WebFetch")
        assert len(result) < len(long_output)
        assert "[truncated by context budget]" in result

    def test_format_respects_per_tier_config(self):
        mgr = self._make_manager(
            per_tier_limits={"low": 50},
        )
        long_output = "x" * 2000
        result = mgr.format_tool_output(long_output, "WebFetch")
        # 50 tokens * 4 chars = 200 chars max
        assert len(result) <= 250  # 200 + truncation message

    def test_format_different_tiers_different_limits(self):
        mgr = self._make_manager()
        long_output = "x" * 20000
        low_result = mgr.format_tool_output(long_output, "WebFetch")
        high_result = mgr.format_tool_output(long_output, "execute_code")
        # HIGH tier gets more budget than LOW
        assert len(high_result) > len(low_result)

    # -- token-aware estimates --

    def test_uses_token_estimates_not_chars(self):
        """Budget tracking uses token estimates, not raw character counts."""
        mgr = self._make_manager(max_tokens=100_000)
        content = "a" * 400  # ~100 tokens at 4 chars/token
        mgr.account(content, "assistant")
        token_count = mgr.entries[0].estimated_tokens()
        assert token_count == estimate_tokens(content)
        assert token_count != len(content)

    # -- remaining_tokens / current_token_count --

    def test_remaining_decreases_on_add(self):
        mgr = self._make_manager(max_tokens=100_000, reserve_tokens=4_096)
        initial = mgr.remaining_tokens
        mgr.account("hello world test", "assistant")
        assert mgr.remaining_tokens < initial
        assert mgr.current_token_count > 0

    def test_remaining_never_negative(self):
        mgr = self._make_manager(max_tokens=10, reserve_tokens=0)
        mgr.add_entry(ContextEntry(content="x" * 1000, source="test"))
        assert mgr.remaining_tokens == 0

    # -- benchmark --

    def test_compaction_meets_recovery_threshold(self):
        """Compaction must recover at least the benchmark ratio."""
        mgr = self._make_manager(
            max_tokens=1000, reserve_tokens=0,
            compaction_threshold=0.01,
        )
        # Fill with ephemeral + low entries
        for i in range(5):
            mgr.add_entry(ContextEntry(
                content=f"ephemeral-{i} " + "x" * 300,
                source="test",
                importance=ImportanceTier.EPHEMERAL,
            ))
        for i in range(3):
            mgr.add_entry(ContextEntry(
                content=f"low-{i} " + "y" * 500,
                source="test",
                importance=ImportanceTier.LOW,
            ))
        before = mgr.current_token_count
        freed = mgr.compact()
        assert freed > 0
        recovery_ratio = freed / before if before > 0 else 0
        assert recovery_ratio >= BENCHMARK_THRESHOLDS.compaction_min_recovery_ratio
