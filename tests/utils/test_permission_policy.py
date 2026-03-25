"""Tests for subagent isolation and per-tool permission policy (task-029).

Covers:
  - PermissionPolicy verdict resolution (allow/ask/deny)
  - Per-tool and per-class overrides
  - Allowlist and denylist enforcement
  - Unknown tool fallback
  - Subagent policy factory (create_subagent_policy)
  - SubagentRuntime isolation properties
  - dispatch_tool permission_policy integration
  - Isolation: subagent does NOT mutate parent state
  - High-risk tools and isolation modes are explicit in metadata
"""

import asyncio
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Guard: require harness env-var
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Permission policy tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

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

from omicverse.utils.ovagent import (
    PermissionDecision,
    PermissionPolicy,
    PermissionVerdict,
    SubagentController,
    SubagentRuntime,
    ToolRuntime,
    create_default_policy,
    create_subagent_policy,
)
from omicverse.utils.ovagent.permission_policy import (
    PermissionDecision as _PD,
    PermissionPolicy as _PP,
    PermissionVerdict as _PV,
)
from omicverse.utils.ovagent.tool_registry import (
    ApprovalClass,
    IsolationMode,
    OutputTier,
    ParallelClass,
    ToolMetadata,
    ToolRegistry,
    build_default_registry,
)
from omicverse.utils.ovagent.tool_runtime import LEGACY_AGENT_TOOLS
from omicverse.utils.ovagent.context_budget import create_subagent_budget_manager

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------

class _DummyCtx:
    """Minimal AgentContext stub."""

    LEGACY_AGENT_TOOLS = LEGACY_AGENT_TOOLS

    def __init__(self, session_id="test-perm-session"):
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
        self._web_session_id = session_id
        self._managed_api_env = {}
        self._code_only_mode = False
        self._code_only_captured_code = ""
        self._code_only_captured_history = []
        self.use_notebook_execution = False
        self.enable_filesystem_context = False

    def _get_runtime_session_id(self):
        return self._web_session_id

    def _emit(self, level, message, category=""):
        pass

    def _get_harness_session_id(self):
        return self._web_session_id

    def _get_visible_agent_tools(self, *, allowed_names=None):
        return []

    def _get_loaded_tool_names(self):
        return []

    def _refresh_runtime_working_directory(self):
        return "."

    def _tool_blocked_in_plan_mode(self, tool_name):
        return False

    def _detect_repo_root(self, cwd=None):
        return None

    def _resolve_local_path(self, file_path, *, allow_relative=False):
        raise NotImplementedError

    def _ensure_server_tool_mode(self, tool_name):
        pass

    def _request_interaction(self, payload):
        raise NotImplementedError

    def _request_tool_approval(self, tool_name, *, reason, payload):
        raise NotImplementedError

    def _load_skill_guidance(self, slug):
        return ""

    def _extract_python_code(self, text):
        return text

    def _normalize_registry_entry_for_codegen(self, entry):
        return entry

    def _build_agentic_system_prompt(self):
        return "test"

    async def _run_agentic_loop(self, request, adata, **kwargs):
        return adata

    @contextmanager
    def _temporary_api_keys(self):
        yield


class _DummyExecutor:
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def default_policy(registry):
    return create_default_policy(registry)


# ===================================================================
# 1. PermissionVerdict enum
# ===================================================================

class TestPermissionVerdict:

    def test_verdict_values(self):
        assert PermissionVerdict.allow == "allow"
        assert PermissionVerdict.ask == "ask"
        assert PermissionVerdict.deny == "deny"

    def test_verdict_is_string_enum(self):
        assert isinstance(PermissionVerdict.allow, str)


# ===================================================================
# 2. PermissionDecision
# ===================================================================

class TestPermissionDecision:

    def test_is_allowed(self):
        d = PermissionDecision(
            verdict=PermissionVerdict.allow, tool_name="Read"
        )
        assert d.is_allowed is True
        assert d.is_denied is False

    def test_is_denied(self):
        d = PermissionDecision(
            verdict=PermissionVerdict.deny, tool_name="Bash"
        )
        assert d.is_denied is True
        assert d.is_allowed is False

    def test_ask_is_neither_allowed_nor_denied(self):
        d = PermissionDecision(
            verdict=PermissionVerdict.ask, tool_name="Edit"
        )
        assert d.is_allowed is False
        assert d.is_denied is False

    def test_to_dict(self):
        d = PermissionDecision(
            verdict=PermissionVerdict.deny,
            tool_name="Bash",
            reason="test reason",
            requires_isolation=IsolationMode.sandbox,
        )
        as_dict = d.to_dict()
        assert as_dict["verdict"] == "deny"
        assert as_dict["tool_name"] == "Bash"
        assert as_dict["reason"] == "test reason"
        assert as_dict["requires_isolation"] == "sandbox"

    def test_frozen(self):
        d = PermissionDecision(
            verdict=PermissionVerdict.allow, tool_name="Read"
        )
        with pytest.raises(AttributeError):
            d.verdict = PermissionVerdict.deny  # type: ignore[misc]


# ===================================================================
# 3. PermissionPolicy — registry defaults
# ===================================================================

class TestPermissionPolicyRegistryDefaults:

    def test_readonly_tools_are_allowed(self, default_policy):
        for name in ["Read", "Glob", "Grep", "ToolSearch"]:
            decision = default_policy.check(name)
            assert decision.verdict == PermissionVerdict.allow, (
                f"Expected {name} to be allowed"
            )

    def test_high_risk_tools_are_ask(self, default_policy):
        for name in ["Bash", "Edit", "Write"]:
            decision = default_policy.check(name)
            assert decision.verdict == PermissionVerdict.ask, (
                f"Expected {name} to be ask"
            )

    def test_execute_code_requires_sandbox(self, default_policy):
        decision = default_policy.check("execute_code")
        assert decision.verdict == PermissionVerdict.ask
        assert decision.requires_isolation == IsolationMode.sandbox

    def test_unknown_tool_is_denied(self, default_policy):
        decision = default_policy.check("nonexistent_tool_xyz")
        assert decision.is_denied
        assert "unknown tool" in decision.reason

    def test_registry_property(self, default_policy, registry):
        assert default_policy.registry is registry

    def test_reason_populated(self, default_policy):
        decision = default_policy.check("Read")
        assert decision.reason != ""


# ===================================================================
# 4. PermissionPolicy — deny list
# ===================================================================

class TestPermissionPolicyDenyList:

    def test_denied_tool_overrides_registry(self, registry):
        policy = PermissionPolicy(
            registry,
            denied_tools=frozenset({"Read"}),
        )
        decision = policy.check("Read")
        assert decision.is_denied
        assert "deny list" in decision.reason

    def test_denied_tool_overrides_allowlist(self, registry):
        policy = PermissionPolicy(
            registry,
            denied_tools=frozenset({"Read"}),
            allowed_tools=frozenset({"Read", "Glob"}),
        )
        decision = policy.check("Read")
        assert decision.is_denied


# ===================================================================
# 5. PermissionPolicy — allowlist
# ===================================================================

class TestPermissionPolicyAllowlist:

    def test_tool_not_in_allowlist_is_denied(self, registry):
        policy = PermissionPolicy(
            registry,
            allowed_tools=frozenset({"Read", "Glob"}),
        )
        decision = policy.check("Bash")
        assert decision.is_denied
        assert "not in the allowed set" in decision.reason

    def test_tool_in_allowlist_passes_through(self, registry):
        policy = PermissionPolicy(
            registry,
            allowed_tools=frozenset({"Read", "Glob"}),
        )
        decision = policy.check("Read")
        assert decision.is_allowed

    def test_none_allowlist_means_all_allowed(self, registry):
        policy = PermissionPolicy(registry, allowed_tools=None)
        decision = policy.check("Read")
        assert decision.is_allowed


# ===================================================================
# 6. PermissionPolicy — per-tool overrides
# ===================================================================

class TestPermissionPolicyToolOverrides:

    def test_per_tool_override_changes_verdict(self, registry):
        policy = PermissionPolicy(
            registry,
            tool_overrides={"Read": ApprovalClass.deny},
        )
        decision = policy.check("Read")
        assert decision.is_denied
        assert "per-tool override" in decision.reason

    def test_per_tool_override_to_allow(self, registry):
        policy = PermissionPolicy(
            registry,
            tool_overrides={"Bash": ApprovalClass.allow},
        )
        decision = policy.check("Bash")
        assert decision.is_allowed


# ===================================================================
# 7. PermissionPolicy — per-class overrides
# ===================================================================

class TestPermissionPolicyClassOverrides:

    def test_class_override_affects_all_tools_in_class(self, registry):
        policy = PermissionPolicy(
            registry,
            class_overrides={ApprovalClass.ask: PermissionVerdict.deny},
        )
        # Bash has ApprovalClass.ask in registry
        decision = policy.check("Bash")
        assert decision.is_denied
        assert "class override" in decision.reason

    def test_class_override_does_not_affect_other_classes(self, registry):
        policy = PermissionPolicy(
            registry,
            class_overrides={ApprovalClass.ask: PermissionVerdict.deny},
        )
        # Read has ApprovalClass.allow — unaffected
        decision = policy.check("Read")
        assert decision.is_allowed


# ===================================================================
# 8. PermissionPolicy — unknown tool default
# ===================================================================

class TestPermissionPolicyUnknownDefault:

    def test_unknown_default_deny(self, registry):
        policy = PermissionPolicy(
            registry,
            unknown_tool_default=PermissionVerdict.deny,
        )
        decision = policy.check("imaginary_tool")
        assert decision.is_denied

    def test_unknown_default_ask(self, registry):
        policy = PermissionPolicy(
            registry,
            unknown_tool_default=PermissionVerdict.ask,
        )
        decision = policy.check("imaginary_tool")
        assert decision.verdict == PermissionVerdict.ask


# ===================================================================
# 9. Policy summary
# ===================================================================

class TestPermissionPolicySummary:

    def test_summary_includes_all_registered_tools(self, default_policy, registry):
        summary = default_policy.summary()
        for entry in registry.all_entries():
            assert entry.canonical_name in summary

    def test_summary_values_are_strings(self, default_policy):
        summary = default_policy.summary()
        for v in summary.values():
            assert isinstance(v, str)


# ===================================================================
# 10. create_subagent_policy factory
# ===================================================================

class TestCreateSubagentPolicy:

    def test_only_allowed_tools_pass(self, registry):
        policy = create_subagent_policy(
            registry,
            allowed_tools=frozenset({"inspect_data", "finish"}),
        )
        assert policy.check("inspect_data").is_allowed
        assert policy.check("finish").is_allowed
        assert policy.check("Bash").is_denied
        assert policy.check("execute_code").is_denied

    def test_unknown_tools_denied(self, registry):
        policy = create_subagent_policy(
            registry,
            allowed_tools=frozenset({"Read"}),
        )
        assert policy.check("nonexistent").is_denied

    def test_explore_subagent_tool_set(self, registry):
        explore_tools = frozenset({
            "inspect_data", "run_snippet", "search_functions",
            "web_fetch", "web_search", "finish",
        })
        policy = create_subagent_policy(registry, allowed_tools=explore_tools)
        for tool in explore_tools:
            decision = policy.check(tool)
            assert not decision.is_denied, f"{tool} should not be denied"
        # execute_code NOT in explore set
        assert policy.check("execute_code").is_denied
        assert policy.check("Bash").is_denied


# ===================================================================
# 11. create_default_policy factory
# ===================================================================

class TestCreateDefaultPolicy:

    def test_returns_permission_policy(self, registry):
        policy = create_default_policy(registry)
        assert isinstance(policy, PermissionPolicy)

    def test_default_policy_matches_registry_metadata(self, registry):
        policy = create_default_policy(registry)
        for entry in registry.all_entries():
            decision = policy.check(entry.canonical_name)
            expected = PermissionVerdict(entry.approval_class.value)
            assert decision.verdict == expected, (
                f"{entry.canonical_name}: expected {expected}, got {decision.verdict}"
            )


# ===================================================================
# 12. SubagentRuntime isolation
# ===================================================================

class TestSubagentRuntime:

    def _make_runtime(self, registry, allowed=None):
        allowed = allowed or frozenset({"inspect_data", "finish"})
        policy = create_subagent_policy(registry, allowed_tools=allowed)
        budget = create_subagent_budget_manager(model="test")
        return SubagentRuntime(
            agent_type="explore",
            max_turns=5,
            permission_policy=policy,
            budget_manager=budget,
            tool_schemas=[{"name": "inspect_data"}, {"name": "finish"}],
        )

    def test_construction(self, registry):
        rt = self._make_runtime(registry)
        assert rt.agent_type == "explore"
        assert rt.max_turns == 5
        assert rt.last_usage is None

    def test_record_usage_is_local(self, registry):
        rt = self._make_runtime(registry)
        rt.record_usage({"prompt_tokens": 100, "completion_tokens": 50})
        assert rt.last_usage == {"prompt_tokens": 100, "completion_tokens": 50}

    def test_check_tool_permission_allowed(self, registry):
        rt = self._make_runtime(registry)
        decision = rt.check_tool_permission("inspect_data")
        assert not decision.is_denied

    def test_check_tool_permission_denied(self, registry):
        rt = self._make_runtime(registry)
        decision = rt.check_tool_permission("Bash")
        assert decision.is_denied

    def test_can_mutate_adata_default_false(self, registry):
        rt = self._make_runtime(registry)
        assert rt.can_mutate_adata is False

    def test_tool_schemas_are_snapshot(self, registry):
        rt = self._make_runtime(registry)
        original_len = len(rt.tool_schemas)
        rt.tool_schemas.append({"name": "injected"})
        assert len(rt.tool_schemas) == original_len + 1
        # A new runtime gets a fresh list
        rt2 = self._make_runtime(registry)
        assert len(rt2.tool_schemas) == original_len


# ===================================================================
# 13. dispatch_tool with permission_policy
# ===================================================================

class TestDispatchToolPermission:

    def _make_runtime(self):
        ctx = _DummyCtx()
        rt = ToolRuntime(ctx, _DummyExecutor())
        return rt

    def test_dispatch_without_policy_works(self):
        rt = self._make_runtime()
        tc = SimpleNamespace(name="finish", id="tc-1", arguments={"summary": "done"})
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        # finish handler returns a dict
        assert isinstance(result, (str, dict))

    def test_dispatch_with_allow_policy_works(self):
        rt = self._make_runtime()
        policy = create_default_policy(rt.registry)
        tc = SimpleNamespace(name="finish", id="tc-1", arguments={"summary": "done"})
        result = asyncio.run(
            rt.dispatch_tool(tc, None, "test", permission_policy=policy)
        )
        assert isinstance(result, (str, dict))

    def test_dispatch_with_deny_policy_blocks(self):
        rt = self._make_runtime()
        policy = PermissionPolicy(
            rt.registry,
            denied_tools=frozenset({"finish"}),
        )
        tc = SimpleNamespace(name="finish", id="tc-1", arguments={"summary": "done"})
        result = asyncio.run(
            rt.dispatch_tool(tc, None, "test", permission_policy=policy)
        )
        assert isinstance(result, str)
        assert "Permission denied" in result

    def test_dispatch_subagent_policy_restricts_tools(self):
        rt = self._make_runtime()
        policy = create_subagent_policy(
            rt.registry,
            allowed_tools=frozenset({"inspect_data"}),
        )
        # finish not in allowed set → denied
        tc = SimpleNamespace(name="finish", id="tc-1", arguments={"summary": "done"})
        result = asyncio.run(
            rt.dispatch_tool(tc, None, "test", permission_policy=policy)
        )
        assert "Permission denied" in result


# ===================================================================
# 14. Subagent does NOT mutate parent state
# ===================================================================

class TestSubagentIsolationFromParent:

    def test_subagent_runtime_does_not_share_parent_usage(self, registry):
        ctx = _DummyCtx()
        ctx.last_usage = {"prompt_tokens": 999}

        rt = SubagentRuntime(
            agent_type="explore",
            max_turns=3,
            permission_policy=create_subagent_policy(
                registry, frozenset({"finish"})
            ),
            budget_manager=create_subagent_budget_manager(model="test"),
        )
        rt.record_usage({"prompt_tokens": 42})

        # Parent unchanged
        assert ctx.last_usage == {"prompt_tokens": 999}
        # Subagent has its own
        assert rt.last_usage == {"prompt_tokens": 42}

    def test_subagent_controller_creates_isolated_runtime(self, registry):
        ctx = _DummyCtx()
        from omicverse.utils.ovagent.prompt_builder import PromptBuilder

        pb = PromptBuilder(ctx)
        rt = ToolRuntime(ctx, _DummyExecutor())
        sc = SubagentController(ctx, pb, rt)

        subagent_rt = sc._create_subagent_runtime(
            agent_type="explore",
            allowed_tools=frozenset({"inspect_data", "finish"}),
            max_turns=5,
        )
        assert isinstance(subagent_rt, SubagentRuntime)
        assert subagent_rt.agent_type == "explore"
        assert subagent_rt.max_turns == 5
        assert subagent_rt.can_mutate_adata is False
        assert subagent_rt.last_usage is None


# ===================================================================
# 15. High-risk tools and isolation modes are explicit in metadata
# ===================================================================

class TestHighRiskToolMetadata:

    def test_execute_code_is_ask_and_sandbox(self, registry):
        meta = registry.get("execute_code")
        assert meta is not None
        assert meta.approval_class == ApprovalClass.ask
        assert meta.isolation_mode == IsolationMode.sandbox

    def test_bash_is_ask_and_sandbox(self, registry):
        meta = registry.get("Bash")
        assert meta is not None
        assert meta.approval_class == ApprovalClass.ask
        assert meta.isolation_mode == IsolationMode.sandbox

    def test_enter_worktree_is_ask_and_worktree(self, registry):
        meta = registry.get("EnterWorktree")
        assert meta is not None
        assert meta.approval_class == ApprovalClass.ask
        assert meta.isolation_mode == IsolationMode.worktree

    def test_read_only_tools_have_no_isolation(self, registry):
        for name in ["Read", "Glob", "Grep", "ToolSearch"]:
            meta = registry.get(name)
            assert meta is not None
            assert meta.isolation_mode == IsolationMode.none, (
                f"{name} should have no isolation requirement"
            )

    def test_all_registered_tools_have_explicit_isolation(self, registry):
        for entry in registry.all_entries():
            assert isinstance(entry.isolation_mode, IsolationMode), (
                f"{entry.canonical_name} missing IsolationMode"
            )
            assert isinstance(entry.approval_class, ApprovalClass), (
                f"{entry.canonical_name} missing ApprovalClass"
            )


# ===================================================================
# 16. Exports contract
# ===================================================================

class TestExportsContract:

    def test_permission_types_in_ovagent_all(self):
        import omicverse.utils.ovagent as pkg
        all_names = set(pkg.__all__)
        expected = {
            "PermissionDecision", "PermissionPolicy", "PermissionVerdict",
            "SubagentRuntime",
            "create_default_policy", "create_subagent_policy",
        }
        missing = expected - all_names
        assert not missing, f"Missing from __all__: {missing}"

    def test_permission_types_importable(self):
        from omicverse.utils.ovagent import (
            PermissionDecision,
            PermissionPolicy,
            PermissionVerdict,
            SubagentRuntime,
            create_default_policy,
            create_subagent_policy,
        )
        assert PermissionDecision is not None
        assert PermissionPolicy is not None
        assert PermissionVerdict is not None
        assert SubagentRuntime is not None
        assert create_default_policy is not None
        assert create_subagent_policy is not None


# ===================================================================
# 17. Priority ordering: deny > allowlist > override > class > default
# ===================================================================

class TestPriorityOrdering:

    def test_deny_beats_everything(self, registry):
        policy = PermissionPolicy(
            registry,
            denied_tools=frozenset({"Read"}),
            allowed_tools=frozenset({"Read"}),
            tool_overrides={"Read": ApprovalClass.allow},
        )
        assert policy.check("Read").is_denied

    def test_allowlist_beats_override(self, registry):
        policy = PermissionPolicy(
            registry,
            allowed_tools=frozenset({"Glob"}),
            tool_overrides={"Bash": ApprovalClass.allow},
        )
        # Bash not in allowlist → denied despite per-tool allow override
        assert policy.check("Bash").is_denied

    def test_tool_override_beats_class_override(self, registry):
        policy = PermissionPolicy(
            registry,
            tool_overrides={"Bash": ApprovalClass.allow},
            class_overrides={ApprovalClass.ask: PermissionVerdict.deny},
        )
        decision = policy.check("Bash")
        assert decision.is_allowed

    def test_class_override_beats_registry_default(self, registry):
        policy = PermissionPolicy(
            registry,
            class_overrides={ApprovalClass.allow: PermissionVerdict.ask},
        )
        decision = policy.check("Read")
        assert decision.verdict == PermissionVerdict.ask
