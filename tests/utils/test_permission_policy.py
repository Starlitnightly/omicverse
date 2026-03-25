"""
Tests for subagent isolation and per-tool permission policy (task-014).

Covers:
  - PermissionPolicy allow/ask/deny semantics
  - ToolPermissionRule per-tool overrides
  - IsolatedSubagentContext read-only snapshot
  - SubagentResult structured handoff
  - build_subagent_policy factory
  - ToolRuntime permission gate integration
  - SubagentController isolated path dispatch

No LLM calls, no heavy imports, no server requirement.
"""

import asyncio
import dataclasses
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load modules in isolation (same pattern as test_runtime_contracts)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

_STUB_NAMES = [
    "omicverse", "omicverse.utils", "omicverse.utils.ovagent",
    "omicverse.utils.harness",
]
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


# Load contracts (stdlib only)
contracts_mod = _load_module(
    "omicverse.utils.ovagent.contracts",
    PACKAGE_ROOT / "utils" / "ovagent" / "contracts.py",
)

# Load permission_policy
permission_policy_mod = _load_module(
    "omicverse.utils.ovagent.permission_policy",
    PACKAGE_ROOT / "utils" / "ovagent" / "permission_policy.py",
)

# Restore original modules
for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


# Pull symbols
ApprovalClass = contracts_mod.ApprovalClass
IsolationMode = contracts_mod.IsolationMode

PermissionPolicy = permission_policy_mod.PermissionPolicy
PermissionDecision = permission_policy_mod.PermissionDecision
PermissionVerdict = permission_policy_mod.PermissionVerdict
ToolPermissionRule = permission_policy_mod.ToolPermissionRule
build_subagent_policy = permission_policy_mod.build_subagent_policy
_resolve_legacy_approval = permission_policy_mod._resolve_legacy_approval
_approval_to_verdict = permission_policy_mod._approval_to_verdict
_resolve_isolation = permission_policy_mod._resolve_isolation


# =====================================================================
# 1. PermissionVerdict enum
# =====================================================================

class TestPermissionVerdict:

    def test_values_match_approval_class(self):
        """PermissionVerdict values mirror ApprovalClass values."""
        assert PermissionVerdict.ALLOW.value == "allow"
        assert PermissionVerdict.ASK.value == "ask"
        assert PermissionVerdict.DENY.value == "deny"

    def test_all_members(self):
        assert len(PermissionVerdict) == 3


# =====================================================================
# 2. PermissionDecision
# =====================================================================

class TestPermissionDecision:

    def test_is_frozen(self):
        d = PermissionDecision(
            verdict=PermissionVerdict.ALLOW, tool_name="inspect_data",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.verdict = PermissionVerdict.DENY  # type: ignore[misc]

    def test_default_isolation_is_in_process(self):
        d = PermissionDecision(
            verdict=PermissionVerdict.ALLOW, tool_name="test",
        )
        assert d.required_isolation == IsolationMode.IN_PROCESS

    def test_custom_fields(self):
        d = PermissionDecision(
            verdict=PermissionVerdict.DENY,
            tool_name="Bash",
            reason="high risk tool",
            required_isolation=IsolationMode.SUBPROCESS,
        )
        assert d.verdict == PermissionVerdict.DENY
        assert d.tool_name == "Bash"
        assert "high risk" in d.reason
        assert d.required_isolation == IsolationMode.SUBPROCESS


# =====================================================================
# 3. ToolPermissionRule
# =====================================================================

class TestToolPermissionRule:

    def test_is_frozen(self):
        rule = ToolPermissionRule(tool_name="Bash")
        with pytest.raises(dataclasses.FrozenInstanceError):
            rule.tool_name = "Edit"  # type: ignore[misc]

    def test_defaults_to_allow(self):
        rule = ToolPermissionRule(tool_name="inspect_data")
        assert rule.approval == ApprovalClass.ALLOW

    def test_custom_approval(self):
        rule = ToolPermissionRule(
            tool_name="execute_code",
            approval=ApprovalClass.ASK,
            required_isolation=IsolationMode.SUBPROCESS,
            reason="code execution requires review",
        )
        assert rule.approval == ApprovalClass.ASK
        assert rule.required_isolation == IsolationMode.SUBPROCESS


# =====================================================================
# 4. PermissionPolicy — core check logic
# =====================================================================

class TestPermissionPolicyCheck:

    def test_default_allows_everything(self):
        """Default policy with no rules allows all tools."""
        policy = PermissionPolicy()
        d = policy.check("inspect_data")
        assert d.verdict == PermissionVerdict.ALLOW

    def test_global_deny_list(self):
        """Tools in the deny list are always denied."""
        policy = PermissionPolicy(denied_tools=frozenset({"Bash", "Write"}))
        d = policy.check("Bash")
        assert d.verdict == PermissionVerdict.DENY
        assert "deny list" in d.reason

        # Non-denied tools still pass
        d2 = policy.check("inspect_data")
        assert d2.verdict == PermissionVerdict.ALLOW

    def test_allowlist_restricts_scope(self):
        """When allowed_tools is set, unlisted tools are denied."""
        policy = PermissionPolicy(
            allowed_tools=frozenset({"inspect_data", "finish"}),
        )
        d = policy.check("inspect_data")
        assert d.verdict == PermissionVerdict.ALLOW

        d2 = policy.check("execute_code")
        assert d2.verdict == PermissionVerdict.DENY
        assert "not in the allowed set" in d2.reason

    def test_per_tool_rule_override(self):
        """Per-tool rules take precedence over policy metadata."""
        rule = ToolPermissionRule(
            tool_name="execute_code",
            approval=ApprovalClass.ASK,
            reason="needs human review",
        )
        policy = PermissionPolicy(rules=[rule])
        d = policy.check("execute_code")
        assert d.verdict == PermissionVerdict.ASK
        assert "needs human review" in d.reason

    def test_tool_policy_metadata_fallback(self):
        """When no rule exists, tool_approval_class is consulted."""
        policy = PermissionPolicy()
        d = policy.check("Bash", tool_approval_class="high_risk")
        assert d.verdict == PermissionVerdict.DENY

        d2 = policy.check("Read", tool_approval_class="none")
        assert d2.verdict == PermissionVerdict.ALLOW

        d3 = policy.check("Edit", tool_approval_class="standard")
        assert d3.verdict == PermissionVerdict.ASK

    def test_deny_takes_priority_over_allowlist(self):
        """A tool in both allowed and denied is denied."""
        policy = PermissionPolicy(
            allowed_tools=frozenset({"Bash", "inspect_data"}),
            denied_tools=frozenset({"Bash"}),
        )
        d = policy.check("Bash")
        assert d.verdict == PermissionVerdict.DENY

    def test_rule_overrides_policy_metadata(self):
        """Explicit rule wins over tool_approval_class."""
        rule = ToolPermissionRule(
            tool_name="Bash",
            approval=ApprovalClass.ALLOW,
            reason="explicitly allowed for this context",
        )
        policy = PermissionPolicy(rules=[rule])
        # Even though we pass high_risk, the rule takes precedence
        d = policy.check("Bash", tool_approval_class="high_risk")
        assert d.verdict == PermissionVerdict.ALLOW

    def test_default_approval_as_fallback(self):
        """When nothing else matches, the default_approval is used."""
        policy = PermissionPolicy(default_approval=ApprovalClass.ASK)
        d = policy.check("unknown_tool")
        assert d.verdict == PermissionVerdict.ASK

    def test_isolation_from_tool_policy(self):
        """isolation_mode from tool policy metadata propagates."""
        policy = PermissionPolicy()
        d = policy.check(
            "execute_code",
            tool_approval_class="none",
            tool_isolation_mode="subprocess",
        )
        assert d.required_isolation == IsolationMode.SUBPROCESS


# =====================================================================
# 5. PermissionPolicy — batch check
# =====================================================================

class TestPermissionPolicyBatch:

    def test_check_batch(self):
        policy = PermissionPolicy(denied_tools=frozenset({"Bash"}))
        results = policy.check_batch(
            ["inspect_data", "Bash", "finish"],
            policies={
                "inspect_data": {"approval_class": "none"},
                "Bash": {"approval_class": "high_risk"},
            },
        )
        assert results["inspect_data"].verdict == PermissionVerdict.ALLOW
        assert results["Bash"].verdict == PermissionVerdict.DENY
        assert results["finish"].verdict == PermissionVerdict.ALLOW

    def test_check_batch_empty(self):
        policy = PermissionPolicy()
        assert policy.check_batch([]) == {}


# =====================================================================
# 6. Factory methods
# =====================================================================

class TestPolicyFactories:

    def test_for_subagent(self):
        policy = PermissionPolicy.for_subagent(
            frozenset({"inspect_data", "finish"}),
        )
        assert policy.allowed_tools == frozenset({"inspect_data", "finish"})
        d = policy.check("inspect_data")
        assert d.verdict == PermissionVerdict.ALLOW
        d2 = policy.check("execute_code")
        assert d2.verdict == PermissionVerdict.DENY

    def test_permissive(self):
        policy = PermissionPolicy.permissive()
        d = policy.check("anything")
        assert d.verdict == PermissionVerdict.ALLOW

    def test_restrictive(self):
        policy = PermissionPolicy.restrictive()
        d = policy.check("anything")
        assert d.verdict == PermissionVerdict.ASK

    def test_build_subagent_policy_basic(self):
        policy = build_subagent_policy(["inspect_data", "finish"])
        assert policy.allowed_tools == frozenset({"inspect_data", "finish"})
        d = policy.check("inspect_data")
        assert d.verdict == PermissionVerdict.ALLOW
        d2 = policy.check("Bash")
        assert d2.verdict == PermissionVerdict.DENY

    def test_build_subagent_policy_deny_mutations(self):
        policy = build_subagent_policy(
            ["inspect_data", "execute_code", "finish"],
            deny_mutations=True,
        )
        # execute_code is in allowed list but also denied by mutation policy
        d = policy.check("execute_code")
        assert d.verdict == PermissionVerdict.DENY

        # inspect_data still allowed
        d2 = policy.check("inspect_data")
        assert d2.verdict == PermissionVerdict.ALLOW


# =====================================================================
# 7. PermissionPolicy — property queries
# =====================================================================

class TestPolicyQueries:

    def test_has_rule(self):
        rule = ToolPermissionRule(tool_name="Bash", approval=ApprovalClass.DENY)
        policy = PermissionPolicy(rules=[rule])
        assert policy.has_rule("Bash") is True
        assert policy.has_rule("Read") is False

    def test_rules_returns_copy(self):
        rule = ToolPermissionRule(tool_name="Bash")
        policy = PermissionPolicy(rules=[rule])
        rules = policy.rules
        rules["new"] = rule  # mutating the copy
        assert not policy.has_rule("new")


# =====================================================================
# 8. Helper functions
# =====================================================================

class TestHelpers:

    def test_resolve_legacy_approval(self):
        assert _resolve_legacy_approval("none") == ApprovalClass.ALLOW
        assert _resolve_legacy_approval("standard") == ApprovalClass.ASK
        assert _resolve_legacy_approval("high_risk") == ApprovalClass.DENY
        # Unknown falls back to ASK
        assert _resolve_legacy_approval("unknown") == ApprovalClass.ASK

    def test_approval_to_verdict(self):
        assert _approval_to_verdict(ApprovalClass.ALLOW) == PermissionVerdict.ALLOW
        assert _approval_to_verdict(ApprovalClass.ASK) == PermissionVerdict.ASK
        assert _approval_to_verdict(ApprovalClass.DENY) == PermissionVerdict.DENY

    def test_resolve_isolation(self):
        assert _resolve_isolation(None) == IsolationMode.IN_PROCESS
        assert _resolve_isolation("in_process") == IsolationMode.IN_PROCESS
        assert _resolve_isolation("subprocess") == IsolationMode.SUBPROCESS
        assert _resolve_isolation("worktree") == IsolationMode.WORKTREE
        # Unknown falls back to IN_PROCESS
        assert _resolve_isolation("unknown") == IsolationMode.IN_PROCESS


# =====================================================================
# 9. IsolatedSubagentContext
# =====================================================================

class TestIsolatedSubagentContext:

    def _make_ctx(self, **overrides):
        from omicverse.utils.ovagent.subagent_controller import IsolatedSubagentContext
        defaults = dict(
            agent_type="explore",
            allowed_tools=frozenset({"inspect_data", "finish"}),
            permission_policy=PermissionPolicy.for_subagent(
                frozenset({"inspect_data", "finish"}),
            ),
        )
        defaults.update(overrides)
        return IsolatedSubagentContext(**defaults)

    def test_is_tool_allowed(self):
        ctx = self._make_ctx()
        assert ctx.is_tool_allowed("inspect_data") is True
        assert ctx.is_tool_allowed("execute_code") is False

    def test_check_tool_permission_returns_decision(self):
        ctx = self._make_ctx()
        d = ctx.check_tool_permission("inspect_data")
        assert isinstance(d, PermissionDecision)
        assert d.verdict == PermissionVerdict.ALLOW

    def test_default_isolation_mode(self):
        ctx = self._make_ctx()
        assert ctx.isolation_mode == IsolationMode.IN_PROCESS

    def test_custom_isolation_mode(self):
        ctx = self._make_ctx(isolation_mode=IsolationMode.SUBPROCESS)
        assert ctx.isolation_mode == IsolationMode.SUBPROCESS

    def test_can_mutate_adata_default_false(self):
        ctx = self._make_ctx()
        assert ctx.can_mutate_adata is False

    def test_snapshot_fields(self):
        ctx = self._make_ctx(
            system_prompt="You are a helper.",
            visible_tool_schemas=[{"name": "inspect_data"}],
        )
        assert ctx.system_prompt == "You are a helper."
        assert len(ctx.visible_tool_schemas) == 1


# =====================================================================
# 10. SubagentResult
# =====================================================================

class TestSubagentResult:

    def test_default_fields(self):
        from omicverse.utils.ovagent.subagent_controller import SubagentResult
        r = SubagentResult(result="done")
        assert r.result == "done"
        assert r.adata is None
        assert r.turns_used == 0
        assert r.tool_calls_made == 0
        assert r.isolation_mode == IsolationMode.IN_PROCESS
        assert r.denied_tool_calls == []

    def test_custom_fields(self):
        from omicverse.utils.ovagent.subagent_controller import SubagentResult
        r = SubagentResult(
            result="completed",
            turns_used=3,
            tool_calls_made=5,
            isolation_mode=IsolationMode.SUBPROCESS,
            denied_tool_calls=["Bash", "Write"],
        )
        assert r.turns_used == 3
        assert r.isolation_mode == IsolationMode.SUBPROCESS
        assert len(r.denied_tool_calls) == 2


# =====================================================================
# 11. ToolRuntime permission policy integration
# =====================================================================

class TestToolRuntimePermissionGate:
    """Verify that ToolRuntime.dispatch_tool respects PermissionPolicy."""

    def test_dispatch_without_policy_allows_all(self):
        """Without a permission policy, dispatch proceeds normally."""
        from omicverse.utils.ovagent.tool_runtime import (
            ToolRuntime, ToolDispatchRegistry, ToolRegistryEntry, ToolPolicy,
        )

        call_log = []

        def _fake_executor(args, adata):
            call_log.append(args)
            return "ok"

        # Build minimal runtime with a mock registry
        ctx = SimpleNamespace(
            _get_runtime_session_id=lambda: "test",
            _approval_handler=None,
        )
        executor = SimpleNamespace()
        rt = ToolRuntime(ctx, executor)

        # Manually register a test tool
        rt._dispatch_registry.register(ToolRegistryEntry(
            name="test_tool",
            executor=_fake_executor,
            schema={},
            policy=ToolPolicy(),
        ))

        tc = SimpleNamespace(name="test_tool", arguments={"x": 1}, id="t1")
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert result == "ok"
        assert len(call_log) == 1

    def test_dispatch_with_deny_policy(self):
        """PermissionPolicy DENY prevents tool execution."""
        from omicverse.utils.ovagent.tool_runtime import (
            ToolRuntime, ToolRegistryEntry, ToolPolicy,
        )

        policy = PermissionPolicy(denied_tools=frozenset({"blocked_tool"}))
        ctx = SimpleNamespace(
            _get_runtime_session_id=lambda: "test",
            _approval_handler=None,
        )
        executor = SimpleNamespace()
        rt = ToolRuntime(ctx, executor, permission_policy=policy)

        rt._dispatch_registry.register(ToolRegistryEntry(
            name="blocked_tool",
            executor=lambda args, adata: "should not reach",
            schema={},
            policy=ToolPolicy(),
        ))

        tc = SimpleNamespace(name="blocked_tool", arguments={}, id="t1")
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert "Permission denied" in result
        assert "blocked_tool" in result

    def test_dispatch_high_risk_denied_via_policy_metadata(self):
        """Tool with approval_class=high_risk is denied through policy."""
        from omicverse.utils.ovagent.tool_runtime import (
            ToolRuntime, ToolRegistryEntry, ToolPolicy,
        )

        policy = PermissionPolicy()  # default allows, but reads policy metadata
        ctx = SimpleNamespace(
            _get_runtime_session_id=lambda: "test",
            _approval_handler=None,
        )
        executor = SimpleNamespace()
        rt = ToolRuntime(ctx, executor, permission_policy=policy)

        rt._dispatch_registry.register(ToolRegistryEntry(
            name="risky_tool",
            executor=lambda args, adata: "should not reach",
            schema={},
            policy=ToolPolicy(approval_class="high_risk"),
        ))

        tc = SimpleNamespace(name="risky_tool", arguments={}, id="t1")
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert "Permission denied" in result

    def test_dispatch_ask_without_handler_errors(self):
        """ASK verdict without approval handler returns error."""
        from omicverse.utils.ovagent.tool_runtime import (
            ToolRuntime, ToolRegistryEntry, ToolPolicy,
        )

        policy = PermissionPolicy()
        ctx = SimpleNamespace(
            _get_runtime_session_id=lambda: "test",
            _approval_handler=None,
        )
        executor = SimpleNamespace()
        rt = ToolRuntime(ctx, executor, permission_policy=policy)

        rt._dispatch_registry.register(ToolRegistryEntry(
            name="ask_tool",
            executor=lambda args, adata: "should not reach",
            schema={},
            policy=ToolPolicy(approval_class="standard"),
        ))

        tc = SimpleNamespace(name="ask_tool", arguments={}, id="t1")
        result = asyncio.run(rt.dispatch_tool(tc, None, "test"))
        assert "requires approval" in result

    def test_get_tool_policy(self):
        """get_tool_policy returns the ToolPolicy for a known tool."""
        from omicverse.utils.ovagent.tool_runtime import (
            ToolRuntime, ToolRegistryEntry, ToolPolicy,
        )

        ctx = SimpleNamespace(_get_runtime_session_id=lambda: "test")
        executor = SimpleNamespace()
        rt = ToolRuntime(ctx, executor)

        rt._dispatch_registry.register(ToolRegistryEntry(
            name="test_tool",
            executor=lambda a, b: None,
            schema={},
            policy=ToolPolicy(read_only=True, approval_class="standard"),
        ))

        p = rt.get_tool_policy("test_tool")
        assert p is not None
        assert p.read_only is True
        assert p.approval_class == "standard"

        assert rt.get_tool_policy("nonexistent") is None

    def test_permission_policy_property(self):
        """permission_policy getter/setter works."""
        from omicverse.utils.ovagent.tool_runtime import ToolRuntime

        ctx = SimpleNamespace(_get_runtime_session_id=lambda: "test")
        executor = SimpleNamespace()
        rt = ToolRuntime(ctx, executor)
        assert rt.permission_policy is None

        policy = PermissionPolicy.permissive()
        rt.permission_policy = policy
        assert rt.permission_policy is policy


# =====================================================================
# 12. Security contract explicitness
# =====================================================================

class TestSecurityContractExplicit:
    """Verify that security contracts remain explicit — no implicit defaults
    that could silently allow dangerous operations."""

    def test_deny_list_is_frozen(self):
        """denied_tools cannot be mutated after construction."""
        policy = PermissionPolicy(denied_tools=frozenset({"Bash"}))
        with pytest.raises(AttributeError):
            policy.denied_tools.add("new")  # type: ignore[attr-defined]

    def test_allowlist_is_frozen(self):
        """allowed_tools cannot be mutated after construction."""
        policy = PermissionPolicy(
            allowed_tools=frozenset({"inspect_data"}),
        )
        with pytest.raises(AttributeError):
            policy.allowed_tools.add("new")  # type: ignore[attr-defined]

    def test_subagent_explore_denies_execute_code(self):
        """Explore subagent policy denies mutation tools."""
        policy = build_subagent_policy(
            ["inspect_data", "run_snippet", "search_functions", "finish"],
            deny_mutations=True,
        )
        d = policy.check("execute_code")
        assert d.verdict == PermissionVerdict.DENY

    def test_subagent_execute_allows_execute_code(self):
        """Execute subagent without deny_mutations allows execute_code."""
        policy = build_subagent_policy(
            ["inspect_data", "execute_code", "finish"],
            deny_mutations=False,
        )
        d = policy.check("execute_code")
        assert d.verdict == PermissionVerdict.ALLOW

    def test_isolated_context_denies_unallowed_tools(self):
        """IsolatedSubagentContext rejects tools not in allowlist."""
        from omicverse.utils.ovagent.subagent_controller import IsolatedSubagentContext
        ctx = IsolatedSubagentContext(
            agent_type="explore",
            allowed_tools=frozenset({"inspect_data", "finish"}),
            permission_policy=PermissionPolicy.for_subagent(
                frozenset({"inspect_data", "finish"}),
            ),
        )
        assert ctx.is_tool_allowed("inspect_data") is True
        assert ctx.is_tool_allowed("Bash") is False
        assert ctx.is_tool_allowed("execute_code") is False

    def test_no_implicit_allow_for_unknown_tools(self):
        """A restrictive policy does not implicitly allow unknown tools."""
        policy = PermissionPolicy.restrictive()
        d = policy.check("never_heard_of_this_tool")
        assert d.verdict == PermissionVerdict.ASK
        assert d.verdict != PermissionVerdict.ALLOW
