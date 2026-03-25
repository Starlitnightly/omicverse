"""PermissionPolicy — per-tool allow/ask/deny permission engine.

Bridges the existing ``ToolPolicy.approval_class`` string field and the
typed ``ApprovalClass`` enum from ``contracts.py`` into a unified decision
engine that SubagentController, ToolRuntime, and ToolScheduler can query
before dispatching a tool call.

Design goals:
- Tool classes expose explicit allow/ask/deny semantics
- Security contracts remain explicit (no implicit defaults)
- Subagent isolation modes can restrict permission scope
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Sequence

from .contracts import ApprovalClass, IsolationMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permission verdict
# ---------------------------------------------------------------------------

class PermissionVerdict(str, Enum):
    """Result of a permission check."""
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionDecision:
    """Structured result from a permission policy check.

    Attributes
    ----------
    verdict : PermissionVerdict
        The allow/ask/deny outcome.
    tool_name : str
        Which tool was checked.
    reason : str
        Human-readable explanation for the decision.
    required_isolation : IsolationMode
        Minimum isolation mode required for this tool invocation.
    """
    verdict: PermissionVerdict
    tool_name: str
    reason: str = ""
    required_isolation: IsolationMode = IsolationMode.IN_PROCESS


# ---------------------------------------------------------------------------
# Per-tool permission rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolPermissionRule:
    """Per-tool permission rule that overrides the default policy.

    When a rule exists for a tool name, it takes precedence over the
    tool's own ``ToolPolicy.approval_class`` field.
    """
    tool_name: str
    approval: ApprovalClass = ApprovalClass.ALLOW
    required_isolation: IsolationMode = IsolationMode.IN_PROCESS
    reason: str = ""


# ---------------------------------------------------------------------------
# Permission policy
# ---------------------------------------------------------------------------

# Map legacy ToolPolicy.approval_class strings → ApprovalClass
_LEGACY_APPROVAL_MAP: Dict[str, ApprovalClass] = {
    "none": ApprovalClass.ALLOW,
    "standard": ApprovalClass.ASK,
    "high_risk": ApprovalClass.DENY,
}


def _resolve_legacy_approval(approval_class: str) -> ApprovalClass:
    """Convert a legacy approval_class string to an ApprovalClass enum."""
    return _LEGACY_APPROVAL_MAP.get(approval_class, ApprovalClass.ASK)


class PermissionPolicy:
    """Per-tool permission policy engine.

    The policy is configured with:
    1. A default approval class for tools without explicit rules
    2. Per-tool override rules
    3. A set of globally denied tool names

    The ``check`` method returns a ``PermissionDecision`` by consulting
    (in order): global deny list → per-tool rule → tool policy metadata →
    default.

    Parameters
    ----------
    default_approval : ApprovalClass
        Fallback approval level for tools with no rule and no policy metadata.
    rules : sequence of ToolPermissionRule
        Per-tool override rules.
    denied_tools : frozenset of str
        Tool names that are always denied regardless of other rules.
    allowed_tools : frozenset of str or None
        If set, only these tools are allowed; everything else is denied.
        This enables subagent-scoped allowlists.
    """

    def __init__(
        self,
        *,
        default_approval: ApprovalClass = ApprovalClass.ALLOW,
        rules: Sequence[ToolPermissionRule] = (),
        denied_tools: FrozenSet[str] = frozenset(),
        allowed_tools: Optional[FrozenSet[str]] = None,
    ) -> None:
        self._default_approval = default_approval
        self._rules: Dict[str, ToolPermissionRule] = {
            r.tool_name: r for r in rules
        }
        self._denied_tools = denied_tools
        self._allowed_tools = allowed_tools

    # -- queries --

    @property
    def default_approval(self) -> ApprovalClass:
        return self._default_approval

    @property
    def denied_tools(self) -> FrozenSet[str]:
        return self._denied_tools

    @property
    def allowed_tools(self) -> Optional[FrozenSet[str]]:
        return self._allowed_tools

    @property
    def rules(self) -> Dict[str, ToolPermissionRule]:
        return dict(self._rules)

    def has_rule(self, tool_name: str) -> bool:
        return tool_name in self._rules

    # -- core check --

    def check(
        self,
        tool_name: str,
        *,
        tool_approval_class: Optional[str] = None,
        tool_isolation_mode: Optional[str] = None,
    ) -> PermissionDecision:
        """Evaluate whether *tool_name* is allowed, needs approval, or is denied.

        Parameters
        ----------
        tool_name : str
            The canonical tool name.
        tool_approval_class : str or None
            The tool's ``ToolPolicy.approval_class`` string (from registry).
            Used as fallback when no explicit rule exists.
        tool_isolation_mode : str or None
            The tool's ``ToolPolicy.isolation_mode`` string (from registry).

        Returns
        -------
        PermissionDecision
        """
        # 1. Global deny list
        if tool_name in self._denied_tools:
            return PermissionDecision(
                verdict=PermissionVerdict.DENY,
                tool_name=tool_name,
                reason=f"Tool '{tool_name}' is in the global deny list.",
            )

        # 2. Allowlist scope (subagent restriction)
        if self._allowed_tools is not None and tool_name not in self._allowed_tools:
            return PermissionDecision(
                verdict=PermissionVerdict.DENY,
                tool_name=tool_name,
                reason=f"Tool '{tool_name}' is not in the allowed set.",
            )

        # 3. Per-tool rule override
        rule = self._rules.get(tool_name)
        if rule is not None:
            verdict = _approval_to_verdict(rule.approval)
            return PermissionDecision(
                verdict=verdict,
                tool_name=tool_name,
                reason=rule.reason or f"Per-tool rule: {rule.approval.value}",
                required_isolation=rule.required_isolation,
            )

        # 4. Tool policy metadata fallback
        if tool_approval_class is not None:
            resolved = _resolve_legacy_approval(tool_approval_class)
            isolation = _resolve_isolation(tool_isolation_mode)
            verdict = _approval_to_verdict(resolved)
            return PermissionDecision(
                verdict=verdict,
                tool_name=tool_name,
                reason=f"Tool policy: approval_class={tool_approval_class}",
                required_isolation=isolation,
            )

        # 5. Default
        verdict = _approval_to_verdict(self._default_approval)
        return PermissionDecision(
            verdict=verdict,
            tool_name=tool_name,
            reason=f"Default policy: {self._default_approval.value}",
        )

    def check_batch(
        self,
        tool_names: Sequence[str],
        *,
        policies: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, PermissionDecision]:
        """Check permissions for multiple tools at once.

        Parameters
        ----------
        tool_names : sequence of str
            Tool names to check.
        policies : dict or None
            Maps tool_name → {"approval_class": str, "isolation_mode": str}.

        Returns
        -------
        dict mapping tool_name → PermissionDecision
        """
        results: Dict[str, PermissionDecision] = {}
        for name in tool_names:
            meta = (policies or {}).get(name, {})
            results[name] = self.check(
                name,
                tool_approval_class=meta.get("approval_class"),
                tool_isolation_mode=meta.get("isolation_mode"),
            )
        return results

    # -- factory helpers --

    @classmethod
    def for_subagent(
        cls,
        allowed_tools: FrozenSet[str],
        *,
        denied_tools: FrozenSet[str] = frozenset(),
        default_approval: ApprovalClass = ApprovalClass.ALLOW,
    ) -> "PermissionPolicy":
        """Create a scoped policy for a subagent with an explicit allowlist."""
        return cls(
            default_approval=default_approval,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
        )

    @classmethod
    def permissive(cls) -> "PermissionPolicy":
        """Create a permissive policy that allows everything."""
        return cls(default_approval=ApprovalClass.ALLOW)

    @classmethod
    def restrictive(cls) -> "PermissionPolicy":
        """Create a restrictive policy that asks for everything."""
        return cls(default_approval=ApprovalClass.ASK)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approval_to_verdict(approval: ApprovalClass) -> PermissionVerdict:
    """Map ApprovalClass → PermissionVerdict."""
    return PermissionVerdict(approval.value)


def _resolve_isolation(mode_str: Optional[str]) -> IsolationMode:
    """Convert a ToolPolicy.isolation_mode string to IsolationMode enum."""
    if mode_str is None:
        return IsolationMode.IN_PROCESS
    try:
        return IsolationMode(mode_str)
    except ValueError:
        return IsolationMode.IN_PROCESS


def build_subagent_policy(
    allowed_tool_names: Sequence[str],
    *,
    deny_mutations: bool = False,
) -> PermissionPolicy:
    """Build a PermissionPolicy scoped to a subagent's tool allowlist.

    Parameters
    ----------
    allowed_tool_names : sequence of str
        Tools the subagent may call.
    deny_mutations : bool
        If True, tools that mutate data (execute_code, Edit, Write, Bash)
        are added to the deny list even if they appear in allowed_tool_names.
    """
    denied = frozenset()
    if deny_mutations:
        denied = frozenset({"execute_code", "Edit", "Write", "Bash", "NotebookEdit"})
    return PermissionPolicy.for_subagent(
        allowed_tools=frozenset(allowed_tool_names),
        denied_tools=denied,
    )
