"""Per-tool permission policy for OVAgent tool dispatch.

Maps tool names to allow/ask/deny permission verdicts based on registry
metadata, explicit overrides, and allowlist/denylist constraints.  The
policy layer sits between the turn loop / subagent controller and tool
dispatch, enforcing per-tool and per-class permission decisions.

Design for forward compatibility
---------------------------------
The ``PermissionPolicy`` *evaluates* verdicts; it does NOT *enforce*
them (that is the caller's job).  This separation lets a future
process-backed isolation layer inject its own enforcement without
changing the evaluation contract.

``create_subagent_policy`` produces a restricted policy that a
``SubagentRuntime`` can carry to filter tool calls before dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional

from .tool_registry import ApprovalClass, IsolationMode, ToolRegistry


# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------


class PermissionVerdict(str, Enum):
    """Result of a permission evaluation."""

    allow = "allow"
    ask = "ask"
    deny = "deny"


@dataclass(frozen=True)
class PermissionDecision:
    """A resolved permission decision for a specific tool invocation."""

    verdict: PermissionVerdict
    tool_name: str
    reason: str = ""
    requires_isolation: IsolationMode = IsolationMode.none

    @property
    def is_allowed(self) -> bool:
        return self.verdict == PermissionVerdict.allow

    @property
    def is_denied(self) -> bool:
        return self.verdict == PermissionVerdict.deny

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "tool_name": self.tool_name,
            "reason": self.reason,
            "requires_isolation": self.requires_isolation.value,
        }


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------


class PermissionPolicy:
    """Evaluate per-tool permission decisions from metadata and overrides.

    Resolution order (highest to lowest priority):

    1. Explicit deny list
    2. Allowlist restriction (if set, only listed tools are reachable)
    3. Per-tool overrides
    4. Per-class overrides (keyed by ``ApprovalClass``)
    5. Registry metadata defaults (``ToolMetadata.approval_class``)
    6. Unknown-tool fallback (configurable, defaults to ``deny``)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        tool_overrides: Optional[Dict[str, ApprovalClass]] = None,
        class_overrides: Optional[Dict[ApprovalClass, PermissionVerdict]] = None,
        denied_tools: Optional[FrozenSet[str]] = None,
        allowed_tools: Optional[FrozenSet[str]] = None,
        unknown_tool_default: PermissionVerdict = PermissionVerdict.deny,
    ) -> None:
        self._registry = registry
        self._tool_overrides = dict(tool_overrides or {})
        self._class_overrides = dict(class_overrides or {})
        self._denied_tools = denied_tools or frozenset()
        self._allowed_tools = allowed_tools
        self._unknown_default = unknown_tool_default

    @property
    def registry(self) -> ToolRegistry:
        """The underlying tool registry."""
        return self._registry

    def check(self, tool_name: str) -> PermissionDecision:
        """Evaluate permission for a tool invocation.

        Returns a ``PermissionDecision`` describing the verdict, the
        reason for the decision, and any isolation requirement drawn
        from the tool's registry metadata.
        """
        # 1. Explicit deny list — absolute priority
        if tool_name in self._denied_tools:
            return PermissionDecision(
                verdict=PermissionVerdict.deny,
                tool_name=tool_name,
                reason="tool is in the explicit deny list",
            )

        # 2. Allowlist restriction — resolve aliases before checking so that an
        # aliased name whose canonical is in the allowlist is not incorrectly denied.
        canonical_name = self._registry.resolve_name(tool_name) or tool_name
        if self._allowed_tools is not None:
            if tool_name not in self._allowed_tools and canonical_name not in self._allowed_tools:
                return PermissionDecision(
                    verdict=PermissionVerdict.deny,
                    tool_name=tool_name,
                    reason="tool is not in the allowed set",
                )

        # 3. Per-tool override
        if tool_name in self._tool_overrides:
            override = self._tool_overrides[tool_name]
            return PermissionDecision(
                verdict=PermissionVerdict(override.value),
                tool_name=tool_name,
                reason=f"per-tool override: {override.value}",
                requires_isolation=self._get_isolation(tool_name),
            )

        # 4–5. Registry metadata lookup
        meta = self._registry.get(tool_name)
        if meta is None:
            # 6. Unknown tool fallback
            return PermissionDecision(
                verdict=self._unknown_default,
                tool_name=tool_name,
                reason="unknown tool, applying default policy",
            )

        # 4. Class-level override
        if meta.approval_class in self._class_overrides:
            return PermissionDecision(
                verdict=self._class_overrides[meta.approval_class],
                tool_name=tool_name,
                reason=f"class override for {meta.approval_class.value}",
                requires_isolation=meta.isolation_mode,
            )

        # 5. Registry default
        return PermissionDecision(
            verdict=PermissionVerdict(meta.approval_class.value),
            tool_name=tool_name,
            reason="registry metadata default",
            requires_isolation=meta.isolation_mode,
        )

    def _get_isolation(self, tool_name: str) -> IsolationMode:
        meta = self._registry.get(tool_name)
        return meta.isolation_mode if meta else IsolationMode.none

    def summary(self) -> Dict[str, str]:
        """Return ``{tool_name: verdict_description}`` for all registered tools."""
        result: Dict[str, str] = {}
        for entry in self._registry.all_entries():
            decision = self.check(entry.canonical_name)
            result[entry.canonical_name] = (
                f"{decision.verdict.value} ({decision.reason})"
            )
        return result


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_default_policy(registry: ToolRegistry) -> PermissionPolicy:
    """Create the default permission policy from registry metadata alone."""
    return PermissionPolicy(registry)


def create_subagent_policy(
    registry: ToolRegistry,
    allowed_tools: FrozenSet[str],
) -> PermissionPolicy:
    """Create a restricted permission policy for subagent execution.

    Subagents receive only the tools in their allowed set; everything
    else is denied.  Unknown tools are denied by default.
    """
    return PermissionPolicy(
        registry,
        allowed_tools=allowed_tools,
        unknown_tool_default=PermissionVerdict.deny,
    )


__all__ = [
    "PermissionDecision",
    "PermissionPolicy",
    "PermissionVerdict",
    "create_default_policy",
    "create_subagent_policy",
]
