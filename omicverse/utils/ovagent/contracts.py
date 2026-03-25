"""OVAgent runtime contracts — target interfaces for the Claude/Codex-gap roadmap.

This module defines the *target* runtime contracts that downstream tasks will
implement against.  Each contract is a frozen dataclass or typing Protocol
specifying the minimal required surface.  No behavioral implementation lives
here — only shapes.

Contracts defined:

1. **ToolPolicyMetadata** — per-tool policy fields that extend ToolDefinition
2. **ContextBudgetContract** — token-aware context budgeting interface
3. **ImportanceTier** / **ContextEntry** — importance-based context items
4. **ExecutionFailureEnvelope** — structured failure payload for recovery loop
5. **RepairHint** — machine-readable hint for self-healing retry
6. **EventContract** — structured event emission interface
7. **PromptLayer** / **PromptCompositionContract** — template-based prompt assembly
8. **RemoteReviewConfig** — ngagent remote review configuration shape

All contracts are intentionally minimal and stable.  Downstream tasks
(tool registry, context engine, repair loop, etc.) depend on these shapes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)


# =====================================================================
# 1. Tool Policy Metadata
# =====================================================================

class ApprovalClass(str, Enum):
    """How a tool invocation must be approved before execution."""
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class IsolationMode(str, Enum):
    """Execution isolation level for a tool."""
    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"
    WORKTREE = "worktree"


class ParallelClass(str, Enum):
    """Whether a tool can safely run in parallel with other tools."""
    SAFE = "safe"
    UNSAFE = "unsafe"
    CONDITIONAL = "conditional"


class OutputTier(str, Enum):
    """How much context budget a tool's output typically consumes."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    VERBOSE = "verbose"
    UNBOUNDED = "unbounded"


@dataclass(frozen=True)
class ToolPolicyMetadata:
    """Per-tool policy metadata that extends the existing ToolDefinition.

    This is the *target* contract for Phase 2 tool registry migration.
    The existing ``ToolDefinition`` in ``harness/tool_catalog.py`` maps to
    this via: ``high_risk=True`` → ``approval=ASK``,
    ``server_only=True`` → ``isolation=SUBPROCESS``, etc.
    """
    approval: ApprovalClass = ApprovalClass.ALLOW
    isolation: IsolationMode = IsolationMode.IN_PROCESS
    parallel: ParallelClass = ParallelClass.SAFE
    output_tier: OutputTier = OutputTier.STANDARD
    read_only: bool = False
    max_output_tokens: Optional[int] = None
    timeout_seconds: Optional[float] = None

    # Lifecycle hook slots — callables are supplied at registration time.
    # Stored as None here; the registry populates them.
    pre_exec_hook: Optional[str] = None
    post_exec_hook: Optional[str] = None
    normalize_result_hook: Optional[str] = None


@dataclass(frozen=True)
class ToolContract:
    """Complete tool contract: identity + schema + policy.

    Merges the existing ToolDefinition fields with the new policy metadata
    into one target shape.  Downstream task (tool-registry conversion) will
    produce these from current ToolDefinition + ToolPolicyMetadata.
    """
    name: str
    group: str
    description: str
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    policy: ToolPolicyMetadata = field(default_factory=ToolPolicyMetadata)
    keywords: tuple[str, ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def requires_approval(self) -> bool:
        return self.policy.approval != ApprovalClass.ALLOW

    def is_parallel_safe(self) -> bool:
        return self.policy.parallel == ParallelClass.SAFE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "description": self.description,
            "parameters_schema": dict(self.parameters_schema),
            "policy": {
                "approval": self.policy.approval.value,
                "isolation": self.policy.isolation.value,
                "parallel": self.policy.parallel.value,
                "output_tier": self.policy.output_tier.value,
                "read_only": self.policy.read_only,
                "max_output_tokens": self.policy.max_output_tokens,
                "timeout_seconds": self.policy.timeout_seconds,
            },
            "keywords": list(self.keywords),
            "aliases": list(self.aliases),
        }


# =====================================================================
# 2. Context Budget Contract
# =====================================================================

class ImportanceTier(str, Enum):
    """Importance level for context entries — drives compaction priority."""
    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"
    EPHEMERAL = "ephemeral"


@dataclass
class ContextEntry:
    """One item in the context window with importance metadata."""
    content: str
    source: str
    importance: ImportanceTier = ImportanceTier.STANDARD
    token_count: Optional[int] = None
    compactable: bool = True
    created_at: float = field(default_factory=time.time)

    def estimated_tokens(self) -> int:
        """Return token_count if set, else rough char/4 estimate."""
        if self.token_count is not None:
            return self.token_count
        return max(1, len(self.content) // 4)


class OverflowPolicy(str, Enum):
    """What to do when the context budget is exceeded."""
    TRUNCATE_OLDEST = "truncate_oldest"
    COMPACT_LOW_IMPORTANCE = "compact_low_importance"
    SUMMARIZE_AND_DROP = "summarize_and_drop"
    REJECT = "reject"


@dataclass
class ContextBudgetConfig:
    """Configuration for the context budget manager."""
    max_tokens: int = 128_000
    reserve_tokens: int = 4_096
    compaction_threshold: float = 0.85
    overflow_policy: OverflowPolicy = OverflowPolicy.COMPACT_LOW_IMPORTANCE
    per_tier_limits: Dict[str, Optional[int]] = field(default_factory=dict)

    @property
    def usable_tokens(self) -> int:
        return self.max_tokens - self.reserve_tokens


@runtime_checkable
class ContextBudgetManager(Protocol):
    """Interface contract for the token-aware context budget manager.

    Downstream task will implement this protocol.  The existing
    ``ContextCompactor`` in ``context_compactor.py`` is the migration
    starting point.
    """

    @property
    def config(self) -> ContextBudgetConfig: ...

    @property
    def current_token_count(self) -> int: ...

    @property
    def remaining_tokens(self) -> int: ...

    def add_entry(self, entry: ContextEntry) -> bool:
        """Add an entry.  Returns False if rejected by overflow policy."""
        ...

    def compact(self) -> int:
        """Run compaction.  Returns number of tokens freed."""
        ...

    def checkpoint(self) -> Dict[str, Any]:
        """Snapshot current state for rollback or summary."""
        ...

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from a previous checkpoint."""
        ...


# =====================================================================
# 3. Execution Failure / Recovery Contract
# =====================================================================

class FailurePhase(str, Enum):
    """Which phase of tool execution the failure occurred in."""
    PRE_EXEC = "pre_exec"
    EXECUTION = "execution"
    POST_EXEC = "post_exec"
    NORMALIZATION = "normalization"
    TIMEOUT = "timeout"


@dataclass
class RepairHint:
    """Machine-readable hint for the recovery loop."""
    strategy: str
    description: str
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionFailureEnvelope:
    """Normalized failure payload fed back into the runtime loop.

    This replaces ad-hoc error string parsing with structured data that
    the LLM and repair loop can act on.
    """
    tool_name: str
    phase: FailurePhase
    exception_type: str
    message: str
    stderr_summary: str = ""
    traceback_summary: str = ""
    retry_count: int = 0
    max_retries: int = 3
    repair_hints: List[RepairHint] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def retryable(self) -> bool:
        return self.retry_count < self.max_retries

    def to_llm_message(self) -> str:
        """Format as a message the LLM can reason about."""
        parts = [
            f"Tool `{self.tool_name}` failed during {self.phase.value}.",
            f"Error: {self.exception_type}: {self.message}",
        ]
        if self.stderr_summary:
            parts.append(f"stderr: {self.stderr_summary[:500]}")
        if self.repair_hints:
            hints = "; ".join(h.description for h in self.repair_hints)
            parts.append(f"Repair hints: {hints}")
        parts.append(
            f"Attempt {self.retry_count + 1}/{self.max_retries}."
            if self.retryable
            else "No retries remaining."
        )
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "phase": self.phase.value,
            "exception_type": self.exception_type,
            "message": self.message,
            "stderr_summary": self.stderr_summary,
            "traceback_summary": self.traceback_summary,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "retryable": self.retryable,
            "repair_hints": [
                {"strategy": h.strategy, "description": h.description,
                 "confidence": h.confidence, "metadata": h.metadata}
                for h in self.repair_hints
            ],
            "timestamp": self.timestamp,
        }


# =====================================================================
# 4. Event Stream Contract
# =====================================================================

# Re-export the canonical event types from harness.contracts so that
# downstream consumers import from one place.
EVENT_CATEGORIES = (
    "lifecycle",
    "tool",
    "code",
    "llm",
    "approval",
    "question",
    "task",
    "trace",
)

EventCategory = Literal[
    "lifecycle",
    "tool",
    "code",
    "llm",
    "approval",
    "question",
    "task",
    "trace",
]


@runtime_checkable
class EventEmitter(Protocol):
    """Interface for structured event emission.

    Replaces ad-hoc ``print()`` and ``_emit()`` calls with a typed
    contract that the observability subsystem implements.
    """

    def emit(
        self,
        event_type: str,
        content: Any,
        *,
        category: str = "",
        step_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a structured event."""
        ...

    def emit_failure(self, envelope: ExecutionFailureEnvelope) -> None:
        """Emit a structured failure event from an envelope."""
        ...


# =====================================================================
# 5. Prompt Composition Contract
# =====================================================================

class PromptLayerKind(str, Enum):
    """Kind of prompt layer in the composition stack."""
    BASE_SYSTEM = "base_system"
    PROVIDER = "provider"
    WORKFLOW = "workflow"
    SKILL = "skill"
    RUNTIME_STATE = "runtime_state"
    CONTEXT = "context"


@dataclass(frozen=True)
class PromptLayer:
    """One layer in the prompt composition stack."""
    kind: PromptLayerKind
    content: str
    priority: int = 0
    source: str = ""
    token_estimate: Optional[int] = None

    def estimated_tokens(self) -> int:
        if self.token_estimate is not None:
            return self.token_estimate
        return max(1, len(self.content) // 4)


@runtime_checkable
class PromptComposer(Protocol):
    """Interface for template-based prompt composition.

    Replaces giant string concatenation in PromptBuilder with a layered
    composition model.
    """

    def add_layer(self, layer: PromptLayer) -> None: ...

    def compose(self) -> str:
        """Assemble all layers into the final prompt string."""
        ...

    def layers(self) -> Sequence[PromptLayer]:
        """Return current layers in priority order."""
        ...

    def total_tokens(self) -> int:
        """Estimated total token count of the composed prompt."""
        ...


# =====================================================================
# 6. Remote Review Contract
# =====================================================================

@dataclass(frozen=True)
class RemoteReviewConfig:
    """Shape of the ngagent remote review configuration.

    This is *not* committed to the repo — it lives in operator-local
    ``.orchestrator/record.json.config.remote`` or is passed via
    ``ngagent init --remote-*`` / ``ngagent remote set`` CLI.

    Defined here so downstream tasks and docs can reference the shape.
    """
    host: str
    user: str = ""
    key_path: str = ""
    workspace: str = ""
    activate_cmd: str = ""
    timeout_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "user": self.user,
            "key_path": self.key_path,
            "workspace": self.workspace,
            "activate_cmd": self.activate_cmd,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoteReviewConfig":
        return cls(
            host=data.get("host", ""),
            user=data.get("user", ""),
            key_path=data.get("key_path", ""),
            workspace=data.get("workspace", ""),
            activate_cmd=data.get("activate_cmd", ""),
            timeout_seconds=data.get("timeout_seconds", 300),
        )


# =====================================================================
# 7. Benchmark Thresholds
# =====================================================================

@dataclass(frozen=True)
class BenchmarkThresholds:
    """Acceptance thresholds for the OVAgent runtime benchmarks.

    Each field defines a measurable target.  The benchmark test suite
    asserts that the runtime meets these thresholds.
    """
    # Tool dispatch: registry lookup must complete within this time (seconds)
    tool_lookup_max_seconds: float = 0.01

    # Context budget: compaction must free at least this fraction of tokens
    compaction_min_recovery_ratio: float = 0.20

    # Event stream: event emission overhead per event (seconds)
    event_emit_max_seconds: float = 0.001

    # Recovery: failure envelope construction overhead (seconds)
    failure_envelope_max_seconds: float = 0.005

    # Prompt composition: full prompt assembly overhead (seconds)
    prompt_compose_max_seconds: float = 0.05

    # Remote review: config round-trip serialization (seconds)
    remote_config_roundtrip_max_seconds: float = 0.001

    # Contract completeness: minimum required fields per contract
    tool_contract_min_fields: int = 3
    context_entry_min_fields: int = 3
    failure_envelope_min_fields: int = 4
    prompt_layer_min_fields: int = 2


# Singleton for tests to import
BENCHMARK_THRESHOLDS = BenchmarkThresholds()


# =====================================================================
# Exports
# =====================================================================

__all__ = [
    # Tool policy
    "ApprovalClass",
    "IsolationMode",
    "ParallelClass",
    "OutputTier",
    "ToolPolicyMetadata",
    "ToolContract",
    # Context budget
    "ImportanceTier",
    "ContextEntry",
    "OverflowPolicy",
    "ContextBudgetConfig",
    "ContextBudgetManager",
    # Execution failure
    "FailurePhase",
    "RepairHint",
    "ExecutionFailureEnvelope",
    # Event stream
    "EVENT_CATEGORIES",
    "EventCategory",
    "EventEmitter",
    # Prompt composition
    "PromptLayerKind",
    "PromptLayer",
    "PromptComposer",
    # Remote review
    "RemoteReviewConfig",
    # Benchmarks
    "BenchmarkThresholds",
    "BENCHMARK_THRESHOLDS",
]
