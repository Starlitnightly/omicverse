"""Token-aware context budget manager for OVAgent.

Replaces per-tool and per-turn character clipping with a budget layer
that accounts for prompt segments, tool output tiers, conversation
history, and subagent contexts.  Supports incremental sliding summaries
instead of one-shot compaction only.

Contract
--------
* Prompt and tool outputs consume **typed budget slices**.
* Overflow policy is chosen by **content tier** (OutputTier from the
  tool registry), not a single hard-coded constant.
* Both the main-agent and subagent paths use the same budgeting model.
* Compaction supports **incremental summaries / checkpoints** in
  addition to the existing one-shot ``ContextCompactor``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tool_registry import OutputTier


# ---------------------------------------------------------------------------
# Token estimator (bounded, no external dependency)
# ---------------------------------------------------------------------------

# CJK regex for character-aware estimation
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")


def estimate_tokens(text: str) -> int:
    """Bounded token estimate without tiktoken.

    Uses ~4 chars/token for ASCII, ~2 chars/token for CJK.
    This is deliberately conservative to avoid under-counting.
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    ascii_chars = len(text) - cjk
    return max(1, cjk // 2 + ascii_chars // 4)


# ---------------------------------------------------------------------------
# Model context windows (canonical source)
# ---------------------------------------------------------------------------

MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-5": 256_000,
    "gpt-5-mini": 128_000,
    "gemini/gemini-2.5-flash": 1_000_000,
    "gemini/gemini-2.5-pro": 1_000_000,
    "anthropic/claude-sonnet-4-20250514": 200_000,
    "anthropic/claude-opus-4-20250514": 200_000,
    "deepseek/deepseek-chat": 64_000,
}

_DEFAULT_CONTEXT_WINDOW = 32_000


def get_context_window(model: str) -> int:
    """Return context window size for *model*, falling back to default."""
    return MODEL_CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)


# ---------------------------------------------------------------------------
# Budget slice types
# ---------------------------------------------------------------------------


class BudgetSliceType(str, Enum):
    """Category of content consuming context budget."""

    system_prompt = "system_prompt"
    user_message = "user_message"
    assistant_message = "assistant_message"
    conversation_history = "conversation_history"
    tool_output = "tool_output"
    compaction_checkpoint = "compaction_checkpoint"


# ---------------------------------------------------------------------------
# Truncation policy per output tier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TruncationPolicy:
    """Defines how overflow is handled for a given content tier.

    Attributes
    ----------
    max_tokens : int
        Maximum token budget for a single piece of content.
    strategy : str
        ``"tail"`` keeps the head and truncates the tail (default).
        ``"head"`` keeps the tail.
        ``"middle"`` keeps head + tail, drops middle.
    suffix : str
        Text appended when truncation occurs.
    """

    max_tokens: int
    strategy: str = "tail"
    suffix: str = "\n... (truncated)"


# Default truncation policies keyed by OutputTier.
# These replace the hard-coded 4000/6000/8000 char constants.
_DEFAULT_TIER_POLICIES: Dict[OutputTier, TruncationPolicy] = {
    OutputTier.minimal: TruncationPolicy(max_tokens=500),
    OutputTier.standard: TruncationPolicy(max_tokens=2000),
    OutputTier.verbose: TruncationPolicy(max_tokens=3000),
}

# Subagent tier policies are tighter to fit the smaller budget.
_SUBAGENT_TIER_POLICIES: Dict[OutputTier, TruncationPolicy] = {
    OutputTier.minimal: TruncationPolicy(max_tokens=300),
    OutputTier.standard: TruncationPolicy(max_tokens=1200),
    OutputTier.verbose: TruncationPolicy(max_tokens=1800),
}


# ---------------------------------------------------------------------------
# Budget slice
# ---------------------------------------------------------------------------


@dataclass
class BudgetSlice:
    """A single allocation unit within the context budget.

    Each slice records what type of content it holds, how many tokens
    it consumes, and which output tier governs its overflow policy.
    """

    slice_type: BudgetSliceType
    tokens: int
    content_key: str = ""
    tier: OutputTier = OutputTier.standard


# ---------------------------------------------------------------------------
# Compaction checkpoint (incremental summaries)
# ---------------------------------------------------------------------------


@dataclass
class CompactionCheckpoint:
    """Snapshot of conversation state for incremental compaction.

    Instead of compacting the entire conversation in one shot,
    checkpoints capture a rolling summary at a specific turn,
    allowing the budget manager to drop old messages while
    retaining their compressed essence.
    """

    turn_index: int
    summary: str
    tokens: int
    timestamp: float = field(default_factory=time.time)
    messages_covered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "summary_preview": self.summary[:200],
            "tokens": self.tokens,
            "timestamp": self.timestamp,
            "messages_covered": self.messages_covered,
        }


# ---------------------------------------------------------------------------
# Context budget manager
# ---------------------------------------------------------------------------


class ContextBudgetManager:
    """Token-aware context budget manager.

    Tracks token consumption across typed budget slices, applies
    tier-driven truncation policies for tool outputs, and maintains
    incremental compaction checkpoints for sliding summaries.

    Parameters
    ----------
    model : str
        Model identifier (used to look up context window size).
    context_window : int, optional
        Override the model-derived context window.
    tier_policies : dict, optional
        Override the default per-tier truncation policies.
    reserve_fraction : float
        Fraction of the context window reserved for the model's
        response generation (default 0.25 = 25%).
    compaction_threshold : float
        Fraction of the usable budget at which compaction triggers
        (default 0.75 = 75%).
    """

    def __init__(
        self,
        model: str = "",
        *,
        context_window: int = 0,
        tier_policies: Optional[Dict[OutputTier, TruncationPolicy]] = None,
        reserve_fraction: float = 0.25,
        compaction_threshold: float = 0.75,
    ) -> None:
        self._model = model
        self._context_window = context_window or get_context_window(model)
        self._tier_policies = dict(
            tier_policies or _DEFAULT_TIER_POLICIES
        )
        self._reserve_fraction = reserve_fraction
        self._compaction_threshold = compaction_threshold

        # Usable budget = context window minus response reserve
        self._usable_budget = int(
            self._context_window * (1.0 - self._reserve_fraction)
        )

        # Tracking
        self._slices: List[BudgetSlice] = []
        self._checkpoints: List[CompactionCheckpoint] = []
        self._total_consumed: int = 0

    # -- properties ---------------------------------------------------------

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def usable_budget(self) -> int:
        return self._usable_budget

    @property
    def total_consumed(self) -> int:
        return self._total_consumed

    @property
    def remaining_budget(self) -> int:
        return max(0, self._usable_budget - self._total_consumed)

    @property
    def utilization(self) -> float:
        """Fraction of usable budget currently consumed (0.0–1.0+)."""
        if self._usable_budget <= 0:
            return 1.0
        return self._total_consumed / self._usable_budget

    @property
    def checkpoints(self) -> Tuple[CompactionCheckpoint, ...]:
        return tuple(self._checkpoints)

    @property
    def tier_policies(self) -> Dict[OutputTier, TruncationPolicy]:
        return dict(self._tier_policies)

    # -- budget tracking ----------------------------------------------------

    def record(
        self,
        slice_type: BudgetSliceType,
        content: str,
        *,
        content_key: str = "",
        tier: OutputTier = OutputTier.standard,
    ) -> BudgetSlice:
        """Record a piece of content consuming budget.

        Returns the ``BudgetSlice`` that was created.
        """
        tokens = estimate_tokens(content)
        bs = BudgetSlice(
            slice_type=slice_type,
            tokens=tokens,
            content_key=content_key,
            tier=tier,
        )
        self._slices.append(bs)
        self._total_consumed += tokens
        return bs

    def consumption_by_type(self) -> Dict[str, int]:
        """Return total tokens consumed per slice type."""
        totals: Dict[str, int] = {}
        for s in self._slices:
            key = s.slice_type.value
            totals[key] = totals.get(key, 0) + s.tokens
        return totals

    # -- tier-driven truncation ---------------------------------------------

    def get_truncation_policy(self, tier: OutputTier) -> TruncationPolicy:
        """Return the truncation policy for *tier*."""
        return self._tier_policies.get(
            tier,
            TruncationPolicy(max_tokens=2000),  # safe fallback
        )

    def truncate_output(
        self,
        content: str,
        tier: OutputTier,
    ) -> str:
        """Apply tier-driven truncation to *content*.

        Returns the (possibly truncated) content string.
        """
        if not content:
            return content

        policy = self.get_truncation_policy(tier)
        content_tokens = estimate_tokens(content)

        if content_tokens <= policy.max_tokens:
            return content

        # Convert token limit back to approximate char budget
        # Use 4 chars/token as inverse of our estimator
        char_budget = policy.max_tokens * 4
        suffix_len = len(policy.suffix)

        # Guard: if budget is too small to fit even the suffix, return the suffix.
        if char_budget <= suffix_len:
            return policy.suffix[:char_budget] if char_budget > 0 else ""

        if policy.strategy == "head":
            # Keep the tail
            tail_chars = char_budget - suffix_len
            return policy.suffix + content[-tail_chars:]

        if policy.strategy == "middle":
            # Keep head + tail, drop middle
            half = max((char_budget - suffix_len) // 2, 0)
            if half == 0:
                return policy.suffix
            return content[:half] + policy.suffix + content[-half:]

        # Default: "tail" strategy — keep head, truncate tail
        keep = max(char_budget - suffix_len, 0)
        return content[:keep] + policy.suffix

    # -- compaction checkpoints ---------------------------------------------

    def should_compact(self) -> bool:
        """True when utilization exceeds the compaction threshold."""
        return self.utilization >= self._compaction_threshold

    def add_checkpoint(
        self,
        turn_index: int,
        summary: str,
        *,
        messages_covered: int = 0,
    ) -> CompactionCheckpoint:
        """Record an incremental compaction checkpoint.

        Checkpoints capture a rolling summary of conversation state
        at a specific turn, enabling sliding-window compaction.
        """
        tokens = estimate_tokens(summary)
        cp = CompactionCheckpoint(
            turn_index=turn_index,
            summary=summary,
            tokens=tokens,
            messages_covered=messages_covered,
        )
        self._checkpoints.append(cp)
        return cp

    def get_latest_checkpoint(self) -> Optional[CompactionCheckpoint]:
        """Return the most recent checkpoint, or None."""
        return self._checkpoints[-1] if self._checkpoints else None

    def build_sliding_summary(self) -> str:
        """Combine all checkpoints into a sliding summary string.

        Returns an empty string if no checkpoints exist.
        """
        if not self._checkpoints:
            return ""
        parts = []
        for cp in self._checkpoints:
            parts.append(
                f"[Turn {cp.turn_index}] {cp.summary}"
            )
        return "\n".join(parts)

    def compact_history(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        keep_recent: int = 4,
    ) -> Tuple[List[Dict[str, Any]], Optional[CompactionCheckpoint]]:
        """Replace older messages with the latest checkpoint summary.

        Keeps the system prompt (index 0) and the *keep_recent* most
        recent messages.  Everything in between is replaced by the
        sliding summary if checkpoints exist.

        Returns (compacted_messages, checkpoint_used_or_None).
        """
        if len(messages) <= keep_recent + 1:
            return list(messages), None

        latest_cp = self.get_latest_checkpoint()
        if latest_cp is None:
            # No checkpoint available; return messages unchanged
            return list(messages), None

        # Keep system prompt + inject summary + keep recent messages
        system_msg = messages[0] if messages else None
        recent = list(messages[-keep_recent:])

        compacted: List[Dict[str, Any]] = []
        if system_msg is not None:
            compacted.append(system_msg)

        # Insert the sliding summary as a system-injected context message
        summary_text = self.build_sliding_summary()
        if summary_text:
            compacted.append({
                "role": "user",
                "content": (
                    "[Context summary from earlier turns]\n" + summary_text
                ),
            })

        compacted.extend(recent)

        # Track the compaction in budget
        self.record(
            BudgetSliceType.compaction_checkpoint,
            summary_text,
            content_key="sliding_summary",
        )

        return compacted, latest_cp

    # -- summary / introspection --------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise budget state for debugging / tracing."""
        return {
            "model": self._model,
            "context_window": self._context_window,
            "usable_budget": self._usable_budget,
            "total_consumed": self._total_consumed,
            "remaining": self.remaining_budget,
            "utilization": round(self.utilization, 4),
            "should_compact": self.should_compact(),
            "slice_count": len(self._slices),
            "checkpoint_count": len(self._checkpoints),
            "consumption_by_type": self.consumption_by_type(),
            "tier_policies": {
                tier.value: {
                    "max_tokens": p.max_tokens,
                    "strategy": p.strategy,
                }
                for tier, p in self._tier_policies.items()
            },
        }

    def reset(self) -> None:
        """Clear all tracked slices and checkpoints."""
        self._slices.clear()
        self._checkpoints.clear()
        self._total_consumed = 0


# ---------------------------------------------------------------------------
# Factory for subagent budget managers
# ---------------------------------------------------------------------------


def create_subagent_budget_manager(
    model: str = "",
    *,
    context_window: int = 0,
) -> ContextBudgetManager:
    """Create a budget manager with tighter policies for subagents.

    Subagents operate with smaller context budgets and stricter
    truncation limits, but share the same budgeting model.
    """
    return ContextBudgetManager(
        model=model,
        context_window=context_window,
        tier_policies=_SUBAGENT_TIER_POLICIES,
        reserve_fraction=0.25,
        compaction_threshold=0.70,
    )


__all__ = [
    "BudgetSlice",
    "BudgetSliceType",
    "CompactionCheckpoint",
    "ContextBudgetManager",
    "TruncationPolicy",
    "create_subagent_budget_manager",
    "estimate_tokens",
    "get_context_window",
]
