"""Token-aware context budget manager for OVAgent.

Replaces fixed character truncation with importance-tiered, token-aware
budgeting.  The manager tracks context entries with their token costs and
importance levels, and can incrementally compact before overflow.

Implements the ``ContextBudgetManager`` protocol from ``contracts.py``.

Interface contract
------------------
The budget manager exposes four interfaces consumed by prompt assembly
and turn control:

- **classify** — determine an ``ImportanceTier`` for content
- **account** — add content to the budget with token tracking
- **compact** — incrementally free budget space by tier
- **render** — return entries in display order
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .contracts import (
    ContextBudgetConfig,
    ContextEntry,
    ImportanceTier,
    OverflowPolicy,
)
from ..context_compactor import estimate_tokens


# ---------------------------------------------------------------------------
# Classification tables
# ---------------------------------------------------------------------------

# Tool name → importance tier (higher = kept longer during compaction)
_TOOL_IMPORTANCE: Dict[str, ImportanceTier] = {
    "finish": ImportanceTier.CRITICAL,
    "execute_code": ImportanceTier.HIGH,
    "Agent": ImportanceTier.HIGH,
    "delegate": ImportanceTier.HIGH,
    "inspect_data": ImportanceTier.STANDARD,
    "search_functions": ImportanceTier.STANDARD,
    "search_skills": ImportanceTier.STANDARD,
    "run_snippet": ImportanceTier.STANDARD,
    "Read": ImportanceTier.STANDARD,
    "Bash": ImportanceTier.STANDARD,
    "Grep": ImportanceTier.STANDARD,
    "Edit": ImportanceTier.STANDARD,
    "Write": ImportanceTier.STANDARD,
    "WebFetch": ImportanceTier.LOW,
    "web_fetch": ImportanceTier.LOW,
    "WebSearch": ImportanceTier.LOW,
    "web_search": ImportanceTier.LOW,
    "web_download": ImportanceTier.LOW,
    "ToolSearch": ImportanceTier.LOW,
    "Glob": ImportanceTier.LOW,
}

# Source type → importance tier (for non-tool entries)
_SOURCE_IMPORTANCE: Dict[str, ImportanceTier] = {
    "system_prompt": ImportanceTier.CRITICAL,
    "user_message": ImportanceTier.HIGH,
    "assistant": ImportanceTier.STANDARD,
    "history": ImportanceTier.LOW,
    "compacted": ImportanceTier.EPHEMERAL,
    "steering": ImportanceTier.LOW,
}

# ImportanceTier → default max token budget for tool outputs
_TIER_MAX_TOKENS: Dict[ImportanceTier, int] = {
    ImportanceTier.CRITICAL: 4000,
    ImportanceTier.HIGH: 2000,
    ImportanceTier.STANDARD: 2000,
    ImportanceTier.LOW: 1000,
    ImportanceTier.EPHEMERAL: 500,
}

# ToolPolicy.output_tier (from tool_runtime) → ImportanceTier
_REGISTRY_OUTPUT_TIER_MAP: Dict[str, ImportanceTier] = {
    "compact": ImportanceTier.STANDARD,
    "normal": ImportanceTier.STANDARD,
    "verbose": ImportanceTier.LOW,
    # OutputTier contract enum values
    "minimal": ImportanceTier.HIGH,
    "standard": ImportanceTier.STANDARD,
    "unbounded": ImportanceTier.LOW,
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def classify(
    content: str,
    source: str,
    *,
    tool_name: str = "",
    output_tier: str = "",
) -> ImportanceTier:
    """Determine the importance tier of a context entry.

    Parameters
    ----------
    content : str
        The content to classify.
    source : str
        Source type (``"tool"``, ``"system_prompt"``, ``"assistant"``, etc.).
    tool_name : str, optional
        Canonical tool name when *source* is ``"tool"``.
    output_tier : str, optional
        The tool's ``output_tier`` from its registry policy, if known.

    Returns
    -------
    ImportanceTier
    """
    if tool_name:
        return _TOOL_IMPORTANCE.get(tool_name, ImportanceTier.STANDARD)
    if output_tier:
        return _REGISTRY_OUTPUT_TIER_MAP.get(output_tier, ImportanceTier.STANDARD)
    return _SOURCE_IMPORTANCE.get(source, ImportanceTier.STANDARD)


def max_tokens_for_tier(tier: ImportanceTier) -> int:
    """Return the default maximum token budget for an importance tier."""
    return _TIER_MAX_TOKENS.get(tier, 2000)


# ---------------------------------------------------------------------------
# DefaultContextBudgetManager
# ---------------------------------------------------------------------------

class DefaultContextBudgetManager:
    """Concrete token-aware context budget manager.

    Satisfies the ``ContextBudgetManager`` protocol from ``contracts.py``
    and adds the ``classify``, ``account``, ``compact``, and ``render``
    interfaces consumed by prompt assembly and turn control.
    """

    def __init__(self, config: Optional[ContextBudgetConfig] = None) -> None:
        self._config = config or ContextBudgetConfig()
        self._entries: List[ContextEntry] = []
        self._compaction_count: int = 0

    # -- protocol properties -----------------------------------------------

    @property
    def config(self) -> ContextBudgetConfig:
        return self._config

    @property
    def current_token_count(self) -> int:
        return sum(e.estimated_tokens() for e in self._entries)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self._config.usable_tokens - self.current_token_count)

    # -- classify ----------------------------------------------------------

    def classify(
        self,
        content: str,
        source: str,
        *,
        tool_name: str = "",
        output_tier: str = "",
    ) -> ImportanceTier:
        """Determine importance tier for content."""
        return classify(
            content, source, tool_name=tool_name, output_tier=output_tier,
        )

    # -- account -----------------------------------------------------------

    def account(
        self,
        content: str,
        source: str,
        *,
        importance: Optional[ImportanceTier] = None,
        tool_name: str = "",
        output_tier: str = "",
        compactable: bool = True,
    ) -> bool:
        """Classify and add an entry to the budget.

        Returns ``False`` if the entry is rejected by the overflow policy.
        """
        if importance is None:
            importance = self.classify(
                content, source, tool_name=tool_name, output_tier=output_tier,
            )

        entry = ContextEntry(
            content=content,
            source=source if not tool_name else f"tool:{tool_name}",
            importance=importance,
            token_count=estimate_tokens(content),
            compactable=compactable and importance != ImportanceTier.CRITICAL,
        )
        return self.add_entry(entry)

    # -- add_entry (protocol) ----------------------------------------------

    def add_entry(self, entry: ContextEntry) -> bool:
        """Add an entry.  Returns ``False`` if rejected by overflow policy.

        If adding would exceed the budget, attempts incremental compaction
        first.  If still over budget, applies the configured overflow policy.
        """
        needed = entry.estimated_tokens()

        if needed <= self.remaining_tokens:
            self._entries.append(entry)
            return True

        # Try compaction first
        self.compact()
        if needed <= self.remaining_tokens:
            self._entries.append(entry)
            return True

        # Apply overflow policy
        policy = self._config.overflow_policy
        if policy == OverflowPolicy.REJECT:
            return False

        if policy == OverflowPolicy.TRUNCATE_OLDEST:
            self._truncate_oldest(needed - self.remaining_tokens)
        elif policy == OverflowPolicy.COMPACT_LOW_IMPORTANCE:
            self._compact_tier(ImportanceTier.EPHEMERAL)
            self._compact_tier(ImportanceTier.LOW)
            if needed > self.remaining_tokens:
                self._truncate_oldest(needed - self.remaining_tokens)
        elif policy == OverflowPolicy.SUMMARIZE_AND_DROP:
            self._drop_lowest()

        self._entries.append(entry)
        return True

    # -- compact (protocol) ------------------------------------------------

    def compact(self) -> int:
        """Incremental compaction by tier.  Returns tokens freed.

        Compaction phases (executed in order, stopping early if budget
        drops below the compaction threshold):

        1. Drop EPHEMERAL entries.
        2. Truncate LOW entries to half their content.
        3. Leave STANDARD and above untouched.
        """
        if not self.needs_compaction():
            return 0

        before = self.current_token_count
        self._compaction_count += 1

        # Phase 1: Drop EPHEMERAL
        self._entries = [
            e for e in self._entries
            if e.importance != ImportanceTier.EPHEMERAL
        ]
        if not self.needs_compaction():
            return before - self.current_token_count

        # Phase 2: Truncate compactable LOW entries to half
        new_entries: List[ContextEntry] = []
        for e in self._entries:
            if (
                e.importance == ImportanceTier.LOW
                and e.compactable
                and len(e.content) > 200
            ):
                half = len(e.content) // 2
                truncated = e.content[:half] + "\n... [compacted]"
                new_entries.append(ContextEntry(
                    content=truncated,
                    source=e.source,
                    importance=e.importance,
                    token_count=estimate_tokens(truncated),
                    compactable=False,  # prevent re-compaction
                    created_at=e.created_at,
                ))
            else:
                new_entries.append(e)
        self._entries = new_entries

        return max(0, before - self.current_token_count)

    def needs_compaction(self) -> bool:
        """Check if current usage exceeds the compaction threshold."""
        threshold = int(
            self._config.usable_tokens * self._config.compaction_threshold
        )
        return self.current_token_count > threshold

    # -- render ------------------------------------------------------------

    def render(self) -> List[ContextEntry]:
        """Return entries sorted by importance (highest first), then time."""
        tier_order = {
            ImportanceTier.CRITICAL: 0,
            ImportanceTier.HIGH: 1,
            ImportanceTier.STANDARD: 2,
            ImportanceTier.LOW: 3,
            ImportanceTier.EPHEMERAL: 4,
        }
        return sorted(
            self._entries,
            key=lambda e: (tier_order.get(e.importance, 5), e.created_at),
        )

    # -- checkpoint / restore (protocol) -----------------------------------

    def checkpoint(self) -> Dict[str, Any]:
        """Snapshot current state for rollback."""
        return {
            "entries": [
                {
                    "content": e.content,
                    "source": e.source,
                    "importance": e.importance.value,
                    "token_count": e.token_count,
                    "compactable": e.compactable,
                    "created_at": e.created_at,
                }
                for e in self._entries
            ],
            "compaction_count": self._compaction_count,
        }

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Restore from a previous checkpoint."""
        self._entries = [
            ContextEntry(
                content=d["content"],
                source=d["source"],
                importance=ImportanceTier(d["importance"]),
                token_count=d.get("token_count"),
                compactable=d.get("compactable", True),
                created_at=d.get("created_at", time.time()),
            )
            for d in checkpoint.get("entries", [])
        ]
        self._compaction_count = checkpoint.get("compaction_count", 0)

    # -- tool output formatting --------------------------------------------

    def format_tool_output(
        self, output: str, tool_name: str, *, output_tier: str = "",
    ) -> str:
        """Token-aware tool output truncation.

        Replaces the fixed character truncation with tier-based limits
        derived from the tool's importance classification.
        """
        tier = self.classify(output, "tool", tool_name=tool_name, output_tier=output_tier)
        max_toks = max_tokens_for_tier(tier)

        # Per-tier limit override from config
        config_limit = self._config.per_tier_limits.get(tier.value)
        if config_limit is not None:
            max_toks = config_limit

        current_toks = estimate_tokens(output)
        if current_toks <= max_toks:
            return output

        # Truncate to fit within token budget (~4 chars per token)
        max_chars = max_toks * 4
        return output[:max_chars] + "\n... [truncated by context budget]"

    # -- convenience accessors ---------------------------------------------

    @property
    def entries(self) -> List[ContextEntry]:
        """Read-only copy of current entries."""
        return list(self._entries)

    @property
    def compaction_count(self) -> int:
        """Number of compaction passes executed."""
        return self._compaction_count

    # -- internal helpers --------------------------------------------------

    def _truncate_oldest(self, tokens_needed: int) -> None:
        """Remove oldest compactable entries until *tokens_needed* is freed."""
        freed = 0
        remaining: List[ContextEntry] = []
        for e in sorted(self._entries, key=lambda e: e.created_at):
            if freed < tokens_needed and e.compactable:
                freed += e.estimated_tokens()
            else:
                remaining.append(e)
        self._entries = remaining

    def _compact_tier(self, tier: ImportanceTier) -> int:
        """Remove all entries of *tier*.  Returns tokens freed."""
        before = self.current_token_count
        self._entries = [
            e for e in self._entries if e.importance != tier
        ]
        return before - self.current_token_count

    def _drop_lowest(self) -> int:
        """Drop entries from lowest tier upward until under threshold."""
        freed = 0
        for tier in (ImportanceTier.EPHEMERAL, ImportanceTier.LOW):
            freed += self._compact_tier(tier)
            if not self.needs_compaction():
                break
        return freed
