"""Tests for the token-aware context budget manager.

Validates that:
- Token estimation is bounded and consistent (no external dependency).
- Budget slices track typed consumption by category.
- Truncation is policy-driven by OutputTier, not hard-coded constants.
- Compaction supports incremental checkpoints and sliding summaries.
- Main-agent and subagent paths share a consistent budgeting model.
- Budget utilization and compaction thresholds work correctly.
- History compaction replaces old messages with checkpoint summaries.
- The manager integrates with the existing ToolRegistry OutputTier enum.

These tests run under ``OV_AGENT_RUN_HARNESS_TESTS=1``.
"""

import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_RUN_HARNESS = os.environ.get("OV_AGENT_RUN_HARNESS_TESTS", "").lower() in {
    "1", "true", "yes", "on",
}
pytestmark = pytest.mark.skipif(
    not _RUN_HARNESS,
    reason="Context budget tests require OV_AGENT_RUN_HARNESS_TESTS=1.",
)

# ---------------------------------------------------------------------------
# Bootstrap: lightweight package stubs (matches test_runtime_contracts.py)
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

from omicverse.utils.ovagent.context_budget import (
    BudgetSlice,
    BudgetSliceType,
    CompactionCheckpoint,
    ContextBudgetManager,
    TruncationPolicy,
    create_subagent_budget_manager,
    estimate_tokens,
    get_context_window,
)
from omicverse.utils.ovagent.tool_registry import OutputTier

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


# ===================================================================
# 1. Token estimator
# ===================================================================


class TestEstimateTokens:
    """Token estimator uses a bounded heuristic, not raw char count."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_ascii_text(self):
        # ~4 chars per token for ASCII
        text = "Hello world this is a test"
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens == len(text) // 4

    def test_cjk_text(self):
        # ~2 chars per token for CJK
        text = "\u4f60\u597d\u4e16\u754c"  # 4 CJK chars
        tokens = estimate_tokens(text)
        assert tokens == 2  # 4 // 2

    def test_mixed_text(self):
        text = "Hello \u4f60\u597d"  # 6 ASCII + 2 CJK
        tokens = estimate_tokens(text)
        # 6 ASCII // 4 = 1, 2 CJK // 2 = 1 → 2
        assert tokens == 2

    def test_minimum_one_token(self):
        assert estimate_tokens("a") >= 1

    def test_returns_int(self):
        assert isinstance(estimate_tokens("test content"), int)


# ===================================================================
# 2. Context window lookup
# ===================================================================


class TestGetContextWindow:
    def test_known_model(self):
        assert get_context_window("gpt-4o") == 128_000

    def test_unknown_model_fallback(self):
        assert get_context_window("unknown-model-xyz") == 32_000

    def test_anthropic_model(self):
        assert get_context_window("anthropic/claude-opus-4-20250514") == 200_000


# ===================================================================
# 3. Budget slice tracking
# ===================================================================


class TestBudgetSliceTracking:
    """Budget manager tracks typed slices of consumed context."""

    def test_record_returns_slice(self):
        mgr = ContextBudgetManager(context_window=10_000)
        bs = mgr.record(
            BudgetSliceType.system_prompt,
            "You are a test agent.",
            content_key="sys",
        )
        assert isinstance(bs, BudgetSlice)
        assert bs.slice_type == BudgetSliceType.system_prompt
        assert bs.tokens > 0

    def test_total_consumed_updates(self):
        mgr = ContextBudgetManager(context_window=10_000)
        assert mgr.total_consumed == 0
        mgr.record(BudgetSliceType.system_prompt, "prompt text")
        assert mgr.total_consumed > 0

    def test_remaining_budget_decreases(self):
        mgr = ContextBudgetManager(context_window=10_000)
        initial = mgr.remaining_budget
        mgr.record(BudgetSliceType.user_message, "user request")
        assert mgr.remaining_budget < initial

    def test_consumption_by_type(self):
        mgr = ContextBudgetManager(context_window=10_000)
        mgr.record(BudgetSliceType.system_prompt, "prompt")
        mgr.record(BudgetSliceType.tool_output, "output1")
        mgr.record(BudgetSliceType.tool_output, "output2")
        by_type = mgr.consumption_by_type()
        assert "system_prompt" in by_type
        assert "tool_output" in by_type
        assert by_type["tool_output"] > 0

    def test_reset_clears_state(self):
        mgr = ContextBudgetManager(context_window=10_000)
        mgr.record(BudgetSliceType.system_prompt, "prompt")
        mgr.add_checkpoint(0, "summary")
        mgr.reset()
        assert mgr.total_consumed == 0
        assert len(mgr.checkpoints) == 0


# ===================================================================
# 4. Tier-driven truncation (acceptance criterion)
# ===================================================================


class TestTierDrivenTruncation:
    """Tool-output truncation is policy-driven by output tier,
    not a single hard-coded constant."""

    def test_minimal_tier_has_tight_limit(self):
        mgr = ContextBudgetManager(context_window=100_000)
        policy = mgr.get_truncation_policy(OutputTier.minimal)
        assert policy.max_tokens == 500

    def test_standard_tier_moderate_limit(self):
        mgr = ContextBudgetManager(context_window=100_000)
        policy = mgr.get_truncation_policy(OutputTier.standard)
        assert policy.max_tokens == 2000

    def test_verbose_tier_larger_limit(self):
        mgr = ContextBudgetManager(context_window=100_000)
        policy = mgr.get_truncation_policy(OutputTier.verbose)
        assert policy.max_tokens == 3000

    def test_different_tiers_different_limits(self):
        """Each tier has a distinct truncation policy."""
        mgr = ContextBudgetManager(context_window=100_000)
        limits = set()
        for tier in OutputTier:
            p = mgr.get_truncation_policy(tier)
            limits.add(p.max_tokens)
        # All three tiers should have different limits
        assert len(limits) == 3

    def test_short_content_not_truncated(self):
        mgr = ContextBudgetManager(context_window=100_000)
        short = "This is short."
        result = mgr.truncate_output(short, OutputTier.minimal)
        assert result == short

    def test_long_content_truncated_with_suffix(self):
        mgr = ContextBudgetManager(context_window=100_000)
        # Create content much larger than the minimal tier limit
        long_content = "x" * 10_000
        result = mgr.truncate_output(long_content, OutputTier.minimal)
        assert len(result) < len(long_content)
        assert result.endswith("... (truncated)")

    def test_verbose_tier_allows_more_than_minimal(self):
        mgr = ContextBudgetManager(context_window=100_000)
        # Content that exceeds minimal but not verbose
        content = "x" * 5000  # ~1250 tokens
        result_minimal = mgr.truncate_output(content, OutputTier.minimal)
        result_verbose = mgr.truncate_output(content, OutputTier.verbose)
        assert len(result_minimal) < len(result_verbose)

    def test_empty_content_returns_empty(self):
        mgr = ContextBudgetManager(context_window=100_000)
        assert mgr.truncate_output("", OutputTier.standard) == ""

    def test_custom_tier_policies(self):
        """Custom policies override defaults."""
        custom = {
            OutputTier.minimal: TruncationPolicy(max_tokens=100),
            OutputTier.standard: TruncationPolicy(max_tokens=200),
            OutputTier.verbose: TruncationPolicy(max_tokens=400),
        }
        mgr = ContextBudgetManager(
            context_window=10_000, tier_policies=custom
        )
        assert mgr.get_truncation_policy(OutputTier.minimal).max_tokens == 100

    def test_truncation_strategy_tail(self):
        """Default 'tail' strategy keeps the head."""
        mgr = ContextBudgetManager(context_window=100_000)
        content = "HEADER_" + "x" * 10_000
        result = mgr.truncate_output(content, OutputTier.minimal)
        assert result.startswith("HEADER_")

    def test_truncation_strategy_head(self):
        """'head' strategy keeps the tail."""
        custom = {
            OutputTier.standard: TruncationPolicy(
                max_tokens=100, strategy="head"
            ),
        }
        mgr = ContextBudgetManager(
            context_window=100_000, tier_policies=custom
        )
        content = "x" * 10_000 + "_TAIL"
        result = mgr.truncate_output(content, OutputTier.standard)
        assert result.endswith("_TAIL")

    def test_truncation_strategy_middle(self):
        """'middle' strategy keeps head + tail, drops middle."""
        custom = {
            OutputTier.standard: TruncationPolicy(
                max_tokens=100, strategy="middle"
            ),
        }
        mgr = ContextBudgetManager(
            context_window=100_000, tier_policies=custom
        )
        content = "HEAD" + "x" * 10_000 + "TAIL"
        result = mgr.truncate_output(content, OutputTier.standard)
        assert "HEAD" in result
        assert "TAIL" in result
        assert "... (truncated)" in result


# ===================================================================
# 5. Incremental compaction checkpoints (acceptance criterion)
# ===================================================================


class TestCompactionCheckpoints:
    """Compaction supports incremental summaries, not just one-shot."""

    def test_add_checkpoint(self):
        mgr = ContextBudgetManager(context_window=10_000)
        cp = mgr.add_checkpoint(
            turn_index=3,
            summary="Called Read and Grep on dataset files.",
            messages_covered=8,
        )
        assert isinstance(cp, CompactionCheckpoint)
        assert cp.turn_index == 3
        assert cp.messages_covered == 8
        assert cp.tokens > 0

    def test_multiple_checkpoints(self):
        mgr = ContextBudgetManager(context_window=10_000)
        mgr.add_checkpoint(0, "first summary")
        mgr.add_checkpoint(3, "second summary")
        mgr.add_checkpoint(6, "third summary")
        assert len(mgr.checkpoints) == 3

    def test_latest_checkpoint(self):
        mgr = ContextBudgetManager(context_window=10_000)
        assert mgr.get_latest_checkpoint() is None
        mgr.add_checkpoint(0, "first")
        mgr.add_checkpoint(5, "latest")
        latest = mgr.get_latest_checkpoint()
        assert latest is not None
        assert latest.turn_index == 5
        assert latest.summary == "latest"

    def test_build_sliding_summary(self):
        mgr = ContextBudgetManager(context_window=10_000)
        mgr.add_checkpoint(0, "Explored dataset structure")
        mgr.add_checkpoint(3, "Ran QC filtering")
        summary = mgr.build_sliding_summary()
        assert "[Turn 0]" in summary
        assert "[Turn 3]" in summary
        assert "Explored dataset" in summary
        assert "QC filtering" in summary

    def test_empty_sliding_summary(self):
        mgr = ContextBudgetManager(context_window=10_000)
        assert mgr.build_sliding_summary() == ""

    def test_checkpoint_to_dict(self):
        mgr = ContextBudgetManager(context_window=10_000)
        cp = mgr.add_checkpoint(2, "summary text", messages_covered=5)
        d = cp.to_dict()
        assert d["turn_index"] == 2
        assert d["messages_covered"] == 5
        assert "summary_preview" in d
        assert d["tokens"] > 0


# ===================================================================
# 6. Compaction threshold and should_compact
# ===================================================================


class TestCompactionThreshold:
    def test_initially_no_compaction_needed(self):
        mgr = ContextBudgetManager(context_window=100_000)
        assert not mgr.should_compact()

    def test_compaction_triggers_at_threshold(self):
        mgr = ContextBudgetManager(
            context_window=1000,
            compaction_threshold=0.5,
            reserve_fraction=0.0,
        )
        # Fill over half the budget
        mgr.record(BudgetSliceType.system_prompt, "x" * 2200)
        assert mgr.should_compact()

    def test_utilization_property(self):
        mgr = ContextBudgetManager(
            context_window=1000, reserve_fraction=0.0
        )
        assert mgr.utilization == 0.0
        mgr.record(BudgetSliceType.system_prompt, "x" * 2000)
        assert mgr.utilization > 0.0


# ===================================================================
# 7. History compaction with checkpoints
# ===================================================================


class TestHistoryCompaction:
    """compact_history replaces old messages with checkpoint summaries."""

    def test_short_history_unchanged(self):
        mgr = ContextBudgetManager(context_window=10_000)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result, cp = mgr.compact_history(messages, keep_recent=4)
        assert len(result) == 3
        assert cp is None

    def test_long_history_compacted(self):
        mgr = ContextBudgetManager(context_window=10_000)
        mgr.add_checkpoint(2, "Explored data and ran QC")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "resp3"},
            {"role": "user", "content": "msg4"},
            {"role": "assistant", "content": "resp4"},
        ]
        result, cp = mgr.compact_history(messages, keep_recent=2)
        assert cp is not None
        # Should have: system + summary + 2 recent
        assert result[0]["role"] == "system"
        assert "[Context summary" in result[1]["content"]
        assert len(result) == 4  # system + summary + 2 recent

    def test_no_checkpoint_no_compaction(self):
        mgr = ContextBudgetManager(context_window=10_000)
        messages = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg{i}"} for i in range(10)
        ]
        result, cp = mgr.compact_history(messages, keep_recent=2)
        assert cp is None
        assert len(result) == 11  # unchanged


# ===================================================================
# 8. Subagent budget manager (shared model, tighter policies)
# ===================================================================


class TestSubagentBudgetManager:
    """Main-agent and subagent share the same budgeting model
    but with different tier policies."""

    def test_subagent_uses_same_class(self):
        mgr = create_subagent_budget_manager()
        assert isinstance(mgr, ContextBudgetManager)

    def test_subagent_tighter_standard_limit(self):
        main = ContextBudgetManager()
        sub = create_subagent_budget_manager()
        main_std = main.get_truncation_policy(OutputTier.standard)
        sub_std = sub.get_truncation_policy(OutputTier.standard)
        assert sub_std.max_tokens < main_std.max_tokens

    def test_subagent_tighter_verbose_limit(self):
        main = ContextBudgetManager()
        sub = create_subagent_budget_manager()
        main_v = main.get_truncation_policy(OutputTier.verbose)
        sub_v = sub.get_truncation_policy(OutputTier.verbose)
        assert sub_v.max_tokens < main_v.max_tokens

    def test_subagent_consistent_api(self):
        """Subagent manager has all the same methods."""
        sub = create_subagent_budget_manager()
        assert hasattr(sub, "truncate_output")
        assert hasattr(sub, "record")
        assert hasattr(sub, "should_compact")
        assert hasattr(sub, "add_checkpoint")
        assert hasattr(sub, "compact_history")

    def test_subagent_lower_compaction_threshold(self):
        """Subagent compacts earlier than main agent."""
        main = ContextBudgetManager()
        sub = create_subagent_budget_manager()
        # Subagent threshold is 0.70, main is 0.75
        assert sub._compaction_threshold <= main._compaction_threshold


# ===================================================================
# 9. Manager introspection / to_dict
# ===================================================================


class TestManagerIntrospection:
    def test_to_dict_structure(self):
        mgr = ContextBudgetManager(model="gpt-4o")
        mgr.record(BudgetSliceType.system_prompt, "sys prompt")
        mgr.add_checkpoint(0, "cp1")
        d = mgr.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["context_window"] == 128_000
        assert d["slice_count"] == 1
        assert d["checkpoint_count"] == 1
        assert "consumption_by_type" in d
        assert "tier_policies" in d
        assert "should_compact" in d

    def test_usable_budget_less_than_window(self):
        mgr = ContextBudgetManager(context_window=100_000)
        assert mgr.usable_budget < mgr.context_window
        assert mgr.usable_budget == int(100_000 * 0.75)


# ===================================================================
# 10. Acceptance criterion: token-aware, not raw char limits
# ===================================================================


class TestAcceptanceCriterion:
    """Validates the full acceptance criterion:

    Context budgeting is token-aware or uses a bounded estimator
    rather than raw char limits; tool-output truncation is
    policy-driven by output tier; compaction supports incremental
    summaries or checkpoints; main-agent and subagent paths share
    a consistent budgeting model.
    """

    def test_token_aware_not_char_based(self):
        """Budget tracking uses token estimates, not raw char counts."""
        mgr = ContextBudgetManager(context_window=10_000)
        content = "x" * 100  # 100 chars → ~25 tokens
        bs = mgr.record(BudgetSliceType.tool_output, content)
        assert bs.tokens == 25  # token estimate, not 100

    def test_truncation_is_tier_driven(self):
        """Different OutputTiers produce different truncation limits."""
        mgr = ContextBudgetManager(context_window=100_000)
        long_content = "data " * 5000  # ~25000 chars

        results = {}
        for tier in OutputTier:
            results[tier] = mgr.truncate_output(long_content, tier)

        # Minimal should be shortest, verbose longest
        assert len(results[OutputTier.minimal]) < len(results[OutputTier.standard])
        assert len(results[OutputTier.standard]) < len(results[OutputTier.verbose])

    def test_compaction_supports_incremental_checkpoints(self):
        """Multiple checkpoints build a sliding summary."""
        mgr = ContextBudgetManager(context_window=10_000)
        mgr.add_checkpoint(0, "Loaded dataset")
        mgr.add_checkpoint(3, "Ran preprocessing")
        mgr.add_checkpoint(6, "Completed clustering")

        summary = mgr.build_sliding_summary()
        assert "Loaded dataset" in summary
        assert "Ran preprocessing" in summary
        assert "Completed clustering" in summary
        assert len(mgr.checkpoints) == 3

    def test_main_and_subagent_share_model(self):
        """Both paths use ContextBudgetManager with same API."""
        main = ContextBudgetManager(model="gpt-4o")
        sub = create_subagent_budget_manager(model="gpt-4o")

        assert type(main) is type(sub)
        assert main.context_window == sub.context_window

        # Both can truncate, record, checkpoint
        for mgr in (main, sub):
            mgr.record(BudgetSliceType.system_prompt, "prompt")
            result = mgr.truncate_output("x" * 50000, OutputTier.standard)
            assert "truncated" in result
            mgr.add_checkpoint(0, "test")
            assert len(mgr.checkpoints) == 1

    def test_overflow_policy_chosen_by_tier(self):
        """Overflow policy is chosen by content tier, not a single
        hard-coded constant — the core interface contract."""
        mgr = ContextBudgetManager(context_window=100_000)

        # Each tier has its own distinct policy
        policies = {
            tier: mgr.get_truncation_policy(tier) for tier in OutputTier
        }
        max_tokens_set = {p.max_tokens for p in policies.values()}
        assert len(max_tokens_set) == len(OutputTier), (
            "Each tier should have a distinct max_tokens limit"
        )

    def test_budget_records_tool_output_with_tier(self):
        """Record method accepts tier parameter for tool outputs."""
        mgr = ContextBudgetManager(context_window=100_000)
        bs = mgr.record(
            BudgetSliceType.tool_output,
            "tool result data",
            content_key="Read",
            tier=OutputTier.standard,
        )
        assert bs.tier == OutputTier.standard
        assert bs.slice_type == BudgetSliceType.tool_output


# ===================================================================
# 11. Export contract
# ===================================================================


class TestExportContract:
    """Context budget types are exported from ovagent.__init__."""

    def test_exports_available(self):
        import omicverse.utils.ovagent as ovagent_pkg
        for name in [
            "BudgetSlice",
            "BudgetSliceType",
            "CompactionCheckpoint",
            "ContextBudgetManager",
            "TruncationPolicy",
            "create_subagent_budget_manager",
        ]:
            assert name in ovagent_pkg.__all__, (
                f"{name} missing from ovagent.__all__"
            )
            assert getattr(ovagent_pkg, name, None) is not None, (
                f"{name} not importable from ovagent"
            )
