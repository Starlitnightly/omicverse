"""Tests for the context compactor soft-threshold logic."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from omicverse.utils.context_compactor import (
    ContextCompactor,
    estimate_tokens,
    get_context_window,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_ascii_only(self):
        # 40 ASCII chars → ~10 tokens
        assert estimate_tokens("a" * 40) == 10

    def test_cjk_only(self):
        # 10 CJK chars → ~5 tokens
        assert estimate_tokens("你" * 10) == 5

    def test_mixed(self):
        text = "hello你好world"
        # 10 ASCII + 2 CJK → 10//4 + 2//2 = 2 + 1 = 3
        assert estimate_tokens(text) == 3

    def test_empty(self):
        assert estimate_tokens("") == 0


# ---------------------------------------------------------------------------
# get_context_window
# ---------------------------------------------------------------------------

class TestGetContextWindow:
    def test_known_model(self):
        assert get_context_window("gpt-4o") == 128_000

    def test_gemini(self):
        assert get_context_window("gemini/gemini-3-flash-preview") == 1_000_000

    def test_unknown_model_returns_default(self):
        assert get_context_window("some-unknown-model") == 32_000


# ---------------------------------------------------------------------------
# ContextCompactor.needs_compaction
# ---------------------------------------------------------------------------

class TestNeedsCompaction:
    def _make_compactor(self, soft=120_000, hard=170_000):
        llm = MagicMock()
        return ContextCompactor(llm, model="gpt-4o", soft_threshold=soft, hard_threshold=hard)

    def test_below_soft_returns_none(self):
        c = self._make_compactor(soft=100, hard=200)
        # 10 tokens total → well below 100
        assert c.needs_compaction("a" * 40, "b" * 40) == "none"

    def test_between_soft_and_hard_returns_light(self):
        c = self._make_compactor(soft=10, hard=200)
        # 20 tokens → above soft=10, below hard=200
        assert c.needs_compaction("a" * 40, "a" * 40) == "light"

    def test_above_hard_returns_aggressive(self):
        c = self._make_compactor(soft=5, hard=10)
        # 20 tokens → above hard=10
        assert c.needs_compaction("a" * 40, "a" * 40) == "aggressive"


# ---------------------------------------------------------------------------
# ContextCompactor.compact
# ---------------------------------------------------------------------------

class TestCompact:
    def _make_compactor(self, soft=10, hard=50):
        llm = MagicMock()
        llm.run = AsyncMock(return_value="compressed prompt")
        return ContextCompactor(llm, model="gpt-4o", soft_threshold=soft, hard_threshold=hard)

    def test_no_compaction_needed(self):
        c = self._make_compactor(soft=9999, hard=99999)
        result = asyncio.run(c.compact("short prompt"))
        assert result == "short prompt"
        c._llm.run.assert_not_called()

    def test_light_compaction(self):
        c = self._make_compactor(soft=5, hard=9999)
        result = asyncio.run(c.compact("a" * 40))  # 10 tokens > soft=5
        assert result == "compressed prompt"
        c._llm.run.assert_called_once()
        call_arg = c._llm.run.call_args[0][0]
        assert "compact reference" in call_arg.lower()

    def test_aggressive_compaction(self):
        c = self._make_compactor(soft=5, hard=8)
        result = asyncio.run(c.compact("a" * 40))  # 10 tokens > hard=8
        assert result == "compressed prompt"
        call_arg = c._llm.run.call_args[0][0]
        assert "aggressively" in call_arg.lower()

    def test_cache_hit(self):
        c = self._make_compactor(soft=5, hard=9999)
        prompt = "a" * 40
        # First call → LLM invoked
        asyncio.run(c.compact(prompt))
        assert c._llm.run.call_count == 1
        # Second call with same prompt → cache hit
        result = asyncio.run(c.compact(prompt))
        assert c._llm.run.call_count == 1
        assert result == "compressed prompt"

    def test_cache_invalidation(self):
        c = self._make_compactor(soft=5, hard=9999)
        prompt = "a" * 40
        asyncio.run(c.compact(prompt))
        assert c._llm.run.call_count == 1
        c.invalidate_cache()
        asyncio.run(c.compact(prompt))
        assert c._llm.run.call_count == 2
