"""
Context window compaction for OmicVerse Agent (P2-1).

When the system prompt (function registry + skill guidance) approaches
the model's context limit, ``ContextCompactor`` asks the LLM to produce
a compressed summary that preserves function signatures and prerequisite
chains while shedding verbose descriptions and examples.

Inspired by Codex ``compact.rs``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .agent_backend import OmicVerseLLMBackend


# Rough token estimate without requiring tiktoken
def estimate_tokens(text: str) -> int:
    """Quick token estimate: ~4 chars/token for ASCII, ~2 for CJK."""
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", text))
    ascii_chars = len(text) - cjk
    return cjk // 2 + ascii_chars // 4


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
    return MODEL_CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)


COMPACTION_PROMPT = (
    "You are compacting OmicVerse agent context for a later handoff. "
    "Produce a compact but high-signal reference that preserves: "
    "function names, parameter signatures, prerequisite chains, "
    "active workflow constraints, unresolved issues, and any user-specific "
    "requirements that would change future tool selection. "
    "Remove repetitive examples, verbose prose, and duplicate explanations. "
    "Do not invent capabilities or results that were not present in the source.\n\n"
    "Source context:\n"
)

HANDOFF_PROMPT = (
    "The following is compacted context from an earlier OVAgent turn. "
    "Treat it as authoritative prior context. Reconstruct intent, constraints, "
    "and relevant tool knowledge from it, but do not assume omitted details are true. "
    "Prefer this summary over recomputing the full original prompt.\n\n"
    "Compacted context:\n"
)


@dataclass
class CompactionResult:
    summary: str
    handoff_text: str
    original_tokens: int
    compacted_tokens: int


class ContextCompactor:
    """Compresses the system prompt when it nears the context window limit."""

    COMPACT_THRESHOLD = 0.75   # trigger at 75 % of window
    MAX_COMPACT_INPUT = 20_000  # token limit for the compaction request itself

    def __init__(self, llm_backend: "OmicVerseLLMBackend", model: str) -> None:
        self._llm = llm_backend
        self._model = model
        self._context_window = get_context_window(model)

    def needs_compaction(self, system_prompt: str, user_prompt: str) -> bool:
        total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        return total > self._context_window * self.COMPACT_THRESHOLD

    async def compact(self, system_prompt: str) -> str:
        """Return a handoff-ready compressed version of *system_prompt*."""
        return (await self.compact_bundle(system_prompt)).handoff_text

    async def compact_bundle(self, system_prompt: str) -> CompactionResult:
        """Return both the raw summary and the handoff-wrapped compacted prompt."""
        char_limit = self.MAX_COMPACT_INPUT * 4  # rough token → char
        truncated = system_prompt[:char_limit]
        summary = await self._llm.run(COMPACTION_PROMPT + truncated)
        summary = summary.strip()
        handoff_text = HANDOFF_PROMPT + summary
        return CompactionResult(
            summary=summary,
            handoff_text=handoff_text,
            original_tokens=estimate_tokens(system_prompt),
            compacted_tokens=estimate_tokens(handoff_text),
        )
