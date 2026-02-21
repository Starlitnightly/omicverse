"""
Context window compaction for OmicVerse Agent (P2-1).

When the system prompt (function registry + skill guidance) approaches
the model's context limit, ``ContextCompactor`` asks the LLM to produce
a compressed summary that preserves function signatures and prerequisite
chains while shedding verbose descriptions and examples.

Inspired by Codex ``compact.rs``.
"""

from __future__ import annotations

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
        """Return a compressed version of *system_prompt*."""
        char_limit = self.MAX_COMPACT_INPUT * 4  # rough token → char
        truncated = system_prompt[:char_limit]
        prompt = (
            "Summarize the following OmicVerse function registry and skill "
            "instructions into a compact reference.  Keep: all function names, "
            "parameter signatures, prerequisite chains.  Remove: verbose "
            "descriptions, examples, related-function lists.\n\n"
            + truncated
        )
        return await self._llm.run(prompt)
