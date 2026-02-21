"""
Context window compaction for OmicVerse Agent.

Implements a **soft-threshold** strategy: when the estimated token count of
the system prompt + user prompt reaches a configurable band (default
120 000 – 170 000 tokens), the compactor summarises the system prompt via an
LLM call so that subsequent requests stay within the model's context window.

Two compression levels are supported:

* **light** (triggered at the *soft* threshold) – removes verbose
  descriptions and examples while preserving all function signatures,
  parameter details, and prerequisite chains.
* **aggressive** (triggered at the *hard* threshold) – additionally strips
  parameter descriptions and collapses each function to a single-line
  signature.

The compacted prompt is cached so repeated calls within the same session
do not re-invoke the LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .agent_backend import OmicVerseLLMBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Quick token estimate: ~4 chars/token for ASCII, ~2 for CJK."""
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", text))
    ascii_chars = len(text) - cjk
    return cjk // 2 + ascii_chars // 4


# ---------------------------------------------------------------------------
# Model context windows
# ---------------------------------------------------------------------------

MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-5": 256_000,
    "gpt-5-mini": 128_000,
    "gemini/gemini-3-flash-preview": 1_000_000,
    "gemini/gemini-2.5-flash": 1_000_000,
    "gemini/gemini-2.5-pro": 1_000_000,
    "anthropic/claude-sonnet-4-20250514": 200_000,
    "anthropic/claude-opus-4-20250514": 200_000,
    "deepseek/deepseek-chat": 64_000,
}

_DEFAULT_CONTEXT_WINDOW = 32_000


def get_context_window(model: str) -> int:
    return MODEL_CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------

@dataclass
class _CompactionCache:
    """Holds a cached compaction result keyed by the source prompt hash."""
    source_hash: int
    level: str            # "light" | "aggressive"
    compacted: str


class ContextCompactor:
    """Compresses the system prompt when it nears configurable token thresholds.

    Parameters
    ----------
    llm_backend : OmicVerseLLMBackend
        The LLM backend used for summarisation calls.
    model : str
        Model identifier (used to look up the context window size).
    soft_threshold : int
        Token count at which *light* compression is triggered (default 120 000).
    hard_threshold : int
        Token count at which *aggressive* compression is triggered (default 170 000).
    """

    def __init__(
        self,
        llm_backend: "OmicVerseLLMBackend",
        model: str,
        soft_threshold: int = 120_000,
        hard_threshold: int = 170_000,
    ) -> None:
        self._llm = llm_backend
        self._model = model
        self._context_window = get_context_window(model)
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self._cache: Optional[_CompactionCache] = None

    # ----- public API -------------------------------------------------------

    def needs_compaction(self, system_prompt: str, user_prompt: str = "") -> str:
        """Return the required compaction level.

        Returns
        -------
        str
            ``"none"`` – total tokens below the soft threshold.
            ``"light"`` – total tokens between the soft and hard thresholds.
            ``"aggressive"`` – total tokens above the hard threshold.
        """
        total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        if total >= self.hard_threshold:
            return "aggressive"
        if total >= self.soft_threshold:
            return "light"
        return "none"

    async def compact(self, system_prompt: str, user_prompt: str = "") -> str:
        """Return a compressed version of *system_prompt* if needed.

        If the estimated tokens are below the soft threshold the original
        prompt is returned unchanged.  Results are cached so that repeated
        calls with the same system prompt do not re-invoke the LLM.
        """
        level = self.needs_compaction(system_prompt, user_prompt)
        if level == "none":
            return system_prompt

        # Check cache
        src_hash = hash(system_prompt)
        if (
            self._cache is not None
            and self._cache.source_hash == src_hash
            and self._cache.level == level
        ):
            logger.debug("Context compaction cache hit (%s)", level)
            return self._cache.compacted

        logger.info(
            "Context compaction triggered (%s): estimated %d tokens",
            level,
            estimate_tokens(system_prompt) + estimate_tokens(user_prompt),
        )

        compacted = await self._compress(system_prompt, level)
        self._cache = _CompactionCache(
            source_hash=src_hash,
            level=level,
            compacted=compacted,
        )

        saved = estimate_tokens(system_prompt) - estimate_tokens(compacted)
        logger.info(
            "Compaction complete: ~%d tokens saved (%s)",
            saved,
            level,
        )
        return compacted

    def invalidate_cache(self) -> None:
        """Clear the compaction cache (e.g. after _setup_agent rebuilds)."""
        self._cache = None

    # ----- internal ---------------------------------------------------------

    async def _compress(self, system_prompt: str, level: str) -> str:
        """Ask the LLM to produce a compressed system prompt."""
        # Cap the input we send to the compactor itself
        max_chars = 80_000 * 4  # ~80k tokens
        truncated = system_prompt[:max_chars]

        if level == "aggressive":
            instruction = (
                "Aggressively compress the following OmicVerse function registry "
                "and skill instructions into a minimal reference.\n"
                "KEEP: every function name, its parameter names (no descriptions), "
                "prerequisite chain (which function must run before which).\n"
                "REMOVE: all descriptions, examples, related-function lists, "
                "code patterns, notes, and verbose text.\n"
                "Format each function as a ONE-LINE signature.\n\n"
            )
        else:
            instruction = (
                "Summarize the following OmicVerse function registry and skill "
                "instructions into a compact reference.\n"
                "KEEP: all function names, parameter signatures with types, "
                "prerequisite chains, and critical code patterns.\n"
                "REMOVE: verbose descriptions, usage examples, "
                "related-function lists, and repeated boilerplate.\n\n"
            )

        prompt = instruction + truncated
        return await self._llm.run(prompt)
