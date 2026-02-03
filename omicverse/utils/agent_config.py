"""
Grouped configuration dataclasses for OmicVerse Agent (P0-2).

Replaces the 15-parameter ``OmicVerseAgent.__init__`` with four focused
config objects.  ``AgentConfig.from_flat_kwargs`` preserves full backward
compatibility with the original constructor signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class SandboxFallbackPolicy(Enum):
    """What to do when notebook execution fails (P2-3)."""
    RAISE = "raise"                # Do not fall back; raise SandboxDeniedError
    WARN_AND_FALLBACK = "warn"     # Emit warning event, then fall back to in-process
    SILENT = "silent"              # Fall back silently (legacy behaviour)


@dataclass
class LLMConfig:
    """LLM connection settings."""
    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    reasoning_effort: str = "high"  # "low" | "medium" | "high" for GPT-5


@dataclass
class ReflectionConfig:
    """Code reflection & result review settings."""
    enabled: bool = True
    iterations: int = 1           # clamped to 1-3 in __post_init__
    result_review: bool = True

    def __post_init__(self) -> None:
        self.iterations = max(1, min(3, self.iterations))


@dataclass
class ExecutionConfig:
    """Code execution environment settings."""
    use_notebook: bool = True
    max_prompts_per_session: int = 5
    storage_dir: Optional[Path] = None
    keep_notebooks: bool = True
    timeout: int = 600
    strict_kernel_validation: bool = True
    sandbox_fallback_policy: SandboxFallbackPolicy = SandboxFallbackPolicy.WARN_AND_FALLBACK


@dataclass
class ContextConfig:
    """Filesystem context management settings."""
    enabled: bool = True
    storage_dir: Optional[Path] = None


@dataclass
class AgentConfig:
    """Aggregated agent configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    verbose: bool = True
    history_enabled: bool = False
    history_path: Optional[Path] = None

    # --------------- backward-compat factory ---------------

    @classmethod
    def from_flat_kwargs(cls, **kw: Any) -> "AgentConfig":
        """Build from the original OmicVerseAgent.__init__ keyword args."""
        sd = kw.get("notebook_storage_dir")
        cd = kw.get("context_storage_dir")
        return cls(
            llm=LLMConfig(
                model=kw.get("model", "gemini-2.5-flash"),
                api_key=kw.get("api_key"),
                endpoint=kw.get("endpoint"),
            ),
            reflection=ReflectionConfig(
                enabled=kw.get("enable_reflection", True),
                iterations=kw.get("reflection_iterations", 1),
                result_review=kw.get("enable_result_review", True),
            ),
            execution=ExecutionConfig(
                use_notebook=kw.get("use_notebook_execution", True),
                max_prompts_per_session=kw.get("max_prompts_per_session", 5),
                storage_dir=Path(sd) if sd else None,
                keep_notebooks=kw.get("keep_execution_notebooks", True),
                timeout=kw.get("notebook_timeout", 600),
                strict_kernel_validation=kw.get("strict_kernel_validation", True),
            ),
            context=ContextConfig(
                enabled=kw.get("enable_filesystem_context", True),
                storage_dir=Path(cd) if cd else None,
            ),
        )
