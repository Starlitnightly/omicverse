"""Domain research utilities for LLM models.

This submodule exposes high-level interfaces like :class:`ResearchManager`.
"""

try:
    from .research_manager import ResearchManager
except Exception:  # pragma: no cover - implementation may be optional
    ResearchManager = None  # type: ignore

__all__ = ["ResearchManager"]

