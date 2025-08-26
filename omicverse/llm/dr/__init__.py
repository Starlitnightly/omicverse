"""Domain research utilities for LLM models.

This submodule exposes high-level interfaces like :class:`ResearchManager`.

Deprecated: `omicverse.llm.dr` is deprecated and will be removed in a
future release. Please import from `omicverse.llm.domain_research` instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "'omicverse.llm.dr' is deprecated; use 'omicverse.llm.domain_research'",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from .research_manager import ResearchManager
except ImportError:  # pragma: no cover - implementation may be optional
    # Only treat genuine import-time missing-dependency issues as optional
    ResearchManager = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .vector_store import (
        create_store,
        add_documents,
        query_store,
        delete_documents,
    )
except Exception:  # pragma: no cover - if chromadb or GPT4All is missing
    create_store = add_documents = query_store = delete_documents = None  # type: ignore

__all__ = [
    "ResearchManager",
    "create_store",
    "add_documents",
    "query_store",
    "delete_documents",
]
