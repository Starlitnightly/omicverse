"""Domain research utilities for LLM models.

This submodule exposes high-level interfaces like :class:`ResearchManager`.
"""

try:
    from .research_manager import ResearchManager
except Exception:  # pragma: no cover - implementation may be optional
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

