"""Domain Research (alias).

This package is an alias for `omicverse.llm.dr`, provided to expose the
domain research helpers under a clearer module name while maintaining
backward compatibility. It re-exports the primary public API.
"""

from __future__ import annotations

from ..dr import (  # type: ignore F401
    ResearchManager,
    create_store,
    add_documents,
    query_store,
    delete_documents,
)

__all__ = [
    "ResearchManager",
    "create_store",
    "add_documents",
    "query_store",
    "delete_documents",
]

