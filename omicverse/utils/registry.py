"""Compatibility shim for the OmicVerse function registry.

Historically the public registry helpers were imported from
``omicverse.utils.registry``. The canonical implementation now lives in
``omicverse._registry``. Re-export the stable API here so older imports,
tests, and docs continue to work.
"""

from .._registry import (
    FunctionRegistry,
    _global_registry,
    export_registry,
    find_function,
    get_function_help,
    import_registry,
    list_functions,
    recommend_function,
    register_function,
)


def get_registry() -> FunctionRegistry:
    """Return the global function registry singleton."""
    return _global_registry


__all__ = [
    "register_function",
    "find_function",
    "list_functions",
    "get_function_help",
    "recommend_function",
    "export_registry",
    "import_registry",
    "get_registry",
    "FunctionRegistry",
    "_global_registry",
]
