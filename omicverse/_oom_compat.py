"""
Compatibility shim for the optional ``anndataoom`` dependency.

omicverse's Rust-backed out-of-memory AnnData support is provided by the
external ``anndataoom`` package (install via ``pip install omicverse[rust]``).
If that package is not installed, this module provides no-op fallbacks so
omicverse's core functions keep working with regular ``anndata.AnnData``.

Usage inside omicverse:

    from .._oom_compat import oom_guard, is_oom

    @oom_guard(materialize=True, result_keys_uns=['*'])
    def my_function(adata, ...):
        ...
"""

from __future__ import annotations

try:
    from anndataoom import (
        AnnDataOOM,
        BackedArray,
        BackedLayers,
        TransformedBackedArray,
        ScaledBackedArray,
        oom_guard,
        is_oom,
    )
    HAS_OOM = True
except ImportError:
    HAS_OOM = False

    # Stub classes — never instantiated, but usable as the second argument to
    # ``isinstance(x, AnnDataOOM)`` without raising TypeError. Since anndataoom
    # is missing, no real OOM object can exist, so isinstance will always be
    # False, which is the desired behaviour.
    class _UnavailableOOMType:
        """Placeholder class used when anndataoom is not installed."""

    class AnnDataOOM(_UnavailableOOMType): pass
    class BackedArray(_UnavailableOOMType): pass
    class BackedLayers(_UnavailableOOMType): pass
    class TransformedBackedArray(_UnavailableOOMType): pass
    class ScaledBackedArray(_UnavailableOOMType): pass

    def is_oom(adata) -> bool:
        """Fallback when anndataoom is not installed — always False."""
        return False

    def oom_guard(**kwargs):
        """Fallback decorator: identity (no-op) when anndataoom is unavailable."""
        def decorator(func):
            return func
        return decorator


__all__ = [
    "HAS_OOM",
    "AnnDataOOM",
    "BackedArray",
    "BackedLayers",
    "TransformedBackedArray",
    "ScaledBackedArray",
    "oom_guard",
    "is_oom",
]
