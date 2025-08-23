"""Caching utilities for the research workflow."""

from functools import lru_cache
from typing import Callable, Optional


def cache(func: Optional[Callable] = None, *, maxsize: int = 128):
    """Decorator that applies an LRU cache to ``func``.

    Parameters
    ----------
    func:
        The function to decorate. If ``None``, returns a decorator configured
        with ``maxsize``.
    maxsize:
        Maximum size of the underlying cache.
    """

    def decorator(f: Callable) -> Callable:
        return lru_cache(maxsize=maxsize)(f)

    if func is not None:
        return decorator(func)
    return decorator


__all__ = ["cache"]
