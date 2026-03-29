"""Deprecated plotting compatibility wrappers for ``omicverse.utils._venn``."""

from __future__ import annotations

import importlib
import warnings

_REMOVAL_VERSION = "2.2"


def _warn(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"`ov.utils.{old_name}` is deprecated and will be removed in omicverse "
        f"{_REMOVAL_VERSION}. Use `ov.pl.{new_name}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def _wrap(name: str, target_name: str):
    def _target():
        backend = importlib.import_module("omicverse.pl._venn_backend")
        return getattr(backend, name)

    def wrapper(*args, **kwargs):
        _warn(name, target_name)
        return _target()(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper


get_shared = _wrap("get_shared", "venn")
get_unique = _wrap("get_unique", "venn")
venny4py = _wrap("venny4py", "venn")

__all__ = ["get_shared", "get_unique", "venny4py"]
