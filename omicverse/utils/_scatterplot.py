"""Deprecated plotting compatibility wrappers for ``omicverse.utils._scatterplot``."""

from __future__ import annotations

import importlib
import warnings

_REMOVAL_VERSION = "2.2"
_MESSAGE = (
    "`ov.utils.{old_name}` is deprecated and will be removed in omicverse "
    "{version}. Use `{replacement}` instead."
)


def _warn(old_name: str, replacement: str) -> None:
    warnings.warn(
        _MESSAGE.format(
            old_name=old_name,
            version=_REMOVAL_VERSION,
            replacement=replacement,
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def _wrap(name: str, replacement: str):
    def _target():
        backend = importlib.import_module("omicverse.pl._scatterplot_backend")
        return getattr(backend, name)

    def wrapper(*args, **kwargs):
        _warn(name, replacement)
        return _target()(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper


_REPLACEMENTS = {
    "embedding": "ov.pl.embedding",
    "umap": "ov.pl.umap",
    "tsne": "ov.pl.tsne",
    "pca": "ov.pl.pca",
    "spatial": "ov.pl.spatial",
    "diffmap": "ov.pl.embedding",
    "draw_graph": "ov.pl.embedding",
    "_get_vector_friendly": "ov.pl.plot_set",
    "_get_vboundnorm": "ov.pl.embedding",
    "_add_categorical_legend": "ov.pl.embedding",
    "_get_basis": "ov.pl.embedding",
    "_get_color_source_vector": "ov.pl.embedding",
    "_get_palette": "ov.pl.embedding",
    "_color_vector": "ov.pl.embedding",
    "_basis2name": "ov.pl.embedding",
    "_components_to_dimensions": "ov.pl.embedding",
    "_embedding": "ov.pl.embedding",
}

for _name, _replacement in _REPLACEMENTS.items():
    globals()[_name] = _wrap(_name, _replacement)

__all__ = sorted(_REPLACEMENTS)
