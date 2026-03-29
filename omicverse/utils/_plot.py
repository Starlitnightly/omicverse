"""Deprecated plotting compatibility wrappers for ``omicverse.utils._plot``."""

from __future__ import annotations

import importlib
import warnings

_REMOVAL_VERSION = "2.2"


def _warn(old_name: str, replacement: str) -> None:
    warnings.warn(
        f"`ov.utils.{old_name}` is deprecated and will be removed in omicverse "
        f"{_REMOVAL_VERSION}. Use `{replacement}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def _wrap(name: str, replacement: str):
    def _target():
        backend = importlib.import_module("omicverse.pl._plot_backend")
        return getattr(backend, name)

    def wrapper(*args, **kwargs):
        _warn(name, replacement)
        return _target()(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper


_FUNCTION_REPLACEMENTS = {
    "plot_set": "ov.pl.plot_set",
    "plotset": "ov.pl.plotset",
    "ov_plot_set": "ov.pl.ov_plot_set",
    "style": "ov.pl.style",
    "palette": "ov.pl.palette",
    "plot_text_set": "ov.pl.plot_text_set",
    "ticks_range": "ov.pl.ticks_range",
    "plot_boxplot": "ov.pl.plot_boxplot",
    "plot_network": "ov.pl.plot_network",
    "plot_cellproportion": "ov.pl.plot_cellproportion",
    "plot_embedding_celltype": "ov.pl.plot_embedding_celltype",
    "geneset_wordcloud": "ov.pl.geneset_wordcloud",
    "plot_pca_variance_ratio": "ov.pl.plot_pca_variance_ratio",
    "plot_pca_variance_ratio1": "ov.pl.plot_pca_variance_ratio1",
    "gen_mpl_labels": "ov.pl.gen_mpl_labels",
}

for _name, _replacement in _FUNCTION_REPLACEMENTS.items():
    globals()[_name] = _wrap(_name, _replacement)

_FORWARDED_ATTRS = {
    "_vector_friendly",
    "pyomic_palette",
    "blue_palette",
    "orange_palette",
    "red_palette",
    "green_palette",
    "sc_color",
    "red_color",
    "green_color",
    "orange_color",
    "blue_color",
    "purple_color",
    "palette_28",
    "cet_g_bw",
    "palette_112",
    "palette_56",
    "vibrant_palette",
    "earth_palette",
    "pastel_palette",
}


def __getattr__(name: str):
    if name in _FORWARDED_ATTRS:
        backend = importlib.import_module("omicverse.pl._plot_backend")
        return getattr(backend, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = sorted(
    list(_FUNCTION_REPLACEMENTS)
    + [
        "_vector_friendly",
        "pyomic_palette",
        "blue_palette",
        "orange_palette",
        "red_palette",
        "green_palette",
        "sc_color",
        "red_color",
        "green_color",
        "orange_color",
        "blue_color",
        "purple_color",
        "palette_28",
        "cet_g_bw",
        "palette_112",
        "palette_56",
        "vibrant_palette",
        "earth_palette",
        "pastel_palette",
    ]
)
