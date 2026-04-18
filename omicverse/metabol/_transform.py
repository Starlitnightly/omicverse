r"""Feature-level transformations for metabolomics.

Between-feature scaling is separate from between-sample normalization
(`_norm.py`) because the aims are different: sample normalization
removes dilution bias, feature transformation makes features
statistically comparable regardless of their absolute intensity range.

Methods
-------
- **log / glog** — compress the dynamic range. ``log`` is the simple
  ``log(x + c)`` with a small pseudocount; ``glog`` is the generalized
  log ``log((x + sqrt(x²+c²))/2)`` that behaves more like linear near
  zero (Durbin et al 2002, matches MetaboAnalyst).
- **autoscaling** (= z-score) — mean-center and divide by SD. Puts all
  features on the same variance scale; often too aggressive for low-
  abundance features.
- **Pareto scaling** — mean-center and divide by ``sqrt(SD)``. Most
  commonly used in PLS-DA / OPLS-DA because it dampens but doesn't
  eliminate high-intensity features' contribution. MetaboAnalyst default.

Returns a *new* AnnData with transformed ``.X``; the pre-transform
matrix is kept in ``adata.layers['raw']``.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from anndata import AnnData


TransformMethod = Literal["log", "glog", "autoscale", "pareto", "none"]


def transform(
    adata: AnnData,
    *,
    method: TransformMethod = "log",
    pseudocount: float = 1.0,
    stash_raw: bool = True,
) -> AnnData:
    """Apply a feature-level transformation to ``adata.X``.

    Parameters
    ----------
    method
        ``"log"``, ``"glog"``, ``"autoscale"``, ``"pareto"``, or ``"none"``.
    pseudocount
        For ``"log"`` and ``"glog"``. Default 1.0 (the MetaboAnalyst
        default for concentration data).
    stash_raw
        If True, stash the pre-transform matrix in ``layers['raw']``
        so later steps (e.g. absolute-value plots) can retrieve it.
    """
    out = adata.copy()
    if stash_raw and "raw" not in out.layers:
        out.layers["raw"] = out.X.copy()
    X = out.X.astype(np.float64, copy=True)

    if method == "log":
        X = np.log2(np.where(X > 0, X, pseudocount) + pseudocount)
    elif method == "glog":
        X = _glog(X, pseudocount=pseudocount)
    elif method == "autoscale":
        X = _autoscale(X)
    elif method == "pareto":
        X = _pareto(X)
    elif method == "none":
        pass
    else:
        raise ValueError(f"unknown method={method!r}")

    out.X = X
    out.uns.setdefault("metabol", {})["transform"] = method
    return out


def _glog(X: np.ndarray, *, pseudocount: float) -> np.ndarray:
    """Generalized log: log2((x + sqrt(x^2 + c^2)) / 2).

    Matches MetaboAnalyst's glog (Durbin-Rocke 2002). Equivalent to
    ``log2`` at large x, linear near zero — avoids the variance
    inflation that plain ``log`` gives when x ≈ 0.
    """
    c = float(pseudocount)
    return np.log2((X + np.sqrt(X * X + c * c)) / 2.0)


def _autoscale(X: np.ndarray) -> np.ndarray:
    """Z-score each column (feature)."""
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def _pareto(X: np.ndarray) -> np.ndarray:
    """Mean-center and divide by sqrt(SD) — MetaboAnalyst / OPLS-DA default."""
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / np.sqrt(sd)
