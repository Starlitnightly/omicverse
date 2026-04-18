r"""Sample normalization for metabolomics.

All methods scale each **sample** (row) to make inter-sample
comparisons valid despite differences in dilution, injection volume,
or total extracted material. They do NOT touch between-feature scaling
(that's `_transform.py`).

Methods
-------
- **PQN** (Probabilistic Quotient Normalization; Dieterle 2006) — the
  de-facto standard for untargeted metabolomics. Divides each sample
  by the median of its ratios to a reference sample (median or mean
  of all samples).
- **TIC** (Total Ion Current) — divides each sample by its sum.
  Simple but assumes global intensity is biologically irrelevant.
- **median** — divides each sample by its median. Robust to a few
  high outlier features.
- **MSTUS** (MS Total Useful Signal) — TIC but only counting features
  that pass a missing-fraction threshold. Useful when a few noisy
  features would otherwise dominate the sum.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from anndata import AnnData


NormMethod = Literal["pqn", "tic", "median", "mstus"]


def normalize(
    adata: AnnData,
    *,
    method: NormMethod = "pqn",
    reference: Literal["median", "mean"] = "median",
    missing_threshold: float = 0.5,
) -> AnnData:
    """Normalize each sample (row) of ``adata.X`` to correct for dilution.

    Parameters
    ----------
    method
        ``"pqn"`` (Dieterle 2006, recommended), ``"tic"``, ``"median"``,
        or ``"mstus"``.
    reference
        Only used by ``"pqn"``. The reference sample is the
        element-wise median (robust) or mean (noisier) of all samples.
        MetaboAnalyst uses median by default; we match.
    missing_threshold
        Only used by ``"mstus"``. Features missing in a higher fraction
        of samples are excluded from the denominator sum.
    """
    out = adata.copy()
    X = out.X.astype(np.float64, copy=True)

    if method == "pqn":
        X = _pqn(X, reference=reference)
    elif method == "tic":
        tic = np.nansum(X, axis=1, keepdims=True)
        tic = np.where(tic == 0, 1.0, tic)
        X = X / tic * np.nanmedian(tic)
    elif method == "median":
        med = np.nanmedian(X, axis=1, keepdims=True)
        med = np.where(med == 0, 1.0, med)
        X = X / med * np.nanmedian(med)
    elif method == "mstus":
        stable = (np.isnan(X) | (X == 0)).mean(axis=0) <= missing_threshold
        tic = np.nansum(X[:, stable], axis=1, keepdims=True)
        tic = np.where(tic == 0, 1.0, tic)
        X = X / tic * np.nanmedian(tic)
    else:
        raise ValueError(f"unknown method={method!r}")

    out.X = X
    out.uns.setdefault("metabol", {})["normalization"] = method
    return out


def _pqn(X: np.ndarray, *, reference: str = "median") -> np.ndarray:
    """Probabilistic Quotient Normalization — Dieterle et al 2006."""
    # Reference spectrum
    if reference == "median":
        ref = np.nanmedian(X, axis=0)
    elif reference == "mean":
        ref = np.nanmean(X, axis=0)
    else:
        raise ValueError(f"unknown reference={reference!r}")

    # Avoid divide-by-zero on reference features that are all zero
    ref_safe = np.where(ref > 0, ref, np.nan)
    # Per-sample: median of non-zero, finite ratios to the reference
    quotients = X / ref_safe[None, :]
    quotients = np.where(np.isfinite(quotients) & (quotients > 0),
                         quotients, np.nan)
    scale = np.nanmedian(quotients, axis=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return X / scale
