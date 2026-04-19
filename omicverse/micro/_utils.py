"""Shared helpers for ``ov.micro`` submodules.

Tiny utilities that used to be copy-pasted across ``_da.py`` / ``_diversity.py``
/ ``_pp.py``. Kept private (underscore-prefixed) to avoid locking the API.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse


def dense(X) -> np.ndarray:
    """Return ``X`` as a dense ``np.ndarray`` (no-op if already dense)."""
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def rarefy_counts(
    X: np.ndarray,
    depth: Optional[int],
    seed: int = 0,
) -> np.ndarray:
    """Row-wise subsample without replacement to ``depth``.

    Rows whose library size is already ≤ ``depth`` are returned unchanged.
    This intentionally does *not* drop rows — callers filter upstream if
    they want that behaviour.
    """
    if depth is None:
        return np.asarray(X, dtype=np.int64)
    X = np.asarray(X, dtype=np.int64)
    rng = np.random.default_rng(seed)
    depth = int(depth)
    rar = np.zeros_like(X)
    for i, row in enumerate(X):
        tot = int(row.sum())
        if tot <= depth:
            rar[i] = row
        else:
            idx = np.repeat(np.arange(len(row)), row.astype(int))
            pick = rng.choice(idx, size=depth, replace=False)
            rar[i] = np.bincount(pick, minlength=len(row))
    return rar
