r"""Small shared utilities for ``omicverse.metabol``.

Everything here is an internal implementation detail — not part of the
public API. Keeps the same numerical routine (BH-FDR) from being
copy-pasted across ``_stats`` / ``_msea`` / ``_mummichog`` / ``_lipidomics``.
"""
from __future__ import annotations

import numpy as np


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR adjustment, pure NumPy.

    Monotonic in the original p-values (after BH-step-up correction),
    capped at 1.0. Accepts NaNs as NaNs in the output.
    """
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    # Step-up monotone transform: reverse cumulative min.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(ranked, 1.0)
    return out
