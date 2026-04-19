r"""Differential correlation for metabolomics.

Classical differential analysis asks *which metabolites change in
abundance?* Differential correlation asks a complementary question:
*which metabolite-metabolite relationships change between
conditions?* Two metabolites might have identical mean levels in A
and B yet be tightly co-regulated in one and uncorrelated in the
other — evidence for a rewired pathway.

The implementation follows DGCA (McKenzie et al., *BMC Genomics*
2016): Fisher-z transformation of each pair's correlation in each
condition, z-test on the difference, Benjamini-Hochberg FDR, and a
categorical class label describing the direction of rewiring (gain,
loss, inversion).

For moderate ``p`` (< ~500 metabolites) the full p × p matrix is fast
because the engine is vectorised ``np.corrcoef``. For larger panels
pass ``features=<list>`` to restrict the pair enumeration.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats

from .._registry import register_function
from ._utils import bh_fdr


@register_function(
    aliases=[
        'dgca',
        'differential_correlation',
        '差异相关',
    ],
    category='metabolomics',
    description='Differential Gene/metabolite Correlation Analysis (McKenzie 2016) — compare metabolite-metabolite correlations between two groups. Fisher-z-transformed z-test per pair, BH FDR, class labels (+/+, +/0, +/-, ...).',
    examples=[
        "ov.metabol.dgca(adata, group_col='group', group_a='case', group_b='ctrl')",
        "ov.metabol.dgca(adata, group_col='group', features=top20, method='spearman')",
    ],
    related=[
        'metabol.differential',
        'metabol.msea_ora',
    ],
)
def dgca(
    adata: AnnData,
    *,
    group_col: str,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
    method: str = "pearson",
    abs_r_threshold: float = 0.3,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """Differential correlation between two groups.

    Parameters
    ----------
    group_col
        Column in ``adata.obs`` with the two-class labels.
    group_a, group_b
        Labels to contrast. ``None`` → first two unique values.
        The z-diff direction is ``z_a - z_b`` (so positive ``z_diff``
        means correlation is stronger in group A).
    features
        Optional list of metabolite names to restrict the pair
        enumeration to. ``None`` → all. For ``p`` features the output
        has ``p*(p-1)/2`` rows.
    method
        ``"pearson"`` (default) or ``"spearman"``. Spearman is
        rank-based and more appropriate for non-Gaussian metabolite
        distributions; Pearson is faster and matches the DGCA default.
    abs_r_threshold
        Threshold for calling a correlation "significant in group X"
        when assigning the ``dc_class`` label. Default 0.3 matches
        MetaboAnalyst's network module.
    layer
        AnnData layer (default ``None`` → ``adata.X``).

    Returns
    -------
    pd.DataFrame
        Long format, one row per feature pair, columns:

        - ``feature_a``, ``feature_b`` — metabolite names
        - ``r_a``, ``r_b`` — correlation in each group
        - ``z_diff`` — Fisher-z-transformed difference test statistic
        - ``pvalue``, ``padj`` — two-sided p and BH-FDR
        - ``dc_class`` — ``"+/+"``, ``"+/0"``, ``"+/-"``, ``"0/+"``,
          ``"0/-"``, ``"-/+"``, ``"-/-"``, or ``"0/0"``. The first
          symbol is group A, the second group B. ``+`` / ``-`` means
          ``|r| ≥ abs_r_threshold`` with that sign; ``0`` means below
          threshold.

        The result is sorted by ``padj`` ascending.
    """
    if group_col not in adata.obs.columns:
        raise KeyError(f"adata.obs has no column {group_col!r}")

    groups = adata.obs[group_col].astype(str).to_numpy()
    if group_a is None or group_b is None:
        unique = list(pd.unique(adata.obs[group_col]))
        if len(unique) < 2:
            raise ValueError(f"{group_col!r} has fewer than 2 levels")
        group_a = group_a or str(unique[0])
        group_b = group_b or str(unique[1])

    mask_a = groups == group_a
    mask_b = groups == group_b
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    if n_a < 4 or n_b < 4:
        raise ValueError(
            f"each group needs ≥4 samples; got {n_a} ({group_a}) / "
            f"{n_b} ({group_b})"
        )

    Xraw = adata.X if layer is None else adata.layers[layer]
    X = np.asarray(Xraw, dtype=np.float64)

    # Subset features
    var_names = list(adata.var_names)
    if features is not None:
        feat_list = list(features)
        miss = [f for f in feat_list if f not in var_names]
        if miss:
            raise KeyError(f"features not in adata.var_names: {miss[:5]}...")
        idx = np.array([var_names.index(f) for f in feat_list])
        X = X[:, idx]
        names = feat_list
    else:
        names = var_names

    X_a = X[mask_a]
    X_b = X[mask_b]

    if method == "spearman":
        X_a = _rankdata_cols(X_a)
        X_b = _rankdata_cols(X_b)
    elif method != "pearson":
        raise ValueError(f"unknown method={method!r}")

    # Vectorised correlation matrices. np.corrcoef needs features in rows.
    R_a = _fast_corr(X_a)
    R_b = _fast_corr(X_b)
    # Clamp to avoid atanh(±1) → ∞
    clip = 1.0 - 1e-10
    R_a_c = np.clip(R_a, -clip, clip)
    R_b_c = np.clip(R_b, -clip, clip)
    Z_a = np.arctanh(R_a_c)
    Z_b = np.arctanh(R_b_c)
    se = np.sqrt(1.0 / (n_a - 3) + 1.0 / (n_b - 3))
    Z_diff = (Z_a - Z_b) / se
    pvals = 2.0 * stats.norm.sf(np.abs(Z_diff))

    # Extract upper triangle only
    p = R_a.shape[0]
    iu, ju = np.triu_indices(p, k=1)
    r_a_arr = R_a[iu, ju]
    r_b_arr = R_b[iu, ju]
    z_arr = Z_diff[iu, ju]
    p_arr = pvals[iu, ju]
    padj = bh_fdr(p_arr)

    def _cls(r: float) -> str:
        if abs(r) < abs_r_threshold:
            return "0"
        return "+" if r > 0 else "-"

    dc = np.array([f"{_cls(ra)}/{_cls(rb)}"
                   for ra, rb in zip(r_a_arr, r_b_arr)])

    out = pd.DataFrame({
        "feature_a": [names[i] for i in iu],
        "feature_b": [names[j] for j in ju],
        "r_a": r_a_arr,
        "r_b": r_b_arr,
        "z_diff": z_arr,
        "pvalue": p_arr,
        "padj": padj,
        "dc_class": dc,
    })
    out = out.sort_values("padj", kind="stable").reset_index(drop=True)
    out.attrs["group_a"] = group_a
    out.attrs["group_b"] = group_b
    out.attrs["n_a"] = n_a
    out.attrs["n_b"] = n_b
    out.attrs["method"] = method
    return out


def _fast_corr(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix (feature × feature). NaN-safe via
    constant-feature guard; falls back to ``np.corrcoef`` otherwise."""
    # X is (n_samples, n_features)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(Xc, axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    Xn = Xc / sd
    # If any NaNs remain, zero them — contributes nothing to covariance
    Xn = np.nan_to_num(Xn, nan=0.0)
    n = X.shape[0]
    R = (Xn.T @ Xn) / (n - 1)
    # Clip numerical noise
    np.fill_diagonal(R, 1.0)
    return R


def _rankdata_cols(X: np.ndarray) -> np.ndarray:
    """Column-wise rank transform for Spearman."""
    out = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        out[:, j] = stats.rankdata(X[:, j], nan_policy="omit")
    return out
