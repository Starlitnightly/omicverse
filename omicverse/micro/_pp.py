"""Compositional preprocessing for microbiome AnnData.

All functions operate on a ``samples × features`` AnnData with int counts in
``X``. Most write the result back into ``X`` (returning a new AnnData is not
default; pass ``copy=True`` to avoid in-place modification).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import anndata as ad
except ImportError as exc:  # pragma: no cover
    raise ImportError("anndata is required for ov.micro._pp") from exc

from .._registry import register_function


def _dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


@register_function(
    aliases=["rarefy", "subsample_depth"],
    category="microbiome",
    description="Subsample each sample's counts to a common depth without replacement (rarefaction).",
    examples=["ov.micro.rarefy(adata, depth=None, seed=0)"],
    related=["micro.pyAlpha", "micro.pyBeta"],
)
def rarefy(
    adata: "ad.AnnData",
    depth: Optional[int] = None,
    seed: int = 0,
    drop_shallow: bool = True,
    layer_out: str = "rarefied",
    copy: bool = False,
) -> "ad.AnnData":
    """Rarefy counts to a common depth.

    Parameters
    ----------
    adata
        Samples × features AnnData with int counts.
    depth
        Target depth. If ``None``, uses the minimum sample library size.
    seed
        Random seed for reproducibility.
    drop_shallow
        If True, samples with library size < depth are DROPPED. If False,
        their original counts are kept (no subsampling possible below depth).
    layer_out
        Layer name where the rarefied matrix is stored. ``X`` is overwritten.
        Set to an empty string to skip storing the original counts.
    copy
        Return a copy instead of modifying in place.
    """
    if copy:
        adata = adata.copy()

    X = _dense(adata.X).astype(np.int64)
    depths = X.sum(axis=1)
    if depth is None:
        depth = int(depths.min())

    rng = np.random.default_rng(seed)
    rarefied = np.zeros_like(X)
    keep = np.ones(X.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        row = X[i]
        tot = int(row.sum())
        if tot <= depth:
            if drop_shallow and tot < depth:
                keep[i] = False
                continue
            rarefied[i] = row
        else:
            idx = np.repeat(np.arange(len(row)), row.astype(int))
            pick = rng.choice(idx, size=depth, replace=False)
            rarefied[i] = np.bincount(pick, minlength=len(row))

    if not keep.all():
        adata._inplace_subset_obs(keep)
        rarefied = rarefied[keep]

    if layer_out:
        adata.layers["counts_raw"] = adata.X.copy()
    adata.X = sparse.csr_matrix(rarefied, dtype=np.int32)
    adata.uns.setdefault("micro", {})["rarefaction_depth"] = int(depth)
    return adata


@register_function(
    aliases=["filter_by_prevalence", "filter_rare_taxa"],
    category="microbiome",
    description="Drop features (ASVs/taxa) present in fewer than `min_prevalence` of samples.",
    examples=["ov.micro.filter_by_prevalence(adata, min_prevalence=0.1)"],
    related=["micro.rarefy", "micro.collapse_taxa"],
)
def filter_by_prevalence(
    adata: "ad.AnnData",
    min_prevalence: float = 0.1,
    min_count: int = 1,
    copy: bool = False,
) -> "ad.AnnData":
    """Filter rare features by prevalence.

    Parameters
    ----------
    min_prevalence
        Minimum fraction of samples in which a feature must have
        ``>= min_count`` reads. 0.1 = 10 % of samples.
    min_count
        Per-sample count threshold used to define "present".
    """
    if copy:
        adata = adata.copy()
    X = _dense(adata.X)
    presence = (X >= min_count).sum(axis=0) / X.shape[0]
    keep = presence >= min_prevalence
    adata._inplace_subset_var(keep)
    return adata


@register_function(
    aliases=["collapse_taxa", "agglomerate_taxa", "group_by_rank"],
    category="microbiome",
    description="Collapse ASV counts to a taxonomic rank (phylum / class / order / family / genus / species).",
    examples=["collapsed = ov.micro.collapse_taxa(adata, rank='genus')"],
    related=["micro.filter_by_prevalence"],
)
def collapse_taxa(
    adata: "ad.AnnData",
    rank: str = "genus",
    unassigned_label: str = "Unassigned",
) -> "ad.AnnData":
    """Collapse ASVs to a taxonomic rank.

    Returns a NEW AnnData where ``var_names`` are taxonomic labels at the
    chosen rank and counts are summed across ASVs sharing that label.
    """
    if rank not in adata.var.columns:
        raise ValueError(
            f"rank {rank!r} not found in adata.var columns "
            f"(available: {list(adata.var.columns)})"
        )
    labels = adata.var[rank].replace({"": unassigned_label}).fillna(unassigned_label).values
    X = _dense(adata.X)
    df = pd.DataFrame(X, index=adata.obs_names, columns=labels)
    agg = df.T.groupby(level=0).sum().T
    new = ad.AnnData(
        X=sparse.csr_matrix(agg.values, dtype=adata.X.dtype),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=agg.columns.rename(rank)),
    )
    new.uns["micro"] = dict(adata.uns.get("micro", {}))
    new.uns["micro"]["collapsed_rank"] = rank
    return new


# ---- compositional transforms ------------------------------------------------


def _pseudo_count(X, method: str = "multiplicative") -> np.ndarray:
    """Replace zeros before log-ratio transforms.

    method='multiplicative' — small fraction of the next-smallest non-zero
    value; standard in compositional data analysis.
    """
    X = X.astype(np.float64).copy()
    if method == "multiplicative":
        rowsum = X.sum(axis=1, keepdims=True)
        # per-row minimum non-zero proportion / 2
        for i in range(X.shape[0]):
            row = X[i]
            nz = row[row > 0]
            if len(nz) == 0:
                continue
            eps = nz.min() / 2.0
            row[row == 0] = eps
            X[i] = row
    else:
        X[X == 0] = 1e-6
    return X


@register_function(
    aliases=["clr_transform", "centered_log_ratio"],
    category="microbiome",
    description="Centred log-ratio transform of sample counts.",
    examples=["ov.micro.clr(adata, layer_out='clr')"],
    related=["micro.ilr", "micro.pyBeta"],
)
def clr(
    adata: "ad.AnnData",
    layer_out: str = "clr",
    copy: bool = False,
) -> "ad.AnnData":
    """CLR transform: ``log(x_i) - mean(log(x))`` per sample (post pseudo-count).

    Result is written to ``adata.layers[layer_out]``. Negative values are
    expected; this is a vector-space transform that removes closure.
    """
    if copy:
        adata = adata.copy()
    X = _dense(adata.X)
    Xp = _pseudo_count(X)
    logx = np.log(Xp)
    mean = logx.mean(axis=1, keepdims=True)
    adata.layers[layer_out] = logx - mean
    return adata


@register_function(
    aliases=["ilr_transform", "isometric_log_ratio"],
    category="microbiome",
    description="Isometric log-ratio transform of sample counts (orthonormal coords).",
    examples=["ov.micro.ilr(adata, layer_out='ilr')"],
    related=["micro.clr"],
)
def ilr(
    adata: "ad.AnnData",
    layer_out: str = "ilr",
    copy: bool = False,
) -> "ad.AnnData":
    """ILR transform — orthonormal coordinate system after closure removal.

    Stores an (n_samples × (n_features - 1)) matrix in ``obsm[layer_out]``
    (not ``layers`` because ILR changes dimensionality).
    """
    try:
        from skbio.stats.composition import ilr as _skbio_ilr
    except ImportError as exc:
        raise ImportError(
            "ov.micro.ilr requires scikit-bio (pip install scikit-bio)."
        ) from exc
    if copy:
        adata = adata.copy()
    X = _pseudo_count(_dense(adata.X))
    # normalise to proportions
    X = X / X.sum(axis=1, keepdims=True)
    coords = np.vstack([_skbio_ilr(X[i]) for i in range(X.shape[0])])
    adata.obsm[layer_out] = coords
    return adata
