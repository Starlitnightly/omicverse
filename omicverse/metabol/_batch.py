r"""QC-based batch / drift correction for LC-MS metabolomics.

ComBat (``ov.bulk.batch_correction``) handles batch *shifts* when the
batch variable is a known categorical — but the dominant confounder in
LC-MS untargeted runs is *within-batch signal drift* caused by
ionization-source fouling, column aging, and evaporation over an 8–24 h
run. Drift is per-feature, non-linear, and continuous in injection
order; categorical batch correction misses it.

SERRF (Fan et al., *Anal. Chem.* 2019) is the community-standard
answer. It uses the pooled quality-control (QC) injections that every
well-run study interleaves (every 5–10 real samples) to learn a
feature's expected intensity from its **correlated co-features** via a
random forest, then normalises every sample by the ratio of observed
to predicted QC abundance.

Recommended usage:

>>> adata = ov.metabol.drift_correct(adata, qc_col="sample_type",
...                                   qc_label="QC", order_col="order")
>>> adata = ov.metabol.serrf(adata, qc_col="sample_type", qc_label="QC",
...                           batch_col="batch")

SERRF subsumes LOESS drift correction in most cases — the RF picks up
non-monotonic drifts that LOESS misses — but the two are complementary:
LOESS corrects *per-feature in isolation*, SERRF *borrows strength*
from correlated features. Use both if your data has strong drift.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function


@register_function(
    aliases=[
        'serrf',
        'SERRF',
        'qc_rf_correction',
        'QC批次校正',
    ],
    category='metabolomics',
    description='SERRF (Fan 2019) — per-feature Random-Forest QC-based drift correction. Uses top-k correlated co-features as predictors of each feature, trains on QC samples, scales every sample by raw/predicted * mean(QC). The LC-MS community standard; supersedes LOESS drift for most runs.',
    examples=[
        "ov.metabol.serrf(adata, qc_col='sample_type', qc_label='QC')",
        "ov.metabol.serrf(adata, qc_col='sample_type', qc_label='QC', batch_col='batch', top_k=10)",
    ],
    related=[
        'metabol.drift_correct',
        'metabol.normalize',
    ],
)
def serrf(
    adata: AnnData,
    *,
    qc_col: str,
    qc_label: str = "QC",
    batch_col: Optional[str] = None,
    top_k: int = 10,
    n_estimators: int = 100,
    min_qc_samples: int = 5,
    layer: Optional[str] = None,
    seed: int = 0,
) -> AnnData:
    """SERRF — QC-based Random Forest drift correction (Fan 2019).

    For each feature and batch:

    1. Rank other features by absolute correlation with the target
       feature **across QC samples only**.
    2. Fit ``RandomForestRegressor`` with the target as response and the
       top-``top_k`` correlated features as predictors, trained on QC
       injections only.
    3. Predict the target abundance for every sample in the batch from
       its co-feature vector.
    4. Corrected value = ``raw / predicted * mean(QC_raw)``.

    The ratio scales each sample onto the QC baseline; features whose
    QC mean is zero or negative, or whose predictions collapse, are
    left uncorrected.

    Parameters
    ----------
    qc_col
        Column in ``adata.obs`` that tags QC vs real samples.
    qc_label
        Value in ``qc_col`` marking QC rows. Default ``"QC"``.
    batch_col
        Optional column in ``adata.obs`` for per-batch correction.
        Without a batch column the full run is treated as one batch.
    top_k
        Number of co-features to use as RF predictors. Fan 2019 uses
        10; larger values add runtime and marginal accuracy.
    n_estimators
        RF tree count. Default 100 (the Fan R package uses 500 but 100
        is within 1% AUC on benchmarks at 5× the speed).
    min_qc_samples
        If a batch has fewer QC samples than this, correction is
        skipped for that batch (features left as raw). Default 5.
    layer
        Which ``adata.layers`` entry to correct. ``None`` → use
        ``adata.X``.
    seed
        RandomForest seed.

    Returns
    -------
    AnnData
        New AnnData with corrected ``.X``. The original matrix is
        preserved in ``out.layers['raw']``. Per-feature CV% before /
        after is stored in ``out.var['cv_qc_raw']`` / ``cv_qc_serrf'']``
        to help users audit improvement.
    """
    if qc_col not in adata.obs.columns:
        raise KeyError(f"adata.obs has no column {qc_col!r}")

    out = adata.copy()
    Xraw = adata.X if layer is None else adata.layers[layer]
    X = np.asarray(Xraw, dtype=np.float64)
    out.layers["raw"] = X.copy()

    qc_mask = (adata.obs[qc_col].astype(str).to_numpy() == str(qc_label))
    if qc_mask.sum() < min_qc_samples:
        raise ValueError(
            f"need ≥{min_qc_samples} QC samples, got {qc_mask.sum()}"
        )

    if batch_col is None:
        batches = np.zeros(adata.n_obs, dtype=int)
    else:
        if batch_col not in adata.obs.columns:
            raise KeyError(f"adata.obs has no column {batch_col!r}")
        batches = adata.obs[batch_col].astype(str).to_numpy()

    corrected = X.copy()
    n, p = X.shape
    # Impute NaNs for RF — use per-feature median across all samples
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    nan_idx = np.isnan(X)
    if nan_idx.any():
        X = np.where(nan_idx, np.broadcast_to(col_med, X.shape), X)

    # Per-feature CV on QC (before correction)
    qc_raw = X[qc_mask]
    cv_before = _cv_pct(qc_raw)

    # RF import — kept inside function so importing this module is cheap
    from sklearn.ensemble import RandomForestRegressor

    # Precompute QC correlation matrix for top-k selection (features × features)
    # Correlation is global across all QC samples, not per-batch, since
    # per-batch correlations on typically 5-20 QC samples are noisy.
    qc_centered = qc_raw - qc_raw.mean(axis=0, keepdims=True)
    qc_std = qc_centered.std(axis=0, ddof=1)
    # Guard against constant QC features (would break corrcoef)
    valid = qc_std > 1e-12
    corr = np.zeros((p, p), dtype=np.float32)
    if valid.sum() > 1:
        sub = qc_centered[:, valid] / qc_std[valid]
        sub_corr = (sub.T @ sub) / (qc_raw.shape[0] - 1)
        idx_valid = np.where(valid)[0]
        for i_local, i_global in enumerate(idx_valid):
            corr[i_global, idx_valid] = sub_corr[i_local]

    for batch in np.unique(batches):
        b_mask = batches == batch
        b_qc_mask = b_mask & qc_mask
        if b_qc_mask.sum() < min_qc_samples:
            continue
        b_idx = np.where(b_mask)[0]
        b_qc_idx = np.where(b_qc_mask)[0]
        X_batch = X[b_idx]
        X_batch_qc = X[b_qc_idx]

        for j in range(p):
            qc_mean_j = float(X[b_qc_idx, j].mean())
            if not np.isfinite(qc_mean_j) or qc_mean_j <= 0:
                continue
            # Top-k co-features (exclude self)
            c = np.abs(corr[j]).copy()
            c[j] = -1.0
            top_idx = np.argsort(c)[::-1][:top_k]
            top_idx = top_idx[c[top_idx] > 0]
            if top_idx.size == 0:
                continue
            X_train = X_batch_qc[:, top_idx]
            y_train = X_batch_qc[:, j]
            if np.allclose(y_train.std(), 0.0):
                continue
            try:
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=1,
                )
                rf.fit(X_train, y_train)
                pred = rf.predict(X_batch[:, top_idx])
            except Exception:
                continue
            # Guard against near-zero predictions
            safe = pred > 1e-12 * max(qc_mean_j, 1.0)
            if not safe.any():
                continue
            corrected_batch_j = X_batch[:, j].copy()
            corrected_batch_j[safe] = (
                X_batch[safe, j] / pred[safe] * qc_mean_j
            )
            corrected[b_idx, j] = corrected_batch_j

    # Restore NaNs that were originally missing
    if nan_idx.any():
        corrected = np.where(nan_idx, np.nan, corrected)

    out.X = corrected
    # Per-feature QC CV after correction (used for reporting)
    cv_after = _cv_pct(corrected[qc_mask])
    out.var["cv_qc_raw"] = cv_before
    out.var["cv_qc_serrf"] = cv_after
    out.uns.setdefault("metabol", {})["batch_correction"] = "serrf"
    return out


def _cv_pct(X: np.ndarray) -> np.ndarray:
    """Per-column coefficient of variation in percent."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mu > 0, 100.0 * sd / mu, np.nan)
    return cv
