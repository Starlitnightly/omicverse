r"""Missing-value imputation for metabolomics.

Metabolomics data contains two kinds of missingness that call for
different imputation strategies:

1. **"Missing at random" (MAR)** — technical dropout in LC-MS due to
   ion suppression, poor ionization, imperfect peak picking. Best
   handled by distance-based methods (kNN) or iterative regression
   (Random Forest).
2. **"Missing not at random" (MNAR)** — concentration below the limit
   of detection. Best handled by half-minimum or QRILC (Quantile
   Regression Imputation of Left-Censored data) which draw from the
   left tail of the feature's distribution.

Choosing the wrong method for the wrong missingness type is a known
source of false positives in downstream stats. A reasonable default
for LC-MS is ``method='qrilc'``; for NMR data where MNAR is rare,
``method='knn'`` is safer.

All functions return a new AnnData; nothing is done in place.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from anndata import AnnData
from scipy import stats

from .._registry import register_function


ImputeMethod = Literal["knn", "half_min", "qrilc", "zero"]


@register_function(
    aliases=[
        'impute',
        '代谢组填补',
        'qrilc',
        'knn_impute',
    ],
    category='metabolomics',
    description='Impute missing values with kNN / half-min / QRILC / zero — QRILC is the recommended MNAR default for LC-MS, kNN for MAR-dominant NMR.',
    examples=[
        "ov.metabol.impute(adata, method='qrilc', seed=0)",
    ],
    related=[
        'metabol.normalize',
        'metabol.transform',
    ],
)
def impute(
    adata: AnnData,
    *,
    method: ImputeMethod = "qrilc",
    missing_threshold: float = 0.5,
    n_neighbors: int = 5,
    q: float = 0.01,
    seed: int = 0,
) -> AnnData:
    """Impute missing values (NaN / 0) in ``adata.X``.

    Parameters
    ----------
    method
        - ``"knn"`` — per-feature kNN on the sample × sample Euclidean
          distance restricted to non-missing positions
        - ``"half_min"`` — replace missing with half the minimum non-missing
          value of that feature (classic MNAR-friendly default)
        - ``"qrilc"`` — Quantile Regression Imputation of Left-Censored:
          draws from ``TruncatedNormal(mean=mu_below_q, sd=sigma)`` where
          mu and sigma are estimated from values below the ``q``-quantile
        - ``"zero"`` — replace missing with 0 (for downstream methods
          that treat 0 as a valid observation)
    missing_threshold
        Before imputation, drop features whose missingness exceeds this
        fraction. Default 0.5 (drop a feature missing in >50% of samples).
    n_neighbors
        kNN neighborhood size (only used when ``method='knn'``).
    q
        Quantile defining the "below-detection-limit" band for QRILC.
    seed
        RNG seed for the QRILC truncated-normal draws. Default 0 — pass
        a different integer to bootstrap or change the imputation
        realization across pipeline runs.

    Returns
    -------
    AnnData
        New object with imputed ``.X`` and a ``var['missing_frac']``
        column recording the pre-imputation missingness of each feature.
    """
    out = adata.copy()
    X = out.X.astype(np.float64, copy=True)
    # Both NaN and 0 are treated as missing — MetaboAnalyst convention.
    missing = np.isnan(X) | (X == 0)
    frac = missing.mean(axis=0)
    keep = frac <= missing_threshold
    out = out[:, keep].copy()
    X = X[:, keep]
    missing = missing[:, keep]
    out.var["missing_frac"] = frac[keep]

    if method == "zero":
        X[missing] = 0.0
    elif method == "half_min":
        for j in range(X.shape[1]):
            col_missing = missing[:, j]
            if col_missing.all():
                X[col_missing, j] = 0.0
                continue
            mn = np.nanmin(np.where(missing[:, j], np.nan, X[:, j]))
            X[col_missing, j] = mn / 2.0 if np.isfinite(mn) and mn > 0 else 0.0
    elif method == "knn":
        X = _knn_impute(X, missing, n_neighbors=n_neighbors)
    elif method == "qrilc":
        X = _qrilc_impute(X, missing, q=q, seed=seed)
    else:
        raise ValueError(f"unknown method={method!r}")

    out.X = X
    return out


def _knn_impute(X: np.ndarray, missing: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Feature-wise kNN on sample distances (non-missing positions only)."""
    from sklearn.impute import KNNImputer

    # KNNImputer expects NaN; convert back and forth.
    X_nan = np.where(missing, np.nan, X)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    return imputer.fit_transform(X_nan)


def _qrilc_impute(X: np.ndarray, missing: np.ndarray, q: float, seed: int = 0) -> np.ndarray:
    """Quantile Regression Imputation of Left-Censored data.

    For each feature: estimate the left-tail mean μ and sd σ from the
    ``q``-quantile of the log-transformed observed values, then impute
    missing entries as ``exp(TruncatedNormal(μ, σ, upper=log(q-quantile)))``.
    Works in log-space to keep intensities non-negative.

    ``seed`` is threaded through from ``impute(..., seed=...)``. Deterministic.
    """
    rng = np.random.default_rng(seed)
    out = X.copy()
    for j in range(X.shape[1]):
        col_missing = missing[:, j]
        n_miss = int(col_missing.sum())
        if n_miss == 0:
            continue
        obs = X[~col_missing, j]
        if obs.size < 2 or np.all(obs <= 0):
            out[col_missing, j] = 0.0
            continue
        # Work in log-space so draws can't go negative
        log_obs = np.log(obs[obs > 0])
        if log_obs.size < 2:
            out[col_missing, j] = obs.min() / 2.0
            continue
        threshold = np.quantile(log_obs, q)
        below = log_obs[log_obs <= threshold] if np.any(log_obs <= threshold) else log_obs
        mu = below.mean()
        sigma = below.std(ddof=1) if below.size >= 2 else log_obs.std(ddof=1)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1e-3
        # Truncated normal — upper bound at the q-quantile
        a = (-np.inf - mu) / sigma
        b = (threshold - mu) / sigma
        draws = stats.truncnorm.rvs(a, b, loc=mu, scale=sigma,
                                    size=n_miss, random_state=rng)
        out[col_missing, j] = np.exp(draws)
    return out
