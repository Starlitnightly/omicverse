r"""MS-specific quality control.

Untargeted LC-MS data without proper QC gives unreliable downstream
results — this module implements the three filters that every serious
metabolomics pipeline does:

1. **CV%-based feature filter** (``cv_filter``) — drop features whose
   technical variation (measured on pooled QC samples) exceeds a
   threshold. 30% is the community standard; 20% for very stringent
   analysis.
2. **Signal-drift correction** (``drift_correct``) — LC-MS response
   drifts over long injection sequences; we fit a LOESS curve against
   injection order using the QC samples and divide it out.
3. **Blank filtering** (``blank_filter``) — features whose sample
   intensity isn't > ``ratio`` times the blank intensity are likely
   contamination peaks and get dropped.

Each function takes an AnnData, operates on ``.X``, and returns a
*new* AnnData (subset or transformed). Nothing is done in-place, so
the caller can chain these without surprises.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function


@register_function(
    aliases=[
        'cv_filter',
        'CV过滤',
        'qc_cv_filter',
    ],
    category='metabolomics',
    description='Drop metabolite features with coefficient-of-variation above a threshold across pooled QC samples (metabolomics-specific QC).',
    examples=[
        "ov.metabol.cv_filter(adata, qc_mask='is_qc', cv_threshold=0.30)",
    ],
    related=[
        'metabol.drift_correct',
        'metabol.blank_filter',
    ],
)
def cv_filter(
    adata: AnnData,
    *,
    qc_mask: str | np.ndarray,
    cv_threshold: float = 0.30,
) -> AnnData:
    """Drop features with coefficient-of-variation above ``cv_threshold`` in QC samples.

    Parameters
    ----------
    qc_mask
        Either the name of a boolean column in ``adata.obs`` (True = QC
        pool sample), or a boolean array of length ``adata.n_obs``.
    cv_threshold
        Features with ``std/mean > cv_threshold`` across QC samples are
        dropped. Default 0.30 is the community standard for untargeted
        LC-MS; lower values (0.20) for targeted / high-stringency.

    Returns
    -------
    AnnData
        Subset to the features that passed. Features dropped:
        ``adata.n_vars - out.n_vars``.
    """
    mask = _resolve_sample_mask(adata, qc_mask)
    qc_X = adata.X[mask]
    if qc_X.shape[0] < 3:
        raise ValueError(
            f"Need ≥3 QC samples for a meaningful CV filter, got {qc_X.shape[0]}"
        )
    mu = np.nanmean(qc_X, axis=0)
    sd = np.nanstd(qc_X, axis=0, ddof=1)
    # Avoid divide-by-zero on all-zero features
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mu > 0, sd / mu, np.inf)
    keep = cv <= cv_threshold
    out = adata[:, keep].copy()
    out.var["qc_cv"] = cv[keep]
    return out


@register_function(
    aliases=[
        'drift_correct',
        'LOESS漂移校正',
    ],
    category='metabolomics',
    description='Correct systematic LC-MS intensity drift across injection order via LOESS regression on pooled QC samples.',
    examples=[
        "ov.metabol.drift_correct(adata, injection_order='run_order', qc_mask='is_qc')",
    ],
    related=[
        'metabol.cv_filter',
        'metabol.blank_filter',
    ],
)
def drift_correct(
    adata: AnnData,
    *,
    injection_order: str | np.ndarray,
    qc_mask: str | np.ndarray,
    frac: float = 0.5,
) -> AnnData:
    """Correct systematic signal drift using LOESS regression on QC samples.

    Fits ``log1p(intensity) ~ injection_order`` per feature on QC
    samples only, then divides the raw intensities in both QC and real
    samples by the fitted drift curve.

    Parameters
    ----------
    injection_order
        Name of a numeric column in ``adata.obs`` (or a 1-D array) that
        gives the run order of each sample.
    qc_mask
        Bool column/array marking QC pool samples.
    frac
        LOESS smoothing fraction (``statsmodels.lowess`` ``frac``
        argument). 0.5 is robust for runs of <500 samples; tighten to
        0.3 for larger runs.

    Returns
    -------
    AnnData
        Intensity matrix corrected in place on a copy; ``uns['qc_drift']``
        records the per-feature LOESS fit for reproducibility.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "drift_correct needs statsmodels; `pip install statsmodels`."
        ) from exc

    order = _resolve_sample_array(adata, injection_order).astype(float)
    qmask = _resolve_sample_mask(adata, qc_mask)

    # Warn if real samples fall outside the QC injection range — np.interp
    # clamps at the boundaries rather than extrapolating, so the drift
    # correction for those edge samples is effectively ignored.
    qc_min, qc_max = order[qmask].min(), order[qmask].max()
    outside = ((order < qc_min) | (order > qc_max)) & ~qmask
    if outside.any():
        import warnings
        warnings.warn(
            f"{int(outside.sum())} real sample(s) were injected outside the "
            f"QC range [{qc_min:g}, {qc_max:g}]. Their drift correction uses "
            "the nearest-edge LOESS value (np.interp clamps; no extrapolation). "
            "Consider bracketing your run with QC pools.",
            UserWarning, stacklevel=2,
        )

    out = adata.copy()
    X = out.X.astype(np.float64, copy=True)
    fits = np.zeros_like(X)
    for j in range(X.shape[1]):
        y_log = np.log1p(np.where(X[qmask, j] > 0, X[qmask, j], 0))
        smooth = lowess(
            endog=y_log, exog=order[qmask], frac=frac, it=0, return_sorted=False
        )
        # Evaluate at all sample injection orders
        fit_all = np.interp(order, np.sort(order[qmask]),
                            smooth[np.argsort(order[qmask])])
        # Divide by drift curve (centered at QC median so magnitudes stay comparable)
        ref = np.median(fit_all[qmask])
        fits[:, j] = np.expm1(fit_all) / max(np.expm1(ref), 1e-6)
    X_corrected = np.where(fits > 0, X / fits, X)
    out.X = X_corrected
    out.uns["qc_drift"] = {"frac": frac, "ref": "median_qc"}
    return out


@register_function(
    aliases=[
        'blank_filter',
        '空白过滤',
    ],
    category='metabolomics',
    description="Drop features whose sample-mean intensity isn't at least ratio× the blank-mean intensity (contamination filter).",
    examples=[
        "ov.metabol.blank_filter(adata, blank_mask='is_blank', ratio=3.0)",
    ],
    related=[
        'metabol.cv_filter',
        'metabol.drift_correct',
    ],
)
def blank_filter(
    adata: AnnData,
    *,
    blank_mask: str | np.ndarray,
    ratio: float = 3.0,
) -> AnnData:
    """Drop features whose sample-mean intensity isn't at least ``ratio``×
    the blank-mean intensity.

    Parameters
    ----------
    blank_mask
        Column name (bool) or bool array marking blank / extraction-control
        samples in ``adata.obs``.
    ratio
        Features with ``mean(sample)/mean(blank) < ratio`` are dropped.
        3× is the community default; 5× for more stringent filtering.
    """
    bmask = _resolve_sample_mask(adata, blank_mask)
    if bmask.sum() == 0:
        raise ValueError("No blank samples found — pass a valid blank_mask.")
    real = ~bmask
    sample_mean = np.nanmean(adata.X[real], axis=0)
    blank_mean = np.nanmean(adata.X[bmask], axis=0)
    # Features never detected in blanks pass automatically.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(blank_mean > 0, sample_mean / blank_mean, np.inf)
    keep = r >= ratio
    out = adata[:, keep].copy()
    out.var["blank_ratio"] = r[keep]
    return out


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _resolve_sample_mask(adata: AnnData, mask_or_col: str | np.ndarray) -> np.ndarray:
    if isinstance(mask_or_col, str):
        if mask_or_col not in adata.obs.columns:
            raise KeyError(f"adata.obs has no column {mask_or_col!r}")
        return adata.obs[mask_or_col].astype(bool).to_numpy()
    arr = np.asarray(mask_or_col, dtype=bool)
    if arr.shape != (adata.n_obs,):
        raise ValueError(
            f"mask shape {arr.shape} does not match n_obs={adata.n_obs}"
        )
    return arr


def _resolve_sample_array(adata: AnnData, col_or_arr: str | np.ndarray) -> np.ndarray:
    if isinstance(col_or_arr, str):
        if col_or_arr not in adata.obs.columns:
            raise KeyError(f"adata.obs has no column {col_or_arr!r}")
        return adata.obs[col_or_arr].to_numpy()
    arr = np.asarray(col_or_arr)
    if arr.shape != (adata.n_obs,):
        raise ValueError(
            f"array shape {arr.shape} does not match n_obs={adata.n_obs}"
        )
    return arr
