"""Unit tests for ``omicverse.metabol._qc`` — CV% filter, drift correction,
blank filter. These are the most data-type-sensitive pieces of the
metabolomics pipeline (they operate on raw intensities before any
normalization) so they deserve focused tests beyond the end-to-end suite.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _adata(X: np.ndarray, obs_cols: dict | None = None, var_names=None):
    """Thin wrapper around AnnData so test bodies stay readable."""
    obs = pd.DataFrame(obs_cols or {}, index=[f"s{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=var_names or [f"m{j}" for j in range(X.shape[1])])
    return ad.AnnData(X=X.astype(np.float64), obs=obs, var=var)


# --------------------------------------------------------------------------- #
# cv_filter
# --------------------------------------------------------------------------- #
class TestCvFilter:
    def test_drops_high_cv_features_keeps_low_cv(self):
        from omicverse.metabol import cv_filter

        # Use deterministic values so the test doesn't depend on n-sample
        # sampling error. Feature 0 is rock-stable in QCs (CV=1%),
        # feature 1 is wildly variable (CV>=50%); cv_filter(threshold=0.30)
        # must drop feature 1.
        X_real = np.array([[100.0, 50.0]] * 5)                     # real cells
        X_qc = np.array([[100.0, 10.0],
                         [101.0, 80.0],
                         [ 99.0, 200.0],
                         [100.5, 50.0],
                         [ 99.5, 30.0]])
        X = np.vstack([X_real, X_qc])
        adata = _adata(X, obs_cols={"is_qc": [False] * 5 + [True] * 5})
        out = cv_filter(adata, qc_mask="is_qc", cv_threshold=0.30)
        assert out.n_vars == 1                       # only stable feature kept
        assert "qc_cv" in out.var.columns
        assert out.n_obs == adata.n_obs              # cv_filter never drops samples

    def test_accepts_bool_array_mask(self):
        from omicverse.metabol import cv_filter

        rng = np.random.default_rng(1)
        X = np.vstack([
            rng.normal(100, 5, size=(3, 3)),          # real
            rng.normal(100, 0.5, size=(4, 3)),        # very-stable QCs
        ])
        adata = _adata(X)
        mask = np.array([False, False, False, True, True, True, True])
        out = cv_filter(adata, qc_mask=mask, cv_threshold=0.10)
        assert out.n_vars == 3     # all survive the tight QC threshold

    def test_needs_at_least_3_qc_samples(self):
        from omicverse.metabol import cv_filter

        adata = _adata(np.ones((5, 2)), obs_cols={
            "is_qc": [True, True, False, False, False],
        })
        # Match the unicode "≥" that the error message actually uses
        with pytest.raises(ValueError, match="3 QC samples"):
            cv_filter(adata, qc_mask="is_qc", cv_threshold=0.30)

    def test_missing_column_raises_keyerror(self):
        from omicverse.metabol import cv_filter

        adata = _adata(np.ones((5, 2)))
        with pytest.raises(KeyError, match="no column"):
            cv_filter(adata, qc_mask="nonexistent", cv_threshold=0.30)


# --------------------------------------------------------------------------- #
# drift_correct
# --------------------------------------------------------------------------- #
class TestDriftCorrect:
    def test_corrects_linear_drift(self):
        from omicverse.metabol import drift_correct

        # Simulate linear drift: every sample multiplied by a factor that
        # grows with injection order. After correction the QC samples should
        # all come back to roughly the same intensity.
        n = 30
        order = np.arange(n, dtype=float)
        drift = 1.0 + 0.02 * order              # up to 60% drift over 30 samples
        true = np.full((n, 2), 100.0)
        X = true * drift[:, None]
        is_qc = (order % 3 == 0)                # every 3rd sample is a pool QC
        adata = _adata(X, obs_cols={"run": order, "is_qc": is_qc.tolist()})
        out = drift_correct(adata, injection_order="run", qc_mask="is_qc", frac=0.5)
        # Post-correction, QC samples should all be near-identical — the
        # span (max/min) should shrink toward 1.0 (true constant value).
        qc_after = out.X[is_qc]
        qc_before = adata.X[is_qc]
        span_before = qc_before.max() / qc_before.min()
        span_after = qc_after.max() / qc_after.min()
        assert span_before > 1.3                # pre-correction drift is visible
        assert span_after < 1.05                # post-correction QCs within 5%

    def test_warns_when_real_samples_outside_qc_range(self):
        from omicverse.metabol import drift_correct

        n = 10
        order = np.arange(n, dtype=float)
        X = np.full((n, 2), 100.0) * (1 + 0.01 * order[:, None])
        # QC samples *only* in the middle — first and last 2 real samples
        # are outside the QC bracket
        is_qc = (order >= 3) & (order <= 6)
        adata = _adata(X, obs_cols={"run": order, "is_qc": is_qc.tolist()})
        with pytest.warns(UserWarning, match="outside the QC range"):
            drift_correct(adata, injection_order="run", qc_mask="is_qc")


# --------------------------------------------------------------------------- #
# blank_filter
# --------------------------------------------------------------------------- #
class TestBlankFilter:
    def test_drops_features_close_to_blank(self):
        from omicverse.metabol import blank_filter

        # Feature 0: sample mean 100, blank mean 2 → ratio 50 (keep)
        # Feature 1: sample mean 10,  blank mean 8 → ratio 1.25 (drop at ratio=3)
        X_samples = np.array([[100, 10], [110, 12], [95, 9]], dtype=float)
        X_blanks  = np.array([[2, 8], [2.5, 9], [1.5, 7]], dtype=float)
        X = np.vstack([X_samples, X_blanks])
        is_blank = np.array([False] * 3 + [True] * 3)
        adata = _adata(X, obs_cols={"is_blank": is_blank.tolist()})
        out = blank_filter(adata, blank_mask="is_blank", ratio=3.0)
        assert out.n_vars == 1        # feature 1 dropped
        assert "blank_ratio" in out.var.columns

    def test_errors_without_blanks(self):
        from omicverse.metabol import blank_filter

        adata = _adata(np.ones((4, 2)), obs_cols={"is_blank": [False] * 4})
        with pytest.raises(ValueError, match="No blank samples"):
            blank_filter(adata, blank_mask="is_blank", ratio=3.0)

    def test_feature_never_in_blanks_passes_unconditionally(self):
        from omicverse.metabol import blank_filter

        # Feature 0 is undetected in blanks (all zero) — should pass regardless
        # of ratio threshold since sample/blank = inf.
        X = np.vstack([
            np.array([[100, 1], [120, 2]]),    # samples
            np.array([[0,   0], [0,   0]]),    # blanks
        ])
        adata = _adata(X, obs_cols={"is_blank": [False, False, True, True]})
        out = blank_filter(adata, blank_mask="is_blank", ratio=10.0)
        assert "m0" in out.var_names
        # feature 1 might or might not pass depending on sample/0 handling
