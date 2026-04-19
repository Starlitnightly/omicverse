"""Offline tests for ``omicverse.metabol.serrf`` and ``dgca``.
Synthetic data with planted drift / planted differential correlation
keeps assertions deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


# --------------------------------------------------------------------------- #
# SERRF
# --------------------------------------------------------------------------- #
def _make_drift_adata(
    n_real_per_batch: int = 20,
    n_qc_per_batch: int = 6,
    n_batches: int = 1,
    n_features: int = 30,
    drift_features: int = 10,
    drift_strength: float = 2.0,
    seed: int = 0,
) -> AnnData:
    """Simulate an LC-MS run with QC-trackable drift.

    A subset of features has a steady injection-order drift
    (multiplicative, ``exp(k * t)``) so CV on QC pools gets large before
    correction. Non-drift features stay stable. Real samples have a
    small group effect on a *different* subset so SERRF doesn't wipe
    biology.
    """
    rng = np.random.default_rng(seed)
    rows_obs: list[dict] = []
    idx: list[str] = []
    order = 0
    for b in range(n_batches):
        # Sprinkle QC injections evenly
        plan = (["QC"] * n_qc_per_batch
                + ["real"] * n_real_per_batch)
        rng.shuffle(plan)
        for k, kind in enumerate(plan):
            group = "case" if rng.random() > 0.5 else "ctrl"
            rows_obs.append({
                "sample_type": kind,
                "group": "QC" if kind == "QC" else group,
                "batch": f"b{b}",
                "order": order,
            })
            idx.append(f"b{b}_s{k}")
            order += 1

    obs = pd.DataFrame(rows_obs, index=idx)
    n = len(obs)
    base = rng.lognormal(mean=4.0, sigma=0.3, size=(n, n_features))

    # Drift applied to first ``drift_features``: exp(drift_strength *
    # order/max_order) so later injections have systematically higher
    # signal on those features. Sanity-check with a single batch.
    max_order = float(obs["order"].max())
    t = obs["order"].to_numpy() / max(max_order, 1.0)
    multiplier = np.exp(drift_strength * t)
    base[:, :drift_features] *= multiplier[:, None]

    # Small biology effect on a different feature block — SERRF must NOT
    # erase this.
    case_mask = (obs["group"] == "case").to_numpy()
    base[case_mask, 20:25] *= 1.8

    var = pd.DataFrame(index=[f"feat{i}" for i in range(n_features)])
    return AnnData(X=base, obs=obs, var=var)


class TestSERRF:
    def test_reduces_qc_cv_on_drift_features(self):
        from omicverse.metabol import serrf

        adata = _make_drift_adata(n_real_per_batch=30, n_qc_per_batch=10,
                                   drift_features=10, drift_strength=1.5,
                                   seed=1)
        corrected = serrf(adata, qc_col="sample_type", qc_label="QC",
                          top_k=8, n_estimators=50, seed=0)
        cv_raw = corrected.var["cv_qc_raw"].to_numpy()
        cv_ser = corrected.var["cv_qc_serrf"].to_numpy()
        # Drift features are 0..9 — their CV% should drop substantially
        drift_block_raw = cv_raw[:10]
        drift_block_ser = cv_ser[:10]
        assert np.nanmean(drift_block_ser) < np.nanmean(drift_block_raw)
        # Reduction should be meaningful (at least 30% on average)
        improvement = (np.nanmean(drift_block_raw)
                       - np.nanmean(drift_block_ser)) / np.nanmean(drift_block_raw)
        assert improvement > 0.30, f"only {improvement:.2%} CV reduction"

    def test_preserves_biology_signal(self):
        from omicverse.metabol import serrf

        adata = _make_drift_adata(n_real_per_batch=30, n_qc_per_batch=10,
                                   drift_features=5, drift_strength=1.0,
                                   seed=2)
        corrected = serrf(adata, qc_col="sample_type", qc_label="QC",
                          top_k=8, n_estimators=50, seed=0)
        # Real samples only, check case vs ctrl on the biology block
        # (features 20..24 are boosted in case by 80%)
        real_mask = (corrected.obs["sample_type"] == "real").to_numpy()
        real_obs = corrected.obs[real_mask]
        X_real = np.asarray(corrected.X)[real_mask]
        case = real_obs["group"].to_numpy() == "case"
        mean_case = X_real[case, 20:25].mean(axis=0)
        mean_ctrl = X_real[~case, 20:25].mean(axis=0)
        # case should remain noticeably higher
        assert (mean_case > mean_ctrl).mean() >= 0.6, "biology erased"

    def test_stores_raw_and_meta(self):
        from omicverse.metabol import serrf

        adata = _make_drift_adata(seed=3)
        corrected = serrf(adata, qc_col="sample_type", qc_label="QC",
                          top_k=5, n_estimators=30, seed=0)
        assert "raw" in corrected.layers
        assert corrected.layers["raw"].shape == corrected.X.shape
        assert "cv_qc_raw" in corrected.var.columns
        assert "cv_qc_serrf" in corrected.var.columns
        assert corrected.uns["metabol"]["batch_correction"] == "serrf"

    def test_multi_batch(self):
        from omicverse.metabol import serrf

        adata = _make_drift_adata(n_real_per_batch=20, n_qc_per_batch=8,
                                   n_batches=2, drift_features=8,
                                   drift_strength=1.2, seed=4)
        corrected = serrf(adata, qc_col="sample_type", qc_label="QC",
                          batch_col="batch", top_k=6, n_estimators=30,
                          seed=0)
        assert np.isfinite(corrected.var["cv_qc_serrf"]).any()

    def test_raises_on_too_few_qc(self):
        from omicverse.metabol import serrf

        adata = _make_drift_adata(n_qc_per_batch=2, seed=5)
        with pytest.raises(ValueError, match="QC samples"):
            serrf(adata, qc_col="sample_type", qc_label="QC",
                  min_qc_samples=5)


# --------------------------------------------------------------------------- #
# DGCA
# --------------------------------------------------------------------------- #
def _make_dgca_adata(n_per_group: int = 30, seed: int = 0) -> AnnData:
    """Two-group synthetic where features 0-1 are correlated only in
    group A, features 2-3 only in group B, features 4-5 invert sign."""
    rng = np.random.default_rng(seed)
    n = n_per_group * 2
    p = 8
    X = rng.standard_normal((n, p))
    # Group A: first n_per_group samples
    # Feature 0, 1 → highly correlated in A, uncorrelated in B
    shared = rng.standard_normal(n_per_group)
    X[:n_per_group, 0] = shared + 0.1 * rng.standard_normal(n_per_group)
    X[:n_per_group, 1] = shared + 0.1 * rng.standard_normal(n_per_group)
    # Feature 2, 3 → uncorrelated in A, highly correlated in B
    shared_b = rng.standard_normal(n_per_group)
    X[n_per_group:, 2] = shared_b + 0.1 * rng.standard_normal(n_per_group)
    X[n_per_group:, 3] = shared_b + 0.1 * rng.standard_normal(n_per_group)
    # Feature 4, 5 → positive correlation in A, negative in B
    shared_both = rng.standard_normal(n_per_group)
    X[:n_per_group, 4] = shared_both + 0.1 * rng.standard_normal(n_per_group)
    X[:n_per_group, 5] = shared_both + 0.1 * rng.standard_normal(n_per_group)
    shared_b2 = rng.standard_normal(n_per_group)
    X[n_per_group:, 4] = shared_b2 + 0.1 * rng.standard_normal(n_per_group)
    X[n_per_group:, 5] = -shared_b2 + 0.1 * rng.standard_normal(n_per_group)

    labels = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    obs = pd.DataFrame({"group": labels}, index=[f"s{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"feat{i}" for i in range(p)])
    return AnnData(X=X, obs=obs, var=var)


class TestDGCA:
    def test_basic_shape_and_columns(self):
        from omicverse.metabol import dgca

        adata = _make_dgca_adata(seed=10)
        df = dgca(adata, group_col="group")
        p = adata.n_vars
        assert len(df) == p * (p - 1) // 2
        for col in ["feature_a", "feature_b", "r_a", "r_b",
                    "z_diff", "pvalue", "padj", "dc_class"]:
            assert col in df.columns

    def test_detects_planted_pairs(self):
        from omicverse.metabol import dgca

        adata = _make_dgca_adata(n_per_group=80, seed=11)
        df = dgca(adata, group_col="group", abs_r_threshold=0.4)
        # Pair feat0-feat1 should have high |r_a|, low |r_b|
        pair_01 = df[((df["feature_a"] == "feat0") & (df["feature_b"] == "feat1"))
                     | ((df["feature_a"] == "feat1") & (df["feature_b"] == "feat0"))]
        assert len(pair_01) == 1
        row = pair_01.iloc[0]
        assert abs(row["r_a"]) > 0.7
        assert abs(row["r_b"]) < 0.4
        # Expected class: "+/0"
        assert row["dc_class"] == "+/0"
        assert row["padj"] < 0.01

    def test_detects_inversion(self):
        from omicverse.metabol import dgca

        adata = _make_dgca_adata(n_per_group=80, seed=12)
        df = dgca(adata, group_col="group", abs_r_threshold=0.4)
        pair_45 = df[((df["feature_a"] == "feat4") & (df["feature_b"] == "feat5"))
                     | ((df["feature_a"] == "feat5") & (df["feature_b"] == "feat4"))]
        assert len(pair_45) == 1
        row = pair_45.iloc[0]
        assert row["r_a"] > 0.7
        assert row["r_b"] < -0.7
        assert row["dc_class"] == "+/-"

    def test_spearman_vs_pearson_close(self):
        from omicverse.metabol import dgca

        adata = _make_dgca_adata(n_per_group=40, seed=13)
        df_p = dgca(adata, group_col="group", method="pearson")
        df_s = dgca(adata, group_col="group", method="spearman")
        # Ranking of padj should correlate highly
        common_key = df_p["feature_a"] + "~" + df_p["feature_b"]
        common_key_s = df_s["feature_a"] + "~" + df_s["feature_b"]
        assert set(common_key) == set(common_key_s)

    def test_feature_subset(self):
        from omicverse.metabol import dgca

        adata = _make_dgca_adata(seed=14)
        df = dgca(adata, group_col="group",
                  features=["feat0", "feat1", "feat4", "feat5"])
        assert len(df) == 4 * 3 // 2
        assert set(df["feature_a"]).union(df["feature_b"]) == {
            "feat0", "feat1", "feat4", "feat5"
        }

    def test_raises_on_unknown_feature(self):
        from omicverse.metabol import dgca

        adata = _make_dgca_adata(seed=15)
        with pytest.raises(KeyError):
            dgca(adata, group_col="group", features=["feat0", "nope"])
