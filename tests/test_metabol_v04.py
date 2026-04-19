"""v0.4 tests for ov.metabol — anova, meba, sample_qc, corr_network, run_mofa."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


# --------------------------------------------------------------------------- #
# anova — 3+ groups
# --------------------------------------------------------------------------- #
def _make_multigroup_adata(n_per=15, n_features=20, n_groups=4,
                            effect_features=5, effect_size=2.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.concatenate([[f"g{g}"] * n_per for g in range(n_groups)])
    n = n_per * n_groups
    X = rng.standard_normal((n, n_features)) * 0.5
    # Linear dose-response on the first ``effect_features`` features
    for g in range(n_groups):
        X[labels == f"g{g}", :effect_features] += g * effect_size
    obs = pd.DataFrame({"group": labels},
                       index=[f"s{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"feat{i}" for i in range(n_features)])
    return AnnData(X=X, obs=obs, var=var)


class TestANOVA:
    def test_welch_anova_detects_planted(self):
        from omicverse.metabol import anova
        adata = _make_multigroup_adata(seed=0)
        df = anova(adata, group_col="group", method="welch_anova")
        sig = df[df["padj"] < 0.05].index.tolist()
        planted = {f"feat{i}" for i in range(5)}
        assert len(set(sig) & planted) >= 4
        for col in ["stat", "pvalue", "padj", "n_groups",
                    "mean_g0", "mean_g1", "mean_g2", "mean_g3"]:
            assert col in df.columns
        assert (df["n_groups"] == 4).all()

    def test_kruskal(self):
        from omicverse.metabol import anova
        adata = _make_multigroup_adata(seed=1)
        df = anova(adata, group_col="group", method="kruskal")
        assert df["padj"].min() < 0.01

    def test_classic_anova(self):
        from omicverse.metabol import anova
        adata = _make_multigroup_adata(seed=2)
        df = anova(adata, group_col="group", method="anova")
        assert df["padj"].min() < 0.01

    def test_two_groups_raises(self):
        from omicverse.metabol import anova
        adata = _make_multigroup_adata(n_groups=2, seed=3)
        with pytest.raises(ValueError, match="needs ≥3 groups"):
            anova(adata, group_col="group")


# --------------------------------------------------------------------------- #
# MEBA — time-series Hotelling T²
# --------------------------------------------------------------------------- #
def _make_time_series_adata(n_subj=8, n_times=4, n_features=15,
                             time_effect=1.0, group_effect=2.0, seed=0):
    rng = np.random.default_rng(seed)
    subjects = [f"s{i}" for i in range(n_subj)]
    group_a = subjects[:n_subj // 2]
    group_b = subjects[n_subj // 2:]
    rows, idx = [], []
    for s in subjects:
        g = "A" if s in group_a else "B"
        for t in range(n_times):
            rows.append({"subject": s, "group": g, "time": f"t{t}"})
            idx.append(f"{s}_t{t}")
    obs = pd.DataFrame(rows, index=idx)
    n = len(obs)
    X = rng.standard_normal((n, n_features)) * 0.3
    # Time effect on features 5..9: linear increase
    for t in range(n_times):
        tmask = (obs["time"] == f"t{t}").to_numpy()
        X[tmask, 5:10] += t * time_effect
    # Group × time interaction on features 0..4: drug group gains
    # more over time
    for t in range(n_times):
        tmask = (obs["time"] == f"t{t}").to_numpy()
        gmask = (obs["group"] == "B").to_numpy()
        X[tmask & gmask, 0:5] += t * group_effect
    var = pd.DataFrame(index=[f"feat{i}" for i in range(n_features)])
    return AnnData(X=X, obs=obs, var=var)


class TestMEBA:
    def test_detects_group_by_time_pattern(self):
        from omicverse.metabol import meba
        adata = _make_time_series_adata(seed=0)
        df = meba(adata, group_col="group", time_col="time",
                  subject_col="subject")
        planted = {f"feat{i}" for i in range(5)}
        sig = df[df["padj"] < 0.05].index.tolist()
        assert len(set(sig) & planted) >= 3, (
            f"Expected >=3 of 5 planted features in sig, got {set(sig) & planted}"
        )
        # Pure-time block (5..9) shares same trajectory across groups →
        # lower power but can still show up; check planted beats pure-time
        mean_pt = df.iloc[:5]["F"].mean()
        mean_time = df.iloc[5:10]["F"].mean()
        assert mean_pt > mean_time

    def test_returns_correct_schema(self):
        from omicverse.metabol import meba
        adata = _make_time_series_adata(n_subj=6, n_times=3, seed=1)
        df = meba(adata, group_col="group", time_col="time",
                  subject_col="subject")
        for col in ["T2", "F", "df1", "df2", "pvalue",
                    "padj", "n_a", "n_b", "k"]:
            assert col in df.columns
        assert (df["k"] == 3).all()
        assert (df["n_a"] + df["n_b"] == 6).all()

    def test_drops_unbalanced_subjects(self):
        from omicverse.metabol import meba
        adata = _make_time_series_adata(n_subj=6, n_times=3, seed=2)
        # Kill one cell — the first subject's t1 observation
        keep = adata.obs_names != "s0_t1"
        adata = adata[keep].copy()
        df = meba(adata, group_col="group", time_col="time",
                  subject_col="subject")
        assert "s0" in df.attrs["dropped_subjects"]
        assert df["n_a"].iloc[0] == 2  # s0 dropped → 2 remaining in A

    def test_insufficient_subjects_raises(self):
        from omicverse.metabol import meba
        adata = _make_time_series_adata(n_subj=2, n_times=4, seed=3)
        with pytest.raises(ValueError, match="balanced-design subjects"):
            meba(adata, group_col="group", time_col="time",
                 subject_col="subject")


# --------------------------------------------------------------------------- #
# sample_qc — Hotelling T² + DModX outliers
# --------------------------------------------------------------------------- #
def _make_outlier_adata(n=40, n_features=20, n_outliers=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # Plant outliers that deviate in a subspace AND normal to PC
    for i in range(n_outliers):
        X[i] += rng.standard_normal(n_features) * 5.0
    obs = pd.DataFrame(index=[f"s{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"feat{i}" for i in range(n_features)])
    return AnnData(X=X, obs=obs, var=var)


class TestSampleQC:
    def test_flags_outliers(self):
        from omicverse.metabol import sample_qc
        adata = _make_outlier_adata(seed=0)
        df = sample_qc(adata, n_components=3, alpha=0.95)
        flagged = df[df["is_outlier"]].index.tolist()
        planted = {"s0", "s1", "s2"}
        assert len(set(flagged) & planted) >= 2, (
            f"Expected >=2 of s0/s1/s2, got flagged={flagged}"
        )

    def test_returns_expected_schema(self):
        from omicverse.metabol import sample_qc
        adata = _make_outlier_adata(seed=1)
        df = sample_qc(adata, n_components=2)
        for col in ["T2", "DModX", "T2_crit", "DModX_crit",
                    "T2_flag", "DModX_flag", "is_outlier"]:
            assert col in df.columns
        assert df.shape[0] == adata.n_obs
        assert "variance_explained" in df.attrs

    def test_handles_nans(self):
        from omicverse.metabol import sample_qc
        adata = _make_outlier_adata(seed=2)
        adata.X[5, 0] = np.nan
        adata.X[10, 3] = np.nan
        df = sample_qc(adata, n_components=2)
        assert np.isfinite(df["T2"]).all()


# --------------------------------------------------------------------------- #
# corr_network
# --------------------------------------------------------------------------- #
def _make_correlated_adata(n=60, seed=0):
    rng = np.random.default_rng(seed)
    p = 10
    X = rng.standard_normal((n, p))
    # Plant: feat0, feat1 strongly correlated
    shared = rng.standard_normal(n)
    X[:, 0] = shared + 0.05 * rng.standard_normal(n)
    X[:, 1] = shared + 0.05 * rng.standard_normal(n)
    # feat2, feat3 strongly correlated
    s2 = rng.standard_normal(n)
    X[:, 2] = s2 + 0.05 * rng.standard_normal(n)
    X[:, 3] = s2 + 0.05 * rng.standard_normal(n)
    obs = pd.DataFrame({"group": ["A"] * (n // 2) + ["B"] * (n - n // 2)},
                       index=[f"s{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"feat{i}" for i in range(p)])
    return AnnData(X=X, obs=obs, var=var)


class TestCorrNetwork:
    def test_detects_planted_pairs(self):
        from omicverse.metabol import corr_network
        adata = _make_correlated_adata(seed=0)
        edges = corr_network(adata, abs_r_threshold=0.8, padj_threshold=0.01)
        pair_names = {frozenset([r["feature_a"], r["feature_b"]])
                      for _, r in edges.iterrows()}
        assert frozenset({"feat0", "feat1"}) in pair_names
        assert frozenset({"feat2", "feat3"}) in pair_names

    def test_group_filter(self):
        from omicverse.metabol import corr_network
        adata = _make_correlated_adata(seed=1)
        edges_a = corr_network(adata, group_col="group", group="A",
                                abs_r_threshold=0.5)
        edges_b = corr_network(adata, group_col="group", group="B",
                                abs_r_threshold=0.5)
        assert edges_a.attrs["n_samples"] <= adata.n_obs // 2 + 1
        assert edges_b.attrs["group"] == "B"

    def test_partial_group_args_raise(self):
        from omicverse.metabol import corr_network
        adata = _make_correlated_adata(seed=2)
        with pytest.raises(ValueError, match="both"):
            corr_network(adata, group_col="group")
        with pytest.raises(ValueError, match="both"):
            corr_network(adata, group="A")

    def test_feature_subset(self):
        from omicverse.metabol import corr_network
        adata = _make_correlated_adata(seed=3)
        edges = corr_network(adata, features=["feat0", "feat1", "feat4"],
                              abs_r_threshold=0.3)
        used = set()
        for _, r in edges.iterrows():
            used.update([r["feature_a"], r["feature_b"]])
        assert used <= {"feat0", "feat1", "feat4"}


# --------------------------------------------------------------------------- #
# run_mofa — bridge (needs mofapy2; skipped if not installed)
# --------------------------------------------------------------------------- #
class TestRunMOFA:
    def test_misaligned_obs_raises(self):
        from omicverse.metabol import run_mofa
        rng = np.random.default_rng(0)
        ad_m = AnnData(X=rng.standard_normal((10, 5)),
                       obs=pd.DataFrame(index=[f"s{i}" for i in range(10)]))
        ad_r = AnnData(X=rng.standard_normal((10, 7)),
                       obs=pd.DataFrame(index=[f"t{i}" for i in range(10)]))
        with pytest.raises(ValueError, match="obs_names do not match"):
            run_mofa({"metabol": ad_m, "rna": ad_r}, n_factors=3)

    def test_single_view_raises(self):
        from omicverse.metabol import run_mofa
        rng = np.random.default_rng(0)
        ad = AnnData(X=rng.standard_normal((10, 5)))
        with pytest.raises(ValueError, match="expects >=2 views"):
            run_mofa({"metabol": ad}, n_factors=3)

    @pytest.mark.skipif(
        not os.environ.get("OV_METABOL_MOFA_TESTS"),
        reason="mofa+ training is slow; enable with OV_METABOL_MOFA_TESTS=1",
    )
    def test_trains_and_returns_factors(self, tmp_path):
        pytest.importorskip("mofapy2")
        from omicverse.metabol import run_mofa
        rng = np.random.default_rng(0)
        n = 30
        samples = [f"s{i}" for i in range(n)]
        obs = pd.DataFrame(index=samples)
        ad_m = AnnData(X=rng.standard_normal((n, 20)), obs=obs.copy(),
                       var=pd.DataFrame(index=[f"m{i}" for i in range(20)]))
        ad_r = AnnData(X=rng.standard_normal((n, 30)), obs=obs.copy(),
                       var=pd.DataFrame(index=[f"g{i}" for i in range(30)]))
        factors = run_mofa(
            {"metabol": ad_m, "rna": ad_r},
            n_factors=3,
            outfile=tmp_path / "mofa.hdf5",
            max_iter=50,
            seed=0,
        )
        assert factors.shape[0] == n
        assert factors.shape[1] >= 1
        assert list(factors.index) == samples
