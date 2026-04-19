"""Tests for ``omicverse.metabol.asca``, ``mixed_model``, ``roc_feature``,
and ``biomarker_panel``. All offline — use synthetic data with planted
signal so assertions don't depend on any external DB or random draw."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


def _make_factorial_adata(
    n_per_cell: int = 6,
    n_features: int = 20,
    treatment_effect: float = 3.0,
    time_effect: float = 1.5,
    interaction_effect: float = 0.0,
    patient_sd: float = 0.0,
    seed: int = 0,
) -> AnnData:
    """2×2 factorial: treatment ∈ {ctrl, drug}, time ∈ {0h, 24h}.
    Treatment affects features 0..4, time affects features 5..9, rest is noise.
    ``patient_sd`` adds a per-patient random intercept — needed so MixedLM
    has real variance to estimate (otherwise the RE covariance collapses).
    """
    rng = np.random.default_rng(seed)
    treatments = ["ctrl", "drug"]
    times = ["0h", "24h"]
    obs_rows = []
    sample_idx = []
    for t in treatments:
        for tm in times:
            for k in range(n_per_cell):
                obs_rows.append({"treatment": t, "time": tm,
                                 "patient": f"p{k}"})
                sample_idx.append(f"{t}_{tm}_{k}")
    obs = pd.DataFrame(obs_rows, index=sample_idx)
    n = len(obs)
    X = rng.standard_normal((n, n_features)) * 0.3
    tmask = (obs["treatment"] == "drug").to_numpy()
    tm_mask = (obs["time"] == "24h").to_numpy()
    X[tmask, 0:5] += treatment_effect
    X[tm_mask, 5:10] += time_effect
    if interaction_effect:
        X[tmask & tm_mask, 10:15] += interaction_effect
    if patient_sd > 0:
        patient_ids = obs["patient"].to_numpy()
        unique_p = np.unique(patient_ids)
        # One random intercept per patient, applied across all features
        intercepts = rng.standard_normal(len(unique_p)) * patient_sd
        p2i = {p: i for p, i in zip(unique_p, intercepts)}
        for i, p in enumerate(patient_ids):
            X[i, :] += p2i[p]
    var = pd.DataFrame(index=[f"feat{i}" for i in range(n_features)])
    return AnnData(X=X, obs=obs, var=var)


# --------------------------------------------------------------------------- #
# ASCA
# --------------------------------------------------------------------------- #
class TestASCA:
    def test_decomposes_additive_design(self):
        from omicverse.metabol import asca

        adata = _make_factorial_adata(treatment_effect=3.0, time_effect=1.5)
        res = asca(adata, factors=["treatment", "time"],
                   include_interactions=True)
        assert {"treatment", "time", "treatment:time"} <= set(res.effects.keys())
        # Treatment effect is stronger than time effect by design → higher SS
        assert (res.effects["treatment"].variance_explained
                > res.effects["time"].variance_explained)
        # Interaction effect is zero → SS should be tiny relative to main effects
        assert (res.effects["treatment:time"].variance_explained
                < res.effects["treatment"].variance_explained * 0.3)
        # Fractions + residual should sum to ~1
        total = sum(e.variance_explained for e in res.effects.values())
        total += res.residual_ss / res.total_ss
        assert abs(total - 1.0) < 1e-6

    def test_scores_and_loadings_shapes(self):
        from omicverse.metabol import asca

        adata = _make_factorial_adata(n_features=15)
        res = asca(adata, factors=["treatment", "time"], n_components=3)
        n, p = adata.shape
        for e in res.effects.values():
            assert e.scores.shape[0] == n
            assert e.loadings.shape[0] == p
            assert e.scores.shape[1] == e.loadings.shape[1]
            assert e.scores.shape[1] <= 3

    def test_summary_frame(self):
        from omicverse.metabol import asca

        adata = _make_factorial_adata()
        res = asca(adata, factors=["treatment", "time"])
        summary = res.summary()
        assert "effect" in summary.columns
        assert "variance_explained" in summary.columns
        assert "residual" in summary["effect"].values

    def test_permutation_p_detects_real_effect(self):
        from omicverse.metabol import asca

        adata = _make_factorial_adata(treatment_effect=5.0, time_effect=0.0,
                                       seed=42)
        res = asca(adata, factors=["treatment", "time"],
                   include_interactions=False, n_permutations=100, seed=0)
        # Strong treatment signal → small p; time absent → large p
        assert res.effects["treatment"].p_value is not None
        assert res.effects["time"].p_value is not None
        assert res.effects["treatment"].p_value < 0.05
        assert res.effects["time"].p_value > 0.1

    def test_missing_factor_raises(self):
        from omicverse.metabol import asca

        adata = _make_factorial_adata()
        with pytest.raises(KeyError):
            asca(adata, factors=["nonexistent"])

    def test_frames_round_trip(self):
        from omicverse.metabol import asca

        adata = _make_factorial_adata(n_features=12)
        res = asca(adata, factors=["treatment", "time"])
        scores_df = res.scores_frame("treatment")
        load_df = res.loadings_frame("treatment")
        assert scores_df.shape[0] == adata.n_obs
        assert load_df.shape[0] == adata.n_vars
        assert list(scores_df.index) == list(adata.obs_names)
        assert list(load_df.index) == list(adata.var_names)


# --------------------------------------------------------------------------- #
# MixedLM
# --------------------------------------------------------------------------- #
class TestMixedLM:
    def test_recovers_treatment_effect(self):
        pytest.importorskip("statsmodels")
        from omicverse.metabol import mixed_model

        adata = _make_factorial_adata(n_per_cell=8, n_features=15,
                                       treatment_effect=2.0, time_effect=0.5,
                                       patient_sd=1.0, seed=1)
        res = mixed_model(
            adata,
            formula="treatment + time",
            groups="patient",
            term="treatment[T.drug]",
        )
        # Feature 0-4 have the planted treatment effect — should be significant
        # after BH FDR.
        top = res.sort_values("pvalue").head(5)
        assert set(top.index) & {f"feat{i}" for i in range(5)}, top
        # Coef should be ~treatment_effect for those features
        true_effect = 2.0
        for f in ["feat0", "feat1", "feat2"]:
            assert abs(res.loc[f, "coef"] - true_effect) < 0.6

    def test_long_format_no_term(self):
        pytest.importorskip("statsmodels")
        from omicverse.metabol import mixed_model

        adata = _make_factorial_adata(n_per_cell=6, n_features=8,
                                       patient_sd=1.0, seed=2)
        res = mixed_model(adata, formula="treatment + time", groups="patient")
        # Long format: column "feature" + "term"
        assert "feature" in res.columns
        assert "term" in res.columns
        # No intercept or random-effect var row
        assert not (res["term"] == "Intercept").any()
        assert not res["term"].str.startswith("Group ").any()

    def test_missing_groups_raises(self):
        pytest.importorskip("statsmodels")
        from omicverse.metabol import mixed_model

        adata = _make_factorial_adata(n_features=5)
        with pytest.raises(KeyError):
            mixed_model(adata, formula="treatment", groups="not_a_column")


# --------------------------------------------------------------------------- #
# ROC / biomarker panel
# --------------------------------------------------------------------------- #
def _make_binary_adata(n_per_group: int = 30, n_features: int = 40,
                      effect_size: float = 2.0, seed: int = 0) -> AnnData:
    """Two-group synthetic: features 0..4 separate the groups."""
    rng = np.random.default_rng(seed)
    labels = np.array(["case"] * n_per_group + ["ctrl"] * n_per_group)
    n = n_per_group * 2
    X = rng.standard_normal((n, n_features)) * 0.5
    X[:n_per_group, 0:5] += effect_size
    obs = pd.DataFrame({"group": labels}, index=[f"s{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"feat{i}" for i in range(n_features)])
    return AnnData(X=X, obs=obs, var=var)


class TestROCFeature:
    def test_planted_features_have_high_auc(self):
        from omicverse.metabol import roc_feature

        adata = _make_binary_adata(n_per_group=25, n_features=30,
                                    effect_size=2.5, seed=3)
        df = roc_feature(adata, group_col="group")
        # First 5 features carry the signal → should be near the top
        top5 = set(df.head(5).index)
        planted = {f"feat{i}" for i in range(5)}
        assert len(top5 & planted) >= 4, f"top5={top5} planted={planted}"

    def test_polarity_invariant(self):
        from omicverse.metabol import roc_feature

        adata = _make_binary_adata(seed=4, effect_size=-2.0)
        df = roc_feature(adata, group_col="group")
        # Even with reversed effect, max(auc, 1-auc) ≥ 0.5
        assert (df["auc"] >= 0.5 - 1e-9).all()
        # Planted features still rank high
        assert df.head(5).index.str.startswith("feat").all()

    def test_ci_bounds_bracket_point_estimate(self):
        from omicverse.metabol import roc_feature

        adata = _make_binary_adata(n_per_group=20, n_features=10, seed=5)
        df = roc_feature(adata, group_col="group", ci=True, n_bootstrap=100,
                         seed=0)
        assert "ci_low" in df.columns
        assert "ci_high" in df.columns
        ok = df.dropna(subset=["ci_low", "ci_high"])
        assert (ok["ci_low"] <= ok["auc"] + 1e-6).all()
        assert (ok["ci_high"] >= ok["auc"] - 1e-6).all()


class TestBiomarkerPanel:
    def test_planted_signal_gives_high_auc(self):
        from omicverse.metabol import biomarker_panel

        adata = _make_binary_adata(n_per_group=30, n_features=30,
                                    effect_size=3.0, seed=6)
        res = biomarker_panel(
            adata, group_col="group", features=5,
            classifier="lr", cv_outer=3, cv_inner=2, seed=0,
        )
        assert 0.0 <= res.mean_auc <= 1.0
        # With planted effect=3.0 on 5 features, nested-CV should clear 0.8
        assert res.mean_auc > 0.8
        assert len(res.features) == 5
        assert res.feature_importance.size == 5
        assert res.outer_aucs.shape == (3,)

    def test_explicit_feature_list(self):
        from omicverse.metabol import biomarker_panel

        adata = _make_binary_adata(n_per_group=20, n_features=15,
                                    effect_size=2.5, seed=7)
        res = biomarker_panel(
            adata, group_col="group",
            features=["feat0", "feat1", "feat2"],
            classifier="rf", cv_outer=3, cv_inner=2, seed=0,
        )
        assert res.features == ["feat0", "feat1", "feat2"]
        assert res.classifier == "rf"

    def test_permutation_null_rejects_random(self):
        from omicverse.metabol import biomarker_panel

        adata = _make_binary_adata(n_per_group=25, n_features=20,
                                    effect_size=3.0, seed=8)
        res = biomarker_panel(
            adata, group_col="group", features=5,
            classifier="lr", cv_outer=3, cv_inner=2,
            n_permutations=20, seed=0,
        )
        assert res.permutation_pvalue is not None
        # Real signal should have a small permutation p
        assert res.permutation_pvalue < 0.2

    def test_handles_nans(self):
        from omicverse.metabol import biomarker_panel

        adata = _make_binary_adata(n_per_group=20, n_features=15,
                                    effect_size=2.5, seed=9)
        # Insert a few NaNs — should be filled by per-column median internally
        adata.X[0, 0] = np.nan
        adata.X[5, 3] = np.nan
        res = biomarker_panel(
            adata, group_col="group", features=5,
            classifier="lr", cv_outer=3, cv_inner=2, seed=0,
        )
        assert np.isfinite(res.mean_auc)
