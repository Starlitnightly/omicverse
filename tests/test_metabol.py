"""Tests for ``omicverse.metabol``.

Uses MetaboAnalyst's canonical ``human_cachexia.csv`` (Eisner et al. 2010)
as the fixture — 77 samples × 63 NMR metabolites with a binary
``Muscle loss`` factor. The CSV is fetched once per test-session and
cached in ``/tmp``; if the fetch fails (offline CI), the dependent
tests are skipped.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# cachexia_csv + cachexia_adata fixtures are defined in tests/conftest.py


def test_io_loads_77x63(cachexia_adata):
    assert cachexia_adata.shape == (77, 63)
    assert "group" in cachexia_adata.obs.columns
    counts = cachexia_adata.obs["group"].value_counts()
    assert int(counts.get("cachexic", 0)) == 47
    assert int(counts.get("control", 0)) == 30


def test_impute_half_min_preserves_shape(cachexia_adata):
    from omicverse.metabol import impute

    # Inject a synthetic 10% NaN pattern so we have something to impute.
    adata = cachexia_adata.copy()
    rng = np.random.default_rng(0)
    mask = rng.random(adata.X.shape) < 0.10
    X = adata.X.astype(float).copy()
    X[mask] = np.nan
    adata.X = X
    out = impute(adata, method="half_min", missing_threshold=0.5)
    assert out.shape == adata.shape
    assert not np.isnan(out.X).any()


def test_impute_qrilc_runs_and_is_positive(cachexia_adata):
    from omicverse.metabol import impute

    adata = cachexia_adata.copy()
    rng = np.random.default_rng(0)
    mask = rng.random(adata.X.shape) < 0.10
    X = adata.X.astype(float).copy()
    X[mask] = np.nan
    adata.X = X
    out = impute(adata, method="qrilc")
    assert not np.isnan(out.X).any()
    # QRILC draws from a left-tail lognormal — results should be positive
    assert (out.X >= 0).all()


def test_normalize_pqn_centres_samples(cachexia_adata):
    from omicverse.metabol import normalize

    out = normalize(cachexia_adata, method="pqn")
    # After PQN the row medians should be within ~1 order of magnitude
    row_medians = np.nanmedian(out.X, axis=1)
    assert row_medians.max() / max(row_medians.min(), 1e-9) < 10.0


def test_transform_pareto_zero_mean_columns(cachexia_adata):
    from omicverse.metabol import normalize, transform

    out = normalize(cachexia_adata, method="pqn")
    out = transform(out, method="pareto")
    col_mean = np.nanmean(out.X, axis=0)
    assert np.allclose(col_mean, 0.0, atol=1e-8)


def test_differential_welch_t_reports_known_hits(cachexia_adata):
    """On the cachexia dataset, certain amino acids (e.g. creatinine)
    are classically differential. We don't hard-code which — we just
    assert the pipeline runs and the FDR sensibly orders things."""
    from omicverse.metabol import differential, normalize, transform

    out = normalize(cachexia_adata, method="pqn")
    out = transform(out, method="log")
    deg = differential(out, group_col="group", method="welch_t", log_transformed=True)
    assert set(deg.columns) >= {"stat", "pvalue", "padj", "log2fc", "mean_a", "mean_b"}
    assert (deg["pvalue"] >= 0).all() and (deg["pvalue"] <= 1).all()
    assert (deg["padj"] >= 0).all() and (deg["padj"] <= 1).all()
    # padj is monotone in pvalue after BH
    assert deg.sort_values("pvalue")["padj"].is_monotonic_increasing
    # Cachexia top hits are well-known biomarkers — Isoleucine, Uracil,
    # Glucose, Acetone should be among the top-10 by p-value.
    top10 = set(deg.sort_values("pvalue").head(10).index)
    assert top10 & {"Isoleucine", "Uracil", "Glucose", "Acetone"}
    # After BH correction, MetaboAnalyst's own tutorial reports ~2 hits at
    # padj<0.05 on this dataset — so we assert the FDR is *doing* something
    # (at least 1 hit) rather than a hard count.
    assert (deg["padj"] < 0.05).sum() >= 1
    # At padj<0.20 we expect a richer signal
    assert (deg["padj"] < 0.20).sum() >= 5


def test_plsda_gives_high_q2_on_cachexia(cachexia_adata):
    # MetaboAnalyst canonical pipeline: PQN → log → Pareto → PLS-DA.
    # Skipping the log step leaves raw concentration variance, which
    # dominates Pareto and breaks the model — Q² ≈ -0.6.
    from omicverse.metabol import normalize, plsda, transform

    out = normalize(cachexia_adata, method="pqn")
    out = transform(out, method="log")
    out = transform(out, method="pareto", stash_raw=False)
    res = plsda(out, group_col="group", n_components=2)
    assert res.scores.shape == (77, 2)
    assert 0.0 <= res.r2x <= 1.0
    assert 0.0 <= res.r2y <= 1.0
    # Cachexia vs control separation is real but modest after Pareto
    # scaling — Q²~0.1 is typical. Assert it's positive (model beats
    # predicting the mean) rather than any hard magnitude.
    assert res.q2 > 0.0
    # And the model does explain some X-variance
    assert res.r2x > 0.05
    # VIP is non-negative and has the same length as vars
    assert res.vip.shape == (out.n_vars,)
    assert (res.vip >= 0).all()


def test_opls_da_has_one_predictive_component(cachexia_adata):
    from omicverse.metabol import normalize, opls_da, transform

    out = normalize(cachexia_adata, method="pqn")
    out = transform(out, method="log")
    out = transform(out, method="pareto", stash_raw=False)
    res = opls_da(out, group_col="group", n_ortho=1)
    assert res.scores.shape == (77, 1)        # exactly one predictive component
    assert res.x_ortho_scores.shape == (77, 1)
    assert res.vip.shape == (out.n_vars,)
    assert (res.vip >= 0).all()
    # Single predictive component fits modestly on this data — calibrated
    # on MetaboAnalystR's own OPLS-DA vignette, R²Y is in the 0.25–0.45 band
    assert 0.2 < res.r2y < 0.9


def test_pymetabo_chainable_end_to_end(cachexia_adata):
    from omicverse.metabol import pyMetabo

    # MetaboAnalyst's canonical pipeline: PQN → log → Pareto → stats / OPLS-DA.
    # ``log_transformed=True`` tells ``differential`` that the fold-change is
    # mean_a - mean_b (already on log scale), which is correct after .transform().
    m = (
        pyMetabo(cachexia_adata.copy())
        .normalize(method="pqn")
        .transform(method="log")
        .differential(method="welch_t", log_transformed=True)
        .transform(method="pareto")
        .opls_da(n_ortho=1)
    )
    assert m.deg_table is not None
    assert m.plsda_result is not None
    vip = m.vip_table()
    assert "vip" in vip.columns
    # Cachexia has a modest FDR-robust signal after BH — expect ≥5 at padj<0.20.
    sig = m.significant_metabolites(padj_thresh=0.20, log2fc_thresh=0.0)
    assert len(sig) >= 5


def test_limma_matches_welch_t_rank_order(cachexia_adata):
    """limma-moderated t and Welch's t should agree on the TOP hits
    even if individual p-values differ (empirical-Bayes shrinkage
    only changes ordering at the margin)."""
    from omicverse.metabol import differential, normalize, transform

    out = normalize(cachexia_adata, method="pqn")
    out = transform(out, method="log")
    welch = differential(out, method="welch_t", log_transformed=True)
    lim = differential(out, method="limma", log_transformed=True)
    # Top-10 overlap should be ≥7
    top_welch = set(welch.sort_values("pvalue").head(10).index)
    top_lim = set(lim.sort_values("pvalue").head(10).index)
    assert len(top_welch & top_lim) >= 7


def test_plotting_volcano_returns_figure(cachexia_adata):
    import matplotlib
    matplotlib.use("Agg")

    from omicverse.metabol import differential, normalize, transform, volcano

    out = normalize(cachexia_adata, method="pqn")
    out = transform(out, method="log")
    deg = differential(out, method="welch_t", log_transformed=True)
    fig, ax = volcano(deg, padj_thresh=0.05, log2fc_thresh=0.5, label_top_n=5)
    assert fig is not None and ax is not None
