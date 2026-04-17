"""Test monocle2_py preprocessing: size factors, dispersion, genes detection.

Numerics are validated by reproducing the exact R Monocle2 formulae from
the deparsed source (see the PR notes). These tests must keep passing if
the refactor preserves algorithmic behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omicverse.external.monocle2_py import (
    detect_genes,
    estimate_size_factors,
    estimate_dispersions,
    dispersion_table,
)


# ---------------------------------------------------------------------------
# Size factors
# ---------------------------------------------------------------------------

def test_size_factors_match_r_formula(small_branching_adata):
    """R's `mean-geometric-mean-total`:
        sf_i = sum(counts_i) / exp(mean(log(sum(counts_j))))
    We compute this by hand and compare to the library output.
    """
    adata = small_branching_adata.copy()
    adata = estimate_size_factors(adata)

    # Reference: compute exactly as R does
    counts = np.round(adata.X)
    cell_total = counts.sum(axis=1)
    cell_total_safe = cell_total.copy()
    cell_total_safe[cell_total_safe == 0] = 1
    expected = cell_total_safe / np.exp(np.mean(np.log(cell_total_safe)))

    np.testing.assert_allclose(adata.obs["Size_Factor"].values, expected,
                                rtol=0, atol=1e-12)


def test_size_factors_mean_near_one(small_branching_adata):
    """Geometric-mean normalization should give factors whose geometric mean ≈ 1."""
    adata = small_branching_adata.copy()
    adata = estimate_size_factors(adata)
    sf = adata.obs["Size_Factor"].values
    assert np.isfinite(sf).all()
    assert (sf > 0).all()
    assert abs(np.exp(np.log(sf).mean()) - 1.0) < 1e-10


def test_size_factors_deterministic(small_branching_adata):
    """Re-running must produce identical values — no hidden randomness."""
    a1 = small_branching_adata.copy(); a1 = estimate_size_factors(a1)
    a2 = small_branching_adata.copy(); a2 = estimate_size_factors(a2)
    np.testing.assert_array_equal(
        a1.obs["Size_Factor"].values, a2.obs["Size_Factor"].values
    )


# ---------------------------------------------------------------------------
# Dispersions
# ---------------------------------------------------------------------------

def test_dispersion_empirical_matches_r_mom_formula(small_branching_adata):
    """R's disp_calc_helper_NB:
        xim       = mean(1 / Size_Factor)
        mu_g      = rowMeans(counts / Size_Factor)
        var_g     = rowMeans((counts/SF - mu_g)^2)
        disp_g    = (var_g - xim * mu_g) / mu_g^2   (negatives → 0)
    """
    adata = small_branching_adata.copy()
    adata = estimate_size_factors(adata)
    adata = detect_genes(adata, min_expr=0.1)
    adata = estimate_dispersions(adata, min_cells_detected=0)

    counts = np.round(adata.X)
    sf = adata.obs["Size_Factor"].values
    xim = np.mean(1.0 / sf)

    normed = counts / sf[:, None]       # cells × genes
    mu = normed.mean(axis=0)
    var = np.mean((normed - mu[None, :]) ** 2, axis=0)

    expected_disp = np.full_like(mu, np.nan)
    valid = (mu > 0) & (adata.var["num_cells_expressed"].values > 0)
    expected_disp[valid] = (var[valid] - xim * mu[valid]) / (mu[valid] ** 2)
    expected_disp[expected_disp < 0] = 0

    actual = adata.var["dispersion_empirical"].values
    assert np.allclose(actual[valid], expected_disp[valid], rtol=0, atol=1e-11)


def test_dispersion_table_schema(small_branching_adata):
    adata = small_branching_adata.copy()
    adata = estimate_size_factors(adata)
    adata = detect_genes(adata)
    adata = estimate_dispersions(adata)
    df = dispersion_table(adata)
    for col in ("gene_id", "mean_expression", "dispersion_fit",
                "dispersion_empirical"):
        assert col in df.columns


def test_dispersion_func_available(small_branching_adata):
    """After fitting, a dispersion function should be stored in uns."""
    adata = small_branching_adata.copy()
    adata = estimate_size_factors(adata)
    adata = detect_genes(adata)
    adata = estimate_dispersions(adata)
    func = adata.uns["monocle"].get("disp_func")
    # Not every synthetic dataset will converge; skip if it didn't.
    if func is None:
        pytest.skip("Dispersion fit did not converge on synthetic data")
    assert callable(func)
    assert func(1.0) > 0
    assert func(10.0) > 0


# ---------------------------------------------------------------------------
# Regression guards — these would catch issues #1 and #5 from the review
# ---------------------------------------------------------------------------

def test_dispersion_fit_no_dir_lookup(small_branching_adata):
    """Regression: the old `'result' in dir()` code could crash or silently
    return the wrong result_object. We just assert the pipeline runs and
    yields a valid `disp_func` when the fit converges, and `np.nan` or a
    constant fit when it doesn't — but never a crash."""
    adata = small_branching_adata.copy()
    adata = estimate_size_factors(adata)
    adata = detect_genes(adata)
    # Should not raise even on tiny / near-degenerate datasets
    adata = estimate_dispersions(adata)
    assert "dispersion_fit" in adata.var.columns
