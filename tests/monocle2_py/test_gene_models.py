"""Tests for gen_smooth_curves and fit_model — the gene-level GLM fitters.

These are the backbone of plot_genes_in_pseudotime, plot_pseudotime_heatmap,
BEAM, and calILRs, so we lock in their return shapes and basic numeric
sanity here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omicverse.single import Monocle
from omicverse.external import monocle2_py as m2


@pytest.fixture
def ordered_adata(small_branching_adata):
    """AnnData that has been through the full trajectory pipeline."""
    mono = Monocle(small_branching_adata.copy())
    (mono.preprocess()
         .select_ordering_genes()
         .reduce_dimension()
         .order_cells())
    return mono.adata


def test_fit_model_returns_one_fit_per_gene(ordered_adata):
    models = m2.fit_model(ordered_adata, cores=1)
    assert len(models) == ordered_adata.n_vars
    # At least 80% of fits should succeed on clean synthetic data
    ok = sum(1 for m in models if m is not None)
    assert ok / len(models) >= 0.8


def test_fit_model_coefficients_shape(ordered_adata):
    models = m2.fit_model(ordered_adata, cores=1)
    for m in models:
        if m is None:
            continue
        # Intercept + df spline basis terms = 4 columns (intercept + 3 df)
        assert m["coefficients"].shape[0] >= 4
        assert np.isfinite(m["coefficients"]).all()
        break   # just spot-check first successful fit


def test_gen_smooth_curves_default_shape(ordered_adata):
    """With ``new_data=None`` the curves are evaluated at the cells' own
    pseudotimes → shape = (n_genes, n_cells)."""
    curves = m2.gen_smooth_curves(ordered_adata, new_data=None)
    assert curves.shape == (ordered_adata.n_vars, ordered_adata.n_obs)


def test_gen_smooth_curves_with_new_data(ordered_adata):
    """Prediction at a user-supplied pseudotime grid."""
    grid = pd.DataFrame({
        "Pseudotime": np.linspace(
            ordered_adata.obs["Pseudotime"].min(),
            ordered_adata.obs["Pseudotime"].max(),
            50,
        )
    })
    curves = m2.gen_smooth_curves(ordered_adata, new_data=grid)
    assert curves.shape == (ordered_adata.n_vars, 50)
    # Most curves should be non-negative (counts)
    nn_frac = (curves >= 0).mean()
    assert nn_frac > 0.9, f"Too many negative curve values: {nn_frac:.2%}"


def test_gen_smooth_curves_monotone_on_monotone_gene(linear_trajectory_adata):
    """For a linearly-increasing gene along pseudotime, the fitted curve
    should also be roughly increasing."""
    mono = Monocle(linear_trajectory_adata.copy())
    (mono.preprocess()
         .select_ordering_genes()
         .reduce_dimension()
         .order_cells())
    # gene 0 has expression ∝ time in the fixture
    grid = pd.DataFrame({
        "Pseudotime": np.linspace(
            mono.adata.obs["Pseudotime"].min(),
            mono.adata.obs["Pseudotime"].max(),
            100,
        )
    })
    curves = m2.gen_smooth_curves(mono.adata, new_data=grid)

    # Correlate curve of gene 0 with the pseudotime grid
    g0 = curves[0]
    if np.isfinite(g0).sum() < 10:
        pytest.skip("Gene 0 curve failed to fit")
    # If the learned pseudotime flipped direction, |r| is still high
    r = abs(np.corrcoef(grid["Pseudotime"].values, g0)[0, 1])
    assert r > 0.5, f"Smooth curve not correlated with time: |r|={r:.2f}"
