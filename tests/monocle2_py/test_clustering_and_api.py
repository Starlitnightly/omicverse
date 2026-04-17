"""Tests for clustering, plotting, and module-level API surface."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from omicverse.single import Monocle
from omicverse.external import monocle2_py as m2


# ---------------------------------------------------------------------------
# __all__ contract (issue #7 from the review)
# ---------------------------------------------------------------------------

def test_all_export_contains_everything_imported():
    """Every top-level name we import in __init__.py should be in __all__."""
    public_names = [n for n in dir(m2) if not n.startswith("_")
                    and n not in {"core", "ddrtree", "dimension_reduction",
                                   "ordering", "differential", "clustering",
                                   "plotting", "utils"}]
    missing = set(public_names) - set(m2.__all__)
    assert not missing, (
        f"Names exported but not in __all__: {sorted(missing)}"
    )


def test_all_exported_names_are_actually_importable():
    """Every name listed in __all__ should resolve to something."""
    for name in m2.__all__:
        assert hasattr(m2, name), f"__all__ lists {name!r} but it's missing"


# ---------------------------------------------------------------------------
# uns['monocle'] state-sharing regression (issue #5)
# ---------------------------------------------------------------------------

def test_densitypeak_writes_to_actual_uns(small_branching_adata):
    """Regression: the old code kept a stale local copy of uns['monocle']
    that disconnected from adata.uns once a new key was added. Ensure
    rho/delta land in adata.uns['monocle'] and can be re-read."""
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes()
    mono.reduce_dimension(reduction_method="tSNE", num_dim=5,
                           perplexity=8)
    mono.cluster_cells(method="densityPeak", num_clusters=3)

    assert "rho" in mono.adata.uns["monocle"]
    assert "delta" in mono.adata.uns["monocle"]
    assert len(mono.adata.uns["monocle"]["rho"]) == mono.adata.n_obs


# ---------------------------------------------------------------------------
# Plotting smoke tests — the dendrogram recursion fix (previous review round)
# ---------------------------------------------------------------------------

def test_plot_cell_trajectory_smoke(small_branching_adata):
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes().reduce_dimension().order_cells()
    fig, ax = mono.plot_trajectory(color_by="State")
    assert fig is not None
    plt.close(fig)


def test_plot_pseudotime_heatmap_many_genes():
    """Large heatmaps must not hit Python recursion (leaves_list fix)."""
    rng = np.random.default_rng(22)
    # Build an adata that will get many ordering genes
    n, g = 80, 500
    X = rng.poisson(5.0, (n, g)).astype(np.float64)
    X[:40, :250] += rng.poisson(8.0, (40, 250)).astype(np.float64)
    import anndata as ad
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame({"gene_short_name": [f"g{i}" for i in range(g)]},
                         index=[f"g{i}" for i in range(g)]),
    )
    mono = Monocle(adata)
    mono.preprocess().select_ordering_genes().reduce_dimension().order_cells()
    all_genes = mono.adata.var_names.tolist()
    # Big heatmap — used to blow up with dendrogram recursion
    fig = mono.plot_pseudotime_heatmap(genes=all_genes, num_clusters=4)
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# Misc: estimate_t + relative2abs now exported (issue #7)
# ---------------------------------------------------------------------------

def test_relative2abs_roundtrip_preserves_shape(small_branching_adata):
    abs_adata = m2.relative2abs(small_branching_adata.copy(),
                                 method="num_genes")
    assert abs_adata.shape == small_branching_adata.shape
    # Values are non-negative integers
    vals = np.asarray(abs_adata.X)
    assert (vals >= 0).all()
    assert np.allclose(vals, np.round(vals))


def test_estimate_t_returns_per_cell_values(small_branching_adata):
    t = m2.estimate_t(small_branching_adata.X)
    assert t.shape == (small_branching_adata.n_obs,)
    assert (t > 0).all()
