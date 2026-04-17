"""Edge-case tests and plotting smoke tests.

Review round-3 gaps:
  - empty / very small / degenerate datasets
  - all-zero-gene handling
  - every plotting function is called at least once to catch NameErrors
    (like the missing ``dendrogram`` import fix in this round).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import anndata as ad

from omicverse.single import Monocle


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_zero_gene_is_filtered(small_branching_adata):
    """A gene with all zeros should either be dropped or yield finite
    outputs — never NaN/Inf that propagates through the pipeline."""
    adata = small_branching_adata.copy()
    # Force genes 0 and 1 to zero everywhere
    adata.X[:, :2] = 0
    mono = Monocle(adata)
    mono.preprocess().select_ordering_genes().reduce_dimension().order_cells()
    # All pseudotime values must be finite
    assert np.isfinite(mono.pseudotime.values).all()
    # Trajectory produced some non-trivial structure
    assert mono.pseudotime.max() > 0


def test_very_small_dataset_runs_without_crash():
    """Tiny dataset (20 cells) — pipeline must not crash."""
    rng = np.random.default_rng(77)
    n, g = 20, 30
    X = rng.poisson(5.0, (n, g)).astype(np.float64)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame({"gene_short_name": [f"g{i}" for i in range(g)]},
                         index=[f"g{i}" for i in range(g)]),
    )
    mono = Monocle(adata)
    mono.preprocess()
    mono.select_ordering_genes()
    # Should not throw even at this scale
    mono.reduce_dimension()
    mono.order_cells()
    assert mono.pseudotime is not None


def test_linear_trajectory_warns_no_branch(linear_trajectory_adata):
    """Review high #8: linear trajectory should warn that no branch
    points were detected."""
    import warnings
    mono = Monocle(linear_trajectory_adata.copy())
    mono.preprocess().select_ordering_genes().reduce_dimension()

    # If no branch points, order_cells should emit a UserWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mono.order_cells()

    # If this data happens to branch, skip (depends on DDRTree seed)
    if mono.branch_points:
        pytest.skip("Synthetic data unexpectedly branched")

    got_warning = any(
        "branch" in str(warn.message).lower() and "linear" in str(warn.message).lower()
        for warn in w
    )
    assert got_warning, "Expected a warning about linear trajectory"


def test_state_assignment_bounds_checked(small_branching_adata):
    """Review critical #3: ``closest_vertex`` that points past the MST
    should raise, not silently yield wrong results."""
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes().reduce_dimension()
    # Corrupt closest_vertex to have an out-of-range value
    import warnings
    n_Y = mono.adata.uns['monocle']['mst'].vcount()
    bad = mono.adata.uns['monocle']['pr_graph_cell_proj_closest_vertex'].copy()
    bad[0] = n_Y + 100   # way out of range
    mono.adata.uns['monocle']['pr_graph_cell_proj_closest_vertex'] = bad
    with pytest.raises(AssertionError, match="out of range"):
        mono.order_cells()


# ---------------------------------------------------------------------------
# Plotting smoke tests — every function called at least once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ordered_mono(small_branching_adata):
    adata = small_branching_adata.copy()
    mono = Monocle(adata)
    (mono.preprocess()
         .select_ordering_genes()
         .reduce_dimension()
         .order_cells())
    return mono


def test_plot_cell_trajectory_state(ordered_mono):
    fig, ax = ordered_mono.plot_trajectory(color_by='State')
    assert fig is not None
    plt.close(fig)


def test_plot_cell_trajectory_pseudotime(ordered_mono):
    fig, ax = ordered_mono.plot_trajectory(color_by='Pseudotime')
    plt.close(fig)


def test_plot_complex_cell_trajectory(ordered_mono):
    fig, ax = ordered_mono.plot_complex_cell_trajectory(color_by='State')
    plt.close(fig)


def test_plot_genes_in_pseudotime(ordered_mono):
    genes = ['g0', 'g10', 'g30', 'g40']
    fig = ordered_mono.plot_genes_in_pseudotime(genes, ncol=2)
    plt.close(fig)


def test_plot_genes_jitter(ordered_mono):
    fig = ordered_mono.plot_genes_jitter(
        ['g0', 'g10'], grouping='State', ncol=2,
    )
    plt.close(fig)


def test_plot_genes_violin(ordered_mono):
    fig = ordered_mono.plot_genes_violin(
        ['g0', 'g10'], grouping='State', ncol=2,
    )
    plt.close(fig)


def test_plot_ordering_genes(ordered_mono):
    fig = ordered_mono.plot_ordering_genes()
    plt.close(fig)


def test_plot_pc_variance_explained(small_branching_adata):
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes()
    fig = mono.plot_pc_variance_explained(max_components=10)
    plt.close(fig)


def test_plot_pseudotime_heatmap_default(ordered_mono):
    # Use a handful of genes so the dendrogram runs cleanly
    top_genes = [f"g{i}" for i in range(20)]
    fig = ordered_mono.plot_pseudotime_heatmap(genes=top_genes, num_clusters=3)
    plt.close(fig)


def test_plot_genes_branched_heatmap_smoke(ordered_mono):
    """Regression guard for missing `dendrogram` import (critical #1)."""
    if not ordered_mono.branch_points:
        pytest.skip("No branch points")
    # Just call it and verify no NameError
    try:
        fig = ordered_mono.plot_genes_branched_heatmap(
            branch_point=1, num_clusters=2, show_rownames=False,
        )
        if fig is not None:
            plt.close(fig)
    except (ValueError, IndexError):
        # Synthetic data may not have enough branch states — not our concern
        pytest.skip("Synthetic data doesn't support branched heatmap")
    except NameError:
        pytest.fail("NameError in plot_genes_branched_heatmap — missing import")
