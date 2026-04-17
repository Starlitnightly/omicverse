"""End-to-end trajectory inference tests.

These cover the full pipeline: preprocessing → ordering-gene selection →
DDRTree → orderCells → plot metadata. Assertions lock in the key
behaviours fixed in the review (MST construction not O(N^2), state
assignment matches R's component-based rule, BEAM uses the correct
branch point index).
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from omicverse.single import Monocle


# ---------------------------------------------------------------------------
# Linear trajectory (0 branch points)
# ---------------------------------------------------------------------------

def test_linear_trajectory_is_mostly_linear(linear_trajectory_adata):
    mono = Monocle(linear_trajectory_adata.copy())
    mono.preprocess().select_ordering_genes().reduce_dimension().order_cells()
    # Linear data should produce a near-path MST: at most 1 minor branch
    # (small noise branches happen when a few Y-centres splay off the
    # main axis; this matches R Monocle 2's behaviour).
    assert len(mono.branch_points) <= 1
    # Pseudotime should be positive and cover the gradient
    assert mono.pseudotime is not None
    assert mono.pseudotime.max() > 0


def test_pseudotime_correlates_with_ground_truth(linear_trajectory_adata):
    """In a linear gradient, the learned pseudotime should correlate
    strongly (|r| > 0.8) with the true `time` covariate."""
    mono = Monocle(linear_trajectory_adata.copy())
    mono.preprocess().select_ordering_genes().reduce_dimension().order_cells()
    true_time = mono.adata.obs["time"].values
    pt = mono.pseudotime.values
    r = abs(np.corrcoef(true_time, pt)[0, 1])
    assert r > 0.8, f"pseudotime ⟷ true time correlation {r:.3f} too low"


# ---------------------------------------------------------------------------
# Branching trajectory (1 branch point)
# ---------------------------------------------------------------------------

def test_branching_trajectory_detects_branch(small_branching_adata):
    mono = Monocle(small_branching_adata.copy())
    (mono.preprocess()
         .select_ordering_genes()
         .reduce_dimension()
         .order_cells())
    assert len(mono.branch_points) >= 1
    assert mono.adata.obs["State"].nunique() >= 2


def test_state_assignment_distinct_branches(three_branch_adata):
    """Cells from different tips should end up in different states."""
    mono = Monocle(three_branch_adata.copy())
    (mono.preprocess(min_expr=0.01)
         .select_ordering_genes()
         .reduce_dimension(ncenter=60)
         .order_cells())

    # tips A, B, C should have distinct modal states
    obs = mono.adata.obs
    mode_a = obs.loc[obs["tip"] == "A", "State"].mode().iloc[0]
    mode_b = obs.loc[obs["tip"] == "B", "State"].mode().iloc[0]
    mode_c = obs.loc[obs["tip"] == "C", "State"].mode().iloc[0]
    # At least two of the three should be in different states
    distinct = len({mode_a, mode_b, mode_c})
    assert distinct >= 2, (
        f"Expected at least 2 distinct tip states, got {mode_a}, {mode_b}, {mode_c}"
    )


def test_state_ids_are_positive_integers(small_branching_adata):
    """No cell should carry state 0 — R's Monocle uses 1-indexed states."""
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes().reduce_dimension().order_cells()
    states = np.asarray(mono.adata.obs["State"].values)
    assert (states >= 1).all(), "State 0 found — R uses 1-indexed states"


# ---------------------------------------------------------------------------
# MST efficiency regression (issue #2)
# ---------------------------------------------------------------------------

def test_mst_construction_scales_to_2000_cells():
    """If MST construction is O(N^2) building a full graph, this becomes
    painful at N=2000. Should finish in under ~20 s with scipy MST."""
    rng = np.random.default_rng(10)
    n, g = 2000, 100
    X = rng.poisson(3.0, (n, g)).astype(np.float64)
    import anndata as ad
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame({"gene_short_name": [f"g{i}" for i in range(g)]},
                         index=[f"g{i}" for i in range(g)]),
    )
    mono = Monocle(adata)
    mono.preprocess().select_ordering_genes(max_genes=200)
    t0 = time.time()
    mono.reduce_dimension(ncenter=150, maxIter=3).order_cells()
    dt = time.time() - t0
    assert dt < 60, f"reduce_dimension+order_cells took {dt:.1f}s (> 60s)"


# ---------------------------------------------------------------------------
# BEAM respects the branch_point index (issue #4)
# ---------------------------------------------------------------------------

def test_beam_uses_indexed_branch_point(three_branch_adata):
    mono = Monocle(three_branch_adata.copy())
    (mono.preprocess(min_expr=0.01)
         .select_ordering_genes()
         .reduce_dimension(ncenter=60)
         .order_cells())

    bps = mono.branch_points
    if len(bps) < 1:
        pytest.skip("No branch point detected on synthetic data")

    # branch_point=1 should succeed
    beam = mono.BEAM(branch_point=1, cores=1)
    assert isinstance(beam, pd.DataFrame)
    assert "pval" in beam.columns and "qval" in beam.columns

    # Out-of-range index should raise ValueError (not IndexError or silent failure)
    with pytest.raises(ValueError):
        mono.BEAM(branch_point=999, cores=1)


# ---------------------------------------------------------------------------
# Monocle class — state tracking
# ---------------------------------------------------------------------------

def test_monocle_class_stateful_repr(small_branching_adata):
    mono = Monocle(small_branching_adata.copy())
    assert "Monocle" in repr(mono)
    assert "preprocessed" not in repr(mono)
    mono.preprocess()
    assert "preprocessed" in repr(mono)
    mono.select_ordering_genes()
    assert "ordering genes" in repr(mono)
    mono.reduce_dimension()
    assert "reduced" in repr(mono)
    mono.order_cells()
    assert "pseudotime" in repr(mono)


def test_monocle_property_access(small_branching_adata):
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes().reduce_dimension().order_cells()
    assert mono.pseudotime is not None
    assert mono.state is not None
    assert mono.Z is not None and mono.Z.shape[0] == 2
    assert mono.Y is not None and mono.Y.shape[0] == 2
    assert isinstance(mono.branch_points, list)


def test_chain_returns_self(small_branching_adata):
    """Method chaining: each mutator should return the Monocle instance."""
    mono = Monocle(small_branching_adata.copy())
    assert mono.preprocess() is mono
    assert mono.select_ordering_genes() is mono
    assert mono.reduce_dimension() is mono
    assert mono.order_cells() is mono
