"""Regression guard for the project2MST pseudotime-scale bug.

An earlier optimisation replaced the full N×N pairwise distance
matrix in `_project_cells_to_mst` with a kNN-sparse graph and used
`sparse.minimum(sparse.T)` to symmetrise. That silently produced
zero-weight MST edges (missing entries in a sparse matrix are 0,
which dominates the pointwise minimum), collapsing pseudotime to
near zero (1.89 vs R's 29.89 on the pancreas dataset).

This test locks in the fix: pseudotime on a branching trajectory must
have a non-trivial range that reflects the actual MST edge lengths.
"""
from __future__ import annotations

import numpy as np
import pytest

from omicverse.single import Monocle


def test_pseudotime_range_non_collapsed(small_branching_adata):
    """Pseudotime max should be at least 0.5 on the branching fixture.
    Before the fix, max was consistently under 0.1."""
    mono = Monocle(small_branching_adata.copy())
    (mono.preprocess()
         .select_ordering_genes()
         .reduce_dimension()
         .order_cells())
    pt_max = float(mono.pseudotime.max())
    assert pt_max >= 0.5, (
        f"Pseudotime collapsed to {pt_max:.3f}; "
        "project2MST likely produced zero-weight MST edges"
    )


def test_pseudotime_reflects_edge_lengths(three_branch_adata):
    """Pseudotime spread should be of the same order of magnitude as
    the diameter of the DDRTree point cloud."""
    mono = Monocle(three_branch_adata.copy())
    (mono.preprocess(min_expr=0.01)
         .select_ordering_genes()
         .reduce_dimension(ncenter=60)
         .order_cells())

    Z = mono.Z                                        # (dim, N)
    cloud_diameter = np.linalg.norm(
        Z.max(axis=1) - Z.min(axis=1)
    )
    pt_span = float(mono.pseudotime.max() - mono.pseudotime.min())
    # Pseudotime should be on the same order as the point-cloud
    # diameter: not more than 3× larger, not less than 10× smaller.
    assert pt_span >= cloud_diameter / 10, (
        f"Pseudotime span {pt_span:.3f} << cloud diameter "
        f"{cloud_diameter:.3f}; MST edge weights likely collapsed"
    )
    assert pt_span <= cloud_diameter * 3, (
        f"Pseudotime span {pt_span:.3f} >> cloud diameter "
        f"{cloud_diameter:.3f} — MST edges suspiciously inflated"
    )
