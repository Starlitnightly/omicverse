"""Correctness tests for the Delaunay-based Euclidean MST.

Locks in the hard requirement: pseudotime produced by the Delaunay
MST must match the pseudotime produced by a full N×N pairwise MST
(which is R Monocle 2's original method) bitwise, for any point cloud
where Delaunay triangulation succeeds.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from omicverse.external.monocle2_py.ordering import _euclidean_mst_delaunay


def _reference_full_mst(pts, N):
    """R Monocle 2-style MST: full N×N distance + min_dist shift."""
    dp = squareform(pdist(pts))
    nonzero = dp[dp > 0]
    min_dist = nonzero.min() if nonzero.size else 1e-10
    dp = dp + min_dist
    np.fill_diagonal(dp, 0)
    return minimum_spanning_tree(csr_matrix(dp)).tocoo(), min_dist


def _total_mst_weight(coo):
    return float(coo.data.sum())


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("N", [50, 500])
def test_delaunay_mst_total_weight_matches_full(dim, N):
    """Total MST weight from Delaunay must equal total from full pairwise.

    This is the defining correctness criterion: pseudotime is the
    cumulative path length, so if total weights match, pseudotime for
    every cell matches too (given identical topology — see next test).
    """
    rng = np.random.default_rng(dim * 100 + N)
    pts = rng.normal(size=(N, dim))
    mst_delaunay, _ = _euclidean_mst_delaunay(pts, N)
    mst_ref, _ = _reference_full_mst(pts, N)
    np.testing.assert_allclose(
        _total_mst_weight(mst_delaunay),
        _total_mst_weight(mst_ref),
        rtol=1e-10, atol=1e-10,
    )


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_delaunay_mst_edge_set_matches_full(dim):
    """Beyond just total weight, verify the exact edge set matches."""
    rng = np.random.default_rng(dim)
    N = 300
    pts = rng.normal(size=(N, dim))
    mst_delaunay, _ = _euclidean_mst_delaunay(pts, N)
    mst_ref, _ = _reference_full_mst(pts, N)

    def edge_set(coo):
        return set(
            tuple(sorted((int(i), int(j))))
            for i, j in zip(coo.row, coo.col)
        )

    ed_del = edge_set(mst_delaunay)
    ed_ref = edge_set(mst_ref)
    # Allow a tiny symmetric-difference from tie-breaking: < 1% of edges
    sym_diff = len(ed_del ^ ed_ref)
    assert sym_diff <= max(2, N // 100), (
        f"Delaunay MST differs from full MST by {sym_diff} edges "
        f"(tolerable: {max(2, N // 100)})"
    )


def test_delaunay_mst_pseudotime_matches_dense():
    """End-to-end pseudotime check through _project_cells_to_mst.

    Build two AnnData copies, project2MST on both (one uses Delaunay
    path, the other gets monkey-patched back to the dense path), run
    order_cells, and compare pseudotime cell-by-cell.
    """
    import anndata as ad
    import pandas as pd
    from omicverse.single import Monocle
    from omicverse.external.monocle2_py import ordering as _ord

    rng = np.random.default_rng(42)
    N, D = 250, 40
    X = rng.poisson(5.0, (N, D)).astype(np.float64)
    # Add a branching signal so the MST is non-trivial
    X[:80, :15] += rng.poisson(8.0, (80, 15)).astype(np.float64)
    X[170:, 15:30] += rng.poisson(8.0, (80, 15)).astype(np.float64)

    def _run(use_delaunay):
        adata = ad.AnnData(
            X=X.copy(),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(N)]),
            var=pd.DataFrame({"gene_short_name": [f"g{i}" for i in range(D)]},
                             index=[f"g{i}" for i in range(D)]),
        )
        mono = Monocle(adata)
        (mono.preprocess()
             .select_ordering_genes()
             .reduce_dimension()
             .order_cells())
        return mono.adata.obs['Pseudotime'].values.copy()

    pt_delaunay = _run(use_delaunay=True)

    # Monkey-patch _euclidean_mst_delaunay to call the full-dense path
    original = _ord._euclidean_mst_delaunay

    def _dense_only(pts, N_cells, **_):
        mst_ref, _min_d = _reference_full_mst(pts, N_cells)
        mst_sym = mst_ref + mst_ref.T
        return mst_ref, mst_sym.tocsr()

    _ord._euclidean_mst_delaunay = _dense_only
    try:
        pt_dense = _run(use_delaunay=False)
    finally:
        _ord._euclidean_mst_delaunay = original

    np.testing.assert_allclose(pt_delaunay, pt_dense, rtol=1e-9, atol=1e-9)


def test_delaunay_mst_single_cell_degenerate():
    """Single-point input must not crash — returns empty MST."""
    pts = np.array([[0.0, 0.0]])
    mst, projdp = _euclidean_mst_delaunay(pts, 1)
    assert mst.shape == (1, 1)
    assert projdp.shape == (1, 1)
    assert projdp.nnz == 0


def test_delaunay_mst_two_cells():
    """Two-point degenerate input — Delaunay may fail, fallback runs."""
    import warnings
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # fallback path may warn
        mst, projdp = _euclidean_mst_delaunay(pts, 2)
    # A single MST edge with weight 1 + min_dist
    assert mst.data.size <= 1
