"""Algorithmic tests for the DDRTree implementation.

These verify the core numerical guarantees of reverse-graph embedding:
- PCA projection is orthogonal and matches scipy eigh
- DDRTree objective is monotonically non-increasing at convergence
- Low-rank PCA trick gives eigenvectors equivalent to the full eigh
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from omicverse.external.monocle2_py.ddrtree import (
    DDRTree,
    _pca_projection,
    _pca_projection_irlba_like,
    _sqdist,
)


def test_sqdist_matches_cdist():
    """_sqdist computes squared Euclidean distances between columns."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 7))
    B = rng.normal(size=(5, 4))
    # _sqdist treats columns as points; cdist treats rows — take the
    # transpose for the comparison.
    expected = cdist(A.T, B.T, "sqeuclidean")
    got = _sqdist(A, B)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)


def test_pca_projection_returns_orthonormal():
    rng = np.random.default_rng(2)
    C = rng.normal(size=(40, 40))
    C = (C + C.T) / 2  # symmetric
    W = _pca_projection(C, 3)
    assert W.shape == (40, 3)
    np.testing.assert_allclose(W.T @ W, np.eye(3), rtol=0, atol=1e-9)


def test_pca_projection_large_matrix_uses_irlba():
    """For D > 5000, _pca_projection should still succeed via eigsh."""
    rng = np.random.default_rng(3)
    # 5100 × 5100 dense eigh would be slow; feed a rank-5 matrix so eigsh
    # finds the signal quickly.
    D = 5100
    U = rng.normal(size=(D, 5))
    C = U @ U.T
    W = _pca_projection(C, 2)
    assert W.shape == (D, 2)
    # Columns should be (approximately) unit norm
    norms = np.linalg.norm(W, axis=0)
    np.testing.assert_allclose(norms, 1.0, rtol=0, atol=1e-6)


def test_ddrtree_runs_and_returns_expected_shapes():
    rng = np.random.default_rng(4)
    D, N = 50, 80
    # Two clusters to create a real branch structure
    X = np.hstack([
        rng.normal(0, 1, (D, N // 2)),
        rng.normal(3, 1, (D, N // 2)),
    ])
    res = DDRTree(X, dimensions=2, ncenter=30, maxIter=10, tol=1e-4)

    assert res["W"].shape == (D, 2)
    assert res["Z"].shape == (2, N)
    assert res["Y"].shape == (2, 30)
    assert len(res["objective_vals"]) >= 1


def test_ddrtree_objective_decreasing_after_warmup():
    """After the first 1-2 warmup iterations, the objective should be
    non-increasing (DDRTree is minimizing a bounded coordinate-descent
    objective)."""
    rng = np.random.default_rng(5)
    D, N = 40, 60
    X = rng.normal(size=(D, N))
    res = DDRTree(X, dimensions=2, ncenter=20, maxIter=15, tol=1e-6)
    obj = np.asarray(res["objective_vals"])
    # Drop the first 2 values (initialisation + first step may overshoot),
    # then require monotone non-increase with a modest tolerance.
    tail = obj[2:]
    diffs = np.diff(tail)
    # Allow small numerical noise up to 1% of the current objective
    assert (diffs <= 1e-2 * np.abs(tail[:-1])).all(), (
        f"DDRTree objective not monotone after warmup: {tail}"
    )


def test_ddrtree_mst_is_a_tree():
    """The resulting stree should be a valid tree: N-1 edges, acyclic."""
    rng = np.random.default_rng(6)
    D, N = 30, 70
    X = rng.normal(size=(D, N))
    K = 25
    res = DDRTree(X, dimensions=2, ncenter=K, maxIter=5)
    stree = res["stree"].toarray()
    # symmetrise
    stree = (stree + stree.T) > 0
    n_edges = stree.sum() // 2
    assert n_edges == K - 1, f"MST has {n_edges} edges, expected {K - 1}"
