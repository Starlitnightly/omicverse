"""Shared fixtures for monocle2_py tests.

We build small synthetic branching datasets with known structure so
that tests can assert exact algorithmic properties (preprocessing
numerics, DDRTree topology, state assignment) without requiring the
external R Monocle2 runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad


@pytest.fixture(scope="session")
def small_branching_adata():
    """Small synthetic 150 cells × 80 genes with a clear Y-shaped trajectory.

    Structure:
      * cells 0-49    : branch A (genes 0-29 upregulated)
      * cells 50-99   : trunk (modest expression of both)
      * cells 100-149 : branch B (genes 30-59 upregulated)
    """
    rng = np.random.default_rng(42)
    n_cells, n_genes = 150, 80
    X = rng.poisson(5.0, (n_cells, n_genes)).astype(np.float64)

    # Branch A markers
    X[:50, :30] += rng.poisson(10.0, (50, 30)).astype(np.float64)
    # Branch B markers
    X[100:, 30:60] += rng.poisson(10.0, (50, 30)).astype(np.float64)

    obs = pd.DataFrame(
        {"group": ["A"] * 50 + ["Trunk"] * 50 + ["B"] * 50},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_short_name": [f"g{i}" for i in range(n_genes)]},
        index=[f"g{i}" for i in range(n_genes)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture(scope="session")
def linear_trajectory_adata():
    """80 cells, 40 genes, linear (unbranched) gradient.

    Expression of gene 0..19 monotonically increases with cell index;
    gene 20..39 is random noise. DDRTree should find no branch points.
    """
    rng = np.random.default_rng(7)
    n, g = 80, 40
    t = np.linspace(0, 1, n)
    X = rng.poisson(3.0, (n, g)).astype(np.float64)
    X[:, :20] += (20.0 * t)[:, None]
    X[:, :20] = np.maximum(X[:, :20], 0)
    obs = pd.DataFrame({"time": t}, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(
        {"gene_short_name": [f"g{i}" for i in range(g)]},
        index=[f"g{i}" for i in range(g)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture(scope="session")
def three_branch_adata():
    """3-way branching (one branch point, three tips) — 180 cells × 60 genes."""
    rng = np.random.default_rng(13)
    n, g = 180, 60
    X = rng.poisson(4.0, (n, g)).astype(np.float64)
    # three tips of 40 cells each; tip markers are genes [0..9], [10..19], [20..29]
    X[:40, :10] += rng.poisson(12.0, (40, 10)).astype(np.float64)
    X[40:80, 10:20] += rng.poisson(12.0, (40, 10)).astype(np.float64)
    X[80:120, 20:30] += rng.poisson(12.0, (40, 10)).astype(np.float64)
    # trunk cells 120-179 share modest expression
    X[120:, :30] += rng.poisson(3.0, (60, 30)).astype(np.float64)

    obs = pd.DataFrame(
        {"tip": (["A"] * 40 + ["B"] * 40 + ["C"] * 40 + ["Trunk"] * 60)},
        index=[f"c{i}" for i in range(n)],
    )
    var = pd.DataFrame(
        {"gene_short_name": [f"g{i}" for i in range(g)]},
        index=[f"g{i}" for i in range(g)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)
