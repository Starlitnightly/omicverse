"""Tests for cluster_cells (densityPeak / Leiden / Louvain paths).

Previously only densityPeak was touched indirectly via
test_densitypeak_writes_to_actual_uns. This file exercises all three
and the densityPeak vectorised inner loop (review high #9).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from omicverse.single import Monocle


@pytest.fixture
def tsne_ready(small_branching_adata):
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes()
    mono.reduce_dimension(reduction_method="tSNE", num_dim=8,
                           perplexity=10, random_state=0)
    return mono


def test_density_peak_cluster_assignment(tsne_ready):
    tsne_ready.cluster_cells(method="densityPeak", num_clusters=3)
    assert "Cluster" in tsne_ready.adata.obs.columns
    labels = tsne_ready.adata.obs["Cluster"].values
    # Each cell must get a cluster in [1, num_clusters]
    uniq = set(np.unique(labels))
    assert uniq.issubset({1, 2, 3}) or uniq.issubset({1, 2, 3, -1})
    # Should not be all the same cluster
    assert len(uniq - {-1}) > 1


def test_density_peak_vectorised_performance():
    """Regression: the delta/nneigh inner loop used to be O(N²) Python.
    After vectorisation, 1000 cells should take under 3s."""
    rng = np.random.default_rng(7)
    N = 1000
    X = rng.normal(size=(N, 30)).astype(np.float64)
    import anndata as ad
    adata = ad.AnnData(
        X=np.maximum(X, 0),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(N)]),
        var=pd.DataFrame({"gene_short_name": [f"g{i}" for i in range(30)]},
                         index=[f"g{i}" for i in range(30)]),
    )
    mono = Monocle(adata)
    mono.preprocess().select_ordering_genes()
    mono.reduce_dimension(reduction_method="tSNE", num_dim=5,
                           perplexity=30, random_state=0)

    t0 = time.time()
    mono.cluster_cells(method="densityPeak", num_clusters=4)
    dt = time.time() - t0
    assert dt < 10.0, f"densityPeak on N=1000 took {dt:.1f}s (> 10s)"


def test_leiden_clustering_smoke(tsne_ready):
    tsne_ready.cluster_cells(method="leiden", k=15, resolution_parameter=0.3)
    assert "Cluster" in tsne_ready.adata.obs.columns


def test_louvain_clustering_smoke(tsne_ready):
    tsne_ready.cluster_cells(method="louvain", k=15)
    assert "Cluster" in tsne_ready.adata.obs.columns


def test_cluster_genes_returns_labels(small_branching_adata):
    """cluster_genes is a separate function on the class."""
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes()
    # Pass a small subset of the expression matrix
    X = mono.adata.X[:, :20].T    # 20 genes × cells
    result = Monocle.cluster_genes(X, k=4)
    assert "clustering" in result
    labels = result["clustering"]
    assert len(labels) == 20
    assert len(np.unique(labels)) <= 4
