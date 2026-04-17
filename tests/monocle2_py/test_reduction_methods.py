"""Smoke tests for the non-DDRTree reduction methods: tSNE and ICA.

These paths had no test coverage in the original PR. We assert that the
pipeline runs end-to-end and produces sensible outputs in ``obsm`` /
``uns['monocle']``.
"""
from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from omicverse.single import Monocle


def test_tsne_reduction_smoke(small_branching_adata):
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes()
    mono.reduce_dimension(
        max_components=2,
        reduction_method="tSNE",
        num_dim=8,
        perplexity=10,
        random_state=0,
    )
    # tSNE populates obsm['X_tSNE']
    assert "X_tSNE" in mono.adata.obsm
    assert mono.adata.obsm["X_tSNE"].shape == (mono.adata.n_obs, 2)
    # reducedDimA is stored in uns for downstream clustering
    assert "reducedDimA" in mono.adata.uns["monocle"]
    assert mono.adata.uns["monocle"]["dim_reduce_type"] == "tSNE"


def test_tsne_deterministic_with_seed(small_branching_adata):
    """Passing the same random_state must produce the same embedding."""
    a = Monocle(small_branching_adata.copy())
    a.preprocess().select_ordering_genes()
    a.reduce_dimension(reduction_method="tSNE", num_dim=8,
                        perplexity=10, random_state=42)

    b = Monocle(small_branching_adata.copy())
    b.preprocess().select_ordering_genes()
    b.reduce_dimension(reduction_method="tSNE", num_dim=8,
                        perplexity=10, random_state=42)

    np.testing.assert_allclose(a.adata.obsm["X_tSNE"],
                                b.adata.obsm["X_tSNE"],
                                rtol=0, atol=1e-10)


def test_ica_reduction_smoke(small_branching_adata):
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes()
    mono.reduce_dimension(
        max_components=2,
        reduction_method="ICA",
        random_state=0,
    )
    assert "X_ICA" in mono.adata.obsm
    assert mono.adata.obsm["X_ICA"].shape == (mono.adata.n_obs, 2)
    # ICA populates reducedDimS / W and builds an MST on the cells
    assert "reducedDimS" in mono.adata.uns["monocle"]
    assert "mst" in mono.adata.uns["monocle"]
    assert mono.adata.uns["monocle"]["dim_reduce_type"] == "ICA"


def test_ica_order_cells_runs(small_branching_adata):
    """ICA → orderCells should work (uses cell-level MST directly)."""
    mono = Monocle(small_branching_adata.copy())
    mono.preprocess().select_ordering_genes()
    mono.reduce_dimension(reduction_method="ICA", random_state=0)
    mono.order_cells()
    assert "Pseudotime" in mono.adata.obs.columns
    assert "State" in mono.adata.obs.columns


def test_reduce_dimension_does_not_touch_global_rng():
    """Regression: previously ``np.random.seed(2016)`` was called inside
    ``reduce_dimension``. Verify the global RNG is unchanged after the call.
    """
    import numpy as np
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(12345)
    rng_legacy = np.random.RandomState(12345)
    # Snapshot the global RNG state
    before = np.random.get_state()

    X = np.random.default_rng(1).poisson(5, (50, 30)).astype(np.float64)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(50)]),
        var=pd.DataFrame({"gene_short_name": [f"g{i}" for i in range(30)]},
                         index=[f"g{i}" for i in range(30)]),
    )
    mono = Monocle(adata)
    mono.preprocess().select_ordering_genes().reduce_dimension()

    after = np.random.get_state()
    # Global RNG state should be untouched
    assert before[0] == after[0]
    # The keys (state arrays) should be identical element-wise
    np.testing.assert_array_equal(before[1], after[1])
