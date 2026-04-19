"""Unit tests for ov.micro (pure-Python, no external CLI required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_adata(n_samples: int = 6, n_features: int = 8, seed: int = 0):
    import anndata as ad
    from scipy import sparse
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 100, size=(n_samples, n_features))
    obs = pd.DataFrame(
        {"group": ["A"] * (n_samples // 2) + ["B"] * (n_samples - n_samples // 2)},
        index=[f"S{i}" for i in range(n_samples)],
    )
    half = n_features // 2
    var = pd.DataFrame(
        {
            "phylum":     ["Firmicutes"] * half + ["Bacteroidetes"] * (n_features - half),
            "genus":      [f"G{i % 3}" for i in range(n_features)],
            "taxonomy":   [f"tax{i}" for i in range(n_features)],
            "sequence":   ["ACGT"] * n_features,
            "sintax_confidence": [0.9] * n_features,
        },
        index=[f"ASV{i}" for i in range(n_features)],
    )
    return ad.AnnData(
        X=sparse.csr_matrix(counts.astype(np.int32)), obs=obs, var=var
    )


def test_dense_sparse_passthrough():
    from omicverse.micro._utils import dense
    from scipy import sparse
    X = np.arange(12, dtype=np.int32).reshape(3, 4)
    dense_X = dense(X)
    assert dense_X is X or np.array_equal(dense_X, X)
    sp = sparse.csr_matrix(X)
    out = dense(sp)
    assert np.array_equal(out, X)


def test_rarefy_counts_caps_row_sums():
    from omicverse.micro._utils import rarefy_counts
    X = np.array([[10, 20, 30], [5, 5, 0]], dtype=np.int64)
    out = rarefy_counts(X, depth=10, seed=0)
    assert out[0].sum() == 10
    np.testing.assert_array_equal(out[1], X[1])


def test_rarefy_counts_none_depth_identity():
    from omicverse.micro._utils import rarefy_counts
    X = np.array([[3, 0, 4]], dtype=np.int64)
    np.testing.assert_array_equal(rarefy_counts(X, depth=None), X)


def test_bh_fdr_matches_monotone_property():
    from omicverse.micro._da import _bh_fdr
    p = np.array([0.001, 0.01, 0.03, 0.5, 1.0])
    q = _bh_fdr(p)
    assert all(q[i] <= q[j] for i, j in zip(range(4), range(1, 5)))
    assert (q >= 0).all() and (q <= 1).all()
    assert q[0] < q[-1]


def test_collapse_taxa_by_phylum():
    from omicverse.micro import collapse_taxa
    adata = _make_adata(n_samples=4, n_features=8)
    ag = collapse_taxa(adata, rank="phylum")
    assert ag.shape[1] == 2
    assert set(ag.var_names) == {"Firmicutes", "Bacteroidetes"}
    assert ag.X.toarray().sum() == adata.X.toarray().sum()


def test_alpha_shannon_observed_runs_and_writes_obs():
    from omicverse.micro import Alpha
    adata = _make_adata()
    a = Alpha(adata).run(metrics=["shannon", "observed_otus"])
    assert "shannon" in a.columns and "observed_otus" in a.columns
    assert a.shape[0] == adata.n_obs
    assert "shannon" in adata.obs.columns


def test_beta_does_not_mutate_rarefy_depth():
    """Regression test for PR#637 review: Beta.run() mutated self.rarefy_depth."""
    from omicverse.micro import Beta
    adata = _make_adata()
    b = Beta(adata, rarefy_depth=None)
    b.run("braycurtis", rarefy=True)
    assert b.rarefy_depth is None

    b2 = Beta(adata, rarefy_depth=50)
    b2.run("braycurtis", rarefy=False)
    assert b2.rarefy_depth == 50


def test_rarefy_preserves_counts_raw_layer():
    from omicverse.micro import rarefy
    adata = _make_adata(n_samples=3, n_features=4)
    original_X = adata.X.toarray().copy()
    out = rarefy(adata, depth=None, copy=True, save_original=True)
    assert "counts_raw" in out.layers
    np.testing.assert_array_equal(out.layers["counts_raw"].toarray(), original_X)
    depths = out.X.toarray().sum(axis=1)
    assert depths.min() == depths.max()


def test_da_wilcoxon_returns_expected_columns():
    from omicverse.micro import DA
    adata = _make_adata(n_samples=6, n_features=6)
    out = DA(adata).wilcoxon(group_key="group", group_a="A", group_b="B", rank=None)
    for col in ("feature", "U_stat", "p_value", "fdr_bh", "prevalence"):
        assert col in out.columns
    assert (out["p_value"] >= 0).all() and (out["p_value"] <= 1).all()
