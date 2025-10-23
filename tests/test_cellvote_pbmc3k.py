import os
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse


def _load_pbmc3k():
    """Load pbmc3k AnnData from a local .h5ad file or skip if missing."""
    import anndata as ad

    env_path = os.getenv("CELLVOTE_PBMC3K")
    candidates = [
        env_path,
        os.path.join("data", "pbmc3k.h5ad"),
        os.path.join(".", "pbmc3k.h5ad"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return ad.read_h5ad(p)

    pytest.skip(
        "pbmc3k .h5ad not found. Set CELLVOTE_PBMC3K or place file at ./data/pbmc3k.h5ad"
    )


def _ensure_clusters(adata, key="leiden"):
    import scanpy as sc
    if key in adata.obs:
        return key
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=12, n_pcs=30)
    sc.tl.leiden(adata, key_added=key, resolution=0.6)
    return key


def _compute_simple_markers(adata, cluster_key: str, topn: int = 15) -> Dict[str, List[str]]:
    markers = {}
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    for ct in adata.obs[cluster_key].cat.categories:
        idx_in = np.where(adata.obs[cluster_key].values == ct)[0]
        idx_out = np.where(adata.obs[cluster_key].values != ct)[0]
        mu_in = np.asarray(X[idx_in].mean(axis=0)).ravel()
        mu_out = np.asarray(X[idx_out].mean(axis=0)).ravel()
        scores = mu_in - mu_out
        top_idx = np.argsort(scores)[-topn:][::-1]
        markers[ct] = [adata.var_names[i] for i in top_idx]
    return markers


def _simulate_obs_annotations(adata, cluster_key: str, p_noise=(0.10, 0.15, 0.20)):
    rng = np.random.default_rng(42)
    clusters = adata.obs[cluster_key].astype(str)
    cats = adata.obs[cluster_key].cat.categories
    base_labels = {c: f"Type_{c}" for c in cats}

    def noisy(labels, p):
        values = []
        all_names = list(set(base_labels.values()))
        for cl in labels:
            if rng.random() < p:
                values.append(rng.choice(all_names))
            else:
                values.append(base_labels[cl])
        return pd.Categorical(values)

    adata.obs["scsa_annotation"] = noisy(clusters, p_noise[0])
    adata.obs["gpt_celltype"] = noisy(clusters, p_noise[1])
    adata.obs["gbi_celltype"] = noisy(clusters, p_noise[2])
    return base_labels


@pytest.fixture(scope="module")
def pbmc3k_adata():
    adata = _load_pbmc3k()
    _ensure_clusters(adata, key="leiden")
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")
    return adata


def test_vote_with_dummy_provider(pbmc3k_adata, monkeypatch):
    from omicverse.single import CellVote
    import omicverse.single._cellvote as cvmod

    adata = pbmc3k_adata.copy()
    cluster_key = "leiden"
    base_labels = _simulate_obs_annotations(adata, cluster_key)
    markers = _compute_simple_markers(adata, cluster_key, topn=10)

    def dummy_get_cluster_celltype(cluster_celltypes, cluster_markers, species, organization, model, base_url, provider, api_key=None, **kwargs):
        out = {}
        for cl, candidates in cluster_celltypes.items():
            if not candidates:
                out[cl] = "unknown"
                continue
            s = pd.Series(candidates).str.lower()
            out[cl] = s.value_counts().idxmax()
        return out

    monkeypatch.setattr(cvmod, "get_cluster_celltype", dummy_get_cluster_celltype)

    cv = CellVote(adata)
    result_map = cv.vote(
        clusters_key=cluster_key,
        cluster_markers=markers,
        celltype_keys=["scsa_annotation", "gpt_celltype", "gbi_celltype"],
        species="human",
        organization="PBMC",
        provider="openai",
        model="gpt-4o-mini",
        result_key="CellVote_celltype",
    )

    assert set(result_map.keys()) == set(adata.obs[cluster_key].cat.categories)
    assert "CellVote_celltype" in adata.obs
    assert pd.api.types.is_categorical_dtype(adata.obs["CellVote_celltype"]) or isinstance(
        adata.obs["CellVote_celltype"].dtype, pd.CategoricalDtype
    )


def test_get_cluster_celltype_timeout_fallback(monkeypatch):
    import requests
    import omicverse.single._cellvote as cvmod

    def fake_post(*args, **kwargs):
        raise requests.Timeout("simulated timeout")

    monkeypatch.setattr(cvmod.requests, "post", fake_post)

    cluster_celltypes = {"0": ["t cell", "t cell", "b cell"], "1": []}
    markers = {"0": ["CD3D", "CD3E"], "1": ["MS4A1"]}

    result = cvmod.get_cluster_celltype(
        cluster_celltypes,
        markers,
        species="human",
        organization="PBMC",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        provider="openai",
        api_key="test",
        timeout=1,
        max_retries=1,
        verbose=False,
    )

    assert result["0"] == "t cell"
    assert result["1"] == "unknown"


def test_get_cluster_celltype_malformed_response(monkeypatch):
    import omicverse.single._cellvote as cvmod

    class FakeResp:
        status_code = 200

        def json(self):
            return {"foo": "bar"}

    def fake_post(*a, **k):
        return FakeResp()

    monkeypatch.setattr(cvmod.requests, "post", fake_post)

    cluster_celltypes = {"0": ["myeloid", "t cell"]}
    result = cvmod.get_cluster_celltype(
        cluster_celltypes,
        cluster_markers={"0": ["LYZ"]},
        species="human",
        organization="PBMC",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        provider="openai",
        api_key="test",
        timeout=1,
        max_retries=0,
        verbose=False,
    )
    assert result["0"] == "myeloid"

