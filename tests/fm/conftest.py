"""Shared fixtures for ``ov.fm`` tests."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import scanpy as sc
import anndata as ad
from sklearn.metrics import silhouette_score  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_adata_path(tmp_path_factory):
    """Create a synthetic 100-cell x 200-gene .h5ad for testing.

    Gene names are HGNC-style symbols (``GENE0`` .. ``GENE199``),
    species is ``human``, and ``obs`` contains ``celltype`` and ``batch``.
    """
    tmp_dir = tmp_path_factory.mktemp("fm_test_data")
    path = str(tmp_dir / "test.h5ad")

    rng = np.random.default_rng(42)
    n_obs, n_vars = 100, 200

    X = rng.poisson(lam=3, size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_vars)]
    adata.obs_names = [f"CELL{i}" for i in range(n_obs)]
    adata.obs["celltype"] = rng.choice(["TypeA", "TypeB", "TypeC"], size=n_obs)
    adata.obs["batch"] = rng.choice(["batch1", "batch2"], size=n_obs)
    adata.uns["species"] = "human"

    adata.write_h5ad(path)
    return path


@pytest.fixture(scope="module")
def test_adata_with_embeddings(tmp_path_factory):
    """Synthetic .h5ad that already has UCE-like embeddings and provenance.

    The embeddings are 1280-dim with clusters aligned to cell type labels
    so that silhouette score is high.
    """
    tmp_dir = tmp_path_factory.mktemp("fm_test_embed")
    path = str(tmp_dir / "test_embed.h5ad")

    rng = np.random.default_rng(42)
    n_obs, n_vars, emb_dim = 100, 200, 1280

    X = rng.poisson(lam=3, size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE{i}" for i in range(n_vars)]
    adata.obs_names = [f"CELL{i}" for i in range(n_obs)]

    labels = rng.choice(["TypeA", "TypeB", "TypeC"], size=n_obs)
    adata.obs["celltype"] = labels
    adata.obs["batch"] = rng.choice(["batch1", "batch2"], size=n_obs)
    adata.uns["species"] = "human"

    # Create clustered embeddings (one hot-ish per cell type)
    embeddings = rng.normal(0, 0.1, size=(n_obs, emb_dim)).astype(np.float32)
    for i, label in enumerate(labels):
        if label == "TypeA":
            embeddings[i, 0] += 5.0
        elif label == "TypeB":
            embeddings[i, 1] += 5.0
        else:
            embeddings[i, 2] += 5.0
    adata.obsm["X_uce"] = embeddings

    # Add provenance
    provenance = {
        "model_name": "uce",
        "version": "1.0.0",
        "task": "embed",
        "output_keys": ["X_uce"],
        "timestamp": "2025-01-01T00:00:00",
        "backend": "local",
    }
    adata.uns["fm"] = {
        "runs_json": [json.dumps(provenance)],
        "latest_json": json.dumps(provenance),
    }

    adata.write_h5ad(path)
    return path


@pytest.fixture(scope="session")
def real_data_path():
    """Path to the real sample .h5ad file for heavy inference tests."""
    path = Path(__file__).resolve().parents[2] / "sample" / "rna_test.h5ad"
    assert path.exists(), f"Real sample data not found: {path}"
    return str(path)


# ---------------------------------------------------------------------------
# Checkpoint fixtures — auto-download if not already present
# ---------------------------------------------------------------------------


def _resolve_or_download(model_name: str, download_key: str) -> str:
    """Resolve checkpoint from env var or auto-download."""
    env_key = f"OV_FM_CHECKPOINT_DIR_{model_name.upper()}"
    env = os.environ.get(env_key)
    if env and Path(env).exists():
        return env
    cached = _resolve_from_default_models_dir(model_name, download_key)
    if cached is not None:
        os.environ[env_key] = cached
        return cached
    from omicverse.llm.model_download import download_model
    path = download_model(download_key)
    # Also set the env var so adapters can find it
    os.environ[env_key] = str(path)
    return str(path)


def _resolve_from_default_models_dir(*candidate_names: str) -> str | None:
    from omicverse.llm.model_download import get_default_models_dir

    root = get_default_models_dir()
    for name in candidate_names:
        if not name:
            continue
        candidate = root / name
        if candidate.exists() and (candidate.is_file() or any(candidate.iterdir())):
            return str(candidate)
    return None


@pytest.fixture(scope="session")
def scgpt_checkpoint_dir():
    """Resolve or auto-download scGPT checkpoint directory."""
    return _resolve_or_download("scgpt", "scgpt-whole-human")


@pytest.fixture(scope="session")
def geneformer_checkpoint_dir():
    """Resolve or auto-download Geneformer checkpoint directory."""
    return _resolve_or_download("geneformer", "geneformer")


@pytest.fixture(scope="session")
def uce_checkpoint_dir():
    """Resolve UCE checkpoint directory.

    UCE requires 5 asset files. Try env var first, then attempt
    auto-download from omicverse.llm if a UCE download spec exists.
    """
    env = os.environ.get("OV_FM_CHECKPOINT_DIR_UCE")
    if env and Path(env).exists():
        return env
    cached = _resolve_from_default_models_dir("uce")
    if cached is not None:
        os.environ["OV_FM_CHECKPOINT_DIR_UCE"] = cached
        return cached
    # Attempt auto-download — UCE assets may be registered
    from omicverse.llm.model_download import MODEL_REGISTRY, download_model
    if "uce" in MODEL_REGISTRY:
        path = download_model("uce")
        os.environ["OV_FM_CHECKPOINT_DIR_UCE"] = str(path)
        return str(path)
    pytest.skip(
        "UCE checkpoint not found. Set OV_FM_CHECKPOINT_DIR_UCE or add 'uce' "
        "to omicverse.llm.model_download.MODEL_REGISTRY"
    )


@pytest.fixture(scope="session")
def scfoundation_checkpoint_dir():
    """Resolve scFoundation checkpoint directory."""
    env = os.environ.get("OV_FM_CHECKPOINT_DIR_SCFOUNDATION")
    if env and Path(env).exists():
        return env
    cached = _resolve_from_default_models_dir("scfoundation")
    if cached is not None:
        os.environ["OV_FM_CHECKPOINT_DIR_SCFOUNDATION"] = cached
        return cached
    from omicverse.llm.model_download import MODEL_REGISTRY, download_model
    if "scfoundation" in MODEL_REGISTRY:
        path = download_model("scfoundation")
        os.environ["OV_FM_CHECKPOINT_DIR_SCFOUNDATION"] = str(path)
        return str(path)
    pytest.skip(
        "scFoundation checkpoint not found. Set OV_FM_CHECKPOINT_DIR_SCFOUNDATION"
    )


@pytest.fixture(scope="session")
def cellplm_checkpoint_dir():
    """Resolve CellPLM checkpoint directory."""
    env = os.environ.get("OV_FM_CHECKPOINT_DIR_CELLPLM")
    if env and Path(env).exists():
        return env
    cached = _resolve_from_default_models_dir("cellplm")
    if cached is not None:
        os.environ["OV_FM_CHECKPOINT_DIR_CELLPLM"] = cached
        return cached
    from omicverse.llm.model_download import MODEL_REGISTRY, download_model
    if "cellplm" in MODEL_REGISTRY:
        path = download_model("cellplm")
        os.environ["OV_FM_CHECKPOINT_DIR_CELLPLM"] = str(path)
        return str(path)
    pytest.skip(
        "CellPLM checkpoint not found. Set OV_FM_CHECKPOINT_DIR_CELLPLM"
    )


@pytest.fixture()
def fresh_registry():
    """Create a fresh ModelRegistry instance (not the global singleton)."""
    from omicverse.fm.registry import ModelRegistry

    return ModelRegistry()
