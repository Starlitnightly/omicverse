"""
Tests for the Rust-backed out-of-memory (OOM) AnnData path.

Covers the integration points that exist in omicverse itself. Deep checks of
the backend semantics live in the `anndataoom` package's own test suite.
"""

import importlib
import sys

import numpy as np
import pytest


anndataoom = pytest.importorskip("anndataoom")


# ──────────────────────────────────────────────────────────────────────
# Compat shim — must work whether or not anndataoom is installed
# ──────────────────────────────────────────────────────────────────────

def test_oom_compat_exports():
    from omicverse import _oom_compat

    for attr in (
        "HAS_OOM",
        "AnnDataOOM",
        "BackedArray",
        "BackedLayers",
        "TransformedBackedArray",
        "ScaledBackedArray",
        "oom_guard",
        "is_oom",
    ):
        assert hasattr(_oom_compat, attr), f"_oom_compat missing {attr!r}"


def test_oom_compat_fallback_when_anndataoom_missing(monkeypatch):
    """When anndataoom cannot be imported, the shim must provide no-op stubs."""
    saved = {
        k: v for k, v in sys.modules.items()
        if k == "anndataoom" or k.startswith("anndataoom.")
    }
    for name in saved:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "anndataoom", None)

    if "omicverse._oom_compat" in sys.modules:
        monkeypatch.delitem(sys.modules, "omicverse._oom_compat")
    compat = importlib.import_module("omicverse._oom_compat")

    assert compat.HAS_OOM is False
    assert compat.AnnDataOOM is None
    assert compat.is_oom(object()) is False

    @compat.oom_guard(materialize=True)
    def identity(x):
        return x

    assert identity(123) == 123


# ──────────────────────────────────────────────────────────────────────
# Fixtures — small h5ad written to a temp path
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_h5ad(tmp_path_factory):
    """A dense-enough synthetic h5ad that survives default QC filtering."""
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(0)
    n_obs, n_vars = 1000, 600
    # Poisson-like counts, high density so genes clear min_cells/robust filters.
    X = rng.poisson(lam=2.0, size=(n_obs, n_vars)).astype(np.float32)
    X = csr_matrix(X)

    obs = pd.DataFrame(
        {"batch": rng.choice(["A", "B"], size=n_obs)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    mt_idx = rng.choice(n_vars, size=20, replace=False)
    var_names = [
        f"MT-{i}" if i in mt_idx else f"gene_{i}" for i in range(n_vars)
    ]
    var = pd.DataFrame(index=var_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    path = tmp_path_factory.mktemp("oom") / "tiny.h5ad"
    adata.write_h5ad(path)
    return str(path)


# ──────────────────────────────────────────────────────────────────────
# ov.read(backend='rust')
# ──────────────────────────────────────────────────────────────────────

def test_read_rust_backend_returns_oom(tiny_h5ad):
    import omicverse as ov

    adata = ov.read(tiny_h5ad, backend="rust")
    try:
        assert type(adata).__name__ == "AnnDataOOM"
        assert getattr(adata, "_is_oom", False) is True
        assert adata.shape == (1000, 600)
    finally:
        adata.close()


def test_read_rust_backend_raises_without_anndataoom(tiny_h5ad, monkeypatch):
    """If anndataoom is not importable, backend='rust' must fail cleanly."""
    # Mark anndataoom as unavailable in sys.modules — a subsequent `import
    # anndataoom` inside _read_h5ad_rust will raise ImportError.
    for name in list(sys.modules):
        if name == "anndataoom" or name.startswith("anndataoom."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "anndataoom", None)

    from omicverse.io.single._read import _read_h5ad_rust

    with pytest.raises(ImportError, match="anndataoom"):
        _read_h5ad_rust(tiny_h5ad)


def test_read_python_backend_unaffected(tiny_h5ad):
    """backend='python' must not touch anndataoom at all."""
    import omicverse as ov

    adata = ov.read(tiny_h5ad, backend="python")
    assert adata.shape == (1000, 600)
    assert getattr(adata, "_is_oom", False) is False


def test_read_invalid_backend_raises(tiny_h5ad):
    import omicverse as ov

    with pytest.raises(ValueError, match="backend"):
        ov.read(tiny_h5ad, backend="julia")


# ──────────────────────────────────────────────────────────────────────
# Preprocess on OOM backend
# ──────────────────────────────────────────────────────────────────────

def test_preprocess_seurat_raises_on_oom(tiny_h5ad):
    """shiftlog|seurat is not implemented for OOM — it must raise, not run
    the Pearson chunked function in silence."""
    import omicverse as ov

    adata = ov.read(tiny_h5ad, backend="rust")
    try:
        ov.pp.qc(
            adata,
            mode="seurat",
            min_cells=0,
            min_genes=0,
            tresh={"mito_perc": 1.0, "nUMIs": 0, "detected_genes": 0},
            mt_startswith="MT-",
            doublets=False,
        )
        with pytest.raises(NotImplementedError, match="[Ss]eurat"):
            ov.pp.preprocess(adata, mode="shiftlog|seurat", n_HVGs=50)
    finally:
        adata.close()


def test_pca_varm_shape_matches_adata_n_vars(tiny_h5ad):
    """After the full OOM pipeline, adata.varm['PCs'] must have shape
    (adata.n_vars, n_comps). Regression test for issue #7."""
    import omicverse as ov

    adata = ov.read(tiny_h5ad, backend="rust")
    try:
        ov.pp.qc(
            adata,
            mode="seurat",
            min_cells=0,
            min_genes=0,
            tresh={"mito_perc": 1.0, "nUMIs": 0, "detected_genes": 0},
            mt_startswith="MT-",
            doublets=False,
        )
        assert adata.n_obs > 0 and adata.n_vars > 0, "QC filtered everything"

        ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=50)
        ov.pp.scale(adata)
        ov.pp.pca(adata, layer="scaled", n_pcs=10)

        assert adata.obsm["X_pca"].shape == (adata.n_obs, 10)
        assert adata.varm["PCs"].shape == (adata.n_vars, 10)
    finally:
        adata.close()
