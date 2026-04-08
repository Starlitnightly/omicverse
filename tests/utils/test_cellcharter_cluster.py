import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score

import omicverse as ov
from omicverse.external.cellcharter import Cluster as OptimizedCluster
from omicverse.external.cellcharter import aggregate_neighbors as optimized_aggregate_neighbors


REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_CELLCHARTER_SRC = REPO_ROOT.parent / "cellcharter" / "src" / "cellcharter"


def _clear_module_prefixes(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)


def _make_test_adata(seed: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    n_per_group = 12
    centers = np.array(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [2.5, 4.0],
        ],
        dtype=np.float64,
    )

    coords = np.vstack([center + rng.normal(scale=0.25, size=(n_per_group, 2)) for center in centers])
    embedding = np.vstack(
        [
            np.column_stack(
                [
                    np.full(n_per_group, idx, dtype=np.float32),
                    rng.normal(loc=idx * 3.0, scale=0.2, size=n_per_group),
                    rng.normal(loc=idx * 5.0, scale=0.2, size=n_per_group),
                ]
            )
            for idx in range(len(centers))
        ]
    ).astype(np.float32)

    adata = AnnData(X=embedding.copy())
    adata.obsm["X_pca"] = embedding
    adata.obsm["spatial"] = coords
    return adata


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_squidpy_stubs():
    squidpy_pkg = types.ModuleType("squidpy")
    squidpy_pkg.__path__ = []
    sys.modules.setdefault("squidpy", squidpy_pkg)

    constants_pkg = types.ModuleType("squidpy._constants")
    constants_pkg.__path__ = []
    sys.modules.setdefault("squidpy._constants", constants_pkg)

    class _ObspKey:
        @staticmethod
        def spatial_conn(key=None):
            return "spatial_connectivities" if key is None else key

        @staticmethod
        def spatial_dist(key=None):
            return "spatial_distances" if key is None else key

    class _UnsKey:
        @staticmethod
        def spatial_neighs(key=None):
            return "spatial_neighbors" if key is None else key

    class _Key:
        obsp = _ObspKey()
        uns = _UnsKey()

    pkg_constants = types.ModuleType("squidpy._constants._pkg_constants")
    pkg_constants.Key = _Key
    sys.modules["squidpy._constants._pkg_constants"] = pkg_constants

    docs_mod = types.ModuleType("squidpy._docs")

    class _Docs:
        @staticmethod
        def dedent(func):
            return func

    docs_mod.d = _Docs()
    sys.modules["squidpy._docs"] = docs_mod

    gr_pkg = types.ModuleType("squidpy.gr")
    gr_pkg.__path__ = []
    sys.modules.setdefault("squidpy.gr", gr_pkg)

    gr_utils = types.ModuleType("squidpy.gr._utils")

    def _assert_connectivity_key(adata, key):
        if key not in adata.obsp:
            raise KeyError(key)

    gr_utils._assert_connectivity_key = _assert_connectivity_key
    sys.modules["squidpy.gr._utils"] = gr_utils


def _load_original_cellcharter_modules():
    if not ORIGINAL_CELLCHARTER_SRC.exists():
        pytest.skip("Sibling `cellcharter` repository was not found.")

    _install_squidpy_stubs()

    cellcharter_pkg = types.ModuleType("cellcharter")
    cellcharter_pkg.__path__ = [str(ORIGINAL_CELLCHARTER_SRC)]
    sys.modules["cellcharter"] = cellcharter_pkg

    constants_pkg = types.ModuleType("cellcharter._constants")
    constants_pkg.__path__ = [str(ORIGINAL_CELLCHARTER_SRC / "_constants")]
    sys.modules["cellcharter._constants"] = constants_pkg

    gr_pkg = types.ModuleType("cellcharter.gr")
    gr_pkg.__path__ = [str(ORIGINAL_CELLCHARTER_SRC / "gr")]
    sys.modules["cellcharter.gr"] = gr_pkg

    tl_pkg = types.ModuleType("cellcharter.tl")
    tl_pkg.__path__ = [str(ORIGINAL_CELLCHARTER_SRC / "tl")]
    sys.modules["cellcharter.tl"] = tl_pkg

    _load_module("cellcharter._utils", ORIGINAL_CELLCHARTER_SRC / "_utils.py")
    _load_module(
        "cellcharter._constants._pkg_constants",
        ORIGINAL_CELLCHARTER_SRC / "_constants" / "_pkg_constants.py",
    )
    aggr_mod = _load_module("cellcharter.gr._aggr", ORIGINAL_CELLCHARTER_SRC / "gr" / "_aggr.py")
    gmm_mod = _load_module("cellcharter.tl._gmm", ORIGINAL_CELLCHARTER_SRC / "tl" / "_gmm.py")
    return aggr_mod.aggregate_neighbors, gmm_mod.Cluster


@pytest.fixture(autouse=True)
def _restore_stubbed_modules():
    saved = {
        name: sys.modules.get(name)
        for name in list(sys.modules)
        if name == "cellcharter" or name.startswith("cellcharter.") or name == "squidpy" or name.startswith("squidpy.")
    }
    yield
    _clear_module_prefixes(("cellcharter", "squidpy"))
    for name, module in saved.items():
        if module is not None:
            sys.modules[name] = module


def test_spatial_neighbors_delaunay_builds_expected_graph():
    adata = AnnData(X=np.ones((4, 2), dtype=np.float32))
    adata.obsm["spatial"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    ov.space.spatial_neighbors(adata, delaunay=True)

    conn = adata.obsp["spatial_connectivities"]
    dist = adata.obsp["spatial_distances"]

    assert conn.shape == (4, 4)
    assert dist.shape == (4, 4)
    assert conn.nnz == dist.nnz
    assert np.allclose(conn.toarray(), conn.toarray().T)
    assert np.allclose(dist.toarray(), dist.toarray().T)
    assert adata.uns["spatial_neighbors"]["params"]["delaunay"] is True


def test_utils_cluster_cellcharter_creates_spatial_labels():
    adata = _make_test_adata()

    model = ov.utils.cluster(
        adata,
        method="cellcharter",
        n_components=3,
        use_rep="X_pca",
        n_layers=2,
        trim_long_links=False,
        delaunay=True,
    )

    assert model is not None
    assert "X_cellcharter" in adata.obsm
    assert "cellcharter" in adata.obs
    assert str(adata.obs["cellcharter"].dtype) == "category"
    assert len(adata.obs["cellcharter"].cat.categories) == 3
    assert adata.obsm["X_cellcharter"].shape[0] == adata.n_obs
    assert adata.uns["_cellcharter"]["backend"] in {"torchgmm", "sklearn"}


@pytest.mark.skipif(importlib.util.find_spec("torchgmm") is None, reason="requires torchgmm")
def test_optimized_matches_original_cellcharter_torchgmm_backend():
    original_aggregate_neighbors, OriginalCluster = _load_original_cellcharter_modules()

    optimized_adata = _make_test_adata(seed=7)
    original_adata = optimized_adata.copy()

    ov.space.spatial_neighbors(optimized_adata, delaunay=True)
    original_adata.obsp["spatial_connectivities"] = optimized_adata.obsp["spatial_connectivities"].copy()
    original_adata.obsp["spatial_distances"] = optimized_adata.obsp["spatial_distances"].copy()
    original_adata.uns["spatial_neighbors"] = optimized_adata.uns["spatial_neighbors"].copy()

    optimized_aggregate_neighbors(
        optimized_adata,
        n_layers=2,
        aggregations="mean",
        connectivity_key="spatial_connectivities",
        use_rep="X_pca",
        out_key="X_cellcharter",
    )
    original_aggregate_neighbors(
        original_adata,
        n_layers=2,
        aggregations="mean",
        connectivity_key="spatial_connectivities",
        use_rep="X_pca",
        out_key="X_cellcharter",
    )

    np.testing.assert_allclose(
        optimized_adata.obsm["X_cellcharter"],
        original_adata.obsm["X_cellcharter"],
        rtol=1e-6,
        atol=1e-6,
    )

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    original_model = OriginalCluster(n_clusters=3, random_state=seed)
    original_model.fit(original_adata, use_rep="X_cellcharter")
    original_labels = np.asarray(original_model.predict(original_adata, use_rep="X_cellcharter")).astype(str)

    np.random.seed(seed)
    torch.manual_seed(seed)
    optimized_model = OptimizedCluster(n_clusters=3, random_state=seed, backend="torchgmm")
    optimized_model.fit(optimized_adata, use_rep="X_cellcharter")
    optimized_labels = np.asarray(optimized_model.predict(optimized_adata, use_rep="X_cellcharter")).astype(str)

    assert optimized_model.backend_ == "torchgmm"
    assert adjusted_rand_score(original_labels, optimized_labels) == pytest.approx(1.0)
    assert sorted(np.unique(original_labels, return_counts=True)[1].tolist()) == sorted(
        np.unique(optimized_labels, return_counts=True)[1].tolist()
    )
    assert float(original_model.nll_) == pytest.approx(float(optimized_model.nll_), rel=1e-6, abs=1e-6)
