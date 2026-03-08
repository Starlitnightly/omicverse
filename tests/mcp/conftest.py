"""
Shared fixtures for MCP tests.

Provides a mock FunctionRegistry and mock functions so tests run offline
without anndata/scanpy installed.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from omicverse._registry import FunctionRegistry


# ---------------------------------------------------------------------------
# Mock functions that simulate real OmicVerse functions
# ---------------------------------------------------------------------------


def mock_read(path: str, backend: str = "python", **kwargs):
    """Mock ov.utils.read — returns a mock AnnData."""
    adata = _make_mock_adata(100, 500)
    adata._source_path = path
    return adata


def mock_qc(adata, mode: str = "seurat", **kwargs):
    """Mock ov.pp.qc — adds QC columns, returns adata."""
    adata.obs["n_genes"] = [200] * adata.shape[0]
    adata.obs["n_counts"] = [5000] * adata.shape[0]
    adata.obs["pct_counts_mt"] = [0.05] * adata.shape[0]
    adata.var["mt"] = [False] * adata.shape[1]
    return adata


def mock_scale(adata, max_value: float = 10, layers_add: str = "scaled", **kwargs):
    """Mock ov.pp.scale — adds 'scaled' layer."""
    adata.layers["scaled"] = adata.X.copy() if hasattr(adata.X, "copy") else adata.X
    return adata


def mock_pca(adata, n_pcs: int = 50, layer: str = "scaled", **kwargs):
    """Mock ov.pp.pca — adds X_pca, PCs, pca."""
    import numpy as np
    n = adata.shape[0]
    adata.obsm["X_pca"] = np.zeros((n, min(n_pcs, 50)))
    adata.varm["PCs"] = np.zeros((adata.shape[1], min(n_pcs, 50)))
    adata.uns["pca"] = {"params": {"n_pcs": n_pcs}}
    return adata


def mock_neighbors(adata, n_neighbors: int = 15, **kwargs):
    """Mock ov.pp.neighbors — adds connectivities, distances, neighbors."""
    adata.obsp["connectivities"] = "mock"
    adata.obsp["distances"] = "mock"
    adata.uns["neighbors"] = {"params": {"n_neighbors": n_neighbors}}
    return adata


def mock_umap(adata, **kwargs):
    """Mock ov.pp.umap — adds X_umap."""
    import numpy as np
    adata.obsm["X_umap"] = np.zeros((adata.shape[0], 2))
    return adata


def mock_leiden(adata, resolution: float = 1.0, key_added: str = "leiden", **kwargs):
    """Mock ov.pp.leiden — adds leiden column."""
    import numpy as np
    adata.obs[key_added] = np.random.choice(["0", "1", "2"], size=adata.shape[0])
    return adata


def mock_store_layers(adata, layers: str = "counts"):
    """Mock ov.utils.store_layers."""
    adata.uns[f"layers_{layers}"] = adata.X


def mock_retrieve_layers(adata, layers: str = "counts"):
    """Mock ov.utils.retrieve_layers."""
    key = f"layers_{layers}"
    if key in adata.uns:
        adata.X = adata.uns[key]


def mock_find_markers(adata, groupby: str, method: str = "cosg", n_genes: int = 50, **kwargs):
    """Mock ov.single.find_markers — writes to uns."""
    adata.uns["rank_genes_groups"] = {"names": ["gene1", "gene2"], "params": {"groupby": groupby}}


def mock_get_markers(adata, n_genes: int = 10, key: str = "rank_genes_groups", **kwargs):
    """Mock ov.single.get_markers — returns a mock DataFrame."""
    import types
    df = types.SimpleNamespace()
    df.__class__ = type("DataFrame", (), {"__name__": "DataFrame"})
    df.columns = ["group", "names", "scores"]
    df.shape = (n_genes, 3)
    df.values = type("Values", (), {"tolist": lambda self: [["0", "g1", 1.0]]})()
    df.head = lambda n=5: df
    return df


def mock_stateless_func(x: int = 1, y: str = "hello") -> dict:
    """A simple stateless function for testing."""
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# Mock classes for class-adapter testing (Phase 2)
# ---------------------------------------------------------------------------


class MockDEG:
    """Mimics pyDEG for class adapter tests."""

    def __init__(self, raw_data=None):
        self.raw_data = raw_data
        self._result = None
        self._deg_run = False

    def deg_analysis(
        self, treatment_groups=None, control_groups=None,
        method="DEseq2", **kwargs
    ):
        self._deg_run = True
        self._result = {
            "n_deg": 42,
            "method": method,
            "treatment": treatment_groups,
            "control": control_groups,
        }
        return self._result

    def foldchange_set(self, fc_threshold=2.0, pval_threshold=0.05):
        pass

    def results(self, n_genes=0):
        return self._result or {"n_deg": 0}


class MockAnnotator:
    """Mimics pySCSA for class adapter tests."""

    def __init__(self, adata=None, foldchange=1.5, pvalue=0.05,
                 species="Human", tissue="All", target="cellmarker"):
        self.adata = adata
        self.foldchange = foldchange
        self.species = species

    def cell_anno(self, clustertype="leiden", cluster="all", key_added="scsa_celltype"):
        return {
            "n_clusters": 5,
            "clustertype": clustertype,
            "key_added": key_added,
        }


class MockMetaCell:
    """Mimics MetaCell for class adapter tests."""

    def __init__(self, adata=None, use_rep="X_pca", n_metacells=None,
                 use_gpu=False, n_neighbors=15):
        self.adata = adata
        self.use_rep = use_rep
        self._trained = False

    def initialize_archetypes(self):
        pass

    def train(self, min_iter=10, max_iter=50):
        self._trained = True
        return {"converged": True, "iterations": 20}

    def predicted(self, method="soft", celltype_label="celltype"):
        return _make_mock_adata(20, 500)  # metacell adata


# ---------------------------------------------------------------------------
# Mock AnnData
# ---------------------------------------------------------------------------


class MockAnnData:
    """Minimal AnnData-like object for testing without anndata installed."""

    __name__ = "AnnData"

    def __init__(self, n_obs: int = 100, n_vars: int = 500):
        import numpy as np
        self.X = np.random.randn(n_obs, n_vars).astype("float32")
        self.obs = _DictLikeFrame(n_obs)
        self.var = _DictLikeFrame(n_vars)
        self.obsm: dict = {}
        self.obsp: dict = {}
        self.varm: dict = {}
        self.uns: dict = {}
        self.layers: dict = {}
        self._n_obs = n_obs
        self._n_vars = n_vars

    @property
    def shape(self):
        return (self._n_obs, self._n_vars)

    @property
    def var_names(self):
        return [f"Gene_{i}" for i in range(self._n_vars)]

    def write_h5ad(self, path):
        """Mock write — creates a JSON file for testing."""
        import json as _json
        with open(path, "w") as f:
            _json.dump({"mock": True, "shape": list(self.shape)}, f)


# Make isinstance checks work for type(obj).__name__ == "AnnData"
MockAnnData.__name__ = "AnnData"


class _DictLikeFrame:
    """Mimics pandas DataFrame .columns / dict-like access for obs/var."""

    def __init__(self, n_rows: int):
        self._data: dict = {}
        self._n = n_rows

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    @property
    def columns(self):
        return list(self._data.keys())

    def keys(self):
        return self._data.keys()


def _make_mock_adata(n_obs: int = 100, n_vars: int = 500) -> MockAnnData:
    return MockAnnData(n_obs, n_vars)


# ---------------------------------------------------------------------------
# Registry fixtures
# ---------------------------------------------------------------------------

# Registry entries matching the real OmicVerse structure
MOCK_ENTRIES = [
    {
        "function": mock_read,
        "full_name": "omicverse.utils._data.read",
        "short_name": "read",
        "module": "omicverse.utils._data",
        "aliases": ["read", "load_data"],
        "category": "utils",
        "description": "Universal file reader",
        "examples": [],
        "related": [],
        "signature": "(path, backend='python', **kwargs)",
        "parameters": ["path", "backend", "**kwargs"],
        "docstring": "",
        "prerequisites": {},
        "requires": {},
        "produces": {},
        "auto_fix": "none",
    },
    {
        "function": mock_store_layers,
        "full_name": "omicverse.utils._data.store_layers",
        "short_name": "store_layers",
        "module": "omicverse.utils._data",
        "aliases": ["store_layers"],
        "category": "utils",
        "description": "Store X matrix in adata.uns",
        "examples": [],
        "related": [],
        "signature": "(adata, layers='counts')",
        "parameters": ["adata", "layers"],
        "docstring": "",
        "prerequisites": {},
        "requires": {},
        "produces": {},
        "auto_fix": "none",
    },
    {
        "function": mock_retrieve_layers,
        "full_name": "omicverse.utils._data.retrieve_layers",
        "short_name": "retrieve_layers",
        "module": "omicverse.utils._data",
        "aliases": ["retrieve_layers"],
        "category": "utils",
        "description": "Retrieve stored X matrix from adata.uns",
        "examples": [],
        "related": [],
        "signature": "(adata, layers='counts')",
        "parameters": ["adata", "layers"],
        "docstring": "",
        "prerequisites": {},
        "requires": {},
        "produces": {},
        "auto_fix": "none",
    },
    {
        "function": mock_qc,
        "full_name": "omicverse.pp._qc.qc",
        "short_name": "qc",
        "module": "omicverse.pp._qc",
        "aliases": ["qc", "quality_control"],
        "category": "preprocessing",
        "description": "Quality control on single-cell data",
        "examples": [],
        "related": [],
        "signature": "(adata, **kwargs)",
        "parameters": ["adata", "**kwargs"],
        "docstring": "",
        "prerequisites": {},
        "requires": {},
        "produces": {"obs": ["n_genes", "n_counts", "pct_counts_mt"], "var": ["mt"]},
        "auto_fix": "none",
    },
    {
        "function": mock_scale,
        "full_name": "omicverse.pp._preprocess.scale",
        "short_name": "scale",
        "module": "omicverse.pp._preprocess",
        "aliases": ["scale", "scaling"],
        "category": "preprocessing",
        "description": "Scale data to unit variance and zero mean",
        "examples": [],
        "related": [],
        "signature": "(adata, max_value=10, layers_add='scaled', to_sparse=True, **kwargs)",
        "parameters": ["adata", "max_value", "layers_add", "to_sparse", "**kwargs"],
        "docstring": "",
        "prerequisites": {"optional_functions": ["normalize", "qc"]},
        "requires": {},
        "produces": {"layers": ["scaled"]},
        "auto_fix": "none",
    },
    {
        "function": mock_pca,
        "full_name": "omicverse.pp._preprocess.pca",
        "short_name": "pca",
        "module": "omicverse.pp._preprocess",
        "aliases": ["pca", "PCA"],
        "category": "preprocessing",
        "description": "Principal Component Analysis",
        "examples": [],
        "related": [],
        "signature": "(adata, n_pcs=50, layer='scaled', inplace=True, **kwargs)",
        "parameters": ["adata", "n_pcs", "layer", "inplace", "**kwargs"],
        "docstring": "",
        "prerequisites": {"functions": ["scale"]},
        "requires": {"layers": ["scaled"]},
        "produces": {"obsm": ["X_pca"], "varm": ["PCs"], "uns": ["pca"]},
        "auto_fix": "escalate",
    },
    {
        "function": mock_neighbors,
        "full_name": "omicverse.pp._preprocess.neighbors",
        "short_name": "neighbors",
        "module": "omicverse.pp._preprocess",
        "aliases": ["neighbors", "knn"],
        "category": "preprocessing",
        "description": "Compute neighborhood graph",
        "examples": [],
        "related": [],
        "signature": "(adata, n_neighbors=15, **kwargs)",
        "parameters": ["adata", "n_neighbors", "**kwargs"],
        "docstring": "",
        "prerequisites": {"optional_functions": ["pca"]},
        "requires": {"obsm": ["X_pca"]},
        "produces": {"obsp": ["distances", "connectivities"], "uns": ["neighbors"]},
        "auto_fix": "auto",
    },
    {
        "function": mock_umap,
        "full_name": "omicverse.pp._preprocess.umap",
        "short_name": "umap",
        "module": "omicverse.pp._preprocess",
        "aliases": ["umap", "UMAP"],
        "category": "preprocessing",
        "description": "Compute UMAP embedding",
        "examples": [],
        "related": [],
        "signature": "(adata, **kwargs)",
        "parameters": ["adata", "**kwargs"],
        "docstring": "",
        "prerequisites": {"functions": ["neighbors"]},
        "requires": {"uns": ["neighbors"], "obsp": ["connectivities", "distances"]},
        "produces": {"obsm": ["X_umap"]},
        "auto_fix": "auto",
    },
    {
        "function": mock_leiden,
        "full_name": "omicverse.pp._preprocess.leiden",
        "short_name": "leiden",
        "module": "omicverse.pp._preprocess",
        "aliases": ["leiden", "clustering"],
        "category": "preprocessing",
        "description": "Leiden community detection",
        "examples": [],
        "related": [],
        "signature": "(adata, resolution=1.0, random_state=0, key_added='leiden', **kwargs)",
        "parameters": ["adata", "resolution", "random_state", "key_added", "**kwargs"],
        "docstring": "",
        "prerequisites": {"functions": ["neighbors"]},
        "requires": {"uns": ["neighbors"], "obsp": ["connectivities"]},
        "produces": {"obs": ["leiden"]},
        "auto_fix": "auto",
    },
    {
        "function": mock_find_markers,
        "full_name": "omicverse.single._markers.find_markers",
        "short_name": "find_markers",
        "module": "omicverse.single._markers",
        "aliases": ["find_markers"],
        "category": "single",
        "description": "Find marker genes per cluster",
        "examples": [],
        "related": [],
        "signature": "(adata, groupby, method='cosg', n_genes=50, **kwargs)",
        "parameters": ["adata", "groupby", "method", "n_genes", "**kwargs"],
        "docstring": "",
        "prerequisites": {},
        "requires": {},
        "produces": {"uns": ["rank_genes_groups"]},
        "auto_fix": "none",
    },
    {
        "function": mock_get_markers,
        "full_name": "omicverse.single._markers.get_markers",
        "short_name": "get_markers",
        "module": "omicverse.single._markers",
        "aliases": ["get_markers"],
        "category": "single",
        "description": "Extract top marker genes as DataFrame",
        "examples": [],
        "related": [],
        "signature": "(adata, n_genes=10, key='rank_genes_groups', **kwargs)",
        "parameters": ["adata", "n_genes", "key", "**kwargs"],
        "docstring": "",
        "prerequisites": {},
        "requires": {},
        "produces": {},
        "auto_fix": "none",
    },
    # A stateless function for testing function_adapter
    {
        "function": mock_stateless_func,
        "full_name": "omicverse.utils._test.mock_stateless",
        "short_name": "mock_stateless",
        "module": "omicverse.utils._test",
        "aliases": ["mock_stateless"],
        "category": "utils",
        "description": "A test stateless function",
        "examples": [],
        "related": [],
        "signature": "(x=1, y='hello')",
        "parameters": ["x", "y"],
        "docstring": "",
        "prerequisites": {},
        "requires": {},
        "produces": {},
        "auto_fix": "none",
    },
    # Class-backed entries (Phase 2)
    {
        "function": MockDEG,
        "full_name": "omicverse.bulk._Deseq2.pyDEG",
        "short_name": "pyDEG",
        "module": "omicverse.bulk._Deseq2",
        "aliases": ["pyDEG"],
        "category": "bulk",
        "description": "Differential expression analysis for bulk RNA-seq",
        "examples": [],
        "related": [],
        "signature": "(raw_data)",
        "parameters": ["raw_data"],
        "docstring": "",
        "prerequisites": {},
        "requires": {},
        "produces": {},
        "auto_fix": "none",
    },
    {
        "function": MockAnnotator,
        "full_name": "omicverse.single._anno.pySCSA",
        "short_name": "pySCSA",
        "module": "omicverse.single._anno",
        "aliases": ["pySCSA"],
        "category": "single",
        "description": "Automated cell type annotation",
        "examples": [],
        "related": [],
        "signature": "(adata, foldchange=1.5)",
        "parameters": ["adata", "foldchange"],
        "docstring": "",
        "prerequisites": {},
        "requires": {},
        "produces": {},
        "auto_fix": "none",
    },
    {
        "function": MockMetaCell,
        "full_name": "omicverse.single._metacell.MetaCell",
        "short_name": "MetaCell",
        "module": "omicverse.single._metacell",
        "aliases": ["MetaCell"],
        "category": "single",
        "description": "Metacell construction",
        "examples": [],
        "related": [],
        "signature": "(adata, use_rep='X_pca')",
        "parameters": ["adata", "use_rep"],
        "docstring": "",
        "prerequisites": {},
        "requires": {},
        "produces": {},
        "auto_fix": "none",
    },
]


def build_mock_registry() -> FunctionRegistry:
    """Build a FunctionRegistry pre-populated with mock entries.

    Directly injects entries to preserve the intended full_name
    (registry.register() would use func.__module__ which points to conftest).
    """
    reg = FunctionRegistry()
    for entry in MOCK_ENTRIES:
        full_name = entry["full_name"]
        short_name = entry["short_name"]
        category = entry["category"]

        # Directly inject into internal structures
        reg._registry[short_name.lower()] = entry
        reg._registry[full_name.lower()] = entry
        for alias in entry["aliases"]:
            reg._registry[alias.lower()] = entry

        reg._function_map[entry["function"]] = full_name

        if category not in reg._categories:
            reg._categories[category] = []
        reg._categories[category].append(full_name)
    return reg


@pytest.fixture
def mock_registry():
    """Fixture providing a populated FunctionRegistry."""
    return build_mock_registry()


@pytest.fixture
def mock_manifest(mock_registry):
    """Fixture providing a manifest built from the mock registry."""
    from omicverse.mcp.manifest import build_registry_manifest
    return build_registry_manifest(registry=mock_registry)


@pytest.fixture
def session_store():
    """Fixture providing a clean SessionStore."""
    from omicverse.mcp.session_store import SessionStore
    return SessionStore()


@pytest.fixture
def mock_adata():
    """Fixture providing a mock AnnData object."""
    return _make_mock_adata(100, 500)


@pytest.fixture
def executor_with_mock(mock_registry):
    """Fixture providing an McpExecutor wired to mock registry."""
    from omicverse.mcp.manifest import build_registry_manifest
    from omicverse.mcp.session_store import SessionStore
    from omicverse.mcp.executor import McpExecutor

    manifest = build_registry_manifest(registry=mock_registry)
    store = SessionStore()
    return McpExecutor(manifest, store)


def register_mock_class_specs():
    """Register ClassWrapperSpecs that reference the mock classes.

    The class_adapter imports classes by full_name via importlib, but mock
    classes live in conftest.  We monkeypatch ``_import_class`` in tests
    instead.  This function just registers the specs so the adapter finds them.
    """
    from omicverse.mcp.class_specs import (
        ClassWrapperSpec, ActionSpec, register_spec, _CLASS_SPECS,
    )

    # pyDEG
    spec_deg = ClassWrapperSpec(
        full_name="omicverse.bulk._Deseq2.pyDEG",
        tool_name="ov.bulk.pydeg",
        description="Mock DEG analysis",
        runtime_requirements={"packages": [], "modules": ["omicverse.bulk._Deseq2"]},
        actions={
            "create": ActionSpec(
                name="create", method="__init__",
                needs_instance=False, needs_adata=True,
                returns="instance_id",
                description="Create pyDEG instance",
                params={
                    "layer": {"type": "string", "default": "X"},
                },
            ),
            "run": ActionSpec(
                name="run", method="deg_analysis",
                needs_instance=True, returns="json",
                description="Run DEG analysis",
                params={
                    "treatment_groups": {"type": "array", "items": {"type": "string"}},
                    "control_groups": {"type": "array", "items": {"type": "string"}},
                    "method": {"type": "string", "default": "DEseq2"},
                },
                required_params=["treatment_groups", "control_groups"],
            ),
            "results": ActionSpec(
                name="results", method="results",
                needs_instance=True, returns="json",
                description="Get DEG results",
            ),
            "destroy": ActionSpec(
                name="destroy", method="destroy",
                needs_instance=True, returns="void",
            ),
        },
    )

    # pySCSA
    spec_scsa = ClassWrapperSpec(
        full_name="omicverse.single._anno.pySCSA",
        tool_name="ov.single.pyscsa",
        description="Mock cell annotation",
        runtime_requirements={"packages": [], "modules": ["omicverse.single._anno"]},
        actions={
            "create": ActionSpec(
                name="create", method="__init__",
                needs_instance=False, needs_adata=True,
                returns="instance_id",
                description="Create pySCSA instance",
                params={
                    "foldchange": {"type": "number", "default": 1.5},
                    "species": {"type": "string", "default": "Human"},
                },
            ),
            "annotate": ActionSpec(
                name="annotate", method="cell_anno",
                needs_instance=True, needs_adata=False,
                returns="json",
                description="Run cell type annotation",
                params={
                    "clustertype": {"type": "string", "default": "leiden"},
                },
            ),
            "destroy": ActionSpec(
                name="destroy", method="destroy",
                needs_instance=True, returns="void",
            ),
        },
    )

    # MetaCell
    spec_mc = ClassWrapperSpec(
        full_name="omicverse.single._metacell.MetaCell",
        tool_name="ov.single.metacell",
        description="Mock metacell construction",
        runtime_requirements={"packages": [], "modules": ["omicverse.single._metacell"]},
        actions={
            "create": ActionSpec(
                name="create", method="__init__",
                needs_instance=False, needs_adata=True,
                returns="instance_id",
                description="Create MetaCell instance",
                params={
                    "use_rep": {"type": "string", "default": "X_pca"},
                },
            ),
            "train": ActionSpec(
                name="train", method="train",
                needs_instance=True, returns="json",
                description="Train SEACells model",
                params={
                    "min_iter": {"type": "integer", "default": 10},
                    "max_iter": {"type": "integer", "default": 50},
                },
            ),
            "predict": ActionSpec(
                name="predict", method="predicted",
                needs_instance=True, returns="object_ref",
                description="Generate metacell AnnData",
                params={
                    "method": {"type": "string", "default": "soft"},
                },
            ),
            "destroy": ActionSpec(
                name="destroy", method="destroy",
                needs_instance=True, returns="void",
            ),
        },
    )

    register_spec(spec_deg)
    register_spec(spec_scsa)
    register_spec(spec_mc)

    return {
        "pyDEG": spec_deg,
        "pySCSA": spec_scsa,
        "MetaCell": spec_mc,
    }


# Map full_name → mock class for monkeypatching _import_class
MOCK_CLASS_MAP = {
    "omicverse.bulk._Deseq2.pyDEG": MockDEG,
    "omicverse.single._anno.pySCSA": MockAnnotator,
    "omicverse.single._metacell.MetaCell": MockMetaCell,
}


@pytest.fixture
def class_adapter_setup(mock_adata, session_store):
    """Set up ClassAdapter with mock specs, monkeypatched import.

    Returns (adapter, store, adata_id, specs).
    """
    from omicverse.mcp.adapters.class_adapter import ClassAdapter

    specs = register_mock_class_specs()

    adapter = ClassAdapter()

    # Monkeypatch _import_class to return mock classes
    original_import = adapter._import_class

    def mock_import(full_name):
        if full_name in MOCK_CLASS_MAP:
            return MOCK_CLASS_MAP[full_name]
        return original_import(full_name)

    adapter._import_class = mock_import

    # Store an adata for tests that need it
    adata_id = session_store.create_adata(mock_adata)

    return adapter, session_store, adata_id, specs


@pytest.fixture
def server_with_p2(mock_registry):
    """Create a mock RegistryMcpServer including P2 class tools."""
    from omicverse.mcp.manifest import build_registry_manifest
    from omicverse.mcp.session_store import SessionStore
    from omicverse.mcp.executor import McpExecutor
    from omicverse.mcp.server import RegistryMcpServer

    # Ensure mock class specs are registered
    register_mock_class_specs()

    manifest = build_registry_manifest(registry=mock_registry, phase="P0+P0.5+P2")

    srv = RegistryMcpServer.__new__(RegistryMcpServer)
    srv._phase = "P0+P0.5+P2"
    srv._store = SessionStore()
    srv._manifest = manifest
    srv._executor = McpExecutor(manifest, srv._store)
    return srv
