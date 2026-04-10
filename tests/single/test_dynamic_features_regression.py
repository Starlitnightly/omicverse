from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _bootstrap_omicverse_single_packages():
    saved = {
        name: sys.modules.get(name)
        for name in [
            "omicverse",
            "omicverse.single",
            "omicverse._registry",
            "omicverse._settings",
        ]
    }
    for name in saved:
        sys.modules.pop(name, None)

    ov_pkg = types.ModuleType("omicverse")
    ov_pkg.__path__ = [str(PACKAGE_ROOT)]
    ov_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse", loader=None, is_package=True)
    sys.modules["omicverse"] = ov_pkg

    single_pkg = types.ModuleType("omicverse.single")
    single_pkg.__path__ = [str(PACKAGE_ROOT / "single")]
    single_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse.single", loader=None, is_package=True)
    sys.modules["omicverse.single"] = single_pkg
    ov_pkg.single = single_pkg

    registry_mod = types.ModuleType("omicverse._registry")

    def register_function(**_kwargs):
        def decorator(func):
            return func

        return decorator

    registry_mod.register_function = register_function
    sys.modules["omicverse._registry"] = registry_mod
    ov_pkg._registry = registry_mod

    settings_mod = types.ModuleType("omicverse._settings")

    class _Colors:
        HEADER = ""
        BOLD = ""
        CYAN = ""
        GREEN = ""
        ENDC = ""

    settings_mod.Colors = _Colors
    settings_mod.EMOJI = {"start": "", "done": ""}
    sys.modules["omicverse._settings"] = settings_mod
    ov_pkg._settings = settings_mod

    return saved


_SAVED = _bootstrap_omicverse_single_packages()
dynamic_features_mod = importlib.import_module("omicverse.single._dynamic_features")

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


def test_dynamic_features_accepts_full_length_weights_after_pseudotime_filter(monkeypatch):
    adata = AnnData(np.array([[1.0], [2.0], [3.0]]))
    adata.var_names = pd.Index(["gene_a"])
    adata.obs["pseudotime"] = [0.0, np.nan, 2.0]

    fit_calls = []

    class FakeGAM:
        statistics_ = {}

        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self, x, y, weights=None):
            fit_calls.append(
                {
                    "x": np.asarray(x).reshape(-1).tolist(),
                    "y": np.asarray(y).reshape(-1).tolist(),
                    "weights": None if weights is None else np.asarray(weights).reshape(-1).tolist(),
                }
            )
            return self

        def predict(self, x):
            x = np.asarray(x).reshape(-1)
            return np.linspace(0.0, 1.0, len(x))

        def confidence_intervals(self, x, width=0.95):
            x = np.asarray(x).reshape(-1)
            lower = np.zeros(len(x), dtype=float)
            upper = np.ones(len(x), dtype=float)
            return np.column_stack([lower, upper])

    monkeypatch.setattr(
        dynamic_features_mod,
        "_require_pygam",
        lambda: {
            ("normal", "identity"): FakeGAM,
            "s": lambda *args, **kwargs: ("term", args, kwargs),
        },
    )

    result = dynamic_features_mod.dynamic_features(
        adata,
        genes=["gene_a"],
        pseudotime="pseudotime",
        weights=[1.0, 2.0, 3.0],
        min_cells=2,
        grid_size=5,
    )

    assert fit_calls, "Expected the fake GAM to be fit once"
    assert fit_calls[0]["x"] == [0.0, 2.0]
    assert fit_calls[0]["weights"] == [1.0, 3.0]
    assert result.stats["success"].tolist() == [True]


def test_dynamic_features_uns_tables_are_h5ad_serializable(monkeypatch, tmp_path):
    adata = AnnData(np.array([[1.0], [2.0], [3.0]]))
    adata.var_names = pd.Index(["gene_a"])
    adata.obs["pseudotime"] = [0.0, 1.0, 2.0]

    class FakeGAM:
        statistics_ = {}

        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self, x, y, weights=None):
            return self

        def predict(self, x):
            x = np.asarray(x).reshape(-1)
            return np.linspace(0.0, 1.0, len(x))

        def confidence_intervals(self, x, width=0.95):
            x = np.asarray(x).reshape(-1)
            lower = np.zeros(len(x), dtype=float)
            upper = np.ones(len(x), dtype=float)
            return np.column_stack([lower, upper])

    monkeypatch.setattr(
        dynamic_features_mod,
        "_require_pygam",
        lambda: {
            ("normal", "identity"): FakeGAM,
            "s": lambda *args, **kwargs: ("term", args, kwargs),
        },
    )

    dynamic_features_mod.dynamic_features(
        adata,
        genes=["gene_a", "missing_gene"],
        pseudotime="pseudotime",
        min_cells=2,
        grid_size=5,
    )

    output_path = tmp_path / "dynamic_features.h5ad"
    adata.write_h5ad(output_path)
    assert output_path.exists()


def test_dynamic_features_can_split_single_adata_by_group(monkeypatch):
    adata = AnnData(np.array([[1.0], [2.0], [4.0], [5.0]]))
    adata.var_names = pd.Index(["gene_a"])
    adata.obs["pseudotime"] = [0.0, 1.0, 0.0, 1.0]
    adata.obs["cell_type"] = ["TypeA", "TypeA", "TypeB", "TypeB"]

    fit_calls = []

    class FakeGAM:
        statistics_ = {}

        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self, x, y, weights=None):
            fit_calls.append((np.asarray(x).reshape(-1).tolist(), np.asarray(y).reshape(-1).tolist()))
            return self

        def predict(self, x):
            x = np.asarray(x).reshape(-1)
            return np.linspace(0.0, 1.0, len(x))

        def confidence_intervals(self, x, width=0.95):
            x = np.asarray(x).reshape(-1)
            lower = np.zeros(len(x), dtype=float)
            upper = np.ones(len(x), dtype=float)
            return np.column_stack([lower, upper])

    monkeypatch.setattr(
        dynamic_features_mod,
        "_require_pygam",
        lambda: {
            ("normal", "identity"): FakeGAM,
            "s": lambda *args, **kwargs: ("term", args, kwargs),
        },
    )

    result = dynamic_features_mod.dynamic_features(
        adata,
        genes=["gene_a"],
        pseudotime="pseudotime",
        groupby="cell_type",
        min_cells=2,
        grid_size=5,
    )

    assert len(fit_calls) == 2
    assert set(result.stats["dataset"]) == {"TypeA", "TypeB"}
    assert "source_dataset" not in result.stats.columns
    assert set(result.stats["groupby_key"]) == {"cell_type"}
    assert set(result.stats["group"]) == {"TypeA", "TypeB"}
