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


def _bootstrap_omicverse_packages():
    saved = {
        name: sys.modules.get(name)
        for name in [
            "omicverse",
            "omicverse.external",
            "omicverse.external.palantir",
        ]
    }
    for name in saved:
        sys.modules.pop(name, None)

    ov_pkg = types.ModuleType("omicverse")
    ov_pkg.__path__ = [str(PACKAGE_ROOT)]
    ov_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse", loader=None, is_package=True)
    sys.modules["omicverse"] = ov_pkg

    external_pkg = types.ModuleType("omicverse.external")
    external_pkg.__path__ = [str(PACKAGE_ROOT / "external")]
    external_pkg.__spec__ = importlib.machinery.ModuleSpec("omicverse.external", loader=None, is_package=True)
    sys.modules["omicverse.external"] = external_pkg
    ov_pkg.external = external_pkg

    pal_pkg = types.ModuleType("omicverse.external.palantir")
    pal_pkg.__path__ = [str(PACKAGE_ROOT / "external" / "palantir")]
    pal_pkg.__spec__ = importlib.machinery.ModuleSpec(
        "omicverse.external.palantir",
        loader=None,
        is_package=True,
    )
    sys.modules["omicverse.external.palantir"] = pal_pkg
    external_pkg.palantir = pal_pkg

    return saved


_SAVED = _bootstrap_omicverse_packages()
core = importlib.import_module("omicverse.external.palantir.core")
presults = importlib.import_module("omicverse.external.palantir.presults")

for name, mod in _SAVED.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


def test_run_palantir_preserves_dataframe_obsm_when_requested(monkeypatch):
    adata = AnnData(np.zeros((3, 1)))
    adata.obs_names = ["c0", "c1", "c2"]
    adata.obsm["DM_EigenVectors_multiscaled"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )

    monkeypatch.setattr(core, "_max_min_sampling", lambda data_df, num_waypoints, seed: pd.Index(["c1", "c2"]))
    monkeypatch.setattr(
        core,
        "_compute_pseudotime",
        lambda data_df, start_cell, knn, waypoints, n_jobs, max_iterations: (
            pd.Series([0.0, 1.0, 2.0], index=data_df.index),
            pd.DataFrame(
                np.eye(len(data_df.index)),
                index=data_df.index,
                columns=data_df.index,
            ),
        ),
    )
    monkeypatch.setattr(
        core,
        "_differentiation_entropy",
        lambda data_df, terminal_cells, knn, n_jobs, pseudotime: (
            pd.Series([0.0, 0.0, 0.0], index=data_df.index),
            pd.DataFrame(
                {
                    "branch_a": [0.8, 0.6, 0.2],
                    "branch_b": [0.2, 0.4, 0.8],
                },
                index=data_df.index,
            ),
        ),
    )

    core.run_palantir(
        adata,
        early_cell="c0",
        num_waypoints=2,
        save_as_df=True,
        scale_components=False,
    )

    stored = adata.obsm["palantir_fate_probabilities"]
    assert isinstance(stored, pd.DataFrame)
    assert list(stored.columns) == ["branch_a", "branch_b"]
    assert list(stored.index) == list(adata.obs_names)


def test_select_branch_cells_preserves_dataframe_masks_when_requested():
    adata = AnnData(np.zeros((4, 1)))
    adata.obs_names = ["c0", "c1", "c2", "c3"]
    adata.obs["palantir_pseudotime"] = [0.0, 1.0, 2.0, 3.0]
    adata.obsm["palantir_fate_probabilities"] = pd.DataFrame(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        index=adata.obs_names,
        columns=["branch_a", "branch_b"],
    )

    masks = presults.select_branch_cells(adata, save_as_df=True)

    assert isinstance(masks, np.ndarray)
    stored = adata.obsm["branch_masks"]
    assert isinstance(stored, pd.DataFrame)
    assert list(stored.columns) == ["branch_a", "branch_b"]
    assert list(stored.index) == list(adata.obs_names)
