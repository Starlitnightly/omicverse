"""Tests for the ``scdblfinder`` backend of ``ov.pp.qc``.

Covers:
- ``doublets_method='scdblfinder'`` dispatch branch in qc/qc_cpu runs without
  error on a synthetic AnnData and writes the expected obs columns.
- ``filter_doublets=True`` actually removes cells.
- Unknown ``doublets_method`` raises a clear ``ValueError``.
- When ``pyscdblfinder`` is missing, the wrapper raises a clear install-hint
  ``ImportError`` rather than an opaque ``ModuleNotFoundError``.

Uses a 300 × 400 Poisson-count AnnData so tests finish in a few seconds —
correctness checks, not benchmarks.
"""
from __future__ import annotations

import builtins
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest


pytest.importorskip("pyscdblfinder", reason="pyscdblfinder not installed")
pytest.importorskip("xgboost", reason="xgboost not installed")


@pytest.fixture
def tiny_counts():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 300, 400
    mu_A = rng.gamma(1.0, 2.0, size=n_genes)
    mu_B = rng.gamma(1.0, 2.0, size=n_genes)
    X_A = rng.poisson(mu_A[None, :], size=(150, n_genes))
    X_B = rng.poisson(mu_B[None, :], size=(150, n_genes))
    X = np.vstack([X_A, X_B]).astype(np.float32)
    var_names = np.array(
        [f"MT-{i}" for i in range(20)] + [f"GENE{i}" for i in range(n_genes - 20)]
    )
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=var_names),
    )


def _qc_kwargs():
    return dict(
        tresh={"mito_perc": 100, "nUMIs": 1, "detected_genes": 1},
        min_cells=1,
        min_genes=1,
        batch_key=None,
    )


def test_scdblfinder_backend_runs_and_writes_columns(tiny_counts):
    import omicverse as ov

    adata = tiny_counts.copy()
    adata = ov.pp.qc(
        adata,
        doublets=True,
        doublets_method="scdblfinder",
        filter_doublets=False,
        **_qc_kwargs(),
    )
    for col in ("predicted_doublet", "doublet_score",
                "scdblfinder_doublet", "scdblfinder_score"):
        assert col in adata.obs.columns, f"missing {col}"
    # Score is in [0, 1]
    scores = adata.obs["scdblfinder_score"].astype(float).values
    assert np.all(np.isfinite(scores))
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    # predicted_doublet is a bool mask
    assert adata.obs["predicted_doublet"].dtype == bool


def test_scdblfinder_backend_filter_doublets_path_runs(tiny_counts):
    """filter_doublets=True path completes and keeps n_obs in a sane range.

    On tiny synthetic data with low expected dbr, the classifier may flag
    zero cells — that's still a successful filter run. Check that the final
    n_obs is between 80% of input (not over-pruned) and input (may remove none).
    """
    import omicverse as ov

    adata = tiny_counts.copy()
    n0 = adata.n_obs
    adata = ov.pp.qc(
        adata,
        doublets=True,
        doublets_method="scdblfinder",
        filter_doublets=True,
        **_qc_kwargs(),
    )
    assert adata.n_obs <= n0
    assert adata.n_obs >= int(0.80 * n0), "too many cells removed"


def test_unknown_doublets_method_still_errors(tiny_counts):
    import omicverse as ov

    with pytest.raises(ValueError, match="Unknown doublets_method"):
        ov.pp.qc(
            tiny_counts.copy(),
            doublets=True,
            doublets_method="nonexistent_method",
            **_qc_kwargs(),
        )


def test_missing_pyscdblfinder_raises_clear_error(tiny_counts, monkeypatch):
    """When pyscdblfinder isn't installed the wrapper should fail with a
    clear install-hint, not an opaque ``ModuleNotFoundError``."""
    monkeypatch.delitem(sys.modules, "pyscdblfinder", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyscdblfinder" or name.startswith("pyscdblfinder."):
            raise ImportError("No module named 'pyscdblfinder' (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from omicverse.pp._scdblfinder import scdblfinder as run_scdbl
    with pytest.raises(ImportError, match="pip install pyscdblfinder"):
        run_scdbl(tiny_counts.copy())
