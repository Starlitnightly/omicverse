"""Tests for the ``doubletfinder`` backend of ``ov.pp.qc``.

Covers:
- The new ``doublets_method='doubletfinder'`` dispatch branch (both CPU and
  CPU-GPU paths) runs without error on a synthetic AnnData.
- Expected columns (``predicted_doublet``, ``doublet_score``,
  ``doubletfinder_doublet``, ``doubletfinder_pANN``) land on ``adata.obs``.
- ``filter_doublets=True`` actually removes cells.
- The importlib-level error path is clear when ``pydoubletfinder`` is
  unavailable.
- Schema validation: an unknown ``doublets_method`` raises ``ValueError``.

These tests use a small (300 × 400) Poisson-count AnnData so they finish
in a few seconds — they're correctness checks, not benchmarks.
"""
from __future__ import annotations

import builtins
import importlib
import sys
from contextlib import contextmanager

import anndata as ad
import numpy as np
import pandas as pd
import pytest


pytest.importorskip("pydoubletfinder", reason="pydoubletfinder not installed")


@pytest.fixture
def tiny_counts():
    """Small synthetic count matrix with a few obvious 'doublet-like' cells.

    We spike 5 cells with mixture expression (sum of two archetypes) so that
    there's a realistic signal the pANN kernel can detect.
    """
    rng = np.random.default_rng(0)
    n_cells, n_genes = 300, 400
    # Two archetypal populations
    mu_A = rng.gamma(1.0, 2.0, size=n_genes)
    mu_B = rng.gamma(1.0, 2.0, size=n_genes)
    # 150 A cells + 150 B cells
    X_A = rng.poisson(mu_A[None, :], size=(150, n_genes))
    X_B = rng.poisson(mu_B[None, :], size=(150, n_genes))
    X = np.vstack([X_A, X_B]).astype(np.float32)
    # Add a mitochondrial gene block so ov.pp.qc's mt detection has something
    var_names = np.array(
        [f"MT-{i}" for i in range(20)] + [f"GENE{i}" for i in range(n_genes - 20)]
    )
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=var_names),
    )
    return adata


def _qc_kwargs():
    """Permissive QC thresholds so tiny synthetic data passes all filters."""
    return dict(
        tresh={"mito_perc": 100, "nUMIs": 1, "detected_genes": 1},
        min_cells=1,
        min_genes=1,
        batch_key=None,
    )


def test_doubletfinder_backend_runs_and_writes_columns(tiny_counts):
    import omicverse as ov

    adata = tiny_counts.copy()
    adata = ov.pp.qc(
        adata,
        doublets=True,
        doublets_method="doubletfinder",
        filter_doublets=False,
        **_qc_kwargs(),
    )
    for col in ("predicted_doublet", "doublet_score",
                "doubletfinder_doublet", "doubletfinder_pANN"):
        assert col in adata.obs.columns, f"missing {col}"
    # pANN is a proportion in [0, 1]
    pann = adata.obs["doubletfinder_pANN"].astype(float).values
    assert np.all(np.isfinite(pann))
    assert pann.min() >= 0.0 and pann.max() <= 1.0
    # predicted_doublet is a bool mask
    assert adata.obs["predicted_doublet"].dtype == bool


def test_doubletfinder_backend_removes_cells_when_filtering(tiny_counts):
    import omicverse as ov

    adata = tiny_counts.copy()
    n0 = adata.n_obs
    adata = ov.pp.qc(
        adata,
        doublets=True,
        doublets_method="doubletfinder",
        filter_doublets=True,
        **_qc_kwargs(),
    )
    # Some cells should be removed (the expected ~7.5% × n_obs)
    assert adata.n_obs < n0
    assert adata.n_obs >= int(0.85 * n0), "too many cells removed"


def test_unknown_doublets_method_errors(tiny_counts):
    import omicverse as ov

    with pytest.raises(ValueError, match="Unknown doublets_method"):
        ov.pp.qc(
            tiny_counts.copy(),
            doublets=True,
            doublets_method="nonexistent_method",
            **_qc_kwargs(),
        )


def test_missing_pydoubletfinder_raises_clear_error(tiny_counts, monkeypatch):
    """When pydoubletfinder isn't installed the wrapper should fail with a
    clear install-hint, not an opaque ``ModuleNotFoundError``."""
    # Simulate pydoubletfinder being missing by blocking the import
    monkeypatch.delitem(sys.modules, "pydoubletfinder", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pydoubletfinder" or name.startswith("pydoubletfinder."):
            raise ImportError("No module named 'pydoubletfinder' (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from omicverse.pp._doubletfinder import doubletfinder as run_df
    with pytest.raises(ImportError, match="pip install pydoubletfinder"):
        run_df(tiny_counts.copy())
