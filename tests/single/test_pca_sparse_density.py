"""
Tests for PCA sparse-to-dense conversion on CPU (fix for #613).

When scaled data is stored as a high-density sparse matrix, the CPU PCA
path should detect this and convert to dense before calling sklearn,
avoiding the ~100x slowdown from sparse covariance computation.
"""

import time
import numpy as np
import pytest
import scipy.sparse as sp
import anndata as ad

from omicverse.pp._pca import pca as _pca, _sparse_density
from omicverse.utils._memory import HIGH_DENSITY_SPARSE_THRESHOLD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def high_density_sparse_adata():
    """AnnData with 100%-dense data stored as sparse (the bug scenario)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2000, 500)).astype(np.float32)
    return ad.AnnData(X=sp.csr_matrix(X))


@pytest.fixture
def low_density_sparse_adata():
    """AnnData with genuinely sparse data (10% density)."""
    X = sp.random(2000, 500, density=0.1, format="csr",
                  dtype=np.float32, random_state=42)
    X.data[:] = np.random.default_rng(42).standard_normal(X.nnz).astype(np.float32)
    return ad.AnnData(X=X)


@pytest.fixture
def dense_adata():
    """AnnData with dense ndarray."""
    rng = np.random.default_rng(42)
    return ad.AnnData(X=rng.standard_normal((2000, 500)).astype(np.float32))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSparseDensityHelper:
    """Tests for the _sparse_density utility."""

    def test_full_density(self):
        X = sp.csr_matrix(np.ones((10, 5), dtype=np.float32))
        assert _sparse_density(X) == pytest.approx(1.0)

    def test_zero_density(self):
        X = sp.csr_matrix((10, 5), dtype=np.float32)
        assert _sparse_density(X) == pytest.approx(0.0)

    def test_partial_density(self):
        X = sp.eye(100, format="csr", dtype=np.float32)
        assert _sparse_density(X) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Integration tests: high-density sparse → dense conversion
# ---------------------------------------------------------------------------

class TestPCAHighDensitySparse:
    """Verify that high-density sparse matrices are converted to dense on CPU."""

    def test_produces_valid_results(self, high_density_sparse_adata):
        """PCA on high-density sparse should produce valid embeddings."""
        adata = high_density_sparse_adata
        _pca(adata, n_comps=20, use_gpu=False)

        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (2000, 20)
        assert "pca" in adata.uns
        assert "variance_ratio" in adata.uns["pca"]
        assert len(adata.uns["pca"]["variance_ratio"]) == 20
        # Variance ratios should be positive and sum to <= 1
        vr = adata.uns["pca"]["variance_ratio"]
        assert np.all(vr > 0)
        assert np.sum(vr) <= 1.0 + 1e-6

    def test_matches_dense_input(self, high_density_sparse_adata, dense_adata):
        """Sparse (100% dense) and dense inputs should give same results."""
        _pca(high_density_sparse_adata, n_comps=20, use_gpu=False)
        _pca(dense_adata, n_comps=20, use_gpu=False)

        vr_sparse = high_density_sparse_adata.uns["pca"]["variance_ratio"]
        vr_dense = dense_adata.uns["pca"]["variance_ratio"]
        np.testing.assert_allclose(vr_sparse, vr_dense, rtol=1e-3)

        # PCA coordinates (compare absolute values to handle sign flips)
        pca_sparse = np.abs(high_density_sparse_adata.obsm["X_pca"])
        pca_dense = np.abs(dense_adata.obsm["X_pca"])
        np.testing.assert_allclose(pca_sparse, pca_dense, rtol=1e-2, atol=1e-3)

    @pytest.mark.slow
    def test_performance_not_degraded(self, high_density_sparse_adata, dense_adata):
        """High-density sparse PCA should not be drastically slower than dense."""
        t0 = time.perf_counter()
        _pca(dense_adata, n_comps=20, use_gpu=False)
        t_dense = time.perf_counter() - t0

        t0 = time.perf_counter()
        _pca(high_density_sparse_adata, n_comps=20, use_gpu=False)
        t_sparse = time.perf_counter() - t0

        ratio = t_sparse / max(t_dense, 0.01)
        # After fix, sparse should be within 5x of dense (before fix: 100x+)
        assert ratio < 5, (
            f"High-density sparse PCA is {ratio:.1f}x slower than dense. "
            f"Expected <5x. sparse={t_sparse:.2f}s, dense={t_dense:.2f}s"
        )


# ---------------------------------------------------------------------------
# Integration tests: low-density sparse stays sparse
# ---------------------------------------------------------------------------

class TestPCALowDensitySparse:
    """Verify that low-density sparse matrices are handled correctly."""

    def test_produces_valid_results(self, low_density_sparse_adata):
        """PCA on genuinely sparse data should still work."""
        adata = low_density_sparse_adata
        _pca(adata, n_comps=20, use_gpu=False)

        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (2000, 20)

    def test_density_below_threshold(self, low_density_sparse_adata):
        """Confirm test data is actually below the conversion threshold."""
        density = _sparse_density(low_density_sparse_adata.X)
        assert density < HIGH_DENSITY_SPARSE_THRESHOLD


# ---------------------------------------------------------------------------
# Issue #615: torch_pca sparse + covariance_eigh should not crash
# ---------------------------------------------------------------------------

class TestTorchPCASparseCovarEigh:
    """Verify torch_pca falls back to lobpcg instead of raising ValueError."""

    def test_covariance_eigh_with_sparse_succeeds(self):
        """covariance_eigh + sparse torch tensor should work natively (#615)."""
        torch = pytest.importorskip("torch")
        from omicverse.external.torch_pca import PCA

        # Use a torch sparse tensor directly to bypass scipy auto-densification
        rng = np.random.default_rng(42)
        X = sp.random(500, 100, density=0.1, format="csr", dtype=np.float32,
                      random_state=42)
        X.data[:] = rng.standard_normal(X.nnz).astype(np.float32)
        coo = X.tocoo()
        indices = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        X_torch = torch.sparse_coo_tensor(indices, values, size=X.shape).coalesce()

        pca = PCA(n_components=10, svd_solver="covariance_eigh")
        result = pca.fit_transform(X_torch)
        assert result.shape == (500, 10)
        # Verify it actually used covariance_eigh, not a fallback
        assert pca.svd_solver_ == "covariance_eigh"

    def test_unsupported_solver_with_sparse_falls_back(self):
        """Unsupported solver + sparse should fall back to covariance_eigh."""
        torch = pytest.importorskip("torch")
        from omicverse.external.torch_pca import PCA

        rng = np.random.default_rng(42)
        X = sp.random(500, 100, density=0.1, format="csr", dtype=np.float32,
                      random_state=42)
        X.data[:] = rng.standard_normal(X.nnz).astype(np.float32)
        coo = X.tocoo()
        indices = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        X_torch = torch.sparse_coo_tensor(indices, values, size=X.shape).coalesce()

        pca = PCA(n_components=10, svd_solver="full")
        with pytest.warns(UserWarning, match="falling back to 'covariance_eigh'"):
            result = pca.fit_transform(X_torch)
        assert result.shape == (500, 10)

    def test_lobpcg_with_sparse_works(self):
        """lobpcg + sparse should work."""
        torch = pytest.importorskip("torch")
        from omicverse.external.torch_pca import PCA

        rng = np.random.default_rng(42)
        X = sp.random(500, 100, density=0.1, format="csr", dtype=np.float32,
                      random_state=42)
        X.data[:] = rng.standard_normal(X.nnz).astype(np.float32)
        pca = PCA(n_components=10, svd_solver="lobpcg")
        result = pca.fit_transform(X)
        assert result.shape == (500, 10)


# ---------------------------------------------------------------------------
# Issue #615: chunked PCA with layer should read from correct layer
# ---------------------------------------------------------------------------

class TestPCALayerCorrectness:
    """Verify that PCA reads from the specified layer, not .X."""

    def test_pca_reads_from_layer(self):
        """When layer is specified, PCA should use that layer's data."""
        rng = np.random.default_rng(42)
        # .X is zeros, layer is random — if PCA reads from .X, variance will be ~0
        n_obs, n_vars = 500, 100
        adata = ad.AnnData(X=sp.csr_matrix((n_obs, n_vars), dtype=np.float32))
        adata.layers["scaled"] = rng.standard_normal(
            (n_obs, n_vars)
        ).astype(np.float32)

        _pca(adata, n_comps=10, layer="scaled", use_gpu=False)

        vr = adata.uns["pca"]["variance_ratio"]
        # If layer was read correctly, explained variance should be meaningful
        assert np.sum(vr) > 0.01, (
            f"Variance ratio sum {np.sum(vr):.6f} is near zero — "
            "PCA may be reading from .X instead of the specified layer"
        )
