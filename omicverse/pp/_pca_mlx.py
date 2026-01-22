"""
MLX-based PCA implementation for Apple Silicon GPU acceleration.
Automatically detects MPS device and uses MLX for computation when available.
Falls back to sklearn when MLX is not available or device is not MPS.

This module is designed to be integrated with omicverse's PCA pipeline.
"""

import numpy as np
import warnings
from typing import Optional, Union, Tuple

# Try to import MLX, fall back gracefully if not available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

# Always import sklearn as fallback
try:
    from sklearn.decomposition import PCA as SklearnPCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SklearnPCA = None


def _detect_device() -> str:
    """Detect the best available device for computation."""
    if not MLX_AVAILABLE:
        return "cpu"

    # MLX automatically uses Metal GPU on Apple Silicon
    # No need for explicit device specification
    try:
        # Test if we can create and use an array
        test_array = mx.array([1.0])
        _ = test_array + 1.0
        return "metal"
    except Exception:
        return "cpu"


class MLXPCA:
    """
    MLX-based PCA implementation optimized for Apple Silicon GPUs.
    
    This class automatically detects the best available device and uses MLX
    for computation when MPS (Apple Silicon GPU) is available, otherwise
    falls back to sklearn.
    
    Parameters
    ----------
    n_components : int, float, None or str, default=None
        Number of components to keep. If n_components is not set then all 
        components are kept.
    device : str, optional
        Device to use for computation ('auto', 'mps', 'cpu'). 
        If 'auto', automatically detects the best device.
    """
    
    def __init__(self, n_components: Optional[Union[int, float, str]] = None,
                 device: str = "auto"):
        self.n_components = n_components

        # Normalize device parameter
        if device == "auto":
            self.device = _detect_device()
        elif device in ["mps", "metal"]:
            # Both mps and metal refer to Apple Silicon GPU
            self.device = "metal" if MLX_AVAILABLE else "cpu"
        else:
            self.device = device

        # Initialize attributes
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_ = None
        self.n_samples_ = None

        # Determine computation backend
        # Use MLX if available and device is metal/cpu
        self._use_mlx = MLX_AVAILABLE and self.device in ["metal", "cpu"]
        self._use_sklearn = not self._use_mlx

        if self._use_sklearn and not SKLEARN_AVAILABLE:
            raise ImportError("Neither MLX nor sklearn is available. Please install at least one.")
    
    def _to_mlx_array(self, X: np.ndarray) -> 'mx.array':
        """Convert numpy array to MLX array on the specified device."""
        if not self._use_mlx:
            raise RuntimeError("MLX not available")
        
        # MLX automatically uses the default device (Metal GPU on Apple Silicon)
        # No need to explicitly set device for individual arrays
        return mx.array(X)
    
    def _from_mlx_array(self, X: 'mx.array') -> np.ndarray:
        """Convert MLX array back to numpy array."""
        return np.array(X)
    
    def _mlx_svd(self, X: 'mx.array', n_components: int) -> Tuple['mx.array', 'mx.array', 'mx.array']:
        """Perform SVD using MLX.

        Note: MLX's svd requires CPU stream for now. While basic operations
        can run on GPU, linalg operations need explicit CPU stream.
        """
        # Center the data (can run on GPU)
        self.mean_ = mx.mean(X, axis=0)
        X_centered = X - self.mean_

        # Use MLX SVD with CPU stream (required for linalg operations)
        U, S, Vt = mx.linalg.svd(X_centered, stream=mx.cpu)
        # Ensure computation is complete
        mx.eval(U, S, Vt)

        # Select the first n_components
        U = U[:, :n_components]
        S = S[:n_components]
        Vt = Vt[:n_components, :]

        return U, S, Vt
    
    def _mlx_eigh(self, X: 'mx.array', n_components: int) -> Tuple['mx.array', 'mx.array']:
        """Perform eigenvalue decomposition using MLX.

        Note: MLX linalg operations require CPU stream. The computation still
        benefits from MLX's optimized CPU kernels and unified memory architecture.
        """
        # Center the data (can run on default stream)
        self.mean_ = mx.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix (can run on default stream)
        cov_matrix = mx.matmul(X_centered.T, X_centered) / (X.shape[0] - 1)

        # Use SVD for symmetric matrix decomposition (requires CPU stream)
        # For symmetric positive semidefinite matrix: cov = U @ diag(S) @ U^T
        # So eigenvalues = S, eigenvectors = U
        U, S, Vt = mx.linalg.svd(cov_matrix, stream=mx.cpu)
        # Ensure computation is complete
        mx.eval(U, S, Vt)

        eigenvalues = S
        eigenvectors = U

        # Sort in descending order (eigenvalues from SVD are already sorted)
        # But we keep this for consistency
        idx = mx.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select the first n_components
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

        return eigenvalues, eigenvectors
    
    def fit(self, X: np.ndarray) -> 'MLXPCA':
        """
        Fit the PCA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be dense or sparse array.

        Returns
        -------
        self : MLXPCA
            Returns the instance itself.
        """
        # Handle sparse matrices
        from scipy import sparse
        if sparse.issparse(X):
            X = X.toarray()
        else:
            X = np.asarray(X)

        self.n_samples_, self.n_features_ = X.shape
        
        # Determine number of components
        if self.n_components is None:
            n_components = min(self.n_samples_, self.n_features_)
        elif isinstance(self.n_components, float):
            n_components = int(self.n_components * self.n_features_)
        else:
            n_components = min(self.n_components, self.n_features_)
        
        if self._use_mlx:
            try:
                # Convert to MLX array
                X_mlx = self._to_mlx_array(X)
                
                # Choose method based on matrix dimensions
                if self.n_features_ > self.n_samples_:
                    # Wide matrix: use SVD
                    U, S, Vt = self._mlx_svd(X_mlx, n_components)
                    self.components_ = self._from_mlx_array(Vt)
                    self.explained_variance_ = self._from_mlx_array(S ** 2 / (self.n_samples_ - 1))
                else:
                    # Tall matrix: use eigenvalue decomposition
                    eigenvalues, eigenvectors = self._mlx_eigh(X_mlx, n_components)
                    self.components_ = self._from_mlx_array(eigenvectors.T)
                    self.explained_variance_ = self._from_mlx_array(eigenvalues)
                
                # Convert mean back to numpy
                self.mean_ = self._from_mlx_array(self.mean_)
                
            except Exception as e:
                warnings.warn(f"MLX computation failed, falling back to sklearn: {e}")
                self._use_sklearn = True
        
        if self._use_sklearn:
            # Fallback to sklearn
            sklearn_pca = SklearnPCA(n_components=n_components)
            sklearn_pca.fit(X)
            
            self.components_ = sklearn_pca.components_
            self.explained_variance_ = sklearn_pca.explained_variance_
            self.explained_variance_ratio_ = sklearn_pca.explained_variance_ratio_
            self.mean_ = sklearn_pca.mean_
        
        # Calculate explained variance ratio
        if self.explained_variance_ratio_ is None:
            total_variance = np.sum(self.explained_variance_)
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_new : array of shape (n_samples, n_components)
            Transformed data.
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transform")
        
        X = np.asarray(X)
        X_centered = X - self.mean_
        
        if self._use_mlx and not self._use_sklearn:
            try:
                X_mlx = self._to_mlx_array(X_centered)
                components_mlx = self._to_mlx_array(self.components_)
                result = mx.matmul(X_mlx, components_mlx.T)
                return self._from_mlx_array(result)
            except Exception as e:
                warnings.warn(f"MLX transform failed, falling back to numpy: {e}")
        
        # Fallback to numpy computation
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        X_new : array of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data back to its original space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data to transform back.
            
        Returns
        -------
        X_original : array of shape (n_samples, n_features)
            Data in original space.
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before inverse_transform")
        
        X = np.asarray(X)
        
        if self._use_mlx and not self._use_sklearn:
            try:
                X_mlx = self._to_mlx_array(X)
                components_mlx = self._to_mlx_array(self.components_)
                mean_mlx = self._to_mlx_array(self.mean_)
                result = mx.matmul(X_mlx, components_mlx) + mean_mlx
                return self._from_mlx_array(result)
            except Exception as e:
                warnings.warn(f"MLX inverse transform failed, falling back to numpy: {e}")
        
        # Fallback to numpy computation
        return np.dot(X, self.components_) + self.mean_


class MockPCA:
    """
    Mock PCA class that wraps MLXPCA to provide sklearn-compatible interface.
    
    This class provides the same interface as sklearn's PCA class but uses
    MLXPCA internally for computation.
    """
    def __init__(self, mlx_pca: MLXPCA):
        self.components_ = mlx_pca.components_
        self.explained_variance_ = mlx_pca.explained_variance_
        self.explained_variance_ratio_ = mlx_pca.explained_variance_ratio_
        self.mean_ = mlx_pca.mean_
        self.n_features_ = mlx_pca.n_features_
        self.n_samples_ = mlx_pca.n_samples_
        self._mlx_pca = mlx_pca
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X."""
        return self._mlx_pca.fit_transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X."""
        return self._mlx_pca.transform(X)
    
    def fit(self, X: np.ndarray) -> 'MockPCA':
        """Fit the PCA model."""
        self._mlx_pca.fit(X)
        return self


# Convenience function for easy import
def create_pca(n_components: Optional[Union[int, float, str]] = None, 
               device: str = "auto") -> MLXPCA:
    """
    Create a PCA instance with automatic device detection.
    
    Parameters
    ----------
    n_components : int, float, None or str, default=None
        Number of components to keep.
    device : str, optional
        Device to use ('auto', 'mps', 'cpu').
        
    Returns
    -------
    MLXPCA instance
    """
    return MLXPCA(n_components=n_components, device=device)


# Compatibility function for omicverse integration
def pca_mlx(
    data,
    n_comps: Optional[int] = None,
    zero_center: bool = True,
    random_state = None,
    return_info: bool = False,
    dtype = "float32",
    copy: bool = False,
    **kwargs
):
    """
    MLX-based PCA function compatible with omicverse's PCA interface.
    
    This function provides a drop-in replacement for sklearn's PCA when using
    Apple Silicon devices with MLX acceleration.
    
    Parameters
    ----------
    data : array-like
        Input data matrix
    n_comps : int, optional
        Number of components to compute
    zero_center : bool, default=True
        Whether to zero-center the data
    random_state : int, optional
        Random state for reproducibility
    return_info : bool, default=False
        Whether to return additional information
    dtype : str, default="float32"
        Data type for computation
    copy : bool, default=False
        Whether to copy the input data
    **kwargs
        Additional keyword arguments (ignored for compatibility)
        
    Returns
    -------
    tuple or array
        If return_info=True, returns (X_pca, components, variance_ratio, variance)
        Otherwise returns X_pca
    """
    # Convert data to numpy array if needed
    if hasattr(data, 'X'):
        X = data.X
    else:
        X = np.asarray(data)
    
    # Determine number of components
    if n_comps is None:
        n_comps = min(X.shape[0], X.shape[1])
    
    # Create MLX PCA instance
    mlx_pca = MLXPCA(n_components=n_comps, device="auto")
    
    # Fit and transform
    X_pca = mlx_pca.fit_transform(X)
    
    # Convert to specified dtype
    if X_pca.dtype != np.dtype(dtype):
        X_pca = X_pca.astype(dtype)
    
    if return_info:
        return (
            X_pca,
            mlx_pca.components_,
            mlx_pca.explained_variance_ratio_,
            mlx_pca.explained_variance_,
        )
    else:
        return X_pca


# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import time
    
    print("MLX PCA Implementation Test")
    print("=" * 50)
    
    # Generate test data
    X, _ = make_classification(n_samples=1000, n_features=50, n_informative=10, random_state=42)
    
    # Test MLX PCA
    print(f"Device detection: {_detect_device()}")
    print(f"MLX available: {MLX_AVAILABLE}")
    print(f"Sklearn available: {SKLEARN_AVAILABLE}")
    
    # Test MLX PCA
    start_time = time.time()
    pca_mlx_instance = MLXPCA(n_components=10)
    X_transformed = pca_mlx_instance.fit_transform(X)
    mlx_time = time.time() - start_time
    
    print(f"\nMLX PCA Results:")
    print(f"Time: {mlx_time:.4f} seconds")
    print(f"Shape: {X_transformed.shape}")
    print(f"Explained variance ratio: {pca_mlx_instance.explained_variance_ratio_[:5]}")
    
    # Compare with sklearn
    if SKLEARN_AVAILABLE:
        start_time = time.time()
        pca_sklearn = SklearnPCA(n_components=10)
        X_sklearn = pca_sklearn.fit_transform(X)
        sklearn_time = time.time() - start_time
        
        print(f"\nSklearn PCA Results:")
        print(f"Time: {sklearn_time:.4f} seconds")
        print(f"Shape: {X_sklearn.shape}")
        print(f"Explained variance ratio: {pca_sklearn.explained_variance_ratio_[:5]}")
        
        # Compare results
        correlation = np.corrcoef(X_transformed.flatten(), X_sklearn.flatten())[0, 1]
        print(f"\nCorrelation between results: {correlation:.6f}")
        
        if mlx_time > 0:
            speedup = sklearn_time / mlx_time
            print(f"Speedup: {speedup:.2f}x")
