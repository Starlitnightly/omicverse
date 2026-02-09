import numpy as np
import warnings


def init_pca(X, no_dims, contri, use_gpu=True, verbose=False):
    """
    This function preprocesses data with excessive size and dimensions.
    Now supports GPU acceleration via MLX (Apple Silicon) or torch_pca (CUDA).

    Parameters
    ----------
    X : array-like
        N by D matrix. Each row in X represents an observation.
    no_dims : int
        A positive integer specifying the number of dimension of the representation Y.
    contri : float
        Threshold of PCA variance contribution.
    use_gpu : bool, optional
        Whether to use GPU acceleration when available. Default: True.
    verbose : bool, optional
        Whether to print device selection information. Default: False.

    Returns
    -------
    mappedX : array-like
        PCA-transformed data with optimal dimensionality.
    """
    # Detect optimal device and PCA backend
    device_info = _detect_optimal_pca_backend(use_gpu, verbose)

    from ..._settings import settings
    omicverse_mode = getattr(settings, 'mode', 'cpu')

    if device_info['backend'] == 'mlx' and device_info['device'] == 'mps' and omicverse_mode != 'cpu':
        return _init_pca_mlx(X, no_dims, contri, verbose)
    elif device_info['backend'] == 'torch_pca' and device_info['device'] == 'cuda' and omicverse_mode != 'cpu':
        return _init_pca_torch_pca(X, no_dims, contri, verbose)
    else:
        # Fallback to CPU implementation
        return _init_pca_cpu(X, no_dims, contri, verbose)


def _detect_optimal_pca_backend(use_gpu=True, verbose=False):
    """
    Detect the optimal PCA backend based on available hardware and omicverse settings.
    
    Parameters
    ----------
    use_gpu : bool
        Whether to prefer GPU acceleration
    verbose : bool
        Whether to print detection information
        
    Returns
    -------
    dict
        Dictionary containing backend and device information
    """
    # Import omicverse settings
    try:
        from ..._settings import get_optimal_device, settings
        device = get_optimal_device(prefer_gpu=use_gpu, verbose=verbose)
        
        # Check omicverse mode
        omicverse_mode = getattr(settings, 'mode', 'cpu')
        
        if verbose:
            print(f"   Omicverse mode: {omicverse_mode}")
            print(f"   Detected device: {device}")
        
        # Determine backend based on device and omicverse mode
        if hasattr(device, 'type'):
            device_type = device.type
        else:
            device_type = str(device)
            
        if device_type == 'mps' and use_gpu:
            # Try MLX for Apple Silicon
            try:
                import mlx.core as mx
                if mx.metal.is_available():
                    return {'backend': 'mlx', 'device': 'mps', 'available': True}
            except ImportError:
                pass
                
        elif device_type == 'cuda' and use_gpu:
            # Use torch_pca for CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    return {'backend': 'torch_pca', 'device': 'cuda', 'available': True}
            except ImportError:
                pass
        
        # Fallback to CPU
        return {'backend': 'cpu', 'device': 'cpu', 'available': True}
        
    except ImportError:
        # Fallback if omicverse settings not available
        return {'backend': 'cpu', 'device': 'cpu', 'available': True}


def _init_pca_mlx(X, no_dims, contri, verbose=False):
    """
    MLX-based PCA implementation for Apple Silicon MPS devices.
    
    Parameters
    ----------
    X : array-like
        Input data matrix
    no_dims : int
        Number of dimensions
    contri : float
        Variance contribution threshold
    verbose : bool
        Whether to print information
        
    Returns
    -------
    mappedX : array-like
        PCA-transformed data
    """
    if verbose:
        print(f"   🚀 Using MLX PCA for Apple Silicon MPS acceleration")
    
    from ...pp._pca_mlx import MLXPCA
    
    # Convert to numpy if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    
    # Determine optimal number of components
    m = X.shape[1]
    optimal_comps = _determine_optimal_components(X, contri, no_dims, verbose)
    
    # Create MLX PCA instance
    mlx_pca = MLXPCA(n_components=optimal_comps, device="metal")
    
    # Fit and transform
    mappedX = mlx_pca.fit_transform(X)
    
    if verbose:
        print(f"   ✅ MLX PCA completed: {X.shape} -> {mappedX.shape}")
        
    return mappedX
        
    

def _init_pca_torch_pca(X, no_dims, contri, verbose=False):
    """
    torch_pca-based PCA implementation for CUDA devices.

    Parameters
    ----------
    X : array-like
        Input data matrix
    no_dims : int
        Number of dimensions
    contri : float
        Variance contribution threshold
    verbose : bool
        Whether to print information

    Returns
    -------
    mappedX : array-like
        PCA-transformed data
    """
    if verbose:
        print(f"   🚀 Using torch_pca PCA for CUDA acceleration")

    import torch
    from ...external.torch_pca import PCA

    # Determine optimal number of components
    optimal_comps = _determine_optimal_components(X, contri, no_dims, verbose)

    # torch_pca supports sparse matrices natively
    from scipy import sparse
    if sparse.issparse(X):
        # For sparse matrices, use ARPACK solver
        pca = PCA(n_components=optimal_comps, svd_solver='arpack')
        mappedX = pca.fit_transform(X)
        # Convert to numpy if tensor
        if hasattr(mappedX, 'cpu'):
            mappedX = mappedX.cpu().numpy()
    else:
        # For dense arrays, convert to torch tensor on GPU
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')

        # Create torch_pca PCA
        pca = PCA(n_components=optimal_comps, svd_solver='randomized')

        # Fit and transform
        mappedX_tensor = pca.fit_transform(X_tensor)

        # Move PCA model to GPU
        pca.to('cuda')

        # Convert back to numpy
        mappedX = mappedX_tensor.cpu().numpy()

    if verbose:
        print(f"   ✅ torch_pca PCA completed: {X.shape} -> {mappedX.shape}")

    return mappedX
        
    

def _init_pca_cpu(X, no_dims, contri, verbose=False):
    """
    Original CPU-based PCA implementation.
    
    Parameters
    ----------
    X : array-like
        Input data matrix
    no_dims : int
        Number of dimensions
    contri : float
        Variance contribution threshold
    verbose : bool
        Whether to print information
        
    Returns
    -------
    mappedX : array-like
        PCA-transformed data
    """
    if verbose:
        print(f"   🖥️ Using CPU PCA implementation")
    
    # Convert to numpy if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    
    m = X.shape[1]
    X = X - np.mean(X, axis=0)
    
    # Compute covariance matrix C
    C = np.cov(X, rowvar=False)
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    
    lamda, M = np.linalg.eig(C)
    lamda = np.real(lamda)
    
    # Obtain the best PCA dimension
    if m < 2001:
        ind = np.where(np.cumsum(lamda) / sum(lamda) > contri)
    else:
        ind = np.where(np.cumsum(lamda) / sum(lamda[:2000]) > contri)
    bestDim = max(no_dims + 1, int(ind[0][0]))
    
    # Apply mapping on the data
    mappedX = X @ np.real(M)[:, :bestDim]
    
    if verbose:
        print(f"   ✅ CPU PCA completed: {X.shape} -> {mappedX.shape}")
    
    return mappedX


def _determine_optimal_components(X, contri, no_dims, verbose=False):
    """
    Determine the optimal number of PCA components based on variance contribution.
    
    Parameters
    ----------
    X : array-like
        Input data matrix
    contri : float
        Variance contribution threshold
    no_dims : int
        Minimum number of dimensions
    verbose : bool
        Whether to print information
        
    Returns
    -------
    int
        Optimal number of components
    """
    m = X.shape[1]
    
    # For small matrices, compute full eigendecomposition
    if m < 2001:
        X_centered = X - np.mean(X, axis=0)
        C = np.cov(X_centered, rowvar=False)
        C[np.isnan(C)] = 0
        C[np.isinf(C)] = 0
        lamda, _ = np.linalg.eig(C)
        lamda = np.real(lamda)
        
        # Find components needed for variance threshold
        cumsum_ratio = np.cumsum(lamda) / sum(lamda)
        ind = np.where(cumsum_ratio > contri)
        optimal_comps = max(no_dims + 1, int(ind[0][0]))
    else:
        # For large matrices, use a heuristic
        optimal_comps = max(no_dims + 1, min(50, m // 4))
    
    if verbose:
        print(f"   📊 Optimal components determined: {optimal_comps}")
    
    return optimal_comps
