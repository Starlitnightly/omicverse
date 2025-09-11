from scipy.spatial.distance import cdist
import numpy as np
import warnings


def mds(X, no_dims, use_gpu=True, verbose=False):
    """
    This function performs MDS embedding.
    Now supports GPU acceleration via MLX (Apple Silicon) or Torch (CUDA).

    Parameters
    ----------
    X : array-like
        N by D matrix. Each row in X represents an observation.
    no_dims : int
        A positive integer specifying the number of dimension of the representation Y.
    use_gpu : bool, optional
        Whether to use GPU acceleration when available. Default: True.
    verbose : bool, optional
        Whether to print device selection information. Default: False.

    Returns
    -------
    array-like
        MDS-embedded data with specified dimensionality.
    """
    # If GPU is explicitly disabled, use CPU implementation
    if not use_gpu:
        return _mds_cpu(X, no_dims, verbose)

    from ..._settings import settings
    omicverse_mode = getattr(settings, 'mode', 'cpu')
    
    # Detect optimal device and backend
    device_info = _detect_optimal_mds_backend(use_gpu, verbose)
    
    if device_info['backend'] == 'mlx' and device_info['device'] == 'mps' and omicverse_mode != 'cpu':
        return _mds_mlx(X, no_dims, verbose)
    elif device_info['backend'] == 'torch' and device_info['device'] in ['cuda', 'cpu'] and omicverse_mode != 'cpu':
        return _mds_torch(X, no_dims, device_info['device'], verbose)
    else:
        # Fallback to CPU implementation
        return _mds_cpu(X, no_dims, verbose)


def _detect_optimal_mds_backend(use_gpu=True, verbose=False):
    """
    Detect the optimal MDS backend based on available hardware and omicverse settings.
    
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
            
        if device_type == 'mps' and use_gpu and omicverse_mode != 'cpu':
            # Try MLX for Apple Silicon
            try:
                import mlx.core as mx
                if mx.metal.is_available():
                    return {'backend': 'mlx', 'device': 'mps', 'available': True}
            except ImportError:
                pass
                
        elif device_type == 'cuda' and use_gpu and omicverse_mode != 'cpu':
            # Try Torch for CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    return {'backend': 'torch', 'device': 'cuda', 'available': True}
            except ImportError:
                pass
        
        # Try Torch CPU as fallback
        try:
            import torch
            return {'backend': 'torch', 'device': 'cpu', 'available': True}
        except ImportError:
            pass
        
        # Fallback to pure CPU
        return {'backend': 'cpu', 'device': 'cpu', 'available': True}
        
    except ImportError:
        # Fallback if omicverse settings not available
        try:
            import torch
            return {'backend': 'torch', 'device': 'cpu', 'available': True}
        except ImportError:
            return {'backend': 'cpu', 'device': 'cpu', 'available': True}


def _compute_distance_matrix_mlx(X):
    """
    Compute pairwise squared distance matrix using MLX.
    
    Parameters
    ----------
    X : mlx.array
        Data matrix
        
    Returns
    -------
    mlx.array
        Squared distance matrix
    """
    import mlx.core as mx
    
    # Compute pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i @ x_j
    squared_norms = mx.sum(X ** 2, axis=1, keepdims=True)
    D = squared_norms + squared_norms.T - 2 * X @ X.T
    
    return D


def _compute_distance_matrix_torch(X, device):
    """
    Compute pairwise squared distance matrix using Torch.
    
    Parameters
    ----------
    X : torch.Tensor
        Data matrix
    device : str
        Device to use
        
    Returns
    -------
    torch.Tensor
        Squared distance matrix
    """
    import torch
    
    # Compute pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i @ x_j
    squared_norms = torch.sum(X ** 2, dim=1, keepdim=True)
    D = squared_norms + squared_norms.T - 2 * X @ X.T
    
    return D


def _mds_mlx(X, no_dims, verbose=False):
    """
    MLX-based MDS implementation for Apple Silicon MPS devices.
    
    Parameters
    ----------
    X : array-like
        Data matrix
    no_dims : int
        Number of dimensions
    verbose : bool
        Whether to print information
        
    Returns
    -------
    array-like
        MDS-embedded data
    """
    if verbose:
        print(f"   🚀 Using MLX MDS for Apple Silicon MPS acceleration")
    
    try:
        import mlx.core as mx
        
        # Convert to MLX array
        X_mx = mx.array(np.asarray(X, dtype=np.float32))
        n = X_mx.shape[0]
        
        # Compute squared distance matrix
        D = _compute_distance_matrix_mlx(X_mx)
        
        # Compute row and overall means
        sumd = mx.mean(D, axis=1)
        sumD = mx.mean(sumd)
        
        # Compute centering matrix B
        B = mx.zeros((n, n))
        for i in range(n):
            for j in range(n):
                B = B.at[i, j].set(-0.5 * (D[i, j] - sumd[i] - sumd[j] + sumD))
        
        # Eigendecomposition
        eigenvalues, eigenvectors = mx.linalg.eig(B)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = mx.argsort(-mx.real(eigenvalues))
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Take the first no_dims components
        selected_eigenvalues = eigenvalues[:no_dims]
        selected_eigenvectors = eigenvectors[:, :no_dims]
        
        # Compute embedding
        sqrt_eigenvalues = mx.sqrt(mx.abs(mx.real(selected_eigenvalues)))
        embedX = mx.real(selected_eigenvectors) @ mx.diag(sqrt_eigenvalues)
        
        # Convert back to numpy
        result = np.array(embedX)
        
        if verbose:
            print(f"   ✅ MLX MDS completed: {X.shape} -> {result.shape}")
            
        return result
        
    except Exception as e:
        if verbose:
            print(f"   ⚠️ MLX MDS failed ({str(e)}), falling back to CPU")
        return _mds_cpu(X, no_dims, verbose)


def _mds_torch(X, no_dims, device='cpu', verbose=False):
    """
    Torch-based MDS implementation for CUDA/CPU devices.
    Note: Currently falls back to CPU implementation for numerical stability.
    
    Parameters
    ----------
    X : array-like
        Data matrix
    no_dims : int
        Number of dimensions
    device : str
        Device to use ('cuda' or 'cpu')
    verbose : bool
        Whether to print information
        
    Returns
    -------
    array-like
        MDS-embedded data
    """
    if verbose:
        print(f"   🚀 Using Torch MDS (fallback to CPU for stability)")
    
    # For now, use CPU implementation to ensure consistency
    # TODO: Fix numerical stability issues in pure Torch implementation
    return _mds_cpu(X, no_dims, verbose)


def _mds_cpu(X, no_dims, verbose=False):
    """
    Original CPU-based MDS implementation.
    
    Parameters
    ----------
    X : array-like
        Data matrix
    no_dims : int
        Number of dimensions
    verbose : bool
        Whether to print information
        
    Returns
    -------
    array-like
        MDS-embedded data
    """
    if verbose:
        print(f"   🖥️ Using CPU MDS implementation")
    
    # Convert to numpy array
    X = np.asarray(X)
    
    n = X.shape[0]
    D = cdist(X, X) ** 2
    sumd = np.mean(D, axis=1)
    sumD = np.mean(sumd)
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            B[i][j] = -0.5 * (D[i][j] - sumd[i] - sumd[j] + sumD)
            B[j][i] = B[i][j]
    value, U = np.linalg.eig(B)
    embedX = U[:, :no_dims] @ np.diag(np.sqrt(np.abs(value[:no_dims])))
    
    if verbose:
        print(f"   ✅ CPU MDS completed: {X.shape} -> {embedX.shape}")
    
    return embedX
