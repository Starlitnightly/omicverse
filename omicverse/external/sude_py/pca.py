import numpy as np
import warnings


def pca(X, no_dims, use_gpu=True, verbose=False):
    """
    This function performs PCA embedding.
    Now supports GPU acceleration via MLX (Apple Silicon) or TorchDR (CUDA).

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
    mappedX : array-like
        PCA-transformed data with specified dimensionality.
    """
    # Detect optimal device and PCA backend
    device_info = _detect_optimal_pca_backend(use_gpu, verbose)

    from ..._settings import settings
    omicverse_mode = getattr(settings, 'mode', 'cpu')
    
    if device_info['backend'] == 'mlx' and device_info['device'] == 'mps' and omicverse_mode != 'cpu':
        return _pca_mlx(X, no_dims, verbose)
    elif device_info['backend'] == 'torchdr' and device_info['device'] == 'cuda' and omicverse_mode != 'cpu':
        return _pca_torchdr(X, no_dims, verbose)
    else:
        # Fallback to CPU implementation
        return _pca_cpu(X, no_dims, verbose)


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
            
        if device_type == 'mps' and use_gpu and omicverse_mode != 'cpu':
            # Try MLX for Apple Silicon
            try:
                import mlx.core as mx
                if mx.metal.is_available():
                    return {'backend': 'mlx', 'device': 'mps', 'available': True}
            except ImportError:
                pass
                
        elif device_type == 'cuda' and use_gpu and omicverse_mode != 'cpu':
            # Try TorchDR for CUDA
            try:
                import torchdr
                import torch
                if torch.cuda.is_available():
                    return {'backend': 'torchdr', 'device': 'cuda', 'available': True}
            except ImportError:
                pass
        
        # Fallback to CPU
        return {'backend': 'cpu', 'device': 'cpu', 'available': True}
        
    except ImportError:
        # Fallback if omicverse settings not available
        return {'backend': 'cpu', 'device': 'cpu', 'available': True}


def _pca_mlx(X, no_dims, verbose=False):
    """
    MLX-based PCA implementation for Apple Silicon MPS devices.
    
    Parameters
    ----------
    X : array-like
        Input data matrix
    no_dims : int
        Number of dimensions
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
    
    # Create MLX PCA instance
    mlx_pca = MLXPCA(n_components=no_dims, device="metal")
    
    # Fit and transform
    mappedX = mlx_pca.fit_transform(X)
    
    if verbose:
        print(f"   ✅ MLX PCA completed: {X.shape} -> {mappedX.shape}")
        
    return mappedX
        

def _pca_torchdr(X, no_dims, verbose=False):
    """
    TorchDR-based PCA implementation for CUDA devices.
    
    Parameters
    ----------
    X : array-like
        Input data matrix
    no_dims : int
        Number of dimensions
    verbose : bool
        Whether to print information
        
    Returns
    -------
    mappedX : array-like
        PCA-transformed data
    """
    if verbose:
        print(f"   🚀 Using TorchDR PCA for CUDA acceleration")
    
    import torch
    import torchdr
    
    # Convert to torch tensor
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
    
    # Create TorchDR PCA
    pca = torchdr.MLXPCA(n_components=no_dims)
    
    # Fit and transform
    mappedX_tensor = pca.fit_transform(X_tensor)
    
    # Convert back to numpy
    mappedX = mappedX_tensor.cpu().numpy()
    
    if verbose:
        print(f"   ✅ TorchDR PCA completed: {X.shape} -> {mappedX.shape}")
        
    return mappedX
        
    

def _pca_cpu(X, no_dims, verbose=False):
    """
    Original CPU-based PCA implementation.
    
    Parameters
    ----------
    X : array-like
        Input data matrix
    no_dims : int
        Number of dimensions
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
    
    X = X - np.mean(X, axis=0)
    C = np.cov(X, rowvar=False)
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    evalues, evectors = np.linalg.eig(C)
    mappedX = X @ np.real(evectors)[:, :no_dims]
    
    if verbose:
        print(f"   ✅ CPU PCA completed: {X.shape} -> {mappedX.shape}")
    
    return mappedX
