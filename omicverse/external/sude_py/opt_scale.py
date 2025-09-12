from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import numpy as np
import warnings


def opt_scale(X, Y, k_num, use_gpu=True, verbose=False):
    """
    This function computes the optimal scales of landmarks.
    Now supports GPU acceleration via MLX (Apple Silicon) or Torch (CUDA).

    Parameters
    ----------
    X : array-like
        N by D matrix. Each row in X represents high-dimensional features of a landmark.
    Y : array-like
        N by d matrix. Each row in Y represents low-dimensional embedding of a landmark.
    k_num : int
        A non-negative integer specifying the number of KNN.
    use_gpu : bool, optional
        Whether to use GPU acceleration when available. Default: True.
    verbose : bool, optional
        Whether to print device selection information. Default: False.

    Returns
    -------
    array-like
        Optimal scales for landmarks.
    """
    # If GPU is explicitly disabled, use CPU implementation
    if not use_gpu:
        return _opt_scale_cpu(X, Y, k_num, verbose)
    
    # Detect optimal device and backend
    device_info = _detect_optimal_opt_scale_backend(use_gpu, verbose)

    from ..._settings import settings
    omicverse_mode = getattr(settings, 'mode', 'cpu')

    if device_info['backend'] == 'mlx' and device_info['device'] == 'mps' and omicverse_mode != 'cpu':
        return _opt_scale_mlx(X, Y, k_num, verbose)
    elif device_info['backend'] == 'torch' and device_info['device'] in ['cuda', 'cpu'] and omicverse_mode != 'cpu':
        return _opt_scale_torch(X, Y, k_num, device_info['device'], verbose)
    else:
        # Fallback to CPU implementation
        return _opt_scale_cpu(X, Y, k_num, verbose)


def _detect_optimal_opt_scale_backend(use_gpu=True, verbose=False):
    """
    Detect the optimal opt_scale backend based on available hardware and omicverse settings.
    
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


def _compute_pairwise_distances_mlx(points):
    """
    Compute pairwise distances using MLX.
    
    Parameters
    ----------
    points : mlx.array
        Points matrix
        
    Returns
    -------
    mlx.array
        Pairwise distance matrix
    """
    import mlx.core as mx
    
    # Compute pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i @ x_j
    squared_norms = mx.sum(points ** 2, axis=1, keepdims=True)
    distances_squared = squared_norms + squared_norms.T - 2 * points @ points.T
    
    # Take square root and ensure non-negative
    distances = mx.sqrt(mx.maximum(distances_squared, 0.0))
    
    return distances


def _compute_pairwise_distances_torch(points, device):
    """
    Compute pairwise distances using Torch.
    
    Parameters
    ----------
    points : torch.Tensor
        Points matrix
    device : str
        Device to use
        
    Returns
    -------
    torch.Tensor
        Pairwise distance matrix
    """
    import torch
    
    # Compute pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i @ x_j
    squared_norms = torch.sum(points ** 2, dim=1, keepdim=True)
    distances_squared = squared_norms + squared_norms.T - 2 * points @ points.T
    
    # Take square root and ensure non-negative
    distances = torch.sqrt(torch.clamp(distances_squared, min=0.0))
    
    return distances


def _opt_scale_mlx(X, Y, k_num, verbose=False):
    """
    MLX-based opt_scale implementation for Apple Silicon MPS devices.
    
    Parameters
    ----------
    X : array-like
        High-dimensional features matrix
    Y : array-like
        Low-dimensional embeddings matrix
    k_num : int
        Number of nearest neighbors
    verbose : bool
        Whether to print information
        
    Returns
    -------
    array-like
        Optimal scales
    """
    if verbose:
        print(f"   🚀 Using MLX opt_scale for Apple Silicon MPS acceleration")
    
    import mlx.core as mx
    
    # Convert to MLX arrays
    X_mx = mx.array(np.asarray(X, dtype=np.float32))
    Y_mx = mx.array(np.asarray(Y, dtype=np.float32))
    
    n = X_mx.shape[0]
    
    # Find KNN using sklearn (CPU-based, but fast for small k_num)
    get_knn = NearestNeighbors(n_neighbors=k_num + 1).fit(np.array(Y_mx)).kneighbors(np.array(Y_mx), return_distance=False)
    
    scale = mx.zeros((n, 1))
    
    for i in range(n):
        # Get KNN indices for point i
        knn_indices = get_knn[i]
        
        # Convert indices to MLX array for proper indexing
        knn_indices_mx = mx.array(knn_indices)
        
        # Extract KNN points
        X_knn = X_mx[knn_indices_mx]
        Y_knn = Y_mx[knn_indices_mx]
        
        # Compute pairwise distances
        X_dis = _compute_pairwise_distances_mlx(X_knn)
        Y_dis = _compute_pairwise_distances_mlx(Y_knn)
        
        # Compute scale
        numerator = mx.sum(X_dis * Y_dis)
        denominator = mx.maximum(mx.sum(X_dis ** 2), 1e-16)  # Equivalent to np.finfo(float).tiny
        # Use proper MLX .at[] syntax for updates
        scale = scale.at[i, 0].add((numerator / denominator) - scale[i, 0])
    
    # Convert back to numpy
    result = np.array(scale)
    
    if verbose:
        print(f"   ✅ MLX opt_scale completed")
        
    return result
        
    
def _opt_scale_torch(X, Y, k_num, device='cpu', verbose=False):
    """
    Torch-based opt_scale implementation for CUDA/CPU devices.
    
    Parameters
    ----------
    X : array-like
        High-dimensional features matrix
    Y : array-like
        Low-dimensional embeddings matrix
    k_num : int
        Number of nearest neighbors
    device : str
        Device to use ('cuda' or 'cpu')
    verbose : bool
        Whether to print information
        
    Returns
    -------
    array-like
        Optimal scales
    """
    if verbose:
        print(f"   🚀 Using Torch opt_scale for {device.upper()} acceleration")
    
    import torch
    
    # Convert to torch tensors
    X_torch = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    Y_torch = torch.tensor(np.asarray(Y, dtype=np.float32), device=device)
    
    n = X_torch.shape[0]
    
    # Find KNN using sklearn (CPU-based, but fast for small k_num)
    get_knn = NearestNeighbors(n_neighbors=k_num + 1).fit(Y_torch.cpu().numpy()).kneighbors(Y_torch.cpu().numpy(), return_distance=False)
    
    scale = torch.zeros((n, 1), device=device)
    
    for i in range(n):
        # Get KNN indices for point i
        knn_indices = get_knn[i]
        
        # Extract KNN points
        X_knn = X_torch[knn_indices]
        Y_knn = Y_torch[knn_indices]
        
        # Compute pairwise distances
        X_dis = _compute_pairwise_distances_torch(X_knn, device)
        Y_dis = _compute_pairwise_distances_torch(Y_knn, device)
        
        # Compute scale
        numerator = torch.sum(X_dis * Y_dis)
        denominator = torch.clamp(torch.sum(X_dis ** 2), min=torch.finfo(torch.float32).tiny)
        scale[i, 0] = numerator / denominator
    
    # Convert back to numpy
    result = scale.cpu().numpy()
    
    if verbose:
        print(f"   ✅ Torch opt_scale completed")
        
    return result
        
    

def _opt_scale_cpu(X, Y, k_num, verbose=False):
    """
    Original CPU-based opt_scale implementation.
    
    Parameters
    ----------
    X : array-like
        High-dimensional features matrix
    Y : array-like
        Low-dimensional embeddings matrix
    k_num : int
        Number of nearest neighbors
    verbose : bool
        Whether to print information
        
    Returns
    -------
    array-like
        Optimal scales
    """
    if verbose:
        print(f"   🖥️ Using CPU opt_scale implementation")
    
    # Convert to numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    n = X.shape[0]
    get_knn = NearestNeighbors(n_neighbors=k_num + 1).fit(Y).kneighbors(Y, return_distance=False)
    scale = np.zeros((n, 1))
    for i in range(n):
        XDis = cdist(X[get_knn[i]], X[get_knn[i]])
        YDis = cdist(Y[get_knn[i]], Y[get_knn[i]])
        scale[i] = np.sum(XDis * YDis) / max(np.sum(XDis ** 2), np.finfo(float).tiny)

    if verbose:
        print(f"   ✅ CPU opt_scale completed")

    return scale
