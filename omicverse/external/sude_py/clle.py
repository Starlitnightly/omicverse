import numpy as np
import warnings


def clle(X_samp, Y_samp, X_i, N_dis, use_gpu=True, verbose=False):
    """
    Constrained Locally Linear Embedding (CLLE)
    This function returns representation of point X_i.
    Now supports GPU acceleration via MLX (Apple Silicon) or Torch (CUDA).

    Parameters
    ----------
    X_samp : array-like
        High-dimensional features of KNN of point X_i. Each row denotes an observation.
    Y_samp : array-like
        Low-dimensional embeddings of KNN of point X_i.
    X_i : array-like
        Current non-landmark point.
    N_dis : float
        Distance between point X_i and its nearest neighbor in lower-dimensional space.
    use_gpu : bool, optional
        Whether to use GPU acceleration when available. Default: True.
    verbose : bool, optional
        Whether to print device selection information. Default: False.

    Returns
    -------
    array-like
        Low-dimensional representation of point X_i.
    """
    # If GPU is explicitly disabled, use CPU implementation
    if not use_gpu:
        return _clle_cpu(X_samp, Y_samp, X_i, N_dis, verbose)
    
    # Detect optimal device and backend
    device_info = _detect_optimal_clle_backend(use_gpu, verbose)

    from ..._settings import settings
    omicverse_mode = getattr(settings, 'mode', 'cpu')

    if device_info['backend'] == 'mlx' and device_info['device'] == 'mps' and omicverse_mode != 'cpu':
        return _clle_mlx(X_samp, Y_samp, X_i, N_dis, verbose)
    elif device_info['backend'] == 'torch' and device_info['device'] in ['cuda', 'cpu'] and omicverse_mode != 'cpu':
        return _clle_torch(X_samp, Y_samp, X_i, N_dis, device_info['device'], verbose)
    else:
        # Fallback to CPU implementation
        return _clle_cpu(X_samp, Y_samp, X_i, N_dis, verbose)


def _detect_optimal_clle_backend(use_gpu=True, verbose=False):
    """
    Detect the optimal CLLE backend based on available hardware and omicverse settings.
    
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


def _clle_mlx(X_samp, Y_samp, X_i, N_dis, verbose=False):
    """
    MLX-based CLLE implementation for Apple Silicon MPS devices.
    
    Parameters
    ----------
    X_samp : array-like
        High-dimensional features matrix
    Y_samp : array-like
        Low-dimensional embeddings matrix
    X_i : array-like
        Current point
    N_dis : float
        Distance parameter
    verbose : bool
        Whether to print information
        
    Returns
    -------
    array-like
        Low-dimensional representation
    """
    if verbose:
        print(f"   🚀 Using MLX CLLE for Apple Silicon MPS acceleration")
    
    try:
        import mlx.core as mx
        
        # Convert to MLX arrays
        X_samp_mx = mx.array(np.asarray(X_samp, dtype=np.float32))
        Y_samp_mx = mx.array(np.asarray(Y_samp, dtype=np.float32))
        X_i_mx = mx.array(np.asarray(X_i, dtype=np.float32))
        
        n = X_samp_mx.shape[0]
        
        # Compute S matrix: (X_samp - X_i) @ (X_samp - X_i).T
        diff = X_samp_mx - X_i_mx
        S = diff @ diff.T
        
        # Check determinant and regularize if needed
        det_S = mx.linalg.det(S)
        if mx.abs(det_S) <= 1e-15:  # mx.finfo equivalent
            trace_S = mx.trace(S)
            regularization = (0.1 ** 2 / n) * trace_S * mx.eye(n)
            S = S + regularization
        
        # Compute weights W
        ones_n = mx.ones((n, 1))
        S_inv = mx.linalg.inv(S)
        numerator = S_inv @ ones_n
        denominator = ones_n.T @ S_inv @ ones_n
        W = numerator / denominator
        
        # Compute Y_0
        Y_0 = W.T @ Y_samp_mx
        
        # Compute distance dd (match CPU implementation exactly)
        dd = mx.sqrt((Y_samp_mx[0] - Y_0) @ (Y_samp_mx[0] - Y_0).T)
        dd = float(dd)  # Convert to scalar
        
        # Compute final result
        if dd != 0:
            Y_i = Y_samp_mx[0] + N_dis * (Y_0.flatten() - Y_samp_mx[0]) / dd
        else:
            Y_i = Y_samp_mx[0]
        
        # Convert back to numpy and ensure shape matches CPU implementation
        result = np.array(Y_i)
        if result.ndim == 1:
            result = result.reshape(1, -1)  # Make it (1, n_dims) like CPU
        
        if verbose:
            print(f"   ✅ MLX CLLE completed")
            
        return result
        
    except Exception as e:
        if verbose:
            print(f"   ⚠️ MLX CLLE failed ({str(e)}), falling back to CPU")
        return _clle_cpu(X_samp, Y_samp, X_i, N_dis, verbose)


def _clle_torch(X_samp, Y_samp, X_i, N_dis, device='cpu', verbose=False):
    """
    Torch-based CLLE implementation for CUDA/CPU devices.
    
    Parameters
    ----------
    X_samp : array-like
        High-dimensional features matrix
    Y_samp : array-like
        Low-dimensional embeddings matrix
    X_i : array-like
        Current point
    N_dis : float
        Distance parameter
    device : str
        Device to use ('cuda' or 'cpu')
    verbose : bool
        Whether to print information
        
    Returns
    -------
    array-like
        Low-dimensional representation
    """
    if verbose:
        print(f"   🚀 Using Torch CLLE for {device.upper()} acceleration")
    
    try:
        import torch
        
        # Convert to torch tensors
        X_samp_torch = torch.tensor(np.asarray(X_samp, dtype=np.float32), device=device)
        Y_samp_torch = torch.tensor(np.asarray(Y_samp, dtype=np.float32), device=device)
        X_i_torch = torch.tensor(np.asarray(X_i, dtype=np.float32), device=device)
        
        n = X_samp_torch.shape[0]
        
        # Compute S matrix: (X_samp - X_i) @ (X_samp - X_i).T
        diff = X_samp_torch - X_i_torch
        S = diff @ diff.T
        
        # Check determinant and regularize if needed
        det_S = torch.det(S)
        if torch.abs(det_S) <= torch.finfo(torch.float32).eps:
            trace_S = torch.trace(S)
            regularization = (0.1 ** 2 / n) * trace_S * torch.eye(n, device=device)
            S = S + regularization
        
        # Compute weights W
        ones_n = torch.ones((n, 1), device=device)
        S_inv = torch.inverse(S)
        numerator = S_inv @ ones_n
        denominator = ones_n.T @ S_inv @ ones_n
        W = numerator / denominator
        
        # Compute Y_0
        Y_0 = W.T @ Y_samp_torch
        
        # Compute distance dd (match CPU implementation exactly)
        Y_0_flat = Y_0.flatten()
        diff = Y_samp_torch[0] - Y_0_flat
        dd = torch.sqrt(diff @ diff.T)
        dd = float(dd.item())  # Convert to scalar
        
        # Compute final result
        if dd != 0:
            Y_i = Y_samp_torch[0] + N_dis * (Y_0.flatten() - Y_samp_torch[0]) / dd
        else:
            Y_i = Y_samp_torch[0]
        
        # Convert back to numpy and ensure shape matches CPU implementation
        result = Y_i.cpu().numpy()
        if result.ndim == 1:
            result = result.reshape(1, -1)  # Make it (1, n_dims) like CPU
        
        if verbose:
            print(f"   ✅ Torch CLLE completed")
            
        return result
        
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Torch CLLE failed ({str(e)}), falling back to CPU")
        return _clle_cpu(X_samp, Y_samp, X_i, N_dis, verbose)


def _clle_cpu(X_samp, Y_samp, X_i, N_dis, verbose=False):
    """
    Original CPU-based CLLE implementation.
    
    Parameters
    ----------
    X_samp : array-like
        High-dimensional features matrix
    Y_samp : array-like
        Low-dimensional embeddings matrix
    X_i : array-like
        Current point
    N_dis : float
        Distance parameter
    verbose : bool
        Whether to print information
        
    Returns
    -------
    array-like
        Low-dimensional representation
    """
    if verbose:
        print(f"   🖥️ Using CPU CLLE implementation")
    
    # Convert to numpy arrays
    X_samp = np.asarray(X_samp)
    Y_samp = np.asarray(Y_samp)
    X_i = np.asarray(X_i)
    
    n = X_samp.shape[0]
    S = (X_samp - X_i) @ (X_samp - X_i).transpose()
    if np.abs(np.linalg.det(S)) <= np.finfo(float).eps:
        S = S + (0.1 ** 2 / n) * np.trace(S) * np.eye(n)
    W = (np.linalg.inv(S) @ np.ones((n, 1))) / (np.ones((1, n)) @ np.linalg.inv(S) @ np.ones((n, 1)))
    Y_0 = W.transpose() @ Y_samp
    dd = np.sqrt((Y_samp[0] - Y_0) @ (Y_samp[0] - Y_0).transpose())
    if dd != 0:
        Y_i = Y_samp[0] + N_dis * (Y_0 - Y_samp[0]) / dd
    else:
        Y_i = Y_samp[0]

    if verbose:
        print(f"   ✅ CPU CLLE completed")

    return Y_i