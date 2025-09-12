import numpy as np
import warnings


def pps(knn, rnn, order, use_gpu=True, verbose=True):
    """
    Plum Pudding Sampling (PPS)
    This function returns the ID of landmarks.
    Now supports GPU acceleration via MLX (Apple Silicon) or Torch (CUDA).

    Parameters
    ----------
    knn : array-like
        A N by k matrix. Each row represents the KNN of each point.
    rnn : array-like
        A vector of length N. Each row represents the RNN of each point.
    order : int
        A positive integer specifying the order of KNN. Once a landmark is selected, its KNN will be removed
        from the queue. PPS supports the removal of multi-order KNN.
    use_gpu : bool, optional
        Whether to use GPU acceleration when available. Default: True.
    verbose : bool, optional
        Whether to print device selection information. Default: False.

    Returns
    -------
    list
        List of landmark IDs.
    """
    # If GPU is explicitly disabled, use CPU implementation
    if not use_gpu:
        return _pps_cpu(knn, rnn, order, verbose)
    
    # Detect optimal device and backend
    device_info = _detect_optimal_pps_backend(use_gpu, verbose)
    #print(device_info)
    
    from ..._settings import settings
    omicverse_mode = getattr(settings, 'mode', 'cpu')

    if device_info['backend'] == 'mlx' and device_info['device'] == 'mps' and omicverse_mode != 'cpu':
        return _pps_mlx(knn, rnn, order, verbose)
        #return _pps_torch(knn, rnn, order, 'cpu', verbose)
    elif device_info['backend'] == 'torch' and device_info['device'] in ['cuda', 'cpu'] and omicverse_mode != 'cpu':
        return _pps_torch(knn, rnn, order, device_info['device'], verbose)
    else:
        # Fallback to CPU implementation
        return _pps_cpu(knn, rnn, order, verbose)


def _detect_optimal_pps_backend(use_gpu=True, verbose=False):
    """
    Detect the optimal PPS backend based on available hardware and omicverse settings.
    
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


def _pps_mlx(knn, rnn, order, verbose=True):
    """
    MLX-based PPS implementation for Apple Silicon MPS devices.
    
    Parameters
    ----------
    knn : array-like
        KNN matrix
    rnn : array-like
        RNN vector
    order : int
        Order parameter
    verbose : bool
        Whether to print information
        
    Returns
    -------
    list
        List of landmark IDs
    """
    if verbose:
        print(f"   🚀 Using MLX PPS for Apple Silicon MPS acceleration")
    
    import mlx.core as mx
    
    # Convert to MLX arrays
    knn_mx = mx.array(np.asarray(knn))
    rnn_mx = mx.array(np.asarray(rnn))
    
    id_samp = []
    # Create sorted indices based on rnn values (descending)
    sorted_indices = mx.argsort(-rnn_mx)  # Negative for descending order
    id_sort_mx = sorted_indices
    
    from tqdm import tqdm
    
    total_points = int(id_sort_mx.size)
    pbar = tqdm(total=total_points, desc="   PPS Progress", disable=not verbose)
    
    while id_sort_mx.size > 0:
        # Get the first element (highest rnn value)
        current_id = int(id_sort_mx[0])
        id_samp.append(current_id)
        
        # Collect points to remove
        rm_pts = mx.array([current_id])
        
        # Expand removal points based on order
        for _ in range(order):
            if rm_pts.size > 0:
                # Get KNN of current removal points
                knn_indices = knn_mx[rm_pts].flatten()
                # Remove duplicates and concatenate (use numpy unique for compatibility)
                rm_pts_concat = mx.concatenate([rm_pts, knn_indices])
                rm_pts_np = np.unique(np.array(rm_pts_concat))
                rm_pts = mx.array(rm_pts_np)
        
        # Find which indices in id_sort need to be removed
        # Use broadcasting to find matches
        rm_pts_np = np.array(rm_pts)
        id_sort_np = np.array(id_sort_mx)
        
        # Create mask for points to keep
        keep_mask = ~np.isin(id_sort_np, rm_pts_np)
        
        # Update id_sort with remaining indices
        if np.any(keep_mask):
            id_sort_mx = mx.array(id_sort_np[keep_mask])
        else:
            break
        # Update progress bar per selected landmark
        pbar.update(1)
    pbar.close()
    
    if verbose:
        print(f"   ✅ MLX PPS completed: {len(id_samp)} landmarks selected")
        
    return id_samp
        
    

def _pps_torch(knn, rnn, order, device='cpu', verbose=False):
    """
    Torch-based PPS implementation for CUDA/CPU devices.
    
    Parameters
    ----------
    knn : array-like
        KNN matrix
    rnn : array-like
        RNN vector
    order : int
        Order parameter
    device : str
        Device to use ('cuda' or 'cpu')
    verbose : bool
        Whether to print information
        
    Returns
    -------
    list
        List of landmark IDs
    """
    if verbose:
        print(f"   🚀 Using Torch PPS for {device.upper()} acceleration")
    
    import torch
    
    # Convert to torch tensors with proper dtype
    knn_torch = torch.tensor(np.asarray(knn), dtype=torch.long, device=device)
    rnn_torch = torch.tensor(np.asarray(rnn), dtype=torch.float32, device=device)
    
    id_samp = []
    # Create sorted indices based on rnn values (descending) with stable sort
    # To match CPU behavior exactly, we need stable sorting
    # torch.sort is not stable, so we use argsort with stable=True if available
    try:
        sorted_indices = torch.argsort(rnn_torch, descending=True, stable=True)
    except TypeError:
        # Fallback for older PyTorch versions without stable parameter
        # Use CPU numpy sorting and convert back
        rnn_np = rnn_torch.cpu().numpy()
        cpu_sorted = sorted(range(len(rnn_np)), key=lambda k: rnn_np[k], reverse=True)
        sorted_indices = torch.tensor(cpu_sorted, device=device)
    id_sort_torch = sorted_indices

    from tqdm import tqdm
    
    total_points = int(id_sort_torch.numel())
    pbar = tqdm(total=total_points, desc="   PPS Progress", disable=not verbose)
    
    
    while id_sort_torch.numel() > 0:
        # Get the first element (highest rnn value)
        current_id = int(id_sort_torch[0].item())
        id_samp.append(current_id)
        
        # Collect points to remove
        rm_pts = torch.tensor([current_id], device=device)
        
        # Expand removal points based on order
        for _ in range(order):
            if rm_pts.numel() > 0:
                # Get KNN of current removal points
                knn_indices = knn_torch[rm_pts].flatten()
                # Remove duplicates and concatenate
                rm_pts = torch.unique(torch.cat([rm_pts, knn_indices]))
        
        # Find which indices in id_sort need to be removed
        # Create mask for points to keep
        rm_pts_expanded = rm_pts.unsqueeze(1)  # Shape: (rm_pts_size, 1)
        id_sort_expanded = id_sort_torch.unsqueeze(0)  # Shape: (1, id_sort_size)
        
        # Broadcasting comparison
        matches = (rm_pts_expanded == id_sort_expanded).any(dim=0)
        keep_mask = ~matches
        
        # Update id_sort with remaining indices
        if keep_mask.any():
            id_sort_torch = id_sort_torch[keep_mask]
        else:
            break
        pbar.update(1)
    pbar.close()
    
    if verbose:
        print(f"   ✅ Torch PPS completed: {len(id_samp)} landmarks selected")
        
    return id_samp
        


def _pps_cpu(knn, rnn, order, verbose=False):
    """
    Original CPU-based PPS implementation.
    
    Parameters
    ----------
    knn : array-like
        KNN matrix
    rnn : array-like
        RNN vector
    order : int
        Order parameter
    verbose : bool
        Whether to print information
        
    Returns
    -------
    list
        List of landmark IDs
    """
    if verbose:
        print(f"   🖥️ Using CPU PPS implementation")
    
    # Convert to numpy arrays
    knn = np.asarray(knn)
    rnn = np.asarray(rnn)
    
    id_samp = []
    id_sort = sorted(range(len(rnn)), key=lambda k: rnn[k], reverse=True)

    # Progress bar (align with _pps_mlx)
    pbar = None
    try:
        from tqdm import tqdm
        total_points = int(len(rnn))
        pbar = tqdm(total=total_points, desc="   PPS Progress", disable=not verbose)
    except Exception:
        pbar = None  # tqdm not available; proceed without progress bar
    while len(id_sort) > 0:
        id_samp.append(id_sort[0])
        rm_pts = [id_sort[0]]
        for _ in range(order):
            rm_pts.extend(knn[rm_pts].flatten().tolist())
        rm_pts = set(rm_pts)
        rm_id = np.where(np.isin(id_sort, list(rm_pts)))[0]
        id_sort = [id_sort[i] for i in range(len(id_sort)) if i not in rm_id]
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    
    if verbose:
        print(f"   ✅ CPU PPS completed: {len(id_samp)} landmarks selected")
    
    return id_samp
