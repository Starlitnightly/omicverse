from sklearn.neighbors import NearestNeighbors
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from .init_pca import init_pca
from .pca import pca
from .mds import mds
import scipy.sparse.linalg as sp_linalg
import numpy as np
import math
from tqdm import tqdm

from scipy.sparse.linalg import eigsh, LinearOperator


def learning_l(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, use_gpu=True, verbose=True):
    """
    This function returns representation of the landmarks in the lower-dimensional space and the number of nearest
    neighbors of landmarks. It computes the gradient using probability matrix P and Q of data blocks.
    Now supports GPU acceleration via MLX (Apple Silicon) or Torch (CUDA).

    Parameters
    ----------
    X_samp : array-like
        Landmark samples data matrix.
    k1 : int
        Number of nearest neighbors parameter.
    get_knn : array-like
        K-nearest neighbors indices.
    rnn : array-like
        Reverse nearest neighbors matrix.
    id_samp : array-like
        Sample indices.
    no_dims : int
        Number of dimensions for the output embedding.
    initialize : str
        Initialization method ('le', 'pca', or 'mds').
    agg_coef : float
        Aggregation coefficient.
    T_epoch : int
        Number of training epochs.
    use_gpu : bool, optional
        Whether to use GPU acceleration when available. Default: True.
    verbose : bool, optional
        Whether to print device selection information. Default: False.

    Returns
    -------
    tuple
        (Y, k2) where Y is the embedding and k2 is the number of nearest neighbors.
    """
    # If GPU is explicitly disabled, use CPU implementation
    if not use_gpu:
        return _learning_l_cpu(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, verbose)

    from ..._settings import settings
    omicverse_mode = getattr(settings, 'mode', 'cpu')
    
    # Detect optimal device and backend
    device_info = _detect_optimal_learning_backend(use_gpu, verbose)
    
    if device_info['backend'] == 'mlx' and device_info['device'] == 'mps' and omicverse_mode != 'cpu':
        return _learning_l_mlx(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, verbose)
        #return _learning_l_torch(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, device_info['device'], verbose)
    elif device_info['backend'] == 'torch' and device_info['device'] in ['cuda', 'cpu'] and omicverse_mode != 'cpu':
        return _learning_l_torch(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, device_info['device'], verbose)
    else:
        # Fallback to CPU implementation
        return _learning_l_cpu(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, verbose)


def _detect_optimal_learning_backend(use_gpu=True, verbose=False):
    """
    Detect the optimal learning backend based on available hardware and omicverse settings.
    
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


def _learning_l_cpu(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, verbose=False):
    """
    Original CPU-based learning_l implementation.
    """
    if verbose:
        print(f"   🖥️ Using CPU learning_l implementation")
    
    # Obtain size and dimension of landmarks
    N, dim = X_samp.shape

    # Compute the number of nearest neighbors of landmarks adaptively
    if N < 9:
        k2 = N
    else:
        if N > 1000:
            k2 = int(np.ceil(np.log2(N)) + 18)
        elif N > 50:
            k2 = int(np.ceil(0.02 * N)) + 8
        else:
            k2 = 9

    # Compute high-dimensional probability matrix P
    if k1 > 0:
        row = []
        col = []
        Pval = []
        knn_rnn_mat = rnn[get_knn[id_samp]]
        for i in range(N):
            snn_id = np.isin(get_knn[id_samp], get_knn[id_samp[i]]).astype(int)
            nn_id = np.where(np.max(snn_id, axis=1) == 1)[0]
            snn = np.zeros((1, N))
            snn[:, nn_id] = np.sum(knn_rnn_mat[nn_id] * snn_id[nn_id], axis=1)
            mod_dis = (1 - snn / max(np.max(snn), np.finfo(float).tiny)) ** agg_coef * cdist(X_samp[i:i + 1, :], X_samp)
            sort_dis = np.sort(mod_dis, axis=1)
            idx = np.argsort(mod_dis, axis=1)
            mean_samp_dis_squared = np.square(np.mean(sort_dis[0, :k2]))
            Pval.extend(
                np.exp(-0.5 * np.square(sort_dis[0, :k2]) / np.maximum(mean_samp_dis_squared, np.finfo(float).tiny)))
            row.extend((i * np.ones((k2, 1))).flatten().tolist())
            col.extend(idx[0, :k2])
        P = csr_matrix((Pval, (row, col)), shape=(N, N))
    else:
        if N > 5000 and dim > 50:
            xx = init_pca(X_samp, no_dims, 0.8)
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(xx).kneighbors(xx)
        else:
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(X_samp).kneighbors(X_samp)
        mean_samp_dis_squared = np.square(np.mean(samp_dis, axis=1))
        Pval = np.exp(-0.5 * np.square(samp_dis) / np.maximum(mean_samp_dis_squared[:, np.newaxis],
                                                              np.finfo(float).tiny))
        P = csr_matrix((Pval.flatten(), ([i for i in range(N) for _ in range(k2)], samp_knn.flatten())), shape=(N, N))
    # Symmetrize matrix P
    P = (P + P.transpose()) / 2

    # Initialize embedding Y of landmarks
    if initialize == 'le':
        # 归一化图拉普拉斯：L = I - D^{-1/2} P D^{-1/2}
        # 注：若 P 对称，axis=0/1 求和等价；此处沿用你原实现的 axis=0。
        deg = np.asarray(P.sum(axis=0)).ravel()

        # 避免度为0导致的除零
        d_inv_sqrt = np.zeros_like(deg, dtype=float)
        nz = deg > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])

        n = P.shape[0]

        # 通过 LinearOperator 隐式实现 L 的 matvec，避免显式构造大矩阵
        def matvec(x):
            # y = D^{-1/2} P D^{-1/2} x
            y = P @ (d_inv_sqrt * x)
            y = d_inv_sqrt * y
            return x - y  # (I - D^{-1/2} P D^{-1/2}) x

        Lop = LinearOperator((n, n), matvec=matvec, dtype=float)

        # 取最小的 k 个特征对（含零特征值），丢掉首个零特征向量
        k = min(no_dims + 1, max(2, n - 1))
        vals, vecs = eigsh(Lop, k=k, which='SM')

        # 保险起见按特征值从小到大排序后丢第一列
        order = np.argsort(vals)
        Y = np.ascontiguousarray(vecs[:, order[1:no_dims+1]])
    elif initialize == 'pca':
        Y = pca(X_samp, no_dims)
    elif initialize == 'mds':
        Y = mds(X_samp, no_dims, use_gpu=False)  # Disable GPU for CPU mode

    # Normalize matrix P
    P = P / (np.sum(P) - N)

    # Compute the start and end markers of each data block
    no_blocks = math.ceil(N / 3000)
    mark = np.zeros((no_blocks, 2))
    for i in range(no_blocks):
        mark[i, :] = [i * math.ceil(N / no_blocks), min((i + 1) * math.ceil(N / no_blocks) - 1, N - 1)]

    # Initialization
    max_alpha = 2.5 * N
    min_alpha = 2 * N
    warm_step = 10
    preGrad = np.zeros((N, no_dims))
    
    # Use tqdm for epoch progress tracking
    for epoch in tqdm(range(1, T_epoch + 1), desc="Training epochs", unit="epoch"):
        # Update learning rate
        if epoch <= warm_step:
            alpha = max_alpha
        else:
            alpha = min_alpha + 0.5 * (max_alpha - min_alpha) * (
                        1 + np.cos(np.pi * ((epoch - warm_step) / (T_epoch - warm_step))))
        Pgrad = np.zeros((N, no_dims))
        Qgrad = np.zeros((N, no_dims))
        sumQ = 0
        # Compute gradient

        for i in range(no_blocks):
            idx = [j for j in range(int(mark[i, 0]), int(mark[i, 1]) + 1)]
            D = cdist(Y[idx], Y) ** 2
            Q1 = 1 / (1 + np.log(1 + D))
            QQ1 = 1 / (1 + D)
            del D
            Pmat = -4 * P[idx, :].multiply(Q1).multiply(QQ1).toarray()
            Qmat = -4 * Q1 ** 2 * QQ1
            del QQ1
            len_blk = len(idx)
            idPQ = np.column_stack((np.array(range(len_blk)), idx[0] + np.array(range(len_blk))))
            Pmat[idPQ[:, 0], idPQ[:, 1]] = Pmat[idPQ[:, 0], idPQ[:, 1]] - np.sum(Pmat, axis=1)
            Qmat[idPQ[:, 0], idPQ[:, 1]] = Qmat[idPQ[:, 0], idPQ[:, 1]] - np.sum(Qmat, axis=1)
            Pgrad[idx] = Pmat @ Y
            Qgrad[idx] = Qmat @ Y
            del Pmat, Qmat
            sumQ = sumQ + np.sum(Q1)
        # Update embedding Y
        Y = Y - alpha * (Pgrad - Qgrad / (sumQ - N) + (epoch - 1) / (epoch + 2) * preGrad)
        preGrad = Pgrad - Qgrad / (sumQ - N)

    if verbose:
        print(f"   ✅ CPU learning_l completed: {X_samp.shape} -> {Y.shape}")
    return Y, k2


def _compute_pairwise_distances_mlx(Y_subset, Y_full):
    """
    Compute pairwise squared distances using MLX.
    
    Parameters
    ----------
    Y_subset : mlx.array
        Subset of data points
    Y_full : mlx.array
        Full dataset
        
    Returns
    -------
    mlx.array
        Squared distance matrix
    """
    import mlx.core as mx
    
    # Compute squared distance matrix using efficient broadcasting
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a@b.T
    norms_subset = mx.sum(Y_subset ** 2, axis=1, keepdims=True)  # (n_subset, 1)
    norms_full = mx.sum(Y_full ** 2, axis=1, keepdims=True).T     # (1, n_full)
    dot_product = Y_subset @ Y_full.T                             # (n_subset, n_full)
    
    D_squared = norms_subset + norms_full - 2 * dot_product
    return mx.maximum(D_squared, 0.0)  # Ensure non-negative


def _compute_pairwise_distances_torch(Y_subset, Y_full, device):
    """
    Compute pairwise squared distances using Torch.
    
    Parameters
    ----------
    Y_subset : torch.Tensor
        Subset of data points
    Y_full : torch.Tensor
        Full dataset
    device : str
        Device to use
        
    Returns
    -------
    torch.Tensor
        Squared distance matrix
    """
    import torch
    
    # Compute squared distance matrix using efficient broadcasting
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a@b.T
    norms_subset = torch.sum(Y_subset ** 2, dim=1, keepdim=True)  # (n_subset, 1)
    norms_full = torch.sum(Y_full ** 2, dim=1, keepdim=True).T     # (1, n_full)
    dot_product = Y_subset @ Y_full.T                             # (n_subset, n_full)
    
    D_squared = norms_subset + norms_full - 2 * dot_product
    return torch.clamp(D_squared, min=0.0)  # Ensure non-negative


def _learning_l_mlx(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, verbose=False):
    """
    MLX-based learning_l implementation for Apple Silicon MPS devices.
    Optimized to minimize CPU-GPU transfers by keeping all computations on GPU.
    """
    if verbose:
        print(f"   🚀 Using MLX learning_l for Apple Silicon MPS acceleration (Optimized)")
    
    import mlx.core as mx
        
    # Convert to numpy first for initial computations that don't benefit from GPU
    X_samp = np.asarray(X_samp, dtype=np.float32)
    N, dim = X_samp.shape

    # Compute k2 (same as CPU)
    if N < 9:
        k2 = N
    else:
        if N > 1000:
            k2 = int(np.ceil(np.log2(N)) + 18)
        elif N > 50:
            k2 = int(np.ceil(0.02 * N)) + 8
        else:
            k2 = 9

    # Initial P matrix computation (CPU for complexity)
    if k1 > 0:
        row = []
        col = []
        Pval = []
        knn_rnn_mat = rnn[get_knn[id_samp]]
        for i in tqdm(range(N), desc="Computing P matrix"):
            snn_id = np.isin(get_knn[id_samp], get_knn[id_samp[i]]).astype(int)
            nn_id = np.where(np.max(snn_id, axis=1) == 1)[0]
            snn = np.zeros((1, N))
            snn[:, nn_id] = np.sum(knn_rnn_mat[nn_id] * snn_id[nn_id], axis=1)
            mod_dis = (1 - snn / max(np.max(snn), np.finfo(float).tiny)) ** agg_coef * cdist(X_samp[i:i + 1, :], X_samp)
            sort_dis = np.sort(mod_dis, axis=1)
            idx = np.argsort(mod_dis, axis=1)
            mean_samp_dis_squared = np.square(np.mean(sort_dis[0, :k2]))
            Pval.extend(
                np.exp(-0.5 * np.square(sort_dis[0, :k2]) / np.maximum(mean_samp_dis_squared, np.finfo(float).tiny)))
            row.extend((i * np.ones((k2, 1))).flatten().tolist())
            col.extend(idx[0, :k2])
        P = csr_matrix((Pval, (row, col)), shape=(N, N))
    else:
        if N > 5000 and dim > 50:
            xx = init_pca(X_samp, no_dims, 0.8)
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(xx).kneighbors(xx)
        else:
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(X_samp).kneighbors(X_samp)
        mean_samp_dis_squared = np.square(np.mean(samp_dis, axis=1))
        Pval = np.exp(-0.5 * np.square(samp_dis) / np.maximum(mean_samp_dis_squared[:, np.newaxis], np.finfo(float).tiny))
        P = csr_matrix((Pval.flatten(), ([i for i in range(N) for _ in range(k2)], samp_knn.flatten())), shape=(N, N))
    
    P = (P + P.transpose()) / 2

    # Initialize embedding Y
    if initialize == 'le':
        deg = np.asarray(P.sum(axis=0)).ravel()
        d_inv_sqrt = np.zeros_like(deg, dtype=float)
        nz = deg > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        n = P.shape[0]
        
        def matvec(x):
            y = P @ (d_inv_sqrt * x)
            y = d_inv_sqrt * y
            return x - y
        
        Lop = LinearOperator((n, n), matvec=matvec, dtype=float)
        k = min(no_dims + 1, max(2, n - 1))
        vals, vecs = eigsh(Lop, k=k, which='SM')
        order = np.argsort(vals)
        Y = np.ascontiguousarray(vecs[:, order[1:no_dims+1]], dtype=np.float32)
    elif initialize == 'pca':
        Y = pca(X_samp, no_dims).astype(np.float32)
    elif initialize == 'mds':
        Y = mds(X_samp, no_dims, use_gpu=True, verbose=False).astype(np.float32)

    # Convert to MLX arrays for GPU computation and keep everything on GPU
    Y_mx = mx.array(Y)
    P = P / (np.sum(P) - N)

    # Compute data blocks
    no_blocks = math.ceil(N / 3000)
    mark = np.zeros((no_blocks, 2))
    for i in range(no_blocks):
        mark[i, :] = [i * math.ceil(N / no_blocks), min((i + 1) * math.ceil(N / no_blocks) - 1, N - 1)]

    # Training loop with fully GPU-accelerated computation
    max_alpha = 2.5 * N
    min_alpha = 2 * N
    warm_step = 10
    preGrad_mx = mx.zeros((N, no_dims))
    
    # Pre-allocate MLX arrays for gradients to avoid repeated allocations
    Pgrad_mx = mx.zeros((N, no_dims))
    Qgrad_mx = mx.zeros((N, no_dims))
    
    for epoch in tqdm(range(1, T_epoch + 1), desc="Training epochs (MLX-Optimized)", unit="epoch"):
        if epoch <= warm_step:
            alpha = max_alpha
        else:
            alpha = min_alpha + 0.5 * (max_alpha - min_alpha) * (
                        1 + np.cos(np.pi * ((epoch - warm_step) / (T_epoch - warm_step))))
        
        # Reset gradients on GPU
        Pgrad_mx = mx.zeros((N, no_dims))
        Qgrad_mx = mx.zeros((N, no_dims))
        sumQ = 0.0
        
        for i in range(no_blocks):
            start_idx = int(mark[i, 0])
            end_idx = int(mark[i, 1]) + 1
            idx = list(range(start_idx, end_idx))
            len_blk = len(idx)
            
            # GPU-accelerated distance computation
            Y_subset = Y_mx[start_idx:end_idx]
            D_squared = _compute_pairwise_distances_mlx(Y_subset, Y_mx)
            
            # GPU-accelerated element-wise operations
            Q1 = 1 / (1 + mx.log(1 + D_squared))
            QQ1 = 1 / (1 + D_squared)
            
            # Convert sparse matrix operations to dense for GPU
            P_block = mx.array(P[idx, :].toarray().astype(np.float32))
            Pmat = -4 * P_block * Q1 * QQ1
            Qmat = -4 * Q1 ** 2 * QQ1
            
            # GPU-based diagonal correction using MLX operations
            # Compute row sums on GPU
            Pmat_row_sums = mx.sum(Pmat, axis=1)
            Qmat_row_sums = mx.sum(Qmat, axis=1)
            
            # Apply diagonal correction efficiently on GPU using proper MLX syntax
            for j in range(len_blk):
                diag_col = start_idx + j
                Pmat = Pmat.at[j, diag_col].add(-Pmat_row_sums[j])
                Qmat = Qmat.at[j, diag_col].add(-Qmat_row_sums[j])
            
            # GPU-accelerated matrix multiplication - keep everything on GPU
            Pgrad_block = Pmat @ Y_mx
            Qgrad_block = Qmat @ Y_mx
            
            # Update gradients directly on GPU using proper MLX indexing
            Pgrad_mx = Pgrad_mx.at[start_idx:end_idx].add(Pgrad_block - Pgrad_mx[start_idx:end_idx])
            Qgrad_mx = Qgrad_mx.at[start_idx:end_idx].add(Qgrad_block - Qgrad_mx[start_idx:end_idx])
            
            # Accumulate sumQ on GPU
            sumQ += float(mx.sum(Q1))
        
        # Update embedding - all operations stay on GPU
        grad_update = Pgrad_mx - Qgrad_mx / (sumQ - N) + (epoch - 1) / (epoch + 2) * preGrad_mx
        Y_mx = Y_mx - alpha * grad_update
        preGrad_mx = Pgrad_mx - Qgrad_mx / (sumQ - N)
    
    # Final conversion back to numpy only at the end
    Y_result = np.array(Y_mx)
    
    if verbose:
        print(f"   ✅ MLX learning_l completed (GPU-optimized): {X_samp.shape} -> {Y_result.shape}")
    
    return Y_result, k2
        
    

def _learning_l_torch(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, device='cpu', verbose=False):
    """
    Torch-based learning_l implementation for CUDA/CPU devices.
    """
    if verbose:
        print(f"   🚀 Using Torch learning_l on {device.upper()}")
    
    import torch
    
    # Convert to tensor
    X_samp = torch.tensor(np.asarray(X_samp, dtype=np.float32), device=device)
    N, dim = X_samp.shape

    # Compute k2 (same as CPU)
    if N < 9:
        k2 = N
    else:
        if N > 1000:
            k2 = int(np.ceil(np.log2(N)) + 18)
        elif N > 50:
            k2 = int(np.ceil(0.02 * N)) + 8
        else:
            k2 = 9

    # Initial P matrix computation (CPU for complexity, then transfer to GPU)
    X_np = X_samp.cpu().numpy()
    if k1 > 0:
        row = []
        col = []
        Pval = []
        knn_rnn_mat = rnn[get_knn[id_samp]]
        for i in range(N):
            snn_id = np.isin(get_knn[id_samp], get_knn[id_samp[i]]).astype(int)
            nn_id = np.where(np.max(snn_id, axis=1) == 1)[0]
            snn = np.zeros((1, N))
            snn[:, nn_id] = np.sum(knn_rnn_mat[nn_id] * snn_id[nn_id], axis=1)
            mod_dis = (1 - snn / max(np.max(snn), np.finfo(float).tiny)) ** agg_coef * cdist(X_np[i:i + 1, :], X_np)
            sort_dis = np.sort(mod_dis, axis=1)
            idx = np.argsort(mod_dis, axis=1)
            mean_samp_dis_squared = np.square(np.mean(sort_dis[0, :k2]))
            Pval.extend(
                np.exp(-0.5 * np.square(sort_dis[0, :k2]) / np.maximum(mean_samp_dis_squared, np.finfo(float).tiny)))
            row.extend((i * np.ones((k2, 1))).flatten().tolist())
            col.extend(idx[0, :k2])
        P = csr_matrix((Pval, (row, col)), shape=(N, N))
    else:
        if N > 5000 and dim > 50:
            xx = init_pca(X_np, no_dims, 0.8)
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(xx).kneighbors(xx)
        else:
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(X_np).kneighbors(X_np)
        mean_samp_dis_squared = np.square(np.mean(samp_dis, axis=1))
        Pval = np.exp(-0.5 * np.square(samp_dis) / np.maximum(mean_samp_dis_squared[:, np.newaxis], np.finfo(float).tiny))
        P = csr_matrix((Pval.flatten(), ([i for i in range(N) for _ in range(k2)], samp_knn.flatten())), shape=(N, N))
    
    P = (P + P.transpose()) / 2

    # Initialize embedding Y
    if initialize == 'le':
        deg = np.asarray(P.sum(axis=0)).ravel()
        d_inv_sqrt = np.zeros_like(deg, dtype=float)
        nz = deg > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        n = P.shape[0]
        
        def matvec(x):
            y = P @ (d_inv_sqrt * x)
            y = d_inv_sqrt * y
            return x - y
        
        Lop = LinearOperator((n, n), matvec=matvec, dtype=float)
        k = min(no_dims + 1, max(2, n - 1))
        vals, vecs = eigsh(Lop, k=k, which='SM')
        order = np.argsort(vals)
        Y = np.ascontiguousarray(vecs[:, order[1:no_dims+1]], dtype=np.float32)
    elif initialize == 'pca':
        Y = pca(X_np, no_dims).astype(np.float32)
    elif initialize == 'mds':
        Y = mds(X_np, no_dims, use_gpu=(device!='cpu'), verbose=False).astype(np.float32)

    # Convert to torch tensors for GPU computation
    Y_torch = torch.tensor(Y, device=device)
    P = P / (np.sum(P) - N)

    # Compute data blocks
    no_blocks = math.ceil(N / 3000)
    mark = np.zeros((no_blocks, 2))
    for i in range(no_blocks):
        mark[i, :] = [i * math.ceil(N / no_blocks), min((i + 1) * math.ceil(N / no_blocks) - 1, N - 1)]

    # Training loop with GPU acceleration
    max_alpha = 2.5 * N
    min_alpha = 2 * N
    warm_step = 10
    preGrad_torch = torch.zeros((N, no_dims), device=device)
    
    for epoch in tqdm(range(1, T_epoch + 1), desc=f"Training epochs (Torch-{device.upper()})", unit="epoch"):
        if epoch <= warm_step:
            alpha = max_alpha
        else:
            alpha = min_alpha + 0.5 * (max_alpha - min_alpha) * (
                        1 + np.cos(np.pi * ((epoch - warm_step) / (T_epoch - warm_step))))
        
        Pgrad_torch = torch.zeros((N, no_dims), device=device)
        Qgrad_torch = torch.zeros((N, no_dims), device=device)
        sumQ = 0
        
        for i in range(no_blocks):
            idx = list(range(int(mark[i, 0]), int(mark[i, 1]) + 1))
            idx_tensor = torch.tensor(idx, device=device)
            
            # GPU-accelerated distance computation
            Y_subset = Y_torch[idx_tensor]
            D_squared = _compute_pairwise_distances_torch(Y_subset, Y_torch, device)
            
            # GPU-accelerated element-wise operations
            Q1 = 1 / (1 + torch.log(1 + D_squared))
            QQ1 = 1 / (1 + D_squared)
            
            # Convert sparse matrix operations to dense for GPU
            P_block = torch.tensor(P[idx, :].toarray().astype(np.float32), device=device)
            Pmat = -4 * P_block * Q1 * QQ1
            Qmat = -4 * Q1 ** 2 * QQ1
            
            # Diagonal correction
            len_blk = len(idx)
            for j in range(len_blk):
                row_sum_P = torch.sum(Pmat[j])
                row_sum_Q = torch.sum(Qmat[j])
                Pmat[j, idx[0] + j] = Pmat[j, idx[0] + j] - row_sum_P
                Qmat[j, idx[0] + j] = Qmat[j, idx[0] + j] - row_sum_Q
            
            # GPU-accelerated matrix multiplication
            Pgrad_block = Pmat @ Y_torch
            Qgrad_block = Qmat @ Y_torch
            
            # Update gradients
            Pgrad_torch[idx_tensor] = Pgrad_block
            Qgrad_torch[idx_tensor] = Qgrad_block
            
            sumQ += torch.sum(Q1)
        
        # Update embedding
        grad_update = Pgrad_torch - Qgrad_torch / (sumQ - N) + (epoch - 1) / (epoch + 2) * preGrad_torch
        Y_torch = Y_torch - alpha * grad_update
        preGrad_torch = Pgrad_torch - Qgrad_torch / (sumQ - N)
    
    # Convert back to numpy
    Y_result = Y_torch.cpu().numpy()
    
    if verbose:
        print(f"   ✅ Torch learning_l completed: {X_samp.shape} -> {Y_result.shape}")
    
    return Y_result, k2
        
