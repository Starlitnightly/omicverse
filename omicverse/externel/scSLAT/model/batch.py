r"""
Batch correction module
"""
import os
from typing import List, Optional

import torch
from torch import Tensor
import numpy as np
from sklearn.utils.extmath import randomized_svd

from ..utils import get_free_gpu

def dual_pca(X:np.ndarray, Y:np.ndarray, 
            dim:Optional[int]=50,
            singular:Optional[bool]=False,
            backend:Optional[str]='sklearn',
            use_gpu:Optional[bool]=True
    ) -> List[Tensor]:
    r"""
    Dual PCA for batch correction
    
    Parameters:
    ----------
    X
        expr matrix 1 in shape of (cells, genes)
    Y
        expr matrix 2 in shape of (cells, genes)
    dim
        dimension of embedding
    singular
        if multiply the singular value
    backend
        backend to calculate singular value
    use_gpu
        if calculate in gpu
    Returns:
    ----------
    embd1, embd2: Tensors of embedding
    
    References:
    ----------
    Thanks Xin-Ming Tu for his [blog](https://xinmingtu.cn/blog/2022/CCA_dual_PCA/)
    """
    assert X.shape[1] == Y.shape[1]
    device = torch.device(f'cuda:{get_free_gpu()}' if torch.cuda.is_available() and use_gpu else 'cpu')
    X = torch.Tensor(X).to(device=device)
    Y = torch.Tensor(Y).to(device=device)
    cor_var = X @ Y.T
    if backend == 'torch':
        U, S, Vh = torch.linalg.svd(cor_var)
        if not singular:
            return U[:, :dim], Vh.T[:, :dim]
        Z_x = U[:, :dim] @ torch.sqrt(torch.diag(S[:dim]))
        Z_y = Vh.T[:, :dim] @ torch.sqrt(torch.diag(S[:dim]))
        return Z_x.cpu(), Z_y.cpu()
        # torch.dist(cor_var, Z_x @ Z_y.T)  # check the information loss
    elif backend == 'sklearn':
        cor_var = cor_var.cpu().numpy()
        U, S, Vh = randomized_svd(cor_var, n_components=dim, random_state=0)
        if not singular:
            return Tensor(U), Tensor(Vh.T)
        Z_x = U @ np.sqrt(np.diag(S))
        Z_y = Vh.T @ np.sqrt(np.diag(S))
        return Tensor(Z_x), Tensor(Z_y)