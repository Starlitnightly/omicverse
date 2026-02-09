"""Utilities for converting between scipy sparse matrices and PyTorch tensors."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
from typing import Union
import numpy as np
import torch
from torch import Tensor
from scipy import sparse as sp


def scipy_sparse_to_torch_sparse(
    matrix: Union[sp.csr_matrix, sp.csc_matrix],
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert scipy sparse matrix to PyTorch sparse COO tensor.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix or csc_matrix
        Sparse matrix to convert
    device : torch.device
        Target device for tensor (can be 'cuda' for GPU)
    dtype : torch.dtype
        Target dtype for tensor

    Returns
    -------
    torch.Tensor
        Sparse PyTorch tensor in COO format

    Notes
    -----
    This preserves sparsity and can be placed on GPU for accelerated operations.
    """
    # Convert to COO format for efficient conversion
    coo = matrix.tocoo()

    # Create indices tensor
    indices = torch.from_numpy(
        np.vstack([coo.row, coo.col])
    ).long()

    # Create values tensor
    values = torch.from_numpy(coo.data).to(dtype=dtype)

    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=coo.shape,
        dtype=dtype,
        device=device
    )

    return sparse_tensor.coalesce()


def scipy_sparse_to_torch_dense(
    matrix: Union[sp.csr_matrix, sp.csc_matrix],
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> Tensor:
    """Convert scipy sparse matrix to dense PyTorch tensor.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix or csc_matrix
        Sparse matrix to convert
    device : torch.device
        Target device for tensor
    dtype : torch.dtype
        Target dtype for tensor

    Returns
    -------
    Tensor
        Dense PyTorch tensor

    Notes
    -----
    This function densifies the sparse matrix. Use only when necessary
    (e.g., for small matrices like components, singular values).
    """
    # Convert to dense numpy array
    dense_array = matrix.toarray() if sp.issparse(matrix) else matrix

    # Convert to torch tensor
    tensor = torch.from_numpy(dense_array).to(dtype=dtype, device=device)

    return tensor


def numpy_to_torch(
    array: np.ndarray,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> Tensor:
    """Convert numpy array to PyTorch tensor.

    Parameters
    ----------
    array : np.ndarray
        Numpy array to convert
    device : torch.device
        Target device for tensor
    dtype : torch.dtype
        Target dtype for tensor

    Returns
    -------
    Tensor
        PyTorch tensor

    Notes
    -----
    If the numpy array has negative strides (e.g., from slicing with ::-1),
    a copy is made to ensure compatibility with PyTorch.
    """
    # Handle negative strides or non-contiguous arrays by making a copy
    # This ensures compatibility with torch.from_numpy
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    return torch.from_numpy(array).to(dtype=dtype, device=device)


def compute_sparse_mean_torch(
    sparse_tensor: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """Compute mean along dimension for PyTorch sparse tensor.

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        Sparse input tensor (in COO format)
    dim : int
        Dimension along which to compute mean (0 for column means, 1 for row means)

    Returns
    -------
    torch.Tensor
        Mean values (dense tensor on same device as input)

    Notes
    -----
    This operates on GPU if the sparse tensor is on GPU.
    """
    # Sum along dimension and divide by size
    if not sparse_tensor.is_sparse:
        return sparse_tensor.mean(dim=dim)

    # For sparse tensor, convert to dense for mean (more efficient for this operation)
    dense = sparse_tensor.to_dense()
    return dense.mean(dim=dim)


def compute_sparse_mean(
    matrix: Union[sp.csr_matrix, sp.csc_matrix],
    axis: int = 0
) -> np.ndarray:
    """Compute mean along axis for scipy sparse matrix efficiently.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        Sparse input matrix
    axis : int
        Axis along which to compute mean (0 for column means, 1 for row means)

    Returns
    -------
    np.ndarray
        Mean values
    """
    return np.asarray(matrix.mean(axis=axis)).ravel()


def torch_sparse_to_scipy(
    sparse_tensor: torch.Tensor
) -> sp.csr_matrix:
    """Convert PyTorch sparse tensor to scipy CSR matrix.

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        PyTorch sparse tensor (COO format)

    Returns
    -------
    scipy.sparse.csr_matrix
        Scipy sparse matrix in CSR format

    Notes
    -----
    This moves the tensor to CPU for conversion.
    Used when we need to call scipy functions (e.g., svds).
    """
    # Ensure tensor is coalesced and on CPU
    sparse_tensor = sparse_tensor.coalesce().cpu()

    # Get indices and values
    indices = sparse_tensor.indices().numpy()
    values = sparse_tensor.values().numpy()
    shape = sparse_tensor.shape

    # Create COO matrix then convert to CSR
    coo = sp.coo_matrix(
        (values, (indices[0], indices[1])),
        shape=shape
    )
    return coo.tocsr()


def compute_sparse_variance(
    matrix: Union[sp.csr_matrix, sp.csc_matrix],
    n_samples: int
) -> float:
    """Compute total variance for sparse matrix.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        Sparse input matrix (should be centered for accurate variance)
    n_samples : int
        Number of samples

    Returns
    -------
    float
        Total variance

    Notes
    -----
    For a centered sparse matrix X, variance is computed as:
    var = sum(X_ij^2) / (n_samples - 1)

    This uses the efficient sparse matrix multiply operation.
    """
    # For sparse matrix, compute sum of squared elements efficiently
    return matrix.multiply(matrix).sum() / (n_samples - 1)
