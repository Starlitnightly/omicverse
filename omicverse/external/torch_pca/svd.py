"""Functions related to SVD."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
# Inspired from https://github.com/scikit-learn (BSD-3-Clause License)
# Copyright (c) Scikit-learn developers. All Rights Reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from scipy import sparse as sp

from .ncompo import NComponentsType
from .random_svd import randomized_range_finder


def choose_svd_solver(
    inputs: Union[Tensor, sp.spmatrix],
    n_components: NComponentsType,
    is_sparse: bool = False
) -> str:
    """Choose the SVD solver based on the input shape and type.

    Parameters
    ----------
    inputs : Tensor or scipy.sparse matrix
        Input data
    n_components : NComponentsType
        Number of components
    is_sparse : bool
        Whether input is a scipy sparse matrix

    Returns
    -------
    str
        Selected solver name ('full', 'covariance_eigh', 'randomized', or 'arpack')
    """
    # For sparse matrices, prefer ARPACK
    if is_sparse:
        return "arpack"

    # Original logic for dense tensors
    if inputs.shape[-1] <= 1_000 and inputs.shape[-2] >= 10 * inputs.shape[-1]:
        return "covariance_eigh"
    if max(inputs.shape[-2:]) <= 500 or n_components == "mle":
        return "full"
    if isinstance(n_components, int) and 1 <= n_components < 0.8 * min(inputs.shape):
        return "randomized"
    return "full"


def randomized_svd(
    inputs: Tensor,
    n_components: int,
    n_oversamples: int,
    n_iter: Union[str, int],
    power_iteration_normalizer: str,
    random_state: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Randomized SVD using Halko, et al. (2009) method.

    Returns
    -------
    u_mat : Tensor
        Left singular vectors.
    coefs : Tensor
        Singular values.
    vh_mat : Tensor
        Right singular vectors.

    References
    ----------
    .. [1] :arxiv:`"Finding structure with randomness:
      Stochastic algorithms for constructing approximate matrix decompositions"
      <0909.4061>`
      Halko, et al. (2009)

    .. [2] A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    .. [3] An implementation of a randomized algorithm for principal component
      analysis A. Szlam et al. 2014
    """
    n_random = n_components + n_oversamples
    n_samples, n_features = inputs.shape
    if n_iter == "auto":
        # Compromise found by Sklearn
        n_iter = 7 if n_components < 0.1 * min(inputs.shape) else 4
    if isinstance(n_iter, str):
        raise ValueError(
            f"`iterated_power` must be an integer or 'auto'. Found '{n_iter}'."
        )
    if n_samples < n_features:
        inputs = inputs.T
    proj_mat = randomized_range_finder(
        inputs,
        size=n_random,
        n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
    )
    pseudo_inputs = proj_mat.T @ inputs
    u_mat, coefs, vh_mat = torch.linalg.svd(pseudo_inputs, full_matrices=False)
    u_mat = proj_mat @ u_mat
    if n_samples < n_features:
        return (
            vh_mat[:n_components, :].T,
            coefs[:n_components],
            u_mat[:, :n_components].T,
        )
    return u_mat[:, :n_components], coefs[:n_components], vh_mat[:n_components, :]


def arpack_svd(
    inputs: Union[sp.csr_matrix, sp.csc_matrix],
    n_components: int,
    mean_vector: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ARPACK-based SVD for sparse matrices using scipy.sparse.linalg.svds.

    Parameters
    ----------
    inputs : scipy.sparse.csr_matrix or csc_matrix
        Sparse input matrix (n_samples, n_features)
    n_components : int
        Number of components to compute
    mean_vector : np.ndarray, optional
        Mean vector for centering. If None, data is not centered.
        Note: Centering sparse matrices destroys sparsity, so we use
        a LinearOperator to apply centering implicitly.
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    u_mat : np.ndarray
        Left singular vectors (n_samples, n_components)
    coefs : np.ndarray
        Singular values (n_components,)
    vh_mat : np.ndarray
        Right singular vectors (n_components, n_features)

    Notes
    -----
    ARPACK computes SVD using Implicitly Restarted Lanczos Method.
    If mean_vector is provided, the matrix is centered implicitly
    without densifying (using LinearOperator).

    References
    ----------
    .. [1] scipy.sparse.linalg.svds documentation
    .. [2] Lehoucq, R. B., Sorensen, D. C., & Yang, C. (1998).
           ARPACK users' guide: solution of large-scale eigenvalue problems
           with implicitly restarted Arnoldi methods.
    """
    from scipy.sparse.linalg import svds, LinearOperator

    n_samples, n_features = inputs.shape

    # Validate n_components
    max_components = min(n_samples, n_features) - 1
    if n_components >= max_components:
        raise ValueError(
            f"n_components must be < min(n_samples, n_features) for ARPACK. "
            f"Got n_components={n_components}, but max is {max_components}"
        )

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)
        v0 = np.random.uniform(-1.0, 1.0, min(n_samples, n_features))
    else:
        v0 = None

    # If centering is needed, use LinearOperator to avoid densification
    if mean_vector is not None:
        # Create implicit centered matrix using LinearOperator
        mean_vector = mean_vector.ravel()

        def matvec(x):
            # Compute (A - mean_row) @ x where each row of A is centered by mean
            # (A - ones @ mean.T) @ x = A @ x - ones * (mean @ x)
            # Result shape: (n_samples,)
            # Force x to 1D to avoid broadcasting issues
            x = np.asarray(x).ravel()
            result = inputs @ x - np.dot(mean_vector, x)
            return result

        def rmatvec(x):
            # Compute (A - mean_row).T @ x
            # = A.T @ x - mean @ (ones.T @ x) = A.T @ x - mean * sum(x)
            # Result shape: (n_features,)
            # Force x to 1D to avoid broadcasting issues
            x = np.asarray(x).ravel()
            result = inputs.T @ x - mean_vector * np.sum(x)
            return result

        centered_op = LinearOperator(
            shape=(n_samples, n_features),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=inputs.dtype
        )

        # Compute SVD on centered operator
        u_mat, coefs, vh_mat = svds(
            centered_op,
            k=n_components,
            v0=v0,
            return_singular_vectors=True
        )
    else:
        # Compute SVD directly on sparse matrix
        u_mat, coefs, vh_mat = svds(
            inputs,
            k=n_components,
            v0=v0,
            return_singular_vectors=True
        )

    # svds returns singular values in ascending order, need to reverse
    # to match torch.linalg.svd convention (descending order)
    # Use .copy() to avoid negative strides (ascontiguousarray doesn't always work with PyTorch)
    u_mat = u_mat[:, ::-1].copy()
    coefs = coefs[::-1].copy()
    vh_mat = vh_mat[::-1, :].copy()

    return u_mat, coefs, vh_mat


def svd_flip(u_mat: Optional[Tensor], vh_mat: Tensor) -> Tuple[Tensor, Tensor]:
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that
    the loadings in the rows in V^H that are largest in absolute
    value are always positive.

    Parameters
    ----------
    u_mat : ndarray
        U matrix in the SVD output (U * diag(S) * V^H)

    vh_mat : ndarray
        V^H matrix in the SVD output (U * diag(S) * V^H)

    Returns
    -------
    u_mat : ndarray
        Adjusted U v.

    vh_mat : ndarray
         Adjusted V^H matrix.
    """
    max_abs_v_rows = torch.argmax(torch.abs(vh_mat), dim=1)
    shift = torch.arange(vh_mat.shape[0], device=vh_mat.device)
    indices = max_abs_v_rows + shift * vh_mat.shape[1]
    flat_vh = torch.reshape(vh_mat, (-1,))
    signs = torch.sign(torch.take_along_dim(flat_vh, indices, dim=0))
    if u_mat is not None:
        u_mat *= signs[None, :]
    vh_mat *= signs[:, None]
    return u_mat, vh_mat