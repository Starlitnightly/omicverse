"""Functions related to SVD."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
# Inspired from https://github.com/scikit-learn (BSD-3-Clause License)
# Copyright (c) Scikit-learn developers. All Rights Reserved.
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from .ncompo import NComponentsType
from .random_svd import randomized_range_finder


def choose_svd_solver(inputs: Tensor, n_components: NComponentsType) -> str:
    """Choose the SVD solver based on the input shape."""
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