"""Functions related to randomized SVD."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
from typing import Optional, Tuple

import torch
from torch import Tensor


def randomized_range_finder(
    inputs: Tensor,
    *,
    size: int,
    n_iter: int,
    power_iteration_normalizer: str,
    random_state: Optional[int],
) -> Tensor:
    """Compute an orthonormal matrix whose range approximates the range of inputs.

    Returns
    -------
    proj_mat : Tensor
        Orthonormal matrix whose range approximates the range of inputs.
    """

    def no_normalizer(inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Disable normalizer."""
        return inputs, None

    def lu_normalizer(inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """LU normalizer."""
        p_mat, l_mat, u_mat = torch.linalg.lu(inputs)
        return p_mat @ l_mat, u_mat

    if random_state is not None:
        torch.manual_seed(random_state)
    proj_mat = torch.randn(
        inputs.shape[-1], size, device=inputs.device, dtype=inputs.dtype
    )
    if power_iteration_normalizer == "auto":
        power_iteration_normalizer = "none" if n_iter <= 2 else "QR"
    qr_normalizer = torch.linalg.qr
    if power_iteration_normalizer == "QR":
        normalizer = qr_normalizer
    elif power_iteration_normalizer == "LU":
        normalizer = lu_normalizer
    else:
        normalizer = no_normalizer
    for _ in range(n_iter):
        proj_mat, _ = normalizer(inputs @ proj_mat)
        proj_mat, _ = normalizer(inputs.T @ proj_mat)
    proj_mat, _ = qr_normalizer(inputs @ proj_mat, mode="reduced")
    return proj_mat