"""Functions related to n_component inference from fit data."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
# Modified from https://github.com/scikit-learn (BSD-3-Clause License)
# Copyright (c) Scikit-learn developers. All Rights Reserved.
from math import log
from typing import Union

import torch
from torch import Tensor

NComponentsType = Union[int, float, None, str]


def find_ncomponents(
    n_components: NComponentsType,
    inputs: Tensor,
    n_samples: int,
    explained_variance: Tensor,
    explained_variance_ratio: Tensor,
) -> int:
    """Find the number of components to keep."""
    if n_components is None:
        return min(inputs.shape[-2:])
    if isinstance(n_components, int) and int(n_components) == n_components:
        return n_components
    if isinstance(n_components, float) and 0.0 < n_components <= 1.0:
        variance_cumsum = torch.cumsum(explained_variance_ratio, dim=0)
        return (
            torch.searchsorted(
                variance_cumsum,
                n_components,
                right=True,
            ).item()
            + 1
        )
    if isinstance(n_components, str) and n_components == "mle":
        return infer_mle(
            spectrum=explained_variance,
            n_samples=n_samples,
        )
    raise ValueError(
        "`n_components` should be an int, a float in (0, 1] or 'mle'. "
        f"Found '{n_components}'."
    )


def infer_mle(spectrum: Tensor, n_samples: int) -> int:
    """Infer the dimension of a dataset with a given spectrum.

    Based on:
    `Minka, T. P.. "Automatic choice of dimensionality for PCA".
    In NIPS, pp. 598-604 <https://tminka.github.io/papers/pca/minka-pca.pdf>`_
    The returned value will be in [1, n_features - 1].
    """
    log_likelihood = torch.zeros_like(spectrum)
    n_features = spectrum.shape[0]
    # Prevent returning 0:
    log_likelihood[0] = -torch.inf
    for rank in range(1, spectrum.shape[0]):
        if spectrum[rank - 1] < 1e-15:
            # No need to compute the likelihood if the rank is too low
            log_likelihood[rank] = -torch.inf
        else:
            p_u = -rank * log(2.0)
            for i in range(1, rank + 1):
                p_u += (
                    torch.lgamma(torch.tensor(n_features - i + 1) / 2.0).item()
                    - log(torch.pi) * (n_features - i + 1) / 2.0
                )
            p_l = torch.sum(torch.log(spectrum[:rank]))
            p_l = -p_l * n_samples / 2.0

            var = max(1e-15, torch.sum(spectrum[rank:]) / (n_features - rank))
            p_v = -log(var) * n_samples * (n_features - rank) / 2.0

            mean = n_features * rank - rank * (rank + 1.0) / 2.0
            p_p = log(2.0 * torch.pi) * (mean + rank) / 2.0

            p_a = 0.0
            spectrum_ = torch.clone(spectrum)
            spectrum_[rank:n_features] = var
            for i in range(rank):
                for j in range(i + 1, spectrum.shape[0]):
                    p_a += log(
                        (spectrum[i] - spectrum[j])
                        * (1.0 / spectrum_[j] - 1.0 / spectrum_[i])
                    ) + log(n_samples)

            log_likelihood[rank] = (
                p_u + p_l + p_v + p_p - p_a / 2.0 - rank * log(n_samples) / 2.0
            )
    return torch.argmax(log_likelihood).item()