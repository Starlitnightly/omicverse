import warnings

import numpy as np
import torch
from pyro.distributions import Gamma, Poisson
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all, probs_to_logits

# NB distribution parameterisation with mu and theta parametrisation is copied over from scVI:
#    Copyright (c) 2020 Romain Lopez, Adam Gayoso, Galen Xing, Yosef Lab
#    All rights reserved


def log_nb_positive(value, mu, theta, eps=1e-8):
    """NB loss with mu and theta parametrisation - Copied over from scVI
    Note: All inputs should be torch Tensors
    log likelihood (scalar) of a minibatch according to a nb model.

    Variables:

    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + value * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(value + theta)
        - torch.lgamma(theta)
        - torch.lgamma(value + 1)
    )

    return res


def log_nb_pymc3(value, mu, alpha, eps=1e-8):
    """NB loss with mu and alpha parametrisation as defined in pymc3"""

    def logpow(x, m):
        """
        Calculates log(x**m) since m*log(x) will fail when m, x = 0.
        """
        return torch.where(
            torch.eq(x, torch.tensor(0)),
            torch.where(torch.eq(m, torch.tensor(0)), torch.tensor(0.0), torch.tensor(-np.inf)),
            m * torch.log(x),
        )

    def factln(n):
        return torch.lgamma(n + 1)

    def binomln(n, k):
        return factln(n) - factln(k) - factln(n - k)

    negbinom = (
        binomln(value + alpha - 1, value) + logpow(mu / (mu + alpha), value) + logpow(alpha / (mu + alpha), alpha)
    )

    # Return Poisson when alpha gets very large.
    return torch.where(torch.gt(alpha, 1e10), (mu.log() * value) - mu - (value + 1).lgamma(), negbinom)


def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""NB parameterizations conversion  - Copied over from scVI
    :param mu: mean of the NB distribution.
    :param theta: inverse overdispersion.
    :param eps: constant used for numerical log stability.
    :return: the number of failures until the experiment is stopped
        and the success probability.
    """
    assert (mu is None) == (
        theta is None
    ), "If using the mu/theta NB parameterization, both parameters must be specified"
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits


def _convert_counts_logits_to_mean_disp(total_count, logits):
    """NB parameterizations conversion  - Copied over from scVI
    :param total_count: Number of failures until the experiment is stopped.
    :param logits: success logits.
    :return: the mean and inverse overdispersion of the NB distribution.
    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta


class NegativeBinomial(Distribution):
    r"""Negative Binomial(NB) distribution using two parameterizations:  - Copied over from scVI

    - (`total_count`, `probs`) where `total_count` is the number of failures
        until the experiment is stopped
        and `probs` the success probability.
    - The (`mu`, `theta`) parameterization is the one used by scVI. These parameters respectively
    control the mean and overdispersion of the distribution.
    `_convert_mean_disp_to_counts_logits` and `_convert_counts_logits_to_mean_disp` provide ways to convert
    one parameterization to another.

    """
    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: torch.Tensor = None,
        probs: torch.Tensor = None,
        logits: torch.Tensor = None,
        mu: torch.Tensor = None,
        theta: torch.Tensor = None,
        validate_args=True,
    ):
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation for more information."
            )

        using_param_1 = total_count is not None and (logits is not None or probs is not None)
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            mu, theta = _convert_counts_logits_to_mean_disp(total_count, logits)
        else:
            mu, theta = broadcast_all(mu, theta)
        self.mu = mu
        self.theta = theta
        super().__init__(validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        gamma_d = self._gamma()
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        counts = Poisson(l_train).sample()  # Shape : (n_samples, n_cells_batch, n_genes)
        return counts

    def log_prob(self, value):
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                )
        return log_nb_positive(value, mu=self.mu, theta=self.theta, eps=self._eps)
        # return log_nb_pymc3(value, mu=self.mu, alpha=self.theta, eps=self._eps)

    def _gamma(self):
        concentration = self.theta
        rate = self.theta / self.mu
        # Important remark: Gamma is parametrized by the rate = 1/scale!
        gamma_d = Gamma(concentration=concentration, rate=rate)
        return gamma_d
