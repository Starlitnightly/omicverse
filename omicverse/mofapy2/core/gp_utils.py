"""
Utility funcions for the Gaussian process calculations of the Sigma Node
"""

import scipy as s
import scipy.spatial as SS
import numpy as np
# import gpytorch

def SE(X, l, zeta = 1e-3):
    """
    squared exponential covariance function on input X with lengthscale l
    """
    tmp = SS.distance.pdist(X, 'euclidean') ** 2.
    tmp = SS.distance.squareform(tmp)
    if l == 0:
        cov = (1-zeta) * (tmp ==0).astype(float)
    else:
        cov = (1-zeta) * np.exp(-tmp/ (2 * l ** 2.))
    cov += zeta * np.eye(X.shape[0])

    return cov

def Cauchy(X, l, zeta =  1e-3):
    """
    Cauchy covariance function on input X with lengthscale l
    """
    tmp = SS.distance.pdist(X,'euclidean') ** 2.
    tmp = SS.distance.squareform(tmp)
    if l == 0:
        cov = (1-zeta) * (tmp ==0).astype(float)
    else:
        cov = (1-zeta) * 1/(1 + tmp/ (l ** 2.))
    cov += zeta * np.eye(X.shape[0])

    return cov


# def PE(X, period, zeta =  1e-3, ls = 1):
#     """
#     periodic covariance function on input X with period period and lengthscale ls
#     """
#     dist_per_dim = [SS.distance.pdist(X[:,i], 'euclidean') for i in range(X.shape[1])]
#     dist_per_dim = [SS.distance.squareform(tmp) for tmp in dist_per_dim]
#     if period == 0:
#         arg_per_dim =[(dd ==0).astype(float) for dd in dist_per_dim]
#     else:
#         arg_per_dim = [np.pi * dd / period for dd in dist_per_dim]  # argument of sin
#     sum_coord = sum([-0.5 * (np.sin(arg) ** 2 / ls ** 2) for arg in arg_per_dim])  # term in exponent of peridoic kernel
#     cov = (1-zeta) * np.exp(sum_coord)
#
#     cov += zeta * np.eye(X.shape[0])
#
#     return cov


def get_l_limits(X, idx = None):
    """
    Get boundaries for the grid of lengthscales to optimize over (as implemented in spatialDE) 
    Boundaries of the grid are the shortest observed distance, divided by 2, and the longest observed distance multiplied by 2
    """
    if not idx is None: # calculate based on distances in the reference group
        X = X[idx, :]
    tmp = SS.distance.pdist(X,'euclidean')**2.
    tmp = SS.distance.squareform(tmp)
    tmp_vals = np.unique(tmp.flatten())
    tmp_vals = tmp_vals[tmp_vals > 1e-8]

    l_min = np.sqrt(tmp_vals.min()) / 2.
    l_max = np.sqrt(tmp_vals.max()) * 2.

    return l_min, l_max


def get_l_grid(X, n_grid = 5, idx = None):
    """
    Function to get points in a logarithmic grid for lengthscales (as implemented in spatialDE)
    """
    l_min, l_max = get_l_limits(X, idx)
    return np.logspace(np.log10(l_min), np.log10(l_max), n_grid)


# # does not account for posterior variance of z, need to adapt likelihood
# # K = K_GG \otimes K_CC
# class MultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, n_tasks, rank_x):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.MultitaskMean(
#             gpytorch.means.ConstantMean(), num_tasks=n_tasks
#         )
#         self.covar_module = gpytorch.kernels.MultitaskKernel(
#             gpytorch.kernels.RBFKernel(), num_tasks=n_tasks, rank=rank_x
#         )
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# utility functions from spatialDE
# def covar_rescaling_factor(C):
#     """
#     Returns the rescaling factor for the Gower normalizion on covariance matrix C
#     the rescaled covariance matrix has sample variance of 1 s.t. one obtaines an unbiased estimate of the variance explained
#     (based on https://github.com/PMBio/limix/blob/master/limix/utils/preprocess.py - covar_rescaling_factor_efficient;
#     https://limix.readthedocs.io/en/stable/api/limix.qc.normalise_covariance.html)
#     """
#     n = C.shape[0]
#     P = np.eye(n) - np.ones((n,n))/float(n) # Gower’s centering matrix
#     CP = C - C.mean(0)[:, np.newaxis]
#     trPCP = np.sum(P * CP) # trace of doubly centered covariance matrix
#     r = (n-1) / trPCP
#     return r
#
def covar_to_corr(C):
    """
    Transforms the covariance matrix into a correlation matrix
    """
    Cdiag = np.diag(C)
    Ccor = np.diag(1/np.sqrt(Cdiag)) @ C @ np.diag(1/np.sqrt(Cdiag))
    return Ccor


def set_inducing_points(data, sample_cov, groups, dims, n_inducing, random = False, seed_inducing = 0):
    """
    Method to select samples to use as inducing points for the GP
    """

    missing_sample_per_view = np.ones((dims["N"], dims["M"]))
    for m in range(len(data)):
        missing_sample_per_view[:,m] = np.isnan(data[m]).all(axis = 1)
    nonmissing_samples = np.where(missing_sample_per_view.sum(axis = 1) != dims["M"])[0]
    N_nonmissing = len(nonmissing_samples)
    n_inducing = min(n_inducing, N_nonmissing)
    if random:
        if not seed_inducing is None:
            np.random.seed(int(seed_inducing))
        idx_inducing = np.random.choice(dims["N"], n_inducing, replace = False)
        idx_inducing.sort()
    else:
        N = dims["N"]
        loc = sample_cov.sum(axis = 1)
        nonmissing_samples_tiesshuffled = nonmissing_samples[np.lexsort((np.random.random(N_nonmissing), loc[nonmissing_samples]))] # shuffle tienp.randomly (e.g. between groups)
        grid_ix = np.floor(np.arange(0, N_nonmissing, step=N_nonmissing / n_inducing)).astype('int')
        if grid_ix[-1] == N_nonmissing: # avoid out of bound
            grid_ix = grid_ix[:-1]
        idx_inducing = nonmissing_samples_tiesshuffled[grid_ix]

    return idx_inducing
