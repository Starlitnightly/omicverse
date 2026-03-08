''' Main underlying functions for SpatialDE functionality.

Optimizations over the original:
  - null_fits / const_fits fully vectorized (no Python gene loops)
  - lengthscale_fits: batch UTy = exp_tab.T @ U, then joblib-parallel lbfgsb
  - dyn_de / run accept n_jobs kwarg (default 1 = sequential)
'''
import sys
import logging
from time import time
import warnings

import numpy as np
from scipy import optimize
from scipy import linalg
from scipy import stats
from scipy.misc import derivative
from scipy.special import logsumexp

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm

import pandas as pd

from .util import qvalue


def get_l_limits(X):
    X = np.asarray(X)
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    R_vals = np.unique(R2.flatten())
    R_vals = R_vals[R_vals > 1e-8]

    l_min = np.sqrt(R_vals.min()) / 2.
    l_max = np.sqrt(R_vals.max()) * 2.

    return l_min, l_max

## Kernels ##

def SE_kernel(X, l):
    X = np.asarray(X)
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.exp(-R2 / (2 * l ** 2))


def linear_kernel(X):
    K = np.dot(X, X.T)
    return K / K.max()


def cosine_kernel(X, p):
    X = np.asarray(X)
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.cos(2 * np.pi * np.sqrt(R2) / p)


def gower_scaling_factor(K):
    n = K.shape[0]
    P = np.eye(n) - np.ones((n, n)) / n
    KP = K - K.mean(0)[:, np.newaxis]
    trPKP = np.sum(P * KP)
    return trPKP / (n - 1)


def factor(K):
    S, U = np.linalg.eigh(K)
    return U, np.clip(S, 1e-8, None)


def get_UT1(U):
    return U.sum(0)


def get_UTy(U, y):
    return y.dot(U)


def mu_hat(delta, UTy, UT1, S, n, Yvar=None):
    if Yvar is None:
        Yvar = np.ones_like(S)
    UT1_scaled = UT1 / (S + delta * Yvar)
    return UT1_scaled.dot(UTy) / UT1_scaled.dot(UT1)


def s2_t_hat(delta, UTy, S, n, Yvar=None):
    if Yvar is None:
        Yvar = np.ones_like(S)
    UTy_scaled = UTy / (S + delta * Yvar)
    return UTy_scaled.dot(UTy) / n


def LL(delta, UTy, UT1, S, n, Yvar=None):
    mu_h = mu_hat(delta, UTy, UT1, S, n, Yvar)
    if Yvar is None:
        Yvar = np.ones_like(S)
    sum_1 = (np.square(UTy - UT1 * mu_h) / (S + delta * Yvar)).sum()
    sum_2 = np.log(S + delta * Yvar).sum()
    with np.errstate(divide='ignore'):
        return -0.5 * (n * np.log(2 * np.pi) + n * np.log(sum_1 / n) + sum_2 + n)


def logdelta_prior_lpdf(log_delta):
    s2p = 100.
    return -np.log(np.sqrt(2 * np.pi * s2p)) - np.square(log_delta - 20.) / (2 * s2p)


def make_objective(UTy, UT1, S, n, Yvar=None):
    def LL_obj(log_delta):
        return -LL(np.exp(log_delta), UTy, UT1, S, n, Yvar)
    return LL_obj


def brent_max_LL(UTy, UT1, S, n):
    LL_obj = make_objective(UTy, UT1, S, n)
    o = optimize.minimize_scalar(LL_obj, bounds=[-10, 10], method='bounded',
                                 options={'maxiter': 32})
    max_ll = -o.fun
    max_delta = np.exp(o.x)
    return max_ll, max_delta, mu_hat(max_delta, UTy, UT1, S, n), s2_t_hat(max_delta, UTy, S, n)


def lbfgsb_max_LL(UTy, UT1, S, n, Yvar=None):
    LL_obj = make_objective(UTy, UT1, S, n, Yvar)
    min_boundary = -10
    max_boundary = 20.
    x, f, d = optimize.fmin_l_bfgs_b(LL_obj, 0., approx_grad=True,
                                      bounds=[(min_boundary, max_boundary)],
                                      maxfun=64, factr=1e12, epsilon=1e-4)
    max_ll = -f
    max_delta = np.exp(x[0])

    for bd in (max_boundary, min_boundary):
        bll = -LL_obj(bd)
        if bll > max_ll:
            max_ll = bll
            max_delta = np.exp(bd)

    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n, Yvar)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, S, n, Yvar)
    s2_logdelta = 1. / (derivative(LL_obj, np.log(max_delta), n=2) ** 2)

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat, s2_logdelta


def search_max_LL(UTy, UT1, S, n, num=32):
    min_obj = np.inf
    max_log_delta = np.nan
    LL_obj = make_objective(UTy, UT1, S, n)
    for log_delta in np.linspace(start=-10, stop=20, num=num):
        cur_obj = LL_obj(log_delta)
        if cur_obj < min_obj:
            min_obj = cur_obj
            max_log_delta = log_delta
    max_delta = np.exp(max_log_delta)
    return (-min_obj, max_delta,
            mu_hat(max_delta, UTy, UT1, S, n),
            s2_t_hat(max_delta, UTy, S, n))


def make_FSV(UTy, S, n, Gower):
    def FSV(log_delta):
        s2_t = s2_t_hat(np.exp(log_delta), UTy, S, n)
        s2_t_g = s2_t * Gower
        return s2_t_g / (s2_t_g + np.exp(log_delta) * s2_t)
    return FSV


def _fit_one_gene(g, gene_name, UTy, UT1, S, n, Gower):
    """Fit a single gene's GP model; used by both serial and parallel paths."""
    t0 = time()
    max_reg_ll, max_delta, max_mu_hat, max_s2_t_hat, s2_logdelta = lbfgsb_max_LL(UTy, UT1, S, n)
    t = time() - t0

    FSV = make_FSV(UTy, S, n, Gower)
    s2_FSV = derivative(FSV, np.log(max_delta), n=1) ** 2 * s2_logdelta

    return {
        'g': gene_name,
        'max_ll': max_reg_ll,
        'max_delta': max_delta,
        'max_mu_hat': max_mu_hat,
        'max_s2_t_hat': max_s2_t_hat,
        'time': t,
        'n': n,
        'FSV': FSV(np.log(max_delta)),
        's2_FSV': s2_FSV,
        's2_logdelta': s2_logdelta,
    }


def lengthscale_fits(exp_tab, U, UT1, S, Gower, n_jobs=1, num=64):
    '''Fit GPs after pre-processing for a particular lengthscale.

    Optimizations vs original:
      - Batch UTy computation: one matrix multiply instead of G dot products
      - n_jobs > 1: joblib thread-pool over genes (lbfgsb releases the GIL)
    '''
    n, G = exp_tab.shape
    vals = exp_tab.values  # avoid repeated pandas overhead

    # Batch: compute UTy for ALL genes at once → G × n
    UTY = vals.T.dot(U)   # equivalent to: [y_g.dot(U) for each gene]

    gene_names = list(exp_tab.columns)

    if n_jobs == 1:
        results = []
        for g in tqdm(range(G), leave=False):
            results.append(_fit_one_gene(g, gene_names[g], UTY[g], UT1, S, n, Gower))
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_fit_one_gene)(g, gene_names[g], UTY[g], UT1, S, n, Gower)
            for g in tqdm(range(G), leave=False)
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Vectorized baseline models (no Python gene loop)
# ---------------------------------------------------------------------------

def null_fits(exp_tab):
    '''Get maximum LL for null model (vectorized over all genes).'''
    n, G = exp_tab.shape
    vals = exp_tab.values
    max_s2_e = np.square(vals).sum(0) / n          # shape (G,)
    max_ll = -0.5 * (n * np.log(2 * np.pi) + n + n * np.log(max_s2_e))
    return pd.DataFrame({
        'g': exp_tab.columns,
        'max_ll': max_ll,
        'max_delta': np.inf,
        'max_mu_hat': 0.,
        'max_s2_t_hat': 0.,
        'time': 0,
        'n': n,
    })


def const_fits(exp_tab):
    '''Get maximum LL for const model (vectorized over all genes).'''
    n, G = exp_tab.shape
    vals = exp_tab.values
    means = vals.mean(0)                            # shape (G,)
    vars_ = vals.var(0)
    vars_ = np.where(vars_ == 0, 1e-10, vars_)     # guard against zero variance
    sum1 = np.square(vals - means).sum(0)
    max_ll = -0.5 * (n * np.log(vars_) + sum1 / vars_ + n * np.log(2 * np.pi))
    return pd.DataFrame({
        'g': exp_tab.columns,
        'max_ll': max_ll,
        'max_delta': np.inf,
        'max_mu_hat': means,
        'max_s2_t_hat': 0.,
        'time': 0,
        'n': n,
    })


def simulate_const_model(MLL_params, N):
    dfm = np.zeros((N, MLL_params.shape[0]))
    for i, params in enumerate(MLL_params.iterrows()):
        params = params[1]
        s2_e = params.max_s2_t_hat * params.max_delta
        dfm[:, i] = np.random.normal(params.max_mu_hat, s2_e, N)
    return pd.DataFrame(dfm)


def get_mll_results(results, null_model='const'):
    null_lls = results.query('model == "{}"'.format(null_model))[['g', 'max_ll']]
    model_results = results.query('model != "{}"'.format(null_model))
    model_results = model_results[
        model_results.groupby(['g'])['max_ll'].transform(max) == model_results['max_ll']
    ]
    mll_results = model_results.merge(null_lls, on='g', suffixes=('', '_null'))
    mll_results['LLR'] = mll_results['max_ll'] - mll_results['max_ll_null']
    return mll_results


def dyn_de(X, exp_tab, kernel_space=None, n_jobs=1):
    if kernel_space is None:
        kernel_space = {'SE': [5., 25., 50.]}

    results = []

    if 'null' in kernel_space:
        result = null_fits(exp_tab)
        result['l'] = np.nan
        result['M'] = 1
        result['model'] = 'null'
        results.append(result)

    if 'const' in kernel_space:
        result = const_fits(exp_tab)
        result['l'] = np.nan
        result['M'] = 2
        result['model'] = 'const'
        results.append(result)

    logging.info("Pre-calculating USU^T = K's ...")
    US_mats = []
    t0 = time()
    X = np.asarray(X)

    if 'linear' in kernel_space:
        K = linear_kernel(X)
        U, S = factor(K)
        US_mats.append({'model': 'linear', 'M': 3, 'l': np.nan,
                        'U': U, 'S': S, 'UT1': get_UT1(U), 'Gower': gower_scaling_factor(K)})

    if 'SE' in kernel_space:
        for lengthscale in kernel_space['SE']:
            K = SE_kernel(X, lengthscale)
            U, S = factor(K)
            US_mats.append({'model': 'SE', 'M': 4, 'l': lengthscale,
                            'U': U, 'S': S, 'UT1': get_UT1(U), 'Gower': gower_scaling_factor(K)})

    if 'PER' in kernel_space:
        for period in kernel_space['PER']:
            K = cosine_kernel(X, period)
            U, S = factor(K)
            US_mats.append({'model': 'PER', 'M': 4, 'l': period,
                            'U': U, 'S': S, 'UT1': get_UT1(U), 'Gower': gower_scaling_factor(K)})

    logging.info('Done: {0:.2f}s'.format(time() - t0))
    logging.info('Fitting gene models')

    for cov in tqdm(US_mats, desc='Models: '):
        result = lengthscale_fits(exp_tab, cov['U'], cov['UT1'], cov['S'], cov['Gower'],
                                  n_jobs=n_jobs)
        result['l'] = cov['l']
        result['M'] = cov['M']
        result['model'] = cov['model']
        results.append(result)

    logging.info('Finished fitting {} models to {} genes'.format(len(US_mats), exp_tab.shape[1]))

    results = pd.concat(results, sort=True).reset_index(drop=True)
    results['BIC'] = -2 * results['max_ll'] + results['M'] * np.log(results['n'])
    return results


def run(X, exp_tab, kernel_space=None, n_jobs=1):
    '''Perform SpatialDE test.

    Parameters
    ----------
    X : array-like, shape (n, 2)
        Spatial coordinates.
    exp_tab : DataFrame, shape (n, G)
        Normalised expression (cells × genes).
    kernel_space : dict, optional
        Covariance matrices to search. Default: 10 SE lengthscales + const.
    n_jobs : int, default 1
        Number of parallel threads for gene-level GP fitting.
        -1 = use all available cores.
    '''
    X = np.asarray(X)
    if kernel_space is None:
        l_min, l_max = get_l_limits(X)
        kernel_space = {
            'SE': np.logspace(np.log10(l_min), np.log10(l_max), 10),
            'const': 0,
        }

    logging.info('Performing DE test')
    results = dyn_de(X, exp_tab, kernel_space, n_jobs=n_jobs)
    mll_results = get_mll_results(results)

    mll_results['pval'] = 1 - stats.chi2.cdf(mll_results['LLR'], df=1)
    mll_results['qval'] = qvalue(mll_results['pval'])

    return mll_results


def model_search(X, exp_tab, DE_mll_results, kernel_space=None, n_jobs=1):
    if kernel_space is None:
        P_min, P_max = get_l_limits(X)
        kernel_space = {
            'PER': np.logspace(np.log10(P_min), np.log10(P_max), 10),
            'linear': 0,
        }

    de_exp_tab = exp_tab[DE_mll_results['g']]
    logging.info('Performing model search')
    results = dyn_de(X, de_exp_tab, kernel_space, n_jobs=n_jobs)
    new_and_old_results = pd.concat((results, DE_mll_results), sort=True)

    mask = (new_and_old_results.groupby(['g', 'model'])['BIC'].transform(min)
            == new_and_old_results['BIC'])
    log_p_data_Hi = -new_and_old_results[mask].pivot_table(
        values='BIC', index='g', columns='model')
    log_Z = logsumexp(log_p_data_Hi, 1)
    log_p_Hi_data = (log_p_data_Hi.T - log_Z).T
    p_Hi_data = np.exp(log_p_Hi_data).add_suffix('_prob')

    mask = new_and_old_results.groupby('g')['BIC'].transform(min) == new_and_old_results['BIC']
    ms_results = new_and_old_results[mask]
    ms_results = ms_results.join(p_Hi_data, on='g')

    transfer_columns = ['pval', 'qval', 'max_ll_null']
    ms_results = (ms_results.drop(transfer_columns, 1)
                  .merge(DE_mll_results[transfer_columns + ['g']], on='g'))

    return ms_results
