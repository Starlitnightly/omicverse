import logging

import numpy as np
import pandas as pd

from . import base


def optimize_gp_params(Ymean, Yvar, Us, UT1s, Ss, Ks, N):
    ''' Optimize GP parameters for pattern of (Ymean, Yvar)
        over the covariance structure grid.
    '''
    ll_max = -np.inf
    top_k = 0
    par_max = [-np.inf, 1., 1.]
    k = 0
    for U, UT1, S, K in zip(Us, UT1s, Ss, Ks):
        UTy = base.get_UTy(U, Ymean[:, 0])
        ll, delta, mu, s2s, _ = base.lbfgsb_max_LL(UTy, UT1, S, N, Yvar)

        if ll > ll_max:
            ll_max = ll
            top_k = k
            par_max = [ll, delta, s2s]

        k += 1

    ll, delta, s2s = par_max
    
    return ll, delta, s2s, top_k


def gp_smoothing(Ymean, Yvar, K, s2s, delta):
    noise_K = s2s * K + delta * s2s * np.diag(Yvar)
    Y_hat = s2s * K.dot(np.linalg.solve(noise_K, Ymean))
    return Y_hat


def aeh(X, Y, C, Ks=None, max_iter=10):
    # We're only interested in patterns, so center data
    Y = (Y.T - Y.mean(1)).T

    # Pre-factor kernel search space
    Us, Ss, UT1s = [], [], []
    for K in Ks:
        U, S = base.factor(K)
        Us.append(U)
        Ss.append(S)
        UT1 = base.get_UT1(U)
        UT1s.append(UT1)

    N = K.shape[0]
    G = Y.shape[0]

    # Initialize by random assignment
    idx = np.arange(G)
    np.random.shuffle(idx)
    cidxs = np.array_split(idx, C)
    new_clusts = np.zeros(len(idx))

    for j in range(max_iter):
        finish = False
        Yhats = []
        ll_sum = 0
        params = []
        for genes in cidxs:
            if len(genes) == 0:
                continue

            # Make average pattern per cluster, with observation variance
            Ymean = Y[genes].mean(0)[:, None]
            Yvar = Y[genes].var(0) + 1.

            # Learn noise level and lengthscale of pattern
            ll, delta, s2s, top_k = optimize_gp_params(Ymean, Yvar, Us, UT1s, Ss, Ks, N)
            K = Ks[top_k]


            # Create predicted average pattern
            Yhat = gp_smoothing(Ymean, Yvar, K, s2s, delta)
            Yhats.append(Yhat)

        # Calculate residuals for each pattern and gene
        tYhats = np.dstack(Yhats)
        tY = np.transpose(Y[:, :, None], (1, 0, 2))
        cost = np.square(tY - tYhats).sum(0)

        # Switch clusters for genes to minimize residuals
        if not finish:
            old_clusts = new_clusts.copy()
            new_clusts = cost.argmin(1)

            if np.array_equal(old_clusts, new_clusts):
                logging.info('Converged! Finishing...')
                finish = True

            cidxs = [np.where(new_clusts == i)[0] for i in range(C)]

        # Calculate the new data likelihood
        log_likelihood = -(N + G) / 2. * np.log(2. * np.pi) - \
                          (N + G) / 2. * np.log(cost.min(1).sum() / (N + G)) - \
                          (N + G) / 2.

        logging.info('Iteration {}, log likelihood: {:.2f}'.format(j + 1, log_likelihood))

        if finish:
            break

    return new_clusts, cost, Yhats, log_likelihood


def spatial_patterns(X, exp_mat, DE_mll_results, C, max_iter=10, kernel_space=None):
    ''' Group spatially variable genes into spatial patterns using automatic
     histology analysis (AHA).

    Returns
    
    pattern_results : A DataFrame with pattern membership information
        for each gene

    patterns : The spatial patterns underlying the expression values
        for the genes in the given pattern.

    '''
    if kernel_space == None:
        l_min, l_max = base.get_l_limits(X)
        l_range = np.logspace(np.log10(l_min), np.log10(l_max), 10)

        Ks = [base.SE_kernel(X, l) for l in l_range]
        
    else:
        raise NotImplementedError('Custom kernels not supported for AEH.')
        
    Y = exp_mat[DE_mll_results['g']].values.T
    N = Y.shape[1]
    
    patterns, cost, Yhats, ll = aeh(X, Y, C, Ks=Ks, max_iter=max_iter)
    
    cres = pd.DataFrame({'g': DE_mll_results['g'],
                         'pattern': patterns,
                         'membership': cost.min(1) / N})
    
    Yhats = np.hstack(Yhats)
    Yhats = pd.DataFrame.from_records(Yhats)
    Yhats.index = exp_mat.index
    
    return cres, Yhats

