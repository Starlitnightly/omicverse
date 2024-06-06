import scipy as s
import numpy as np
import scipy.stats as stats
import sys
import scipy.spatial as SS
import math
import seaborn as sns
import matplotlib.pyplot as plt
from ..core.gp_utils import covar_to_corr

def simulate_data(N=200, seed=1234567, views = ["0", "1", "2", "3"], D = [500, 200, 500, 200], noise_level = 1,
                  K = 4, G = 1, lscales = [0.2, 0.8, 0.0, 0.0], sample_cov = "equidistant", scales = [1, 0.8, 0, 0],
                  shared = True, plot = False, alpha = None):
    """
    Function to simulate test data for MOFA (without ARD or spike-and-slab on factors)

    N: Number of time points/ samples per group
    seed: seed to use for simulation
    views: list of view names
    K: number of factors
    G: Number of groups
    D: list of number of features per view (same length as views)
    noise_level: variance of the residuals (1/tau);
                 per feature it is multiplied by a uniform random number in [0.5, 1.5] to model differences in features' noise
    scales, lscales: hyperparameters of the GP per factor (length as given by K)
    sample_cov: sample_covariates to use (can be of shape N X C) or "equidistant" or None
    shared: A list or single boolean indicating for each factor whether it is perfectly shared across groups or not.
    For non-shared ones pairwise group-group correlations are simulated by a Bernoulli distribution.
    Only relevant for factors with lengthscale and scale > 0.
    plot: If True, simulation results are plotted
    alpha: Pre-set activity for different views per factor. Needs to be a list of length M with arrays of length K. Default is None and will be drawn at random.
    """

    # simulate some test data
    np.random.seed(seed)
    M = len(views)
    N = int(N)
    if type(shared) == bool:
        shared = [shared]
    if len(shared) == 1:
        shared = [shared] * K

    groupidx = np.repeat(range(G), N) # kronecker structure
    if not sample_cov is None:
        if sample_cov == "equidistant":
            sample_cov = np.linspace(0,1,N)
            sample_cov = sample_cov.reshape(N, 1)
        else:
            assert sample_cov.shape[0] == N, "Number of rows of sample_cov and N does not match"
            if len(np.repeat(np.arange(0,100,1),2).shape) == 1:
                sample_cov = sample_cov.reshape(N, 1)
        distC = SS.distance.pdist(sample_cov, 'euclidean')**2.
        distC = SS.distance.squareform(distC)

    else:
        lscales = [0]* K

    Gmats = []
    for k in range(K):
        if scales[k] == 0 or lscales[k] == 0: # group structure not modelled
            Gmat = np.eye(G)
        else:
            if shared[k]:
                Gmat = np.ones([G,G])
            else:
                x = np.random.uniform(-1,1, G)
                Gmat = np.outer(x,x) + 0.5 * np.eye(G)
                Gmat = covar_to_corr(Gmat)
        Gmats.append(Gmat)

    # simulate Sigma
    Sigma =[]
    for k in range(K):
        if lscales[k] > 0:
            Kmat = scales[k] * np.exp(-distC / (2 * lscales[k] ** 2))
            Kmat = np.kron(Gmats[k], Kmat)
            Sigma.append( Kmat + (1-scales[k]) * np.eye(N*G))
        elif lscales[k] == 0:
            Kmat = scales[k] * (distC == 0).astype(float)
            Kmat = np.kron(Gmats[k], Kmat)
            Sigma.append(Kmat + (1-scales[k]) * np.eye(N*G))
            # Sigma.append(np.eye(N*G))
        else:
            sys.exit("All lengthscales need to be non-negative")

    # plot covariance structure
    if plot:
        fig, axs = plt.subplots(1, K, sharex=True, sharey=True)
        for k in range(K):
            sns.heatmap(Sigma[k], ax =axs[k])

    # simulate factor values
    Zks = []
    for k in range(K):
        sig = Sigma[k]
        Zks.append(np.random.multivariate_normal(np.zeros(N * G), sig, 1))
    Zks = np.vstack(Zks).transpose()

    Z = []
    for g in range(G):
        Z.append(Zks[groupidx == g,])

    # simulate alpha and theta, each factor should be active in at least one view
    theta = 0.5 * np.ones([M, K])
    if alpha is None:
        inactive = 1000
        active = 1
        alpha_tmp = [np.ones(M) * inactive]*K
        for k in range(K):
            while np.all(alpha_tmp[k]==inactive):
                alpha_tmp[k] = np.random.choice([active,inactive], size=M, replace=True)
        alpha = [ np.array(alpha_tmp)[:,m] for m in range(M) ]
    else:
        assert len(alpha) == M
        assert len(alpha[0]) == K

    # simulate weights
    W = []
    for m in range(M):
        W.append(np.column_stack(
            [np.random.normal(0, np.sqrt(1/alpha[m][k]), D[m]) * np.random.binomial(1, theta[m][k], D[m]) for k in range(K)]))

    # simulate heteroscedastic noise
    noise = []
    for m in range(M):
        tau_m = stats.uniform.rvs(loc=0.5, scale=1, size=D[m]) * 1/noise_level # uniform between 0.5 and 1.5 scaled by noise level
        noise.append(np.random.multivariate_normal(np.zeros(D[m]), np.eye(D[m]) * 1 / tau_m, N))

    # generate data
    data = []
    for m in range(M):
        tmp = []
        for g in range(G):
            tmp.append(Z[g].dot(W[m].transpose()) + noise[m])
        data.append(tmp)

    # store as list of groups
    if not sample_cov is None:
        sample_cov = [sample_cov] * G

    return {'data': data, 'W': W, 'Z': Z, 'noise': noise, 'sample_cov': sample_cov, 'Sigma': Sigma,
            'views': views, 'lscales': lscales, 'N': N, 'Gmats' : Gmats }



def mask_samples(sim, perc = 0.2, perc_all_views = 0):
    """
    Function to mask values at randomly sampled time points in each group and view.

    Param:
    perc: this fraction of time points are drawn in each view and group independently and all feature values at this time point set to NaN
    perc_all_views: this fraction of time points are drawn in each group independently and all feature values in all views are set to NaN
    """

    data = sim['data']
    N = sim['N']
    M = len(sim['views'])
    G = len(sim['data'][0])
    masked_samples = [[np.random.choice(N, math.floor(N * perc), replace = False) for g in range(G)] for m in range(M)]
    for m in range(M):
        for g in range(G):
            data[m][g][masked_samples[m][g],:] = s.nan

    if perc_all_views > 0:
        masked_samples.all_views = [np.random.choice(N, math.floor(N * perc_all_views), replace = False) for g in range(G)]
        for m in range(len(data)):
            for g in range(G):
                data[m][g][masked_samples.all_views[g], :] = s.nan


    return data
