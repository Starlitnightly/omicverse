import numpy as np
from numba import jit
from tqdm import tqdm
import pandas as pd
from scipy.stats import norm
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import multiprocessing

from . import danb_model
from . import bernoulli_model
from . import normal_model
from . import none_model

from .knn import compute_node_degree
from .utils import center_values


@jit(nopython=True)
def local_cov_weights(x, neighbors, weights):
    out = 0

    for i in range(len(x)):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            xi = x[i]
            xj = x[j]
            if xi == 0 or xj == 0 or w_ij == 0:
                out += 0
            else:
                out += xi * xj * w_ij

    return out


@jit(nopython=True)
def compute_moments_weights_slow(mu, x2, neighbors, weights):
    """
    This version exaustively iterates over all |E|^2 terms
    to compute the expected moments exactly.  Used to test
    the more optimized formulations that follow
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij * mu[i] * mu[j]

    # Calculate E[G^2]
    EG2 = 0
    for i in range(N):

        EG2_i = 0

        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            for x in range(N):
                for z in range(K):
                    y = neighbors[x, z]
                    wxy = weights[x, z]

                    s = wij * wxy
                    if s == 0:
                        continue

                    if i == x:
                        if j == y:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[y]
                    elif i == y:
                        if j == x:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[x]
                    else:  # i is unique since i can't equal j

                        if j == x:
                            t1 = mu[i] * x2[j] * mu[y]
                        elif j == y:
                            t1 = mu[i] * x2[j] * mu[x]
                        else:  # i and j are unique, no shared nodes
                            t1 = mu[i] * mu[j] * mu[x] * mu[y]

                    EG2_i += s * t1

        EG2 += EG2_i

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights(mu, x2, neighbors, weights):

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij * mu[i] * mu[j]

    # Calculate E[G^2]
    EG2 = 0

    #   Get the x^2*y*z terms
    t1 = np.zeros(N)
    t2 = np.zeros(N)

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            if wij == 0:
                continue

            t1[i] += wij * mu[j]
            t2[i] += wij**2 * mu[j] ** 2

            t1[j] += wij * mu[i]
            t2[j] += wij**2 * mu[i] ** 2

    t1 = t1**2

    for i in range(N):
        EG2 += (x2[i] - mu[i] ** 2) * (t1[i] - t2[i])

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]

            EG2 += wij**2 * (x2[i] * x2[j] - (mu[i] ** 2) * (mu[j] ** 2))

    EG2 += EG**2

    return EG, EG2


@jit(nopython=True)
def compute_local_cov_max(node_degrees, vals):
    tot = 0.0

    for i in range(node_degrees.size):
        tot += node_degrees[i] * (vals[i] ** 2)

    return tot / 2

def initializer(neighbors, weights, num_umi, model, centered, Wtot2, D):
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_centered
    global g_Wtot2
    global g_D
    g_neighbors = neighbors
    g_weights = weights
    g_num_umi = num_umi
    g_model = model
    g_centered = centered
    g_Wtot2 = Wtot2
    g_D = D

def compute_hs(
    counts, neighbors, weights, num_umi, model, genes, centered=False, jobs=1
):

    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    def data_iter():
        for i in range(counts.shape[0]):
            vals = counts[i]
            if issparse(vals):
                vals = vals.toarray().ravel()
            vals = vals.astype("double")
            yield vals

    if jobs > 1:
        with multiprocessing.Pool(
            processes=jobs, 
            initializer=initializer, 
            initargs=[neighbors, weights, num_umi, model, centered, Wtot2, D]
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        _map_fun_parallel,
                        data_iter()
                    ), 
                    total=counts.shape[0]
                )
            )
    else:

        def _map_fun(vals):
            return _compute_hs_inner(
                vals, neighbors, weights, num_umi, model, centered, Wtot2, D
            )

        results = list(tqdm(map(_map_fun, data_iter()), total=counts.shape[0]))

    results = pd.DataFrame(results, index=genes, columns=["G", "EG", "stdG", "Z", "C"])

    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]

    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"

    results = results[["C", "Z", "Pval", "FDR"]]  # Remove other columns

    return results


def _compute_hs_inner(vals, neighbors, weights, num_umi, model, centered, Wtot2, D):
    """
    Note, since this is an inner function, for parallelization to work well
    none of the contents of the function can use MKL or OPENBLAS threads.
    Or else we open too many.  Because of this, some simple numpy operations
    are re-implemented using numba instead as it's difficult to control
    the number of threads in numpy after it's imported
    """

    if model == "bernoulli":
        vals = (vals > 0).astype("double")
        mu, var, x2 = bernoulli_model.fit_gene_model(vals, num_umi)
    elif model == "danb":
        mu, var, x2 = danb_model.fit_gene_model(vals, num_umi)
    elif model == "normal":
        mu, var, x2 = normal_model.fit_gene_model(vals, num_umi)
    elif model == "none":
        mu, var, x2 = none_model.fit_gene_model(vals, num_umi)
    else:
        raise Exception("Invalid Model: {}".format(model))

    if centered:
        vals = center_values(vals, mu, var)

    G = local_cov_weights(vals, neighbors, weights)

    if centered:
        EG, EG2 = 0, Wtot2
    else:
        EG, EG2 = compute_moments_weights(mu, x2, neighbors, weights)

    stdG = (EG2 - EG * EG) ** 0.5

    Z = (G - EG) / stdG

    G_max = compute_local_cov_max(D, vals)
    C = (G - EG) / G_max

    return [G, EG, stdG, Z, C]


def _map_fun_parallel(vals):
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_centered
    global g_Wtot2
    global g_D
    return _compute_hs_inner(
        vals, g_neighbors, g_weights, g_num_umi, g_model, g_centered, g_Wtot2, g_D
    )
