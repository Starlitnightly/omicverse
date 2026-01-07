import numpy as np
from numba import jit, njit
from tqdm import tqdm
import pandas as pd
import multiprocessing
import itertools

from . import danb_model
from . import bernoulli_model
from . import normal_model
from . import none_model
from .local_stats import compute_local_cov_max
from .knn import compute_node_degree
from .utils import center_values


@jit(nopython=True)
def conditional_eg2(x, neighbors, weights):
    """
    Computes EG2 for the conditional correlation
    """
    N = neighbors.shape[0]
    K = neighbors.shape[1]

    t1x = np.zeros(N)

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            if wij == 0:
                continue

            t1x[i] += wij*x[j]
            t1x[j] += wij*x[i]

    out_eg2 = (t1x**2).sum()

    return out_eg2


def conditional_eg2_slow(x, neighbors, weights):
    """
    Computes EG2 for the conditional correlation
    This version is slower and more explicit for debugging purposes
    """
    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # enumerate ALL neighbors of each node and the edge weight

    neighbors_all = [[] for _ in range(N)]
    weights_all = [[] for _ in range(N)]

    for i in range(N):
        neighbors_i = neighbors_all[i]
        weights_i = weights_all[i]

        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]
            if wij == 0:
                continue

            neighbors_j = neighbors_all[j]
            weights_j = weights_all[j]

            neighbors_i.append(j)
            weights_i.append(wij)

            neighbors_j.append(i)
            weights_j.append(wij)

    node_totals = np.zeros(N)
    node_totals_diag = np.zeros(N)

    for i in range(N):
        neighbors_i = neighbors_all[i]
        weights_i = weights_all[i]

        for j, wij in zip(neighbors_i, weights_i):
            node_totals[i] += x[j]*wij
            node_totals_diag[i] += (x[j]*wij)**2

    node_totals = node_totals**2

    dup = 0

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]
            if wij == 0:
                continue

            dup += wij**2*(x[i]**2+x[j]**2)

    return node_totals.sum() - node_totals_diag.sum() + dup


@jit(nopython=True)
def local_cov_pair(x, y, neighbors, weights):
    """Test statistic for local pair-wise autocorrelation"""
    out = 0

    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        if xi == 0 and yi == 0:
            continue
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            xj = x[j]
            yj = y[j]

            out += w_ij*(xi*yj + yi*xj)/2

    return out


@jit(nopython=True)
def compute_moments_weights_pairs_slow(muA, a2, muB, b2, neighbors, weights):
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

            EG += wij * (muA[i] * muB[j] + muB[i] * muA[j]) / 2

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
                            t1 = (
                                a2[i] * b2[j]
                                + muA[i] * muB[j] * muA[j] * muB[i]
                                + muA[j] * muB[i] * muA[i] * muB[j]
                                + a2[j] * b2[i]
                            )
                        else:
                            t1 = (
                                a2[i] * muB[j] * muB[y]
                                + muA[i] * muB[j] * muA[y] * muB[i]
                                + muA[j] * muB[i] * muA[i] * muB[y]
                                + muA[j] * b2[i] * muA[y]
                            )
                    elif i == y:
                        if j == x:
                            t1 = (
                                muA[i] * muB[j] * muA[j] * muB[i]
                                + a2[i] * b2[j]
                                + a2[j] * b2[i]
                                + muA[j] * muB[i] * muA[i] * muB[j]
                            )
                        else:
                            t1 = (
                                muA[i] * muB[j] * muA[x] * muB[i]
                                + a2[i] * muB[j] * muB[x]
                                + muA[j] * b2[i] * muA[x]
                                + muA[j] * muB[i] * muA[i] * muB[x]
                            )
                    else:  # i is unique since i can't equal j

                        if j == x:
                            t1 = (
                                muA[i] * muB[j] * muA[j] * muB[y]
                                + muA[i] * b2[j] * muA[y]
                                + a2[j] * muB[i] * muB[y]
                                + muA[j] * muB[i] * muA[y] * muB[j]
                            )
                        elif j == y:
                            t1 = (
                                muA[i] * b2[j] * muA[x]
                                + muA[i] * muB[j] * muA[j] * muB[x]
                                + muA[j] * muB[i] * muA[x] * muB[j]
                                + a2[j] * muB[i] * muB[x]
                            )
                        else:  # i and j are unique, no shared nodes
                            t1 = (
                                muA[i] * muB[j] * muA[x] * muB[y]
                                + muA[i] * muB[j] * muA[y] * muB[x]
                                + muA[j] * muB[i] * muA[x] * muB[y]
                                + muA[j] * muB[i] * muA[y] * muB[x]
                            )

                    EG2_i += s * t1 / 4

        EG2 += EG2_i

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights_pairs(
        muX, x2,
        muY, y2,
        neighbors, weights):
    """Computes the expectations of the local pair-wise test statistic"""

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij*(muX[i]*muY[j] + muY[i]*muX[j])/2

    # Calculate E[G^2]
    EG2 = 0
    EG2 += (EG**2)

    #   Get the x^2*y*z terms
    t1x = np.zeros(N)
    t2x = np.zeros(N)

    t1y = np.zeros(N)
    t2y = np.zeros(N)

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            if wij == 0:
                continue

            t1x[i] += wij*muX[j]
            t2x[i] += wij**2*muX[j]**2

            t1x[j] += wij*muX[i]
            t2x[j] += wij**2*muX[i]**2

            t1y[i] += wij*muY[j]
            t2y[i] += wij**2*muY[j]**2

            t1y[j] += wij*muY[i]
            t2y[j] += wij**2*muY[i]**2

    t1x = t1x**2
    t1y = t1y**2

    for i in range(N):
        EG2 += (y2[i] - muY[i]**2)*(t1x[i] - t2x[i])/4
        EG2 += (x2[i] - muX[i]**2)*(t1y[i] - t2y[i])/4

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]

            EG2 += wij**2*(
                x2[i]*y2[j] - (muX[i]**2)*(muY[j]**2) +
                y2[i]*x2[j] - (muY[i]**2)*(muX[j]**2)
            )/4

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights_pairs_fast(
        muX, x2,
        muY, y2,
        neighbors, weights):
    """
    Computes the expectations of the local pair-wise test statistic

    About 2x as fast as compute_moments_weights_pairs
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    #   Get the x^2*y*z terms
    t1x = np.zeros(N)
    t2x = np.zeros(N)

    t1y = np.zeros(N)
    t2y = np.zeros(N)

    EG = 0
    EG2 = 0

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]
            wij_2 = wij**2

            if wij == 0:
                continue

            muX_i2 = muX[i]**2
            muX_j2 = muX[j]**2
            muY_i2 = muY[i]**2
            muY_j2 = muY[j]**2

            EG += wij*(muX[i]*muY[j] + muY[i]*muX[j])/2

            t1x[i] += wij*muX[j]
            t2x[i] += wij_2*muX_j2

            t1x[j] += wij*muX[i]
            t2x[j] += wij_2*muX_i2

            t1y[i] += wij*muY[j]
            t2y[i] += wij_2*muY_j2

            t1y[j] += wij*muY[i]
            t2y[j] += wij_2*muY_i2

            EG2 += wij_2*(
                x2[i]*y2[j] - (muX_i2)*(muY_j2) +
                y2[i]*x2[j] - (muY_i2)*(muX_j2)
            )/4

    # Calculate E[G^2]
    t1x = t1x**2
    t1y = t1y**2

    for i in range(N):
        EG2 += (y2[i] - muY[i]**2)*(t1x[i] - t2x[i])/4
        EG2 += (x2[i] - muX[i]**2)*(t1y[i] - t2y[i])/4

    EG2 += (EG**2)

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights_pairs_std(neighbors, weights):
    """
    Computes the expectations of the local pair-wise test statistic

    This version assumes variables are standardized,
    and so the moments are actually the same for all
    pairs of variables
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0

    # Calculate E[G^2]
    EG2 = 0

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            wij = weights[i, k]

            EG2 += (wij**2)/2

    return EG, EG2


def create_centered_counts(counts, model, num_umi):
    """
    Creates a matrix of centered/standardized counts given
    the selected statistical model
    """
    out = np.zeros_like(counts, dtype='double')

    for i in tqdm(range(out.shape[0])):

        vals_x = counts[i]

        out_x = create_centered_counts_row(
            vals_x, model, num_umi)

        out[i] = out_x

    return out


def create_centered_counts_row(vals_x, model, num_umi):

    if model == 'bernoulli':
        vals_x = (vals_x > 0).astype('double')
        mu_x, var_x, x2_x = bernoulli_model.fit_gene_model(
            vals_x, num_umi)

    elif model == 'danb':
        mu_x, var_x, x2_x = danb_model.fit_gene_model(
            vals_x, num_umi)

    elif model == 'normal':
        mu_x, var_x, x2_x = normal_model.fit_gene_model(
            vals_x, num_umi)

    elif model == 'none':
        mu_x, var_x, x2_x = none_model.fit_gene_model(
            vals_x, num_umi)

    else:
        raise Exception("Invalid Model: {}".format(model))

    var_x[var_x == 0] = 1
    out_x = (vals_x-mu_x)/(var_x**0.5)
    out_x[out_x == 0] = 0

    return out_x


def _compute_hs_pairs_inner(row_i, counts, neighbors, weights, num_umi,
                            model, centered, Wtot2, D):

    vals_x = counts[row_i]

    lc_out = np.zeros(counts.shape[0])
    lc_z_out = np.zeros(counts.shape[0])

    if model == 'bernoulli':
        vals_x = (vals_x > 0).astype('double')
        mu_x, var_x, x2_x = bernoulli_model.fit_gene_model(
            vals_x, num_umi)

    elif model == 'danb':
        mu_x, var_x, x2_x = danb_model.fit_gene_model(
            vals_x, num_umi)

    elif model == 'normal':
        mu_x, var_x, x2_x = normal_model.fit_gene_model(
            vals_x, num_umi)

    elif model == 'none':
        mu_x, var_x, x2_x = none_model.fit_gene_model(
            vals_x, num_umi)

    else:
        raise Exception("Invalid Model: {}".format(model))

    if centered:
        vals_x = center_values(vals_x, mu_x, var_x)

    for row_j in range(counts.shape[0]):

        if row_j > row_i:
            continue

        vals_y = counts[row_j]

        if model == 'bernoulli':
            vals_y = (vals_y > 0).astype('double')
            mu_y, var_y, x2_y = bernoulli_model.fit_gene_model(
                vals_y, num_umi)

        elif model == 'danb':
            mu_y, var_y, x2_y = danb_model.fit_gene_model(
                vals_y, num_umi)

        elif model == 'normal':
            mu_y, var_y, x2_y = normal_model.fit_gene_model(
                vals_y, num_umi)

        elif model == 'none':
            mu_x, var_x, x2_x = none_model.fit_gene_model(
                vals_x, num_umi)

        else:
            raise Exception("Invalid Model: {}".format(model))

        if centered:
            vals_y = center_values(vals_y, mu_y, var_y)

        if centered:
            EG, EG2 = 0, Wtot2/2
        else:
            EG, EG2 = compute_moments_weights_pairs_fast(mu_x, x2_x,
                                                         mu_y, x2_y,
                                                         neighbors, weights)

        lc = local_cov_pair(vals_x, vals_y,
                            neighbors, weights)

        stdG = (EG2 - EG**2)**.5

        Z = (lc - EG) / stdG

        lc_out[row_j] = lc
        lc_z_out[row_j] = Z

    return (lc_out, lc_z_out)


def _compute_hs_pairs_inner_centered(
        rowpair, counts, neighbors, weights, Wtot2, D):
    """
    This version assumes that the counts have already been modeled
    and centered
    """
    row_i, row_j = rowpair

    vals_x = counts[row_i]
    vals_y = counts[row_j]

    EG, EG2 = 0, Wtot2/2

    lc = local_cov_pair(vals_x, vals_y,
                        neighbors, weights)

    stdG = (EG2 - EG**2)**.5

    Z = (lc - EG) / stdG

    return (lc, Z)


@jit(nopython=True)
def _compute_hs_pairs_inner_centered_cond_sym(
    rowpair, counts, neighbors, weights, eg2s
):
    """
    This version assumes that the counts have already been modeled
    and centered
    """
    row_i, row_j = rowpair

    vals_x = counts[row_i]
    vals_y = counts[row_j]

    lc = local_cov_pair(vals_x, vals_y, neighbors, weights)*2

    # Compute xy
    EG, EG2 = 0, eg2s[row_i]

    stdG = (EG2 - EG ** 2) ** 0.5

    Zxy = (lc - EG) / stdG

    # Compute yx
    EG, EG2 = 0, eg2s[row_j]

    stdG = (EG2 - EG ** 2) ** 0.5

    Zyx = (lc - EG) / stdG

    if abs(Zxy) < abs(Zyx):
        Z = Zxy
    else:
        Z = Zyx

    return (lc, Z)


def initializer1(counts, neighbors, weights, num_umi, model, centered, Wtot2, D):
    global g_counts
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_centered
    global g_Wtot2
    global g_D
    g_counts = counts
    g_neighbors = neighbors
    g_weights = weights
    g_num_umi = num_umi
    g_model = model
    g_centered = centered
    g_Wtot2 = Wtot2
    g_D = D

def compute_hs_pairs(counts, neighbors, weights,
                     num_umi, model, centered=False, jobs=1):

    genes = counts.index

    counts = counts.values
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    if jobs > 1:
        with multiprocessing.Pool(
            processes=jobs, 
            initializer=initializer1, 
            initargs=[counts, neighbors, weights, num_umi, model, centered, Wtot2, D]
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        _map_fun_parallel_pairs, 
                        range(counts.shape[0])
                    ), 
                    total=counts.shape[0]
                )
            )
    else:
        def _map_fun(row_i):
            return _compute_hs_pairs_inner(
                row_i, counts, neighbors, weights, num_umi,
                model, centered, Wtot2, D)
        results = list(
            tqdm(
                map(_map_fun, range(counts.shape[0])),
                total=counts.shape[0]
            )
        )

    # Only have the lower triangle so we must rebuild the rest
    lcs = [x[0] for x in results]
    lc_zs = [x[1] for x in results]

    lcs = np.vstack(lcs)
    lc_zs = np.vstack(lc_zs)

    lcs = lcs + lcs.T
    lc_zs = lc_zs + lc_zs.T

    np.fill_diagonal(lcs, lcs.diagonal() / 2)
    np.fill_diagonal(lc_zs, lc_zs.diagonal() / 2)

    lc_maxs = compute_local_cov_pairs_max(D, counts)
    lcs = lcs / lc_maxs

    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)

    return lcs, lc_zs

def initializer2(counts, neighbors, weights, Wtot2, D):
    global g_counts
    global g_neighbors
    global g_weights
    global g_Wtot2
    global g_D
    g_counts = counts
    g_neighbors = neighbors
    g_weights = weights
    g_Wtot2 = Wtot2
    g_D = D

def compute_hs_pairs_centered(counts, neighbors, weights,
                              num_umi, model, jobs=1):

    genes = counts.index

    counts = counts.values
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    counts = create_centered_counts(counts, model, num_umi)

    pairs = list(itertools.combinations(range(counts.shape[0]), 2))

    if jobs > 1:
        with multiprocessing.Pool(
            processes=jobs, 
            initializer=initializer2,
            initargs=[counts, neighbors, weights, Wtot2, D]
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        _map_fun_parallel_pairs_centered, 
                        pairs
                    ),
                    total=len(pairs)
                )
            )
    else:
        def _map_fun(rowpair):
            return _compute_hs_pairs_inner_centered(
                rowpair, counts, neighbors, weights, Wtot2, D)
        results = list(
            tqdm(
                map(_map_fun, pairs),
                total=len(pairs)
            )
        )

    N = counts.shape[0]
    pairs = np.array(pairs)
    vals_lc = np.array([x[0] for x in results])
    vals_z = np.array([x[1] for x in results])
    lcs = expand_pairs(pairs, vals_lc, N)
    lc_zs = expand_pairs(pairs, vals_z, N)

    lc_maxs = compute_local_cov_pairs_max(D, counts)
    lcs = lcs / lc_maxs

    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)

    return lcs, lc_zs


def initializer3(counts, neighbors, weights, eg2s):
    global g_counts
    global g_neighbors
    global g_weights
    global g_eg2s
    g_counts = counts
    g_neighbors = neighbors
    g_weights = weights
    g_eg2s = eg2s


def compute_hs_pairs_centered_cond(counts, neighbors, weights,
                                   num_umi, model, jobs=1):

    genes = counts.index

    counts = counts.values
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)

    counts = create_centered_counts(counts, model, num_umi)

    eg2s = np.array(
        [
            conditional_eg2(counts[i], neighbors, weights)
            for i in range(counts.shape[0])
        ]
    )

    pairs = list(itertools.combinations(range(counts.shape[0]), 2))

    if jobs > 1:
        with multiprocessing.Pool(
            processes=jobs, 
            initializer=initializer3,
            initargs=[counts, neighbors, weights, eg2s]
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        _map_fun_parallel_pairs_centered_cond,
                        pairs
                    ),
                    total=len(pairs)
                )
            )
    else:
        def _map_fun(rowpair):
            return _compute_hs_pairs_inner_centered_cond_sym(
                rowpair, counts, neighbors, weights, eg2s)
        results = list(
            tqdm(
                map(_map_fun, pairs),
                total=len(pairs)
            )
        )

    N = counts.shape[0]
    pairs = np.array(pairs)
    vals_lc = np.array([x[0] for x in results])
    vals_z = np.array([x[1] for x in results])
    lcs = expand_pairs(pairs, vals_lc, N)
    lc_zs = expand_pairs(pairs, vals_z, N)

    lc_maxs = compute_local_cov_pairs_max(D, counts)
    lcs = lcs / lc_maxs

    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)

    return lcs, lc_zs


def _map_fun_parallel_pairs(row_i):
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_centered
    global g_Wtot2
    global g_D
    global g_counts
    return _compute_hs_pairs_inner(
        row_i, g_counts, g_neighbors, g_weights, g_num_umi,
        g_model, g_centered, g_Wtot2, g_D)


def _map_fun_parallel_pairs_centered(rowpair):
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_Wtot2
    global g_D
    global g_counts
    return _compute_hs_pairs_inner_centered(
        rowpair, g_counts, g_neighbors, g_weights, g_Wtot2, g_D)


def _map_fun_parallel_pairs_centered_cond(rowpair):
    global g_neighbors
    global g_weights
    global g_counts
    global g_eg2s
    return _compute_hs_pairs_inner_centered_cond_sym(
        rowpair, g_counts, g_neighbors, g_weights, g_eg2s
    )


@njit
def expand_pairs(pairs, vals, N):

    out = np.zeros((N, N))

    for i in range(len(pairs)):

        x = pairs[i, 0]
        y = pairs[i, 1]
        v = vals[i]

        out[x, y] = v
        out[y, x] = v

    return out


def compute_local_cov_pairs_max(node_degrees, counts):
    """
    For a Genes x Cells count matrix, compute the maximal pair-wise correlation
    between any two genes
    """

    N_GENES = counts.shape[0]

    gene_maxs = np.zeros(N_GENES)
    for i in range(N_GENES):
        gene_maxs[i] = compute_local_cov_max(node_degrees, counts[i])

    result = gene_maxs.reshape((-1, 1)) + gene_maxs.reshape((1, -1))
    result = result / 2
    return result
