from numba import jit
import pandas as pd
import numpy as np
from numba import njit

N_BIN_TARGET = 30


@jit(nopython=True)
def find_gene_p(num_umi, D):
    """
    Finds gene_p such that sum of expected detects
    matches our data

    Performs a binary search on p in the space of log(p)
    """

    low = 1e-12
    high = 1

    if D == 0:
        return 0

    for ITER in range(40):

        attempt = (high*low)**0.5
        tot = 0

        for i in range(len(num_umi)):
            tot = tot + 1-(1-attempt)**num_umi[i]

        if abs(tot-D)/D < 1e-3:
            break

        if tot > D:
            high = attempt
        else:
            low = attempt

    return (high*low)**0.5


def fit_gene_model_scaled(gene_detects, umi_counts):

    D = gene_detects.sum()

    gene_p = find_gene_p(umi_counts, D)

    detect_p = 1-(1-gene_p)**umi_counts

    mu = detect_p
    var = detect_p * (1 - detect_p)
    x2 = detect_p

    return mu, var, x2


def true_params_scaled(gene_p, umi_counts):

    detect_p = 1-(1-gene_p/10000)**umi_counts

    mu = detect_p
    var = detect_p * (1 - detect_p)
    x2 = detect_p

    return mu, var, x2


def fit_gene_model_linear(gene_detects, umi_counts):

    umi_count_bins, bins = pd.qcut(
        np.log10(umi_counts), N_BIN_TARGET, labels=False, retbins=True,
        duplicates='drop'
    )
    bin_centers = np.array(
        [bins[i] / 2 + bins[i + 1] / 2 for i in range(len(bins) - 1)]
    )

    N_BIN = len(bin_centers)

    bin_detects = bin_gene_detection(gene_detects, umi_count_bins, N_BIN)

    lbin_detects = logit(bin_detects)

    X = np.ones((N_BIN, 2))
    X[:, 1] = bin_centers
    Y = lbin_detects

    b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    detect_p = ilogit(b[0] + b[1] * np.log10(umi_counts))

    mu = detect_p
    var = detect_p * (1 - detect_p)
    x2 = detect_p

    return mu, var, x2


fit_gene_model = fit_gene_model_linear


@njit
def logit(p):
    return np.log(p / (1 - p))


@njit
def ilogit(q):
    return np.exp(q) / (1 + np.exp(q))


@njit
def bin_gene_detection(gene_detects, umi_count_bins, N_BIN):
    bin_detects = np.zeros(N_BIN)
    bin_totals = np.zeros(N_BIN)

    for i in range(len(gene_detects)):
        x = gene_detects[i]
        bin_i = umi_count_bins[i]
        bin_detects[bin_i] += x
        bin_totals[bin_i] += 1

    # Need to account for 0% detects
    #    Add 1 to numerator and denominator
    # Need to account for 100% detects
    #    Add 1 to denominator

    return (bin_detects+1) / (bin_totals+2)
