import numpy as np
from numba import jit, njit


@njit
def center_values(vals, mu, var):
    out = np.zeros_like(vals)

    for i in range(len(vals)):
        std = var[i]**0.5
        if std == 0:
            out[i] = 0
        else:
            out[i] = (vals[i] - mu[i])/std

    return out


@njit
def neighbor_smoothing(vals, neighbors, weights, _lambda=.9):
    """

    output is (neighborhood average) * _lambda + self * (1-_lambda)


    vals: expression matrix (genes x cells)
    neighbors: neighbor indices (cells x K)
    weights: neighbor weights (cells x K)
    _lambda: ratio controlling self vs. neighborhood
    """

    out = np.zeros_like(vals, dtype=np.float64)

    G = vals.shape[0]       # Genes

    for g in range(G):

        row_vals = vals[g, :]
        smooth_row_vals = neighbor_smoothing_row(
            row_vals, neighbors, weights, _lambda)

        out[g, :] = smooth_row_vals

    return out


@njit
def neighbor_smoothing_row(vals, neighbors, weights, _lambda=.9):
    """

    output is (neighborhood average) * _lambda + self * (1-_lambda)


    vals: expression matrix (genes x cells)
    neighbors: neighbor indices (cells x K)
    weights: neighbor weights (cells x K)
    _lambda: ratio controlling self vs. neighborhood
    """

    out = np.zeros_like(vals, dtype=np.float64)
    out_denom = np.zeros_like(vals, dtype=np.float64)

    N = neighbors.shape[0]  # Cells
    K = neighbors.shape[1]  # Neighbors

    for i in range(N):

        xi = vals[i]

        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            xj = vals[j]

            out[i] += xj*wij
            out[j] += xi*wij

            out_denom[i] += wij
            out_denom[j] += wij

    out /= out_denom

    out = (out * _lambda) + (1 - _lambda) * vals

    return out
