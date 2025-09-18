import logging
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse


def linear_embedding_velocity(V, PCs):
    return V @ PCs


def _estimate_dt(X, V, nbrs_idx):
    """Estimates the time step (`dt`) based on local density for each cell.

    This function calculates an estimate of the time step for each cell in the dataset by evaluating the 
    local density of neighboring cells. It assumes that the time step is inversely proportional to the 
    velocity magnitude relative to the local density of neighboring points.

    The process involves computing the Euclidean distance between each cell and its neighbors within local patch,
    then averaging these distances to estimate the local density. The time step is then estimated as the 
    median of the density-to-velocity ratio across neighbors for each cell.

    Parameters
    ----------
    X : :class:`~numpy.ndarray`
        The gene expression data matrix of shape `(n_cells, n_genes)`.
    V : :class:`~numpy.ndarray`
        The velocity matrix of shape `(n_cells, n_genes)`, containing the velocity vectors for each cell.
    nbrs_idx : list of list of int
        A list where each element contains the indices of the nearest neighbors for a given cell.
        Each cell's neighbors are identified using a nearest-neighbor search algorithm, typically based 
        on the gene expression data.

    Returns
    -------
    :class:`~numpy.ndarray`
        A column vector of shape `(n_cells, 1)` representing the estimated time steps (`dt`) for each cell.
    """
    logging.info('Estimate dt based on local density...')
    dt = np.zeros(X.shape[0])
    for i, ind in enumerate(nbrs_idx):
        delta_i = X[ind] - X[i]
        density = np.mean(np.linalg.norm(delta_i, axis=1))
        dt[i] = np.median(density / np.linalg.norm(V[i]))
    return dt.reshape(-1, 1)


def binary_corr(xi, Xj, vi):
    return np.mean(np.sign(vi) == np.sign(Xj - xi), axis=1)

def inner_product(xi, Xj, vi):
    D = Xj - xi
    return D @ vi 


def pearson_corr(xi, Xj, vi):
    D = Xj - xi
    Vi = np.tile(vi, (D.shape[0], 1))
    c = np.zeros(D.shape[0])
    for i in range(D.shape[0]):
        c[i] = np.corrcoef(D[i], Vi[i])[0, 1]

    return c


def cos_corr(xi, Xj, vi):
    D = Xj - xi
    dist = np.linalg.norm(D, axis=1)
    dist[dist == 0] = 1
    D /= dist[:, None]

    v_norm = np.linalg.norm(vi)
    if v_norm == 0:
        v_norm = 1
    
    return D @ vi / v_norm


def corr_kernel(X, V, nbrs, sigma=None, corr_func=cos_corr, softmax_adjusted=False):
    if softmax_adjusted:
        if sigma is None:
            logging.info("param sigma is `None`, estimate sigma automatically...")
            sigma = _estimate_sigma(X, V, nbrs, corr_func)
        logging.info(f"Using `sigma={sigma:.4f}`")
    P = np.zeros((X.shape[0], X.shape[0]))
    for i, x in enumerate(X):
        v, idx = V[i], nbrs[i]
        c = corr_func(x, X[idx], v)
        if softmax_adjusted:
            c = np.exp(c*sigma)
            c /= np.sum(c)
        P[i][idx] = c
    return P


def projection_with_transition_matrix(T, X_emb, correct_density=True, norm_diff=False):
    n = T.shape[0]
    V = np.zeros((n, X_emb.shape[1]))

    if not issparse(T):
        T = csr_matrix(T)

    for i in range(n):
        idx = T[i].indices
        diff_emb = X_emb[idx] - X_emb[i, None]
        if norm_diff:
            diff_emb /= np.linalg.norm(diff_emb, axis=1)[:, None]
        if np.isnan(diff_emb).sum() != 0:
            diff_emb[np.isnan(diff_emb)] = 0
        T_i = T[i].data
        V[i] = T_i.dot(diff_emb)
        if correct_density:
            V[i] -= T_i.mean() * diff_emb.sum(0)

    return V


def density_corrected_transition_matrix(T):
    T = sp.csr_matrix(T, copy=True)

    for i in range(T.shape[0]):
        idx = T[i].indices
        T_i = T[i].data
        T_i -= T_i.mean()
        T[i, idx] = T_i

    return T


def _estimate_sigma(X, V, nbrs, corr_func) -> float:
    """
    Adapted from cellrank
    """
    logits = np.zeros((X.shape[0], X.shape[0]))
    for i, x in enumerate(X):
        v, idx = V[i], nbrs[i]
        c = corr_func(x, X[idx], v)
        logits[i][idx] = c
    logits_abs = np.abs(logits)
    return 1.0 / np.median(logits_abs[logits_abs>0])