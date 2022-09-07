import numpy as np


def generate_bigclam(F, B=None, D=None, p_no_comm=0, seed=None):
    """Produce the graph according to the BigCLAM model.

    Parameters
    ----------
    F : np.ndarray, shape [N, K]
        Nonnegative community affiliation matrix.
    B : np.ndarray, shape [K, K]
        Compatiblity matrix.
    D : np.ndarray, shape [N]
        Degree vector.
    p_no_comm : float
        Background edge probability.
    seed : int or None
        Random seed.

    Returns
    -------
    A : np.ndarray, shape [num_nodes, num_nodes]
        Adjacency matrix of the generated graph.

    """
    N, K = F.shape
    if B is None and D is not None:
        raise ValueError("Can only use D if B is provided.")
    if B is None:
        B = np.eye(K)
    if np.any(F < 0):
        raise ValueError("F must be nonnegative.")
    if np.any(B < 0):
        raise ValueError("B must be nonnegative.")
    if D is not None:
        D[D == 0]= 1
        if np.any(D < 0):
            raise ValueError("D must be nonnegative.")
        F = F * D[:, None]

    if seed is not None:
        np.random.seed(seed)
    eps = np.log(1 / (1 - p_no_comm))
    probas = 1 - np.exp(-F @ B @ F.T - eps)
    A = np.tril(np.random.binomial(1, probas))
    A = A + A.T
    np.fill_diagonal(A, 0)
    return A

