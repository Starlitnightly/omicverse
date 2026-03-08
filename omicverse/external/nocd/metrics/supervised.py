import numpy as np
import warnings

from ..utils import coms_list_to_matrix

__all__ = [
    'symmetric_jaccard',
    'overlapping_nmi',
]


def symmetric_jaccard(coms_1, coms_2):
    """Quantify agreement between two community assignments based on symmetric Jaccard similarity.

    Computed as in the CESNA paper as
    0.5 * (1 / |C1| * sum_{c1 in C1} max_{c2 in C2} jac(c1, c2) +
           1 / |C2| * sum_{c2 in C2} max_{c1 in C1} jac(c1, c2))

    Parameters
    ----------
    coms_1 : list of len [num_coms] or array-like of shape [num_nodes, num_coms]
        First community assignment to compare.
    coms_2 : list of len [num_coms] or array-like of shape [num_nodes, num_coms]
        Second community assignment to compare.

    Returns
    -------
    symmetric_jaccard_similarity : float
        Symmetric average best Jaccard similarity between two community assignments.

    """
    # Convert community assignments to matrix format
    if isinstance(coms_1, list):
        F1 = coms_list_to_matrix(coms_1)
    elif len(coms_1.shape) == 2:
        F1 = coms_1
    else:
        raise ValueError("coms_1 must be either a list or a matrix.")
    if isinstance(coms_2, list):
        F2 = coms_list_to_matrix(coms_2)
    elif len(coms_2.shape) == 2:
        F2 = coms_2
    else:
        raise ValueError("coms_2 must be either a list or a matrix.")

    intersections = F1.T.dot(F2)
    sum_F1 = F1.sum(0)
    sum_F2 = F2.sum(0)
    unions = (np.ravel(sum_F2) + np.ravel(sum_F1)[:, None]) - intersections
    jacs = intersections / unions
    return 0.5 * (jacs.max(0).mean() + jacs.max(1).mean())


def overlapping_nmi(X, Y):
    """Compute NMI between two overlapping community covers.

    Parameters
    ----------
    X : array-like, shape [N, m]
        Matrix with samples stored as columns.
    Y : array-like, shape [N, n]
        Matrix with samples stored as columns.

    Returns
    -------
    nmi : float
        Float in [0, 1] quantifying the agreement between the two partitions.
        Higher is better.

    References
    ----------
    McDaid, Aaron F., Derek Greene, and Neil Hurley.
    "Normalized mutual information to evaluate overlapping
    community finding algorithms."
    arXiv preprint arXiv:1110.2515 (2011).

    """
    if not ((X == 0) | (X == 1)).all():
        raise ValueError("X should be a binary matrix")
    if not ((Y == 0) | (Y == 1)).all():
        raise ValueError("Y should be a binary matrix")

    if X.shape[1] > X.shape[0] or Y.shape[1] > Y.shape[0]:
        warnings.warn("It seems that you forgot to transpose the F matrix")
    X = X.T
    Y = Y.T
    def cmp(x, y):
        """Compare two binary vectors."""
        a = (1 - x).dot(1 - y)
        d = x.dot(y)
        c = (1 - y).dot(x)
        b = (1 - x).dot(y)
        return a, b, c, d

    def h(w, n):
        """Compute contribution of a single term to the entropy."""
        if w == 0:
            return 0
        else:
            return -w * np.log2(w / n)

    def H(x, y):
        """Compute conditional entropy between two vectors."""
        a, b, c, d = cmp(x, y)
        n = len(x)
        if h(a, n) + h(d, n) >= h(b, n) + h(c, n):
            return h(a, n) + h(b, n) + h(c, n) + h(d, n) - h(b + d, n) - h(a + c, n)
        else:
            return h(c + d, n) + h(a + b, n)
    def H_uncond(X):
        """Compute unconditional entropy of a single binary matrix."""
        return sum(h(x.sum(), len(x)) + h(len(x) - x.sum(), len(x)) for x in X)

    def H_cond(X, Y):
        """Compute conditional entropy between two binary matrices."""
        m, n = X.shape[0], Y.shape[0]
        scores = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                scores[i, j] = H(X[i], Y[j])
        return scores.min(axis=1).sum()

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimensions of X and Y don't match. (Samples must be stored as COLUMNS)")
    H_X = H_uncond(X)
    H_Y = H_uncond(Y)
    I_XY = 0.5 * (H_X + H_Y - H_cond(X, Y) - H_cond(Y, X))
    return I_XY / max(H_X, H_Y)
