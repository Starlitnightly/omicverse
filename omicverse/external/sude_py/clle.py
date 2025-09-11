import numpy as np


def clle(X_samp, Y_samp, X_i, N_dis):
    """
    Constrained Locally Linear Embedding (CLLE)
    This function returns representation of point X_i.

    Parameters are:

    'X_samp'      - High-dimensional features of KNN of point X_i. Each row denotes an observation.
    'Y_samp'      - Low-dimensional embeddings of KNN of point X_i.
    'X_i'         - Current non-landmark point.
    'N_dis'       - Distance between point X_i and its nearest neighbor in lower-dimensional space.

    """
    n = X_samp.shape[0]
    S = (X_samp - X_i) @ (X_samp - X_i).transpose()
    if np.abs(np.linalg.det(S)) <= np.finfo(float).eps:
        S = S + (0.1 ** 2 / n) * np.trace(S) * np.eye(n)
    W = (np.linalg.inv(S) @ np.ones((n, 1))) / (np.ones((1, n)) @ np.linalg.inv(S) @ np.ones((n, 1)))
    Y_0 = W.transpose() @ Y_samp
    dd = np.sqrt((Y_samp[0] - Y_0) @ (Y_samp[0] - Y_0).transpose())
    if dd != 0:
        Y_i = Y_samp[0] + N_dis * (Y_0 - Y_samp[0]) / dd
    else:
        Y_i = Y_samp[0]

    return Y_i