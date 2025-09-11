import numpy as np


def init_pca(X, no_dims, contri):
    """
    This function preprocesses data with excessive size and dimensions.

    Parameters are:

    'X'          - N by D matrix. Each row in X represents an observation.
    'no_dims'    - A positive integer specifying the number of dimension of the representation Y.
    'contri'     - Threshold of PCA variance contribution.

    """
    m = X.shape[1]
    X = X - np.mean(X, axis=0)
    # Compute covariance matrix C
    C = np.cov(X, rowvar=False)
    # Compute covariance matrix C
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    lamda, M = np.linalg.eig(C)
    lamda = np.real(lamda)
    # Obtain the best PCA dimension
    if m < 2001:
        ind = np.where(np.cumsum(lamda) / sum(lamda) > contri)
    else:
        ind = np.where(np.cumsum(lamda) / sum(lamda[:2000]) > contri)
    bestDim = max(no_dims + 1, int(ind[0][0]))
    # Apply mapping on the data
    mappedX = X @ np.real(M)[:, :bestDim]

    return mappedX
