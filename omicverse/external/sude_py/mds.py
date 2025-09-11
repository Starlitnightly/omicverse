from scipy.spatial.distance import cdist
import numpy as np


def mds(X, no_dims):
    """
      This function performs MDS embedding.

      Parameters are:

      'X'          - N by D matrix. Each row in X represents an observation.
      'no_dims'    - A positive integer specifying the number of dimension of the representation Y.

    """
    n = X.shape[0]
    D = cdist(X, X) ** 2
    sumd = np.mean(D, axis=1)
    sumD = np.mean(sumd)
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            B[i][j] = -0.5 * (D[i][j] - sumd[i] - sumd[j] + sumD)
            B[j][i] = B[i][j]
    value, U = np.linalg.eig(B)
    embedX = U[:, :no_dims] @ np.diag(np.sqrt(np.abs(value[:no_dims])))
    return embedX
