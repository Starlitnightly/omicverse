import numpy as np


def pca(X, no_dims):
    """
     This function performs PCA embedding.

     Parameters are:

     'X'          - N by D matrix. Each row in X represents an observation.
     'no_dims'    - A positive integer specifying the number of dimension of the representation Y.

     """
    X = X - np.mean(X, axis=0)
    C = np.cov(X, rowvar=False)
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    evalues, evectors = np.linalg.eig(C)
    mappedX = X @ np.real(evectors)[:, :no_dims]
    return mappedX
