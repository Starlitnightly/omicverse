from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import numpy as np


def opt_scale(X, Y, k_num):
    """
      This function computes the optimal scales of landmarks.

      Parameters are:

      'X'       - N by D matrix. Each row in X represents high-dimensional features of a landmark.
      'Y'       - N by d matrix. Each row in Y represents low-dimensional embedding of a landmark.
      'k_num'   - A non-negative integer specifying the number of KNN.

    """
    n = X.shape[0]
    get_knn = NearestNeighbors(n_neighbors=k_num + 1).fit(Y).kneighbors(Y, return_distance=False)
    scale = np.zeros((n, 1))
    for i in range(n):
        XDis = cdist(X[get_knn[i]], X[get_knn[i]])
        YDis = cdist(Y[get_knn[i]], Y[get_knn[i]])
        scale[i] = np.sum(XDis * YDis) / max(np.sum(XDis ** 2), np.finfo(float).tiny)

    return scale
