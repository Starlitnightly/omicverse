from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from .init_pca import init_pca
from .pps import pps
from .learning_s import learning_s
from .learning_l import learning_l
from .opt_scale import opt_scale
from .clle import clle
import numpy as np
from scipy import sparse


def sude(
        X,
        no_dims=2,
        k1=20,
        normalize=True,
        large=False,
        initialize='le',
        agg_coef=1.2,
        T_epoch=50,
        return_neighbors=False
):
    """
    This function returns representation of the N by D matrix X in the lower-dimensional space. Each row in X
    represents an observation.

    Parameters are:

    'no_dims'      - A positive integer specifying the number of dimension of the representation Y.
                   Default: 2
    'k1'           - A non-negative integer specifying the number of nearest neighbors for PPS to
                   sample landmarks. It must be smaller than N.
                   Default: adaptive
    'normalize'    - Logical scalar. If true, normalize X using min-max normalization. If features in
                   X are on different scales, 'Normalize' should be set to true because the learning
                   process is based on nearest neighbors and features with large scales can override
                   the contribution of features with small scales.
                   Default: True
    'large'        - Logical scalar. If true, the data can be split into multiple blocks to avoid the problem
                   of memory overflow, and the gradient can be computed block by block using 'learning_l' function.
                   Default: False
    'initialize'   - A string specifying the method for initializing Y before manifold learning.
        'le'       - Laplacian eigenmaps.
        'pca'      - Principal component analysis.
        'mds'      - Multidimensional scaling.
                   Default: 'le'
    'agg_coef'     - A positive scalar specifying the aggregation coefficient.
                   Default: 1.2
    'T_epoch'      - Maximum number of epochs to take.
                   Default: 50
    'return_neighbors' - If True, also return neighborhood graph information (connectivities and distances).
                   Default: False

    Returns:
        If return_neighbors=False: Y (embedding)
        If return_neighbors=True: (Y, connectivities, distances)
    """
    # Remove duplicate observations
    X, orig_id = np.unique(X, axis=0, return_inverse=True)

    # Obtain size and dimension of data
    n, dim = X.shape

    # Normalize the data
    if normalize:
        X = MinMaxScaler().fit(X).transform(X)

    # Initialize neighborhood information
    connectivities = None
    distances = None
    
    # Perform PPS to obtain the landmarks
    if k1 > 0:
        if n > 5000 and dim > 50:
            xx = init_pca(X, no_dims, 0.8)
            # Compute KNN once and reuse results
            knn_distances, knn_indices = NearestNeighbors(n_neighbors=k1 + 1).fit(xx).kneighbors(xx)
            get_knn = knn_indices  # Use indices for PPS
            if return_neighbors:
                connectivities, distances = _build_neighborhood_matrices(knn_indices, knn_distances, n)
        else:
            # Compute KNN once and reuse results
            knn_distances, knn_indices = NearestNeighbors(n_neighbors=k1 + 1).fit(X).kneighbors(X)
            get_knn = knn_indices  # Use indices for PPS
            if return_neighbors:
                connectivities, distances = _build_neighborhood_matrices(knn_indices, knn_distances, n)
        _, rnn = np.unique(get_knn, return_counts=True)
        id_samp = pps(get_knn, rnn, 1)
    else:
        get_knn = []
        rnn = []
        id_samp = list(range(n))

    X_samp = X[id_samp]

    # Compute embedding of landmarks
    if not large:
        Y_samp, k2 = learning_s(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch)
    else:
        Y_samp, k2 = learning_l(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch)

    # Compute embedding of non-landmarks
    if k1 > 0:
        id_rest = np.setdiff1d(range(n), id_samp)
        X_rest = X[id_rest]
        Y_rest = np.zeros((len(id_rest), no_dims))
        # Compute the optimal scale for each landmark
        scale = opt_scale(X_samp, Y_samp, k2)
        top_k = no_dims + 1
        near_dis, near_samp = NearestNeighbors(n_neighbors=top_k).fit(X_samp).kneighbors(X_rest)
        for i in range(len(id_rest)):
            near_top_k = near_samp[i]
            top_X = X_samp[near_top_k]
            top_Y = Y_samp[near_top_k]
            N_dis = near_dis[i, 0] * scale[near_top_k[0]]
            # Perform CLLE
            Y_rest[i] = clle(top_X, top_Y, X_rest[i], N_dis)
        YY = np.zeros((n, no_dims))
        YY[id_rest] = Y_rest
        YY[id_samp] = Y_samp
    else:
        YY = Y_samp
    Y = YY[orig_id]
    
    # Reorder neighborhood matrices to match original data ordering
    if return_neighbors and connectivities is not None:
        connectivities = connectivities[orig_id][:, orig_id]
        distances = distances[orig_id][:, orig_id]
        return Y, connectivities, distances
    else:
        return Y


def _build_neighborhood_matrices(knn_indices, knn_distances, n_obs):
    """Build sparse connectivity and distance matrices from KNN results.
    
    Parameters
    ----------
    knn_indices : np.ndarray
        KNN indices array of shape (n_obs, k+1)
    knn_distances : np.ndarray
        KNN distances array of shape (n_obs, k+1)
    n_obs : int
        Number of observations
        
    Returns
    -------
    connectivities : scipy.sparse.csr_matrix
        Sparse connectivity matrix
    distances : scipy.sparse.csr_matrix
        Sparse distance matrix
    """
    # Remove self-connections (first column is usually the point itself)
    knn_indices_no_self = knn_indices[:, 1:]  # Remove first column (self)
    knn_distances_no_self = knn_distances[:, 1:]  # Remove first column (self)
    
    n_neighbors = knn_indices_no_self.shape[1]
    
    # Create row and column indices for sparse matrix
    row_indices = np.repeat(np.arange(n_obs), n_neighbors)
    col_indices = knn_indices_no_self.ravel()
    
    # Create connectivity matrix (binary: 1 if connected, 0 otherwise)
    connectivities_data = np.ones(len(row_indices), dtype=np.float32)
    connectivities = sparse.csr_matrix(
        (connectivities_data, (row_indices, col_indices)),
        shape=(n_obs, n_obs),
        dtype=np.float32
    )
    
    # Create distance matrix
    distances_data = knn_distances_no_self.ravel().astype(np.float32)
    distances = sparse.csr_matrix(
        (distances_data, (row_indices, col_indices)),
        shape=(n_obs, n_obs),
        dtype=np.float32
    )
    
    # Make matrices symmetric by taking maximum of (i,j) and (j,i)
    connectivities = connectivities.maximum(connectivities.T)
    distances = distances.maximum(distances.T)
    
    return connectivities, distances
