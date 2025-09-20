import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from umap.umap_ import fuzzy_simplicial_set


def get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors):
    """
    Copied out of scanpy.neighbors
    """
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()


def compute_connectivities_umap(
    knn_indices, knn_dists, n_obs, n_neighbors, set_op_mix_ratio=1.0, local_connectivity=1.0
):
    r"""
    Copied out of scanpy.neighbors

    This is from umap.fuzzy_simplicial_set [McInnes18]_.
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """
    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))

    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    distances = get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors)

    return distances, connectivities.tocsr()


# ####################--------########################


def spatial_neighbours(coords, n_sp_neighbors=7, radius=None, include_source_location=True, sample_id=None):
    """
    Find spatial neighbours using the number of neighbours or radius (KDTree approach).

    :param coords: numpy.ndarray with x,y positions of spots.
    :param n_sp_neighbors: how many spatially-adjacent neighbors to report for each spot (including the source spot).
     Use 7 for hexagonal grid.
    :param radius: Supersedes `n_sp_neighbors` - radius within which to report spatially-adjacent neighbors for each spot. Pick radius based on spot size.
    :param include_source_location: include the observation itself into the list of neighbours.
    :param sample_id: pd.Series or np.array listing sample membership for each observation (each row of coords).
    """

    # create and query spatial proximity tree within each sample
    if radius is None:
        if include_source_location:
            coord_ind = np.zeros((coords.shape[0], n_sp_neighbors))
        else:
            coord_ind = np.zeros((coords.shape[0], n_sp_neighbors - 1))
    else:
        coord_ind = np.zeros(coords.shape[0])

    if sample_id is None:
        sample_id = np.array(["sample" for i in range(coords.shape[0])])

    total_ind = np.arange(0, coords.shape[0]).astype(int)

    for sam in np.unique(sample_id):
        sam_ind = np.isin(sample_id, [sam])
        coord_tree = KDTree(coords[sam_ind, :])
        if radius is None:
            n_list = coord_tree.query(coords[sam_ind, :], k=n_sp_neighbors, return_distance=False)
            n_list = np.array(n_list)
            # replace sample-specific indices with a global index
            for c in range(n_list.shape[1]):
                n_list[:, c] = total_ind[sam_ind][n_list[:, c]]

            if include_source_location:
                coord_ind[sam_ind, :] = n_list
            else:
                n_list_sel = n_list != np.arange(sam_ind.sum()).reshape(sam_ind.sum(), 1)
                coord_ind[sam_ind, :] = n_list[n_list_sel].reshape((sam_ind.sum(), n_sp_neighbors - 1))

        else:
            coord_ind[sam_ind] = coord_tree.query_radius(coords[sam_ind, :], radius, count_only=False)

    return coord_ind.astype(int)


def sum_neighbours(X_data, neighbours):
    """
    Sum X_data values across neighbours.

    :param coords: numpy.ndarray with variable measurements for each observation  (observations * variables)
    :param neighbours: numpy.ndarray with neigbour indices for each observation (observations * neigbours)
    """

    return np.sum([X_data[neighbours[:, n], :] for n in range(neighbours.shape[1])], axis=0)


# ####################--------########################


def spatial_knn(
    coords, expression, n_neighbors=14, n_sp_neighbors=7, radius=None, which_exprs_dims=None, sample_id=None
):
    """
    A variant on the standard knn neighbor graph inference procedure that also includes the spatial neighbors of each spot.
    With help from Krzysztof Polanski.

    :param coords: numpy.ndarray with x,y positions of spots.
    :param expression: numpy.ndarray with expression of programmes / cluster expression (cols) of spots (rows).
    :param n_neighbors: how many non-spatially-adjacent neighbors to report for each spot
    :param n_sp_neighbors: how many spatially-adjacent neighbors to report for each spot. Use 7 for hexagonal grid.
    :param radius: Supercedes `n_sp_neighbors` - radius within which to report spatially-adjacent neighbors for each spot. Pick radius based on spot size.
    :param which_exprs_dims: which expression dimensions to use (cols)?
    """

    # create and query spatial proximity tree within each sample
    if radius is None:
        coord_ind = np.zeros((coords.shape[0], n_sp_neighbors))
    else:
        coord_ind = np.zeros(coords.shape[0])

    for sam in sample_id.unique():
        coord_tree = KDTree(coords[sample_id.isin([sam]), :])
        if radius is None:
            coord_ind[sample_id.isin([sam]), :] = coord_tree.query(
                coords[sample_id.isin([sam]), :], k=n_sp_neighbors, return_distance=False
            )
        else:
            coord_ind[sample_id.isin([sam])] = coord_tree.query_radius(
                coords[sample_id.isin([sam]), :], radius, count_only=False
            )

    # if selected dimensions not provided choose all
    if which_exprs_dims is None:
        which_exprs_dims = np.arange(expression.shape[1])
    # print(which_exprs_dims)

    # extract and index the appropriate bit of the PCA
    pca = expression[:, which_exprs_dims]
    ckd = cKDTree(pca)
    # the actual number of neighbours - you'll get seven extra spatial neighbours in the thing
    knn = n_neighbors + n_sp_neighbors

    # identify the knn for each spot. this is guaranteed to contain at least n_neighbors non-adjacent spots
    # this is exactly what we're after
    ckdout = ckd.query(x=pca, k=knn, n_jobs=-1)

    # create numeric vectors for subsetting later
    numtemp = np.arange(expression.shape[0])
    rowtemp = np.arange(knn)

    # rejigger the neighour pool by including the spatially adjacent ones
    for i in np.arange(expression.shape[0]):
        # identify the spatial neighbours for the spot and compute their distance
        mask = np.isin(numtemp, coord_ind[i])

        # filter spatial neighbours by sample
        if sample_id is not None:
            mask = mask & sample_id.isin([sample_id[i]])

        neigh = numtemp[mask]
        ndist_temp = pca[mask, :] - pca[i, :]
        ndist_temp = ndist_temp.reshape((mask.sum(), pca.shape[1]))
        ndist = np.linalg.norm(ndist_temp, axis=1)

        # how many non-adjacent neighbours will we get to keep?
        # (this fluctuates as e.g. edge spots will have fewer hex neighbours)
        kpoint = knn - len(neigh)

        # the indices of the top kpoint number of non-adjacent neighbours (by excluding adjacent ones from the set)
        inds = rowtemp[[i not in neigh for i in ckdout[1][0, :]]][:kpoint]

        # keep the identified top non-adjacent neighbours
        ckdout[0][i, :kpoint] = ckdout[0][i, inds]
        ckdout[1][i, :kpoint] = ckdout[1][i, inds]

        # add the spatial neighbours in the remaining spots of the knn graph
        ckdout[0][i, kpoint:] = ndist
        ckdout[1][i, kpoint:] = neigh

    # sort each row of the graph in ascending distance order
    # (sometimes spatially adjacent neighbours are some of the top ones)
    knn_distances, knn_indices = ckdout
    newidx = np.argsort(knn_distances, axis=1)
    knn_indices = knn_indices[np.arange(np.shape(knn_indices)[0])[:, np.newaxis], newidx]
    knn_distances = knn_distances[np.arange(np.shape(knn_distances)[0])[:, np.newaxis], newidx]

    # compute connectivities and export as a dictionary
    dist, cnts = compute_connectivities_umap(knn_indices, knn_distances, knn_indices.shape[0], knn_indices.shape[1])
    neighbors = {
        "distances": dist,
        "connectivities": cnts,
        "params": {"n_neighbors": n_neighbors + n_sp_neighbors, "method": "spot_factors2knn", "metric": "euclidean"},
    }
    return neighbors


# ####################--------########################


def spot_factors2knn(
    adata,
    coord_col=["x", "y"],
    sample_col="sample.x",
    node_name="nUMI_factors",
    sample_type="mean",
    n_neighbors=14,
    n_sp_neighbors=7,
    which_exprs_dims=None,
    which_sample=None,
):
    r"""Construct spatially aware KNN graph using W spot weights
    :param adata: anndata object with spot weights.
    :param coord_col: anndata.obs columns containing spatial coordinates.
    :param sample_col: anndata.obs columns containing individual Visium sample identifier.
    :param node_name: model paramter representing spot contributions of cell types W.
    :param sample_type: use 'means' or 5% quantile ('q05') of the posterior of W?
    :param n_neighbors: number of expression neighbours, between spots within and across samples.
    :param n_sp_neighbors: number of spatial neighbours, only within spots of the same sample. Defaults to 7 to get
        direct neighbours in the hexagonal grid. 13 and 19 give the next rows of spatial neighbours.
    :param which_exprs_dims: select specific cell states from W
    :param which_sample: select one or several samples to construct the graph
    :return: updated anndata with neighbour graph in adata.uns['neighbors']
    """

    if sample_type not in ["mean", "sds", "q05"]:
        raise ValueError("sample_type should be one of `['mean', 'sds', 'q05']`.")

    obs = adata.obs

    # select sample
    if which_sample is not None:
        obs = obs[which_sample, :]

    obs_col = obs.columns

    # extract needed info
    coords = obs[coord_col].values
    col_ind = [sample_type + "_" + node_name in i for i in obs_col.tolist()]
    exprs = obs.loc[:, col_ind].values
    sample_id = obs[sample_col]

    adata.uns["neighbors"] = spatial_knn(
        coords,
        expression=exprs,
        n_neighbors=n_neighbors,
        n_sp_neighbors=n_sp_neighbors,
        which_exprs_dims=which_exprs_dims,
        sample_id=sample_id,
    )

    return adata
