from typing import Optional, List, Dict, Tuple, Union
import os
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from numba import njit
import scipy.sparse as sp
from sklearn.metrics import accuracy_score


def flatten(arr: Union[pd.Series, sp.csr_matrix, np.ndarray]) -> np.ndarray:
    if type(arr) == pd.core.series.Series:
        ret = arr.values.flatten()
    elif sp.issparse(arr):
        ret = arr.A.flatten()
    else:
        ret = arr.flatten()
    return ret


def uniform_downsample_cells(X: np.ndarray, downsample):
    n_cells = X.shape[0]
    if 0 < downsample < 1:
        target = int(n_cells * downsample)
    else:
        target = int(downsample)
    
    n_bins = int(np.ceil(np.sqrt(target)))
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)
    x_bin = np.minimum(np.digitize(X[:, 0], x_edges) - 1, n_bins - 1)
    y_bin = np.minimum(np.digitize(X[:, 1], y_edges) - 1, n_bins - 1)
    bins = list(zip(x_bin, y_bin))

    selected_indices = []
    unique_bins = np.unique(bins, axis=0)
    
    for b in unique_bins:
        idxs = [i for i, bin_val in enumerate(bins) if bin_val == tuple(b)]
        if len(idxs) == 0:
            continue
        ix, iy = b
        x_center = (x_edges[ix] + x_edges[ix+1]) / 2
        y_center = (y_edges[iy] + y_edges[iy+1]) / 2
        distances = [np.sqrt((X[i, 0] - x_center)**2 + (X[i, 1] - y_center)**2) for i in idxs]
        best_idx = idxs[np.argmin(distances)]
        selected_indices.append(best_idx)
    
    selected_indices = np.array(selected_indices)
    if selected_indices.size > target:
        selected_indices = np.random.choice(selected_indices, size=target, replace=False)
    
    return selected_indices


def knn(
    X: np.ndarray,
    k: int,
    query_X: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    exclude_self: bool = True,
    knn_dim: int = 10,
    pynn_num: int = 5000,
    pynn_dim: int = 2,
    pynn_rand_state: int = 0,
    n_jobs: int = -1,
    return_nbrs: bool = False,
    **kwargs,
):
    if method is None:
        logging.info("method arg is None, choosing methods automatically...")
        if X.shape[0] > pynn_num and X.shape[1] > pynn_dim:
            method = "pynn"
        else:
            if X.shape[1] > knn_dim:
                method = "ball_tree"
            else:
                method = "kd_tree"
        logging.info("method %s selected" % (method))

    if query_X is None:
        query_X = X

    if method.lower() in ["pynn", "umap"]:
        from pynndescent import NNDescent

        nbrs = NNDescent(
            X,
            n_neighbors=k + 1,
            n_jobs=n_jobs,
            random_state=pynn_rand_state,
            **kwargs,
        )
        nbrs_idx, dists = nbrs.query(query_X, k=k + 1)
    elif method in ["ball_tree", "kd_tree"]:
        from sklearn.neighbors import NearestNeighbors

        # print("***debug X_data:", X_data)
        nbrs = NearestNeighbors(
            n_neighbors=k + 1,
            algorithm=method,
            n_jobs=n_jobs,
            **kwargs,
        ).fit(X)
        dists, nbrs_idx = nbrs.kneighbors(query_X)
    else:
        raise ImportError(f"nearest neighbor search method {method} is not supported")

    nbrs_idx = np.array(nbrs_idx)
    if exclude_self:
        nbrs_idx = nbrs_idx[:, 1:]
        dists = dists[:, 1:]
    if return_nbrs:
        return nbrs_idx, dists, nbrs, method
    return nbrs_idx, dists    


def adj_to_knn(adj: np.ndarray, n_neighbors: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the adjacency matrix of a nearest neighbor graph to the indices and weights for a knn graph.

    Args:
        adj: Adjacency matrix (n x n) of the nearest neighbor graph.
        n_neighbors: The number of nearest neighbors of the kNN graph. Defaults to 15.

    Returns:
        A tuple (idx, wgt) where idx is the matrix (n x n_neighbors) storing the indices for each node's n_neighbors
        nearest neighbors and wgt is the matrix (n x n_neighbors) storing the wights on the edges for each node's
        n_neighbors nearest neighbors.
    """

    n_cells = adj.shape[0]
    idx = np.zeros((n_cells, n_neighbors), dtype=int)
    wgt = np.zeros((n_cells, n_neighbors), dtype=adj.dtype)

    for cur_cell in range(n_cells):
        # returns the coordinate tuple for non-zero items
        cur_neighbors = adj[cur_cell, :].nonzero()

        # set itself as the nearest neighbor
        idx[cur_cell, :] = cur_cell
        wgt[cur_cell, :] = 0

        # there could be more or less than n_neighbors because of an approximate search
        cur_n_neighbors = len(cur_neighbors[1])

        if cur_n_neighbors > n_neighbors - 1:
            sorted_indices = np.argsort(adj[cur_cell][:, cur_neighbors[1]].A)[0][: (n_neighbors - 1)]
            idx[cur_cell, 1:] = cur_neighbors[1][sorted_indices]
            wgt[cur_cell, 1:] = adj[cur_cell][0, cur_neighbors[1][sorted_indices]].A
        else:
            idx_ = np.arange(1, (cur_n_neighbors + 1))
            idx[cur_cell, idx_] = cur_neighbors[1]
            wgt[cur_cell, idx_] = adj[cur_cell][:, cur_neighbors[1]].A

    return idx, wgt


def angle(vector1, vector2):
    """Returns the angle in radians between given vectors"""
    v1_norm, v1_u = unit_vector(vector1)
    v2_norm, v2_u = unit_vector(vector2)

    if v1_norm == 0 or v2_norm == 0:
        return np.nan
    else:
        minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
        if minor == 0:
            sign = 1
        else:
            sign = -np.sign(minor)
        dot_p = np.dot(v1_u, v2_u)
        dot_p = min(max(dot_p, -1.0), 1.0)
        return sign * np.arccos(dot_p)


# @njit(cache=True, nogil=True) # causing numba error_write issue
def unit_vector(vector):
    """Returns the unit vector of the vector."""
    vec_norm = np.linalg.norm(vector)
    if vec_norm == 0:
        return vec_norm, vector
    else:
        return vec_norm, vector / vec_norm


def velo_angle(v1, v2):
    if v1.shape[0] != v2.shape[0]:
        raise ValueError('Cell number inconsistent ERROR.')
    if v1.shape[1] != v2.shape[1]:
        raise ValueError('Velocity dimension inconsistent ERROR.')
    N = v1.shape[0]
    v1 = v1.A if sp.issparse(v1) else v1
    v2 = v2.A if sp.issparse(v2) else v2

    cell_angles = np.zeros(N, dtype=float)
    for i in range(N):
        cell_angles[i] = angle(v1[i].astype('float64'), v2[i].astype('float64'))
        
    return cell_angles


def velocity_consistency(adata, vkey, gvkey, V_threshold):
    """ Compute velocity consistency between original one and graphvelo output. """
    v1 = adata.layers[vkey].A if sp.issparse(adata.layers[vkey]) else adata.layers[vkey]
    v2 = adata.layers[gvkey].A if sp.issparse(adata.layers[gvkey]) else adata.layers[gvkey]
    threshold_v1 = np.percentile(abs(v1), V_threshold)
    threshold_v2 = np.percentile(abs(v2), V_threshold)
    v1_sign = v1 > -threshold_v1
    v2_sign = v2 > -threshold_v2
    score = [
        accuracy_score(v2_sign[:, var_id], v1_sign[:, var_id])
        for var_id in range(v1.shape[1])
    ]
    return np.array(score)


# sampling-related
def sample_edge_cells(adata, cluster_key: str, edges: list, nbrs_idx: list=None, exclude_target: bool=False):
    if cluster_key not in adata.obs.keys():
        raise ValueError(f'`{cluster_key}` not found in adata.obs.')
    if 'neighbors' not in adata.uns.keys() and nbrs_idx is None:
        raise ValueError(f'Please run `dyn.tl.neighbors` before sample edge cells.')
    edge_cells_idx = {}

    def keep_type(adata, nodes, cluster, cluster_key):
        return nodes[adata.obs[cluster_key][nodes].values == cluster]
    
    for s, t in edges:
        s_idx = np.where(adata.obs[cluster_key] == s)[0]
        s_nbrs = adata.uns['neighbors']['indices'][s_idx] if nbrs_idx is None else nbrs_idx[s_idx]
        t_idx_list = list(map(lambda nodes:keep_type(adata, nodes, t, cluster_key), s_nbrs))
        filtered_s_idx = set([idx for idx, neighbors in zip(s_idx, t_idx_list) if len(neighbors) > 0])
        t_idx = set([idx for i in t_idx_list for idx in i])
        if not exclude_target:
            cells_idx = filtered_s_idx | t_idx
        else:
            cells_idx = filtered_s_idx
        edge_cells_idx[f'{s}->{t}'] = list(cells_idx)

        logging.info(f'Sample {len(cells_idx)} cells in edge `{s}->{t}`.')
    return edge_cells_idx


def is_meg(x, y, trend=0, num_bins=100, min_samples=10, alpha=0.05):
    '''
    Adapted from phylovelo criteria. 
    Determine whether a gene is MEG (monotonically expressed gene) for continuous data.
    
    Args:
        x: 
            Time
        y:
            Gene expression levels (list or array-like, continuous)
        trend:
            Positive for increasing MEGs, negative for decreasing MEGs
        num_bins:
            Number of bins to use for discretizing the continuous data
        min_samples:
            Minimum number of samples required to perform the test
        alpha:
            Significance level for the Mann-Whitney U test
    
    Returns:
        bool:
            True if the gene is MEG, False otherwise
    '''
    from scipy.stats import mannwhitneyu
    x = np.array(x)
    y = np.array(y)
    
    # Compute quantile-based bin edges for x
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(x, quantiles)

    # Digitize x based on quantile edges
    digitized_x = np.digitize(x, bin_edges) - 1  # Bin indices
    
    # Define the condition based on the trend
    alternative = 'greater' if trend > 0 else 'less'
    
    # Calculate the number of tests to be performed
    num_tests = num_bins - 1
    adjusted_alpha = alpha / num_tests  # Bonferroni correction
    
    # Iterate through the range of binned expression levels
    for k in range(num_bins - 1):
        group1 = y[digitized_x == k]
        group2 = y[digitized_x == k + 1]
        
        if len(group1) < min_samples or len(group2) < min_samples:
            continue
        
        # Perform Mann-Whitney U test between consecutive binned levels
        p_value = mannwhitneyu(group1, group2, alternative=alternative)[1]
        
        if p_value < adjusted_alpha:
            return False

    return True


def gene_wise_confidence(
    adata,
    group: str,
    lineage_dict: Optional[Dict[str, str]] = None,
    genes: Optional[List] = None,
    ekey: str = "M_s",
    vkey: str = "velocity_S",
    X_data: Optional[np.ndarray] = None,
    V_data: Optional[np.ndarray] = None,
    V_threshold: float = 1,
) -> None:
    """Adapted from `dynamo`.

    Args:
        adata: An AnnData object.
        group: The column key/name that identifies the cell state grouping information of cells. This will be used for
            calculating gene-wise mack_val score in each cell state.
        lineage_dict: A dictionary describes lineage priors. Keys correspond to the group name from `group` that
            corresponding to the state of one progenitor type while values correspond to the group names from `group`
            that corresponding to the states of one or multiple terminal cell states. The best practice for determining
            terminal cell states are those fully functional cells instead of intermediate cell states. Note that in
            python a dictionary key cannot be a list, so if you have two progenitor types converge into one terminal
            cell state, you need to create two records each with the same terminal cell as value but different
            progenitor as the key. Value can be either a string for one cell group or a list of string for multiple cell
            groups. Defaults to None.
        genes: The list of genes that will be used to gene-wise mack_val score calculation. If `None`, all genes that
            go through velocity estimation will be used. Defaults to None.
        ekey: The layer that will be used to retrieve data for identifying the gene is in induction or repression phase
            at each cell state. If `None`, `.X` is used. Defaults to "M_s".
        vkey: The layer that will be used to retrieve velocity data for calculating gene-wise mack_val. If `None`,
            `velocity_S` is used. Defaults to "velocity_S".
        X_data: The user supplied data that will be used for identifying the gene is in induction or repression phase at
            each cell state directly. Defaults to None.
        V_data: The user supplied data that will be used for calculating gene-wise mack_val directly. Defaults to None.
        V_threshold: The threshold of velocity to calculate the gene wise mack_val. Defaults to 1.

    Raises:
        ValueError: `X_data` is provided but `genes` does not correspond to its columns.
        ValueError: `X_data` is provided but `genes` does not correspond to its columns.
        Exception: The progenitor cell extracted from lineage_dict is not in `adata.obs[group]`.
        Exception: The terminal cell extracted from lineage_dict is not in `adata.obs[group]`.
    """
    if genes is not None:
        genes = adata.var_names.intersection(genes).to_list()
        if len(genes) == 0:
            raise ValueError("No genes from your genes list appear in your adata object.")
    else:
        tmp_V = adata.layers[vkey].A if sp.issparse(adata.layers[vkey]) else adata.layers[vkey]
        genes = adata[:, ~np.isnan(tmp_V.sum(0))].var_names

    if X_data is None or V_data is None:
        X_data = adata[:, genes].layers[ekey]
        V_data = adata[:, genes].layers[vkey]
    else:
        if V_data.shape[1] != X_data.shape[1] or len(genes != X_data.shape[1]):
            raise ValueError(
                f"When providing X_data, a list of genes name that corresponds to the columns of X_data "
                f"must be provided")

    sparse, sparse_v = sp.issparse(X_data), sp.issparse(V_data)

    mack_val = []
    for i_gene, gene in tqdm(
        enumerate(genes),
        desc="calculating gene velocity vectors mack_val based on phase "
        "portrait location with priors of progenitor/mature cell types",
    ):
        all_vals = X_data[:, i_gene].A if sparse else X_data[:, i_gene]
        all_vals_v = V_data[:, i_gene].A if sparse_v else V_data[:, i_gene]

        for progenitors_groups, mature_cells_groups in lineage_dict.items():
            progenitors_groups = [progenitors_groups]
            if type(mature_cells_groups) is str:
                mature_cells_groups = [mature_cells_groups]

            for i, progenitor in enumerate(progenitors_groups):
                prog_vals = all_vals[adata.obs[group] == progenitor]
                prog_vals_v = all_vals_v[adata.obs[group] == progenitor]
                if len(prog_vals_v) == 0:
                    logging.error(f"The progenitor cell type {progenitor} is not in adata.obs[{group}].")
                    raise Exception()

                threshold_val = np.percentile(abs(all_vals_v), V_threshold)

                for j, mature in enumerate(mature_cells_groups):
                    mature_vals = all_vals[adata.obs[group] == mature]
                    mature_vals_v = all_vals_v[adata.obs[group] == mature]
                    if len(mature_vals_v) == 0:
                        logging.error(f"The terminal cell type {progenitor} is not in adata.obs[{group}].")
                        raise Exception()

                    if np.nanmedian(prog_vals) - np.nanmedian(mature_vals) > 0:
                        # repression phase (bottom curve -- phase curve below the linear line indicates steady states)
                        prog_confidence = 1 - sum(prog_vals_v > -threshold_val) / len(
                            prog_vals_v
                        )  # most cells should downregulate / ss
                        mature_confidence = 1 - sum(mature_vals_v > -threshold_val) / len(
                            mature_vals_v
                        )  # most cell should downregulate / ss
                    else:
                        # induction phase (upper curve -- phase curve above the linear line indicates steady states)
                        prog_confidence = 1 - sum(prog_vals_v < threshold_val) / len(
                            prog_vals_v
                        )  # most cells should upregulate / ss
                        mature_confidence = 1 - sum(mature_vals_v < threshold_val) / len(
                            mature_vals_v
                        )  # most cell should upregulate / ss

                    mack_val.append(
                        (
                            gene,
                            progenitor,
                            mature,
                            prog_confidence,
                            mature_confidence,
                        )
                    )

    mack_val = pd.DataFrame(
        mack_val,
        columns=[
            "gene",
            "progenitor",
            "mature",
            "prog_confidence",
            "mature_confidence",
        ],
    )
    mack_val.astype(dtype={"prog_confidence": "float64", "prog_confidence": "float64"})
    adata.var["avg_prog_confidence"], adata.var["avg_mature_confidence"] = (
        np.nan,
        np.nan,
    )
    avg = mack_val.groupby("gene")[["prog_confidence", "mature_confidence"]].mean()
    avg = avg.reset_index().set_index("gene")
    adata.var.loc[genes, "avg_prog_confidence"] = avg.loc[genes, "prog_confidence"]
    adata.var.loc[genes, "avg_mature_confidence"] = avg.loc[genes, "mature_confidence"]

    adata.uns["gene_wise_confidence"] = mack_val


@njit
def compute_confidence_numba(x, v, nbrs_idx, t, eps=1e-5):
    N, k = nbrs_idx.shape  # N cells and k neighbors per cell
    scores = np.empty(N, dtype=np.float64)
    for i in range(N):
        count = 0
        for j in range(k):
            nbr = nbrs_idx[i, j]
            dt = t[nbr] - t[i] + eps
            d_val = x[nbr] - x[i]
            # Compute sign of d_val/dt
            ratio = d_val / dt
            if ratio > 0:
                s = 1
            elif ratio < 0:
                s = -1
            else:
                s = 0
            # Compute sign of v[i]
            if v[i] > 0:
                s_v = 1
            elif v[i] < 0:
                s_v = -1
            else:
                s_v = 0
            if s == s_v:
                count += 1
        scores[i] = count / k  # average accuracy for cell i
    return scores

def compute_confidence(gene_i, gene, X_data, V_data, nbrs_idx, N, t, return_score):
    # Extract the gene-specific expression and velocity as 1D arrays.
    x = flatten(X_data[:, gene_i])
    v = flatten(V_data[:, gene_i])
    # Call the Numba-accelerated function.
    scores = compute_confidence_numba(x, v, nbrs_idx, t)
    if not return_score:
        return (gene, scores.mean())
    else:
        return (gene, scores.mean(), scores)

def mack_score(
    adata,
    n_neighbors: Optional[int] = None,
    basis: Optional[str] = None,
    tkey: Optional[str] = None,
    genes: Optional[List] = None,
    ekey: str = "M_s",
    vkey: str = "velocity_S",
    X_data = None,
    V_data = None,
    n_jobs= -1,
    add_prefix: Optional[str] = None,
    return_score: bool = False,
):   
    # Determine the number of jobs to use.
    if (n_jobs is None or not isinstance(n_jobs, int) or n_jobs < 0 or
            n_jobs > os.cpu_count()):
        n_jobs = os.cpu_count()

    # Restrict genes if provided.
    if genes is not None:
        genes = adata.var_names.intersection(genes).to_list()
        if len(genes) == 0:
            raise ValueError("No genes from your genes list appear in your adata object.")
    else:
        tmp_V = adata.layers[vkey].A if sp.issparse(adata.layers[vkey]) else adata.layers[vkey]
        genes = adata[:, ~np.isnan(tmp_V.sum(0))].var_names

    # Get X_data and V_data if not provided.
    if X_data is None or V_data is None:
        X_data = adata[:, genes].layers[ekey]
        V_data = adata[:, genes].layers[vkey]
    else:
        if V_data.shape[1] != X_data.shape[1] or len(genes) != X_data.shape[1]:
            raise ValueError(
                f"When providing X_data, a list of gene names that corresponds to the columns of X_data "
                f"must be provided")

    # Get kNN indices.
    if n_neighbors is None:
        nbrs_idx = adata.uns['neighbors']['indices']
    else:
        basis_for_knn = 'X_' + basis
        if basis_for_knn in adata.obsm.keys():
            logging.info(f"Compute knn in {basis.upper()} basis...")
            nbrs_idx, _ = knn(adata.obsm[basis_for_knn], n_neighbors)
        else:
            logging.info(f"Compute knn in original basis...")
            X_for_knn = adata.X.A if sp.issparse(adata.X) else adata.X
            nbrs_idx, _ = knn(X_for_knn, n_neighbors)
    
    # Ensure nbrs_idx is a 2D NumPy array.
    if not isinstance(nbrs_idx, np.ndarray):
        nbrs_idx = np.array(nbrs_idx)
    
    N = adata.n_obs
    t = adata.obs[tkey].to_numpy() if hasattr(adata.obs[tkey], "to_numpy") else np.array(adata.obs[tkey])

    res = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_confidence)(
            gene_i, 
            gene,
            X_data, 
            V_data, 
            nbrs_idx,
            N,
            t,
            return_score,
        )
        for gene_i, gene in tqdm(
            enumerate(genes),
            total=len(genes),
            desc=f"calculating manifold-consistent scores in {n_jobs} cpu(s)",
        )
    )
        
    mack_val = pd.DataFrame(
        res,
        columns=[
            "gene",
            "mack_score",
        ]
    )

    mack_val = mack_val.reset_index().set_index("gene")
    mack_val = mack_val.astype({"mack_score": "float64"})

    velo_conf_key = "mack_score" if add_prefix is None else add_prefix+"_mack_score"
    adata.var[velo_conf_key] = np.nan
    adata.var.loc[genes, velo_conf_key] = mack_val.loc[genes, "mack_score"]
    
    if return_score:
        return mack_val