"""
Clustering functions for monocle2_py.

Implements clusterCells (Louvain, Leiden) and clusterGenes.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
import igraph as ig


def _jaccard_coeff(nn_matrix, weighted=False):
    """
    Compute Jaccard coefficient between nearest-neighbor sets.

    Parameters
    ----------
    nn_matrix : np.ndarray, (N, k)
        Matrix of k nearest neighbor indices for each point.
    weighted : bool

    Returns
    -------
    np.ndarray, (M, 3) with columns [from, to, weight]
    """
    N, k = nn_matrix.shape
    links = []

    # Build neighbor sets
    neighbor_sets = [set(nn_matrix[i]) for i in range(N)]

    for i in range(N):
        for j_idx in range(k):
            j = nn_matrix[i, j_idx]
            if j <= i:
                continue
            # Jaccard similarity
            intersection = len(neighbor_sets[i] & neighbor_sets[j])
            union = len(neighbor_sets[i] | neighbor_sets[j])
            if union > 0:
                jac = intersection / union
            else:
                jac = 0
            if jac > 0:
                links.append([i, j, jac])

    return np.array(links) if links else np.empty((0, 3))


def cluster_cells(adata, method='leiden', k=50, resolution_parameter=0.1,
                  louvain_iter=1, verbose=False, **kwargs):
    """
    Cluster cells using graph-based methods.

    Parameters
    ----------
    adata : AnnData
    method : str
        'leiden', 'louvain', or 'DDRTree'
    k : int
        Number of nearest neighbors.
    resolution_parameter : float
        Resolution for Leiden/Louvain.
    louvain_iter : int
        Number of Louvain iterations.
    verbose : bool

    Returns
    -------
    adata with 'Cluster' in .obs
    """
    monocle = adata.uns.get('monocle', {})

    if method == 'densityPeak':
        # Density peak clustering (Rodriguez-Laio 2014) — used in Monocle2
        # tutorial when data has been embedded via tSNE.
        if 'X_tSNE' in adata.obsm:
            data = adata.obsm['X_tSNE']
        elif 'reducedDimA' in monocle:
            data = monocle['reducedDimA'].T
        else:
            raise ValueError("densityPeak requires a prior tSNE embedding")

        num_clusters = kwargs.get('num_clusters', 5)
        rho_threshold = kwargs.get('rho_threshold', None)
        delta_threshold = kwargs.get('delta_threshold', None)

        from scipy.spatial.distance import cdist
        N = data.shape[0]
        dists = cdist(data, data)
        # dc = distance at 2% percentile
        dc = np.percentile(dists[dists > 0], 2)
        rho = np.exp(-(dists / dc) ** 2).sum(axis=1) - 1  # exclude self

        # delta = min distance to a point with higher rho
        rho_order = np.argsort(rho)[::-1]
        delta = np.zeros(N)
        nneigh = np.zeros(N, dtype=int)
        delta[rho_order[0]] = dists[rho_order[0]].max()
        nneigh[rho_order[0]] = rho_order[0]
        for i in range(1, N):
            cur = rho_order[i]
            higher = rho_order[:i]
            d = dists[cur, higher]
            best = np.argmin(d)
            delta[cur] = d[best]
            nneigh[cur] = higher[best]

        adata.uns.setdefault('monocle', {})['rho'] = rho
        adata.uns['monocle']['delta'] = delta

        # Select cluster centers by rho*delta
        gamma = rho * delta
        center_idx = np.argsort(gamma)[-num_clusters:][::-1]
        cluster_labels = np.full(N, -1, dtype=int)
        for ci, cell in enumerate(center_idx):
            cluster_labels[cell] = ci + 1

        # Propagate: each non-center inherits the cluster of its nneigh
        for cell in rho_order:
            if cluster_labels[cell] == -1:
                cluster_labels[cell] = cluster_labels[nneigh[cell]]

        adata.obs['Cluster'] = pd.Categorical(cluster_labels)
        return adata

    if method == 'DDRTree':
        from .dimension_reduction import reduce_dimension
        num_clusters = kwargs.get('num_clusters', None)
        adata = reduce_dimension(adata, max_components=2, reduction_method='DDRTree',
                                 verbose=verbose, ncenter=num_clusters,
                                 param_gamma=100, **kwargs)
        closest = monocle.get('pr_graph_cell_proj_closest_vertex')
        if closest is not None:
            adata.obs['Cluster'] = pd.Categorical(closest.astype(str))
        return adata

    # Get data for clustering
    if 'X_DDRTree' in adata.obsm:
        data = adata.obsm['X_DDRTree']
    elif 'X_tSNE' in adata.obsm:
        data = adata.obsm['X_tSNE']
    elif 'reducedDimA' in monocle:
        data = monocle['reducedDimA'].T
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, min(adata.shape) - 1))
        data = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

    N = data.shape[0]
    k_use = min(k, N - 1)

    if verbose:
        print(f"Finding {k_use} nearest neighbors for {N} cells...")

    nn = NearestNeighbors(n_neighbors=k_use + 1, algorithm='ball_tree')
    nn.fit(data)
    distances, indices = nn.kneighbors(data)
    neighbor_matrix = indices[:, 1:]  # Exclude self

    if verbose:
        print("Computing Jaccard coefficients...")

    links = _jaccard_coeff(neighbor_matrix, weighted=False)

    if len(links) == 0:
        adata.obs['Cluster'] = pd.Categorical(np.ones(N, dtype=int))
        return adata

    # Build igraph graph
    g = ig.Graph()
    g.add_vertices(N)
    g.vs['name'] = list(adata.obs_names)
    edges = [(int(links[i, 0]), int(links[i, 1])) for i in range(len(links))]
    weights = links[:, 2].tolist()
    g.add_edges(edges)
    g.es['weight'] = weights

    if method == 'leiden':
        if verbose:
            print("Running Leiden clustering...")
        try:
            import leidenalg
            partition = leidenalg.find_partition(
                g, leidenalg.CPMVertexPartition,
                resolution_parameter=resolution_parameter,
                n_iterations=louvain_iter,
            )
            membership = partition.membership
        except ImportError:
            # Fallback to igraph's community detection
            partition = g.community_leiden(
                objective_function='CPM',
                resolution=resolution_parameter,
                n_iterations=louvain_iter,
            )
            membership = partition.membership

    elif method == 'louvain':
        if verbose:
            print("Running Louvain clustering...")
        best_partition = None
        best_modularity = -1

        for _iter in range(louvain_iter):
            partition = g.community_multilevel(weights='weight')
            mod = partition.modularity
            if mod > best_modularity:
                best_modularity = mod
                best_partition = partition

        membership = best_partition.membership

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    adata.obs['Cluster'] = pd.Categorical(np.array(membership) + 1)

    if verbose:
        n_clusters = len(set(membership))
        print(f"Found {n_clusters} clusters")

    return adata


def cluster_genes(expression_matrix, k, method='correlation'):
    """
    Cluster genes by expression pattern.

    Parameters
    ----------
    expression_matrix : np.ndarray or pd.DataFrame
        Genes x cells expression matrix.
    k : int
        Number of clusters.
    method : str
        Distance method: 'correlation' (default), 'euclidean'.

    Returns
    -------
    dict with:
        'clustering': np.ndarray of cluster assignments
        'exprs': np.ndarray expression matrix
        'medoids': indices of medoid genes
    """
    if isinstance(expression_matrix, pd.DataFrame):
        gene_names = expression_matrix.index
        expr = expression_matrix.values
    else:
        expr = np.array(expression_matrix)
        gene_names = None

    # Remove rows with NaN
    valid_mask = ~np.any(np.isnan(expr), axis=1)
    expr = expr[valid_mask]

    if method == 'correlation':
        # Correlation distance
        corr = np.corrcoef(expr)
        corr[np.isnan(corr)] = 0
        dist_matrix = (1 - corr) / 2
    else:
        dist_matrix = squareform(pdist(expr, metric='euclidean'))

    # PAM-like clustering
    try:
        from sklearn_extra.cluster import KMedoids
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
        labels = kmedoids.fit_predict(dist_matrix)
        medoids = kmedoids.medoid_indices_
    except (ImportError, Exception):
        # Fallback: use hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        condensed = squareform(dist_matrix, checks=False)
        condensed[condensed < 0] = 0
        Z = linkage(condensed, method='ward')
        labels = fcluster(Z, k, criterion='maxclust')
        medoids = None

    result = {
        'clustering': labels,
        'exprs': expr,
        'medoids': medoids,
    }

    if gene_names is not None:
        result['gene_names'] = gene_names[valid_mask]

    return result
