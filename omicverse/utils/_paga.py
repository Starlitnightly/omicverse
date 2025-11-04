import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, issparse

from scanpy.tools._paga import PAGA

# TODO: Add docstrings
class PAGA_tree(PAGA):
    """TODO."""

    def __init__(
        self,
        adata,
        groups=None,
        vkey=None,
        use_time_prior=None,
        root_key=None,
        end_key=None,
        threshold_root_end_prior=None,
        minimum_spanning_tree=None,
    ):
        super().__init__(adata=adata, groups=groups, model="v1.2")
        self.groups = groups
        self.vkey = vkey
        self.use_time_prior = use_time_prior
        self.root_key = root_key
        self.end_key = end_key
        self.threshold_root_end_prior = threshold_root_end_prior
        if self.threshold_root_end_prior is None:
            self.threshold_root_end_prior = 0.9
        self.minimum_spanning_tree = minimum_spanning_tree

    # TODO: Add docstrings
    def compute_transitions(self):
        """TODO."""
        try:
            import igraph
        except ImportError:
            raise ImportError("To run paga, you need to install `pip install igraph`")
        vkey = f"{self.vkey}_graph"
        if vkey not in self._adata.uns:
            raise ValueError(
                "The passed AnnData needs to have an `uns` annotation "
                "with key 'velocity_graph' - a sparse matrix from RNA velocity."
            )
        if self._adata.uns[vkey].shape != (self._adata.n_obs, self._adata.n_obs):
            raise ValueError(
                f"The passed 'velocity_graph' has shape {self._adata.uns[vkey].shape} "
                f"but shoud have shape {(self._adata.n_obs, self._adata.n_obs)}"
            )

        clusters = self._adata.obs[self.groups]
        cats = clusters.cat.categories
        vgraph = self._adata.uns[vkey] > 0.1
        time_prior = self.use_time_prior

        if isinstance(time_prior, str) and time_prior in self._adata.obs.keys():
            vpt = self._adata.obs[time_prior].values
            vpt_mean = self._adata.obs.groupby(self.groups)[time_prior].mean()
            vpt_means = np.array([vpt_mean[cat] for cat in clusters])
            rows, cols, vals = [], [], []
            for i in range(vgraph.shape[0]):
                indices = vgraph[i].indices
                idx_bool = vpt[i] < vpt[indices]
                idx_bool &= vpt_means[indices] > vpt_means[i] - 0.1
                cols.extend(indices[idx_bool])
                vals.extend(vgraph[i].data[idx_bool])
                rows.extend([i] * np.sum(idx_bool))
            vgraph = vals_to_csr(vals, rows, cols, shape=vgraph.shape)

        lb = self.threshold_root_end_prior  # cells to be consider as terminal states
        if isinstance(self.end_key, str) and self.end_key in self._adata.obs.keys():
            set_row_csr(vgraph, rows=np.where(self._adata.obs[self.end_key] > lb)[0])
        if isinstance(self.root_key, str) and self.root_key in self._adata.obs.keys():
            vgraph[:, np.where(self._adata.obs[self.root_key] > lb)[0]] = 0
            vgraph.eliminate_zeros()

        membership = self._adata.obs[self.groups].cat.codes.values
        g = get_igraph_from_adjacency(vgraph, directed=True)
        vc = igraph.VertexClustering(g, membership=membership)
        cg_full = vc.cluster_graph(combine_edges="sum")
        transitions = get_sparse_from_igraph(cg_full, weight_attr="weight")
        transitions = transitions - transitions.T
        transitions_conf = transitions.copy()
        transitions = transitions.tocoo()
        total_n = self._neighbors.n_neighbors * np.array(vc.sizes())
        for i, j, v in zip(transitions.row, transitions.col, transitions.data):
            reference = np.sqrt(total_n[i] * total_n[j])
            transitions_conf[i, j] = 0 if v < 0 else v / reference
        transitions_conf.eliminate_zeros()

        # remove non-confident direct paths if more confident indirect path is found.
        T = transitions_conf.toarray()
        threshold = max(np.nanmin(np.nanmax(T / (T > 0), axis=0)) - 1e-6, 0.01)
        T *= T > threshold
        for i in range(len(T)):
            idx = T[i] > 0
            if np.any(idx):
                indirect = np.clip(T[idx], None, T[i][idx][:, None]).max(0)
                T[i, T[i] < indirect] = 0

        if self.minimum_spanning_tree:
            T_tmp = T.copy()
            T_num = T > 0
            T_sum = np.sum(T_num, 0)
            T_max = np.max(T_tmp)
            for i in range(len(T_tmp)):
                if T_sum[i] == 1:
                    T_tmp[np.where(T_num[:, i])[0][0], i] = T_max
            from scipy.sparse.csgraph import minimum_spanning_tree

            T_tmp = np.abs(minimum_spanning_tree(-T_tmp).toarray()) > 0
            T = T_tmp * T

        transitions_conf = csr_matrix(T)
        self.transitions_confidence = transitions_conf.T

        # set threshold for minimal spanning tree.
        df = pd.DataFrame(T, index=cats, columns=cats)
        self.threshold = np.nanmin(np.nanmax(df.values / (df.values > 0), axis=0))
        self.threshold = max(self.threshold - 1e-6, 0.01)


# TODO: Add docstrings
def vals_to_csr(vals, rows, cols, shape, split_negative=False):
    """TODO."""
    graph = coo_matrix((vals, (rows, cols)), shape=shape)

    if split_negative:
        graph_neg = graph.copy()

        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()

        return graph.tocsr(), graph_neg.tocsr()

    else:
        return graph.tocsr()
    
# TODO: Finish docstrings
def set_row_csr(csr, rows, value=0):
    """Set all nonzero elements to the given value. Useful to set to 0 mostly."""
    for row in rows:
        start = csr.indptr[row]
        end = csr.indptr[row + 1]
        csr.data[start:end] = value
    if value == 0:
        csr.eliminate_zeros()


# TODO: Finish docstrings
def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shap[0] vertices
    g.add_edges(list(zip(sources, targets)))
    g.es["weight"] = weights
    if g.vcount() != adjacency.shape[0]:
        print(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g

# TODO: Add docstrings
def get_sparse_from_igraph(graph, weight_attr=None):
    """TODO."""
    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        # Convert to list to ensure proper extend behavior
        try:
            weights = list(graph.es[weight_attr])
        except (KeyError, AttributeError):
            # If weight attribute doesn't exist or is malformed, use default weights
            weights = [1] * len(edges)

    if not graph.is_directed():
        # Create a copy of the current weights before extending
        current_weights = weights.copy() if hasattr(weights, 'copy') else list(weights)
        edges.extend([(v, u) for u, v in edges])
        weights.extend(current_weights)

    shape = graph.vcount()
    shape = (shape, shape)

    # Check that we have valid edges and matching weights
    if len(edges) > 0 and len(weights) == len(edges):
        # Unpack edges into separate row and column index lists
        try:
            row_indices, col_indices = zip(*edges)
            # Convert to lists to ensure compatibility with scipy
            row_indices = list(row_indices)
            col_indices = list(col_indices)
            return csr_matrix((weights, (row_indices, col_indices)), shape=shape)
        except (ValueError, TypeError) as e:
            # If unpacking fails, return empty sparse matrix
            print(f"Warning: Failed to unpack edges: {e}. Returning empty sparse matrix.")
            return csr_matrix(shape)
    else:
        # Return empty sparse matrix with the correct shape
        return csr_matrix(shape)


def cal_paga(
    adata,
    groups=None,
    vkey="velocity",
    use_time_prior=True,
    root_key=None,
    end_key=None,
    threshold_root_end_prior=None,
    minimum_spanning_tree=True,
    copy=False,
):
    """PAGA graph with velocity-directed edges.

    Mapping out the coarse-grained connectivity structures of complex manifolds
    :cite:p:`Wolf19`. By quantifying the connectivity of partitions (groups, clusters) of the
    single-cell graph, partition-based graph abstraction (PAGA) generates a much
    simpler abstracted graph (*PAGA graph*) of partitions, in which edge weights
    represent confidence in the presence of connections.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        An annotated data matrix.
    groups : key for categorical in `adata.obs`, optional (default: 'louvain')
        You can pass your predefined groups by choosing any categorical
        annotation of observations (`adata.obs`).
    vkey: `str` or `None` (default: `None`)
        Key for annotations of observations/cells or variables/genes.
    use_time_prior : `str` or bool, optional (default: True)
        Obs key for pseudo-time values.
        If True, 'velocity_pseudotime' is used if available.
    root_key : `str` or bool, optional (default: None)
        Obs key for root states.
    end_key : `str` or bool, optional (default: None)
        Obs key for end states.
    threshold_root_end_prior : `float` (default: 0.9)
        Threshold for root and final states priors, to be in the range of [0,1].
        Values above the threshold will be considered as terminal and included as prior.
    minimum_spanning_tree : bool, optional (default: True)
        Whether to prune the tree such that a path from A-to-B
        is removed if another more confident path exists.
    copy : `bool`, optional (default: `False`)
        Copy `adata` before computation and return a copy.
        Otherwise, perform computation inplace and return `None`.

    Returns
    -------
    connectivities: `.uns`
        The full adjacency matrix of the abstracted graph, weights correspond to
        confidence in the connectivities of partitions.
    connectivities_tree: `.uns`
        The adjacency matrix of the tree-like subgraph that best explains the topology.
    transitions_confidence: `.uns`
        The adjacency matrix of the abstracted directed graph, weights correspond to
        confidence in the transitions between partitions.
    """
    if "neighbors" not in adata.uns:
        raise ValueError(
            "You need to run `pp.neighbors` first to compute a neighborhood graph."
        )

    adata = adata.copy() if copy else adata
    adata.uns['paga_graph']=adata.obsp['connectivities']
    strings_to_categoricals(adata)

    if groups is None:
        groups = (
            "clusters"
            if "clusters" in adata.obs.keys()
            else "louvain"
            if "louvain" in adata.obs.keys()
            else None
        )
    

    priors = [p for p in [use_time_prior, root_key, end_key] if p in adata.obs.keys()]
    print(
        "running PAGA",
        f"using priors: {priors}" if len(priors) > 0 else "",
    )
    paga = PAGA_tree(
        adata,
        groups,
        vkey=vkey,
        use_time_prior=use_time_prior,
        root_key=root_key,
        end_key=end_key,
        threshold_root_end_prior=threshold_root_end_prior,
        minimum_spanning_tree=minimum_spanning_tree,
    )

    if "paga" not in adata.uns:
        adata.uns["paga"] = {}

    paga.compute_connectivities()
    adata.uns["paga"]["connectivities"] = paga.connectivities
    adata.uns["paga"]["connectivities_tree"] = paga.connectivities_tree
    adata.uns[f"{groups}_sizes"] = np.array(paga.ns)

    paga.compute_transitions()
    adata.uns["paga"]["transitions_confidence"] = paga.transitions_confidence
    adata.uns["paga"]["threshold"] = paga.threshold
    adata.uns["paga"]["groups"] = groups

    print("    finished")
    print(
        "added\n" + "    'paga/connectivities', connectivities adjacency (adata.uns)\n"
        "    'paga/connectivities_tree', connectivities subtree (adata.uns)\n"
        "    'paga/transitions_confidence', velocity transitions (adata.uns)"
    )

    return adata if copy else None

def strings_to_categoricals(adata):
    """Transform string annotations to categoricals."""
    from pandas import Categorical
    from pandas.api.types import is_bool_dtype, is_integer_dtype, is_string_dtype

    def is_valid_dtype(values):
        return (
            is_string_dtype(values) or is_integer_dtype(values) or is_bool_dtype(values)
        )

    df = adata.obs
    df_keys = [key for key in df.columns if is_valid_dtype(df[key])]
    for key in df_keys:
        c = df[key]
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c

    df = adata.var
    df_keys = [key for key in df.columns if is_string_dtype(df[key])]
    for key in df_keys:
        c = df[key].astype("U")
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c

from pandas import Index

# TODO: Add docstrings
def make_unique_list(key, allow_array=False):
    """TODO."""
    from collections import abc
    if isinstance(key, (Index, abc.KeysView)):
        key = list(key)
    is_list = (
        isinstance(key, (list, tuple, np.record))
        if allow_array
        else isinstance(key, (list, tuple, np.ndarray, np.record))
    )
    is_list_of_str = is_list and all(isinstance(item, str) for item in key)
    return key if is_list_of_str else key if is_list and len(key) < 20 else [key]

# TODO: Add docstrings
def check_basis(adata, basis):
    """TODO."""
    if basis in adata.obsm.keys() and f"X_{basis}" not in adata.obsm.keys():
        adata.obsm[f"X_{basis}"] = adata.obsm[basis]
        print(f"Renamed '{basis}' to convention 'X_{basis}' (adata.obsm).")

def make_unique_valid_list(adata, keys):
    """TODO."""
    keys = make_unique_list(keys)
    if all(isinstance(item, str) for item in keys):
        for i, key in enumerate(keys):
            if key.startswith("X_"):
                keys[i] = key = key[2:]
            check_basis(adata, key)
        valid_keys = np.hstack(
            [
                adata.obs.keys(),
                adata.var.keys(),
                adata.varm.keys(),
                adata.obsm.keys(),
                [key[2:] for key in adata.obsm.keys()],
                list(adata.layers.keys()),
            ]
        )
        keys_ = keys
        keys = [key for key in keys if key in valid_keys or key in adata.var_names]
        keys_ = [key for key in keys_ if key not in keys]
        if len(keys_) > 0:
            print(", ".join(keys_), "not found.")
    return keys

def default_basis(adata, **kwargs):
    """TODO."""
    if "x" in kwargs and "y" in kwargs:
        keys, x, y = ["embedding"], kwargs.pop("x"), kwargs.pop("y")
        adata.obsm["X_embedding"] = np.stack([x, y]).T
        if "velocity_embedding" in adata.obsm.keys():
            del adata.obsm["velocity_embedding"]
    else:
        keys = [
            key for key in ["pca", "tsne", "umap"] if f"X_{key}" in adata.obsm.keys()
        ]
    if not keys:
        raise ValueError("No basis specified.")
    return keys[-1] if len(keys) > 0 else None

def _safe_categorical_to_str(adata, color_key):
    """
    Safely convert categorical values to string format for plotting.
    
    This function addresses pandas compatibility issues where direct assignment
    to cat.categories is no longer allowed in newer pandas versions.
    """
    if color_key in adata.obs.columns and hasattr(adata.obs[color_key], 'cat'):
        # Create a copy to avoid modifying the original data
        adata_copy = adata.copy()
        # Convert categorical values to string using cat.rename_categories
        if hasattr(adata_copy.obs[color_key].cat, 'categories'):
            str_categories = adata_copy.obs[color_key].cat.categories.astype(str)
            adata_copy.obs[color_key] = adata_copy.obs[color_key].cat.rename_categories(str_categories)
        return adata_copy
    return adata

def plot_paga(adata,
    basis=None,
    vkey="velocity",
    color=None,
    layer=None,
    title=None,
    threshold=None,
    layout=None,
    layout_kwds=None,
    init_pos=None,
    root=0,
    labels=None,
    single_component=False,
    dashed_edges="connectivities",
    solid_edges="transitions_confidence",
    transitions="transitions_confidence",
    node_size_scale=1,
    node_size_power=0.5,
    edge_width_scale=0.4,
    min_edge_width=None,
    max_edge_width=2,
    arrowsize=15,
    random_state=0,
    pos=None,
    node_colors=None,
    normalize_to_color=False,
    cmap=None,
    cax=None,
    cb_kwds=None,
    add_pos=True,
    export_to_gexf=False,
    plot=True,
    use_raw=None,
    size=None,
    groups=None,
    components=None,
    figsize=None,
    dpi=None,
    show=None,
    save=None,
    ax=None,
    ncols=None,
    scatter_flag=None,
    **kwargs,):

    if layout is not None:
        basis = None
    
    # Fix for pandas compatibility: when using basis with categorical colors,
    # scvelo may fail due to read-only categories in newer pandas versions
    adata_processed = adata
    if basis is not None and color is not None:
        adata_processed = _safe_categorical_to_str(adata, color)
    
    import scvelo as scv
    return scv.pl.paga(adata_processed,
    basis=basis,
    vkey=vkey,
    color=color,
    layer=layer,
    title=title,
    threshold=threshold,
    layout=layout,
    layout_kwds=layout_kwds,
    init_pos=init_pos,
    root=root,
    labels=labels,
    single_component=single_component,
    dashed_edges=dashed_edges,
    solid_edges=solid_edges,
    transitions=transitions,
    node_size_scale=node_size_scale,
    node_size_power=node_size_power,
    edge_width_scale=edge_width_scale,
    min_edge_width=min_edge_width,
    max_edge_width=max_edge_width,
    arrowsize=arrowsize,
    random_state=random_state,
    pos=pos,
    node_colors=node_colors,
    normalize_to_color=normalize_to_color,
    cmap=cmap,
    cax=cax,
    cb_kwds=cb_kwds,
    add_pos=add_pos,
    export_to_gexf=export_to_gexf,
    plot=plot,
    use_raw=use_raw,
    size=size,
    groups=groups,
    components=components,
    figsize=figsize,
    dpi=dpi,
    show=show,
    save=save,
    ax=ax,
    ncols=ncols,
    scatter_flag=scatter_flag,
    **kwargs,)

