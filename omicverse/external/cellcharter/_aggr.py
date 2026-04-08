from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as sps
from anndata import AnnData
from scipy.sparse import spdiags

from ._utils import sample_obs_key, spatial_connectivity_key, str2list


def _aggregate_mean(adj, x):
    return adj @ x


def _aggregate_var(adj, x):
    mean = adj @ x
    mean_squared = adj @ (x * x)
    return mean_squared - mean * mean


def _aggregate(adj, x, method):
    if method == "mean":
        return _aggregate_mean(adj, x)
    if method == "var":
        return _aggregate_var(adj, x)
    raise NotImplementedError(f"Unsupported aggregation `{method}`.")


def _mul_broadcast(mat1, mat2):
    return spdiags(mat2, 0, len(mat2), len(mat2)) * mat1


def _hop(adj_hop, adj, adj_visited=None):
    adj_hop = adj_hop @ adj

    if adj_visited is not None:
        adj_hop = adj_hop > adj_visited
        adj_visited = adj_visited + adj_hop

    return adj_hop, adj_visited


def _normalize(adj):
    deg = np.array(np.sum(adj, axis=1)).squeeze()

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        deg_inv = 1 / deg
    deg_inv[deg_inv == float("inf")] = 0

    return _mul_broadcast(adj, deg_inv)


def _setdiag(array, value):
    if isinstance(array, sps.csr_matrix):
        array = array.tolil()
    array.setdiag(value)
    array = array.tocsr()
    if value == 0:
        array.eliminate_zeros()
    return array


def _aggregate_neighbors(
    adj: sps.spmatrix,
    X: np.ndarray,
    nhood_layers: list,
    aggregations: Optional[Union[str, list]] = "mean",
) -> np.ndarray:
    adj = adj.astype(bool)
    adj = _setdiag(adj, 0)
    adj_hop = adj.copy()
    adj_visited = _setdiag(adj.copy(), 1)

    Xs = []
    for i in range(0, max(nhood_layers) + 1):
        if i not in nhood_layers:
            continue
        if i == 0:
            Xs.append(X)
            continue
        if i > 1:
            adj_hop, adj_visited = _hop(adj_hop, adj, adj_visited)
        adj_hop_norm = _normalize(adj_hop)

        for agg in aggregations:
            Xs.append(_aggregate(adj_hop_norm, X, agg))

    if sps.issparse(X):
        return sps.hstack(Xs)
    return np.hstack(Xs)


def aggregate_neighbors(
    adata: AnnData,
    n_layers: Union[int, list],
    aggregations: Optional[Union[str, list]] = "mean",
    connectivity_key: Optional[str] = None,
    use_rep: Optional[str] = None,
    sample_key: Optional[str] = None,
    out_key: Optional[str] = "X_cellcharter",
    copy: bool = False,
) -> np.ndarray | None:
    """Aggregate multi-hop neighborhood features for CellCharter clustering."""
    connectivity_key = spatial_connectivity_key(connectivity_key)
    sample_key = sample_obs_key(sample_key)
    aggregations = str2list(aggregations)

    X = adata.X if use_rep is None else adata.obsm[use_rep]

    if isinstance(n_layers, int):
        n_layers = list(range(n_layers + 1))

    if sps.issparse(X):
        X_aggregated = sps.dok_matrix(
            (X.shape[0], X.shape[1] * ((len(n_layers) - 1) * len(aggregations) + 1)),
            dtype=np.float32,
        )
    else:
        X_aggregated = np.empty(
            (X.shape[0], X.shape[1] * ((len(n_layers) - 1) * len(aggregations) + 1)),
            dtype=np.float32,
        )

    if sample_key in adata.obs:
        sample_indices = [
            np.flatnonzero(adata.obs[sample_key].to_numpy() == sample)
            for sample in adata.obs[sample_key].unique()
        ]
    else:
        sample_indices = [np.arange(adata.shape[0])]

    for rows in sample_indices:
        X_sample_aggregated = _aggregate_neighbors(
            adj=adata.obsp[connectivity_key][rows][:, rows],
            X=X[rows],
            nhood_layers=n_layers,
            aggregations=aggregations,
        )
        X_aggregated[rows] = X_sample_aggregated

    if isinstance(X_aggregated, sps.dok_matrix):
        X_aggregated = X_aggregated.tocsr()

    if copy:
        return X_aggregated

    adata.obsm[out_key] = X_aggregated
    return None
