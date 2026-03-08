import numpy as np


__all__ = [
    'evaluate_unsupervised',
    'clustering_coef',
    'coverage',
    'density',
    'conductance',
]


def evaluate_unsupervised(Z_pred, adj):
    return {'coverage': coverage(Z_pred, adj),
            'density': density(Z_pred, adj),
            'conductance': conductance(Z_pred, adj),
            'clustering_coef': clustering_coef(Z_pred, adj)}


def clustering_coef(Z_pred, adj):
    """Compute weighted average of clustering coefficients of communities."""
    def clustering_coef_community(ind, adj):
        """Compute clustering coefficient of a single community."""
        adj_com = adj[ind][:, ind]
        n = ind.sum()
        if n < 3:
            return 0
        # Number of possible triangles
        possible = (n - 2) * (n - 1) * n / 6
        # Number of existing triangles
        existing = (adj_com @ adj_com @ adj_com).diagonal().sum() / 6
        return existing / possible

    Z_pred = Z_pred.astype(bool)
    com_sizes = Z_pred.sum(0)
    clust_coefs = np.array([clustering_coef_community(Z_pred[:, c], adj) for c in range(Z_pred.shape[1])])
    return clust_coefs @ com_sizes / com_sizes.sum()


def coverage(Z_pred, adj):
    """What fraction of edges are explained by at least 1 community?

    Args:
        Z_pred: Binary community affiliation matrix
        adj : Unweighted symmetric adjacency matrix of a graph.
    """
    u, v = adj.nonzero()
    return ((Z_pred[u] * Z_pred[v]).sum(1) > 0).sum() / adj.nnz


def density(Z_pred, adj):
    """Average density of communities (weighted by size).

    Higher is better.

        (\sum_i density(C_i) * |C_i|) / (\sum_j |C_j|)

    Args:
        Z_pred: Binary community affiliation matrix
        adj : Unweighted symmetric adjacency matrix of a graph.
    """
    def density_community(ind, adj):
        ind = ind.astype(bool)
        n = ind.sum()
        if n  < 2:
            return 0.0
        else:
            return adj[ind][:, ind].nnz / (n**2 - n)
    Z_pred = Z_pred.astype(bool)
    com_sizes = Z_pred.sum(0) / Z_pred.sum()
    densities = np.array([density_community(Z_pred[:, c], adj) for c in range(Z_pred.shape[1])])
    return densities @ com_sizes


def conductance(Z_pred, adj):
    """Compute weight average of conductances of communities.

    Conductance of each community is weighted by its size.

        (\sum_i conductance(C_i) * |C_i|) / (\sum_j |C_j|)

    Args:
        Z_pred: Binary community affiliation matrix
        adj : Unweighted symmetric adjacency matrix of a graph.
    """
    def conductance_community(ind, adj):
        """Compute conductance of a single community.

        Args:
            ind: Binary indicator vector for the community.
            adj: Adjacency matrix in scipy.sparse format.
        """
        ind = ind.astype(bool)
        inside = adj[ind, :][:, ind].nnz
        outside = adj[~ind, :][:, ind].nnz
        if inside + outside == 0:
            return 1
        return outside / (inside + outside)

    Z_pred = Z_pred.astype(bool)
    com_sizes = Z_pred.sum(0)
    conductances = np.array([conductance_community(Z_pred[:, c], adj) for c in range(Z_pred.shape[1])])
    return conductances @ com_sizes / com_sizes.sum()
