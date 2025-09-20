import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def compute_cluster_averages(adata, labels, use_raw=True, layer=None):
    """
    Compute average expression of each gene in each cluster

    Parameters
    ----------
    adata
        AnnData object of reference single-cell dataset
    labels
        Name of adata.obs column containing cluster labels
    use_raw
        Use raw slow in adata.
    layer
        Use layer in adata, provide layer name.

    Returns
    -------
    pd.DataFrame of cluster average expression of each gene

    """

    if layer is not None:
        x = adata.layers[layer]
        var_names = adata.var_names
    else:
        if not use_raw:
            x = adata.X
            var_names = adata.var_names
        else:
            if not adata.raw:
                raise ValueError("AnnData object has no raw data, change `use_raw=True, layer=None` or fix your object")
            x = adata.raw.X
            var_names = adata.raw.var_names

    if sum(adata.obs.columns == labels) != 1:
        raise ValueError("`labels` is absent in adata_ref.obs or not unique")

    all_clusters = np.unique(adata.obs[labels])
    averages_mat = np.zeros((1, x.shape[1]))

    for c in all_clusters:
        sparse_subset = csr_matrix(x[np.isin(adata.obs[labels], c), :])
        aver = sparse_subset.mean(0)
        averages_mat = np.concatenate((averages_mat, aver))
    averages_mat = averages_mat[1:, :].T
    averages_df = pd.DataFrame(data=averages_mat, index=var_names, columns=all_clusters)

    return averages_df


def get_cluster_variances(adata, labels, use_raw=True, layer=None):
    """
    Compute variance of each gene in each cluster

    Parameters
    ----------

    labels
        Name of adata.obs column containing cluster labels
    use_raw
        Use raw slow in adata.
    layer
        Use layer in adata, provide layer name.

    Returns
    -------
    pd.DataFrame of within cluster variance of each gene
    """
    if layer is not None:
        x = adata.layers[layer]
        var_names = adata.var_names
    else:
        if not use_raw:
            x = adata.X
            var_names = adata.var_names
        else:
            if not adata.raw:
                raise ValueError("AnnData object has no raw data, change `use_raw=True, layer=None` or fix your object")
            x = adata.raw.X
            var_names = adata.raw.var_names

    if sum(adata.obs.columns == labels) != 1:
        raise ValueError("`labels` is absent in adata_ref.obs or not unique")

    all_clusters = np.unique(adata.obs[labels])
    var_mat = np.zeros((1, x.shape[1]))

    for c in all_clusters:
        sparse_subset = csr_matrix(x[np.isin(adata.obs[labels], c), :])
        c = sparse_subset.copy()
        c.data **= 2
        var = c.mean(0) - (np.array(sparse_subset.mean(0)) ** 2)
        del c
        var_mat = np.concatenate((var_mat, var))
    var_mat = var_mat[1:, :].T
    var_df = pd.DataFrame(data=var_mat, index=var_names, columns=all_clusters)

    return var_df


def get_cluster_averages_df(X, cluster_col):
    """
    :param X: DataFrame with spots / cells in rows and expression dimensions in columns
    :param cluster_col: pd.Series object containing cluster labels
    :returns: pd.DataFrame of cluster average expression of each gene
    """

    all_clusters = np.unique(cluster_col)
    averages_mat = np.zeros((1, X.shape[1]))

    for c in all_clusters:
        aver = X.loc[np.isin(cluster_col, c), :].values.mean(0)
        averages_mat = np.concatenate((averages_mat, aver.reshape((1, X.shape[1]))))
    averages_mat = averages_mat[1:, :].T
    averages_df = pd.DataFrame(data=averages_mat, index=X.columns, columns=all_clusters)

    return averages_df


def get_cluster_variances_df(X, cluster_col):
    """
    :param X: DataFrame with spots / cells in rows and expression dimensions in columns
    :param cluster_col: pd.Series object containing cluster labels
    :returns: pd.DataFrame of within cluster variances of each gene
    """

    all_clusters = np.unique(cluster_col)
    averages_mat = np.zeros((1, X.shape[1]))

    for c in all_clusters:
        aver = X.loc[np.isin(cluster_col, c), :].values.var(0)
        averages_mat = np.concatenate((averages_mat, aver.reshape((1, X.shape[1]))))
    averages_mat = averages_mat[1:, :].T
    averages_df = pd.DataFrame(data=averages_mat, index=X.columns, columns=all_clusters)

    return averages_df
