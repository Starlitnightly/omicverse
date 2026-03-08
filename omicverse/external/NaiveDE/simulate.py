import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy import stats


def softplus(data):
    return np.log(1. + np.exp(data))


def simulate_cell_types(num_clusters, num_cells, num_markers_per_cluster, num_bg_genes, marker_expr=1.0, marker_covar=0.8):
    total_genes = num_clusters * num_markers_per_cluster + num_bg_genes
    cells_per_cluster = num_cells // num_clusters

    data = np.zeros((num_cells, total_genes))

    for cluster in range(num_clusters):
        mu = np.zeros(total_genes)
        Sigma = np.zeros((total_genes, total_genes))

        g_slice = np.s_[cluster * num_markers_per_cluster:(cluster + 1) * num_markers_per_cluster]
        mu[g_slice] = marker_expr
        Sigma[g_slice, g_slice] = marker_covar
        np.fill_diagonal(Sigma, 1.)

        c_slice = np.s_[cluster * cells_per_cluster:(cluster + 1) * cells_per_cluster]
        data[c_slice, :] = stats.multivariate_normal(mu, Sigma).rvs(cells_per_cluster)

    expression = pd.DataFrame.from_records(softplus(data))
    expression = expression.T.add_prefix('cell_') \
                           .T.add_prefix('gene_')
    return expression


def simulate_bilevel_cell_types(num_l1_clusters, num_l2_clusters, num_cells, num_markers_per_cluster,
                                num_bg_genes, marker_expr=1.0, marker_covar=0.8):
    ''' Simulate cells from a hierarchy of cell types
    '''
    total_genes = num_l1_clusters * (1 + num_l2_clusters) * num_markers_per_cluster + num_bg_genes
    cells_per_cluster = num_cells // (num_l1_clusters * num_l2_clusters)

    data = np.zeros((num_cells, total_genes))

    g2_offset = num_l1_clusters * num_markers_per_cluster
    clust_num = 0
    l2c_num = 0
    for l1c in range(num_l1_clusters):
        g1_slice = np.s_[l1c * num_markers_per_cluster:(l1c + 1) * num_markers_per_cluster]
        for l2c in range(num_l2_clusters):
            g2_slice = np.s_[g2_offset + (l2c_num) * num_markers_per_cluster:g2_offset + (l2c_num + 1) * num_markers_per_cluster]

            mu = np.zeros(total_genes)
            Sigma = np.zeros((total_genes, total_genes))

            mu[g1_slice] = marker_expr
            mu[g2_slice] = marker_expr
            Sigma[g1_slice, g1_slice] = marker_covar
            Sigma[g2_slice, g2_slice] = marker_covar
            np.fill_diagonal(Sigma, 1.)

            c_slice = np.s_[clust_num * cells_per_cluster:(clust_num + 1) * cells_per_cluster]
            data[c_slice, :] = stats.multivariate_normal(mu, Sigma).rvs(cells_per_cluster)
            clust_num += 1
            l2c_num += 1

    expression = pd.DataFrame.from_records(softplus(data))
    expression = expression.T.add_prefix('cell_') \
                           .T.add_prefix('gene_')
    return expression


def plot_data(data):
    plt.pcolormesh(data.T, cmap=cm.gray_r)
    plt.ylabel('Genes')
    plt.xlabel('Cells')
    plt.colorbar(label='expression')

