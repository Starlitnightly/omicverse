import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_expected_vs_obs(
    mu, data, gene1="ENSG00000081237", gene1_lab="PTPRC", gene2="ENSG00000167286", gene2_lab="CD3D"
):
    r"""Plot expected vs observed values of a pair of genes (2D histogram).

    :param mu: ndarray of values expected by the model
    :param data: anndata object containing observed data
    :param gene1: gene 1 in anndata.varnames
    :param gene1_lab: gene names to show on the plot
    :param gene2: gene 2 in anndata.varnames
    :param gene2_lab: gene names to show on the plot
    """

    # remove the cell with maximal value
    # max1 = mu[:, np.where(data.var_names == gene1)].max()

    # extract from anndata and convert to numpy if needed
    x = data[:, gene1].X
    y = data[:, gene2].X
    x_mu = mu[:, np.where(data.var_names == gene1)][mu[:, np.where(data.var_names == gene1)] > 0].flatten()
    y_mu = mu[:, np.where(data.var_names == gene2)][mu[:, np.where(data.var_names == gene2)] > 0].flatten()

    from anndata._core.views import SparseCSRView
    from scipy.sparse.csr import csr_matrix

    if type(x) is csr_matrix or type(x) is SparseCSRView:
        x = x.toarray().flatten()
        y = y.toarray().flatten()

    # find number of bins per
    bins = (np.int64(np.ceil(x_mu.max())), np.int64(np.ceil(y_mu.max())))

    bins_adata = (np.int64(np.ceil(x.max())), np.int64(np.ceil(y.max())))

    # create subplot panels
    plt.subplot(2, 2, 1)
    plt.hist2d(x, y, bins=bins_adata, range=[[0, bins_adata[0]], [0, bins_adata[1]]], norm=matplotlib.colors.LogNorm())
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0, bins_adata[0])
    plt.ylim(0, bins_adata[1])
    plt.xlabel("Observed " + gene1_lab)
    plt.ylabel("Observed " + gene2_lab)

    plt.subplot(2, 2, 2)
    plt.hist2d(
        x_mu, y_mu, bins=bins_adata, range=[[0, bins_adata[0]], [0, bins_adata[1]]], norm=matplotlib.colors.LogNorm()
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0, bins_adata[0])
    plt.ylim(0, bins_adata[1])
    plt.xlabel("Imputed " + gene1_lab)
    plt.ylabel("Imputed " + gene2_lab)

    plt.subplot(2, 2, 3)
    plt.hist2d(
        x,
        x_mu,
        bins=(bins_adata[0], bins[0]),
        range=[[0, bins_adata[0]], [0, bins[0]]],
        norm=matplotlib.colors.LogNorm(),
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0, bins_adata[0])
    plt.ylim(0, bins_adata[0])
    plt.xlabel("Observed " + gene1_lab)
    plt.ylabel("Imputed " + gene1_lab)

    plt.subplot(2, 2, 4)
    plt.hist2d(
        y,
        y_mu,
        bins=(bins_adata[1], bins[1]),
        range=[[0, bins_adata[1]], [0, bins[1]]],
        norm=matplotlib.colors.LogNorm(),
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0, bins_adata[1])
    plt.ylim(0, bins_adata[1])
    plt.xlabel("Observed " + gene2_lab)
    plt.ylabel("Imputed " + gene2_lab)

    plt.tight_layout()
