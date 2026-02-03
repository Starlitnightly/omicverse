import numpy as np


def fit_gene_model(gene_counts, umi_counts):

    N = gene_counts.size

    mu = np.zeros(N)
    var = np.ones(N)
    x2 = np.ones(N)

    return mu, var, x2
