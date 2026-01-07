"""
Simplest Model - just assumes expression data is normal
UMI counts are regressed out
"""
import numpy as np


def fit_gene_model(gene_counts, umi_counts):

    X = np.vstack((np.ones(len(umi_counts)), umi_counts)).T
    y = gene_counts.reshape((-1, 1))

    if umi_counts.var() == 0:

        mu = gene_counts.mean()
        var = gene_counts.var()
        mu = np.repeat(mu, len(umi_counts))
        var = np.repeat(var, len(umi_counts))
        x2 = mu**2 + var

        return mu, var, x2

    B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    mu = X.dot(B)

    var = (y - mu).var()
    var = np.repeat(var, len(umi_counts))

    mu = mu.ravel()

    x2 = mu**2 + var

    return mu, var, x2
