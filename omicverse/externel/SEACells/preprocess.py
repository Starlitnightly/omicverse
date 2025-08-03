import numpy as np


def log_transform(ad, ps=0.1):
    """Compute log transformation of AnnData data in place.

    :param ad: anndata.AnnData object to be normalized
    :param ps: (float) pseudo-count for log transformation to avoid log(0) errors
    """
    ad.X = np.log2(ad.X + ps) - np.log2(ps)
