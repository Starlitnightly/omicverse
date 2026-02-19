from __future__ import annotations

from multiprocessing import Pool
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd
from diptest import diptest as diptst
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn import mixture

if TYPE_CHECKING:
    import numpy.typing as npt


def derive_threshold(
    auc_mtx: pd.DataFrame,
    regulon_name: str,
    seed: float | None = None,
    method: str = "hdt",
) -> float:
    """
    Derive threshold on the AUC values of the given regulon to binarize the cells in
    two clusters: "on" versus "off" state of the regulator.

    :param auc_mtx: The dataframe with the AUC values for all cells and regulons
        (n_cells x n_regulons).
    :param regulon_name: the name of the regulon for which to predict the threshold.
    :param method: The method to use to decide if the distribution of AUC values for
        the given regulon is not unimodel.
        Can be either Hartigan's Dip Test (HDT) or Bayesian Information Content (BIC).
        The former method performs better.
        The BIC compares the BIC for two Gaussian Mixture Models: single versus two
        components.
    :return: The threshold on the AUC values.
    """  # noqa: D205
    assert auc_mtx is not None
    assert not auc_mtx.empty
    assert regulon_name in auc_mtx.columns
    assert method in {"hdt", "bic"}

    data = auc_mtx[regulon_name].values

    if seed:
        np.random.seed(seed=seed)

    def isbimodal(data: npt.NDarray, method: str) -> bool:
        if method == "hdt":
            # Use Hartigan's dip statistic to decide if distribution deviates from
            # unimodality.
            _, pval = diptst(
                np.sort(data, axis=0),
                sort_x=False,
                full_output=False,
                allow_zero=False,
            )
            return pval <= 0.05
        else:
            # Compare Bayesian Information Content of two Gaussian Mixture Models.
            X = data.reshape(-1, 1)
            gmm2 = mixture.GaussianMixture(
                n_components=2, covariance_type="full", random_state=seed
            ).fit(X)
            gmm1 = mixture.GaussianMixture(
                n_components=1, covariance_type="full", random_state=seed
            ).fit(X)
            return gmm2.bic(X) <= gmm1.bic(X)

    if not isbimodal(data, method):
        # For an unimodal distribution the threshold is set as mean plus two standard
        # deviations.
        return data.mean() + 2.0 * data.std()
    else:
        # Fit a two component Gaussian Mixture model on the AUC distribution using an
        # Expectation-Maximization algorithm to identify the peaks in the distribution.
        gmm2 = mixture.GaussianMixture(
            n_components=2, covariance_type="full", random_state=seed
        ).fit(data.reshape(-1, 1))
        # For a bimodal distribution the threshold is defined as the "trough" in
        # between the two peaks. This is solved as a minimization problem on the kernel
        # smoothed density.
        return minimize_scalar(
            fun=stats.gaussian_kde(data), bounds=sorted(gmm2.means_), method="bounded"
        ).x[0]


def binarize(
    auc_mtx: pd.DataFrame,
    threshold_overides: Mapping[str, float] | None = None,
    seed: float | None = None,
    num_workers: int = 1,
) -> (pd.DataFrame, pd.Series):
    """
    "Binarize" the supplied AUC matrix, i.e. decide if for each cells in the matrix a
    regulon is active or not based on the bimodal distribution of the AUC values for
    that regulon.

    :param auc_mtx: The dataframe with the AUC values for all cells and regulons
        (n_cells x n_regulons).
    :param threshold_overides: A dictionary that maps name of regulons to manually set
        thresholds.
    :return: A "binarized" dataframe and a series containing the AUC threshold used for
        each regulon.
    """  # noqa: D205

    def derive_thresholds(
        auc_mtx: pd.DataFrame, seed: float | None = seed
    ) -> pd.Series:
        with Pool(processes=num_workers) as p:
            thrs = p.starmap(
                derive_threshold, [(auc_mtx, c, seed) for c in auc_mtx.columns]
            )
        return pd.Series(index=auc_mtx.columns, data=thrs)

    thresholds = derive_thresholds(auc_mtx)
    if threshold_overides is not None:
        thresholds[list(threshold_overides.keys())] = list(threshold_overides.values())
    return (auc_mtx > thresholds).astype(int), thresholds
