from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_binarization(
    auc_mtx: pd.DataFrame, regulon_name: str, threshold: float, bins: int = 200, ax=None
) -> None:
    """
    Plot the "binarization" process for the given regulon.

    :param auc_mtx: The dataframe with the AUC values for all cells and regulons (n_cells x n_regulons).
    :param regulon_name: The name of the regulon.
    :param bins: The number of bins to use in the AUC histogram.
    :param threshold: The threshold to use for binarization.
    """
    if ax is None:
        ax = plt.gca()
    sns.distplot(auc_mtx[regulon_name], ax=ax, norm_hist=True, bins=bins)

    ylim = ax.get_ylim()
    ax.plot([threshold] * 2, ylim, "r:")
    ax.set_ylim(ylim)
    ax.set_xlabel("AUC")
    ax.set_ylabel("#")
    ax.set_title(regulon_name)


def plot_rss(rss, cell_type, top_n=5, max_n=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))
    if max_n is None:
        max_n = rss.shape[1]
    data = rss.T[cell_type].sort_values(ascending=False)[0:max_n]
    ax.plot(np.arange(len(data)), data, ".")
    ax.set_ylim([floor(data.min() * 100.0) / 100.0, ceil(data.max() * 100.0) / 100.0])
    ax.set_ylabel("RSS")
    ax.set_xlabel("Regulon")
    ax.set_title(cell_type)
    ax.set_xticklabels([])

    font = {
        "color": "red",
        "weight": "normal",
        "size": 4,
    }

    for idx, (regulon_name, rss_val) in enumerate(
        zip(data[0:top_n].index, data[0:top_n].values)
    ):
        ax.plot([idx, idx], [rss_val, rss_val], "r.")
        ax.text(
            idx + (max_n / 25),
            rss_val,
            regulon_name,
            fontdict=font,
            horizontalalignment="left",
            verticalalignment="center",
        )
