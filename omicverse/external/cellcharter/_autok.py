"""Automatic K selection for CellCharter via clustering stability.

Implements the ClusterAutoK algorithm from the CellCharter package:
for each candidate K, runs clustering multiple times and measures
pairwise stability (Fowlkes-Mallows score) between adjacent K values.
The K with highest stability is selected.
"""

from __future__ import annotations

import logging
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import fowlkes_mallows_score

from ._gmm import Cluster
from ._utils import AnyRandom

logger = logging.getLogger(__name__)


class ClusterAutoK:
    """Automatic selection of cluster number K via stability analysis.

    For each K in ``n_clusters``, the clustering is repeated ``max_runs``
    times. Stability is measured as the Fowlkes-Mallows score between
    labels at K vs K-1 and K vs K+1. The K with the highest mean
    stability is selected as ``best_k``.

    Parameters
    ----------
    n_clusters
        Range of K values to test. Either a tuple ``(min, max)``
        (inclusive) or a list of specific K values.
    max_runs
        Maximum number of clustering repetitions per K.
    convergence_tol
        Early stopping: if the MAPE of the stability curve between
        consecutive iterations is below this threshold, stop.
    model_params
        Keyword arguments passed to :class:`Cluster`.
    similarity_function
        Pairwise label comparison metric. Default: Fowlkes-Mallows score.
    """

    def __init__(
        self,
        n_clusters: tuple[int, int] | list[int] = (2, 10),
        max_runs: int = 10,
        convergence_tol: float = 0.01,
        model_params: dict | None = None,
        similarity_function: Callable = fowlkes_mallows_score,
    ):
        if isinstance(n_clusters, tuple):
            # Expand range by 1 on each side for boundary comparisons
            self._k_range = list(range(n_clusters[0] - 1, n_clusters[1] + 2))
        else:
            k_sorted = sorted(n_clusters)
            self._k_range = list(range(k_sorted[0] - 1, k_sorted[-1] + 2))

        self._inner_range = self._k_range[1:-1]  # reported K values
        self.max_runs = max_runs
        self.convergence_tol = convergence_tol
        self.model_params = model_params or {}
        self.similarity_function = similarity_function

        # Filled after fit
        self._labels: dict[int, list[np.ndarray]] = {k: [] for k in self._k_range}
        self._stability: np.ndarray | None = None
        self._models: dict[int, Cluster] = {}

    @property
    def best_k(self) -> int:
        """K with highest mean stability."""
        if self._stability is None:
            raise RuntimeError("Call `fit` first.")
        idx = int(np.argmax(self._stability.mean(axis=0)))
        return self._inner_range[idx]

    @property
    def peaks(self) -> np.ndarray:
        """All K values at local stability peaks."""
        if self._stability is None:
            raise RuntimeError("Call `fit` first.")
        from scipy.signal import find_peaks
        mean_stab = self._stability.mean(axis=0)
        peak_idx, _ = find_peaks(mean_stab)
        return np.array([self._inner_range[i] for i in peak_idx])

    def fit(self, adata: ad.AnnData, use_rep: str = "X_cellcharter"):
        """Run repeated clustering for each K and compute stability."""
        prev_stability = None

        for run in range(self.max_runs):
            # Cluster for each K
            for k in self._k_range:
                seed = self.model_params.get("random_state", 0)
                model = Cluster(
                    n_clusters=k,
                    random_state=seed + run * 1000 + k,
                    **{k_: v for k_, v in self.model_params.items() if k_ != "random_state"},
                )
                model.fit(adata, use_rep=use_rep)
                labels = model.predict(adata, use_rep=use_rep)
                self._labels[k].append(np.asarray(labels))
                self._models[k] = model

            # Compute stability (need at least 2 runs)
            if run < 1:
                continue

            stability = self._compute_stability()
            self._stability = stability

            # Early stopping via MAPE
            if prev_stability is not None and stability.shape[0] >= 2:
                mean_curr = stability.mean(axis=0)
                mean_prev = prev_stability.mean(axis=0)
                denom = np.clip(np.abs(mean_prev), 1e-10, None)
                mape = np.mean(np.abs(mean_curr - mean_prev) / denom)
                if mape < self.convergence_tol:
                    logger.info(f"ClusterAutoK converged after {run + 1} runs (MAPE={mape:.4f})")
                    break

            prev_stability = stability

        return self

    def _compute_stability(self) -> np.ndarray:
        """Compute mirrored stability matrix (runs x inner_K)."""
        n_inner = len(self._inner_range)
        n_runs = min(len(self._labels[k]) for k in self._k_range)

        # For each pair of runs, compute similarity between K and K±1
        stab_down = np.zeros((n_runs, n_inner))  # K vs K-1
        stab_up = np.zeros((n_runs, n_inner))    # K vs K+1

        for r in range(n_runs):
            for j, k in enumerate(self._inner_range):
                k_prev = self._k_range[j]      # K-1
                k_next = self._k_range[j + 2]   # K+1
                stab_down[r, j] = self.similarity_function(
                    self._labels[k][r], self._labels[k_prev][r]
                )
                stab_up[r, j] = self.similarity_function(
                    self._labels[k][r], self._labels[k_next][r]
                )

        # Mirror: stability[j] = stab_down[j] + stab_up[j-1] (when applicable)
        stability = np.zeros((n_runs, n_inner))
        for j in range(n_inner):
            stability[:, j] = stab_down[:, j]
            if j > 0:
                stability[:, j] += stab_up[:, j - 1]

        return stability

    def predict(
        self, adata: ad.AnnData, use_rep: str = "X_cellcharter", k: int | None = None
    ) -> pd.Categorical:
        """Predict labels using the model at ``k`` (default: ``best_k``)."""
        if k is None:
            k = self.best_k
        if k not in self._models:
            raise ValueError(f"K={k} not in fitted range {self._inner_range}")
        return self._models[k].predict(adata, use_rep=use_rep)


def plot_autok_stability(
    autok: ClusterAutoK,
    ax=None,
    figsize: tuple[float, float] = (6, 4),
    color: str = "#2176AE",
    save: str | None = None,
):
    """Plot the stability curve from ClusterAutoK.

    Parameters
    ----------
    autok
        Fitted ClusterAutoK instance.
    ax
        Matplotlib axes. If None, a new figure is created.
    figsize
        Figure size when creating a new axes.
    color
        Line color.
    save
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if autok._stability is None:
        raise RuntimeError("ClusterAutoK has not been fitted yet.")

    k_vals = autok._inner_range
    mean_stab = autok._stability.mean(axis=0)
    std_stab = autok._stability.std(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(k_vals, mean_stab, "o-", color=color, linewidth=2, markersize=5)
    ax.fill_between(k_vals, mean_stab - std_stab, mean_stab + std_stab,
                    alpha=0.2, color=color)

    best = autok.best_k
    best_idx = k_vals.index(best)
    ax.axvline(best, color="#E63946", linestyle="--", linewidth=1.5,
               label=f"Best K = {best}")
    ax.scatter([best], [mean_stab[best_idx]], color="#E63946", s=80,
               zorder=5, edgecolors="white", linewidth=2)

    ax.set_xlabel("N. clusters", fontsize=12)
    ax.set_ylabel("Stability", fontsize=12)
    ax.set_title("CellCharter Auto-K Stability", fontsize=13)
    ax.set_xticks(k_vals)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11)

    if save:
        ax.get_figure().savefig(save, dpi=150, bbox_inches="tight")

    return ax
