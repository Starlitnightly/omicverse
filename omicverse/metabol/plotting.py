r"""Metabolomics-specific plots.

Three core plots every metabolomics paper has:

- **Volcano plot** — log2FC vs -log10(padj), metabolite labels on
  significant hits.
- **S-plot** — OPLS-DA signature plot of p(corr) vs p(cov) used to
  interpret which metabolites drive the group separation.
- **VIP bar** — top-N metabolites by VIP score.

These intentionally return ``(fig, ax)`` tuples in omicverse's
convention, so downstream code can layer additional annotations and
ship a single publication-ready figure.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._plsda import PLSDAResult


def volcano(
    deg: pd.DataFrame,
    *,
    padj_thresh: float = 0.05,
    log2fc_thresh: float = 1.0,
    label_top_n: int = 10,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (5.5, 4.5),
):
    """Metabolomics volcano plot — log2FC vs -log10(padj)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    x = deg["log2fc"].to_numpy()
    y = -np.log10(np.clip(deg["padj"].to_numpy(), 1e-300, 1.0))
    is_sig = (deg["padj"] < padj_thresh) & (deg["log2fc"].abs() >= log2fc_thresh)
    up = is_sig & (deg["log2fc"] > 0)
    down = is_sig & (deg["log2fc"] < 0)
    ax.scatter(x[~is_sig], y[~is_sig], c="#bfbfbf", s=8, alpha=0.6, label="n.s.")
    ax.scatter(x[up], y[up], c="#c0392b", s=14, alpha=0.85, label="up")
    ax.scatter(x[down], y[down], c="#2980b9", s=14, alpha=0.85, label="down")
    ax.axhline(-np.log10(padj_thresh), ls="--", c="grey", lw=0.8)
    ax.axvline(+log2fc_thresh, ls="--", c="grey", lw=0.8)
    ax.axvline(-log2fc_thresh, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("log$_2$ fold-change")
    ax.set_ylabel("-log$_{10}$ (padj)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    # Label top hits
    if label_top_n > 0:
        top = deg[is_sig].assign(_y=y[is_sig]).nlargest(label_top_n, "_y")
        for name, row in top.iterrows():
            ax.text(row["log2fc"], -np.log10(max(row["padj"], 1e-300)),
                    str(name), fontsize=7, ha="left", va="bottom")
    return fig, ax


def s_plot(
    result: PLSDAResult,
    adata,
    *,
    label_top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (5.5, 4.5),
):
    """OPLS-DA S-plot: p(cov) vs p(corr), i.e. covariance vs correlation
    between each feature and the predictive component.

    This is the classic visualization for interpreting OPLS-DA loadings
    (Wiklund et al 2008). Features in the two "arms" of the S
    (high-covariance, high-correlation) are the strongest drivers of
    the case/control separation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    X = np.asarray(adata.X, dtype=np.float64)
    t = result.scores[:, 0]
    # p(cov)_j = cov(X_j, t) / sd(t)
    t_centered = t - t.mean()
    x_centered = X - X.mean(axis=0)
    cov = (x_centered.T @ t_centered) / (t_centered.size - 1)
    pcov = cov / (t.std(ddof=1) if t.std(ddof=1) > 0 else 1.0)
    # p(corr)_j = cov(X_j, t) / (sd(X_j) * sd(t))
    sd_x = x_centered.std(axis=0, ddof=1)
    pcorr = np.where(sd_x > 0, cov / (sd_x * (t.std(ddof=1) if t.std(ddof=1) > 0 else 1.0)), 0.0)

    ax.scatter(pcov, pcorr, c=np.abs(result.vip), cmap="viridis", s=14, alpha=0.75)
    ax.axhline(0, c="grey", lw=0.6); ax.axvline(0, c="grey", lw=0.6)
    ax.set_xlabel("p(cov) — covariance w/ predictive component")
    ax.set_ylabel("p(corr) — correlation w/ predictive component")
    # Label tallest-|pcorr| × |pcov| points
    if label_top_n > 0:
        score = np.abs(pcov) * np.abs(pcorr)
        idx = np.argsort(-score)[:label_top_n]
        for i in idx:
            ax.text(pcov[i], pcorr[i], str(adata.var_names[i]),
                    fontsize=7, ha="left", va="bottom")
    return fig, ax


def vip_bar(
    result: PLSDAResult,
    var_names,
    *,
    top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (5.0, 5.0),
):
    """Horizontal bar chart of top-``top_n`` VIP metabolites."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    tbl = result.to_vip_table(var_names).head(top_n)
    # Color by sign of the regression coefficient
    colors = ["#c0392b" if c > 0 else "#2980b9" for c in tbl["coef"]]
    ax.barh(tbl.index[::-1], tbl["vip"].iloc[::-1], color=colors[::-1])
    ax.axvline(1.0, c="grey", lw=0.6, ls="--")   # Wold VIP>1 threshold
    ax.set_xlabel("VIP score")
    ax.set_ylabel("")
    return fig, ax
