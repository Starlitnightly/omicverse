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
    use_pvalue: bool = False,
    clip_log2fc: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (5.5, 4.5),
):
    """Metabolomics volcano plot — log2FC vs -log10(padj) (or pvalue).

    Parameters
    ----------
    use_pvalue
        Plot against the raw ``pvalue`` column instead of the BH-adjusted
        ``padj``. Useful for small-n untargeted LC-MS where the 5000+
        feature FDR burden means no peak survives FDR; raw p-value is
        the honest axis for the volcano in that regime.
    clip_log2fc
        Clip the x-axis to ±this value. On LC-MS data with below-detection
        zeros, a handful of features can have log2fc up to ±25 after
        pseudo-count logging and completely dominate the plot. Set
        e.g. ``clip_log2fc=5`` to keep the volcano interpretable.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    x = deg["log2fc"].to_numpy(dtype=float)
    stat_col = "pvalue" if use_pvalue else "padj"
    y = -np.log10(np.clip(deg[stat_col].to_numpy(dtype=float), 1e-300, 1.0))
    is_sig = (deg[stat_col] < padj_thresh) & (deg["log2fc"].abs() >= log2fc_thresh)
    up = is_sig & (deg["log2fc"] > 0)
    down = is_sig & (deg["log2fc"] < 0)
    # Clip if requested — keeps the x-axis readable on LC-MS-zero-inflated data
    if clip_log2fc is not None:
        x = np.clip(x, -clip_log2fc, clip_log2fc)
    ax.scatter(x[~is_sig], y[~is_sig], c="#bfbfbf", s=8, alpha=0.6, label="n.s.")
    ax.scatter(x[up], y[up], c="#c0392b", s=14, alpha=0.85, label="up")
    ax.scatter(x[down], y[down], c="#2980b9", s=14, alpha=0.85, label="down")
    ax.axhline(-np.log10(padj_thresh), ls="--", c="grey", lw=0.8)
    ax.axvline(+log2fc_thresh, ls="--", c="grey", lw=0.8)
    ax.axvline(-log2fc_thresh, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("log$_2$ fold-change")
    ax.set_ylabel("-log$_{10}$ (pvalue)" if use_pvalue else "-log$_{10}$ (padj)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    # Label top hits
    if label_top_n > 0 and is_sig.any():
        top = deg[is_sig].assign(_y=y[is_sig]).nlargest(label_top_n, "_y")
        for name, row in top.iterrows():
            x_lab = row["log2fc"]
            if clip_log2fc is not None:
                x_lab = max(min(x_lab, clip_log2fc), -clip_log2fc)
            ax.text(x_lab, -np.log10(max(row[stat_col], 1e-300)),
                    str(name), fontsize=7, ha="left", va="bottom")
    return fig, ax


def pathway_bar(
    enrichment: pd.DataFrame,
    *,
    term_col: Optional[str] = None,
    score_col: str = "pvalue",
    top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6.0, 5.0),
):
    """Horizontal bar chart of pathway enrichment p-values.

    Works for both ``msea_ora`` output (``pathway`` / ``pvalue``) and
    ``lion_enrichment`` output (``term`` / ``pvalue``); auto-detects
    the term column if ``term_col`` isn't given.

    Parameters
    ----------
    enrichment
        DataFrame returned by :func:`msea_ora`, :func:`msea_gsea`, or
        :func:`lion_enrichment`.
    term_col
        Column holding the pathway / term name. Auto-picks from
        ``pathway`` / ``term`` / ``Term`` / the index if None.
    score_col
        Column to plot. ``"pvalue"`` (default) → ``-log10(pvalue)`` bars;
        ``"padj"`` for FDR-adjusted; ``"nes"`` for GSEA normalized
        enrichment score (plotted as-is, signed).
    top_n
        Show the top-``top_n`` terms by the score.
    """
    df = enrichment.copy()
    if term_col is None:
        for candidate in ("pathway", "term", "Term"):
            if candidate in df.columns:
                term_col = candidate
                break
        else:
            df = df.reset_index().rename(columns={"index": "pathway"})
            term_col = "pathway"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if score_col in ("pvalue", "padj", "pval", "fdr"):
        df["_score"] = -np.log10(np.clip(df[score_col].astype(float), 1e-300, 1.0))
        xlabel = f"-log$_{{10}}$ ({score_col})"
    else:
        df["_score"] = df[score_col].astype(float)
        xlabel = score_col

    df = df.sort_values("_score", ascending=False).head(top_n)
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in df["_score"]]
    ax.barh(df[term_col][::-1], df["_score"][::-1], color=colors[::-1])
    ax.set_xlabel(xlabel)
    return fig, ax


def pathway_dot(
    enrichment: pd.DataFrame,
    *,
    term_col: Optional[str] = None,
    size_col: str = "overlap",
    x_col: str = "odds_ratio",
    color_col: str = "pvalue",
    top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6.5, 5.5),
):
    """Dot plot of pathway enrichment — the de-facto standard figure.

    Matches the layout of `scanpy`'s / `clusterProfiler`'s dot plot:
    each row is a pathway, dot **size** encodes overlap count, **x**
    position encodes fold-enrichment (odds ratio or NES), **color**
    encodes -log10(p-value).

    Parameters
    ----------
    size_col
        Column mapped to dot size. Default ``"overlap"`` (ORA output);
        use ``"matched_size"`` for GSEA output.
    x_col
        Column mapped to x position. Default ``"odds_ratio"`` (ORA);
        use ``"nes"`` for GSEA.
    color_col
        Column mapped to color (via ``-log10``). Default ``"pvalue"``.
    """
    df = enrichment.copy()
    if term_col is None:
        for candidate in ("pathway", "term", "Term"):
            if candidate in df.columns:
                term_col = candidate
                break
        else:
            df = df.reset_index().rename(columns={"index": "pathway"})
            term_col = "pathway"

    for col in (size_col, x_col, color_col):
        if col not in df.columns:
            raise KeyError(
                f"column {col!r} not in enrichment DataFrame "
                f"(columns: {list(df.columns)})"
            )

    df["_neglog10"] = -np.log10(np.clip(df[color_col].astype(float), 1e-300, 1.0))
    df = df.sort_values("_neglog10", ascending=False).head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y = np.arange(len(df))
    sizes = df[size_col].astype(float).to_numpy()
    sizes_scaled = 30 + (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1) * 200
    scatter = ax.scatter(
        df[x_col].astype(float), y, s=sizes_scaled,
        c=df["_neglog10"], cmap="viridis", edgecolor="black", linewidth=0.4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df[term_col])
    ax.invert_yaxis()
    ax.set_xlabel(x_col)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.04)
    cbar.set_label(f"-log$_{{10}}$ ({color_col})")
    # Size legend
    for s in np.percentile(sizes, [25, 75, 100]):
        ax.scatter([], [], s=30 + (s - sizes.min()) / max(sizes.max() - sizes.min(), 1) * 200,
                   c="grey", label=f"{size_col}={int(s)}")
    ax.legend(loc="center left", bbox_to_anchor=(1.35, 0.5), fontsize=8,
              frameon=False, title=size_col)
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
