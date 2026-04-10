from __future__ import annotations

import warnings
from math import ceil
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .._registry import register_function
from .._settings import Colors, EMOJI


def _normalize_gene_list(genes, fitted) -> list[str]:
    available = list(dict.fromkeys(fitted["gene"].astype(str)))
    if genes is None:
        return available
    if isinstance(genes, str):
        genes = [genes]
    genes = [str(gene) for gene in genes]
    missing = [gene for gene in genes if gene not in available]
    if missing:
        raise KeyError(f"Genes not present in fitted trend table: {missing}.")
    return genes


def _default_colors() -> list[str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    return list(colors)


def _resolve_palette(labels: Sequence[str], palette=None) -> dict[str, str]:
    if palette is None:
        colors = _default_colors()
        return {label: colors[i % len(colors)] for i, label in enumerate(labels)}
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        if len(labels) == 1:
            return {labels[0]: cmap(0.6)}
        return {label: cmap(i / max(len(labels) - 1, 1)) for i, label in enumerate(labels)}
    if isinstance(palette, dict):
        return {label: palette[label] for label in labels}
    palette = list(palette)
    return {label: palette[i % len(palette)] for i, label in enumerate(labels)}


def _resolve_linestyles(labels: Sequence[str], linestyles=None) -> dict[str, str]:
    if linestyles is None:
        base = ["-", "--", ":", "-."]
        return {label: base[i % len(base)] for i, label in enumerate(labels)}
    if isinstance(linestyles, dict):
        return {label: linestyles[label] for label in labels}
    linestyles = list(linestyles)
    return {label: linestyles[i % len(linestyles)] for i, label in enumerate(labels)}


def _series_label(group_name: str, gene_name: str, compare_features: bool, compare_groups: bool) -> str:
    if compare_features and compare_groups:
        return f"{group_name} | {gene_name}"
    if compare_features:
        return gene_name
    return group_name


def _default_panel_title(
    panel_mode: str,
    panel_label: str,
    *,
    gene_list: Sequence[str],
    group_list: Sequence[str],
    compare_features: bool,
    compare_groups: bool,
):
    if panel_mode == "groups":
        if compare_features and len(gene_list) > 1 and len(group_list) == 1:
            return ""
        return panel_label
    return panel_label


def _draw_legend(axis, *, legend: bool, legend_outside: bool):
    if not legend:
        return False
    handles, labels = axis.get_legend_handles_labels()
    if not labels:
        return False
    dedup = dict(zip(labels, handles))
    if legend_outside:
        axis.legend(
            dedup.values(),
            dedup.keys(),
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
    else:
        axis.legend(dedup.values(), dedup.keys(), frameon=False)
    return True


@register_function(
    aliases=["dynamic_trends", "plot_gam_trends", "动态趋势图", "gam趋势图"],
    category="pl",
    description="Plot fitted pseudotime GAM trends from `ov.single.dynamic_features` for one or multiple genes across datasets.",
    examples=[
        "res = ov.single.dynamic_features({'A': adata_a, 'B': adata_b}, genes=['tbxta'], pseudotime='palantir_pseudotime')",
        "ov.pl.dynamic_trends(res, genes='tbxta', figsize=(2, 2))",
    ],
    related=["single.dynamic_features", "pl.dynamic_heatmap"],
)
def dynamic_trends(
    result,
    genes=None,
    *,
    groups=None,
    datasets=None,
    compare_features: bool = False,
    compare_groups: bool = False,
    add_line: bool = True,
    add_interval: bool = True,
    add_point: bool = False,
    line_palette=None,
    line_palcolor=None,
    line_style_by: str | None = None,
    line_styles=None,
    figsize: tuple = (3, 3),
    nrows: int | None = None,
    ncols: int | None = None,
    scatter_size: float = 8,
    scatter_alpha: float = 0.2,
    linewidth: float = 2.0,
    legend: bool = True,
    legend_outside: bool = True,
    sharey: bool = False,
    xlabel: str = "Pseudotime",
    ylabel: str = "Expression",
    title=None,
    title_fontsize: float | None = None,
    add_grid: bool = True,
    grid_alpha: float = 0.3,
    grid_linewidth: float = 0.6,
    show: bool | None = None,
    return_axes: bool = False,
    return_fig: bool = False,
    ax=None,
    verbose: bool = True,
):
    """Plot GAM-fitted pseudotime trends for one or more genes.

    Parameters
    ----------
    result
        Result returned by :func:`ov.single.dynamic_features`.
    genes
        Gene name or list of gene names to plot. When ``None``, all fitted
        genes available in ``result`` are shown.
    groups
        Optional fitted group names to include when ``result`` contains
        multiple lineages, cell types, or conditions.
    datasets
        Backward-compatible alias of ``groups``. Prefer ``groups`` in new
        code.
    compare_features
        Whether to compare multiple selected features on the same panel. When
        enabled for a single fitted group, the default panel title is left
        blank to avoid generic titles such as ``adata``.
    compare_groups
        Whether to compare multiple fitted groups on the same panel. Groups are
        the fitted series labels returned by :func:`ov.single.dynamic_features`,
        for example lineages, cell types, or condition names.
    add_line
        Whether to draw the fitted GAM trend lines.
    add_interval
        Whether to draw confidence interval ribbons when available.
    add_point
        Whether to overlay observed expression values. This requires that
        ``result`` was computed with :func:`ov.single.dynamic_features` using
        ``store_raw=True``.
    line_palette
        Optional named matplotlib colormap or palette-like sequence describing
        line colors.
    line_palcolor
        Explicit color mapping or sequence overriding ``line_palette``.
    line_style_by
        Optional semantic used to vary line styles when a comparison panel
        contains multiple series. Choose from ``'features'`` or ``'groups'``.
    line_styles
        Optional line-style mapping or sequence used with ``line_style_by``.
    figsize
        Size of each panel in inches. For multi-gene plots this is interpreted
        as the per-panel size before tiling.
    nrows
        Number of subplot rows for multi-panel layouts.
    ncols
        Number of subplot columns for multi-gene layouts.
    scatter_size
        Marker size used for observed points when ``add_point=True``.
    scatter_alpha
        Alpha transparency for observed points when ``add_point=True``.
    linewidth
        Line width for fitted trend curves.
    legend
        Whether to draw a legend on each axis.
    legend_outside
        Whether to place the legend outside the plotting area on the right.
    sharey
        Whether subplots should share the y-axis in multi-gene layouts.
    xlabel
        Label used for the x-axis.
    ylabel
        Label used for the y-axis.
    title
        Optional title applied to the plotted axis or axes. Pass a list/tuple
        to specify per-panel titles in multi-panel layouts. By default, panel
        titles use the plotted gene or group label automatically. Pass ``''``
        to hide titles explicitly.
    title_fontsize
        Font size used for panel or figure titles. When ``None``, matplotlib's
        default title size is used.
    add_grid
        Whether to draw a light background grid.
    grid_alpha
        Alpha transparency used for the background grid.
    grid_linewidth
        Line width used for the background grid.
    show
        Whether to call ``plt.show()`` before returning. By default, plots are
        shown only when neither ``return_fig`` nor ``return_axes`` is requested.
    return_axes
        Whether to return the created axis or axes. When ``False`` and
        ``return_fig=False``, the function returns ``None`` to avoid notebook
        repr noise.
    return_fig
        Whether to return the figure together with the created axes.
    ax
        Existing matplotlib axis used when plotting a single gene.
    verbose
        Whether to print a short plotting summary.

    Returns
    -------
    None | matplotlib.axes.Axes | list[matplotlib.axes.Axes] | tuple
        Returns ``None`` by default after drawing so notebook cells do not
        print raw axis representations. When ``return_axes=True``, returns the
        created axis or axes. When ``return_fig=True``, returns
        ``(figure, axes)``.
    """
    if not hasattr(result, "fitted"):
        raise TypeError("`result` must be the output of `ov.single.dynamic_features`.")

    groups = groups if groups is not None else datasets
    fitted = result.get_fitted(genes=genes, datasets=groups)
    if fitted.empty:
        raise ValueError("No fitted trends remained after filtering.")

    gene_list = _normalize_gene_list(genes, fitted)
    group_list = list(dict.fromkeys(fitted["dataset"].astype(str)))
    if verbose:
        print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Dynamic trend plotting:{Colors.ENDC}")
        print(f"   {Colors.CYAN}Features: {Colors.BOLD}{len(gene_list)}{Colors.ENDC}{Colors.CYAN} | Groups: {Colors.BOLD}{len(group_list)}{Colors.ENDC}")
        print(f"   {Colors.CYAN}compare_features={Colors.BOLD}{compare_features}{Colors.ENDC}{Colors.CYAN} | compare_groups={Colors.BOLD}{compare_groups}{Colors.ENDC}")
    if line_palcolor is None:
        line_palcolor = line_palette
    compare_groups = bool(compare_groups and len(group_list) > 1)

    raw = result.get_raw(genes=gene_list, datasets=groups) if add_point and hasattr(result, "get_raw") else None
    if add_point and raw is None:
        raise ValueError("`add_point=True` requires `ov.single.dynamic_features(..., store_raw=True)`.")

    if line_style_by not in {None, "features", "groups"}:
        raise ValueError("`line_style_by` must be one of {None, 'features', 'groups'}.")

    if compare_features and compare_groups:
        panel_mode = "combined"
        panel_labels = ["combined"]
    elif compare_features:
        panel_mode = "groups"
        panel_labels = group_list
    else:
        panel_mode = "features"
        panel_labels = gene_list

    if panel_mode == "combined":
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        axes = [ax]
    elif len(panel_labels) == 1:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        axes = [ax]
    else:
        if nrows is None and ncols is None:
            ncols = min(3, len(panel_labels))
            nrows = int(ceil(len(panel_labels) / ncols))
        elif nrows is None:
            ncols = int(ncols)
            nrows = int(ceil(len(panel_labels) / ncols))
        elif ncols is None:
            nrows = int(nrows)
            ncols = int(ceil(len(panel_labels) / nrows))
        else:
            nrows = int(nrows)
            ncols = int(ncols)
            if nrows * ncols < len(panel_labels):
                raise ValueError("`nrows * ncols` must be large enough for the requested panels.")
        panel_width, panel_height = figsize
        fig, axes_arr = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(panel_width * ncols, panel_height * nrows),
            sharey=sharey,
        )
        axes = list(np.ravel(axes_arr))

    if panel_mode == "combined":
        color_labels = group_list if compare_groups else gene_list
        if line_style_by is None and compare_groups and len(gene_list) > 1:
            line_style_by = "features"
    elif panel_mode == "groups":
        color_labels = gene_list
    else:
        color_labels = group_list

    color_map = _resolve_palette(color_labels, palette=line_palcolor)
    if line_style_by == "groups":
        style_map = _resolve_linestyles(group_list, linestyles=line_styles)
    elif line_style_by == "features":
        style_map = _resolve_linestyles(gene_list, linestyles=line_styles)
    else:
        style_map = {}

    legend_drawn = False

    for idx, panel_label in enumerate(panel_labels):
        axis = axes[0] if panel_mode == "combined" else axes[idx]
        if panel_mode == "combined":
            panel_genes = gene_list
            panel_groups = group_list
        elif panel_mode == "groups":
            panel_genes = gene_list
            panel_groups = [panel_label]
        else:
            panel_genes = [panel_label]
            panel_groups = group_list

        for gene in panel_genes:
            gene_fitted = fitted[fitted["gene"].astype(str) == gene]
            gene_raw = None if raw is None else raw[raw["gene"].astype(str) == gene]
            for group_name in panel_groups:
                dataset_fit = gene_fitted[gene_fitted["dataset"].astype(str) == group_name].sort_values("pseudotime")
                if dataset_fit.empty:
                    continue
                if panel_mode == "combined":
                    color = color_map[group_name] if compare_groups else color_map[gene]
                elif panel_mode == "groups":
                    color = color_map[gene]
                else:
                    color = color_map[group_name]

                if line_style_by == "groups":
                    linestyle = style_map[group_name]
                elif line_style_by == "features":
                    linestyle = style_map[gene]
                else:
                    linestyle = "-"

                label = _series_label(group_name, gene, compare_features, compare_groups)
                if panel_mode == "groups" and not compare_groups:
                    label = gene
                elif panel_mode == "features" and not compare_features:
                    label = group_name

                if add_point and gene_raw is not None:
                    dataset_raw = gene_raw[gene_raw["dataset"].astype(str) == group_name]
                    if not dataset_raw.empty:
                        axis.scatter(
                            dataset_raw["pseudotime"],
                            dataset_raw["expression"],
                            s=scatter_size,
                            alpha=scatter_alpha,
                            color=color,
                            linewidths=0,
                        )
                if add_line:
                    axis.plot(
                        dataset_fit["pseudotime"],
                        dataset_fit["fitted"],
                        color=color,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        label=label,
                    )
                if add_interval and dataset_fit["lower"].notna().any() and dataset_fit["upper"].notna().any():
                    axis.fill_between(
                        dataset_fit["pseudotime"].to_numpy(dtype=float),
                        dataset_fit["lower"].to_numpy(dtype=float),
                        dataset_fit["upper"].to_numpy(dtype=float),
                        color=color,
                        alpha=0.2,
                    )

        if panel_mode != "combined":
            if title is None:
                panel_title = _default_panel_title(
                    panel_mode,
                    panel_label,
                    gene_list=gene_list,
                    group_list=group_list,
                    compare_features=compare_features,
                    compare_groups=compare_groups,
                )
            elif isinstance(title, (list, tuple)):
                panel_title = title[idx]
            else:
                panel_title = title
            if title_fontsize is None:
                axis.set_title("" if panel_title is None else str(panel_title))
            else:
                axis.set_title("" if panel_title is None else str(panel_title), fontsize=title_fontsize)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(add_grid, alpha=grid_alpha, linewidth=grid_linewidth)
            legend_drawn = _draw_legend(axis, legend=legend, legend_outside=legend_outside) or legend_drawn

    if panel_mode == "combined":
        axis = axes[0]
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if title_fontsize is None:
            axis.set_title("" if title is None else str(title))
        else:
            axis.set_title("" if title is None else str(title), fontsize=title_fontsize)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(add_grid, alpha=grid_alpha, linewidth=grid_linewidth)
        legend_drawn = _draw_legend(axis, legend=legend, legend_outside=legend_outside) or legend_drawn

    for axis in axes[len(panel_labels):]:
        axis.set_visible(False)

    fig.tight_layout(rect=(0, 0, 0.84, 1) if legend_drawn and legend_outside else None)
    axes_out = axes[0] if panel_mode == "combined" or len(panel_labels) == 1 else axes[: len(panel_labels)]
    if show is None:
        show = not return_fig and not return_axes
    if show:
        backend = plt.get_backend().lower()
        if "agg" in backend:
            fig.canvas.draw()
        else:
            plt.show()
    if return_fig:
        if verbose:
            print(f"{Colors.GREEN}{EMOJI['done']} Dynamic trend plotting completed!{Colors.ENDC}")
        return fig, axes_out
    if return_axes:
        if verbose:
            print(f"{Colors.GREEN}{EMOJI['done']} Dynamic trend plotting completed!{Colors.ENDC}")
        return axes_out
    if verbose:
        print(f"{Colors.GREEN}{EMOJI['done']} Dynamic trend plotting completed!{Colors.ENDC}")
    return None


plot_gam_trends = dynamic_trends
