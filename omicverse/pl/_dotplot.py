from typing import Sequence, Mapping, Literal, Union, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import marsilea as ma
import marsilea.plotter as mp
from matplotlib.colors import Normalize, Colormap
from matplotlib.axes import Axes as _AxesSubplot
from anndata import AnnData
import functools
import operator
from typing import Any
from ._palette import palette_28, palette_56

_VarNames = Union[str, Sequence[str]]

def dotplot(
    adata: AnnData,
    var_names: Union[_VarNames, Mapping[str, _VarNames]],
    groupby: Union[str, Sequence[str]],
    *,
    use_raw: Optional[bool] = None,
    log: bool = False,
    num_categories: int = 7,
    categories_order: Optional[Sequence[str]] = None,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    standard_scale: Optional[Literal['var', 'group']] = None,
    title: Optional[str] = None,
    colorbar_title: Optional[str] = 'Mean expression\nin group',
    size_title: Optional[str] = 'Fraction of cells\nin group (%)',
    figsize: Optional[Tuple[float, float]] = None,
    dendrogram: Union[bool, str] = False,
    gene_symbols: Optional[str] = None,
    var_group_positions: Optional[Sequence[Tuple[int, int]]] = None,
    var_group_labels: Optional[Sequence[str]] = None,
    var_group_rotation: Optional[float] = None,
    layer: Optional[str] = None,
    swap_axes: Optional[bool] = False,
    dot_color_df: Optional[pd.DataFrame] = None,
    show: Optional[bool] = None,
    save: Optional[Union[str, bool]] = None,
    ax: Optional[_AxesSubplot] = None,
    return_fig: Optional[bool] = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    norm: Optional[Normalize] = None,
    cmap: Union[Colormap, str, None] = 'Reds',
    dot_max: Optional[float] = None,
    dot_min: Optional[float] = None,
    smallest_dot: float = 0.0,
    fontsize: int = 12,
    **kwds,
) -> Optional[Union[Dict, 'DotPlot']]:
    """
    Make a dot plot of the expression values of `var_names`.
    
    For each var_name and each `groupby` category a dot is plotted.
    Each dot represents two values: mean expression within each category
    (visualized by color) and fraction of cells expressing the `var_name` in the
    category (visualized by the size of the dot).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    var_names : str or list of str or dict
        Variables to plot.
    groupby : str or list of str
        The key of the observation grouping to consider.
    use_raw : bool, optional
        Use `raw` attribute of `adata` if present.
    standard_scale : {'var', 'group'} or None
        Whether to standardize data.
    fontsize : int, optional (default: 12)
        Font size for labels and legends. Titles will be one point larger.
    
    Returns
    -------
    If `return_fig` is True, returns the figure object.
    If `show` is False, returns axes dictionary.
    """
    # Convert var_names to list if string
    if isinstance(var_names, str):
        var_names = [var_names]
    elif isinstance(var_names, Mapping):
        # Get gene groups
        gene_groups = []
        var_names_list = []
        
        # Get cell type order
        if categories_order is not None:
            group_order = categories_order
        elif pd.api.types.is_categorical_dtype(adata.obs[groupby]):
            group_order = list(adata.obs[groupby].cat.categories)
        else:
            group_order = list(adata.obs[groupby].unique())
        
        # Order gene groups according to cell types
        for group in group_order:
            if group in var_names:
                genes = var_names[group]
                if isinstance(genes, str):
                    genes = [genes]
                var_names_list.extend(genes)
                gene_groups.extend([group] * len(genes))
        
        # Add any remaining groups that weren't in the cell types
        for group, genes in var_names.items():
            if group not in group_order:
                if isinstance(genes, str):
                    genes = [genes]
                var_names_list.extend(genes)
                gene_groups.extend([group] * len(genes))
        
        var_names = var_names_list
    
    # Get expression matrix
    if use_raw and adata.raw is not None:
        matrix = adata.raw.X
        var_names_idx = [adata.raw.var_names.get_loc(name) for name in var_names]
    else:
        matrix = adata.X if layer is None else adata.layers[layer]
        var_names_idx = [adata.var_names.get_loc(name) for name in var_names]
    
    # Determine category order
    if categories_order is not None:
        cats = categories_order
    else:
        # Use the categorical order from adata if available
        if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
            cats = adata.obs[groupby].cat.categories
        else:
            # If not categorical, get unique values
            cats = adata.obs[groupby].unique()
    
    # Get aggregated data with specified order
    agg = adata.obs[groupby].value_counts().reindex(cats)
    cell_counts = agg.to_numpy()
    
    # Calculate mean expression and fraction of expressing cells
    means = np.zeros((len(agg), len(var_names)))
    fractions = np.zeros_like(means)
    
    for i, group in enumerate(agg.index):
        mask = adata.obs[groupby] == group
        group_matrix = matrix[mask][:, var_names_idx]
        
        # Calculate mean expression
        if mean_only_expressed:
            expressed = group_matrix > expression_cutoff
            means[i] = np.array([
                group_matrix[:, j][expressed[:, j]].mean() if expressed[:, j].any() else 0
                for j in range(group_matrix.shape[1])
            ])
        else:
            means[i] = np.mean(group_matrix, axis=0)
        
        # Calculate fraction of expressing cells
        fractions[i] = np.mean(group_matrix > expression_cutoff, axis=0)
    
    # Scale if requested
    if standard_scale == 'group':
        means = (means - means.min(axis=1, keepdims=True)) / (means.max(axis=1, keepdims=True) - means.min(axis=1, keepdims=True))
    elif standard_scale == 'var':
        means = (means - means.min(axis=0)) / (means.max(axis=0) - means.min(axis=0))
    
    # Handle dot size limits
    if dot_max is not None:
        fractions = np.minimum(fractions, dot_max)
    if dot_min is not None:
        fractions = np.maximum(fractions, dot_min)
    
    # Scale dot sizes to account for smallest_dot
    if smallest_dot > 0:
        fractions = smallest_dot + (1 - smallest_dot) * fractions
    
    # Create the plot
    h, w = means.shape
    
    # Get colors if available
    try:
        color_key = f"{groupby}_colors"
        colors = adata.uns[color_key]
        color_dict = dict(zip(agg.index, colors))
        cell_colors = [color_dict[i] for i in agg.index]
    except KeyError:
        cell_colors = None
    
    # Handle color normalization
    if norm is None and (vmin is not None or vmax is not None or vcenter is not None):
        if vcenter is not None:
            vmin = vmin if vmin is not None else means.min()
            vmax = vmax if vmax is not None else means.max()
            norm = mp.DivergingNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
        else:
            norm = mp.Normalize(vmin=vmin if vmin is not None else means.min(),
                              vmax=vmax if vmax is not None else means.max())
    
    # Create SizedHeatmap
    m = ma.SizedHeatmap(
        size=fractions,
        color=means,
        cluster_data=fractions if dendrogram else None,
        height=h / 3,
        width=w / 3,
        edgecolor="lightgray",
        cmap=cmap,
        norm=norm,
        size_legend_kws=dict(
            colors="#c2c2c2",
            title=size_title,
            labels=[f"{int(x*100)}%" for x in [0.2, 0.4, 0.6, 0.8, 1.0]],
            show_at=[0.2, 0.4, 0.6, 0.8, 1.0],
            fontsize=fontsize,
            ncol=3,
            title_fontproperties={"size": fontsize + 1, "weight": 100}
        ),
        color_legend_kws=dict(
            title=colorbar_title,
            fontsize=fontsize,
            orientation="horizontal",
            title_fontproperties={"size": fontsize + 1, "weight": 100}
        ),
    )
    
    # Add labels
    m.add_top(mp.Labels(var_names, fontsize=fontsize), pad=0.1)
    
    # Group genes if var_names was a dictionary
    if 'gene_groups' in locals():
        # Get colors for gene groups
        try:
            # Try to get colors from adata.uns
            group_colors = adata.uns[f"{groupby}_colors"]
            color_dict = dict(zip(
                adata.obs[groupby].cat.categories,
                group_colors
            ))
            # Get unique groups and check which ones are not in color_dict
            unique_groups = list(dict.fromkeys(gene_groups))
            missing_groups = [g for g in unique_groups if g not in color_dict]
            
            # If there are missing groups, add colors from palette
            if missing_groups:
                if len(palette_28) >= len(color_dict) + len(missing_groups):
                    extra_colors = palette_28[len(color_dict):len(color_dict) + len(missing_groups)]
                else:
                    extra_colors = palette_56[len(color_dict):len(color_dict) + len(missing_groups)]
                color_dict.update(dict(zip(missing_groups, extra_colors)))
        except (KeyError, AttributeError):
            # If colors not found in uns, use default palette
            unique_groups = list(dict.fromkeys(gene_groups))
            if len(unique_groups) <= 28:
                palette = palette_28
            else:
                palette = palette_56
            color_dict = dict(zip(unique_groups, palette[:len(unique_groups)]))
        
        # Add color bars with matching order
        # Add group labels
        # Only show used colors in legend and increase group spacing
        used_groups = list(dict.fromkeys(gene_groups))
        used_color_dict = {k: color_dict[k] for k in used_groups}
        m.add_top(
            mp.Colors(gene_groups, palette=used_color_dict),
            pad=0.1,
            size=0.15,
        )
        # Add group labels with increased spacing
        m.group_cols(gene_groups)
    
    # Add cell type colors if available
    try:
        color_key = f"{groupby}_colors"
        if color_key in adata.uns:
            # Create color dictionary
            color_dict = dict(zip(cats, adata.uns[color_key]))
            # Add color bar
            m.add_left(
                mp.Colors(agg.index, palette=color_dict),
                size=0.15,
                pad=0.1,
                legend=False,
            )
    except KeyError:
        pass
    
    # Add cell type labels
    m.add_left(mp.Labels(agg.index, align="right", fontsize=fontsize), pad=0.1)
    
    # Add cell counts
    m.add_right(
        mp.Numbers(
            cell_counts,
            color="#EEB76B",
            label="Count",
            label_props={'size': fontsize},
            props={'size': fontsize},
            show_value=False
        ),
        size=0.5,
        pad=0.1,
    )
    
    # Add dendrogram if requested
    if dendrogram:
        m.add_dendrogram("right", pad=0.1)
    
    # Add legends
    m.add_legends(box_padding=2)
    
    # Render the plot
    fig = m.render()
    
    if return_fig:
        return fig
    elif not show:
        return m
    return None

def rank_genes_groups_df(
    adata: AnnData,
    group: str,
    key: str = "rank_genes_groups",
    gene_symbols: str | None = None,
    log2fc_min: float | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with the results of rank_genes_groups."""
    d = pd.DataFrame()
    for k in ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj']:
        if k in adata.uns[key]:
            d[k] = pd.DataFrame(adata.uns[key][k])[group]
    
    if log2fc_min is not None:
        d = d[d['logfoldchanges'].abs() > log2fc_min]
    
    return d

def _get_values_to_plot(
    adata: AnnData,
    values_to_plot: str,
    var_names: Sequence[str],
    key: str = 'rank_genes_groups',
    gene_symbols: str | None = None,
) -> pd.DataFrame:
    """Get values to plot from rank_genes_groups results."""
    if values_to_plot not in adata.uns[key]:
        raise ValueError(
            f'The key {values_to_plot} is not available in adata.uns["{key}"]'
        )
    
    # Get the values for each group
    values = pd.DataFrame(adata.uns[key][values_to_plot])
    values.index = pd.DataFrame(adata.uns[key]['names']).iloc[:, 0]
    values = values.loc[var_names]
    
    return values

def rank_genes_groups_dotplot(
    adata: AnnData,
    plot_type: str = "dotplot",
    *,
    groups: Optional[Union[str, Sequence[str]]] = None,
    n_genes: Optional[int] = None,
    groupby: Optional[str] = None,
    values_to_plot: Optional[str] = None,
    var_names: Optional[Union[Sequence[str], Mapping[str, Sequence[str]]]] = None,
    min_logfoldchange: Optional[float] = None,
    key: Optional[str] = None,
    show: Optional[bool] = None,
    save: Optional[bool] = None,
    return_fig: bool = False,
    gene_symbols: Optional[str] = None,
    **kwds: Any,
) -> Optional[Union[Dict, Any]]:
    """
    Create a dot plot from rank_genes_groups results.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    plot_type : str
        Currently only 'dotplot' is supported.
    groups : str or list of str, optional
        Groups to include in the plot.
    n_genes : int, optional
        Number of genes to include in the plot.
    groupby : str, optional
        Key in `adata.obs` to group by.
    values_to_plot : str, optional
        Key in rank_genes_groups results to plot (e.g. 'logfoldchanges', 'scores').
    var_names : str or list of str or dict, optional
        Variables to include in the plot. Can be:
        - A list of gene names: ['gene1', 'gene2', ...]
        - A dictionary mapping group names to gene lists: {'group1': ['gene1', 'gene2'], 'group2': ['gene3', 'gene4']}
        When a dictionary is provided, genes will be grouped and labeled accordingly in the plot.
    min_logfoldchange : float, optional
        Minimum log fold change to include in the plot.
    key : str, optional
        Key in `adata.uns` to use for rank_genes_groups results.
    show : bool, optional
        Whether to show the plot.
    save : bool, optional
        Whether to save the plot.
    return_fig : bool
        Whether to return the figure object.
    gene_symbols : str, optional
        Key for gene symbols in `adata.var`.
    **kwds : dict
        Additional keyword arguments to pass to dotplot.
    
    Returns
    -------
    If `return_fig` is True, returns the figure object.
    If `show` is False, returns axes dictionary.
    
    Examples
    --------
    >>> # Basic usage with top genes
    >>> sc.pl.rank_genes_groups_dotplot(adata, n_genes=5)
    
    >>> # Using logfoldchanges for coloring
    >>> sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, values_to_plot='logfoldchanges')
    
    >>> # Grouping genes manually
    >>> gene_groups = {
    ...     'Group1': ['gene1', 'gene2'],
    ...     'Group2': ['gene3', 'gene4']
    ... }
    >>> sc.pl.rank_genes_groups_dotplot(adata, var_names=gene_groups)
    """
    if plot_type != "dotplot":
        raise ValueError("Only 'dotplot' is currently supported")
        
    if var_names is not None and n_genes is not None:
        msg = (
            "The arguments n_genes and var_names are mutually exclusive. Please "
            "select only one."
        )
        raise ValueError(msg)

    if key is None:
        key = "rank_genes_groups"

    if groupby is None:
        groupby = str(adata.uns[key]["params"]["groupby"])
    group_names = adata.uns[key]["names"].dtype.names if groups is None else groups

    if var_names is not None:
        if isinstance(var_names, Mapping):
            # get a single list of all gene names in the dictionary
            var_names_list = functools.reduce(
                operator.iadd, [list(x) for x in var_names.values()], []
            )
        elif isinstance(var_names, str):
            var_names_list = [var_names]
        else:
            var_names_list = var_names
    else:
        # set n_genes = 10 as default when none of the options is given
        if n_genes is None:
            n_genes = 10

        # dict in which each group is the key and the n_genes are the values
        var_names = {}
        var_names_list = []
        for group in group_names:
            df = rank_genes_groups_df(
                adata,
                group,
                key=key,
                gene_symbols=gene_symbols,
                log2fc_min=min_logfoldchange,
            )

            if gene_symbols is not None:
                df["names"] = df[gene_symbols]

            genes_list = df.names[df.names.notnull()].tolist()

            if len(genes_list) == 0:
                print(f"Warning: No genes found for group {group}")
                continue
            genes_list = genes_list[n_genes:] if n_genes < 0 else genes_list[:n_genes]
            var_names[group] = genes_list
            var_names_list.extend(genes_list)

    # by default add dendrogram to plots
    kwds.setdefault("dendrogram", True)

    # Get values to plot if specified
    title = None
    values_df = None
    if values_to_plot is not None:
        values_df = _get_values_to_plot(
            adata,
            values_to_plot,
            var_names_list,
            key=key,
            gene_symbols=gene_symbols,
        )
        title = values_to_plot
        if values_to_plot == "logfoldchanges":
            title = "log fold change"
        else:
            title = values_to_plot.replace("_", " ").replace("pvals", "p-value")

    # Create the plot
    _pl = dotplot(
        adata,
        var_names,
        groupby,
        dot_color_df=values_df,
        return_fig=True,
        gene_symbols=gene_symbols,
        **kwds,
    )
    
    if title is not None and "colorbar_title" not in kwds:
        _pl.legend(colorbar_title=title)
    
    if return_fig:
        return _pl
    elif not show:
        return _pl
    return None