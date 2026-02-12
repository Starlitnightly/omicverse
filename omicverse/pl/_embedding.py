import collections.abc as cabc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors, colormaps, rcParams, patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap, Normalize
from typing import Union, Sequence, Optional, Mapping, Any, Tuple, Literal
from anndata import AnnData
from cycler import Cycler
from scipy.sparse import issparse

from .._registry import register_function
from ..utils._scatterplot import (
    _get_basis,
    _get_color_source_vector,
    _color_vector,
    _get_palette,
    _get_vboundnorm,
    _add_categorical_legend,
    _basis2name,
    _components_to_dimensions,
)

try:
    from scanpy.plotting._utils import ColorLike, VBound, _FontWeight, _FontSize
except ImportError:
    ColorLike = Any
    VBound = Any
    _FontWeight = Any
    _FontSize = Any


@register_function(
    aliases=["大规模嵌入可视化", "datashader_embedding", "atlas_plot", "百万细胞可视化"],
    category="pl",
    description="High-resolution embedding visualization using Datashader for large-scale datasets (millions of cells)",
    examples=[
        "# Basic large-scale UMAP visualization",
        "ov.pl.embedding_atlas(adata, basis='X_umap', color='leiden')",
        "# Continuous gene expression on large datasets",
        "ov.pl.embedding_atlas(adata, basis='X_umap', color='CD14', cmap='viridis')",
        "# Custom plot settings",
        "ov.pl.embedding_atlas(adata, basis='X_tsne', color='celltype',",
        "                       figsize=(8,8), frameon=False, plot_width=1000)",
        "# Save high-resolution plot",
        "ax = ov.pl.embedding_atlas(adata, basis='X_umap', color='batch',",
        "                            plot_width=1200, plot_height=1200, show=False)",
        "plt.savefig('atlas.png', dpi=300, bbox_inches='tight')"
    ],
    related=["utils.embedding", "pl.umap", "pl.tsne", "utils.mde"]
)
def embedding_atlas(
    adata: AnnData,
    basis: str,
    color: Union[str, Sequence[str], None] = None,
    *,
    gene_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    groups: Optional[str] = None,
    title: Union[str, Sequence[str], None] = None,
    figsize: Tuple[float, float] = (4, 4),
    ax: Optional[Axes] = None,
    cmap: Union[str, Colormap] = 'RdBu_r',
    palette: Union[str, Sequence[str], Cycler, None] = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    legend_loc: str = 'right margin',
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = 'bold',
    legend_fontoutline: Optional[int] = None,
    frameon: Optional[Union[bool, str]] = 'small',
    colorbar_loc: Optional[str] = "right",
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    plot_width: int = 800,
    plot_height: int = 800,
    spread_px: int = 0,
    how: str = 'eq_hist',
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    return_fig: Optional[bool] = None,
    ncols: int = 4,
    wspace: Optional[float] = None,
    hspace: float = 0.25,
    **kwargs,
) -> Union[Figure, Axes, None]:
    r"""Create high-resolution embedding plots using Datashader for large datasets.

    Uses Datashader to render embeddings at high resolution, suitable for datasets
    with millions of cells where standard scatter plots become ineffective.

    Arguments:
        adata: Annotated data object with embedding coordinates
        basis: Key in adata.obsm containing embedding coordinates (e.g., 'X_umap')
        color: Gene name(s) or obs column(s) to color cells by (None)
        gene_symbols: Column name in .var DataFrame for gene symbols (None)
        use_raw: Whether to use .raw attribute of adata (None)
        layer: Layer to use for coloring (None)
        groups: Restrict to a subset of groups (None)
        title: Plot title (None, uses color name)
        figsize: Figure dimensions as (width, height) ((4, 4))
        ax: Existing matplotlib axes object (None)
        cmap: Colormap for continuous values ('RdBu_r')
        palette: Colors to use for categorical variables (None)
        na_color: Color for missing values ('lightgray')
        na_in_legend: Include missing values in legend (True)
        legend_loc: Legend position ('right margin')
        legend_fontsize: Font size for legend (None)
        legend_fontweight: Font weight for legend ('bold')
        legend_fontoutline: Font outline width for legend (None)
        frameon: Frame style - False, 'small', or True ('small')
        colorbar_loc: Location of colorbar ('right')
        vmax: Maximum color scale value (None)
        vmin: Minimum color scale value (None)
        vcenter: Center color scale value (None)
        norm: Normalization for color scale (None)
        plot_width: Datashader canvas width in pixels (800)
        plot_height: Datashader canvas height in pixels (800)
        spread_px: Spread pixels for better visibility (0)
        how: Datashader color aggregation method ('eq_hist')
        show: Show the plot (None)
        save: Save the plot (None)
        return_fig: Return figure object (None)
        ncols: Number of columns for multi-panel plots (4)
        wspace: Width spacing between subplots (None)
        hspace: Height spacing between subplots (0.25)
        **kwargs: Additional arguments

    Returns:
        Matplotlib axes or figure object if show=False, otherwise None
    """
    try:
        import datashader as ds
        import datashader.transfer_functions as tf
        from datashader.colors import Sets1to3
    except ImportError:
        raise ImportError(
            "Datashader is required for embedding_atlas. "
            "Install it with: pip install datashader"
        )

    try:
        import bokeh.palettes
    except ImportError:
        raise ImportError(
            "Bokeh is required for embedding_atlas. "
            "Install it with: pip install bokeh"
        )

    # Handle default fontsize
    if legend_fontsize is None:
        legend_fontsize = 12

    # Handle use_raw default
    if use_raw is None:
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            f"Cannot use both a layer and the raw representation. "
            f"Was passed: use_raw={use_raw}, layer={layer}."
        )
    if use_raw and adata.raw is None:
        raise ValueError(
            "`use_raw` is set to True but AnnData object does not have raw."
        )

    # Convert color to list
    if color is None:
        color_list = [None]
    elif isinstance(color, str):
        color_list = [color]
    else:
        color_list = list(color)

    # Convert title to list
    if title is not None:
        if isinstance(title, str):
            title_list = [title]
        else:
            title_list = list(title)
    else:
        title_list = [None] * len(color_list)

    # Ensure title list matches color list length
    if len(title_list) < len(color_list):
        title_list.extend([None] * (len(color_list) - len(title_list)))

    # Get embedding coordinates
    embedding = _get_basis(adata, basis)

    # Handle multiple panels
    if len(color_list) > 1:
        if ax is not None:
            raise ValueError(
                "Cannot specify `ax` when plotting multiple panels."
            )

        # Create multi-panel figure
        from matplotlib import gridspec

        n_panels_x = min(ncols, len(color_list))
        n_panels_y = int(np.ceil(len(color_list) / n_panels_x))

        if wspace is None:
            wspace = 0.75 / rcParams['figure.figsize'][0] + 0.02

        fig = plt.figure(
            figsize=(
                n_panels_x * figsize[0] * (1 + wspace),
                n_panels_y * figsize[1],
            ),
        )

        left = 0.2 / n_panels_x
        bottom = 0.13 / n_panels_y
        gs = gridspec.GridSpec(
            nrows=n_panels_y,
            ncols=n_panels_x,
            left=left,
            right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
            bottom=bottom,
            top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
            hspace=hspace,
            wspace=wspace,
        )

        axs = []
        for i, (value_to_plot, plot_title) in enumerate(zip(color_list, title_list)):
            ax_i = plt.subplot(gs[i])
            axs.append(ax_i)
            _plot_single_embedding_atlas(
                adata=adata,
                basis=basis,
                embedding=embedding,
                color=value_to_plot,
                ax=ax_i,
                title=plot_title,
                cmap=cmap,
                palette=palette,
                na_color=na_color,
                na_in_legend=na_in_legend,
                legend_loc=legend_loc,
                legend_fontsize=legend_fontsize,
                legend_fontweight=legend_fontweight,
                legend_fontoutline=legend_fontoutline,
                frameon=frameon,
                colorbar_loc=colorbar_loc,
                vmax=vmax,
                vmin=vmin,
                vcenter=vcenter,
                norm=norm,
                plot_width=plot_width,
                plot_height=plot_height,
                spread_px=spread_px,
                how=how,
                gene_symbols=gene_symbols,
                use_raw=use_raw,
                layer=layer,
                groups=groups,
            )

        if show is None:
            show = True
        if show:
            plt.show()
        if save:
            if isinstance(save, str):
                fig.savefig(save, dpi=300, bbox_inches='tight')
            else:
                fig.savefig(f'{basis}_atlas.png', dpi=300, bbox_inches='tight')

        if return_fig:
            return fig
        elif not show:
            return axs
        return None

    else:
        # Single panel
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        _plot_single_embedding_atlas(
            adata=adata,
            basis=basis,
            embedding=embedding,
            color=color_list[0],
            ax=ax,
            title=title_list[0],
            cmap=cmap,
            palette=palette,
            na_color=na_color,
            na_in_legend=na_in_legend,
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight,
            legend_fontoutline=legend_fontoutline,
            frameon=frameon,
            colorbar_loc=colorbar_loc,
            vmax=vmax,
            vmin=vmin,
            vcenter=vcenter,
            norm=norm,
            plot_width=plot_width,
            plot_height=plot_height,
            spread_px=spread_px,
            how=how,
            gene_symbols=gene_symbols,
            use_raw=use_raw,
            layer=layer,
            groups=groups,
        )

        if show is None:
            show = True
        if show:
            plt.show()
        if save:
            if isinstance(save, str):
                fig.savefig(save, dpi=300, bbox_inches='tight')
            else:
                fig.savefig(f'{basis}_atlas.png', dpi=300, bbox_inches='tight')

        if return_fig:
            return fig
        elif not show:
            return ax
        return None


def _plot_single_embedding_atlas(
    adata: AnnData,
    basis: str,
    embedding: np.ndarray,
    color: Optional[str],
    ax: Axes,
    title: Optional[str],
    cmap: Union[str, Colormap],
    palette: Union[str, Sequence[str], Cycler, None],
    na_color: ColorLike,
    na_in_legend: bool,
    legend_loc: str,
    legend_fontsize: Union[int, float],
    legend_fontweight: Union[int, str],
    legend_fontoutline: Optional[int],
    frameon: Optional[Union[bool, str]],
    colorbar_loc: Optional[str],
    vmax: Union[VBound, Sequence[VBound], None],
    vmin: Union[VBound, Sequence[VBound], None],
    vcenter: Union[VBound, Sequence[VBound], None],
    norm: Union[Normalize, Sequence[Normalize], None],
    plot_width: int,
    plot_height: int,
    spread_px: int,
    how: str,
    gene_symbols: Optional[str],
    use_raw: Optional[bool],
    layer: Optional[str],
    groups: Optional[str],
):
    """Helper function to plot a single embedding atlas panel."""
    import datashader as ds
    import datashader.transfer_functions as tf
    import bokeh.palettes

    # Create Datashader canvas
    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)

    # Create DataFrame with coordinates
    df = pd.DataFrame(embedding[:, :2], columns=['x', 'y'])

    # Get color values
    if color is None:
        # Just plot points without coloring
        agg = cvs.points(df, 'x', 'y')
        img = tf.shade(agg, cmap=['lightgray'])
        categorical = False
        color_source_vector = None
    else:
        color_source_vector = _get_color_source_vector(
            adata,
            color,
            layer=layer,
            use_raw=use_raw,
            gene_symbols=gene_symbols,
            groups=groups,
        )
        color_vector, categorical = _color_vector(
            adata,
            color,
            color_source_vector,
            palette=palette,
            na_color=na_color,
        )

        df['label'] = color_source_vector

        if categorical or isinstance(color_source_vector.dtype, pd.CategoricalDtype):
            # Categorical coloring
            df['label'] = df['label'].astype('category')
            agg = cvs.points(df, 'x', 'y', ds.count_cat('label'))

            # Get color mapping
            color_map = _get_palette(adata, color, palette=palette)

            # Ensure all categories are in color_map
            # Convert to 6-digit hex (Datashader doesn't support alpha in hex)
            categories = df['label'].cat.categories
            color_key = []
            for cat in categories:
                hex_color = color_map.get(str(cat), na_color)
                # Convert to RGB if it's a hex color with alpha
                if isinstance(hex_color, str) and hex_color.startswith('#'):
                    if len(hex_color) == 9:  # #RRGGBBAA format
                        hex_color = hex_color[:7]  # Remove alpha
                    elif len(hex_color) not in [7, 4]:  # Not standard #RRGGBB or #RGB
                        # Convert via matplotlib to ensure valid format
                        from matplotlib.colors import to_hex
                        hex_color = to_hex(hex_color, keep_alpha=False)
                color_key.append(hex_color)

            img = tf.shade(
                tf.spread(agg, px=spread_px) if spread_px > 0 else agg,
                color_key=color_key,
                how=how
            )
        else:
            # Continuous coloring
            # Convert to numeric
            try:
                df['label'] = pd.to_numeric(df['label'], errors='coerce')
            except:
                pass

            agg = cvs.points(df, 'x', 'y', ds.mean('label'))

            # Handle colormap
            if isinstance(cmap, str):
                if cmap in bokeh.palettes.all_palettes.keys():
                    num = list(bokeh.palettes.all_palettes[cmap].keys())[-1]
                    cmap_colors = bokeh.palettes.all_palettes[cmap][num]
                else:
                    try:
                        mpl_cmap = colormaps.get_cmap(cmap) if matplotlib.__version__ < "3.7.0" else matplotlib.colormaps[cmap]
                        cmap_colors = [colors.rgb2hex(mpl_cmap(i/255)) for i in range(256)]
                    except:
                        cmap_colors = cmap
            else:
                cmap_colors = [colors.rgb2hex(cmap(i/255)) for i in range(256)]

            # Apply vmin/vmax/vcenter if specified
            if vmin is not None or vmax is not None or vcenter is not None:
                vmin_val, vmax_val, vcenter_val, norm_obj = _get_vboundnorm(
                    [vmin] if not isinstance(vmin, (list, tuple)) else vmin,
                    [vmax] if not isinstance(vmax, (list, tuple)) else vmax,
                    [vcenter] if not isinstance(vcenter, (list, tuple)) else vcenter,
                    [norm] if not isinstance(norm, (list, tuple)) else norm,
                    0,
                    df['label'].values
                )
                # Datashader doesn't support norm directly, but we can clip values
                if vmin_val is not None:
                    df.loc[df['label'] < vmin_val, 'label'] = vmin_val
                if vmax_val is not None:
                    df.loc[df['label'] > vmax_val, 'label'] = vmax_val

            img = tf.shade(agg, cmap=cmap_colors)

    # Display image on axis
    ax.imshow(img.to_pil(), aspect='auto', interpolation='nearest')

    # Set title
    if title is None and color is not None:
        title = color
    if title is not None:
        ax.set_title(title, fontsize=legend_fontsize + 1)

    # Handle frame style
    if frameon == False:
        ax.axis('off')
    elif frameon == 'small':
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        # Set spine positions for "small" style
        ax.spines['bottom'].set_bounds(0, plot_width * 0.2)
        ax.spines['left'].set_bounds(plot_height * 0.8, plot_height)

        # Set axis labels
        name = _basis2name(basis)
        ax.set_xlabel(f'{name}1', loc='left', fontsize=legend_fontsize)
        ax.set_ylabel(f'{name}2', loc='bottom', fontsize=legend_fontsize)

        # Adjust line width
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
    else:
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)

        name = _basis2name(basis)
        ax.set_xlabel(f'{name}1', loc='center', fontsize=legend_fontsize)
        ax.set_ylabel(f'{name}2', loc='center', fontsize=legend_fontsize)

    # Add legend for categorical data
    if color is not None and categorical and legend_loc is not None:
        if color_source_vector is not None:
            if legend_fontoutline is not None:
                path_effect = [
                    patheffects.withStroke(linewidth=legend_fontoutline, foreground='w')
                ]
            else:
                path_effect = None

            # Use the helper function from _scatterplot
            _add_categorical_legend(
                ax=ax,
                color_source_vector=color_source_vector,
                palette=_get_palette(adata, color, palette=palette),
                scatter_array=embedding[:, :2],
                legend_loc=legend_loc,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=na_color,
                na_in_legend=na_in_legend,
                multi_panel=False,
            )

    # Add colorbar for continuous data
    if color is not None and not categorical and colorbar_loc is not None:
        if frameon == 'small' or frameon == False:
            from matplotlib.ticker import MaxNLocator
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize as mpl_Normalize

            # Get position
            pos = ax.get_position()

            # Calculate colorbar dimensions
            cb_height = pos.height * 0.3
            cb_bottom = pos.y0

            # Create colorbar axis
            cax = plt.gcf().add_axes([pos.x1 + 0.02, cb_bottom, 0.02, cb_height])

            # Create ScalarMappable for colorbar
            if vmin is not None or vmax is not None:
                vmin_val = vmin[0] if isinstance(vmin, (list, tuple)) else vmin
                vmax_val = vmax[0] if isinstance(vmax, (list, tuple)) else vmax
                norm_obj = mpl_Normalize(vmin=vmin_val, vmax=vmax_val)
            else:
                norm_obj = mpl_Normalize(
                    vmin=df['label'].min(),
                    vmax=df['label'].max()
                )

            # Get matplotlib colormap
            if isinstance(cmap, str):
                mpl_cmap = colormaps.get_cmap(cmap) if matplotlib.__version__ < "3.7.0" else matplotlib.colormaps[cmap]
            else:
                mpl_cmap = cmap

            sm = ScalarMappable(norm=norm_obj, cmap=mpl_cmap)
            sm.set_array([])

            cb = plt.colorbar(sm, cax=cax, orientation="vertical")
            cb.locator = MaxNLocator(nbins=3, integer=False)
            cb.update_ticks()
        else:
            # Standard colorbar
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize as mpl_Normalize

            if vmin is not None or vmax is not None:
                vmin_val = vmin[0] if isinstance(vmin, (list, tuple)) else vmin
                vmax_val = vmax[0] if isinstance(vmax, (list, tuple)) else vmax
                norm_obj = mpl_Normalize(vmin=vmin_val, vmax=vmax_val)
            else:
                norm_obj = mpl_Normalize(
                    vmin=df['label'].min(),
                    vmax=df['label'].max()
                )

            if isinstance(cmap, str):
                mpl_cmap = colormaps.get_cmap(cmap) if matplotlib.__version__ < "3.7.0" else matplotlib.colormaps[cmap]
            else:
                mpl_cmap = cmap

            sm = ScalarMappable(norm=norm_obj, cmap=mpl_cmap)
            sm.set_array([])

            plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.08, aspect=30, location=colorbar_loc)
