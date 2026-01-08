import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list

# Optional marsilea import for enhanced visualization
try:
    import marsilea as ma
    import marsilea.plotter as mp
    from matplotlib.colors import Normalize
    MARSILEA_AVAILABLE = True
except ImportError:
    MARSILEA_AVAILABLE = False


def local_correlation_plot(
            local_correlation_z, modules, linkage,
            mod_cmap='tab10', vmin=-8, vmax=8,
            z_cmap='RdBu_r', yticklabels=False
):

    row_colors = None
    colors = list(plt.get_cmap(mod_cmap).colors)
    module_colors = {i: colors[(i-1) % len(colors)] for i in modules.unique()}
    module_colors[-1] = '#ffffff'

    row_colors1 = pd.Series(
        [module_colors[i] for i in modules],
        index=local_correlation_z.index,
    )

    row_colors = pd.DataFrame({
        "Modules": row_colors1,
    })

    cm = sns.clustermap(
        local_correlation_z,
        row_linkage=linkage,
        col_linkage=linkage,
        vmin=vmin,
        vmax=vmax,
        cmap=z_cmap,
        xticklabels=False,
        yticklabels=yticklabels,
        row_colors=row_colors,
        rasterized=True,
    )

    fig = plt.gcf()
    plt.sca(cm.ax_heatmap)
    plt.ylabel("")
    plt.xlabel("")

    cm.ax_row_dendrogram.remove()

    # Add 'module X' annotations
    ii = leaves_list(linkage)

    mod_reordered = modules.iloc[ii]

    mod_map = {}
    y = np.arange(modules.size)

    for x in mod_reordered.unique():
        if x == -1:
            continue

        mod_map[x] = y[mod_reordered == x].mean()

    plt.sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        plt.text(-.5, y=mod_y, s="Module {}".format(mod),
                 horizontalalignment='right',
                 verticalalignment='center')
    plt.xticks([])

    # Find the colorbar 'child' and modify
    min_delta = 1e99
    min_aa = None
    for aa in fig.get_children():
        try:
            bbox = aa.get_position()
            delta = (0-bbox.xmin)**2 + (1-bbox.ymax)**2
            if delta < min_delta:
                delta = min_delta
                min_aa = aa
        except AttributeError:
            pass

    min_aa.set_ylabel('Z-Scores')
    min_aa.yaxis.set_label_position("left")

    return cm


def local_correlation_plot_marsilea(
            local_correlation_z, modules, linkage,
            mod_cmap='tab10', vmin=-8, vmax=8,
            z_cmap='RdBu_r', width=10, height=10,
            font_size=10, add_dendrogram=True,
            add_module_colors=True, add_module_labels=True,
            show_values=False, value_fontsize=6,
            title="Local Gene Correlations"
):
    """
    Create an enhanced local correlation plot using Marsilea (optional visualization)

    Parameters
    ----------
    local_correlation_z : pd.DataFrame
        Z-scored local correlation matrix
    modules : pd.Series
        Module assignments for genes
    linkage : ndarray
        Linkage matrix from hierarchical clustering
    mod_cmap : str
        Colormap for module colors
    vmin : float
        Minimum value for correlation Z-scores
    vmax : float
        Maximum value for correlation Z-scores
    z_cmap : str
        Colormap for correlation heatmap
    width : float
        Figure width
    height : float
        Figure height
    font_size : int
        Font size for labels
    add_dendrogram : bool
        Whether to add dendrogram
    add_module_colors : bool
        Whether to add module color bar
    add_module_labels : bool
        Whether to add module labels
    show_values : bool
        Whether to show correlation values in cells
    value_fontsize : int
        Font size for cell values
    title : str
        Plot title

    Returns
    -------
    h : marsilea plot object
        The marsilea heatmap object
    """
    if not MARSILEA_AVAILABLE:
        raise ImportError(
            "marsilea package is not available. "
            "Please install it with: pip install marsilea\n"
            "Or use the default seaborn visualization by setting use_marsilea=False"
        )

    # Prepare data
    ii = leaves_list(linkage)
    data_reordered = local_correlation_z.iloc[ii, ii]
    modules_reordered = modules.iloc[ii]

    # Create module colors
    colors = list(plt.get_cmap(mod_cmap).colors)
    module_colors_dict = {i: colors[(i-1) % len(colors)] for i in modules.unique()}
    module_colors_dict[-1] = '#ffffff'

    row_colors = [module_colors_dict[i] for i in modules_reordered]
    col_colors = row_colors.copy()

    # Create marsilea heatmap
    h = ma.Heatmap(
        data_reordered,
        cmap=z_cmap,
        label="Z-Score",
        width=width * 0.7,
        height=height * 0.7,
        linewidth=0,
        vmin=vmin,
        vmax=vmax
    )

    # Add values if requested
    if show_values:
        try:
            text_matrix = data_reordered.values
            text_array = np.array([[f"{val:.1f}" if not np.isnan(val) else ""
                                   for val in row] for row in text_matrix])
            h.add_layer(ma.plotter.TextMesh(text_array, fontsize=value_fontsize,
                                           color="black", alpha=0.7))
        except Exception as e:
            print(f"Warning: Failed to add text values: {e}")

    # Add module color bars
    if add_module_colors:
        try:
            # Top color bar
            h.add_top(
                ma.plotter.Colors(
                    modules_reordered.values,
                    palette=row_colors
                ),
                size=0.1,
                pad=0.01
            )

            # Left color bar
            h.add_left(
                ma.plotter.Colors(
                    modules_reordered.values,
                    palette=row_colors
                ),
                size=0.1,
                pad=0.01
            )
        except Exception as e:
            print(f"Warning: Failed to add module colors: {e}")

    # Add dendrograms
    if add_dendrogram:
        try:
            # Use the linkage for clustering
            h.add_dendrogram("left", linkage=linkage, colors="#2ECC71")
            h.add_dendrogram("top", linkage=linkage, colors="#9B59B6")
        except Exception as e:
            print(f"Warning: Failed to add dendrograms: {e}")

    # Add module labels on the left
    if add_module_labels:
        try:
            # Create module labels
            mod_map = {}
            y_positions = np.arange(len(modules_reordered))

            for x in modules_reordered.unique():
                if x == -1:
                    continue
                mod_map[x] = y_positions[modules_reordered == x].mean()

            # Create label text
            module_label_list = [""] * len(modules_reordered)
            for mod, mod_y_pos in mod_map.items():
                idx = int(mod_y_pos)
                if 0 <= idx < len(module_label_list):
                    module_label_list[idx] = f"Module {mod}"

            h.add_left(
                ma.plotter.Labels(
                    module_label_list,
                    rotation=0,
                    fontsize=font_size,
                    align="right"
                ),
                size=0.4,
                pad=0.05
            )
        except Exception as e:
            print(f"Warning: Failed to add module labels: {e}")

    # Add legend for modules
    try:
        from matplotlib.patches import Patch
        unique_modules = [m for m in modules.unique() if m != -1]
        legend_elements = [
            Patch(facecolor=module_colors_dict[m], label=f'Module {m}')
            for m in sorted(unique_modules)
        ]

        # Try new API first, fallback to old API
        try:
            h.add_legends(legend_elements, stack_by="row", align_stacks="right")
        except TypeError:
            # Older marsilea version
            h.add_legends(
                side="right",
                stack_by="row",
                frameon=True,
                fontsize=font_size-1,
                title="Modules"
            )
    except Exception as e:
        # Silently skip legend if it fails
        pass

    # Render the plot
    try:
        h.render()
        if title:
            plt.suptitle(title, fontsize=font_size+4, fontweight='bold', y=0.98)
    except Exception as e:
        print(f"Warning: Failed to render plot: {e}")

    return h
