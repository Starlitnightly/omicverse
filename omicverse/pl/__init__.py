r"""
Plotting utilities for omics data visualization.

This module provides comprehensive plotting functions for visualizing various types
of omics data including single-cell, bulk, spatial transcriptomics, multi-omics,
and network data. It offers publication-ready plots with customizable aesthetics.

Visualization categories:
    Single-cell plots: UMAP, t-SNE, PCA, violin plots, dot plots
    Bulk data plots: Heatmaps, volcano plots, MA plots, PCA plots
    Spatial plots: Spatial feature plots, domain visualization
    Multi-omics plots: Factor plots, integration visualizations
    Network plots: Protein-protein interactions, pathway networks
    General plots: Heatmaps, scatter plots, bar plots
    
Key modules:
    _single: Single-cell specific plotting functions
    _bulk: Bulk RNA-seq plotting functions  
    _space: Spatial transcriptomics visualizations
    _multi: Multi-omics integration plots
    _heatmap: Heatmap and clustering visualizations
    _general: General-purpose plotting utilities
    _palette: Color palette and aesthetic functions
    _cpdb: Cell-cell communication network plots
    _flowsig: Flow cytometry-style visualizations
    _embedding: Dimensionality reduction visualizations
    _density: Density and distribution plots

Features:
    - Publication-ready figures with customizable aesthetics
    - Integration with matplotlib and seaborn
    - Support for interactive plots
    - Consistent color schemes and themes
    - Export capabilities for various formats
    - Integration with AnnData objects

Examples:
    >>> import omicverse as ov
    >>> # Single-cell visualization
    >>> ov.pl.embedding(adata, basis='umap', color='cell_type')
    >>> ov.pl.violin(adata, keys=['CD3D', 'CD8A'], groupby='cell_type')
    >>> 
    >>> # Bulk data visualization  
    >>> ov.pl.volcano(deg_results, pval_threshold=0.05)
    >>> ov.pl.heatmap(adata, var_names=marker_genes)
    >>> 
    >>> # Spatial visualization
    >>> ov.pl.spatial(adata, color='total_counts')
    >>> ov.pl.spatial_domains(adata, color='domain')
"""
from ._palette import (
    ForbiddenCity,
    Forbidden_Cmap,
    Forbiddencity,
    blue_color,
    cet_g_bw,
    colormaps_palette,
    earth_palette,
    get_forbidden,
    green_color,
    optim_palette,
    orange_color,
    palette_112,
    palette_28,
    palette_56,
    pastel_palette,
    purple_color,
    red_color,
    sc_color,
    vibrant_palette,
)
from ._single import (
    ConvexHull,
    add_arrow,
    bardotplot,
    cellproportion,
    cellstackarea,
    contour,
    dotplot_doublegroup,
    embedding,
    embedding_adjust,
    embedding_celltype,
    embedding_density,
    mde,
    pca,
    plot_boxplots,
    single_group_boxplot,
    tsne,
    umap,
    violin_box,
    violin_old,
)
from ._general import (
    add_palue,
    create_custom_colormap,
    create_transparent_gradient_colormap,
)
from ._heatmap import (
    check_pycomplexheatmap,
    complexheatmap,
    global_imports,
    marker_heatmap,
)
# Import pycomplexheatmap_install as a module-level variable, not a function
from ._heatmap import pycomplexheatmap_install
from ._multi import embedding_multi
from ._bulk import boxplot, plot_grouped_fractions, venn, volcano
from ._space import (
    add_pie2spatial,
    add_pie_charts_to_spatial,
    create_colormap,
    get_rgb_function,
    html_to_rgb,
    plot_spatial,
    plot_spatial_general,
    rgb_to_ryb,
    ryb_to_rgb,
    spatial_value,
)
from ._cpdb import (
    cpdb_chord,
    cpdb_group_heatmap,
    cpdb_heatmap,
    cpdb_interacting_heatmap,
    cpdb_interacting_network,
    cpdb_network,
    curved_graph,
    curved_line,
    plot_curve_network,
)
from ._flowsig import curved_graph, curved_line, plot_curve_network, plot_flowsig_network
from ._embedding import embedding_atlas
from ._density import add_density_contour, calculate_gene_density
from ._cpdbviz import CellChatViz
from ._dotplot import dotplot, rank_genes_groups_dotplot, rank_genes_groups_df
from ._spatial import spatial_segment,spatial_segment_overlay
from ._violin import violin
from ._animation_lines import (
    Streamlines,
    add_streamplot,
    animate_streamplot,
    compute_velocity_on_grid,
    nan_helper,
)

# Explicit public exports for stable, non-wildcard imports
__all__ = [
    # @ _palette
    "ForbiddenCity",
    "Forbidden_Cmap",
    "Forbiddencity",
    "blue_color",
    "cet_g_bw",
    "colormaps_palette",
    "earth_palette",
    "get_forbidden",
    "green_color",
    "optim_palette",
    "orange_color",
    "palette_112",
    "palette_28",
    "palette_56",
    "pastel_palette",
    "purple_color",
    "red_color",
    "sc_color",
    "vibrant_palette",
    # @ _single
    "ConvexHull",
    "add_arrow",
    "bardotplot",
    "cellproportion",
    "cellstackarea",
    "contour",
    "dotplot_doublegroup",
    "embedding",
    "embedding_adjust",
    "embedding_celltype",
    "embedding_density",
    "mde",
    "pca",
    "plot_boxplots",
    "single_group_boxplot",
    "tsne",
    "umap",
    "violin_box",
    "violin_old",
    # @ _general
    "add_palue",
    "create_custom_colormap",
    "create_transparent_gradient_colormap",
    # @ _heatmap
    "check_pycomplexheatmap",
    "complexheatmap",
    "global_imports",
    "marker_heatmap",
    "pycomplexheatmap_install",
    # @ _multi
    "embedding_multi",
    # @ _bulk
    "boxplot",
    "plot_grouped_fractions",
    "venn",
    "volcano",
    # @ _space
    "add_pie2spatial",
    "add_pie_charts_to_spatial",
    "create_colormap",
    "get_rgb_function",
    "html_to_rgb",
    "plot_spatial",
    "plot_spatial_general",
    "rgb_to_ryb",
    "ryb_to_rgb",
    "spatial_value",
    # @ _cpdb
    "cpdb_chord",
    "cpdb_group_heatmap",
    "cpdb_heatmap",
    "cpdb_interacting_heatmap",
    "cpdb_interacting_network",
    "cpdb_network",
    "curved_graph",
    "curved_line",
    "plot_curve_network",
    # @ _flowsig
    "plot_flowsig_network",
    # @ _embedding
    "embedding_atlas",
    # @ _density
    "add_density_contour",
    "calculate_gene_density",
    # @ _cpdbviz
    "CellChatViz",
    # @ _dotplot
    "dotplot",
    "rank_genes_groups_dotplot",
    "rank_genes_groups_df",
    # @ _spatial
    "spatial_segment",
    "spatial_segment_overlay",
    # @ _violin
    "violin",
    # @ _animation_lines
    "Streamlines",
    "add_streamplot",
    "animate_streamplot",
    "compute_velocity_on_grid",
    "nan_helper",
]

