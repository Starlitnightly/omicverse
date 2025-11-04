---
name: omicverse-visualization-for-bulk-color-systems-and-single-cell-d
title: OmicVerse visualization for bulk, color systems, and single-cell data
description: Guide users through OmicVerse plotting utilities showcased in the bulk, color system, and single-cell visualization tutorials, including venn/volcano charts, palette selection, and advanced embedding layouts.
---

# OmicVerse visualization for bulk, color systems, and single-cell data

## Overview
Leverage this skill when a user wants help recreating or adapting plots from the OmicVerse plotting tutorials:
- [`t_visualize_bulk.ipynb`](../../omicverse_guide/docs/Tutorials-plotting/t_visualize_bulk.ipynb)
- [`t_visualize_colorsystem.ipynb`](../../omicverse_guide/docs/Tutorials-plotting/t_visualize_colorsystem.ipynb)
- [`t_visualize_single.ipynb`](../../omicverse_guide/docs/Tutorials-plotting/t_visualize_single.ipynb)

It covers how to configure OmicVerse's plotting style, choose colors from the Forbidden City palette, and generate bulk as well as single-cell specific figures.

## Instructions
1. **Set up the plotting environment**
   - Import `omicverse as ov`, `matplotlib.pyplot as plt`, and other libraries required by the user's request (`pandas`, `seaborn`, `scanpy`, etc.).
   - Call `ov.ov_plot_set()` (or `ov.plot_set()` depending on the installed version) to apply OmicVerse's default styling before generating figures.
   - Load example data via `ov.read(...)`/`ov.pp.preprocess(...)` or instruct users to supply their own AnnData/CSV files.
2. **Bulk RNA-seq visuals (`t_visualize_bulk`)**
   - Use `ov.pl.venn(sets=..., palette=...)` to display overlaps among DEG lists (no more than 4 groups). Encourage setting `sets` as a dictionary of set names → gene lists.
   - For volcano plots, load the DEG table (`result = ov.read('...csv')`) and call `ov.pl.volcano(result, pval_name='qvalue', fc_name='log2FoldChange', ...)`. Explain optional keyword arguments such as `sig_pvalue`, `sig_fc`, `palette`, and label formatting.
   - To compare group distributions with box plots, gather long-form data (e.g., from `seaborn.load_dataset('tips')`) and invoke `ov.pl.boxplot(data, x_value=..., y_value=..., hue=..., ax=ax, palette=...)`. Mention how to adjust figure size, legend placement, and significance annotations.
3. **Color management (`t_visualize_colorsystem`)**
   - Introduce the color book via `fb = ov.pl.ForbiddenCity()` and demonstrate `fb.get_color(name='凝夜紫')` for specific hues.
   - Show how to pull predefined palettes (`ov.pl.green_color`, `ov.pl.red_color`, etc.) and build dicts mapping cell types/groups to color hex codes.
   - For segmented gradients, combine colors and call `ov.pl.get_cmap_seg(colors, name='custom')`, then pass the colormap into Matplotlib/Scanpy plotting functions.
   - Highlight using these palettes in embeddings: `ov.pl.embedding(adata, basis='X_umap', color='clusters', palette=color_dict, ax=ax)`.
4. **Single-cell visualizations (`t_visualize_single`)**
   - Remind users to preprocess AnnData if needed (`adata = ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)`).
   - **IMPORTANT - Data validation**: Before plotting, always verify that required data exists:
     ```python
     # Before plotting by clustering or other categorical variable
     color_col = 'leiden'  # or 'clusters', 'celltype', etc.
     if color_col not in adata.obs.columns:
         raise ValueError(f"Column '{color_col}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")

     # Before plotting embeddings
     basis = 'X_umap'  # or 'X_pca', 'X_tsne', etc.
     if basis not in adata.obsm.keys():
         raise ValueError(f"Embedding '{basis}' not found in adata.obsm. Available embeddings: {list(adata.obsm.keys())}")
     ```
   - For palette optimization, use `ov.pl.optim_palette(adata, basis='X_umap', colors='clusters')` to auto-generate color schemes when categories clash.
   - Reproduce stacked proportions with `ov.pl.cellproportion(adata, groupby='clusters', celltype_clusters='celltype', ax=ax)` and transform into stacked area charts by setting `kind='area'`.
   - Showcase compound embedding utilities:
     - `ov.pl.embedding_celltype` to place counts/proportions alongside UMAPs.
     - `ov.pl.ConvexHull` or `ov.pl.contour` for highlighting regions of interest.
     - `ov.pl.embedding_adjust` to reposition legends automatically.
     - `ov.pl.embedding_density` for density overlays, controlling smoothness with `adjust`.
   - For spatial gene density, describe the workflow: `ov.pl.calculate_gene_density(adata, genes=[...], basis='spatial')`, then overlay with `ov.pl.embedding(..., layer='gene_density', cmap='...')`.
   - Cover additional charts like `ov.pl.single_group_boxplot`, `ov.pl.bardotplot`, `ov.pl.dotplot`, and `ov.pl.marker_heatmap`, emphasizing input formats (long-form DataFrame vs. AnnData with `.obs` annotations) and optional helpers such as `ov.pl.add_palue` for manual p-value annotations.
5. **Finishing touches and exports**
   - Encourage adding titles, axis labels, and `fig.tight_layout()` to prevent clipping.
   - Suggest saving figures with `fig.savefig('plot.png', dpi=300, bbox_inches='tight')` and documenting color mappings for reproducibility.
   - Troubleshoot common issues:
     - **Missing AnnData keys**: Always validate `adata.obs` columns and `adata.obsm` embeddings exist before plotting
     - **Palette names not found**: Verify color dictionaries match actual category values
     - **Matplotlib font rendering**: When using Chinese characters, ensure appropriate fonts are installed
     - **"Could not find X in adata.obs"**: Check that clustering or annotation has been performed before trying to visualize results. Use defensive checks to compute missing prerequisites on-the-fly.

## Examples
- "Plot a three-set Venn diagram of overlapping DEG lists and reuse Forbidden City colors for consistency."
- "Load the dentate gyrus AnnData, color clusters with `fb.get_color` selections, and render an embedding with adjusted legend placement."
- "Generate single-cell proportion bar/area plots plus gene-density overlays using OmicVerse helper functions."

## References
- Bulk tutorial: [`t_visualize_bulk.ipynb`](../../omicverse_guide/docs/Tutorials-plotting/t_visualize_bulk.ipynb)
- Color system tutorial: [`t_visualize_colorsystem.ipynb`](../../omicverse_guide/docs/Tutorials-plotting/t_visualize_colorsystem.ipynb)
- Single-cell tutorial: [`t_visualize_single.ipynb`](../../omicverse_guide/docs/Tutorials-plotting/t_visualize_single.ipynb)
- Quick reference snippets: [`reference.md`](reference.md)
