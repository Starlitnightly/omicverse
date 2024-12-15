```
# Line 1:  Import the omicverse library as ov. -- import omicverse as ov
# Line 2:  Import the scanpy library as sc. -- import scanpy as sc
# Line 3:  Import the matplotlib.pyplot library as plt. -- import matplotlib.pyplot as plt
# Line 4:  Set the plotting style using omicverse utilities. -- ov.utils.ov_plot_set()
# Line 6:  Load the scRNA hematopoiesis dataset from omicverse into an AnnData object named adata. -- adata = ov.single.scRNA_hematopoiesis()
# Line 7:  Perform principal component analysis on the AnnData object. -- sc.tl.pca(adata, svd_solver='arpack', n_comps=200)
# Line 8:  Display the modified AnnData object. -- adata
# Line 10:  Initialize a pyVIA object with specified parameters for trajectory inference. -- v0 = ov.single.pyVIA(adata=adata,adata_key='X_pca',adata_ncomps=80, basis='tsne',
# Line 11:                           clusters='label',knn=30,random_seed=4,root_user=[4823],)
# Line 13:  Run the pyVIA trajectory inference. -- v0.run()
# Line 15:  Create a figure and an axes for plotting with a specified size. -- fig, ax = plt.subplots(1,1,figsize=(4,4))
# Line 16:  Generate an embedding plot with color based on the 'label' of the adata. -- sc.pl.embedding(
# Line 17:      adata,
# Line 18:     basis="tsne",
# Line 19:     color=['label'],
# Line 20:     frameon=False,
# Line 21:    ncols=1,
# Line 22:    wspace=0.5,
# Line 23:    show=False,
# Line 24:   ax=ax
# Line 25:  Save the figure as a PNG file with specified DPI and tight bounding box. -- fig.savefig('figures/via_fig1.png',dpi=300,bbox_inches = 'tight')
# Line 27:  Generate a pie chart graph with clusters colored using a red colormap and specified parameters. -- fig, ax, ax1 = v0.plot_piechart_graph(clusters='label',cmap='Reds',dpi=80,
# Line 28:                                   show_legend=False,ax_text=False,fontsize=4)
# Line 29:  Save the figure as a PNG file with specified DPI and tight bounding box. -- fig.savefig('figures/via_fig2.png',dpi=300,bbox_inches = 'tight')
# Line 32: Extract the pseudotime from pyVIA model. -- v0.get_pseudotime(v0.adata)
# Line 33: Display the modified AnnData object from pyVIA. -- v0.adata
# Line 35:  Define a list of genes for analysis. -- gene_list_magic = ['IL3RA', 'IRF8', 'GATA1', 'GATA2', 'ITGA2B', 'MPO', 'CD79B', 'SPI1', 'CD34', 'CSF1R', 'ITGAX']
# Line 36:  Plot the cluster graph for the first four genes and save as PNG. -- fig,axs=v0.plot_clustergraph(gene_list=gene_list_magic[:4],figsize=(12,3),)
# Line 37:  Save the figure as a PNG file with specified DPI and tight bounding box. -- fig.savefig('figures/via_fig2_1.png',dpi=300,bbox_inches = 'tight')
# Line 39: Plot trajectory gams based on tsne coordinates and label clusters -- fig,ax1,ax2=v0.plot_trajectory_gams(basis='tsne',clusters='label',draw_all_curves=False)
# Line 40: Save the trajectory figure as a PNG file with specified DPI and tight bounding box. -- fig.savefig('figures/via_fig3.png',dpi=300,bbox_inches = 'tight')
# Line 42:  Generate a stream plot based on tsne coordinates and label clusters. -- fig,ax=v0.plot_stream(basis='tsne',clusters='label',
# Line 43:               density_grid=0.8, scatter_size=30, scatter_alpha=0.3, linewidth=0.5)
# Line 44:  Save the figure as a PNG file with specified DPI and tight bounding box. -- fig.savefig('figures/via_fig4.png',dpi=300,bbox_inches = 'tight')
# Line 46: Generate a stream plot, color by time with other customizations. -- fig,ax=v0.plot_stream(basis='tsne',density_grid=0.8, scatter_size=30, color_scheme='time', linewidth=0.5,
# Line 47:                             min_mass = 1, cutoff_perc = 5, scatter_alpha=0.3, marker_edgewidth=0.1,
# Line 48:                             density_stream = 2, smooth_transition=1, smooth_grid=0.5)
# Line 49: Save the stream figure as a PNG file with specified DPI and tight bounding box. -- fig.savefig('figures/via_fig5.png',dpi=300,bbox_inches = 'tight')
# Line 51:  Plot the lineage probability and save as PNG. -- fig,axs=v0.plot_lineage_probability(figsize=(8,4),)
# Line 52:  Save the figure as a PNG file with specified DPI and tight bounding box. -- fig.savefig('figures/via_fig6.png',dpi=300,bbox_inches = 'tight')
# Line 54: Plot the lineage probability and save as PNG, with marked lineages. -- fig,axs=v0.plot_lineage_probability(figsize=(6,3),marker_lineages = [2,3])
# Line 55: Save the lineage probability plot as a PNG with specified DPI and tight bounding box. -- fig.savefig('figures/via_fig7.png',dpi=300,bbox_inches = 'tight')
# Line 57:  Plot the gene trend for a given list of genes and save as PNG. -- fig,axs=v0.plot_gene_trend(gene_list=gene_list_magic,figsize=(8,6),)
# Line 58:  Save the gene trend plot as a PNG with specified DPI and tight bounding box. -- fig.savefig('figures/via_fig8.png',dpi=300,bbox_inches = 'tight')
# Line 60: Plot the gene trend heatmap, highlighting lineage 2 and save as PNG. -- fig,ax=v0.plot_gene_trend_heatmap(gene_list=gene_list_magic,figsize=(4,4),
# Line 61:                          marker_lineages=[2])
# Line 62: Save the gene trend heatmap as a PNG with specified DPI and tight bounding box. -- fig.savefig('figures/via_fig9.png',dpi=300,bbox_inches = 'tight')
```
