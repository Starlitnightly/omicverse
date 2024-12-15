```
# Line 1:  Import the omicverse library as ov. -- import omicverse as ov
# Line 2:  Import the scvelo library as scv. -- import scvelo as scv
# Line 3:  Import the matplotlib.pyplot library as plt. -- import matplotlib.pyplot as plt
# Line 4:  Set the plotting parameters using omicverse's ov_plot_set function. -- ov.ov_plot_set()
# Line 5:  Load the dentategyrus dataset from scvelo into an AnnData object named adata. -- adata = scv.datasets.dentategyrus()
# Line 6:  Display the adata object. -- adata
# Line 7:  Apply quality control filtering to the adata object using omicverse's qc function, with specified thresholds for mitochondrial percentage, number of UMIs, and number of detected genes. -- adata=ov.pp.qc(adata,
# Line 8: Store the 'counts' layer of the adata object using omicverse's store_layers function. --               tresh={'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250},
# Line 9: --               )
# Line 10: Store the 'counts' layer of the adata object using omicverse's store_layers function. -- ov.utils.store_layers(adata,layers='counts')
# Line 11: Display the adata object. -- adata
# Line 12: Preprocess the adata object using omicverse's preprocess function, applying shiftlog and pearson normalization, selecting 2000 highly variable genes. -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',
# Line 13: --                        n_HVGs=2000)
# Line 14: Store the current state of adata into adata.raw before performing further operations. -- adata.raw = adata
# Line 15: Subset the adata object to only include highly variable genes. -- adata = adata[:, adata.var.highly_variable_features]
# Line 16: Display the adata object. -- adata
# Line 17: Scale the data in the adata object using omicverse's scale function. -- ov.pp.scale(adata)
# Line 18: Perform principal component analysis (PCA) on the scaled data in the adata object, keeping 50 PCs, using omicverse's pca function. -- ov.pp.pca(adata,layer='scaled',n_pcs=50)
# Line 19: Compute the minimum distortion embedding (MDE) using omicverse's utils.mde function, using PCA embedding result, and stores it in the obsm slot of the adata. -- adata.obsm["X_mde_pca"] = ov.utils.mde(adata.obsm["scaled|original|X_pca"])
# Line 20: Convert the raw counts stored in adata.raw into an AnnData object and assigns it back to adata. -- adata=adata.raw.to_adata()
# Line 21: Create a figure and an axes object using matplotlib for plotting. -- fig, ax = plt.subplots(figsize=(3,3))
# Line 22: Generate an embedding plot using omicverse's embedding function, with 'X_mde_pca' as basis, displaying clusters and setting plotting options. -- ov.utils.embedding(adata,
# Line 23: --                 basis='X_mde_pca',frameon='small',
# Line 24: --                 color=['clusters'],show=False,ax=ax)
# Line 25: Import the SEACells library. -- import SEACells
# Line 26: Subset the adata object to remove 'Endothelial' cells based on their cluster assignment. -- adata=adata[adata.obs['clusters']!='Endothelial']
# Line 27: Initialize a SEACells model with specified parameters such as kernel building basis, number of SEACells, and number of waypoint eigenvectors. -- model = SEACells.core.SEACells(adata, 
# Line 28: --                   build_kernel_on='scaled|original|X_pca', 
# Line 29: --                   n_SEACells=200, 
# Line 30: --                   n_waypoint_eigs=10,
# Line 31: --                   convergence_epsilon = 1e-5)
# Line 32: Construct the kernel matrix for the SEACells model. -- model.construct_kernel_matrix()
# Line 33: Store the kernel matrix in the variable M. -- M = model.kernel_matrix
# Line 34: Initialize archetypes for the SEACells model. -- # Initialize archetypes
# Line 35: Initialize archetypes for the SEACells model. -- model.initialize_archetypes()
# Line 36: Fit the SEACells model with minimum and maximum iteration limits. -- model.fit(min_iter=10, max_iter=50)
# Line 37: Enable inline plotting for matplotlib. -- # Check for convergence 
# Line 38: Enable inline plotting for matplotlib. -- %matplotlib inline
# Line 39: Plot the convergence of the SEACells model using model.plot_convergence(). -- model.plot_convergence()
# Line 40: Print the number of iterations the model has run. -- # You can force the model to run additional iterations step-wise using the .step() function
# Line 41: Print the number of iterations the model has run. -- print(f'Run for {len(model.RSS_iters)} iterations')
# Line 42: Run the model for 10 additional steps. -- for _ in range(10):
# Line 43: Run the model for 10 additional steps. --     model.step()
# Line 44: Print the updated number of iterations. -- print(f'Run for {len(model.RSS_iters)} iterations')
# Line 45: Enable inline plotting for matplotlib. -- # Check for convergence 
# Line 46: Enable inline plotting for matplotlib. -- %matplotlib inline
# Line 47: Plot the convergence of the SEACells model after additional iterations. -- model.plot_convergence()
# Line 48: Enable inline plotting for matplotlib. -- %matplotlib inline
# Line 49: Generate a 2D plot using SEACells.plot.plot_2D function, visualizing the mde_pca embedding with specified parameters, while disabling the plotting of meta cells -- SEACells.plot.plot_2D(adata, key='X_mde_pca', colour_metacells=False,
# Line 50: --                      figsize=(4,4),cell_size=20,title='Dentategyrus Metacells',
# Line 51: --                      )
# Line 52: Store the current state of adata into adata.raw for later use. -- adata.raw=adata.copy()
# Line 53: Generate a soft SEACell representation of the data by summarizing the data according to the soft SEACell matrix and celltype labels. -- SEACell_soft_ad = SEACells.core.summarize_by_soft_SEACell(adata, model.A_, 
# Line 54: --                                                           celltype_label='clusters',
# Line 55: --                                                           summarize_layer='raw', minimum_weight=0.05)
# Line 56: Display the resulting AnnData object, SEACell_soft_ad. -- SEACell_soft_ad
# Line 57: Import scanpy library as sc. -- import scanpy as sc
# Line 58: Store a copy of the SEACell_soft_ad into the raw attribute. -- SEACell_soft_ad.raw=SEACell_soft_ad.copy()
# Line 59: Calculate highly variable genes using Scanpy's function and stores the result inplace. -- sc.pp.highly_variable_genes(SEACell_soft_ad, n_top_genes=2000, inplace=True)
# Line 60: Subset the SEACell_soft_ad object to include only highly variable genes. -- SEACell_soft_ad=SEACell_soft_ad[:,SEACell_soft_ad.var.highly_variable]
# Line 61: Scale the data in SEACell_soft_ad using omicverse's scale function. -- ov.pp.scale(SEACell_soft_ad)
# Line 62: Perform PCA on the scaled data in SEACell_soft_ad, using omicverse's pca function. -- ov.pp.pca(SEACell_soft_ad,layer='scaled',n_pcs=50)
# Line 63: Compute a neighborhood graph of the SEACell_soft_ad using Scanpy's neighbor function, using the pca embedding. -- sc.pp.neighbors(SEACell_soft_ad, use_rep='scaled|original|X_pca')
# Line 64: Calculate UMAP embedding of the SEACell_soft_ad using Scanpy's umap function. -- sc.tl.umap(SEACell_soft_ad)
# Line 65: Convert the 'celltype' column in the obs attribute of the SEACell_soft_ad object to a categorical type. -- SEACell_soft_ad.obs['celltype']=SEACell_soft_ad.obs['celltype'].astype('category')
# Line 66: Reorder the categories in the 'celltype' column to match the order of categories from the original adata clusters. -- SEACell_soft_ad.obs['celltype']=SEACell_soft_ad.obs['celltype'].cat.reorder_categories(adata.obs['clusters'].cat.categories)
# Line 67: Copy the color mapping from the original adata's clusters to the celltype colors in the SEACell_soft_ad object. -- SEACell_soft_ad.uns['celltype_colors']=adata.uns['clusters_colors']
# Line 68: Import matplotlib.pyplot as plt. -- import matplotlib.pyplot as plt
# Line 69: Create a figure and an axes object using matplotlib. -- fig, ax = plt.subplots(figsize=(3,3))
# Line 70: Generate an embedding plot using omicverse's embedding function, visualizing the UMAP embedding and coloring by celltype. -- ov.utils.embedding(SEACell_soft_ad,
# Line 71: --                    basis='X_umap',
# Line 72: --                    color=["celltype"],
# Line 73: --                    title='Meta Celltype',
# Line 74: --                    frameon='small',
# Line 75: --                    legend_fontsize=12,
# Line 76: --                    #palette=ov.utils.palette()[11:],
# Line 77: --                    ax=ax,
# Line 78: --                    show=False)
# Line 79: Initialize a pyVIA object using omicverse's single.pyVIA function with specified parameters for trajectory inference. -- v0 = ov.single.pyVIA(adata=SEACell_soft_ad,adata_key='scaled|original|X_pca',
# Line 80: --                          adata_ncomps=50, basis='X_umap',
# Line 81: --                          clusters='celltype',knn=10, root_user=['nIPC','Neuroblast'],
# Line 82: --                          dataset='group', 
# Line 83: --                          random_seed=112,is_coarse=True, 
# Line 84: --                          preserve_disconnected=True,
# Line 85: --                          piegraph_arrow_head_width=0.05,piegraph_edgeweight_scalingfactor=2.5,
# Line 86: --                          gene_matrix=SEACell_soft_ad.X,velo_weight=0.5,
# Line 87: --                          edgebundle_pruning_twice=False, edgebundle_pruning=0.15, 
# Line 88: --                          jac_std_global=0.05,too_big_factor=0.05,
# Line 89: --                          cluster_graph_pruning_std=1,
# Line 90: --                          time_series=False,
# Line 91: --                         )
# Line 92: Run the pyVIA trajectory inference. -- v0.run()
# Line 93: Calculate and store pseudotime in the SEACell_soft_ad object using the pyVIA results. -- v0.get_pseudotime(SEACell_soft_ad)
# Line 94: Import matplotlib.pyplot as plt. -- #v0.get_pseudotime(SEACell_soft_ad)
# Line 95: Import matplotlib.pyplot as plt. -- import matplotlib.pyplot as plt
# Line 96: Create a figure and an axes object using matplotlib for plotting. -- fig, ax = plt.subplots(figsize=(3,3))
# Line 97: Generate an embedding plot using omicverse's embedding function, visualizing the UMAP embedding and coloring by pseudotime. -- ov.utils.embedding(SEACell_soft_ad,
# Line 98: --                    basis='X_umap',
# Line 99: --                    color=["pt_via"],
# Line 100: --                    title='Pseudotime',
# Line 101: --                    frameon='small',
# Line 102: --                    cmap='Reds',
# Line 103: --                    #size=40,
# Line 104: --                    legend_fontsize=12,
# Line 105: --                    #palette=ov.utils.palette()[11:],
# Line 106: --                    ax=ax,
# Line 107: --                    show=False)
# Line 108: Write the SEACell_soft_ad object to an h5ad file with gzip compression. -- SEACell_soft_ad.write_h5ad('data/tutorial_meta_den.h5ad',compression='gzip')
# Line 109: Read the h5ad file into the SEACell_soft_ad object using omicverse's utils.read function. -- SEACell_soft_ad=ov.utils.read('data/tutorial_meta_den.h5ad')
# Line 110: Initialize a cellfategenie object using omicverse's single.cellfategenie function with pseudotime. -- cfg_obj=ov.single.cellfategenie(SEACell_soft_ad,pseudotime='pt_via')
# Line 111: Initialize the cellfategenie model. -- cfg_obj.model_init()
# Line 112: Run the ATR filtering method to filter the data with specified stop and flux parameters. -- cfg_obj.ATR(stop=500,flux=0.01)
# Line 113: Generate and display the filtering plot using the cellfategenie object. -- fig,ax=cfg_obj.plot_filtering(color='#5ca8dc')
# Line 114: Add a title to the filtering plot. -- ax.set_title('Dentategyrus Metacells\nCellFateGenie')
# Line 115: Fit the cellfategenie model. -- res=cfg_obj.model_fit()
# Line 116: Plot the gene fitting curves using the raw gene expression for each cell type. -- cfg_obj.plot_color_fitting(type='raw',cluster_key='celltype')
# Line 117: Plot the gene fitting curves using the filtered gene expression for each cell type. -- cfg_obj.plot_color_fitting(type='filter',cluster_key='celltype')
# Line 118: Calculate the Kendall Tau correlation for each gene after filtering and return results. -- kt_filter=cfg_obj.kendalltau_filter()
# Line 119: Display the top few rows of the Kendall Tau filtering results. -- kt_filter.head()
# Line 120: Select the variable names (genes) whose p-value is less than average p-value in the kendall tau filtered table. -- var_name=kt_filter.loc[kt_filter['pvalue']<kt_filter['pvalue'].mean()].index.tolist()
# Line 121: Initialize a gene trends object using omicverse's single.gene_trends function with specified pseudotime and variable names. -- gt_obj=ov.single.gene_trends(SEACell_soft_ad,'pt_via',var_name)
# Line 122: Calculate the trends of the selected genes along the pseudotime, smoothing the curve using a convolution window of length 10. -- gt_obj.calculate(n_convolve=10)
# Line 123: Print the number of selected variable names. -- print(f"Dimension: {len(var_name)}")
# Line 124: Generate and display a trend plot of the selected genes, coloring with a specific color from omicverse's utils, setting the title, and fontsize. -- fig,ax=gt_obj.plot_trend(color=ov.utils.blue_color[3])
# Line 125: Set the title of the trend plot. -- ax.set_title(f'Dentategyrus meta\nCellfategenie',fontsize=13)
# Line 126: Generate a heatmap using omicverse's plot_heatmap function, visualizing the selected genes ordered by pseudotime, and colored by cell type. -- g=ov.utils.plot_heatmap(SEACell_soft_ad,var_names=var_name,
# Line 127: --                   sortby='pt_via',col_color='celltype',
# Line 128: --                  n_convolve=10,figsize=(1,6),show=False)
# Line 129: Set the size of the heatmap figure. -- g.fig.set_size_inches(2, 6)
# Line 130: Add a title to the heatmap figure with specified alignment and fontsize. -- g.fig.suptitle('CellFateGenie',x=0.25,y=0.83,
# Line 131: --                horizontalalignment='left',fontsize=12,fontweight='bold')
# Line 132: Set the font size of the y-axis labels on the heatmap. -- g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(),fontsize=12)
# Line 133: Display the heatmap. -- plt.show()
# Line 134: Calculate and add the border cell position, based on the pseudotime and cell type, into SEACell_soft_ad object. -- gt_obj.cal_border_cell(SEACell_soft_ad,'pt_via','celltype')
# Line 135: Get genes with border expression between different cell types from the pseudotime using multi-border approach. -- bordgene_dict=gt_obj.get_multi_border_gene(SEACell_soft_ad,'celltype',
# Line 136: --                                           threshold=0.5)
# Line 137: Get genes with border expression between 'Granule immature' and 'Granule mature' cells with specified threshold. -- gt_obj.get_border_gene(SEACell_soft_ad,'celltype','Granule immature','Granule mature',
# Line 138: --                       threshold=0.5)
# Line 139: Get specific border genes between 'Granule immature' and 'Granule mature' cells. -- gt_obj.get_special_border_gene(SEACell_soft_ad,'celltype','Granule immature','Granule mature')
# Line 140: Import matplotlib.pyplot. -- import matplotlib.pyplot as plt
# Line 141: Generate a heatmap using omicverse's plot_heatmap function to display the border genes identified, sorting cells by pseudotime, and coloring by cell type. -- g=ov.utils.plot_heatmap(SEACell_soft_ad,
# Line 142: --                         var_names=gt_obj.get_border_gene(SEACell_soft_ad,'celltype','Granule immature','Granule mature'),
# Line 143: --                   sortby='pt_via',col_color='celltype',yticklabels=True,
# Line 144: --                  n_convolve=10,figsize=(1,6),show=False)
# Line 145: Set the figure size of the heatmap plot. -- g.fig.set_size_inches(2, 4)
# Line 146: Set the font size of the y-axis labels in the heatmap plot. -- g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(),fontsize=12)
# Line 147: Display the heatmap plot. -- plt.show()
# Line 148: Get special kernel genes for 'Granule immature' cell type, based on the pseudotime. -- gt_obj.get_special_kernel_gene(SEACell_soft_ad,'celltype','Granule immature')
# Line 149: Get kernel genes for 'Granule immature' cell type based on the pseudotime, with specified threshold and number of genes. -- gt_obj.get_kernel_gene(SEACell_soft_ad,
# Line 150: --                        'celltype','Granule immature',
# Line 151: --                        threshold=0.3,
# Line 152: --                       num_gene=10)
# Line 153: Import matplotlib.pyplot. -- import matplotlib.pyplot as plt
# Line 154: Generate a heatmap using omicverse's plot_heatmap function, displaying the kernel genes for 'Granule immature' cells, sorting by pseudotime, and coloring by cell type. -- g=ov.utils.plot_heatmap(SEACell_soft_ad,
# Line 155: --                         var_names=gt_obj.get_kernel_gene(SEACell_soft_ad,
# Line 156: --                        'celltype','Granule immature',
# Line 157: --                        threshold=0.3,
# Line 158: --                       num_gene=10),
# Line 159: --                   sortby='pt_via',col_color='celltype',yticklabels=True,
# Line 160: --                  n_convolve=10,figsize=(1,6),show=False)
# Line 161: Set the size of the heatmap figure. -- g.fig.set_size_inches(2, 4)
# Line 162: Set the font size of the y-axis labels in the heatmap. -- g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(),fontsize=12)
# Line 163: Display the heatmap. -- plt.show()
```
