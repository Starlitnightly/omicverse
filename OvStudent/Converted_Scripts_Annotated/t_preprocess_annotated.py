```
# Line 1:  # Line 1: Import the omicverse library as ov -- import omicverse as ov
# Line 2:  # Line 2: Import the scanpy library as sc -- import scanpy as sc
# Line 3:  # Line 3: Set the plotting style for omicverse. -- ov.ov_plot_set()
# Line 8:  # Line 8: Read 10x matrix data into an AnnData object. -- adata = sc.read_10x_mtx(
# Line 9:  # Line 9: Specify the directory containing the .mtx file. --     'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file
# Line 10:  # Line 10: Use gene symbols for variable names. --     var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
# Line 11:  # Line 11: Enable caching for faster reading. --     cache=True)                              # write a cache file for faster subsequent reading
# Line 12:  # Line 12: Display the AnnData object. -- adata
# Line 14:  # Line 14: Make variable names unique. -- adata.var_names_make_unique()
# Line 15:  # Line 15: Make observation names unique. -- adata.obs_names_make_unique()
# Line 17:  # Line 17: Perform quality control on the AnnData object using specified thresholds. -- adata=ov.pp.qc(adata,
# Line 18:  # Line 18: Set threshold parameters for mito_perc, nUMIs, and detected_genes. --               tresh={'mito_perc': 0.05, 'nUMIs': 500, 'detected_genes': 250})
# Line 19:  # Line 19: Display the AnnData object after QC. -- adata
# Line 21:  # Line 21: Store counts layer in the AnnData object. -- ov.utils.store_layers(adata,layers='counts')
# Line 22:  # Line 22: Display the AnnData object after storing the layer. -- adata
# Line 24:  # Line 24: Preprocess the AnnData object using shiftlog and pearson mode, selecting 2000 HVGs. -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
# Line 25:  # Line 25: Display the AnnData object after preprocessing. -- adata
# Line 27:  # Line 27: Store raw counts in the `raw` attribute. -- adata.raw = adata
# Line 28:  # Line 28: Filter AnnData object to keep only highly variable features. -- adata = adata[:, adata.var.highly_variable_features]
# Line 29:  # Line 29: Display the filtered AnnData object. -- adata
# Line 31:  # Line 31: Create a copy of the AnnData object named adata_counts. -- adata_counts=adata.copy()
# Line 32:  # Line 32: Retrieve counts layer from adata_counts. -- ov.utils.retrieve_layers(adata_counts,layers='counts')
# Line 33:  # Line 33: Print the maximum value of normalized adata.X. -- print('normalize adata:',adata.X.max())
# Line 34:  # Line 34: Print the maximum value of raw count adata_counts.X. -- print('raw count adata:',adata_counts.X.max())
# Line 36:  # Line 36: Display the adata_counts object. -- adata_counts
# Line 38:  # Line 38: Create a copy of the raw data as an AnnData object into adata_counts. -- adata_counts=adata.raw.to_adata().copy()
# Line 39:  # Line 39: Retrieve the counts layer in adata_counts -- ov.utils.retrieve_layers(adata_counts,layers='counts')
# Line 40:  # Line 40: Print the maximum value of normalized adata.X. -- print('normalize adata:',adata.X.max())
# Line 41:  # Line 41: Print the maximum value of raw count adata_counts.X. -- print('raw count adata:',adata_counts.X.max())
# Line 42:  # Line 42: Display the adata_counts object. -- adata_counts
# Line 44:  # Line 44: Scale the AnnData object. -- ov.pp.scale(adata)
# Line 45:  # Line 45: Display the AnnData object after scaling. -- adata
# Line 47:  # Line 47: Perform PCA on scaled layer, using 50 principal components. -- ov.pp.pca(adata,layer='scaled',n_pcs=50)
# Line 48:  # Line 48: Display the AnnData object after PCA. -- adata
# Line 50:  # Line 50: Assign the scaled pca to the X_pca embedding -- adata.obsm['X_pca']=adata.obsm['scaled|original|X_pca']
# Line 51:  # Line 51: Generate an embedding plot based on X_pca with CST3 coloring. -- ov.utils.embedding(adata,
# Line 52:  # Line 52: Set the basis to X_pca for embedding. --                   basis='X_pca',
# Line 53:  # Line 53: Set color to CST3 gene. --                   color='CST3',
# Line 54:  # Line 54: Set the frame style of the plot to small. --                   frameon='small')
# Line 56:  # Line 56: Compute neighborhood graph, using scaled PCA representation. -- sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50,
# Line 57:  # Line 57: Use the scaled PCA representation as the input for the neighborhood graph. --                use_rep='scaled|original|X_pca')
# Line 59:  # Line 59: Calculate Multidimensional Energy scaling embedding. -- adata.obsm["X_mde"] = ov.utils.mde(adata.obsm["scaled|original|X_pca"])
# Line 60:  # Line 60: Display the AnnData object with X_mde calculated. -- adata
# Line 62:  # Line 62: Generate an embedding plot based on X_mde, with CST3 coloring. -- ov.utils.embedding(adata,
# Line 63:  # Line 63: Set the basis to X_mde for the embedding plot. --                 basis='X_mde',
# Line 64:  # Line 64: Set color to CST3 gene. --                 color='CST3',
# Line 65:  # Line 65: Set the frame style to small for the plot. --                 frameon='small')
# Line 67:  # Line 67: Run UMAP dimensionality reduction. -- sc.tl.umap(adata)
# Line 69:  # Line 69: Generate an embedding plot based on X_umap, with CST3 coloring. -- ov.utils.embedding(adata,
# Line 70:  # Line 70: Set the basis to X_umap. --                 basis='X_umap',
# Line 71:  # Line 71: Set the color to CST3. --                 color='CST3',
# Line 72:  # Line 72: Set the frame style to small for the plot. --                 frameon='small')
# Line 74:  # Line 74: Run Leiden clustering. -- sc.tl.leiden(adata)
# Line 76:  # Line 76: Generate an embedding plot based on X_mde, with Leiden, CST3, and NKG7 coloring. -- ov.utils.embedding(adata,
# Line 77:  # Line 77: Set basis to X_mde. --                 basis='X_mde',
# Line 78:  # Line 78: Color by Leiden cluster, CST3 gene expression and NKG7 gene expression --                 color=['leiden', 'CST3', 'NKG7'],
# Line 79:  # Line 79: Set the frame style of the plot to small. --                 frameon='small')
# Line 81:  # Line 81: Import the matplotlib plotting library -- import matplotlib.pyplot as plt
# Line 82:  # Line 82: Create a Matplotlib figure and an axes object. -- fig,ax=plt.subplots( figsize = (4,4))
# Line 84:  # Line 84: Generate an embedding plot based on X_mde with Leiden coloring. -- ov.utils.embedding(adata,
# Line 85:  # Line 85: Set the basis to X_mde. --                 basis='X_mde',
# Line 86:  # Line 86: Color the embedding by Leiden clusters. --                 color=['leiden'],
# Line 87:  # Line 87: Do not show the plot. --                 show=False,
# Line 88:  # Line 88: Set the axis of the embedding plot. --                 ax=ax)
# Line 90:  # Line 90: Generate a convex hull plot on top of the X_mde embedding based on leiden clusters. -- ov.utils.plot_ConvexHull(adata,
# Line 91:  # Line 91: Set the basis for the convex hull plot to X_mde. --                 basis='X_mde',
# Line 92:  # Line 92: Set the cluster key to Leiden. --                 cluster_key='leiden',
# Line 93:  # Line 93: Generate a hull for cluster '0'. --                 hull_cluster='0',
# Line 94:  # Line 94: Set the axis of the convex hull plot. --                 ax=ax)
# Line 97:  # Line 97: Import patheffects module from matplotlib. -- from matplotlib import patheffects
# Line 98:  # Line 98: Import the matplotlib plotting library. -- import matplotlib.pyplot as plt
# Line 99:  # Line 99: Create a matplotlib figure and axes. -- fig, ax = plt.subplots(figsize=(4,4))
# Line 101:  # Line 101: Generate an embedding plot based on X_mde, with Leiden coloring. -- ov.utils.embedding(adata,
# Line 102:  # Line 102: Set the embedding basis to X_mde. --                   basis='X_mde',
# Line 103:  # Line 103: Color the points according to leiden cluster. --                   color=['leiden'],
# Line 104:  # Line 104: Do not show the plot, do not show a legend, do not add an outline, set the frame to small, set the legend font outline and set the axis of the plot. --                    show=False, legend_loc=None, add_outline=False, 
# Line 105:  # Line 105: Set the frame to small, set the legend font outline and set the axis of the plot. --                    frameon='small',legend_fontoutline=2,ax=ax
# Line 106:  # Line 106: close embedding function --                  )
# Line 108:  # Line 108: Generate labels for the given clusters. -- ov.utils.gen_mpl_labels(
# Line 109:  # Line 109: Use Leiden as cluster key. --     adata,
# Line 110:  # Line 110: Use Leiden clusters --     'leiden',
# Line 111:  # Line 111: Exclude the "None" cluster. --     exclude=("None",),  
# Line 112:  # Line 112: Set the embedding basis to X_mde. --     basis='X_mde',
# Line 113:  # Line 113: Set the axis of the generated label plot --     ax=ax,
# Line 114:  # Line 114: Set the label arrow props. --     adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')),
# Line 115:  # Line 115: Set the text properties for the generated labels. --     text_kwargs=dict(fontsize= 12 ,weight='bold',
# Line 116:  # Line 116: Set the path effect of the label. --                      path_effects=[patheffects.withStroke(linewidth=2, foreground='w')] ),
# Line 118:  # Line 118: Define a list of marker genes. -- marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
# Line 119:  # Line 119: Define a list of marker genes. --                 'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
# Line 120:  # Line 120: Define a list of marker genes. --                 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']
# Line 122:  # Line 122: Generate a dotplot of marker genes by leiden cluster. -- sc.pl.dotplot(adata, marker_genes, groupby='leiden',
# Line 123:  # Line 123: Standard scale the dotplot by variable. --              standard_scale='var');
# Line 125:  # Line 125: Compute a dendrogram of leiden clusters. -- sc.tl.dendrogram(adata,'leiden',use_rep='scaled|original|X_pca')
# Line 126:  # Line 126: Rank genes for each leiden cluster using t-test on the scaled PCA embeddings. -- sc.tl.rank_genes_groups(adata, 'leiden', use_rep='scaled|original|X_pca',
# Line 127:  # Line 127: Set the method, use_raw, and key_added for the ranked gene t-test analysis --                         method='t-test',use_raw=False,key_added='leiden_ttest')
# Line 128:  # Line 128: Generate a dotplot of top ranked genes by leiden cluster from t-test. -- sc.pl.rank_genes_groups_dotplot(adata,groupby='leiden',
# Line 129:  # Line 129: Set color, the key for results, the standard_scale and number of genes to display for the dot plot. --                                 cmap='Spectral_r',key='leiden_ttest',
# Line 130:  # Line 130: Set the standard scale for the dotplot by variable, display 3 genes. --                                 standard_scale='var',n_genes=3)
# Line 132:  # Line 132: Rank genes for each leiden cluster using t-test on the scaled PCA embeddings. -- sc.tl.rank_genes_groups(adata, groupby='leiden', 
# Line 133:  # Line 133: Set the method and use_rep for the ranked genes t-test analysis. --                         method='t-test',use_rep='scaled|original|X_pca',)
# Line 134:  # Line 134: Run consensus scoring of gene groups by leiden cluster -- ov.single.cosg(adata, key_added='leiden_cosg', groupby='leiden')
# Line 135:  # Line 135: Generate a dotplot of top ranked genes from cosg by leiden cluster. -- sc.pl.rank_genes_groups_dotplot(adata,groupby='leiden',
# Line 136:  # Line 136: Set the color map, key, standard scaling, and number of genes to display for the dot plot. --                                 cmap='Spectral_r',key='leiden_cosg',
# Line 137:  # Line 137: Set the standard scaling by variable and number of genes to display for the dot plot. --                                 standard_scale='var',n_genes=3)
# Line 139:  # Line 139: Create an empty dictionary to store rank genes group data. -- data_dict={}
# Line 140:  # Line 140: Iterate over each leiden category. -- for i in adata.obs['leiden'].cat.categories:
# Line 141:  # Line 141: Retrieve ranked genes for each cluster based on t-test pvalues and store it to the dictionary. --     data_dict[i]=sc.get.rank_genes_groups_df(adata, group=i, key='leiden_ttest',
# Line 142:  # Line 142: Set cutoff values for pvalue and logfoldchanges --                                             pval_cutoff=None,log2fc_min=None)
# Line 144:  # Line 144: Print the keys of the data dictionary. -- data_dict.keys()
# Line 146:  # Line 146: Display the head of the data dictionary for the last category. -- data_dict[i].head()
# Line 148:  # Line 148: Create a color dictionary using leiden categories and colors. -- type_color_dict=dict(zip(adata.obs['leiden'].cat.categories,
# Line 149:  # Line 149: Use leiden color categories for the type_color_dict. --                          adata.uns['leiden_colors']))
# Line 150:  # Line 150: Print type_color_dict -- type_color_dict
# Line 152:  # Line 152: Create a stacked volcano plot based on the ranked gene results. -- fig,axes=ov.utils.stacking_vol(data_dict,type_color_dict,
# Line 153:  # Line 153: Set the p-value threshold for significance. --             pval_threshold=0.01,
# Line 154:  # Line 154: Set the log2 fold change threshold for significance. --             log2fc_threshold=2,
# Line 155:  # Line 155: Set the figure size. --             figsize=(8,4),
# Line 156:  # Line 156: Set the color for significant genes. --             sig_color='#a51616',
# Line 157:  # Line 157: Set the color for non-significant genes. --             normal_color='#c7c7c7',
# Line 158:  # Line 158: Set the number of genes to plot. --             plot_genes_num=2,
# Line 159:  # Line 159: Set the fontsize of the plot genes. --             plot_genes_fontsize=6,
# Line 160:  # Line 160: Set the font weight for the plotted genes. --             plot_genes_weight='bold',
# Line 161:  # Line 161: close stacking vol function --             )
# Line 163:  # Line 163: Set initial y min and y max values for stacking plots. -- y_min,y_max=0,0
# Line 164:  # Line 164: Iterate over each cluster in the data dict. -- for i in data_dict.keys():
# Line 165:  # Line 165: Update y min by taking the min of current ymin and logfoldchanges minimum value. --     y_min=min(y_min,data_dict[i]['logfoldchanges'].min())
# Line 166:  # Line 166: Update y max by taking the max of current ymax and logfoldchanges maximum value. --     y_max=max(y_max,data_dict[i]['logfoldchanges'].max())
# Line 167:  # Line 167: Iterate over each leiden category. -- for i in adata.obs['leiden'].cat.categories:
# Line 168:  # Line 168: Set the y axis limits for each subplot using calculated y min and max --     axes[i].set_ylim(y_min,y_max)
# Line 169:  # Line 169: Set the suptitle for the whole plot figure. -- plt.suptitle('Stacking_vol',fontsize=12)
```