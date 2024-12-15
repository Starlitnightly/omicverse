```
# Line 1: Import the omicverse library as ov -- import omicverse as ov
# Line 2: Print the version of the omicverse library. -- print(f'omicverse version:{ov.__version__}')
# Line 3: Import the scanpy library as sc -- import scanpy as sc
# Line 4: Print the version of the scanpy library. -- print(f'scanpy version:{sc.__version__}')
# Line 5: Set plotting defaults for omicverse. -- ov.ov_plot_set()
# Line 11: Read 10X data into an AnnData object from the specified directory, using gene symbols as variable names, and enabling caching. -- adata = sc.read_10x_mtx(
# Line 12: This is a comment describing the directory with the .mtx file --     'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file
# Line 13: This is a comment indicating that gene symbols are used for variable names --     var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
# Line 14: This is a comment indicating that a cache file is written for faster subsequent reading --     cache=True)                              # write a cache file for faster subsequent reading
# Line 18: Perform quality control on the AnnData object, filtering cells based on mitochondrial percentage, number of UMIs, and detected genes. -- adata=ov.pp.qc(adata,
# Line 19: This is a comment describing the quality control parameters --               tresh={'mito_perc': 0.05, 'nUMIs': 500, 'detected_genes': 250})
# Line 21: Preprocess the AnnData object, including normalization and calculation of highly variable genes. -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
# Line 23: Save the original data in adata.raw and then filter the data to retain only highly variable genes. -- adata.raw = adata
# Line 24: Filter the AnnData object to keep only highly variable genes. -- adata = adata[:, adata.var.highly_variable_features]
# Line 27: Scale the expression data in adata.X. -- ov.pp.scale(adata)
# Line 30: Perform principal component analysis (PCA) on the scaled data, retaining the top 50 principal components. -- ov.pp.pca(adata,layer='scaled',n_pcs=50)
# Line 33: Construct a neighborhood graph for the AnnData object using the specified number of neighbors, PCs, and representation. -- sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50,
# Line 34: This is a comment indicating the representation to use for the graph construction. --                use_rep='scaled|original|X_pca')
# Line 37: Perform Leiden clustering on the AnnData object. -- sc.tl.leiden(adata)
# Line 40: Reduce the dimensionality of the AnnData object using MDE for visualization and store it in adata.obsm. -- adata.obsm["X_mde"] = ov.utils.mde(adata.obsm["scaled|original|X_pca"])
# Line 41: Display the AnnData object. -- adata
# Line 43: Create a pySCSA object for cell annotation using cellmarker database. -- scsa=ov.single.pySCSA(adata=adata,
# Line 44: This is a comment describing the fold change parameter. --                       foldchange=1.5,
# Line 45: This is a comment describing the p-value parameter. --                       pvalue=0.01,
# Line 46: This is a comment describing the celltype parameter. --                       celltype='normal',
# Line 47: This is a comment describing the target parameter. --                       target='cellmarker',
# Line 48: This is a comment describing the tissue parameter. --                       tissue='All',
# Line 49: This is a comment describing the model path parameter. --                       model_path='temp/pySCSA_2023_v2_plus.db'                    
# Line 51: Annotate the cells based on leiden clusters using cellmarker database. -- anno=scsa.cell_anno(clustertype='leiden',
# Line 52: This is a comment describing the cluster parameter. --                cluster='all',rank_rep=True)
# Line 54: Automatically annotate cells based on the cellmarker annotations and store in adata. -- scsa.cell_auto_anno(adata,key='scsa_celltype_cellmarker')
# Line 56: Create a pySCSA object for cell annotation using panglaodb database. -- scsa=ov.single.pySCSA(adata=adata,
# Line 57: This is a comment describing the fold change parameter. --                           foldchange=1.5,
# Line 58: This is a comment describing the p-value parameter. --                           pvalue=0.01,
# Line 59: This is a comment describing the celltype parameter. --                           celltype='normal',
# Line 60: This is a comment describing the target parameter. --                           target='panglaodb',
# Line 61: This is a comment describing the tissue parameter. --                           tissue='All',
# Line 62: This is a comment describing the model path parameter. --                           model_path='temp/pySCSA_2023_v2_plus.db'
# Line 66: Annotate the cells based on leiden clusters using panglaodb database. -- res=scsa.cell_anno(clustertype='leiden',
# Line 67: This is a comment describing the cluster parameter. --                cluster='all',rank_rep=True)
# Line 69: Print the cell annotations. -- scsa.cell_anno_print()
# Line 71: Automatically annotate cells based on the panglaodb annotations and store in adata. -- scsa.cell_auto_anno(adata,key='scsa_celltype_panglaodb')
# Line 73: Generate and display an embedding plot of cells colored by Leiden clusters, cellmarker annotations, and panglaodb annotations. -- ov.utils.embedding(adata,
# Line 74: This is a comment describing the basis of the embedding. --                    basis='X_mde',
# Line 75: This is a comment describing the colors of the plot. --                    color=['leiden','scsa_celltype_cellmarker','scsa_celltype_panglaodb'], 
# Line 76: This is a comment describing the legend location. --                    legend_loc='on data', 
# Line 77: This is a comment describing the frame and legend_fontoutline parameters. --                    frameon='small',
# Line 78: This is a comment describing the legend font outline --                    legend_fontoutline=2,
# Line 79: This is a comment describing the color palette --                    palette=ov.utils.palette()[14:],
# Line 82: Add a 'group' column to adata.obs and set it to 'A' for all cells initially. -- adata.obs['group']='A'
# Line 83: Set the 'group' column to 'B' for the first 1000 cells. -- adata.obs.loc[adata.obs.index[:1000],'group']='B'
# Line 85: Generate and display an embedding plot of cells colored by group, using red color palette. -- ov.utils.embedding(adata,
# Line 86: This is a comment describing the basis of the embedding. --                    basis='X_mde',
# Line 87: This is a comment describing the color of the plot. --                    color=['group'], 
# Line 88: This is a comment describing the frame and legend_fontoutline parameters. --                    frameon='small',legend_fontoutline=2,
# Line 89: This is a comment describing the red color palette --                    palette=ov.utils.red_color,
# Line 92: Generate and display a cell proportion plot based on cellmarker cell types and sample groups. -- ov.utils.plot_cellproportion(adata=adata,celltype_clusters='scsa_celltype_cellmarker',
# Line 93: This is a comment describing the visual clusters parameter. --                     visual_clusters='group',
# Line 94: This is a comment describing the visual_name parameter. --                     visual_name='group',figsize=(2,4))
# Line 96: Generate and display an embedding plot showing the cell type annotations, and adjust title and cell type and embedding range parameters -- ov.utils.plot_embedding_celltype(adata,figsize=None,basis='X_mde',
# Line 97: This is a comment describing the celltype_key parameter --                             celltype_key='scsa_celltype_cellmarker',
# Line 98: This is a comment describing the title parameter --                             title='            Cell type',
# Line 99: This is a comment describing the celltype_range parameter --                             celltype_range=(2,6),
# Line 100: This is a comment describing the embedding_range parameter --                             embedding_range=(4,10),)
# Line 102: Calculate the ratio of observed to expected cell proportions for each cell type in each group. -- roe=ov.utils.roe(adata,sample_key='group',cell_type_key='scsa_celltype_cellmarker')
# Line 104: Import the seaborn plotting library as sns -- import seaborn as sns
# Line 105: Import the pyplot module from matplotlib library as plt -- import matplotlib.pyplot as plt
# Line 106: Create a new figure and an axes for the heatmap with a specified figsize. -- fig, ax = plt.subplots(figsize=(2,4))
# Line 108: Copy the roe data for transformation. -- transformed_roe = roe.copy()
# Line 109: Transform the roe data based on different thresholds into a symbolic data. -- transformed_roe = transformed_roe.applymap(
# Line 110: This is a comment describing transformation of value >=2 to '+++', value >=1.5 to '++', value >=1 to '+', else to '+/-' --     lambda x: '+++' if x >= 2 else ('++' if x >= 1.5 else ('+' if x >= 1 else '+/-')))
# Line 112: Create a heatmap with the transformed Ro/e data and add symbolic annotations. -- sns.heatmap(roe, annot=transformed_roe, cmap='RdBu_r', fmt='', 
# Line 113: This is a comment describing cbar and ax parameters --             cbar=True, ax=ax,vmin=0.5,vmax=1.5,cbar_kws={'shrink':0.5})
# Line 114: Adjust the size of the xtick labels. -- plt.xticks(fontsize=12)
# Line 115: Adjust the size of the ytick labels. -- plt.yticks(fontsize=12)
# Line 117: Label the x axis as 'Group'. -- plt.xlabel('Group',fontsize=13)
# Line 118: Label the y axis as 'Cell type'. -- plt.ylabel('Cell type',fontsize=13)
# Line 119: Set the title of heatmap as 'Ro/e'. -- plt.title('Ro/e',fontsize=13)
# Line 121: This is a dictionary defining marker genes for cell types. -- res_marker_dict={
# Line 131: Compute and store a dendrogram for the leiden clusters -- sc.tl.dendrogram(adata,'leiden')
# Line 132: Create a dotplot of gene expression for the specified marker genes, grouped by Leiden cluster -- sc.pl.dotplot(adata, res_marker_dict, 'leiden', 
# Line 133: This is a comment describing dendrogram and standard_scale parameters --               dendrogram=True,standard_scale='var')
# Line 136: Create a dictionary mapping cluster ID to cell type labels. -- cluster2annotation = {
# Line 150: Annotate the AnnData object with cell types based on the cluster2annotation dictionary. -- ov.single.scanpy_cellanno_from_dict(adata,anno_dict=cluster2annotation,
# Line 151: This is a comment describing the clustertype parameter --                                        clustertype='leiden')
# Line 153: Generate and display an embedding plot with major cell types and scsa cellmarker annotations. -- ov.utils.embedding(adata,
# Line 154: This is a comment describing the basis of the embedding. --                    basis='X_mde',
# Line 155: This is a comment describing the colors of the plot. --                    color=['major_celltype','scsa_celltype_cellmarker'], 
# Line 156: This is a comment describing legend_loc, frameon and legend_fontoutline parameters. --                    legend_loc='on data', frameon='small',legend_fontoutline=2,
# Line 157: This is a comment describing the color palette --                    palette=ov.utils.palette()[14:],
# Line 160: Get a dictionary of marker genes for each cell type based on scsa_celltype_cellmarker annotations. -- marker_dict=ov.single.get_celltype_marker(adata,clustertype='scsa_celltype_cellmarker')
# Line 161: Print the keys of the marker dictionary. -- marker_dict.keys()
# Line 163: Print the marker genes for the 'B cell' cell type. -- marker_dict['B cell']
# Line 165: Get a list of tissues in the pySCSA database. -- scsa.get_model_tissue()
```
