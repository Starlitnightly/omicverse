```
# Line 1:  Imports the omicverse library, aliased as ov. -- import omicverse as ov
# Line 2:  Imports the scanpy library, aliased as sc. -- import scanpy as sc
# Line 3:  Imports the scvelo library, aliased as scv. -- import scvelo as scv
# Line 5:  Sets the plotting style for omicverse. -- ov.utils.ov_plot_set()
# Line 7:  Loads the pancreas dataset from scvelo and stores it in adata. -- adata = scv.datasets.pancreas()
# Line 8:  Displays the adata object. -- adata
# Line 10:  Finds the maximum value in the adata.X matrix. -- adata.X.max()
# Line 12:  Performs quality control on the adata object using the specified thresholds. -- adata=ov.pp.qc(adata,
# Line 13:               tresh={'mito_perc': 0.05, 'nUMIs': 500, 'detected_genes': 250})
# Line 14:  Preprocesses the adata object using shiftlog normalization and Pearson residuals, selecting 2000 high variable genes. -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
# Line 17:  Saves a copy of the original adata object to adata.raw. -- adata.raw = adata
# Line 18:  Filters the adata object to keep only highly variable genes. -- adata = adata[:, adata.var.highly_variable_features]
# Line 20:  Scales the data matrix in adata.X. -- ov.pp.scale(adata)
# Line 22:  Performs PCA dimensionality reduction using the scaled layer with 50 components. -- ov.pp.pca(adata,layer='scaled',n_pcs=50)
# Line 24:  Finds the maximum value in the adata.X matrix after scaling and PCA. -- adata.X.max()
# Line 26:  Creates a new adata object containing only cells from Alpha and Beta clusters. -- test_adata=adata[adata.obs['clusters'].isin(['Alpha','Beta'])]
# Line 27:  Displays the test_adata object. -- test_adata
# Line 29:  Performs differential expression analysis using pyDEG on the log-normalized data. -- dds=ov.bulk.pyDEG(test_adata.to_df(layer='lognorm').T)
# Line 31:  Removes duplicate indices from the dds object. -- dds.drop_duplicates_index()
# Line 32:  Prints a success message after removing duplicate indices. -- print('... drop_duplicates_index success')
# Line 34:  Creates a list of cell indices for the Alpha treatment group. -- treatment_groups=test_adata.obs[test_adata.obs['clusters']=='Alpha'].index.tolist()
# Line 35:  Creates a list of cell indices for the Beta control group. -- control_groups=test_adata.obs[test_adata.obs['clusters']=='Beta'].index.tolist()
# Line 36:  Performs differential expression analysis between treatment and control groups using a t-test. -- result=dds.deg_analysis(treatment_groups,control_groups,method='ttest')
# Line 39:  Sorts the results by q-value and displays the top entries. -- result.sort_values('qvalue').head()
# Line 41:  Sets the fold change, p-value, and log p-value thresholds for the dds object. -- dds.foldchange_set(fc_threshold=-1,
# Line 42:                    pval_threshold=0.05,
# Line 43:                    logp_max=10)
# Line 45:  Generates a volcano plot for differential expression analysis results. -- dds.plot_volcano(title='DEG Analysis',figsize=(4,4),
# Line 46:                  plot_genes_num=8,plot_genes_fontsize=12,)
# Line 48:  Generates a box plot of the expression of Irx1 and Adra2a genes for treatment and control groups. -- dds.plot_boxplot(genes=['Irx1','Adra2a'],treatment_groups=treatment_groups,
# Line 49:                 control_groups=control_groups,figsize=(2,3),fontsize=12,
# Line 50:                  legend_bbox=(2,0.55))
# Line 52:  Generates an embedding plot with cells colored by cluster, Irx1, and Adra2a expression. -- ov.utils.embedding(adata,
# Line 53:                    basis='X_umap',
# Line 54:                     frameon='small',
# Line 55:                    color=['clusters','Irx1','Adra2a'])
# Line 57:  Creates a MetaCell object for single-cell analysis. -- meta_obj=ov.single.MetaCell(adata,use_rep='scaled|original|X_pca',n_metacells=150,
# Line 58:                            use_gpu=True)
# Line 60:  Initializes the archetypes for the MetaCell object. -- meta_obj.initialize_archetypes()
# Line 62:  Trains the MetaCell object. -- meta_obj.train(min_iter=10, max_iter=50)
# Line 64:  Saves the trained MetaCell model to a file. -- meta_obj.save('seacells/model.pkl')
# Line 66:  Loads the trained MetaCell model from a file. -- meta_obj.load('seacells/model.pkl')
# Line 68:  Generates predicted cell type labels using the soft method. -- ad=meta_obj.predicted(method='soft',celltype_label='clusters',
# Line 69:                      summarize_layer='lognorm')
# Line 71:  Prints the minimum and maximum values in the predicted cell type matrix. -- ad.X.min(),ad.X.max()
# Line 73:  Imports the matplotlib.pyplot module, aliased as plt. -- import matplotlib.pyplot as plt
# Line 74:  Creates a figure and an axes object for plotting. -- fig, ax = plt.subplots(figsize=(4,4))
# Line 75:  Generates an embedding plot of meta-cells colored by cluster, with specified customizations. -- ov.utils.embedding(
# Line 76:     meta_obj.adata,
# Line 77:     basis="X_umap",
# Line 78:     color=['clusters'],
# Line 79:     frameon='small',
# Line 80:     title="Meta cells",
# Line 81:     #legend_loc='on data',
# Line 82:     legend_fontsize=14,
# Line 83:     legend_fontoutline=2,
# Line 84:     size=10,
# Line 85:     ax=ax,
# Line 86:     alpha=0.2,
# Line 87:     #legend_loc='', 
# Line 88:     add_outline=False, 
# Line 89:     #add_outline=True,
# Line 90:     outline_color='black',
# Line 91:     outline_width=1,
# Line 92:     show=False,
# Line 93:     #palette=ov.utils.blue_color[:],
# Line 94:     #legend_fontweight='normal'
# Line 95:  Plots the meta-cells with a red color. -- ov.single._metacell.plot_metacells(ax,meta_obj.adata,color='#CB3E35',
# Line 96:                                   )
# Line 98:  Creates a new adata object containing only meta-cells with Alpha and Beta cell type labels. -- test_adata=ad[ad.obs['celltype'].isin(['Alpha','Beta'])]
# Line 99:  Displays the test_adata object. -- test_adata
# Line 101: Performs differential expression analysis using pyDEG on the meta-cell data. -- dds_meta=ov.bulk.pyDEG(test_adata.to_df().T)
# Line 103: Removes duplicate indices from the dds_meta object. -- dds_meta.drop_duplicates_index()
# Line 104: Prints a success message after removing duplicate indices. -- print('... drop_duplicates_index success')
# Line 106: Creates a list of meta-cell indices for the Alpha treatment group. -- treatment_groups=test_adata.obs[test_adata.obs['celltype']=='Alpha'].index.tolist()
# Line 107: Creates a list of meta-cell indices for the Beta control group. -- control_groups=test_adata.obs[test_adata.obs['celltype']=='Beta'].index.tolist()
# Line 108: Performs differential expression analysis on meta-cells between treatment and control groups using a t-test. -- result=dds_meta.deg_analysis(treatment_groups,control_groups,method='ttest')
# Line 110: Sorts the meta-cell DEG results by q-value and displays the top entries. -- result.sort_values('qvalue').head()
# Line 112: Sets the fold change, p-value, and log p-value thresholds for the dds_meta object. -- dds_meta.foldchange_set(fc_threshold=-1,
# Line 113:                    pval_threshold=0.05,
# Line 114:                    logp_max=10)
# Line 116: Generates a volcano plot for meta-cell differential expression analysis results. -- dds_meta.plot_volcano(title='DEG Analysis',figsize=(4,4),
# Line 117:                  plot_genes_num=8,plot_genes_fontsize=12,)
# Line 119: Generates a box plot of the expression of Ctxn2 and Mnx1 genes for treatment and control groups. -- dds_meta.plot_boxplot(genes=['Ctxn2','Mnx1'],treatment_groups=treatment_groups,
# Line 120:                 control_groups=control_groups,figsize=(2,3),fontsize=12,
# Line 121:                  legend_bbox=(2,0.55))
# Line 123: Generates an embedding plot with cells colored by cluster, Ctxn2, and Mnx1 expression. -- ov.utils.embedding(adata,
# Line 124:                    basis='X_umap',
# Line 125:                     frameon='small',
# Line 126:                    color=['clusters','Ctxn2','Mnx1'])
```