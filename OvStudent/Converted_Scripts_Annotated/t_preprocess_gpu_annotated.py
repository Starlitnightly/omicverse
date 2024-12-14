```python
# Line 1:  import omicverse as ov -- import omicverse as ov
# Line 2:  import scanpy as sc -- import scanpy as sc
# Line 3:  ov.plot_set() -- ov.plot_set()
# Line 4:  ov.settings.gpu_init() -- ov.settings.gpu_init()
# Line 9: adata = sc.read_10x_mtx( -- adata = sc.read_10x_mtx(
# Line 10:     'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file --     'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file
# Line 11:     var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index) --     var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
# Line 12:     cache=True)                              # write a cache file for faster subsequent reading --     cache=True)                              # write a cache file for faster subsequent reading
# Line 13: adata -- adata
# Line 15: adata.var_names_make_unique() -- adata.var_names_make_unique()
# Line 16: adata.obs_names_make_unique() -- adata.obs_names_make_unique()
# Line 18: ov.pp.anndata_to_GPU(adata) -- ov.pp.anndata_to_GPU(adata)
# Line 20: adata=ov.pp.qc(adata, -- adata=ov.pp.qc(adata,
# Line 21:               tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}, --               tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250},
# Line 22:               batch_key=None) --               batch_key=None)
# Line 23: adata -- adata
# Line 25: adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,) -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
# Line 26: adata -- adata
# Line 28: adata.raw = adata -- adata.raw = adata
# Line 29: adata = adata[:, adata.var.highly_variable_features] -- adata = adata[:, adata.var.highly_variable_features]
# Line 30: adata -- adata
# Line 32: ov.pp.scale(adata) -- ov.pp.scale(adata)
# Line 33: adata -- adata
# Line 35: ov.pp.pca(adata,layer='scaled',n_pcs=50) -- ov.pp.pca(adata,layer='scaled',n_pcs=50)
# Line 36: adata -- adata
# Line 38: adata.obsm['X_pca']=adata.obsm['scaled|original|X_pca'] -- adata.obsm['X_pca']=adata.obsm['scaled|original|X_pca']
# Line 39: ov.utils.embedding(adata, -- ov.utils.embedding(adata,
# Line 40:                   basis='X_pca', --                   basis='X_pca',
# Line 41:                   color='CST3', --                   color='CST3',
# Line 42:                   frameon='small') --                   frameon='small')
# Line 44: ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50, -- ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50,
# Line 45:                use_rep='scaled|original|X_pca',method='cagra') --                use_rep='scaled|original|X_pca',method='cagra')
# Line 47: adata.obsm["X_mde"] = ov.utils.mde(adata.obsm["scaled|original|X_pca"]) -- adata.obsm["X_mde"] = ov.utils.mde(adata.obsm["scaled|original|X_pca"])
# Line 48: adata -- adata
# Line 50: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 51:                 basis='X_mde', --                 basis='X_mde',
# Line 52:                 color='CST3', --                 color='CST3',
# Line 53:                 frameon='small') --                 frameon='small')
# Line 55: ov.pp.umap(adata) -- ov.pp.umap(adata)
# Line 57: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 58:                 basis='X_umap', --                 basis='X_umap',
# Line 59:                 color='CST3', --                 color='CST3',
# Line 60:                 frameon='small') --                 frameon='small')
# Line 62: ov.pp.leiden(adata) -- ov.pp.leiden(adata)
# Line 64: ov.pp.anndata_to_CPU(adata) -- ov.pp.anndata_to_CPU(adata)
# Line 66: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 67:                 basis='X_mde', --                 basis='X_mde',
# Line 68:                 color=['leiden', 'CST3', 'NKG7'], --                 color=['leiden', 'CST3', 'NKG7'],
# Line 69:                 frameon='small') --                 frameon='small')
# Line 71: import matplotlib.pyplot as plt -- import matplotlib.pyplot as plt
# Line 72: fig,ax=plt.subplots( figsize = (4,4)) -- fig,ax=plt.subplots( figsize = (4,4))
# Line 74: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 75:                 basis='X_mde', --                 basis='X_mde',
# Line 76:                 color=['leiden'], --                 color=['leiden'],
# Line 77:                 frameon='small', --                 frameon='small',
# Line 78:                 show=False, --                 show=False,
# Line 79:                 ax=ax) --                 ax=ax)
# Line 81: ov.pl.ConvexHull(adata, -- ov.pl.ConvexHull(adata,
# Line 82:                 basis='X_mde', --                 basis='X_mde',
# Line 83:                 cluster_key='leiden', --                 cluster_key='leiden',
# Line 84:                 hull_cluster='0', --                 hull_cluster='0',
# Line 85:                 ax=ax) --                 ax=ax)
# Line 88: from matplotlib import patheffects -- from matplotlib import patheffects
# Line 89: import matplotlib.pyplot as plt -- import matplotlib.pyplot as plt
# Line 90: fig, ax = plt.subplots(figsize=(4,4)) -- fig, ax = plt.subplots(figsize=(4,4))
# Line 92: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 93:                   basis='X_mde', --                   basis='X_mde',
# Line 94:                   color=['leiden'], --                   color=['leiden'],
# Line 95:                    show=False, legend_loc=None, add_outline=False,  --                    show=False, legend_loc=None, add_outline=False, 
# Line 96:                    frameon='small',legend_fontoutline=2,ax=ax --                    frameon='small',legend_fontoutline=2,ax=ax
# Line 99: ov.utils.gen_mpl_labels( -- ov.utils.gen_mpl_labels(
# Line 100:     adata, --     adata,
# Line 101:     'leiden', --     'leiden',
# Line 102:     exclude=("None",),   --     exclude=("None",),  
# Line 103:     basis='X_mde', --     basis='X_mde',
# Line 104:     ax=ax, --     ax=ax,
# Line 105:     adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')), --     adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')),
# Line 106:     text_kwargs=dict(fontsize= 12 ,weight='bold', --     text_kwargs=dict(fontsize= 12 ,weight='bold',
# Line 107:                      path_effects=[patheffects.withStroke(linewidth=2, foreground='w')] ), --                      path_effects=[patheffects.withStroke(linewidth=2, foreground='w')] ),
# Line 109: marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14', -- marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
# Line 110:                 'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1', --                 'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
# Line 111:                 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP'] --                 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']
# Line 113: sc.pl.dotplot(adata, marker_genes, groupby='leiden', -- sc.pl.dotplot(adata, marker_genes, groupby='leiden',
# Line 114:              standard_scale='var'); --              standard_scale='var');
# Line 116: sc.tl.dendrogram(adata,'leiden',use_rep='scaled|original|X_pca') -- sc.tl.dendrogram(adata,'leiden',use_rep='scaled|original|X_pca')
# Line 117: sc.tl.rank_genes_groups(adata, 'leiden', use_rep='scaled|original|X_pca', -- sc.tl.rank_genes_groups(adata, 'leiden', use_rep='scaled|original|X_pca',
# Line 118:                         method='t-test',use_raw=False,key_added='leiden_ttest') --                         method='t-test',use_raw=False,key_added='leiden_ttest')
# Line 119: sc.pl.rank_genes_groups_dotplot(adata,groupby='leiden', -- sc.pl.rank_genes_groups_dotplot(adata,groupby='leiden',
# Line 120:                                 cmap='Spectral_r',key='leiden_ttest', --                                 cmap='Spectral_r',key='leiden_ttest',
# Line 121:                                 standard_scale='var',n_genes=3) --                                 standard_scale='var',n_genes=3)
# Line 123: sc.tl.rank_genes_groups(adata, groupby='leiden',  -- sc.tl.rank_genes_groups(adata, groupby='leiden', 
# Line 124:                         method='t-test',use_rep='scaled|original|X_pca',) --                         method='t-test',use_rep='scaled|original|X_pca',)
# Line 125: ov.single.cosg(adata, key_added='leiden_cosg', groupby='leiden') -- ov.single.cosg(adata, key_added='leiden_cosg', groupby='leiden')
# Line 126: sc.pl.rank_genes_groups_dotplot(adata,groupby='leiden', -- sc.pl.rank_genes_groups_dotplot(adata,groupby='leiden',
# Line 127:                                 cmap='Spectral_r',key='leiden_cosg', --                                 cmap='Spectral_r',key='leiden_cosg',
# Line 128:                                 standard_scale='var',n_genes=3) --                                 standard_scale='var',n_genes=3)
# Line 130: data_dict={} -- data_dict={}
# Line 131: for i in adata.obs['leiden'].cat.categories: -- for i in adata.obs['leiden'].cat.categories:
# Line 132:     data_dict[i]=sc.get.rank_genes_groups_df(adata, group=i, key='leiden_ttest', --     data_dict[i]=sc.get.rank_genes_groups_df(adata, group=i, key='leiden_ttest',
# Line 133:                                             pval_cutoff=None,log2fc_min=None) --                                             pval_cutoff=None,log2fc_min=None)
# Line 135: data_dict.keys() -- data_dict.keys()
# Line 137: data_dict[i].head() -- data_dict[i].head()
# Line 139: type_color_dict=dict(zip(adata.obs['leiden'].cat.categories, -- type_color_dict=dict(zip(adata.obs['leiden'].cat.categories,
# Line 140:                          adata.uns['leiden_colors'])) --                          adata.uns['leiden_colors']))
# Line 141: type_color_dict -- type_color_dict
# Line 143: fig,axes=ov.utils.stacking_vol(data_dict,type_color_dict, -- fig,axes=ov.utils.stacking_vol(data_dict,type_color_dict,
# Line 144:             pval_threshold=0.01, --             pval_threshold=0.01,
# Line 145:             log2fc_threshold=2, --             log2fc_threshold=2,
# Line 146:             figsize=(8,4), --             figsize=(8,4),
# Line 147:             sig_color='#a51616', --             sig_color='#a51616',
# Line 148:             normal_color='#c7c7c7', --             normal_color='#c7c7c7',
# Line 149:             plot_genes_num=2, --             plot_genes_num=2,
# Line 150:             plot_genes_fontsize=6, --             plot_genes_fontsize=6,
# Line 151:             plot_genes_weight='bold', --             plot_genes_weight='bold',
# Line 155: for i in data_dict.keys(): -- for i in data_dict.keys():
# Line 156:     y_min=min(y_min,data_dict[i]['logfoldchanges'].min()) --     y_min=min(y_min,data_dict[i]['logfoldchanges'].min())
# Line 157:     y_max=max(y_max,data_dict[i]['logfoldchanges'].max()) --     y_max=max(y_max,data_dict[i]['logfoldchanges'].max())
# Line 158: for i in adata.obs['leiden'].cat.categories: -- for i in adata.obs['leiden'].cat.categories:
# Line 159:     axes[i].set_ylim(y_min,y_max) --     axes[i].set_ylim(y_min,y_max)
# Line 160: plt.suptitle('Stacking_vol',fontsize=12) -- plt.suptitle('Stacking_vol',fontsize=12)
```