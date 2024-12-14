```
# Line 1:  import scanpy as sc -- import scanpy as sc
# Line 2:  import omicverse as ov -- import omicverse as ov
# Line 3:  ov.plot_set() -- ov.plot_set()
# Line 5:  !wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O data/pbmc3k_filtered_gene_bc_matrices.tar.gz -- !wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O data/pbmc3k_filtered_gene_bc_matrices.tar.gz
# Line 6:  !cd data; tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz -- !cd data; tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz
# Line 8: adata = sc.read_10x_mtx( -- adata = sc.read_10x_mtx(
# Line 9:     'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file --     'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file
# Line 10:     var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index) --     var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
# Line 11:     cache=True)                              # write a cache file for faster subsequent reading --     cache=True)                              # write a cache file for faster subsequent reading
# Line 12: adata -- adata
# Line 14: adata.var_names_make_unique() -- adata.var_names_make_unique()
# Line 15: adata.obs_names_make_unique() -- adata.obs_names_make_unique()
# Line 17: %%time -- %%time
# Line 18: adata=ov.pp.qc(adata, -- adata=ov.pp.qc(adata,
# Line 19:               tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}, --               tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250},
# Line 20:                doublets_method='sccomposite', --                doublets_method='sccomposite',
# Line 21:               batch_key=None) --               batch_key=None)
# Line 22: adata -- adata
# Line 24: %%time -- %%time
# Line 25: adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,) -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
# Line 26: adata -- adata
# Line 28: %%time -- %%time
# Line 29: adata.raw = adata -- adata.raw = adata
# Line 30: adata = adata[:, adata.var.highly_variable_features] -- adata = adata[:, adata.var.highly_variable_features]
# Line 31: adata -- adata
# Line 33: %%time -- %%time
# Line 34: ov.pp.scale(adata) -- ov.pp.scale(adata)
# Line 35: adata -- adata
# Line 37: %%time -- %%time
# Line 38: ov.pp.pca(adata,layer='scaled',n_pcs=50) -- ov.pp.pca(adata,layer='scaled',n_pcs=50)
# Line 39: adata -- adata
# Line 41: adata.obsm['X_pca']=adata.obsm['scaled|original|X_pca'] -- adata.obsm['X_pca']=adata.obsm['scaled|original|X_pca']
# Line 42: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 43:                   basis='X_pca', --                   basis='X_pca',
# Line 44:                   color='CST3', --                   color='CST3',
# Line 45:                   frameon='small') --                   frameon='small')
# Line 47: %%time -- %%time
# Line 48: ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50, -- ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50,
# Line 49:                use_rep='scaled|original|X_pca') --                use_rep='scaled|original|X_pca')
# Line 51: %%time -- %%time
# Line 52: ov.pp.umap(adata) -- ov.pp.umap(adata)
# Line 54: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 55:                 basis='X_umap', --                 basis='X_umap',
# Line 56:                 color='CST3', --                 color='CST3',
# Line 57:                 frameon='small') --                 frameon='small')
# Line 59: ov.pp.mde(adata,embedding_dim=2,n_neighbors=15, basis='X_mde', -- ov.pp.mde(adata,embedding_dim=2,n_neighbors=15, basis='X_mde',
# Line 60:           n_pcs=50, use_rep='scaled|original|X_pca',) --           n_pcs=50, use_rep='scaled|original|X_pca',)
# Line 62: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 63:                 basis='X_mde', --                 basis='X_mde',
# Line 64:                 color='CST3', --                 color='CST3',
# Line 65:                 frameon='small') --                 frameon='small')
# Line 67: adata_raw=adata.raw.to_adata() -- adata_raw=adata.raw.to_adata()
# Line 68: ov.pp.score_genes_cell_cycle(adata_raw,species='human') -- ov.pp.score_genes_cell_cycle(adata_raw,species='human')
# Line 70: ov.pl.embedding(adata_raw, -- ov.pl.embedding(adata_raw,
# Line 71:                 basis='X_mde', --                 basis='X_mde',
# Line 72:                 color='phase', --                 color='phase',
# Line 73:                 frameon='small') --                 frameon='small')
# Line 75: ov.pp.leiden(adata,resolution=1) -- ov.pp.leiden(adata,resolution=1)
# Line 77: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 78:                 basis='X_mde', --                 basis='X_mde',
# Line 79:                 color=['leiden', 'CST3', 'NKG7'], --                 color=['leiden', 'CST3', 'NKG7'],
# Line 80:                 frameon='small') --                 frameon='small')
# Line 82: import matplotlib.pyplot as plt -- import matplotlib.pyplot as plt
# Line 83: fig,ax=plt.subplots( figsize = (4,4)) -- fig,ax=plt.subplots( figsize = (4,4))
# Line 85: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 86:                 basis='X_mde', --                 basis='X_mde',
# Line 87:                 color=['leiden'], --                 color=['leiden'],
# Line 88:                 frameon='small', --                 frameon='small',
# Line 89:                 show=False, --                 show=False,
# Line 90:                 ax=ax) --                 ax=ax)
# Line 92: ov.pl.ConvexHull(adata, -- ov.pl.ConvexHull(adata,
# Line 93:                 basis='X_mde', --                 basis='X_mde',
# Line 94:                 cluster_key='leiden', --                 cluster_key='leiden',
# Line 95:                 hull_cluster='0', --                 hull_cluster='0',
# Line 96:                 ax=ax) --                 ax=ax)
# Line 99: from matplotlib import patheffects -- from matplotlib import patheffects
# Line 100: import matplotlib.pyplot as plt -- import matplotlib.pyplot as plt
# Line 101: fig, ax = plt.subplots(figsize=(4,4)) -- fig, ax = plt.subplots(figsize=(4,4))
# Line 103: ov.pl.embedding(adata, -- ov.pl.embedding(adata,
# Line 104:                   basis='X_mde', --                   basis='X_mde',
# Line 105:                   color=['leiden'], --                   color=['leiden'],
# Line 106:                    show=False, legend_loc=None, add_outline=False,  --                    show=False, legend_loc=None, add_outline=False, 
# Line 107:                    frameon='small',legend_fontoutline=2,ax=ax --                    frameon='small',legend_fontoutline=2,ax=ax
# Line 110: ov.utils.gen_mpl_labels( -- ov.utils.gen_mpl_labels(
# Line 111:     adata, --     adata,
# Line 112:     'leiden', --     'leiden',
# Line 113:     exclude=("None",),  --     exclude=("None",),  
# Line 114:     basis='X_mde', --     basis='X_mde',
# Line 115:     ax=ax, --     ax=ax,
# Line 116:     adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')), --     adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')),
# Line 117:     text_kwargs=dict(fontsize= 12 ,weight='bold', --     text_kwargs=dict(fontsize= 12 ,weight='bold',
# Line 118:                      path_effects=[patheffects.withStroke(linewidth=2, foreground='w')] ), --                      path_effects=[patheffects.withStroke(linewidth=2, foreground='w')] ),
# Line 121: marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14', -- marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
# Line 122:                 'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1', --                 'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
# Line 123:                 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP'] --                 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']
# Line 125: sc.pl.dotplot(adata, marker_genes, groupby='leiden', -- sc.pl.dotplot(adata, marker_genes, groupby='leiden',
# Line 126:              standard_scale='var'); --              standard_scale='var');
# Line 128: sc.tl.dendrogram(adata,'leiden',use_rep='scaled|original|X_pca') -- sc.tl.dendrogram(adata,'leiden',use_rep='scaled|original|X_pca')
# Line 129: sc.tl.rank_genes_groups(adata, 'leiden', use_rep='scaled|original|X_pca', -- sc.tl.rank_genes_groups(adata, 'leiden', use_rep='scaled|original|X_pca',
# Line 130:                         method='t-test',use_raw=False,key_added='leiden_ttest') --                         method='t-test',use_raw=False,key_added='leiden_ttest')
# Line 131: sc.pl.rank_genes_groups_dotplot(adata,groupby='leiden', -- sc.pl.rank_genes_groups_dotplot(adata,groupby='leiden',
# Line 132:                                 cmap='Spectral_r',key='leiden_ttest', --                                 cmap='Spectral_r',key='leiden_ttest',
# Line 133:                                 standard_scale='var',n_genes=3) --                                 standard_scale='var',n_genes=3)
# Line 135: sc.tl.rank_genes_groups(adata, groupby='leiden',  -- sc.tl.rank_genes_groups(adata, groupby='leiden', 
# Line 136:                         method='t-test',use_rep='scaled|original|X_pca',) --                         method='t-test',use_rep='scaled|original|X_pca',)
# Line 137: ov.single.cosg(adata, key_added='leiden_cosg', groupby='leiden') -- ov.single.cosg(adata, key_added='leiden_cosg', groupby='leiden')
# Line 138: sc.pl.rank_genes_groups_dotplot(adata,groupby='leiden', -- sc.pl.rank_genes_groups_dotplot(adata,groupby='leiden',
# Line 139:                                 cmap='Spectral_r',key='leiden_cosg', --                                 cmap='Spectral_r',key='leiden_cosg',
# Line 140:                                 standard_scale='var',n_genes=3) --                                 standard_scale='var',n_genes=3)
# Line 142: data_dict={} -- data_dict={}
# Line 143: for i in adata.obs['leiden'].cat.categories: -- for i in adata.obs['leiden'].cat.categories:
# Line 144:     data_dict[i]=sc.get.rank_genes_groups_df(adata, group=i, key='leiden_ttest', --     data_dict[i]=sc.get.rank_genes_groups_df(adata, group=i, key='leiden_ttest',
# Line 145:                                             pval_cutoff=None,log2fc_min=None) --                                             pval_cutoff=None,log2fc_min=None)
# Line 147: data_dict.keys() -- data_dict.keys()
# Line 149: data_dict[i].head() -- data_dict[i].head()
# Line 151: type_color_dict=dict(zip(adata.obs['leiden'].cat.categories, -- type_color_dict=dict(zip(adata.obs['leiden'].cat.categories,
# Line 152:                          adata.uns['leiden_colors'])) --                          adata.uns['leiden_colors']))
# Line 153: type_color_dict -- type_color_dict
# Line 155: fig,axes=ov.utils.stacking_vol(data_dict,type_color_dict, -- fig,axes=ov.utils.stacking_vol(data_dict,type_color_dict,
# Line 156:             pval_threshold=0.01, --             pval_threshold=0.01,
# Line 157:             log2fc_threshold=2, --             log2fc_threshold=2,
# Line 158:             figsize=(8,4), --             figsize=(8,4),
# Line 159:             sig_color='#a51616', --             sig_color='#a51616',
# Line 160:             normal_color='#c7c7c7', --             normal_color='#c7c7c7',
# Line 161:             plot_genes_num=2, --             plot_genes_num=2,
# Line 162:             plot_genes_fontsize=6, --             plot_genes_fontsize=6,
# Line 163:             plot_genes_weight='bold', --             plot_genes_weight='bold',
# Line 166: y_min,y_max=0,0 -- y_min,y_max=0,0
# Line 167: for i in data_dict.keys(): -- for i in data_dict.keys():
# Line 168:     y_min=min(y_min,data_dict[i]['logfoldchanges'].min()) --     y_min=min(y_min,data_dict[i]['logfoldchanges'].min())
# Line 169:     y_max=max(y_max,data_dict[i]['logfoldchanges'].max()) --     y_max=max(y_max,data_dict[i]['logfoldchanges'].max())
# Line 170: for i in adata.obs['leiden'].cat.categories: -- for i in adata.obs['leiden'].cat.categories:
# Line 171:     axes[i].set_ylim(y_min,y_max) --     axes[i].set_ylim(y_min,y_max)
# Line 172: plt.suptitle('Stacking_vol',fontsize=12) -- plt.suptitle('Stacking_vol',fontsize=12)
```