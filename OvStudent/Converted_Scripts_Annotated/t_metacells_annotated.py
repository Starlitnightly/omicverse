```
# Line 1:  Import the omicverse library as ov. -- import omicverse as ov
# Line 2:  Import the scanpy library as sc. -- import scanpy as sc
# Line 3:  Import the scvelo library as scv. -- import scvelo as scv
# Line 5:  Set plotting parameters using ov.plot_set(). -- ov.plot_set()
# Line 7:  Load the pancreas dataset using scv and assign it to adata. -- adata = scv.datasets.pancreas()
# Line 8:  Display the loaded AnnData object. -- adata
# Line 11: Perform quality control on the AnnData object using ov.pp.qc, filtering based on mito percentage, number of UMIs, and detected genes and filtering mitochondrial genes. -- adata=ov.pp.qc(adata,
# Line 12: Perform quality control on the AnnData object using ov.pp.qc, filtering based on mito percentage, number of UMIs, and detected genes and filtering mitochondrial genes. --               tresh={'mito_perc': 0.20, 'nUMIs': 500, 'detected_genes': 250},
# Line 13: Perform quality control on the AnnData object using ov.pp.qc, filtering based on mito percentage, number of UMIs, and detected genes and filtering mitochondrial genes. --               mt_startswith='mt-')
# Line 15: Preprocess the AnnData object using ov.pp.preprocess with shiftlog normalization and Pearson residuals, calculating 2000 highly variable genes. -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=2000,)
# Line 18: Store the original AnnData object in the raw attribute. -- adata.raw = adata
# Line 19: Filter the AnnData object to keep only the highly variable genes. -- adata = adata[:, adata.var.highly_variable_features]
# Line 22: Scale the expression data in adata.X using ov.pp.scale(). -- ov.pp.scale(adata)
# Line 25: Perform PCA dimensionality reduction using ov.pp.pca, using scaled data and 50 principal components. -- ov.pp.pca(adata,layer='scaled',n_pcs=50)
# Line 27: Create a MetaCell object using ov.single.MetaCell with the scaled, original and pca data, no specific number of metacells, and using GPU if available. -- meta_obj=ov.single.MetaCell(adata,use_rep='scaled|original|X_pca',
# Line 28: Create a MetaCell object using ov.single.MetaCell with the scaled, original and pca data, no specific number of metacells, and using GPU if available. --                             n_metacells=None,
# Line 29: Create a MetaCell object using ov.single.MetaCell with the scaled, original and pca data, no specific number of metacells, and using GPU if available. --                            use_gpu='cuda:0')
# Line 31: Initialize the archetypes for the MetaCell object. -- meta_obj.initialize_archetypes()
# Line 33: Train the MetaCell model with a minimum of 10 and maximum of 50 iterations. -- meta_obj.train(min_iter=10, max_iter=50)
# Line 35: Save the trained MetaCell model to the specified file path. -- meta_obj.save('seacells/model.pkl')
# Line 37: Load the trained MetaCell model from the specified file path. -- meta_obj.load('seacells/model.pkl')
# Line 39: Predict cell assignments using the trained MetaCell model, assigning soft memberships, using cluster labels and summarizing normalized log data. -- ad=meta_obj.predicted(method='soft',celltype_label='clusters',
# Line 40: Predict cell assignments using the trained MetaCell model, assigning soft memberships, using cluster labels and summarizing normalized log data. --                      summarize_layer='lognorm')
# Line 42: Compute cell type purity scores based on clusters labels. -- SEACell_purity = meta_obj.compute_celltype_purity('clusters')
# Line 43: Calculate separation scores using specified representations and nearest neighbor. -- separation = meta_obj.separation(use_rep='scaled|original|X_pca',nth_nbr=1)
# Line 44: Calculate compactness scores using specified representations. -- compactness = meta_obj.compactness(use_rep='scaled|original|X_pca')
# Line 46: Import the seaborn library as sns. -- import seaborn as sns
# Line 47: Import the matplotlib.pyplot library as plt. -- import matplotlib.pyplot as plt
# Line 48: Set plot parameters with omicverse. -- ov.plot_set()
# Line 49: Create a figure and axes for subplots for evaluation metrics. -- fig, axes = plt.subplots(1,3,figsize=(4,4))
# Line 50: Create a box plot of the SEACell purity data on the first subplot using a blue color from the ov.utils palette. -- sns.boxplot(data=SEACell_purity, y='clusters_purity',ax=axes[0],
# Line 51: Create a box plot of the SEACell purity data on the first subplot using a blue color from the ov.utils palette. --            color=ov.utils.blue_color[3])
# Line 52: Create a box plot of the compactness data on the second subplot using a blue color from the ov.utils palette. -- sns.boxplot(data=compactness, y='compactness',ax=axes[1],
# Line 53: Create a box plot of the compactness data on the second subplot using a blue color from the ov.utils palette. --            color=ov.utils.blue_color[4])
# Line 54: Create a box plot of the separation data on the third subplot using a blue color from the ov.utils palette. -- sns.boxplot(data=separation, y='separation',ax=axes[2],
# Line 55: Create a box plot of the separation data on the third subplot using a blue color from the ov.utils palette. --            color=ov.utils.blue_color[4])
# Line 56: Adjust the spacing between subplots to avoid overlapping. -- plt.tight_layout()
# Line 57: Set the title of the entire figure and adjust vertical positioning. -- plt.suptitle('Evaluate of MetaCells',fontsize=13,y=1.05)
# Line 58: Iterate through each of the axes to customize the appearance of each plot. -- for ax in axes:
# Line 59: Disable grid lines for the current plot. --     ax.grid(False)
# Line 60: Make the top spine of the current plot invisible. --     ax.spines['top'].set_visible(False)
# Line 61: Make the right spine of the current plot invisible. --     ax.spines['right'].set_visible(False)
# Line 62: Make the bottom spine of the current plot visible. --     ax.spines['bottom'].set_visible(True)
# Line 63: Make the left spine of the current plot visible. --     ax.spines['left'].set_visible(True)
# Line 65: Import the matplotlib.pyplot library as plt. -- import matplotlib.pyplot as plt
# Line 66: Create a single figure and axes object for embedding. -- fig, ax = plt.subplots(figsize=(4,4))
# Line 67: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. -- ov.pl.embedding(
# Line 68: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     meta_obj.adata,
# Line 69: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     basis="X_umap",
# Line 70: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     color=['clusters'],
# Line 71: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     frameon='small',
# Line 72: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     title="Meta cells",
# Line 73: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     #legend_loc='on data',
# Line 74: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     legend_fontsize=14,
# Line 75: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     legend_fontoutline=2,
# Line 76: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     size=10,
# Line 77: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     ax=ax,
# Line 78: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     alpha=0.2,
# Line 79: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     #legend_loc='',
# Line 80: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     add_outline=False,
# Line 81: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     #add_outline=True,
# Line 82: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     outline_color='black',
# Line 83: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     outline_width=1,
# Line 84: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     show=False,
# Line 85: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     #palette=ov.utils.blue_color[:],
# Line 86: Generate an embedding plot using ov.pl.embedding using umap coordinates, colored by 'clusters' and customized legend, marker size, transparency, and outline. --     #legend_fontweight='normal'
# Line 87: Overlay meta cells using ov.single.plot_metacells on the existing plot with a specified color. -- ov.single.plot_metacells(ax,meta_obj.adata,color='#CB3E35',
# Line 89: Get mean S score values from meta cells using ov.single.get_obs_value function. -- ov.single.get_obs_value(ad,adata,groupby='S_score',
# Line 90: Get mean S score values from meta cells using ov.single.get_obs_value function. --                        type='mean')
# Line 91: Show the head of the annotation data. -- ad.obs.head()
# Line 93: Import the scanpy library as sc. -- import scanpy as sc
# Line 94: Create a copy of the AnnData object to the raw attribute. -- ad.raw=ad.copy()
# Line 95: Calculate highly variable genes using scanpy's highly_variable_genes function, selecting the top 2000 genes. -- sc.pp.highly_variable_genes(ad, n_top_genes=2000, inplace=True)
# Line 96: Filter the AnnData object to keep only the highly variable genes. -- ad=ad[:,ad.var.highly_variable]
# Line 98: Scale the expression data in ad.X using ov.pp.scale(). -- ov.pp.scale(ad)
# Line 99: Perform PCA dimensionality reduction using ov.pp.pca, using scaled data and 30 principal components. -- ov.pp.pca(ad,layer='scaled',n_pcs=30)
# Line 100: Compute neighborhood graph using ov.pp.neighbors, for specified parameters. -- ov.pp.neighbors(ad, n_neighbors=15, n_pcs=20,
# Line 101: Compute neighborhood graph using ov.pp.neighbors, for specified parameters. --                use_rep='scaled|original|X_pca')
# Line 103: Compute UMAP embedding coordinates using ov.pp.umap(). -- ov.pp.umap(ad)
# Line 105: Cast the 'celltype' column in ad.obs as a category type. -- ad.obs['celltype']=ad.obs['celltype'].astype('category')
# Line 106: Reorder the categories in the 'celltype' column of ad.obs to match the order of clusters in adata.obs. -- ad.obs['celltype']=ad.obs['celltype'].cat.reorder_categories(adata.obs['clusters'].cat.categories)
# Line 107: Copy color palette associated with 'clusters' from adata to ad under the name 'celltype'. -- ad.uns['celltype_colors']=adata.uns['clusters_colors']
# Line 109: Generate embedding plot using ov.pl.embedding using umap coordinates, colored by 'celltype' and 'S_score', with specified title and layout adjustments. -- ov.pl.embedding(ad, basis='X_umap',
# Line 110: Generate embedding plot using ov.pl.embedding using umap coordinates, colored by 'celltype' and 'S_score', with specified title and layout adjustments. --                 color=["celltype","S_score"],
# Line 111: Generate embedding plot using ov.pl.embedding using umap coordinates, colored by 'celltype' and 'S_score', with specified title and layout adjustments. --                 frameon='small',cmap='RdBu_r',
# Line 112: Generate embedding plot using ov.pl.embedding using umap coordinates, colored by 'celltype' and 'S_score', with specified title and layout adjustments. --                wspace=0.5)
```