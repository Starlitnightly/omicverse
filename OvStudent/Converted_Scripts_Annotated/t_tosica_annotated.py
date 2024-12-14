```
# Line 1: Import the omicverse library as ov. -- import omicverse as ov
# Line 2: Import the scanpy library as sc. -- import scanpy as sc
# Line 3: Set the plotting style for omicverse. -- ov.utils.ov_plot_set()
# Line 4: Read the reference data from a h5ad file into an AnnData object. -- ref_adata = sc.read('demo_train.h5ad')
# Line 5: Select all rows and columns with original var_names for ref_adata. -- ref_adata = ref_adata[:,ref_adata.var_names]
# Line 6: Print the ref_adata AnnData object. -- print(ref_adata)
# Line 7: Print the value counts of the 'Celltype' column in ref_adata's obs. -- print(ref_adata.obs.Celltype.value_counts())
# Line 8: Read the query data from a h5ad file into an AnnData object. -- query_adata = sc.read('demo_test.h5ad')
# Line 9: Select all rows and columns of query_adata based on ref_adata's var_names. -- query_adata = query_adata[:,ref_adata.var_names]
# Line 10: Print the query_adata AnnData object. -- print(query_adata)
# Line 11: Print the value counts of the 'Celltype' column in query_adata's obs. -- print(query_adata.obs.Celltype.value_counts())
# Line 12: Make the variable names of ref_adata unique. -- ref_adata.var_names_make_unique()
# Line 13: Make the variable names of query_adata unique. -- query_adata.var_names_make_unique()
# Line 14: Find the intersection of variable names between query_adata and ref_adata and store as a list. -- ret_gene=list(set(query_adata.var_names) & set(ref_adata.var_names))
# Line 15: Calculate the length of the ret_gene list. -- len(ret_gene)
# Line 16: Subset query_adata to keep only the genes in ret_gene. -- query_adata=query_adata[:,ret_gene]
# Line 17: Subset ref_adata to keep only the genes in ret_gene. -- ref_adata=ref_adata[:,ret_gene]
# Line 18: Print the maximum value of the X matrix for both ref_adata and query_adata. -- print(f"The max of ref_adata is {ref_adata.X.max()}, query_data is {query_adata.X.max()}",)
# Line 19: Download the TOSICA gene set file from the omicverse utils. -- ov.utils.download_tosica_gmt()
# Line 20: Initialize a pyTOSICA object using ref_adata, specifying gene set path, depth, label, project path and batch size. -- tosica_obj=ov.single.pyTOSICA(adata=ref_adata,
# Line 21:   gmt_path='genesets/GO_bp.gmt', depth=1,
# Line 22:   label_name='Celltype',
# Line 23:   project_path='hGOBP_demo',
# Line 24:   batch_size=8)
# Line 25: Train the TOSICA model for 5 epochs. -- tosica_obj.train(epochs=5)
# Line 26: Save the trained TOSICA model. -- tosica_obj.save()
# Line 27: Load the trained TOSICA model. -- tosica_obj.load()
# Line 28: Predict cell states for query_adata using the trained TOSICA model, saving results into new_adata. -- new_adata=tosica_obj.predicted(pre_adata=query_adata)
# Line 29: Scale the query_adata object. -- ov.pp.scale(query_adata)
# Line 30: Perform PCA on the scaled data of query_adata, keeping 50 components. -- ov.pp.pca(query_adata,layer='scaled',n_pcs=50)
# Line 31: Compute the neighborhood graph of query_adata using the scaled PCA data with 15 neighbors. -- sc.pp.neighbors(query_adata, n_neighbors=15, n_pcs=50,
# Line 32:   use_rep='scaled|original|X_pca')
# Line 33: Compute the multidimensional embedding (MDE) of query_adata's scaled PCA data and save as "X_mde". -- query_adata.obsm["X_mde"] = ov.utils.mde(query_adata.obsm["scaled|original|X_pca"])
# Line 34: Print the modified query_adata object. -- query_adata
# Line 35: Copy the obsm from query_adata to new_adata based on the overlapping obs indices. -- new_adata.obsm=query_adata[new_adata.obs.index].obsm.copy()
# Line 36: Copy the obsp from query_adata to new_adata based on the overlapping obs indices. -- new_adata.obsp=query_adata[new_adata.obs.index].obsp.copy()
# Line 37: Print the modified new_adata object. -- new_adata
# Line 38: Import the numpy library as np. -- import numpy as np
# Line 39: Create a numpy array of hex color codes as a string of unicode characters, and set the dtype. -- col = np.array([
# Line 40: "#98DF8A","#E41A1C" ,"#377EB8", "#4DAF4A" ,"#984EA3" ,"#FF7F00" ,"#FFFF33" ,"#A65628" ,"#F781BF" ,"#999999","#1F77B4","#FF7F0E","#279E68","#FF9896"
# Line 41: ]).astype('<U7')
# Line 42: Create a tuple of cell type names. -- celltype = ("alpha","beta","ductal","acinar","delta","PP","PSC","endothelial","epsilon","mast","macrophage","schwann",'t_cell')
# Line 43: Convert the 'Prediction' column in new_adata's obs to a categorical type. -- new_adata.obs['Prediction'] = new_adata.obs['Prediction'].astype('category')
# Line 44: Reorder the categories of 'Prediction' in new_adata based on the celltype tuple. -- new_adata.obs['Prediction'] = new_adata.obs['Prediction'].cat.reorder_categories(list(celltype))
# Line 45: Set the color palette for 'Prediction' in new_adata.uns using colors from the col array starting from index 1. -- new_adata.uns['Prediction_colors'] = col[1:]
# Line 46: Create a tuple of cell type names including MHC class II. -- celltype = ("MHC class II","alpha","beta","ductal","acinar","delta","PP","PSC","endothelial","epsilon","mast")
# Line 47: Convert the 'Celltype' column in new_adata's obs to a categorical type. -- new_adata.obs['Celltype'] = new_adata.obs['Celltype'].astype('category')
# Line 48: Reorder the categories of 'Celltype' in new_adata based on the updated celltype tuple. -- new_adata.obs['Celltype'] = new_adata.obs['Celltype'].cat.reorder_categories(list(celltype))
# Line 49: Set the color palette for 'Celltype' in new_adata.uns using the first 11 colors from the col array. -- new_adata.uns['Celltype_colors'] = col[:11]
# Line 50: Create an embedding plot using scanpy, visualizing "X_mde" and coloring by 'Celltype' and 'Prediction'. -- sc.pl.embedding(
# Line 51:  new_adata,
# Line 52:  basis="X_mde",
# Line 53:  color=['Celltype', 'Prediction'],
# Line 54:  frameon=False,
# Line 55:  #ncols=1,
# Line 56:  wspace=0.5,
# Line 57:  #palette=ov.utils.pyomic_palette()[11:],
# Line 58:  show=False,
# Line 59: Identify cell types in the 'Prediction' column that have fewer than 5 counts. -- cell_idx=new_adata.obs['Prediction'].value_counts()[new_adata.obs['Prediction'].value_counts()<5].index
# Line 60: Remove cells from new_adata that have a predicted cell type found in cell_idx. -- new_adata=new_adata[~new_adata.obs['Prediction'].isin(cell_idx)]
# Line 61: Perform differential gene expression analysis using Wilcoxon rank-sum test based on 'Prediction'. -- sc.tl.rank_genes_groups(new_adata, 'Prediction', method='wilcoxon')
# Line 62: Create a dotplot showing top ranked genes for each prediction using scanpy's rank_genes_groups results. -- sc.pl.rank_genes_groups_dotplot(new_adata,
# Line 63:  n_genes=3,standard_scale='var',)
# Line 64: Get a dataframe of DEGs for the 'PP' group, from rank_genes_groups results, filtered by p-value. -- degs = sc.get.rank_genes_groups_df(new_adata, group='PP', key='rank_genes_groups',
# Line 65:  pval_cutoff=0.05)
# Line 66: Print the first 5 rows of the degs dataframe. -- degs.head()
# Line 67: Create an embedding plot using scanpy, visualizing "X_mde" and coloring by 'Prediction' and 'GOBP_REGULATION_OF_MUSCLE_SYSTEM_PROCESS'. -- sc.pl.embedding(
# Line 68:  new_adata,
# Line 69:  basis="X_mde",
# Line 70:  color=['Prediction','GOBP_REGULATION_OF_MUSCLE_SYSTEM_PROCESS'],
# Line 71:  frameon=False,
# Line 72:  #ncols=1,
# Line 73:  wspace=0.5,
# Line 74:  #palette=ov.utils.pyomic_palette()[11:],
# Line 75:  show=False,
```