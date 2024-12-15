```
# Line 1: Imports the csr_matrix class from the scipy.sparse module for creating sparse matrices. -- from scipy.sparse import csr_matrix
# Line 2: Imports the omicverse library as ov. -- import omicverse as ov
# Line 3: Imports the scanpy library as sc. -- import scanpy as sc
# Line 4: Imports the anndata library as ad. -- import anndata as ad
# Line 5: Imports the pandas library as pd. -- import pandas as pd
# Line 6: Imports the os module for interacting with the operating system. -- import os
# Line 8: Sets the plotting style for omicverse. -- ov.utils.ov_plot_set()
# Line 10: Initializes an empty list called Batch_list. -- Batch_list = []
# Line 11: Initializes an empty list called adj_list. -- adj_list = []
# Line 12: Defines a list of section IDs, likely corresponding to different datasets. -- section_ids = ['Slide-seqV2_MoB', 'Stereo-seq_MoB']
# Line 13: Prints the section IDs. -- print(section_ids)
# Line 14: Defines a variable 'pathway' which is a string representing the path to the STAligner directory. -- pathway = '/storage/zengjianyangLab/hulei/scRNA-seq/scripts/STAligner'
# Line 16: Starts a loop that iterates through each section ID. -- for section_id in section_ids:
# Line 17: Prints the current section ID. -- print(section_id)
# Line 18: Reads an AnnData object from an h5ad file based on the section ID. -- adata = sc.read_h5ad(os.path.join(pathway,section_id+".h5ad"))
# Line 20: Checks if the data matrix (adata.X) is a pandas DataFrame. -- if isinstance(adata.X, pd.DataFrame):
# Line 21: If adata.X is a DataFrame, converts it to a sparse matrix in CSR format. -- adata.X = csr_matrix(adata.X)
# Line 22: If adata.X is not a DataFrame, pass does nothing and continue -- else:
# Line 23: Does nothing if the if statement on Line 20 is false -- pass
# Line 25: Makes the variable names in adata unique by appending '++' to duplicates. -- adata.var_names_make_unique(join="++")
# Line 28: Creates unique observation names by appending the section ID. -- adata.obs_names = [x+'_'+section_id for x in adata.obs_names]
# Line 31: Calculates spatial network based on spot coordinates and saves to adata.uns['adj']. -- ov.space.Cal_Spatial_Net(adata, rad_cutoff=50) # the spatial network are saved in adata.uns[‘adj’]
# Line 34: Identifies highly variable genes using the seurat_v3 method, selecting the top 10000. -- sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=10000)
# Line 35: Normalizes counts such that each cell has a total count equal to target_sum. -- sc.pp.normalize_total(adata, target_sum=1e4)
# Line 36: Applies a log transformation to normalized counts. -- sc.pp.log1p(adata)
# Line 38: Subsets the AnnData object to only keep highly variable genes. -- adata = adata[:, adata.var['highly_variable']]
# Line 39: Appends the adjacency matrix to adj_list. -- adj_list.append(adata.uns['adj'])
# Line 40: Appends the processed AnnData object to the Batch_list. -- Batch_list.append(adata)
# Line 43: Prints the list of AnnData Objects that have been loaded -- Batch_list
# Line 45: Concatenates the AnnData objects in Batch_list into a single object, adding a 'slice_name' column and using section_ids as keys. -- adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
# Line 46: Creates a batch_name column from the slice_name column and converts to category type -- adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
# Line 47: Prints the shape of the concatenated AnnData object. -- print('adata_concat.shape: ', adata_concat.shape)
# Line 49: Comment: Measures the time for this code block. -- %%time
# Line 50: Creates a list of tuples indicating the order of slice integration (iter_comb). -- iter_comb = [(i, i + 1) for i in range(len(section_ids) - 1)]
# Line 53: Creates an STAligner object for integrating spatial transcriptomics data. -- STAligner_obj = ov.space.pySTAligner(adata_concat, verbose=True, knn_neigh = 100, n_epochs = 600, iter_comb = iter_comb,
# Line 54: Initializes the STAligner object with the concatenated adata, sets training parameters, and specifies the batch key. --  batch_key = 'batch_name',  key_added='STAligner', Batch_list = Batch_list)
# Line 56: Trains the STAligner model. -- STAligner_obj.train()
# Line 58: Gets the predicted latent representation from the trained model. -- adata = STAligner_obj.predicted()
# Line 60: Computes the nearest neighbors graph using the STAligner embedding. -- sc.pp.neighbors(adata, use_rep='STAligner', random_state=666)
# Line 61: Clusters the cells using the Leiden algorithm based on the STAligner representation. -- ov.utils.cluster(adata,use_rep='STAligner',method='leiden',resolution=0.4)
# Line 62: Runs UMAP for dimensionality reduction using the STAligner embedding. -- sc.tl.umap(adata, random_state=666)
# Line 63: Generates and displays a UMAP plot, coloring by batch and cluster, and sets plot space. -- sc.pl.umap(adata, color=['batch_name',"leiden"],wspace=0.5)
# Line 66: Imports the matplotlib.pyplot module as plt for plotting. -- import matplotlib.pyplot as plt
# Line 67: Sets spot size for the spatial plots. -- spot_size = 50
# Line 68: Sets title size for the spatial plots. -- title_size = 15
# Line 69: Creates a figure and two subplots for spatial visualization. -- fig, ax = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'wspace': 0.05, 'hspace': 0.2})
# Line 70: Generates spatial plots for 'Slide-seqV2_MoB' colored by cluster and removes the legend. -- _sc_0 = sc.pl.spatial(adata[adata.obs['batch_name'] == 'Slide-seqV2_MoB'], img_key=None, color=['leiden'], title=['Slide-seqV2'],
# Line 71: Sets parameters for the first spatial plot such as legend font size, show plot flag, axis object, and removes plot frame --                       legend_fontsize=10, show=False, ax=ax[0], frameon=False, spot_size=spot_size, legend_loc=None)
# Line 72: Sets the title of the first spatial plot and sets the size. -- _sc_0[0].set_title('Slide-seqV2', size=title_size)
# Line 74: Generates spatial plots for 'Stereo-seq_MoB' colored by cluster and removes the legend. -- _sc_1 = sc.pl.spatial(adata[adata.obs['batch_name'] == 'Stereo-seq_MoB'], img_key=None, color=['leiden'], title=['Stereo-seq'],
# Line 75: Sets parameters for the second spatial plot such as legend font size, show plot flag, axis object, and removes plot frame --                       legend_fontsize=10, show=False, ax=ax[1], frameon=False, spot_size=spot_size)
# Line 76: Sets the title of the second spatial plot and sets the size. -- _sc_1[0].set_title('Stereo-seq',size=title_size)
# Line 77: Inverts the y-axis of the second spatial plot. -- _sc_1[0].invert_yaxis()
# Line 78: Displays the generated plots. -- plt.show()
```
