```
# Line 1:  import omicverse as ov -- import omicverse as ov
# Line 3:  import scanpy as sc -- import scanpy as sc
# Line 5:  ov.utils.ov_plot_set() -- Sets the plotting style for omicverse.
# Line 7:  adata1=ov.read('neurips2021_s1d3.h5ad') -- Reads the first AnnData object from the specified H5AD file.
# Line 8:  adata1.obs['batch']='s1d3' -- Adds a 'batch' column to the observations of the first AnnData object with value 's1d3'.
# Line 9:  adata2=ov.read('neurips2021_s2d1.h5ad') -- Reads the second AnnData object from the specified H5AD file.
# Line 10: adata2.obs['batch']='s2d1' -- Adds a 'batch' column to the observations of the second AnnData object with value 's2d1'.
# Line 11: adata3=ov.read('neurips2021_s3d7.h5ad') -- Reads the third AnnData object from the specified H5AD file.
# Line 12: adata3.obs['batch']='s3d7' -- Adds a 'batch' column to the observations of the third AnnData object with value 's3d7'.
# Line 14: adata=sc.concat([adata1,adata2,adata3],merge='same') -- Concatenates the three AnnData objects into a single AnnData object.
# Line 15: adata -- Displays the concatenated AnnData object.
# Line 17: adata.obs['batch'].unique() -- Displays the unique values of the 'batch' column in the observations of the concatenated AnnData object.
# Line 19: import numpy as np -- Imports the NumPy library as np.
# Line 20: adata.X=adata.X.astype(np.int64) -- Casts the data matrix of the AnnData object to a 64-bit integer type.
# Line 22: adata=ov.pp.qc(adata,  tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250}, batch_key='batch') -- Performs quality control filtering on the AnnData object based on specified thresholds and batch key.
# Line 24: adata -- Displays the quality controlled AnnData object.
# Line 26: adata=ov.pp.preprocess(adata,mode='shiftlog|pearson', n_HVGs=3000,batch_key=None) -- Preprocesses the AnnData object, including shift-log transformation and highly variable gene selection.
# Line 28: adata -- Displays the preprocessed AnnData object.
# Line 30: adata.raw = adata -- Stores a copy of the current AnnData object in the .raw attribute.
# Line 31: adata = adata[:, adata.var.highly_variable_features] -- Subsets the AnnData object to only include highly variable genes.
# Line 32: adata -- Displays the AnnData object after subsetting.
# Line 34: adata.write_h5ad('neurips2021_batch_normlog.h5ad',compression='gzip') -- Writes the AnnData object to a compressed H5AD file.
# Line 36: ov.pp.scale(adata) -- Scales the data matrix of the AnnData object.
# Line 37: ov.pp.pca(adata,layer='scaled',n_pcs=50,mask_var='highly_variable_features') -- Performs PCA on the scaled data using the highly variable features mask.
# Line 39: adata.obsm["X_mde_pca"] = ov.utils.mde(adata.obsm["scaled|original|X_pca"]) -- Computes an MDE embedding from the PCA results and stores it in the .obsm.
# Line 41: ov.utils.embedding(adata, basis='X_mde_pca',frameon='small', color=['batch','cell_type'],show=False) -- Generates an embedding plot using the MDE transformed PCA coordinates, color coded by batch and cell type.
# Line 43: adata_harmony=ov.single.batch_correction(adata,batch_key='batch', methods='harmony',n_pcs=50) -- Performs batch correction using Harmony.
# Line 44: adata -- Displays the AnnData object after Harmony batch correction.
# Line 46: adata.obsm["X_mde_harmony"] = ov.utils.mde(adata.obsm["X_harmony"]) -- Computes an MDE embedding from the Harmony corrected data and stores it in .obsm.
# Line 48: ov.utils.embedding(adata, basis='X_mde_harmony',frameon='small', color=['batch','cell_type'],show=False) -- Generates an embedding plot using the MDE transformed Harmony coordinates, color coded by batch and cell type.
# Line 50: adata_combat=ov.single.batch_correction(adata,batch_key='batch', methods='combat',n_pcs=50) -- Performs batch correction using ComBat.
# Line 51: adata -- Displays the AnnData object after ComBat batch correction.
# Line 53: adata.obsm["X_mde_combat"] = ov.utils.mde(adata.obsm["X_combat"]) -- Computes an MDE embedding from the ComBat corrected data and stores it in .obsm.
# Line 55: ov.utils.embedding(adata, basis='X_mde_combat',frameon='small', color=['batch','cell_type'],show=False) -- Generates an embedding plot using the MDE transformed ComBat coordinates, color coded by batch and cell type.
# Line 57: adata_scanorama=ov.single.batch_correction(adata,batch_key='batch', methods='scanorama',n_pcs=50) -- Performs batch correction using Scanorama.
# Line 58: adata -- Displays the AnnData object after Scanorama batch correction.
# Line 60: adata.obsm["X_mde_scanorama"] = ov.utils.mde(adata.obsm["X_scanorama"]) -- Computes an MDE embedding from the Scanorama corrected data and stores it in .obsm.
# Line 62: ov.utils.embedding(adata, basis='X_mde_scanorama',frameon='small', color=['batch','cell_type'],show=False) -- Generates an embedding plot using the MDE transformed Scanorama coordinates, color coded by batch and cell type.
# Line 64: adata_scvi=ov.single.batch_correction(adata,batch_key='batch', methods='scVI',n_layers=2, n_latent=30, gene_likelihood="nb") -- Performs batch correction using scVI.
# Line 65: adata -- Displays the AnnData object after scVI batch correction.
# Line 67: adata.obsm["X_mde_scVI"] = ov.utils.mde(adata.obsm["X_scVI"]) -- Computes an MDE embedding from the scVI corrected data and stores it in .obsm.
# Line 69: ov.utils.embedding(adata, basis='X_mde_scVI',frameon='small', color=['batch','cell_type'],show=False) -- Generates an embedding plot using the MDE transformed scVI coordinates, color coded by batch and cell type.
# Line 71: LDA_obj=ov.utils.LDA_topic(adata,feature_type='expression', highly_variable_key='highly_variable_features', layers='counts',batch_key='batch',learning_rate=1e-3) -- Initializes an LDA topic model using expression data and considering batch effects.
# Line 73: LDA_obj.plot_topic_contributions(6) -- Plots the contribution of the top 6 topics.
# Line 75: LDA_obj.predicted(15) -- Predicts the topic for each cell using the LDA model with 15 topics.
# Line 77: adata.obsm["X_mde_mira_topic"] = ov.utils.mde(adata.obsm["X_topic_compositions"]) -- Computes an MDE embedding of topic compositions and stores it in the .obsm.
# Line 78: adata.obsm["X_mde_mira_feature"] = ov.utils.mde(adata.obsm["X_umap_features"]) -- Computes an MDE embedding of UMAP features and stores it in the .obsm.
# Line 80: ov.utils.embedding(adata, basis='X_mde_mira_topic',frameon='small', color=['batch','cell_type'],show=False) -- Generates an embedding plot using the MDE transformed topic compositions, color coded by batch and cell type.
# Line 83: ov.utils.embedding(adata, basis='X_mde_mira_feature',frameon='small', color=['batch','cell_type'],show=False) -- Generates an embedding plot using the MDE transformed UMAP features, color coded by batch and cell type.
# Line 85: adata.write_h5ad('neurips2021_batch_all.h5ad',compression='gzip') -- Writes the AnnData object containing all batch correction results to a compressed H5AD file.
# Line 87: adata=sc.read('neurips2021_batch_all.h5ad') -- Reads the AnnData object back from the specified H5AD file.
# Line 89: adata.obsm['X_pca']=adata.obsm['scaled|original|X_pca'].copy() -- Copies the PCA embedding from scaled data into a new '.obsm' key.
# Line 90: adata.obsm['X_mira_topic']=adata.obsm['X_topic_compositions'].copy() -- Copies the topic composition embedding to a new '.obsm' key.
# Line 91: adata.obsm['X_mira_feature']=adata.obsm['X_umap_features'].copy() -- Copies the UMAP feature embedding to a new '.obsm' key.
# Line 93: from scib_metrics.benchmark import Benchmarker -- Imports the Benchmarker class from the scib_metrics library.
# Line 94: bm = Benchmarker( adata, batch_key="batch", label_key="cell_type", embedding_obsm_keys=["X_pca", "X_combat", "X_harmony", 'X_scanorama','X_mira_topic','X_mira_feature','X_scVI'], n_jobs=8, ) -- Initializes a Benchmarker object for evaluating batch correction methods.
# Line 99: bm.benchmark() -- Runs the benchmark to evaluate the batch correction results.
# Line 101: bm.plot_results_table(min_max_scale=False) -- Plots a table summarizing the benchmark results.
```