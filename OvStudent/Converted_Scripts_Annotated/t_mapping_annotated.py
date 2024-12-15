```
# Line 1: Imports the omicverse library and assigns it the alias ov. -- import omicverse as ov
# Line 3: Imports the scanpy library and assigns it the alias sc. -- import scanpy as sc
# Line 5: Sets the plotting parameters for omicverse. -- ov.utils.ov_plot_set()
# Line 7: Reads a single-cell data file into an AnnData object using omicverse. -- adata_sc=ov.read('data/sc.h5ad')
# Line 8: Imports the matplotlib.pyplot module and assigns it the alias plt. -- import matplotlib.pyplot as plt
# Line 9: Creates a matplotlib figure and axes with a specified size. -- fig, ax = plt.subplots(figsize=(3,3))
# Line 10: Generates and displays an embedding plot using omicverse, visualizing the 'Subset' annotation on the UMAP coordinates. -- ov.utils.embedding(
# Line 20: Prints the maximum value of raw expression data before normalization. -- print("RAW",adata_sc.X.max())
# Line 21: Preprocesses the single-cell data using omicverse, applying shifting and log transformation, selecting highly variable genes, and targeting a specific total sum. -- adata_sc=ov.pp.preprocess(adata_sc,mode='shiftlog|pearson',n_HVGs=3000,target_sum=1e4)
# Line 22: Stores the preprocessed data as raw attribute. -- adata_sc.raw = adata_sc
# Line 23: Subsets the AnnData object to include only the highly variable genes. -- adata_sc = adata_sc[:, adata_sc.var.highly_variable_features]
# Line 24: Prints the maximum value of normalized expression data. -- print("Normalize",adata_sc.X.max())
# Line 26: Loads a Visium spatial transcriptomics dataset from scanpy. -- adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
# Line 27: Adds sample information to the obs attribute of the spatial transcriptomics data. -- adata.obs['sample'] = list(adata.uns['spatial'].keys())[0]
# Line 28: Makes the gene names unique in the spatial transcriptomics data. -- adata.var_names_make_unique()
# Line 30: Calculates quality control metrics for the spatial transcriptomics data using scanpy. -- sc.pp.calculate_qc_metrics(adata, inplace=True)
# Line 31: Filters the spatial transcriptomics data by total counts. -- adata = adata[:,adata.var['total_counts']>100]
# Line 32: Calculates spatial variable genes for the spatial transcriptomics data using omicverse. -- adata=ov.space.svg(adata,mode='prost',n_svgs=3000,target_sum=1e4,platform="visium",)
# Line 33: Stores the spatial data as raw attribute. -- adata.raw = adata
# Line 34: Subsets the spatial AnnData object to include only spatially variable genes. -- adata = adata[:, adata.var.space_variable_features]
# Line 35: Creates a copy of the processed spatial transcriptomics data. -- adata_sp=adata.copy()
# Line 36: Displays the spatial AnnData object. -- adata_sp
# Line 38: Initializes the Tangram object for spatial mapping using the single-cell and spatial data. -- tg=ov.space.Tangram(adata_sc,adata_sp,clusters='Subset')
# Line 40: Trains the Tangram model using the specified settings. -- tg.train(mode="clusters",num_epochs=500,device="cuda:0")
# Line 42: Performs cell-to-location mapping using the trained Tangram model. -- adata_plot=tg.cell2location()
# Line 43: Displays the column names of observation data for mapped object. -- adata_plot.obs.columns
# Line 45: Defines a list of cell type annotations for plotting. -- annotation_list=['B_Cycling', 'B_GC_LZ', 'T_CD4+_TfH_GC', 'FDC',
# Line 48: Generates and displays a spatial plot using scanpy, visualizing cell types. -- sc.pl.spatial(adata_plot, cmap='magma',
# Line 57: Creates a dictionary mapping the single-cell 'Subset' categories to colors. -- color_dict=dict(zip(adata_sc.obs['Subset'].cat.categories,
# Line 60: Imports the matplotlib module. -- import matplotlib as mpl
# Line 61: Creates a subset of cell types and transforms to string. -- clust_labels = annotation_list[:5]
# Line 62: Converts cell type labels to strings for column name compatibility. -- clust_col = ['' + str(i) for i in clust_labels] # in case column names differ from labels
# Line 64: Creates a context for matplotlib rc parameters for specific plot configurations. -- with mpl.rc_context({'figure.figsize': (8, 8),'axes.grid': False}):
# Line 65: Generates and displays a spatial plot using omicverse, visualizing cell types. -- fig = ov.pl.plot_spatial(
```
