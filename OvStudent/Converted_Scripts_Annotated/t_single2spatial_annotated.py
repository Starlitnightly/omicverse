```
# Line 1:  Import the scanpy library for single-cell analysis. -- import scanpy as sc
# Line 2:  Import the pandas library for data manipulation. -- import pandas as pd
# Line 3:  Import the numpy library for numerical operations. -- import numpy as np
# Line 4:  Import the omicverse library for spatial omics analysis. -- import omicverse as ov
# Line 5:  Import the matplotlib.pyplot library for plotting. -- import matplotlib.pyplot as plt
# Line 7:  Set the plotting style using omicverse utilities. -- ov.utils.ov_plot_set()
# Line 9: Import the anndata library for handling annotated data objects. -- import anndata
# Line 10: Read single-cell gene expression data from a CSV file into a pandas DataFrame. -- raw_data=pd.read_csv('data/pdac/sc_data.csv', index_col=0)
# Line 11: Create an AnnData object from the transposed single-cell gene expression DataFrame. -- single_data=anndata.AnnData(raw_data.T)
# Line 12: Read single-cell metadata and assign the 'Cell_type' column to the AnnData object's obs attribute. -- single_data.obs = pd.read_csv('data/pdac/sc_meta.csv', index_col=0)[['Cell_type']]
# Line 13: Display the single_data AnnData object. -- single_data
# Line 15: Read spatial transcriptomics gene expression data from a CSV file into a pandas DataFrame. -- raw_data=pd.read_csv('data/pdac/st_data.csv', index_col=0)
# Line 16: Create an AnnData object from the transposed spatial transcriptomics gene expression DataFrame. -- spatial_data=anndata.AnnData(raw_data.T)
# Line 17: Read spatial transcriptomics metadata and assign it to the AnnData object's obs attribute. -- spatial_data.obs = pd.read_csv('data/pdac/st_meta.csv', index_col=0)
# Line 18: Display the spatial_data AnnData object. -- spatial_data
# Line 20: Initialize the Single2Spatial model from omicverse to integrate single-cell and spatial data. -- st_model=ov.bulk2single.Single2Spatial(single_data=single_data,
# Line 21: Specify the spatial data for the Single2Spatial model. -- spatial_data=spatial_data,
# Line 22: Specify the cell type annotation key. -- celltype_key='Cell_type',
# Line 23: Specify the spot coordinate keys. -- spot_key=['xcoord','ycoord'],
# Line 27: Train the Single2Spatial model and create an AnnData object containing spatial predictions. -- sp_adata=st_model.train(spot_num=500,
# Line 28: Specify the cell number parameter for training. -- cell_num=10,
# Line 29: Specify the directory for saving the model. -- df_save_dir='data/pdac/predata_net/save_model',
# Line 30: Specify the file name for saving the model. -- df_save_name='pdac_df',
# Line 31: Specify training parameters k, num_epochs, batch_size, and predicted_size. -- k=10,num_epochs=1000,batch_size=1000,predicted_size=32)
# Line 34: Load a pre-trained Single2Spatial model from a saved file. -- sp_adata=st_model.load(modelsize=14478,df_load_dir='data/pdac/predata_net/save_model/pdac_df.pth',
# Line 35: Specify loading parameters k and predicted_size for the model. -- k=10,predicted_size=32)
# Line 37: Perform spatial spot assessment using the trained model. -- sp_adata_spot=st_model.spot_assess()
# Line 39: Create a spatial embedding plot for gene expression using scanpy's embedding function. -- sc.pl.embedding(
# Line 40: Specify the spatial embedding basis. -- sp_adata,
# Line 41: Specify the genes to color the embedding plot by. -- basis="X_spatial",
# Line 42: Turn off frame and set the number of columns for the plot. -- color=['REG1A', 'CLDN1', 'KRT16', 'MUC5B'],
# Line 43: Display the plot. -- frameon=False,
# Line 44: Turn off displaying plot. -- ncols=4,
# Line 49: Create a spatial embedding plot for spatial spots using scanpy's embedding function. -- sc.pl.embedding(
# Line 50: Specify the spatial embedding basis. -- sp_adata_spot,
# Line 51: Specify the genes to color the embedding plot by. -- basis="X_spatial",
# Line 52: Turn off frame and set the number of columns for the plot. -- color=['REG1A', 'CLDN1', 'KRT16', 'MUC5B'],
# Line 53: Turn off displaying plot. -- frameon=False,
# Line 54: Turn off displaying plot. -- ncols=4,
# Line 59: Create a spatial embedding plot for cell types in spatial spots using scanpy's embedding function. -- sc.pl.embedding(
# Line 60: Specify the spatial embedding basis. -- sp_adata_spot,
# Line 61: Specify the cell types to color the embedding plot by. -- basis="X_spatial",
# Line 62: Turn off frame and set the number of columns for the plot. -- color=['Acinar cells','Cancer clone A','Cancer clone B','Ductal'],
# Line 63: Turn off frame and set the number of columns for the plot. -- frameon=False,
# Line 64: Turn off displaying plot. -- ncols=4,
# Line 65: Turn off displaying plot. -- show=False,
# Line 70: Create a spatial embedding plot for cell types using scanpy's embedding function. -- sc.pl.embedding(
# Line 71: Specify the spatial embedding basis. -- sp_adata,
# Line 72: Specify the cell type annotation to color the embedding plot by. -- basis="X_spatial",
# Line 73: Turn off frame and set the number of columns for the plot. -- color=['Cell_type'],
# Line 74: Turn off frame and set the number of columns for the plot. -- frameon=False,
# Line 75: Turn off displaying plot. -- ncols=4,
# Line 76: Use a specific color palette from omicverse for the plot. -- show=False,
# Line 77: Use a specific color palette from omicverse for the plot. -- palette=ov.utils.ov_palette()[11:]
```
