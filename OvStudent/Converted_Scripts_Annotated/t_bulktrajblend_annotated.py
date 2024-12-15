```
# Line 1:  Import the omicverse library as ov -- import omicverse as ov
# Line 2:  Import the mde function from omicverse.utils -- from omicverse.utils import mde
# Line 3:  Import the scanpy library as sc -- import scanpy as sc
# Line 4:  Import the scvelo library as scv -- import scvelo as scv
# Line 5:  Set plotting style using ov.plot_set() -- ov.plot_set()
# Line 8:  Load the dentategyrus dataset from scvelo into adata -- adata=scv.datasets.dentategyrus()
# Line 9: Display the AnnData object -- adata
# Line 11: Import the numpy library as np -- import numpy as np
# Line 12: Read bulk RNA-seq data from a file using ov.utils.read -- bulk=ov.utils.read('data/GSE74985_mergedCount.txt.gz',index_col=0)
# Line 13: Map gene IDs in bulk data using ov.bulk.Matrix_ID_mapping -- bulk=ov.bulk.Matrix_ID_mapping(bulk,'genesets/pair_GRCm39.tsv')
# Line 14: Display the first few rows of the bulk data -- bulk.head()
# Line 16: Create a BulkTrajBlend object using bulk and single-cell data -- bulktb=ov.bulk2single.BulkTrajBlend(bulk_seq=bulk,single_seq=adata,
# Line 17: Specify bulk groups and cell type key for the BulkTrajBlend object --                                    bulk_group=['dg_d_1','dg_d_2','dg_d_3'],
# Line 18: Specify cell type key for the BulkTrajBlend object --                                    celltype_key='clusters',)
# Line 20: Configure the VAE model within BulkTrajBlend with 100 target cells -- bulktb.vae_configure(cell_target_num=100)
# Line 23: Train the VAE model using specific parameters and save it -- vae_net=bulktb.vae_train(
# Line 24: Set batch size to 512 --     batch_size=512,
# Line 25: Set learning rate to 1e-4 --     learning_rate=1e-4,
# Line 26: Set hidden size to 256 --     hidden_size=256,
# Line 27: Set the number of training epochs to 3500 --     epoch_num=3500,
# Line 28: Set the directory to save the VAE model --     vae_save_dir='data/bulk2single/save_model',
# Line 29: Set the filename to save the VAE model --     vae_save_name='dg_btb_vae',
# Line 30: Set the directory to save generated data --     generate_save_dir='data/bulk2single/output',
# Line 31: Set the filename to save generated data --     generate_save_name='dg_btb')
# Line 33: Load the pretrained VAE model from specified path -- bulktb.vae_load('data/bulk2single/save_model/dg_btb_vae.pth')
# Line 35: Generate new data using the loaded VAE and specified leiden size -- generate_adata=bulktb.vae_generate(leiden_size=25)
# Line 37: Plot the cell proportion after bulk-to-single-cell mapping using generate_adata -- ov.bulk2single.bulk2single_plot_cellprop(generate_adata,celltype_key='clusters',
# Line 38: Close parenthesis --                                         )
# Line 40: Configure the GNN model in BulkTrajBlend with specific parameters -- bulktb.gnn_configure(max_epochs=2000,use_rep='X',
# Line 41: Specify the neighbor representation for the GNN --                      neighbor_rep='X_pca')
# Line 43: Train the GNN model -- bulktb.gnn_train()
# Line 45: Load the trained GNN model from the specified path -- bulktb.gnn_load('save_model/gnn.pth')
# Line 47: Generate the results from the GNN model -- res_pd=bulktb.gnn_generate()
# Line 48: Display the first few rows of the GNN results -- res_pd.head()
# Line 50: Compute and store MDE coordinates in the 'X_mde' slot of the AnnData object -- bulktb.nocd_obj.adata.obsm["X_mde"] = mde(bulktb.nocd_obj.adata.obsm["X_pca"])
# Line 51: Plot the MDE embedding colored by clusters and nocd_n with specified parameters -- sc.pl.embedding(bulktb.nocd_obj.adata,basis='X_mde',color=['clusters','nocd_n'],wspace=0.4,
# Line 52: Specify palette --           palette=ov.utils.pyomic_palette())
# Line 54: Plot the MDE embedding for cells without '-' in 'nocd_n', colored by clusters and nocd_n -- sc.pl.embedding(bulktb.nocd_obj.adata[~bulktb.nocd_obj.adata.obs['nocd_n'].str.contains('-')],
# Line 55: Specify basis, color, and spacing for plot --                 basis='X_mde',
# Line 56: Specify color and spacing --            color=['clusters','nocd_n'],
# Line 57: Specify spacing and palette --            wspace=0.4,palette=sc.pl.palettes.default_102)
# Line 59: Print the number of raw cells -- print('raw cells: ',bulktb.single_seq.shape[0])
# Line 61: Interpolate cells based on 'OPC' -- adata1=bulktb.interpolation('OPC')
# Line 62: Print the number of interpolated cells -- print('interpolation cells: ',adata1.shape[0])
# Line 64: Store the raw data in the raw slot of adata1 -- adata1.raw = adata1
# Line 65: Identify highly variable genes in adata1 -- sc.pp.highly_variable_genes(adata1, min_mean=0.0125, max_mean=3, min_disp=0.5)
# Line 66: Subset adata1 to keep only highly variable genes -- adata1 = adata1[:, adata1.var.highly_variable]
# Line 67: Scale the gene expression data in adata1 -- sc.pp.scale(adata1, max_value=10)
# Line 69: Perform PCA on adata1 with 100 components -- sc.tl.pca(adata1, n_comps=100, svd_solver="auto")
# Line 71: Normalize the total counts in the original AnnData object -- sc.pp.normalize_total(adata, target_sum=1e4)
# Line 72: Apply log1p transformation to the normalized counts -- sc.pp.log1p(adata)
# Line 73: Store the raw data in the raw slot of adata -- adata.raw = adata
# Line 74: Identify highly variable genes in the original AnnData object -- sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# Line 75: Subset adata to keep only highly variable genes -- adata = adata[:, adata.var.highly_variable]
# Line 76: Scale the gene expression data in the original AnnData object -- sc.pp.scale(adata, max_value=10)
# Line 78: Perform PCA on the original AnnData object with 100 components -- sc.tl.pca(adata, n_comps=100, svd_solver="auto")
# Line 80: Compute and store MDE coordinates in the 'X_mde' slot of the original AnnData object -- adata.obsm["X_mde"] = mde(adata.obsm["X_pca"])
# Line 81: Compute and store MDE coordinates in the 'X_mde' slot of the interpolated AnnData object -- adata1.obsm["X_mde"] = mde(adata1.obsm["X_pca"])
# Line 83: Generate and display an MDE embedding plot for the original data, colored by clusters -- ov.utils.embedding(adata,
# Line 84: Specify basis, color, frameon, spacing, and palette for plot --                 basis='X_mde',
# Line 85: Specify color for plot --            color=['clusters'],
# Line 86: Specify frameon --                    frameon='small',
# Line 87: Specify spacing and palette --            wspace=0.4,palette=sc.pl.palettes.default_102)
# Line 89: Generate and display an MDE embedding plot for the interpolated data, colored by clusters -- ov.utils.embedding(adata1,
# Line 90: Specify basis, color, frameon, spacing, and palette for plot --                 basis='X_mde',
# Line 91: Specify color for plot --            color=['clusters'],
# Line 92: Specify frameon --                 frameon='small',
# Line 93: Specify spacing and palette --            wspace=0.4,palette=sc.pl.palettes.default_102)
# Line 95: Create a pyVIA object for the original data with specified parameters -- v0 = ov.single.pyVIA(adata=adata,adata_key='X_pca',adata_ncomps=100, basis='X_mde',
# Line 96: Specify clusters, knn, random seed, root user, and dataset --                          clusters='clusters',knn=20,random_seed=4,root_user=['nIPC'],
# Line 97: Specify dataset --                     dataset='group')
# Line 98: Run the pyVIA analysis for the original data -- v0.run()
# Line 100: Create a pyVIA object for the interpolated data with specified parameters -- v1 = ov.single.pyVIA(adata=adata1,adata_key='X_pca',adata_ncomps=100, basis='X_mde',
# Line 101: Specify clusters, knn, random seed, root user and dataset --                          clusters='clusters',knn=15,random_seed=4,root_user=['Neuroblast'],
# Line 103: Specify dataset --                     dataset='group')
# Line 105: Run the pyVIA analysis for the interpolated data -- v1.run()
# Line 107: Import the matplotlib.pyplot module as plt -- import matplotlib.pyplot as plt
# Line 108: Create and display a stream plot for the original data -- fig,ax=v0.plot_stream(basis='X_mde',clusters='clusters',
# Line 109: Set plotting parameters for the stream plot --                density_grid=0.8, scatter_size=30, scatter_alpha=0.3, linewidth=0.5)
# Line 110: Set the title of the stream plot for the original data -- plt.title('Raw Dentategyrus',fontsize=12)
# Line 113: Create and display a stream plot for the interpolated data -- fig,ax=v1.plot_stream(basis='X_mde',clusters='clusters',
# Line 114: Set plotting parameters for the stream plot --                density_grid=0.8, scatter_size=30, scatter_alpha=0.3, linewidth=0.5)
# Line 115: Set the title of the stream plot for the interpolated data -- plt.title('Interpolation Dentategyrus',fontsize=12)
# Line 118: Create and display a stream plot of pseudo time for the original data -- fig,ax=v0.plot_stream(basis='X_mde',density_grid=0.8, scatter_size=30, color_scheme='time', linewidth=0.5,
# Line 119: Set plotting parameters for pseudo time stream plot --                              min_mass = 1, cutoff_perc = 5, scatter_alpha=0.3, marker_edgewidth=0.1,
# Line 120: Set plotting parameters for pseudo time stream plot --                             density_stream = 2, smooth_transition=1, smooth_grid=0.5)
# Line 121: Set the title of the pseudo time stream plot for the original data -- plt.title('Raw Dentategyrus\nPseudoTime',fontsize=12)
# Line 123: Create and display a stream plot of pseudo time for the interpolated data -- fig,ax=v1.plot_stream(basis='X_mde',density_grid=0.8, scatter_size=30, color_scheme='time', linewidth=0.5,
# Line 124: Set plotting parameters for pseudo time stream plot --                              min_mass = 1, cutoff_perc = 5, scatter_alpha=0.3, marker_edgewidth=0.1,
# Line 125: Set plotting parameters for pseudo time stream plot --                             density_stream = 2, smooth_transition=1, smooth_grid=0.5)
# Line 126: Set the title of the pseudo time stream plot for the interpolated data -- plt.title('Interpolation Dentategyru\nPseudoTime',fontsize=12)
# Line 128: Compute pseudotime using pyVIA and store in the original AnnData -- v0.get_pseudotime(adata)
# Line 129: Compute neighbors using PCA embeddings for the original AnnData -- sc.pp.neighbors(adata,n_neighbors= 15,use_rep='X_pca')
# Line 130: Calculate PAGA graph using pseudotime as prior for the original AnnData -- ov.utils.cal_paga(adata,use_time_prior='pt_via',vkey='paga',
# Line 131: Specify group --                  groups='clusters')
# Line 133: Generate and display a PAGA graph for the original AnnData -- ov.utils.plot_paga(adata,basis='mde', size=50, alpha=.1,title='PAGA LTNN-graph',
# Line 134: Set plotting parameters --             min_edge_width=2, node_size_scale=1.5,show=False,legend_loc=False)
# Line 136: Compute pseudotime using pyVIA and store in the interpolated AnnData -- v1.get_pseudotime(adata1)
# Line 137: Compute neighbors using PCA embeddings for the interpolated AnnData -- sc.pp.neighbors(adata1,n_neighbors= 15,use_rep='X_pca')
# Line 138: Calculate PAGA graph using pseudotime as prior for the interpolated AnnData -- ov.utils.cal_paga(adata1,use_time_prior='pt_via',vkey='paga',
# Line 139: Specify group --                  groups='clusters')
# Line 141: Generate and display a PAGA graph for the interpolated AnnData -- ov.utils.plot_paga(adata1,basis='mde', size=50, alpha=.1,title='PAGA LTNN-graph',
# Line 142: Set plotting parameters --             min_edge_width=2, node_size_scale=1.5,show=False,legend_loc=False)
```
