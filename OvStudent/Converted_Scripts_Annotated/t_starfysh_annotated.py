```
# Line 1:  # Line 1: Import the scanpy library for single-cell analysis. -- import scanpy as sc
# Line 2:  # Line 2: Import the omicverse library for multi-omics analysis. -- import omicverse as ov
# Line 3:  # Line 3: Set plotting parameters using omicverse. -- ov.plot_set()
# Line 5:  # Line 5: Import specific modules from the starfysh package within omicverse. -- from omicverse.externel.starfysh import (AA, utils, plot_utils, post_analysis)
# Line 6:  # Line 6: Import the starfysh model implementation. -- from omicverse.externel.starfysh import _starfysh as sf_model
# Line 8:  # Line 8: Specify the path to the data directory. -- data_path = 'data/star_data'
# Line 9:  # Line 9: Specify the sample ID. -- sample_id = 'CID44971_TNBC'
# Line 10: # Line 10: Specify the name of the signature gene file. -- sig_name = 'bc_signatures_version_1013.csv'
# Line 12: # Line 12: Load the AnnData object and normalized data using the specified paths and sample id, keeping 2000 highly variable genes. -- adata, adata_normed = utils.load_adata(data_folder=data_path,
# Line 13: # Line 13: Specify the sample id (comment). --                                       sample_id=sample_id, # sample id
# Line 14: # Line 14: Specify the number of highly variable genes to keep. --                                       n_genes=2000  # number of highly variable genes to keep
# Line 16: # Line 16: Import the pandas library for data manipulation. -- import pandas as pd
# Line 17: # Line 17: Import the os library for file system interactions. -- import os
# Line 18: # Line 18: Read the gene signature file into a pandas DataFrame. -- gene_sig = pd.read_csv(os.path.join(data_path, sig_name))
# Line 19: # Line 19: Filter the gene signature DataFrame based on genes present in the AnnData object. -- gene_sig = utils.filter_gene_sig(gene_sig, adata.to_df())
# Line 20: # Line 20: Display the head of the gene signature DataFrame. -- gene_sig.head()
# Line 22: # Line 22: Preprocess the image data, extracting spatial information. -- img_metadata = utils.preprocess_img(data_path,
# Line 23: # Line 23: Pass the sample ID for preprocessing. --                                    sample_id,
# Line 24: # Line 24: Pass the adata index for preprocessing. --                                    adata_index=adata.obs.index,
# Line 26: # Line 26: Extract the image, map information, and scaling factor from the processed metadata. -- img, map_info, scalefactor = img_metadata['img'], img_metadata['map_info'], img_metadata['scalefactor']
# Line 27: # Line 27: Calculate the UMAP embeddings for the AnnData object. -- umap_df = utils.get_umap(adata, display=True)
# Line 30: # Line 30: Import the matplotlib library for plotting. -- import matplotlib.pyplot as plt
# Line 31: # Line 31: Create a new figure with a specific size and resolution for the image. -- plt.figure(figsize=(6, 6), dpi=80)
# Line 32: # Line 32: Display the loaded image. -- plt.imshow(img)
# Line 34: # Line 34: Display the head of the spatial mapping information DataFrame. -- map_info.head()
# Line 36: # Line 36: Define the arguments for the Visium analysis, including adata, gene signatures, and spatial data. -- visium_args = utils.VisiumArguments(adata,
# Line 37: # Line 37: Include the normalized adata. --                                    adata_normed,
# Line 38: # Line 38: Include the gene signatures. --                                    gene_sig,
# Line 39: # Line 39: Include the img_metadata. --                                    img_metadata,
# Line 40: # Line 40: Specify the number of anchor spots. --                                    n_anchors=60,
# Line 41: # Line 41: Specify the window size for spatial analysis. --                                    window_size=3,
# Line 42: # Line 42: Specify the sample ID. --                                    sample_id=sample_id
# Line 44: # Line 44: Get the modified AnnData and normalized data using the VisiumArguments object. -- adata, adata_normed = visium_args.get_adata()
# Line 45: # Line 45: Get the anchor spot DataFrame using the VisiumArguments object. -- anchors_df = visium_args.get_anchors()
# Line 47: # Line 47: Add a log library size column to the AnnData's observation data using the VisiumArguments object. -- adata.obs['log library size']=visium_args.log_lib
# Line 48: # Line 48: Add a windowed log library size column to the AnnData's observation data using the VisiumArguments object. -- adata.obs['windowed log library size']=visium_args.win_loglib
# Line 50: # Line 50: Plot the spatial distribution of 'log library size' using scanpy. -- sc.pl.spatial(adata, cmap='magma',
# Line 52: # Line 52: Specify the feature to color by, and the number of columns. --                   color='log library size',
# Line 53: # Line 53: Specify plot parameters. --                   ncols=4, size=1.3,
# Line 54: # Line 54: Specify image key. --                   img_key='hires',
# Line 59: # Line 59: Plot the spatial distribution of 'windowed log library size' using scanpy. -- sc.pl.spatial(adata, cmap='magma',
# Line 61: # Line 61: Specify the feature to color by, and the number of columns. --                   color='windowed log library size',
# Line 62: # Line 62: Specify plot parameters. --                   ncols=4, size=1.3,
# Line 63: # Line 63: Specify image key. --                   img_key='hires',
# Line 68: # Line 68: Plot the spatial distribution of 'IL7R' gene expression using scanpy. -- sc.pl.spatial(adata, cmap='magma',
# Line 70: # Line 70: Specify the feature to color by, and the number of columns. --                   color='IL7R',
# Line 71: # Line 71: Specify plot parameters. --                   ncols=4, size=1.3,
# Line 72: # Line 72: Specify image key. --                   img_key='hires',
# Line 77: # Line 77: Plot the anchor spots and their corresponding signatures using the plot_utils. -- plot_utils.plot_anchor_spots(umap_df,
# Line 78: # Line 78: Pass pure spots. --                              visium_args.pure_spots,
# Line 79: # Line 79: Pass the signature means. --                              visium_args.sig_mean,
# Line 80: # Line 80: Specify the bounding box x coordinate. --                              bbox_x=2
# Line 82: # Line 82: Initialize an ArchetypalAnalysis object using the normalized AnnData object. -- aa_model = AA.ArchetypalAnalysis(adata_orig=adata_normed)
# Line 83: # Line 83: Compute archetypes and return archetypal scores, dictionary of archetypes, major index, and explained variance. -- archetype, arche_dict, major_idx, evs = aa_model.compute_archetypes(cn=40)
# Line 85: # Line 85: Find the archetypal spots, using major archetypes. -- arche_df = aa_model.find_archetypal_spots(major=True)
# Line 87: # Line 87: Find marker genes associated with each archetypal cluster. -- markers_df = aa_model.find_markers(n_markers=30, display=False)
# Line 89: # Line 89: Map the archetypes to the closest anchors. -- map_df, map_dict = aa_model.assign_archetypes(anchors_df)
# Line 91: # Line 91: Find the most distant archetypes that are not assigned to any annotated cell types. -- distant_arches = aa_model.find_distant_archetypes(anchors_df, n=3)
# Line 93: # Line 93: Plot the explained variance ratios. -- plot_utils.plot_evs(evs, kmin=aa_model.kmin)
# Line 95: # Line 95: Plot the archetypes. -- aa_model.plot_archetypes(do_3d=False, major=True, disp_cluster=False)
# Line 97: # Line 97: Plot the archetype mapping results. -- aa_model.plot_mapping(map_df)
# Line 99: # Line 99: Refine the anchor spots based on the results of archetypal analysis. -- visium_args = utils.refine_anchors(
# Line 100: # Line 100: Pass visium arguments. --     visium_args,
# Line 101: # Line 101: Pass archetypal analysis model. --     aa_model,
# Line 103: # Line 103: Specify number of genes. --     n_genes=5,
# Line 106: # Line 106: Get the updated AnnData object and normalized data. -- adata, adata_normed = visium_args.get_adata()
# Line 107: # Line 107: Get the updated gene signatures. -- gene_sig = visium_args.gene_sig
# Line 108: # Line 108: Get the cell type names from gene signature. -- cell_types = gene_sig.columns
# Line 110: # Line 110: Import the torch library for deep learning. -- import torch
# Line 111: # Line 111: Specify the number of repeats for model training. -- n_repeats = 3
# Line 112: # Line 112: Specify the number of epochs for model training. -- epochs = 200
# Line 113: # Line 113: Specify the patience for early stopping. -- patience = 50
# Line 114: # Line 114: Set the device to GPU if available, otherwise use CPU. -- device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Line 116: # Line 116: Run the starfysh model training using the provided parameters. -- model, loss = utils.run_starfysh(visium_args,
# Line 117: # Line 117: Specify the number of repeats for model training. --                                  n_repeats=n_repeats,
# Line 118: # Line 118: Specify the number of epochs. --                                  epochs=epochs,
# Line 120: # Line 120: Pass the device. --                                  device=device
# Line 122: # Line 122: Get updated adata objects after training. -- adata, adata_normed = visium_args.get_adata()
# Line 123: # Line 123: Evaluate the trained starfysh model and get inference and generative outputs. -- inference_outputs, generative_outputs,adata_ = sf_model.model_eval(model,
# Line 124: # Line 124: Pass the adata object. --                                                            adata,
# Line 125: # Line 125: Pass the visium arguments. --                                                            visium_args,
# Line 126: # Line 126: Specify poe. --                                                            poe=False,
# Line 127: # Line 127: Pass the device to use. --                                                            device=device)
# Line 129: # Line 129: Import the numpy library for numerical operations. -- import numpy as np
# Line 130: # Line 130: Get the number of cell types from the gene signature. -- n_cell_types = gene_sig.shape[1]
# Line 131: # Line 131: Select a random index for cell type plotting. -- idx = np.random.randint(0, n_cell_types)
# Line 132: # Line 132: Plot the mean expression versus inferred proportion for a random cell type. -- post_analysis.gene_mean_vs_inferred_prop(inference_outputs,
# Line 133: # Line 133: Pass the VisiumArguments. --                                         visium_args,
# Line 134: # Line 134: Pass the index. --                                         idx=idx,
# Line 135: # Line 135: Specify figure size. --                                         figsize=(4,4)
# Line 137: # Line 137: Plot the spatial distribution of inferred expression for the 'ql_m' feature. -- plot_utils.pl_spatial_inf_feature(adata_, feature='ql_m', cmap='Blues')
# Line 139: # Line 139: Define a function to convert cell data to proportion data. -- def cell2proportion(adata):
# Line 140: # Line 140: Create a new AnnData object for plotting using the expression matrix of the given adata. --     adata_plot=sc.AnnData(adata.X)
# Line 141: # Line 141: Copy the observation data to the new AnnData object. --     adata_plot.obs=utils.extract_feature(adata_, 'qc_m').obs.copy()
# Line 142: # Line 142: Copy the variable data. --     adata_plot.var=adata.var.copy()
# Line 143: # Line 143: Copy the observation matrix. --     adata_plot.obsm=adata.obsm.copy()
# Line 144: # Line 144: Copy the observation pair wise data. --     adata_plot.obsp=adata.obsp.copy()
# Line 145: # Line 145: Copy the unstructured data. --     adata_plot.uns=adata.uns.copy()
# Line 146: # Line 146: Return the new AnnData object. --     return adata_plot
# Line 147: # Line 147: Convert the adata_ object to a proportion object by calling cell2proportion function. -- adata_plot=cell2proportion(adata_)
# Line 149: # Line 149: Show the adata_plot object. -- adata_plot
# Line 151: # Line 151: Plot the spatial distribution of Basal, LumA, LumB. -- sc.pl.spatial(adata_plot, cmap='Spectral_r',
# Line 153: # Line 153: Specify the features to color by and number of columns. --                   color=['Basal','LumA','LumB'],
# Line 154: # Line 154: Specify plot parameters. --                   ncols=4, size=1.3,
# Line 155: # Line 155: Specify the image key. --                   img_key='hires',
# Line 156: # Line 156: Specify the min and max value for coloring. --                   vmin=0, vmax='p90'
# Line 159: # Line 159: Plot UMAP embeddings colored by the expression of Basal, LumA, MBC, and Normal epithelial. -- ov.pl.embedding(adata_plot,
# Line 160: # Line 160: Specify the basis. --                basis='z_umap',
# Line 161: # Line 161: Specify the features to color by. --                 color=['Basal', 'LumA', 'MBC', 'Normal epithelial'],
# Line 162: # Line 162: Specify frameon parameter. --                frameon='small',
# Line 163: # Line 163: Specify the min and max values, as well as cmap. --                 vmin=0, vmax='p90',
# Line 164: # Line 164: Specify the color map. --                cmap='Spectral_r',
# Line 167: # Line 167: Predict cell type-specific expression using the trained model. -- pred_exprs = sf_model.model_ct_exp(model,
# Line 168: # Line 168: Pass the adata object. --                                   adata,
# Line 169: # Line 169: Pass the visium arguments. --                                   visium_args,
# Line 170: # Line 170: Pass the device. --                                   device=device)
# Line 172: # Line 172: Specify the gene and celltype for visualization. -- gene='IL7R'
# Line 173: # Line 173: Specify the gene and celltype for visualization. -- gene_celltype='Tem'
# Line 174: # Line 174: Add inferred expression of specified gene/cell type to adata_.layers. -- adata_.layers[f'infer_{gene_celltype}']=pred_exprs[gene_celltype]
# Line 176: # Line 176: Plot the spatial distribution of the predicted 'IL7R' expression. -- sc.pl.spatial(adata_, cmap='Spectral_r',
# Line 178: # Line 178: Specify the color to use for plotting, plot title, and layer name. --                   color=gene,
# Line 179: # Line 179: Specify the title for the plot. --                   title=f'{gene} (Predicted expression)\n{gene_celltype}',
# Line 180: # Line 180: Specify the layer to color by. --                   layer=f'infer_{gene_celltype}',
# Line 181: # Line 181: Specify plot parameters. --                   ncols=4, size=1.3,
# Line 182: # Line 182: Specify image key. --                   img_key='hires',
# Line 187: # Line 187: Specify the output directory. -- outdir = './results/'
# Line 188: # Line 188: Create the output directory if it doesn't exist. -- if not os.path.exists(outdir):
# Line 189: # Line 189: Create output directory. --     os.mkdir(outdir)
# Line 191: # Line 191: Save the trained model's state dictionary to disk. -- torch.save(model.state_dict(), os.path.join(outdir, 'starfysh_model.pt'))
# Line 193: # Line 193: Save the AnnData object to disk in h5ad format. -- adata.write(os.path.join(outdir, 'st.h5ad'))
```