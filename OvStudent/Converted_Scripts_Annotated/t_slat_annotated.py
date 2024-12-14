```
# Line 1: Import the omicverse library as ov. -- import omicverse as ov
# Line 2: Import the os module. -- import os
# Line 3: Import the scanpy library as sc. -- import scanpy as sc
# Line 4: Import the numpy library as np. -- import numpy as np
# Line 5: Import the pandas library as pd. -- import pandas as pd
# Line 6: Import the torch library. -- import torch
# Line 7: Set the plot style using omicverse. -- ov.plot_set()
# Line 9: Import specific modules from the omicverse external scSLAT library. -- from omicverse.externel.scSLAT.model import load_anndatas, Cal_Spatial_Net, run_SLAT, scanpy_workflow, spatial_match
# Line 10: Import specific visualization modules from the omicverse external scSLAT library. -- from omicverse.externel.scSLAT.viz import match_3D_multi, hist, Sankey, match_3D_celltype, Sankey,Sankey_multi,build_3D
# Line 11: Import the region_statistics module from the omicverse external scSLAT library. -- from omicverse.externel.scSLAT.metrics import region_statistics
# Line 12: Read the first AnnData object from an H5AD file. -- adata1 = sc.read_h5ad('data/E115_Stereo.h5ad')
# Line 13: Read the second AnnData object from an H5AD file. -- adata2 = sc.read_h5ad('data/E125_Stereo.h5ad')
# Line 14: Add a 'week' observation to adata1, setting it to 'E11.5'. -- adata1.obs['week']='E11.5'
# Line 15: Add a 'week' observation to adata2, setting it to 'E12.5'. -- adata2.obs['week']='E12.5'
# Line 16: Generate a spatial plot for adata1 colored by 'annotation' with a spot size of 3. -- sc.pl.spatial(adata1, color='annotation', spot_size=3)
# Line 17: Generate a spatial plot for adata2 colored by 'annotation' with a spot size of 3. -- sc.pl.spatial(adata2, color='annotation', spot_size=3)
# Line 18: Calculate the spatial network for adata1 using KNN with a k_cutoff of 20. -- Cal_Spatial_Net(adata1, k_cutoff=20, model='KNN')
# Line 19: Calculate the spatial network for adata2 using KNN with a k_cutoff of 20. -- Cal_Spatial_Net(adata2, k_cutoff=20, model='KNN')
# Line 20: Load edges and features from a list of AnnData objects, using DPCA features and not checking order. -- edges, features = load_anndatas([adata1, adata2], feature='DPCA', check_order=False)
# Line 21: Run the SLAT algorithm to get embeddings and time information. -- embd0, embd1, time = run_SLAT(features, edges, LGCN_layer=5)
# Line 22: Perform spatial matching between the embeddings of the two timepoints, not reordering, returning best match, index and distances. -- best, index, distance = spatial_match([embd0, embd1], reorder=False, adatas=[adata1,adata2])
# Line 23: Create a numpy array representing the matching pairs between the two timepoints. -- matching = np.array([range(index.shape[0]), best])
# Line 24: Extract the best match distances from the distance matrix. -- best_match = distance[:,0]
# Line 25: Calculate and print region statistics based on the matching distances. -- region_statistics(best_match, start=0.5, number_of_interval=10)
# Line 27: Import the matplotlib.pyplot module as plt. -- import matplotlib.pyplot as plt
# Line 28: Create a list containing the matching information. -- matching_list=[matching]
# Line 29: Build a 3D model using the two AnnData objects and matching list. -- model = build_3D([adata1,adata2], matching_list,subsample_size=300, )
# Line 30: Draw a 3D visualization of the model, hiding axes. -- ax=model.draw_3D(hide_axis=True, line_color='#c2c2c2', height=1, size=[6,6], line_width=1)
# Line 32: Add low quality index as obs to adata2 -- adata2.obs['low_quality_index']= best_match
# Line 33: Convert low quality index to float -- adata2.obs['low_quality_index'] = adata2.obs['low_quality_index'].astype(float)
# Line 35: Access spatial coordinates stored in adata2.obsm. -- adata2.obsm['spatial']
# Line 36: Generate a spatial plot of adata2, coloring by the 'low_quality_index', spot size 3, with the title "Quality". -- sc.pl.spatial(adata2, color='low_quality_index', spot_size=3, title='Quality')
# Line 38: Create a Sankey plot for the two AnnData objects based on annotation and given matching. -- fig=Sankey_multi(adata_li=[adata1,adata2],
# Line 39: Set the prefixes for each object to E11.5, E12.5. --              prefix_li=['E11.5','E12.5'],
# Line 40: Set the matching list to the previously defined matching array. --              matching_li=[matching],
# Line 41: Set the clusters to annotation and filter number to 10. --                 clusters='annotation',filter_num=10,
# Line 42: Set node opacity to 0.8 --              node_opacity = 0.8,
# Line 43: Set link opacity to 0.2 --              link_opacity = 0.2,
# Line 44: Set layout to specified size. --                 layout=[800,500],
# Line 45: Set font size to 12 --            font_size=12,
# Line 46: Set font color to black. --            font_color='Black',
# Line 47: Set save name to none. --            save_name=None,
# Line 48: Set format to png. --            format='png',
# Line 49: Set width to 1200. --            width=1200,
# Line 50: Set height to 1000. --            height=1000,
# Line 51: Return the figure object. --            return_fig=True)
# Line 52: Display the created Sankey plot. -- fig.show()
# Line 54: Save the Sankey plot as an HTML file. -- fig.write_html("slat_sankey.html")
# Line 56: Create a color dictionary for adata1's annotations. -- color_dict1=dict(zip(adata1.obs['annotation'].cat.categories,
# Line 57: Map the colors to adata1's annotations. --                     adata1.uns['annotation_colors'].tolist()))
# Line 58: Create a pandas DataFrame for adata1 containing spatial information and celltype information. -- adata1_df = pd.DataFrame({'index':range(embd0.shape[0]),
# Line 59: Get x spatial coordinates from adata1 --                           'x': adata1.obsm['spatial'][:,0],
# Line 60: Get y spatial coordinates from adata1 --                           'y': adata1.obsm['spatial'][:,1],
# Line 61: Get celltype information from adata1 --                           'celltype':adata1.obs['annotation'],
# Line 62: Get color based on the celltype for each cell in adata1. --                          'color':adata1.obs['annotation'].map(color_dict1)
# Line 63: End of the dataframe declaration for adata1 --                         }
# Line 64: Create a color dictionary for adata2's annotations. -- color_dict2=dict(zip(adata2.obs['annotation'].cat.categories,
# Line 65: Map the colors to adata2's annotations. --                     adata2.uns['annotation_colors'].tolist()))
# Line 66: Create a pandas DataFrame for adata2 containing spatial information and celltype information. -- adata2_df = pd.DataFrame({'index':range(embd1.shape[0]),
# Line 67: Get x spatial coordinates from adata2 --                           'x': adata2.obsm['spatial'][:,0],
# Line 68: Get y spatial coordinates from adata2 --                           'y': adata2.obsm['spatial'][:,1],
# Line 69: Get celltype information from adata2 --                           'celltype':adata2.obs['annotation'],
# Line 70: Get color based on the celltype for each cell in adata2. --                          'color':adata2.obs['annotation'].map(color_dict2)
# Line 71: End of the dataframe declaration for adata2 --                         }
# Line 73: Create a 3D celltype-specific alignment visualization. -- kidney_align = match_3D_celltype(adata1_df, adata2_df, matching, meta='celltype', 
# Line 74: Highlight specific cell types during the alignment visualization. --                                  highlight_celltype = [['Urogenital ridge'],['Kidney','Ovary']],
# Line 75: Set the subsample size for the alignment to 10000, the highlight line color to blue and to scale the coordinate. --                                  subsample_size=10000, highlight_line = ['blue'], scale_coordinate = True )
# Line 76: Draw the 3D alignment visualization, specifying size, line width, point sizes, and hiding axes. -- kidney_align.draw_3D(size= [6, 6], line_width =0.8, point_size=[0.6,0.6], hide_axis=True)
# Line 78: Define a function to calculate matching cells based on a specific query cell. -- def cal_matching_cell(target_adata,query_adata,matching,query_cell,clusters='annotation',):
# Line 79: Create a DataFrame for target_adata containing spatial information and celltype information. --     adata1_df = pd.DataFrame({'index':range(target_adata.shape[0]),
# Line 80: Get x spatial coordinates from target_adata. --                           'x': target_adata.obsm['spatial'][:,0],
# Line 81: Get y spatial coordinates from target_adata. --                           'y': target_adata.obsm['spatial'][:,1],
# Line 82: Get celltype information from target_adata based on given cluster. --                           'celltype':target_adata.obs[clusters]})
# Line 83: Create a DataFrame for query_adata containing spatial information and celltype information. --     adata2_df = pd.DataFrame({'index':range(query_adata.shape[0]),
# Line 84: Get x spatial coordinates from query_adata. --                               'x': query_adata.obsm['spatial'][:,0],
# Line 85: Get y spatial coordinates from query_adata. --                               'y': query_adata.obsm['spatial'][:,1],
# Line 86: Get celltype information from query_adata based on given cluster. --                               'celltype':query_adata.obs[clusters]})
# Line 87: Create a new anndata based on matching of the celltype in query_cell in the query_adata based on the matching from the target adata --     query_adata = target_adata[matching[1,adata2_df.loc[adata2_df.celltype==query_cell,'index'].values],:]
# Line 88: Commented out code which would add the target cell type and index to the query adata dataframe. --     #adata2_df['target_celltype'] = adata1_df.iloc[matching[1,:],:]['celltype'].to_list()
# Line 89: Commented out code which would add the target cell type and index to the query adata dataframe. --     #adata2_df['target_obs_names'] = adata1_df.iloc[matching[1,:],:].index.to_list()
# Line 91: Returns the query adata containing the matched cells. --     return query_adata
# Line 94: Call cal_matching_cell to extract the target cells corresponding to 'Kidney' cells in the second adata. -- query_adata=cal_matching_cell(target_adata=adata1,
# Line 95: Pass adata2 as the query adata to the cal_matching_cell. --                               query_adata=adata2,
# Line 96: Pass the matching array to the cal_matching_cell. --                               matching=matching,
# Line 97: Specify the query cell as Kidney. --                               query_cell='Kidney',clusters='annotation')
# Line 98: Returns the query_adata. -- query_adata
# Line 100: Initialize the column 'kidney_anno' in adata1 to empty strings. -- adata1.obs['kidney_anno']=''
# Line 101: Set the 'kidney_anno' of the matching cell in adata1 according to its annotation in the corresponding cell from query_adata. -- adata1.obs.loc[query_adata.obs.index,'kidney_anno']=query_adata.obs['annotation']
# Line 103: Generate a spatial plot of adata1, coloring by 'kidney_anno', spot size 3, using a specified palette. -- sc.pl.spatial(adata1, color='kidney_anno', spot_size=3,
# Line 104: Specify the palette for the spatial plot. --              palette=['#F5F5F5','#ff7f0e', 'green',])
# Line 106: Concatenate the query_adata with the kidney cell from adata2 and combine the anndata objects. -- kidney_lineage_ad=sc.concat([query_adata,adata2[adata2.obs['annotation']=='Kidney']],merge='same')
# Line 107: Preprocess the concatenated AnnData object using shiftlog|pearson method, HVGs=3000 and target_sum=1e4. -- kidney_lineage_ad=ov.pp.preprocess(kidney_lineage_ad,mode='shiftlog|pearson',n_HVGs=3000,target_sum=1e4)
# Line 108: Store the original count data in raw object. -- kidney_lineage_ad.raw = kidney_lineage_ad
# Line 109: Select the highly variable features from the AnnData object. -- kidney_lineage_ad = kidney_lineage_ad[:, kidney_lineage_ad.var.highly_variable_features]
# Line 110: Scale the data in the AnnData object. -- ov.pp.scale(kidney_lineage_ad)
# Line 111: Perform PCA on the scaled data. -- ov.pp.pca(kidney_lineage_ad)
# Line 112: Calculate neighbors using scaled,original,X_pca and cosine distance. -- ov.pp.neighbors(kidney_lineage_ad,use_rep='scaled|original|X_pca',metric="cosine")
# Line 113: Perform Leiden clustering on the AnnData object. -- ov.utils.cluster(kidney_lineage_ad,method='leiden',resolution=1)
# Line 114: Calculate UMAP embeddings. -- ov.pp.umap(kidney_lineage_ad)
# Line 116: Generate a UMAP plot colored by 'annotation','week','leiden' with small frame. -- ov.pl.embedding(kidney_lineage_ad,basis='X_umap',
# Line 117: Specify the colors for the embedding plot. --                color=['annotation','week','leiden'],
# Line 118: Specify the frameon for the embedding plot. --                frameon='small')
# Line 120: Generate a dotplot for specified genes grouped by Leiden clusters. -- sc.pl.dotplot(kidney_lineage_ad,{'nephron progenitors':['Wnt9b','Osr1','Nphs1','Lhx1','Pax2','Pax8'],
# Line 121: Define a second gene group for the dotplot. --                          'metanephric':['Eya1','Shisa3','Foxc1'], 
# Line 122: Define a third gene group for the dotplot. --                          'kidney':['Wt1','Wnt4','Nr2f2','Dach1','Cd44']} ,
# Line 123: Specify the group to show the dotplot on and hide the dendrogram and specify colorbar title. --               'leiden',dendrogram=False,colorbar_title='Expression')
# Line 125: Add a column called re_anno to the kidney lineage adata. -- kidney_lineage_ad.obs['re_anno'] = 'Unknown'
# Line 126: Sets the re_anno category for leiden cluster 4. -- kidney_lineage_ad.obs.loc[kidney_lineage_ad.obs.leiden.isin(['4']),'re_anno'] = 'Nephron progenitors (E11.5)'
# Line 127: Sets the re_anno category for leiden clusters 2,3,1,5. -- kidney_lineage_ad.obs.loc[kidney_lineage_ad.obs.leiden.isin(['2','3','1','5']),'re_anno'] = 'Metanephron progenitors (E11.5)'
# Line 128: Sets the re_anno category for leiden cluster 0. -- kidney_lineage_ad.obs.loc[kidney_lineage_ad.obs.leiden=='0','re_anno'] = 'Kidney (E12.5)'
# Line 130: Commented out line that was supposed to filter cells by leiden cluster 3 -- # kidney_all = kidney_all[kidney_all.obs.leiden!='3',:]
# Line 131: Convert leiden cluster column to list -- kidney_lineage_ad.obs.leiden = list(kidney_lineage_ad.obs.leiden)
# Line 132: Generate a UMAP plot colored by 'annotation', 're_anno' with small frame. -- ov.pl.embedding(kidney_lineage_ad,basis='X_umap',
# Line 133: Specify the colors for the embedding plot --                color=['annotation','re_anno'],
# Line 134: Specify the frameon for the embedding plot. --                frameon='small')
# Line 136: Initialize the column 'kidney_anno' in adata1 to empty strings. -- adata1.obs['kidney_anno']=''
# Line 137: Set the 'kidney_anno' of the cells in adata1 where week is E11.5 according to re_anno of the kidney lineage cells of E11.5. -- adata1.obs.loc[kidney_lineage_ad[kidney_lineage_ad.obs['week']=='E11.5'].obs.index,'kidney_anno']=kidney_lineage_ad[kidney_lineage_ad.obs['week']=='E11.5'].obs['re_anno']
# Line 139: Import matplotlib.pyplot as plt. -- import matplotlib.pyplot as plt
# Line 140: Create a subplots object with a size of 8x8. -- fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# Line 141: Generate a spatial plot of adata1 colored by 'kidney_anno', size 1.5, using the palette specified and specify show=False. -- sc.pl.spatial(adata1, color='kidney_anno', spot_size=1.5,
# Line 142: Specify the palette to use for the spatial plot and do not show the spatial plot immediately. --              palette=['#F5F5F5','#ff7f0e', 'green',],show=False,ax=ax)
# Line 144: Assign kidney_lineage_ad to test_adata. -- test_adata=kidney_lineage_ad
# Line 145: Create a pyDEG object from the transposed lognorm data of test_adata. -- dds=ov.bulk.pyDEG(test_adata.to_df(layer='lognorm').T)
# Line 146: Remove duplicates from the index. -- dds.drop_duplicates_index()
# Line 147: Print the success of dropping duplicates from the index. -- print('... drop_duplicates_index success')
# Line 148: Get the list of index for the cells in test_adata where the week is E12.5 as the treatment group. -- treatment_groups=test_adata.obs[test_adata.obs['week']=='E12.5'].index.tolist()
# Line 149: Get the list of index for the cells in test_adata where the week is E11.5 as the control group. -- control_groups=test_adata.obs[test_adata.obs['week']=='E11.5'].index.tolist()
# Line 150: Run deg analysis with ttest method with treatment groups and control groups. -- result=dds.deg_analysis(treatment_groups,control_groups,method='ttest')
# Line 151: Sets the foldchange threshold, pvalue threshold and max log pvalue for the foldchange set. -- # -1 means automatically calculates
# Line 152: Sets the foldchange threshold, pvalue threshold and logp_max. -- dds.foldchange_set(fc_threshold=-1,
# Line 153:  Sets the pvalue and max logp threshold. --                    pval_threshold=0.05,
# Line 154: Sets the logp_max. --                    logp_max=10)
# Line 156: Generate a volcano plot for the DEG analysis with title "DEG Analysis". -- dds.plot_volcano(title='DEG Analysis',figsize=(4,4),
# Line 157: Set number of genes to plot to 8 and the font size to 12. --                  plot_genes_num=8,plot_genes_fontsize=12,)
# Line 159: Gets the index of the top 3 up regulated genes from the DEG result. -- up_gene=dds.result.loc[dds.result['sig']=='up'].sort_values('qvalue')[:3].index.tolist()
# Line 160: Gets the index of the top 3 down regulated genes from the DEG result. -- down_gene=dds.result.loc[dds.result['sig']=='down'].sort_values('qvalue')[:3].index.tolist()
# Line 161: Combines the up regulated and down regulated gene lists. -- deg_gene=up_gene+down_gene
# Line 163: Generate a dotplot of specified genes grouped by 're_anno'. -- sc.pl.dotplot(kidney_lineage_ad,deg_gene,
# Line 164: Specify the group for the dotplot. --              groupby='re_anno')
# Line 166: Compute the dendrogram based on re_anno on the specified scale data. -- sc.tl.dendrogram(kidney_lineage_ad,'re_anno',use_rep='scaled|original|X_pca')
# Line 167: Perform ranked gene group analysis using t-test based on re_anno and scaled data. -- sc.tl.rank_genes_groups(kidney_lineage_ad, 're_anno', use_rep='scaled|original|X_pca',
# Line 168:  Specify method to use for gene ranking. --                         method='t-test',use_raw=False,key_added='re_anno_ttest')
# Line 169: Generate a dotplot of the ranked gene groups with the group name set to re_anno. -- sc.pl.rank_genes_groups_dotplot(kidney_lineage_ad,groupby='re_anno',
# Line 170: Specify the cmap, key and standard scale for the dotplot and the number of genes to show for each group. --                                cmap='RdBu_r',key='re_anno_ttest',
# Line 171: Set the standard scale and number of genes. --                                standard_scale='var',n_genes=3)
```