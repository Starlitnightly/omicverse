```python
# Line 1:  Import the omicverse library as ov. -- import omicverse as ov
# Line 3:  Import the scanpy library as sc. -- import scanpy as sc
# Line 5:  Set plotting parameters using omicverse. -- ov.plot_set()
# Line 7:  Read Visium spatial data using scanpy. -- adata = sc.read_visium(path='data', count_file='151676_filtered_feature_bc_matrix.h5')
# Line 8:  Make variable names unique in the AnnData object. -- adata.var_names_make_unique()
# Line 10: Calculate quality control metrics using scanpy. -- sc.pp.calculate_qc_metrics(adata, inplace=True)
# Line 11: Filter out genes with total counts less than or equal to 100. -- adata = adata[:,adata.var['total_counts']>100]
# Line 12: Perform spatial variable gene selection using omicverse. -- adata=ov.space.svg(adata,mode='prost',n_svgs=3000,target_sum=1e4,platform="visium",)
# Line 13: Display the AnnData object. -- adata
# Line 15: Write the AnnData object to a file with gzip compression. -- adata.write('data/cluster_svg.h5ad',compression='gzip')
# Line 17: Read the AnnData object from a file with gzip compression. -- adata=ov.read('data/cluster_svg.h5ad',compression='gzip')
# Line 20: Import the pandas library as pd. -- import pandas as pd
# Line 21: Import the os library. -- import os
# Line 22: Read the ground truth annotations from a tab-separated file into a pandas DataFrame. -- Ann_df = pd.read_csv(os.path.join('data', '151676_truth.txt'), sep='\t', header=None, index_col=0)
# Line 23: Assign the column name 'Ground Truth' to the DataFrame. -- Ann_df.columns = ['Ground Truth']
# Line 24: Add the ground truth annotation as an observation in the AnnData object. -- adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
# Line 25: Create a spatial plot with annotations using scanpy. -- sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"])
# Line 27: Initialize a dictionary to store keyword arguments for methods. -- methods_kwargs={}
# Line 28: Set parameters for the GraphST method in the methods_kwargs dictionary. -- methods_kwargs['GraphST']={
# Line 31: Perform clustering using the GraphST method using omicverse. -- adata=ov.space.clusters(adata,
# Line 35: Cluster data using mclust on the GraphST representation with omicverse. -- ov.utils.cluster(adata,use_rep='graphst|original|X_pca',method='mclust',n_components=10,
# Line 38: Refine cluster labels using omicverse based on the mclust results. -- adata.obs['mclust_GraphST'] = ov.utils.refine_label(adata, radius=50, key='mclust')
# Line 39: Convert the mclust_GraphST column to categorical data type. -- adata.obs['mclust_GraphST']=adata.obs['mclust_GraphST'].astype('category')
# Line 41: Merge clusters based on the mclust_GraphST labels using omicverse. -- res=ov.space.merge_cluster(adata,groupby='mclust_GraphST',use_rep='graphst|original|X_pca',
# Line 44: Create a spatial plot showing several cluster annotations using scanpy. -- sc.pl.spatial(adata, color=['mclust_GraphST','mclust_GraphST_tree','mclust','Ground Truth'])
# Line 46: Cluster data using mclust_R on the GraphST representation using omicverse. -- ov.utils.cluster(adata,use_rep='graphst|original|X_pca',method='mclust_R',n_components=10,
# Line 49: Refine cluster labels using omicverse based on the mclust_R results. -- adata.obs['mclust_R_GraphST'] = ov.utils.refine_label(adata, radius=30, key='mclust_R')
# Line 50: Convert the mclust_R_GraphST column to categorical data type. -- adata.obs['mclust_R_GraphST']=adata.obs['mclust_R_GraphST'].astype('category')
# Line 51: Merge clusters based on the mclust_R_GraphST labels using omicverse. -- res=ov.space.merge_cluster(adata,groupby='mclust_R_GraphST',use_rep='graphst|original|X_pca',
# Line 54: Create a spatial plot showing several cluster annotations using scanpy. -- sc.pl.spatial(adata, color=['mclust_R_GraphST','mclust_R_GraphST_tree','mclust','Ground Truth'])
# Line 56: Re-initialize the methods_kwargs dictionary. -- methods_kwargs={}
# Line 57: Set parameters for the BINARY method in the methods_kwargs dictionary. -- methods_kwargs['BINARY']={
# Line 73: Perform clustering using the BINARY method using omicverse. -- adata=ov.space.clusters(adata,
# Line 77: Cluster data using mclust_R on the BINARY representation using omicverse. -- ov.utils.cluster(adata,use_rep='BINARY',method='mclust_R',n_components=10,
# Line 80: Refine cluster labels using omicverse based on the mclust_R results. -- adata.obs['mclust_BINARY'] = ov.utils.refine_label(adata, radius=30, key='mclust_R')
# Line 81: Convert the mclust_BINARY column to categorical data type. -- adata.obs['mclust_BINARY']=adata.obs['mclust_BINARY'].astype('category')
# Line 83: Merge clusters based on the mclust_BINARY labels using omicverse. -- res=ov.space.merge_cluster(adata,groupby='mclust_BINARY',use_rep='BINARY',
# Line 86: Create a spatial plot showing several cluster annotations using scanpy. -- sc.pl.spatial(adata, color=['mclust_BINARY','mclust_BINARY_tree','mclust','Ground Truth'])
# Line 88: Cluster data using mclust on the BINARY representation using omicverse. -- ov.utils.cluster(adata,use_rep='BINARY',method='mclust',n_components=10,
# Line 91: Refine cluster labels using omicverse based on the mclust results. -- adata.obs['mclustpy_BINARY'] = ov.utils.refine_label(adata, radius=30, key='mclust')
# Line 92: Convert the mclustpy_BINARY column to categorical data type. -- adata.obs['mclustpy_BINARY']=adata.obs['mclustpy_BINARY'].astype('category')
# Line 94: Convert the mclustpy_BINARY column to categorical data type. -- adata.obs['mclustpy_BINARY']=adata.obs['mclustpy_BINARY'].astype('category')
# Line 95: Merge clusters based on the mclustpy_BINARY labels using omicverse. -- res=ov.space.merge_cluster(adata,groupby='mclustpy_BINARY',use_rep='BINARY',
# Line 98: Create a spatial plot showing several cluster annotations using scanpy. -- sc.pl.spatial(adata, color=['mclustpy_BINARY','mclustpy_BINARY_tree','mclust','Ground Truth'])
# Line 102: Re-initialize the methods_kwargs dictionary. -- methods_kwargs={}
# Line 103: Set parameters for the STAGATE method in the methods_kwargs dictionary. -- methods_kwargs['STAGATE']={
# Line 110: Perform clustering using the STAGATE method using omicverse. -- adata=ov.space.clusters(adata,
# Line 114: Cluster data using mclust_R on the STAGATE representation using omicverse. -- ov.utils.cluster(adata,use_rep='STAGATE',method='mclust_R',n_components=10,
# Line 117: Refine cluster labels using omicverse based on the mclust_R results. -- adata.obs['mclust_R_STAGATE'] = ov.utils.refine_label(adata, radius=30, key='mclust_R')
# Line 118: Convert the mclust_R_STAGATE column to categorical data type. -- adata.obs['mclust_R_STAGATE']=adata.obs['mclust_R_STAGATE'].astype('category')
# Line 119: Merge clusters based on the mclust_R_STAGATE labels using omicverse. -- res=ov.space.merge_cluster(adata,groupby='mclust_R_STAGATE',use_rep='STAGATE',
# Line 122: Create a spatial plot showing several cluster annotations using scanpy. -- sc.pl.spatial(adata, color=['mclust_R_STAGATE','mclust_R_STAGATE_tree','mclust_R','Ground Truth'])
# Line 124: Display the top 5 genes with highest PI values. -- adata.var.sort_values('PI',ascending=False).head(5)
# Line 126: Set the name of the gene to plot. -- plot_gene = 'MBP'
# Line 127: Import the matplotlib library as plt. -- import matplotlib.pyplot as plt
# Line 128: Create a figure and a set of subplots for spatial plotting. -- fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# Line 129: Create a spatial plot showing raw expression of the specified gene using scanpy. -- sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[0], title='RAW_'+plot_gene, vmax='p99')
# Line 130: Create a spatial plot showing STAGATE-transformed expression of the specified gene using scanpy. -- sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[1], title='STAGATE_'+plot_gene, layer='STAGATE_ReX', vmax='p99')
# Line 133: Re-initialize the methods_kwargs dictionary. -- methods_kwargs={}
# Line 134: Set parameters for the CAST method in the methods_kwargs dictionary. -- methods_kwargs['CAST']={
# Line 139: Perform clustering using the CAST method using omicverse. -- adata=ov.space.clusters(adata,
# Line 142: Cluster data using mclust on the CAST representation using omicverse. -- ov.utils.cluster(adata,use_rep='X_cast',method='mclust',n_components=10,
# Line 145: Refine cluster labels using omicverse based on the mclust results. -- adata.obs['mclust_CAST'] = ov.utils.refine_label(adata, radius=50, key='mclust')
# Line 146: Convert the mclust_CAST column to categorical data type. -- adata.obs['mclust_CAST']=adata.obs['mclust_CAST'].astype('category')
# Line 148: Merge clusters based on the mclust_CAST labels using omicverse. -- res=ov.space.merge_cluster(adata,groupby='mclust_CAST',use_rep='X_cast',
# Line 151: Create a spatial plot showing several cluster annotations using scanpy. -- sc.pl.spatial(adata, color=['mclust_CAST','mclust_CAST_tree','mclust','Ground Truth'])
# Line 153: Display the AnnData object. -- adata
# Line 155: Import the adjusted rand score from sklearn. -- from sklearn.metrics.cluster import adjusted_rand_score
# Line 157: Create a subset of adata's obs dataframe that does not contain any NA values. -- obs_df = adata.obs.dropna()
# Line 159: Calculate the adjusted rand index for mclust_GraphST compared to the Ground Truth and print it. -- ARI = adjusted_rand_score(obs_df['mclust_GraphST'], obs_df['Ground Truth'])
# Line 160: Print the ARI for mclust_GraphST. -- print('mclust_GraphST: Adjusted rand index = %.2f' %ARI)
# Line 162: Calculate the adjusted rand index for mclust_R_GraphST compared to the Ground Truth and print it. -- ARI = adjusted_rand_score(obs_df['mclust_R_GraphST'], obs_df['Ground Truth'])
# Line 163: Print the ARI for mclust_R_GraphST. -- print('mclust_R_GraphST: Adjusted rand index = %.2f' %ARI)
# Line 165: Calculate the adjusted rand index for mclust_R_STAGATE compared to the Ground Truth and print it. -- ARI = adjusted_rand_score(obs_df['mclust_R_STAGATE'], obs_df['Ground Truth'])
# Line 166: Print the ARI for mclust_STAGATE. -- print('mclust_STAGATE: Adjusted rand index = %.2f' %ARI)
# Line 168: Calculate the adjusted rand index for mclust_BINARY compared to the Ground Truth and print it. -- ARI = adjusted_rand_score(obs_df['mclust_BINARY'], obs_df['Ground Truth'])
# Line 169: Print the ARI for mclust_BINARY. -- print('mclust_BINARY: Adjusted rand index = %.2f' %ARI)
# Line 171: Calculate the adjusted rand index for mclustpy_BINARY compared to the Ground Truth and print it. -- ARI = adjusted_rand_score(obs_df['mclustpy_BINARY'], obs_df['Ground Truth'])
# Line 172: Print the ARI for mclustpy_BINARY. -- print('mclustpy_BINARY: Adjusted rand index = %.2f' %ARI)
# Line 174: Calculate the adjusted rand index for mclust_CAST compared to the Ground Truth and print it. -- ARI = adjusted_rand_score(obs_df['mclust_CAST'], obs_df['Ground Truth'])
# Line 175: Print the ARI for mclust_CAST. -- print('mclust_CAST: Adjusted rand index = %.2f' %ARI)
```