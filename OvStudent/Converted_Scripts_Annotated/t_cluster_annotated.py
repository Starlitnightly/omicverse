```
# Line 1:  Import the omicverse library and alias it as ov. -- import omicverse as ov
# Line 2:  Import the scanpy library and alias it as sc. -- import scanpy as sc
# Line 3:  Import the scvelo library and alias it as scv. -- import scvelo as scv
# Line 4:  Set the plotting style using the omicverse library. -- ov.plot_set()
# Line 6:  Import the scvelo library and alias it as scv (again, which is redundant). -- import scvelo as scv
# Line 7:  Load the dentategyrus dataset from scvelo into an AnnData object called adata. -- adata=scv.datasets.dentategyrus()
# Line 8:  Display the adata AnnData object. -- adata
# Line 10: Preprocess the adata AnnData object using a 'shiftlog|pearson' method and selecting the top 3000 highly variable genes using the omicverse library. -- adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',n_HVGs=3000,)
# Line 11:  Store the processed adata AnnData object in the .raw attribute. -- adata.raw = adata
# Line 12:  Subset the adata AnnData object to only keep the highly variable genes. -- adata = adata[:, adata.var.highly_variable_features]
# Line 13: Scale the adata AnnData object using the omicverse library. -- ov.pp.scale(adata)
# Line 14:  Perform Principal Component Analysis (PCA) on the scaled data with 50 principal components using the omicverse library. -- ov.pp.pca(adata,layer='scaled',n_pcs=50)
# Line 16:  Plot the variance ratio explained by each principal component using the omicverse library. -- ov.utils.plot_pca_variance_ratio(adata)
# Line 18:  Compute the neighborhood graph with 15 neighbors using the top 50 PCs of the scaled, original or X_pca representations, using the scanpy library. -- sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50,
# Line 19: Cluster the data using the Leiden algorithm with resolution 1, using the omicverse library. --                use_rep='scaled|original|X_pca')
# Line 20:  Generate a UMAP embedding colored by the 'clusters' and 'leiden' columns, using the omicverse library. -- ov.utils.cluster(adata,method='leiden',resolution=1)
# Line 21:  Compute the neighborhood graph with 15 neighbors using the top 50 PCs of the scaled, original or X_pca representations, using the scanpy library (repeated). -- ov.utils.embedding(adata,basis='X_umap',
# Line 22: Cluster the data using the Louvain algorithm with resolution 1, using the omicverse library. --                    color=['clusters','leiden'],
# Line 23:  Generate a UMAP embedding colored by the 'clusters' and 'louvain' columns, using the omicverse library. --                    frameon='small',wspace=0.5)
# Line 25:  Compute the neighborhood graph with 15 neighbors using the top 50 PCs of the scaled, original or X_pca representations, using the scanpy library (repeated). -- sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50,
# Line 26: Cluster the data using the Louvain algorithm with resolution 1, using the omicverse library (repeated). --                use_rep='scaled|original|X_pca')
# Line 27:  Generate a UMAP embedding colored by the 'clusters' and 'louvain' columns, using the omicverse library (repeated). -- ov.utils.cluster(adata,method='louvain',resolution=1)
# Line 29: Cluster the data using Gaussian Mixture Model (GMM) with 21 components on the scaled, original or X_pca representations, using the omicverse library. -- ov.utils.embedding(adata,basis='X_umap',
# Line 30:  Generate a UMAP embedding colored by the 'clusters' and 'gmm_cluster' columns, using the omicverse library. --                    color=['clusters','louvain'],
# Line 31: Perform Latent Dirichlet Allocation (LDA) topic modeling on the gene expression, using the omicverse library. --                   frameon='small',wspace=0.5)
# Line 33: Generate a plot of the topic contributions from the LDA model using the omicverse library. -- ov.utils.cluster(adata,use_rep='scaled|original|X_pca',
# Line 35: Generate a plot of the topic contributions from the LDA model using the omicverse library (for topic 13). --                  method='GMM',n_components=21,
# Line 37:  Set the plot style using the omicverse library again (redundant). --                  covariance_type='full',tol=1e-9, max_iter=1000, )
# Line 38: Generate a UMAP embedding colored by the LDA topic columns, using the omicverse library. -- ov.utils.embedding(adata,basis='X_umap',
# Line 41:  Generate a UMAP embedding colored by the 'clusters' and 'LDA_cluster' columns, using the omicverse library. --                    color=['clusters','gmm_cluster'],
# Line 43: Run a Random Forest Classifier (RFC) based on LDA topic results using the omicverse library. --                    frameon='small',wspace=0.5)
# Line 45: Generate a UMAP embedding colored by the Random Forest classification and cluster assignments from LDA results, using the omicverse library. -- LDA_obj=ov.utils.LDA_topic(adata,feature_type='expression',
# Line 47:  Convert the sparse matrix of the AnnData object to a dense array using numpy --                   highly_variable_key='highly_variable_features',
# Line 49:  Import the numpy library. --                  layers='counts',batch_key=None,learning_rate=1e-3)
# Line 50: Initialize a cNMF object to perform consensus non-negative matrix factorization. -- LDA_obj.plot_topic_contributions(6)
# Line 51: Launch a cNMF worker with id 0, among 4 total workers, using the omicverse library. -- LDA_obj.predicted(13)
# Line 52:  Combine the results of all the cNMF workers skipping any missing file using the omicverse library. -- ov.plot_set()
# Line 53: Generate a plot of the cNMF K-selection, closing the figure after generation using the omicverse library. -- ov.utils.embedding(adata, basis='X_umap',color = LDA_obj.model.topic_cols, cmap='BuPu', ncols=4,
# Line 55: Set the number of selected components to 7. --            add_outline=True,  frameon='small',)
# Line 56: Set the density threshold for the consensus cNMF step. -- ov.utils.embedding(adata,basis='X_umap',
# Line 57: Perform the consensus step for cNMF, and show clustering on the generated heatmap, using the omicverse library. --                    color=['clusters','LDA_cluster'],
# Line 58: Load the cNMF results for the specified k and density threshold using the omicverse library. --                    frameon='small',wspace=0.5)
# Line 59:  Add the cNMF results into the adata AnnData object using the omicverse library. -- LDA_obj.get_results_rfc(adata,use_rep='scaled|original|X_pca',
# Line 61:  Generate a UMAP embedding colored by the cNMF normalized usage matrix, using the omicverse library. --                         LDA_threshold=0.4,num_topics=13)
# Line 63: Run a Random Forest Classifier (RFC) based on cNMF results using the omicverse library. -- ov.utils.embedding(adata,basis='X_umap',
# Line 65: Generate a UMAP embedding colored by the Random Forest classification and cluster assignments from cNMF results, using the omicverse library. --                    color=['LDA_cluster_rfc','LDA_cluster_clf'],
# Line 66:  Convert the sparse matrix of the AnnData object to a dense array using numpy (redundant since done in line 47). --                   frameon='small',wspace=0.5)
# Line 68: Calculate the Adjusted Rand Index (ARI) comparing the "clusters" to the "leiden" labels, using sklearn. -- adata.X.toarray()
# Line 69: Print the Adjusted Rand Index (ARI) for "leiden". -- import numpy as np
# Line 71: Calculate the Adjusted Rand Index (ARI) comparing the "clusters" to the "louvain" labels, using sklearn. -- ## Initialize the cnmf object that will be used to run analyses
# Line 72: Print the Adjusted Rand Index (ARI) for "louvain". -- cnmf_obj = ov.single.cNMF(adata,components=np.arange(5,11), n_iter=20, seed=14, num_highvar_genes=2000,
# Line 74: Calculate the Adjusted Rand Index (ARI) comparing the "clusters" to the "gmm_cluster" labels, using sklearn. --                           output_dir='example_dg1/cNMF', name='dg_cNMF')
# Line 75: Print the Adjusted Rand Index (ARI) for "GMM". -- ## Specify that the jobs are being distributed over a single worker (total_workers=1) and then launch that worker
# Line 77: Calculate the Adjusted Rand Index (ARI) comparing the "clusters" to the "LDA_cluster" labels, using sklearn. -- cnmf_obj.factorize(worker_i=0, total_workers=4)
# Line 78: Print the Adjusted Rand Index (ARI) for "LDA". -- cnmf_obj.combine(skip_missing_files=True)
# Line 80: Calculate the Adjusted Rand Index (ARI) comparing the "clusters" to the "LDA_cluster_rfc" labels, using sklearn. -- cnmf_obj.k_selection_plot(close_fig=False)
# Line 81: Print the Adjusted Rand Index (ARI) for "LDA_rfc". -- 
# Line 83: Calculate the Adjusted Rand Index (ARI) comparing the "clusters" to the "LDA_cluster_clf" labels, using sklearn. -- selected_K = 7
# Line 84: Print the Adjusted Rand Index (ARI) for "LDA_clf". -- density_threshold = 2.00
# Line 86: Calculate the Adjusted Rand Index (ARI) comparing the "clusters" to the "cNMF_cluster_rfc" labels, using sklearn. -- cnmf_obj.consensus(k=selected_K, 
# Line 87: Print the Adjusted Rand Index (ARI) for "cNMF_rfc". --                    density_threshold=density_threshold, 
# Line 89: Calculate the Adjusted Rand Index (ARI) comparing the "clusters" to the "cNMF_cluster_clf" labels, using sklearn. --                    show_clustering=True, 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --                    close_clustergram_fig=False)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- result_dict = cnmf_obj.load_results(K=selected_K, density_threshold=density_threshold)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- cnmf_obj.get_results(adata,result_dict)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- ov.pl.embedding(adata, basis='X_umap',color=result_dict['usage_norm'].columns,
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --            use_raw=False, ncols=3, vmin=0, vmax=1,frameon='small')
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- cnmf_obj.get_results_rfc(adata,result_dict,
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --                          use_rep='scaled|original|X_pca',
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --                         cNMF_threshold=0.5)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- ov.pl.embedding(
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     adata,
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     basis="X_umap",
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     color=['cNMF_cluster_rfc','cNMF_cluster_clf'],
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     frameon='small',
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     #title="Celltypes",
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     #legend_loc='on data',
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     legend_fontsize=14,
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     legend_fontoutline=2,
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     #size=10,
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     #legend_loc=True, 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     add_outline=False, 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     #add_outline=True,
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     outline_color='black',
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     outline_width=1,
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". --     show=False,
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- )
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- from sklearn.metrics.cluster import adjusted_rand_score
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- ARI = adjusted_rand_score(adata.obs['clusters'], adata.obs['leiden'])
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- print('Leiden, Adjusted rand index = %.2f' %ARI)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- ARI = adjusted_rand_score(adata.obs['clusters'], adata.obs['louvain'])
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- print('Louvain, Adjusted rand index = %.2f' %ARI)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- ARI = adjusted_rand_score(adata.obs['clusters'], adata.obs['gmm_cluster'])
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- print('GMM, Adjusted rand index = %.2f' %ARI)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- ARI = adjusted_rand_score(adata.obs['clusters'], adata.obs['LDA_cluster'])
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- print('LDA, Adjusted rand index = %.2f' %ARI)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- ARI = adjusted_rand_score(adata.obs['clusters'], adata.obs['LDA_cluster_rfc'])
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- print('LDA_rfc, Adjusted rand index = %.2f' %ARI)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- ARI = adjusted_rand_score(adata.obs['clusters'], adata.obs['LDA_cluster_clf'])
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- print('LDA_clf, Adjusted rand index = %.2f' %ARI)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- ARI = adjusted_rand_score(adata.obs['clusters'], adata.obs['cNMF_cluster_rfc'])
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- print('cNMF_rfc, Adjusted rand index = %.2f' %ARI)
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- 
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- ARI = adjusted_rand_score(adata.obs['clusters'], adata.obs['cNMF_cluster_clf'])
# Line 90: Print the Adjusted Rand Index (ARI) for "cNMF_clf". -- print('cNMF_clf, Adjusted rand index = %.2f' %ARI)
```