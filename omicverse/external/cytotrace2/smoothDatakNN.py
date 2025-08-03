import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import os

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError("File not found.")
    
    # Load data
    data = pd.read_csv(filepath, sep="\t", header=0)
    
    # Set row names and clean them
    data.index = data.iloc[:, 0]
    data.index = data.index.str.replace('.', '-', regex=False)
    data = data.iloc[:, 1:]  # Remove the first column
    
    return data

def smooth_data_kNN(output_dir, suffix, max_pcs, seed):
    # Load data
    predicted_df = load_data(f"{output_dir}/binned_df{suffix}.txt")
    ranked_df = load_data(f"{output_dir}/ranked_df{suffix}.txt")
    with open(f"{output_dir}/top_var_genes{suffix}.txt", 'r') as file:
        top_genes = [line.strip() for line in file]
    
    np.random.seed(seed)
    
    def smoothing_by_KNN(score, umap_coordinates):
        maxk = min(30, max(3, round(0.005 * len(score))))
        nn = NearestNeighbors(n_neighbors=maxk)
        nn.fit(umap_coordinates)
        distances, indices = nn.kneighbors(umap_coordinates)
        
        smoothed_scores = np.array([np.median(score[indices[i]]) for i in range(len(score))])
        return smoothed_scores

    # Create Seurat-like object using Scanpy
    adata = sc.AnnData(X=ranked_df.values.T)
    adata.var_names = ranked_df.index
    adata.obs_names = ranked_df.columns

    # Set top variable features
    adata.var['highly_variable'] = adata.var_names.isin(top_genes)

    # Standardize (scale) data
    sc.pp.scale(adata, max_value=10)

    # Run PCA
    sc.tl.pca(adata, n_comps=min(adata.shape[1] - 1, max_pcs))

    threshold_var_explained = 0.5
    stdev_explained = np.std(adata.obsm['X_pca'], axis=0)
    var_explained = stdev_explained ** 2
    var_explained /= np.sum(var_explained)

    if np.sum(var_explained) < threshold_var_explained:
        num_pcs = min(max_pcs, adata.shape[1] - 1)
    else:
        num_pcs = np.argmax(np.cumsum(var_explained) > threshold_var_explained) + 1
    num_pcs = max(2, num_pcs)

    umap_coordinates = adata.obsm['X_pca'][:, :num_pcs]

    knn_score = smoothing_by_KNN(predicted_df['preKNN_CytoTRACE2_Score'].values, umap_coordinates)
    predicted_df['CytoTRACE2_Score'] = knn_score

    # Remap potency categories
    ranges = np.linspace(0, 1, 7)
    labels = ['Differentiated', 'Unipotent', 'Oligopotent', 'Multipotent', 'Pluripotent', 'Totipotent']
    predicted_df['CytoTRACE2_Potency'] = pd.cut(predicted_df['CytoTRACE2_Score'], bins=ranges, labels=labels)

    predicted_df.to_csv(os.path.join(output_dir, f'smoothbykNNresult{suffix}.txt'), sep='\t', index=True)

