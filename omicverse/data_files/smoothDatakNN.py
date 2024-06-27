import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
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
    np.random.seed(seed)

    # Load data
    predicted_df = load_data(os.path.join(output_dir, f'binned_df{suffix}.txt'))
    ranked_df = load_data(os.path.join(output_dir, f'ranked_df{suffix}.txt'))
    with open(os.path.join(output_dir, f'top_var_genes{suffix}.txt'), 'r') as f:
        top_genes = f.read().splitlines()

    def smoothing_by_KNN(score, umap_coordinates):
        maxk = min(30, max(3, round(0.005 * len(score))))
        nbrs = NearestNeighbors(n_neighbors=maxk).fit(umap_coordinates)
        _, indices = nbrs.kneighbors(umap_coordinates)
        smoothed_scores = [np.median(score[indices[i]]) for i in range(len(score))]
        return smoothed_scores

    # Convert ranked_df to numpy array
    ranked_data = ranked_df.to_numpy()

    # Perform PCA
    pca = PCA(n_components=min(ranked_data.shape[1] - 1, max_pcs))
    pca_result = pca.fit_transform(ranked_data)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    threshold_var_explained = 0.5
    num_pcs = np.argmax(cumulative_variance >= threshold_var_explained) + 1
    num_pcs = max(2, num_pcs)

    umap_coordinates = pca_result[:, :num_pcs]

    # Apply kNN smoothing
    knn_score = smoothing_by_KNN(predicted_df['preKNN_CytoTRACE2_Score'], umap_coordinates)
    predicted_df['CytoTRACE2_Score'] = knn_score

    # Re-map potency categories
    ranges = np.linspace(0, 1, 7)
    labels = [
        'Differentiated',
        'Unipotent',
        'Oligopotent',
        'Multipotent',
        'Pluripotent',
        'Totipotent'
    ]
    order_vector = pd.cut(predicted_df['CytoTRACE2_Score'], bins=ranges, labels=labels, include_lowest=True)
    predicted_df['CytoTRACE2_Potency'] = order_vector

    # Save the result
    predicted_df.to_csv(os.path.join(output_dir, f'smoothbykNNresult{suffix}.txt'), sep='\t', index=False, quoting=3)

# Argument parsing
parser = argparse.ArgumentParser()

parser.add_argument("--output-dir", type=str, default="cytotrace2_results", help="Output directory containing intermediate files")
parser.add_argument("--suffix", type=str, default="", help="Suffix of intermediate files")
parser.add_argument("--max-pcs", type=int, default=200, help="Integer, indicating the maximum number of principal components to use in the smoothing by kNN step (default is 200)")
parser.add_argument("--seed", type=int, default=14, help="Integer, specifying the seed for reproducibility in random processes (default is 14).")

args = parser.parse_args()
smooth_data_kNN(output_dir=args.output_dir, suffix=args.suffix, max_pcs=args.max_pcs, seed=args.seed)
