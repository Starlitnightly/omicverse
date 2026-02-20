import os
import csv
import gzip
import torch
import anndata

import numpy as np
import pandas as pd
import scanpy as sc
import tensorly as tl
import sklearn.neighbors

from scipy.io import mmread
from scipy.sparse import coo_matrix

tl.set_backend('pytorch')


def preprocessing(raw_data_path, PPI_data_path, load_labels=False,
                  use_coexpression=True, use_PPI=True, 
                  use_highly_variable=True, use_all_entries=False,
                  apply_normalization=True, n_pcs=15, n_neighbors=10, 
                  n_top_genes=3000):
    
    '''
    Parameters:
    
    raw_data_path: the path to raw spatial transcriptomics data
                   (Please organize data into the following structure: 
                   
                    . <data-folder>
                    ├── ...
                    ├── <tissue-folder>
                    │   ├── filtered_feature_bc_matrix
                    │   │   ├── barcodes.tsv.gz
                    │   │   ├── features.tsv.gz
                    │   │   └── matrix.mtx.gz
                    │   ├── spatial
                    │   │   ├── tissue_positions_list.csv
                    │   │   └── cluster_labels.csv (optional)
                    └── ...)
                    
    PPI_data_path: the path to PPI graph 
                   (Please download corresponding PPI graph 
                    for differet species from BioGRID: 
                    https://downloads.thebiogrid.org/BioGRID)
                    
    load_labels: the flag to determine whether to load spot labels. 
                 (Note that this flag is only available to DFLPC 
                  sections from spatialLIBD project)
                  
    use_coexpression: the flag to determine whether to use co-expression 
                      in spatial graph construction. (Note that 
                      co-expression is drived from k nearest 
                      neighbors (kNN) graph on normalized expression 
                      profile of highly variable genes)
    
    use_PPI: the flag to determine whether to involve PPI graph in 
             Cartesian product graph regularization
    
    use_highly_variable: the flag to determine whether to impute expression
                         profile of highly variable genes only (Note that 
                         the number of highly varibale genes is determined 
                         by the parameter n_top_genes)
                         
    use_all_entries: the flag to determine whether to include all entries in 
                     the tensor for model training
    
    apply_normalization: the flag to determine whether to apply normalization 
                         to raw UMI counts (Note that raw UMI counts will be 
                         normalized per spot, pleasse see details in the 
                         following link: 
                         https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.normalize_total.html)
                    
    n_pcs: the number of principal components used in kNN graph for co-expression 
           if use_coexpression is enabled
    
    n_neighbors: the number of k nearest neighbors used in kNN graph for 
                 co-expression if use_coexpression is enabled
    
    n_top_genes: the number of highly variable genes used in kNN graph for 
                 co-expression if use_coexpression is enabled, and the number 
                 of genes in the imputed expression tensor if use_highly_variable 
                 is set to be True.
                 
                 
    Returns:
    
    expr_tensor: expression tensor in sparse format
    
    A_g: adjacency matrix of PPI graph in sparse format
    
    A_xy: adjacency matrix of PPI graph in sparse format
    
    feature_ids_PPI: ensembl IDs of genes in the expression tensor
    
    gene_names_PPI: offical names of genes in the expression tensor
    
    mapping: mapping between original coordinates (both spatial and pixel) 
             and indexes of corresponding entries in the tesnor, where the 
             last column indicates the spot label if load_label is  
             set to be True, otherwise, it indicates whether the spot 
             is overlapped with tissue
    
    
    '''
    
    # Set the shape of spot array
    n_x = 78; n_y = 64
    
    matrix_dir = os.path.join(raw_data_path, "filtered_feature_bc_matrix")
    features_path = os.path.join(matrix_dir, "features.tsv.gz")
    barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
    sp_info_dir = os.path.join(raw_data_path, "spatial")
    sp_info_path = os.path.join(sp_info_dir, "tissue_positions_list.csv")

    # Read filtered feature-barcode matrix data
    raw_expr_mat = mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))
    feature_ids = np.array([row[0] for row in csv.reader(gzip.open(features_path, 'rt'), delimiter="\t")])
    gene_names = np.char.lower(np.array([row[1] for row in csv.reader(gzip.open(features_path, 'rt'), delimiter="\t")]))
    feature_types = np.array([row[2] for row in csv.reader(gzip.open(features_path, 'rt'), delimiter="\t")])
    barcodes = np.array([row[0] for row in csv.reader(gzip.open(barcodes_path, 'rt'), delimiter="\t")])

    # Read spatial coordinates
    
    # -2 indicate spot outside tissue
    # -1 indicate spot overlapped with tissue
    # >=0 indicate spot label
    
    mapping = np.vstack((np.repeat(np.arange(n_x), n_y), np.tile(np.arange(n_y), n_x), 
                         np.zeros(n_x*n_y), np.zeros(n_x*n_y), 
                         np.zeros(n_x*n_y), np.zeros(n_x*n_y),
                         -2 * np.ones(n_x*n_y))).T
    
    sp_barcodes = np.array([row[0] for row in csv.reader(open(sp_info_path, 'rt'), delimiter=",")])
    sp_in_tissue = np.array([row[1] for row in csv.reader(open(sp_info_path, 'rt'), delimiter=",")], dtype=np.int32)
    sp_x_coords = np.array([row[2] for row in csv.reader(open(sp_info_path, 'rt'), delimiter=",")], dtype=np.int32)
    sp_y_coords = np.array([row[3] for row in csv.reader(open(sp_info_path, 'rt'), delimiter=",")], dtype=np.int32)
    px_x_coords = np.array([row[4] for row in csv.reader(open(sp_info_path, 'rt'), delimiter=",")], dtype=np.int32)
    px_y_coords = np.array([row[5] for row in csv.reader(open(sp_info_path, 'rt'), delimiter=",")], dtype=np.int32)
    
    # Align spatial coordinates between rows
    idx = [np.where(np.array(sp_barcodes) == barcode)[0][0] for barcode in barcodes]
    x_coords = sp_x_coords[idx]
    y_coords = sp_y_coords[idx]
    x_aligned_coords = x_coords
    y_aligned_coords = y_coords//2
    aligned_coords = x_aligned_coords*n_y + y_aligned_coords
    mapping[aligned_coords, 2] = sp_x_coords[idx]; mapping[aligned_coords, 3] = sp_y_coords[idx];
    mapping[aligned_coords, 4] = px_x_coords[idx]; mapping[aligned_coords, 5] = px_y_coords[idx];
    mapping[aligned_coords, -1] = sp_in_tissue[idx] - 2
    
    # Note that this code snippet only works
    if load_labels:
        
        cl_info_path = os.path.join(sp_info_dir, 'cluster_labels.csv')
        cl_barcodes = np.array([row['key'].split('_')[1] for row in csv.DictReader(open(cl_info_path, 'rt'), delimiter=",")])
        cl_labels = np.array([row['ground_truth'] for row in csv.DictReader(open(cl_info_path, 'rt'), delimiter=",")])
        cl_labels[np.where(cl_labels == 'WM')] = "Layer_0"
        cl_barcodes = cl_barcodes[np.where(cl_labels != 'NA')]
        cl_labels = cl_labels[np.where(cl_labels != 'NA')]
        cl_labels = np.array([int(label.split('_')[1]) for label in cl_labels])

        # Align spatial coordinates between rows
        idx = [np.where(np.array(sp_barcodes) == barcode)[0][0] for barcode in cl_barcodes]
        x_coords = sp_x_coords[idx]
        y_coords = sp_y_coords[idx]
        x_aligned_coords = x_coords
        y_aligned_coords = y_coords//2
        aligned_coords = x_aligned_coords*n_y + y_aligned_coords
        mapping[aligned_coords, -1] = cl_labels
    
    
    # Load PPI graph
    biogrid_PPI = pd.read_csv(PPI_data_path, delimiter='\t')
    
    # Construct PPI graph
    gene_names_A = biogrid_PPI['Official Symbol Interactor A'].str.lower().to_numpy().astype('<U16') 
    gene_names_B = biogrid_PPI['Official Symbol Interactor B'].str.lower().to_numpy().astype('<U16')
    matched_indices = np.where(np.isin(gene_names_A, gene_names) & np.isin(gene_names_B, gene_names))[0]
    gene_names_A = gene_names_A[matched_indices]; gene_names_B = gene_names_B[matched_indices]
    gene_names_PPI = np.unique(np.union1d(gene_names_A, gene_names_B))
    row = np.array([np.where(gene_names_PPI == gene_name)[0][0] for gene_name in gene_names_A])
    col = np.array([np.where(gene_names_PPI == gene_name)[0][0] for gene_name in gene_names_B])
    data = np.ones(len(row) + len(col)); n_g = len(gene_names_PPI)
    A_g = coo_matrix((data, (np.concatenate([row, col]), 
                             np.concatenate([col, row]))), 
                      shape=(n_g, n_g)).toarray()
    A_g[np.where(A_g > 0)] = 1 # convert weighted adjacent matrix to adjacent matrix
    np.fill_diagonal(A_g, 0) # remove self connections
    
    if not use_PPI:
        A_g = np.eye(A_g.shape[0])

    # Potenial issue: BioGRID provides Entrez gene ID in protein-protein interaction network, while 10x Visium
    # spatial transcriptomics data uses Ensembl gene ID, however, multiple Ensembl gene ID might be mapped to 
    # the same Entrez gene ID, the current startegy in this preprocessing script is removing genes if their 
    # Entrez gene IDs are duplicated
    gene_indices = np.array([np.where(gene_names == gene_name)[0][0] for gene_name in gene_names_PPI])
    feature_ids_PPI = feature_ids[gene_indices]
    
    expr_mat = np.stack([coo_matrix((raw_expr_mat.data[np.where(raw_expr_mat.row==row)[0]],
                                     (x_aligned_coords[raw_expr_mat.col[np.where(raw_expr_mat.row==row)[0]]],
                                      y_aligned_coords[raw_expr_mat.col[np.where(raw_expr_mat.row==row)[0]]])),
                                    shape=(n_x, n_y)).toarray()
                         for row in gene_indices], axis = 0)
    
    # Crop the region overlapping with tissue
    nonzero_index = np.where(np.sum(expr_mat != 0, axis = tuple([0, 2]))>0)[0]
    expr_mat = expr_mat[:, nonzero_index, :]
    mapping = mapping[np.where([i in nonzero_index for i in mapping[:, 0]])]
    nonzero_index = np.where(np.sum(expr_mat != 0, axis = tuple([0, 1]))>0)[0]
    expr_mat = expr_mat[:, :, nonzero_index]
    mapping = mapping[np.where([i in nonzero_index for i in mapping[:, 1]])]
    n_g, n_x, n_y = np.shape(expr_mat) # new shape of tensor after cropping
    
    # Find spots overlapped with tissue
    spot_idx = np.where(mapping[:, -1] != -2)
    empty_spot_idx = np.where(mapping[:, -1] == -2)
    if load_labels:
        spot_idx = np.where(mapping[:, -1] >= 0)
        empty_spot_idx = np.where(mapping[:, -1] < 0)
        
    # Construct spatial graph
    A_x = np.eye(n_x, k=-1) + np.eye(n_x, k=1)
    A_y = np.eye(n_y, k=-1) + np.eye(n_y, k=1)
    A_xy_s = np.kron(A_x, np.eye(n_y)) + np.kron(np.eye(n_x), A_y)    
    A_xy_s[empty_spot_idx[0], :] = 0; A_xy_s[:, empty_spot_idx[0]] = 0
    
    A_xy = A_xy_s
    
    if use_coexpression:
        
        #expr_mat_norm = torch.from_numpy(expr_mat)
        #expr_mat_norm = tl.unfold(expr_mat_norm, 0).numpy().T
        expr_mat_norm = expr_mat.reshape(n_g, -1).T
        expr_mat_norm = anndata.AnnData(expr_mat_norm)
        sc.pp.highly_variable_genes(expr_mat_norm, flavor="seurat_v3", n_top_genes=n_top_genes)
        sc.pp.normalize_total(expr_mat_norm, target_sum=1e4)
        sc.pp.log1p(expr_mat_norm)
        
        sc.pp.pca(expr_mat_norm, n_comps=n_pcs, use_highly_variable=True)
        A_xy_c = construct_knn_graph(expr_mat_norm.obsm['X_pca'], n_neighbors=n_neighbors)
        A_xy_c[empty_spot_idx[0], :] = 0; A_xy_c[:, empty_spot_idx[0]] = 0
        
        A_xy = A_xy_s + A_xy_c
        A_xy[A_xy>1] = 1
    
    # Apply normalization
    #expr_mat = torch.from_numpy(expr_mat)
    #expr_mat = tl.unfold(expr_mat, 0).numpy().T
    expr_mat = expr_mat.reshape(n_g, -1).T
    expr_mat = anndata.AnnData(expr_mat)
    sc.pp.highly_variable_genes(expr_mat, flavor="seurat_v3", n_top_genes=n_top_genes)
    
    # Normalize counts per spot
    if apply_normalization:
        sc.pp.normalize_total(expr_mat, target_sum=1e4)
    # Take log transformation
    sc.pp.log1p(expr_mat)
    
    # Keep highly variable genes only
    if use_highly_variable:
        index = np.where(expr_mat.var['highly_variable'])[0]
        n_g = len(index)
        A_g = A_g[np.ix_(index, index)]
        feature_ids_PPI = feature_ids_PPI[index]
        gene_names_PPI = gene_names_PPI[index]
        expr_mat = expr_mat.X[:, index].T
    else:
        expr_mat = expr_mat.X.T
    
    # Convert normalization expression data into a sparse tensor format
    if use_all_entries:
        expr_mat_ = expr_mat[:, spot_idx[0]]
        expr_mat_[np.where(expr_mat_==0)] = -1
        expr_mat[:, spot_idx[0]] = expr_mat_
    
    #expr_mat = torch.from_numpy(expr_mat)  
    #expr_mat = tl.fold(expr_mat, 0, (n_g, n_x, n_y))
    expr_mat = expr_mat.reshape(n_g, n_x, n_y)
    expr_mat = torch.from_numpy(expr_mat) 
    expr_tensor = expr_mat.to_sparse()
    
    return expr_tensor, A_g, A_xy, feature_ids_PPI, gene_names_PPI, mapping


def construct_knn_graph(X, n_neighbors):
    n = X.shape[0]
    A = np.zeros((n, n))
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    neighborhood = nbrs.kneighbors(X, return_distance=False)
    
    for i in range(n):
        A[i, neighborhood[i, :]] = 1
        A[neighborhood[i, :], i] = 1
        
    np.fill_diagonal(A, 0)
    
    return A