import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scanpy.external as sce
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.io import mmread
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import svds
import gc
import sys
from sklearn.neighbors import NearestNeighbors



def fit_knn(mat_train, mat_holdout, n_neighbors, algorithm = 'kd_tree'):
    
    # fit knn using mat_train
    # return nn indices and distances in train set for holdout set
    knn = NearestNeighbors(n_neighbors = n_neighbors, algorithm = algorithm).fit(mat_train)
    distances, indices = knn.kneighbors(mat_holdout)
    indices = indices[:,1:]
    distances = distances[:,1:]

    return indices, distances



def calc_knn_prop(knn_indices, labels_train, label_categories):

    # knn_indices: shape = (n_holdout_samples, (knn-1)), np.array
    # labels_train: shape = (n_train_samples, ), pd.object
    # label_categories: shape = (n_label_categories, ), np.array
    n = knn_indices.shape[0]
    n_category = label_categories.shape[0]
    nn_prop = np.zeros(shape = (n, n_category))

    for i in range(n):
        knn_labels = labels_train[knn_indices[i,]]
        for k in range(n_category):
            nn_prop[i, k] = sum(knn_labels == label_categories[k]) 

    nn_prop = nn_prop / knn_indices.shape[1]
    return nn_prop


def calc_oobNN(adata_orig, batch_key, condition_key, n_neighbors=15):
    ''' Compute out-of-batch k-nearest-neighbor composition 

    Parameters
    ----------
    adata_orig : anndata object
        Expression data stored in adata_orig.X, based on which to compute out of batch nearest neighbors.
    batch_key : str
        Variable name indicating batch. Should be a column name of adata.obs.
    condition_key : str
        Variable name indicating condition. Should be a column name of adata.obs. 
        We compute out-of-batch proportion of each condition level within each cell's neighborhood.
    n_neighbors : int, optional
        Number of k-nearest neighbors.
    
    Returns
    ----------
    res: anndata object
        One new attribute added. 
        res.obsm['knn_prop'] : pd.DataFrame, out-of-batch k-nearest-neighbor composition, cell-by-condition
    '''

    np.random.seed(123)
    
    list_holdout = []
    for holdout_idx in np.unique(adata_orig.obs[batch_key]):

        adata_train = adata_orig[~adata_orig.obs[batch_key].isin([holdout_idx])]
        adata_holdout = adata_orig[adata_orig.obs[batch_key].isin([holdout_idx])]
        num_cells = adata_train.obs[condition_key].value_counts().min()
    
        a_list = []
        for x in np.unique(adata_train.obs[condition_key]):
            a1 = adata_train[adata_train.obs[condition_key].isin([x])]
            random_indices = np.random.choice(a1.shape[0], size=num_cells, replace = False)
            a1 = a1[random_indices,:]
            a_list.append(a1)
 
        adata_train = ad.concat(a_list)
        adata = ad.concat([adata_train, adata_holdout])
    
        mat = sc.tl.pca(adata.X, n_comps = 20)
        mat_train = mat[~adata.obs[batch_key].isin([holdout_idx]),]
        mat_holdout = mat[adata.obs[batch_key].isin([holdout_idx]),]
    
        # fit knn
        indices, distances = fit_knn(mat_train=mat_train, mat_holdout=mat_holdout, n_neighbors=n_neighbors, algorithm = 'kd_tree')

        # compute proprotion
        labels_train = adata_train.obs[condition_key].astype('object')
        label_categories = np.unique(labels_train)
        result = calc_knn_prop(indices, labels_train, label_categories)
        knn_df = pd.DataFrame(data=result, 
                              index =  adata_holdout.obs_names,
                              columns = label_categories)
        adata_holdout.obsm['knn_prop'] = knn_df
        list_holdout.append(adata_holdout)

    res = ad.concat(list_holdout)
    return res


