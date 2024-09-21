import scanpy as sc
import numpy as np

import torch
#from tensorflow_probability import math as tm
#tfk = tm.psd_kernels
#import spatial_factorization as sf
from typing import Optional
from scipy.sparse import csr_matrix
from ._townes_nsf_utils import *
from sklearn.decomposition import NMF

def construct_gems_using_pyliger(adata: sc.AnnData,
                                n_gems: int,
                                layer_key: str,
                                condition_key: str):
    import pyliger
    conditions = adata.obs[condition_key].unique().tolist()
    ad = adata.copy()

    ad.X = csr_matrix(ad.layers[layer_key].copy())
    ad.obs.index.name = 'index'
    ad.var.index.name = 'index'

    # Create LIGER object
    adata_list = []
    for cond in conditions:
        adata_cond =  adata[adata.obs[condition_key] == cond].copy()
        adata_cond.uns['sample_name'] = cond
        adata_list.append(adata_cond)

    adata_liger = pyliger.create_liger(adata_list, make_sparse=True)

    pyliger.normalize(adata_liger)
    pyliger.select_genes(adata_liger)
    pyliger.scale_not_center(adata_liger)

    # Save the var_names that were used for the NMF
    # adata_burkhardt.uns['pyliger_vars'] = burkhardt_liger.adata_list[0].var_names.tolist()

    pyliger.optimize_ALS(adata_liger, k = n_gems)

    # # Save results to adata
    X_gem = np.zeros((adata.n_obs, n_gems))
    pyliger_info = {}

    for i, cond in enumerate(conditions):
        cond_indices = np.where(adata.obs[condition_key] == cond)[0]
        X_gem[cond_indices] = adata_liger.adata_list[i].obsm['H']

        pyliger_info[cond] = {'H': adata_liger.adata_list[i].obsm['H'],
                            'W': adata_liger.adata_list[i].varm['W'],
                            'V': adata_liger.adata_list[i].varm['V']}

    adata.uns['pyliger_info'] = pyliger_info
    adata.uns['pyliger_info']['vars'] = adata_liger.adata_list[0].var_names.tolist()        
    adata.uns['pyliger_info']['n_gems'] =  n_gems
    adata.obsm['X_gem'] = X_gem

def construct_gems_using_nsf(adata: sc.AnnData,
                            n_gems: int,
                            layer_key: str,
                            spatial_key: str = "spatial",
                            n_inducing_pts: int = 500,
                            length_scale: float = 10.0):
    import spatial_factorization as sf
    import gpytorch
    ad = adata.copy()
    
    X = ad.obsm[spatial_key]
    # Take raw count data for NSF
    training_fraction = 1.0
    D, Dval = anndata_to_train_val(ad,
                                   layer=layer_key,
                                   train_frac=training_fraction,
                                   flip_yaxis=True)
    Ntr, J = D["Y"].shape
    Xtr = D["X"]
    ad = adata[:Ntr, :]
    
    # Convert to PyTorch tensors
    Dtf = prepare_datasets_torch(D, Dval=Dval)

    Z = kmeans_inducing_pts(Xtr, n_inducing_pts)
    M = Z.shape[0]  # number of inducing points

    # Define the Matern 3/2 kernel using gpytorch
    class MaternThreeHalvesKernel(gpytorch.kernels.MaternKernel):
        def __init__(self, **kwargs):
            super(MaternThreeHalvesKernel, self).__init__(nu=1.5, **kwargs)

    ker = MaternThreeHalvesKernel(lengthscale=length_scale)

    fit = sf.SpatialFactorization(J, n_gems, Z, psd_kernel=ker, length_scale=length_scale, nonneg=True, lik="poi")
    fit.init_loadings(D["Y"], X=Xtr, sz=D["sz"], shrinkage=0.3)
    tro = sf.ModelTrainer(fit)
    tro.train_model(*Dtf, status_freq=50)  # about 3 mins

    insf = interpret_nsf(fit, Xtr, S=100, lda_mode=False)

    adata.uns['nsf_info'] = insf
    adata.uns['nsf_info']['vars'] = adata.var_names.tolist()
    adata.uns['nsf_info']['n_gems'] = n_gems
    adata.obsm['X_gem'] = insf['factors']

# Helper function to convert datasets to PyTorch tensors
def prepare_datasets_torch(D, Dval=None):
    Dtorch = {
        "X": torch.tensor(D["X"], dtype=torch.float32),
        "Y": torch.tensor(D["Y"], dtype=torch.float32),
        "sz": torch.tensor(D["sz"], dtype=torch.float32)
    }
    if Dval is not None:
        Dtorch["Xval"] = torch.tensor(Dval["X"], dtype=torch.float32)
        Dtorch["Yval"] = torch.tensor(Dval["Y"], dtype=torch.float32)
        Dtorch["szval"] = torch.tensor(Dval["sz"], dtype=torch.float32)
    return Dtorch

def construct_gems_using_nmf(adata: sc.AnnData,
                                n_gems: int,
                                layer_key: str, 
                                random_state: int = 0,
                                max_iter: int = 1000):

    X_expr = adata.layers[layer_key].copy()
    
    model = NMF(n_components=n_gems, init='random', random_state=random_state, max_iter=max_iter)

    W = model.fit_transform(X_expr)
    H = model.components_

    W_sum = W.sum(axis=0)
    W_lda = W / W_sum

    H_scaled = H.T * W_sum
    H_sum = H_scaled.sum(axis=1)
    H_lda = (H_scaled.T / H_sum).T

    fact_orders = np.argsort(-H_lda.sum(axis=0))

    W_lda = W_lda[:, fact_orders]
    H_lda = H_lda[:, fact_orders].T

    adata.uns['nmf_info'] = {'n_gems': n_gems,
                             'vars': adata.var_names.tolist(),
                             'factors':W_lda,
                             'loadings':H_lda,
                             'totals':W_sum}

    adata.obsm['X_gem'] = W_lda
