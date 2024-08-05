from typing import Optional
import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse
from scipy.spatial import distance_matrix
import anndata
from .._optimal_transport import cot_sparse
from .._optimal_transport import cot_combine_sparse
from .._optimal_transport import uot
from .._optimal_transport import usot

def pairwise_scc(X1, X2):
    X1 = X1.argsort(axis=1).argsort(axis=1)
    X2 = X2.argsort(axis=1).argsort(axis=1)
    X1 = (X1-X1.mean(axis=1, keepdims=True))/X1.std(axis=1, keepdims=True)
    X2 = (X2-X2.mean(axis=1, keepdims=True))/X2.std(axis=1, keepdims=True)
    sccmat = np.empty([X1.shape[0], X2.shape[0]], float)
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            c = np.dot( X1[i,:], X2[j,:]) / float(X1.shape[1])
            sccmat[i,j] = c
    return sccmat

def infer_spatial_information(
    adata_sc: anndata.AnnData,
    adata_sp: anndata.AnnData,
    cost_sc_sp: Optional[np.ndarray] = None,
    cost_sc: Optional[np.ndarray] = None,
    cost_sp: Optional[np.ndarray] = None,
    ot_alpha: float = 0.2,
    ot_rho: float = 0.05,
    ot_epsilon: float = 0.01,
    exp_pred_prone: float = 0.0,
    loc_pred_k: int = 1,
    return_gamma: bool = False
):
    """
    Infer spatial information.

    Given a pair of spatial data and scRNA-seq data, estimate spatial origin
    of scRNA-seq data and impute gene expression for spatial data [Cang2020]_.

    Parameters
    ----------
    adata_sc
        The data matrix for scRNA-seq data of shape ``n_obs(sc)`` × ``n_vars(sc)``.
        Rows corresponds to cells and columns to genes.
    adata_sp
        The data matrix for spatial data of shape ``n_obs(sp)`` × ``n_vars(sp)``.
        Rows corresponds to positions and columns to genes.
    cost_sc_sp
        The dissimilarity matrix between scRNA-seq data and spatial data of shape ``n_obs(sc)`` × ``n_obs(sp)``.
        If not given, 1 - Spearman's r on common genes is used.
    cost_sc
        The dissimilarity matrix within scRNA-seq data of shape ``n_obs(sc)`` × ``n_obs(sc)``.
        Only needed when structured optimal transport is used (ot_alpha > 0).
        If not given, the Euclidean distance in PCA space is used.
    cost_sp
        The distance matrix within spatial data of shape ``n_obs(sp)`` × ``n_obs(sp)``.
        Only needed when structured optimal transport is used (ot_alpha > 0).
        If not given, the spatial distance among spatial locations is used.
    ot_alpha
        Weight for the structured component in optimal transport in [0,1]. 
    ot_rho
        Marginal relaxtion term (>0). Traditional OT when ot_rho=inf.
    ot_epsilon
        Entropy regularization term (>0). A higher value will generate a denser mapping matrix.
    exp_pred_prone
        The percentage of cells with low weights to ignore when predicing gene expression for each spatial data. 
        A higher percentage will increase the sparseness of the predicted spatial data due to the sparseness in the scRNA-seq data.
    loc_pred_k
        Number of top spatial matches for predicting spatial origin of cells. 
    return_gamma
        Whether to return the optimal transport plan (gamma matrix)

    Returns
    -------
    adata_sc_pred : anndata.AnnData
        The scRNA-seq data with predicted spatial origins in ``.obsm['spatial']``.
    adata_sp_pred : anndata.AnnData
        The spatial data with imputed gene expression.
    gamma : np.ndarray
        The connectivity matrix between scRNA-seq data and spatial data which is used as weights to generate the predicted datasets adata_sc_pred and adata_sp_pred.


    References
    ----------

    .. [Cang2020] Cang, Z., & Nie, Q. (2020). Inferring spatial and signaling 
        relationships between cells from single cell transcriptomic data. 
        Nature communications, 11(1), 1-13.

    """
    
    # Cost matrices for optimal transport
    if cost_sc_sp is None:
        common_genes = list(set(adata_sc.var_names).intersection(set(adata_sp.var_names)))
        adata_sc_common = adata_sc[:, common_genes]
        adata_sp_common = adata_sp[:, common_genes]
        cor_sc_sp = pairwise_scc(np.array(adata_sc_common.X.toarray()), np.array(adata_sp_common.X.toarray()))
        cost_sc_sp = 0.5 * ( 1.0 - cor_sc_sp )
    if cost_sc is None and ot_alpha != 0.0:
        adata_sc_processed = adata_sc.copy()
        sc.pp.highly_variable_genes(adata_sc_processed, n_top_genes=1000)
        adata_sc_processed = adata_sc_processed[:, adata_sc_processed.var.highly_variable]
        sc.pp.scale(adata_sc_processed, max_value=10)
        sc.tl.pca(adata_sc_processed, svd_solver='arpack', n_comps=20)
        x_pca = adata_sc_processed.obsm["X_pca"]
        cost_sc = distance_matrix(x_pca, x_pca)
    if cost_sp is None and ot_alpha != 0.0:
        cost_sp = distance_matrix(adata_sp.obsm["spatial"], adata_sp.obsm["spatial"])
    
    if ot_alpha != 0:
        cost_sp = cost_sp / np.max(cost_sp)
        cost_sc = cost_sc / np.max(cost_sc)
    cost_sc_sp = cost_sc_sp / np.max(cost_sc_sp)

    # Optimal transport mapping for the two datasets
    w_mat = np.exp(-cost_sc_sp)
    w_sc = np.sum(w_mat, axis=1); w_sp = np.sum(w_mat, axis=0)
    w_sc = w_sc / np.sum(w_sc); w_sp = w_sp / np.sum(w_sp)

    if ot_alpha == 0.0 and np.isinf(ot_rho):
        import ot
        gamma = ot.sinkhorn(w_sc, w_sp, cost_sc_sp, ot_epsilon)
    elif ot_alpha == 0.0 and not np.isinf(ot_rho):
        gamma = uot(w_sc, w_sp, cost_sc_sp, ot_epsilon, rho = ot_rho)
    else:
        gamma = usot(w_sc, w_sp, cost_sc_sp, cost_sc, cost_sp, ot_alpha,
                            epsilon = ot_epsilon, rho = ot_rho)

    # Spatial data with predicted gene expression
    gamma_sp = gamma.copy()
    if exp_pred_prone > 0.0:
        tmp_quantile = np.quantile(gamma_sp, exp_pred_prone, axis=0)
        tmp_gamma_sp = gamma_sp - tmp_quantile
        gamma_sp[np.where(tmp_gamma_sp < 0)] = 0
    gamma_sp = gamma_sp / gamma_sp.sum(axis=0)
    if isinstance(adata_sc.X, np.ndarray):
        X_sp_pred = np.matmul(gamma_sp.T, np.array(adata_sc.X))
    elif isinstance(adata_sc.X, sparse.csr_matrix):
        X_sp_pred = sparse.csr_matrix( gamma_sp.T * adata_sc.X )
    adata_sp_pred = anndata.AnnData(X=X_sp_pred, var=pd.DataFrame(index=adata_sc.var_names))
    adata_sp_pred.obsm["spatial"] = adata_sp.obsm["spatial"]

    # Single-cell data with predicted spatial location
    gamma_sc = gamma.copy()
    gamma_sc_sorted = -np.sort(-gamma_sc, axis=1)
    k = loc_pred_k
    for i in range(gamma_sc.shape[0]):
        gamma_sc[i,np.where(gamma_sc[i,:] < gamma_sc_sorted[i,k])[0]] = 0
        gamma_sc[i,:] = gamma_sc[i,:] / np.sum(gamma_sc[i,:])
    pos_pred = np.matmul(gamma_sc, adata_sp.obsm["spatial"])
    adata_sc_pred = adata_sc.copy()
    adata_sc_pred.obsm["spatial"] = pos_pred

    if return_gamma:
        return adata_sc_pred, adata_sp_pred, gamma
    else:
        return adata_sc_pred, adata_sp_pred
