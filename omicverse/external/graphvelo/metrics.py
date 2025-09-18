import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import cosine_similarity


def v_delta_regress_score(X, V, nbrs, alpha=1.0):
    P = np.zeros((X.shape[0], X.shape[0]))
    for i, x in enumerate(X):
        v = V[i]
        idx = nbrs[i]

        U = X[idx] - x

        clf = Lasso(alpha=alpha)
        clf.fit(U, v)
        #print(clf.coef_)
        P[i][idx] = clf.coef_
    return P.sum(1)


def cross_boundary_correctness(adata, 
                               xkey:str='Ms',
                               vkey:str='velocity',
                               cluster_key:str='clusters', 
                               cluster_edges:list=None, 
                               basis:str='pca',
                               neighbor_key:str='neighbors',
                               vector:str='velocity', 
                               corr_func='cosine',
                               return_raw:bool=False):
    scores = {}
    all_scores = {}
    
    if basis != 'raw' and 'X_' + basis in adata.obsm.keys():
        X = adata.obsm['X_'+basis]
        V = adata.obsm[vector+'_'+basis]
    else:
        X = adata.layers[xkey].A if issparse(adata.layers[xkey]) else adata.layers[xkey]
        V = adata.layers[vkey].A if issparse(adata.layers[vkey]) else adata.layers[vkey]
        V[np.isnan(V)] = 0
    nbrs_idx = adata.uns[neighbor_key]['indices'] # [n * 30]
        
    def keep_type(adata, nodes, cluster, cluster_key):
        return nodes[adata.obs[cluster_key][nodes].values == cluster]

    for u, v in cluster_edges:
        sel = adata.obs[cluster_key] == u
        nbrs = nbrs_idx[sel]
        
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, cluster_key), nbrs)
        x_points = X[sel]
        x_velocities = V[sel]
        
        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0: continue

            position_dif = X[nodes] - x_pos
            if corr_func == 'cosine':
                dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
            elif corr_func == 'pearson':
                dir_scores = np.zeros(position_dif.shape[0])
                for i in range(position_dif.shape[0]):
                    dir_scores[i] = np.corrcoef(position_dif[i], x_vel)[0, 1]
            else:
                raise ValueError("corr_func should be one of `cosine` and `pearson`.")
            type_score.append(np.mean(dir_scores))
        
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
        
    if return_raw:
        return all_scores 
    
    return scores, np.mean([sc for sc in scores.values()])


def relative_flux_correctness(
    adata, 
    k_cluster, 
    k_transition_matrix, 
    cluster_transitions):
    """Relative Flux Direction Correctness Score (A->B) on the transition matrix
    Adapted from topicvelo
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame
        k_transition_matrix (str): 
            key to the transition matrix in adata.obsp
        cluster_transitions (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        
    Returns:
        rel_flux (dict):
            relative flux from A->B
        flux (dict): 
            forward and reverse flux between A and B
    """
    flux = {}
    rel_flux = {}
    for A, B in cluster_transitions:
        A_inds = np.where(adata.obs[k_cluster] == A)[0]
        B_inds = np.where(adata.obs[k_cluster] == B)[0]
        A_to_B = 0
        for b in B_inds:
            A_to_B += np.sum(adata.obsp[k_transition_matrix][A_inds,b])  
        B_to_A = 0
        for a in A_inds:
            B_to_A += np.sum(adata.obsp[k_transition_matrix][B_inds,a])  
        #normalization
        # A_to_B = A_to_B/len(A_inds)
        # B_to_A = B_to_A/len(B_inds)
        flux[(A, B)] = A_to_B
        flux[(B, A)] = B_to_A
        rel_flux[(A,B)] = (A_to_B-B_to_A)/(A_to_B+B_to_A)
    adata.uns[k_transition_matrix+'_flux'] = flux
    adata.uns[k_transition_matrix+'_rel_flux']=rel_flux
    return rel_flux, flux