import numpy as np
import scanpy as sc
import scvelo as scv
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr, sem

import pandas as pd
from ..utils import paired_cosine_similarity, average_velocity

from .metrics import pseudotime_correlation, velocity_magnitude_correlation, true_velocity_cosine, cross_boundary_correctness, inner_cluster_coh, nn_velo, true_velocity_correlation


def format_benchmark_adata(adata, latent_adata, estimated_vkey='velocity', sim_vkey = 'rna_velocity', latent_tkey = 'latent_time', sim_tkey = 'sim_time', use_velo_genes = False, latent=True):
    """
    Format dataset to run benchmarks for synthetic data (based on dyngen simulations)
    """
    if use_velo_genes:
        adata.layers[sim_vkey][:,adata.var['velocity_genes'] == False] = np.nan
        adata.layers['gene_velocity'][:,adata.var['velocity_genes'] == False] = np.nan
    
    if latent:
        adata.obsm['X_latent'] = latent_adata.X
        adata.obsm['gene_velocity_latent'] = latent_adata.layers[estimated_vkey]
    latent_adata.obsm['X_pca'] = adata.obsm['X_pca']
    latent_adata.obsm['X_pca_20'] = adata.obsm['X_pca'][:,:20]
    latent_adata.obsm['X_pca_10'] = adata.obsm['X_pca'][:,:10]
    latent_adata.obsm['X_pca_3'] = adata.obsm['X_pca'][:,:3]
    adata.obsm['X_pca_20'] = adata.obsm['X_pca'][:,:20]
    adata.obsm['X_pca_10'] = adata.obsm['X_pca'][:,:10]
    adata.obsm['X_pca_3'] = adata.obsm['X_pca'][:,:3]
    scv.tl.velocity_embedding(latent_adata, basis='pca', vkey=estimated_vkey)
    scv.tl.velocity_embedding(latent_adata, basis='pca_20', vkey=estimated_vkey)
    scv.tl.velocity_embedding(latent_adata, basis='pca_10', vkey=estimated_vkey)
    scv.tl.velocity_embedding(latent_adata, basis='pca_3', vkey=estimated_vkey)
    scv.tl.velocity_embedding(latent_adata, basis='umap', vkey=estimated_vkey)
    adata.obsm['gene_velocity_pca'] = latent_adata.obsm[estimated_vkey+'_pca']
    adata.obsm['gene_velocity_pca_20'] = latent_adata.obsm[estimated_vkey+'_pca_20']
    adata.obsm['gene_velocity_pca_10'] = latent_adata.obsm[estimated_vkey+'_pca_10']
    adata.obsm['gene_velocity_pca_3'] = latent_adata.obsm[estimated_vkey+'_pca_3']
    adata.obsm['gene_velocity_umap'] = latent_adata.obsm[estimated_vkey+'_umap']

    scv.tl.velocity_graph(adata, vkey='rna_velocity')
    if latent:
        scv.tl.velocity_embedding(adata, vkey='rna_velocity', basis='latent')
    scv.tl.velocity_embedding(adata, vkey='rna_velocity', basis='pca')
    scv.tl.velocity_embedding(adata, vkey='rna_velocity', basis='pca_20')
    scv.tl.velocity_embedding(adata, vkey='rna_velocity', basis='pca_10')
    scv.tl.velocity_embedding(adata, vkey='rna_velocity', basis='pca_3')
    scv.tl.velocity_embedding(adata, vkey='rna_velocity', basis='umap')
    
    adata.obs['latent_time'] = latent_adata.obs[latent_tkey].values
    
    adata.uns['neighbors_latent'] = latent_adata.uns['neighbors'].copy()
    adata.obsp['connectivities_latent'] = latent_adata.obsp['connectivities'].copy()
    adata.obsp['distances_latent'] = latent_adata.obsp['distances'].copy()
    adata.uns['neighbors_latent']['connectivities_key'] = 'connectivities_latent'


def benchmark_synthetic(adata, layer=True, estimated_vkey='gene_velocity', sim_vkey = 'rna_velocity', latent_tkey = 'latent_time', sim_tkey = 'sim_time', basis_list = ['latent'], cluster_edges = None, cluster_key = None, majority_vote=True, batch_key=None, avg_velocity=False, n_neighbors_avg = 100, n_pcs_avg=30):
    """
    Run benchmarks for synthetic data
    """
    results = {}
    results_all = {}

    # pseudotime correlation
    pt_correlation = pseudotime_correlation(adata, latent_time_key=latent_tkey,
                                         pseudotime_key = sim_tkey)
    results['pseudotime correlation'] = [pt_correlation]
    
    
    # velocity magnitude
    """
    if not avg_velocity:
        mag_correlation = velocity_magnitude_correlation(adata, layer=layer,
                                                         estimated_key=estimated_vkey,
                                                         true_key = sim_vkey,
                                                         plot=False)
        results['velocity magnitude correlation'] = [mag_correlation]
    """

    # velocity cosine correlation
    for basis in basis_list:

        if basis != '':
            cosine, cosine_mean = true_velocity_cosine(adata, layer=False,
                                                     estimated_key=estimated_vkey + '_'+basis,
                                                     true_key = sim_vkey + '_'+basis,
                                                     plot=False)
            results['velocity_cosine_'+basis] = cosine
            
            # cross boundary correctness
            if cluster_edges != None:
                transition_scores = cross_boundary_correctness(adata, cluster_key=cluster_key, velocity_key = estimated_vkey+'_'+basis, cluster_edges = cluster_edges, x_emb = 'X_'+basis, return_raw=True, majority_vote=majority_vote) # do umap for now..
                results['CBDir_' + basis] = np.concatenate([np.array(x) for x in transition_scores.values()])

            
            if cluster_key != None:
                coh_scores = inner_cluster_coh(adata, cluster_key=cluster_key, velocity_key=estimated_vkey+'_'+basis, return_raw=True, layer=False)
                results['ICCoh_' + basis] = np.concatenate([np.array(x) for x in transition_scores.values()])

            
            if batch_key != None:
                _, _, nn_velo_score = nn_velo(adata, batch_key=batch_key, vkey=estimated_vkey+'_'+basis, all=True, layer=False, neighbor_key='neighbors_latent')
                results['nn_velo_'+basis] = nn_velo_score
                
        else:

            if avg_velocity:
                adata.layers['avg_velo'] = average_velocity(adata, vkey=sim_vkey, n_pcs = n_pcs_avg,
                                                                       n_neighbors=n_neighbors_avg)
                
                cosine, cosine_mean = true_velocity_cosine(adata, layer=True,
                                                           estimated_key=estimated_vkey,
                                                           true_key = 'avg_velo',
                                                           plot=False)
                
                corr, corr_mean = true_velocity_correlation(adata, layer=True, estimated_key=estimated_vkey,
                                                            true_key='avg_velo', plot=False)
                """
                mag_correlation = velocity_magnitude_correlation(adata, layer=layer,
                                                                 estimated_key=estimated_vkey,
                                                                 true_key = 'avg_velo',
                                                                 plot=False)
                results['velocity magnitude correlation'] = [mag_correlation]
                """
                
                
            else:
                cosine, cosine_mean = true_velocity_cosine(adata, layer=True,
                                                           estimated_key=estimated_vkey,
                                                           true_key = sim_vkey,
                                                           plot=False)

                corr, corr_mean = true_velocity_correlation(adata, layer=True, estimated_key=estimated_vkey,
                                                            true_key=sim_vkey, plot=False)
            
            results['velocity_cosine'] = cosine
            results['velociy_correlation'] = corr


    max_len = max([len(results[key]) for key in results.keys()])
    for key in results.keys():
        if len(results[key]) < max_len:
            new_array = np.ones(max_len) * np.nan
            new_array[:len(results[key])] = results[key]
            results[key] = new_array
        
    return pd.DataFrame(results)
