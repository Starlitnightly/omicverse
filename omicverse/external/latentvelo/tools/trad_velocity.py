"""
These are 'traditional' RNA velocity methods computed with scVelo.
"""

import scanpy as sc
import scvelo as scv
import numpy as np
import scipy as scp
import anndata as ad
import tempfile
import shutil

def compute_scvelo(adata_raw, min_shared_counts=30, n_top_genes=2000, n_pcs=30, n_neighbors=30, mode='both', n_jobs=1, embeddings=['umap', 'pca'], enforce=False):
    
    scv.settings.verbosity = 1
    
    adata = adata_raw.copy()
    
    scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes, enforce=enforce)
    scv.pp.pca(adata)
    scv.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    sc.tl.umap(adata)
    
    if mode == 'dynamical' or mode=='both':
        scv.tl.recover_dynamics(adata, n_jobs=n_jobs)
        scv.tl.velocity(adata, mode='dynamical', vkey='dynvelo', n_jobs=n_jobs)
        scv.tl.velocity_graph(adata, vkey='dynvelo', n_jobs=n_jobs)
        for embedding in list(embeddings):
            scv.tl.velocity_embedding(adata, basis=embedding, vkey='dynvelo')
        
    elif mode == 'stochastic' or mode =='both':
        scv.tl.velocity(adata, mode='stochastic', vkey='stocvelo', n_jobs=n_jobs)
        scv.tl.velocity_graph(adata, vkey='stocvelo', n_jobs=n_jobs)
        for embedding in list(embeddings):
            scv.tl.velocity_embedding(adata, basis=embedding, vkey='stocvelo')

    else:
        print('Error, choose mode="dynamical", "stochastic", or "both"')
    
    return adata


def compute_unitvelo(adata_raw, min_shared_counts=30, n_top_genes=2000, n_pcs=30, n_neighbors=30, n_jobs=1, embeddings=['umap', 'pca'], enforce=False, mode=1):
    import unitvelo as utv

    scv.settings.verbosity = 1
    
    adata = adata_raw.copy()
    
    scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes, enforce=enforce)
    scv.pp.pca(adata)
    scv.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    sc.tl.umap(adata)
    
    
    velo_config = utv.config.Configuration()
    velo_config.R2_ADJUST = True
    velo_config.IROOT = None
    velo_config.FIT_OPTION = str(mode)
    velo_config.N_TOP_GENES = adata.shape[1]
    velo_config.BASIS='umap'

    #dirpath = tempfile.mkdtemp()

    adata.obs['label'] = 0
    #adata.write(dirpath+'temp_adata.h5ad')
    adata.write('temp_adata.h5ad')
    
    #adata = utv.run_model(dirpath+'/temp_adata.h5ad', 'label', config_file=velo_config)
    adata = utv.run_model('./temp_adata.h5ad', 'label', config_file=velo_config)

    scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=n_jobs)
    for embedding in list(embeddings):
        scv.tl.velocity_embedding(adata, basis=embedding, vkey='velocity')
    
    return adata
