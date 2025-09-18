import scanpy as sc
import scvelo as scv
import numpy as np
import scipy as scp
import pandas as pd

from scipy.stats import ranksums
from scipy.spatial.distance import cdist

def cell_trajectories(traj, times, latent_adata, adata, cells,  min_time):
    
    Z = np.concatenate((latent_adata.layers['spliced'],
                        latent_adata.layers['unspliced'],
                        latent_adata.obsm['zr']), axis=-1)
    
    X_traj = []
    ids = []
    for i in range(traj.shape[0]):
        dist = cdist(np.array(traj)[i], Z)
        selected = np.argmin(dist, axis=1)
        
        selected_traj = np.array(adata[selected].layers['spliced'])
        max_time = latent_adata[cells[i]].obs['latent_time'].values[0]
        selected_traj = selected_traj[(times[i,:,0] > min_time) & (times[i,:,0] < max_time)]
        
        X_traj.append(selected_traj)
        ids.append(selected[None])
    X_traj = np.concatenate(X_traj, axis=0)
    return X_traj

def de_genes(adata, df, celltype, celltype_key='clusters', mode='greater'):
    
    genes = []
    statistic = []
    pvals = []
    for i,gene in enumerate(adata.var.index.values):
        test = ranksums(df[df[celltype_key] == celltype][gene], 
                        df[df[celltype_key] != celltype][gene],
                        alternative=mode)
        genes.append(gene)
        statistic.append(test.statistic)
        pvals.append(test.pvalue)
    return pd.DataFrame({'gene': genes, 'statistic': statistic, 'pval': pvals})
    
