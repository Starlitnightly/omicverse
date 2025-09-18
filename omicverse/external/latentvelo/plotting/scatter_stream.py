import numpy as np
import scanpy as sc
import scvelo as scv
import anndata as ad
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, sem
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

def scatter_stream(adata, gene, cluster_key = None, save=False, name_pre = '',
                  min_density= 1,ax=None):
    
    plt.rcParams.update({'font.size': 18})
    
    s_gene = np.array(adata[:,adata.var.index==gene].layers['spliced'])[:,0]
    u_gene = np.array(adata[:,adata.var.index==gene].layers['unspliced'])[:,0]
    
    adata.obsm['X_'+gene] = np.concatenate((s_gene[:,None], u_gene[:,None]), axis=-1)
    
    vs_gene = np.array(adata[:,adata.var.index==gene].layers['velo_s'])[:,0]
    vu_gene = np.array(adata[:,adata.var.index==gene].layers['velo_u'])[:,0]
    
    
    x = np.linspace(s_gene.min()*1.1, s_gene.max()*0.9, 250)
    y = np.linspace(u_gene.min()*1.1, u_gene.max()*0.9, 250)

    S, U = np.meshgrid(x, y)
    
    VS = griddata((s_gene, u_gene), vs_gene, (S, U), method='linear')
    VU = griddata((s_gene, u_gene), vu_gene, (S, U), method='linear')
    
    kernel = gaussian_kde(adata.obsm['X_'+gene][(s_gene > 0) & (u_gene > 0)].T)
    
    positions = np.vstack([S.ravel(), U.ravel()])
    density = np.reshape(kernel(positions).T, S.shape)
    
    VS[density< min_density] = np.nan
    VU[density< min_density] = np.nan
    
    adata.obs['gene_s_'+gene] = s_gene
    adata.obs['gene_u_'+gene] = u_gene
    
    if ax == None:
        fig,ax=plt.subplots()
        plt.streamplot(S, U, VS, VU, color='k', linewidth=1.5, arrowsize=1.5)
        
        scv.pl.scatter(adata, x='gene_s_'+gene,y='gene_u_'+gene,  color=cluster_key, ax=ax, title = gene, 
                  frameon='artist', legend_loc='none', show=False, size=100, legend_fontsize=18,
                  xlabel='spliced', ylabel='unspliced')
        
        if save:
            plt.savefig('figures/' + name_pre + str(gene) + '.pdf')
        else:
            return fig
        
    else:
        ax.streamplot(S, U, VS, VU, color='k', linewidth=1.5, arrowsize=1.5)
        
        scv.pl.scatter(adata, x='gene_s_'+gene,y='gene_u_'+gene,  color=cluster_key, ax=ax, title = gene, 
                  frameon='artist', legend_loc='none', show=False, size=100, legend_fontsize=18,
                  xlabel='spliced', ylabel='unspliced')


def scatter_stream_chromatin(adata, gene, cluster_key = None, save=False, name_pre = '',
                  min_density= 1,ax=None):
    
    plt.rcParams.update({'font.size': 18})
    
    u_gene = np.array(adata[:,adata.var.index==gene].layers['unspliced'])[:,0] #[:,None]
    c_gene = np.array(adata[:,adata.var.index==gene].layers['Mc'].todense())[:,0] #[:,None]
    
    adata.obsm['X_'+gene] = np.concatenate((u_gene[:,None], c_gene[:,None]), axis=-1)
    
    vu_gene = np.array(adata[:,adata.var.index==gene].layers['velo_u'])[:,0]
    vc_gene = np.array(adata[:,adata.var.index==gene].layers['velo_c'])[:,0]
    
    
    x = np.linspace(u_gene.min()*1.1, u_gene.max()*0.9, 250)
    y = np.linspace(c_gene.min()*1.1, c_gene.max()*0.9, 250)

    U, C = np.meshgrid(x, y)
    
    VU = griddata((u_gene, c_gene), vu_gene, (U, C), method='linear')
    VC = griddata((u_gene, c_gene), vc_gene, (U, C), method='linear')
    
    kernel = gaussian_kde(adata.obsm['X_'+gene][(u_gene > 0) & (c_gene > 0)].T)
    
    positions = np.vstack([U.ravel(), C.ravel()])
    density = np.reshape(kernel(positions).T, C.shape)
    
    VS[density< min_density] = np.nan
    VU[density< min_density] = np.nan
    
    adata.obs['gene_u_'+gene] = u_gene
    adata.obs['gene_c_'+gene] = c_gene
    
    if ax == None:
        fig,ax=plt.subplots()
        plt.streamplot(U, C, VU, VC, color='k', linewidth=1.5, arrowsize=1.5)
        
        scv.pl.scatter(adata, x='gene_u_'+gene,y='gene_c_'+gene,  color=cluster_key, ax=ax, title = gene, 
                  frameon='artist', legend_loc='none', show=False, size=100, legend_fontsize=18,
                  xlabel='spliced', ylabel='unspliced')
        
        if save:
            plt.savefig('figures/' + name_pre + str(gene) + '.pdf')
        else:
            return fig
        
    else:
        ax.streamplot(U, C, VU, VC, color='k', linewidth=1.5, arrowsize=1.5)
        
        scv.pl.scatter(adata, x='gene_u_'+gene,y='gene_c_'+gene,  color=cluster_key, ax=ax, title = gene, 
                  frameon='artist', legend_loc='none', show=False, size=100, legend_fontsize=18,
                  xlabel='spliced', ylabel='unspliced')
