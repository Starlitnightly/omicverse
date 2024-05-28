from ..STT import tl,pl
import scanpy as sc
import numpy as np
import pandas as pd

class STT(object):
    
    def __init__(self,adata,spatial_loc='xy_loc',region='Region'):
        
        self.adata=adata
        self.adata_aggr=None
        self.spatial_loc=spatial_loc
        self.region=region
        if 'attractor' not in self.adata.obs.keys():
            self.adata.obs['attractor'] = self.adata.obs[region]
        
    def stage_estimate(self):
        U = self.adata.layers['unspliced']
        S = self.adata.layers['spliced']
        if 'toarray' in dir(U):
            U = U.toarray()
            S = S.toarray()
        X_all = np.concatenate((U,S),axis = 1)
        adata_aggr = sc.AnnData(X=X_all)
        sc.tl.pca(adata_aggr, svd_solver='arpack')
        sc.pp.neighbors(adata_aggr)
        sc.tl.leiden(adata_aggr,resolution = 0.15)
        self.adata.obs['joint_leiden'] = adata_aggr.obs['leiden'].values
        print(f"...estimate stage: {len(self.adata.obs['joint_leiden'].unique())}")
        
    def train(self,n_states = 9, n_iter = 15, weight_connectivities = 0.5, 
                   n_neighbors = 50,thresh_ms_gene = 0.2, spa_weight =0.3,**kwargs):
        self.adata_aggr = tl.dynamical_iteration(self.adata,n_states = n_states, 
                                       n_iter = n_iter, weight_connectivities = weight_connectivities, 
                                       n_neighbors = n_neighbors,thresh_ms_gene = thresh_ms_gene, spa_weight =spa_weight,
                                                   **kwargs)
        self.adata.obsm[f'X_{self.spatial_loc}'] = self.adata.obsm[self.spatial_loc]
        self.adata_aggr.obsm[f'X_{self.spatial_loc}']=self.adata.obsm[self.spatial_loc]
        self.adata_aggr.obsm[f'X_{self.spatial_loc}_aggr']=self.adata.obsm[self.spatial_loc]
        self.adata.obsm[f'X_{self.spatial_loc}_aggr']=self.adata.obsm[self.spatial_loc]

    def load(self,adata,adata_aggr):
        self.adata=adata 
        self.adata_aggr=adata_aggr

        
    def compute_pathway(self,pathway_dict):
        return tl.compute_pathway(self.adata,self.adata_aggr,pathway_dict)
        
    def plot_pathway(self,label_fontsize=20,**kwargs):
        fig = pl.plot_pathway(self.adata,**kwargs)
        for ax in fig.axes:
            ax.set_xlabel('Embedding 1', fontsize=label_fontsize)  # Adjust font size as needed
            ax.set_ylabel('Embedding 2', fontsize=label_fontsize)  # Adjust font size as needed
        return fig
    
    def plot_tensor_pathway(self,pathway_name,**kwargs):
        ax=pl.plot_tensor_pathway(self.adata,self.adata_aggr, 
                                  pathway_name = pathway_name,**kwargs)
        return ax
    
    def plot_tensor(self,list_attractor,**kwargs):
        return pl.plot_tensor(self.adata, self.adata_aggr, 
                          list_attractor = list_attractor,basis = self.spatial_loc,**kwargs)

    
    def construct_landscape(self,coord_key = 'X_xy_loc',**kwargs):
        tl.construct_landscape(self.adata, coord_key = coord_key,**kwargs)
        self.adata.obsm['trans_coord'] = self.adata.uns['land_out']['trans_coord']

    def infer_lineage(self,**kwargs):
        return pl.infer_lineage(self.adata,**kwargs)

    def plot_landscape(self,**kwargs):
        return pl.plot_landscape(self.adata,**kwargs)
    
    def plot_sankey(self,vector1, vector2):
        return pl.plot_sankey(vector1, vector2)
    
    def plot_top_genes(self,**kwargs):
        return pl.plot_top_genes(self.adata,**kwargs)
    
