import scanpy as sc
import pandas as pd
import anndata
import numpy as np
from ._cosg import cosg
from ..palantir.plot import plot_palantir_results,plot_branch_selection,plot_gene_trends
from ..palantir.utils import run_diffusion_maps,determine_multiscale_space,run_magic_imputation
from ..palantir.core import run_palantir
from ..palantir.presults import select_branch_cells,compute_gene_trends


class TrajInfer(object):
    
    def __init__(self,adata:anndata.AnnData,
                 basis:str='X_umap',use_rep:str='X_pca',n_comps:int=50,
                 n_neighbors:int=15,
                groupby:str='clusters',):
        self.adata=adata
        self.use_rep=use_rep
        self.n_comps=n_comps
        self.basis=basis
        self.groupby=groupby
        self.n_neighbors=n_neighbors
        
        self.origin=None
        self.terminal=None
        
    def set_terminal_cells(self,terminal:list):
        self.terminal=terminal
        
    def set_origin_cells(self,origin:str):
        self.origin=origin
        
    def inference(self,method:str='palantir',**kwargs):
        
        if method=='palantir':

            dm_res = run_diffusion_maps(self.adata,
                                                       pca_key=self.use_rep, 
                                                       n_components=self.n_comps)
            ms_data = determine_multiscale_space(self.adata)
            imputed_X = run_magic_imputation(self.adata)

            sc.tl.rank_genes_groups(self.adata, groupby=self.groupby, 
                        method='t-test',use_rep=self.use_rep,)
            cosg(self.adata, key_added=f'{self.groupby}_cosg', groupby=self.groupby)
            
            ## terminal cells calculation
            terminal_index=[]
            for t in self.terminal:
                gene=sc.get.rank_genes_groups_df(self.adata, group=t, key=f'{self.groupby}_cosg')['names'][0]
                terminal_index.append(self.adata[self.adata.obs[self.groupby]==t].to_df()[gene].sort_values().index[-1])
            
            terminal_states = pd.Series(
                self.terminal,
                index=terminal_index,
            )
            #return terminal_states
            
            ## origin cells calculation
            origin_cell=self.origin
            gene=sc.get.rank_genes_groups_df(self.adata, group=origin_cell, key=f'{self.groupby}_cosg')['names'][0]
            origin_cell_index=self.adata[self.adata.obs[self.groupby]==origin_cell].to_df()[gene].sort_values().index[-1]
            
            start_cell = origin_cell_index
            pr_res = run_palantir(
                self.adata, early_cell=start_cell, terminal_states=terminal_states,
                **kwargs
            )
            
            self.adata.obs['palantir_pseudotime']=pr_res.pseudotime
            return pr_res
        elif method=='diffusion_map':
            sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, n_pcs=self.n_comps,
               use_rep=self.use_rep)
            sc.tl.diffmap(self.adata)
            sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, use_rep='X_diffmap')
            sc.tl.draw_graph(self.adata)
            self.adata.uns['iroot'] = np.flatnonzero(self.adata.obs[self.groupby]  == self.origin)[0]
            sc.tl.dpt(self.adata)
            sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, n_pcs=self.n_comps,
               use_rep=self.use_rep)
        else:
            print('Please input the correct method name, such as `palantir` or `diffusion_map`')
            return
        
    def palantir_plot_pseudotime(self,**kwargs):

        plot_palantir_results(self.adata,**kwargs)
        
    def palantir_cal_branch(self,**kwargs):

        masks = select_branch_cells(self.adata, **kwargs)
        plot_branch_selection(self.adata)

    def palantir_cal_gene_trends(self,layers:str="MAGIC_imputed_data"):

        gene_trends = compute_gene_trends(
            self.adata,
            expression_key=layers,
        )
        return gene_trends
        
    def palantir_plot_gene_trends(self,genes):
        #genes = ['Cdca3','Rasl10a','Mog','Aqp4']

        return plot_gene_trends(self.adata, genes)
