r"""
Data preprocess and build graph
"""
from typing import Optional, Union, List

import pandas as pd
import numpy as np
import scanpy as sc
import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
from anndata import AnnData


def Cal_Spatial_Net(adata:AnnData,
                    rad_cutoff:Optional[Union[None,int]]=None,
                    k_cutoff:Optional[Union[None,int]]=None, 
                    model:Optional[str]='Radius',
                    return_data:Optional[bool]=False,
                    verbose:Optional[bool]=True
    ) -> None:
    r"""
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. 
        When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('Calculating spatial neighbor graph ...')

    if model == 'KNN':
        edge_index = knn_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source',
                                k=k_cutoff, loop=True, num_workers=8)
        edge_index = to_undirected(edge_index, num_nodes=adata.shape[0]) # ensure the graph is undirected
    elif model == 'Radius':
        edge_index = radius_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source',
                                    r=rad_cutoff, loop=True, num_workers=8) 

    graph_df = pd.DataFrame(edge_index.numpy().T, columns=['Cell1', 'Cell2'])
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    graph_df['Cell1'] = graph_df['Cell1'].map(id_cell_trans)
    graph_df['Cell2'] = graph_df['Cell2'].map(id_cell_trans)
    adata.uns['Spatial_Net'] = graph_df
    
    if verbose:
        print(f'The graph contains {graph_df.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{graph_df.shape[0]/adata.n_obs} neighbors per cell on average.')

    if return_data:
        return adata


def scanpy_workflow(adata:AnnData,
                    filter_cell:Optional[bool]=False,
                    min_gene:Optional[int]=200,
                    min_cell:Optional[int]=30,
                    call_hvg:Optional[bool]=True,
                    n_top_genes:Optional[Union[int, List]]=2500,
                    batch_key:Optional[str]=None,
                    n_comps:Optional[int]=50,
                    viz:Optional[bool]=False,
                    resolution:Optional[float]=0.8
    ) -> AnnData:
    r"""
    Scanpy workflow using Seurat HVG
    
    Parameters
    ----------
    adata
        adata
    filter_cell
        whether to filter cells and genes
    min_gene
        min number of genes per cell
    min_cell
        min number of cells per gene
    call_hvg
        whether to call highly variable genes (only support seurat_v3 method)
    n_top_genes
        n top genes or gene list
    n_comps
        n PCA components
    viz
        whether to run visualize steps
    resolution
        resolution for leiden clustering (used when viz=True)
        
    Return
    ----------
    anndata object
    """
    if 'counts' not in adata.layers.keys():
        adata.layers["counts"] = adata.X.copy()
        
    if filter_cell:
        sc.pp.filter_cells(adata, min_genes=min_gene)
        sc.pp.filter_genes(adata, min_cells=min_cell)

    if call_hvg:
        if isinstance(n_top_genes, int):
            if adata.n_vars > n_top_genes:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3", batch_key=batch_key)
            else:
                adata.var['highly_variable'] = True
                print("All genes are highly variable.")
        elif isinstance(n_top_genes, list):
            adata.var['highly_variable'] = False
            n_top_genes = list(set(adata.var.index).intersection(set(n_top_genes)))
            adata.var.loc[n_top_genes, 'highly_variable'] = True
    else:
        print("Skip calling highly variable genes.")
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    if n_comps > 0:
        sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
        
    if viz:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_comps)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)
    return adata