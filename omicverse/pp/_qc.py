"""
Copy from pegasus and cellual

"""


import os
import logging
import pandas as pd 
from functools import reduce
import anndata
import numpy as np 
from random import seed, sample
import scanpy as sc
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib import rcParams
import seaborn as sns
from typing import Union, Optional, Sequence, Tuple, List, Dict
from scipy.sparse import issparse


def mads(meta, cov, nmads=5, lt=None): 
    """
    Given a certain array, it calculate its Median Absolute Deviation (MAD).
    """
    x = meta[cov]
    mad = np.median(np.absolute(x - np.median(x)))
    t1 = np.median(x) - (nmads * mad) 
    t1 = t1 if t1 > 0 else lt[cov]
    t2 = np.median(x) + (nmads * mad) 
    return t1, t2

def mads_test(meta, cov, nmads=5, lt=None):
    """
    Given a certain array, it returns a boolean array with True values only at indeces 
    from entries within x < n_mads*mad and x > n_mads*mad.
    """
    tresholds = mads(meta, cov, nmads=nmads, lt=lt)
    return (meta[cov] > tresholds[0]) & (meta[cov] < tresholds[1])

def quantity_control(adatas, mode='seurat', min_cells=3, min_genes=200, nmads=5, path_viz=None, tresh=None):
    """
    Perform quality control on a dictionary of AnnData objects.
    
    This function calculates several QC metrics, including mitochondrial percentage, nUMIs, 
    and detected genes, and produces several plots visualizing the QC metrics for each sample. 
    The function performs doublet detection using scrublet and filtering using either 
    Seurat or MADs. The function returns a merged AnnData object with cells that passed QC filters
    and a list of cells that did not pass QC on all samples.

    Parameters
    ----------
    adatas : dict
        A dictionary of AnnData objects, one for each sample.
    mode : str, optional
        The filtering method to use. Valid options are 'seurat' and 'mads'. Default is 'seurat'.
    min_cells : int, optional
        The minimum number of cells for a sample to pass QC. Default is 3.
    min_genes : int, optional
        The minimum number of genes for a cell to pass QC. Default is 200.
    nmads : int, optional
        The number of MADs to use for MADs filtering. Default is 5.
    path_viz : str, optional
        The path to save the QC plots. Default is None.
    tresh : dict, optional
        A dictionary of QC thresholds. The keys should be 'mito_perc', 'nUMIs', and 'detected_genes'.
        Only used if mode is 'seurat'. Default is None.

    Returns
    -------
    adata : AnnData
        An AnnData object containing cells that passed QC filters.
    removed_cells : list
        List of cells that did not pass QC on all samples.
    """
    # Logging 
    logger = logging.getLogger("my_logger")  
    if tresh is None:
        tresh={'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}
    
    # For each adata, produce a figure
    # with PdfPages(path_viz + 'original_QC_by_sample.pdf') as pdf: 
    removed_cells = []
    for s, adata in adatas.items():


        print(f'Sample {s} QC...')

        # QC metrics
        print('Calculate QC metrics')
        adata.var_names_make_unique()
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        if issparse(adata.X):
            adata.obs['nUMIs'] = adata.X.toarray().sum(axis=1)  
            adata.obs['mito_perc'] = adata[:, adata.var["mt"]].X.toarray().sum(axis=1) / adata.obs['nUMIs'].values
            adata.obs['detected_genes'] = (adata.X.toarray() > 0).sum(axis=1)  
        else:
            adata.obs['nUMIs'] = adata.X.sum(axis=1)  
            adata.obs['mito_perc'] = adata[:, adata.var["mt"]].X.sum(axis=1) / adata.obs['nUMIs'].values
            adata.obs['detected_genes'] = (adata.X > 0).sum(axis=1)  
        adata.obs['cell_complexity'] = adata.obs['detected_genes'] / adata.obs['nUMIs']
        print(f'End calculation of QC metrics.')

        # Original QC plot
        n0 = adata.shape[0]
        print(f'Original cell number: {n0}')

        # Post doublets removal QC plot
        print('Begin of post doublets removal and QC plot')
        sc.external.pp.scrublet(adata, random_state=1234)
        adata_remove = adata[adata.obs['predicted_doublet'], :]
        removed_cells.extend(list(adata_remove.obs_names))
        adata = adata[~adata.obs['predicted_doublet'], :]
        n1 = adata.shape[0]
        print(f'Cells retained after scrublet: {n1}, {n0-n1} removed.')
        print(f'End of post doublets removal and QC plots.')

        # Post seurat or mads filtering QC plot

        # Filters
        print('Filters application (seurat or mads)')
        if mode == 'seurat':
            adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
            adata.obs['passing_nUMIs'] = adata.obs['nUMIs'] > tresh['nUMIs']
            adata.obs['passing_ngenes'] = adata.obs['detected_genes'] > tresh['detected_genes']
        elif mode == 'mads':
            adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
            adata.obs['passing_nUMIs'] = mads_test(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
            adata.obs['passing_ngenes'] = mads_test(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)  

        # Report 
        if mode == 'seurat':
            print(f'Lower treshold, nUMIs: {tresh["nUMIs"]}; filtered-out-cells: {n1-np.sum(adata.obs["passing_nUMIs"])}')
            print(f'Lower treshold, n genes: {tresh["detected_genes"]}; filtered-out-cells: {n1-np.sum(adata.obs["passing_ngenes"])}')
            print(f'Lower treshold, mito %: {tresh["mito_perc"]}; filtered-out-cells: {n1-np.sum(adata.obs["passing_mt"])}')
        elif mode == 'mads':
            nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
            n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
            print(f'Tresholds used, nUMIs: ({nUMIs_t[0]}, {nUMIs_t[1]}); filtered-out-cells: {n1-np.sum(adata.obs["passing_nUMIs"])}')
            print(f'Tresholds used, n genes: ({n_genes_t[0]}, {n_genes_t[1]}); filtered-out-cells: {n1-np.sum(adata.obs["passing_ngenes"])}')
            print(f'Lower treshold, mito %: {tresh["mito_perc"]}; filtered-out-cells: {n1-np.sum(adata.obs["passing_mt"])}')
        print(f'Filters applicated.')

        # QC plot
        QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
        removed = QC_test.loc[lambda x : x == False]
        removed_cells.extend(list(removed.index.values))
        print(f'Total cell filtered out with this last --mode {mode} QC (and its chosen options): {n1-np.sum(QC_test)}')
        adata = adata[QC_test, :]
        n2 = adata.shape[0]
            


        # Store cleaned adata
        print(f'Cells retained after scrublet and {mode} filtering: {n2}, {n0-n2} removed.')
        adatas[s] = adata
        print(adatas[s])


    # Concenate
    universe = sorted(
        list(reduce(lambda x,y: x&y, [ set(adatas[k].var_names) for k in adatas ]))
    )
    seed(1234)
    universe = sample(universe, len(universe))
    adata = anndata.concat([ adatas[k][:, universe] for k in adatas ], axis=0)

    # Last gene and cell filter
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    return adata, removed_cells


def qc(adata:anndata.AnnData, mode='seurat', 
       min_cells=3, min_genes=200, nmads=5, 
       max_cells_ratio=1,max_genes_ratio=1,
       batch_key=None,doublets=True,
       path_viz=None, tresh=None,mt_startswith='MT-'):
    """
    Perform quality control on a dictionary of AnnData objects.
    
    This function calculates several QC metrics, including mitochondrial percentage, nUMIs, 
    and detected genes, and produces several plots visualizing the QC metrics for each sample. 
    The function performs doublet detection using scrublet and filtering using either 
    Seurat or MADs. The function returns a merged AnnData object with cells that passed QC filters
    and a list of cells that did not pass QC on all samples.

    Arguments:
        adatas : AnnData objects
        mode : The filtering method to use. Valid options are 'seurat' and 'mads'. Default is 'seurat'.
        min_cells : The minimum number of cells for a sample to pass QC. Default is 3.
        min_genes : The minimum number of genes for a cell to pass QC. Default is 200.
        max_cells_ratio : The maximum number of cells ratio for a sample to pass QC. Default is 1.
        max_genes_ratio : The maximum number of genes ratio for a cell to pass QC. Default is 1.
        nmads : The number of MADs to use for MADs filtering. Default is 5.
        path_viz : The path to save the QC plots. Default is None.
        tresh : A dictionary of QC thresholds. The keys should be 'mito_perc', 'nUMIs', and 'detected_genes'.
            Only used if mode is 'seurat'. Default is None.
        mt_startswith : The prefix of mitochondrial genes. Default is 'MT-'.

    Returns:
        adata : An AnnData object containing cells that passed QC filters.

    """
    # Logging 
    if tresh is None:
        tresh={'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}
    
    # For each adata, produce a figure
    # with PdfPages(path_viz + 'original_QC_by_sample.pdf') as pdf: 
    removed_cells = []

    # QC metrics
    print('Calculate QC metrics')
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith(mt_startswith)
    if issparse(adata.X):
        adata.obs['nUMIs'] = adata.X.toarray().sum(axis=1)  
        adata.obs['mito_perc'] = adata[:, adata.var["mt"]].X.toarray().sum(axis=1) / adata.obs['nUMIs'].values
        adata.obs['detected_genes'] = (adata.X.toarray() > 0).sum(axis=1)  
    else:
        adata.obs['nUMIs'] = adata.X.sum(axis=1)  
        adata.obs['mito_perc'] = adata[:, adata.var["mt"]].X.sum(axis=1) / adata.obs['nUMIs'].values
        adata.obs['detected_genes'] = (adata.X > 0).sum(axis=1)  
    adata.obs['cell_complexity'] = adata.obs['detected_genes'] / adata.obs['nUMIs']
    print(f'End calculation of QC metrics.')

    # Original QC plot
    n0 = adata.shape[0]
    print(f'Original cell number: {n0}')

    if doublets==True:
        # Post doublets removal QC plot
        print('Begin of post doublets removal and QC plot')
        sc.external.pp.scrublet(adata, random_state=1234,batch_key=batch_key)
        adata_remove = adata[adata.obs['predicted_doublet'], :]
        removed_cells.extend(list(adata_remove.obs_names))
        adata = adata[~adata.obs['predicted_doublet'], :]
        n1 = adata.shape[0]
        print(f'Cells retained after scrublet: {n1}, {n0-n1} removed.')
        print(f'End of post doublets removal and QC plots.')

    # Post seurat or mads filtering QC plot

    # Filters
    print('Filters application (seurat or mads)')
    if mode == 'seurat':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = adata.obs['nUMIs'] > tresh['nUMIs']
        adata.obs['passing_ngenes'] = adata.obs['detected_genes'] > tresh['detected_genes']
    elif mode == 'mads':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = mads_test(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        adata.obs['passing_ngenes'] = mads_test(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)  

    # Report 
    if mode == 'seurat':
        print(f'Lower treshold, nUMIs: {tresh["nUMIs"]}; filtered-out-cells: {n1-np.sum(adata.obs["passing_nUMIs"])}')
        print(f'Lower treshold, n genes: {tresh["detected_genes"]}; filtered-out-cells: {n1-np.sum(adata.obs["passing_ngenes"])}')
        print(f'Lower treshold, mito %: {tresh["mito_perc"]}; filtered-out-cells: {n1-np.sum(adata.obs["passing_mt"])}')
    elif mode == 'mads':
        nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
        print(f'Tresholds used, nUMIs: ({nUMIs_t[0]}, {nUMIs_t[1]}); filtered-out-cells: {n1-np.sum(adata.obs["passing_nUMIs"])}')
        print(f'Tresholds used, n genes: ({n_genes_t[0]}, {n_genes_t[1]}); filtered-out-cells: {n1-np.sum(adata.obs["passing_ngenes"])}')
        print(f'Lower treshold, mito %: {tresh["mito_perc"]}; filtered-out-cells: {n1-np.sum(adata.obs["passing_mt"])}')
    print(f'Filters applicated.')

    # QC plot
    QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
    removed = QC_test.loc[lambda x : x == False]
    removed_cells.extend(list(removed.index.values))
    print(f'Total cell filtered out with this last --mode {mode} QC (and its chosen options): {n1-np.sum(QC_test)}')
    adata = adata[QC_test, :]
    n2 = adata.shape[0]
        


    # Store cleaned adata
    print(f'Cells retained after scrublet and {mode} filtering: {n2}, {n0-n2} removed.')

    # Last gene and cell filter
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, max_genes=max_genes_ratio*adata.shape[1])
    sc.pp.filter_genes(adata, max_cells=max_cells_ratio*adata.shape[0])

    return adata

def filter_cells(adata: anndata.AnnData,
    min_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_genes: Optional[int] = None,
    inplace: bool = True,):
    """
    Filter cell outliers based on counts and numbers of genes expressed.
    
    For instance, only keep cells with at least `min_counts` counts or
    `min_genes` genes expressed. This is to filter measurement outliers,
    i.e. “unreliable” observations.
    
    Only provide one of the optional parameters `min_counts`, `min_genes`,
    `max_counts`, `max_genes` per call.
    
    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_genes
        Minimum number of genes expressed required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_genes
        Maximum number of genes expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.
    
    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix:
    
    cells_subset
        Boolean index mask that does filtering. `True` means that the
        cell is kept. `False` means the cell is removed.
    number_per_cell
        Depending on what was tresholded (`counts` or `genes`),
        the array stores `n_counts` or `n_cells` per gene.
    
    """

    sc.pp.filter_cells(adata, min_genes=min_genes,min_counts=min_counts, max_counts=max_counts,
                       max_genes=max_genes, inplace=inplace)
    
def filter_genes(adata: anndata.AnnData,
    min_counts: Optional[int] = None,
    min_cells: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_cells: Optional[int] = None,
    inplace: bool = True,):
    """
    Filter genes based on number of cells or counts.
    
    Keep genes that have at least `min_counts` counts or are expressed in at
    least `min_cells` cells or have at most `max_counts` counts or are expressed
    in at most `max_cells` cells.
    
    Only provide one of the optional parameters `min_counts`, `min_cells`,
    `max_counts`, `max_cells` per call.
    
    Parameters
    ----------
    data
        An annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    min_counts
        Minimum number of counts required for a gene to pass filtering.
    min_cells
        Minimum number of cells expressed required for a gene to pass filtering.
    max_counts
        Maximum number of counts required for a gene to pass filtering.
    max_cells
        Maximum number of cells expressed required for a gene to pass filtering.
    inplace
        Perform computation inplace or return result.
    
    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix
    
    gene_subset
        Boolean index mask that does filtering. `True` means that the
        gene is kept. `False` means the gene is removed.
    number_per_gene
        Depending on what was tresholded (`counts` or `cells`), the array stores
        `n_counts` or `n_cells` per gene.
    
    """

    sc.pp.filter_genes(adata, min_counts=min_counts, min_cells=min_cells, max_counts=max_counts,
                          max_cells=max_cells, inplace=inplace)
    