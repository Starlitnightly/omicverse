"""
Copy from pegasus and cellual

"""


import os
import logging
from functools import reduce
from random import seed, sample
from typing import Union, Optional, Sequence, Tuple, List, Dict
import pandas as pd 
import anndata
import numpy as np 
import scanpy as sc
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib import rcParams
import seaborn as sns
from scipy.sparse import issparse

from .._settings import settings,print_gpu_usage_color,EMOJI,add_reference



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

def quantity_control(adatas, mode='seurat', min_cells=3, min_genes=200,\
    nmads=5, path_viz=None, tresh=None):
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
        A dictionary of QC thresholds. The keys should be 'mito_perc', 'nUMIs', 
        and 'detected_genes'.
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
            adata.obs['mito_perc'] = adata[:, adata.var["mt"]].X.toarray().sum(axis=1) / \
            adata.obs['nUMIs'].values
            adata.obs['detected_genes'] = (adata.X.toarray() > 0).sum(axis=1)
        else:
            adata.obs['nUMIs'] = adata.X.sum(axis=1)
            adata.obs['mito_perc'] = adata[:, adata.var["mt"]].X.sum(axis=1) / \
            adata.obs['nUMIs'].values
            adata.obs['detected_genes'] = (adata.X > 0).sum(axis=1)
        adata.obs['cell_complexity'] = adata.obs['detected_genes'] / adata.obs['nUMIs']
        print('End calculation of QC metrics.')

        # Original QC plot
        n0 = adata.shape[0]
        print(f'Original cell number: {n0}')

        # Post doublets removal QC plot
        print('Begin of post doublets removal and QC plot')
        sc.pp.scrublet(adata, random_state=1234)
        adata_remove = adata[adata.obs['predicted_doublet'], :]
        removed_cells.extend(list(adata_remove.obs_names))
        adata = adata[~adata.obs['predicted_doublet'], :]
        n1 = adata.shape[0]
        print(f'Cells retained after scrublet: {n1}, {n0-n1} removed.')
        print('End of post doublets removal and QC plots.')

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
            adata.obs['passing_ngenes'] = mads_test(adata.obs, 'detected_genes', \
            nmads=nmads, lt=tresh)

        # Report
        if mode == 'seurat':
            print(f'Lower treshold, nUMIs: {tresh["nUMIs"]}; filtered-out-cells: \
            {n1-np.sum(adata.obs["passing_nUMIs"])}')
            print(f'Lower treshold, n genes: {tresh["detected_genes"]}; \
            filtered-out-cells: {n1-np.sum(adata.obs["passing_ngenes"])}')
            print(f'Lower treshold, mito %: {tresh["mito_perc"]}; \
            filtered-out-cells: {n1-np.sum(adata.obs["passing_mt"])}')
        elif mode == 'mads':
            nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
            n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
            print(f'Tresholds used, nUMIs: ({nUMIs_t[0]}, {nUMIs_t[1]}); \
            filtered-out-cells: {n1-np.sum(adata.obs["passing_nUMIs"])}')
            print(f'Tresholds used, n genes: ({n_genes_t[0]}, {n_genes_t[1]}); \
                filtered-out-cells: {n1-np.sum(adata.obs["passing_ngenes"])}')
            print(f'Lower treshold, mito %: {tresh["mito_perc"]}; \
            filtered-out-cells: {n1-np.sum(adata.obs["passing_mt"])}')
        print('Filters applicated.')

        # QC plot
        QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & \
        (adata.obs['passing_ngenes'])
        removed = QC_test.loc[lambda x : x is False]
        removed_cells.extend(list(removed.index.values))
        print(f'Total cell filtered out with this last --mode {mode} QC (and its \
        chosen options): {n1-np.sum(QC_test)}')
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


def qc(adata,**kwargs):
    '''
    qc
    '''


    if settings.mode == 'gpu':
        print(f"{EMOJI['gpu']} Using RAPIDS GPU to calculate QC...")
        return qc_gpu(adata,**kwargs)
    elif settings.mode == 'cpu-gpu-mixed':
        print(f"{EMOJI['mixed']} Using torch CPU/GPU mixed mode...")
        print_gpu_usage_color()
        return qc_cpu_gpu_mixed(adata,**kwargs)
    else:
        print(f"{EMOJI['cpu']} Using torch CPU mode...")
        return qc_cpu(adata,**kwargs)
    

def qc_cpu_gpu_mixed(adata:anndata.AnnData, mode='seurat',
       min_cells=3, min_genes=200, nmads=5,
       max_cells_ratio=1,max_genes_ratio=1,
       batch_key=None,doublets=True,doublets_method='scrublet',
       path_viz=None, tresh=None,mt_startswith='MT-',mt_genes=None,use_gpu=True):
    """
    Perform quality control on a dictionary of AnnData objects.
    
    This function calculates several QC metrics, including mitochondrial percentage, nUMIs, 
    and detected genes, and produces several plots visualizing the QC metrics for each sample. 
    The function performs doublet detection using scrublet and filtering using either 
    Seurat or MADs. The function returns a merged AnnData object with cells that passed QC filters
    and a list of cells that did not pass QC on all samples.

    Arguments:
        adatas : AnnData objects
        mode : The filtering method to use. Valid options are 'seurat' 
        and 'mads'. Default is 'seurat'.
        min_cells : The minimum number of cells for a sample to pass QC. Default is 3.
        min_genes : The minimum number of genes for a cell to pass QC. Default is 200.
        max_cells_ratio : The maximum number of cells ratio for a sample to pass QC. Default is 1.
        max_genes_ratio : The maximum number of genes ratio for a cell to pass QC. Default is 1.
        nmads : The number of MADs to use for MADs filtering. Default is 5.
        path_viz : The path to save the QC plots. Default is None.
        tresh : A dictionary of QC thresholds. The keys should be 'mito_perc', 
        'nUMIs', and 'detected_genes'.
            Only used if mode is 'seurat'. Default is None.
        mt_startswith : The prefix of mitochondrial genes. Default is 'MT-'.
        mt_genes : The list of mitochondrial genes. Default is None. 
        if mt_genes is not None, mt_startswith will be ignored.

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
    if mt_genes is not None:
        adata.var['mt']=False
        adata.var.loc[list(set(adata.var_names) & set(mt_genes)),'mt']=True
    else:
        adata.var["mt"] = adata.var_names.str.startswith(mt_startswith)
    if issparse(adata.X):
        adata.obs['nUMIs'] = np.array(adata.X.sum(axis=1)).reshape(-1)
        adata.obs['mito_perc'] = np.array(adata[:, adata.var["mt"]].X.sum(axis=1)).reshape(-1) / \
        adata.obs['nUMIs'].values
        adata.obs['detected_genes'] = adata.X.getnnz(axis=1)
    else:
        adata.obs['nUMIs'] = adata.X.sum(axis=1)
        adata.obs['mito_perc'] = adata[:, adata.var["mt"] is True].X.sum(axis=1) / \
        adata.obs['nUMIs'].values
        adata.obs['detected_genes'] = np.count_nonzero(adata.X, axis=1)
    adata.obs['cell_complexity'] = adata.obs['detected_genes'] / adata.obs['nUMIs']
    print('End calculation of QC metrics.')

    # Original QC plot
    n0 = adata.shape[0]
    print(f'Original cell number: {n0}')

    if doublets is True:
        if doublets_method=='scrublet':
            # Post doublets removal QC plot
            print('!!!It should be noted that the `scrublet` detection is too old and \
            may not work properly.!!!')
            print('!!!if you want to use novel doublet detection, \
            please set `doublets_method=sccomposite`!!!')
            print('Begin of post doublets removal and QC plot using`scrublet`')
            from ._scrublet import scrublet
            scrublet(adata, random_state=1234,batch_key=batch_key,use_gpu=use_gpu)

            adata_remove = adata[adata.obs['predicted_doublet'], :]
            removed_cells.extend(list(adata_remove.obs_names))
            adata = adata[~adata.obs['predicted_doublet'], :]
            n1 = adata.shape[0]
            print(f'Cells retained after scrublet: {n1}, {n0-n1} removed.')
            print('End of post doublets removal and QC plots.')
        elif doublets_method=='sccomposite':
            print('!!!It should be noted that the `sccomposite` will remove more cells than \
            `scrublet`!!!')
            print('Begin of post doublets removal and QC plot using `sccomposite`')
            adata.obs['sccomposite_doublet']=0
            adata.obs['sccomposite_consistency']=0
            if batch_key is None:
                from ._sccomposite import composite_rna
                multiplet_classification, consistency = composite_rna(adata)
                adata.obs['sccomposite_doublet']=multiplet_classification
                adata.obs['sccomposite_consistency']=consistency
            else:
                for batch in adata.obs[batch_key].unique():
                    from ._sccomposite import composite_rna
                    adata_batch=adata[adata.obs[batch_key]==batch]
                    multiplet_classification, consistency = composite_rna(adata_batch)
                    adata.obs.loc[adata_batch.obs.index,'sccomposite_doublet']=\
                    multiplet_classification
                    adata.obs.loc[adata_batch.obs.index,'sccomposite_consistency']=consistency
            adata_remove = adata[adata.obs['sccomposite_doublet']!=0, :]
            removed_cells.extend(list(adata_remove.obs_names))
            adata = adata[adata.obs['sccomposite_doublet']==0, :]
            n1 = adata.shape[0]
            print(f'Cells retained after sccomposite: {n1}, {n0-n1} removed.')
            print('End of post sccomposite removal and QC plots.')
    else:
        n1 = adata.shape[0]
    # Fix bug where n1 is not defined when doublets=False
    if not doublets:
        n1 = n0


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
        print(f'Lower treshold, nUMIs: {tresh["nUMIs"]}; filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_nUMIs"])}')
        print(f'Lower treshold, n genes: {tresh["detected_genes"]}; filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_ngenes"])}')
        print(f'Lower treshold, mito %: {tresh["mito_perc"]}; filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_mt"])}')
    elif mode == 'mads':
        nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
        print(f'Tresholds used, nUMIs: ({nUMIs_t[0]}, {nUMIs_t[1]}); filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_nUMIs"])}')
        print(f'Tresholds used, n genes: ({n_genes_t[0]}, {n_genes_t[1]}); filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_ngenes"])}')
        print(f'Lower treshold, mito %: {tresh["mito_perc"]}; filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_mt"])}')
    print('Filters applicated.')

    # QC plot
    QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
    removed = QC_test.loc[lambda x : x == False]
    removed_cells.extend(list(removed.index.values))
    print(f'Total cell filtered out with this last --mode {mode} QC (and its \
    chosen options): {n1-np.sum(QC_test)}')
    adata = adata[QC_test, :]
    n2 = adata.shape[0]
    # Store cleaned adata
    print(f'Cells retained after scrublet and {mode} filtering: {n2}, {n0-n2} removed.')

    # Last gene and cell filter
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, max_genes=max_genes_ratio*adata.shape[1])
    sc.pp.filter_genes(adata, max_cells=max_cells_ratio*adata.shape[0])

    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}

    adata.uns['status']['qc']=True
    adata.uns['status_args']['qc'] = {
        'mode': mode,
        'min_cells': min_cells,
        'min_genes': min_genes,
        'nmads': nmads,
        'max_cells_ratio': max_cells_ratio,
        'max_genes_ratio': max_genes_ratio,
        'batch_key': batch_key,
        'doublets': doublets,
        'doublets_method': doublets_method,
        'path_viz': path_viz,
        'mito_perc': tresh['mito_perc'],
        'nUMIs': tresh['nUMIs'],
        'detected_genes': tresh['detected_genes'],
    }

    return adata


def qc_cpu(adata:anndata.AnnData, mode='seurat',
       min_cells=3, min_genes=200, nmads=5,
       max_cells_ratio=1,max_genes_ratio=1,
       batch_key=None,doublets=True,doublets_method='scrublet',
       path_viz=None, tresh=None,mt_startswith='MT-',mt_genes=None):
    """
    Perform quality control on a dictionary of AnnData objects.
    
    This function calculates several QC metrics, including mitochondrial percentage, nUMIs, 
    and detected genes, and produces several plots visualizing the QC metrics for each sample. 
    The function performs doublet detection using scrublet and filtering using either 
    Seurat or MADs. The function returns a merged AnnData object with cells that passed QC filters
    and a list of cells that did not pass QC on all samples.

    Arguments:
        adatas : AnnData objects
        mode : The filtering method to use. Valid options are 'seurat' 
        and 'mads'. Default is 'seurat'.
        min_cells : The minimum number of cells for a sample to pass QC. Default is 3.
        min_genes : The minimum number of genes for a cell to pass QC. Default is 200.
        max_cells_ratio : The maximum number of cells ratio for a sample to pass QC. Default is 1.
        max_genes_ratio : The maximum number of genes ratio for a cell to pass QC. Default is 1.
        nmads : The number of MADs to use for MADs filtering. Default is 5.
        path_viz : The path to save the QC plots. Default is None.
        tresh : A dictionary of QC thresholds. The keys should be 'mito_perc', 
        'nUMIs', and 'detected_genes'.
            Only used if mode is 'seurat'. Default is None.
        mt_startswith : The prefix of mitochondrial genes. Default is 'MT-'.
        mt_genes : The list of mitochondrial genes. Default is None. 
        if mt_genes is not None, mt_startswith will be ignored.

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
    if mt_genes is not None:
        adata.var['mt']=False
        adata.var.loc[list(set(adata.var_names) & set(mt_genes)),'mt']=True
    else:
        adata.var["mt"] = adata.var_names.str.startswith(mt_startswith)
    if issparse(adata.X):
        adata.obs['nUMIs'] = np.array(adata.X.sum(axis=1)).reshape(-1)
        adata.obs['mito_perc'] = np.array(adata[:, adata.var["mt"]].X.sum(axis=1)).reshape(-1) / \
        adata.obs['nUMIs'].values
        adata.obs['detected_genes'] = adata.X.getnnz(axis=1)
    else:
        adata.obs['nUMIs'] = adata.X.sum(axis=1)
        adata.obs['mito_perc'] = adata[:, adata.var["mt"] is True].X.sum(axis=1) / \
        adata.obs['nUMIs'].values
        adata.obs['detected_genes'] = np.count_nonzero(adata.X, axis=1)
    adata.obs['cell_complexity'] = adata.obs['detected_genes'] / adata.obs['nUMIs']
    print('End calculation of QC metrics.')

    # Original QC plot
    n0 = adata.shape[0]
    print(f'Original cell number: {n0}')

    if doublets is True:
        if doublets_method=='scrublet':
            # Post doublets removal QC plot
            print('!!!It should be noted that the `scrublet` detection is too old and \
            may not work properly.!!!')
            print('!!!if you want to use novel doublet detection, \
            please set `doublets_method=sccomposite`!!!')
            print('Begin of post doublets removal and QC plot using`scrublet`')
            sc.pp.scrublet(adata, random_state=1234,batch_key=batch_key)

            adata_remove = adata[adata.obs['predicted_doublet'], :]
            removed_cells.extend(list(adata_remove.obs_names))
            adata = adata[~adata.obs['predicted_doublet'], :]
            n1 = adata.shape[0]
            print(f'Cells retained after scrublet: {n1}, {n0-n1} removed.')
            print('End of post doublets removal and QC plots.')
        elif doublets_method=='sccomposite':
            print('!!!It should be noted that the `sccomposite` will remove more cells than \
            `scrublet`!!!')
            print('Begin of post doublets removal and QC plot using `sccomposite`')
            adata.obs['sccomposite_doublet']=0
            adata.obs['sccomposite_consistency']=0
            if batch_key is None:
                from ._sccomposite import composite_rna
                multiplet_classification, consistency = composite_rna(adata)
                adata.obs['sccomposite_doublet']=multiplet_classification
                adata.obs['sccomposite_consistency']=consistency
            else:
                for batch in adata.obs[batch_key].unique():
                    from ._sccomposite import composite_rna
                    adata_batch=adata[adata.obs[batch_key]==batch]
                    multiplet_classification, consistency = composite_rna(adata_batch)
                    adata.obs.loc[adata_batch.obs.index,'sccomposite_doublet']=\
                    multiplet_classification
                    adata.obs.loc[adata_batch.obs.index,'sccomposite_consistency']=consistency
            adata_remove = adata[adata.obs['sccomposite_doublet']!=0, :]
            removed_cells.extend(list(adata_remove.obs_names))
            adata = adata[adata.obs['sccomposite_doublet']==0, :]
            n1 = adata.shape[0]
            print(f'Cells retained after sccomposite: {n1}, {n0-n1} removed.')
            print('End of post sccomposite removal and QC plots.')
    else:
        n1 = adata.shape[0]
    # Fix bug where n1 is not defined when doublets=False
    if not doublets:
        n1 = n0


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
        print(f'Lower treshold, nUMIs: {tresh["nUMIs"]}; filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_nUMIs"])}')
        print(f'Lower treshold, n genes: {tresh["detected_genes"]}; filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_ngenes"])}')
        print(f'Lower treshold, mito %: {tresh["mito_perc"]}; filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_mt"])}')
    elif mode == 'mads':
        nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
        print(f'Tresholds used, nUMIs: ({nUMIs_t[0]}, {nUMIs_t[1]}); filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_nUMIs"])}')
        print(f'Tresholds used, n genes: ({n_genes_t[0]}, {n_genes_t[1]}); filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_ngenes"])}')
        print(f'Lower treshold, mito %: {tresh["mito_perc"]}; filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_mt"])}')
    print('Filters applicated.')

    # QC plot
    QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
    removed = QC_test.loc[lambda x : x == False]
    removed_cells.extend(list(removed.index.values))
    print(f'Total cell filtered out with this last --mode {mode} QC (and its \
    chosen options): {n1-np.sum(QC_test)}')
    adata = adata[QC_test, :]
    n2 = adata.shape[0]
    # Store cleaned adata
    print(f'Cells retained after scrublet and {mode} filtering: {n2}, {n0-n2} removed.')

    # Last gene and cell filter
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, max_genes=max_genes_ratio*adata.shape[1])
    sc.pp.filter_genes(adata, max_cells=max_cells_ratio*adata.shape[0])

    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}

    adata.uns['status']['qc']=True
    adata.uns['status_args']['qc'] = {
        'mode': mode,
        'min_cells': min_cells,
        'min_genes': min_genes,
        'nmads': nmads,
        'max_cells_ratio': max_cells_ratio,
        'max_genes_ratio': max_genes_ratio,
        'batch_key': batch_key,
        'doublets': doublets,
        'doublets_method': doublets_method,
        'path_viz': path_viz,
        'mito_perc': tresh['mito_perc'],
        'nUMIs': tresh['nUMIs'],
        'detected_genes': tresh['detected_genes'],
    }
    add_reference(adata,'qc','QC with scanpy')
    return adata


def qc_gpu(adata, mode='seurat',
       min_cells=3, min_genes=200, nmads=5,
       max_cells_ratio=1,max_genes_ratio=1,
       batch_key=None,doublets=True,doublets_method='scrublet',
       path_viz=None, tresh=None,mt_startswith='MT-',mt_genes=None):
    '''
    qc
    
    '''
    import rapids_singlecell as rsc
     # Logging
    if tresh is None:
        tresh={'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}
    # For each adata, produce a figure
    # with PdfPages(path_viz + 'original_QC_by_sample.pdf') as pdf:
    removed_cells = []
    rsc.get.anndata_to_GPU(adata)
    # QC metrics
    print('Calculate QC metrics')
    adata.var_names_make_unique()
    if mt_genes is not None:
        adata.var['mt']=False
        adata.var.loc[list(set(adata.var_names) & set(mt_genes)),'mt']=True
    else:
        rsc.pp.flag_gene_family(adata, gene_family_name="mt", gene_family_prefix=mt_startswith)
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"])
    adata.obs['nUMIs'] = adata.obs['total_counts']
    adata.obs['mito_perc'] = adata.obs['pct_counts_mt']/100
    adata.obs['detected_genes'] = adata.obs['n_genes_by_counts']
    adata.obs['cell_complexity'] = adata.obs['detected_genes'] / adata.obs['nUMIs']
    print('End calculation of QC metrics.')

    # Original QC plot
    n0 = adata.shape[0]
    print(f'Original cell number: {n0}')

    if doublets is True:
        if doublets_method=='scrublet':
            # Post doublets removal QC plot
            print('Begin of post doublets removal and QC plot')
            rsc.pp.scrublet(adata, random_state=1234,batch_key=batch_key)
            adata_remove = adata[adata.obs['predicted_doublet'], :]
            removed_cells.extend(list(adata_remove.obs_names))
            adata = adata[~adata.obs['predicted_doublet'], :]
            n1 = adata.shape[0]
            print(f'Cells retained after scrublet: {n1}, {n0-n1} removed.')
            print('End of post doublets removal and QC plots.')

        elif doublets_method=='sccomposite':
            adata.obs['sccomposite_doublet']=0
            adata.obs['sccomposite_consistency']=0
            if batch_key is None:
                from ._sccomposite import composite_rna
                multiplet_classification, consistency = composite_rna(adata)
                adata.obs['sccomposite_doublet']=multiplet_classification
                adata.obs['sccomposite_consistency']=consistency
            else:
                for batch in adata.obs[batch_key].unique():
                    from ._sccomposite import composite_rna
                    adata_batch=adata[adata.obs[batch_key]==batch]
                    multiplet_classification, consistency = composite_rna(adata_batch)
                    adata.obs.loc[adata_batch.obs.index,'sccomposite_doublet']=\
                    multiplet_classification
                    adata.obs.loc[adata_batch.obs.index,'sccomposite_consistency']=consistency
            adata_remove = adata[adata.obs['sccomposite_doublet']!=0, :]
            removed_cells.extend(list(adata_remove.obs_names))
            adata = adata[adata.obs['sccomposite_doublet']==0, :]
            n1 = adata.shape[0]
            print(f'Cells retained after sccomposite: {n1}, {n0-n1} removed.')
            print('End of post sccomposite removal and QC plots.')

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
        print(f'Lower treshold, nUMIs: {tresh["nUMIs"]}; filtered-out-cells: \
            {n1-np.sum(adata.obs["passing_nUMIs"])}')
        print(f'Lower treshold, n genes: {tresh["detected_genes"]}; \
            filtered-out-cells: {n1-np.sum(adata.obs["passing_ngenes"])}')
        print(f'Lower treshold, mito %: {tresh["mito_perc"]}; \
            filtered-out-cells: {n1-np.sum(adata.obs["passing_mt"])}')
    elif mode == 'mads':
        nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
        print(f'Tresholds used, nUMIs: ({nUMIs_t[0]}, {nUMIs_t[1]}); filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_nUMIs"])}')
        print(f'Tresholds used, n genes: ({n_genes_t[0]}, {n_genes_t[1]}); filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_ngenes"])}')
        print(f'Lower treshold, mito %: {tresh["mito_perc"]}; filtered-out-cells: \
        {n1-np.sum(adata.obs["passing_mt"])}')
    print(f'Filters applicated.')

    # QC plot
    QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
    removed = QC_test.loc[lambda x : x is False]
    removed_cells.extend(list(removed.index.values))
    print(f'Total cell filtered out with this last --mode {mode} QC (and its chosen options): \
    {n1-np.sum(QC_test)}')
    adata = adata[QC_test, :]
    n2 = adata.shape[0]
    # Store cleaned adata
    print(f'Cells retained after scrublet and {mode} filtering: {n2}, {n0-n2} removed.')

    # Last gene and cell filter
    rsc.pp.filter_cells(adata,qc_var='detected_genes', min_count=min_genes, \
        max_count=max_genes_ratio*adata.shape[1])
    rsc.pp.filter_genes(adata, min_count=min_cells, max_count=max_cells_ratio*adata.shape[0])

    if adata.uns['status'] is None:
        adata.uns['status'] = {}
    adata.uns['status']['qc']=True
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
    