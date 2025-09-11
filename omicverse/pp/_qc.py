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

from .._settings import settings,print_gpu_usage_color,EMOJI,add_reference,Colors
from ..utils.registry import register_function





def mads(meta, cov, nmads=5, lt=None, batch_key=None):
    """
    Calculate Median Absolute Deviation (MAD) thresholds.
    
    If batch_key is provided, calculates MAD per batch and returns
    a dictionary mapping batch values to (lower, upper) thresholds.
    If batch_key is None, calculates global MAD as before.
    
    Parameters
    ----------
    meta : pd.DataFrame
        Metadata dataframe containing the column to analyze
    cov : str
        Column name to calculate MAD for
    nmads : float, default 5
        Number of MADs for threshold calculation
    lt : dict, optional
        Lower thresholds to use if calculated threshold <= 0
    batch_key : str, optional
        Column name for batch information. If provided, calculates per-batch MAD
        
    Returns
    -------
    If batch_key is None:
        tuple: (lower_threshold, upper_threshold)
    If batch_key is provided:
        dict: {batch_value: (lower_threshold, upper_threshold)}
    """
    if batch_key is None:
        # Original global MAD calculation
        x = meta[cov]
        mad = np.median(np.absolute(x - np.median(x)))
        t1 = np.median(x) - (nmads * mad)
        t1 = t1 if t1 > 0 else lt[cov]
        t2 = np.median(x) + (nmads * mad)
        return t1, t2
    else:
        # Batch-wise MAD calculation
        batch_thresholds = {}
        for batch in meta[batch_key].unique():
            batch_mask = meta[batch_key] == batch
            x_batch = meta.loc[batch_mask, cov]
            
            if len(x_batch) == 0:
                continue
                
            mad = np.median(np.absolute(x_batch - np.median(x_batch)))
            t1 = np.median(x_batch) - (nmads * mad)
            t1 = t1 if t1 > 0 else lt[cov]
            t2 = np.median(x_batch) + (nmads * mad)
            batch_thresholds[batch] = (t1, t2)
            
        return batch_thresholds

def mads_test(meta, cov, nmads=5, lt=None, batch_key=None):
    """
    Apply MAD-based filtering with optional batch awareness.
    
    Returns a boolean array with True values for entries within MAD thresholds.
    If batch_key is provided, applies per-batch MAD thresholds.
    
    Parameters
    ----------
    meta : pd.DataFrame
        Metadata dataframe containing the column to analyze
    cov : str
        Column name to test
    nmads : float, default 5
        Number of MADs for threshold calculation
    lt : dict, optional
        Lower thresholds to use if calculated threshold <= 0
    batch_key : str, optional
        Column name for batch information. If provided, applies per-batch MAD
        
    Returns
    -------
    pd.Series
        Boolean series indicating which cells pass the MAD filter
    """
    if batch_key is None:
        # Original global MAD test
        thresholds = mads(meta, cov, nmads=nmads, lt=lt)
        return (meta[cov] > thresholds[0]) & (meta[cov] < thresholds[1])
    else:
        # Batch-wise MAD test
        batch_thresholds = mads(meta, cov, nmads=nmads, lt=lt, batch_key=batch_key)
        result = pd.Series(False, index=meta.index, dtype=bool)
        
        for batch, (t1, t2) in batch_thresholds.items():
            batch_mask = meta[batch_key] == batch
            batch_test = (meta.loc[batch_mask, cov] > t1) & (meta.loc[batch_mask, cov] < t2)
            result.loc[batch_mask] = batch_test
            
        return result

@register_function(
    aliases=["质控", "qc", "quality_control", "质量控制"],
    category="preprocessing",
    description="Perform comprehensive quality control on single-cell data. For seurat mode, use tresh dict with keys: 'mito_perc', 'nUMIs', 'detected_genes'",
    examples=[
        "ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250})",
        "ov.pp.qc(adata, mode='mads', nmads=5, doublets=True)"
    ],
    related=["preprocess", "filter_cells", "filter_genes", "scrublet"]
)
def qc(adata,**kwargs):
    '''
    qc
    '''

    if settings.mode == 'gpu':
        print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['gpu']} Using RAPIDS GPU to calculate QC...{Colors.ENDC}")
        return qc_gpu(adata,**kwargs)
    elif settings.mode == 'cpu-gpu-mixed':
        print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['mixed']} Using CPU/GPU mixed mode for QC...{Colors.ENDC}")
        print_gpu_usage_color()
        return qc_cpu_gpu_mixed(adata,**kwargs)
    else:
        print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['cpu']} Using CPU mode for QC...{Colors.ENDC}")
        return qc_cpu(adata,**kwargs)
    

def qc_cpu_gpu_mixed(adata:anndata.AnnData, mode='seurat',
       min_cells=3, min_genes=200, nmads=5,
       max_cells_ratio=1,max_genes_ratio=1,
       batch_key=None,doublets=True,doublets_method='scrublet',
       path_viz=None, tresh=None,mt_startswith='MT-',mt_genes=None,use_gpu=True,
       batch_wise_mad=None):
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
    
    removed_cells = []

    print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Quality Control Analysis (CPU-GPU Mixed):{Colors.ENDC}")
    print(f"   {Colors.CYAN}Dataset shape: {Colors.BOLD}{adata.shape[0]:,} cells × {adata.shape[1]:,} genes{Colors.ENDC}")
    print(f"   {Colors.BLUE}QC mode: {Colors.BOLD}{mode}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Doublet detection: {Colors.BOLD}{doublets_method if doublets else 'disabled'}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Mitochondrial genes: {Colors.BOLD}{mt_startswith if mt_genes is None else 'custom list'}{Colors.ENDC}")

    # QC metrics
    print(f"\n{Colors.HEADER}{Colors.BOLD}📊 Step 1: Calculating QC Metrics{Colors.ENDC}")
    adata.var_names_make_unique()
    if mt_genes is not None:
        adata.var['mt']=False
        adata.var.loc[list(set(adata.var_names) & set(mt_genes)),'mt']=True
        mt_genes_found = sum(adata.var['mt'])
        print(f"   {Colors.CYAN}Custom mitochondrial genes: {Colors.BOLD}{mt_genes_found}/{len(mt_genes)}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    else:
        adata.var["mt"] = adata.var_names.str.startswith(mt_startswith)
        mt_genes_found = sum(adata.var["mt"])
        print(f"   {Colors.CYAN}Mitochondrial genes (prefix '{mt_startswith}'): {Colors.BOLD}{mt_genes_found}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    
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
    
    # Display QC statistics
    print(f"   {Colors.GREEN}✓ QC metrics calculated:{Colors.ENDC}")
    print(f"     {Colors.BLUE}• Mean nUMIs: {Colors.BOLD}{adata.obs['nUMIs'].mean():.0f}{Colors.ENDC}{Colors.BLUE} (range: {adata.obs['nUMIs'].min():.0f}-{adata.obs['nUMIs'].max():.0f}){Colors.ENDC}")
    print(f"     {Colors.BLUE}• Mean genes: {Colors.BOLD}{adata.obs['detected_genes'].mean():.0f}{Colors.ENDC}{Colors.BLUE} (range: {adata.obs['detected_genes'].min():.0f}-{adata.obs['detected_genes'].max():.0f}){Colors.ENDC}")
    print(f"     {Colors.BLUE}• Mean mitochondrial %: {Colors.BOLD}{adata.obs['mito_perc'].mean()*100:.1f}%{Colors.ENDC}{Colors.BLUE} (max: {adata.obs['mito_perc'].max()*100:.1f}%){Colors.ENDC}")

    # Original QC plot
    n0 = adata.shape[0]
    
    # Post seurat or mads filtering QC plot

    # Filters
    print(f"\n{Colors.HEADER}{Colors.BOLD}🔧 Step 2: Quality Filtering ({mode.upper()}){Colors.ENDC}")
    print(f"   {Colors.CYAN}Thresholds: mito≤{tresh['mito_perc']}, nUMIs≥{tresh['nUMIs']}, genes≥{tresh['detected_genes']}{Colors.ENDC}")
    
    if mode == 'seurat':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = adata.obs['nUMIs'] > tresh['nUMIs']
        adata.obs['passing_ngenes'] = adata.obs['detected_genes'] > tresh['detected_genes']
    elif mode == 'mads':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = mads_test(adata.obs, 'nUMIs', nmads=nmads, lt=tresh, batch_key=batch_key)
        adata.obs['passing_ngenes'] = mads_test(adata.obs, 'detected_genes', nmads=nmads, lt=tresh, batch_key=batch_key)

    # Report
    n1 = adata.shape[0]
    mt_failed = n1-np.sum(adata.obs["passing_mt"])
    umis_failed = n1-np.sum(adata.obs["passing_nUMIs"])
    genes_failed = n1-np.sum(adata.obs["passing_ngenes"])
    
    if mode == 'seurat':
        print(f"   {Colors.BLUE}📊 Seurat Filter Results:{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs filter (≥{tresh['nUMIs']}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({umis_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes filter (≥{tresh['detected_genes']}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({genes_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({mt_failed/n1*100:.1f}%){Colors.ENDC}")
    elif mode == 'mads':
        nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
        print(f"   {Colors.BLUE}📊 MADs Filter Results (±{nmads} MADs):{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs range ({nUMIs_t[0]:.0f}, {nUMIs_t[1]:.0f}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes range ({n_genes_t[0]:.0f}, {n_genes_t[1]:.0f}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")

    # QC plot
    QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
    removed = QC_test.loc[lambda x : x == False]
    removed_cells.extend(list(removed.index.values))
    total_qc_failed = n1-np.sum(QC_test)
    adata = adata[QC_test, :]
    n2 = adata.shape[0]
    
    print(f"   {Colors.GREEN}✓ Combined QC filters: {Colors.BOLD}{total_qc_failed:,}{Colors.ENDC}{Colors.GREEN} cells removed ({total_qc_failed/n1*100:.1f}%){Colors.ENDC}")

    # Last gene and cell filter
    print(f"\n{Colors.HEADER}{Colors.BOLD}🎯 Step 3: Final Filtering{Colors.ENDC}")
    print(f"   {Colors.CYAN}Parameters: min_genes={min_genes}, min_cells={min_cells}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Ratios: max_genes_ratio={max_genes_ratio}, max_cells_ratio={max_cells_ratio}{Colors.ENDC}")
    
    cells_before_final = adata.shape[0]
    genes_before_final = adata.shape[1]
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, max_genes=max_genes_ratio*adata.shape[1])
    sc.pp.filter_genes(adata, max_cells=max_cells_ratio*adata.shape[0])
    
    cells_final_filtered = cells_before_final - adata.shape[0]
    genes_final_filtered = genes_before_final - adata.shape[1]
    
    print(f"   {Colors.GREEN}✓ Final filtering: {Colors.BOLD}{cells_final_filtered:,}{Colors.ENDC}{Colors.GREEN} cells, {Colors.BOLD}{genes_final_filtered:,}{Colors.ENDC}{Colors.GREEN} genes removed{Colors.ENDC}")
    
    n_after_final_filt = adata.shape[0]
    
    if doublets is True:
        print(f"\n{Colors.HEADER}{Colors.BOLD}🔍 Step 4: Doublet Detection{Colors.ENDC}")
        if doublets_method=='scrublet':
            # Post doublets removal QC plot
            print(f"   {Colors.WARNING}⚠️  Note: 'scrublet' detection is legacy and may not work optimally{Colors.ENDC}")
            print(f"   {Colors.CYAN}💡 Consider using 'doublets_method=sccomposite' for better results{Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running scrublet doublet detection...{Colors.ENDC}")
            from ._scrublet import scrublet
            scrublet(adata, random_state=1234,batch_key=batch_key,use_gpu=use_gpu)

            adata_remove = adata[adata.obs['predicted_doublet'], :]
            removed_cells.extend(list(adata_remove.obs_names))
            adata = adata[~adata.obs['predicted_doublet'], :]
            n1 = adata.shape[0]
            doublets_removed = n_after_final_filt-n1
            print(f"   {Colors.GREEN}✓ Scrublet completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            
        elif doublets_method=='sccomposite':
            print(f"   {Colors.WARNING}⚠️  Note: 'sccomposite' typically removes more cells than 'scrublet'{Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running sccomposite doublet detection...{Colors.ENDC}")
            adata.obs['sccomposite_doublet']=0
            adata.obs['sccomposite_consistency']=0
            if batch_key is None:
                from ._sccomposite import composite_rna
                multiplet_classification, consistency = composite_rna(adata)
                adata.obs['sccomposite_doublet']=multiplet_classification
                adata.obs['sccomposite_consistency']=consistency
            else:
                print(f"   {Colors.CYAN}Processing {len(adata.obs[batch_key].unique())} batches separately...{Colors.ENDC}")
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
            doublets_removed = n_after_final_filt-n1
            print(f"   {Colors.GREEN}✓ sccomposite completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
    else:
        print(f"\n{Colors.BLUE}📊 Step 4: Doublet detection disabled{Colors.ENDC}")
        n1 = adata.shape[0]

    # Store status
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
    print(f"\n{Colors.HEADER}{Colors.BOLD}📊 Step 1: Calculating QC Metrics{Colors.ENDC}")
    adata.var_names_make_unique()
    if mt_genes is not None:
        adata.var['mt']=False
        adata.var.loc[list(set(adata.var_names) & set(mt_genes)),'mt']=True
        mt_genes_found = sum(adata.var['mt'])
        print(f"   {Colors.CYAN}Custom mitochondrial genes: {Colors.BOLD}{mt_genes_found}/{len(mt_genes)}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    else:
        adata.var["mt"] = adata.var_names.str.startswith(mt_startswith)
        mt_genes_found = sum(adata.var["mt"])
        print(f"   {Colors.CYAN}Mitochondrial genes (prefix '{mt_startswith}'): {Colors.BOLD}{mt_genes_found}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    
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
    
    # Display QC statistics
    print(f"   {Colors.GREEN}✓ QC metrics calculated:{Colors.ENDC}")
    print(f"     {Colors.BLUE}• Mean nUMIs: {Colors.BOLD}{adata.obs['nUMIs'].mean():.0f}{Colors.ENDC}{Colors.BLUE} (range: {adata.obs['nUMIs'].min():.0f}-{adata.obs['nUMIs'].max():.0f}){Colors.ENDC}")
    print(f"     {Colors.BLUE}• Mean genes: {Colors.BOLD}{adata.obs['detected_genes'].mean():.0f}{Colors.ENDC}{Colors.BLUE} (range: {adata.obs['detected_genes'].min():.0f}-{adata.obs['detected_genes'].max():.0f}){Colors.ENDC}")
    print(f"     {Colors.BLUE}• Mean mitochondrial %: {Colors.BOLD}{adata.obs['mito_perc'].mean()*100:.1f}%{Colors.ENDC}{Colors.BLUE} (max: {adata.obs['mito_perc'].max()*100:.1f}%){Colors.ENDC}")

    # Original QC plot
    n0 = adata.shape[0]
    print(f"   {Colors.CYAN}📈 Original cell count: {Colors.BOLD}{n0:,}{Colors.ENDC}")

    # Post seurat or mads filtering QC plot

    # Filters
    print(f"\n{Colors.HEADER}{Colors.BOLD}🔧 Step 2: Quality Filtering ({mode.upper()}){Colors.ENDC}")
    print(f"   {Colors.CYAN}Thresholds: mito≤{tresh['mito_perc']}, nUMIs≥{tresh['nUMIs']}, genes≥{tresh['detected_genes']}{Colors.ENDC}")
    
    if mode == 'seurat':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = adata.obs['nUMIs'] > tresh['nUMIs']
        adata.obs['passing_ngenes'] = adata.obs['detected_genes'] > tresh['detected_genes']
    elif mode == 'mads':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = mads_test(adata.obs, 'nUMIs', nmads=nmads, lt=tresh, batch_key=batch_key)
        adata.obs['passing_ngenes'] = mads_test(adata.obs, 'detected_genes', nmads=nmads, lt=tresh, batch_key=batch_key)

    # Report
    n1 = adata.shape[0]
    mt_failed = n1-np.sum(adata.obs["passing_mt"])
    umis_failed = n1-np.sum(adata.obs["passing_nUMIs"])
    genes_failed = n1-np.sum(adata.obs["passing_ngenes"])
    
    if mode == 'seurat':
        print(f"   {Colors.BLUE}📊 Seurat Filter Results:{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs filter (≥{tresh['nUMIs']}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({umis_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes filter (≥{tresh['detected_genes']}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({genes_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({mt_failed/n1*100:.1f}%){Colors.ENDC}")
    elif mode == 'mads':
        nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
        print(f"   {Colors.BLUE}📊 MADs Filter Results (±{nmads} MADs):{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs range ({nUMIs_t[0]:.0f}, {nUMIs_t[1]:.0f}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes range ({n_genes_t[0]:.0f}, {n_genes_t[1]:.0f}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Filters applied successfully{Colors.ENDC}")

    # QC plot
    QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
    removed = QC_test.loc[lambda x : x == False]
    removed_cells.extend(list(removed.index.values))
    total_qc_failed = n1-np.sum(QC_test)
    adata = adata[QC_test, :]
    n2 = adata.shape[0]
    
    print(f"   {Colors.GREEN}✓ Combined QC filters: {Colors.BOLD}{total_qc_failed:,}{Colors.ENDC}{Colors.GREEN} cells removed ({total_qc_failed/n1*100:.1f}%){Colors.ENDC}")

    # Last gene and cell filter
    print(f"\n{Colors.HEADER}{Colors.BOLD}🎯 Step 3: Final Filtering{Colors.ENDC}")
    print(f"   {Colors.CYAN}Parameters: min_genes={min_genes}, min_cells={min_cells}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Ratios: max_genes_ratio={max_genes_ratio}, max_cells_ratio={max_cells_ratio}{Colors.ENDC}")
    
    cells_before_final = adata.shape[0]
    genes_before_final = adata.shape[1]
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, max_genes=max_genes_ratio*adata.shape[1])
    sc.pp.filter_genes(adata, max_cells=max_cells_ratio*adata.shape[0])
    
    cells_final_filtered = cells_before_final - adata.shape[0]
    genes_final_filtered = genes_before_final - adata.shape[1]
    
    print(f"   {Colors.GREEN}✓ Final filtering: {Colors.BOLD}{cells_final_filtered:,}{Colors.ENDC}{Colors.GREEN} cells, {Colors.BOLD}{genes_final_filtered:,}{Colors.ENDC}{Colors.GREEN} genes removed{Colors.ENDC}")
    
    n_after_final_filt = adata.shape[0]
    
    if doublets is True:
        print(f"\n{Colors.HEADER}{Colors.BOLD}🔍 Step 4: Doublet Detection{Colors.ENDC}")
        if doublets_method=='scrublet':
            from ._scrublet import scrublet
            # Post doublets removal QC plot
            print(f"   {Colors.WARNING}⚠️  Note: 'scrublet' detection is too old and may not work properly{Colors.ENDC}")
            print(f"   {Colors.CYAN}💡 Consider using 'doublets_method=sccomposite' for better results{Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running scrublet doublet detection...{Colors.ENDC}")
            scrublet(adata, random_state=1234,batch_key=batch_key)

            adata_remove = adata[adata.obs['predicted_doublet'], :]
            removed_cells.extend(list(adata_remove.obs_names))
            adata = adata[~adata.obs['predicted_doublet'], :]
            n1 = adata.shape[0]
            doublets_removed = n_after_final_filt-n1
            print(f"   {Colors.GREEN}✓ Scrublet completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            
        elif doublets_method=='sccomposite':
            print(f"   {Colors.WARNING}⚠️  Note: the `sccomposite` will remove more cells than `scrublet`{Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running sccomposite doublet detection...{Colors.ENDC}")
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
            doublets_removed = n_after_final_filt-n1
            print(f"   {Colors.GREEN}✓ sccomposite completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
    else:
        print(f"\n{Colors.BLUE}📊 Step 4: Doublet detection disabled{Colors.ENDC}")
        n1 = adata.shape[0]

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
    GPU-accelerated quality control using RAPIDS
    '''
    import rapids_singlecell as rsc
    
    # Logging
    if tresh is None:
        tresh={'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}
    
    removed_cells = []
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Quality Control Analysis (GPU-Accelerated):{Colors.ENDC}")
    print(f"   {Colors.CYAN}Dataset shape: {Colors.BOLD}{adata.shape[0]:,} cells × {adata.shape[1]:,} genes{Colors.ENDC}")
    print(f"   {Colors.BLUE}QC mode: {Colors.BOLD}{mode}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Doublet detection: {Colors.BOLD}{doublets_method if doublets else 'disabled'}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Mitochondrial genes: {Colors.BOLD}{mt_startswith if mt_genes is None else 'custom list'}{Colors.ENDC}")
    
    print(f"   {Colors.GREEN}{EMOJI['gpu']} Loading data to GPU...{Colors.ENDC}")
    rsc.get.anndata_to_GPU(adata)
    
    # QC metrics
    print(f"\n{Colors.HEADER}{Colors.BOLD}📊 Step 1: Calculating QC Metrics{Colors.ENDC}")
    adata.var_names_make_unique()
    if mt_genes is not None:
        adata.var['mt']=False
        adata.var.loc[list(set(adata.var_names) & set(mt_genes)),'mt']=True
        mt_genes_found = sum(adata.var['mt'])
        print(f"   {Colors.CYAN}Custom mitochondrial genes: {Colors.BOLD}{mt_genes_found}/{len(mt_genes)}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    else:
        rsc.pp.flag_gene_family(adata, gene_family_name="mt", gene_family_prefix=mt_startswith)
        mt_genes_found = sum(adata.var["mt"])
        print(f"   {Colors.CYAN}Mitochondrial genes (prefix '{mt_startswith}'): {Colors.BOLD}{mt_genes_found}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"])
    adata.obs['nUMIs'] = adata.obs['total_counts']
    adata.obs['mito_perc'] = adata.obs['pct_counts_mt']/100
    adata.obs['detected_genes'] = adata.obs['n_genes_by_counts']
    adata.obs['cell_complexity'] = adata.obs['detected_genes'] / adata.obs['nUMIs']
    
    # Display QC statistics
    print(f"   {Colors.GREEN}✓ QC metrics calculated:{Colors.ENDC}")
    print(f"     {Colors.BLUE}• Mean nUMIs: {Colors.BOLD}{adata.obs['nUMIs'].mean():.0f}{Colors.ENDC}{Colors.BLUE} (range: {adata.obs['nUMIs'].min():.0f}-{adata.obs['nUMIs'].max():.0f}){Colors.ENDC}")
    print(f"     {Colors.BLUE}• Mean genes: {Colors.BOLD}{adata.obs['detected_genes'].mean():.0f}{Colors.ENDC}{Colors.BLUE} (range: {adata.obs['detected_genes'].min():.0f}-{adata.obs['detected_genes'].max():.0f}){Colors.ENDC}")
    print(f"     {Colors.BLUE}• Mean mitochondrial %: {Colors.BOLD}{adata.obs['mito_perc'].mean()*100:.1f}%{Colors.ENDC}{Colors.BLUE} (max: {adata.obs['mito_perc'].max()*100:.1f}%){Colors.ENDC}")

    # Original QC plot
    n0 = adata.shape[0]
    print(f"   {Colors.CYAN}📈 Original cell count: {Colors.BOLD}{n0:,}{Colors.ENDC}")

    # Filters
    print(f"\n{Colors.HEADER}{Colors.BOLD}🔧 Step 2: Quality Filtering ({mode.upper()}){Colors.ENDC}")
    print(f"   {Colors.CYAN}Thresholds: mito≤{tresh['mito_perc']}, nUMIs≥{tresh['nUMIs']}, genes≥{tresh['detected_genes']}{Colors.ENDC}")
    
    if mode == 'seurat':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = adata.obs['nUMIs'] > tresh['nUMIs']
        adata.obs['passing_ngenes'] = adata.obs['detected_genes'] > tresh['detected_genes']
    elif mode == 'mads':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = mads_test(adata.obs, 'nUMIs', nmads=nmads, lt=tresh, batch_key=batch_key)
        adata.obs['passing_ngenes'] = mads_test(adata.obs, 'detected_genes', nmads=nmads, lt=tresh, batch_key=batch_key)
    
    # Report
    n1 = adata.shape[0]
    mt_failed = n1-np.sum(adata.obs["passing_mt"])
    umis_failed = n1-np.sum(adata.obs["passing_nUMIs"])
    genes_failed = n1-np.sum(adata.obs["passing_ngenes"])
    
    if mode == 'seurat':
        print(f"   {Colors.BLUE}📊 Seurat Filter Results:{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs filter (≥{tresh['nUMIs']}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({umis_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes filter (≥{tresh['detected_genes']}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({genes_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({mt_failed/n1*100:.1f}%){Colors.ENDC}")
    elif mode == 'mads':
        nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
        print(f"   {Colors.BLUE}📊 MADs Filter Results (±{nmads} MADs):{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs range ({nUMIs_t[0]:.0f}, {nUMIs_t[1]:.0f}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes range ({n_genes_t[0]:.0f}, {n_genes_t[1]:.0f}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")

    # QC plot
    QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
    removed = QC_test.loc[~QC_test.values]
    removed_cells.extend(list(removed.index.values))
    total_qc_failed = n1-np.sum(QC_test)
    adata = adata[QC_test, :]
    n2 = adata.shape[0]
    
    print(f"   {Colors.GREEN}✓ Combined QC filters: {Colors.BOLD}{total_qc_failed:,}{Colors.ENDC}{Colors.GREEN} cells removed ({total_qc_failed/n1*100:.1f}%){Colors.ENDC}")

    # Last gene and cell filter
    print(f"\n{Colors.HEADER}{Colors.BOLD}🎯 Step 3: Final Filtering{Colors.ENDC}")
    print(f"   {Colors.CYAN}Parameters: min_genes={min_genes}, min_cells={min_cells}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Ratios: max_genes_ratio={max_genes_ratio}, max_cells_ratio={max_cells_ratio}{Colors.ENDC}")
    
    cells_before_final = adata.shape[0]
    genes_before_final = adata.shape[1]
    
    rsc.pp.filter_cells(adata, min_counts=min_genes)
    rsc.pp.filter_cells(adata, max_counts=max_genes_ratio*adata.shape[1])
    rsc.pp.filter_genes(adata, min_counts=min_cells)
    rsc.pp.filter_genes(adata, max_counts=max_cells_ratio*adata.shape[0])
    
    cells_final_filtered = cells_before_final - adata.shape[0]
    genes_final_filtered = genes_before_final - adata.shape[1]
    
    print(f"   {Colors.GREEN}✓ Final filtering: {Colors.BOLD}{cells_final_filtered:,}{Colors.ENDC}{Colors.GREEN} cells, {Colors.BOLD}{genes_final_filtered:,}{Colors.ENDC}{Colors.GREEN} genes removed{Colors.ENDC}")
    
    n_after_final_filt = adata.shape[0]
    
    if doublets is True:
        print(f"\n{Colors.HEADER}{Colors.BOLD}🔍 Step 4: Doublet Detection{Colors.ENDC}")
        if doublets_method=='scrublet':
            print(f"   {Colors.GREEN}{EMOJI['start']} Running GPU-accelerated scrublet...{Colors.ENDC}")
            rsc.pp.scrublet(adata, random_state=1234,batch_key=batch_key)
            adata_remove = adata[adata.obs['predicted_doublet'], :]
            removed_cells.extend(list(adata_remove.obs_names))
            adata = adata[~adata.obs['predicted_doublet'], :]
            n1 = adata.shape[0]
            doublets_removed = n_after_final_filt-n1
            print(f"   {Colors.GREEN}✓ Scrublet completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            
        elif doublets_method=='sccomposite':
            print(f"   {Colors.GREEN}{EMOJI['start']} Running sccomposite doublet detection...{Colors.ENDC}")
            adata.obs['sccomposite_doublet']=0
            adata.obs['sccomposite_consistency']=0
            if batch_key is None:
                from ._sccomposite import composite_rna
                multiplet_classification, consistency = composite_rna(adata)
                adata.obs['sccomposite_doublet']=multiplet_classification
                adata.obs['sccomposite_consistency']=consistency
            else:
                print(f"   {Colors.CYAN}Processing {len(adata.obs[batch_key].unique())} batches separately...{Colors.ENDC}")
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
            doublets_removed = n_after_final_filt-n1
            print(f"   {Colors.GREEN}✓ sccomposite completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")

    # Store status
    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    adata.uns['status']['qc']=True
    
    # Final summary
    print(f"\n{Colors.GREEN}{EMOJI['done']} GPU Quality Control Analysis Completed!{Colors.ENDC}")
    print(f"\n{Colors.HEADER}{Colors.BOLD}📈 Final Summary:{Colors.ENDC}")
    print(f"   {Colors.CYAN}📊 Original: {Colors.BOLD}{n0:,}{Colors.ENDC}{Colors.CYAN} cells × {Colors.BOLD}{genes_before_final:,}{Colors.ENDC}{Colors.CYAN} genes{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Final: {Colors.BOLD}{adata.shape[0]:,}{Colors.ENDC}{Colors.GREEN} cells × {Colors.BOLD}{adata.shape[1]:,}{Colors.ENDC}{Colors.GREEN} genes{Colors.ENDC}")
    print(f"   {Colors.BLUE}📉 Total removed: {Colors.BOLD}{n0-adata.shape[0]:,}{Colors.ENDC}{Colors.BLUE} cells ({(n0-adata.shape[0])/n0*100:.1f}%){Colors.ENDC}")
    print(f"   {Colors.BLUE}📉 Total removed: {Colors.BOLD}{genes_before_final-adata.shape[1]:,}{Colors.ENDC}{Colors.BLUE} genes ({(genes_before_final-adata.shape[1])/genes_before_final*100:.1f}%){Colors.ENDC}")
    
    # Quality assessment
    final_retention_rate = adata.shape[0] / n0
    if final_retention_rate >= 0.8:
        quality_color = Colors.GREEN
        quality_msg = "Excellent retention rate"
    elif final_retention_rate >= 0.6:
        quality_color = Colors.CYAN
        quality_msg = "Good retention rate"
    else:
        quality_color = Colors.WARNING
        quality_msg = "Low retention rate - consider relaxing thresholds"
    
    print(f"   {quality_color}💯 Quality: {Colors.BOLD}{final_retention_rate*100:.1f}%{Colors.ENDC}{quality_color} retention ({quality_msg}){Colors.ENDC}")
    print(f"\n{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
    
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
    i.e. "unreliable" observations.
    
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
    print(f"{Colors.GREEN}{EMOJI['start']} Filtering cells...{Colors.ENDC}")
    filter_params = []
    if min_counts is not None:
        filter_params.append(f"min_counts≥{min_counts}")
    if min_genes is not None:
        filter_params.append(f"min_genes≥{min_genes}")
    if max_counts is not None:
        filter_params.append(f"max_counts≤{max_counts}")
    if max_genes is not None:
        filter_params.append(f"max_genes≤{max_genes}")
    
    if filter_params:
        print(f"   {Colors.CYAN}Parameters: {', '.join(filter_params)}{Colors.ENDC}")
    
    cells_before = adata.shape[0]
    sc.pp.filter_cells(adata, min_genes=min_genes,min_counts=min_counts, max_counts=max_counts,
                       max_genes=max_genes, inplace=inplace)
    cells_filtered = cells_before - adata.shape[0]
    print(f"   {Colors.GREEN}✓ Filtered: {Colors.BOLD}{cells_filtered:,}{Colors.ENDC}{Colors.GREEN} cells removed{Colors.ENDC}")

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
    print(f"{Colors.GREEN}{EMOJI['start']} Filtering genes...{Colors.ENDC}")
    filter_params = []
    if min_counts is not None:
        filter_params.append(f"min_counts≥{min_counts}")
    if min_cells is not None:
        filter_params.append(f"min_cells≥{min_cells}")
    if max_counts is not None:
        filter_params.append(f"max_counts≤{max_counts}")
    if max_cells is not None:
        filter_params.append(f"max_cells≤{max_cells}")
    
    if filter_params:
        print(f"   {Colors.CYAN}Parameters: {', '.join(filter_params)}{Colors.ENDC}")
    
    genes_before = adata.shape[1]
    sc.pp.filter_genes(adata, min_counts=min_counts, min_cells=min_cells, max_counts=max_counts,
                          max_cells=max_cells, inplace=inplace)
    genes_filtered = genes_before - adata.shape[1]
    print(f"   {Colors.GREEN}✓ Filtered: {Colors.BOLD}{genes_filtered:,}{Colors.ENDC}{Colors.GREEN} genes removed{Colors.ENDC}")
    
