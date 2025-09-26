import pdb,sys,os
import anndata
import scanpy as sc
import argparse
import copy
import torch
import numpy as np
import gc
import pandas as pd
import timeit
import scipy
import warnings
warnings.filterwarnings('ignore')
import faiss
from sklearn.cluster import KMeans
import sklearn
from scipy import stats
from sklearn.neighbors import kneighbors_graph
from matplotlib.pyplot import figure

from typing import Tuple,Union
from torch.utils.data import Dataset



def hamster_to_human(hamster_gene_list):
    f=open('scSemiProfiler/hamster_to_human_gene.txt','r')
    lines = f.readlines()
    dic = {}
    for l in lines:
        l = l.strip().split()
        if len(l)==2:
            dic[l[0]]=l[1]
    human_gene_list = []
    for g in hamster_gene_list:
        if g in dic.keys():
            human_gene_list.append(dic[g])
        else:
            human_gene_list.append(g)
    
    return human_gene_list

def gen_tf_gene_table(genes, tf_list, dTD):
    """
    Adapted from:
    Author: Jun Ding
    Project: SCDIFF2
    Ref: Ding, J., Aronow, B. J., Kaminski, N., Kitzmiller, J., Whitsett, J. A., & Bar-Joseph, Z.
    (2018). Reconstructing differentiation networks and their regulation from time series
    single-cell expression data. Genome research, 28(3), 383-395.
    """
    

    gene_names = [g.upper() for g in genes]
    TF_names = [g.upper() for g in tf_list]
    tf_gene_table = dict.fromkeys(tf_list)

    for i, tf in enumerate(tf_list):
        tf_gene_table[tf] = np.zeros(len(gene_names))
        _genes = dTD[tf]

        _existed_targets = list(set(_genes).intersection(gene_names))
        _idx_targets = map(lambda x: gene_names.index(x), _existed_targets)

        for _g in _idx_targets:
            tf_gene_table[tf][_g] = 1

    del gene_names
    del TF_names
    del _genes
    del _existed_targets
    del _idx_targets

    gc.collect()

    return tf_gene_table



def getGeneSetMatrix(_name, genes_upper, gene_sets_path):
    """

    Adapted from:
    Author: Jun Ding
    Project: SCDIFF2
    Ref: Ding, J., Aronow, B. J., Kaminski, N., Kitzmiller, J., Whitsett, J. A., & Bar-Joseph, Z.
    (2018). Reconstructing differentiation networks and their regulation from time series
    single-cell expression data. Genome research, 28(3), 383-395.

    """
    if _name[-3:] == 'gmt':
        print(f"GMT file {_name} loading ... ")
        filename = _name
        filepath = os.path.join(gene_sets_path, f"{filename}")

        with open(filepath) as genesets:
            pathway2gene = {line.strip().split("\t")[0]: line.strip().split("\t")[2:]
                            for line in genesets.readlines()}

        print(len(pathway2gene))

        gs = []
        for k, v in pathway2gene.items():
            gs += v

        print(f"Number of genes in {_name} {len(set(gs).intersection(genes_upper))}")

        pathway_list = pathway2gene.keys()
        pathway_gene_table = gen_tf_gene_table(genes_upper, pathway_list, pathway2gene)
        gene_set_matrix = np.array(list(pathway_gene_table.values()))
        keys = pathway_gene_table.keys()

        del pathway2gene
        del gs
        del pathway_list
        del pathway_gene_table

        gc.collect()


    elif _name == 'TF-DNA':

        # get TF-DNA dictionary
        # TF->DNA
        def getdTD(tfDNA):
            dTD = {}
            with open(tfDNA, 'r') as f:
                tfRows = f.readlines()
                tfRows = [item.strip().split() for item in tfRows]
                for row in tfRows:
                    itf = row[0].upper()
                    itarget = row[1].upper()
                    if itf not in dTD:
                        dTD[itf] = [itarget]
                    else:
                        dTD[itf].append(itarget)

            del tfRows
            del itf
            del itarget
            gc.collect()

            return dTD

        from collections import defaultdict

        def getdDT(dTD):
            gene_tf_dict = defaultdict(lambda: [])
            for key, val in dTD.items():
                for v in val:
                    gene_tf_dict[v.upper()] += [key.upper()]

            return gene_tf_dict

        tfDNA_file = os.path.join(gene_sets_path, f"Mouse_TF_targets.txt")
        dTD = getdTD(tfDNA_file)
        dDT = getdDT(dTD)

        tf_list = list(sorted(dTD.keys()))
        tf_list.remove('TF')

        tf_gene_table = gen_tf_gene_table(genes_upper, tf_list, dTD)
        gene_set_matrix = np.array(list(tf_gene_table.values()))
        keys = tf_gene_table.keys()

        del dTD
        del dDT
        del tf_list
        del tf_gene_table

        gc.collect()

    else:
        gene_set_matrix = None

    return gene_set_matrix, keys

        
def fast_cellgraph(adata: anndata.AnnData,k: int = 15,diagw: float=1.0) -> Tuple[anndata.AnnData, np.ndarray]:
    """
    Augment an anndata object using a cell neighbor graph. 

    Parameters
    ----------
    adata
        The dataset to be augmented
    k
        The number of neighbors to consider
    diagw
        The weight of the original cell when agregating the information

    Returns
    -------
    adata
        The augmented anndata object. 
    adj
        The adjacency matrix of the cell neighbor graph.
    """
    
    
    adj = kneighbors_graph(np.array(adata.X), k, mode='connectivity', include_self=True)
    adj = adj.toarray()
    diag = np.array(np.identity(adj.shape[0]).astype('float32'))*diagw
    adj = adj + diag
    adj = adj/adj.sum(axis=1)        
    selfw = np.zeros(adj.shape[0])
    for i in range(adj.shape[0]):
        selfw[i] = adj[i,i]
    selfw=selfw.astype('float32')
    adata.obs['selfw']=selfw
    #remove self so that not in neighbors
    for i in range(adj.shape[0]):
        adj[i,i]=0
        
    adata.obsm['adj'] = adj
    adj = torch.from_numpy(adj.astype('float32'))#.type(torch.FloatTensor)
    neighboridx = np.where(adj!=0)
    xs = neighboridx[0]
    ys = neighboridx[1]
        
    maxn=k
    neighbors = np.zeros((adj.shape[0],maxn-1)) - 1
    for i in range(len(adata.obs)):
        ns=np.zeros(maxn-1)-1
        flag=0
        j=0
        k=0
        while flag!=2 and j<xs.shape[0]:
            if xs[j]==i:
                if k < maxn-1:  # Add bounds check
                    ns[k] = (ys[j])
                    k+=1
                    flag=1
                else:
                    flag=2  # Stop if we've filled all neighbor slots
            elif flag==1:
                flag=2
            j+=1
        neighbors[i] = ns
        
    neighbors = neighbors.astype(int)
    adata.obsm['neighbors']=neighbors
    neighborx = np.array(adata.X)
    
    normchoice = 0
    if normchoice == 0:
        neighborx = np.log(1+neighborx)*selfw[0] + np.log(1+neighborx[neighbors,:]).sum(axis=1)*(1-selfw[0])/(maxn-1)
    else:                                       ## 2*
        neighborx = neighborx*selfw[0] + neighborx[neighbors].sum(axis=1)*(1-selfw[0])/(maxn-1)
        neighborx = np.log(1+neighborx)
    
    adata.obsm['neighborx']=neighborx
    
    return adata,adj



    
def scprocess(
    name:str,singlecell:str, 
    logged:bool = False, normed:bool = True, 
    cellfilter:bool = False, threshold:float = 1e-3, 
    geneset:Union[bool,str] = True, 
    weight:float=0.5, k:int=15) -> None:
    """
    Process the reprsentatives' single-cell data, including preprocessing and feature augmentations. 

    Parameters
    ----------
    name
        Project name.
    singlecell
        Path to representatives' single-cell data.
    logged
        Whether the data has been logged or not
    normed
        Whether the library size has been normalized or not
    cellfilter
        Whether to perform standard cell filtering.
    threshold
        Threshold for background noise removal.
    geneset
        Whether to use gene set to augment gene expression features or no.
    weight
        The proportion of top features to increase importance weight.
    k
        K for the K-NN graph built for cells.
        
    Returns
    -------
        None
        
    Example
    -------
    >>> scSemiProfiler.scprocess(name = 'project_name', singlecell = name+'/representative_sc.h5ad', logged = False, normed = True, cellfilter = False, threshold=1e-3, geneset=True, weight = 0.5, k = 15)
    
    
    """
    
    print('Processing representative single-cell data')
    
    scdata = anndata.read_h5ad(singlecell)
    sids = np.unique(scdata.obs['sample_ids'])
    
    # cell filtering
    if cellfilter == True:
        print('Filtering cells')
        sc.pp.filter_cells(scdata, min_genes=200)
        scdata.var['mt'] = scdata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(scdata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        scdata = scdata[scdata.obs.n_genes_by_counts < 2500, :]
        scdata = scdata[scdata.obs.pct_counts_mt < 5, :]
    
    if logged == True:
        print('recovering log-transformed data to count data')
        adata = scdata
        bdata = anndata.AnnData((np.exp(adata.X)-1))
        bdata.obs = adata.obs
        bdata.var = adata.var
        bdata.obsm = adata.obsm
        bdata.uns = adata.uns
        scdata = bdata
    
    if normed == False:
        print('Library size normalization.')
        sc.pp.normalize_total(scdata, target_sum=1e4)    
    
    # convert to dense if sparse
    if scipy.sparse.issparse(scdata.X):
        X = np.array(scdata.X.todense())
        tempdata = anndata.AnnData(X)
        tempdata.obs = scdata.obs
        tempdata.var = scdata.var
        scdata = tempdata
    
    
    # norm remove noise
    if float(threshold) > 0:
        print('Removing background noise')
        X = np.array(scdata.X)
        cutoff = 1e4*threshold
        X = X * np.array(X>cutoff)
        nscdata = anndata.AnnData(X)
        nscdata.obs = scdata.obs
        nscdata.obsm = scdata.obsm
        nscdata.var = scdata.var
        nscdata.uns = scdata.uns
        scdata = nscdata
    
    
    
    # store singlecell data, geneset score
    if (os.path.isdir(name + '/sample_sc')) == False:
        os.system('mkdir ' + name + '/sample_sc')

    
    
    
    
    if geneset != False:
        if (os.path.isdir(name + '/geneset_scores')) == False:
            os.system('mkdir ' + name + '/geneset_scores')
        
        prior_name = "c2.cp.v7.4.symbols.gmt" 
        if (geneset == True) or (geneset == 'human'):
            print('Computing human geneset scores')
        elif geneset == 'hamster':
            print('Computing hamster geneset scores')
        zps=[]
        for sid in sids:
            adata = scdata[scdata.obs['sample_ids'] == sid]
            X = adata.X

            gene_sets_path = "genesets/"
            genes = list(adata.var.index)
            
            if geneset == 'hamster':
                genes = hamster_to_human(genes)
            genes_upper = [g.upper() for g in genes]
            N = adata.X.shape[0]
            G = len(genes_upper)
            gene_set_matrix, keys_all = getGeneSetMatrix(prior_name, genes_upper, gene_sets_path)

            zp = X.dot(np.array(gene_set_matrix).T)
            eps = 1e-6
            den = (np.array(gene_set_matrix.sum(axis=1))+eps)
            zp = (zp+eps)/den
            zp = zp - eps/den
            np.save(name + '/geneset_scores/' + sid,zp)
            zps.append(zp)

        if 'hvset.npy' not in os.listdir(name):
            zps=np.concatenate(zps,axis=0)
            zdata = anndata.AnnData(zps)
            sc.pp.log1p(zdata)
            sc.pp.highly_variable_genes(zdata)
            hvset = zdata.var.highly_variable
            np.save(name + '/hvset.npy',hvset)
        
        # select highly variable genes (genes in preprocessed bulk data)
        hvgenes = np.load(name + '/hvgenes.npy', allow_pickle = True)
        
        for g in hvgenes:
            if g not in scdata.var.index:
                print('Error. Bulk data contains genes that are not in single-cell data. Please remove those genes from the bulk data and try again.')
                return
        
        hvmask = []
        for i in scdata.var.index:
            if i in hvgenes:
                hvmask.append(True)
            else:
                hvmask.append(False)
        hvmask = np.array(hvmask)
        scdata = scdata[:,hvmask]
        np.save(name + '/hvmask.npy',hvmask)
    

    print('Augmenting and saving single-cell data.')
    for sid in sids:
        adata = scdata[scdata.obs['sample_ids'] == sid]
        
        # gcn
        adata.obs['cellidx']=range(len(adata.obs))
        adata,adj = fast_cellgraph(adata,k=k,diagw=1.0)
        
        
        if geneset == True:
            # # importance weight
            sample_geneset = np.load(name + '/geneset_scores/'+sid+'.npy')
            setmask = np.load(name + '/hvset.npy')
            sample_geneset = sample_geneset[:,setmask]
            sample_geneset = sample_geneset.astype('float32')

            features = np.concatenate([adata.X,sample_geneset],1)
        else:
            features = adata.X
            
        variances = np.var(features,axis=0)
        adata.uns['feature_var'] = variances
        
        adata.write(name + '/sample_sc/' + sid + '.h5ad')
    
    print('Finished processing representative single-cell data')
    return 




def main():
    parser=argparse.ArgumentParser(description="scSemiProfiler scprocess")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('--singlecell',required=True,help="Input representatives' single-cell data as a h5ad file. Sample IDs should be stored in obs.['sample_ids']. Cell IDs should be stored in obs.index. Gene symbols should be stored in var.index. Values should either be raw read counts or normalized expression.")
    
    required.add_argument('--name',required=True, help="Project name.")
    
    optional.add_argument('--normed',required=False, default='no', help="Whether the library size normalization has already been done (Default: no)") ###
    
    
    optional.add_argument('--cellfilter',required=False, default='yes', help="Whether to perform cell filtering: 'yes' or 'no'. (Default: yes)")
    optional.add_argument('--threshold',required=False, default='1e-3', help="The threshold for removing extremely low expressed background noise, as a proportion of the library size. (Default: 1e-3)")
    optional.add_argument('--geneset',required=False, default='human', help="Specify the gene set file: 'human', 'mouse', 'none', or path to the file (Default: 'human')")
    optional.add_argument('--weight',required=False, default=0.5, help="The proportion of top highly variable features to increase importance weight. (Default: 0.5)")
    optional.add_argument('--k',required=False, default=15, help="K-nearest cell neighbors used for cell graph convolution. (Default: 15)")
    
    args = parser.parse_args()
    singlecell = args.singlecell
    normed = args.normed
    name = args.name
    cellfilter = args.cellfilter
    threshold = float(args.threshold)
    geneset = args.geneset
    weight = args.weight
    k = args.k
    
    scprocess(name,singlecell,normed,cellfilter,threshold,geneset,weight,k)

if __name__=="__main__":
    main()
