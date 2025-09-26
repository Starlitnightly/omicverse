import pdb,sys,os
import anndata
import scanpy as sc
import argparse
import copy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from typing import Union

import matplotlib.pyplot as plt



def initsetup(
    name:str, bulk:str,logged:bool=False,
    normed:bool = True, 
    geneselection:Union[bool,int]=True,
    batch:int=4) -> None:
    """
    Initial setup of the semi-profiling pipeline, including processing the bulk data, clustering for finding the initial representatives. Bulk data should be provided as an 'h5ad' file. Sample IDs should be stored in adata.obs['sample_ids'] and gene names should be stored in adata.var.index. If not using active learning for iterative representative selection, directly set the batch size to be the total number of representatives desired.
    
    Parameters
    ----------
    name
        Project name. 
    bulk
        Path to bulk data as an h5ad file. Sample IDs should be stored in adata.obs['sample_ids'] and gene names should be stored in adata.var.index. 
    logged
        Whether the data has been logged or not
    normed
        Whether the library size has been normalized or not
    geneselection
        Either a boolean value indicating whether to perform gene selection using the bulk data or not, or a integer specifying the number of highly variable genes should be selected.
    batch 
        Representative selection batch size. 
    
    Returns
    -------
        None
    
    Example
    --------
    >>> import scSemiProfiler
    >>> name = 'runexample'
    >>> bulk = 'example_data/bulkdata.h5ad'
    >>> logged = False
    >>> normed = True
    >>> geneselection = False
    >>> batch = 2
    >>> scSemiProfiler.initsetup(name, bulk,logged,normed,geneselection,batch)

    """
    
    print('Start initial setup')
    
    if (os.path.isdir(name)) == False:
        os.system('mkdir '+name)
    else:
        print(name + ' exists. Please choose another name.')
        return
    
    if (os.path.isdir(name+'/figures')) == False:
        os.system('mkdir '+name+'/figures')
    
    bulkdata = anndata.read_h5ad(bulk)
    
    
    if normed == False:
        if logged == True:
            print('Bad data preprocessing. Please normalize the library size before log-transformation.')
            return
        sc.pp.normalize_total(bulkdata, target_sum=1e4)
    
    if logged == False:
        sc.pp.log1p(bulkdata)
        
    # write sample ids
    sids = list(bulkdata.obs['sample_ids'])
    f = open(name+'/sids.txt','w')
    for sid in sids:
        f.write(sid+'\n')
    f.close()
    
    
    if geneselection == False:
        hvgenes = np.array(bulkdata.var.index)
    elif geneselection == True:
        sc.pp.highly_variable_genes(bulkdata, n_top_genes=6000)
        #sc.pp.highly_variable_genes(bulkdata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        bulkdata = bulkdata[:, bulkdata.var.highly_variable]
        hvgenes = (np.array(bulkdata.var.index))[bulkdata.var.highly_variable]
    else:
        sc.pp.highly_variable_genes(bulkdata, n_top_genes = int(geneselection))
        #sc.pp.highly_variable_genes(bulkdata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        bulkdata = bulkdata[:, bulkdata.var.highly_variable]
        hvgenes = (np.array(bulkdata.var.index))[bulkdata.var.highly_variable]
    np.save(name+'/hvgenes.npy',hvgenes)
    
    #dim reduction and clustering
    
    if bulkdata.X.shape[0]>100:
        n_comps = 100
    else:
        n_comps = bulkdata.X.shape[0]-1
    
    sc.tl.pca(bulkdata,n_comps=n_comps)
    
    bulkdata.write(name + '/processed_bulkdata.h5ad')
    
    #cluster
    BATCH_SIZE = batch
    kmeans = KMeans(n_clusters=BATCH_SIZE, random_state=1).fit(bulkdata.obsm['X_pca'])
    cluster_labels = kmeans.labels_
    #find representatives and cluster labels
    pnums = []
    for i in range(len(bulkdata.X)):
        pnums.append(i)
    pnums=np.array(pnums)
    centers=[]
    representatives=[]
    repredic={}
    for i in range(len(np.unique(cluster_labels))):
        mask = (cluster_labels==i)
        cluster = bulkdata.obsm['X_pca'][mask]
        cluster_patients = pnums[mask]
        center = cluster.mean(axis=0)
        centers.append(center)
        # find the closest patient
        sqdist = ((cluster - center)**2).sum(axis=1)
        cluster_representative = cluster_patients[np.argmin(sqdist)]
        representatives.append(cluster_representative)
        repredic[i] = cluster_representative
    centers = np.array(centers)
    #store representatives cluster labels
    if (os.path.isdir(name + '/status')) == False:
        os.system('mkdir ' + name + '/status')
    

    f=open(name + '/status/init_cluster_labels.txt','w')
    for i in range(len(cluster_labels)):
        f.write(str(cluster_labels[i])+'\n')
    f.close()

    f=open(name + '/status/init_representatives.txt','w')
    for i in range(len(representatives)):
        f.write(str(representatives[i])+'\n')
    f.close()
    
    print('Initial setup finished. Among ' + str(len(sids)) + ' total samples, selected '+str(batch)+' representatives:')
    for i in range(batch):
        print(sids[representatives[i]])
    
    return





def main():
    parser=argparse.ArgumentParser(description="scSemiProfiler initsetup")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('--bulk',required=True,help="Input bulk data as a h5ad file. Sample IDs should be stored in obs.['sample_ids']. Gene symbols should be stored in var.index.")
    
    required.add_argument('--name',required=True, help="Project name.")
    
    optional.add_argument('--normed',required=False, default='no', help="Whether the library size normalization has already been done (Default: no)") ###
    
    optional.add_argument('--geneselection',required=False,default='yes', help="Whether to perform highly variable gene selection: 'yes' or 'no'. (Default: yes)")
    
    optional.add_argument('--batch',required=False, default=4, help="The representative sample batch size (Default: 4)")
    
    args = parser.parse_args()
    bulk = args.bulk
    name = args.name
    geneselection = args.geneselection
    normed = args.normed
    batch = int(args.batch)
    
    initsetup(name,bulk,normed,geneselection,batch)

if __name__=="__main__":
    main()
