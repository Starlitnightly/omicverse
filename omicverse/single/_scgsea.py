import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import seaborn as sns
from .._settings import add_reference
from ._aucell import derive_auc_threshold,fast_rank,_rank_sparse_row,aucell


def geneset_aucell_tmp(adata, geneset_name, geneset, AUC_threshold=0.01, seed=42, chunk_size=10000):
    r"""Calculate the AUC-ell score for a given gene set.

    Arguments:
        adata: AnnData object containing gene expression data.
        geneset_name: Name of the gene set.
        geneset: List of gene symbols for the gene set.
        AUC_threshold: AUC threshold used to determine significant interactions. (0.01)
        seed: Seed used to initialize the random number generator. (42)
        chunk_size: The number of cells to process in each chunk. (10000)

    Returns:
        None: Adds a column to the 'obs' attribute of the adata object containing the AUC-ell score for the gene set.
    """
    from ..external.ctxcore.recovery import aucs

    matrix = adata.X.copy()
    percentiles = derive_auc_threshold(matrix, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]

    n_cells = matrix.shape[0]
    auc_results = np.zeros(n_cells, dtype=np.float64)

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = matrix[start:end]
        np_rnk_sparse = fast_rank(chunk, seed=seed)
        rnk = pd.DataFrame(np_rnk_sparse[:, np.where(adata.var_names.isin(geneset))[0]])

        if rnk.empty or (float(len(np.where(adata.var_names.isin(geneset))[0])) / float(len(geneset))) < 0.80:
            print(
                f"Less than 80% of the genes in {geneset_name} are present in the "
                "expression matrix."
            )
        else:
            weights = np.array([1 for _ in geneset])
            auc_results[start:end] = aucs(rnk, np_rnk_sparse.shape[1], weights, auc_threshold)

    adata.obs[f'{geneset_name}_aucell'] = auc_results

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

def geneset_aucell(adata,geneset_name,geneset,AUC_threshold=0.01,seed=42):
    r"""Calculate the AUC-ell score for a given gene set.

    Arguments:
        adata: AnnData object containing gene expression data.
        geneset_name: Name of the gene set.
        geneset: List of gene symbols for the gene set.
        AUC_threshold: AUC threshold used to determine significant interactions. (0.01)
        seed: Seed used to initialize the random number generator. (42)

    Returns:
        None: Adds a column to the 'obs' attribute of the adata object containing the AUC-ell score for the gene set.
    """
    from ..external.ctxcore.recovery import aucs

    matrix = adata.X.copy()
    percentiles = derive_auc_threshold(matrix, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]

    np_rnk_sparse=fast_rank(matrix, seed= seed)

    rnk  = pd.DataFrame(np_rnk_sparse[:, np.where(adata.var_names.isin(geneset))[0]])

    if rnk.empty or (float(np_rnk_sparse.shape[1]) / float(len(geneset))) < 0.80:
        print(
            f"Less than 80% of the genes in {geneset_name} are present in the "
            "expression matrix."
        )
        adata.obs['{}_aucell'.format(geneset_name)]=np.zeros(shape=(rnk.shape[0]), dtype=np.float64)
    else:
        weights = np.array([1 for i in geneset])
        adata.obs['{}_aucell'.format(geneset_name)]=aucs(rnk,
        np_rnk_sparse.shape[1],weights,auc_threshold
        )
    
def pathway_aucell(adata,pathway_names,pathways_dict,AUC_threshold=0.01,seed=42):
    r"""Calculate the area under the curve (AUC) for a set of pathways in an AnnData object.

    Arguments:
        adata: AnnData object containing the data.
        pathway_names: Names of the pathways to analyze.
        pathways_dict: Dictionary containing the gene sets for each pathway.
        AUC_threshold: AUC threshold to use for determining significant gene-pathway associations. (0.01)
        seed: Random seed for reproducibility. (42)

    Returns:
        None: The function modifies the `adata.obs` attribute of the input AnnData object.
    """
    from ..external.ctxcore.recovery import aucs

    matrix = adata.X.copy()
    percentiles = derive_auc_threshold(matrix, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]

    np_rnk_sparse=fast_rank(matrix, seed= seed)

    for pathway_name in pathway_names:
        pathway_genes=pathways_dict[pathway_name]

        rnk  = pd.DataFrame(np_rnk_sparse[:, np.where(adata.var_names.isin(pathway_genes))[0]]).copy()
        if rnk.empty or (float(np_rnk_sparse.shape[1]) / float(len(pathway_genes))) < 0.80:
            print(
                f"Less than 80% of the genes in {pathway_name} are present in the "
                "expression matrix."
            )
            adata.obs['{}_aucell'.format(pathway_name)]=np.zeros(shape=(rnk.shape[0]), dtype=np.float64)
        else:
            weights = np.array([1 for i in pathway_genes])
            adata.obs['{}_aucell'.format(pathway_name)]=aucs(rnk,
            np_rnk_sparse.shape[1],weights,auc_threshold
            )
    add_reference(adata,'AUCell','pathway activity score with AUCell')
            
def pathway_aucell_tmp(adata, pathway_names, pathways_dict, AUC_threshold=0.01, seed=42, chunk_size=10000):
    r"""Calculate the area under the curve (AUC) for a set of pathways in an AnnData object.

    Arguments:
        adata: AnnData object containing the data.
        pathway_names: Names of the pathways to analyze.
        pathways_dict: Dictionary containing the gene sets for each pathway.
        AUC_threshold: AUC threshold to use for determining significant gene-pathway associations. (0.01)
        seed: Random seed for reproducibility. (42)
        chunk_size: The number of cells to process in each chunk. (10000)

    Returns:
        None: The function modifies the `adata.obs` attribute of the input AnnData object.
    """
    from ..external.ctxcore.recovery import aucs

    matrix = adata.X.copy()
    percentiles = derive_auc_threshold(matrix, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]
    
    n_cells = matrix.shape[0]
    
    for pathway_name in pathway_names:
        pathway_genes = pathways_dict[pathway_name]
        auc_results = np.zeros(n_cells, dtype=np.float64)
        
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = matrix[start:end]
            np_rnk_sparse = fast_rank(chunk, seed=seed)
            
            rnk = pd.DataFrame(np_rnk_sparse[:, np.where(adata.var_names.isin(pathway_genes))[0]])
            if rnk.empty or (float(len(np.where(adata.var_names.isin(pathway_genes))[0])) / float(len(pathway_genes))) < 0.80:
                print(
                    f"Less than 80% of the genes in {pathway_name} are present in the "
                    "expression matrix."
                )
            else:
                weights = np.array([1 for _ in pathway_genes])
                auc_results[start:end] = aucs(rnk, np_rnk_sparse.shape[1], weights, auc_threshold)

        adata.obs[f'{pathway_name}_aucell'] = auc_results
            
def pathway_aucell_enrichment(adata,pathways_dict,AUC_threshold=0.01,seed=42,num_workers=1,gene_overlap_threshold=0.80):
    r"""Enrich cell annotations with pathway activity scores using the AUC-ell method.

    Arguments:
        adata: AnnData object containing the expression matrix.
        pathways_dict: A dictionary where keys are pathway names and values are lists of genes associated with each pathway.
        AUC_threshold: The threshold for calculating the area under the curve (AUC) values using the AUC-ell method. (0.01)
        seed: The seed to use for the random number generator. (42)
        num_workers: The number of workers to use for parallel processing. (1)
        gene_overlap_threshold: Minimum fraction of pathway genes that must be present in expression matrix (0.80 = 80%).

    Returns:
        adata_aucs: AnnData object containing the pathway activity scores for each cell in the input AnnData object.
    """
    from ..external.ctxcore.genesig import GeneSignature

    test_gmt=[]
    for i in pathways_dict.keys():
        test_gmt.append(GeneSignature(name=i,gene2weight=dict(zip(pathways_dict[i],[1 for i in pathways_dict[i]]))))

    # Use sparse matrix for both derive_auc_threshold and aucell
    matrix_sparse = adata.X.copy()
    percentiles = derive_auc_threshold(matrix_sparse, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]

    # Pass sparse matrix directly to aucell with index and columns
    aucs_mtx = aucell(matrix_sparse, signatures=test_gmt, auc_threshold=auc_threshold,
                     num_workers=num_workers, seed=seed,
                     index=adata.obs_names, columns=adata.var_names,
                     gene_overlap_threshold=gene_overlap_threshold)
    
    adata_aucs=anndata.AnnData(aucs_mtx)
    add_reference(adata,'AUCell','pathway activity score with AUCell')
    return adata_aucs

def pathway_aucell_enrichment_tmp(adata, pathways_dict, AUC_threshold=0.01, seed=42, 
                              num_workers=1, chunk_size=10000):
    r"""Enrich cell annotations with pathway activity scores using the AUC-ell method.

    Arguments:
        adata: AnnData object containing the expression matrix.
        pathways_dict: A dictionary where keys are pathway names and values are lists of genes associated with each pathway.
        AUC_threshold: The threshold for calculating the area under the curve (AUC) values using the AUC-ell method. (0.01)
        seed: The seed to use for the random number generator. (42)
        num_workers: The number of workers to use for parallel processing. (1)
        chunk_size: The number of cells to process in each chunk. (10000)

    Returns:
        adata_aucs: AnnData object containing the pathway activity scores for each cell in the input AnnData object.
    """
    from tqdm import tqdm
    from ..external.ctxcore.genesig import GeneSignature

    test_gmt = []
    for i in pathways_dict.keys():
        test_gmt.append(GeneSignature(name=i, gene2weight=dict(zip(pathways_dict[i], [1 for _ in pathways_dict[i]]))))

    # Use sparse matrix for derive_auc_threshold
    matrix_sparse = adata.X.copy()
    percentiles = derive_auc_threshold(matrix_sparse, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]

    # Process in chunks using sparse matrix slicing
    aucs_mtx_list = []
    n_cells = matrix_sparse.shape[0]
    
    for start in tqdm(range(0, n_cells, chunk_size)):
        end = min(start + chunk_size, n_cells)
        # Slice sparse matrix directly
        chunk_sparse = matrix_sparse[start:end]
        chunk_index = adata.obs_names[start:end]
        
        aucs_chunk = aucell(chunk_sparse, signatures=test_gmt, auc_threshold=auc_threshold, 
                           num_workers=num_workers, seed=seed,
                           index=chunk_index, columns=adata.var_names)
        aucs_mtx_list.append(aucs_chunk)

    # Concatenate the results
    aucs_mtx = pd.concat(aucs_mtx_list)

    adata_aucs = anndata.AnnData(aucs_mtx)
    return adata_aucs


def pathway_enrichment(adata, pathways_dict,organism='Human',group_by='louvain', 
                       cutoff=0.05, logfc_threshold=2,pvalue_type='adjust',plot=True):
    r"""Perform pathway enrichment analysis on gene expression data.
    
    Arguments:
        adata: Annotated data matrix containing gene expression data.
        pathways_dict: A dictionary of gene sets with their gene members.
        organism: The organism to be used for the enrichment analysis. Can be either 'Human' or 'Mouse'. ('Human')
        group_by: The group label of the cells in adata.obs to perform the enrichment analysis on. ('louvain')
        cutoff: The adjusted p-value cutoff used to filter enriched pathways. (0.05)
        logfc_threshold: The log2 fold change cutoff used to define differentially expressed genes. (2)
        pvalue_type: The type of p-value used for filtering enriched pathways. Can be either 'adjust' or 'raw'. ('adjust')
        plot: If True, generate a bar plot for each cluster of enriched pathways. (True)
    
    Returns:
        enrich_res: A pandas dataframe containing the enriched pathways information including term name, p-value, adjusted p-value, overlap genes, odds ratio, log2 fold change, and cluster name.
    
    Examples:
        >>> # Perform pathway enrichment analysis on adata using pathways_dict
        >>> res = pathway_enrichment(adata, pathways_dict)
    
    Reference:
        The code for pathway_enrichment() function was adapted from scanpy workflows: 
        https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html#Gene-set-enrichment-analysis.
    """

    #import gseapy as gp
    from ..external.gseapy import enrichr
    df_list = []
    cluster_list = []
    celltypes = sorted(adata.obs[group_by].unique())

    for celltype in celltypes:
        degs = sc.get.rank_genes_groups_df(adata, group=celltype, key='rank_genes_groups', log2fc_min=logfc_threshold, 
                                    pval_cutoff=cutoff)['names'].squeeze()
        if isinstance(degs, str):
            degs = [degs.strip()]
        else:
            degs = degs.str.strip().tolist()
        
        if not degs:
            continue
        if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
            background='mmusculus_gene_ensembl'
        elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
            background='hsapiens_gene_ensembl'
        else:
            background=adata.var.index.tolist()
        enr = enrichr(gene_list=degs,
                description='',
                gene_sets=pathways_dict,
                organism=organism,
                background=background,
                cutoff=0.5,
                )
        if (enr is not None) and hasattr(enr, 'res2d') and (enr.res2d.shape[0] > 0):
            df_list.append(enr.res2d)
            cluster_list.append(celltype)

    columns = ['Cluster', 'Gene_set', 'Term', 'Overlap', 'Odds Ratio','P-value', 'Adjusted P-value', 'Genes']

    df = pd.DataFrame(columns = columns)
    for cluster_ind, df_ in zip(cluster_list, df_list):
        df_ = df_[df_['Adjusted P-value'] <= cutoff]
        df_ = df_.assign(Cluster = cluster_ind)
        if (df_.shape[0] > 0):
            df = pd.concat([df, df_[columns]], sort=False)
            df_tmp = df_.loc[:, ['Term', 'Adjusted P-value']][:min(10, df_.shape[0])]
            df_tmp['Term'] = [x.split('(',1)[0] for x in df_tmp['Term']]
            df_tmp['-log_adj_p'] = - np.log10(df_tmp['Adjusted P-value'])
            df_tmp = df_tmp.sort_values(by='-log_adj_p', ascending=True)
            if plot==True:
                ax = df_tmp.plot.barh(y='-log_adj_p', x='Term', legend=False, grid=False, figsize=(12,4))
                ax.set_title('Cluster {}'.format(cluster_ind))
                ax.set_ylabel('')
                ax.set_xlabel('-log(Adjusted P-value)')
            #pp.savefig(ax.figure, bbox_inches='tight')
            #plt.close()
        else:
            print('No pathway with an adjusted P-value less than the cutoff (={}) for cluster {}'.format(cutoff, cluster_ind))
    
    if pvalue_type=='adjust':
        enrich_res=df[df['Adjusted P-value']<cutoff]
        enrich_res['logp']=-np.log(enrich_res['Adjusted P-value'])
    else:
        enrich_res=df[df['P-value']<cutoff]
        enrich_res['logp']=-np.log(enrich_res['P-value'])
    enrich_res['logc']=np.log(enrich_res['Odds Ratio'])
    enrich_res['num']=[int(i.split('/')[0]) for i in enrich_res['Overlap']]
    enrich_res['fraction']=[int(i.split('/')[0])/int(i.split('/')[1]) for i in enrich_res['Overlap']]
    add_reference(adata,'GSEApy','pathway enrichment analysis with gseapy')
    
    return enrich_res

def pathway_enrichment_plot(enrich_res,term_num=5,return_table=False,figsize=(3,10),plot_title='',**kwds):
    r"""Visualize the pathway enrichment analysis results as a heatmap.

    Arguments:
        enrich_res: The output from the pathway_enrichment() function.
        term_num: The number of enriched terms to display for each cluster. (5)
        return_table: Whether to return the heatmap table as a DataFrame. (False)
        figsize: The size of the plot. ((3,10))
        plot_title: The title of the plot. ('')
        **kwds: Other keyword arguments to pass to the seaborn heatmap function.

    Returns:
        ax: The heatmap plot.

    Examples:
        >>> res = pathway_enrichment(adata, pathways_dict)
        >>> pathway_enrichment_plot(res, term_num=10, return_table=True, figsize=(6,12), cmap='Blues')
    """

    celltypes=enrich_res['Cluster'].unique()
    plot_heatmap=pd.DataFrame(index=celltypes)
    for celltype in celltypes:
        res_test=enrich_res.loc[enrich_res['Cluster']==celltype].iloc[:term_num]
        plot_heatmap_test=pd.DataFrame(res_test[['Term','logp']])
        plot_heatmap_test=plot_heatmap_test.set_index(plot_heatmap_test['Term'])
        del plot_heatmap_test['Term']
        #plot_heatmap[plot_heatmap_test.index]=0
        for i in plot_heatmap_test.index:
            if len(enrich_res.loc[(enrich_res['Cluster']==celltype)&(enrich_res['Term']==i),'logp'].values)!=0:
                plot_heatmap.loc[celltype,i]=enrich_res.loc[(enrich_res['Cluster']==celltype)&(enrich_res['Term']==i),'logp'].values[0]
            else:
                plot_heatmap.loc[celltype,i]=0
    plot_heatmap.fillna(0,inplace=True)
    if return_table==True:
        return plot_heatmap
    
    fig,ax=plt.subplots(figsize=figsize)
    axr=sns.heatmap(plot_heatmap.T,ax=ax,**kwds)
    axr.tick_params(right=True, top=False,left=False, labelright=True, labelleft=False,labeltop=False,rotation=0)
    ax.set_title(plot_title,fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 10)

    
    return ax



