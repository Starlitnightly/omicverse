import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import seaborn as sns

from ._aucell import derive_auc_threshold,create_rankings,aucell
#from ctxcore.recovery import enrichment4cells,aucs
#from ctxcore.genesig import GeneSignature
from ..utils import plot_text_set

ctxcore_install=False

def check_ctxcore():
    """
    
    """
    global ctxcore_install
    try:
        import ctxcore
        ctxcore_install=True
        print('ctxcore have been install version:',ctxcore.__version__)
    except ImportError:
        raise ImportError(
            'Please install the ctxcore: `pip install ctxcore`.'
        )

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

def geneset_aucell(adata,geneset_name,geneset,AUC_threshold=0.01,seed=42):
    """
    Calculate the AUC-ell score for a given gene set.

    Parameters
    ----------
    - adata : `AnnData object`
        Annotated data matrix containing gene expression data.
    - geneset_name : `str`
        Name of the gene set.
    - geneset : `list` of `str`
        List of gene symbols for the gene set.
    - AUC_threshold : `float`, optional
        AUC threshold used to determine significant interactions (default is 0.01).
    - seed : `int`, optional
        Seed used to initialize the random number generator (default is 42).

    Returns
    -------
    None
        Adds a column to the 'obs' attribute of the adata object containing the AUC-ell score for the gene set.
    """
    check_ctxcore()
    global ctxcore_install
    if ctxcore_install==True:
        global aucs
        global GeneSignature
        from ctxcore.recovery import aucs
        from ctxcore.genesig import GeneSignature


    matrix = adata.to_df()
    percentiles = derive_auc_threshold(matrix)
    auc_threshold = percentiles[AUC_threshold]

    df_rnk=create_rankings(matrix, seed)
    rnk=df_rnk.iloc[:, df_rnk.columns.isin(geneset)]

    if rnk.empty or (float(len(rnk.columns)) / float(len(geneset))) < 0.80:
        print(
            f"Less than 80% of the genes in {geneset_name} are present in the "
            "expression matrix."
        )
        adata.obs['{}_aucell'.format(geneset_name)]=np.zeros(shape=(df_rnk.shape[0]), dtype=np.float64)
    else:
        weights = np.array([1 for i in geneset])
        adata.obs['{}_aucell'.format(geneset_name)]=aucs(rnk,
            len(df_rnk.columns),weights,auc_threshold
            )
    
def pathway_aucell(adata,pathway_names,pathways_dict,AUC_threshold=0.01,seed=42):
    """
    Calculates the area under the curve (AUC) for a set of pathways in an AnnData object.

    Parameters
    ----------
    - adata : `AnnData object`
        AnnData object containing the data.
    - pathway_names : `list` of `str`
        Names of the pathways to analyze.
    - pathways_dict : `dict`
        Dictionary containing the gene sets for each pathway.
    - AUC_threshold : `float`, optional (default: 0.01)
        AUC threshold to use for determining significant gene-pathway associations.
    - seed : `int`, optional (default: 42)
        Random seed for reproducibility.

    Returns
    -------
    None
        The function modifies the `adata.obs` attribute of the input AnnData object.
    """
    check_ctxcore()
    global ctxcore_install
    if ctxcore_install==True:
        global aucs
        global GeneSignature
        from ctxcore.recovery import aucs
        from ctxcore.genesig import GeneSignature

    matrix = adata.to_df()
    percentiles = derive_auc_threshold(matrix)
    auc_threshold = percentiles[AUC_threshold]
    df_rnk=create_rankings(matrix, seed)

    for pathway_name in pathway_names:
        pathway_genes=pathways_dict[pathway_name]
        rnk=df_rnk.iloc[:, df_rnk.columns.isin(pathway_genes)]
        if rnk.empty or (float(len(rnk.columns)) / float(len(pathway_genes))) < 0.80:
            print(
                f"Less than 80% of the genes in {pathway_name} are present in the "
                "expression matrix."
            )
            adata.obs['{}_aucell'.format(pathway_name)]=np.zeros(shape=(df_rnk.shape[0]), dtype=np.float64)
        else:
            weights = np.array([1 for i in pathway_genes])
            adata.obs['{}_aucell'.format(pathway_name)]=aucs(rnk,
                len(df_rnk.columns),weights,auc_threshold
                )
            
def pathway_aucell_enrichment(adata,pathways_dict,AUC_threshold=0.01,seed=42,num_workers=1):
    """
    Enriches cell annotations with pathway activity scores using the AUC-ell method.

    Parameters
    ----------
    - adata : `AnnData object`
        AnnData object containing the expression matrix.
    - pathways_dict : `dict`
        A dictionary where keys are pathway names and values are lists of genes associated with each pathway.
    - AUC_threshold : `float`, optional
        The threshold for calculating the area under the curve (AUC) values using the AUC-ell method. The default is 0.01.
    - seed : `int`, optional
        The seed to use for the random number generator. The default is 42.
    - num_workers : `int`, optional
        The number of workers to use for parallel processing. The default is 1.

    Returns
    -------
    - adata_aucs: `AnnData object`
        AnnData object containing the pathway activity scores for each cell in the input AnnData object.

    """
    check_ctxcore()
    global ctxcore_install
    if ctxcore_install==True:
        global aucs
        global GeneSignature
        from ctxcore.recovery import aucs
        from ctxcore.genesig import GeneSignature
        
    test_gmt=[]
    for i in pathways_dict.keys():
        test_gmt.append(GeneSignature(name=i,gene2weight=dict(zip(pathways_dict[i],[1 for i in pathways_dict[i]]))))

    matrix = adata.to_df()
    percentiles = derive_auc_threshold(matrix)
    auc_threshold = percentiles[AUC_threshold]

    aucs_mtx = aucell(matrix, signatures=test_gmt, auc_threshold=auc_threshold, num_workers=num_workers)
    adata_aucs=anndata.AnnData(aucs_mtx)
    return adata_aucs


def pathway_enrichment(adata, pathways_dict,organism='Human',group_by='louvain', 
                       cutoff=0.05, logfc_threshold=2,pvalue_type='adjust',plot=True):
    """
    Perform pathway enrichment analysis on gene expression data.
    
    Parameters
    ----------
    - adata : `anndata.AnnData`
        Annotated data matrix containing gene expression data.
    - pathways_dict : `dict`
        A dictionary of gene sets with their gene members.
    - organism : `str`, optional (default: 'Human')
        The organism to be used for the enrichment analysis. Can be either 'Human' or 'Mouse'.
    - group_by : `str`, optional (default: 'louvain')
        The group label of the cells in adata.obs to perform the enrichment analysis on.
    - cutoff : `float`, optional (default: 0.05)
        The adjusted p-value cutoff used to filter enriched pathways.
    - logfc_threshold : `float`, optional (default: 2)
        The log2 fold change cutoff used to define differentially expressed genes.
    - pvalue_type : `str`, optional (default: 'adjust')
        The type of p-value used for filtering enriched pathways. Can be either 'adjust' or 'raw'.
    - plot : `bool`, optional (default: True)
        If True, generate a bar plot for each cluster of enriched pathways.
    
    Returns
    -------
    - enrich_res : `pandas.DataFrame`
        A pandas dataframe containing the enriched pathways information including term name, 
        p-value, adjusted p-value, overlap genes, odds ratio, log2 fold change, and cluster name.
    
    Examples
    --------
    >>> # Perform pathway enrichment analysis on adata using pathways_dict
    >>> res = pathway_enrichment(adata, pathways_dict)
    
    Reference
    ---------
    The code for pathway_enrichment() function was adapted from scanpy workflows: 
    https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html#Gene-set-enrichment-analysis.
    """
    import gseapy as gp
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
        enr = gp.enrichr(gene_list=degs,
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
    
    return enrich_res

def pathway_enrichment_plot(enrich_res,term_num=5,return_table=False,figsize=(3,10),plot_title='',**kwds):
    """
    Visualize the pathway enrichment analysis results as a heatmap.

    Parameters:
    ----------
    - enrich_res : `pandas DataFrame`
        The output from the pathway_enrichment() function.
    - term_num : `int`, optional
        The number of enriched terms to display for each cluster. Default is 5.
    - return_table : `bool`, optional
        Whether to return the heatmap table as a DataFrame. Default is False.
    - figsize : `tuple`, optional
        The size of the plot. Default is (3,10).
    - plot_title : `str`, optional
        The title of the plot. Default is an empty string.
    - **kwds : optional
        Other keyword arguments to pass to the seaborn heatmap function.

    Returns:
    -------
    - ax : `matplotlib Axes object`
        The heatmap plot.

    Examples:
    --------
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



