import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import seaborn as sns

from ._aucell import derive_auc_threshold,create_rankings,aucell
from ctxcore.recovery import enrichment4cells,aucs
from ctxcore.genesig import GeneSignature
from ..utils import plot_text_set
import gseapy as gp

def geneset_aucell(adata,geneset_name,geneset,AUC_threshold=0.01,seed=42):
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
    plot_heatmap.fillna(0)
    if return_table==True:
        return plot_heatmap
    
    fig,ax=plt.subplots(figsize=figsize)
    axr=sns.heatmap(plot_heatmap.T,ax=ax,**kwds)
    axr.tick_params(right=True, top=False,left=False, labelright=True, labelleft=False,labeltop=False,rotation=0)
    ax.set_title(plot_title,fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 10)

    
    return ax



