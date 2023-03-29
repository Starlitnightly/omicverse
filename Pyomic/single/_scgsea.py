import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from _aucell import derive_auc_threshold,create_rankings
from ctxcore.recovery import enrichment4cells,aucs
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

def geneset_enrichment(gene_list,pathways_dict,pvalue_threshold=0.05,pvalue_type='adjust',
                       organism='Human',description='None',outdir='./enrichr',cutoff=0.5):

    enr = gp.enrichr(gene_list=gene_list,
				 gene_sets=pathways_dict,
				 organism=organism, # don't forget to set organism to the one you desired! e.g. Yeast
				 description=description,
				 outdir=outdir,
				 cutoff=0.5 # test dataset, use lower value from range(0,1)
				)
    if pvalue_type=='adjust':
        enrich_res=enr.res2d[enr.res2d['Adjusted P-value']<pvalue_threshold]
        enrich_res['logp']=-np.log(enrich_res['Adjusted P-value'])
    else:
        enrich_res=enr.res2d[enr.res2d['P-value']<pvalue_threshold]
        enrich_res['logp']=-np.log(enrich_res['P-value'])
    enrich_res['logc']=np.log(enrich_res['Odds Ratio'])
    enrich_res['num']=[int(i.split('/')[0]) for i in enrich_res['Overlap']]
    enrich_res['fraction']=[int(i.split('/')[0])/int(i.split('/')[1]) for i in enrich_res['Overlap']]
    return enrich_res

def geneset_plot(enrich_res,num=10,node_size=[5,10,15],cax_loc=2,cax_fontsize=12,
                 fig_title='',fig_xlabel='Fractions of genes',cmap='YlGnBu'):


    fig, ax = plt.subplots(figsize=(2,4))
    plot_data2=enrich_res.sort_values('P-value')[:num].sort_values('logc')
    st=ax.scatter(plot_data2['fraction'],range(len(plot_data2['logc'])),
            s=plot_data2['num']*10,linewidths=1,edgecolors='black',c=plot_data2['logp'],cmap=cmap)
    ax.yaxis.tick_right()
    plt.yticks(range(len(plot_data2['fraction'])),[plot_text_set(i.split('(')[0]) for i in plot_data2['Term']],
            fontsize=10,)
    plt.xticks(fontsize=12,)
    plt.title(fig_title,fontsize=12)
    plt.xlabel(fig_xlabel,fontsize=12)

    #fig = plt.gcf()
    cax = fig.add_axes([cax_loc, 0.55, 0.5, 0.02])
    cb=fig.colorbar(st,shrink=0.25,cax=cax,orientation='horizontal')
    cb.set_label(r'$−Log_{10}(P_{adjusted})$',fontdict={'size':cax_fontsize})

    gl_li=[]
    for i in node_size:
        gl_li.append(ax.scatter([],[], s=i*10, marker='o', color='white',edgecolors='black'))

    plt.legend(gl_li,
        [str(i) for i in node_size],
        loc='lower left',
        ncol=3,bbox_to_anchor=(-0.45, -13),
        fontsize=cax_fontsize)
    return ax
    
def plot_text_set(text):
    if len(text)>20:
        ty=text.split(' ')
        ty_len=len(ty)
        ty_mid=ty_len//2
        res=''
        for i in range(ty_len):
            if i!=ty_mid:
                res+=ty[i]+' '
            else:
                res+='\n'+ty[i]+' '
        return res
    else:
        return text