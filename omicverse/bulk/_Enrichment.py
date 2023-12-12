import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import networkx as nx
import seaborn as sns
import pandas as pd
from sklearn import decomposition as skldec 

from ..utils import plot_text_set
import matplotlib

"""
def enrichment_KEGG(gene_list,
                    gene_sets=['KEGG_2019_Human'],
                    organism='Human',
                    description='test_name',
                    outdir='enrichment_kegg',
                    cutoff=0.5):

    '''
    Gene enrichment analysis of KEGG

    Parameters
    ----------
    gene_list:list
        The gene set to be enrichment analyzed
    gene_sets:list
        The gene_set of enrichr library
        Input Enrichr Libraries (https://maayanlab.cloud/Enrichr/#stats)
    organism:str
        Select from (human, mouse, yeast, fly, fish, worm)
    description:str
        The title of enrichment
    outdir:str
        The savedir of enrichment
    cutoff:float
        Show enriched terms which Adjusted P-value < cutoff.

    Returns
    ----------
    res:pandas.DataFrame
        stores your last query
    '''

    enr = gp.enrichr(gene_list=gene_list,
                 gene_sets=gene_sets,
                 organism=organism, # don't forget to set organism to the one you desired! e.g. Yeast
                 description=description,
                 outdir=outdir,
                 # no_plot=True,
                 cutoff=cutoff # test dataset, use lower value from range(0,1)
                )
    subp=dotplot(enr.res2d, title=description,cmap='seismic')
    print(subp)
    return enr.res2d

def enrichment_GO(gene_list,
                    go_mode='Bio',
                    organism='Human',
                    description='test_name',
                    outdir='enrichment_go',
                    cutoff=0.5):

    '''
    Gene enrichment analysis of GO

    Parameters
    ----------
    gene_list:list
        The gene set to be enrichment analyzed
    go_mode:str
        The module of GO include:'Bio','Cell','Mole'
    organism:str
        Select from (human, mouse, yeast, fly, fish, worm)
    description:str
        The title of enrichment
    outdir:str
        The savedir of enrichment
    cutoff:float
        Show enriched terms which Adjusted P-value < cutoff.

    Returns
    ----------
    result:pandas.DataFrame
        stores your last query
    '''
    if(go_mode=='Bio'):
        geneset='GO_Biological_Process_2018'
    if(go_mode=='Cell'):
        geneset='GO_Cellular_Component_2018'
    if(go_mode=='Mole'):
        geneset='GO_Molecular_Function_2018'
    enr = gp.enrichr(gene_list=gene_list,
                 gene_sets=geneset,
                 organism=organism, # don't forget to set organism to the one you desired! e.g. Yeast
                 description=description,
                 outdir=outdir,
                 # no_plot=True,
                 cutoff=cutoff # test dataset, use lower value from range(0,1)
                )
    subp=dotplot(enr.res2d, title=description,cmap='seismic')
    print(subp)
    return enr.res2d

def enrichment_GSEA(data,
                   gene_sets='KEGG_2016',
                   processes=4,
                   permutation_num=100,
                   outdir='prerank_report_kegg',
                   seed=6):
    '''
    Gene enrichment analysis of GSEA

    Parameters
    ----------
    data:pandas.DataFrame
        The result of Find_DEG(function in DeGene.py)
    gene_sets:list
        The gene_set of enrichr library
        Input Enrichr Libraries (https://maayanlab.cloud/Enrichr/#stats)
    processes:int
        CPU number
    permutation_num:int
        Number of permutations for significance computation. Default: 1000.
    outdir:str
        The savedir of enrichment
    seed:int
        Random seed

    Returns
    ----------
    result:Return a Prerank obj. 
    All results store to  a dictionary, obj.results,
         where contains::

             | {es: enrichment score,
             |  nes: normalized enrichment score,
             |  p: P-value,
             |  fdr: FDR,
             |  size: gene set size,
             |  matched_size: genes matched to the data,
             |  genes: gene names from the data set
             |  ledge_genes: leading edge genes}
    '''


    rnk=pd.DataFrame(columns=['genename','FoldChange'])
    rnk['genename']=data.index
    rnk['FoldChange']=data['FoldChange'].tolist()
    rnk1=rnk.drop_duplicates(['genename'])
    rnk1=rnk1.sort_values(by='FoldChange', ascending=False)
    
    pre_res = gp.prerank(rnk=rnk1, gene_sets=gene_sets,
                     processes=processes,
                     permutation_num=permutation_num, # reduce number to speed up testing
                     outdir=outdir, format='png', seed=seed)
    pre_res.res2d.sort_index().to_csv('GSEA_result.csv')
    return pre_res

def Plot_GSEA(data,num=0):

    '''
    Plot the GSEA result figure

    Parameters
    ----------
    data:prerank obj
        The result of enrichment_GSEA
    num:int
        The sequence of pathway drawn 
        Default:0(the first pathway)

    '''
    terms = data.res2d.index
    from gseapy.plot import gseaplot
    # to save your figure, make sure that ofname is not None
    gseaplot(rank_metric=data.ranking, term=terms[num], **data.results[terms[num]])
"""
def geneset_enrichment(gene_list:list,pathways_dict:dict,
                       pvalue_threshold:float=0.05,pvalue_type:str='auto',
                       organism:str='Human',description:str='None',
                       background:list=None,
                       outdir:str='./enrichr',cutoff:float=0.5)->pd.DataFrame:
    """
    Performs gene set enrichment analysis using Enrichr API.

    Arguments:
        gene_list: List of gene symbols to be tested for enrichment.
        pathways_dict: Dictionary of pathway library names and corresponding Enrichr API URLs.
        pvalue_threshold: P-value threshold for significant pathways. Default is 0.05.
        pvalue_type: Type of p-value correction to use. 'auto' uses Benjamini-Hochberg correction,for small gene sets (<500 genes) and Bonferroni correction for larger gene sets.,'bh' uses only Benjamini-Hochberg correction. 'bonferroni' uses only Bonferroni correction.,Default is 'auto'.
        organism: Organism of the input gene list. Default is 'Human'.
        description: Description of the input gene list. Default is 'None'.
        background: Background gene list to use for enrichment analysis. Default is None. If None, the background gene list is automatically set to the organism-specific gene list.
        outdir: Output directory for Enrichr results. Default is './enrichr'.
        cutoff: Show enriched terms which Adjusted P-value < cutoff. Default is 0.5.

    Returns:
        enrich_res: A pandas DataFrame containing the enrichment results.


    """
    import gseapy as gp
    if background is None:
        if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
            background='mmusculus_gene_ensembl'
        elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
            background='hsapiens_gene_ensembl'

    enr = gp.enrichr(gene_list=gene_list,
                 gene_sets=pathways_dict,
                 organism=organism, # don't forget to set organism to the one you desired! e.g. Yeast
                 description=description,
                 background=background,
                 outdir=outdir,
                 cutoff=cutoff # test dataset, use lower value from range(0,1)
                )
    if pvalue_type=='auto':
        if enr.res2d.shape[0]>100:
            enrich_res=enr.res2d[enr.res2d['Adjusted P-value']<pvalue_threshold]
            enrich_res['logp']=-np.log(enrich_res['Adjusted P-value'])
        else:
            enrich_res=enr.res2d[enr.res2d['P-value']<pvalue_threshold]
            enrich_res['logp']=-np.log(enrich_res['P-value'])
    elif pvalue_type=='adjust':
        enrich_res=enr.res2d[enr.res2d['Adjusted P-value']<pvalue_threshold]
        enrich_res['logp']=-np.log(enrich_res['Adjusted P-value'])
    else:
        enrich_res=enr.res2d[enr.res2d['P-value']<pvalue_threshold]
        enrich_res['logp']=-np.log(enrich_res['P-value'])
    enrich_res['logc']=np.log(enrich_res['Odds Ratio'])
    enrich_res['num']=[int(i.split('/')[0]) for i in enrich_res['Overlap']]
    enrich_res['fraction']=[int(i.split('/')[0])/int(i.split('/')[1]) for i in enrich_res['Overlap']]
    return enrich_res



def geneset_enrichment_GSEA(gene_rnk:pd.DataFrame,pathways_dict:dict,
                            processes:int=8,
                     permutation_num:int=100, # reduce number to speed up testing
                     outdir:str='./enrichr_gsea', format:str='png', seed:int=112)->dict:
    """
    Enrichment analysis using GSEA

    Arguments:
        gene_rnk: pre-ranked correlation table or pandas DataFrame. Same input with ``GSEA`` .rnk file.
        pathways_dict: Dictionary of pathway library names and corresponding Enrichr API URLs.
        processes: Number of Processes you are going to use. Default: 8.
        permutation_num: Number of permutations for significance computation. Default: 100.
        outdir: Output directory for Enrichr results. Default is './enrichr_gsea'.
        format: Matplotlib figure format. Default: 'png'.
        seed: Random seed. Default: 112.

    Returns:
        pre_res: A prerank object containing the enrichment results.
    
    """
    import gseapy as gp
    pre_res = gp.prerank(rnk=gene_rnk, gene_sets=pathways_dict,
                     processes=processes,
                     permutation_num=permutation_num, # reduce number to speed up testing
                     outdir=outdir, format=format, seed=seed)
    return pre_res
    enrich_res=pre_res.res2d[pre_res.res2d['fdr']<0.05]
    enrich_res['logp']=-np.log(enrich_res['fdr']+0.0001)
    enrich_res['logc']=enrich_res['nes']
    enrich_res['num']=enrich_res['matched_size']
    enrich_res['fraction']=enrich_res['matched_size']/enrich_res['geneset_size']
    enrich_res['Term']=enrich_res.index.tolist()
    enrich_res['P-value']=enrich_res['fdr']
    return enrich_res


def geneset_plot(enrich_res,num:int=10,node_size:list=[5,10,15],
                        cax_loc:int=2,cax_fontsize:int=12,
                        fig_title:str='',fig_xlabel:str='Fractions of genes',
                        figsize:tuple=(2,4),cmap:str='YlGnBu',
                        text_knock:int=2,text_maxsize:int=20)->matplotlib.axes._axes.Axes:
    """
    Plot the gene set enrichment result.

    Arguments:
        num: The number of enriched terms to plot. Default is 10.
        node_size: A list of integers defining the size of nodes in the plot. Default is [5,10,15].
        cax_loc: The location of the colorbar on the plot. Default is 2.
        cax_fontsize: The fontsize of the colorbar label. Default is 12.
        fig_title: The title of the plot. Default is an empty string.
        fig_xlabel: The label of the x-axis. Default is 'Fractions of genes'.
        figsize: The size of the plot. Default is (2,4).
        cmap: The colormap to use for the plot. Default is 'YlGnBu'.
        text_knock: The number of characters to knock off the end of the term name. Default is 2.
        text_maxsize: The maximum fontsize of the term names. Default is 20.

    Returns:
        A matplotlib.axes.Axes object.
    
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_data2=enrich_res.sort_values('P-value')[:num].sort_values('logc')
    st=ax.scatter(plot_data2['fraction'],range(len(plot_data2['logc'])),
            s=plot_data2['num']*10,linewidths=1,edgecolors='black',c=plot_data2['logp'],cmap=cmap)
    ax.yaxis.tick_right()
    plt.yticks(range(len(plot_data2['fraction'])),[plot_text_set(i.split('(')[0],text_knock=text_knock,text_maxsize=text_maxsize) for i in plot_data2['Term']],
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

class pyGSE(object):

    def __init__(self,gene_list:list,pathways_dict:dict,pvalue_threshold:float=0.05,pvalue_type:str='auto',
                 background=None,organism:str='Human',description:str='None',outdir:str='./enrichr',cutoff:float=0.5) -> None:
        """Initialize the pyGSEA class.

        Arguments:
            gene_list: A list of genes.
            pathways_dict: A dictionary of pathways.
            pvalue_threshold: The p-value threshold for enrichment. Default is 0.05.
            pvalue_type: The p-value type. Default is 'auto'.
            organism: The organism. Default is 'Human'.
            description: The description. Default is 'None'.
            outdir: The output directory. Default is './enrichr'.
            cutoff: The cutoff for enrichment. Default is 0.5.
        
        """

        self.gene_list=gene_list
        self.pathways_dict=pathways_dict
        self.pvalue_threshold=pvalue_threshold
        self.pvalue_type=pvalue_type
        self.organism=organism
        self.description=description
        self.outdir=outdir
        self.cutoff=cutoff
        if background is None:
            if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
                background='mmusculus_gene_ensembl'
            elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
                background='hsapiens_gene_ensembl'
            self.background=background
        else:
            self.background=background
    
    def enrichment(self):
        """gene set enrichment analysis.
        
        Returns:
            A pandas.DataFrame object containing the enrichment results.
        """

        enrich_res=geneset_enrichment(self.gene_list,self.pathways_dict,self.pvalue_threshold,self.pvalue_type,
                                  self.organism,self.description,self.background,self.outdir,self.cutoff)
        self.enrich_res=enrich_res
        return enrich_res
    
    
    def plot_enrichment(self,num:int=10,node_size:list=[5,10,15],
                        cax_loc:int=2,cax_fontsize:int=12,
                        fig_title:str='',fig_xlabel:str='Fractions of genes',
                        figsize:tuple=(2,4),cmap:str='YlGnBu',text_knock:int=2,text_maxsize:int=20)->matplotlib.axes._axes.Axes:
        
        """Plot the gene set enrichment result.
        
        Arguments:
            num: The number of enriched terms to plot. Default is 10.
            node_size: A list of integers defining the size of nodes in the plot. Default is [5,10,15].
            cax_loc: The location of the colorbar on the plot. Default is 2.
            cax_fontsize: The fontsize of the colorbar label. Default is 12.
            fig_title: The title of the plot. Default is an empty string.
            fig_xlabel: The label of the x-axis. Default is 'Fractions of genes'.
            figsize: The size of the plot. Default is (2,4).
            cmap: The colormap to use for the plot. Default is 'YlGnBu'.

        Returns:
            A matplotlib.axes.Axes object.
        """
        return geneset_plot(self.enrich_res,num,node_size,cax_loc,cax_fontsize,
                            fig_title,fig_xlabel,figsize,cmap,text_knock,text_maxsize)
    
class pyGSEA(object):

    def __init__(self,gene_rnk:pd.DataFrame,pathways_dict:dict,
                 processes:int=8,permutation_num:int=100,
                 outdir:str='./enrichr_gsea',cutoff:float=0.5) -> None:
        """Initialize the pyGSEA class.

        Arguments:
            gene_rnk: pre-ranked correlation table or pandas DataFrame. Same input with ``GSEA`` .rnk file.
            pathways_dict: Dictionary of pathway library names and corresponding Enrichr API URLs.
            processes: Number of Processes you are going to use. Default: 8.
            permutation_num: Number of permutations for significance computation. Default: 100.
            outdir: Output directory for Enrichr results. Default is './enrichr_gsea'.
            cutoff: The cutoff for enrichment. Default is 0.5.
        """

        self.gene_rnk=gene_rnk
        self.pathways_dict=pathways_dict
        self.processes=processes
        self.permutation_num=permutation_num
        self.outdir=outdir
        self.cutoff=cutoff
    
    
    def enrichment(self,format:str='png', pval=0.05,seed:int=112)->pd.DataFrame:
        """gene set enrichment analysis.
        
        Arguments:
            format: Matplotlib figure format. Default: 'png'.
            seed: Random seed. Default: 112.
        
        Returns:
            enrich_res:A pandas.DataFrame object containing the enrichment results.
        """

        
        pre_res=geneset_enrichment_GSEA(self.gene_rnk,self.pathways_dict,
                                           self.processes,self.permutation_num,
                                           self.outdir,format,seed)
        self.pre_res=pre_res
        enrich_res=pre_res.res2d[pre_res.res2d['fdr']<pval]
        enrich_res['logp']=-np.log(enrich_res['fdr']+0.0001)
        enrich_res['logc']=enrich_res['nes']
        enrich_res['num']=enrich_res['matched_size']
        enrich_res['fraction']=enrich_res['matched_size']/enrich_res['geneset_size']
        enrich_res['Term']=enrich_res.index.tolist()
        enrich_res['P-value']=enrich_res['fdr']
        self.enrich_res=enrich_res
        return enrich_res
    
    def plot_gsea(self,term_num:int=0,
                  gene_set_title:str='',
                  figsize:tuple=(3,4),
                  cmap:str='RdBu_r',
                  title_fontsize:int=12,
                  title_y:float=0.95)->matplotlib.figure.Figure:
        """Plot the gene set enrichment result.
        
        Arguments:
            term_num: The number of enriched terms to plot. Default is 0.
            gene_set_title: The title of the plot. Default is an empty string.
            figsize: The size of the plot. Default is (3,4).
            cmap: The colormap to use for the plot. Default is 'RdBu_r'.
            title_fontsize: The fontsize of the title. Default is 12.
            title_y: The y coordinate of the title. Default is 0.95.

        Returns:
            fig: A matplotlib.figure.Figure object.
        """
        from gseapy.plot import GSEAPlot
        terms = self.enrich_res.index
        g = GSEAPlot(
        rank_metric=self.pre_res.ranking, term=terms[term_num],figsize=figsize,cmap=cmap,
            **self.pre_res.results[terms[term_num]]
            )
        if gene_set_title=='':
            g.fig.suptitle(terms[term_num],fontsize=title_fontsize,y=title_y)
        else:
            g.fig.suptitle(gene_set_title,fontsize=title_fontsize,y=title_y)
        g.add_axes()
        return g.fig
    
    
    def plot_enrichment(self,num:int=10,node_size:list=[5,10,15],
                        cax_loc:int=2,cax_fontsize:int=12,
                        fig_title:str='',fig_xlabel:str='Fractions of genes',
                        figsize:tuple=(2,4),cmap:str='YlGnBu',
                        text_knock:int=2,text_maxsize:int=20)->matplotlib.axes._axes.Axes:
        
        """Plot the gene set enrichment result.
        
        Arguments:
            num: The number of enriched terms to plot. Default is 10.
            node_size: A list of integers defining the size of nodes in the plot. Default is [5,10,15].
            cax_loc: The location of the colorbar on the plot. Default is 2.
            cax_fontsize: The fontsize of the colorbar label. Default is 12.
            fig_title: The title of the plot. Default is an empty string.
            fig_xlabel: The label of the x-axis. Default is 'Fractions of genes'.
            figsize: The size of the plot. Default is (2,4).
            cmap: The colormap to use for the plot. Default is 'YlGnBu'.
            text_knock: The number of terms to knock out for text labels. Default is 2.
            text_maxsize: The maximum fontsize of text labels. Default is 20.

        Returns:
            ax: A matplotlib.axes.Axes object.
        """
        return geneset_plot(self.enrich_res,num,node_size,cax_loc,cax_fontsize,
                            fig_title,fig_xlabel,figsize,cmap,text_knock,text_maxsize)