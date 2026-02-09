import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import networkx as nx
import seaborn as sns
import pandas as pd
from sklearn import decomposition as skldec
from typing import Optional, Any, Dict, List, Tuple

from ..utils import plot_text_set
from .._registry import register_function
import matplotlib

@register_function(
    aliases=["基因集富集", "geneset_enrichment", "enrichr_analysis", "pathway_enrichment", "富集分析"],
    category="bulk",
    description="Perform gene set enrichment analysis. IMPORTANT: pathways_dict must be a dictionary loaded via ov.utils.geneset_prepare(), NOT a file path string!",
    prerequisites={
        'optional_functions': ['download_pathway_database', 'geneset_prepare']
    },
    examples=[
        "# STEP 1: Download pathway database (run once)",
        "ov.utils.download_pathway_database()",
        "",
        "# STEP 2: Load geneset into dictionary - REQUIRED!",
        "pathways_dict = ov.utils.geneset_prepare('genesets/GO_Biological_Process_2021.txt', organism='Human')",
        "",
        "# STEP 3: Run enrichment with the DICTIONARY (NOT file path!)",
        "enr = ov.bulk.geneset_enrichment(",
        "    gene_list=deg_genes,",
        "    pathways_dict=pathways_dict,  # Must be dict, NOT string path!",
        "    pvalue_type='auto',",
        "    organism='Human'",
        ")",
        "",
        "# WRONG - DO NOT DO THIS:",
        "# enr = ov.bulk.geneset_enrichment(gene_list=genes, pathways_dict='file.gmt')  # ERROR!"
    ],
    related=["utils.geneset_prepare", "utils.download_pathway_database", "bulk.geneset_plot", "bulk.pyGSEA"]
)
def geneset_enrichment(gene_list:list,pathways_dict:dict,
                       pvalue_threshold:float=0.05,pvalue_type:str='auto',
                       organism:str='Human',description:str='None',
                       background:list=None,
                       outdir:str='./enrichr',cutoff:float=0.5)->pd.DataFrame:
    r"""Perform gene set enrichment analysis using Enrichr API.

    Arguments:
        gene_list: List of gene symbols to be tested for enrichment.
        pathways_dict: Dictionary of pathway library names and corresponding Enrichr API URLs.
        pvalue_threshold: P-value threshold for significant pathways. (0.05)
        pvalue_type: Type of p-value correction to use. 'auto' uses Benjamini-Hochberg correction for small gene sets (<500 genes) and Bonferroni correction for larger gene sets. 'bh' uses only Benjamini-Hochberg correction. 'bonferroni' uses only Bonferroni correction. ('auto')
        organism: Organism of the input gene list. ('Human')
        description: Description of the input gene list. ('None')
        background: Background gene list to use for enrichment analysis. If None, the background gene list is automatically set to the organism-specific gene list. (None)
        outdir: Output directory for Enrichr results. ('./enrichr')
        cutoff: Show enriched terms which Adjusted P-value < cutoff. (0.5)

    Returns:
        enrich_res: A pandas DataFrame containing the enrichment results.

    """
    from ..external.gseapy import enrichr
    #import gseapy as gp
    if background is None:
        if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
            background='mmusculus_gene_ensembl'
        elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
            background='hsapiens_gene_ensembl'

    enr = enrichr(gene_list=gene_list,
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

def enrichment_multi_concat(enr_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    def process_df(df: pd.DataFrame, term_col_name: str) -> pd.DataFrame:
        new_data = []
        for _, row in df.iterrows():
            genes = row['Genes'].split(';')
            for gene in genes:
                new_data.append({
                    'Gene': gene,
                    term_col_name: row['Term'],

                })
        return pd.DataFrame(new_data)
    new_dict={}
    new_li=[]
    for key in enr_dict.keys():
        new_dict[key]=process_df(enr_dict[key], key)
        new_li.append(new_dict[key])
    # 合并两个DataFrame
    merged_df = pd.concat(new_li, ignore_index=True).fillna('')
    #return merged_df

    #print(dict(zip(enr_dict.keys(),
    #        [lambda x: '|'.join(x.dropna().unique()) for i in range(len(enr_dict.keys()))])))
    
    # 按基因分组，并将相同基因的Term合并
    result_df = merged_df.groupby('Gene').agg(dict(zip(enr_dict.keys(),
            [lambda x: '|'.join(x.dropna().unique()) for i in range(len(enr_dict.keys()))]))
                                             ).reset_index()
    return result_df



def geneset_enrichment_GSEA(gene_rnk:pd.DataFrame,pathways_dict:dict,
                            processes:int=8,
                     permutation_num:int=100, # reduce number to speed up testing
                     outdir:str='./enrichr_gsea', format:str='png', seed:int=112)->dict:
    r"""Enrichment analysis using GSEA.

    Arguments:
        gene_rnk: Pre-ranked correlation table or pandas DataFrame. Same input with ``GSEA`` .rnk file.
        pathways_dict: Dictionary of pathway library names and corresponding Enrichr API URLs.
        processes: Number of Processes you are going to use. (8)
        permutation_num: Number of permutations for significance computation. (100)
        outdir: Output directory for Enrichr results. ('./enrichr_gsea')
        format: Matplotlib figure format. ('png')
        seed: Random seed. (112)

    Returns:
        pre_res: A prerank object containing the enrichment results.
    
    """
    from ..external.gseapy import prerank
    pre_res = prerank(rnk=gene_rnk, gene_sets=pathways_dict,
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

def geneset_plot_multi(enr_dict: Dict[str, pd.DataFrame], colors_dict: Dict[str, str], num: int = 5, fontsize: int = 10,
                        fig_title: str = '', fig_xlabel: str = 'Fractions of genes',
                        figsize: tuple = (2, 4), cmap: str = 'YlGnBu',
                        text_knock: int = 5, text_maxsize: int = 20, ax: Optional[matplotlib.axes._axes.Axes] = None
                        ) -> matplotlib.axes._axes.Axes:
    r"""Enrichment multi genesets analysis using GSEA.

    Arguments:
        enr_dict: A dictionary of enrichment results.
        colors_dict: A dictionary of colors for each gene set.
        num: The number of enriched terms to plot. (5)
        fontsize: The fontsize of the plot. (10)
        fig_title: The title of the plot. ('')
        fig_xlabel: The label of the x-axis. ('Fractions of genes')
        figsize: The size of the plot. ((2,4))
        cmap: The colormap to use for the plot. ('YlGnBu')
        text_knock: The number of characters to knock off the end of the term name. (5)
        text_maxsize: The maximum fontsize of the term names. (20)
        ax: A matplotlib.axes.Axes object. (None)

    Returns:
        ax: The matplotlib axes object

    """
    from PyComplexHeatmap import HeatmapAnnotation,DotClustermapPlotter,anno_label,anno_simple,AnnotationBase
    for key in enr_dict.keys():
        enr_dict[key]['Type']=key
    enr_all=pd.concat([enr_dict[i].iloc[:num] for i in enr_dict.keys()],axis=0)
    enr_all['Term']=[plot_text_set(i.split('(')[0],text_knock=text_knock,text_maxsize=text_maxsize) for i in enr_all.Term.tolist()]
    enr_all.index=enr_all.Term
    enr_all = enr_all.loc[~enr_all.index.duplicated(keep='first')]  # some GO term exist in multi category(BP/CC/MF)
    enr_all['Term1']=[i for i in enr_all.index.tolist()]
    del enr_all['Term']

    colors=colors_dict

    left_ha = HeatmapAnnotation(
                          label=anno_label(enr_all.Type, merge=True,rotation=0,colors=colors,relpos=(1,0.8)),
                          Category=anno_simple(enr_all.Type,cmap='Set1',
                                           add_text=False,legend=False,colors=colors),
                           axis=0,verbose=0,label_kws={'rotation':45,'horizontalalignment':'left','visible':False})
    right_ha = HeatmapAnnotation(
                              label=anno_label(enr_all.Term1, merge=True,rotation=0,relpos=(0,0.5),arrowprops=dict(visible=True),
                                               colors=enr_all.assign(color=enr_all.Type.map(colors)).set_index('Term1').color.to_dict(),
                                              fontsize=fontsize,luminance=0.8,height=2),
                               axis=0,verbose=0,#label_kws={'rotation':45,'horizontalalignment':'left'},
                                orientation='right')
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize) 
    else:
        ax=ax
    #plt.figure(figsize=figsize)
    cm = DotClustermapPlotter(data=enr_all, x='fraction',y='Term1',value='logp',c='logp',s='num',
                              cmap=cmap,
                              row_cluster=True,#col_cluster=True,#hue='Group',
                              #cmap={'Group1':'Greens','Group2':'OrRd'},
                              vmin=-1*np.log10(0.1),vmax=-1*np.log10(1e-10),
                              #colors={'Group1':'yellowgreen','Group2':'orange'},
                              #marker={'Group1':'*','Group2':'$\\ast$'},
                              show_rownames=True,show_colnames=False,row_dendrogram=False,
                              col_names_side='top',row_names_side='right',
                              xticklabels_kws={'labelrotation': 30, 'labelcolor': 'blue','labelsize':fontsize},
                              #yticklabels_kws={'labelsize':10},
                              #top_annotation=col_ha,left_annotation=left_ha,right_annotation=right_ha,
                              left_annotation=left_ha,right_annotation=right_ha,
                              spines=False,
                              row_split=enr_all.Type,# row_split_gap=1,
                              #col_split=df_col.Group,col_split_gap=0.5,
                              verbose=1,legend_gap=10,
                              #dot_legend_marker='*',
                              xlabel='Fractions of genes',xlabel_side="bottom",
                              xlabel_kws=dict(labelpad=8,fontweight='normal',fontsize=fontsize+2),
                              # xlabel_bbox_kws=dict(facecolor=facecolor)
                             )
    tesr=plt.gcf().axes
    for ax in plt.gcf().axes:
        if hasattr(ax, 'get_xlabel'):
            if ax.get_xlabel() == 'Fractions of genes':  # 假设 colorbar 有一个特定的标签
                cbar = ax
                cbar.grid(False)
            if ax.get_ylabel() == 'logp':  # 假设 colorbar 有一个特定的标签
                cbar = ax
                cbar.tick_params(labelsize=fontsize+2)
                cbar.set_ylabel(r'$−Log_{10}(P_{adjusted})$',fontsize=fontsize+2)
                cbar.grid(False)
    return ax

@register_function(
    aliases=["富集分析可视化", "geneset_plot", "enrichment_plot", "通路富集图", "pathway_plot"],
    category="bulk",
    description="Visualize gene set enrichment analysis results with bubble plot",
    examples=[
        "# Basic usage",
        "ov.bulk.geneset_plot(enrich_res, num=10)",
        "# Custom appearance",
        "ov.bulk.geneset_plot(enrich_res, num=15, figsize=(3,5), cmap='RdBu')",
        "# Adjust node sizes and colors",
        "ov.bulk.geneset_plot(enrich_res, node_size=[10,20,30], fig_title='KEGG Pathways')"
    ],
    related=["bulk.geneset_enrichment", "bulk.geneset_plot_multi", "pl.volcano", "pl.dotplot"]
)
def geneset_plot(enrich_res: pd.DataFrame, num: int = 10, node_size: list = [5, 10, 15],
                        cax_loc: list = [2, 0.55, 0.5, 0.02], cax_fontsize: int = 12,
                        fig_title: str = '', fig_xlabel: str = 'Fractions of genes',
                        figsize: tuple = (2, 4), cmap: str = 'YlGnBu',
                        text_knock: int = 5, text_maxsize: int = 20,
                        bbox_to_anchor_used: tuple = (-0.45, -13), node_diameter: int = 10,
                        custom_ticks: list = [5, 10], ax: Optional[matplotlib.axes._axes.Axes] = None) -> matplotlib.axes._axes.Axes:
    r"""Plot the gene set enrichment result.

    Arguments:
        enrich_res: Enrichment results DataFrame.
        num: The number of enriched terms to plot. Default: 10.
        node_size: A list of integers defining the size of nodes in the plot. Default: [5,10,15].
        cax_loc: The location, width and height of the colorbar on the plot. Default: [2, 0.55, 0.5, 0.02].
        cax_fontsize: The fontsize of the colorbar label. Default: 12.
        fig_title: The title of the plot. Default: ''.
        fig_xlabel: The label of the x-axis. Default: 'Fractions of genes'.
        figsize: The size of the plot. Default: (2,4).
        cmap: The colormap to use for the plot. Default: 'YlGnBu'.
        text_knock: The number of characters to knock off the end of the term name. Default: 5.
        text_maxsize: The maximum fontsize of the term names. Default: 20.
        bbox_to_anchor_used: The anchor point for placing the legend. Default: (-0.45, -13).
        node_diameter: The base size for nodes in the plot. Default: 10.
        custom_ticks: Custom tick marks for the plot. Default: [5,10].
        ax: Matplotlib axes object. Default: None.

    Returns:
        ax: A matplotlib.axes.Axes object containing the enrichment bubble plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    plot_data2=enrich_res.sort_values('P-value')[:num].sort_values('logc')
    st=ax.scatter(plot_data2['fraction'],range(len(plot_data2['logc'])),
            s=plot_data2['num']*node_diameter,linewidths=1,edgecolors='black',c=plot_data2['logp'],cmap=cmap)
    ax.yaxis.tick_right()
    plt.yticks(range(len(plot_data2['fraction'])),[plot_text_set(i.split('(')[0],text_knock=text_knock,text_maxsize=text_maxsize) for i in plot_data2['Term']],
            fontsize=10,)
    plt.xticks(fontsize=12,)
    plt.title(fig_title,fontsize=12)
    plt.xlabel(fig_xlabel,fontsize=12)

    fig = plt.gcf()
    cax = fig.add_axes(cax_loc)
    cb=fig.colorbar(st,shrink=0.25,cax=cax,orientation='horizontal')
    cb.set_label(r'$−Log_{10}(P_{adjusted})$',fontdict={'size':cax_fontsize})
    # new code to add custom ticks
    cb.set_ticks(custom_ticks)

    gl_li=[]
    for i in node_size:
        gl_li.append(ax.scatter([],[], s=i*node_diameter, marker='o', color='white',edgecolors='black'))

    plt.legend(gl_li,
        [str(i) for i in node_size],
        loc='lower left',
        ncol=3,bbox_to_anchor=bbox_to_anchor_used,
        fontsize=cax_fontsize)
    return ax

class pyGSE(object):

    def __init__(self,gene_list:list,pathways_dict:dict,pvalue_threshold:float=0.05,pvalue_type:str='auto',
                 background=None,organism:str='Human',description:str='None',outdir:str='./enrichr',cutoff:float=0.5) -> None:
        """Initialize the pyGSE class.

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
    
@register_function(
    aliases=["GSEA分析", "pyGSEA", "gene_set_enrichment", "基因集富集分析"],
    category="bulk",
    description="Gene Set Enrichment Analysis (GSEA) for ranked gene lists",
    examples=[
        "# Initialize GSEA object",
        "gsea_obj = ov.bulk.pyGSEA(ranked_genes, pathway_dict)",
        "# Run enrichment analysis",
        "enrich_res = gsea_obj.enrichment()",
        "# Visualize enrichment results",
        "gsea_obj.plot_enrichment(num=10, figsize=(3,5))",
        "# Plot GSEA for specific term",
        "gsea_obj.plot_gsea(term_num=0, gene_set_title='KEGG Pathway')"
    ],
    related=["bulk.geneset_enrichment", "bulk.pyDEG.ranking2gsea", "utils.geneset_prepare"]
)
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
        from ..external.gseapy.plot import GSEAPlot
        #from gseapy.plot import GSEAPlot
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