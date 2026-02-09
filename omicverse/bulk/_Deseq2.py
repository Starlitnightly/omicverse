r"""
Different Expression analysis in Python
"""

import numpy as np
import pandas as pd

import statsmodels.api as sm
import anndata as ad
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
from typing import Union, Tuple, Optional, Any
from ..utils import plot_boxplot
from .._registry import register_function
from ..pl import volcano

@register_function(
    aliases=["基因ID映射", "gene_id_mapping", "id_mapping", "基因符号转换", "gene_symbol_mapping"],
    category="bulk",
    description="Map gene IDs to gene symbols using a reference table for bulk RNA-seq data",
    examples=[
        "ov.bulk.Matrix_ID_mapping(data, gene_ref_path='gene_reference.txt')",
        "ov.bulk.Matrix_ID_mapping(data, gene_ref_path='gene_ref.tsv', keep_unmapped=False)"
    ],
    related=["bulk.deseq2_normalize", "utils.gene_symbol_to_ensembl", "pp.filter_genes"]
)
def Matrix_ID_mapping(data:pd.DataFrame,gene_ref_path:str,keep_unmapped:bool=True)->pd.DataFrame:
    r"""Map gene IDs in the input data to gene symbols using a reference table.

    Arguments:
        data: The input data containing gene IDs as index.
        gene_ref_path: The path to the reference table containing the mapping from gene IDs to gene symbols.
        keep_unmapped: Whether to keep genes that are not found in the mapping table. If True, unmapped genes retain their original IDs. If False, unmapped genes are removed (original behavior). Default: True.

    Returns:
        data: The input data with gene IDs mapped to gene symbols.

    """
    
    pair=pd.read_csv(gene_ref_path,sep='\t',index_col=0)
    
    if keep_unmapped:
        # Keep all genes, map those that exist in the reference
        all_genes = data.index.tolist()
        mapped_genes = list(set(all_genes) & set(pair.index.tolist()))
        unmapped_genes = list(set(all_genes) - set(pair.index.tolist()))
        
        new_index = []
        
        # Process mapped genes
        for gene in all_genes:
            if gene in pair.index:
                symbol = pair.loc[gene, 'symbol']
                if str(symbol) == 'nan':
                    new_index.append(gene)  # Keep original ID if symbol is NaN
                else:
                    new_index.append(symbol)
            else:
                new_index.append(gene)  # Keep original ID for unmapped genes
        
        data.index = new_index
        print(f"......Mapped {len(mapped_genes)} genes to symbols, kept {len(unmapped_genes)} unmapped genes with original IDs")
    else:
        # Original behavior: only keep genes that can be mapped
        original_genes = data.index.tolist()
        ret_gene=list(set(original_genes) & set(pair.index.tolist()))
        data=data.loc[ret_gene]
        new_index=[]
        for i in ret_gene:
            a=pair.loc[i,'symbol']
            if str(a)=='nan':
                new_index.append(i)
            else:
                new_index.append(a)
        data.index=new_index
        print(f"......Mapped {len(ret_gene)} genes to symbols, removed {len(original_genes) - len(ret_gene)} unmapped genes")
    
    return data


def deseq2_normalize(data:pd.DataFrame)->pd.DataFrame:
    r"""Normalize the data using DESeq2 method.

    Arguments:
        data: The data to be normalized.
    
    Returns:
        data: The normalized data.

    """
    avg1=data.apply(np.log,axis=1).mean(axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    data1=data.loc[avg1.index]
    data_log=data1.apply(np.log,axis=1)
    scale=data_log.sub(avg1.values,axis=0).median(axis=0).apply(np.exp)
    return data/scale


def normalize_bulk(df_counts: pd.DataFrame, df_lengths: pd.DataFrame, normalization_type: str) -> pd.DataFrame:
    r"""Normalize the count data.

    Arguments:
        df_counts: Gene expression count matrix (number of cells x number of genes).
        df_lengths: Vector of gene lengths.
        normalization_type: Type of normalization (e.g., 'CPM', 'TPM', 'FPKM', 'RPKM').

    Returns:
        normalized_data: Normalized data as DataFrame
    """
    counts = df_counts.values
    lengths = df_lengths['feature_length'].astype(float).values.reshape(1, -1)  # Ensure lengths is a column vector
    
    if normalization_type == 'CPM':
        # Counts Per Million
        counts_per_million = counts / counts.sum(axis=1, keepdims=True) * 1e6
        return pd.DataFrame(counts_per_million, index=df_counts.index, columns=df_counts.columns)
    
    elif normalization_type == 'TPM':
        # Transcripts Per Million
        rate = counts / lengths
        tpm = rate / rate.sum(axis=1, keepdims=True) * 1e6
        return pd.DataFrame(tpm, index=df_counts.index, columns=df_counts.columns)
    
    elif normalization_type == 'FPKM' or normalization_type == 'RPKM':
        # Fragments Per Kilobase of transcript per Million mapped reads
        total_counts = counts.sum(axis=1, keepdims=True)
        fpkm = (counts / lengths) / total_counts * 1e9
        return pd.DataFrame(fpkm, index=df_counts.index, columns=df_counts.columns)
    
    else:
        raise ValueError("Unsupported normalization type. Choose from 'CPM', 'TPM', 'FPKM', 'RPKM'.")


def estimateSizeFactors(data:pd.DataFrame)->pd.Series:
    r"""Estimate size factors for data normalization.

    Arguments:
        data: A pandas DataFrame of gene expression data where rows correspond to samples and columns correspond to genes.
    
    Returns:
        scale: A pandas Series of size factors, one for each sample.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> import omicverse as ov
        >>> data = pd.DataFrame(np.random.rand(100, 10), columns=list('abcdefghij'))
        >>> size_factors = ov.bulk.estimateSizeFactors(data)
    """
    avg1=data.apply(np.log,axis=1).mean(axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    data1=data.loc[avg1.index]
    data_log=data1.apply(np.log,axis=1)
    scale=data_log.sub(avg1.values,axis=0).median(axis=0).apply(np.exp)
    return scale


def estimateDispersions(counts:pd.DataFrame)->pd.Series:
    r"""Estimate the dispersion parameter of the Negative Binomial distribution for each gene in the input count matrix.

    Arguments:
        counts: Input count matrix with shape (n_genes, n_samples).

    Returns:
        disp: Array of dispersion values for each gene in the input count matrix.
    """
    # Step 1: Calculate mean and variance of counts for each gene
    mean_counts = np.mean(counts, axis=1)
    var_counts = np.var(counts, axis=1)
    
    # Step 2: Fit trend line to variance-mean relationship using GLM
    mean_expr = sm.add_constant(np.log(mean_counts))
    mod = sm.GLM(np.log(var_counts), mean_expr, family=sm.families.Gamma())
    res = mod.fit()
    fitted_var = np.exp(res.fittedvalues)
    
    # Step 3: Calculate residual variance for each gene
    disp = fitted_var / var_counts
    
    return disp

def data_drop_duplicates_index(data:pd.DataFrame)->pd.DataFrame:
    r"""Drop the duplicated index of data.

    Arguments:
        data: The data to be processed.

    Returns:
        data: The data after dropping the duplicated index.
    """
    # Sort the data by the sum of counts in descending order
    data = data.loc[data.sum(axis=1).sort_values(ascending=False).index]
    
    # Drop duplicates, keeping the first occurrence (which is the highest due to sorting)
    data = data.loc[~data.index.duplicated(keep='first')]
    return data

@register_function(
    aliases=["差异表达分析", "DEG", "differential_expression", "差异基因分析", "pyDEG"],
    category="bulk", 
    description="Python implementation of differential expression analysis for bulk RNA-seq data",
    examples=[
        "# Initialize with raw count data",
        "dds = ov.bulk.pyDEG(raw_count_data)",
        "# Remove duplicate gene IDs",
        "dds.drop_duplicates_index()",
        "# Normalize using DESeq2 method",
        "dds.normalize()",
        "# Perform differential expression analysis",
        "dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')",
        "# Set fold change thresholds",
        "dds.foldchange_set(fc_threshold=2, pval_threshold=0.05)",
        "# Visualize results",
        "dds.plot_volcano(title='DEG Analysis')",
        "dds.plot_boxplot(genes=['GENE1', 'GENE2'], treatment_groups, control_groups)",
        "# Prepare ranking for GSEA",
        "ranked_genes = dds.ranking2gsea()"
    ],
    related=["bulk.deseq2_normalize", "single.rank_genes_groups", "utils.volcano_plot"]
)
class pyDEG(object):


    def __init__(self,raw_data:pd.DataFrame) -> None:
        r"""Initialize the pyDEG class.

        Arguments:
            raw_data: The raw data to be processed.
        
        Returns:
            None
        """
        self.raw_data=raw_data
        self.data=raw_data.copy()
        
    def drop_duplicates_index(self)->pd.DataFrame:
        r"""Drop the duplicated index of data.

        Returns:
            data: The data after dropping the duplicated index.
        """
        self.data=data_drop_duplicates_index(self.data)
        return self.data

    def normalize(self)->pd.DataFrame:
        r"""Normalize the data using DESeq2 method.
        
        Returns:
            data: The normalized data.
        """
        self.size_factors=estimateSizeFactors(self.data)
        self.data=deseq2_normalize(self.data)
        return self.data
    
    def foldchange_set(self, fc_threshold: int = -1, pval_threshold: float = 0.05, logp_max: int = 6, fold_threshold: int = 0) -> None:
        r"""Set fold-change and p-value thresholds to classify differentially expressed genes as up-regulated, down-regulated, or not significant.

        Arguments:
            fc_threshold: Absolute fold-change threshold. If set to -1, the threshold is calculated based on the histogram of log2 fold-changes. (-1)
            pval_threshold: p-value threshold for determining significance. (0.05)
            logp_max: Maximum value for log-transformed p-values. (6)
            fold_threshold: Index of the histogram bin corresponding to the fold-change threshold (only applicable if fc_threshold=-1). (0)

        Returns:
            None
        """
        if fc_threshold==-1:
            foldp=np.histogram(self.result['log2FC'].dropna())
            foldchange=(foldp[1][np.where(foldp[1]>0)[0][fold_threshold]]+foldp[1][np.where(foldp[1]>0)[0][fold_threshold+1]])/2
        else:
            foldchange=fc_threshold
        print('... Fold change threshold: %s'%foldchange)
        fc_max,fc_min=foldchange,0-foldchange
        self.fc_max,self.fc_min=fc_max,fc_min
        self.pval_threshold=pval_threshold
        self.result['sig']='normal'
        self.result.loc[((self.result['log2FC']>fc_max)&(self.result['qvalue']<pval_threshold)),'sig']='up'
        self.result.loc[((self.result['log2FC']<fc_min)&(self.result['qvalue']<pval_threshold)),'sig']='down'
        self.result.loc[self.result['-log(qvalue)']>logp_max,'-log(qvalue)']=logp_max
        self.logp_max=logp_max
    

    def plot_volcano(self, figsize: tuple = (4, 4), pval_name: str = 'qvalue', fc_name: str = 'log2FC',
                     title: str = '', titlefont: dict = {'weight': 'normal', 'size': 14},
                     up_color: str = '#e25d5d', down_color: str = '#7388c1', normal_color: str = '#d7d7d7',
                     up_fontcolor: str = '#e25d5d', down_fontcolor: str = '#7388c1', normal_fontcolor: str = '#d7d7d7',
                     legend_bbox: tuple = (0.8, -0.2), legend_ncol: int = 2, legend_fontsize: int = 12,
                     plot_genes: Optional[list] = None, plot_genes_num: int = 10, plot_genes_fontsize: int = 10,
                     ticks_fontsize: int = 12, ax: Optional[matplotlib.axes._axes.Axes] = None) -> matplotlib.axes._axes.Axes:
        r"""Generate a volcano plot for the differential gene expression analysis results.

        Arguments:
            figsize: The size of the generated figure. ((4,4))
            pval_name: Column name for p-values. ('qvalue')
            fc_name: Column name for fold changes. ('log2FC')
            title: The title of the plot. ('')
            titlefont: A dictionary of font properties for the plot title. ({'weight':'normal','size':14,})
            up_color: The color of the up-regulated genes in the plot. ('#e25d5d')
            down_color: The color of the down-regulated genes in the plot. ('#7388c1')
            normal_color: The color of the non-significant genes in the plot. ('#d7d7d7')
            up_fontcolor: Font color for up-regulated gene labels. ('#e25d5d')
            down_fontcolor: Font color for down-regulated gene labels. ('#7388c1')
            normal_fontcolor: Font color for normal gene labels. ('#d7d7d7')
            legend_bbox: A tuple containing the coordinates of the legend's bounding box. ((0.8, -0.2))
            legend_ncol: The number of columns in the legend. (2)
            legend_fontsize: The font size of the legend. (12)
            plot_genes: A list of genes to be plotted on the volcano plot. (None)
            plot_genes_num: The number of genes to be plotted on the volcano plot. (10)
            plot_genes_fontsize: The font size of the genes to be plotted on the volcano plot. (10)
            ticks_fontsize: The font size of the ticks. (12)
            ax: Matplotlib axis object. (None)

        Returns:
            ax: The generated volcano plot.

        """
        
        ax=volcano(result=self.result,pval_name=pval_name,fc_name=fc_name,pval_max=self.logp_max,
                       figsize=figsize,title=title,titlefont=titlefont,
                       up_color=up_color,down_color=down_color,normal_color=normal_color,
                       up_fontcolor=up_fontcolor,down_fontcolor=down_fontcolor,normal_fontcolor=normal_fontcolor,
                       legend_bbox=legend_bbox,legend_ncol=legend_ncol,legend_fontsize=legend_fontsize,plot_genes=plot_genes,
                       plot_genes_num=plot_genes_num,plot_genes_fontsize=plot_genes_fontsize,
                       ticks_fontsize=ticks_fontsize,ax=ax,
                       pval_threshold=self.pval_threshold,fc_max=self.fc_max,fc_min=self.fc_min)
        return ax
        '''
        fig, ax = plt.subplots(figsize=figsize)
        result=self.result.copy()
        #首先绘制正常基因
        ax.scatter(x=result[result['sig']=='normal']['log2FC'],
                y=result[result['sig']=='normal']['-log(qvalue)'],
                color=normal_color,#颜色
                alpha=.5,#透明度
                )
        #接着绘制上调基因
        ax.scatter(x=result[result['sig']=='up']['log2FC'],
                y=result[result['sig']=='up']['-log(qvalue)'],
                color=up_color,#选择色卡第15个颜色
                alpha=.5,#透明度
                )
        #绘制下调基因
        ax.scatter(x=result[result['sig']=='down']['log2FC'],
                y=result[result['sig']=='down']['-log(qvalue)'],
                color=down_color,#颜色
                alpha=.5,#透明度
                )
        
        ax.plot([result['log2FC'].min(),result['log2FC'].max()],#辅助线的x值起点与终点
                [-np.log10(self.pval_threshold),-np.log10(self.pval_threshold)],#辅助线的y值起点与终点
                linewidth=2,#辅助线的宽度
                linestyle="--",#辅助线类型：虚线
                color='black'#辅助线的颜色
        )
        ax.plot([self.fc_max,self.fc_max],
                [result['-log(qvalue)'].min(),result['-log(qvalue)'].max()],
                linewidth=2, 
                linestyle="--",
                color='black')
        ax.plot([self.fc_min,self.fc_min],
                [result['-log(qvalue)'].min(),result['-log(qvalue)'].max()],
                linewidth=2, 
                linestyle="--",
                color='black')
        #设置横标签与纵标签
        ax.set_ylabel(r'$-log_{10}(qvalue)$',titlefont)                                    
        ax.set_xlabel(r'$log_{2}FC$',titlefont)
        #设置标题
        ax.set_title(title,titlefont)

        #绘制图注
        #legend标签列表，上面的color即是颜色列表
        labels = ['up:{0}'.format(len(result[result['sig']=='up'])),
                'down:{0}'.format(len(result[result['sig']=='down']))]  
        #用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
        color = [up_color,down_color]
        patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i]) ) for i in range(len(color))] 

        ax.legend(handles=patches,
            bbox_to_anchor=legend_bbox, 
            ncol=legend_ncol,
            fontsize=legend_fontsize)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        from adjustText import adjust_text
        import adjustText
        
        if plot_genes is not None:
            hub_gene=plot_genes
        else:
            up_result=result.loc[result['sig']=='up']
            down_result=result.loc[result['sig']=='down']
            hub_gene=up_result.sort_values('qvalue').index[:plot_genes_num//2].tolist()+down_result.sort_values('qvalue').index[:plot_genes_num//2].tolist()

        color_dict={
        'up':up_fontcolor,
            'down':down_fontcolor,
            'normal':normal_fontcolor
        }

        texts=[ax.text(result.loc[i,'log2FC'], 
               result.loc[i,'-log(qvalue)'],
               i,
               fontdict={'size':plot_genes_fontsize,'weight':'bold','color':color_dict[result.loc[i,'sig']]}
               ) for i in hub_gene]
        
        if adjustText.__version__<='0.8':
            adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)
        else:
            adjust_text(texts,only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                        arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_xticks([round(i,2) for i in ax.get_xticks()[1:-1]],#获取x坐标轴内容
              [round(i,2) for i in ax.get_xticks()[1:-1]],#更新x坐标轴内容
              fontsize=ticks_fontsize,
              fontweight='normal'
              )
        return fig,ax
        '''
    
    def plot_boxplot(self, genes: list, treatment_groups: list, control_groups: list,
                     log: bool = True,
                     treatment_name: str = 'Treatment', control_name: str = 'Control',
                     figsize: tuple = (4, 3), palette: list = ["#a64d79", "#674ea7"],
                     title: str = 'Gene Expression', fontsize: int = 12, legend_bbox: tuple = (1, 0.55), legend_ncol: int = 1,
                     **kwarg: Any) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
        r"""
        Plot the boxplot of genes from dds data

        Arguments:
            genes: The genes to plot.
            treatment_groups: The treatment groups.
            control_groups: The control groups.
            figsize: The figure size.
            palette: The color palette.
            title: The title of the plot.
            fontsize: The fontsize of the plot.
            legend_bbox: The bbox of the legend.
            legend_ncol: The number of columns of the legend.
            **kwarg: Other arguments for plot_boxplot function.
        
        Returns:
            fig: The figure of the plot.
            ax: The axis of the plot.
        """
        p_data=pd.DataFrame(columns=['Value','Gene','Type'])
        if log:
            for gene in genes:
                plot_data1=pd.DataFrame()
                plot_data1['Value']=np.log1p(self.data[treatment_groups].loc[gene].values)
                plot_data1['Gene']=gene
                plot_data1['Type']=treatment_name

                plot_data2=pd.DataFrame()
                plot_data2['Value']=np.log1p(self.data[control_groups].loc[gene].values)
                plot_data2['Gene']=gene
                plot_data2['Type']=control_name

                plot_data=pd.concat([plot_data1,plot_data2],axis=0)
                p_data=pd.concat([p_data,plot_data],axis=0)
        else:
            for gene in genes:
                plot_data1=pd.DataFrame()
                plot_data1['Value']=self.data[treatment_groups].loc[gene].values
                plot_data1['Gene']=gene
                plot_data1['Type']=treatment_name

                plot_data2=pd.DataFrame()
                plot_data2['Value']=self.data[control_groups].loc[gene].values
                plot_data2['Gene']=gene
                plot_data2['Type']=control_name

                plot_data=pd.concat([plot_data1,plot_data2],axis=0)
                p_data=pd.concat([p_data,plot_data],axis=0)

        fig,ax=plot_boxplot(p_data,hue='Type',x_value='Gene',y_value='Value',palette=palette,
                          figsize=figsize,fontsize=fontsize,title=title,
                          legend_bbox=legend_bbox,legend_ncol=legend_ncol, **kwarg)
        return fig,ax
    
    def ranking2gsea(self,rank_max:int=200,rank_min:int=274)->pd.DataFrame:
        r"""
        Ranking the result of dds data for gsea analysis

        Arguments:
            rank_max: The max rank of the result.
            rank_min: The min rank of the result.

        Returns:
            rnk: The ranking result.

        """


        result=self.result.copy()
        result['fcsign']=np.sign(result['log2FC'])
        result['logp']=-np.log10(result['pvalue'])
        result['metric']=result['logp']/result['fcsign']
        rnk=pd.DataFrame()
        rnk['gene_name']=result.index
        rnk['rnk']=result['metric'].values
        rnk=rnk.sort_values(by=['rnk'],ascending=False)
        k=1
        total=0
        for i in range(len(rnk)):
            if rnk.loc[i,'rnk']==np.inf: 
                total+=1
        #200跟274根据你的数据进行更改，保证inf比你数据最大的大，-inf比数据最小的小就好
        for i in range(len(rnk)):
            if rnk.loc[i,'rnk']==np.inf: 
                rnk.loc[i,'rnk']=rank_max+(total-k)
                k+=1
            elif rnk.loc[i,'rnk']==-np.inf: 
                rnk.loc[i,'rnk']=-(rank_min+k)
                k+=1
        return rnk

    def deg_analysis(self,group1:list,group2:list,
                 method:str='DEseq2',alpha:float=0.05,
                 multipletests_method:str='fdr_bh',n_cpus:int=8,
                 cooks_filter:bool=True, independent_filter:bool=True)->pd.DataFrame:
        r"""
        Differential expression analysis.

        Arguments:
            group1: The first group to be compared.
            group2: The second group to be compared.
            method: The method to be used for differential expression analysis.
                - `DEseq2`: DEseq2
                - `ttest`: ttest
                - `wilcox`: wilconx test
            alpha: The threshold of p-value.
            multipletests_method:
                - `bonferroni` : one-step correction
                - `sidak` : one-step correction
                - `holm-sidak` : step down method using Sidak adjustments
                - `holm` : step-down method using Bonferroni adjustments
                - `simes-hochberg` : step-up method  (independent)
                - `hommel` : closed method based on Simes tests (non-negative)
                - `fdr_bh` : Benjamini/Hochberg  (non-negative)
                - `fdr_by` : Benjamini/Yekutieli (negative)
                - `fdr_tsbh` : two stage fdr correction (non-negative)
                - `fdr_tsbky` : two stage fdr correction (non-negative)

        Returns
            result: The result of differential expression analysis.
        """
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        from scipy.stats import ttest_ind
        from statsmodels.stats.multitest import multipletests
        print(f"⚙️ You are using {method} method for differential expression analysis.")
        if method=='ttest':
            
            data=self.data

            g1_mean=data[group1].mean(axis=1)
            g2_mean=data[group2].mean(axis=1)
            g=(g2_mean+g1_mean)/2
            g=g.loc[g>0].min()
            fold=(g1_mean+g)/(g2_mean+g)
            #log2fold=np.log2(fold)
            ttest = ttest_ind(data[group1].T.values, data[group2].T.values)
            pvalue=ttest[1]
            print(f"⏰ Start to calculate qvalue...")
            qvalue = multipletests(np.nan_to_num(np.array(pvalue),0), alpha=0.5, 
                               method=multipletests_method, is_sorted=False, returnsorted=False)
            #qvalue=fdrcorrection(np.nan_to_num(np.array(pvalue),0), alpha=0.05, method='indep', is_sorted=False)
            genearray = np.asarray(pvalue)
            result = pd.DataFrame({'pvalue':genearray,'qvalue':qvalue[1],'FoldChange':fold})
            result['MaxBaseMean']=np.max([g1_mean,g2_mean],axis=0)
            result['BaseMean']=(g1_mean+g2_mean)/2
            result['log2(BaseMean)']=np.log2((g1_mean+g2_mean)/2)
            result['log2FC'] = np.log2(result['FoldChange'])
            result['abs(log2FC)'] = abs(np.log2(result['FoldChange']))
            result['size']  =np.abs(result['FoldChange'])/10
            result=result.loc[~result['pvalue'].isnull()]
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['qvalue'])
            #max mean of between each value in group1 and group2
            #result=result[result['padj']<alpha]
            result['sig']='normal'
            result.loc[result['qvalue']<alpha,'sig']='sig'
            print(f"✅ Differential expression analysis completed.")
            
            self.result=result
            return result
        elif method=='wilcox':
            raise ValueError('The method is not supported.')
            print(f"⚙️ You are using {method} method for differential expression analysis.")
        elif method=='DEseq2':
            import pydeseq2
            counts_df = self.data[group1+group2].T
            clinical_df = pd.DataFrame(index=group1+group2)

            clinical_df['condition'] = ['Treatment'] * len(group1) + ['Control'] * len(group2)
            print(f"⏰ Start to create DeseqDataSet...")
            # Determine pydeseq2 version and create the DeseqDataSet accordingly
            if pydeseq2.__version__ <= '0.3.5':
                dds = DeseqDataSet(
                    counts=counts_df,
                    clinical=clinical_df,
                    design_factors="condition",  # compare samples based on "condition"
                    ref_level=["condition", "Control"],
                    refit_cooks=True,
                    n_cpus=n_cpus,
                )
            elif pydeseq2.__version__ <= '0.4.1':
                if ad.__version__ > '0.10.8':
                    raise ImportError(
                        'Please install the 0.10.8 version of anndata: `pip install anndata==0.10.8`.'
                    )
                dds = DeseqDataSet(
                    counts=counts_df,
                    metadata=clinical_df,
                    design_factors="condition",
                    refit_cooks=True,
                    n_cpus=n_cpus,
                )
            else:
                from pydeseq2.default_inference import DefaultInference
                inference = DefaultInference(n_cpus=n_cpus)
                dds = DeseqDataSet(
                    counts=counts_df,
                    metadata=clinical_df,
                    design_factors="condition",
                    refit_cooks=True,
                    inference=inference,
                )
        
            dds.fit_size_factors()
            dds.fit_genewise_dispersions()
            dds.fit_dispersion_trend()
            dds.fit_dispersion_prior()
            print(f"logres_prior={dds.uns['_squared_logres']}, sigma_prior={dds.uns['prior_disp_var']}")
            dds.fit_MAP_dispersions()
            dds.fit_LFC()
            dds.calculate_cooks()
            if dds.refit_cooks:
                dds.refit()
        
            # Add the 'contrast' parameter here:
        # FIX: Adding version check for DeseqStats constructor
            if pydeseq2.__version__<='0.3.5':
                stat_res = DeseqStats(dds, alpha=alpha, cooks_filter=cooks_filter, independent_filter=independent_filter)
            elif pydeseq2.__version__ <= '0.4.1':
                # For newer PyDESeq2 versions that require the contrast parameter
                stat_res = DeseqStats(dds, contrast=["condition", "Treatment", "Control"], 
                                    alpha=alpha, cooks_filter=cooks_filter, independent_filter=independent_filter)
                stat_res.run_wald_test()
                if stat_res.cooks_filter:
                    stat_res._cooks_filtering()
                    
                if stat_res.independent_filter:
                    stat_res._independent_filtering()
                else:
                    stat_res._p_value_adjustment()
            else:
                stat_res=DeseqStats(
                            dds,
                            contrast=["condition", "Treatment", "Control"], 
                            alpha=alpha,
                            cooks_filter=cooks_filter,
                            independent_filter=independent_filter,
                        )
                stat_res.run_wald_test()
                if stat_res.cooks_filter:
                    stat_res._cooks_filtering()
                    
                if stat_res.independent_filter:
                    stat_res._independent_filtering()
                else:
                    stat_res._p_value_adjustment()

                    
            self.stat_res = stat_res
            stat_res.summary()
            result = stat_res.results_df
            result['qvalue'] = result['padj']
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['padj'])
            result['BaseMean'] = result['baseMean']
            result['log2(BaseMean)'] = np.log2(result['baseMean'] + 1)
            result['log2FC'] = result['log2FoldChange']
            result['abs(log2FC)'] = abs(result['log2FC'])
            result['sig'] = 'normal'
            result.loc[result['qvalue'] < alpha, 'sig'] = 'sig'
            self.result = result
            print(f"✅ Differential expression analysis completed.")
            return result
            
        
        elif method == 'edgepy':
            try:
                from inmoose.data.pasilla import pasilla
                from inmoose.edgepy import DGEList, glmLRT, topTags
                from patsy import dmatrix
            except:
                raise ImportError('Please install inmoose: `pip install inmoose`')
            print(f"⏰ Start to create DGEList...")
            anno1=pd.DataFrame(
                index=group1+group2
            )
            anno1['condition']=['treatment' for i in group1]+['control' for i in group2]
            var=pd.DataFrame(index=self.data.index)
            var.index.name='gene_id'

            # build a DGEList object
            dge_list = DGEList(
                counts=self.data[group1+group2].values, 
                samples=anno1, 
                group_col="condition", 
                genes=var
            )
            design1 = dmatrix("~condition", data=anno1)
            dge_list.estimateGLMCommonDisp(design=design1)
            fit = dge_list.glmFit(design=design1)
            lrt = glmLRT(fit)
            lrt.index=var.index

            #	log2FoldChange	lfcSE	logCPM	stat	pvalue		
            # 
            pvalue=lrt['pvalue'].values.reshape(-1)
            qvalue = multipletests(np.nan_to_num(np.array(pvalue),0), alpha=0.5, 
                               method=multipletests_method, is_sorted=False, returnsorted=False)
            
            g1_mean=self.data[group1].mean(axis=1)
            g2_mean=self.data[group2].mean(axis=1)
            g=(g2_mean+g1_mean)/2
            g=g.loc[g>0].min()
            fold=(g1_mean+g)/(g2_mean+g)
            print(f"⏰ Start to calculate qvalue...")

            result = pd.DataFrame({'pvalue':pvalue,'qvalue':qvalue[1],'FoldChange':fold})
            result['MaxBaseMean']=np.max([g1_mean,g2_mean],axis=0)
            result['BaseMean']=(g1_mean+g2_mean)/2
            result['log2(BaseMean)']=np.log2((g1_mean+g2_mean)/2)
            result['log2FC'] = np.log2(result['FoldChange'])
            result['abs(log2FC)'] = abs(result['log2FC'])
            result['size']  =np.abs(result['log2FC'])/10
            result=result.loc[~result['pvalue'].isnull()]
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['qvalue'])
            #max mean of between each value in group1 and group2
            #result=result[result['padj']<alpha]
            result['sig']='normal'
            result.loc[result['qvalue']<alpha,'sig']='sig'
            self.result=result
            print(f"✅ Differential expression analysis completed.")
            return result

        elif method == 'limma':
            try:
                from patsy import dmatrix
                from inmoose.limma import lmFit, makeContrasts, contrasts_fit, eBayes, topTable
            except:
                raise ImportError('Please install inmoose: `pip install inmoose`')
            print(f"⏰ Start to create DGEList...")
            anno1=pd.DataFrame(
                index=group1+group2
            )
            anno1['condition']=['treatment' for i in group1]+['control' for i in group2]

            # 3.1 构建设计矩阵
            design1 = dmatrix("~0 + condition", data=anno1)
            #    列名会是 ['condition[treatment]', 'condition[control]']

            # 3.2 lmFit 拟合线性模型
            #    输入: counts_df 行基因为基因，列为样本
            counts_df = self.data[group1+group2].values
            fit = lmFit(counts_df, design1)
            # 3.3 定义对比——treatment vs control
            contrast_matrix = makeContrasts(
                "condition[treatment] - condition[control]",
                levels=design1
            )

            # 3.4 contrasts_fit 应用对比
            fit_con = contrasts_fit(fit, contrast_matrix)

            # 3.5 经验贝叶斯调整
            print(f"⏰ Start to adjust pvalue...")
            fit_eb = eBayes(fit_con)

            g1_mean=self.data[group1].mean(axis=1)
            g2_mean=self.data[group2].mean(axis=1)
            g=(g2_mean+g1_mean)/2
            g=g.loc[g>0].min()
            fold=(g1_mean+g)/(g2_mean+g)

            pvalue=fit_eb.p_value.values.reshape(-1)
            qvalue = multipletests(np.nan_to_num(np.array(pvalue),0), alpha=0.5, 
                               method=multipletests_method, is_sorted=False, returnsorted=False)
            
            result = pd.DataFrame({'pvalue':pvalue,'qvalue':qvalue[1],'FoldChange':fold})
            result['MaxBaseMean']=np.max([g1_mean,g2_mean],axis=0)
            result['BaseMean']=(g1_mean+g2_mean)/2
            result['log2(BaseMean)']=np.log2((g1_mean+g2_mean)/2)
            result['log2FC'] = np.log2(result['FoldChange'])
            result['abs(log2FC)'] = abs(result['log2FC'])
            result['size']  =np.abs(result['log2FC'])/10
            result['sig']='normal'
            result=result.loc[~result['pvalue'].isnull()]
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['qvalue'])
            #max mean of between each value in group1 and group2
            #result=result[result['padj']<alpha]
            result['sig']='normal'
            result.loc[result['qvalue']<alpha,'sig']='sig'

            result['F']=fit_eb.F.reshape(-1)
            result['t']=fit_eb.t.values.reshape(-1)
            self.result=result
            print(f"✅ Differential expression analysis completed.")
            return result

            # 3.6 提取结果
            
            
            
            
        else:  # This is where the "method" check (not pydeseq2 version check) ends
            raise ValueError('The method is not supported.')