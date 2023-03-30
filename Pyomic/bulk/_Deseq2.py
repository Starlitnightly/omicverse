import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def Matrix_ID_mapping(data,gene_ref_path):
    pair=pd.read_csv(gene_ref_path,sep='\t',index_col=0)
    ret_gene=list(set(data.index.tolist()) & set(pair.index.tolist()))
    data=data.loc[ret_gene]
    new_index=[]
    for i in ret_gene:
        a=pair.loc[i,'symbol']
        if str(a)=='nan':
            new_index.append(i)
        else:
            new_index.append(a)
    data.index=new_index
    return data


def deseq2_normalize(data):
    r"""
    Normalize the data using DESeq2 method:

    Parameters
    ----------
    data: pandas.DataFrame
        The data to be normalized.

    Returns
    -------
    pandas.DataFrame
        The normalized data.
    """
    avg1=data.apply(np.log,axis=1).mean(axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    data1=data.loc[avg1.index]
    data_log=data1.apply(np.log,axis=1)
    scale=data_log.sub(avg1.values,axis=0).median(axis=0).apply(np.exp)
    return data/scale

def estimateSizeFactors(data):
    
    avg1=data.apply(np.log,axis=1).mean(axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    data1=data.loc[avg1.index]
    data_log=data1.apply(np.log,axis=1)
    scale=data_log.sub(avg1.values,axis=0).median(axis=0).apply(np.exp)
    return scale


def estimateDispersions(counts):
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

def data_drop_duplicates_index(data):
    r"""
    Drop the duplicated index of data.

    Parameters
    ----------
    data: pandas.DataFrame
        The data to be processed.

    Returns
    -------
    pandas.DataFrame
        The data after dropping the duplicated index.
    """
    index=data.index
    data=data.loc[~index.duplicated(keep='first')]
    return data

class pyDEseq(object):


    def __init__(self,raw_data) -> None:
        self.raw_data=raw_data
        self.data=raw_data.copy()
        
    def drop_duplicates_index(self):
        r"""
        Drop the duplicated index of data.

        Returns
        -------
        pandas.DataFrame
            The data after dropping the duplicated index.
        """
        self.data=data_drop_duplicates_index(self.data)
        return self.data

    def normalize(self):
        r"""
        Normalize the data using DESeq2 method.
        
        Returns
        -------
        pandas.DataFrame
            The normalized data.
        """
        self.size_factors=estimateSizeFactors(self.data)
        self.data=deseq2_normalize(self.data)
        return self.data
    
    def foldchange_set(self,fc_threshold=-1,pval_threshold=0.05,logp_max=6,fold_threshold=0):
        if fc_threshold==-1:
            foldp=np.histogram(self.result['log2FC'])
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
    

    def plot_volcano(self,figsize=(4,4),title='',titlefont={'weight':'normal','size':14,},
                     up_color='#e25d5d',down_color='#7388c1',normal_color='#d7d7d7',
                     legend_bbox=(0.8, -0.2),legend_ncol=2,legend_fontsize=12,
                     plot_genes=None,plot_genes_num=10,plot_genes_fontsize=10,
                     ticks_fontsize=12):

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
        color = ['#e25d5d','#174785']
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
        if plot_genes is not None:
            hub_gene=plot_genes
        else:
            up_result=result.loc[result['sig']=='up']
            down_result=result.loc[result['sig']=='down']
            hub_gene=up_result.sort_values('qvalue').index[:plot_genes_num//2].tolist()+down_result.sort_values('qvalue').index[:plot_genes_num//2].tolist()

        color_dict={
        'up':'#a51616',
            'down':'#174785',
            'normal':'grey'
        }

        texts=[ax.text(result.loc[i,'log2FC'], 
               result.loc[i,'-log(qvalue)'],
               i,
               fontdict={'size':plot_genes_fontsize,'weight':'bold','color':color_dict[result.loc[i,'sig']]}
               ) for i in hub_gene if 'ENSG' not in i]
        adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)

        ax.set_xticks([round(i,2) for i in ax.get_xticks()[1:-1]],#获取x坐标轴内容
              [round(i,2) for i in ax.get_xticks()[1:-1]],#更新x坐标轴内容
              fontsize=ticks_fontsize,
              fontweight='normal'
              )
        return ax
        

    def deg_analysis(self,group1,group2,method='ttest',alpha=0.05):
        r"""
        Differential expression analysis.

        Parameters
        ----------
        group1: list
            The first group to be compared.
        group2: list
            The second group to be compared.
        method: str
            The method to be used for differential expression analysis.
            The default value is 'ttest'.
        alpha: float
            The threshold of p-value.

        Returns
        -------
        pandas.DataFrame
            The result of differential expression analysis.
        """
        if method=='ttest':
            from scipy.stats import ttest_ind
            from statsmodels.stats.multitest import fdrcorrection
            data=self.data

            g1_mean=data[group1].mean(axis=1)
            g2_mean=data[group2].mean(axis=1)
            g=(g2_mean+g1_mean)/2
            g=g.loc[g>0].min()
            fold=(g1_mean+g)/(g2_mean+g)
            log2fold=np.log2(fold+1)
            ttest = ttest_ind(data[group1].T.values, data[group2].T.values)
            pvalue=ttest[1]

            qvalue=fdrcorrection(np.nan_to_num(np.array(pvalue),0), alpha=0.05, method='indep', is_sorted=False)
            genearray = np.asarray(pvalue)
            result = pd.DataFrame({'pvalue':genearray,'qvalue':qvalue[1],'FoldChange':fold})
            
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['qvalue'])
            result['BaseMean']=g1_mean+g2_mean
            result['log2(BaseMean)']=np.log2((g1_mean+g2_mean)/2)
            result['log2FC'] = np.log2(result['FoldChange'])
            result['abs(log2FC)'] = abs(np.log2(result['FoldChange']))
            result['size']  =np.abs(result['FoldChange'])/10
            #result=result[result['padj']<alpha]
            result['sig']='normal'
            result.loc[result['qvalue']<alpha,'sig']='sig'
            
            self.result=result
            return result
        elif method=='wilcox':
            from scipy.stats import wilcoxon
            from statsmodels.stats.multitest import fdrcorrection
            data=self.data

            g1_mean=data[group1].mean(axis=1)
            g2_mean=data[group2].mean(axis=1)
            fold=(g1_mean+0.00001)/(g2_mean+0.00001)
            log2fold=np.log2(fold+1)
            wilcox = wilcoxon(data[group1].T.values, data[group2].T.values)
            pvalue=wilcox[1]

            qvalue=fdrcorrection(np.nan_to_num(np.array(pvalue),0), alpha=0.05, method='indep', is_sorted=False)
            genearray = np.asarray(pvalue)
            result = pd.DataFrame({'pvalue':genearray,'qvalue':qvalue[1],'FoldChange':fold})
            
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['qvalue'])
            result['BaseMean']=g1_mean+g2_mean
            result['log2(BaseMean)']=np.log2((g1_mean+g2_mean)/2)
            result['log2FC'] = np.log2(result['FoldChange'])
            result['abs(log2FC)'] = abs(np.log2(result['FoldChange']))
            result['size']  =np.abs(result['FoldChange'])/10
            #result=result[result['padj']<alpha]
            result['sig']='normal'
            result.loc[result['qvalue']<alpha,'sig']='sig'
            self.result=result
            return result

        else:
            raise ValueError('The method is not supported.')