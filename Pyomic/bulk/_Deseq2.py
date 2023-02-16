import numpy as np
import pandas as pd
import scanpy as sc


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
        self.data=deseq2_normalize(self.data)
        return self.data

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
            fold=(g1_mean+0.00001)/(g2_mean+0.00001)
            log2fold=np.log2(fold+1)
            ttest = ttest_ind(data[group1].T.values, data[group2].T.values)
            pvalue=ttest[1]

            qvalue=fdrcorrection(np.nan_to_num(np.array(pvalue),0), alpha=0.05, method='indep', is_sorted=False)
            genearray = np.asarray(pvalue)
            result = pd.DataFrame({'pvalue':genearray,'qvalue':qvalue[1],'FoldChange':fold})
            
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['qvalue'])
            result['log2(BaseMean)']=np.log2((g1_mean+g2_mean)/2)
            result['log2FC'] = np.log2(result['FoldChange'])
            result['abs(log2FC)'] = abs(np.log2(result['FoldChange']))
            result['size']  =np.abs(result['FoldChange'])/10
            #result=result[result['padj']<alpha]
            result['sig']='normal'
            result.loc[result['qvalue']<alpha,'sig']='sig'
            return result
        else:
            raise ValueError('The method is not supported.')