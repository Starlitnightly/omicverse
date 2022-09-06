import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy.stats import norm
from scipy import stats
import networkx as nx
import datetime
import seaborn as sns
import pandas as pd
from scipy.cluster import hierarchy  
from scipy import cluster   
from sklearn import decomposition as skldec 


from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,dendrogram

import ERgene
import os

def find_DEG(data,eg,cg,
             log2fc=-1,
             fold_threshold=0,
             pvalue_threshold=0.05,
             cmap="seismic"
            ):
    '''
    Find out the differential expression gene

    Parameters
    ----------
    data:pandas.DataFrame
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=gene_name
    eg:list
        Columns name of the experimental group
        Sample:['lab1','lab2']
    cg:list
        Columns name of the control group
        Sample:['Ctrl1','Ctrl2']
    log2fc:float
        The threshold value of the difference multiple
        If it is -1, then use the column in the middle of the HIST diagram to filter
    fold_threshold:int
        This parameter is only valid when log2fc is -1, representing a multiple of the HIST filter
    pvalue_threshold:float
        Represents the threshold value of pvalue
    cmap:string
        Style of drawing

    Returns
    -------
    result:pandas.DataFrame
        index is data's index, columns=['pvalue','qvalue','FoldChange','log(pvalue)','log2FC','sig','size']
    '''


    #plt_set
    font1 = {'family' :'Arial','weight' :'bold','size' :15}
    my_dpi=300
    fig=plt.figure(figsize=(2000/my_dpi, 1000/my_dpi), dpi=my_dpi)
    grid = plt.GridSpec(1, 4, wspace=1, hspace=0.1)
    
    #cal_mean
    eg_mean=data[eg].mean(axis=1)
    eg_mean.head()
    cg_mean=data[cg].mean(axis=1)
    cg_mean.head()
    
    #cal_fold
    plt.subplot(grid[0,0:2])
    fold=eg_mean/cg_mean
    log2fold=-np.log2(fold)
    foldp=plt.hist(log2fold,color="#384793")
    if log2fc==-1:
        foldchange=(foldp[1][np.where(foldp[1]>0)[0][fold_threshold]]+foldp[1][np.where(foldp[1]>0)[0][fold_threshold+1]])/2
    else:
        foldchange=log2fc
    cmap1=sns.color_palette(cmap)
    plt.title("log2fc",font1)
    plt.ylabel('Density',font1)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.savefig("log2fc.png",dpi=300,bbox_inches = 'tight')
    
    #pvalue
    pvalue = []
    for i in range(0, len(data)):
        ttest = stats.ttest_ind(list(data.iloc[i][eg].values), list(data.iloc[i][cg].values))
        pvalue.append(ttest[1])
    
    #FDR
    from statsmodels.stats.multitest import fdrcorrection
    qvalue=fdrcorrection(np.array(pvalue), alpha=0.05, method='indep', is_sorted=False)
    
    #result
    genearray = np.asarray(pvalue)
    result = pd.DataFrame({'pvalue':genearray,'qvalue':qvalue[1],'FoldChange':fold})
    result['-log(pvalue)'] = -np.log10(result['pvalue'])
    result['log2FC'] = np.log2(result['FoldChange'])
    result['sig'] = 'normal'
    result['size']  =np.abs(result['FoldChange'])/10
    result.loc[(result.log2FC> foldchange )&(result.pvalue < pvalue_threshold),'sig'] = 'up'
    result.loc[(result.log2FC< 0-foldchange )&(result.pvalue < pvalue_threshold),'sig'] = 'down'
    result.to_csv("DEG_result.csv")
    print('up:',len(result[result['sig']=='up']))
    print('down:',len(result[result['sig']=='down']))
    
    #plt
    plt.subplot(grid[0,2:4])
    ax = sns.scatterplot(x="log2FC", y="-log(pvalue)",
                      hue='sig',
                      hue_order = ('up','down','normal'),
                      palette=(cmap1[5],cmap1[0],"grey"),
                      size='sig',sizes=(50, 100),
                      data=result)
    ax.set_ylabel('-log(pvalue)',font1)                                    
    ax.set_xlabel('log2FC',font1)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.legend(bbox_to_anchor=(1.05,0), loc=3, borderaxespad=0)
    ax.tick_params(labelsize=15)
    plt.savefig("DER_fire.png",dpi=300,bbox_inches = 'tight')
    
    #snsclu
    fold_cutoff = foldchange
    pvalue_cutoff = 0.05
    filtered_ids = []
    for i in range(0, len(result)):
        if (abs(np.log2(fold[i])) >= fold_cutoff) and (pvalue[i] <= pvalue_cutoff):
            filtered_ids.append(i)        
    filtered = data.iloc[filtered_ids,:]
    filtered.to_csv('fi.csv')
    a=sns.clustermap(filtered, cmap=cmap, standard_scale = 0)
    plt.savefig("sns2.png",dpi=300,bbox_inches = 'tight')
    return result

def Density_norm(data,
                depth=2,
                legend=False,
                norm_by=0,
                xlim=-1,
                ylim=-1):
    '''
    The batch effect of samples was eliminated and the data was normalized

    Parameters
    ----------
    data:pandas.DataFrame
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=gene_name
    depth:int
        Number of samples used
        Accuracy of endogenous reference gene,must be larger that 2
        The larger the number, the fewer genes are screened out,Accuracy improvement
    legend:bool
        Whether to display the diagram legend
    norm_by:int
        When the value is 0, the first reference gene screened was used for normalization
    xlim:float
        When the value is not -1,the abscissa range of the generated graph is [-xlim,xlim] 
    ylim:float
        When the value is not -1,the ordinate range of the generated graph is [0,ylim] 

    Returns
    ----------
    result:pandas.DataFrame
        A expression data matrix that normalized
    '''

    #plt_set
    font1 = {'family': 'Arial','weight' : 'bold','size'   : 15}
    my_dpi=300
    fig=plt.figure(figsize=(2000/my_dpi, 1000/my_dpi), dpi=my_dpi)
    grid = plt.GridSpec(1, 4, wspace=2, hspace=0.1)


    #raw data
    ax1=plt.subplot(grid[0,0:2])
    data.plot(kind = 'density',legend=legend,fontsize=15,colormap='seismic',ax=ax1)
    plt.ylabel('Density',font1)
    plt.title('Raw Count',font1)
    if (xlim!=-1):
        plt.xlim(0-xlim,xlim)
    if (ylim!=-1):
        plt.ylim(0,ylim)

    #normalized
    ERlist=ERgene.FindERG(data,depth)
    data2=ERgene.normalizationdata(data,ERlist[norm_by])

    #After data
    ax2=plt.subplot(grid[0,2:4])
    data2.plot(kind='density',legend=legend,fontsize=15,colormap='seismic',ax=ax2)
    plt.ylabel('',font1)
    plt.title('After Count',font1)
    if (xlim!=-1):
        plt.xlim(0-xlim,xlim)
    if (ylim!=-1):
        plt.ylim(0,ylim)
    plt.savefig("norm.png",dpi=300,bbox_inches = 'tight')
    return data2


def Plot_gene_expression(data,
                         gene_list,
                         eg,
                         cg,
                         save_name="gene_expression_plot.png",
                        cmap='seismic'):
    '''
    Plot the expression of specific genes in the expression matrix

    Parameters
    ----------
    data:pandas.DataFrame
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=gene_name
    gene_list:list
        The specific gene that needs to be plot
    eg:list
        Columns name of the experimental group
        Sample:['lab1','lab2']
    cg:list
        Columns name of the control group
        Sample:['Ctrl1','Ctrl2']
    save_name:string
        The save path of the figure
    cmap:string
        The color style of the figure

    '''

    ui=pd.DataFrame(columns=['gene','value','class'])
    gene=gene_list
    for ge in gene:
        for i in data.loc[ge][eg].values:
            ui=ui.append({'gene':ge,'value':i,'class':'experiment'},ignore_index=True)
        for i in data.loc[ge][cg].values:
            ui=ui.append({'gene':ge,'value':i,'class':'ctrl'},ignore_index=True)
    df=ui
    dyt=[]
    for i in df['gene']:
        if (i not in dyt):
            dyt.append(i)
    dyt

    #plt.figure(figsize=(10,5))
    plt.rc('font', family='Arial')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.weight"] = "15"
    plt.rcParams["axes.labelweight"] = "bold"
    cmap1=sns.color_palette(cmap)
    ax = sns.violinplot(x="gene", y="value",hue="class", data=df,palette=[cmap1[0],cmap1[5]],
                        scale="area",split=True,
                        )
    ax.set_ylabel('Gene Expression',fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax.set_xlabel('')
    for i in range(len(dyt)):
        ttest = stats.ttest_ind(df[df['gene']==dyt[i]][df[df['gene']==dyt[i]]['class']=='ctrl']['value'], df[df['gene']==dyt[i]][df[df['gene']==dyt[i]]['class']=='experiment']['value'])
        max=df[df['gene']==dyt[i]]['value'].max()
        if(ttest[1]<0.001):
            xing="***"   
        elif(ttest[1]<0.01):
            xing="**"
        elif(ttest[1]<0.05):
            xing="*"
        else:
            xing=' '
        print(ttest[1],xing)
        ax.text(i,max+0.5, xing,ha='center', va='bottom', fontsize=15)
    plt.savefig(save_name,dpi=300,bbox_inches = 'tight')

def ID_mapping(raw_data,
                mapping_data,
                raw_col,
                map_col):
    
    '''
    Universal probe matching conversion gene ID

    Parameters
    ----------
    raw_data:pandas.DataFrame
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=probe
    mapping_data:pandas.DataFrame
        DataFrame of data of a matrix with pre IDs and after-conversion IDs
    raw_col:string
        columns name of mapping_data that is raw-IDs
    map_col:string
        columns name of mapping_data that is after-conversion IDs

    Returns
    ----------
    result:pandas.DataFrame
        A expression data matrix that conversion
    '''

    raw_index=raw_data.index
    #raw_col='queryItem'
    raw_length=raw_data.columns.size
    #map_col='preferredName'
    map_index=[]
    map_data=pd.DataFrame(columns=raw_data.columns)
    mapping=mapping_data
    for i in sorted(set(list(raw_index)),key=list(raw_index).index):
        if(mapping[mapping[raw_col]==i][map_col].values.size==0):
            continue
        else:

            if(raw_data.loc[i].size==raw_length):
                map_index.append(mapping[mapping[raw_col]==i][map_col].values[0])
                map_data=map_data.append(raw_data.loc[i],ignore_index=True)
            else:
                map_index.append(mapping[mapping[raw_col]==i][map_col].values[0])
                testdata=raw_data.loc[i]
                map_data=map_data.append(testdata.iloc[np.where(testdata.mean(axis=1)==testdata.mean(axis=1).max())])
            #print(mapping[mapping[raw_col]==i][map_col].values[0],raw_data.loc[i])
    map_data.index=map_index
    return map_data

def Drop_dupligene(raw_data):
    '''
    Drop the duplicate genes by max

    Parameters
    ----------
    raw_data:pandas.DataFrame
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=probe

    Returns
    ----------
    new_data:pandas.DataFrame
        A expression data matrix that have been duplicated
    '''

    print("...Drop the duplicate genes by max")
    raw_len=len(raw_data.columns)
    new_data=pd.DataFrame(columns=raw_data.columns)
    raw_index=raw_data.index
    new_index=[]
    for i in sorted(set(list(raw_index)),key=list(raw_index).index):
        if(raw_data.loc[i].size==raw_len):
            new_index.append(i)
            new_data=new_data.append(raw_data.loc[i],ignore_index=True)
        else:
            new_index.append(i)
            testdata=raw_data.loc[i]
            new_data=new_data.append(testdata.iloc[np.where(testdata.mean(axis=1)==testdata.mean(axis=1).max())[0][0]],ignore_index=True)       
    new_data.index=new_index
    return new_data