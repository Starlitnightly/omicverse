
from .dynamicTree import cutreeHybrid
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

def Trans_corr_matrix(data,
                    method='pearson',
                    cmap='seismic'):
    '''
    Calculate the correlation adjacent matrix (direct and indirect) (scale-free network)

    Parameters
    ----------
    data:pandas.DataFrame
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=gene_name
    method:string
        The method of calculating correlation coefficient
        method='pearson'/'kendall'/'spearman'
    cmap:string
        color Style of drawing

    Returns
    ----------
    result:pandas.DataFrame
        the gene's correlation adjacent matrix
    '''

    data_len=len(data)
    data_index=data.index
    
    #correlation coefficient
    start = datetime.datetime.now()
    print('...correlation coefficient matrix is being calculated')
    result=data.T.corr(method)
    end = datetime.datetime.now()
    result.to_csv('direction correlation matrix.csv')
    print("...direction correlation have been saved")
    print("...calculate time",end-start)
    
    #indirect correlation add
    start = datetime.datetime.now()
    print('...indirect correlation matrix is being calculated')
    np.fill_diagonal(result.values, 0)
    arr=abs(result)
    arr_len=len(arr)
    temp=np.zeros(arr_len)
    for i in range(1,3):    
        temp=temp+(arr**i)/i
    end = datetime.datetime.now()
    temp.to_csv('indirection correlation matrix.csv')
    print("...indirection correlation have been saved")
    print("...calculate time",end-start)
    
        
    #cal soft_threshold
    plt.rc('font', family='Arial')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.weight"] = "12"
    plt.rcParams["axes.labelweight"] = "bold"
    my_dpi=300
    fig=plt.figure(figsize=(2000/my_dpi, 1000/my_dpi), dpi=my_dpi)
    start = datetime.datetime.now()
    print('...soft_threshold is being calculated')
    soft=6
    re1=pd.DataFrame(columns=['beta','r2','meank'])
    for j in range(1,12):
        result_i=np.float_power(temp,j)
        tt_0=np.sum(abs(result_i),axis=0)-1
        n=plt.hist(x = tt_0), #
        x=n[0][0]
        y=[]
        for i in range(len(n[0][1])-1):
            y.append((n[0][1][i]+n[0][1][i+1])/2)
        x=np.log10(x)
        y=np.log10(y)
        res=stats.linregress(x, y)
        r2=np.float_power(res.rvalue,2)
        k=tt_0.mean()
        re1=re1.append({'beta':j,'r2':r2,'meank':k},ignore_index=True)
    for i in re1['r2']:
        if i>0.85:
            soft=re1[re1['r2']==i]['beta'].iloc[0]
            break
    print('...appropriate soft_thresholds:',soft)
    plt.savefig('soft_threshold_hist.png',dpi=300)
    
    #select soft_threhold
    my_dpi=300
    fig=plt.figure(figsize=(2000/my_dpi, 1000/my_dpi), dpi=my_dpi)
    grid = plt.GridSpec(1, 4, wspace=1, hspace=0.1)
    #fig, (ax0, ax1) = plt.subplots(2, 1)
    plt.subplot(grid[0,0:2])
    cmap1=sns.color_palette(cmap)
    plt.subplot(1,2, 1)
    p1=sns.regplot(x=re1["beta"], y=re1['r2'], fit_reg=False, marker="o", color=cmap1[0])
    p1.axhline(y=0.9,ls=":",c=cmap1[5])

    plt.subplot(grid[0,2:4])
    p1=sns.regplot(x=re1["beta"], y=re1['meank'], fit_reg=False, marker="o", color=cmap1[4])
    plt.savefig('soft_threshold_select.png',dpi=300)
    
    test=np.float_power(temp,soft)
    np.fill_diagonal(test.values, 1.0)
    return test
    
def Select_module(data,
                  linkage_method='ward',
                  minClusterSize=30,
                  deepSplit=2,
                  cmap='seismic',
                  trans=True,
                 ):

    '''
    Select gene module from correlation adjacent matrix

    Parameters
    ----------
    data:pandas.DataFrame
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=gene_name
    linkage_method:string
        The method of clustering
        method='ward'/'average'...
    minClusterSize:int
        The least contained gene in each module
    deepSplit:int
        Degree of split
    cmap:string
        color style of figure

    Returns
    ----------
    mol:pandas.DataFrame
        the gene's module
        columns=['ivl','module','name']
    '''

    #distance
    print("...distance have being calculated")
    if trans==True:
       distances = pdist(1-data, "euclidean")
    else:
        distances = pdist(data, "euclidean")
    
    #geneTree
    print("...geneTree have being calculated")
    geneTree=linkage(distances, linkage_method)
      
    #dynamicMods
    print("...dynamicMods have being calculated")
    dynamicMods=cutreeHybrid(geneTree,distM=distances,minClusterSize = minClusterSize,deepSplit = deepSplit, pamRespectsDendro = False)
    print("...total:",len(set(dynamicMods['labels'])))
    
    plt.rc('font', family='Arial')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.weight"] = "12"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.figure(figsize=(25, 10))
    grid = plt.GridSpec(3, 1, wspace=0.5, hspace=0.1)
    #fig, (ax0, ax1) = plt.subplots(2, 1)
    plt.subplot(grid[0:2,0])
    hierarchy.set_link_color_palette(['#000000'])
    dn=hierarchy.dendrogram(geneTree,color_threshold=0, above_threshold_color='black')
    plt.tick_params( \
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    
    #mol
    x=dn['ivl']
    y=[dynamicMods['labels'][int(x)] for x in dn['ivl']]
    yy=np.array([y])
    z=[data.index[int(x)] for x in dn['ivl']]
    mol=pd.DataFrame(columns=['ivl','module','name'])
    mol['ivl']=x
    mol['module']=y
    mol['name']=z
    
    plt.subplot(grid[2,0])
    ax1=plt.pcolor(yy,cmap=cmap)

    plt.savefig('module_tree.png',dpi=300)
    return mol
     

    

def Analysis_cocharacter(data,
                        character,
                        module):

    '''
    Calculate gene and trait correlation matrix

    Parameters
    ----------
    data:pandas.DataFrame
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=gene_name
    character:pandas.DataFrame
        DataFrame of sample's character, columns=character, index=['sample1','sample2',...]
    module:pandas.DataFrame
        The result of the function Select_module

    Returns
    ----------
    result:pandas.DataFrame
        Character and module correlation matrix
    '''
    print("...PCA analysis have being done")
    pcamol=pd.DataFrame(columns=data.columns)
    set_index=set(module['module'])
    for j in set_index:
        newdata=pd.DataFrame(columns=data.columns)
        for i in list(module[module['module']==j].dropna()['name']):
            newdata=newdata.append(data[data.index==i])
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1) 
        reduced_X = pca.fit_transform(newdata.T)
        tepcamol=pd.DataFrame(reduced_X.T,columns=data.columns)
        pcamol=pcamol.append(tepcamol,ignore_index=True)
    pcamol.index=set_index
    
    print("...co-analysis have being done")
    from scipy.stats import spearmanr,pearsonr,kendalltau
    # seed random number generator
    # calculate spearman's correlation
    result_1=pd.DataFrame(columns=character.columns)
    result_p=pd.DataFrame(columns=character.columns)
    for j in character.columns:
        co=[]
        pvv=[]
        for i in range(len(pcamol)):   
            tempcor=pd.DataFrame(columns=['x','y'])
            tempcor['x']=list(character[j])
            tempcor['y']=list(pcamol.iloc[i])
            tempcor=tempcor.dropna()
            coef,pv=pearsonr(tempcor['x'],tempcor['y'])
            co.append(coef)
            pvv.append(pv)
        result_1[j]=co
        result_p[j]=pvv
            #print(coef)
    result_1=abs(result_1)
    result_1.index=set_index
    
    plt.rc('font', family='Arial')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.weight"] = "12"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.figure(figsize=(10,10))
    sns.heatmap(result_1,vmin=0, vmax=1,center=1,annot=True,square=True)
    plt.savefig('co_character.png',dpi=300)
    return result_1