
from ._dynamicTree import cutreeHybrid
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy import stats
import networkx as nx
import datetime
import seaborn as sns
from scipy.cluster import hierarchy   
from sklearn import decomposition as skldec 
from matplotlib.colors import LinearSegmentedColormap

from typing import Union,Tuple
import matplotlib


from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,dendrogram
from ..utils import pyomic_palette,plot_network
import os

try:
    from ..external.PyWGCNA.wgcna import pyWGCNA
    from ..external.PyWGCNA.utils import readWGCNA
except Exception:  # pragma: no cover - optional dependency
    pyWGCNA = None
    readWGCNA = None


class pyWGCNA_old(object):
    r"""
        pyWGCNA: Weighted correlation network analysis in Python
    """
    def __init__(self,data:pd.DataFrame,save_path:str=''):
        r"""Initialize the pyWGCNA module.
        
        Arguments:
            data: The dataframe of gene expression data
            save_path: The path to save the results. ('')
        
        Returns:
            None
        """
        self.data=data.fillna(0)
        self.data_len=len(data)
        self.data_index=data.index
        self.save_path=save_path

    def mad_filtered(self,gene_num:int=5000):
        r"""Filter genes by MAD to construct a scale-free network.

        Arguments:
            gene_num: The number of genes to be saved. (5000)
        
        Returns:
            None
        """
        from statsmodels import robust #import package
        gene_mad=self.data.T.apply(robust.mad) #use function to calculate MAD
        self.data=self.data.loc[gene_mad.sort_values(ascending=False).index[:gene_num]]

    def calculate_correlation_direct(self,method:str='pearson',save:bool=False):
        r"""Calculate the correlation coefficient matrix.

        Arguments:
            method: The method to calculate the correlation coefficient matrix. ('pearson')
            save: Whether to save the result. (False)
        
        Returns:
            None
        """
        print('...correlation coefficient matrix is being calculated')
        self.result=self.data.T.corr(method)
        if save==True:
            self.result.to_csv(self.save_path+'/'+'direction correlation matrix.csv')
            print("...direction correlation have been saved")


    def calculate_correlation_indirect(self,save:bool=False):
        r"""Calculate the indirect correlation coefficient matrix.
        
        Arguments:
            save: Whether to save the result. (False)
        
        Returns:
            None
        """
        print('...indirect correlation matrix is being calculated')
        np.fill_diagonal(self.result.values, 0)
        arr=abs(self.result)
        arr_len=len(arr)
        self.temp=np.zeros(arr_len)
        for i in range(1,3):    
            self.temp=self.temp+(arr**i)/i

        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minMax = min_max_scaler.fit_transform(self.temp)
        self.normal_corelation=pd.DataFrame(X_minMax,index=self.temp.index,columns=self.temp.columns)

        if save==True:
            self.temp.to_csv(self.save_path+'/'+'indirection correlation matrix.csv')
            print("...indirection correlation have been saved")


    def calculate_soft_threshold(self,threshold_range:int=12,
                                 plot:bool=True,save:bool=False,
                                 figsize:tuple=(6,3))->pd.DataFrame:
        """calculate the soft threshold
        
        Arguments:
            threshold_range: The range of threshold
            plot: Whether to plot the result
            save: Whether to save the result

        Returns:
            The dataframe of soft threshold
        """

        print('...soft_threshold is being calculated')
        soft=6
        re1=pd.DataFrame(columns=['beta','r2','meank'])
        for j in range(1,threshold_range):
            #print('Now is:',j)
            result_i=np.float_power(self.temp,j)
            tt_0=np.sum(abs(result_i),axis=0)-1
            n=np.histogram(tt_0), #
            x=n[0][0]
            y=[]
            for i in range(len(n[0][1])-1):
                y.append((n[0][1][i]+n[0][1][i+1])/2)
            x=np.log10(x)
            y=np.log10(y)
            res=stats.linregress(x, y)
            r2=np.float_power(res.rvalue,2)
            k=tt_0.mean()
            re1.loc[j]={'beta':j,'r2':r2,'meank':k}
        for i in re1['r2']:
            if i>0.85:
                soft=re1[re1['r2']==i]['beta'].iloc[0]
                break
        self.re1=re1
        self.soft=soft
        print('...appropriate soft_thresholds:',soft)
        if plot==True:
            fig, ax = plt.subplots(1,2,figsize=figsize)
            ax[0].scatter(re1['beta'],re1['r2'],c='b')
            ax[0].plot([0,threshold_range],[0.95,0.95],c='r')
            ax[0].set_xlim(0,threshold_range)
            ax[0].set_ylabel('r2',fontsize=14)
            ax[0].set_xlabel('beta',fontsize=14)
            ax[0].set_title('Best Soft threshold')

            ax[1].scatter(re1['beta'],re1['meank'],c='r')
            ax[1].set_ylabel('meank',fontsize=14)
            ax[1].set_xlabel('beta',fontsize=14)
            ax[1].set_title('Best Soft threshold')
            fig.tight_layout()
        if save==True:
            fig.savefig(self.save_path+'/'+'soft_threshold_hist.png',dpi=300)
        return re1



    def calculate_corr_matrix(self):
        """calculate the correlation matrix
        
        """
        self.cor=np.float_power(self.temp,self.soft)
        np.fill_diagonal(self.cor.values, 1.0)

    def calculate_distance(self,trans:bool=True):
        """calculate the distance matrix
        
        Arguments:
            trans: Whether to transpose the correlation matrix
        """
        #distance
        print("...distance have being calculated")
        if trans==True:
            self.distances = pdist(1-self.cor, "euclidean")
        else:
            self.distances = pdist(self.cor, "euclidean")

    def calculate_geneTree(self,linkage_method:str='ward'):
        """
        calculate the geneTree
        
        Arguments:
            linkage_method: The method to calculate the geneTree, it can be found in `scipy.cluster.hierarchy.linkage`
        """

        #geneTree
        print("...geneTree have being calculated")
        self.geneTree=linkage(self.distances, linkage_method)

    def calculate_dynamicMods(self,minClusterSize:int=30,
                  deepSplit:int=2,):
        """calculate the dynamicMods

        Arguments:
            minClusterSize: The minimum size of cluster
            deepSplit: The deep of split
        """

        #dynamicMods
        print("...dynamicMods have being calculated")
        self.dynamicMods=cutreeHybrid(self.geneTree,distM=self.distances,minClusterSize = minClusterSize,
                                            deepSplit = deepSplit, pamRespectsDendro = False)
        print("...total:",len(set(self.dynamicMods['labels'])))

    def calculate_gene_module(self,figsize:tuple=(25,10),save:bool=True,
                              colorlist:list=None)->pd.DataFrame:
        """calculate the gene module
        
        Arguments:
            figsize: The size of figure
            save: Whether to save the figure
            colorlist: The color list of module

        Returns:
            The dataframe of gene module
        """
        plt.figure(figsize=figsize)
        grid = plt.GridSpec(3, 1, wspace=0.5, hspace=0.1)

        plt.subplot(grid[0:2,0])
        hierarchy.set_link_color_palette(['#000000'])
        dn=hierarchy.dendrogram(self.geneTree,color_threshold=0, above_threshold_color='black')
        plt.tick_params( \
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
        
        #mol
        #ivl=dn['ivl']
        mod=[self.dynamicMods['labels'][int(x)] for x in dn['ivl']]
        plot_mod=np.array([mod])
        gene_name=[self.cor.index[int(x)] for x in dn['ivl']]
        mol=pd.DataFrame(columns=['ivl','module','name'])
        mol['ivl']=dn['ivl']
        mol['module']=mod
        mol['name']=gene_name

        res=[]
        for i in plot_mod[0]:
            if i in res:
                continue
            else:
                res.append(i)
        res_len=len(res)
        if colorlist!=None:
            colorlist_cmap=LinearSegmentedColormap.from_list('Custom', colorlist[:res_len], len(colorlist[:res_len]))
            color_dict=dict(zip(range(1,res_len+1),colorlist[:res_len]))
            mol['color']=mol['module'].map(color_dict)
            plt.subplot(grid[2,0])
            ax1=plt.pcolor(plot_mod,cmap=colorlist_cmap)
        else:
            if res_len>28:
                from ..pl._palette import palette_112
                colorlist=palette_112[:res_len]
                #colorlist=sc.pl.palettes.default_102
            else:
                colorlist=pyomic_palette()
            colorlist_cmap=LinearSegmentedColormap.from_list('Custom', colorlist[:res_len], len(colorlist[:res_len]))
            color_dict=dict(zip(range(1,res_len+1),colorlist[:res_len]))
            mol['color']=mol['module'].map(color_dict)
            plt.subplot(grid[2,0])
            ax1=plt.pcolor(plot_mod,cmap=colorlist_cmap)

        self.mol=mol
        if save==True:
            plt.savefig(self.save_path+'/'+'module_tree.png',dpi=300)
        return mol
    
    def get_sub_module(self,mod_list:list)->pd.DataFrame:
        '''
        Get sub-module of a module

        Arguments:
            mod_list: module number

        Returns:
            sub_module:sub-module of a module
        '''
        mol=self.mol
        return mol[mol['module'].isin(mod_list)]
    
    def get_sub_network(self,mod_list:list,correlation_threshold=0.95)->nx.Graph:
        '''
        Get sub-network of a module

        Arguments:
            mod_list: module number

        Returns:
            sub_network:sub-network of a module
        '''
        module1=self.get_sub_module(mod_list)
        gene_net1=abs(self.normal_corelation.loc[module1.name.tolist(),module1.name.tolist()])


        G = nx.Graph()
        for i in gene_net1.index:
            for j in gene_net1.columns:
                if gene_net1.loc[i,j]>correlation_threshold:
                    G.add_weighted_edges_from([(i, j, gene_net1.loc[i,j])])
        return G
    
    def plot_matrix(self,cmap='RdBu_r',save:bool=True,figsize:tuple=(8,9),
                    legene_ncol:int=2,legene_bbox_to_anchor:tuple=(5, 2.95),legene_fontsize:int=12,):
        """plot the matrix of correlation

        Arguments:
            cmap: The color of matrix
            save: Whether to save the figure
            figsize: The size of figure
            legene_ncol: The number of column of legene
            legene_bbox_to_anchor: The position of legene
            legene_fontsize: The size of legene

        Returns:
            ax: The axis of figure
        """

        module_color=self.mol.copy()
        module_color.index=module_color['name']
        a=sns.clustermap(self.normal_corelation,#standard_scale=1,
                    cmap=cmap,yticklabels=False,xticklabels=False,#method='ward',metric='euclidean',
                    #row_cluster=False,col_cluster=False,
                    col_linkage=self.geneTree,row_linkage=self.geneTree,
                    col_colors=module_color.loc[self.normal_corelation.index,'color'].values,
                    row_colors=module_color.loc[self.normal_corelation.index,'color'].values,
                    cbar_kws={"shrink": .5},square=True,
                    dendrogram_ratio=(.1, .2),
                    cbar_pos=(-.1, .32, .03, .2),figsize=figsize
        )
        #add the color of the module
        for i in range(len(set(self.mol['module']))):
            t1=self.mol.loc[self.mol['module']==i+1]
            a.ax_heatmap.add_patch(plt.Rectangle((t1.index[0],t1.index[0]),len(t1),len(t1),fill=False,color='black',lw=1))

        color=[]
        labels=[]
        for i in module_color['module'].unique():
            t1=module_color.loc[module_color['module']==i].iloc[0]
            color.append(t1['color'])
            labels.append(i)
        from matplotlib import patches as mpatches
        #用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
        patches = [ mpatches.Patch(color=color[i], label="ME-{}".format(labels[i]) ) for i in range(len(labels)) ] 
        plt.legend(handles=patches,bbox_to_anchor=legene_bbox_to_anchor, ncol=legene_ncol,fontsize=legene_fontsize)
        if save==True:
            plt.savefig(self.save_path+'/'+'module_matrix.png',dpi=300,bbox_inches = 'tight')
        return a

    
    def plot_sub_network(self,mod_list:list,
                         correlation_threshold:float=0.95,
                         plot_genes=None,
                         plot_gene_num:int=5,**kwargs)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        '''
        plot sub-network of a module

        Arguments:
            mod_list: module number
            correlation_threshold: correlation threshold
            plot_genes: genes to plot in the sub-network. If None, the hub genes will be ploted
            plot_gene_num: number of genes to plot

        Returns:
            fig: figure
            ax: axis
        '''

        module1=self.get_sub_module(mod_list)
        G_type_dict=dict(zip(module1['name'],[str(i) for i in module1['module']]))
        G_color_dict=dict(zip(module1['name'],module1['color']))
        G=self.get_sub_network(mod_list,correlation_threshold)
        G_color_dict=dict(zip(G.nodes,[G_color_dict[i] for i in G.nodes]))
        G_type_dict=dict(zip(G.nodes,[G_type_dict[i] for i in G.nodes]))

        degree_dict = dict(G.degree(G.nodes()))
        de_pd=pd.DataFrame(degree_dict.values(),index=degree_dict.keys(),columns=['Degree'])
        hub_gene=[]
        if plot_genes!=None:
            hub_gene=plot_genes
        else:
            for i in mod_list:
                ret_gene=list(set(de_pd.index) & set(module1.loc[module1['module']==i]['name'].tolist()))
                hub_gene1=de_pd.loc[ret_gene,'Degree'].sort_values(ascending=False)[:plot_gene_num].index.tolist()
                hub_gene+=hub_gene1

        fig,ax=plot_network(G,G_type_dict,G_color_dict,plot_node=hub_gene,**kwargs)
        return fig,ax
        
    def analysis_meta_correlation(self,meta_data:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
        """Analysis meta correlation
        
        Arguments:
            meta_data: meta data of samples

        Returns:
            meta_cor: meta correlation
            meta_p: meta p-value
        """

        print("...PCA analysis have being done")
        data=self.data
        module=self.mol
        pcamol=pd.DataFrame(columns=data.columns)
        set_index=set(module['module'])
        for j in set_index:
            newdata=pd.DataFrame(columns=data.columns)
            for i in list(module[module['module']==j].dropna()['name']):
                newdata.loc[i]=data[data.index==i].values[0]
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1) 
            reduced_X = pca.fit_transform(newdata.T)
            tepcamol=pd.DataFrame(reduced_X.T,columns=data.columns)
            pcamol.loc[j]=tepcamol.values[0]
        pcamol.index=set_index
        
        print("...co-analysis have being done")
        from scipy.stats import spearmanr,pearsonr,kendalltau
        # seed random number generator
        # calculate spearman's correlation
        new_meta=pd.DataFrame()
        new_meta.index=meta_data.index
        for j in meta_data.columns:
            if meta_data[j].dtype=='int64':
                new_meta=pd.concat([new_meta,meta_data[j]],axis=1)
            elif meta_data[j].dtype!='float32': 
                new_meta=pd.concat([new_meta,meta_data[j]],axis=1)
            elif meta_data[j].dtype!='float64':
                new_meta=pd.concat([new_meta,meta_data[j]],axis=1)
            else:
                one_hot = pd.get_dummies(meta_data[j], prefix=j)
                new_meta=pd.concat([new_meta,one_hot],axis=1)
                

        result_1=pd.DataFrame(columns=new_meta.columns)
        result_p=pd.DataFrame(columns=new_meta.columns)
        for j in new_meta.columns:
            co=[]
            pvv=[]
            for i in range(len(pcamol)):   
                tempcor=pd.DataFrame(columns=['x','y'])
                tempcor['x']=list(new_meta[j])
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

        return result_1,result_p
    
    def plot_meta_correlation(self,cor_matrix:tuple,
                              label_fontsize:int=10,label_colors:str='red')->matplotlib.axes._axes.Axes:
        """Plot meta correlation
        
        Arguments:
            cor_matrix: meta correlation and meta p-value
            label_fontsize: label fontsize
            label_colors: label colors

        Returns:
            ax: axis
        """

        cor=cor_matrix[0].copy()
        cor_p=cor_matrix[1].copy()
        cor['module']=cor.index
        cor_p['module']=cor_p.index

        df_melt = pd.melt(cor, id_vars='module', var_name='meta', value_name='correlation')
        df_melt2 = pd.melt(cor_p, id_vars='module', var_name='meta', value_name='pvalue')
        df_melt['pvalue']=df_melt2['pvalue']
        df_melt['logp']=-np.log10(df_melt2['pvalue'])
        df_melt.index=[str(i) for i in df_melt.index]
        df_melt['module']=[str(i) for i in df_melt['module']]

        #new_keys = {'item_key': 'module','group_key': 'meta','sizes_key': 'logp','color_key': 'correlation'}
        
        try:
            import dotplot
        except ImportError:
            print("dotplot is not installed, please install it using `pip install python-dotplot`")
            return None
        new_keys = {'item_key': 'meta','group_key': 'module','sizes_key': 'logp','color_key': 'correlation'}
        dp = dotplot.DotPlot.parse_from_tidy_data(df_melt, **new_keys)
        #fig, ax = plt.subplots(figsize=(10,2))
        ax = dp.plot(size_factor=10, cmap='Spectral_r',vmax=1,vmin=0,
                    dot_title = '–log10(Pvalue)', colorbar_title = 'Correlation',)
        xlabs=ax.get_axes()[0].get_xticklabels()
        ax.get_axes()[0].set_xticklabels(xlabs,fontsize=label_fontsize)
        ylabs=ax.get_axes()[0].get_yticklabels()
        ax.get_axes()[0].set_yticklabels(ylabs,fontsize=label_fontsize)
        #去除配体
        [i.set_text(i.get_text().split('_')[1]) for i in xlabs]
        ax.get_axes()[0].set_xticklabels(xlabs,fontsize=label_fontsize)

        #上色        
        ax.get_axes()[0].tick_params(axis='both',colors=label_colors, which='both')
        return ax






    def __Analysis_cocharacter(self,
                        character,save=True):

        '''
        Calculate gene and trait correlation matrix

        Parameters
        ----------
        - character: `pandas.DataFrame`
            DataFrame of sample's character, columns=character, index=['sample1','sample2',...]
        - save: `bool`
            save the result

        Returns
        -------
        - result: `pandas.DataFrame`
            Character and module correlation matrix
        '''
        print("...PCA analysis have being done")
        data=self.data
        module=self.mol
        pcamol=pd.DataFrame(columns=data.columns)
        set_index=set(module['module'])
        for j in set_index:
            newdata=pd.DataFrame(columns=data.columns)
            for i in list(module[module['module']==j].dropna()['name']):
                newdata.loc[i]=data[data.index==i].values[0]
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1) 
            reduced_X = pca.fit_transform(newdata.T)
            tepcamol=pd.DataFrame(reduced_X.T,columns=data.columns)
            pcamol.loc[j]=tepcamol.values[0]
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
        
        plt.figure(figsize=(10,10))
        sns.heatmap(result_1,vmin=0, vmax=1,center=1,annot=True,square=True)
        if save==True:
            plt.savefig(self.save_path+'/'+'co_character.png',dpi=300)
        return result_1



def Trans_corr_matrix(data,
                    method='pearson',
                    cmap='seismic'):
    '''
    Calculate the correlation adjacent matrix (direct and indirect) (scale-free network)

    Parameters
    ----------
    - data: `pandas.DataFrame`
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=gene_name
    - method: `str`
        The method of calculating correlation coefficient
        method='pearson'/'kendall'/'spearman'
    - cmap: `str`
        color Style of drawing

    Returns
    ----------
    - result: `pandas.DataFrame`
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
    - data: `pandas.DataFrame`
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=gene_name
    - linkage_method: `str`
        The method of clustering
        method='ward'/'average'...
    - minClusterSize: `int`
        The least contained gene in each module
    - deepSplit: `int`
        Degree of split
    - cmap: `str`
        color style of figure

    Returns
    ----------
    - mol: `pandas.DataFrame`
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
    dynamicMods=dynamicTree.cutreeHybrid(geneTree,distM=distances,minClusterSize = minClusterSize,deepSplit = deepSplit, pamRespectsDendro = False)
    #return dynamicMods


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
    - data: `pandas.DataFrame`
        DataFrame of data points with each entry in the form:[sample1','sample2'...],index=gene_name
    - character: `pandas.DataFrame`
        DataFrame of sample's character, columns=character, index=['sample1','sample2',...]
    - module: `pandas.DataFrame`
        The result of the function Select_module

    Returns
    ----------
    - result: `pandas.DataFrame`
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