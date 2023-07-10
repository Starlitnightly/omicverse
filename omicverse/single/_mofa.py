from .. import mofapy2
from ..mofapy2.run.entry_point import entry_point
from ..utils import pyomic_palette
from ..single import get_celltype_marker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
import h5py
import anndata
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from typing import Union,Tuple
import matplotlib

mofax_install=False

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

def check_mofax():
    """
    
    """
    global mofax_install
    try:
        import mofax as mfx
        mofax_install=True
        print('mofax have been install version:',mfx.__version__)
    except ImportError:
        raise ImportError(
            'Please install the mofax: `pip install mofax`.'
        )

class GLUE_pair(object):

    def __init__(self,rna:anndata.AnnData,
              atac:anndata.AnnData) -> None:
        r"""
        Pair the cells between RNA and ATAC using result of GLUE.

        Arguments:
            rna: the AnnData of RNA-seq.
            atac: the AnnData of ATAC-seq.
            depth: the depth of the search for the nearest neighbor.
        
        """
        
        print('......Extract GLUE layer from obs')
        self.rna_loc=pd.DataFrame(rna.obsm['X_glue'], index=rna.obs.index)
        self.atac_loc=pd.DataFrame(atac.obsm['X_glue'], index=atac.obs.index)

    def correlation(self):
        """
        Perform Pearson Correlation analysis in the layer of GLUE
        
        """
        
        print('......Prepare for pair')
        import gc
        len1=(len(self.rna_loc)//5000)+1
        len2=(len(self.atac_loc)//5000)+1
        if len1>len2:
            len1=len2
        p_pd=pd.DataFrame(columns=['rank_'+str(i) for i in range(50)])
        n_pd=pd.DataFrame(columns=['rank_'+str(i) for i in range(50)])
        print('......Start to calculate the Pearson coef')
        for j in range(len1):
            c=pd.DataFrame()
            with trange(len1) as tt:
                for i in tt:
                    t1=self.rna_loc.iloc[5000*(i):5000*(i+1)]
                    t2=self.atac_loc.iloc[5000*(j):5000*(j+1)]
                    a=np.corrcoef(t1,t2)[len(t1):,0:len(t1)]
                    b=pd.DataFrame(a,index=t2.index,columns=t1.index)  
                    c=pd.concat([c,b],axis=1)
                    del t1
                    del t2
                    del a
                    del b
                    gc.collect()
                    tt.set_description('Now Pearson block is {}/{}'.format(i,len1))
            with trange(len(c)) as t:
                for i in t:
                    t_c=c.iloc[i]
                    p_pd.loc[t_c.name]=c.iloc[i].sort_values(ascending=False)[:50].values
                    n_pd.loc[t_c.name]=c.iloc[i].sort_values(ascending=False)[:50].index.tolist()
                    t.set_description('Now rna_index is {}/{}, all is {}'.format(i+j*5000,j*5000+len(c),len(self.atac_loc)))
            print('Now epoch is {}, {}/{}'.format(j,j*5000+len(c),len(self.atac_loc))) 
            del c
            gc.collect()
        self.rna_pd=p_pd
        self.atac_pd=n_pd

    def find_neighbor_cell(self,depth:int=10)->pd.DataFrame:
        """
        Find the neighbor cells between two omics using pearson
        
        Arguments:
            depth: the depth of the search for the nearest neighbor.

        Returns:
            result: the pair result

        """


        if depth>50:
            print('......depth limited to 50')
            depth=50
        rubish_c=[]
        finish_c=[]
        p_pd=self.rna_pd.copy()
        n_pd=self.atac_pd.copy()
        with trange(depth) as dt:
            for d in dt:
                p_pd=p_pd.loc[p_pd['rank_{}'.format(d)]>0.9]
                p_pd=p_pd.sort_values('rank_{}'.format(d),ascending=False)
                for i in p_pd.index:
                    name=n_pd.loc[i,'rank_{}'.format(d)]
                    if name not in rubish_c:
                        finish_c.append(i)
                        rubish_c.append(name)
                    else:
                        continue
                p_pd=p_pd.loc[~p_pd.index.isin(finish_c)]
                n_pd=n_pd.loc[~n_pd.index.isin(finish_c)]
                dt.set_description('Now depth is {}/{}'.format(d,depth))
        result=pd.DataFrame()
        result['omic_1']=rubish_c
        result['omic_2']=finish_c
        result.index=['cell_{}'.format(i) for i in range(len(result))]
        self.pair_res=result
        return result
    
    def pair_omic(self,omic1:anndata.AnnData,omic2:anndata.AnnData)->Tuple[anndata.AnnData,anndata.AnnData]:
        """
        Pair the omics using the result of find_neighbor_cell

        Arguments:
            omic1: the AnnData of omic1.
            omic2: the AnnData of omic2.

        Returns:
            rna1: the paired AnnData of omic1.
            atac1: the paired AnnData of omic2.

        """
        rna1=omic1[self.res_pair['omic_1']].copy()
        atac1=omic2[self.res_pair['omic_2']].copy()
        rna1.obs.index=self.res_pair.index
        atac1.obs.index=self.res_pair.index
        return rna1,atac1


def glue_pair(rna:anndata.AnnData,
              atac:anndata.AnnData,depth:int=20)->pd.DataFrame:
    r"""
    Pair the cells between RNA and ATAC using result of GLUE.

    Arguments:
        rna: the AnnData of RNA-seq.
        atac: the AnnData of ATAC-seq.
        depth: the depth of the search for the nearest neighbor.
    
    """


    #提取GLUE层结果
    print('......Extract GLUE layer from obs')
    rna_loc=pd.DataFrame(rna.obsm['X_glue'], index=rna.obs.index)
    atac_loc=pd.DataFrame(atac.obsm['X_glue'], index=atac.obs.index)

    #对GLUE层进行Pearson系数分析
    print('......Prepare for pair')
    import gc
    len1=(len(rna_loc)//5000)+1
    len2=(len(atac_loc)//5000)+1
    if len1>len2:
        len1=len2
    p_pd=pd.DataFrame(columns=['rank_'+str(i) for i in range(50)])
    n_pd=pd.DataFrame(columns=['rank_'+str(i) for i in range(50)])
    print('......Start to calculate the Pearson coef')
    for j in range(len1):
        c=pd.DataFrame()
        with trange(len1) as tt:
            for i in tt:
                t1=rna_loc.iloc[5000*(i):5000*(i+1)]
                t2=atac_loc.iloc[5000*(j):5000*(j+1)]
                a=np.corrcoef(t1,t2)[len(t1):,0:len(t1)]
                b=pd.DataFrame(a,index=t2.index,columns=t1.index)  
                c=pd.concat([c,b],axis=1)
                del t1
                del t2
                del a
                del b
                gc.collect()
                tt.set_description('Now Pearson block is {}/{}'.format(i,len1))
        with trange(len(c)) as t:
            for i in t:
                t_c=c.iloc[i]
                p_pd.loc[t_c.name]=c.iloc[i].sort_values(ascending=False)[:50].values
                n_pd.loc[t_c.name]=c.iloc[i].sort_values(ascending=False)[:50].index.tolist()
                t.set_description('Now rna_index is {}/{}, all is {}'.format(i+j*5000,i+j*5000+len(c),len(atac_loc)))
        print('Now epoch is {}, {}/{}'.format(j,j*5000+len(c),len(atac_loc))) 
        del c
        gc.collect()
        
    #寻找最近的细胞，其中depth的灵活调整可以使得配对成功的细胞数变大，同时精度有所下降
    def find_neighbor_cell(p_pd,n_pd,depth=10):
        if depth>50:
            print('......depth limited to 50')
            depth=50
        rubish_c=[]
        finish_c=[]
        with trange(depth) as dt:
            for d in dt:
                p_pd=p_pd.loc[p_pd['rank_{}'.format(d)]>0.9]
                p_pd=p_pd.sort_values('rank_{}'.format(d),ascending=False)
                for i in p_pd.index:
                    name=n_pd.loc[i,'rank_{}'.format(d)]
                    if name not in rubish_c:
                        finish_c.append(i)
                        rubish_c.append(name)
                    else:
                        continue
                p_pd=p_pd.loc[~p_pd.index.isin(finish_c)]
                n_pd=n_pd.loc[~n_pd.index.isin(finish_c)]
                dt.set_description('Now depth is {}/{}'.format(d,depth))
        result=pd.DataFrame()
        result['omic_1']=rubish_c
        result['omic_2']=finish_c
        result.index=['cell_{}'.format(i) for i in range(len(result))]
        return result
    print('......Start to find neighbor')
    res_pair=find_neighbor_cell(p_pd,n_pd,depth=depth)
    return res_pair

def normalization(data):
    r"""
    Normalization for data.

    Arguments:
        data: the data to be normalized.

    Returns:
        data: the normalized data.
    """
    _range = np.max(abs(data))
    return data / _range

def get_weights(hdf5_path:str,view:str,
                factor:int,scale:bool=True)->pd.DataFrame:
    r"""
    Get the weights of each feature in a specific factor.

    Arguments:
        hdf5_path: the path of hdf5 file.
        view: the name of view.
        factor: the number of factor.
        scale: whether to scale the weights.

    Returns:
        res: the weights of each feature in a specific factor.

    """
    f = h5py.File(hdf5_path,'r')  
    view_names=f['views']['views'][:]
    group_names=f['groups']['groups'][:]
    feature_names=np.array([f['features'][i][:] for i in view_names])
    sample_names=np.array([f['samples'][i][:] for i in group_names])
    f_name=feature_names[np.where(view_names==str.encode(view))[0][0]]
    f_w=f['expectations']['W'][view][factor-1]
    if scale==True:
        f_w=normalization(f_w)
    res=pd.DataFrame()
    res['feature']=f_name
    res['weights']=f_w
    res['abs_weights']=abs(f_w)
    res['sig']='+'
    res.loc[(res.weights<0),'sig'] = '-'

    return res

def factor_exact(adata:anndata.AnnData,hdf5_path:str)->anndata.AnnData:
    r"""
    Extract the factor information from hdf5 file.

    Arguments:
        adata: The AnnData object.
        hdf5_path: The path of hdf5 file.

    Returns:
        adata: The AnnData object with factor information.

    """
    f_pos = h5py.File(hdf5_path,'r')  
    for i in range(f_pos['expectations']['Z']['group0'].shape[0]):
        adata.obs['factor{0}'.format(i+1)]=f_pos['expectations']['Z']['group0'][i] 
    return adata

def factor_correlation(adata:anndata.AnnData,cluster:str,
                       factor_list:list,p_threshold:int=500)->pd.DataFrame:
    r"""
    Calculate the correlation between factors and cluster.

    Arguments:
        adata: The AnnData object.
        cluster: The name of cluster.
        factor_list: The list of factors.
        p_threshold: The threshold of p-value.

    Returns:
        cell_pd: The correlation between factors and cluster.

    """
    plot_data=adata.obs
    cell_t=list(set(plot_data[cluster]))
    cell_pd=pd.DataFrame(index=cell_t)
    for i in factor_list:
        test=[]
        for j in cell_t:
            a=plot_data[plot_data[cluster]==j]['factor'+str(i)].values
            b=plot_data[~(plot_data[cluster]==j)]['factor'+str(i)].values
            t, p = stats.ttest_ind(a,b)
            logp=-np.log(p)
            if(logp>p_threshold):
                logp=p_threshold
            test.append(logp)
        cell_pd['factor'+str(i)]=test
    return cell_pd

class pyMOFA(object):
    r"""
    MOFA class.
    """
    def __init__(self,omics:list,omics_name:list):
        r"""
        Initialize the MOFA class.
        
        Arguments:
            omics: The list of omics data.
            omics_name: The list of omics name.
        """
        self.omics=omics 
        self.omics_name=omics_name
        self.M=len(omics)

    def mofa_preprocess(self):
        r"""
        Preprocess the data.
        """
        self.data_mat=[[None for g in range(1)] for m in range(len(self.omics))]
        self.feature_name=[]
        for m in range(self.M):
            if issparse(self.omics[m].X)==True:
                self.data_mat[m][0]=self.omics[m].X.toarray()
            else:
                self.data_mat[m][0]=self.omics[m].X
            self.feature_name.append([self.omics_name[m]+'_'+i for i in self.omics[m].var.index])

    def mofa_run(self,outfile:str='res.hdf5',factors:int=20,iter:int = 1000,convergence_mode:str = "fast",
                spikeslab_weights:bool = True,startELBO:int = 1, freqELBO:int = 1, dropR2:float = 0.001, gpu_mode:bool = True, 
                verbose:bool = False, seed:int = 112,scale_groups:bool = False, 
                scale_views:bool = False,center_groups:bool=True,)->None:
        r"""
        Train the MOFA model.

        Arguments:
            outfile: The path of output file.
            factors: The number of factors.
            iter: The number of iterations.
            convergence_mode: The mode of convergence.
            spikeslab_weights: Whether to use spikeslab weights.
            startELBO: The start of ELBO.
            freqELBO: The frequency of ELBO.
            dropR2: The drop of R2.
            gpu_mode: Whether to use gpu mode.
            verbose: Whether to print the information.
            seed: The seed of random number.
            scale_groups: Whether to scale groups.
            scale_views: Whether to scale views.
            center_groups: Whether to center groups.

        """
        ent1 = entry_point()
        ent1.set_data_options(
            scale_groups = scale_groups, 
            scale_views = scale_views,
            center_groups=center_groups,
        )
        ent1.set_data_matrix(self.data_mat, likelihoods = [i for i in ["gaussian"]*self.M],
            views_names=self.omics_name,
            samples_names=[self.omics[0].obs.index],
            features_names=self.feature_name)
        # set param
        ent1.set_model_options(
            factors = factors, 
            spikeslab_weights = spikeslab_weights, 
            ard_factors = True,
            ard_weights = True
        )
        ent1.set_train_options(
            iter = iter, 
            convergence_mode = convergence_mode, 
            startELBO = startELBO, 
            freqELBO = freqELBO, 
            dropR2 = dropR2, 
            gpu_mode = gpu_mode, 
            verbose = verbose, 
            seed = seed
        )
        # 
        ent1.build()
        ent1.run()
        ent1.save(outfile=outfile)

    

class pyMOFAART(object):
    
    def __init__(self,model_path:str):
        """
        Initialize the MOFAART class.

        Arguments:
            model_path: The path of MOFA model.
        """
        check_mofax()
        global mofax_install
        if mofax_install==True:
            global_imports("mofax","mfx")
        
        self.model_path=model_path
        mfx_model=mfx.mofa_model(model_path)
        self.factors=mfx_model.get_factors()
        plot_data=pd.DataFrame()
        for i in mfx_model.get_r2()['View'].unique():
            plot_data[i]=mfx_model.get_r2().loc[mfx_model.get_r2()['View']==i,'R2'].values
        self.r2=plot_data
        mfx_model.close()
        
       
    def get_factors(self,adata:anndata.AnnData):
        """
        Get the factors of MOFA to anndata object.

        Arguments:
            adata: The anndata object.
        
        """
        print('......Add factors to adata and store to adata.obsm["X_mofa"]')
        adata.obsm['X_mofa']=self.factors
        adata=factor_exact(adata,hdf5_path=self.model_path)

    def get_r2(self,)->pd.DataFrame:
        """
        Get the varience of each factor

        Returns:
            r2: the varience of each factor
        """

        return self.r2

    def plot_r2(self,figsize:tuple=(2,3),cmap:str='Greens',
                ticks_fontsize:int=10,labels_fontsize:int=12,
                save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        plot the varience of each factor.

        Arguments:
            figsize: The size of figure.
            cmap: The color map.
            ticks_fontsize: The size of ticks.
            labels_fontsize: The size of labels.
            save: Whether to save the figure.

        Returns:
            fig: The figure of varience.
            ax: The axes of varience.
        
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(self.r2,cmap=cmap,ax=ax,xticklabels=True,yticklabels=True,
                    cbar_kws={'shrink':0.5})
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.ylabel('Factor',fontsize=labels_fontsize)
        plt.xlabel('View',fontsize=labels_fontsize)
        plt.title('Varience',fontsize=labels_fontsize)
        if save:
            fig.savefig("mofa_varience.png",dpi=300,bbox_inches = 'tight')
        return fig,ax
    
    def get_cor(self,adata:anndata.AnnData,cluster:str,factor_list=None)->pd.DataFrame:
        """
        get the correlation of each factor with cluster type in anndata object.

        Arguments:
            adata: The anndata object.
            cluster: The cluster type.
            factor_list: The list of factors.

        Returns:
            plot_data1: The correlation of each factor with cluster type.
        
        """

        if factor_list==None:
            factor_list=[i+1 for i in range(self.r2.shape[0])]
        plot_data1=factor_correlation(adata=adata,cluster=cluster,factor_list=factor_list)
        return plot_data1

    def plot_cor(self,adata:anndata.AnnData,cluster:str,factor_list=None,figsize:tuple=(6,3),
                 cmap:str='Purples',ticks_fontsize:int=10,labels_fontsize:int=12,title:str='Correlation',
                 save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Plot the correlation of each factor with cluster type in anndata object.

        Arguments:
            adata: The anndata object in MOFA pre trained.
            cluster: The cluster type in adata.obs.
            factor_list: The list of factors.
            figsize: The size of figure.
            cmap: The color map.
            ticks_fontsize: The font size of ticks.
            labels_fontsize: The font size of labels.
            title: The title of figure.
            save: Whether to save the figure.

        Returns:
            fig: The figure of correlation.
            ax: The axes of correlation.
        
        """

        if factor_list==None:
            factor_list=[i+1 for i in range(self.r2.shape[0])]
        plot_data1=factor_correlation(adata=adata,cluster=cluster,factor_list=factor_list)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(plot_data1,cmap=cmap,ax=ax,square=True,
                    cbar_kws={'shrink':0.5})
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.xlabel('Factor',fontsize=labels_fontsize)
        plt.ylabel(cluster,fontsize=labels_fontsize)
        plt.title(title,fontsize=labels_fontsize)
        if save:
            fig.savefig("mofa_cor.png",dpi=300,bbox_inches = 'tight')
        return fig,ax

    def plot_factor(self,adata:anndata.AnnData,cluster:str,title:str,figsize:tuple=(3,3),
                    factor1:int=1,factor2:int=2,palette:list=None,
                    save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Plot the factor of MOFA in anndata object.
        
        Arguments:
            adata: The anndata object.
            cluster: The cluster type in adata.obs.
            title: The title of figure.
            figsize: The size of figure.
            factor1: The first factor.
            factor2: The second factor.
            palette: The color map.
            save: Whether to save the figure.
        
        Returns:
            fig: The figure of factor.
            ax: The axes of factor.

        """

        if 'X_mofa' not in adata.obsm.keys():
            self.get_factors(adata)
        if palette==None:
            palette=pyomic_palette()
        fig, ax = plt.subplots(figsize=figsize)
        #factor1,factor2=4,6
        sc.pl.embedding(
            adata=adata,
            basis='X_mofa',
            color=cluster,
            title=title,
            components="{},{}".format(factor1,factor2),
            palette=palette,
            ncols=1,
            ax=ax
        )
        if save:
            fig.savefig("figures/mofa_factor_{}_{}.png".format(factor1,factor2),dpi=300,bbox_inches = 'tight')

        return fig,ax

    def plot_weight_gene_d1(self,view:str,factor1:int,factor2:int,
                            colors_dict:dict=None,plot_gene_num:int=5,title:str='',title_fontsize:int=12,
                            ticks_fontsize:int=12,labels_fontsize:int=12,
                            weith_threshold:float=0.5,figsize:tuple=(3,3),
                            save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Plot the weight of gene in each factor of MOFA in anndata object in dimension 1.

        Arguments:
            view: The view of MOFA.
            factor1: The first factor.
            factor2: The second factor.
            colors_dict: The color dict of up, down and normal. default is {'normal':'#c2c2c2','up':'#a51616','down':'#0d6a3b'}
            plot_gene_num: The number of genes to plot.
            title: The title of figure.
            title_fontsize: The font size of title.
            ticks_fontsize: The font size of ticks.
            labels_fontsize: The font size of labels.
            weith_threshold: The threshold of weight.
            figsize: The size of figure.
            save: Whether to save the figure.

        Returns:
            fig: The figure of weight.
            ax: The axes of weight.
        
        """
        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        #factor1,factor2=6,4
        plot_data3=factor_w[['factor_{}'.format(factor1),'factor_{}'.format(factor2)]]
        plot_data3['sig']='normal'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]>weith_threshold),'sig']='up'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]<-weith_threshold),'sig']='down'

        if colors_dict==None:
            colors_dict={'normal':'#c2c2c2','up':'#a51616','down':'#0d6a3b'}
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(plot_data3.loc[plot_data3['sig']=='normal','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='normal','factor_{}'.format(factor2)],
                color=colors_dict['normal'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='up','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='up','factor_{}'.format(factor2)],
                color=colors_dict['up'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='down','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='down','factor_{}'.format(factor2)],
                color=colors_dict['down'],alpha=0.5)

        plt.vlines(x=weith_threshold,ymin=-1,ymax=1,color=colors_dict['up'],linestyles='dashed')
        plt.hlines(y=weith_threshold,xmin=-1,xmax=1,color=colors_dict['up'],linestyles='dashed')

        plt.vlines(x=-weith_threshold,ymin=-1,ymax=1,color=colors_dict['down'],linestyles='dashed')
        plt.hlines(y=-weith_threshold,xmin=-1,xmax=1,color=colors_dict['down'],linestyles='dashed')

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        plt.grid(False)

        plt.xlabel('factor_{}'.format(factor1),fontsize=labels_fontsize)
        plt.ylabel('factor_{}'.format(factor2),fontsize=labels_fontsize)

        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)

        from adjustText import adjust_text

        for sig,color in zip(['up','down'],
                            [colors_dict['up'],colors_dict['down']]):
            if 'up' in sig:
                hub_gene=plot_data3.loc[plot_data3['sig']==sig].sort_values('factor_{}'.format(factor1),ascending=False).index.tolist()
            else:
                hub_gene=plot_data3.loc[plot_data3['sig']==sig].sort_values('factor_{}'.format(factor1),ascending=True).index.tolist()
            if len(hub_gene)==0:
                continue
            texts=[ax.text(plot_data3.loc[i,'factor_{}'.format(factor1)],
                        plot_data3.loc[i,'factor_{}'.format(factor2)],
                        i,
                        fontdict={'size':10,'weight':'bold','color':'black'}
                        ) for i in hub_gene[:plot_gene_num]]

            adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='grey'),)

        plt.title(title,fontsize=title_fontsize)
        if save:
            fig.savefig("factor_gene_{}.png".format(title),dpi=300,bbox_inches = 'tight')
        return fig,ax
    
    def plot_weight_gene_d2(self,view:str,factor1:int,factor2:int,
                            colors_dict:dict=None,plot_gene_num:int=5,title:str='',title_fontsize:int=12,
                            ticks_fontsize:int=12,labels_fontsize:int=12,
                            weith_threshold:float=0.5,figsize:tuple=(3,3),
                            save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Plot the weight of gene in each factor of MOFA in anndata object in dimension 2.

        Arguments:
            view: The view of MOFA.
            factor1: The first factor.
            factor2: The second factor.
            colors_dict: The color dict. default is {'up-up':'#a51616','up-down':'#e25d5d','down-up':'#1a6e1a','down-down':'#5de25d','normal':'#c2c2c2'}
            plot_gene_num: The number of genes to plot.
            title: The title of figure.
            title_fontsize: The font size of title.
            ticks_fontsize: The font size of ticks.
            labels_fontsize: The font size of labels.
            weith_threshold: The threshold of weight.
            figsize: The size of figure.
            save: Whether to save the figure.

        Returns:
            fig: The figure of weight.
            ax: The axes of weight.
        
        """
        
        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        #factor1,factor2=6,4
        plot_data3=factor_w[['factor_{}'.format(factor1),'factor_{}'.format(factor2)]]
        plot_data3['sig']='normal'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]>weith_threshold)&(plot_data3['factor_{}'.format(factor2)]>weith_threshold),'sig']='up-up'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]>weith_threshold)&(plot_data3['factor_{}'.format(factor2)]<-weith_threshold),'sig']='up-down'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]<-weith_threshold)&(plot_data3['factor_{}'.format(factor2)]>weith_threshold),'sig']='down-up'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]<-weith_threshold)&(plot_data3['factor_{}'.format(factor2)]<-weith_threshold),'sig']='down-down'


        if colors_dict==None:
            colors_dict={'up-up':'#a51616','up-down':'#e25d5d','down-up':'#1a6e1a','down-down':'#5de25d','normal':'#c2c2c2'}
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(plot_data3.loc[plot_data3['sig']=='normal','factor_{}'.format(factor1)],
           plot_data3.loc[plot_data3['sig']=='normal','factor_{}'.format(factor2)],
          color=colors_dict['normal'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='up-up','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='up-up','factor_{}'.format(factor2)],
                color=colors_dict['up-up'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='up-down','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='up-down','factor_{}'.format(factor2)],
                color=colors_dict['up-down'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='down-up','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='down-up','factor_{}'.format(factor2)],
                color=colors_dict['down-up'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='down-down','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='down-down','factor_{}'.format(factor2)],
                color=colors_dict['down-down'],alpha=0.5)

        plt.vlines(x=weith_threshold,ymin=-1,ymax=1,color=colors_dict['up-up'],linestyles='dashed')
        plt.hlines(y=weith_threshold,xmin=-1,xmax=1,color=colors_dict['up-up'],linestyles='dashed')

        plt.vlines(x=-weith_threshold,ymin=-1,ymax=1,color=colors_dict['down-down'],linestyles='dashed')
        plt.hlines(y=-weith_threshold,xmin=-1,xmax=1,color=colors_dict['down-down'],linestyles='dashed')

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        plt.grid(False)

        plt.xlabel('factor_{}'.format(factor1),fontsize=labels_fontsize)
        plt.ylabel('factor_{}'.format(factor2),fontsize=labels_fontsize)

        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)

        from adjustText import adjust_text

        for sig,color in zip(['up-up','up-down','down-up','down-down'],
                     [colors_dict['up-up'],colors_dict['up-down'],colors_dict['down-up'],colors_dict['down-down']]):
            hub_gene=plot_data3.loc[plot_data3['sig']==sig].index.tolist()
            if len(hub_gene)==0:
                continue
            texts=[ax.text(plot_data3.loc[i,'factor_{}'.format(factor1)],
                        plot_data3.loc[i,'factor_{}'.format(factor2)],
                        i,
                        fontdict={'size':10,'weight':'bold','color':'black'}
                        ) for i in hub_gene[:plot_gene_num]]

            adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='grey'),)

        plt.title(title,fontsize=title_fontsize)
        if save:
            fig.savefig("factor_gene_{}.png".format(title),dpi=300,bbox_inches = 'tight')
        return fig,ax

    def plot_weights(self,view:str,factor:int,color:str='#a51616',figsize:tuple=(3,4),
                     plot_gene_num:int=10,ascending:bool=False,
                    labels_fontsize:int=12,ticks_fontsize:int=12,title_fontsize:int=12,
                     title=None,save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Plot the weights of each gene in the factor

        Arguments:
            view: str, the view of the factor
            factor: int, the factor number
            color: str, the color of the plot
            figsize: tuple, the size of the figure
            plot_gene_num: int, the number of genes to plot
            ascending: bool, whether to sort the genes by weights
            labels_fontsize: int, the fontsize of the labels
            ticks_fontsize: int, the fontsize of the ticks
            title_fontsize: int, the fontsize of the title
            title: str, the title of the plot
            save: bool, whether to save the plot

        Returns:
            fig: the figure of the plot
            ax: the axis of the plot

        """

        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        fig, ax = plt.subplots(figsize=figsize)
        plot_data4=pd.DataFrame()
        plot_data4['weight']=factor_w['factor_{}'.format(factor)].sort_values(ascending=ascending)
        plot_data4['rank']=range(len(plot_data4['weight']))
        plt.plot(plot_data4['rank'],plot_data4['weight'],color=color)

        hub_gene=plot_data4.index[:plot_gene_num]
        plt.scatter(plot_data4.loc[hub_gene,'rank'],
                plot_data4.loc[hub_gene,'weight'],color=color,
                    alpha=0.5)

        from adjustText import adjust_text
        texts=[ax.text(plot_data4.loc[i,'rank'],
                        plot_data4.loc[i,'weight'],
                        i,
                        fontdict={'size':10,'weight':'normal','color':'black'}
                        ) for i in hub_gene]

        adjust_text(texts,only_move={'text': 'xy'},
                    arrowprops=dict(arrowstyle='->', color='grey'),)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        if title is None:
            plt.title('factor_{}'.format(factor),fontsize=title_fontsize,)
        else:
            plt.title(title,fontsize=title_fontsize,)
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.xlabel('Feature rank',fontsize=labels_fontsize)
        plt.ylabel('Weight',fontsize=labels_fontsize)

        plt.grid(False)
        if save:
            fig.savefig("factor{}_gene.png".format(factor),dpi=300,bbox_inches = 'tight')
        return fig,ax

    def plot_top_feature_dotplot(self,view:str,cmap:str='bwr',n_genes:int=3)->list:
        """
        Plot the top features of each factor in dotplot
        
        Arguments:
            view: str, the view of the factor
            cmap: str, the color map of the plot
            n_genes: int, the number of genes to plot

        Returns:
            axes: the list of the figure

        """

        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        adata1=anndata.AnnData(pd.concat([factor_w,factor_w],axis=1).T)
        adata1.obs['Factor']=adata1.obs.index
        adata1.obs['Factor']=adata1.obs['Factor'].astype('category')
        sc.tl.rank_genes_groups(adata1, groupby='Factor', method='wilcoxon')
        ax=sc.pl.rank_genes_groups_dotplot(adata1, n_genes=n_genes, 
                                        cmap=cmap,show=False)
        return ax
    
    def plot_top_feature_heatmap(self,view:str,cmap:str='bwr',n_genes:int=3)->list:
        """
        Plot the top features of each factor in dotplot
        
        Arguments:
            view: str, the view of the factor
            cmap: str, the color map of the plot
            n_genes: int, the number of genes to plot

        Returns:
            axes: the list of the figure

        """

        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        adata1=anndata.AnnData(pd.concat([factor_w,factor_w],axis=1).T)
        adata1.obs['Factor']=adata1.obs.index
        adata1.obs['Factor']=adata1.obs['Factor'].astype('category')
        sc.tl.rank_genes_groups(adata1, groupby='Factor', method='wilcoxon')
        ax=sc.pl.rank_genes_groups_matrixplot(adata1, n_genes=n_genes, 
                                        cmap=cmap,show=False)
        return ax
    
    def get_top_feature(self,view:str,log2fc_min:int=3,pval_cutoff:float=0.1)->dict:
        """
        Get the top features of each factor

        Arguments:
            view: str, the view of the factor
            log2fc_min: float, the minimum log2fc of the feature
            pval_cutoff: float, the maximum pval of the feature

        Returns:
            top_feature: dict, the top features of each factor

        """


        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        adata1=anndata.AnnData(pd.concat([factor_w,factor_w],axis=1).T)
        adata1.obs['Factor']=adata1.obs.index
        adata1.obs['Factor']=adata1.obs['Factor'].astype('category')
        top_feature=get_celltype_marker(adata1,clustertype='Factor',
                            log2fc_min=log2fc_min,pval_cutoff=pval_cutoff)
        return top_feature