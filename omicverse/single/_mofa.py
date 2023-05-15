from .. import mofapy2
from ..mofapy2.run.entry_point import entry_point
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
import h5py

def normalization(data):
    r"""
    Normalization for data.

    Parameters
    ----------
    - data : `numpy.ndarray`
        The data to be normalized.

    Returns
    -------
    - data : `numpy.ndarray`
        The normalized data.
    """
    _range = np.max(abs(data))
    return data / _range

def get_weights(hdf5_path,view,factor,scale=True):
    r"""
    Get the weights of each feature in a specific factor.

    Parameters
    ----------
    - hdf5_path : `str`
        The path of hdf5 file.
    - view : `str`  
        The name of view.
    - factor : `int`
        The number of factor.
    - scale : `bool`    
        Whether to scale the weights.
    
    Returns
    -------
    - res : `pandas.DataFrame`
        The weights of each feature in a specific factor.

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

def factor_exact(adata,hdf5_path):
    r"""
    Extract the factor information from hdf5 file.

    Parameters
    ----------
    - adata : `anndata.AnnData`
        The AnnData object.
    - hdf5_path : `str`
        The path of hdf5 file.
    
    Returns
    -------
    - adata : `anndata.AnnData`
        The AnnData object with factor information.

    """
    f_pos = h5py.File(hdf5_path,'r')  
    for i in range(f_pos['expectations']['Z']['group0'].shape[0]):
        adata.obs['factor{0}'.format(i+1)]=f_pos['expectations']['Z']['group0'][i] 
    return adata

def factor_correlation(adata,cluster,factor_list,p_threshold=500):
    r"""
    Calculate the correlation between factors and cluster.

    Parameters
    ----------
    - adata : `anndata.AnnData`
        The AnnData object.
    - cluster : `str`
        The name of cluster.
    - factor_list : `list`
        The list of factors.
    - p_threshold : `float` 
        The threshold of p-value.

    Returns 
    -------
    - cell_pd : `pandas.DataFrame`
        The correlation between factors and cluster.
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

class mofa(object):
    r"""
    MOFA class.
    """
    def __init__(self,omics,omics_name):
        r"""
        Initialize the MOFA class.
        
        Parameters
        ----------
        - omics : `list`
            The list of omics data.
        - omics_name : `list`   
            The list of omics name.
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

    def mofa_run(self,outfile='res.hdf5',factors=20,iter = 1000,convergence_mode = "fast",
                spikeslab_weights = True,startELBO = 1, freqELBO = 1, dropR2 = 0.001, gpu_mode = True, 
                verbose = False, seed = 112,scale_groups = False, scale_views = False,center_groups=True,):
        r"""
        Train the MOFA model.

        Parameters
        ----------
        - outfile : `str`
            The path of output file.
        - factors : `int`
            The number of factors.
        - iter : `int`
            The number of iterations.
        - convergence_mode : `str`
            The mode of convergence.
        - spikeslab_weights : `bool`
            Whether to use spikeslab weights.
        - startELBO : `int`
            The start of ELBO.
        - freqELBO : `int`
            The frequency of ELBO.
        - dropR2 : `float`
            The drop of R2.
        - gpu_mode : `bool`
            Whether to use gpu mode.
        - verbose : `bool`
            Whether to print the information.
        - seed : `int`
            The seed of random number.
        - scale_groups : `bool`
            Whether to scale groups.
        - scale_views : `bool`
            Whether to scale views.
        - center_groups : `bool`
            Whether to center groups.
        

        Returns
        -------
        - None

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

    


