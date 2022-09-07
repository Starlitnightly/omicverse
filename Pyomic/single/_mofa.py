from .. import mofapy2
from ..mofapy2.run.entry_point import entry_point
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
import h5py

def normalization(data):
    _range = np.max(abs(data))
    return data / _range

def get_weights(hdf5_path,view,factor,scale=True):
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
    f_pos = h5py.File(hdf5_path,'r')  
    for i in range(f_pos['expectations']['Z']['group0'].shape[0]):
        adata.obs['factor{0}'.format(i+1)]=f_pos['expectations']['Z']['group0'][i] 
    return adata

def factor_correlation(adata,cluster,factor_list,p_threshold=500):
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

    def __init__(self,omics,omics_name):
        self.omics=omics 
        self.omics_name=omics_name
        self.M=len(omics)

    def mofa_preprocess(self):
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
        ent1 = entry_point()
        ent1.set_data_options(
            scale_groups = False, 
            scale_views = False,
            center_groups=True,
        )
        ent1.set_data_matrix(self.data_mat, likelihoods = [i for i in ["gaussian"]*self.M],
            views_names=self.omics_name,
            samples_names=[self.omics[0].obs.index],
            features_names=self.feature_name)
        # set param
        ent1.set_model_options(
            factors = 20, 
            spikeslab_weights = True, 
            ard_factors = True,
            ard_weights = True
        )
        ent1.set_train_options(
            iter = 1000, 
            convergence_mode = "fast", 
            startELBO = 1, 
            freqELBO = 1, 
            dropR2 = 0.001, 
            gpu_mode = True, 
            verbose = False, 
            seed = 112
        )
        # 
        ent1.build()
        ent1.run()
        ent1.save(outfile=outfile)

    


