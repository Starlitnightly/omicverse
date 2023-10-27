import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import os
import torch
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from tqdm import tqdm


simba_install=False

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

class pySIMBA(object):

    def check_simba(self):
        """
        Check if simba have been installed.
        """
        global simba_install
        try:
            import simba as si
            simba_install=True
            print('simba have been install version:',si.__version__)
        except ImportError:
            raise ImportError(
                """Please install the simba and simba_pdg: `conda install -c bioconda simba` or 
                `pip install git+https://github.com/huidongchen/simba and pip install git+https://github.com/pinellolab/simba_pbg`.'"""
            )

    def __init__(self,adata,workdir="simba_result") -> None:
        """
        SIMBA method for batch correction.

        Arguments:
            adata: AnnData object.
            workdir: The working directory for saving the results.
        
        """

        self.check_simba()
        global simba_install
        if simba_install==True:
            global_imports("simba","si")
        self.adata=adata
        self.workdir=workdir
        pass

    def preprocess(self,batch_key='batch',min_n_cells=3,
                    method='lib_size',n_top_genes=3000,n_bins=5):
        """
        The preprocess of the adata by simba

        Arguments:
            batch_key: The key of batch in adata.obs.
            min_n_cells: The minimum number of cells for a gene to be considered.
            method: The method for normalization.
            n_top_genes: The number of top genes to keep.
            n_bins: The number of bins for discretization.

        
        """
        adata=self.adata
        adata_dict={}
        adata.obs[batch_key]=adata.obs[batch_key].astype('category')
        for batch in adata.obs[batch_key].cat.categories:
            adata_dict[batch]=adata[adata.obs[batch_key]==batch].copy()
            si.pp.filter_genes(adata_dict[batch],min_n_cells=min_n_cells)
            si.pp.cal_qc_rna(adata_dict[batch])
            si.pp.normalize(adata_dict[batch],method=method)
            si.pp.log_transform(adata_dict[batch])
            si.pp.select_variable_genes(adata_dict[batch], n_top_genes=n_top_genes)
            si.tl.discretize(adata_dict[batch],n_bins=n_bins)
        adata.uns['simba_batch_dict']=adata_dict

    def gen_graph(self,n_components=15, k=15,
                    copy=False,dirname='graph0'):
        """
        Generate the graph for batch correction.

        Arguments:
            n_components: The number of components for the graph.
            k: The number of neighbors for the graph.
            copy: Whether to copy the adata.
            dirname: The name of the graph.
        
        """


        batch_size=dict(zip(list(self.adata.uns['simba_batch_dict'].keys()),
                            [self.adata.uns['simba_batch_dict'][i].shape[0] for i in self.adata.uns['simba_batch_dict'].keys()]))
        result_max = max(batch_size,key=lambda x:batch_size[x])
        adata_edge_dict={}
        for batch in self.adata.obs['batch'].cat.categories:
            if batch==result_max:
                continue
            else:
                adata_edge_dict[batch]=si.tl.infer_edges(self.adata.uns['simba_batch_dict'][result_max], 
                                                        self.adata.uns['simba_batch_dict'][batch], n_components=n_components, k=k)
        self.adata.uns['simba_batch_edge_dict']=adata_edge_dict
        si.tl.gen_graph(list_CG=[self.adata.uns['simba_batch_dict'][i] for i in self.adata.uns['simba_batch_dict'].keys()],
                    list_CC=[self.adata.uns['simba_batch_edge_dict'][i] for i in self.adata.uns['simba_batch_edge_dict'].keys()],
                    copy=copy,
                    dirname=dirname)
        
    def train(self,num_workers=12,auto_wd=True, save_wd=True, output='model'):
        """

        Train the model for batch correction.

        Arguments:
            num_workers: The number of workers for training.
            auto_wd: Whether to use the automatic weight decay.
            save_wd: Whether to save the weight decay.
            output: The output directory for saving the model.

        """

        # modify parameters
        dict_config = si.settings.pbg_params.copy()
        # dict_config['wd'] = 0.00477
        dict_config['workers'] = num_workers
        ## start training
        si.tl.pbg_train(pbg_params = dict_config, auto_wd=auto_wd, save_wd=save_wd, output=output)

    def load(self,model_path=None):
        """
        Load the model for batch correction.

        Arguments:
            model_path: The path of the model.

        
        """
        # load in graph ('graph0') info
        if model_path==None:
            si.load_graph_stats()
            si.load_pbg_config()
        else:
            si.load_graph_stats(path=model_path)
            si.load_pbg_config(path=model_path+'/model/')
       

    def batch_correction(self,use_precomputed=False):
        """
        Batch correction by SIMBA

        Arguments:
            use_precomputed: Whether to use the precomputed model.

        """
        dict_adata = si.read_embedding()
        adata_dict={}
        for batch,key in zip(self.adata.obs['batch'].cat.categories,range(1,len(dict_adata.keys()))):
            if key==1:
                key=''
            #adata.uns['simba_batch_dict'][batch].obsm['X_simba']=dict_adata['C{}'.format(key)][adata.uns['simba_batch_dict'][batch].obs.index].to_df().values
            adata_dict[batch]=dict_adata['C{}'.format(key)]
        self.adata.uns['simba_Gen']=adata_dict
        for batch in self.adata.obs['batch'].cat.categories:
            self.adata.uns['simba_Gen'][batch].obsm['X_simba']=self.adata.uns['simba_Gen'][batch].to_df().values
        
        # we choose the largest dataset as a reference
        batch_size_si = dict(zip(list(dict_adata.keys()),
                            [dict_adata[i].shape[0] for i in dict_adata.keys()]))
        adata_ref_si = dict_adata[max(batch_size_si, key=batch_size_si.get)]
        dict_adata.pop(max(batch_size_si, key=batch_size_si.get))
        list_adata_query = list(dict_adata.values())
        adata_all = si.tl.embed(adata_ref = adata_ref_si,
                            list_adata_query = list_adata_query,
                            use_precomputed = use_precomputed)
        
        #all_adata=anndata.concat([adata.uns['simba_Gen'][batch] for batch in adata.uns['simba_Gen'].keys()],merge='same')
        cell_idx=adata_all.obs.index
        self.adata=self.adata[cell_idx]
        self.adata.obsm['X_simba']=adata_all[cell_idx].to_df().values
        return self.adata
        