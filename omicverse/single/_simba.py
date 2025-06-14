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
from .._settings import add_reference

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
        r"""Check if SIMBA package has been installed.
        
        Raises:
            ImportError: If SIMBA package is not installed
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
        r"""Initialize SIMBA object for batch correction and multimodal analysis.

        Arguments:
            adata: AnnData object containing single-cell data
            workdir (str): Working directory for saving results (default: "simba_result")
        
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
        r"""Preprocess the AnnData object for SIMBA analysis.

        Arguments:
            batch_key (str): Key of batch information in adata.obs (default: 'batch')
            min_n_cells (int): Minimum number of cells for a gene to be considered (default: 3)
            method (str): Method for normalization (default: 'lib_size')
            n_top_genes (int): Number of top variable genes to keep (default: 3000)
            n_bins (int): Number of bins for discretization (default: 5)

        
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
        r"""Generate the graph structure for batch correction.

        Arguments:
            n_components (int): Number of components for dimensionality reduction (default: 15)
            k (int): Number of neighbors for graph construction (default: 15)
            copy (bool): Whether to copy the adata object (default: False)
            dirname (str): Directory name for saving graph results (default: 'graph0')
        
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
        r"""Train the SIMBA model for batch correction.

        Arguments:
            num_workers (int): Number of workers for parallel training (default: 12)
            auto_wd (bool): Whether to use automatic weight decay (default: True)
            save_wd (bool): Whether to save weight decay parameters (default: True)
            output (str): Output directory for saving the trained model (default: 'model')

        """

        # modify parameters
        dict_config = si.settings.pbg_params.copy()
        # dict_config['wd'] = 0.00477
        dict_config['workers'] = num_workers
        ## start training
        si.tl.pbg_train(pbg_params = dict_config, auto_wd=auto_wd, save_wd=save_wd, output=output)
        add_reference(self.adata,'SIMBA','batch correction with SIMBA')

    def load(self,model_path=None):
        r"""Load a pre-trained SIMBA model for batch correction.

        Arguments:
            model_path (str): Path to the model directory (default: None)
                            If None, loads from default location

        
        """
        # load in graph ('graph0') info
        if model_path==None:
            si.load_graph_stats()
            si.load_pbg_config()
        else:
            si.load_graph_stats(path=model_path)
            si.load_pbg_config(path=model_path+'/model/')
       

    def batch_correction(self,use_precomputed=False):
        r"""Perform batch correction using the trained SIMBA model.

        Arguments:
            use_precomputed (bool): Whether to use precomputed embeddings (default: False)

        Returns:
            AnnData: Batch-corrected AnnData object with X_simba embedding
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
        dict_adata2 = dict_adata.copy()
        dict_adata2.pop("G") # remove genes
        batch_size_si = dict(zip(list(dict_adata2.keys()),
                [dict_adata2[i].shape[0] for i in dict_adata2.keys()]))
        max_dict_label = max(batch_size_si, key=batch_size_si.get)
        adata_ref_si = dict_adata2[max_dict_label] # select largest dataset
        dict_adata2.pop(max_dict_label) # remove largest dataset
        list_adata_query = list(dict_adata2.values())
        adata_all = si.tl.embed(adata_ref = adata_ref_si,
                            list_adata_query = list_adata_query,
                            use_precomputed = use_precomputed)
        
        #all_adata=anndata.concat([adata.uns['simba_Gen'][batch] for batch in adata.uns['simba_Gen'].keys()],merge='same')
        cell_idx=adata_all.obs.index
        self.adata=self.adata[cell_idx]
        self.adata.obsm['X_simba']=adata_all[cell_idx].to_df().values
        return self.adata
        