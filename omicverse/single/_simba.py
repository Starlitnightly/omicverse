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
from .._registry import register_function

simba_install=False

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

@register_function(
    aliases=['SIMBA整合器', 'pySIMBA', 'single-cell batch integration simba'],
    category="single",
    description="SIMBA wrapper for batch correction and integrated manifold learning across single-cell datasets.",
    prerequisites={'optional_functions': ['pp.preprocess']},
    requires={'obs': ['batch labels']},
    produces={'obsm': ['X_simba'], 'uns': ['simba graph/model']},
    auto_fix='none',
    examples=['simba_object = ov.single.pySIMBA(adata, workdir)', 'simba_object.preprocess(batch_key="batch")'],
    related=['single.batch_correction', 'pp.neighbors', 'pp.umap']
)
class pySIMBA(object):
    """
    SIMBA wrapper for batch correction and integrated manifold learning across single-cell datasets
    
    Parameters
    ----------
    adata : Any
        Configuration argument used when constructing `pySIMBA`.
    workdir : Any, optional, default="simba_result"
        Configuration argument used when constructing `pySIMBA`.
    
    Returns
    -------
    None
        Initialize the class instance.
    
    Notes
    -----
    This class docstring follows the unified OmicVerse help template.
    
    Examples
    --------
    >>> simba_object = ov.single.pySIMBA(adata, workdir)
    """

    def check_simba(self):
        """
        Check if SIMBA package has been installed
        
        Parameters
        ----------
        None
            This callable does not require explicit parameters.
        
        Returns
        -------
        Any
            Output produced by `check_simba`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
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
        """
        Preprocess the AnnData object for SIMBA analysis
        
        Parameters
        ----------
        batch_key : Any, optional, default='batch'
            Input parameter for `preprocess`.
        min_n_cells : Any, optional, default=3
            Input parameter for `preprocess`.
        method : Any, optional, default='lib_size'
            Input parameter for `preprocess`.
        n_top_genes : Any, optional, default=3000
            Input parameter for `preprocess`.
        n_bins : Any, optional, default=5
            Input parameter for `preprocess`.
        
        Returns
        -------
        Any
            Output produced by `preprocess`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
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
        Generate the graph structure for batch correction
        
        Parameters
        ----------
        n_components : Any, optional, default=15
            Input parameter for `gen_graph`.
        k : Any, optional, default=15
            Input parameter for `gen_graph`.
        copy : Any, optional, default=False
            Input parameter for `gen_graph`.
        dirname : Any, optional, default='graph0'
            Input parameter for `gen_graph`.
        
        Returns
        -------
        Any
            Output produced by `gen_graph`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
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
        Train the SIMBA model for batch correction
        
        Parameters
        ----------
        num_workers : Any, optional, default=12
            Input parameter for `train`.
        auto_wd : Any, optional, default=True
            Input parameter for `train`.
        save_wd : Any, optional, default=True
            Input parameter for `train`.
        output : Any, optional, default='model'
            Input parameter for `train`.
        
        Returns
        -------
        Any
            Output produced by `train`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """

        # modify parameters
        dict_config = si.settings.pbg_params.copy()
        # dict_config['wd'] = 0.00477
        dict_config['workers'] = num_workers
        ## start training
        si.tl.pbg_train(pbg_params = dict_config, auto_wd=auto_wd, save_wd=save_wd, output=output)
        add_reference(self.adata,'SIMBA','batch correction with SIMBA')

    def load(self,model_path=None):
        """
        Load a pre-trained SIMBA model for batch correction
        
        Parameters
        ----------
        model_path : Any, optional, default=None
            Input parameter for `load`.
        
        Returns
        -------
        Any
            Output produced by `load`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
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
        Perform batch correction using the trained SIMBA model
        
        Parameters
        ----------
        use_precomputed : Any, optional, default=False
            Input parameter for `batch_correction`.
        
        Returns
        -------
        Any
            Output produced by `batch_correction`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
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
        
