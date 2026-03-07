
import pandas as pd
import numpy as np
import anndata

import torch
from typing import Union, Optional, Any

from ..bulk import data_drop_duplicates_index, deseq2_normalize
from ..bulk2single import Bulk2Single
from ..single import scnocd
from ._vae import train_vae, generate_vae, load_vae
from .._registry import register_function

@register_function(
    aliases=['Bulk轨迹融合', 'BulkTrajBlend', 'bulk trajectory blend'],
    category="bulk2single",
    description="Integrate bulk and single-cell signals to infer transitional cell states and trajectory dynamics through bulk-to-single reconstruction plus graph modeling.",
    prerequisites={'functions': ['Bulk2Single']},
    requires={'obs': ['celltype labels']},
    produces={'obs': ['trajectory-related annotations'], 'uns': ['trajectory blend results']},
    auto_fix='none',
    examples=['bulktb = ov.bulk2single.BulkTrajBlend(bulk_seq=bulk, single_seq=adata, celltype_key="celltype")', 'adata1 = bulktb.train(spot_num=100, cell_num=10)'],
    related=['bulk2single.Bulk2Single', 'utils.cal_paga', 'utils.plot_paga']
)
class BulkTrajBlend(object):
    """
    Integrate bulk and single-cell signals to infer transitional cell states and trajectory dynamics through bulk-to-single reconstruction plus graph modeling
    
    Parameters
    ----------
    bulk_seq : pd.DataFrame
        Configuration argument used when constructing `BulkTrajBlend`.
    single_seq : anndata.AnnData
        Configuration argument used when constructing `BulkTrajBlend`.
    celltype_key : str
        Configuration argument used when constructing `BulkTrajBlend`.
    bulk_group : Optional[Any], optional, default=None
        Configuration argument used when constructing `BulkTrajBlend`.
    max_single_cells : int, optional, default=5000
        Configuration argument used when constructing `BulkTrajBlend`.
    top_marker_num : int, optional, default=500
        Configuration argument used when constructing `BulkTrajBlend`.
    ratio_num : int, optional, default=1
        Configuration argument used when constructing `BulkTrajBlend`.
    gpu : Union[int, str], optional, default=0
        Configuration argument used when constructing `BulkTrajBlend`.
    
    Returns
    -------
    None
        Initialize the class instance.
    
    Notes
    -----
    This class docstring follows the unified OmicVerse help template.
    
    Examples
    --------
    >>> bulktb = ov.bulk2single.BulkTrajBlend(bulk_seq=bulk, single_seq=adata, celltype_key="celltype")
    """

    def __init__(self, bulk_seq: pd.DataFrame, single_seq: anndata.AnnData,
                 celltype_key: str, bulk_group: Optional[Any] = None, max_single_cells: int = 5000,
                 top_marker_num: int = 500, ratio_num: int = 1, gpu: Union[int, str] = 0) -> None:
        r"""
        Initialize BulkTrajBlend for trajectory inference and cell blending.

        Arguments:
            bulk_seq: Bulk RNA-seq data with genes as rows and samples as columns
            single_seq: Single-cell RNA-seq reference data as AnnData object
            celltype_key: Column name in single_seq.obs containing cell type annotations
            bulk_group: Column names in bulk_seq for sample grouping. Default: None
            max_single_cells: Maximum number of single cells to use. Default: 5000
            top_marker_num: Number of top marker genes per cell type. Default: 500
            ratio_num: Cell selection ratio for each cell type. Default: 1
            gpu: GPU device ID; -1 for CPU, 'mps' for Apple Silicon. Default: 0

        Returns:
            None
        """

        self.bulk_seq = bulk_seq.copy()
        self.single_seq = single_seq.copy()
        self.celltype_key=celltype_key
        self.top_marker_num=top_marker_num
        self.ratio_num=ratio_num
        self.gpu=gpu
        self.group=bulk_group
        self.max_single_cells=max_single_cells
        if gpu=='mps' and torch.backends.mps.is_available():
            print('Note that mps may loss will be nan, used it when torch is supported')
            self.used_device = torch.device("mps")
        else:
            self.used_device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        self.history=[]
        data_dg_v=self.bulk_seq.mean(axis=1)
        data_dg=pd.DataFrame(index=data_dg_v.index)
        data_dg['group']=data_dg_v
        self.bulk_seq_group=data_dg
        pass

    def bulk_preprocess_lazy(self,)->None:
        """
        Preprocess bulk RNA-seq data for trajectory analysis
        
        Parameters
        ----------
        None
            This callable does not require explicit parameters.
        
        Returns
        -------
        None
            Output produced by `bulk_preprocess_lazy`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """

        print("......drop duplicates index in bulk data")
        self.bulk_seq=data_drop_duplicates_index(self.bulk_seq)
        print("......deseq2 normalize the bulk data")
        self.bulk_seq=deseq2_normalize(self.bulk_seq)
        print("......log10 the bulk data")
        self.bulk_seq=np.log10(self.bulk_seq+1)
        print("......calculate the mean of each group")
        if self.group is None:
            return None
        else:
            data_dg_v=self.bulk_seq[self.group].mean(axis=1)
            data_dg=pd.DataFrame(index=data_dg_v.index)
            data_dg['group']=data_dg_v
            self.bulk_seq_group=data_dg
        return None
    
    def single_preprocess_lazy(self,target_sum:int=1e4)->None:
        """
        Preprocess single-cell reference data for trajectory analysis
        
        Parameters
        ----------
        target_sum : int, optional, default=1e4
            Input parameter for `single_preprocess_lazy`.
        
        Returns
        -------
        None
            Output produced by `single_preprocess_lazy`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """

        print("......normalize the single data")
        self.single_seq.obs_names_make_unique()
        self.single_seq.var_names_make_unique()
        from ..pp._preprocess import normalize_total,log1p
        normalize_total(self.single_seq, target_sum=target_sum)
        print("......log1p the single data")
        log1p(self.single_seq)
        return None
    
    def vae_configure(self, cell_target_num: Optional[int] = None, **kwargs: Any) -> None:
        """
        Configure the VAE model for bulk-to-single-cell generation
        
        Parameters
        ----------
        cell_target_num : Optional[int], optional, default=None
            Input parameter for `vae_configure`.
        **kwargs : Any
            Input parameter for `vae_configure`.
        
        Returns
        -------
        None
            Output produced by `vae_configure`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """
        self.vae_model=Bulk2Single(bulk_data=self.bulk_seq,single_data=self.single_seq,
                                   celltype_key=self.celltype_key,bulk_group=self.group,
                                      max_single_cells=self.max_single_cells,
                 top_marker_num=self.top_marker_num,ratio_num=self.ratio_num,gpu=self.gpu)
        if cell_target_num!=None:
            self.vae_model.cell_target_num=dict(zip(list(set(self.single_seq.obs[self.celltype_key])),
                                                [cell_target_num]*len(list(set(self.single_seq.obs[self.celltype_key])))))
        else:
            self.cellfract=self.vae_model.predicted_fraction(**kwargs)
        
        self.sc_ref=self.vae_model.sc_ref.copy()
        self.bulk_ref=self.vae_model.bulk_data.T.copy()

        self.vae_model.bulk_preprocess_lazy()
        self.vae_model.single_preprocess_lazy()
        self.vae_model.prepare_input()

        

    def vae_train(self,
                  vae_save_dir:str='save_model',
            vae_save_name:str='vae',
            generate_save_dir:str='output',
            generate_save_name:str='output',
            batch_size:int=512,
            learning_rate:int=1e-4,
            hidden_size:int=256,
            epoch_num:int=5000,
            patience:int=50,save:bool=True):
        """
        Train the VAE model for trajectory-aware single-cell generation
        
        Parameters
        ----------
        vae_save_dir : str, optional, default='save_model'
            Input parameter for `vae_train`.
        vae_save_name : str, optional, default='vae'
            Input parameter for `vae_train`.
        generate_save_dir : str, optional, default='output'
            Input parameter for `vae_train`.
        generate_save_name : str, optional, default='output'
            Input parameter for `vae_train`.
        batch_size : int, optional, default=512
            Input parameter for `vae_train`.
        learning_rate : int, optional, default=1e-4
            Input parameter for `vae_train`.
        hidden_size : int, optional, default=256
            Input parameter for `vae_train`.
        epoch_num : int, optional, default=5000
            Input parameter for `vae_train`.
        patience : int, optional, default=50
            Input parameter for `vae_train`.
        save : bool, optional, default=True
            Input parameter for `vae_train`.
        
        Returns
        -------
        Any
            Output produced by `vae_train`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """
        
        self.vae_net=self.vae_model.train(
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                epoch_num=epoch_num,
                vae_save_dir=vae_save_dir,
                vae_save_name=vae_save_name,
                generate_save_dir=generate_save_dir,
                generate_save_name=generate_save_name,
                patience=patience,save=save)
        
        
    def vae_load(self,vae_load_dir:str,hidden_size:int=256):
        """
        Load a pre-trained VAE model for trajectory analysis
        
        Parameters
        ----------
        vae_load_dir : str
            Input parameter for `vae_load`.
        hidden_size : int, optional, default=256
            Input parameter for `vae_load`.
        
        Returns
        -------
        Any
            Output produced by `vae_load`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """

        print(f'loading model from {vae_load_dir}')
        vae_net = self.vae_model.load(vae_load_dir,hidden_size=hidden_size)
        self.vae_net=vae_net

    def vae_generate(self,highly_variable_genes:bool=True,max_value:float=10,
                     n_comps:int=100,svd_solver:str='auto',leiden_size:int=50)->anndata.AnnData:
        """
        Generate trajectory-aware single-cell data with quality filtering
        
        Parameters
        ----------
        highly_variable_genes : bool, optional, default=True
            Input parameter for `vae_generate`.
        max_value : float, optional, default=10
            Input parameter for `vae_generate`.
        n_comps : int, optional, default=100
            Input parameter for `vae_generate`.
        svd_solver : str, optional, default='auto'
            Input parameter for `vae_generate`.
        leiden_size : int, optional, default=50
            Input parameter for `vae_generate`.
        
        Returns
        -------
        anndata.AnnData
            Output produced by `vae_generate`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """
        import scanpy as sc

        generate_adata=self.vae_model.generate()
        self.generate_adata_raw=generate_adata.copy()
        generate_adata.raw = generate_adata
        if highly_variable_genes:
            sc.pp.highly_variable_genes(generate_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            generate_adata = generate_adata[:, generate_adata.var.highly_variable]
        sc.pp.scale(generate_adata, max_value=max_value)
        sc.tl.pca(generate_adata, n_comps=n_comps, svd_solver=svd_solver)
        sc.pp.neighbors(generate_adata, use_rep="X_pca")
        sc.tl.leiden(generate_adata)
        filter_leiden=list(generate_adata.obs['leiden'].value_counts()[generate_adata.obs['leiden'].value_counts()<leiden_size].index)
        generate_adata.uns['noisy_leiden']=filter_leiden
        print("The filter leiden is ",filter_leiden)
        generate_adata=generate_adata[~generate_adata.obs['leiden'].isin(filter_leiden)]
        self.generate_adata=generate_adata.copy()
        return generate_adata.raw.to_adata()
    
    def gnn_configure(self, use_rep: str = 'X', neighbor_rep: str = 'X_pca',
                      gpu: Union[int, str] = 0, hidden_size: int = 128,
                     weight_decay: float = 1e-2,
                     dropout: float = 0.5,
                     batch_norm: bool = True,
                     lr: float = 1e-3,
                     max_epochs: int = 500,
                     display_step: int = 25,
                     balance_loss: bool = True,
                     stochastic_loss: bool = True,
                     batch_size: int = 2000, num_workers: int = 5) -> None:
        """
        Configure Graph Neural Network for trajectory and transition state analysis
        
        Parameters
        ----------
        use_rep : str, optional, default='X'
            Input parameter for `gnn_configure`.
        neighbor_rep : str, optional, default='X_pca'
            Input parameter for `gnn_configure`.
        gpu : Union[int, str], optional, default=0
            Input parameter for `gnn_configure`.
        hidden_size : int, optional, default=128
            Input parameter for `gnn_configure`.
        weight_decay : float, optional, default=1e-2
            Input parameter for `gnn_configure`.
        dropout : float, optional, default=0.5
            Input parameter for `gnn_configure`.
        batch_norm : bool, optional, default=True
            Input parameter for `gnn_configure`.
        lr : float, optional, default=1e-3
            Input parameter for `gnn_configure`.
        max_epochs : int, optional, default=500
            Input parameter for `gnn_configure`.
        display_step : int, optional, default=25
            Input parameter for `gnn_configure`.
        balance_loss : bool, optional, default=True
            Input parameter for `gnn_configure`.
        stochastic_loss : bool, optional, default=True
            Input parameter for `gnn_configure`.
        batch_size : int, optional, default=2000
            Input parameter for `gnn_configure`.
        num_workers : int, optional, default=5
            Input parameter for `gnn_configure`.
        
        Returns
        -------
        None
            Output produced by `gnn_configure`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """
        nocd_obj=scnocd(self.generate_adata,use_rep=use_rep,
                        neighbor_rep=neighbor_rep,gpu=gpu)
        #nocd_obj.device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        nocd_obj.matrix_transform(clustertype=self.celltype_key)
        nocd_obj.matrix_normalize()
        nocd_obj.GNN_configure(hidden_size=hidden_size,weight_decay=weight_decay,
                             dropout=dropout,batch_norm=batch_norm,lr=lr,
                             max_epochs=max_epochs,display_step=display_step,
                             balance_loss=balance_loss,stochastic_loss=stochastic_loss,
                             batch_size=batch_size)
        nocd_obj.GNN_preprocess(num_workers=num_workers)

        self.nocd_obj=nocd_obj
    
    def gnn_train(self,thresh:float=0.5,gnn_save_dir:str='save_model',
            gnn_save_name:str='gnn'):
        """
        Train the GNN model for trajectory and transition state inference
        
        Parameters
        ----------
        thresh : float, optional, default=0.5
            Input parameter for `gnn_train`.
        gnn_save_dir : str, optional, default='save_model'
            Input parameter for `gnn_train`.
        gnn_save_name : str, optional, default='gnn'
            Input parameter for `gnn_train`.
        
        Returns
        -------
        Any
            Output produced by `gnn_train`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """
        self.nocd_obj.GNN_model()
        self.nocd_obj.GNN_result(thresh=thresh)
        self.nocd_obj.cal_nocd()
        self.nocd_obj.save(gnn_save_dir=gnn_save_dir,gnn_save_name=gnn_save_name)

    def gnn_load(self,gnn_load_dir:str,thresh:float=0.5,):
        """
        Load a pre-trained GNN model for trajectory analysis
        
        Parameters
        ----------
        gnn_load_dir : str
            Input parameter for `gnn_load`.
        thresh : float, optional, default=0.5
            Input parameter for `gnn_load`.
        
        Returns
        -------
        Any
            Output produced by `gnn_load`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """
        self.nocd_obj.load(gnn_load_dir)
        self.nocd_obj.GNN_result(thresh=thresh)
        self.nocd_obj.cal_nocd()
    
    def gnn_generate(self)->pd.DataFrame:
        """
        Generate overlapping cell communities representing transition states
        
        Parameters
        ----------
        None
            This callable does not require explicit parameters.
        
        Returns
        -------
        pd.DataFrame
            Output produced by `gnn_generate`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """
        '''
        pair_dict_r={}
        for i in range(self.nocd_obj.K):
            j=0
            while 1:
                if self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_n']==str(i)].shape[0]==0:
                    break
                if j>=len(self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_n']==str(i)].obs.value_counts(self.celltype_key).index):
                    pair_dict_r[str(i)]=self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_n']==str(i)].obs.value_counts(self.celltype_key).index[j-1]+'_'+str(j)
                    break
                if self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_n']==str(i)].obs.value_counts(self.celltype_key).index[j] not in list(pair_dict_r.values()):
                    pair_dict_r[str(i)]=self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_n']==str(i)].obs.value_counts(self.celltype_key).index[j]
                    break
                else:
                    j+=1
        pair_dict_r
        '''
        unique_adata=self.nocd_obj.adata[~self.nocd_obj.adata.obs['nocd_n'].str.contains('-')]
        pair_dict_r={}
        repeat_celltype=dict(zip(list(set(unique_adata.obs[self.celltype_key])),np.zeros(len(list(set(unique_adata.obs[self.celltype_key]))))))
        for nocd_class in list(set(unique_adata.obs['nocd_n'])):
            now_celltype=unique_adata[unique_adata.obs['nocd_n']==nocd_class].obs.value_counts(self.celltype_key).index[0]
            if (now_celltype in pair_dict_r.values()):
                #print(now_celltype)
                pair_dict_r[str(nocd_class)]=now_celltype+'_'+str(int(repeat_celltype[now_celltype]))
                repeat_celltype[now_celltype]+=1
            else:
                pair_dict_r[str(nocd_class)]=now_celltype
                repeat_celltype[now_celltype]+=1
        
        # Ensure all keys from 0 to K-1 exist in pair_dict_r to prevent KeyError
        for j in range(self.nocd_obj.K):
            if str(j) not in pair_dict_r:
                pair_dict_r[str(j)] = f'unknown_{j}'

        def li_range(li,max_len):
            r=[0]*max_len   
            for i in li:
                r[int(i)]=1
            return r
        
        res_li=[li_range(i.split('-'),self.nocd_obj.K) for i in self.nocd_obj.adata.obs['nocd_n']]
        res_pd=pd.DataFrame(res_li,index=self.nocd_obj.adata.obs.index,columns=['nocd_'+i for i in [pair_dict_r[str(j)] for j in range(self.nocd_obj.K)]])
        print("The nocd result is ",res_pd.sum(axis=0))
        print("The nocd result has been added to adata.obs['nocd_']")
        self.nocd_obj.adata.obs=pd.concat([self.nocd_obj.adata.obs,res_pd],axis=1)
        return res_pd 
    
    def interpolation(self,celltype:str,adata:anndata.AnnData=None,)->anndata.AnnData:
        """
        Interpolate trajectory communities back to original data space
        
        Parameters
        ----------
        celltype : str
            Input parameter for `interpolation`.
        adata : anndata.AnnData, optional, default=None
            Input parameter for `interpolation`.
        
        Returns
        -------
        anndata.AnnData
            Output produced by `interpolation`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """
        if adata is None:
            adata=self.single_seq
        test_adata=self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_{}'.format(celltype)]==1].raw.to_adata()
        if test_adata.shape[0]!=0:
            adata1=anndata.concat([test_adata,
                        adata],merge='same')
        else:
            adata1=adata 
            print("The cell type {} is not in the nocd result".format(celltype))
        return adata1
    


