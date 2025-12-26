
import pandas as pd
import numpy as np
import anndata

import torch
from typing import Union, Optional, Any

from ..bulk import data_drop_duplicates_index, deseq2_normalize
from ..bulk2single import Bulk2Single
from ..single import scnocd
from ._vae import train_vae, generate_vae, load_vae

class BulkTrajBlend(object):
    r"""
    Bulk-to-single-cell trajectory blending using VAE and GNN integration.
    
    This class implements a comprehensive workflow for integrating bulk RNA-seq data
    with single-cell data to infer cellular trajectories and transition states.
    The method combines:
    - VAE-based bulk-to-single-cell deconvolution
    - Graph Neural Network (GNN) analysis for trajectory inference
    - Non-overlapping cell-type decomposition (NOCD) for transition states
    
    The workflow enables identification of intermediate cell states and trajectory
    dynamics that bridge bulk expression profiles with single-cell resolution.
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
        r"""
        Preprocess bulk RNA-seq data for trajectory analysis.
        
        Performs normalization, log transformation, and optional group averaging
        of bulk data for downstream trajectory inference.

        Arguments:
            None
            
        Returns:
            None: Updates self.bulk_seq and self.bulk_seq_group in place
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
        r"""
        Preprocess single-cell reference data for trajectory analysis.
        
        Normalizes single-cell data and makes cell/gene names unique for
        consistent integration with bulk data.

        Arguments:
            target_sum: Target sum for total count normalization (10000)
            
        Returns:
            None: Updates self.single_seq in place
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
        r"""
        Configure the VAE model for bulk-to-single-cell generation.

        Sets up the Bulk2Single model with cell-type target numbers either from
        deconvolution prediction or manual specification.

        Arguments:
            cell_target_num: Number of cells per type to generate. Default: None for auto-prediction
            **kwargs: Additional arguments passed to predicted_fraction method

        Returns:
            None
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
        r"""
        Train the VAE model for trajectory-aware single-cell generation.
        
        Trains the underlying Bulk2Single VAE model to generate synthetic single
        cells that preserve trajectory information from bulk data.

        Arguments:
            vae_save_dir: Directory to save trained VAE model ('save_model')
            vae_save_name: Filename for saved VAE model ('vae')
            generate_save_dir: Directory for generated data output ('output')
            generate_save_name: Filename for generated data ('output')
            batch_size: Training batch size (512)
            learning_rate: Optimizer learning rate (1e-4)
            hidden_size: Hidden layer dimensions (256)
            epoch_num: Maximum training epochs (5000)
            patience: Early stopping patience (50)
            save: Whether to save trained model (True)
            
        Returns:
            None: Updates self.vae_net with trained model
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
        r"""
        Load a pre-trained VAE model for trajectory analysis.
        
        Loads a previously trained VAE model for generating trajectory-aware
        single-cell data.

        Arguments:
            vae_load_dir: Directory containing the trained VAE model
            hidden_size: Hidden layer dimensions matching training (256)
            
        Returns:
            None: Updates self.vae_net with loaded model
        """

        print(f'loading model from {vae_load_dir}')
        vae_net = self.vae_model.load(vae_load_dir,hidden_size=hidden_size)
        self.vae_net=vae_net

    def vae_generate(self,highly_variable_genes:bool=True,max_value:float=10,
                     n_comps:int=100,svd_solver:str='auto',leiden_size:int=50)->anndata.AnnData:
        r"""
        Generate trajectory-aware single-cell data with quality filtering.
        
        Uses the trained VAE to generate synthetic single cells and applies
        quality control filtering to remove noisy clusters.

        Arguments:
            highly_variable_genes: Whether to select highly variable genes (True)
            max_value: Maximum value for data scaling (10)
            n_comps: Number of principal components for PCA (100)
            svd_solver: SVD solver for PCA ('auto')
            leiden_size: Minimum cluster size threshold for filtering (50)

        Returns:
            anndata.AnnData: Generated and filtered single-cell data
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
        r"""
        Configure Graph Neural Network for trajectory and transition state analysis.

        Sets up the NOCD (Non-Overlapping Cell-type Decomposition) GNN model
        for identifying cellular trajectories and intermediate states.

        Arguments:
            use_rep: Representation to use for GNN input. Default: 'X'
            neighbor_rep: Representation for neighbor graph construction. Default: 'X_pca'
            gpu: GPU device ID for training. Default: 0
            hidden_size: Hidden layer dimensions in GNN. Default: 128
            weight_decay: L2 regularization strength. Default: 1e-2
            dropout: Dropout probability. Default: 0.5
            batch_norm: Whether to use batch normalization. Default: True
            lr: Learning rate for GNN training. Default: 1e-3
            max_epochs: Maximum training epochs. Default: 500
            display_step: Frequency of progress updates. Default: 25
            balance_loss: Whether to use balanced loss function. Default: True
            stochastic_loss: Whether to use stochastic loss. Default: True
            batch_size: Training batch size. Default: 2000
            num_workers: Number of data loading workers. Default: 5

        Returns:
            None
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
        r"""
        Train the GNN model for trajectory and transition state inference.
        
        Trains the NOCD model to identify cellular trajectories and overlapping
        cell communities representing transition states.

        Arguments:
            thresh: Threshold for community assignment (0.5)
            gnn_save_dir: Directory to save trained GNN model ('save_model')
            gnn_save_name: Filename for saved GNN model ('gnn')
            
        Returns:
            None: Trains model and computes trajectory results
        """
        self.nocd_obj.GNN_model()
        self.nocd_obj.GNN_result(thresh=thresh)
        self.nocd_obj.cal_nocd()
        self.nocd_obj.save(gnn_save_dir=gnn_save_dir,gnn_save_name=gnn_save_name)

    def gnn_load(self,gnn_load_dir:str,thresh:float=0.5,):
        r"""
        Load a pre-trained GNN model for trajectory analysis.
        
        Loads a previously trained NOCD model and computes trajectory results.

        Arguments:
            gnn_load_dir: Directory containing the trained GNN model
            thresh: Threshold for community assignment (0.5)
            
        Returns:
            None: Loads model and computes trajectory results
        """
        self.nocd_obj.load(gnn_load_dir)
        self.nocd_obj.GNN_result(thresh=thresh)
        self.nocd_obj.cal_nocd()
    
    def gnn_generate(self)->pd.DataFrame:
        r"""
        Generate overlapping cell communities representing transition states.
        
        Identifies and names cell communities based on trajectory analysis,
        creating binary matrix indicating community membership for each cell.

        Arguments:
            None
            
        Returns:
            pd.DataFrame: Binary matrix of cell community assignments with community names
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
        r"""
        Interpolate trajectory communities back to original data space.
        
        Integrates identified cell communities from generated data back with
        original single-cell reference data for downstream analysis.

        Arguments:
            celltype: Cell type or community name to interpolate
            adata: Original data for interpolation; uses self.single_seq if None (None)

        Returns:
            anndata.AnnData: Combined data with interpolated community information
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
    


