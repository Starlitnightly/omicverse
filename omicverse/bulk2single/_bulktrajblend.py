
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import torch
from typing import Union

from ..bulk import data_drop_duplicates_index,deseq2_normalize
from ..bulk2single import Bulk2Single
from ..single import scnocd
from ._vae import train_vae, generate_vae, load_vae

class BulkTrajBlend(object):
    """
    BulkTrajBlend: A class for bulk and single cell data integration and trajectory inference using beta-VAE and GNN.


    """

    def __init__(self,bulk_seq:pd.DataFrame,single_seq:anndata.AnnData,
                 celltype_key:str,bulk_group=None,max_single_cells:int=5000,
                 top_marker_num:int=500,ratio_num:int=1,gpu:Union[int,str]=0) -> None:
        """
        Initialize the BulkTrajBlend class

        Arguments:
            bulk_seq: The bulk data. The index is gene name and the columns is cell name.
            single_seq: The single cell data. The index is cell name and the columns is gene name.
            celltype_key: The key of cell type in the single cell data.
            top_marker_num: The number of top marker genes for each cell type.
            ratio_num: The number of cells to be selected for each cell type.
            gpu: The gpu id or 'cpu' or 'mps'.
            max_single_cells: The maximum number of single cells to be used. Default is 5000.

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
        Preprocess the bulk data

        Arguments:
            group: The group of the bulk data. Default is None. It need to set to calculate the mean of each group.
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
        Preprocess the single data

        Arguments:
            target_sum: The target sum of the normalize. Default is 1e4.

        """

        print("......normalize the single data")
        self.single_seq.obs_names_make_unique()
        self.single_seq.var_names_make_unique()
        sc.pp.normalize_total(self.single_seq, target_sum=target_sum)
        print("......log1p the single data")
        sc.pp.log1p(self.single_seq)
        return None
    
    def vae_configure(self,cell_target_num=None,**kwargs):
        """
        Configure the vae model

        Arguments:
            cell_target_num: The number of cell types to be generated. Default is 100.

        
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
        Train the VAE model of BulkTrajBlend.

        Arguments:
            vae_save_dir: The directory to save the trained VAE model. Default is 'save_model'.
            vae_save_name: The name to save the trained VAE model. Default is 'vae'.
            generate_save_dir: The directory to save the generated single-cell data. Default is 'output'.
            generate_save_name: The name to save the generated single-cell data. Default is 'output'.
            batch_size: The batch size for training the VAE model. Default is 512.
            learning_rate: The learning rate for training the VAE model. Default is 1e-4.
            hidden_size: The hidden size for the encoder and decoder networks. Default is 256.
            epoch_num: The epoch number for training the VAE model. Default is 5000.
            patience: The patience for training the VAE model. Default is 50.
            save: Whether to save the trained VAE model. Default is True.

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
        load the trained VAE model of BulkTrajBlend.

        Arguments:
            vae_load_dir: The directory to load the trained VAE model.
            hidden_size: The hidden size for the encoder and decoder networks. Default is 256.
        """

        print(f'loading model from {vae_load_dir}')
        vae_net = self.vae_model.load(vae_load_dir,hidden_size=hidden_size)
        self.vae_net=vae_net

    def vae_generate(self,highly_variable_genes:bool=True,max_value:float=10,
                     n_comps:int=100,svd_solver:str='auto',leiden_size:int=50)->anndata.AnnData:
        """
        Generate the single-cell data from the trained VAE model.

        Arguments:
            highly_variable_genes: Whether to use highly variable genes. Default is True.
            max_value: The maximum value for the scaled data. Default is 10.
            n_comps: The number of principal components. Default is 100.
            svd_solver: The solver for the PCA. Default is 'auto'.
            leiden_size: The minimum size of the leiden clusters. Default is 50.

        Returns:
            generate_adata: The generated single-cell data.

        """

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
    
    def gnn_configure(self,use_rep='X',neighbor_rep='X_pca',
                      gpu=0,hidden_size:int=128,
                     weight_decay:int=1e-2,
                     dropout:float=0.5,
                     batch_norm:bool=True,
                     lr:int=1e-3,
                     max_epochs:int=500,
                     display_step:int=25,
                     balance_loss:bool=True,
                     stochastic_loss:bool=True,
                     batch_size:int=2000,num_workers:int=5,):
        """
        Configure the GNN model of BulkTrajBlend.

        Arguments:
            gpu: The GPU ID for training the GNN model. Default is 0.
            hidden_size: The hidden size for the GNN model. Default is 128.
            weight_decay: The weight decay for the GNN model. Default is 1e-2.
            dropout: The dropout for the GNN model. Default is 0.5.
            batch_norm: Whether to use batch normalization for the GNN model. Default is True.
            lr: The learning rate for the GNN model. Default is 1e-3.
            max_epochs: The maximum epoch number for training the GNN model. Default is 500.
            display_step: The display step for training the GNN model. Default is 25.
            balance_loss: Whether to use the balance loss for training the GNN model. Default is True.
            stochastic_loss: Whether to use the stochastic loss for training the GNN model. Default is True.
            batch_size: The batch size for training the GNN model. Default is 2000.
            num_workers: The number of workers for training the GNN model. Default is 5.


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
        Train the GNN model of BulkTrajBlend.

        Arguments:
            thresh: The threshold for the GNN model. Default is 0.5.
            gnn_save_dir: The directory for saving the GNN model. Default is 'save_model'.
            gnn_save_name: The name for saving the GNN model. Default is 'gnn'.
        
        """
        self.nocd_obj.GNN_model()
        self.nocd_obj.GNN_result(thresh=thresh)
        self.nocd_obj.cal_nocd()
        self.nocd_obj.save(gnn_save_dir=gnn_save_dir,gnn_save_name=gnn_save_name)

    def gnn_load(self,gnn_load_dir:str,thresh:float=0.5,):
        """
        Load the GNN model of BulkTrajBlend.

        Arguments:
            gnn_load_dir: The directory for loading the GNN model.
            thresh: The threshold for the GNN model. Default is 0.5.
        
        """
        self.nocd_obj.load(gnn_load_dir)
        self.nocd_obj.GNN_result(thresh=thresh)
        self.nocd_obj.cal_nocd()
    
    def gnn_generate(self)->pd.DataFrame:
        """
        Generate the overlap cell community.

        Returns:
            res_pd: The overlap cell community.
        
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
        Interpolate the cell community to raw data.

        Arguments:
            celltype: The cell type for interpolation.
            adata: The raw data for interpolation. If is None, will use the single_seq data. Default is None.

        Returns:
            adata1: The adata after interpolated .
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
    


