import pandas as pd
import torch
import numpy as np
import anndata
import matplotlib.pyplot as plt
from ._utils import load_data, data_process,bulk2single_data_prepare
from ._vae import train_vae, generate_vae, load_vae
from ._map_utils import create_data, DFRunner, joint_analysis, knn,create_data_pyomic
import os
import warnings
import matplotlib
from typing import Union,Tuple
from .._registry import register_function
warnings.filterwarnings("ignore")

@register_function(
    aliases=['单细胞映射空间', 'Single2Spatial', 'single to spatial mapping'],
    category="bulk2single",
    description="Deep-learning mapper that projects single-cell profiles to spatial coordinates using a paired spatial reference.",
    prerequisites={'optional_functions': ['pp.preprocess']},
    requires={'obsm': ['spatial coordinates in spatial_data'], 'obs': ['celltype labels in single_data']},
    produces={'obsm': ['predicted spatial coordinates'], 'uns': ['single2spatial model']},
    auto_fix='none',
    examples=['st_model = ov.bulk2single.Single2Spatial(single_data=single_data, spatial_data=st_data, celltype_key="Cell_type")', 'st_map = st_model.train(spot_num=200, cell_num=8)'],
    related=['bulk2single.Bulk2Single', 'pl.plot_spatial']
)
class Single2Spatial(object):
    """
    Deep-learning mapper that projects single-cell profiles to spatial coordinates using a paired spatial reference
    
    Parameters
    ----------
    single_data : anndata.AnnData
        Configuration argument used when constructing `Single2Spatial`.
    spatial_data : anndata.AnnData
        Configuration argument used when constructing `Single2Spatial`.
    celltype_key : str
        Configuration argument used when constructing `Single2Spatial`.
    spot_key : list, optional, default=['xcoord','ycoord']
        Configuration argument used when constructing `Single2Spatial`.
    top_marker_num : Any, optional, default=500
        Configuration argument used when constructing `Single2Spatial`.
    marker_used : Any, optional, default=True
        Configuration argument used when constructing `Single2Spatial`.
    gpu : Union[int,str], optional, default=0
        Configuration argument used when constructing `Single2Spatial`.
    
    Returns
    -------
    None
        Initialize the class instance.
    
    Notes
    -----
    This class docstring follows the unified OmicVerse help template.
    
    Examples
    --------
    >>> st_model = ov.bulk2single.Single2Spatial(single_data=single_data, spatial_data=st_data, celltype_key="Cell_type")
    """

    def __init__(self,single_data:anndata.AnnData,
                 spatial_data:anndata.AnnData,
                 celltype_key:str,
                 spot_key:list=['xcoord','ycoord'],
                 top_marker_num=500,
                marker_used=True,gpu:Union[int,str]=0) -> None:
        r"""
        Initialize Single2Spatial model for mapping single cells to spatial coordinates.

        Arguments:
            single_data: Single-cell RNA-seq data as AnnData object
            spatial_data: Spatial transcriptomics reference data as AnnData object
            celltype_key: Column name in single_data.obs containing cell type annotations
            spot_key: Column names in spatial_data.obs for spatial coordinates (['xcoord','ycoord'])
            top_marker_num: Number of top marker genes to use in model (500)
            marker_used: Whether to use marker genes for training (True)
            gpu: GPU device ID; -1 for CPU, 'mps' for Apple Silicon (0)
            
        Returns:
            None
        """

        self.single_data = single_data
        self.spatial_data = spatial_data
        self.top_marker_num = top_marker_num
        self.marker_used = marker_used
        self.celltype_key=celltype_key
        if gpu=='mps' and torch.backends.mps.is_available():
            print('Note that mps may loss will be nan, used it when torch is supported')
            self.used_device = torch.device("mps")
        else:
            self.used_device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        self.history=[]
        self.input_data = create_data_pyomic(self.single_data,self.spatial_data,
                                             celltype_key,spot_key,)

    def train(self,spot_num:int,
                    cell_num:int,
                    df_save_dir:str='save_model',
                    df_save_name:str='df',
                    max_cell_in_diff_spot_ratio=None,
                    k:int=10,
                    random_seed:int=112,
                    mul_train:int=1,save=True,
                    n_jobs:int=1,num_epochs=1000,batch_size=1000,predicted_size=32)->anndata.AnnData:
        """
        Train the deep neural network for single-cell to spatial mapping
        
        Parameters
        ----------
        spot_num : int
            Input parameter for `train`.
        cell_num : int
            Input parameter for `train`.
        df_save_dir : str, optional, default='save_model'
            Input parameter for `train`.
        df_save_name : str, optional, default='df'
            Input parameter for `train`.
        max_cell_in_diff_spot_ratio : Any, optional, default=None
            Input parameter for `train`.
        k : int, optional, default=10
            Input parameter for `train`.
        random_seed : int, optional, default=112
            Input parameter for `train`.
        mul_train : int, optional, default=1
            Input parameter for `train`.
        save : Any, optional, default=True
            Input parameter for `train`.
        n_jobs : int, optional, default=1
            Input parameter for `train`.
        num_epochs : Any, optional, default=1000
            Input parameter for `train`.
        batch_size : Any, optional, default=1000
            Input parameter for `train`.
        predicted_size : Any, optional, default=32
            Input parameter for `train`.
        
        Returns
        -------
        anndata.AnnData
            Output produced by `train`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """
        # load data

        xtrain, ytrain = create_data(self.input_data['input_sc_meta'], 
                                     self.input_data['input_sc_data'], self.input_data["input_st_data"], 
                                     spot_num, cell_num,
                                     self.top_marker_num,
                                     self.marker_used, mul_train)
        df_runner = DFRunner(self.input_data['input_sc_data'], self.input_data['input_sc_meta'], 
                             self.input_data['input_st_data'], self.input_data['input_st_meta'], 
                             self.marker_used, self.top_marker_num, random_seed=random_seed,n_jobs=n_jobs,device=self.used_device)
        self.df_runner=df_runner
        df_meta, df_spot = self.df_runner.run(xtrain, ytrain, max_cell_in_diff_spot_ratio, k, df_save_dir, 
                                              df_save_name,num_epochs=num_epochs,batch_size=batch_size,predicted_size=predicted_size)
        
        if save:
            path_save = os.path.join(df_save_dir, f"{df_save_name}.pth")
            if not os.path.exists(df_save_dir):
                os.makedirs(df_save_dir)
            torch.save(df_runner.model.state_dict(), path_save)
            print(f"...save trained net in {path_save}.")

        sp_adata=anndata.AnnData(df_spot.T)
        sp_adata.obs=df_meta
        sp_adata.obs.set_index(sp_adata.obs['Cell'],inplace=True)
        sp_adata.obsm['X_spatial']=sp_adata.obs[['Cell_xcoord','Cell_ycoord']].values
        self.sp_adata=sp_adata
        return sp_adata
        #  save df
        os.makedirs(map_save_dir, exist_ok=True)
        meta_dir = os.path.join(map_save_dir, f'meta_{map_save_name}_{k}.csv')
        spot_dir = os.path.join(map_save_dir, f'data_{map_save_name}_{k}.csv')
        df_meta.to_csv(meta_dir)
        df_spot.to_csv(spot_dir)
        print(f"saving result to {meta_dir} and {spot_dir}")
        return df_meta, df_spot


        return df_meta, df_spot
    
    def save(self,df_save_dir:str='save_model',
                df_save_name:str='df',):
        """
        Save the trained Single2Spatial model
        
        Parameters
        ----------
        df_save_dir : str, optional, default='save_model'
            Input parameter for `save`.
        df_save_name : str, optional, default='df'
            Input parameter for `save`.
        
        Returns
        -------
        Any
            Output produced by `save`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """

        path_save = os.path.join(df_save_dir, f"{df_save_name}.pth")
        if not os.path.exists(df_save_dir):
            os.makedirs(df_save_dir)
        torch.save(self.df_runner.model.state_dict(), path_save)
        print(f"...save trained net in {path_save}.")
        #print("Model have been saved to "+os.path.join(df_save_dir, f"{df_save_name}"))

    
    def load(self,modelsize,
                    df_load_dir:str='save_model/df',
                    
                    max_cell_in_diff_spot_ratio=None,
                    k:int=10,
                    random_seed:int=112,
                    n_jobs:int=1,predicted_size=32)->anndata.AnnData:
        """
        Load a pre-trained Single2Spatial model and perform mapping
        
        Parameters
        ----------
        modelsize : Any
            Input parameter for `load`.
        df_load_dir : str, optional, default='save_model/df'
            Input parameter for `load`.
        max_cell_in_diff_spot_ratio : Any, optional, default=None
            Input parameter for `load`.
        k : int, optional, default=10
            Input parameter for `load`.
        random_seed : int, optional, default=112
            Input parameter for `load`.
        n_jobs : int, optional, default=1
            Input parameter for `load`.
        predicted_size : Any, optional, default=32
            Input parameter for `load`.
        
        Returns
        -------
        anndata.AnnData
            Output produced by `load`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """
        #xtrain, ytrain = create_data(self.input_data['input_sc_meta'], 
        #                             self.input_data['input_sc_data'], self.input_data["input_st_data"], 
        #                             spot_num, cell_num,
        #                             self.top_marker_num,
        #                             self.marker_used, mul_train)
        df_runner = DFRunner(self.input_data['input_sc_data'], self.input_data['input_sc_meta'], 
                             self.input_data['input_st_data'], self.input_data['input_st_meta'], 
                             self.marker_used, self.top_marker_num, random_seed=random_seed,n_jobs=n_jobs)
        self.df_runner=df_runner
        df_meta, df_spot = self.df_runner.load(df_load_dir,modelsize,max_cell_in_diff_spot_ratio, k,
                                              predicted_size=predicted_size)

        sp_adata=anndata.AnnData(df_spot.T)
        sp_adata.obs=df_meta
        sp_adata.obs.set_index(sp_adata.obs['Cell'],inplace=True)
        sp_adata.obsm['X_spatial']=sp_adata.obs[['Cell_xcoord','Cell_ycoord']].values
        self.sp_adata=sp_adata
        return sp_adata
        #  save df
        os.makedirs(map_save_dir, exist_ok=True)
        meta_dir = os.path.join(map_save_dir, f'meta_{map_save_name}_{k}.csv')
        spot_dir = os.path.join(map_save_dir, f'data_{map_save_name}_{k}.csv')
        df_meta.to_csv(meta_dir)
        df_spot.to_csv(spot_dir)
        print(f"saving result to {meta_dir} and {spot_dir}")
        return df_meta, df_spot
    
    def spot_assess(self)->anndata.AnnData:
        """
        Assess and aggregate predicted spatial data at the spot level
        
        Parameters
        ----------
        None
            This callable does not require explicit parameters.
        
        Returns
        -------
        anndata.AnnData
            Output produced by `spot_assess`.
        
        Notes
        -----
        This docstring follows the unified OmicVerse help template.
        """

        # spot-level
        # calculate cell type proportion per spot
        prop = self.sp_adata.obs[['Cell', 'Cell_type', 'Spot']].pivot_table(index=['Spot'], columns=['Cell_type'], aggfunc='count',values = 'Cell', fill_value=0)
        prop = prop.div(prop.sum(axis=1), axis=0)
        prop.columns = pd.Index(list(prop.columns))
        prop['Spot_xcoord'] = np.array(pd.pivot_table(self.sp_adata.obs,values='Spot_xcoord',index=['Spot'])['Spot_xcoord'].values)
        prop['Spot_ycoord'] = np.array(pd.pivot_table(self.sp_adata.obs,values='Spot_ycoord',index=['Spot'])['Spot_ycoord'].values)

        # aggregate gene expression per spot
        pred_spot_new = self.sp_adata.to_df()
        genes = pred_spot_new.columns
        pred_spot_new['Spot'] = self.sp_adata.obs['Spot']
        pred_spot_mean = pred_spot_new.groupby('Spot')[genes].mean()

        sp_adata_spot=anndata.AnnData(pred_spot_mean)
        sp_adata_spot.obs=prop
        sp_adata_spot.obsm['X_spatial']=sp_adata_spot.obs[['Spot_xcoord','Spot_ycoord']].values
        return sp_adata_spot

def spatial_mapping(generate_sc_meta,
                    generate_sc_data,
                    input_st_data_path,
                    input_st_meta_path,
                    map_save_dir='output',  # file_dir
                    map_save_name='map',  # file_name
                    ):
    r"""
    Map generated single-cell data to spatial coordinates using reference data.
    
    Performs spatial mapping by integrating generated single-cell data with
    reference spatial transcriptomics data using batch correction and k-nearest
    neighbor assignment.

    Arguments:
        generate_sc_meta: Metadata for generated single cells
        generate_sc_data: Expression data for generated single cells
        input_st_data_path: Path to spatial transcriptomics expression data
        input_st_meta_path: Path to spatial transcriptomics metadata
        map_save_dir: Directory to save mapping results ('output')
        map_save_name: Filename prefix for saved results ('map')

    Returns:
        tuple: (spatial_metadata, spatial_expression_data) containing mapped results
    """
    input_st_data = pd.read_csv(input_st_data_path, index_col=0)
    input_st_meta = pd.read_csv(input_st_meta_path, index_col=0)
    print('start to process image-based st data...')
    sc_gene_new = generate_sc_data._stat_axis.values.tolist()
    st_gene_new = input_st_data._stat_axis.values.tolist()
    intersect_gene_new = list(set(sc_gene_new).intersection(set(st_gene_new)))
    generate_sc_data_new = generate_sc_data.loc[intersect_gene_new]
    input_st_data_new = input_st_data.loc[intersect_gene_new]

    sc_cell_rename = [f'SC_{i}' for i in range(1, generate_sc_data_new.shape[1] + 1)]
    generate_sc_data.columns = generate_sc_data_new.columns = sc_cell_rename
    generate_sc_meta = generate_sc_meta.drop(['Cell'], axis=1)
    generate_sc_meta.insert(0, 'Cell', sc_cell_rename)
    generate_sc_meta['Batch'] = 'sc'
    generate_sc_meta_new = generate_sc_meta.drop(['Cell_type'], axis=1)
    st_cell_rename = [f'ST_{i}' for i in range(1, input_st_data_new.shape[1] + 1)]
    input_st_data.columns = input_st_data_new.columns = st_cell_rename
    input_st_meta = input_st_meta.drop(['Cell'], axis=1)
    input_st_meta.insert(0, 'Cell', st_cell_rename)
    input_st_meta_new = pd.DataFrame({'Cell': st_cell_rename, 'Batch': 'st'})

    all_data = generate_sc_data_new.join(input_st_data_new)
    all_meta = pd.concat([generate_sc_meta_new, input_st_meta_new], ignore_index=True)
    joint_data = joint_analysis(all_data, all_meta['Batch'], ref_batch="st")
    joint_data[joint_data < 0] = 0
    sc_data_new = joint_data.iloc[:, 0:generate_sc_data_new.shape[1]]
    st_data_new = joint_data.iloc[:, generate_sc_data_new.shape[1]:all_data.shape[1]]

    _, ind = knn(data=sc_data_new.T, query=st_data_new.T, k=10)

    st_data_pred = pd.DataFrame()
    st_meta_pred = pd.DataFrame(columns=['Cell', 'Cell_type'])

    for i in range(len(st_cell_rename)):
        st_data_pred[st_cell_rename[i]] = list(generate_sc_data.iloc[:, ind[i]].mean(axis=1))
        ct_tmp = list(generate_sc_meta.iloc[ind[i], :].Cell_type)
        ct_pred = max(ct_tmp, key=ct_tmp.count)
        st_meta_pred.loc[st_cell_rename[i]] = [st_cell_rename[i], ct_pred]

    st_data_pred.index = generate_sc_data.index
    st_meta_pred = pd.merge(st_meta_pred, input_st_meta, how='left', on='Cell')

    #  save df
    os.makedirs(map_save_dir, exist_ok=True)
    meta_dir = os.path.join(map_save_dir, f'meta_{map_save_name}.csv')
    data_dir = os.path.join(map_save_dir, f'data_{map_save_name}.csv')
    st_meta_pred.to_csv(os.path.join(meta_dir))
    st_data_pred.to_csv(os.path.join(data_dir))
    print(f'saving to {meta_dir} and {data_dir}')
    return st_meta_pred, st_data_pred