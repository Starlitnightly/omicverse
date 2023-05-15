import pandas as pd
import torch
import numpy as np
import anndata
import matplotlib.pyplot as plt
from ._utils import load_data, data_process,bulk2single_data_prepare
from ._vae import train_vae, generate_vae, load_vae
#from .map_utils import create_data, DFRunner, joint_analysis, knn
import os
import warnings
import matplotlib
from typing import Union,Tuple
warnings.filterwarnings("ignore")


class Bulk2Single:
    r"""
    Bulk2Single class.
    
    """
    def __init__(self,bulk_data:pd.DataFrame,single_data:anndata.AnnData,celltype_key:str,
                 top_marker_num:int=500,ratio_num:int=1,gpu:Union[int,str]=0):
        """
        Initializes the Bulk2Single class.

        Arguments:
            bulk_data: The bulk RNA-seq data.
            single_data: The single-cell RNA-seq data.
            celltype_key: The name of the column in the bulk data containing cell types.
            top_marker_num: The number of top markers to select per cell type. Default is 500.
            ratio_num: The ratio between the number of single cells and target number of converted cells. Default is 1.
            gpu: The ID of the GPU to use. Set to -1 to use CPU. Default is 0. If set to 'mps', the MPS backend will be used.

        """
        self.bulk_data=bulk_data
        self.single_data=single_data
        self.celltype_key=celltype_key
        self.input_data=bulk2single_data_prepare(bulk_data,single_data,celltype_key)
        self.cell_target_num = data_process(self.input_data, top_marker_num, ratio_num)
        if gpu=='mps' and torch.backends.mps.is_available():
            print('Note that mps may loss will be nan, used it when torch is supported')
            self.used_device = torch.device("mps")
        else:
            self.used_device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        self.history=[]

    def train(self,
            vae_save_dir:str='save_model',
            vae_save_name:str='vae',
            generate_save_dir:str='output',
            generate_save_name:str='output',
            batch_size:int=512,
            learning_rate:int=1e-4,
            hidden_size:int=256,
            epoch_num:int=5000,
            patience:int=50,save:bool=True)->torch.nn.Module:
        """
        Trains the VAE model.

        Arguments:
            vae_save_dir: The directory to save the trained VAE model. Default is 'save_model'.
            vae_save_name: The name of the saved VAE model. Default is 'vae'.
            generate_save_dir: The directory to save the generated single-cell data. Default is 'output'.
            generate_save_name: The name of the saved generated single-cell data. Default is 'output'.
            batch_size: The batch size for training. Default is 512.
            learning_rate: The learning rate for training. Default is 1e-4.
            hidden_size: The hidden size for the encoder and decoder networks. Default is 256.
            epoch_num: The maximum number of epochs for training. Default is 5.
            patience: The number of epochs to wait before early stopping. Default is 50.
            save: Whether to save the trained VAE model. Default is True.

        Returns:
            vae_net: The trained VAE model.
        """
        single_cell, label, breed_2_list, index_2_gene, cell_number_target_num, \
        nclass, ntrain, feature_size = self.__get_model_input(self.input_data, self.cell_target_num)
        print('...begin vae training')
        vae_net,history = train_vae(single_cell,
                            label,
                            self.used_device,
                            batch_size,
                            feature_size=feature_size,
                            epoch_num=epoch_num,
                            learning_rate=learning_rate,
                            hidden_size=hidden_size,
                            patience=patience,)
        print('...vae training done!')
        if save:
            path_save = os.path.join(vae_save_dir, f"{vae_save_name}.pth")
            if not os.path.exists(vae_save_dir):
                os.makedirs(vae_save_dir)
            torch.save(vae_net.state_dict(), path_save)
            print(f"...save trained vae in {path_save}.")
        self.vae_net=vae_net
        self.history=history
        return vae_net
        print('generating....')
        generate_sc_meta, generate_sc_data = generate_vae(vae_net, -1,
                                                          single_cell, label, breed_2_list,
                                                          index_2_gene, cell_number_target_num, used_device)
        sc_g=anndata.AnnData(generate_sc_data.T)
        sc_g.obs[self.celltype_key] = generate_sc_meta.loc[sc_g.obs.index,self.celltype_key].values
        sc_g.write_h5ad(os.path.join(generate_save_dir, f"{generate_save_name}.h5ad"), compression='gzip')
        self.__save_generation(generate_sc_meta, generate_sc_data,
                               generate_save_dir, generate_save_name)
        return sc_g
    
    def save(self,vae_save_dir:str='save_model',
            vae_save_name:str='vae',):
        """
        Saves the trained VAE model.

        Arguments:
            vae_save_dir: the directory to save the trained VAE model. Default is 'save_model'.
            vae_save_name: the name of the saved VAE model. Default is 'vae'.

        """
        path_save = os.path.join(vae_save_dir, f"{vae_save_name}.pth")
        if not os.path.exists(vae_save_dir):
            os.makedirs(vae_save_dir)
        torch.save(self.vae_net.state_dict(), path_save)
        print(f"...save trained vae in {path_save}.")
    
    def generate(self)->anndata.AnnData:
        r"""
        Generate the single-cell data.

        Returns:
            sc_g: The generated single-cell data.
        """
        single_cell, label, breed_2_list, index_2_gene, cell_number_target_num, \
        nclass, ntrain, feature_size = self.__get_model_input(self.input_data, self.cell_target_num)
        print('...generating')
        generate_sc_meta, generate_sc_data = generate_vae(self.vae_net, -1,
                                                          single_cell, label, breed_2_list,
                                                          index_2_gene, cell_number_target_num, self.used_device)
        generate_sc_meta.set_index('Cell',inplace=True)
        sc_g=anndata.AnnData(generate_sc_data.T)
        sc_g.obs[self.celltype_key] = generate_sc_meta.loc[sc_g.obs.index,'Cell_type'].values
        return sc_g
    
    def load(self,vae_load_dir:str,hidden_size:int=256):
        r"""
        load the trained VAE model of Bulk2Single.

        Arguments:
            vae_load_dir: The directory to load the trained VAE model.
            hidden_size: The hidden size for the encoder and decoder networks. Default is 256.
        """
        single_cell, label, breed_2_list, index_2_gene, cell_number_target_num, \
        nclass, ntrain, feature_size = self.__get_model_input(self.input_data, self.cell_target_num)
        print(f'loading model from {vae_load_dir}')
        vae_net = load_vae(feature_size, hidden_size, vae_load_dir, self.used_device)
        self.vae_net=vae_net

    def load_and_generate(self,
                              vae_load_dir:str,  # load_dir
                              hidden_size:int=256)->anndata.AnnData:
        r"""
        load the trained VAE model of Bulk2Single and generate the single-cell data.

        Arguments:
            vae_load_dir: The directory to load the trained VAE model.
            hidden_size: The hidden size for the encoder and decoder networks. Default is 256.

        Returns:
            sc_g: The generated single-cell data.
        """
        single_cell, label, breed_2_list, index_2_gene, cell_number_target_num, \
        nclass, ntrain, feature_size = self.__get_model_input(self.input_data, self.cell_target_num)
        print(f'loading model from {vae_load_dir}')
        vae_net = load_vae(feature_size, hidden_size, vae_load_dir, self.used_device)
        print('...generating')
        generate_sc_meta, generate_sc_data = generate_vae(vae_net, -1,
                                                          single_cell, label, breed_2_list,
                                                          index_2_gene, cell_number_target_num, self.used_device)
        generate_sc_meta.set_index('Cell',inplace=True)
        #return generate_sc_meta, generate_sc_data
        sc_g=anndata.AnnData(generate_sc_data.T)
        sc_g.obs[self.celltype_key] = generate_sc_meta.loc[sc_g.obs.index,'Cell_type'].values
        
        print('...generating done!')
        return sc_g
    
    def plot_loss(self,figsize:tuple=(4,4))->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        r"""
        plot the loss curve of the trained VAE model.

        Arguments:
            figsize: The size of the figure. Default is (4,4).

        Returns:
            fig: The figure of the loss curve.
            ax: The axes of the figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(len(self.history)),self.history)
        ax.set_title('Beta-VAE')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        return fig,ax


    def __get_model_input(self, data, cell_target_num):
        # input：data， celltype， bulk & output: label, dic, single_cell
        single_cell = data["input_sc_data"].values.T  # single cell data (600 * 6588)
        index_2_gene = (data["input_sc_data"].index).tolist()
        breed = data["input_sc_meta"]['Cell_type']
        breed_np = breed.values
        breed_set = set(breed_np)
        breed_2_list = list(breed_set)
        dic = {}  # breed_set to index {'B cell': 0, 'Monocyte': 1, 'Dendritic cell': 2, 'T cell': 3}
        label = []  # the label of cell (with index correspond)
        nclass = len(breed_set)

        ntrain = single_cell.shape[0]
        # FeaSize = single_cell.shape[1]
        feature_size = single_cell.shape[1]
        assert nclass == len(cell_target_num.keys()), "cell type num no match!!!"

        for i in range(len(breed_set)):
            dic[breed_2_list[i]] = i
        cell = data["input_sc_meta"]["Cell"].values

        for i in range(cell.shape[0]):
            label.append(dic[breed_np[i]])

        label = np.array(label)

        # label index the data size of corresponding target
        cell_number_target_num = {}
        for k, v in cell_target_num.items():
            cell_number_target_num[dic[k]] = v

        return single_cell, label, breed_2_list, index_2_gene, cell_number_target_num, nclass, ntrain, feature_size

    def __save_generation(self, generate_sc_meta, generate_sc_data, generate_save_dir,
                          generate_save_name, ):
        # saving.....
        if not os.path.exists(generate_save_dir):
            os.makedirs(generate_save_dir)
        path_label_generate_csv = os.path.join(generate_save_dir, f"{generate_save_name}_sc_celltype.csv")
        path_cell_generate_csv = os.path.join(generate_save_dir, f"{generate_save_name}_sc_data.csv")

        generate_sc_meta.to_csv(path_label_generate_csv)
        generate_sc_data.to_csv(path_cell_generate_csv)
        print(f"saving to {path_label_generate_csv} and {path_cell_generate_csv}.")
