import pandas as pd
import torch
import numpy as np
import anndata
import matplotlib.pyplot as plt
from ._utils import load_data, data_process, bulk2single_data_prepare
from ._vae import train_vae, generate_vae, load_vae
from ..bulk import data_drop_duplicates_index, deseq2_normalize
#from .map_utils import create_data, DFRunner, joint_analysis, knn
import os
import warnings
import matplotlib
from typing import Union, Tuple, Optional, Any

warnings.filterwarnings("ignore")


class Bulk2Single:
    r"""
    Main class for bulk-to-single-cell deconvolution.
    
    This class implements a VAE-based approach to generate single-cell data from bulk RNA-seq
    data by leveraging reference single-cell data. The method uses deconvolution to predict
    cell-type proportions and then generates synthetic single-cell data matching the bulk
    expression profile.
    
    The workflow includes:
    - Cell fraction prediction using deconvolution methods (SCADEN, TAPE)
    - VAE training for single-cell data generation
    - Quality filtering and analysis of generated cells
    """
    def __init__(self, bulk_data: pd.DataFrame, single_data: anndata.AnnData,
                 celltype_key: str, bulk_group: Optional[Any] = None, max_single_cells: int = 5000,
                 top_marker_num: int = 500, ratio_num: int = 1, gpu: Union[int, str] = 0):
        r"""
        Initialize the Bulk2Single class for bulk-to-single-cell deconvolution.

        Arguments:
            bulk_data: Bulk RNA-seq expression data as DataFrame with genes as rows and samples as columns
            single_data: Reference single-cell RNA-seq data as AnnData object
            celltype_key: Column name in single_data.obs containing cell type annotations
            bulk_group: Column names in bulk_data for grouping samples (None)
            max_single_cells: Maximum number of single cells to use from reference (5000)
            top_marker_num: Number of top marker genes to select per cell type (500)
            ratio_num: Ratio between single cells and target converted cells (1)
            gpu: GPU device ID for computation; -1 for CPU, 'mps' for Apple Silicon (0)

        """
        single_data.var_names_make_unique()
        bulk_data=data_drop_duplicates_index(bulk_data)
        self.bulk_data=bulk_data
        self.single_data=single_data
        
        if self.single_data.shape[0]>max_single_cells:
            print(f"......random select {max_single_cells} single cells")
            import random
            cell_idx=random.sample(self.single_data.obs.index.tolist(),max_single_cells)
            self.single_data=self.single_data[cell_idx,:]
        self.celltype_key=celltype_key
        self.bulk_group=bulk_group
        self.input_data=None
        #self.input_data=bulk2single_data_prepare(bulk_data,single_data,celltype_key)
        #self.cell_target_num = data_process(self.input_data, top_marker_num, ratio_num)

        test2=single_data.to_df()
        sc_ref=pd.DataFrame(columns=test2.columns)
        sc_ref_index=[]
        for celltype in list(set(single_data.obs[celltype_key])):
            sc_ref.loc[celltype]=single_data[single_data.obs[celltype_key]==celltype].to_df().sum()
            sc_ref_index.append(celltype)
        sc_ref.index=sc_ref_index
        self.sc_ref=sc_ref


        if gpu=='mps' and torch.backends.mps.is_available():
            print('Note that mps may loss will be nan, used it when torch is supported')
            self.used_device = torch.device("mps")
        else:
            self.used_device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        self.history=[]

    def predicted_fraction(self,method='scaden',sep='\t', scaler='mms',
                        datatype='counts', genelenfile=None,
                        mode='overall', adaptive=True, variance_threshold=0.98,
                        save_model_name=None,
                        batch_size=128, epochs=128, seed=1,scale_size=2):
        r"""
        Predict cell-type fractions from bulk RNA-seq data using deconvolution.
        
        Uses machine learning approaches (SCADEN or TAPE) to estimate cell-type
        proportions in bulk samples based on single-cell reference data.

        Arguments:
            method: Deconvolution method to use - 'scaden' or 'tape' ('scaden')
            sep: Separator for input files ('\t')
            scaler: Scaling method for normalization ('mms')
            datatype: Type of input data - 'counts' or 'tpm' ('counts')
            genelenfile: Gene length file for TPM calculation (None)
            mode: Analysis mode for TAPE method ('overall')
            adaptive: Whether to use adaptive variance threshold (True)
            variance_threshold: Variance threshold for feature selection (0.98)
            save_model_name: Name for saving trained model (None)
            batch_size: Batch size for training (128)
            epochs: Number of training epochs (128)
            seed: Random seed for reproducibility (1)
            scale_size: Scaling factor for cell number prediction (2)

        Returns:
            pd.DataFrame: Predicted cell-type fractions for each bulk sample
        """
        from ..external.tape import Deconvolution,ScadenDeconvolution
        sc_ref=self.sc_ref.copy()
        if method=='scaden':
            CellFractionPrediction=ScadenDeconvolution(sc_ref, 
                           self.bulk_data.T, sep=sep,
                           batch_size=batch_size, epochs=epochs, scaler=scaler)
        elif method=='tape':
            SignatureMatrix, CellFractionPrediction = \
                Deconvolution(sc_ref, self.bulk_data.T, sep=sep, scaler=scaler,
                            datatype=datatype, genelenfile=genelenfile,
                            mode=mode, adaptive=adaptive, variance_threshold=variance_threshold,
                            save_model_name=save_model_name,
                            batch_size=batch_size, epochs=epochs, seed=seed)
        else:
            raise ValueError('method must be scaden or tape')
        if self.bulk_group!=None:
            cell_total_num=self.single_data.shape[0]*self.bulk_data[self.bulk_group].mean(axis=1).sum()/self.single_data.to_df().sum().sum()
            print('Predicted Total Cell Num:',cell_total_num)
            predicted_fractions=CellFractionPrediction.loc[self.bulk_group].mean()*cell_total_num*scale_size
        else:
            cell_total_num=self.single_data.shape[0]*self.bulk_data.mean(axis=1).sum()/self.single_data.to_df().sum().sum()
            print('Predicted Total Cell Num:',cell_total_num)
            predicted_fractions=CellFractionPrediction.mean()*cell_total_num*scale_size
            
        # Get actual cell types from single-cell reference data
        actual_cell_types = list(set(self.single_data.obs[self.celltype_key]))
        predicted_cell_types = predicted_fractions.index.tolist()
        
        # Create cell_target_num dictionary with matching cell type names
        self.cell_target_num = {}
        
        # First, try exact matching
        for cell_type in actual_cell_types:
            if cell_type in predicted_cell_types:
                self.cell_target_num[cell_type] = int(predicted_fractions[cell_type])
        
        # For cell types not found in predictions, use average of predicted values
        if len(self.cell_target_num) < len(actual_cell_types):
            avg_fraction = int(predicted_fractions.mean())
            for cell_type in actual_cell_types:
                if cell_type not in self.cell_target_num:
                    print(f"Warning: Cell type '{cell_type}' not found in deconvolution predictions. Using average value: {avg_fraction}")
                    self.cell_target_num[cell_type] = avg_fraction
                    
        # Ensure minimum of 1 cell per type to avoid zero cell generation
        for cell_type in self.cell_target_num:
            if self.cell_target_num[cell_type] <= 0:
                self.cell_target_num[cell_type] = 1
        
        return CellFractionPrediction

    def bulk_preprocess_lazy(self,)->None:
        r"""
        Preprocess bulk RNA-seq data for deconvolution.
        
        Performs normalization, log transformation, and group averaging of bulk data.
        Steps include duplicate removal, DESeq2 normalization, log10 transformation,
        and optional group-wise averaging.

        Arguments:
            None
            
        Returns:
            None: Updates self.bulk_seq_group in place
        """

        print("......drop duplicates index in bulk data")
        self.bulk_data=data_drop_duplicates_index(self.bulk_data)
        print("......deseq2 normalize the bulk data")
        self.bulk_data=deseq2_normalize(self.bulk_data)
        print("......log10 the bulk data")
        self.bulk_data=np.log10(self.bulk_data+1)
        print("......calculate the mean of each group")
        if self.bulk_group is None:
            self.bulk_seq_group=self.bulk_data
            return None
        else:
            data_dg_v=self.bulk_data[self.bulk_group].mean(axis=1)
            data_dg=pd.DataFrame(index=data_dg_v.index)
            data_dg['group']=data_dg_v
            self.bulk_seq_group=data_dg
        return None
    
    def single_preprocess_lazy(self,target_sum:int=1e4)->None:
        r"""
        Preprocess single-cell reference data.
        
        Normalizes single-cell data using scanpy's standard preprocessing pipeline
        including total count normalization and log1p transformation.

        Arguments:
            target_sum: Target sum for total count normalization (10000)
            
        Returns:
            None: Updates self.single_data in place
        """

        print("......normalize the single data")
        from ..pp._preprocess import normalize_total,log1p
        normalize_total(self.single_data, target_sum=target_sum)
        print("......log1p the single data")
        log1p(self.single_data)
        return None
    
    def prepare_input(self,):
        r"""
        Prepare input data for VAE training.
        
        Formats and aligns bulk and single-cell data for training the VAE model.
        This step matches genes between datasets and prepares the data structure
        needed for model training.

        Arguments:
            None
            
        Returns:
            None: Updates self.input_data
        """
        print("......prepare the input of bulk2single")
        self.input_data=bulk2single_data_prepare(self.bulk_seq_group,
                                                 self.single_data,
                                                 self.celltype_key)


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
        r"""
        Train the VAE model for single-cell data generation.
        
        Trains a beta-VAE model to learn the mapping from bulk to single-cell expression
        patterns. The model learns to generate synthetic single cells that match the
        bulk expression profile and predicted cell-type proportions.

        Arguments:
            vae_save_dir: Directory to save trained VAE model ('save_model')
            vae_save_name: Filename for saved VAE model ('vae')
            generate_save_dir: Directory for generated data output ('output')
            generate_save_name: Filename for generated data ('output')
            batch_size: Training batch size (512)
            learning_rate: Optimizer learning rate (1e-4)
            hidden_size: Hidden layer dimensions in encoder/decoder (256)
            epoch_num: Maximum training epochs (5000)
            patience: Early stopping patience (50)
            save: Whether to save trained model (True)

        Returns:
            torch.nn.Module: Trained VAE model
        """
        if self.input_data==None:
            self.prepare_input()
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
            import pickle
            #save cell_target_num
            with open(os.path.join(vae_save_dir, f"{vae_save_name}_cell_target_num.pkl"), 'wb') as f:
                pickle.dump(self.cell_target_num, f)
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
    
    def save(self, vae_save_dir: str = 'save_model',
            vae_save_name: str = 'vae') -> None:
        r"""
        Save the trained VAE model and cell target numbers.

        Saves both the model state dict and the predicted cell-type target numbers
        needed for generation.

        Arguments:
            vae_save_dir: Directory to save the trained VAE model. Default: 'save_model'
            vae_save_name: Filename for the saved VAE model. Default: 'vae'

        Returns:
            None
        """
        path_save = os.path.join(vae_save_dir, f"{vae_save_name}.pth")
        if not os.path.exists(vae_save_dir):
            os.makedirs(vae_save_dir)
        torch.save(self.vae_net.state_dict(), path_save)
        import pickle
        #save cell_target_num
        with open(os.path.join(vae_save_dir, f"{vae_save_name}_cell_target_num.pkl"), 'wb') as f:
            pickle.dump(self.cell_target_num, f)
        print(f"...save trained vae in {path_save}.")
    
    def generate(self)->anndata.AnnData:
        r"""
        Generate synthetic single-cell data from trained VAE model.
        
        Uses the trained VAE to generate single-cell expression profiles that match
        the bulk expression and predicted cell-type proportions.

        Arguments:
            None
            
        Returns:
            anndata.AnnData: Generated single-cell data with cell type annotations
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
    
    def load_fraction(self, fraction_path: str) -> None:
        r"""
        Load predicted cell-type target numbers from file.

        Loads previously computed cell-type target numbers that specify how many
        cells of each type to generate.

        Arguments:
            fraction_path: Path to pickled file containing cell target numbers

        Returns:
            None
        """
        #load cell_target_num
        import pickle
        with open(os.path.join(fraction_path), 'rb') as f:
            self.cell_target_num = pickle.load(f)
        
    
    def load(self,vae_load_dir:str,hidden_size:int=256):
        r"""
        Load a pre-trained VAE model.
        
        Loads a previously trained VAE model from disk for generating single-cell data.

        Arguments:
            vae_load_dir: Directory containing the trained VAE model
            hidden_size: Hidden layer dimensions matching training configuration (256)
            
        Returns:
            None: Updates self.vae_net
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
        Load pre-trained VAE model and generate single-cell data.
        
        Convenience method that loads a trained model and immediately generates
        synthetic single-cell data.

        Arguments:
            vae_load_dir: Directory containing the trained VAE model
            hidden_size: Hidden layer dimensions matching training configuration (256)

        Returns:
            anndata.AnnData: Generated single-cell data with cell type annotations
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
    
    def filtered(self,generate_adata,highly_variable_genes:bool=True,max_value:float=10,
                     n_comps:int=100,svd_solver:str='auto',leiden_size:int=50):
        r"""
        Filter generated single-cell data by removing low-quality clusters.
        
        Applies quality control filtering to generated single-cell data by identifying
        and removing small clusters that may represent noise or artifacts.

        Arguments:
            generate_adata: Generated single-cell AnnData object to filter
            highly_variable_genes: Whether to select highly variable genes (True)
            max_value: Maximum value for scaling (10)
            n_comps: Number of principal components (100)
            svd_solver: SVD solver for PCA ('auto')
            leiden_size: Minimum cluster size threshold for filtering (50)
            
        Returns:
            anndata.AnnData: Filtered single-cell data
        """
        generate_adata.raw = generate_adata
        import scanpy as sc
        if highly_variable_genes:
            sc.pp.highly_variable_genes(generate_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            generate_adata = generate_adata[:, generate_adata.var.highly_variable]
        sc.pp.scale(generate_adata, max_value=max_value)
        sc.tl.pca(generate_adata, n_comps=n_comps, svd_solver=svd_solver)
        sc.pp.neighbors(generate_adata, use_rep="X_pca")
        sc.tl.leiden(generate_adata)
        filter_leiden=list(generate_adata.obs['leiden'].value_counts()[generate_adata.obs['leiden'].value_counts()<leiden_size].index)
        print("The filter leiden is ",filter_leiden)
        generate_adata=generate_adata[~generate_adata.obs['leiden'].isin(filter_leiden)]
        self.generate_adata=generate_adata.copy()

        return generate_adata
    
    def plot_loss(self,figsize:tuple=(4,4))->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        r"""
        Plot training loss curve of the VAE model.
        
        Visualizes the training loss progression to assess model convergence.

        Arguments:
            figsize: Figure dimensions as (width, height) (4, 4)

        Returns:
            matplotlib.figure.Figure: Figure object containing the plot
            matplotlib.axes.Axes: Axes object for the plot
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
