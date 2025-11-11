"""Module for integrating spatial transcriptomics data across different conditions.

This module implements STAligner, a method for integrating spatial transcriptomics data
across different experimental conditions, technologies, and developmental stages. The
integration preserves both gene expression patterns and spatial organization.

Key features:
1. Spatial network construction
2. Graph neural network-based integration
3. Mutual nearest neighbor alignment
4. Batch effect correction
5. Cross-condition comparison

References:
    Zhou, X., Dong, K. & Zhang, S. Integrating spatial transcriptomics data 
    across different conditions, technologies and developmental stages. 
    Nat Comput Sci 3, 894–906 (2023)
"""
__author__ = "Xiang Zhou"
__email__ = "xzhou@amss.ac.cn"
__citation__ = "Zhou, X., Dong, K. & Zhang, S. Integrating spatial transcriptomics data across different conditions, technologies and developmental stages. Nat Comput Sci 3, 894–906 (2023)"

from ..external.STAligner.mnn_utils import create_dictionary_mnn
from ..external.STAligner.STALIGNER import STAligner

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import sklearn.neighbors
import random

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .._settings import add_reference
from ..utils.registry import register_function


@register_function(
    aliases=["空间网络构建", "Cal_Spatial_Net", "spatial_network", "空间邻域网络", "构建空间图"],
    category="space",
    description="Construct spatial neighbor networks for spatial transcriptomics integration",
    prerequisites={
        'optional_functions': []
    },
    requires={
        'obsm': ['spatial']  # Spatial coordinates required
    },
    produces={
        'uns': ['Spatial_Net', 'adj']
    },
    auto_fix='none',
    examples=[
        "# Radius-based spatial network",
        "ov.space.Cal_Spatial_Net(adata, rad_cutoff=150, model='Radius')",
        "# K-nearest neighbor network",
        "ov.space.Cal_Spatial_Net(adata, k_cutoff=6, model='KNN')",
        "# Custom parameters",
        "ov.space.Cal_Spatial_Net(adata, rad_cutoff=200, max_neigh=100,",
        "                         model='Radius', verbose=True)",
        "# Access network results",
        "spatial_graph = adata.uns['Spatial_Net']",
        "adjacency_matrix = adata.uns['adj']"
    ],
    related=["space.pySTAligner", "space.clusters", "space.pySTAGATE"]
)
def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):
    r"""Construct spatial neighbor networks for spatial integration.
    
    This function builds a spatial neighborhood graph by connecting spots based
    on their physical distances. It supports both radius-based and k-nearest
    neighbor approaches for network construction.

    Arguments:
        adata: AnnData
            Input spatial data containing:
            - Spatial coordinates in obsm['spatial']
            - Gene expression data in X
        rad_cutoff: float, optional (default=None)
            Maximum distance between neighbors for 'Radius' model.
            Only used when model='Radius'.
        k_cutoff: int, optional (default=None)
            Number of nearest neighbors to connect for 'KNN' model.
            Only used when model='KNN'.
        max_neigh: int, optional (default=50)
            Maximum number of neighbors to consider during graph construction.
            Helps limit memory usage for large datasets.
        model: str, optional (default='Radius')
            Network construction method:
            - 'Radius': Connect spots within rad_cutoff distance
            - 'KNN': Connect k_cutoff nearest neighbors
        verbose: bool, optional (default=True)
            Whether to print network statistics.
        
    Returns:
        None
            Updates adata with:
            - adata.uns['Spatial_Net']: DataFrame of edges and distances
            - adata.uns['adj']: Sparse adjacency matrix

    Notes:
        - For STAligner, adjust rad_cutoff to ensure 5-10 neighbors per spot
        - Includes self-loops in adjacency matrix
        - Uses ball_tree algorithm for efficient neighbor search
        - Memory efficient implementation for large datasets
        - Critical for downstream integration tasks
    """
    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)

    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :],distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print(f'The graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{(Spatial_Net.shape[0] / adata.n_obs):.4f} neighbors per cell on average.')
    adata.uns['Spatial_Net'] = Spatial_Net

    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G


@register_function(
    aliases=["STAligner空间整合", "pySTAligner", "STAligner", "空间数据整合", "空间转录组整合"],
    category="space",
    description="STAligner for integrating spatial transcriptomics data across conditions and technologies",
    prerequisites={
        'functions': ['Cal_Spatial_Net'],
        'optional_functions': []
    },
    requires={
        'obsm': ['spatial'],
        'obs': [],  # Requires batch_name column (user-specified)
        'uns': ['Spatial_Net', 'adj']  # From Cal_Spatial_Net
    },
    produces={
        'obsm': ['STAligner', 'STAligner_embed']
    },
    auto_fix='auto',
    examples=[
        "# Basic STAligner integration",
        "staligner = ov.space.pySTAligner(adata, batch_name='batch',",
        "                                 device='cuda:0')",
        "staligner.train()",
        "adata_integrated = staligner.train_STAligner()",
        "# Custom parameters",
        "staligner = ov.space.pySTAligner(adata, batch_name='condition',",
        "                                 k_cutoff=6, device='cpu')",
        "# Multi-batch integration",
        "staligner.train_STAligner(epochs=800, lr=0.001, weight_decay=1e-4)",
        "# Access integrated results",
        "integrated_embedding = adata.obsm['STAligner']",
        "batch_corrected = adata.obsm['STAligner_embed']"
    ],
    related=["space.Cal_Spatial_Net", "space.clusters", "space.svg"]
)
class pySTAligner(object):
    r"""STAligner for spatial transcriptomics data integration.
    
    STAligner is a deep learning method for integrating spatial transcriptomics
    data across different experimental conditions, technologies, and developmental
    stages. It combines graph neural networks with mutual nearest neighbors to
    preserve both transcriptional and spatial relationships during integration.

    The method works by:
    1. Constructing spatial neighborhood graphs
    2. Learning batch-invariant embeddings
    3. Aligning similar regions across batches
    4. Preserving spatial organization
    5. Enabling cross-condition comparison

    Attributes:
        adata: AnnData
            Combined data containing all batches
        model: STAligner
            Neural network model for integration
        loader: DataLoader
            PyTorch geometric data loader
        device: torch.device
            Computing device (GPU/CPU)
        optimizer: torch.optim.Optimizer
            Adam optimizer for training

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load data
        >>> adata1 = sc.read_visium(...)
        >>> adata2 = sc.read_visium(...)
        >>> # Construct spatial networks
        >>> ov.space.Cal_Spatial_Net(adata1, rad_cutoff=100)
        >>> ov.space.Cal_Spatial_Net(adata2, rad_cutoff=100)
        >>> # Combine data
        >>> adata = adata1.concatenate(adata2)
        >>> # Initialize STAligner
        >>> staligner = ov.space.pySTAligner(
        ...     adata=adata,
        ...     batch_key='batch',
        ...     Batch_list=[adata1, adata2]
        ... )
        >>> # Train model
        >>> staligner.train()
        >>> # Get integrated embeddings
        >>> embeddings = staligner.predicted()
    """
    
    def __init__(self,adata,
                 hidden_dims: list = [512, 30],
                 n_epochs: int = 1000,
                 lr: float = 0.001,
                 batch_key: str = 'batch_name',
                 key_added: str = 'STAligner',
                 gradient_clipping: float = 5,
                 weight_decay: float = 0.0001,
                 margin: float = 1,
                 verbose: bool = False,
                 random_seed: int = 666,
                 iter_comb = None,
                 knn_neigh: int = 100,
                 Batch_list = None,
                 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
             ) -> None:
        r"""Initialize STAligner spatial integration model.
        
        This method sets up the STAligner model by:
        1. Processing input data
        2. Constructing graph neural networks
        3. Initializing optimization parameters
        4. Preparing batch alignment strategy

        Arguments:
            adata: AnnData
                Combined data containing all batches to integrate.
                Must have batch information in obs[batch_key].
            hidden_dims: list, optional (default=[512, 30])
                Dimensions of hidden layers in the neural network.
                The final dimension determines embedding size.
            n_epochs: int, optional (default=1000)
                Number of training epochs.
                More epochs may improve integration but take longer.
            lr: float, optional (default=0.001)
                Learning rate for Adam optimizer.
                Adjust if training is unstable.
            batch_key: str, optional (default='batch_name')
                Column in adata.obs containing batch information.
            key_added: str, optional (default='STAligner')
                Key for storing embeddings in adata.obsm.
            gradient_clipping: float, optional (default=5)
                Maximum gradient norm for stability.
            weight_decay: float, optional (default=0.0001)
                L2 regularization strength.
            margin: float, optional (default=1)
                Margin for triplet loss function.
                Larger values enforce stronger separation.
            verbose: bool, optional (default=False)
                Whether to print training progress.
            random_seed: int, optional (default=666)
                Random seed for reproducibility.
            iter_comb: list, optional (default=None)
                List of batch pairs to compare.
                If None, compares all pairs.
            knn_neigh: int, optional (default=100)
                Number of neighbors for mutual nearest neighbors.
            Batch_list: list, optional (default=None)
                List of individual batch AnnData objects.
                Must be in same order as in combined adata.
            device: torch.device, optional (default=auto)
                Computing device to use.
                Automatically uses GPU if available.

        Notes:
            - Requires pre-computed spatial networks
            - GPU acceleration recommended for large datasets
            - Batch_list order must match batch_key order
            - Memory usage scales with dataset size
            - Consider reducing knn_neigh for large datasets
        """
        self.device = device
        section_ids = np.array(adata.obs[batch_key].unique())

        comm_gene = adata.var_names
        data_list = []
        for adata_tmp in Batch_list:
            adata_tmp = adata_tmp[:, comm_gene].copy()   # line 268 avoid 'ArrayView'
            adata_tmp_X = adata_tmp.X.toarray() if hasattr(adata_tmp.X, 'toarray') else adata_tmp.X
            edge_index = np.nonzero(adata_tmp.uns['adj'])
            data_list.append(
                Data(edge_index=torch.LongTensor(np.array([edge_index[0], edge_index[1]])),
                              prune_edge_index=torch.LongTensor(np.array([])),
                              x=torch.FloatTensor(adata_tmp_X)))

        loader = DataLoader(data_list, batch_size=1, shuffle=True)

        self.loader=loader
        self.adata = adata
        self.data_list = data_list

        # hyper-parameters
        self.lr=lr
        self.section_ids = section_ids
        self.n_epochs = n_epochs
        self.weight_decay=weight_decay
        self.hidden_dims = hidden_dims
        self.key_added = key_added
        self.gradient_clipping = gradient_clipping
        self.random_seed = random_seed
        self.margin = margin
        self.verbose = verbose
        self.iter_comb = iter_comb
        self.knn_neigh = knn_neigh
        self.Batch_list = Batch_list
        self.batch_key = batch_key
        self.model = STAligner(hidden_dims=[adata.X.shape[1], hidden_dims[0],
                                            hidden_dims[1]]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                          weight_decay=weight_decay)

        if verbose:
            print(self.model)

    def train(self):
        r"""Train the STAligner spatial integration model.
        
        This method performs two-stage training of the STAligner model:
        1. Pre-training with STAGATE to learn initial embeddings
        2. Fine-tuning with STAligner using triplet loss and MNN

        The training process:
        1. Sets random seeds for reproducibility
        2. Pre-trains with graph autoencoder
        3. Identifies mutual nearest neighbors
        4. Optimizes embeddings with triplet loss
        5. Monitors training progress
        6. Saves final embeddings

        Notes:
            - Progress shown if verbose=True
            - Uses GPU if available
            - Early stopping not implemented
            - Results stored in adata.obsm[key_added]
            - Memory usage increases during training
            - Consider batch size for large datasets
        """
        seed = self.random_seed
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        print('Pretrain with STAGATE...')
        for epoch in tqdm(range(0, 500)):
            for batch in self.loader:
                self.model.train()
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                z, out = self.model(batch.x, batch.edge_index)

                loss = F.mse_loss(batch.x, out)  # +adv_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()


        with torch.no_grad():
            z_list = []
            for batch in self.data_list:
                z, _ = self.model.cpu()(batch.x, batch.edge_index)
                z_list.append(z.cpu().detach().numpy())
        self.adata.obsm['STAGATE'] = np.concatenate(z_list, axis=0)
        self.model = self.model.to(self.device)


        print('Train with STAligner...')
        for epoch in tqdm(range(500, self.n_epochs)):
            if epoch % 100 == 0 or epoch == 500:
                if self.verbose:
                    print('Update spot triplets at epoch ' + str(epoch))
                with torch.no_grad():
                    z_list = []
                    for batch in self.data_list:
                        z, _ = self.model.cpu()(batch.x, batch.edge_index)
                        z_list.append(z.cpu().detach().numpy())

                self.adata.obsm['STAGATE'] = np.concatenate(z_list, axis=0)
                self.model = self.model.to(self.device)

                pair_data_list = []

                for comb in self.iter_comb:
                    #print(comb)
                    i, j = comb[0], comb[1]
                    batch_pair = self.adata[self.adata.obs[self.batch_key].isin([self.section_ids[i],
                                                                                  self.section_ids[j]])]
                    mnn_dict = create_dictionary_mnn(batch_pair, use_rep='STAGATE', batch_name=self.batch_key,
                                                           k=self.knn_neigh,
                                                           iter_comb=None, verbose=0)

                    batchname_list = batch_pair.obs[self.batch_key]
                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(self.section_ids)):
                        cellname_by_batch_dict[self.section_ids[batch_id]] = batch_pair.obs_names[
                            batch_pair.obs[self.batch_key] == self.section_ids[batch_id]].values
                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for batch_pair_name in mnn_dict.keys():  # pairwise compare for multiple batches
                        for anchor in mnn_dict[batch_pair_name].keys():
                            anchor_list.append(anchor)
                            positive_spot = mnn_dict[batch_pair_name][anchor][0]
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                            negative_list.append(
                                cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                    batch_as_dict = dict(zip(list(batch_pair.obs_names),
                                              range(0, batch_pair.shape[0])))
                    anchor_ind = list(map(lambda _: batch_as_dict[_], anchor_list))
                    positive_ind = list(map(lambda _: batch_as_dict[_], positive_list))
                    negative_ind = list(map(lambda _: batch_as_dict[_], negative_list))

                    edge_list_1 = np.nonzero(self.Batch_list[i].uns['adj'])

                    max_ind = edge_list_1[0].max()
                    edge_list_2 = np.nonzero(self.Batch_list[j].uns['adj'])

                    edge_list_2 = (edge_list_2[0] + max_ind + 1, edge_list_2[1] + max_ind + 1)
                    edge_list = [edge_list_1, edge_list_2]

                    edge_pairs = [np.append(edge_list[0][0], edge_list[1][0]),
                                   np.append(edge_list[0][1], edge_list[1][1])]

                    pair_data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
                                           anchor_ind=torch.LongTensor(np.array(anchor_ind)),
                                           positive_ind=torch.LongTensor(np.array(positive_ind)),
                                           negative_ind=torch.LongTensor(np.array(negative_ind)),
                                           x=batch_pair.X)) 
                    #torch.FloatTensor(batch_pair.X.todense())

                # for temp in pair_data_list:
                #     temp.to(device)
                pair_loader = DataLoader(pair_data_list, batch_size=1, shuffle=True)

            for batch in pair_loader:
                self.model.train()
                self.optimizer.zero_grad()
                torch_data = batch.x[0].copy()
                if hasattr(torch_data, 'toarray'):
                    torch_data = torch_data.toarray()
                batch.x = torch.FloatTensor(torch_data)
                batch = batch.to(self.device)
                z, out = self.model(batch.x, batch.edge_index)
                mse_loss = F.mse_loss(batch.x, out)

                anchor_arr = z[batch.anchor_ind,]
                positive_arr = z[batch.positive_ind,]
                negative_arr = z[batch.negative_ind,]

                triplet_loss = torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='sum')
                tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

                loss = mse_loss + tri_output
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                self.optimizer.step()

        add_reference(self.adata,'STAligner','spatial integration with STAligner')

    def predicted(self):
        r"""Generate and store the final embedding from trained STAligner model.
        
        Returns:
            AnnData object with STAligner embedding stored in obsm[key_added].
        """ 
        self.model.eval()
        with torch.no_grad():
            z_list = []
            for batch in self.data_list:
                z, _ = self.model.cpu()(batch.x, batch.edge_index)
                z_list.append(z.cpu().detach().numpy())

        self.adata.obsm[self.key_added] = np.concatenate(z_list, axis=0)
        add_reference(self.adata,'STAligner','spatial integration with STAligner')
        return self.adata
# End-of-file (EOF)
