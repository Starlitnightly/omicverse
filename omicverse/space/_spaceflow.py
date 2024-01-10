import numpy as np
import pandas as pd
import scanpy as sc
import torch
import random
from tqdm import tqdm
import torch.nn as nn

import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from sklearn.neighbors import kneighbors_graph

sf_install = False

class pySpaceFlow(object):

    def __init__(self,adata) -> None:
        global sf_install
        try:
            import SpaceFlow
            sf_install=True
            #print('mofax have been install version:',mfx.__version__)
        except ImportError:
            raise ImportError(
                'Please install the SpaceFlow: `pip install SpaceFlow`.'
            )
        from SpaceFlow import SpaceFlow
        sf = SpaceFlow.SpaceFlow(adata=adata, 
                         spatial_locs=adata.obsm['spatial'])
        
        
        spatial_locs = adata.obsm['spatial']
        spatial_graph = sf.graph_alpha(spatial_locs, n_neighbors=10)

        sf.adata_preprocessed = adata
        sf.spatial_graph = spatial_graph
        self.sf = sf
        self.adata = adata

    def train(self,spatial_regularization_strength=0.1, 
              z_dim=50, lr=1e-3, epochs=1000, max_patience=50, 
              min_stop=100, random_seed=42, gpu=0, 
              regularization_acceleration=True, edge_subset_sz=1000000):
        
        from SpaceFlow.util import sparse_mx_to_torch_edge_list, corruption

        adata_preprocessed, spatial_graph = self.sf.adata_preprocessed, self.sf.spatial_graph
        if not adata_preprocessed:
            print("The data has not been preprocessed, please run preprocessing_data() method first!")
            return
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        model = DeepGraphInfomax(
            hidden_channels=z_dim, encoder=GraphEncoder(adata_preprocessed.shape[1], z_dim),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption).to(device)

        expr = adata_preprocessed.X.todense() if type(adata_preprocessed.X).__module__ != np.__name__ else adata_preprocessed.X
        expr = torch.tensor(expr).float().to(device)

        edge_list = sparse_mx_to_torch_edge_list(spatial_graph).to(device)

        model.train()
        min_loss = np.inf
        patience = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_params = model.state_dict()

        for epoch in tqdm(range(epochs)):
            train_loss = 0.0
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            z, neg_z, summary = model(expr, edge_list)
            loss = model.loss(z, neg_z, summary)

            coords = torch.tensor(adata_preprocessed.obsm['spatial']).float().to(device)
            if regularization_acceleration or adata_preprocessed.shape[0] > 5000:
                cell_random_subset_1, cell_random_subset_2 = torch.randint(0, z.shape[0], (edge_subset_sz,)).to(
                    device), torch.randint(0, z.shape[0], (edge_subset_sz,)).to(device)
                z1, z2 = torch.index_select(z, 0, cell_random_subset_1), torch.index_select(z, 0, cell_random_subset_2)
                c1, c2 = torch.index_select(coords, 0, cell_random_subset_1), torch.index_select(coords, 0,
                                                                                                 cell_random_subset_1)
                pdist = torch.nn.PairwiseDistance(p=2)

                z_dists = pdist(z1, z2)
                z_dists = z_dists / torch.max(z_dists)

                sp_dists = pdist(c1, c2)
                sp_dists = sp_dists / torch.max(sp_dists)
                n_items = z_dists.size(dim=0)
            else:
                z_dists = torch.cdist(z, z, p=2)
                z_dists = torch.div(z_dists, torch.max(z_dists)).to(device)
                sp_dists = torch.cdist(coords, coords, p=2)
                sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
                n_items = z.size(dim=0) * z.size(dim=0)

            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
            loss = loss + spatial_regularization_strength * penalty_1

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if train_loss > min_loss:
                patience += 1
            else:
                patience = 0
                min_loss = train_loss
                best_params = model.state_dict()
            if patience > max_patience and epoch > min_stop:
                break

        model.load_state_dict(best_params)

        z, _, _ = model(expr, edge_list)
        embedding = z.cpu().detach().numpy()

        self.sf.embedding = embedding
        self.adata.obsm['spaceflow']=self.sf.embedding.copy()

        return embedding
    
    def cal_pSM(self,n_neighbors:int=20,resolution:int=1,
                       max_cell_for_subsampling:int=5000,
                       psm_key='pSM_spaceflow'):
        """
        Calculate the pseudo-spatial map using diffusion pseudotime (DPT) algorithm.

        Parameters
        ----------
        n_neighbors: int
            Number of neighbors for constructing the kNN graph.
        resolution: float
            Resolution for clustering.
        max_cell_for_subsampling: int
            Maximum number of cells for subsampling. If the number of cells is larger than this value, the subsampling will be performed.

        Returns
        -------
        pSM_values: numpy.ndarray
            The pseudo-spatial map values.
        
        """

        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, 
               use_rep='spaceflow')
        sc.tl.umap(self.adata)
        sc.tl.leiden(self.adata, resolution=resolution)
        sc.tl.paga(self.adata)
        max_cell_for_subsampling = max_cell_for_subsampling
        if self.adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = self.adata.obsm['spaceflow']
        else:
            indices = np.arange(self.adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = self.adata[selected_ind, :].obsm['spaceflow']

        from scipy.spatial import distance_matrix
        import numpy as np
        sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
        self.adata.uns['iroot'] = np.argmax(sum_dists)
        sc.tl.diffmap(self.adata)
        sc.tl.dpt(self.adata)
        self.adata.obs.rename({"dpt_pseudotime": psm_key}, axis=1, inplace=True)
        print(f'The pseudo-spatial map values are stored in adata.obs["{psm_key}"].')

        pSM_values = self.adata.obs[psm_key].to_numpy()
        return pSM_values


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x