"""Module providing a encapsulation of pySTAGATE."""
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from ..externel.STAGATE_pyG import Batch_Data,Cal_Spatial_Net,Transfer_pytorch_Data,Stats_Spatial_Net,STAGATE
import scanpy as sc
from anndata import AnnData

class pySTAGATE:
    """Class representing the object of pySTAGATE."""

    def __init__(self,
                 adata: AnnData,
                 num_batch_x,
                 num_batch_y,
                 spatial_key: list = ['X','Y'],
                 batch_size: int = 1,
                rad_cutoff: int = 200,
                num_epoch: int = 1000,
                lr: float = 0.001,
                weight_decay: float = 1e-4,
                hidden_dims: list = [512, 30],
                device: str = 'cuda:0')-> None:
        # Initialize device
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.device=device

        # Create batches
        batch_list = Batch_Data(adata, num_batch_x=num_batch_x, num_batch_y=num_batch_y,
                                    spatial_key=spatial_key, plot_Stats=True)
        for temp_adata in batch_list:
            Cal_Spatial_Net(temp_adata, rad_cutoff=rad_cutoff)

        # Transfer to PyTorch data format
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data_list = [Transfer_pytorch_Data(adata) for adata in batch_list]
        for temp in data_list:
            temp.to(device)

        Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
        data = Transfer_pytorch_Data(adata)
        Stats_Spatial_Net(adata)

        # batch_size=1 or 2
        self.loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

        # hyper-parameters
        self.num_epoch = num_epoch
        self.lr=lr
        self.weight_decay=weight_decay
        self.hidden_dims = hidden_dims
        self.adata=adata
        self.data=data

        # Model and optimizer
        self.model = STAGATE(hidden_dims = [data_list[0].x.shape[1]]+self.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=self.lr,
                                            weight_decay=self.weight_decay)

    def train(self):
        """Train the STAGATE model."""       
        for epoch in tqdm(range(1, self.num_epoch+1)):
            for batch in self.loader:
                self.model.train()
                self.optimizer.zero_grad()
                z, out = self.model(batch.x, batch.edge_index)
                loss = F.mse_loss(batch.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()
        # The total network
        self.data.to(self.device)

    def predicted(self):
        """
        Predict the STAGATE representation and ReX values for all cells.
        """
        self.model.eval()
        z, out = self.model(self.data.x, self.data.edge_index)

        stagate_rep = z.to('cpu').detach().numpy()
        self.adata.obsm['STAGATE'] = stagate_rep
        rex = out.to('cpu').detach().numpy()
        rex[rex<0] = 0
        self.adata.layers['STAGATE_ReX'] = rex

        print('The STAGATE representation values are stored in adata.obsm["STAGATE"].')
        print('The rex values are stored in adata.layers["STAGATE_ReX"].')

    def cal_pSM(self,n_neighbors:int=20,resolution:int=1,
                       max_cell_for_subsampling:int=5000,
                       psm_key='pSM_STAGATE'):
        """
        Calculate the pseudo-spatial map using diffusion pseudotime (DPT) algorithm.

        Parameters
        ----------
        n_neighbors: int
            Number of neighbors for constructing the kNN graph.
        resolution: float
            Resolution for clustering.
        max_cell_for_subsampling: int
            Maximum number of cells for subsampling. 
            If the number of cells is larger than this value, the subsampling will be performed.

        Returns
        -------
        pSM_values: numpy.ndarray
            The pseudo-spatial map values.
        
        """

        from scipy.spatial import distance_matrix
        import numpy as np

        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors,
               use_rep='STAGATE')
        sc.tl.umap(self.adata)
        sc.tl.leiden(self.adata, resolution=resolution)
        sc.tl.paga(self.adata)
       # max_cell_for_subsampling = max_cell_for_subsampling
        if self.adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = self.adata.obsm['STAGATE']
        else:
            indices = np.arange(self.adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = self.adata[selected_ind, :].obsm['STAGATE']

        sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
        self.adata.uns['iroot'] = np.argmax(sum_dists)
        sc.tl.diffmap(self.adata)
        sc.tl.dpt(self.adata)
        self.adata.obs.rename({"dpt_pseudotime": psm_key}, axis=1, inplace=True)
        print(f'The pseudo-spatial map values are stored in adata.obs["{psm_key}"].')

        psm_values = self.adata.obs[psm_key].to_numpy()
        return psm_values
        # End-of-file (EOF)
