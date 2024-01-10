

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from ..STAGATE_pyG import Batch_Data,Cal_Spatial_Net,Transfer_pytorch_Data,Stats_Spatial_Net,STAGATE
import scanpy as sc

class pySTAGATE(object):
    
    def __init__(self,adata,num_batch_x,num_batch_y,
                 spatial_key=['X','Y'],batch_size=1,
                rad_cutoff=200,num_epoch = 1000,lr=0.001,
                weight_decay=1e-4,hidden_dims = [512, 30],
                device='cuda:0'):
        
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.device=device
        
        Batch_list = Batch_Data(adata, num_batch_x=num_batch_x, num_batch_y=num_batch_y,
                                    spatial_key=spatial_key, plot_Stats=True)
        for temp_adata in Batch_list:
            Cal_Spatial_Net(temp_adata, rad_cutoff=rad_cutoff)
        
        
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data_list = [Transfer_pytorch_Data(adata) for adata in Batch_list]
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
        
        self.model = STAGATE(hidden_dims = [data_list[0].x.shape[1]]+self.hidden_dims).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        
    def train(self):
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

        STAGATE_rep = z.to('cpu').detach().numpy()
        self.adata.obsm['STAGATE'] = STAGATE_rep
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX<0] = 0
        self.adata.layers['STAGATE_ReX'] = ReX

        print('The STAGATE representation values are stored in adata.obsm["STAGATE"].')
        print('The ReX values are stored in adata.layers["STAGATE_ReX"].')

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
            Maximum number of cells for subsampling. If the number of cells is larger than this value, the subsampling will be performed.

        Returns
        -------
        pSM_values: numpy.ndarray
            The pseudo-spatial map values.
        
        """

        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, 
               use_rep='STAGATE')
        sc.tl.umap(self.adata)
        sc.tl.leiden(self.adata, resolution=resolution)
        sc.tl.paga(self.adata)
        max_cell_for_subsampling = max_cell_for_subsampling
        if self.adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = self.adata.obsm['STAGATE']
        else:
            indices = np.arange(self.adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = self.adata[selected_ind, :].obsm['STAGATE']

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
        